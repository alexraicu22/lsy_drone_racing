"""Fast racing MPC on an exact model of the simulator, using the attitude interface.

The controller has three layers. Everything is computed from the observations and the config, so it
also works on other layouts, not only on the level-2 track.

1. Model. An exact CasADi replica of crazyflow's attitude interface for the cf21B_500
   (``first_principles`` dynamics + onboard Mellinger controller):
   - the Mellinger torque law (rotation-matrix error, rate and rate-derivative damping, +-32000 PWM
     clip) and the legacy PWM mixing with per-motor clipping to [PWM_MIN, PWM_MAX], which is what
     costs torque authority at high collective thrust,
   - motor force -> rotor speed command, the asymmetric spin-up/spin-down rotor dynamics, the rotor
     thrust and drag-torque curves, propeller gyroscopic torques and the body drag.
   Clips and switches are smoothed for the optimizer. Over 0.4 s open loop it predicts the position
   to ~0.3 cm in moderate and ~1 cm in aggressive flight (an identified lag model: 4 and 26 cm).
2. Planner (cached per track). A smooth path through the gate sequence (IPOPT on a polyline):
   crossings at the gate centres, entry/exit funnels around each gate, clearance from poles and from
   gate frames (extra margin for gates the path is not crossing, whose pose is randomized), no dip
   below (rise above) a gate approached from above (below). Then a minimum-time speed profile with
   a gravity-aware thrust limit, lateral-acceleration and jerk limits; one profile per drone mass.
   (``PLANNER = "mintime"`` instead plans a jerk-limited minimum-time point-mass trajectory near
   that path; it is faster but leaves less margin for corrections.)
3. MPC. acados SQP (MPC_SQP_ITERS iterations) on the exact model at the environment frequency:
   - the reference time follows the drone's progress (closest point on the plan),
   - the drone's mass is estimated in the first ~0.3 s of flight and the profile of that mass used,
     a disturbance observer adds a small acceleration offset to the model,
   - when a gate's true pose is revealed (sensor range) the reference near it moves rigidly with the
     gate, and the MPC keeps the drone inside a funnel around the observed gate just before the
     crossing; revealed poles push the reference away and enter soft clearance constraints,
   - the reference continues past the last gate, so the drone never brakes for the finish.

Run it with:
    pixi run python scripts/sim.py --config level2.toml --controller time_optimal_mpc.py
with ``control_mode = "attitude"`` in the config.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from collections.abc import Iterator

    from crazyflow import Sim
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------
# Knobs (aggressiveness vs. reliability)
# ------------------------------------------------------------------------------------------------
V_MAX = 5.0  # m/s. Speed limit of the reference.
A_LAT = 10.0  # m/s^2. Lateral acceleration limit (sets the speed in turns: v = sqrt(A_LAT /
#               curvature)).
A_TOT = 12.0  # m/s^2. (unused; the thrust limit below includes gravity)
MASS_ADAPT = True  # Estimate the mass in the first ~0.3 s of flight and switch to a speed profile
#                    whose thrust limit (THRUST_BUDGET / mass) that drone can really fly.
THRUST_BUDGET = 0.66  # N.
MASS_GRID = (0.0384, 0.0409, 0.0434, 0.0459, 0.0484)  # kg. Precomputed profiles (43.4 +- 5 g).
MASS_EST_STEPS = (3, 15)  # Airborne steps whose samples are used; the estimate is applied after
#                           them.
MASS_EST_BIAS = 0.0  # kg. Remaining bias of the estimate (identified on the simulator).
MASS_SWITCH_WARM = False  # Keep the MPC's warm start when switching to the mass's profile (same
#                           path).
A_THRUST = 13.4  # m/s^2. Limit of |acceleration + g| (thrust / mass): 0.65 N on the heaviest (48 g)
#                  drone. Braking a descent or climbing needs more thrust than flying level: gravity
#                  counts.
A_ACC = 10.0  # m/s^2. Acceleration limit along the path ...
A_DEC = 10.0  # m/s^2. ... and deceleration limit.
GATE_APPROACH_V = 2.5  # m/s. Speed cap over the last GATE_APPROACH_D before gates whose true pose
#                        is
GATE_APPROACH_D = 0.8  # m.   only revealed late (at 0.7 m): time to react. Early-revealed gates: no
#                        cap.
GATE_APPROACH_PASSES = ()  # Tuple of gate-pass indices the cap applies to (None: all late passes).
J_LAT = 1000.0  # m/s^3. Limit on how fast the sideways acceleration may change (the attitude loop
#                 needs time to re-tilt): v <= (J_LAT / |d curvature / ds|)^(1/3).
KAPPA_WIN = 0.1  # m. Curvature is max-filtered over +-this distance (slow down before turns).
GATE_STRAIGHT = 0.0  # m. Fixed straight piece before/after each gate (0: use the funnels below).
GATE_STRAIGHT_OUT = 0.0  # m.
GATE_STRAIGHT_MIN = 0.15  # m.
FUNNEL_LEN = (0.5, 0.35)  # m. Path length before/after each gate that must stay inside a funnel ...
FUNNEL_W0 = 0.03  # m. ... whose half-width is this at the gate centre ...
FUNNEL_SLOPE = (0.47, 0.84)  # ... and grows with this slope with the distance from the gate plane
#                              (entry 25 deg, exit 40 deg): aligned crossing without kinks.
FUNNEL_NO_UNDERSHOOT = True  # Approach a lower (higher) gate from above (below) without dipping
#                              below (rising above) its centre in the entry funnel: no last-moment
#                              climb.
FUNNEL_SLOPE_Z = None  # (entry, exit) vertical slopes; None: same as FUNNEL_SLOPE. A flat vertical
#                        entry funnel makes the path reach the gate's height early (no dive into low
#                        gates).
GATE_MAX_ANGLE = 60.0  # deg. Max angle between the path and the gate normal at the gate centre.
PATH_H = 0.08  # m. Spacing of the path points optimized between gates.
PATH_LEN_WEIGHT = 0.02  # Weight of path length vs. curvature (squared second differences).
PATH_Z_WEIGHT = 4.0  # Extra weight on vertical curvature (climbs/dives cost a drone more than
#                      turns).
PATH_STARTS = (1.0, 0.6, 1.6)  # Hermite tangent scales of the initial guesses (best lap is kept).
POLE_CLEARANCE_PLAN = 0.24  # m. Horizontal clearance of the path from (nominal) pole axes.
FRAME_CLEARANCE = 0.12  # m. Clearance of the path from gate-frame spheres (bar spheres r = 0.08 m).
FRAME_RAND_MARGIN = 0.18  # m. Extra clearance from frames of gates the path is not crossing right
#                           then (their true pose may differ by +-15 cm and +-0.2 rad).
REVEAL_EARLY = False  # Bend the path past gates it will fly through later, so their true pose is
REVEAL_RADIUS = 0.5  # m. revealed early: pass within this horizontal distance of the nominal centre
#                      ...
REVEAL_DETOUR = 1.0  # m. ... if an earlier part of the path already comes this close.
FLOOR_Z = 0.25  # m. Minimum height of the path after the first TAKEOFF_DIST metres.
TAKEOFF_DIST = 0.5  # m.
MPC_THRUST_MAX = 0.72  # N. Above ~0.7 N the motors have little headroom left for torque.
MPC_CMD_MAX = 0.85  # rad. Bound of the roll/pitch commands. Values above MPC_TILT_MAX let the MPC
#                    overdrive the overdamped (~130 ms) attitude loop, but caused floor crashes.
MPC_TILT_MAX = 1.0  # rad. Soft limit on the drone's actual tilt (angle of the thrust axis).
YAW_CMD_MAX = np.pi / 4 + 0.25  # rad. Bound of the yaw command (gates need at most +-pi/4).
POLE_CLEARANCE_MPC = 0.16  # m. Soft clearance the MPC keeps from observed pole axes.
GATE_TUNNEL = (0.08, 0.10, 0.10)  # m (lateral, up, down). Near a crossing the MPC keeps the drone's
#                                   sideways/vertical offset from
GATE_FUNNEL_SLOPE = 0.4  # the observed gate's centre line below this + slope * distance to the gate
#                          plane
GATE_TUNNEL_T = (0.15, 0.05)  # s. (soft) for prediction steps from this long before to this long
#                               after the planned crossing: a funnel that lines the drone up with
#                               the real gate.
MODEL_SMOOTH = 1.0  # Width factor of the smoothed clips/switches in the MPC model (optimizer-
#                     friendly).
YAW_ALIGN = True  # Turn the (square) drone so its sides are parallel to each gate's frame.

MPC_HORIZON = 35  # Steps of 1/env_freq seconds.
MPC_INT_STEPS = 2  # RK4 steps per MPC interval.
W_POS = (300.0, 300.0, 400.0)  # Tracking weights: position x, y, z
W_VEL = 10.0  # velocity
W_ATT = (2.0, 2.0, 30.0)  # roll, pitch, yaw
W_INPUT = (1.0, 1.0, 30.0)  # deviation from the reference roll, pitch, thrust command
W_RATE = (5.0, 5.0, 50.0)  # command minus its lagged state (roll, pitch, thrust): a rate penalty
W_TERMINAL = 3.0  # terminal cost factor
MPC_SQP_ITERS = 20  # 1: real-time iteration (one QP per step); >1: full SQP with this many
#                     iterations.
MPC_LM = 1.0  # Levenberg-Marquardt regularization of the Gauss-Newton Hessian (damps the steps).
MPC_TOL = None  # Stationarity tolerance for early termination of the SQP (None: acados default).
MPC_EXTRA_SOLVES = 0  # Extra SQP runs (continuing from the last iterate) in steps where the
#                       reference changed (gate/pole revealed): the shifted warm start is far from
#                       the new optimum.
REVEAL_REWARM = False  # Re-initialize the horizon by a fallback rollout on the changed reference.
POLE_SLACK_WEIGHTS = (3e4, 3e6)  # L1 and L2 penalty on pole-clearance violation (in m^2)
SLACK_WEIGHTS = (1e3, 1e5)  # L1 and L2 penalty on the other soft constraints (tilt, gate funnel)
DIST_OBSERVER_GAIN = 0.1  # Per step. 0 disables the disturbance observer.
FALLBACK_KP, FALLBACK_KD = 6.0, 4.0  # PD gains of the fallback controller (1/s^2, 1/s).
START_BLEND = 1.0  # m. The reference is moved onto the actual start position over this distance.
TAKEOFF_DZ = 0.001  # m. Until the drone has risen this much it climbs level at full thrust ...
TAKEOFF_THRUST = 0.8  # N. ... with this thrust (the motors start at 0 rpm; spin them up fast).
TAKEOFF_Z = 0.005  # m. Or until it is this high (absolute) and rising: the drone first drops onto
#                    the floor while the rotors spin up and only then climbs back past its start
#                    height (>0: on).
DIST_MAX = 3.0  # m/s^2. Clamp of the estimated disturbance acceleration.
GATE_DEFORM = True  # Move the reference with revealed gates (else only the MPC's gate constraint).
DEFORM_ROTATION = True  # Also rotate the reference with a revealed gate's yaw (big sideways swings
#                          0.5 m before the gate for a small change of the crossing angle: off).
DEFORM_DEADBAND = 0.0  # m. Sideways/vertical gate shifts up to this are not corrected (the opening
#                         tolerates them); larger ones are corrected minus this amount.
DEFORM_DEADBAND_Z = None  # m. Separate vertical dead band (None: DEFORM_DEADBAND). The opening's
#                           vertical tolerance for the drone's centre (-0.13/+0.09 m) nearly covers
#                           the +-0.1 m.
REVEAL_R1 = 0.5  # m. Reference points this close to a revealed gate move rigidly with the gate ...
REVEAL_R2 = 1.0  # m. ... fading out to no movement at this distance.
POLE_PUSH = 0.25  # m. The reference is pushed out to this horizontal distance from revealed poles.
REF_DT = 0.01  # s. Resolution of the stored reference.
PLANNER = "profile"  # "profile": smoothed path + speed profile; "mintime": minimum-time point-mass
#                      trajectory (jerk-limited, warm-started from the profile reference, same
#                      topology).
MT_N = 40  # Nodes per gate-to-gate segment.
MT_THRUST = 0.70  # N. Thrust limit of the plan (acceleration limit MT_THRUST / mass).
MT_TILT = 60.0  # deg. Tilt limit of the planned thrust vector.
MT_JERK = 0.0  # m/s^3. Jerk limit (0: none). The attitude loop needs time to turn the thrust
#                vector.
MT_WIN = 0.10  # m. Crossing point within this of the gate centre (vertically aimed 2 cm low: drone
#                sphere).
MT_ANGLE = 60.0  # deg. Max angle between the crossing velocity and the gate normal.
MT_Z_AIM = -0.02  # m. Vertical aim of the crossing window (the drone's collision sphere sits 2 cm
#                   up).
MT_FRAME = 0.126  # m. Clearance of the trajectory from gate-frame spheres (drone sphere 0.086 +
#                   margin).
MT_FRAME_RAND = 0.0  # m. Extra clearance from frames of gates not being crossed (randomized poses).
MT_POLE = None  # m. Pole clearance (None: POLE_CLEARANCE_PLAN).
MT_FLOOR = 0.15  # m. Minimum height after the first 0.4 s.
MT_TRUST = 0.5  # m. Each node stays within this of the warm start's node (same route, local
#                 refinement).
MT_SPEED_CAP = None  # {pass index: max speed} over the last MT_CAP_FRAC of the segment into that
#                      gate
MT_CAP_FRAC = 0.3  # (late reveals need time to react).
MT_GATE_THRUST = None  # N. Lower thrust limit over the same approach parts: spare authority for the
#                        correction when the gate's true pose is revealed (None: MT_THRUST).

# ------------------------------------------------------------------------------------------------
# Model of crazyflow's first_principles drone (cf21B_500) + Mellinger attitude controller
# ------------------------------------------------------------------------------------------------
G = 9.81
MASS = 0.0434  # Nominal mass in kg. Level 2 randomizes +-5 g; the disturbance observer absorbs it.
J = np.array([25e-6, 28e-6, 49e-6])  # crazyflow/dynamics/first_principles/params.toml
DRAG = np.array([-0.021643637770852733, -0.021643637770852733, -0.02477138502714662])
KR = np.array([70000.0, 70000.0, 60000.0])  # crazyflow/control/mellinger/params.toml
KW = np.array([20000.0, 20000.0, 12000.0])
KD = np.array([200.0, 200.0, 0.0])  # kd_omega; the firmware zeroes the yaw component
TORQUE_PWM_MAX = 32000.0
PWM_MIN, PWM_MAX = 7000.0, 65535.0
THRUST_MAX_MOTOR = 0.2  # N per motor at PWM_MAX
ARM = 0.035355  # m. Rotor arm (crazyflow/dynamics/first_principles/params.toml, cf21B_500)
# motor force = k1 w + k2 w^2 (w in rpm)
RPM2THRUST = (-3.133427287299859e-7, 4.407354891648379e-10)
RPM2TORQUE = (1.65886356219615e-9, 2.4693477924534137e-12)  # rotor drag torque
ROTOR_DYN = (13.996001897562685, 0.00011093207920685363, 5.933168530682111, 0.00031951312393561264)
PROP_INERTIA = 38.93e-9  # kg m^2
MIXING = np.array([[-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
KRPM = 1000.0  # rotor speeds are states in krpm (numerical scaling)
TAU_D = 0.0226  # s. Time constants of the command filters behind the rate penalty (roll/pitch) ...
TAU_T = 0.0612  # s. ... and thrust.
THRUST_MIN = 4 * 0.02136263065537499  # N. Environment action lower bound.
NX, NU = 19, 3  # the MPC decides roll, pitch and thrust; yaw follows a schedule (parameter)
MAX_POLES = 8
NP = 5 + 2 * MAX_POLES + 9  # [1/m, d(3), yaw_cmd, poles (2 each), gate centre(3), lateral(3),
#                             up(3)]
GATE_P = 5 + 2 * MAX_POLES  # index of the gate parameters
_VERSION = 20  # bump to invalidate cached plans and solvers


def _smin(a: ca.SX, b: ca.SX, eps: float) -> ca.SX:
    """Smooth minimum."""
    return 0.5 * (a + b - ca.sqrt((a - b) ** 2 + eps**2))


def _smax(a: ca.SX, b: ca.SX, eps: float) -> ca.SX:
    """Smooth maximum."""
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps**2))


def _rot(rpy: ca.SX) -> ca.SX:
    """Rotation matrix of scipy's extrinsic "xyz" Euler angles, R = Rz(y) Ry(p) Rx(r)."""
    cr, sr = ca.cos(rpy[0]), ca.sin(rpy[0])
    cp, sp = ca.cos(rpy[1]), ca.sin(rpy[1])
    cy, sy = ca.cos(rpy[2]), ca.sin(rpy[2])
    return ca.vertcat(
        ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        ca.horzcat(-sp, cp * sr, cp * cr),
    )


def rotor_thrust(w_krpm: NDArray) -> NDArray:
    """Collective thrust [N] of rotor speeds in krpm (numpy)."""
    w = np.asarray(w_krpm) * KRPM
    return float(np.sum(RPM2THRUST[0] * w + RPM2THRUST[1] * w**2))


def model_dynamics(x: ca.SX, u: ca.SX, p: ca.SX) -> ca.SX:
    """Continuous dynamics x' = f(x, u, p): a replica of crazyflow's attitude interface.

    x = [pos(3), vel(3), rpy(3), body rates(3), rotor speeds(4) [krpm], filtered commands(3)]
    u = [roll_cmd, pitch_cmd, thrust_cmd]   (the environment action is [roll, pitch, yaw, thrust])
    p = [1/mass, disturbance acceleration (3), yaw_cmd, ...]

    Mellinger attitude law (crazyflow/control/mellinger/control.py): rotation-matrix error, rate and
    rate-derivative damping, +-32000 torque-PWM clip, legacy PWM mixing with per-motor clipping to
    [PWM_MIN, PWM_MAX]; motor force -> rotor speed command; asymmetric rotor spin-up/down dynamics;
    rotor thrust/drag-torque curves, propeller gyroscopic torques, body drag
    (crazyflow/dynamics/first_principles). Clips are smoothed for the optimizer.
    """
    vel, rpy, wb, wr = x[3:6], x[6:9], x[9:12], x[12:16] * KRPM
    inv_m, dist = p[0], p[1:4]
    Rm = _rot(rpy)
    mix = ca.DM(MIXING)
    k1, k2 = RPM2THRUST
    f_mot = k1 * wr + k2 * wr**2
    q_mot = RPM2TORQUE[0] * wr + RPM2TORQUE[1] * wr**2
    h = PROP_INERTIA * (mix[2, :] @ (wr * 2 * np.pi / 60))  # propeller angular momentum
    tau = ca.vertcat(
        ARM * (mix[0, :] @ f_mot) + wb[1] * h,
        ARM * (mix[1, :] @ f_mot) - wb[0] * h,
        mix[2, :] @ q_mot,
    )
    Jd = ca.DM(J)
    w_dot = (tau - ca.cross(wb, Jd * wb)) / Jd  # (+ rotor-acceleration yaw torque, added below)
    Rd = _rot(ca.vertcat(u[0], u[1], p[4]))
    E = Rd.T @ Rm - Rm.T @ Rd
    e_r = ca.vertcat(E[2, 1], E[0, 2], E[1, 0])
    cmd = -ca.DM(KR) * e_r - ca.DM(KW) * wb - ca.DM(KD) * w_dot  # KD yaw = 0: no algebraic loop
    cmd = _smax(
        _smin(cmd, TORQUE_PWM_MAX, 200.0 * MODEL_SMOOTH), -TORQUE_PWM_MAX, 200.0 * MODEL_SMOOTH
    )
    f_pwm = u[2] / 4 / THRUST_MAX_MOTOR * PWM_MAX
    pwm = f_pwm + (ca.vertcat(cmd[0] / 2, cmd[1] / 2, cmd[2]).T @ mix).T
    pwm = _smax(_smin(pwm, PWM_MAX, 300.0 * MODEL_SMOOTH), PWM_MIN, 300.0 * MODEL_SMOOTH)
    f_cmd = pwm / PWM_MAX * THRUST_MAX_MOTOR
    w_cmd = (-k1 + ca.sqrt(k1**2 + 4 * k2 * f_cmd)) / (2 * k2)
    a1, a2, d1, d2 = ROTOR_DYN
    spin_up = 0.5 * (1 + ca.tanh((w_cmd - wr) / (20.0 * MODEL_SMOOTH)))
    wr_dot = spin_up * (a1 * (w_cmd - wr) + a2 * (w_cmd**2 - wr**2)) + (1 - spin_up) * (
        d1 * (w_cmd - wr) + d2 * (w_cmd**2 - wr**2)
    )
    w_dot = w_dot + ca.vertcat(0, 0, PROP_INERTIA * (mix[2, :] @ (wr_dot * 2 * np.pi / 60)) / J[2])
    drag = Rm @ (ca.DM(DRAG) * (Rm.T @ vel))
    acc = ca.sum1(f_mot) * inv_m * Rm[:, 2] + ca.DM([0.0, 0.0, -G]) + drag * inv_m + dist
    r, pt = rpy[0], rpy[1]
    cp = ca.cos(pt)
    cp = 0.5 * (cp + 0.15 + ca.sqrt((cp - 0.15) ** 2 + 1e-4))  # smooth max(cos, 0.15): no blow-up
    tp = ca.sin(pt) / cp
    rpy_dot = ca.vertcat(
        wb[0] + ca.sin(r) * tp * wb[1] + ca.cos(r) * tp * wb[2],
        ca.cos(r) * wb[1] - ca.sin(r) * wb[2],
        (ca.sin(r) * wb[1] + ca.cos(r) * wb[2]) / cp,
    )
    filt_dot = ca.vertcat((u[0:2] - x[16:18]) / TAU_D, (u[2] - x[18]) / TAU_T)
    return ca.vertcat(vel, acc, rpy_dot, w_dot, wr_dot / KRPM, filt_dot)


_RK4: dict = {}


def _rk4_function(steps: int = 4) -> ca.Function:
    """Discrete model x_next = F(x, u, dt, p) with fixed-step RK4 (cached)."""
    if steps not in _RK4:
        x, u, p = ca.SX.sym("x", NX), ca.SX.sym("u", NU), ca.SX.sym("p", 5)
        dt = ca.SX.sym("dt")
        f = ca.Function("f", [x, u, p], [model_dynamics(x, u, p)])
        h, xn = dt / steps, x
        for _ in range(steps):
            k1 = f(xn, u, p)
            k2 = f(xn + h / 2 * k1, u, p)
            k3 = f(xn + h / 2 * k2, u, p)
            k4 = f(xn + h * k3, u, p)
            xn = xn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _RK4[steps] = ca.Function("F_rk4", [x, u, dt, p], [xn])
    return _RK4[steps]


def _rpy_from_thrust(f: NDArray, yaw: NDArray) -> NDArray:
    """Roll/pitch (scipy "xyz") that point the body z axis along f for a given yaw."""
    z = f / np.linalg.norm(f, axis=-1, keepdims=True)
    cy, sy = np.cos(yaw), np.sin(yaw)
    zx, zy = cy * z[:, 0] + sy * z[:, 1], -sy * z[:, 0] + cy * z[:, 1]  # rotate by -yaw
    roll = -np.arcsin(np.clip(zy, -1, 1))
    pitch = np.arctan2(zx, z[:, 2])
    return np.stack([roll, pitch, yaw], axis=-1)


# ------------------------------------------------------------------------------------------------
# Planner: gate-anchored path, obstacle-aware smoothing, curvature-limited speed profile
# ------------------------------------------------------------------------------------------------
def _gate_passes(
    gates_pos: NDArray, gates_quat: NDArray, sequence: NDArray, direction: NDArray
) -> list[dict]:
    passes = []
    for gid, sgn in zip(np.asarray(sequence, int), np.asarray(direction)):
        Rg = R.from_quat(gates_quat[gid]).as_matrix()
        n = Rg[:, 0] * (1.0 if sgn >= 0 else -1.0)
        passes.append(
            {
                "c": np.asarray(gates_pos[gid], float),
                "n": n,
                "l": Rg[:, 1],
                "u": Rg[:, 2],
                "gid": int(gid),
                "yaw": float(np.arctan2(n[1], n[0])),
            }
        )
    return passes


def _yaw_targets(passes: list[dict], yaw0: float) -> list[float]:
    """Yaw per gate pass that aligns the square drone with the gate (mod 90 deg), near the last."""
    out, prev = [], yaw0
    for ps in passes:
        cands = ps["yaw"] + np.arange(-8, 9) * np.pi / 2
        cands = cands[np.abs(cands) <= np.pi / 4 + 0.05]  # stay far from the +-pi/2 action clip
        out.append(float(cands[np.argmin(np.abs(cands - prev))]) if YAW_ALIGN else 0.0)
        prev = out[-1]
    return out


def _gate_spheres(gates_pos: NDArray, gates_quat: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Spheres covering every gate's collision geometry (envs/assets/gate.xml).

    The four bars (0.16 m wide, centred 0.28 m from the gate centre) are chains of r=0.08 spheres,
    the stand below the gate (0.1 m square) a chain of r=0.07 spheres down to the floor.
    """
    centers, radii, gids = [], [], []
    s = np.arange(-0.28, 0.2801, 0.08)
    bar = [(0.0, y, z) for y in s for z in (0.28, -0.28)] + [
        (0.0, y, z) for y in (0.28, -0.28) for z in s
    ]
    for g, (c, q) in enumerate(zip(np.asarray(gates_pos, float), np.asarray(gates_quat, float))):
        Rg = R.from_quat(q).as_matrix()
        local = list(dict.fromkeys(bar))
        centers += [c + Rg @ np.array(pt) for pt in local]
        radii += [0.08] * len(local)
        gids += [g] * len(local)
        for z in np.arange(-0.41, -1.36, -0.1):
            if c[2] + z > -0.05:
                centers.append(c + Rg @ np.array([0.0, 0.0, z]))
                radii.append(0.07)
                gids.append(g)
    return np.array(centers), np.array(radii), np.array(gids)


def _hermite(
    p0: NDArray, d0: NDArray, p1: NDArray, d1: NDArray, s: NDArray, scale: float = 1.0
) -> NDArray:
    s = s[:, None]
    L = np.linalg.norm(p1 - p0) * scale
    return (
        (2 * s**3 - 3 * s**2 + 1) * p0
        + (s**3 - 2 * s**2 + s) * L * d0
        + (-2 * s**3 + 3 * s**2) * p1
        + (s**3 - s**2) * L * d1
    )


def _straight_length(c: NDArray, d: NDArray, poles_xy: NDArray, d_max: float) -> float:
    """Longest straight piece from gate centre c along direction d that keeps pole clearance."""
    length = d_max
    dh = d[:2] / max(np.linalg.norm(d[:2]), 1e-6)
    for o in poles_xy:
        rel = o - c[:2]
        along, lat = rel @ dh, abs(rel @ np.array([-dh[1], dh[0]]))
        if along > 0 and lat < POLE_CLEARANCE_PLAN + 0.05:
            length = min(
                length, along - np.sqrt(max((POLE_CLEARANCE_PLAN + 0.05) ** 2 - lat**2, 0.0))
            )
    return max(length, GATE_STRAIGHT_MIN)


def _initial_polyline(
    start: NDArray, passes: list[dict], scale: float = 1.0, poles_xy: NDArray | None = None
) -> tuple[NDArray, NDArray, list[int]]:
    """Dense initial path: Hermite connectors between gates, a fixed straight piece per gate.

    Each connector leaves the previous gate and enters the next along their normals, so it already
    turns the right way; it gets one point per PATH_H of its length.
    """
    pts, fixed, centers = [np.asarray(start, float)], [True], []

    def straight(a: NDArray, b: NDArray):
        n = max(int(np.ceil(np.linalg.norm(b - a) / PATH_H)), 1)
        pts.extend(a + (b - a) * i / n for i in range(1, n))
        fixed.extend([True] * (n - 1))

    d_prev = None
    poles_xy = np.zeros((0, 2)) if poles_xy is None else np.asarray(poles_xy, float).reshape(-1, 2)
    for ps in passes:
        d_in = _straight_length(ps["c"], -ps["n"], poles_xy, GATE_STRAIGHT) if GATE_STRAIGHT else 0
        d_out = (
            _straight_length(ps["c"], ps["n"], poles_xy, GATE_STRAIGHT_OUT) if GATE_STRAIGHT else 0
        )
        pre, c = ps["c"] - d_in * ps["n"], ps["c"]
        post = ps["c"] + d_out * ps["n"]
        a = pts[-1]
        d0 = (pre - a) / max(np.linalg.norm(pre - a), 1e-6) if d_prev is None else d_prev
        curve = _hermite(a, d0, pre, ps["n"], np.linspace(0, 1, 200), scale)
        length = np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
        n = max(int(np.ceil(length / PATH_H)), 2)
        pts.extend(_hermite(a, d0, pre, ps["n"], np.linspace(0, 1, n + 1)[1:-1], scale))
        fixed.extend([False] * (n - 1))
        if GATE_STRAIGHT > 0:
            pts.append(pre)
            fixed.append(True)
            straight(pre, c)
        pts.append(c)
        fixed.append(True)
        centers.append(len(pts) - 1)
        if GATE_STRAIGHT > 0:
            straight(c, post)
            pts.append(post)
            fixed.append(True)
        d_prev = ps["n"]
    if GATE_STRAIGHT <= 0:  # short straight exit after the finish gate (fly through, no braking)
        end = pts[-1] + 0.3 * passes[-1]["n"]
        straight(pts[-1], end)
        pts.append(end)
        fixed.append(True)
    return np.array(pts), np.array(fixed), centers


def _optimize_path(
    start: NDArray,
    passes: list[dict],
    poles_xy: NDArray,
    spheres: tuple,
    limits: tuple,
    scale: float = 1.0,
) -> tuple[NDArray, list[int]]:
    """Smooth path through the gate pieces with hard clearance constraints (IPOPT).

    Minimizes curvature (squared second differences) plus a little length. Frame/stand spheres are
    constrained for the path points near them; the set is updated from each solution until none is
    violated.
    """
    P0, fixed, centers = _initial_polyline(start, passes, scale, poles_xy)
    free = np.nonzero(~fixed)[0]
    poles = np.asarray(poles_xy, float).reshape(-1, 2)
    for o in poles:  # start on a definite side of each pole
        d = P0[free, :2] - o
        r = np.linalg.norm(d, axis=1)
        m = r < POLE_CLEARANCE_PLAN + 0.05
        P0[free[m], :2] = o + d[m] / np.maximum(r[m, None], 1e-6) * (POLE_CLEARANCE_PLAN + 0.05)
    centers_s, radii_s, sgid_s = spheres
    lo, hi = limits
    # path points near a crossing of gate g (initial-guess arc length within 0.8 m): normal clear.
    # from g's frame; elsewhere the extra random-pose margin applies
    s_init0 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P0, axis=0), axis=1))])
    crossing = {}
    for g in np.unique(sgid_s):
        sc = [s_init0[ci] for ci, ps in zip(centers, passes) if ps["gid"] == g]
        crossing[g] = (
            np.any(np.abs(s_init0[:, None] - np.array(sc)[None]) < 0.8, axis=1)
            if sc
            else np.zeros(len(P0), bool)
        )
    s0 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P0, axis=0), axis=1))])
    airborne = np.nonzero(s0[free] > TAKEOFF_DIST)[0]
    active = np.zeros((len(centers_s), len(free)), bool)
    guess = P0[free]
    for _ in range(4):
        dist = np.linalg.norm(guess[None] - centers_s[:, None], axis=2)
        new = (dist < radii_s[:, None] + FRAME_CLEARANCE + FRAME_RAND_MARGIN + 0.4) & ~active
        if _ > 0 and not new.any():
            break
        active |= new
        opti = ca.Opti()
        Z = opti.variable(3, len(free))
        cols, k = [], 0
        for i in range(len(P0)):
            if fixed[i]:
                cols.append(ca.DM(P0[i]))
            else:
                cols.append(Z[:, k])
                k += 1
        P = ca.horzcat(*cols)
        d2 = P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]
        d1 = P[:, 1:] - P[:, :-1]
        smooth_cost = (
            ca.sumsqr(d2[0:2, :]) + PATH_Z_WEIGHT * ca.sumsqr(d2[2, :])
        ) / PATH_H**3 + PATH_LEN_WEIGHT * ca.sumsqr(d1) / PATH_H
        for o in poles:
            opti.subject_to(
                (Z[0, :] - o[0]) ** 2 + (Z[1, :] - o[1]) ** 2 >= (POLE_CLEARANCE_PLAN + 0.02) ** 2
            )  # +2 cm: spline between points
        for j, (c, r) in enumerate(zip(centers_s, radii_s)):
            near = np.nonzero(active[j])[0]
            if len(near):
                margin = np.where(crossing[sgid_s[j]][free[near]], 0.0, FRAME_RAND_MARGIN)
                opti.subject_to(
                    ca.sum1((Z[:, near] - ca.DM(c)) ** 2).T
                    >= (r + FRAME_CLEARANCE + 0.02 + margin) ** 2
                )
        s_init = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P0, axis=0), axis=1))])
        funnel_cost = 0
        for jp, (ci, ps) in enumerate(zip(centers, passes)):  # funnels: aligned approach and exit
            z_prev = (start if jp == 0 else passes[jp - 1]["c"])[2]
            for side, (length, slope) in enumerate(zip(FUNNEL_LEN, FUNNEL_SLOPE)):
                slope_z = slope if FUNNEL_SLOPE_Z is None else FUNNEL_SLOPE_Z[side]
                idx = [
                    i
                    for i in range(len(P0))
                    if not fixed[i]
                    and (
                        0 < s_init[ci] - s_init[i] <= length
                        if side == 0
                        else 0 < s_init[i] - s_init[ci] <= length
                    )
                ]
                if not idx:
                    continue
                rel = P[:, idx] - ca.repmat(ca.DM(ps["c"]), 1, len(idx))
                d = ca.DM(ps["n"]).T @ rel  # along the normal (negative before the gate)
                dist = -d if side == 0 else d
                opti.subject_to(dist >= 0)
                for axis, sl in ((ps["l"], slope), (ps["u"], slope_z)):
                    off = ca.DM(axis).T @ rel
                    opti.subject_to(off <= FUNNEL_W0 + sl * dist)
                    opti.subject_to(-off <= FUNNEL_W0 + sl * dist)
                if (
                    FUNNEL_NO_UNDERSHOOT
                    and side == 0
                    and abs(z_prev - ps["c"][2]) > 0.1
                    and (FUNNEL_NO_UNDERSHOOT is True or jp in FUNNEL_NO_UNDERSHOOT)
                ):
                    under = -np.sign(z_prev - ps["c"][2]) * (ca.DM(ps["u"]).T @ rel) - FUNNEL_W0
                    funnel_cost += 1e4 * ca.sumsqr(0.5 * (under + ca.sqrt(under**2 + 1e-8)))
        reveal_cost = 0
        if REVEAL_EARLY:  # pass close to gates early so their true pose is revealed in time
            seg_of = np.searchsorted(np.array(centers), np.arange(len(P0)), side="right")
            s_c = s_init[np.array(centers)]
            first_pass = {}
            for j, ps in enumerate(passes):
                first_pass.setdefault(ps["gid"], j)
            free_set = set(free.tolist())
            for gid, j in first_pass.items():
                c = passes[j]["c"]
                cand = [
                    i
                    for i in range(len(P0))
                    if i in free_set
                    and seg_of[i] < j
                    and s_c[j] - s_init[i] > 1.0
                    and np.min(np.abs(s_c - s_init[i])) > 0.6
                ]
                if not cand:
                    continue
                d = np.linalg.norm(P0[cand, :2] - c[:2], axis=1)
                if d.min() >= REVEAL_DETOUR:
                    continue
                i_best = cand[int(np.argmin(d))]
                excess = (P[0, i_best] - c[0]) ** 2 + (P[1, i_best] - c[1]) ** 2 - REVEAL_RADIUS**2
                reveal_cost += 500.0 * (0.5 * (excess + ca.sqrt(excess**2 + 1e-6))) ** 2
        opti.minimize(smooth_cost + reveal_cost + funnel_cost)
        for ci, ps in zip(centers, passes):  # cross every gate forward, within GATE_MAX_ANGLE
            if fixed[ci - 1] and fixed[ci + 1]:
                continue  # straight gate piece: already along the normal
            tangent = P[:, ci + 1] - P[:, ci - 1]
            opti.subject_to(
                ca.dot(tangent, ca.DM(ps["n"]))
                >= np.cos(np.radians(GATE_MAX_ANGLE)) * ca.norm_2(tangent)
            )
        opti.subject_to(Z[2, airborne] >= FLOOR_Z)
        opti.subject_to(Z[2, :] <= hi[2])
        opti.subject_to(opti.bounded(lo[0], Z[0, :], hi[0]))
        opti.subject_to(opti.bounded(lo[1], Z[1, :], hi[1]))
        opti.set_initial(Z, guess.T)
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"max_iter": 3000, "print_level": 0, "sb": "yes", "tol": 1e-6},
        )
        try:
            guess = np.asarray(opti.solve().value(Z)).T
        except RuntimeError:
            logger.warning(f"path optimization: {opti.stats().get('return_status')}")
            guess = np.asarray(opti.debug.value(Z)).T
    pts = P0.copy()
    pts[free] = guess
    return pts, centers


def _speed_profile(
    pts: NDArray,
    centers: list[int],
    ds: float = 0.01,
    late: list[bool] | None = None,
    a_thrust: float | None = None,
) -> dict:
    """Arc-length resampling and a curvature/acceleration-limited minimum-time speed profile."""
    d = np.maximum(np.linalg.norm(np.diff(pts, axis=0), axis=1), 1e-6)
    u_pts = np.concatenate([[0.0], np.cumsum(d)])
    sp = CubicSpline(u_pts, pts, bc_type="natural")
    u_f = np.linspace(0.0, u_pts[-1], int(u_pts[-1] / 0.002) + 2)
    P_f = sp(u_f)
    s_f = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P_f, axis=0), axis=1))])
    s = np.arange(0.0, s_f[-1], ds)
    P = sp(np.interp(s, s_f, u_f))
    T = np.gradient(P, ds, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    K = np.gradient(T, ds, axis=0)  # curvature vector (|K| = curvature)
    kappa = np.linalg.norm(K, axis=1)
    w = max(int(KAPPA_WIN / ds), 1)
    kappa_f = np.array([kappa[max(i - w, 0) : i + w + 1].max() for i in range(len(kappa))])
    v = np.minimum(V_MAX, np.sqrt(A_LAT / np.maximum(kappa_f, 1e-6)))
    # thrust limit across the path: |v^2 K + g_perp| <= A_THRUST (g_perp: gravity normal to path)
    g_vec = np.array([0.0, 0.0, G])
    g_perp = g_vec - (T @ g_vec)[:, None] * T
    kq = np.sum(K * g_perp, axis=1)
    kk = np.maximum(np.sum(K * K, axis=1), 1e-9)
    a_th = A_THRUST if a_thrust is None else a_thrust
    disc = np.maximum(kq**2 - kk * (np.sum(g_perp**2, axis=1) - a_th**2), 0.0)
    v2_max = np.maximum((-kq + np.sqrt(disc)) / kk, 0.0)
    v2_max = np.array([v2_max[max(i - w, 0) : i + w + 1].min() for i in range(len(v2_max))])
    v = np.minimum(v, np.sqrt(v2_max))
    K_s = gaussian_filter1d(
        K, sigma=0.06 / ds, axis=0, mode="nearest"
    )  # spline knots: smooth first
    dK = np.linalg.norm(np.gradient(K_s, ds, axis=0), axis=1)  # change of the curvature vector
    dK_f = np.array([dK[max(i - w, 0) : i + w + 1].max() for i in range(len(dK))])
    v = np.minimum(v, np.cbrt(J_LAT / np.maximum(dK_f, 1e-6)))
    s_gate = np.interp(u_pts[centers], u_f, s_f)
    for j, sc in enumerate(s_gate):
        if GATE_APPROACH_PASSES is not None:  # explicit list of passes overrides the late flags
            if j not in GATE_APPROACH_PASSES:
                continue
        elif late is not None and not late[j]:
            continue
        v[(s >= sc - GATE_APPROACH_D) & (s <= sc)] = np.minimum(
            v[(s >= sc - GATE_APPROACH_D) & (s <= sc)], GATE_APPROACH_V
        )
    v[0] = 0.0

    def along(i: int, vi: float, sign: int) -> float:
        """Max acceleration (sign=+1) or deceleration (sign=-1) along the path.

        At point i and speed vi such that |a_t T + v^2 K + g| <= the thrust limit.
        """
        b = vi**2 * K[i] + g_vec
        tb = T[i] @ b
        root = np.sqrt(max(tb**2 - b @ b + a_th**2, 0.0))
        return max(-tb + root if sign > 0 else tb + root, 0.0)

    for i in range(len(s) - 1):  # forward: acceleration within what the turn and gravity leave
        a_t = min(A_ACC, along(i, v[i], +1))
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_t * ds))
    for i in range(len(s) - 2, -1, -1):  # backward: deceleration
        a_t = min(A_DEC, along(i + 1, v[i + 1], -1))
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_t * ds))
    v = np.maximum(v, 1e-3)
    t = np.concatenate([[0.0], np.cumsum(2 * ds / (v[:-1] + v[1:]))])
    acc = np.gradient(v, t)[:, None] * T + (v**2)[:, None] * K
    s_cross = s_gate
    return {
        "t": t,
        "s": s,
        "pos": P,
        "vel": v[:, None] * T,
        "acc": acc,
        "speed": v,
        "t_cross": np.interp(s_cross, s, t),
    }


def _min_time_profile(
    start: NDArray,
    passes: list[dict],
    poles_xy: NDArray,
    spheres: tuple,
    limits: tuple,
    warm: dict,
    a_max: float,
    steps: list | None = None,
) -> dict:
    """Minimum-time point-mass trajectory through the gate passes (CasADi + IPOPT).

    States position, velocity, acceleration; input jerk (constant per interval, exact integration).
    Constraints: |a + g| <= a_max and its tilt <= MT_TILT, |jerk| <= MT_JERK, each crossing inside
    MT_WIN of the gate centre within MT_ANGLE of the normal, pole clearance, gate-frame spheres (the
    pairs near the current solution, updated each round), floor/ceiling/xy limits. Solved as a
    continuation from loose to the requested limits, warm-started from ``warm`` (same topology).
    Returns a densely sampled profile like _speed_profile.
    """
    S, N = len(passes), MT_N
    K = S * N
    gz = ca.DM([0.0, 0.0, G])
    centers_s, radii_s, gid_s = spheres
    poles = np.asarray(poles_xy, float).reshape(-1, 2)
    pole_r = POLE_CLEARANCE_PLAN if MT_POLE is None else MT_POLE
    t_b = np.concatenate([[0.0], np.asarray(warm["t_cross"], float)])
    tg = np.concatenate(
        [np.linspace(t_b[j], t_b[j + 1], N + 1)[:-1] for j in range(S)] + [[t_b[-1]]]
    )
    xg = np.vstack(
        [
            np.stack([np.interp(tg, warm["t"], warm[key][:, i]) for i in range(3)])
            for key in ("pos", "vel", "acc")
        ]
    )
    xg[6:9] = gaussian_filter1d(xg[6:9], 2.0, axis=1, mode="nearest")
    p_anchor = xg[0:3].copy()  # trust region centre: the warm start's route
    sol = {
        "X": xg,
        "U": np.diff(xg[6:9], axis=1) / np.maximum(np.diff(tg), 1e-3)[None],
        "T": np.diff(t_b),
        "tg": tg,
        "ok": False,
    }
    if steps is None:
        # the warm start (profile plan) already crosses at the gate centres and keeps the
        # random-pose margin from other gates' frames: impose those from the start, then the
        # crossing-angle and jerk limits
        steps = [
            (MT_WIN, max(MT_ANGLE, 60.0), 0.0, MT_FRAME_RAND),
            (MT_WIN, MT_ANGLE, MT_JERK, MT_FRAME_RAND),
        ]
    for win, ang, jmax, frame_rand in steps:
        active = np.zeros((len(centers_s), K + 1), bool)
        for _ in range(2):  # relevance rounds for the frame spheres
            near = (
                np.linalg.norm(sol["X"][0:3].T[None] - centers_s[:, None], axis=2)
                < radii_s[:, None] + MT_FRAME + frame_rand + 0.3
            )
            active |= near
            opti = ca.Opti()
            T, X, U = opti.variable(S), opti.variable(9, K + 1), opti.variable(3, K)
            for j in range(S):
                dt, k0, k1 = T[j] / N, j * N, (j + 1) * N
                p, v, a, jk = X[0:3, k0:k1], X[3:6, k0:k1], X[6:9, k0:k1], U[:, k0:k1]
                opti.subject_to(
                    X[0:3, k0 + 1 : k1 + 1] == p + v * dt + a * dt**2 / 2 + jk * dt**3 / 6
                )
                opti.subject_to(X[3:6, k0 + 1 : k1 + 1] == v + a * dt + jk * dt**2 / 2)
                opti.subject_to(X[6:9, k0 + 1 : k1 + 1] == a + jk * dt)
                ps = passes[j]
                rel = X[0:3, k1] - ca.DM(ps["c"])
                opti.subject_to(ca.dot(ca.DM(ps["n"]), rel) == 0)
                opti.subject_to(opti.bounded(-win, ca.dot(ca.DM(ps["l"]), rel), win))
                opti.subject_to(
                    opti.bounded(MT_Z_AIM - win, ca.dot(ca.DM(ps["u"]), rel), MT_Z_AIM + win)
                )
                vk = X[3:6, k1]
                opti.subject_to(
                    ca.dot(vk, ca.DM(ps["n"]))
                    >= np.cos(np.radians(ang)) * ca.sqrt(ca.sumsqr(vk) + 1e-4)
                )
                if MT_SPEED_CAP and str(j) in {str(key) for key in MT_SPEED_CAP}:
                    cap = {str(key): val for key, val in MT_SPEED_CAP.items()}[str(j)]
                    k_cap = k1 - int(round(MT_CAP_FRAC * N))  # the last part of the approach
                    opti.subject_to(ca.sum1(X[3:6, k_cap : k1 + 1] ** 2) <= cap**2)
                opti.subject_to(opti.bounded(0.1, T[j], 4.0))
            f = X[6:9, :] + ca.repmat(gz, 1, K + 1)
            a_lim = np.full(K + 1, a_max)
            if MT_GATE_THRUST is not None and MT_SPEED_CAP:
                for key in MT_SPEED_CAP:
                    k1j = (int(key) + 1) * N
                    a_lim[k1j - int(round(MT_CAP_FRAC * N)) : k1j + 1] = (
                        a_max * MT_GATE_THRUST / MT_THRUST
                    )
            opti.subject_to(ca.sum1(f**2).T <= ca.DM(a_lim**2))
            opti.subject_to(f[2, :] >= np.cos(np.radians(MT_TILT)) * ca.sqrt(ca.sum1(f**2) + 1e-6))
            if jmax > 0:
                opti.subject_to(ca.sum1(U**2) <= jmax**2)
            opti.subject_to(X[:, 0] == ca.DM([*np.asarray(start, float), 0, 0, 0, 0, 0, 0]))
            for o in poles:
                opti.subject_to((X[0, :] - o[0]) ** 2 + (X[1, :] - o[1]) ** 2 >= pole_r**2)
            airborne = np.nonzero(sol["tg"] >= 0.4)[0].tolist()  # after the take-off climb
            opti.subject_to(X[2, airborne] >= MT_FLOOR)
            opti.subject_to(X[2, :] <= limits[1][2])
            if MT_TRUST:
                opti.subject_to(ca.sum1((X[0:3, :] - ca.DM(p_anchor)) ** 2) <= MT_TRUST**2)
            opti.subject_to(opti.bounded(limits[0][0], X[0, :], limits[1][0]))
            opti.subject_to(opti.bounded(limits[0][1], X[1, :], limits[1][1]))
            # nodes within 0.8 m (path length) of a crossing of gate g: no random margin from g
            s_node = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(sol["X"][0:3], axis=1), axis=0))]
            )
            s_cross = s_node[[(j + 1) * N for j in range(S)]]
            for i_s, (c, r, g) in enumerate(zip(centers_s, radii_s, gid_s)):
                idx = np.nonzero(active[i_s])[0]
                if not len(idx):
                    continue
                own = np.array(
                    [
                        any(
                            abs(s_node[i] - s_cross[j]) < 0.8
                            for j, ps in enumerate(passes)
                            if ps["gid"] == g
                        )
                        for i in idx
                    ]
                )
                rad = r + MT_FRAME + np.where(own, 0.0, frame_rand)
                opti.subject_to(ca.sum1((X[0:3, idx.tolist()] - ca.DM(c)) ** 2).T >= ca.DM(rad**2))
            opti.set_initial(X, sol["X"])
            opti.set_initial(U, sol["U"])
            opti.set_initial(T, sol["T"])
            opti.minimize(ca.sum1(T) + 1e-6 * ca.sumsqr(U) / N)
            opti.solver(
                "ipopt",
                {"expand": True, "print_time": False},
                {"max_iter": 1500, "print_level": 0, "sb": "yes", "tol": 1e-6},
            )
            try:
                res = opti.solve()
            except RuntimeError:
                logger.warning(
                    f"min-time plan: {opti.stats().get('return_status')} (keeping last solution)"
                )
                break
            Tv = np.asarray(res.value(T)).ravel()
            sol = {
                "X": np.asarray(res.value(X)),
                "U": np.asarray(res.value(U)),
                "T": Tv,
                "ok": True,
                "tg": np.concatenate(
                    [np.cumsum(np.r_[0, Tv])[j] + np.arange(N) * Tv[j] / N for j in range(S)]
                    + [[Tv.sum()]]
                ),
            }
    # dense, exact sampling of the piecewise-cubic trajectory
    Xs, Us, Tv = sol["X"], sol["U"], sol["T"]
    ts, P, V, A = [], [], [], []
    for k in range(K):
        dt = Tv[k // N] / N
        tau = np.arange(0.0, dt - 1e-9, 0.002)[:, None]
        p, v, a, jk = Xs[0:3, k], Xs[3:6, k], Xs[6:9, k], Us[:, k]
        ts.append(sol["tg"][k] + tau[:, 0])
        P.append(p + v * tau + a * tau**2 / 2 + jk * tau**3 / 6)
        V.append(v + a * tau + jk * tau**2 / 2)
        A.append(a + jk * tau)
    t = np.concatenate(ts + [[sol["tg"][-1]]])
    P = np.vstack(P + [Xs[0:3, -1:].T])
    V = np.vstack(V + [Xs[3:6, -1:].T])
    A = np.vstack(A + [Xs[6:9, -1:].T])
    s_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return {
        "t": t,
        "s": s_arc,
        "pos": P,
        "vel": V,
        "acc": A,
        "speed": np.linalg.norm(V, axis=1),
        "t_cross": np.cumsum(Tv),
        "ok": sol["ok"],
        "T": Tv,
        "X": Xs,
        "U": Us,
    }


def _time_scaled(prof: dict, mass: float, mass0: float = MASS) -> dict:
    """The same path, its timing stretched (lam > 1) or compressed for another drone mass.

    lam is chosen so that this mass needs the same peak thrust as mass0 on the original profile
    (gravity does not scale with the timing: bisection on lam).
    """
    acc, vel = prof["acc"], prof["vel"]

    def peak(m: float, lam: float) -> float:
        f = m * (acc / lam**2 + np.array([0.0, 0.0, G])) - DRAG * vel / lam
        return float(np.max(np.linalg.norm(f, axis=1)))

    target, lo, hi = peak(mass0, 1.0), 0.5, 2.0
    for _ in range(40):
        lam = 0.5 * (lo + hi)
        lo, hi = (lam, hi) if peak(mass, lam) > target else (lo, lam)
    lam = hi
    return {
        **prof,
        "t": prof["t"] * lam,
        "vel": vel / lam,
        "acc": acc / lam**2,
        "speed": prof["speed"] / lam,
        "t_cross": np.asarray(prof["t_cross"]) * lam,
        "lam": lam,
    }


def _build_reference(
    prof: dict, passes: list[dict], yaw0: float, extend: float = 1.5, mass: float = MASS
) -> dict:
    """Reference on a REF_DT time grid: position, velocity, attitude and input feedforward."""
    t_p, t_cross = prof["t"], prof["t_cross"]
    t_end = float(t_cross[-1])
    t = np.arange(0.0, t_end + extend, REF_DT)
    pos = np.stack([np.interp(t, t_p, prof["pos"][:, i]) for i in range(3)], axis=1)
    vel = np.stack([np.interp(t, t_p, prof["vel"][:, i]) for i in range(3)], axis=1)
    acc = np.stack([np.interp(t, t_p, prof["acc"][:, i]) for i in range(3)], axis=1)
    i_end = int(np.searchsorted(t_p, t_end))
    after = t > t_end  # past the finish: keep flying straight at the final velocity
    pos[after] = prof["pos"][i_end] + prof["vel"][i_end] * (t[after] - t_end)[:, None]
    vel[after], acc[after] = prof["vel"][i_end], 0.0
    s_arc = np.interp(t, t_p, prof["s"])
    s_arc[after] = prof["s"][i_end] + prof["speed"][i_end] * (t[after] - t_end)
    # yaw schedule: each gate's yaw at its crossing, smooth blend in between
    yaw_t = _yaw_targets(passes, yaw0)
    seg = np.minimum(np.searchsorted(t_cross, t, side="right"), len(passes) - 1)
    t_start = np.concatenate([[0.0], t_cross[:-1]])[seg]
    frac = np.clip((t - t_start) / np.maximum(t_cross[seg] - t_start, 1e-6), 0, 1)
    b = np.clip((frac - 0.15) / 0.7, 0, 1)
    b = b * b * (3 - 2 * b)
    yaw_prev = np.array([yaw0] + yaw_t[:-1])[seg]
    yaw = yaw_prev + (np.array(yaw_t)[seg] - yaw_prev) * b
    yaw[after] = yaw_t[-1]
    f = mass * (acc + np.array([0.0, 0.0, G])) - DRAG * vel  # thrust vector incl. drag
    rpy = _rpy_from_thrust(f, yaw)
    thrust = np.clip(np.linalg.norm(f, axis=1), THRUST_MIN, MPC_THRUST_MAX)
    u = np.column_stack([np.clip(rpy[:, 0:2], -MPC_CMD_MAX, MPC_CMD_MAX), yaw, thrust])
    return {
        "t": t,
        "s": s_arc,
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "rpy": rpy,
        "u": u,
        "t_cross": t_cross,
        "gids": np.array([ps["gid"] for ps in passes]),
        "lap": t_end,
    }


_PLANS: dict = {}


def _code_dir() -> Path:
    d = Path(
        os.environ.get("TOMPC_CACHE_DIR", Path.cwd() / "c_generated_code" / "time_optimal_mpc")
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Inter-process lock (POSIX); a no-op where fcntl is unavailable."""
    try:
        import fcntl
    except ImportError:  # Windows
        yield
        return
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def plan_reference(
    start: NDArray,
    yaw0: float,
    gates_pos: NDArray,
    gates_quat: NDArray,
    sequence: NDArray,
    direction: NDArray,
    poles_xy: NDArray,
    limits: tuple,
) -> dict:
    """Plan (or load the cached plan for) the reference through the gate sequence."""
    track = [
        np.round(np.asarray(a, float), 3).tolist()
        for a in (
            start,
            [yaw0],
            gates_pos,
            gates_quat,
            sequence,
            direction,
            poles_xy,
            limits[0],
            limits[1],
        )
    ]
    geometry = [
        FRAME_RAND_MARGIN,
        GATE_STRAIGHT,
        GATE_STRAIGHT_OUT,
        GATE_STRAIGHT_MIN,
        FUNNEL_LEN,
        FUNNEL_W0,
        FUNNEL_SLOPE,
        FUNNEL_SLOPE_Z,
        FUNNEL_NO_UNDERSHOOT,
        GATE_MAX_ANGLE,
        REVEAL_EARLY,
        REVEAL_RADIUS,
        REVEAL_DETOUR,
        PATH_H,
        PATH_Z_WEIGHT,
        PATH_STARTS,
        PATH_LEN_WEIGHT,
        POLE_CLEARANCE_PLAN,
        FRAME_CLEARANCE,
        FLOOR_Z,
        TAKEOFF_DIST,
        _VERSION,
    ]
    speed = [
        V_MAX,
        A_LAT,
        A_TOT,
        A_THRUST,
        A_ACC,
        A_DEC,
        KAPPA_WIN,
        J_LAT,
        GATE_APPROACH_V,
        GATE_APPROACH_D,
        GATE_APPROACH_PASSES,
        MASS_ADAPT,
        THRUST_BUDGET,
        MASS_GRID,
        MPC_THRUST_MAX,
        MPC_CMD_MAX,
        YAW_ALIGN,
        REF_DT,
        PLANNER,
        MT_N,
        MT_THRUST,
        MT_TILT,
        MT_JERK,
        MT_WIN,
        MT_ANGLE,
        MT_FRAME,
        MT_FRAME_RAND,
        MT_POLE,
        MT_FLOOR,
        str(MT_SPEED_CAP),
        MT_CAP_FRAC,
        MT_TRUST,
        MT_GATE_THRUST,
        MT_Z_AIM,
    ]
    path_key = hashlib.sha1(json.dumps(track + geometry + speed).encode()).hexdigest()[:16]
    key = hashlib.sha1(json.dumps([path_key] + speed).encode()).hexdigest()[:16]
    if key in _PLANS:
        return _PLANS[key]
    passes = _gate_passes(gates_pos, gates_quat, sequence, direction)
    path = _code_dir() / f"path_{path_key}.npz"
    with _file_lock(_code_dir() / "plan.lock"):
        if path.exists():
            data = np.load(path)
            pts, centers = data["pts"], data["centers"].tolist()
        else:
            spheres = _gate_spheres(gates_pos, gates_quat)
            best = None
            for scale in PATH_STARTS:  # multi-start: keep the path with the fastest reference
                cand = _optimize_path(
                    np.asarray(start, float),
                    passes,
                    np.asarray(poles_xy, float),
                    spheres,
                    limits,
                    scale,
                )
                lap = _speed_profile(*cand)["t_cross"][-1]
                if best is None or lap < best[0]:
                    best = (lap, cand)
            pts, centers = best[1]
            np.savez(path, pts=pts, centers=np.array(centers))
    # passes that see their gate only late (no path point within ~0.5 m more than 1 m before)
    s_pts = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    late, seen = [], set()
    for ci, ps in zip(centers, passes):
        early = s_pts < s_pts[ci] - 1.0
        d = np.linalg.norm(pts[early, :2] - ps["c"][:2], axis=1) if early.any() else np.array([9.0])
        late.append(ps["gid"] not in seen and d.min() > REVEAL_RADIUS + 0.05)
        seen.add(ps["gid"])
    if PLANNER == "mintime":
        mt = [
            PLANNER,
            MT_N,
            MT_THRUST,
            MT_TILT,
            MT_JERK,
            MT_WIN,
            MT_ANGLE,
            MT_FRAME,
            MT_FRAME_RAND,
            MT_POLE,
            MT_FLOOR,
            MT_SPEED_CAP,
            MT_CAP_FRAC,
            MT_TRUST,
            MT_GATE_THRUST,
            MT_Z_AIM,
            _VERSION,
        ]
        mt_key = hashlib.sha1(json.dumps([path_key, mt], default=str).encode()).hexdigest()[:16]
        file = _code_dir() / f"mintime_{mt_key}.npz"
        spheres = _gate_spheres(gates_pos, gates_quat)
        masses = list(MASS_GRID) if MASS_ADAPT else [MASS]
        with _file_lock(_code_dir() / "plan.lock"):
            if file.exists():
                data = np.load(file, allow_pickle=True)
                profs = data["profs"].item()
            else:
                warm = _speed_profile(pts, centers, late=late)
                nominal = _min_time_profile(
                    start, passes, poles_xy, spheres, limits, warm, MT_THRUST / MASS
                )
                # other masses: the same path, time-scaled to the same peak thrust (no jump when
                # the controller switches to the estimated mass's profile)
                profs = {m: _time_scaled(nominal, m) for m in masses}
                np.savez(file, profs=np.array(profs, dtype=object))
        refs = {m: _build_reference(profs[m], passes, yaw0, mass=m) for m in masses}
        m0 = min(masses, key=lambda m: abs(m - MASS))
        ref = {**refs[m0], "late": late}
        if MASS_ADAPT:
            ref["variants"] = refs
        logger.info(f"time_optimal_mpc: min-time reference lap {ref['lap']:.3f} s")
        _PLANS[key] = ref
        return ref
    ref = _build_reference(_speed_profile(pts, centers, late=late), passes, yaw0)
    ref["late"] = late
    if MASS_ADAPT:  # same path, one speed profile per drone mass (thrust limit THRUST_BUDGET / m)
        ref["variants"] = {
            m: _build_reference(
                _speed_profile(pts, centers, late=late, a_thrust=THRUST_BUDGET / m),
                passes,
                yaw0,
                mass=m,
            )
            for m in MASS_GRID
        }
        ref = {
            **ref["variants"][min(MASS_GRID, key=lambda m: abs(m - MASS))],
            "variants": ref["variants"],
            "late": late,
        }
    logger.info(f"time_optimal_mpc: reference lap {ref['lap']:.3f} s")
    _PLANS[key] = ref
    return ref


# ------------------------------------------------------------------------------------------------
# MPC
# ------------------------------------------------------------------------------------------------
_SOLVERS: dict = {}


def _get_solver(N: int, dt: float) -> AcadosOcpSolver:
    """Build (or load) the acados SQP-RTI solver. Built once per process and cached on disk."""
    settings = [
        N,
        dt,
        MPC_THRUST_MAX,
        MPC_CMD_MAX,
        MPC_TILT_MAX,
        POLE_CLEARANCE_MPC,
        W_POS,
        W_VEL,
        W_ATT,
        W_INPUT,
        W_RATE,
        W_TERMINAL,
        POLE_SLACK_WEIGHTS,
        TAU_D,
        TAU_T,
        MPC_INT_STEPS,
        MODEL_SMOOTH,
        MPC_LM,
        MPC_SQP_ITERS,
        MPC_TOL,
        GATE_TUNNEL,
        GATE_FUNNEL_SLOPE,
        SLACK_WEIGHTS,
        _VERSION,
    ]
    key = hashlib.sha1(json.dumps(settings).encode()).hexdigest()[:10]
    if key in _SOLVERS:
        return _SOLVERS[key]
    name = f"tompc_{key}"
    model = AcadosModel()
    model.name = name
    x, u, p = ca.SX.sym("x", NX), ca.SX.sym("u", NU), ca.SX.sym("p", NP)
    model.x, model.u, model.p = x, u, p
    model.f_expl_expr = model_dynamics(x, u, p)
    poles = [(x[0] - p[5 + 2 * i]) ** 2 + (x[1] - p[6 + 2 * i]) ** 2 for i in range(MAX_POLES)]
    tilt = ca.cos(x[6]) * ca.cos(x[7])  # cosine of the actual tilt of the thrust axis
    rel = x[0:3] - p[GATE_P : GATE_P + 3]
    lat_ax, up_ax = p[GATE_P + 3 : GATE_P + 6], p[GATE_P + 6 : GATE_P + 9]
    gate_lat, gate_up = ca.dot(rel, lat_ax), ca.dot(rel, up_ax)  # offsets from the gate centre line
    dist = ca.sqrt(ca.dot(rel, ca.cross(lat_ax, up_ax)) ** 2 + 1e-4)  # |distance to the gate plane|
    open_ = GATE_FUNNEL_SLOPE * dist
    model.con_h_expr = ca.vertcat(
        *poles, tilt, gate_lat - open_, -gate_lat - open_, gate_up - open_, -gate_up - open_
    )
    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt
    # y = [pos, vel, rpy, roll_cmd, pitch_cmd, thrust_cmd, roll_cmd - lagged, pitch_cmd - lagged,
    #      thrust_cmd - thrust]; the last three penalize fast command changes.
    ny, ny_e = 15, 9
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    Vx = np.zeros((ny, NX))
    Vx[0:9, 0:9] = np.eye(9)
    Vx[12, 16], Vx[13, 17], Vx[14, 18] = -1.0, -1.0, -1.0
    Vu = np.zeros((ny, NU))
    Vu[9:12, :] = np.eye(3)
    Vu[12:15, :] = np.eye(3)
    ocp.cost.Vx, ocp.cost.Vu = Vx, Vu
    ocp.cost.Vx_e = np.eye(ny_e, NX)
    W = np.diag([*W_POS, W_VEL, W_VEL, W_VEL, *W_ATT, *W_INPUT, *W_RATE])
    ocp.cost.W, ocp.cost.W_e = W, W[:ny_e, :ny_e] * W_TERMINAL
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(ny), np.zeros(ny_e)
    ocp.constraints.lbu = np.array([-MPC_CMD_MAX, -MPC_CMD_MAX, THRUST_MIN])
    ocp.constraints.ubu = np.array([MPC_CMD_MAX, MPC_CMD_MAX, MPC_THRUST_MAX])
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.x0 = np.zeros(NX)
    # soft bounds on the predicted roll/pitch (keeps predictions away from the Euler singularity)
    ocp.constraints.idxbx = np.array([6, 7])
    ocp.constraints.lbx, ocp.constraints.ubx = np.full(2, -1.3), np.full(2, 1.3)
    ocp.constraints.idxsbx = np.arange(2)
    # soft pole clearance (horizontal distance to each pole axis) and soft tilt limit
    # (gate offsets: wide open by default, tightened at runtime around each crossing)
    ocp.constraints.lh = np.r_[
        np.full(MAX_POLES, POLE_CLEARANCE_MPC**2), np.cos(MPC_TILT_MAX), np.full(4, -1e4)
    ]
    ocp.constraints.uh = np.r_[np.full(MAX_POLES, 1e8), 2.0, np.full(4, 1e3)]
    ocp.constraints.idxsh = np.arange(MAX_POLES + 5)
    for attr, i in (
        ("zl", 0),
        ("zu", 0),
        ("Zl", 1),
        ("Zu", 1),
    ):  # slacks: bx(2), poles, tilt, gate(4)
        setattr(
            ocp.cost,
            attr,
            np.r_[
                np.full(2, SLACK_WEIGHTS[i]),
                np.full(MAX_POLES, POLE_SLACK_WEIGHTS[i]),
                np.full(5, SLACK_WEIGHTS[i]),
            ],
        )
    ocp.parameter_values = _params(np.zeros(3), 0.0, np.zeros((0, 2)), None)
    opts = ocp.solver_options
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.hessian_approx = "GAUSS_NEWTON"
    opts.integrator_type = "ERK"
    opts.sim_method_num_stages = 4
    opts.sim_method_num_steps = MPC_INT_STEPS
    opts.nlp_solver_type = "SQP_RTI" if MPC_SQP_ITERS <= 1 else "SQP"
    if MPC_SQP_ITERS > 1:
        opts.nlp_solver_max_iter = MPC_SQP_ITERS
        opts.globalization = "MERIT_BACKTRACKING"
        if MPC_TOL is not None:  # stop early once converged
            opts.nlp_solver_tol_stat, opts.nlp_solver_tol_eq = MPC_TOL, MPC_TOL * 1e-2
            opts.nlp_solver_tol_ineq, opts.nlp_solver_tol_comp = MPC_TOL * 1e-2, MPC_TOL * 1e-2
    opts.qp_solver_iter_max = 50
    opts.qp_solver_warm_start = 1
    opts.levenberg_marquardt = MPC_LM
    out = _code_dir() / name
    ocp.code_export_directory = str(out)
    json_file = str(_code_dir() / f"{name}.json")
    builder = None
    if os.name == "nt":  # On Windows, use CMake with MinGW to build the solver DLL.
        builder = ocp_get_default_cmake_builder()
        builder.generator = "MinGW Makefiles"
        builder.build_dir = "build-mingw"
    with _file_lock(_code_dir() / f"{name}.lock"):
        built = (out / "build_done").exists()
        solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
            build=not built,
            generate=not built,
            verbose=False,
            cmake_builder=builder,
        )
        (out / "build_done").touch()
    _SOLVERS[key] = solver
    return solver


def _params(
    dist: NDArray, yaw_cmd: float, poles_xy: NDArray, gate: tuple | None, mass: float = MASS
) -> NDArray:
    p = np.zeros(NP)
    p[0], p[1:4], p[4] = 1.0 / mass, dist, yaw_cmd
    poles = np.full((MAX_POLES, 2), 50.0)  # unused slots: far away
    poles[: len(poles_xy)] = poles_xy[:MAX_POLES]
    p[5:GATE_P] = poles.ravel()
    if gate is None:
        p[GATE_P + 3], p[GATE_P + 8] = 1.0, 1.0
    else:  # (centre, rotation matrix): lateral axis = column 1, up axis = column 2
        p[GATE_P : GATE_P + 3], p[GATE_P + 3 : GATE_P + 6] = gate[0], gate[1][:, 1]
        p[GATE_P + 6 : GATE_P + 9] = gate[1][:, 2]
    return p


def _smoothstep(x: NDArray) -> NDArray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------------------------------------
# Controller
# ------------------------------------------------------------------------------------------------
class TimeOptimalMPC(Controller):
    """Minimum-time plan + white-box MPC with progress sync, gate re-planning and pole avoidance."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the reference (cached per track) and set up the MPC.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("TimeOptimalMPC needs env.control_mode = 'attitude'")
        self._dt = 1.0 / config.env.freq
        self._N = MPC_HORIZON
        # Plan from the nominal start if the drone is near it (shared, cached plan), else from obs
        start = np.asarray(obs["pos"], float)
        try:
            nominal = np.asarray(config.env.track.drones[0]["pos"], float)
            if np.linalg.norm(nominal[:2] - start[:2]) < 0.25:
                start = nominal
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        try:
            sl = config.env.track.safety_limits
            limits = (np.asarray(sl.pos_limit_low) + 0.1, np.asarray(sl.pos_limit_high) - 0.1)
        except AttributeError:
            limits = (np.array([-2.9, -2.9, 0.1]), np.array([2.9, 2.9, 2.4]))
        limits[0][2] = max(limits[0][2], 0.1)
        self._gates_nom_pos = np.array(obs["gates_pos"], float)
        self._gates_nom_quat = np.array(obs["gates_quat"], float)
        self._gates_obs_pos = self._gates_nom_pos.copy()
        self._gates_obs_quat = self._gates_nom_quat.copy()
        self._poles_nom = np.array(obs["obstacles_pos"], float)[:, :2]
        self._poles_obs = self._poles_nom.copy()
        rpy0 = R.from_quat(obs["quat"]).as_euler("xyz")
        plan = plan_reference(
            start,
            0.0,
            self._gates_nom_pos,
            self._gates_nom_quat,
            obs["gate_sequence"],
            obs["gate_sequence_direction"],
            np.asarray(obs["obstacles_pos"])[:, :2],
            limits,
        )
        self._variants = plan.get("variants")
        self._start_pos = np.asarray(obs["pos"], float)
        self._install_reference(plan)
        self._solver = _get_solver(self._N, self._dt)
        self._rk4 = _rk4_function(4)
        self._x_est = np.zeros(NX)
        self._x_est[6:9], self._x_est[16:18] = rpy0, rpy0[:2]  # rotors at rest (0 rpm) at the start
        self.solve_times: list[float] = []
        self.n_fail = 0  # MPC steps where the solver failed and the fallback controller was used
        self.episode_callback()

    # -------------------------------------------------------------------------------- reference
    def _install_reference(self, ref: dict):
        """Use ref, moved onto the actual (randomized) start position over the first metre."""
        s_path = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(ref["pos"], axis=0), axis=1))]
        )
        w0 = 1.0 - _smoothstep(s_path / START_BLEND)
        pos = ref["pos"] + w0[:, None] * (self._start_pos - ref["pos"][0])
        self._ref = {**ref, "pos": pos, "vel": np.gradient(pos, REF_DT, axis=0)}
        self._ref_pos, self._ref_vel = self._ref["pos"].copy(), self._ref["vel"].copy()
        self._force_track_update = True

    def _estimate_mass(self, vel: NDArray, quat: NDArray):
        """Least-squares fit of 1/m from thrust-axis acceleration = (thrust + body z drag) / m.

        Uses airborne steps MASS_EST_STEPS (the first steps after lift-off read heavy while motors
        are still spinning up), skips low-thrust samples, removes the remaining bias (MASS_EST_BIAS)
        and then switches to the speed profile of the estimated mass.
        """
        if self._prev_vel is None or self._prev_F is None or self._mass_locked:
            return
        k0, k1 = MASS_EST_STEPS
        F_cmd = 0.5 * (self._prev_F + rotor_thrust(self._x_pred[12:16]))
        if k0 <= self._airborne_steps < k1 and F_cmd > 0.4:
            acc = (vel - self._prev_vel) / self._dt + np.array([0.0, 0.0, G])
            zb = R.from_quat(quat).as_matrix()[:, 2]
            F = F_cmd + DRAG[2] * float(vel @ zb)
            self._mass_num += float(acc @ zb) * F
            self._mass_den += F * F
            self._mass_n += 1
        if self._airborne_steps >= k1 and self._mass_n >= 5:
            m = self._mass_den / max(self._mass_num, 1e-9) - MASS_EST_BIAS
            self._mass, self._mass_locked = float(np.clip(m, 0.036, 0.051)), True
            if self._variants:
                m_sel = min([g for g in MASS_GRID if g >= self._mass - 0.0005] or [max(MASS_GRID)])
                new = self._variants[m_sel]
                s_now = np.interp(self._t_ref, self._ref["t"], self._ref["s"])
                self._install_reference(new)
                self._t_ref = float(np.interp(s_now, new["s"], new["t"]))
                self._warm = MASS_SWITCH_WARM and self._warm

    def _ref_index(self, t: float | NDArray) -> NDArray:
        return np.clip(np.rint(np.asarray(t) / REF_DT).astype(int), 0, len(self._ref["t"]) - 1)

    def _sync_progress(self, pos: NDArray) -> float:
        """Reference time = time of the closest plan point near the previous reference time."""
        i0 = int(self._ref_index(self._t_ref))
        lo, hi = max(i0 - 3, 0), min(i0 + int(0.15 / REF_DT), len(self._ref["t"]) - 1)
        i = lo + int(np.argmin(np.linalg.norm(self._ref_pos[lo : hi + 1] - pos, axis=1)))
        return max(i, i0) * REF_DT

    def _update_track(self, obs: dict):
        """Deform the reference when the true pose of a gate or pole is revealed.

        Reference points near a revealed gate move rigidly with it (full within REVEAL_R1, fading
        out by REVEAL_R2), so the path keeps its clearance to that gate's frame on the approach,
        through the opening and on the way out. Then the reference is pushed horizontally away
        from revealed poles it would pass closer than POLE_PUSH.
        """
        gp, gq = np.asarray(obs["gates_pos"], float), np.asarray(obs["gates_quat"], float)
        op = np.asarray(obs["obstacles_pos"], float)[:, :2]
        if (
            np.allclose(gp, self._gates_obs_pos, atol=1e-4)
            and np.allclose(gq, self._gates_obs_quat, atol=1e-4)
            and np.allclose(op, self._poles_obs, atol=1e-4)
            and not self._force_track_update
        ):
            return
        self._force_track_update = False
        self._ref_changed = True
        self._gates_obs_pos, self._gates_obs_quat, self._poles_obs = gp.copy(), gq.copy(), op.copy()
        p0 = self._ref["pos"]
        disp, wsum = np.zeros_like(p0), np.zeros(len(p0))
        for g in range(len(gp) if GATE_DEFORM else 0):
            moved = (
                np.max(np.abs(gp[g] - self._gates_nom_pos[g])) > 1e-4
                or np.max(np.abs(gq[g] - self._gates_nom_quat[g])) > 1e-4
            )
            if not moved:
                continue
            r_nom = R.from_quat(self._gates_nom_quat[g]).as_matrix()
            shift = (
                gp[g] - self._gates_nom_pos[g]
            ) @ r_nom  # in the nominal gate frame (n, lat, up)
            band = np.array(
                [
                    DEFORM_DEADBAND,
                    DEFORM_DEADBAND if DEFORM_DEADBAND_Z is None else DEFORM_DEADBAND_Z,
                ]
            )
            shift[1:] = np.sign(shift[1:]) * np.maximum(np.abs(shift[1:]) - band, 0.0)
            if DEFORM_ROTATION:
                r_rel = (
                    R.from_quat(gq[g]) * R.from_quat(self._gates_nom_quat[g]).inv()
                ).as_matrix()
                delta = (
                    self._gates_nom_pos[g]
                    + r_nom @ shift
                    + (p0 - self._gates_nom_pos[g]) @ r_rel.T
                    - p0
                )
            else:
                delta = np.broadcast_to(r_nom @ shift, p0.shape)
            w = _smoothstep(
                (REVEAL_R2 - np.linalg.norm(p0 - self._gates_nom_pos[g], axis=1))
                / (REVEAL_R2 - REVEAL_R1)
            )
            disp += w[:, None] * delta
            wsum += w
        pos = p0 + disp / np.maximum(wsum, 1.0)[:, None]
        d_gate = np.min(np.linalg.norm(pos[:, None] - gp[None], axis=2), axis=1)
        keep = _smoothstep((d_gate - 0.35) / 0.25)  # 0 within 0.35 m of a gate, 1 beyond 0.6 m
        for o, o_nom in zip(op, self._poles_nom):
            if np.max(np.abs(o - o_nom)) < 1e-4:
                continue
            d = pos[:, :2] - o
            r = np.linalg.norm(d, axis=1)
            push = np.maximum(POLE_PUSH - r, 0.0)
            if not push.any():
                continue
            # smooth the push along the reference (moving max, then moving average over ~0.3 s)
            k = int(0.15 / REF_DT)
            push = np.array([push[max(i - k, 0) : i + k + 1].max() for i in range(len(push))])
            push = np.convolve(push, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
            pos[:, :2] += (keep * push)[:, None] * d / np.maximum(r[:, None], 1e-6)
        self._ref_pos = pos
        self._ref_vel = np.gradient(pos, REF_DT, axis=0)

    def _fallback(self, x: NDArray, i: int) -> NDArray:
        """Simple PD position controller on the reference: [roll_cmd, pitch_cmd, thrust]."""
        a = (
            self._ref["acc"][i]
            + FALLBACK_KP * (self._ref_pos[i] - x[0:3])
            + FALLBACK_KD * (self._ref_vel[i] - x[3:6])
        )
        f = MASS * (a + np.array([0.0, 0.0, G])) - DRAG * x[3:6]
        f[2] = max(f[2], 0.2 * MASS * G)
        tilt = np.arccos(np.clip(f[2] / np.linalg.norm(f), -1, 1))
        if tilt > MPC_CMD_MAX:  # limit the tilt, keep the vertical component
            f[:2] *= np.tan(MPC_CMD_MAX) * f[2] / np.linalg.norm(f[:2])
        rpy = _rpy_from_thrust(f[None], np.array([self._ref["u"][i, 2]]))[0]
        thrust = np.clip(np.linalg.norm(f), THRUST_MIN, MPC_THRUST_MAX)
        return np.array([rpy[0], rpy[1], thrust])

    # -------------------------------------------------------------------------------- control
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the attitude command [roll, pitch, yaw, thrust].

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The command [roll, pitch, yaw, thrust] as a numpy array.
        """
        pos, vel = np.asarray(obs["pos"], float), np.asarray(obs["vel"], float)
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        self._update_track(obs)
        # State estimate: measured pos/vel/attitude/rates + predicted torque, cmd lag and thrust
        x = self._x_est.copy() if self._x_pred is None else self._x_pred.copy()
        if self._airborne_steps > 10 and DIST_OBSERVER_GAIN > 0:  # skip takeoff transients
            self._dist += DIST_OBSERVER_GAIN * (vel - self._x_pred[3:6]) / self._dt
            self._dist = np.clip(self._dist, -DIST_MAX, DIST_MAX)
        x[0:3], x[3:6], x[6:9], x[9:12] = pos, vel, rpy, np.asarray(obs["ang_vel"], float)
        self._t_ref = self._sync_progress(pos)
        poles_xy = np.asarray(obs["obstacles_pos"], float)[:, :2]

        if self._z0 is None:
            self._z0 = float(pos[2])
        lifted = pos[2] >= self._z0 + TAKEOFF_DZ or (
            TAKEOFF_Z > 0 and pos[2] >= TAKEOFF_Z and vel[2] > 0.1
        )
        if self._airborne_steps == 0 and not lifted:
            # On the ground with the motors spinning up: climb level at full thrust. The model has
            # no ground contact, so the MPC would otherwise tilt early to "catch up" with the plan.
            yaw0 = float(self._ref["u"][0, 2])
            u = np.array([0.0, 0.0, TAKEOFF_THRUST])
            p0 = _params(self._dist, yaw0, poles_xy, None)
            self._x_pred = np.asarray(self._rk4(x, u, self._dt, p0[0:5])).ravel()
            return np.array([0.0, 0.0, yaw0, TAKEOFF_THRUST], dtype=np.float32)
        self._airborne_steps += 1
        if MASS_ADAPT:
            self._estimate_mass(vel, np.asarray(obs["quat"], float))
        self._prev_vel, self._prev_F = vel.copy(), rotor_thrust(self._x_pred[12:16])

        solver, N = self._solver, self._N
        idx = self._ref_index(self._t_ref + np.arange(N + 1) * self._dt)
        ref_rpy, ref_u4 = self._ref["rpy"][idx], self._ref["u"][idx]
        ref_u = ref_u4[:, [0, 1, 3]]
        yref = np.hstack(
            [self._ref_pos[idx], self._ref_vel[idx], ref_rpy, ref_u, np.zeros((N + 1, 3))]
        )
        # the gate (observed pose) whose crossing is within GATE_TUNNEL_T of each stage, if any
        t_k = self._t_ref + np.arange(N + 1) * self._dt
        rel_t = t_k[:, None] - self._ref["t_cross"][None, :]
        inside = (rel_t >= -GATE_TUNNEL_T[0]) & (rel_t <= GATE_TUNNEL_T[1])
        near = np.where(inside.any(axis=1), np.argmax(inside, axis=1), -1)
        gates = [
            None
            if j < 0
            else (
                self._gates_obs_pos[self._ref["gids"][j]],
                R.from_quat(self._gates_obs_quat[self._ref["gids"][j]]).as_matrix(),
            )
            for j in near
        ]
        params = [
            _params(self._dist, ref_u4[k, 2], poles_xy, gates[k], self._mass) for k in range(N + 1)
        ]
        lh0 = np.r_[np.full(MAX_POLES, POLE_CLEARANCE_MPC**2), np.cos(MPC_TILT_MAX)]
        uh0 = np.r_[np.full(MAX_POLES, 1e8), 2.0]
        for k in range(1, N):
            w_lat, w_up, w_down = GATE_TUNNEL if gates[k] is not None else (1e3, 1e3, 1e3)
            solver.constraints_set(k, "lh", np.r_[lh0, np.full(4, -1e4)])
            solver.constraints_set(k, "uh", np.r_[uh0, w_lat, w_lat, w_up, w_down])
        changed, self._ref_changed = self._ref_changed, False
        if changed and REVEAL_REWARM:
            self._warm = False
        if not self._warm:  # initialize the horizon by simulating the fallback controller
            xk = x.copy()
            for k in range(N):
                uk = self._fallback(xk, idx[k])
                solver.set(k, "x", xk)
                solver.set(k, "u", uk)
                xk = np.asarray(self._rk4(xk, uk, self._dt, params[k][0:5])).ravel()
            solver.set(N, "x", xk)
            self._warm = True
        elif self._sol_x is not None:  # warm start: previous solution shifted by one step
            for k in range(N):
                solver.set(k, "x", self._sol_x[k + 1])
                solver.set(k, "u", self._sol_u[min(k + 1, N - 1)])
            solver.set(N, "x", self._sol_x[N])
        for k in range(N):
            solver.set(k, "yref", yref[k])
            solver.set(k, "p", params[k])
        solver.set(N, "yref", yref[N, 0:9])
        solver.set(N, "p", params[N])
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        status = solver.solve()
        t_solve = solver.get_stats("time_tot")
        for _ in range(MPC_EXTRA_SOLVES if changed else 0):
            if status != 2:
                break
            status = solver.solve()
            t_solve += solver.get_stats("time_tot")
        self.last_status = status
        self.solve_times.append(t_solve)
        u = solver.get(0, "u")
        if status not in (0, 2) or not np.all(np.isfinite(u)):
            logger.debug(f"acados status {status}, using the fallback controller")
            self.n_fail += 1
            u = self._fallback(x, idx[0])
            self._warm = False
        u = np.clip(
            u, [-MPC_CMD_MAX, -MPC_CMD_MAX, THRUST_MIN], [MPC_CMD_MAX, MPC_CMD_MAX, MPC_THRUST_MAX]
        )
        self._sol_x = np.array([solver.get(k, "x") for k in range(N + 1)])
        self._sol_u = np.array([solver.get(k, "u") for k in range(N)])
        self._pred_pos = self._sol_x[::2, 0:3]
        self._x_pred = np.asarray(self._rk4(x, u, self._dt, params[0][0:5])).ravel()
        return np.array([u[0], u[1], ref_u4[0, 2], u[2]], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Never finish early; the environment ends the episode after the last gate."""
        return False

    def episode_callback(self):
        """Reset the per-episode state."""
        self._t_ref, self._dist, self._x_pred, self._warm = 0.0, np.zeros(3), None, False
        self._airborne_steps = 0  # steps since lift-off (0 = still on the ground)
        self._pred_pos = None
        self._z0 = None  # height at the start (randomized)
        self._mass, self._mass_num, self._mass_den, self._mass_n = MASS, 0.0, 0.0, 0
        self._mass_locked, self._prev_vel, self._prev_F = False, None, None
        self._sol_x, self._sol_u = None, None
        self._ref_changed = False

    def render_callback(self, sim: Sim):
        """Draw the (re-planned) reference and the MPC prediction."""
        try:
            from crazyflow.sim.visualize import draw_line

            draw_line(sim, self._ref_pos[::4], rgba=np.array([0.2, 0.5, 0.9, 1.0]))
            if self._pred_pos is not None:
                draw_line(sim, self._pred_pos, rgba=np.array([1.0, 0.3, 0.1, 1.0]))
        except Exception:  # noqa: BLE001 (visualization only)
            pass
