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
MPC_THRUST_MAX = 0.75  # N. Above ~0.7 N the motors have little headroom left for torque.
MPC_CMD_MAX = 1.1  # rad. Bound of the roll/pitch commands. Values above MPC_TILT_MAX let the MPC
#                    overdrive the overdamped (~130 ms) attitude loop, but caused floor crashes.
MPC_TILT_MAX = 1.1  # rad. Soft limit on the drone's actual tilt (angle of the thrust axis).
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
PLANNER = "fullmodel"  # "profile": smoothed path + speed profile; "mintime": minimum-time point-mass
#                      trajectory (jerk-limited, warm-started from the profile reference, same
#                      topology).
MT_N = 40  # Nodes per gate-to-gate segment.
MT_THRUST = 0.66  # N. Thrust limit of the plan (acceleration limit MT_THRUST / mass).
MT_TILT = 55.0  # deg. Tilt limit of the planned thrust vector.
MT_JERK = 0  # m/s^3. Jerk limit (0: none). The attitude loop needs time to turn the thrust
#                vector.
MT_WIN = 0.02  # m. Crossing point within this of the gate centre (vertically aimed 2 cm low: drone
#                sphere).
MT_ANGLE = 60.0  # deg. Max angle between the crossing velocity and the gate normal.
MT_Z_AIM = -0.02  # m. Vertical aim of the crossing window (the drone's collision sphere sits 2 cm
#                   up).
MT_FRAME = 0.126  # m. Clearance of the trajectory from gate-frame spheres (drone sphere 0.086 +
#                   margin).
MT_FRAME_RAND = 0.08  # m. Extra clearance from frames of gates not being crossed (randomized poses).
MT_POLE = None  # m. Pole clearance (None: POLE_CLEARANCE_PLAN).
MT_FLOOR = 0.15  # m. Minimum height after the first 0.4 s.
MT_TRUST = 0.5  # m. Each node stays within this of the warm start's node (same route, local
#                 refinement).
MT_SPEED_CAP = {"0": 3.0, "2": 3.0, "3": 3.0}  # {pass index: max speed} over the last MT_CAP_DIST into that
#                      gate
MT_CAP_FRAC = 0.3  # gate (late reveals need time to react); MT_CAP_DIST None: the last MT_CAP_FRAC
#                   of the segment's nodes.
MT_AIM = None  # e.g. {"0": (0.035, 0.0)}: tested on 2026-11-04, no net gain (other gates got worse)
#  m. {pass index: (sideways, vertical) aim offset of the crossing} along the gate's lateral and up
#  axes: cancels the drone's measured mean crossing offset at that pass (min-time/full-model plans).
MT_CAP_DIST = 0.8  # m. Path length before the crossing over which the speed cap applies.
MT_GATE_THRUST = None  # N. Lower thrust limit over the same approach parts: spare authority for the
#                        correction when the gate's true pose is revealed (None: MT_THRUST).
MTF_LEN = (0.5, 0.35)  # m. Path length before/after each crossing that the min-time trajectory
#                        keeps inside a funnel (None: no funnels): aligned, level approach and exit,
#                        so no climb, dive or swerve happens where the gate's true pose shows up.
MTF_W0 = 0.03  # m. Half-width of the funnels at the gate plane ...
MTF_SLOPE = (0.25, 0.84)  # ... growing with this slope (entry, exit) sideways ...
MTF_SLOPE_Z = (0.15, 0.3)  # ... and with this slope vertically (flat: reach the gate height early
#                            and leave it late; diving right after a crossing clips the bottom bar).
MTF_NO_UNDERSHOOT = True  # In the entry funnel, never below (above) the centre line of a gate
#                           approached from above (below).
FM_N = 30  # Nodes per gate-to-gate segment of the full-model plan (PLANNER = "fullmodel").
FM_RK = 3  # RK4 steps per node interval.
FM_THRUST = 0.68  # N. Collective thrust command limit of the full-model plan.
FM_TILT = 60.0  # deg. Tilt limit of the full-model plan.
FM_W_DU = 1e-3  # Weight of command changes between nodes (smooth commands).
FM_GATE_THRUST = None  # N. Thrust command limit over the last FM_GATE_DIST before the crossings in
FM_GATE_DIST = 0.7  # m. FM_GATE_PASSES (None: no limit): the drone coasts into gates whose true pose
FM_GATE_PASSES = (0, 1, 2, 3)  # is revealed late, with thrust to spare for the correction.
FM_STARTS = (1.0,)  # Warm starts: the point-mass plan with its time stretched by these factors
#                          (the problem has local optima; the fastest solution is kept).

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
_VERSION = 1003  # bump to invalidate cached plans and solvers


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
    # node sets of the speed caps and funnels, by path length along the warm start (fixed for all
    # rounds: sets that follow the solution make the rounds inconsistent)
    s_w = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xg[0:3], axis=1), axis=0))])
    s_wc = s_w[[(j + 1) * N for j in range(S)]]

    def cap_nodes(j: int) -> list[int]:
        """Nodes of the approach to crossing j where its speed cap applies."""
        k1 = (j + 1) * N
        if MT_CAP_DIST is None:
            return list(range(k1 - int(round(MT_CAP_FRAC * N)), k1 + 1))
        return [k for k in range(j * N, k1 + 1) if s_wc[j] - s_w[k] <= MT_CAP_DIST]

    funnel_nodes = []
    for j in range(S if MTF_LEN else 0):
        k1 = (j + 1) * N
        entry = [k for k in range(j * N + 1, k1) if s_wc[j] - s_w[k] <= MTF_LEN[0]]
        exit_ = [k for k in range(k1 + 1, min(k1 + N, K) + 1) if s_w[k] - s_wc[j] <= MTF_LEN[1]]
        funnel_nodes.append((entry, exit_ if j + 1 < S else []))
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
            # path length along the current solution, per node, and at the crossings
            s_node = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(sol["X"][0:3], axis=1), axis=0))]
            )
            s_cross = s_node[[(j + 1) * N for j in range(S)]]

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
                    opti.subject_to(ca.sum1(X[3:6, cap_nodes(j)] ** 2) <= cap**2)
                opti.subject_to(opti.bounded(0.1, T[j], 4.0))
            f = X[6:9, :] + ca.repmat(gz, 1, K + 1)
            a_lim = np.full(K + 1, a_max)
            if MT_GATE_THRUST is not None and MT_SPEED_CAP:
                for key in MT_SPEED_CAP:
                    a_lim[cap_nodes(int(key))] = a_max * MT_GATE_THRUST / MT_THRUST
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
            for j, ps in enumerate(passes if MTF_LEN else []):  # entry/exit funnels
                z_prev = (start if j == 0 else passes[j - 1]["c"])[2]
                for side, ks in enumerate(funnel_nodes[j]):
                    if not ks:
                        continue
                    rel = X[0:3, ks] - ca.repmat(ca.DM(ps["c"]), 1, len(ks))
                    d = ca.DM(ps["n"]).T @ rel  # along the normal (negative before the gate)
                    dist = -d if side == 0 else d
                    opti.subject_to(dist >= 0)
                    lat = ca.DM(ps["l"]).T @ rel
                    up = ca.DM(ps["u"]).T @ rel - MT_Z_AIM
                    for off, sl in ((lat, MTF_SLOPE[side]), (up, MTF_SLOPE_Z[side])):
                        opti.subject_to(off - MTF_W0 - sl * dist <= 0)
                        opti.subject_to(-off - MTF_W0 - sl * dist <= 0)
                    if MTF_NO_UNDERSHOOT and side == 0 and abs(z_prev - ps["c"][2]) > 0.1:
                        opti.subject_to(-np.sign(z_prev - ps["c"][2]) * up <= MTF_W0)
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
            logger.debug(
                f"min-time plan (win {win}, angle {ang}, jerk {jmax}): lap {Tv.sum():.3f} s, "
                f"segments {np.round(Tv, 3).tolist()}"
            )
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


def _full_model_reference(
    start: NDArray,
    passes: list[dict],
    poles_xy: NDArray,
    spheres: tuple,
    limits: tuple,
    warm: dict,
    extend: float = 1.5,
    mass: float = MASS,
) -> dict:
    """Minimum-time trajectory on the exact 19-state model (the MPC's own model), IPOPT.

    The point-mass plan is only feasible for a drone that turns its thrust vector instantly; this
    one includes the attitude loop, the rotor dynamics and the torque authority lost at high thrust,
    so the MPC can track it closely (no lag building up). Same gate windows, funnels, speed caps,
    pole and frame clearances as the point-mass plan (MT_* and MTF_* knobs); thrust command limit
    FM_THRUST, tilt limit FM_TILT. Time 0 is the lift-off: the rotors first spin up on the floor.
    Warm start: ``warm``, a point-mass reference (``_build_reference``) on the same route.
    Returns a reference like ``_build_reference``: states and commands of the model.
    """
    S, N = len(passes), FM_N
    K = S * N
    yaw_w = warm["u"][:, 2]
    # lift-off state: the rotors spin up at TAKEOFF_THRUST while the drone sits on the floor
    f_to = _rk4_function(4)
    x0 = np.zeros(NX)
    x0[0:3], x0[8] = start, float(yaw_w[0])
    p_to = np.array([1.0 / mass, 0.0, 0.0, 0.0, float(yaw_w[0])])
    for _ in range(500):
        if rotor_thrust(x0[12:16]) >= mass * G:
            break
        x0 = np.asarray(f_to(x0, np.array([0.0, 0.0, TAKEOFF_THRUST]), 0.002, p_to)).ravel()
        x0[0:6] = np.r_[start, 0.0, 0.0, 0.0]  # floor contact
    # warm start: the point-mass reference at the node times
    tb = np.concatenate([[0.0], warm["t_cross"]])
    tg = np.concatenate([np.linspace(tb[j], tb[j + 1], N + 1)[:-1] for j in range(S)] + [[tb[-1]]])

    def at(key: str) -> NDArray:
        return np.stack([np.interp(tg, warm["t"], col) for col in warm[key].T], axis=0)

    Xg = np.zeros((NX, K + 1))
    Xg[0:3], Xg[3:6], Xg[6:9] = at("pos"), at("vel"), at("rpy")
    Xg[9:12] = np.gradient(Xg[6:9], tg, axis=1)
    uu = at("u")
    k1, k2 = RPM2THRUST
    Xg[12:16] = ((-k1 + np.sqrt(k1**2 + k2 * uu[3])) / (2 * k2) / KRPM)[None, :]
    Xg[16:18], Xg[18] = uu[0:2], uu[3]
    Xg[:, 0] = x0
    yaw_cmd = uu[2]  # per node (segment fraction: invariant to the segment times)
    s_w = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Xg[0:3], axis=1), axis=0))])
    s_wc = s_w[[(j + 1) * N for j in range(S)]]
    centers_s, radii_s, gid_s = spheres
    pole_r = POLE_CLEARANCE_PLAN if MT_POLE is None else MT_POLE

    opti = ca.Opti()
    X, U, T = opti.variable(NX, K + 1), opti.variable(NU, K), opti.variable(S)
    f_rk = _rk4_function(FM_RK)
    for j in range(S):
        for k in range(j * N, (j + 1) * N):
            pk = ca.DM([1.0 / mass, 0.0, 0.0, 0.0, float(yaw_cmd[k])])
            opti.subject_to(X[:, k + 1] == f_rk(X[:, k], U[:, k], T[j] / N, pk))
        opti.subject_to(opti.bounded(0.1, T[j], 3.0))
    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(opti.bounded(-MPC_CMD_MAX, U[0, :], MPC_CMD_MAX))
    opti.subject_to(opti.bounded(-MPC_CMD_MAX, U[1, :], MPC_CMD_MAX))
    opti.subject_to(opti.bounded(THRUST_MIN, U[2, :], FM_THRUST))
    opti.subject_to(ca.cos(X[6, :]) * ca.cos(X[7, :]) >= np.cos(np.radians(FM_TILT)))
    for j, ps in enumerate(passes):
        kc = (j + 1) * N
        rel = X[0:3, kc] - ca.DM(ps["c"])
        opti.subject_to(ca.dot(ca.DM(ps["n"]), rel) == 0)
        opti.subject_to(opti.bounded(-MT_WIN, ca.dot(ca.DM(ps["l"]), rel), MT_WIN))
        opti.subject_to(
            opti.bounded(MT_Z_AIM - MT_WIN, ca.dot(ca.DM(ps["u"]), rel), MT_Z_AIM + MT_WIN)
        )
        if MTF_LEN:  # entry/exit funnels (node sets from the warm start)
            entry = [k for k in range(j * N + 1, kc) if s_wc[j] - s_w[k] <= MTF_LEN[0]]
            exit_ = [
                k for k in range(kc + 1, min(kc + N, K) + 1) if s_w[k] - s_wc[j] <= MTF_LEN[1]
            ]
            z_prev = (start if j == 0 else passes[j - 1]["c"])[2]
            for side, ks in enumerate((entry, exit_ if j + 1 < S else [])):
                if not ks:
                    continue
                relk = X[0:3, ks] - ca.repmat(ca.DM(ps["c"]), 1, len(ks))
                d = ca.DM(ps["n"]).T @ relk
                dist = -d if side == 0 else d
                opti.subject_to(dist >= 0)
                lat = ca.DM(ps["l"]).T @ relk
                up = ca.DM(ps["u"]).T @ relk - MT_Z_AIM
                for off, sl in ((lat, MTF_SLOPE[side]), (up, MTF_SLOPE_Z[side])):
                    opti.subject_to(off - MTF_W0 - sl * dist <= 0)
                    opti.subject_to(-off - MTF_W0 - sl * dist <= 0)
                if MTF_NO_UNDERSHOOT and side == 0 and abs(z_prev - ps["c"][2]) > 0.1:
                    opti.subject_to(-np.sign(z_prev - ps["c"][2]) * up <= MTF_W0)
        cap = {str(a): b for a, b in (MT_SPEED_CAP or {}).items()}.get(str(j))
        if cap is not None:
            ks = [k for k in range(j * N, kc + 1) if s_wc[j] - s_w[k] <= MT_CAP_DIST]
            opti.subject_to(ca.sum1(X[3:6, ks] ** 2) <= cap**2)
        if FM_GATE_THRUST is not None and j in FM_GATE_PASSES:
            ks = [k for k in range(j * N, kc) if s_wc[j] - s_w[k] <= FM_GATE_DIST]
            opti.subject_to(U[2, ks] <= FM_GATE_THRUST)
    for o in np.asarray(poles_xy, float).reshape(-1, 2):
        opti.subject_to((X[0, :] - o[0]) ** 2 + (X[1, :] - o[1]) ** 2 >= pole_r**2)
    opti.subject_to(X[2, np.nonzero(tg >= 0.4)[0].tolist()] >= MT_FLOOR)
    opti.subject_to(X[2, :] <= limits[1][2])
    opti.subject_to(opti.bounded(limits[0][0], X[0, :], limits[1][0]))
    opti.subject_to(opti.bounded(limits[0][1], X[1, :], limits[1][1]))
    for c, r, g in zip(centers_s, radii_s, gid_s):  # frame spheres near the warm start
        near = np.linalg.norm(Xg[0:3].T - c, axis=1) < r + MT_FRAME + MT_FRAME_RAND + 0.35
        for i in np.nonzero(near)[0]:
            own = any(abs(s_w[i] - s_wc[j]) < 0.8 for j, ps in enumerate(passes) if ps["gid"] == g)
            rad = r + MT_FRAME + (0.0 if own else MT_FRAME_RAND)
            opti.subject_to(ca.sumsqr(X[0:3, int(i)] - ca.DM(c)) >= rad**2)
    opti.minimize(ca.sum1(T) + FM_W_DU * ca.sumsqr(U[:, 1:] - U[:, :-1]))
    opti.set_initial(X, Xg)
    opti.set_initial(U, np.vstack([uu[0:2, :-1], uu[3:4, :-1]]))
    opti.set_initial(T, np.diff(tb))
    opti.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {"max_iter": 3000, "print_level": 0, "sb": "yes", "tol": 1e-6},
    )
    try:
        res = opti.solve()
        get = res.value
    except RuntimeError:
        logger.warning(f"full-model plan: {opti.stats().get('return_status')} (using the iterate)")
        get = opti.debug.value
    Tv = np.asarray(get(T), float).ravel()
    Xv, Uv = np.asarray(get(X), float), np.asarray(get(U), float)
    sol = {"T": Tv, "X": Xv, "U": Uv, "yaw": np.asarray(yaw_cmd, float)}
    return {**_full_model_dense(sol, passes, extend, mass), "sol": sol}


def _full_model_dense(sol: dict, passes: list[dict], extend: float = 1.5, mass: float = MASS) -> dict:
    """Reference on the REF_DT grid from a full-model solution (node states, commands, times)."""
    Tv, Xv, Uv, yaw_cmd = sol["T"], sol["X"], sol["U"], sol["yaw"]
    S, N = len(Tv), Uv.shape[1] // len(Tv)
    K = S * N
    t_nodes = np.concatenate(
        [np.cumsum(np.r_[0.0, Tv])[j] + np.arange(N) * Tv[j] / N for j in range(S)]
        + [[Tv.sum()]]
    )
    # dense reference: re-simulate each interval (exactly the model's trajectory), then fly on
    t_end = float(Tv.sum())
    t = np.arange(0.0, t_end + extend, REF_DT)
    xs, us = np.zeros((len(t), NX)), np.zeros((len(t), 4))
    f_fine = _rk4_function(2)
    for i, ti in enumerate(t):
        k = int(min(np.searchsorted(t_nodes, min(ti, t_end), side="right") - 1, K - 1))
        pk = np.array([1.0 / mass, 0.0, 0.0, 0.0, float(yaw_cmd[k])])
        dt = min(ti, t_end) - t_nodes[k]
        xs[i] = Xv[:, k] if dt < 1e-9 else np.asarray(f_fine(Xv[:, k], Uv[:, k], dt, pk)).ravel()
        us[i] = [Uv[0, k], Uv[1, k], yaw_cmd[k], Uv[2, k]]
    after = t > t_end  # past the finish: straight on at the final velocity, level
    pos, vel = xs[:, 0:3].copy(), xs[:, 3:6].copy()
    pos[after] = xs[~after][-1, 0:3] + xs[~after][-1, 3:6] * (t[after] - t_end)[:, None]
    vel[after] = xs[~after][-1, 3:6]
    rpy = xs[:, 6:9].copy()
    f_lvl = mass * np.array([0.0, 0.0, G]) - DRAG * vel[after]
    rpy[after] = _rpy_from_thrust(f_lvl, np.full(after.sum(), yaw_cmd[-1]))
    us[after] = np.column_stack(
        [rpy[after, 0:2], np.full(after.sum(), yaw_cmd[-1]), np.linalg.norm(f_lvl, axis=1)]
    )
    s_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    return {
        "t": t,
        "s": s_arc,
        "pos": pos,
        "vel": vel,
        "acc": np.gradient(vel, REF_DT, axis=0),
        "rpy": rpy,
        "u": us,
        "t_cross": np.cumsum(Tv),
        "gids": np.array([ps["gid"] for ps in passes]),
        "lap": t_end,
    }


_PLANS: dict = {}


def _code_dir() -> Path:
    base = Path(os.environ.get("TOMPC_CACHE_DIR", Path.cwd() / "c_generated_code"))
    d = base / "time_optimal_mpc_fast"  # separate from time_optimal_mpc.py's plans and solvers
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
    if PLANNER in ("mintime", "fullmodel"):
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
            MT_CAP_DIST,
            *([str(MT_AIM)] if MT_AIM else []),  # (absent when unused: keeps the embedded plan's key)
            MTF_LEN,
            MTF_W0,
            MTF_SLOPE,
            MTF_SLOPE_Z,
            MTF_NO_UNDERSHOOT,
            _VERSION,
        ]
        mt_key = hashlib.sha1(json.dumps([path_key, mt], default=str).encode()).hexdigest()[:16]
        spheres = _gate_spheres(gates_pos, gates_quat)
        masses = list(MASS_GRID) if MASS_ADAPT else [MASS]
        aim = {str(k): v for k, v in (MT_AIM or {}).items()}
        passes_aim = [  # crossing targets shifted by the per-pass aim offsets
            {**ps, "c": ps["c"] + aim.get(str(j), (0.0, 0.0))[0] * ps["l"]
             + aim.get(str(j), (0.0, 0.0))[1] * ps["u"]}
            for j, ps in enumerate(passes)
        ]
        if PLANNER == "fullmodel":
            fm = [
                mt_key, FM_N, FM_RK, FM_THRUST, FM_TILT, FM_W_DU, FM_STARTS, MPC_CMD_MAX,
                TAKEOFF_THRUST, FM_GATE_THRUST, FM_GATE_DIST, FM_GATE_PASSES,
            ]
            fm_key = hashlib.sha1(json.dumps(fm, default=str).encode()).hexdigest()[:16]
            file_fm = _code_dir() / f"fullmodel_{fm_key}.npz"
            shipped = _shipped_plan()
            full = None
            if shipped is not None and shipped["key"] == fm_key:
                # the plan computed offline for this track and these knobs (embedded at the end
                # of this file): no solver run here, and no risk of another local optimum
                full = _full_model_dense(shipped, passes)
                logger.info("time_optimal_mpc_fast: using the embedded full-model plan")
            elif file_fm.exists():
                full = np.load(file_fm, allow_pickle=True)["ref"].item()
            if full is not None:
                ref = {**full, "late": late}  # one plan for all masses (the MPC knows the mass)
                logger.info(f"time_optimal_mpc_fast: full-model reference lap {ref['lap']:.3f} s")
                _PLANS[key] = ref
                return ref
        file = _code_dir() / f"mintime_{mt_key}.npz"
        with _file_lock(_code_dir() / "plan.lock"):
            if file.exists():
                data = np.load(file, allow_pickle=True)
                profs = data["profs"].item()
            else:
                warm = _speed_profile(pts, centers, late=late)
                nominal = _min_time_profile(
                    start, passes_aim, poles_xy, spheres, limits, warm, MT_THRUST / MASS
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
        if PLANNER == "fullmodel":  # the point-mass plan is the warm start of the full-model plan
            with _file_lock(_code_dir() / "plan.lock"):
                if file_fm.exists():
                    full = np.load(file_fm, allow_pickle=True)["ref"].item()
                else:
                    full, w0 = None, refs[m0]
                    for lam in FM_STARTS:  # multi-start: time-stretched warm starts
                        warm = {
                            **w0,
                            "t": w0["t"] * lam,
                            "vel": w0["vel"] / lam,
                            "t_cross": np.asarray(w0["t_cross"]) * lam,
                        }
                        cand = _full_model_reference(
                            start, passes_aim, poles_xy, spheres, limits, warm, mass=MASS
                        )
                        logger.debug(f"full-model plan from x{lam} warm start: {cand['lap']:.3f} s")
                        if full is None or cand["lap"] < full["lap"]:
                            full = cand
                    full["sol"]["key"] = fm_key
                    np.savez(file_fm, ref=np.array(full, dtype=object))
            ref = {**full, "late": late}  # one plan for all masses (the MPC knows the mass)
        logger.info(f"time_optimal_mpc_fast: min-time reference lap {ref['lap']:.3f} s")
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
    logger.info(f"time_optimal_mpc_fast: reference lap {ref['lap']:.3f} s")
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
class TimeOptimalMPCFast(Controller):
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
            raise ValueError("TimeOptimalMPCFast needs env.control_mode = 'attitude'")
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


def _shipped_plan() -> dict | None:
    """The embedded full-model plan (solution of _full_model_reference for level 2), if any."""
    if not _SHIPPED_PLAN:
        return None
    import base64
    import io
    import zlib

    data = np.load(io.BytesIO(zlib.decompress(base64.b64decode("".join(_SHIPPED_PLAN)))))
    return {k: (str(data[k]) if k == "key" else data[k]) for k in ("key", "T", "X", "U", "yaw")}


# Full-model plan for the level-2 track with the knobs above (zlib-compressed npz, base64), computed
# offline with _full_model_reference; plan_reference() uses it only if its key (track + knobs)
# matches, else it solves the plan itself (a few minutes) and caches it.
_SHIPPED_PLAN = (
    "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"
    "IRv7ZSVZtG/geoz9XVl0ymh05ri7LHq7wxN19b0scotRlyILkUXTxZGqR6VkUc5SU53h3n2exQrlmYqyyJg+P3pFSxb5P84a"
    "8+KQQsJprwKvV4uj3mp58S4PCVQSbTv7e0sC9XW9yTY3kURDFFTx525KolZx2tT1x5IoxlQ4UFRLEmlutrlGgziKI6CiCyai"
    "aN1TeKtSVhR5qVrVFsyLosx/ktiYSRItdtueKw+VRBIC9P1qxlLIqayNvTdbCtHWytmcyhRDs7mWW7Ia4ujdVMbKB3JJlHpd"
    "4jJRWBIlFQcI37OTRCHDN6taayQRyYenP2LipZAHb11l1o89P+qbOhT5TRodvNSUGRYhhTQ8Q0oHc6XQ0bA+sptfpdHDLYsH"
    "v/hkkEm2VN3MYRkUffx5wBs2GbTDrVJefVwGNVOqCuQR9+yCPEKKFGVQ367MdWtPGdQzuZ6n80YGdQ2m8Z5Xk0Fpc0wWs4XS"
    "iDe65NoLPWn0W+tNlbq+NDr0os74xpI0eum2S/JYTga5nncnr1aRQTmCaoqPWCTR2k8Vv8vGYii0t4HmmYkwqmc8Kvpx6Aia"
    "8uKc70GHEec7xlGdhycR62s/vvbDkqgssV7l2qok4lDzXHM038OrrV1cHCuJOC1e9OV8l0Z9/WJkfYYSaMS66FpFuwTK4mZ9"
    "SaMrjWxp6o3HkiXRQJmY7vQVKUTVdOBorqg4SjjbnlDtLYVuX1S69+qdNGodp/xULy+DZnZHfqQUyaDVTz0FfmyyKJODP1JQ"
    "WxYdVDkjFW0ui9J/+j74TiaLPvI/yOU6JYPK8ssPvsiTQVTnNyJr4mVQdZxVozS1LFKKCjlQsiiDTLmUVe9EyCAyGUEDb0Vp"
    "lH9foWlEWRKp3NHWfx8mjiKkWnfOBoki3sYHviwjYohRSvrE6RUJ5Enk4H32RAr1Hi9rmS2TQsm/jCJ+HJRCrlFXfy/5SaJh"
    "G7XikXlJpG3Us+MQIIX+mDH/M+uRQsODUbc1xaRRZ95o56f70ihd/XmYdKQ00kg8mpfYKo3mBVoPeO3FDR996v+dtzpWvhp3"
    "P5NCT0U9RPT29k3nrtItGiFplDT9bVdBXwr5qsVZ3YiWQue+vZ7WFJdGCgqVFieCpZFK2amk7ilpBLf6Hrgz7+U3qhzqhH0y"
    "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"
    "XBnf/g8/2BNi"
)
