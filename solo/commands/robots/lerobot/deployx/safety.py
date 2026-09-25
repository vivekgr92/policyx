"""
Safety layer for the DeployX edge agent: per-joint limits loaded from the
follower's existing lerobot calibration file, an action validator, a
predict-response watchdog, and an emergency-stop mechanism.

Joint limits are read from the same calibration file 'solo robo --calibrate
follower' already writes (see solo/commands/robots/lerobot/calibration.py),
rather than a new limits config. lerobot stores each motor's calibrated
travel as raw encoder ticks (range_min/range_max - see Robot._save_calibration
in lerobot/robots/robot.py and MotorCalibration in lerobot/motors/motors_bus.py),
and DeployX's wire protocol carries "raw joint positions" (see
ObservationPayload.state in protocol.py), so no unit conversion is needed
between the calibration file and the values validated here.
"""

import json
import math
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from solo.commands.robots.lerobot.deployx import protocol

# Fallback per-joint velocity limit (ticks/s) used when the caller doesn't
# supply one. SO-101 motors have a full-scale range of ~4095 ticks; this caps
# a single joint's commanded move to roughly one full sweep per second, which
# is already fast for a policy chunk executed at typical control-loop rates.
DEFAULT_MAX_VELOCITY_TICKS_PER_S = 1500.0


class EmergencyStopTriggered(Exception):
    """Raised to unwind the control loop into the safe-shutdown routine.

    Raised either by EmergencyStop.check() (SIGINT/SIGTERM was received) or
    directly by the control loop when validate_action() rejects an action -
    a policy producing garbage is treated the same as an operator e-stop.
    """


@dataclass
class JointLimits:
    """Per-joint position range (raw calibration ticks) and max per-step velocity."""

    names: list[str]
    position_min: list[float]
    position_max: list[float]
    max_velocity: list[float]  # ticks/s, same order as `names`

    def __len__(self) -> int:
        return len(self.names)


def _calibration_fpath(robot_type: str, follower_id: str) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

    return HF_LEROBOT_CALIBRATION / ROBOTS / f"{robot_type}_follower" / f"{follower_id}.json"


def load_joint_limits(
    robot_type: str,
    follower_id: str,
    max_velocity_ticks_per_s: float = DEFAULT_MAX_VELOCITY_TICKS_PER_S,
) -> JointLimits:
    """
    Build JointLimits from the follower's lerobot calibration file, written by
    'solo robo --calibrate follower' (or --calibrate all). Raises
    FileNotFoundError with an actionable message rather than falling back to
    an invented range - running DeployX against an uncalibrated arm is a
    setup error, not something to silently paper over.
    """
    fpath = _calibration_fpath(robot_type, follower_id)
    if not fpath.is_file():
        raise FileNotFoundError(
            f"No calibration file found for {robot_type} follower '{follower_id}' at {fpath}. "
            "Run 'solo robo --calibrate follower' first."
        )

    with open(fpath) as f:
        raw = json.load(f)

    names = list(raw.keys())
    position_min = [float(raw[name]["range_min"]) for name in names]
    position_max = [float(raw[name]["range_max"]) for name in names]
    max_velocity = [max_velocity_ticks_per_s] * len(names)

    return JointLimits(
        names=names,
        position_min=position_min,
        position_max=position_max,
        max_velocity=max_velocity,
    )


def validate_action(
    action: list[float],
    current_state: Optional[list[float]],
    dt: float,
    limits: JointLimits,
) -> tuple[bool, list[float], Optional[str]]:
    """
    Validate one commanded action (a single step of an action_chunk) against
    joint position and velocity limits.

    Policy (deliberately asymmetric - see task spec): a merely out-of-range
    value is clamped back within the calibrated envelope and execution
    proceeds, since small overshoot is expected from a policy and clamping
    keeps the arm moving smoothly. A value that is NaN/Inf, wrong-shaped, or
    extreme (more than one full calibrated range-width past a limit) is
    treated as garbage output rather than overshoot: it is rejected outright
    (is_safe=False) so the caller e-stops instead of executing a clamped
    value that no longer resembles the policy's intent.

    Returns (is_safe, clamped_action, reason). `reason` is set whenever
    clamping occurred (is_safe=True) or explains the rejection (is_safe=False).
    """
    if len(action) != len(limits):
        return False, list(action), f"action length {len(action)} != {len(limits)} calibrated joints"

    if any(math.isnan(v) or math.isinf(v) for v in action):
        return False, list(action), "action contains NaN/Inf"

    for i, v in enumerate(action):
        span = limits.position_max[i] - limits.position_min[i]
        extreme_lo = limits.position_min[i] - span
        extreme_hi = limits.position_max[i] + span
        if v < extreme_lo or v > extreme_hi:
            return (
                False,
                list(action),
                f"joint '{limits.names[i]}' action {v:.1f} is extremely out of range "
                f"[{limits.position_min[i]:.1f}, {limits.position_max[i]:.1f}]",
            )

    clamped = list(action)
    reasons: list[str] = []

    for i, v in enumerate(clamped):
        lo, hi = limits.position_min[i], limits.position_max[i]
        if v < lo or v > hi:
            new_v = min(max(v, lo), hi)
            reasons.append(f"'{limits.names[i]}' position clamped {v:.1f}->{new_v:.1f}")
            clamped[i] = new_v

    if current_state is not None and dt > 0:
        if len(current_state) != len(limits):
            return (
                False,
                list(action),
                f"current_state length {len(current_state)} != {len(limits)} calibrated joints",
            )
        for i, v in enumerate(clamped):
            max_step = limits.max_velocity[i] * dt
            delta = v - current_state[i]
            if abs(delta) > max_step:
                new_v = current_state[i] + math.copysign(max_step, delta)
                reasons.append(f"'{limits.names[i]}' velocity clamped {v:.1f}->{new_v:.1f}")
                clamped[i] = new_v

    reason = "; ".join(reasons) if reasons else None
    return True, clamped, reason


@dataclass
class Watchdog:
    """Tracks time since the last successful server response (pong or action)."""

    timeout_s: float = protocol.PREDICT_TIMEOUT_S
    _last_ok: float = field(default_factory=time.monotonic)

    def reset(self, now: Optional[float] = None) -> None:
        self._last_ok = now if now is not None else time.monotonic()

    def is_timed_out(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        return (now - self._last_ok) > self.timeout_s


class EmergencyStop:
    """
    SIGINT/SIGTERM handler for the edge agent's control loop.

    The signal handler only sets a flag - it never runs shutdown logic
    itself, since safe-stop needs to talk to hardware/telemetry, which is not
    safe to do from inside a signal handler. The control loop calls check()
    once per iteration, which raises EmergencyStopTriggered so the normal
    try/finally safe-shutdown path in edge_agent.py runs.
    """

    def __init__(self) -> None:
        self._triggered = False
        self._prev_handlers: dict[int, object] = {}

    def _handle(self, signum, frame) -> None:  # noqa: ARG002 - signal handler signature
        self._triggered = True

    def arm(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def disarm(self) -> None:
        for sig, handler in self._prev_handlers.items():
            signal.signal(sig, handler)
        self._prev_handlers.clear()

    def trigger(self) -> None:
        """Manually trigger e-stop, e.g. after validate_action() rejects an action."""
        self._triggered = True

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def check(self) -> None:
        """Raise EmergencyStopTriggered if the stop flag is set. Call every loop iteration."""
        if self._triggered:
            raise EmergencyStopTriggered("Emergency stop triggered (signal or safety check)")


def safe_shutdown(robot, reason: str = "") -> None:
    """
    Best-effort safe-stop: disable torque and disconnect. Never raises - this
    runs from exception/finally paths where a second failure must not mask
    the first or prevent telemetry from flushing.

    lerobot's SO-101 follower disables torque on disconnect by default
    (SOFollowerConfig.disable_torque_on_disconnect=True, see
    lerobot/robots/so_follower/config_so_follower.py), so disconnect() alone
    is already the hardware's safe-stop; disable_torque() is called first
    too, in case disconnect() itself raises partway through.
    """
    import typer

    typer.echo(f"🛑 Safe shutdown{f': {reason}' if reason else ''}")

    if robot is None:
        return

    try:
        bus = getattr(robot, "bus", None)
        if bus is not None and hasattr(bus, "disable_torque"):
            try:
                bus.disable_torque()
            except Exception:
                pass

        if getattr(robot, "is_connected", False):
            robot.disconnect()
    except Exception as e:
        typer.echo(f"⚠️  Error during safe shutdown: {e}")
