"""
Star Arm 102 (StarAI / Fashionstar) leader -> SO101 follower teleoperator adapter.

The Star Arm 102 leader is a 6+1 DoF arm driven over the FashionStar UART bus,
exposed to LeRobot by the vendor plugin `lerobot_teleoperator_stararm102`. That
plugin emits actions keyed `Motor_0..Motor_5` + `gripper`, which an SO101
follower cannot consume: it expects `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll` and `gripper`.

This adapter wraps the vendor teleoperator and republishes its action under the
SO101 joint names, so `lerobot.teleoperate` (and record/inference, which use the
same loop) work unchanged.

Joint correspondence
--------------------
The Star Arm 102 has one joint more than the SO101 - a forearm roll (`Motor_3`)
with no SO101 equivalent - so it is dropped by default. Both arms report body
joints in degrees about their own calibrated midpoint, which makes the transfer
a physically meaningful 1:1 by default; per-joint sign, gain and offset are
available for the axes whose zero or direction disagree.
"""

import logging
import time
from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

# Encoder counts per full turn on the StarAI bus (see lerobot_motor_starai.starai).
STARAI_COUNTS_PER_TURN = 4096.0

# SO101 joint <- Star Arm 102 servo. Motor_3 (forearm roll) has no SO101 twin.
DEFAULT_JOINT_MAP: dict[str, str] = {
    "shoulder_pan": "Motor_0",
    "shoulder_lift": "Motor_1",
    "elbow_flex": "Motor_2",
    "wrist_flex": "Motor_4",
    "wrist_roll": "Motor_5",
    "gripper": "gripper",
}

SO101_JOINTS = tuple(DEFAULT_JOINT_MAP)

# Joints that can spin past a half turn, where the raw count wraps at 0/4096.
DEFAULT_CONTINUOUS_JOINTS: tuple[str, ...] = ("wrist_roll",)


@TeleoperatorConfig.register_subclass("stararm102_so101_leader")
@dataclass
class StarArm102SO101LeaderConfig(TeleoperatorConfig):
    # Serial port of the Star Arm 102 leader (its UC-01 board enumerates as a CH340).
    port: str

    # SO101 joint -> StarAI servo. Dropping an entry leaves that follower joint uncommanded.
    joint_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_JOINT_MAP))

    # Per-joint correction applied as: sign * gain * leader_degrees + offset_degrees.
    signs: dict[str, float] = field(default_factory=dict)
    gains: dict[str, float] = field(default_factory=dict)
    offsets: dict[str, float] = field(default_factory=dict)

    # Extra leader servos folded into a follower joint, as
    # {follower_joint: {leader_servo: weight}}. The Star Arm 102 has two twist
    # axes where the SO101 has one, so adding the forearm twist onto the wrist
    # twist keeps the leader's spare joint doing something useful. Weights are in
    # leader degrees per follower degree; use a negative weight when the two axes
    # turn opposite ways. Ignored for the gripper.
    blend: dict[str, dict[str, float]] = field(default_factory=dict)

    # SO101 joint -> [min, max] in the follower's own degree convention. Derived from
    # the follower's calibration file; commands are clamped so a mis-signed axis
    # cannot drive the follower into a hard stop.
    joint_limits: dict[str, list[float]] = field(default_factory=dict)

    # Joints whose raw count may wrap at 0/4096 and needs unwrapping.
    continuous_joints: list[str] = field(default_factory=lambda: list(DEFAULT_CONTINUOUS_JOINTS))

    # Exponential smoothing on the emitted action. 0.0 disables it; 0.3 is a gentle filter.
    smoothing: float = 0.0


class StarArm102SO101Leader(Teleoperator):
    config_class = StarArm102SO101LeaderConfig
    # Share the vendor teleoperator's calibration directory so `solo robo --calibrate
    # leader` and the vendor's own `lerobot-calibrate` write the same file.
    name = "stararm102_leader"

    def __init__(self, config: StarArm102SO101LeaderConfig):
        super().__init__(config)
        self.config = config

        try:
            from lerobot_teleoperator_stararm102 import Stararm102Leader, Stararm102LeaderConfig
        except ImportError as e:
            raise ImportError(
                "The Star Arm 102 leader needs the vendor LeRobot plugin. Install it with:\n"
                "    pip install lerobot_teleoperator_stararm102 lerobot-motor-starai 'fashionstar-uart-sdk>=1.3.6'"
            ) from e

        # use_degrees is left False: the StarAI bus does not convert to degrees in its
        # DEGREES branch, so this adapter reads raw counts and converts them itself.
        self._inner = Stararm102Leader(
            Stararm102LeaderConfig(
                port=config.port,
                id=config.id,
                calibration_dir=config.calibration_dir,
                use_degrees=False,
            )
        )

        self._prev_counts: dict[str, float] = {}
        self._prev_action: dict[str, float] = {}

    # ------------------------------------------------------------------ features

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joint_map}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ------------------------------------------------------------------ lifecycle

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    @property
    def is_calibrated(self) -> bool:
        return bool(self._inner.calibration)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._inner.connect(calibrate=calibrate)
        self.calibration = self._inner.calibration

        if calibrate and not self._inner.calibration:
            # `calibrate=False` is how `lerobot-calibrate` opens the arm before
            # recording ranges, so only a requested-and-failed calibration is an error.
            raise DeviceNotConnectedError(
                f"Star Arm 102 leader '{self.id}' has no calibration. "
                "Run 'solo robo --calibrate leader' first."
            )

        self._prev_counts = {}
        self._prev_action = {}
        self.configure()
        logger.info(f"{self} connected ({len(self.config.joint_map)} mapped joints).")

    def calibrate(self) -> None:
        self._inner.calibrate()
        self.calibration = self._inner.calibration

    def configure(self) -> None:
        self._inner.configure()

    def disconnect(self) -> None:
        self._inner.disconnect()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("The Star Arm 102 leader has no force feedback channel.")

    # ------------------------------------------------------------------ conversion

    def _midpoint(self, motor: str) -> float:
        cal = self._inner.calibration[motor]
        return (cal.range_min + cal.range_max) / 2.0

    def _unwrap(self, motor: str, count: float, continuous: bool) -> float:
        """Undo the 0/4096 wrap on joints that can turn past a half revolution."""
        if not continuous:
            return count
        prev = self._prev_counts.get(motor)
        if prev is not None:
            delta = count - prev
            if delta > STARAI_COUNTS_PER_TURN / 2:
                count -= STARAI_COUNTS_PER_TURN
            elif delta < -STARAI_COUNTS_PER_TURN / 2:
                count += STARAI_COUNTS_PER_TURN
        self._prev_counts[motor] = count
        return count

    def _continuous_motors(self) -> set[str]:
        """Servos that need wrap-unwrapping, from the joints declared continuous."""
        motors: set[str] = set()
        for joint in self.config.continuous_joints:
            primary = self.config.joint_map.get(joint)
            if primary:
                motors.add(primary)
            motors.update(self.config.blend.get(joint, {}))
        return motors

    def _motor_degrees(self, counts: dict[str, float]) -> dict[str, float]:
        """
        Every servo reading as degrees about its calibrated midpoint.

        Computed once per cycle so a servo feeding two follower joints - which is
        what blending does - is unwrapped exactly once.
        """
        continuous = self._continuous_motors()
        degrees: dict[str, float] = {}
        for motor, count in counts.items():
            if count is None or motor not in self._inner.calibration:
                continue
            unwrapped = self._unwrap(motor, float(count), motor in continuous)
            # Matches how LeRobot normalises a follower joint in DEGREES mode.
            degrees[motor] = (unwrapped - self._midpoint(motor)) * 360.0 / STARAI_COUNTS_PER_TURN
        return degrees

    def _gripper_percent(self, motor: str, counts: dict[str, float]) -> float | None:
        """The SO101 gripper is normalised 0..100 over its calibrated travel."""
        count = counts.get(motor)
        if count is None or motor not in self._inner.calibration:
            return None
        cal = self._inner.calibration[motor]
        span = cal.range_max - cal.range_min
        if span == 0:
            return None
        return min(100.0, max(0.0, (count - cal.range_min) / span * 100.0))

    def _leader_value(
        self, joint: str, motor: str, counts: dict[str, float],
        degrees: dict[str, float] | None = None,
    ) -> float | None:
        """The leader's contribution to one SO101 joint, before corrections."""
        if joint == "gripper":
            return self._gripper_percent(motor, counts)

        if degrees is None:
            degrees = self._motor_degrees(counts)

        value = degrees.get(motor)
        if value is None:
            return None

        for source, weight in self.config.blend.get(joint, {}).items():
            contribution = degrees.get(source)
            if contribution is not None:
                value += weight * contribution

        return value

    def _apply_corrections(self, joint: str, value: float) -> float:
        value = self.config.signs.get(joint, 1.0) * self.config.gains.get(joint, 1.0) * value
        value += self.config.offsets.get(joint, 0.0)

        limits = self.config.joint_limits.get(joint)
        if limits:
            lo, hi = float(limits[0]), float(limits[1])
            value = min(hi, max(lo, value))
        elif joint == "gripper":
            value = min(100.0, max(0.0, value))

        return value

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self._inner.calibration:
            raise DeviceNotConnectedError(
                f"Star Arm 102 leader '{self.id}' has no calibration, so its readings "
                "cannot be mapped onto SO101 joints. Run 'solo robo --calibrate leader'."
            )

        start = time.perf_counter()
        counts = self._inner.bus.sync_read("Present_Position", normalize=False)
        degrees = self._motor_degrees(counts)

        action: dict[str, float] = {}
        alpha = self.config.smoothing
        for joint, motor in self.config.joint_map.items():
            raw = self._leader_value(joint, motor, counts, degrees)
            if raw is None:
                # A dropped reading: hold the previous command rather than jumping.
                if joint in self._prev_action:
                    action[f"{joint}.pos"] = self._prev_action[joint]
                continue

            value = self._apply_corrections(joint, raw)
            if alpha > 0.0 and joint in self._prev_action:
                value = alpha * self._prev_action[joint] + (1.0 - alpha) * value

            self._prev_action[joint] = value
            action[f"{joint}.pos"] = value

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action
