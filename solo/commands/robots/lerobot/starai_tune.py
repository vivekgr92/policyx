"""
Interactive tuning for the Star Arm 102 -> SO101 joint mapping.

A Star Arm 102 leader is not an SO101, so some of its axes run the opposite way
or swing further than the follower's matching joint. `solo robo --star-tune`
shows the live leader readings next to the command they produce, and writes the
per-joint sign, gain and offset to `~/.solo/starai_map.json`, which every mode
reads.
"""

import threading
import time
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.prompt import Confirm, Prompt
from rich.table import Table

from solo.commands.robots.lerobot.starai_config import (
    DEFAULT_STARAI_MAP,
    describe_map,
    get_follower_joint_limits,
    get_leader_joint_spans,
    suggest_gains,
    load_starai_map,
    save_starai_map,
)
from solo.commands.robots.lerobot.teleoperators.stararm102_so101 import SO101_JOINTS

console = Console()

STARAI_MOTORS = [f"Motor_{i}" for i in range(6)] + ["gripper"]


def _resolve_leader(config: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Find the Star Arm 102's port and id, plus the follower id to clamp against."""
    from solo.commands.robots.lerobot.config import get_known_ids_by_type
    from solo.commands.robots.lerobot.scan import scan_starai_port

    lerobot_config = config.get("lerobot", {})
    follower_id = lerobot_config.get("follower_id")

    # The saved leader_port may belong to a different leader (an SO101, say), so
    # only trust it if a Star Arm 102 actually answers on it.
    port = lerobot_config.get("leader_port")
    if port and not scan_starai_port(port):
        typer.echo(f"ℹ️  No Star Arm 102 on the saved leader port {port}.")
        port = None

    if not port:
        typer.echo("🔍 Scanning for the Star Arm 102...")
        try:
            from solo.commands.robots.lerobot.scan import auto_detect_single_port
            port, _ = auto_detect_single_port("leader", robot_type="stararm102", verbose=True)
        except Exception as e:
            typer.echo(f"⚠️  Scan failed: {e}")

    if not port:
        port = Prompt.ask("Star Arm 102 leader port")

    # Likewise, the saved leader_id only applies when it was recorded for this arm.
    leader_id = None
    if lerobot_config.get("robot_type") == "stararm102":
        leader_id = lerobot_config.get("leader_id")

    if not leader_id:
        # Only this robot type's ids: get_known_ids() would fall back to every
        # leader id on the machine, which for this prompt is just noise.
        known_leader_ids = (
            get_known_ids_by_type(config).get("stararm102", {}).get("leaders", [])
        )
        if known_leader_ids:
            typer.echo("📇 Known Star Arm 102 leader ids:")
            for i, known in enumerate(known_leader_ids, 1):
                typer.echo(f"   {i}. {known}")
        leader_id = Prompt.ask(
            "Star Arm 102 leader id (must match the one you calibrated)",
            default=known_leader_ids[0] if known_leader_ids else "stararm102_leader",
        )

    return port, leader_id, follower_id


def _live_view(port: str, leader_id: str, follower_id: Optional[str], mapping: Dict) -> None:
    """Stream the leader's joints next to the SO101 command they produce."""
    leader, limits = _open_leader(port, leader_id, follower_id, mapping)
    if leader is None:
        return

    stop = _wait_for_enter()
    typer.echo("\n📋 Move one leader joint at a time. Press ENTER to stop.\n")

    try:
        with Live(console=console, refresh_per_second=8, transient=True) as live:
            while not stop.is_set():
                try:
                    counts = leader._inner.bus.sync_read("Present_Position", normalize=False)
                    # Read each joint once: _leader_value advances the wrap-unwrapping
                    # state, so calling get_action() as well would unwrap twice.
                    degrees = leader._motor_degrees(counts)
                    leader_values = {
                        joint: leader._leader_value(joint, motor, counts, degrees)
                        for joint, motor in mapping["joint_map"].items()
                    }
                    action = {
                        joint: (None if value is None else leader._apply_corrections(joint, value))
                        for joint, value in leader_values.items()
                    }
                except Exception as e:
                    live.update(f"[red]Read failed: {e}[/red]")
                    time.sleep(0.2)
                    continue

                table = Table(title=f"Star Arm 102 '{leader_id}' → SO101", show_header=True)
                table.add_column("SO101 joint", style="bold")
                table.add_column("Leader servo")
                table.add_column("Raw count", justify="right")
                table.add_column("Leader", justify="right")
                table.add_column("Command", justify="right")
                table.add_column("Clamp", justify="center")

                for joint in SO101_JOINTS:
                    motor = mapping["joint_map"].get(joint)
                    if not motor:
                        table.add_row(joint, "[dim]unmapped[/dim]", "-", "-", "-", "-")
                        continue

                    raw = counts.get(motor)
                    leader_value = leader_values.get(joint)
                    command = action.get(joint)
                    unit = "" if joint == "gripper" else "°"
                    limit = limits.get(joint)

                    at_limit = "-"
                    if limit and command is not None:
                        margin = (limit[1] - limit[0]) * 0.01
                        if command <= limit[0] + margin or command >= limit[1] - margin:
                            at_limit = "[bold red]AT LIMIT[/bold red]"
                        else:
                            at_limit = "[green]ok[/green]"

                    blended = mapping.get("blend", {}).get(joint, {})
                    source = motor + "".join(f" {w:+.2f}×{src}" for src, w in sorted(blended.items()))

                    table.add_row(
                        joint,
                        source,
                        "-" if raw is None else f"{raw:.0f}",
                        "-" if leader_value is None else f"{leader_value:+.1f}{unit}",
                        "-" if command is None else f"{command:+.1f}{unit}",
                        at_limit,
                    )

                live.update(table)
                time.sleep(0.1)
    finally:
        leader.disconnect()


def _open_leader(port: str, leader_id: str, follower_id: Optional[str], mapping: Dict):
    """Build and connect the adapter, or return None with an explanation."""
    from solo.commands.robots.lerobot.teleoperators.stararm102_so101 import (
        StarArm102SO101Leader,
        StarArm102SO101LeaderConfig,
    )

    limits = (
        get_follower_joint_limits(follower_id)
        if mapping.get("clamp_to_follower_limits", True)
        else {}
    )

    leader = StarArm102SO101Leader(
        StarArm102SO101LeaderConfig(
            port=port,
            id=leader_id,
            joint_map=dict(mapping["joint_map"]),
            signs=dict(mapping["signs"]),
            gains=dict(mapping["gains"]),
            offsets=dict(mapping["offsets"]),
            blend={j: dict(w) for j, w in mapping.get("blend", {}).items()},
            joint_limits=limits,
            continuous_joints=list(mapping.get("continuous_joints", [])),
            smoothing=float(mapping.get("smoothing", 0.0)),
        )
    )

    try:
        leader.connect(calibrate=False)
    except Exception as e:
        typer.echo(f"❌ Could not open the leader on {port}: {e}")
        return None, limits

    if not leader.is_calibrated:
        typer.echo("❌ This leader is not calibrated yet. Run 'solo robo --calibrate leader' first.")
        leader.disconnect()
        return None, limits

    return leader, limits


def _wait_for_enter() -> threading.Event:
    stop = threading.Event()

    def wait():
        try:
            input()
        except EOFError:
            pass
        stop.set()

    threading.Thread(target=wait, daemon=True).start()
    return stop


def _identify_view(port: str, leader_id: str, follower_id: Optional[str], mapping: Dict) -> None:
    """
    Name the leader's servos by watching which one you move.

    The Star Arm 102 has one joint more than the SO101, so one servo drives
    nothing. Rather than reading that off a table of numbers, move each joint in
    turn: the row that lights up is the servo you just moved, and it says what -
    if anything - that servo controls.
    """
    leader, _ = _open_leader(port, leader_id, follower_id, mapping)
    if leader is None:
        return

    # servo -> the follower joints it feeds, as primary source or blended in
    drives: Dict[str, list] = {motor: [] for motor in STARAI_MOTORS}
    for joint, motor in mapping["joint_map"].items():
        drives.setdefault(motor, []).append(joint)
    for joint, sources in mapping.get("blend", {}).items():
        for source in sources:
            drives.setdefault(source, []).append(f"{joint} (blended)")

    stop = _wait_for_enter()
    typer.echo("\n📋 Move ONE leader joint at a time through its range. Press ENTER to stop.\n")

    mins: Dict[str, float] = {}
    maxes: Dict[str, float] = {}

    try:
        with Live(console=console, refresh_per_second=8, transient=True) as live:
            while not stop.is_set():
                try:
                    counts = leader._inner.bus.sync_read("Present_Position", normalize=False)
                except Exception as e:
                    live.update(f"[red]Read failed: {e}[/red]")
                    time.sleep(0.2)
                    continue

                for motor, count in counts.items():
                    if count is None:
                        continue
                    mins[motor] = min(mins.get(motor, count), count)
                    maxes[motor] = max(maxes.get(motor, count), count)

                swings = {m: maxes[m] - mins[m] for m in maxes}
                most_moved = max(swings, key=swings.get) if swings else None

                table = Table(
                    title="Which leader servo did you just move?", show_header=True
                )
                table.add_column("Leader servo", style="bold")
                table.add_column("Moved", justify="right")
                table.add_column("Drives")

                for motor in STARAI_MOTORS:
                    swing_counts = swings.get(motor, 0)
                    swing_deg = swing_counts * 360.0 / 4096.0
                    targets = drives.get(motor) or []
                    label = ", ".join(targets) if targets else "[yellow]nothing (spare joint)[/yellow]"

                    moved = f"{swing_deg:.0f}°"
                    if motor == most_moved and swing_counts > 40:
                        moved = f"[bold green]{swing_deg:.0f}°  ←[/bold green]"
                    table.add_row(motor, moved, label)

                live.update(table)
                time.sleep(0.1)
    finally:
        leader.disconnect()

    typer.echo("Total movement seen per servo:")
    for motor in STARAI_MOTORS:
        swing = (maxes.get(motor, 0) - mins.get(motor, 0)) * 360.0 / 4096.0
        targets = drives.get(motor) or []
        typer.echo(f"   {motor:<8} {swing:6.0f}°   {', '.join(targets) or 'nothing (spare joint)'}")


def _pick_joint(prompt: str = "Which SO101 joint?") -> Optional[str]:
    typer.echo("")
    for i, joint in enumerate(SO101_JOINTS, 1):
        typer.echo(f"  {i}. {joint}")
    choice = Prompt.ask(prompt, default="1")
    try:
        index = int(choice)
    except ValueError:
        return None
    if 1 <= index <= len(SO101_JOINTS):
        return SO101_JOINTS[index - 1]
    return None


def tune_starai_map(config: dict) -> None:
    """Interactive editor for the Star Arm 102 → SO101 mapping."""
    mapping = load_starai_map()
    port, leader_id, follower_id = _resolve_leader(config)

    if follower_id:
        limits = get_follower_joint_limits(follower_id)
        if limits:
            typer.echo(f"🛡️  Clamping to follower '{follower_id}' travel limits.")
        else:
            typer.echo(
                f"⚠️  No calibration found for follower '{follower_id}' — commands will not be clamped."
            )

    dirty = False

    while True:
        describe_map(mapping, saved=not dirty)

        typer.echo("\n🔧 Star Arm 102 tuning")
        typer.echo("  1. Identify leader joints (move one at a time — names each servo)")
        typer.echo("  2. Live mapping view (leader reading → follower command)")
        typer.echo("  3. Flip a joint's direction")
        typer.echo("  4. Set a joint's gain")
        typer.echo("  5. Set a joint's offset (degrees)")
        typer.echo("  6. Reassign which leader servo drives a joint")
        typer.echo("  7. Blend an extra leader servo into a joint")
        typer.echo("  8. Per-step motion cap")
        typer.echo("  9. Smoothing")
        typer.echo("  a. Auto-fit gains so the leader's sweep covers the follower's travel")
        typer.echo("  0. Reset to defaults")
        typer.echo("  s. Save and exit")
        typer.echo("  q. Exit without saving")

        choice = Prompt.ask("Choose", default="1").strip().lower()

        if choice == "1":
            if not port:
                typer.echo("❌ No leader port.")
                continue
            _identify_view(port, leader_id, follower_id, mapping)

        elif choice == "2":
            if not port:
                typer.echo("❌ No leader port.")
                continue
            _live_view(port, leader_id, follower_id, mapping)

        elif choice == "3":
            joint = _pick_joint("Flip which joint?")
            if joint:
                mapping["signs"][joint] = -1.0 * mapping["signs"].get(joint, 1.0)
                typer.echo(f"✅ {joint} direction is now {mapping['signs'][joint]:+.0f}")
                dirty = True

        elif choice == "4":
            joint = _pick_joint("Set the gain on which joint?")
            if joint:
                current = mapping["gains"].get(joint, 1.0)
                value = Prompt.ask(
                    f"{joint} gain (1.0 = follower moves as far as the leader)",
                    default=str(current),
                )
                try:
                    mapping["gains"][joint] = float(value)
                    dirty = True
                except ValueError:
                    typer.echo("⚠️  Not a number.")

        elif choice == "5":
            joint = _pick_joint("Set the offset on which joint?")
            if joint:
                current = mapping["offsets"].get(joint, 0.0)
                value = Prompt.ask(f"{joint} offset in degrees", default=str(current))
                try:
                    mapping["offsets"][joint] = float(value)
                    dirty = True
                except ValueError:
                    typer.echo("⚠️  Not a number.")

        elif choice == "6":
            joint = _pick_joint("Reassign which joint?")
            if joint:
                typer.echo("")
                for i, motor in enumerate(STARAI_MOTORS, 1):
                    typer.echo(f"  {i}. {motor}")
                typer.echo(f"  {len(STARAI_MOTORS) + 1}. (leave unmapped)")
                pick = Prompt.ask("Driven by which leader servo?", default="1")
                try:
                    index = int(pick)
                except ValueError:
                    continue
                if index == len(STARAI_MOTORS) + 1:
                    mapping["joint_map"].pop(joint, None)
                    typer.echo(f"✅ {joint} is now unmapped (the follower will not move it)")
                    dirty = True
                elif 1 <= index <= len(STARAI_MOTORS):
                    mapping["joint_map"][joint] = STARAI_MOTORS[index - 1]
                    typer.echo(f"✅ {joint} ← {STARAI_MOTORS[index - 1]}")
                    dirty = True

        elif choice == "7":
            joint = _pick_joint("Blend an extra servo into which joint?")
            if joint:
                if joint == "gripper":
                    typer.echo("⚠️  The gripper is a single 0-100 axis; blending does not apply.")
                    continue

                blended = mapping.setdefault("blend", {}).setdefault(joint, {})
                primary = mapping["joint_map"].get(joint)
                typer.echo(
                    f"\n{joint} is driven by {primary or '(nothing)'}"
                    + (f", plus {', '.join(f'{w:+.2f}×{src}' for src, w in sorted(blended.items()))}" if blended else "")
                )
                typer.echo("")
                for i, motor in enumerate(STARAI_MOTORS[:6], 1):
                    typer.echo(f"  {i}. {motor}")
                pick = Prompt.ask("Add which leader servo?", default="4")
                try:
                    index = int(pick)
                except ValueError:
                    continue
                if not 1 <= index <= 6:
                    continue

                source = STARAI_MOTORS[index - 1]
                if source == primary:
                    typer.echo(f"⚠️  {source} already drives {joint} directly.")
                    continue

                typer.echo(
                    "\nWeight: 1.0 adds this servo's motion to the joint, -1.0 subtracts it"
                    "\n(use -1.0 if the two axes turn opposite ways), 0 removes the blend."
                )
                value = Prompt.ask(f"Weight for {source}", default=str(blended.get(source, 1.0)))
                try:
                    weight = float(value)
                except ValueError:
                    typer.echo("⚠️  Not a number.")
                    continue

                if weight == 0:
                    blended.pop(source, None)
                    if not blended:
                        mapping["blend"].pop(joint, None)
                    typer.echo(f"✅ {source} no longer feeds {joint}")
                else:
                    blended[source] = weight
                    typer.echo(f"✅ {joint} ← {primary} {weight:+.2f}×{source}")
                dirty = True

        elif choice == "8":
            current = mapping.get("max_relative_target") or 0
            typer.echo(
                "\nThis caps how far the follower may move in one control step. A low value"
                "\nmakes a mapping mistake a slow drift instead of a lunge; 0 turns it off."
            )
            value = Prompt.ask("Per-step cap in degrees (0 = off)", default=str(current))
            try:
                cap = float(value)
                mapping["max_relative_target"] = None if cap <= 0 else cap
                dirty = True
            except ValueError:
                typer.echo("⚠️  Not a number.")

        elif choice == "9":
            current = mapping.get("smoothing", 0.0)
            value = Prompt.ask("Smoothing, 0.0-0.9 (0 = off)", default=str(current))
            try:
                smoothing = float(value)
                if 0.0 <= smoothing < 1.0:
                    mapping["smoothing"] = smoothing
                    dirty = True
                else:
                    typer.echo("⚠️  Must be between 0.0 and 0.9.")
            except ValueError:
                typer.echo("⚠️  Not a number.")

        elif choice == "a":
            suggested = suggest_gains(leader_id, follower_id, mapping)
            if not suggested:
                typer.echo(
                    "⚠️  Needs both arms calibrated — the leader as "
                    f"'{leader_id}' and the follower as '{follower_id or '(unset)'}'."
                )
                continue

            spans = get_leader_joint_spans(leader_id)
            limits = get_follower_joint_limits(follower_id)

            table = Table(title="Calibrated travel", show_header=True)
            table.add_column("Joint", style="bold")
            table.add_column("Leader", justify="right")
            table.add_column("Follower", justify="right")
            table.add_column("Gain now", justify="right")
            table.add_column("Suggested", justify="right")

            for joint, gain in suggested.items():
                motor = mapping["joint_map"].get(joint, "")
                leader_span = spans.get(motor, 0.0)
                for source, weight in mapping.get("blend", {}).get(joint, {}).items():
                    leader_span += abs(weight) * spans.get(source, 0.0)
                follower_span = limits[joint][1] - limits[joint][0]
                table.add_row(
                    joint,
                    f"{leader_span:.0f}°",
                    f"{follower_span:.0f}°",
                    f"{mapping['gains'].get(joint, 1.0):.2f}",
                    f"[bold]{gain:.2f}[/bold]",
                )
            console.print(table)

            stretched = [j for j, g in suggested.items() if g > 2.0]
            if stretched:
                typer.echo(
                    f"\n⚠️  {', '.join(stretched)} would need a gain above 2 — that usually means"
                    "\n   the joint was not swept through its full travel during leader calibration."
                    "\n   Re-running 'solo robo --calibrate leader' is the better fix; a large gain"
                    "\n   magnifies hand tremor and encoder noise along with the motion."
                )

            if Confirm.ask("\nApply these gains?", default=not stretched):
                mapping["gains"].update(suggested)
                dirty = True
                typer.echo("✅ Gains updated.")

        elif choice == "0":
            if Confirm.ask("Reset the whole mapping to defaults?", default=False):
                import json
                mapping = json.loads(json.dumps(DEFAULT_STARAI_MAP))
                dirty = True

        elif choice == "s":
            save_starai_map(mapping)
            typer.echo("🎮 Run 'solo robo --teleop' to try it.")
            return

        elif choice == "q":
            if dirty and not Confirm.ask("Discard your changes?", default=False):
                continue
            typer.echo("No changes saved.")
            return
