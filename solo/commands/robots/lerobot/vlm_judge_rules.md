# VLM Judge - House Rules

Editable judging criteria for the VLM episode-quality judge (`vlm_judge.py`).
Edit this file to tune what counts as a valid episode - it's loaded fresh on
every judge call, no code change or restart needed.

These rules are checked IN ADDITION to the episode's own task description
(e.g. "Pick A to B") - a verdict should be INVALID if the task wasn't
completed OR if any of these are violated, even if the task direction looks
right.

- **Task completion**: the object was actually picked up, carried to the
  correct target zone, and released there - not just nudged, grazed, or
  partially moved.
- **Gripper ends empty**: by the final frame, the gripper has released the
  object and is not still holding/pinching it.
- **Object placement is upright, not flipped**: the cup (or other object)
  should be right-side-up at the end, not knocked over, tipped, or flipped
  on its side.
- **No drop-and-abandon**: if the object was dropped mid-transit and left
  where it fell (not at the target zone), that's a failure even if the
  gripper ends empty.
- **No collision/knock-over of other objects**: other cups/objects in the
  scene should not have been bumped out of place as a side effect.
