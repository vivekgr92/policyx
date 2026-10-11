# SO-101 assets — vendored copy

**Source**: [google-deepmind/mujoco_menagerie — `robotstudio_so101/`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
**License**: Apache 2.0 (see `LICENSE`)
**Upstream upstream**: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101), commit `aec17bbc256d1a7342d53aaa4950595d4c30b40d`
**Vendored on**: 2026-04-25

## Why vendored (not submodule / runtime download)

- Offline reproducibility (NFR2 in `PRD.md`: clone → first episode ≤ 30 min).
- Asset set is small (~20 STLs + 3 XMLs).
- Upstream is stable; we can re-pull and diff manually when we want.

## Upstream refresh

```bash
BASE=https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotstudio_so101
curl -sSfLo so101.xml "$BASE/so101.xml"
curl -sSfLo scene.xml "$BASE/scene.xml"
# …etc (see scripts if we automate this later)
```

## What Menagerie already did for us

Per Menagerie's README, the vendored MJCF includes:
- Primitive collision geometries for gripper + arm (PRD R1 mitigation).
- Tuned collision solver params for manipulation.
- A camera mount (`wrist_cam` body in `so101.xml`).

## User additions in this directory (NOT from Menagerie)

These files live next to the vendored MJCF so relative paths (includes,
meshdir) resolve identically — MuJoCo anchors nested paths to the
top-level file's directory, so keeping siblings avoids path-juggling.

- `scene_pick_place.xml` — pick-and-place task scene: arm + ground +
  static target container + free-joint graspable cube. Authored here.
