# Fixes a real authoring defect in the upstream SO-ARM101-USD.usd asset,
# confirmed via live USD stage inspection (pxr.Usd.Stage.Traverse +
# GetChildren, not guessed): the robot's visual meshes live under 3 separate
# TOP-LEVEL root prims (/visuals, /meshes, /colliders - parent "/", siblings
# of the default prim) instead of nested under the physics articulation's
# own link Xforms (/so101_new_calib/{base,shoulder,...}). Isaac Lab's
# UsdFileCfg spawn only brings in the default prim's own subtree, so physics
# (joints, collision meshes - which ARE correctly nested under each link)
# loads fine, but the visual meshes are structurally orphaned and never
# appear - confirmed by a black/missing-robot screenshot
# (zero_agent_screenshot.py) despite fully working joint reads and grasp
# detection. Each physics link Xform even has an empty `proxyPrim`
# relationship (USD's own standard slot for exactly this kind of link),
# confirming the exporter meant to wire this up and didn't.
#
# Fix: add an internal USD reference from each physics link Xform to its
# matching /visuals/<link> group, producing a new, separate .usd file
# (leaves the original untouched). Verified: a post-fix live scene-graph
# inspection shows real Mesh prims correctly nested under every link
# (e.g. /World/envs/env_0/Robot/base/visual_mesh/base_so101_v2/mesh).
#
# Needs pxr (USD Python bindings) - run via Isaac Sim's own python:
#   ./isaaclab.sh -p fix-robot-visual-mesh.py
# (from /workspace/isaaclab, after the workshop repo is cloned+LFS-pulled).
from pxr import Usd, Sdf
import shutil

src = "/workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd"
dst = "/workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD-visual-fix.usd"

shutil.copy(src, dst)

stage = Usd.Stage.Open(dst)
links = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]

for link in links:
    link_prim_path = f"/so101_new_calib/{link}"
    link_prim = stage.GetPrimAtPath(link_prim_path)
    if not link_prim.IsValid():
        print(f"SKIP {link}: physics prim not found")
        continue
    visual_src_path = f"/visuals/{link}"
    visual_target_path = f"{link_prim_path}/visual_mesh"
    new_prim = stage.DefinePrim(visual_target_path, "Xform")
    new_prim.GetReferences().AddInternalReference(Sdf.Path(visual_src_path))
    print(f"linked {visual_target_path} -> {visual_src_path}")

stage.GetRootLayer().Save()
print("saved", dst)
