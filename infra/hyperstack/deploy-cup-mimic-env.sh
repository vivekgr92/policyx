#!/usr/bin/env bash
# Run on the Hyperstack VM (after bootstrap.sh + deploy-isaac-launchable.sh)
# to set up the SO-101 cup pick-place Isaac Mimic task for real testing.
#
# Clones NVIDIA's public isaac-sim/Sim-to-Real-SO-101-Workshop repo (the
# robot/scene Mimic simulates against), applies the Isaac Sim 6.0.1
# compat patch in this directory, and drops in the committed env files
# from solo/commands/robots/lerobot/sim_augmentation/isaac_lab_envs/.
#
# Real gotchas this encodes:
#   - isaacsim.core.prims / isaacsim.core.utils (what this repo was
#     originally written against, isaaclab==2.3.1 era) are confirmed
#     fully removed from Isaac Sim 6.0.1's Kit extension registry -
#     sim-to-real-so101-workshop-isaac-sim-6-compat.patch swaps every
#     call site to the current isaacsim.core.experimental.* API.
#   - isaacsim.core.experimental.prims exists but isn't in the default
#     app's enabled-extension set - pass --enable isaacsim.core.experimental.prims
#     to isaaclab.sh/generate_dataset.py, confirmed required even after
#     the patch (isaacsim.core.experimental.utils loads automatically,
#     .prims does not).
#   - isaaclab 3.0.0-beta2's isaaclab_tasks.utils.import_packages only
#     auto-imports __init__.py packages, not bare .py modules - our
#     gym.register() call lives at the bottom of
#     cup_pickplace_mimic_env_cfg.py itself and needs an external
#     callback (register_cup_task.py, written by this script) to
#     explicitly import that module and trigger it.
#
# Usage (inside the vscode container, or via docker exec from the host):
#   PYTHONPATH=/workspace/Sim-to-Real-SO-101-Workshop/source:$PYTHONPATH \
#     ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
#     --task Lerobot-So101-Cup-PickPlace-Mimic \
#     --input_file <source>.hdf5 --output_file <out>.hdf5 \
#     --external_callback register_cup_task.register \
#     --enable isaacsim.core.experimental.prims --headless
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/workspace/Sim-to-Real-SO-101-Workshop}"
SOLO_CLI_ENVS_DIR="${SOLO_CLI_ENVS_DIR:?Set to the local path of solo-cli's isaac_lab_envs/ directory}"

if [ -d "${REPO_DIR}" ]; then
  echo "==> ${REPO_DIR} already exists, leaving as-is (delete it first to re-clone)"
else
  git clone --depth 1 https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop "${REPO_DIR}"
fi

# The robot/scene USD assets are Git LFS objects. A plain clone only fetches
# their tiny text pointer files (~130 bytes) - Isaac Sim will spawn those as
# the robot without erroring, then fail much later and confusingly with
# "Expected exactly one ArticulationRootAPI prim ... found 0" once it
# actually tries to initialize physics on the (effectively empty) asset.
# Confirmed live: this is not an Isaac Sim 6 API drift issue, just missing
# LFS content.
command -v git-lfs >/dev/null || (apt-get update -qq && apt-get install -y -qq git-lfs)
git -C "${REPO_DIR}" lfs install
git -C "${REPO_DIR}" lfs pull

echo "==> Applying Isaac Sim 6.0.1 compat patch..."
git -C "${REPO_DIR}" apply "${SCRIPT_DIR}/sim-to-real-so101-workshop-isaac-sim-6-compat.patch"

echo "==> Fixing the robot's visual mesh (real authoring defect in the"
echo "    upstream USD - see fix-robot-visual-mesh.py's header for details)..."
cp "${SCRIPT_DIR}/fix-robot-visual-mesh.py" /workspace/isaaclab/fix-robot-visual-mesh.py
(cd /workspace/isaaclab && ./isaaclab.sh -p fix-robot-visual-mesh.py)
sed -i 's|usd_path=f"{here}/usd/SO-ARM101-USD.usd"|usd_path=f"{here}/usd/SO-ARM101-USD-visual-fix.usd"|' \
  "${REPO_DIR}/source/sim_to_real_so101/assets/so101.py"

echo "==> Dropping in the cup pick-place env files..."
cp "${SOLO_CLI_ENVS_DIR}/cup_pickplace_mimic_env.py" "${REPO_DIR}/source/sim_to_real_so101/tasks/"
cp "${SOLO_CLI_ENVS_DIR}/cup_pickplace_mimic_env_cfg.py" "${REPO_DIR}/source/sim_to_real_so101/tasks/"
cp "${SOLO_CLI_ENVS_DIR}/cup_mdp.py" "${REPO_DIR}/source/sim_to_real_so101/mdp/"

cat > register_cup_task.py << 'PYEOF'
"""External callback for generate_dataset.py --external_callback.

Directly imports the cup task's env cfg module (rather than relying on
isaaclab_tasks.utils.import_packages' package auto-walk, which - as of
isaaclab 3.0.0-beta2 - only imports __init__.py packages, not bare .py
modules, so the gym.register() call at the bottom of
cup_pickplace_mimic_env_cfg.py would never fire via that path).
"""


def register():
    import sim_to_real_so101.tasks.cup_pickplace_mimic_env_cfg  # noqa: F401

    return []
PYEOF

echo "==> Done. register_cup_task.py written to $(pwd)/register_cup_task.py"
echo "    Run from /workspace/isaaclab with PYTHONPATH=${REPO_DIR}/source:\$PYTHONPATH,"
echo "    and pass --enable isaacsim.core.experimental.prims --external_callback register_cup_task.register"
echo "    to generate_dataset.py (see this script's header for the full command)."
