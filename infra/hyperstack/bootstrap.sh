#!/usr/bin/env bash
# Run on a fresh Hyperstack VM (tested on "Ubuntu Server 22.04 LTS R535 CUDA 12.2
# with Docker") to bring the NVIDIA driver and Docker GPU runtime up to what
# Isaac Sim 6.0.1 actually requires. Idempotent - safe to re-run.
#
# Real gotchas this encodes (found the hard way, see ../../CLAUDE.md sim_augmentation
# section and README.md in this directory):
#   - Isaac Sim 6.0.1 hard-rejects driver < 550.90.07 ("rtx driver verification
#     failed"). Hyperstack's stock image ships 535.183.06.
#   - Installing nvidia-driver-580 on top of the stock image fails with a dpkg
#     file conflict (libnvidia-gl-580 vs libnvidia-extra-535) unless the old
#     driver's leftover packages are purged first.
#   - Docker doesn't know about the nvidia runtime/CDI device injection until
#     nvidia-ctk configures it - needed for Isaac Sim's own container, not just
#     bare test containers.
#
# Usage: ssh onto the VM, then: bash bootstrap.sh
set -euo pipefail

TARGET_DRIVER_SERIES="${TARGET_DRIVER_SERIES:-580}"

echo "==> Current driver (if any):"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "(none loaded yet)"

current_series=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | cut -d. -f1 || echo 0)

if [ "${current_series}" -ge "${TARGET_DRIVER_SERIES}" ] 2>/dev/null; then
  echo "==> Driver series ${current_series} already >= ${TARGET_DRIVER_SERIES}, skipping driver install."
else
  echo "==> Installing nvidia-driver-${TARGET_DRIVER_SERIES}..."
  sudo apt-get update -qq

  echo "==> Purging any older NVIDIA driver series to avoid dpkg file conflicts..."
  old_pkgs=$(dpkg -l | awk '/^ii/{print $2}' | grep -E "^(libnvidia|nvidia)-.*-[0-9]{3}(-server)?(:amd64)?$" | grep -v -- "-${TARGET_DRIVER_SERIES}" || true)
  if [ -n "${old_pkgs}" ]; then
    echo "${old_pkgs}"
    # shellcheck disable=SC2086
    sudo dpkg -P --force-all ${old_pkgs} || true
  fi

  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "nvidia-driver-${TARGET_DRIVER_SERIES}" || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y

  echo "==> Driver installed. A REBOOT IS REQUIRED before the new kernel module loads."
  echo "    Run: sudo reboot"
  echo "    Then re-run this script to finish the Docker runtime setup."
  # Only continue to the docker setup below if the kernel module already matches
  # (e.g. re-running post-reboot); otherwise stop here so the caller reboots.
  loaded_version=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "")
  installed_version=$(dpkg -l | awk -v s="${TARGET_DRIVER_SERIES}" '$2=="nvidia-driver-"s{print $3}' | cut -d- -f1)
  if [ "${loaded_version%%.*}" != "${installed_version%%.*}" ]; then
    exit 0
  fi
fi

echo "==> Registering nvidia runtime + CDI with Docker..."
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Enable Docker's native CDI feature (needed on Docker >= 25 for --device=nvidia.com/gpu=all).
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/etc/docker/daemon.json")
d = json.loads(p.read_text()) if p.exists() else {}
d.setdefault("features", {})["cdi"] = True
d["default-runtime"] = "nvidia"
p.write_text(json.dumps(d, indent=2))
PY
sudo systemctl restart docker
sleep 2

echo "==> Verifying GPU passthrough into a container..."
sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L

echo "==> Verifying Vulkan on the host (install vulkan-tools if missing)..."
if ! command -v vulkaninfo >/dev/null; then
  sudo apt-get install -y -qq vulkan-tools
fi
vulkaninfo --summary 2>&1 | sed -n '/^Devices:/,/^$/p'

echo "==> bootstrap.sh done. Driver + Docker GPU runtime are ready for Isaac Sim."
