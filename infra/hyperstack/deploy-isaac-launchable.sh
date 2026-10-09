#!/usr/bin/env bash
# Run on the Hyperstack VM AFTER bootstrap.sh + reboot. Clones NVIDIA's
# isaac-launchable repo, applies the override patch in this directory, and
# brings up the VSCode + Isaac Sim + web-viewer stack.
#
# Usage: VSCODE_PASSWORD=yourpass bash deploy-isaac-launchable.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/isaac-launchable}"
VSCODE_PASSWORD="${VSCODE_PASSWORD:?Set VSCODE_PASSWORD before running}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "${REPO_DIR}" ]; then
  echo "==> ${REPO_DIR} already exists, pulling latest"
  git -C "${REPO_DIR}" pull --ff-only
else
  git clone https://github.com/isaac-sim/isaac-launchable "${REPO_DIR}"
fi

cp "${SCRIPT_DIR}/docker-compose.override.yml" "${REPO_DIR}/isaac-lab/docker-compose.override.yml"
echo "VSCODE_PASSWORD=${VSCODE_PASSWORD}" > "${REPO_DIR}/isaac-lab/.env"

cd "${REPO_DIR}/isaac-lab"
sudo -E docker compose up -d --build

echo "==> Containers:"
sudo docker ps --filter "name=vscode" --filter "name=nginx" --filter "name=web-viewer"

echo "==> Done. Open http://<floating-ip>/ for VSCode (password above) and"
echo "    http://<floating-ip>/viewer/ for the Isaac Sim stream, once you've"
echo "    launched Isaac Sim inside the vscode container, e.g.:"
echo "    docker exec -d vscode bash -c 'cd /isaac-sim && ACCEPT_EULA=Y nohup ./runheadless.sh > /tmp/isaac.log 2>&1 &'"
echo "    First launch takes ~5-6 min (one-time shader/material cache compile)."
