#!/usr/bin/env bash
# Launch a Hyperstack GPU VM suitable for Isaac Sim, via the Hyperstack API.
# Reads the API key from ~/.solo/config.json (key: hyperstack.api_key), same
# place solo-cli itself stores it.
#
# Usage: ./launch-vm.sh [flavor_name] [vm_name]
#   flavor_name defaults to n3-L40x1 (confirmed working). n3-RTX-A6000x1 also
#   works when in stock - check first with:
#     curl -s -H "api_key: $KEY" "https://infrahub-api.nexgencloud.com/v1/core/flavors?region=CANADA-1"
set -euo pipefail

FLAVOR="${1:-n3-L40x1}"
VM_NAME="${2:-isaac-sim-$(date +%s)}"
ENVIRONMENT_NAME="${HYPERSTACK_ENV:-solo-cli-env}"
KEYPAIR_NAME="${HYPERSTACK_KEYPAIR:-solo-cli-mac}"
IMAGE_NAME="${HYPERSTACK_IMAGE:-Ubuntu Server 22.04 LTS R535 CUDA 12.2 with Docker}"

API_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.solo/config.json'))['hyperstack']['api_key'])")

echo "==> Launching ${FLAVOR} as '${VM_NAME}' in ${ENVIRONMENT_NAME}..."
RESP=$(curl -s -X POST -H "api_key: ${API_KEY}" -H "Content-Type: application/json" \
  https://infrahub-api.nexgencloud.com/v1/core/virtual-machines \
  -d "{
    \"name\": \"${VM_NAME}\",
    \"environment_name\": \"${ENVIRONMENT_NAME}\",
    \"image_name\": \"${IMAGE_NAME}\",
    \"flavor_name\": \"${FLAVOR}\",
    \"key_name\": \"${KEYPAIR_NAME}\",
    \"count\": 1,
    \"assign_floating_ip\": true,
    \"security_rules\": [
      {\"direction\": \"ingress\", \"protocol\": \"tcp\", \"ethertype\": \"IPv4\", \"port_range_min\": 22, \"port_range_max\": 22, \"remote_ip_prefix\": \"0.0.0.0/0\"}
    ]
  }")

echo "${RESP}" | python3 -m json.tool

VM_ID=$(echo "${RESP}" | python3 -c "import json,sys; print(json.load(sys.stdin)['instances'][0]['id'])" 2>/dev/null) || {
  echo "!! Launch failed (see response above - likely out of stock, try a different flavor/region or 'Not enough credit')." >&2
  exit 1
}

echo "==> VM id: ${VM_ID}"
echo "==> Polling for ACTIVE + floating IP..."
for i in $(seq 1 40); do
  INFO=$(curl -s -H "api_key: ${API_KEY}" "https://infrahub-api.nexgencloud.com/v1/core/virtual-machines/${VM_ID}")
  ST=$(echo "${INFO}" | python3 -c "import json,sys; print(json.load(sys.stdin)['instance']['status'])")
  echo "  [$i] status=${ST}"
  if [ "${ST}" = "ACTIVE" ]; then
    IP=$(echo "${INFO}" | python3 -c "import json,sys; print(json.load(sys.stdin)['instance'].get('floating_ip') or '')")
    if [ -n "${IP}" ]; then
      echo "==> Ready. VM id=${VM_ID} floating_ip=${IP}"
      echo "    Next: ./open-firewall.sh ${VM_ID}"
      echo "    Then: ssh -i ~/.ssh/id_ed25519_runpod ubuntu@${IP}"
      exit 0
    fi
  fi
  sleep 15
done
echo "!! Timed out waiting for VM to become ACTIVE with a floating IP." >&2
exit 1
