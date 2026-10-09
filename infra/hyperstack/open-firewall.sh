#!/usr/bin/env bash
# Open the real public ports Isaac Sim's WebRTC viewer needs on a Hyperstack
# VM's security group. The signaling channel can ride over SSH-tunneled port
# 80, but the actual WebRTC media is UDP and needs genuine public reachability
# - see docker-compose.override.yml for the matching client-side config.
#
# Usage: ./open-firewall.sh <vm_id>
set -euo pipefail

VM_ID="${1:?Usage: open-firewall.sh <vm_id>}"
API_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.solo/config.json'))['hyperstack']['api_key'])")

# (port_min, port_max, protocol)
RULES=(
  "80 80 tcp"
  "1024 1024 tcp"
  "47998 47998 tcp"
  "49100 49100 tcp"
  "47998 47998 udp"
  "1024 1024 udp"
)

for rule in "${RULES[@]}"; do
  read -r min max proto <<< "${rule}"
  echo "==> Opening ${proto} ${min}-${max}..."
  curl -s -X POST -H "api_key: ${API_KEY}" -H "Content-Type: application/json" \
    "https://infrahub-api.nexgencloud.com/v1/core/virtual-machines/${VM_ID}/sg-rules" \
    -d "{\"direction\":\"ingress\",\"protocol\":\"${proto}\",\"ethertype\":\"IPv4\",\"port_range_min\":${min},\"port_range_max\":${max},\"remote_ip_prefix\":\"0.0.0.0/0\"}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(' ', d.get('message', d))"
done

echo "==> Note: these rules allow 0.0.0.0/0 (the whole internet) on the viewer"
echo "    port and the WebRTC ports. The viewer page has no login. Fine for a"
echo "    short-lived test VM; tighten remote_ip_prefix if it'll run longer."
