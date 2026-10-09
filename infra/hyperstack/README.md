# Hyperstack + Isaac Sim: repeatable setup

Scripts to spin up a Hyperstack GPU VM and get a working, browser-viewable
Isaac Sim stream on it, without redoing the manual debugging from scratch.

This came out of a real multi-cloud evaluation: Runpod has a structural
GPU-isolation bug that breaks Vulkan (see `../../CLAUDE.md`'s memory notes /
the `runpod-physx-gpu-isolation-limit` note), Lambda Cloud's stock image has a
broken Vulkan driver even after fixing the obvious gaps, and AWS's account
was blocked for all EC2 launches. Hyperstack is the one that actually worked,
with three real, fixable gotchas - encoded in these scripts so they don't
need rediscovering.

## Quickstart

```bash
# 1. Launch a VM (defaults to the L40 flavor, which is in stock more often
#    than the A6000; both work for Isaac Sim's RTX renderer).
./launch-vm.sh n3-L40x1

# 2. Open the WebRTC/viewer ports (takes the vm_id launch-vm.sh printed).
./open-firewall.sh <vm_id>

# 3. SSH in and bring the driver + Docker GPU runtime up to spec.
ssh -i ~/.ssh/id_ed25519_runpod ubuntu@<floating_ip>
# on the VM:
bash bootstrap.sh
sudo reboot
# after it comes back:
bash bootstrap.sh   # finishes the Docker/CDI setup now that the new driver is loaded

# 4. Deploy Isaac Sim.
VSCODE_PASSWORD=yourpass bash deploy-isaac-launchable.sh

# 5. Launch Isaac Sim's streaming app inside the container.
sudo docker exec -d vscode bash -c \
  'cd /isaac-sim && ACCEPT_EULA=Y nohup ./runheadless.sh > /tmp/isaac.log 2>&1 &'
# first launch takes ~5-6 min (one-time shader/material cache compile) -
# tail /tmp/isaac.log inside the container and wait for "app ready".
```

Then open `http://<floating_ip>/viewer/` for the live stream and
`http://<floating_ip>/` for VSCode (the password you set above).

## The three real gotchas these scripts encode

1. **Driver version floor.** Isaac Sim 6.0.1 hard-rejects any driver below
   `550.90.07` with a clean `rtx driver verification failed` log line (not a
   crash). Hyperstack's stock "Ubuntu Server 22.04 LTS R535 CUDA 12.2 with
   Docker" image ships `535.183.06`. `bootstrap.sh` upgrades to the 580
   series. Installing the new driver directly on top of the old one fails
   with a dpkg file conflict (`libnvidia-gl-580` vs `libnvidia-extra-535`
   both claiming the same `.so`) - the script purges the old driver's
   packages first.

2. **Docker doesn't know about the GPU's graphics capability by default.**
   `--gpus all` alone gets you CUDA/compute, but not a working Vulkan ICD
   inside the container - `nvidia_icd.json` isn't mounted, `/dev/dri` isn't
   passed, and (on a *minimal* container - NVIDIA's own Isaac Sim image
   already ships this) `libegl1`/`libgl1`/`libglvnd0`/`libglx0` are often
   missing too. `bootstrap.sh` registers the `nvidia` runtime and generates a
   CDI spec so all of this is handled for you, and sets Docker's default
   runtime to `nvidia` so `isaac-launchable`'s `runtime: nvidia` compose
   setting actually resolves.

3. **The WebRTC viewer needs real public reachability, not just a tunnel.**
   The signaling/control channel can go over SSH-tunneled HTTP, but the
   actual video stream is WebRTC media (UDP) and can't be carried by a plain
   SSH `-L` port forward. `docker-compose.override.yml` switches the viewer
   to `ENV=brev` (auto-detects the VM's public IP for the media server
   config - nothing else Brev-specific about it) and `FORCE_WSS=false`
   (plain `ws://` on port 80, avoiding a self-signed-cert trust dance on
   port 443). `open-firewall.sh` opens the matching real ports (80, 1024,
   47998, 49100, TCP+UDP) to `0.0.0.0/0` - fine for a short-lived test VM,
   tighten it if the VM will run longer.

## Known open item: Isaac Lab Mimic version drift

`isaac-launchable` ships **Isaac Lab 3.0.0-beta2-post1 / isaaclab_mimic
1.3.3**. The SO-101 custom Mimic env code in
`solo/commands/robots/lerobot/sim_augmentation/` (specifically the
`cup_pickplace_mimic_env_cfg.py` / `cup_pickplace_mimic_env.py` / `cup_mdp.py`
files built on the Runpod volume) was written and validated against
**isaaclab==2.3.1 / isaaclab_mimic==1.0.16**. Not yet verified whether the
Mimic API surface shifted between those versions - check this before
assuming `DataGenerator.generate()` will run unmodified on this stack.

Cosmos (photorealistic re-rendering) hasn't been touched on this
infrastructure yet - expect it to need its own container and likely the same
driver-version floor, but unconfirmed.
