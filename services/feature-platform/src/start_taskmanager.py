#!/usr/bin/env python3
"""
start_taskmanager.py — resolves FLINK_HOME, patches flink-conf.yaml,
then starts the TaskManager.

Same approach as start_jobmanager.py — see that file for the full explanation
of why we patch flink-conf.yaml instead of passing -D flags on the CLI.
"""
import os

from pyflink.find_flink_home import _find_flink_home

flink_home = _find_flink_home()
os.environ["FLINK_HOME"] = flink_home

conf_path = os.path.join(flink_home, "conf", "flink-conf.yaml")

rpc_address  = os.environ.get("JOB_MANAGER_RPC_ADDRESS", "flink-jobmanager")
pod_hostname = os.environ.get("HOSTNAME", "localhost")

OVERRIDES = {
    "jobmanager.rpc.address":  rpc_address,
    "taskmanager.bind-host":   "0.0.0.0",
    "taskmanager.host":        pod_hostname,
}

# patch flink-conf.yaml
with open(conf_path, "r") as f:
    lines = f.readlines()

patched_keys = set()
new_lines    = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("#") or ":" not in stripped:
        new_lines.append(line)
        continue
    key = stripped.split(":", 1)[0].strip()
    if key in OVERRIDES:
        new_lines.append(f"{key}: {OVERRIDES[key]}\n")
        patched_keys.add(key)
    else:
        new_lines.append(line)

for key, val in OVERRIDES.items():
    if key not in patched_keys:
        new_lines.append(f"{key}: {val}\n")

with open(conf_path, "w") as f:
    f.writelines(new_lines)

print(f"flink-conf.yaml patched — overrides: {OVERRIDES}")

# exec taskmanager.sh start-foreground
taskmanager_sh = os.path.join(flink_home, "bin", "taskmanager.sh")

os.execv("/bin/bash", ["/bin/bash", taskmanager_sh, "start-foreground"])