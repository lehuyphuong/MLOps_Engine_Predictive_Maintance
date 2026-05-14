#!/usr/bin/env python3
"""
start_jobmanager.py — resolves FLINK_HOME, patches flink-conf.yaml,
then starts the JobManager.

Why we patch flink-conf.yaml instead of passing -D flags to jobmanager.sh:
  jobmanager.sh internally calls the Java StandaloneSessionClusterEntrypoint
  which parses its own CLI flags first:
    -h / --host       <hostname>
    -r / --webui-port <port>
    -D                <property=value>   (repeatable)
  The shell script inserts --host and --webui-port into the argument list
  before our -D flags. The Java CLI parser then consumes the first token
  after --host as the hostname value — and that token turns out to be our
  -Drest.bind-address=0.0.0.0, which it accepts as the hostname string,
  leaving --webui-port with -Drest.address=flink-jobmanager as its port
  value → MissingArgumentException / FlinkParseException.

  Writing directly to flink-conf.yaml sidesteps CLI parsing entirely.
  The file is read before argument parsing and the -D flags from the
  memory-sizing pre-configuration script are then applied on top,
  so our network bindings are never disturbed.
"""
import os
import shutil

from pyflink.find_flink_home import _find_flink_home

flink_home  = _find_flink_home()
os.environ["FLINK_HOME"] = flink_home

conf_path = os.path.join(flink_home, "conf", "flink-conf.yaml")

rpc_address  = os.environ.get("JOB_MANAGER_RPC_ADDRESS", "flink-jobmanager")

# Properties to patch — keyed by the exact property name Flink reads.
# We overwrite any existing value for these keys and append the ones
# that are not already present.
OVERRIDES = {
    "rest.bind-address":      "0.0.0.0",
    "rest.address":           rpc_address,
    "jobmanager.bind-host":   "0.0.0.0",
    "jobmanager.rpc.address": rpc_address,
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

# Append any keys that were not already in the file
for key, val in OVERRIDES.items():
    if key not in patched_keys:
        new_lines.append(f"{key}: {val}\n")

with open(conf_path, "w") as f:
    f.writelines(new_lines)

print(f"flink-conf.yaml patched — overrides: {OVERRIDES}")

# exec jobmanager.sh start-foreground (no extra -D flags needed)
jobmanager_sh = os.path.join(flink_home, "bin", "jobmanager.sh")

os.execv("/bin/bash", ["/bin/bash", jobmanager_sh, "start-foreground"])