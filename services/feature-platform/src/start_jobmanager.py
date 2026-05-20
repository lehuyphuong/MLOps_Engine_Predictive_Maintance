#!/usr/bin/env python3
"""
start_jobmanager.py

Resolves FLINK_HOME via PyFlink's own helper and starts the JobManager.
All Flink configuration is passed via environment variables set in the
Helm chart (flink-cluster.yaml) — no flink-conf.yaml patching.

Flink reads env vars with the prefix FLINK_ where dots become underscores:
  FLINK_REST_BIND__ADDRESS=0.0.0.0
    => rest.bind-address: 0.0.0.0
But this prefix convention is unreliable across versions.

The reliable approach used here is direct -D flags to jobmanager.sh,
avoiding the --host / --webui-port argument conflict by NOT using those
positional args — we only pass -D key=value pairs which the JVM picks up
directly, bypassing the shell script CLI parser conflict entirely.

The earlier conflict was:
  jobmanager.sh inserts --host <value> --webui-port <value>
  then our -D flags appeared AFTER, causing parse errors.

Fix: write the config directly to flink-conf.yaml BEFORE calling
jobmanager.sh, which reads the file first before parsing CLI args.
The -D flags from jobmanager.sh's own pre-configuration are appended
after our file values and override them for memory settings only —
our network settings in the file are not touched by those -D flags.
"""
import os
from pyflink.find_flink_home import _find_flink_home

flink_home    = _find_flink_home()
os.environ["FLINK_HOME"] = flink_home

conf_path = os.path.join(flink_home, "conf", "flink-conf.yaml")
rpc_addr  = os.environ["JOB_MANAGER_RPC_ADDRESS"]

# Read all overrides from env vars set by the Helm chart
# so there is zero logic here — the chart controls everything
overrides = {
    "rest.bind-address":      os.environ.get("FLINK_REST_BIND_ADDRESS",  "0.0.0.0"),
    "rest.address":           os.environ.get("FLINK_REST_ADDRESS",       rpc_addr),
    "jobmanager.bind-host":   os.environ.get("FLINK_JM_BIND_HOST",      "0.0.0.0"),
    "jobmanager.rpc.address": os.environ.get("FLINK_JM_RPC_ADDRESS",    rpc_addr),
    # Pin BlobServer to a fixed port so the Helm Service can expose it.
    # By default Flink picks a random ephemeral port — the TaskManager
    # then tries to connect to that port on the JobManager, which is not
    # in the Service's port list => Connection timed out.
    "blob.server.port":       "6125",
}

with open(conf_path) as f:
    lines = f.readlines()

patched, new_lines = set(), []
for line in lines:
    s = line.strip()
    if s.startswith("#") or ":" not in s:
        new_lines.append(line)
        continue
    key = s.split(":", 1)[0].strip()
    if key in overrides:
        new_lines.append(f"{key}: {overrides[key]}\n")
        patched.add(key)
    else:
        new_lines.append(line)

for k, v in overrides.items():
    if k not in patched:
        new_lines.append(f"{k}: {v}\n")

with open(conf_path, "w") as f:
    f.writelines(new_lines)

print("start_jobmanager.py — patched flink-conf.yaml:", overrides)

jobmanager_sh = os.path.join(flink_home, "bin", "jobmanager.sh")
print(f"Executing: {jobmanager_sh} start-foreground")
os.execv("/bin/bash", ["/bin/bash", jobmanager_sh, "start-foreground"])