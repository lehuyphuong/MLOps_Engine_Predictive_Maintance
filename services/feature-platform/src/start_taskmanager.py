#!/usr/bin/env python3
"""
start_taskmanager.py

Resolves FLINK_HOME and starts the TaskManager with the correct FQDN
that the JobManager can resolve via kube-dns.

StatefulSet pod naming:
  Pod names are ordinal: flink-taskmanager-0, flink-taskmanager-1, ...
  Combined with the headless Service (flink-taskmanager-hl) and namespace,
  kube-dns registers:
    flink-taskmanager-0.flink-taskmanager-hl.feature-platform.svc.cluster.local

  This FQDN resolves correctly from the JobManager pod because StatefulSet
  pods (unlike Deployment pods) are registered in kube-dns via the headless
  Service + serviceName binding.

  HOSTNAME is set to the pod name by Kubernetes automatically
  (e.g. flink-taskmanager-0).
"""
import os
from pyflink.find_flink_home import _find_flink_home

flink_home = _find_flink_home()
os.environ["FLINK_HOME"] = flink_home

conf_path = os.path.join(flink_home, "conf", "flink-conf.yaml")
rpc_addr  = os.environ["JOB_MANAGER_RPC_ADDRESS"]

# StatefulSet pod name is ordinal and stable: flink-taskmanager-0
pod_name  = os.environ.get("HOSTNAME", "flink-taskmanager-0")
subdomain = "flink-taskmanager-hl"
namespace = "feature-platform"
pod_fqdn  = f"{pod_name}.{subdomain}.{namespace}.svc.cluster.local"

print(f"start_taskmanager.py — pod_name={pod_name} fqdn={pod_fqdn}")

overrides = {
    "jobmanager.rpc.address": os.environ.get("FLINK_JM_RPC_ADDRESS",  rpc_addr),
    "taskmanager.bind-host":  os.environ.get("FLINK_TM_BIND_HOST",    "0.0.0.0"),
    "taskmanager.host":       os.environ.get("FLINK_TM_HOST",         pod_fqdn),
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

print(f"start_taskmanager.py — patched flink-conf.yaml: {overrides}")

taskmanager_sh = os.path.join(flink_home, "bin", "taskmanager.sh")
print(f"Executing: {taskmanager_sh} start-foreground")
os.execv("/bin/bash", ["/bin/bash", taskmanager_sh, "start-foreground"])