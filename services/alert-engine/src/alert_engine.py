"""
alert_engine.py — Alert Rule Engine
Polls GCS inference-results/ for new engine_id · RUL · anomaly_score · anomaly_flag events.
Evaluates two alert rules:
  Rule 1 — RUL threshold  : RUL < RUL_ALERT_THRESHOLD (default 20 cycles)
  Rule 2 — Anomaly flag   : anomaly_flag == True

For each triggered rule:
  - Writes alert row to PostgreSQL `alerts` table (replaces Elasticsearch)
  - Logs alert with severity level

Runs as a Kubernetes CronJob (every 5 min, after model-serving writes results).

Environment variables (set via configmap + secret):
  GCS_RAW_BUCKET         phm-raw-data-aide2-494008
  DATASET                FD002
  GCP_PROJECT            aide2-494008
  RUL_ALERT_THRESHOLD    20
  DB_HOST                Cloud SQL private IP — terraform output db_private_ip
  DB_PORT                5432
  DB_NAME                phmdb
  DB_USER                phmadmin
  DB_PASSWORD            (from alert-engine-secrets)

Notifier environment variables (from alert-engine-secrets):
  SENDGRID_API_KEY       SendGrid API key for email delivery
  ALERT_FROM_EMAIL       sender address  e.g. phm-alerts@yourdomain.com
  ALERT_TO_EMAIL         recipient address e.g. engineer@yourdomain.com
  NOTIFY_ENABLED         true | false  (default: true)
  NOTIFY_SEVERITY        critical | warning | all  (default: critical)
                         critical -> only RUL critical alerts
                         warning  -> critical + warning + anomaly
                         all      -> every alert
"""

import os
import json
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from google.cloud import storage
from google.cloud.exceptions import NotFound

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_RAW_BUCKET      = os.environ.get("GCS_RAW_BUCKET",         "phm-raw-data-aide2-494008")
DATASET             = os.environ.get("DATASET",                 "FD002")
GCP_PROJECT         = os.environ.get("GCP_PROJECT",             "aide2-494008")
RUL_ALERT_THRESHOLD = int(os.environ.get("RUL_ALERT_THRESHOLD", "20"))

DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_PORT     = int(os.environ.get("DB_PORT", "5432"))
DB_NAME     = os.environ.get("DB_NAME",     "phmdb")
DB_USER     = os.environ.get("DB_USER",     "phmadmin")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

RESULTS_PREFIX = f"inference-results/{DATASET}/"
CHECKPOINT_KEY = f"checkpoints/{DATASET}/alert_processed_keys.json"

# ---------------------------------------------------------------------------
# Notifier config
# ---------------------------------------------------------------------------
SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY",   "")
ALERT_FROM_EMAIL  = os.environ.get("ALERT_FROM_EMAIL",   "")
ALERT_TO_EMAIL    = os.environ.get("ALERT_TO_EMAIL",     "")
NOTIFY_ENABLED    = os.environ.get("NOTIFY_ENABLED",     "true").lower() == "true"
NOTIFY_SEVERITY   = os.environ.get("NOTIFY_SEVERITY",    "critical")  # critical | warning | all

# ---------------------------------------------------------------------------
# GCS client  (replaces boto3.client("s3"))
# ---------------------------------------------------------------------------
gcs = storage.Client(project=GCP_PROJECT)


def gcs_list_result_keys() -> list[str]:
    blobs = gcs.bucket(GCS_RAW_BUCKET).list_blobs(prefix=RESULTS_PREFIX)
    return [
        b.name for b in blobs
        if b.name.endswith(".json") and "checkpoint" not in b.name
    ]


def gcs_read_json(key: str) -> dict:
    blob = gcs.bucket(GCS_RAW_BUCKET).blob(key)
    return json.loads(blob.download_as_text())


def load_processed_keys() -> set:
    try:
        blob = gcs.bucket(GCS_RAW_BUCKET).blob(CHECKPOINT_KEY)
        data = json.loads(blob.download_as_text())
        return set(data.get("processed", []))
    except NotFound:
        return set()


def save_processed_keys(keys: set) -> None:
    gcs.bucket(GCS_RAW_BUCKET).blob(CHECKPOINT_KEY).upload_from_string(
        json.dumps({
            "processed":  list(keys),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# PostgreSQL — connection + schema bootstrap
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id            SERIAL PRIMARY KEY,
    alert_type    VARCHAR(50)   NOT NULL,
    severity      VARCHAR(20)   NOT NULL,
    engine_id     INTEGER       NOT NULL,
    cycle         INTEGER,
    rul           FLOAT,
    anomaly_score FLOAT,
    anomaly_flag  BOOLEAN,
    rule          TEXT,
    message       TEXT,
    dataset       VARCHAR(10),
    event_time    TIMESTAMPTZ,
    indexed_at    TIMESTAMPTZ   DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alerts_engine
    ON alerts (engine_id, indexed_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON alerts (severity, indexed_at DESC);
"""


def get_pg_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, connect_timeout=10,
    )


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    log.info("PostgreSQL schema ready — table: alerts")


# ---------------------------------------------------------------------------
# Alert Rule Engine — unchanged logic
# ---------------------------------------------------------------------------
def evaluate_rules(event: dict) -> list[dict]:
    alerts        = []
    engine_id     = event.get("engine_id")
    cycle         = event.get("cycle")
    rul           = event.get("RUL")
    anomaly_score = event.get("anomaly_score", 0.0)
    anomaly_flag  = event.get("anomaly_flag",  False)
    timestamp     = event.get("timestamp", datetime.now(timezone.utc).isoformat())

    # Rule 1 — RUL threshold
    if rul is not None and rul < RUL_ALERT_THRESHOLD:
        severity = "critical" if rul < 10 else "warning"
        alerts.append({
            "alert_type":    "rul_threshold",
            "severity":      severity,
            "engine_id":     engine_id,
            "cycle":         cycle,
            "rul":           rul,
            "anomaly_score": anomaly_score,
            "anomaly_flag":  anomaly_flag,
            "rule":          f"RUL < {RUL_ALERT_THRESHOLD}",
            "message":       f"Engine {engine_id} approaching failure — RUL={rul:.1f} cycles",
            "dataset":       DATASET,
            "event_time":    timestamp,
        })
        log.warning(
            f"[{severity.upper()}] Engine {engine_id} cycle {cycle} — "
            f"RUL={rul:.1f} < threshold={RUL_ALERT_THRESHOLD}"
        )

    # Rule 2 — Anomaly flag
    if anomaly_flag:
        alerts.append({
            "alert_type":    "anomaly_detected",
            "severity":      "warning",
            "engine_id":     engine_id,
            "cycle":         cycle,
            "rul":           rul,
            "anomaly_score": anomaly_score,
            "anomaly_flag":  anomaly_flag,
            "rule":          "anomaly_flag == True",
            "message":       (
                f"Engine {engine_id} anomaly detected — "
                f"score={anomaly_score:.4f} (HPC/fan degradation)"
            ),
            "dataset":       DATASET,
            "event_time":    timestamp,
        })
        log.warning(
            f"[WARNING] Engine {engine_id} cycle {cycle} — "
            f"anomaly_flag=True score={anomaly_score:.4f}"
        )

    return alerts


# ---------------------------------------------------------------------------
# Notifier — send email via SendGrid
# ---------------------------------------------------------------------------
def should_notify(alert: dict) -> bool:
    """Decide whether to send email based on NOTIFY_SEVERITY setting."""
    if not NOTIFY_ENABLED:
        return False
    if not SENDGRID_API_KEY or not ALERT_FROM_EMAIL or not ALERT_TO_EMAIL:
        log.warning("Notifier: SENDGRID_API_KEY / ALERT_FROM_EMAIL / ALERT_TO_EMAIL not set — skipping email")
        return False
    if not SENDGRID_AVAILABLE:
        log.warning("Notifier: sendgrid package not installed — skipping email")
        return False

    severity   = alert.get("severity", "")
    alert_type = alert.get("alert_type", "")

    if NOTIFY_SEVERITY == "critical":
        return severity == "critical"
    elif NOTIFY_SEVERITY == "warning":
        return severity in ("critical", "warning") or alert_type == "anomaly_detected"
    else:  # all
        return True


def send_email(alert: dict) -> None:
    """Send alert notification email via SendGrid."""
    engine_id  = alert["engine_id"]
    alert_type = alert["alert_type"]
    severity   = alert["severity"].upper()
    message    = alert["message"]
    rul        = alert.get("rul")
    score      = alert.get("anomaly_score", 0.0)
    timestamp  = alert.get("event_time", datetime.now(timezone.utc).isoformat())

    subject = f"[PHM {severity}] Engine {engine_id} — {alert_type.replace('_', ' ').title()}"

    body = f"""
PHM Engine Predictive Maintenance — Alert Notification
=======================================================

Severity  : {severity}
Alert Type: {alert_type}
Engine ID : {engine_id}
Dataset   : {alert['dataset']}
Timestamp : {timestamp}

Message:
{message}

Details:
  RUL            : {f"{rul:.2f} cycles" if rul is not None else "N/A"}
  Anomaly Score  : {score:.6f}
  Anomaly Flag   : {alert.get("anomaly_flag", False)}
  Rule Triggered : {alert.get("rule", "")}

---
This alert was generated automatically by the PHM Alert Rule Engine.
View the full fleet dashboard at: http://34.71.60.98/grafana
    """.strip()

    try:
        sg  = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        msg = Mail(
            from_email=ALERT_FROM_EMAIL,
            to_emails=ALERT_TO_EMAIL,
            subject=subject,
            plain_text_content=Content("text/plain", body),
        )
        response = sg.send(msg)
        log.info(
            f"Email sent — engine={engine_id} type={alert_type} "
            f"status={response.status_code}"
        )
    except Exception as e:
        log.error(f"Failed to send email for engine {engine_id}: {e}")


# ---------------------------------------------------------------------------
# PostgreSQL — insert alert row — unchanged logic
# ---------------------------------------------------------------------------
INSERT_ALERT_SQL = """
INSERT INTO alerts (
    alert_type, severity, engine_id, cycle, rul,
    anomaly_score, anomaly_flag, rule, message,
    dataset, event_time
) VALUES (
    %(alert_type)s, %(severity)s, %(engine_id)s, %(cycle)s, %(rul)s,
    %(anomaly_score)s, %(anomaly_flag)s, %(rule)s, %(message)s,
    %(dataset)s, %(event_time)s
);
"""


def insert_alert(conn, alert: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(INSERT_ALERT_SQL, alert)
    log.info(
        f"Inserted alert — type={alert['alert_type']} "
        f"engine={alert['engine_id']} severity={alert['severity']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(
        f"Alert engine starting — dataset={DATASET} "
        f"rul_threshold={RUL_ALERT_THRESHOLD}"
    )

    conn = get_pg_conn()
    log.info(f"PostgreSQL connected — {DB_HOST}:{DB_PORT}/{DB_NAME}")
    ensure_schema(conn)

    # List new inference result files from GCS  (replaces s3 paginator)
    all_keys = gcs_list_result_keys()

    if not all_keys:
        log.info("No inference results found. Nothing to evaluate.")
        conn.close()
        return

    processed_keys = load_processed_keys()
    pending_keys   = [k for k in all_keys if k not in processed_keys]

    log.info(f"Total={len(all_keys)} Pending={len(pending_keys)}")

    if not pending_keys:
        log.info("All results already processed.")
        conn.close()
        return

    total_alerts = 0
    errors       = 0

    for key in pending_keys:
        try:
            event  = gcs_read_json(key)
            alerts = evaluate_rules(event)

            for alert in alerts:
                insert_alert(conn, alert)
                total_alerts += 1

                # Notifier — send email if severity meets threshold
                if should_notify(alert):
                    send_email(alert)

            processed_keys.add(key)

            if len(processed_keys) % 50 == 0:
                conn.commit()

        except Exception as e:
            log.error(f"Failed to process {key}: {e}")
            errors += 1

    conn.commit()
    conn.close()

    save_processed_keys(processed_keys)

    log.info(
        f"Alert engine done — "
        f"processed={len(pending_keys) - errors} "
        f"alerts_fired={total_alerts} "
        f"errors={errors}"
    )


if __name__ == "__main__":
    main()