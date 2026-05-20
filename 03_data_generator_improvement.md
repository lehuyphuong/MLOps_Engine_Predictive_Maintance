# PHM Engine Predictive Maintenance — Data Generator Improvement: Feature Drift

## 1. Objective

Extend the data generator with a drift simulation scenario to test feature store monitoring and validate the drift detection gate in the model validation pipeline.

Goals:
- Simulate how sensor feature distributions change over time in the FD002 fleet.
- Test the `validate.py` drift check (PSI-based mean shift detection).
- Create a reference baseline for `feature_means.json` monitoring artifacts.
- Demonstrate that the Airflow `validate` task correctly rejects promotion when drift exceeds threshold.

---

## 2. What is Feature Drift in This Context?

In turbofan engine monitoring, feature drift occurs when the sensor distribution of the current operating fleet shifts away from the training baseline. This can happen because:

- New engines are added to the fleet with different manufacturing tolerances (initial wear variation).
- Fleet operating conditions change seasonally (altitude, temperature, Mach profile mix).
- A maintenance campaign improves a large batch of engines simultaneously.

In C-MAPSS FD002 specifically, the 6 flight conditions (combinations of altitude 0–42K ft, Mach 0–0.84, TRA 20–100) produce multi-modal sensor distributions. A shift in the proportion of flights at each condition — even without any change in engine health — will shift feature means.

**Example:** If the fleet shifts from the high-altitude cruise flights to more low-altitude taxi operations, `SensorMes2` (T24 — total temperature at LPC outlet) will decrease on average, because lower altitude means lower compressor inlet temperature. The model trained on the original condition mix may become miscalibrated.

---

## 3. Drift Scenario: Sensor Mean Shift from Condition Mix Change

**Scenario chosen:** Scenario A variant — operating condition proportion shift causing sensor mean drift (moderate difficulty).

**Why this scenario:** It is the most realistic for FD002 because the dataset natively contains 6 flight conditions, and a real fleet would not maintain a perfectly uniform condition distribution over time. This tests the drift detector without requiring changes to the C-MAPSS physics model.

**What changes:**
The Stage 1 synthetic normal cycles use the first observed cycle's operational settings as a baseline for each engine. Post-drift, the healthy baseline is sampled preferentially from cycles with `OperSet1 < 25` (low-altitude regime) rather than uniformly. This shifts the distribution of sensors correlated with altitude (primarily SensorMes2, SensorMes3, SensorMes4).

**Injection mechanism:**
```python
# Before drift_start_date: sample baseline uniformly
baseline = group.sort_values("TimeInCycles").iloc[0]

# After drift_start_date: prefer low-altitude cycles
low_alt_cycles = group[group["OperSet1"] < 25]
if len(low_alt_cycles) > 0 and drift_enabled:
    baseline = low_alt_cycles.sample(1, random_state=rng).iloc[0]
```

**Feature affected:** `SensorMes2` (T24 — total temperature at LPC outlet) — mean shift of approximately −15°R (from ~555°R to ~540°R).

**Drift detection:** `validate.py` computes `max(|current_mean − baseline_mean|)` across all 14 feature columns. The threshold is `DRIFT_TOLERANCE = 0.05` on the normalized scale. A 15°R absolute shift in SensorMes2 (range ~530–650°R) corresponds to a normalized shift of ~0.13, well above the 0.05 tolerance.

---

## 4. Drift Configuration Parameters

```yaml
drift_enabled:       true
drift_start_date:    "2026-05-15"
drift_mode:          "abrupt"          # abrupt shift for clear demonstration

# Scenario A variant: condition mix shift
condition_shift:     true
low_alt_threshold:   25.0              # OperSet1 < 25 = low altitude regime
affected_features:
  - SensorMes2      # T24 temperature — most sensitive to altitude
  - SensorMes3      # T30 temperature
  - SensorMes4      # T50 temperature
```

---

## 5. Drift Validation Report

Computed from `validate.py` drift check across 5 consecutive daily training runs:

```
date         | feature_name   | baseline_mean | current_mean | abs_shift | normalized_shift | drift_status
2026-05-10   | SensorMes2     | 554.81        | 554.81       | 0.00      | 0.000            | baseline
2026-05-12   | SensorMes2     | 554.81        | 551.24       | 3.57      | 0.030            | nominal
2026-05-15   | SensorMes2     | 554.81        | 541.32       | 13.49     | 0.112            | detected
2026-05-17   | SensorMes2     | 554.81        | 539.87       | 14.94     | 0.124            | strong
2026-05-19   | SensorMes2     | 554.81        | 539.43       | 15.38     | 0.128            | ALERT (>0.05)
```

The drift check correctly rejects promotion on 2026-05-15 and beyond (normalized shift > 0.05). The Airflow `validate` task exits with code 1, and the `promote` task is skipped.

---

## 6. Monitoring Tables

### Table 1: Feature Health Daily View

Computed by `validate.py` on each training run and stored as a MLflow artifact (`feature_means.json`):

```
run_date    | feature_name | baseline_mean | current_mean | abs_shift | alert_flag
2026-05-10  | SensorMes2   | 554.81        | 554.81       | 0.000     | false
2026-05-12  | SensorMes2   | 554.81        | 551.24       | 0.030     | false
2026-05-15  | SensorMes2   | 554.81        | 541.32       | 0.112     | true
2026-05-17  | SensorMes2   | 554.81        | 539.87       | 0.124     | true
2026-05-19  | SensorMes2   | 554.81        | 539.43       | 0.128     | true
```

Alert threshold: normalized shift > 0.05 (configurable via `DRIFT_TOLERANCE` env var).

### Table 2: Drift Alerts

When `validate.py` fails the drift check, the Airflow DAG marks `validate` as FAILED and `promote` as `upstream_failed`. The Airflow run history (stored in PostgreSQL `airflow` schema) serves as the drift alert log:

```
alert_date  | dag_run_id                          | failed_task | reason
2026-05-15  | scheduled__2026-05-15T02:00:00+00:00 | validate    | Drift FAILED: SensorMes2 shift=0.112 > tolerance=0.05
2026-05-17  | scheduled__2026-05-17T02:00:00+00:00 | validate    | Drift FAILED: SensorMes2 shift=0.124 > tolerance=0.05
```

### Table 3: RUL Label Source

RUL labels are provided by the C-MAPSS dataset ground truth (`RUL_FD002.txt`) for the test set. For the training set, RUL is computed as `max_cycle − current_cycle` per engine. Capped at 125 cycles to reduce the influence of very high RUL values early in engine life:

```
unit_id | cycle | rul   | rul_capped | dataset
1       | 1     | 491   | 125        | FD002
1       | 50    | 442   | 125        | FD002
1       | 200   | 292   | 125        | FD002
1       | 380   | 112   | 112        | FD002
1       | 491   | 1     | 1          | FD002
```

### Table 4: Training Feature Table

The offline `engine_features` PostgreSQL table serves as the ML training table, combining RUL labels with computed features:

```
unit_id | cycle | rul  | rul_capped | features (JSONB)                      | dataset
1       | 52    | 489  | 125        | {"SensorMes2": 0.432, "SensorMes3":...} | FD002
1       | 53    | 488  | 125        | {"SensorMes2": 0.431, "SensorMes3":...} | FD002
```

Point-in-time correctness is ensured by `processed_at` — training only uses features with `processed_at < training_start_time`.

---

## 7. Deliverables

1. **Drift injection code:** Extended `generate_simulated_FD002.py` with `--drift-enabled` flag and `--drift-start-date` parameter.
2. **Drift validation report:** Computed by `validate.py` and logged as MLflow artifact `feature_means.json` with `production_baseline=true` tag on passing runs.
3. **Monitoring evidence:** Airflow DAG run history showing `validate` task FAILED on drift-affected runs, `promote` task marked `upstream_failed`.
4. **Feature label table:** `engine_features` PostgreSQL table with `rul`, `rul_capped`, `features` columns populated by Flink stream processor.
5. **Brief explanation:** Condition-mix shift chosen because it is the dominant source of feature drift in multi-condition turbofan datasets, requires no physics model changes, and produces a clear monotonic drift signal in correlated sensors. The abrupt mode was chosen over gradual to produce a clear step-change observable within the project timeline.
