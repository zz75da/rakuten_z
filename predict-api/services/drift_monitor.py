"""
Input distribution drift monitor — Evidently AI.

Design constraints (20 GB disk, memory-limited laptop):
  1. Reference dataset: stratified 5k-row sample (not full 85k × 10k matrix)
     Rebuilt only when text_features_meta.json changes.
  2. Prediction buffer: thread-safe ring buffer, max 2000 rows.
     Prevents unbounded memory growth from live traffic accumulation.
  3. Report generation: background thread, never blocks inference path.
     If Evidently fails or hangs, inference continues unaffected.
  4. Report rotation: keeps last 10 HTML reports, deletes older ones.
     Each report ~2-5 MB; 10 reports = max ~50 MB disk impact.
  5. Reference sample path: /app/data/artifacts/drift_reference.csv
     Rebuilt from training features + labels (stratified, 5k rows max).
"""
import os
import gc
import csv
import json
import threading
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARTIFACTS   = Path("/app/data/artifacts")
REPORT_DIR  = ARTIFACTS / "drift_reports"
REF_PATH    = ARTIFACTS / "drift_reference.csv"
META_PATH   = Path("/app/data/feature_cache/text_features_meta.json")

MAX_BUFFER  = 2000   # max live prediction rows before computing a report
MAX_REPORTS = 10     # max report files kept on disk


# ── Thread-safe ring buffer ───────────────────────────────────────────────────
class _PredictionBuffer:
    def __init__(self, maxlen: int):
        self._buf  = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, record: dict):
        with self._lock:
            self._buf.append(record)

    def drain(self) -> list:
        with self._lock:
            rows = list(self._buf)
            self._buf.clear()
            return rows

    def __len__(self):
        with self._lock:
            return len(self._buf)


_buffer   = _PredictionBuffer(MAX_BUFFER)
_report_lock = threading.Lock()   # prevents concurrent report generation


def record_prediction(features: dict):
    """
    Called from the inference path — non-blocking, sub-millisecond.
    features: small dict of scalar summary stats (not raw high-dim vectors).
    """
    _buffer.append(features)
    # Trigger report in background when buffer is full
    if len(_buffer) >= MAX_BUFFER:
        threading.Thread(target=_generate_report, daemon=True).start()


# ── Reference dataset ────────────────────────────────────────────────────────
def build_reference(n_samples: int = 5000):
    """
    Build a stratified reference sample from the training label file.
    Called once after training completes (via POST /drift-rebuild-reference).
    Saves to drift_reference.csv — ~1 MB.
    """
    y_csv = os.getenv("TRAIN_CSV_Y_PATH", "/app/data/Y_train_CVw08PX.csv")
    x_csv = os.getenv("TRAIN_CSV_X_PATH", "/app/data/X_train_update.csv")
    if not (os.path.exists(y_csv) and os.path.exists(x_csv)):
        log.warning("Training CSVs not found — drift reference not built")
        return False
    try:
        Y = pd.read_csv(y_csv)
        X = pd.read_csv(x_csv, usecols=["Unnamed: 0", "designation"])
        merged = X.merge(Y, on="Unnamed: 0")
        # Stratified sample
        sampled = (
            merged.groupby("prdtypecode", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(n_samples * len(g) / len(merged)))),
                random_state=42,
            ))
        ).head(n_samples)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        sampled[["Unnamed: 0", "designation", "prdtypecode"]].to_csv(REF_PATH, index=False)
        log.info(f"Drift reference built: {len(sampled)} rows → {REF_PATH}")
        del Y, X, merged, sampled
        gc.collect()
        return True
    except Exception as e:
        log.warning(f"Drift reference build failed: {e}")
        return False


def reference_exists() -> bool:
    return REF_PATH.exists() and REF_PATH.stat().st_size > 1000


# ── Report generation ────────────────────────────────────────────────────────
def _rotate_reports():
    """Keep only the last MAX_REPORTS files in REPORT_DIR."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(REPORT_DIR.glob("drift_*.html"), key=lambda p: p.stat().st_mtime)
    for old in reports[:-MAX_REPORTS]:
        try:
            old.unlink()
            log.info(f"Drift report rotated (deleted): {old.name}")
        except Exception:
            pass


def _generate_report():
    """
    Generate an Evidently drift report from the current buffer.
    Runs in a daemon thread — isolated from inference path.
    Any exception here is caught and logged; never propagates to callers.
    """
    if not _report_lock.acquire(blocking=False):
        return   # another report already generating
    try:
        if not reference_exists():
            log.info("Drift report skipped — no reference dataset")
            return

        rows = _buffer.drain()
        if not rows:
            return

        # Build current dataset from buffer rows
        current_df  = pd.DataFrame(rows)
        reference_df = pd.read_csv(REF_PATH)

        # Keep only columns present in both
        shared_cols = [c for c in current_df.columns if c in reference_df.columns]
        if not shared_cols:
            log.warning("No shared columns between reference and current — skipping report")
            return

        try:
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            column_mapping = ColumnMapping(target="prdtypecode" if "prdtypecode" in shared_cols else None)
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=reference_df[shared_cols],
                current_data=current_df[shared_cols],
                column_mapping=column_mapping,
            )

            _rotate_reports()
            ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            path = REPORT_DIR / f"drift_{ts}.html"
            report.save_html(str(path))
            log.info(f"Drift report saved: {path}  ({path.stat().st_size // 1024} KB)")

        except ImportError:
            log.warning("evidently not installed — drift report skipped")
        except Exception as e:
            log.warning(f"Evidently report failed: {e}")

        del current_df, reference_df
        gc.collect()

    except Exception as e:
        log.warning(f"Drift report generation error: {e}")
    finally:
        _report_lock.release()


def trigger_report():
    """Force a report generation regardless of buffer size (e.g., scheduled trigger)."""
    threading.Thread(target=_generate_report, daemon=True).start()


def buffer_size() -> int:
    return len(_buffer)
