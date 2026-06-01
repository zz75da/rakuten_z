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
import threading
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ARTIFACTS  = Path("/app/data/artifacts")
REPORT_DIR = ARTIFACTS / "drift_reports"
REF_PATH   = ARTIFACTS / "drift_reference.csv"   # built by train-api POST /drift-rebuild-reference

MAX_BUFFER  = 2000   # max live prediction rows before computing a report
MIN_BUFFER  = 30     # minimum rows needed for Evidently statistical tests
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


def reference_exists() -> bool:
    """drift_reference.csv is built by train-api POST /drift-rebuild-reference and saved
    to the shared /app/data/artifacts/ volume."""
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
            log.info("Drift report skipped — buffer is empty")
            return
        if len(rows) < MIN_BUFFER:
            log.warning(
                f"Drift report skipped — only {len(rows)} rows in buffer "
                f"(minimum {MIN_BUFFER} required for meaningful statistics). "
                f"Make more predictions then trigger again."
            )
            # Put rows back so they aren't lost
            for r in rows:
                _buffer.append(r)
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


def trigger_report() -> dict:
    """Force a report generation. Returns buffer state before triggering."""
    n = len(_buffer)
    threading.Thread(target=_generate_report, daemon=True).start()
    return {"buffer_size": n, "min_required": MIN_BUFFER, "enough": n >= MIN_BUFFER}


def buffer_size() -> int:
    return len(_buffer)
