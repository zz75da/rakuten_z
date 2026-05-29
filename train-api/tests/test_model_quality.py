"""
Model quality gate — runs after every training job via the DAG.

Asserts minimum acceptable performance on the saved training history and
final metrics. Blocks DAG progression (and therefore deployment) if any
encoder falls below its floor or shows a class collapse.

Run manually (inside train-api container):
    pytest /app/tests/test_model_quality.py -v

Floors (conservative — set well below current best to catch regressions):
    Accuracy:  CV ≥ 0.72  |  CLIP ≥ 0.80  |  MiniLM ≥ 0.70  |  mpnet ≥ 0.72
    MacroF1:   all encoders ≥ 0.65
    Top3:      all encoders ≥ 0.88
    Per-class: no class may have recall = 0.0 (complete collapse)
"""
import json
import os
import pytest

_ARTIFACTS = os.environ.get("ARTIFACTS_PATH", "/app/data/artifacts")

# ── Per-encoder accuracy floors ──────────────────────────────────────────────
_ACC_FLOORS = {
    "cv":     0.72,
    "clip":   0.80,
    "minilm": 0.70,
    "mpnet":  0.72,
}
_MACRO_F1_FLOOR  = 0.65
_TOP3_FLOOR      = 0.88


def _load_history(encoder: str) -> dict:
    suffix = "" if encoder == "cv" else f"_{encoder}"
    path = os.path.join(_ARTIFACTS, f"train_history{suffix}.json")
    if not os.path.exists(path):
        pytest.skip(f"No history file for {encoder} — not yet trained")
    with open(path) as f:
        return json.load(f)


def _best(history: dict, key: str) -> float:
    vals = history.get(key, [])
    return max(vals) if vals else 0.0


def _load_final_metrics(encoder: str) -> dict | None:
    """Load final_metrics from the most recent successful job file for this encoder."""
    jobs_dir = "/app/data/jobs"
    if not os.path.isdir(jobs_dir):
        return None
    model_suffix = {
        "cv":     "neural_network_model.keras",
        "clip":   "neural_network_model_clip.keras",
        "minilm": "neural_network_model_minilm.keras",
        "mpnet":  "neural_network_model_mpnet.keras",
    }[encoder]
    candidates = []
    for fname in os.listdir(jobs_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(jobs_dir, fname)) as f:
                d = json.load(f)
            if d.get("status") == "success" and model_suffix in (d.get("model_path") or ""):
                candidates.append((os.path.getmtime(os.path.join(jobs_dir, fname)), d))
        except Exception:
            continue
    if not candidates:
        return None
    _, best_job = max(candidates, key=lambda x: x[0])
    return best_job.get("final_metrics")


# ── Accuracy floors ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("encoder", list(_ACC_FLOORS))
def test_val_accuracy_floor(encoder):
    history = _load_history(encoder)
    best_acc = _best(history, "val_accuracy")
    floor    = _ACC_FLOORS[encoder]
    assert best_acc >= floor, (
        f"{encoder} best val_accuracy={best_acc:.4f} < floor={floor}. "
        f"Model may have regressed or not converged."
    )


# ── Macro F1 floor ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("encoder", list(_ACC_FLOORS))
def test_macro_f1_floor(encoder):
    history  = _load_history(encoder)
    best_f1  = _best(history, "val_macro_f1")
    if best_f1 == 0.0:
        pytest.skip(f"val_macro_f1 not recorded for {encoder} (older run)")
    assert best_f1 >= _MACRO_F1_FLOOR, (
        f"{encoder} best macro_f1={best_f1:.4f} < floor={_MACRO_F1_FLOOR}."
    )


# ── Top-3 accuracy floor ──────────────────────────────────────────────────────
@pytest.mark.parametrize("encoder", list(_ACC_FLOORS))
def test_top3_accuracy_floor(encoder):
    history   = _load_history(encoder)
    best_top3 = _best(history, "val_top3_accuracy")
    if best_top3 == 0.0:
        pytest.skip(f"val_top3_accuracy not recorded for {encoder} (older run)")
    assert best_top3 >= _TOP3_FLOOR, (
        f"{encoder} best top3_accuracy={best_top3:.4f} < floor={_TOP3_FLOOR}."
    )


# ── No class collapse (recall = 0.0) ─────────────────────────────────────────
@pytest.mark.parametrize("encoder", list(_ACC_FLOORS))
def test_no_class_collapse(encoder):
    fm = _load_final_metrics(encoder)
    if fm is None:
        pytest.skip(f"No final_metrics found for {encoder}")
    report = fm.get("report", {})
    collapsed = []
    for label, metrics in report.items():
        if isinstance(metrics, dict) and metrics.get("recall", 1.0) == 0.0:
            collapsed.append(label)
    assert not collapsed, (
        f"{encoder} has {len(collapsed)} class(es) with recall=0.0: {collapsed}. "
        f"Check for label errors or extreme class imbalance."
    )


# ── Overfitting guard (train_acc - val_acc gap) ───────────────────────────────
@pytest.mark.parametrize("encoder", list(_ACC_FLOORS))
def test_overfit_gap(encoder):
    history   = _load_history(encoder)
    val_accs  = history.get("val_accuracy", [])
    train_accs = history.get("accuracy", [])
    if not val_accs or not train_accs:
        pytest.skip(f"Incomplete history for {encoder}")
    # Use the epoch with best val_accuracy
    best_epoch = val_accs.index(max(val_accs))
    gap = train_accs[best_epoch] - val_accs[best_epoch]
    assert gap <= 0.20, (
        f"{encoder} overfit gap={gap:.3f} at best epoch {best_epoch+1}. "
        f"Gap > 0.20 suggests significant overfitting."
    )
