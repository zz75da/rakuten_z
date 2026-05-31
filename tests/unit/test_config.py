"""
Configuration validation tests
================================
Purpose : Verify that key configuration files are well-formed, contain required
          sections/keys, and have values within safe operating ranges.
          These tests run without Docker, TensorFlow, or any running service.

Covered :
  TestParamsYaml          — params.yaml structure, required keys, safe ranges
  TestDockerCompose       — docker-compose.yml required services and ports
  TestPrometheusConfig    — prometheus.yml scrape targets
  TestAlertRules          — alert-rules.yml 4-encoder accuracy alerts present
  TestEnvTemplate         — .env.template required variables documented

Dependencies : PyYAML, pathlib (stdlib only — no heavy ML dependencies)
"""
import os
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()


def _load_yaml(rel_path: str) -> dict:
    import yaml
    path = PROJECT_ROOT / rel_path
    assert path.exists(), f"Config file not found: {path}"
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# params.yaml
# ---------------------------------------------------------------------------
class TestParamsYaml:
    @pytest.fixture(scope="class")
    def params(self):
        return _load_yaml("params.yaml")

    def test_required_top_level_sections(self, params):
        for section in ("preprocess", "train", "model"):
            assert section in params, f"Missing section: {section}"

    def test_preprocess_required_keys(self, params):
        pre = params["preprocess"]
        for key in ("pca_components", "n_text_pca_components", "cv_max_features",
                    "image_batch_size", "pca_batch_size"):
            assert key in pre, f"Missing preprocess key: {key}"

    def test_pca_components_safe_range(self, params):
        n = params["preprocess"]["pca_components"]
        assert 64 <= n <= 512, f"pca_components={n} outside safe range [64, 512]"

    def test_n_text_pca_components_safe_range(self, params):
        n = params["preprocess"]["n_text_pca_components"]
        assert 128 <= n <= 1024, f"n_text_pca_components={n} outside safe range"

    def test_batch_size_not_too_large(self, params):
        bs = params["train"]["batch_size"]
        assert bs <= 256, f"train.batch_size={bs} may cause OOM (max safe: 256)"

    def test_model_required_keys(self, params):
        m = params["model"]
        for key in ("learning_rate", "hidden_1", "hidden_2",
                    "dropout_1", "dropout_2", "l2_reg",
                    "early_stopping_patience", "focal_gamma"):
            assert key in m, f"Missing model key: {key}"

    def test_focal_gamma_valid_range(self, params):
        g = params["model"]["focal_gamma"]
        assert 0.0 <= g <= 5.0, f"focal_gamma={g} outside valid range [0, 5]"

    def test_learning_rate_reasonable(self, params):
        lr = params["model"]["learning_rate"]
        assert 1e-5 <= lr <= 0.1, f"learning_rate={lr} unreasonable"

    def test_use_late_fusion_is_bool(self, params):
        assert isinstance(params["model"].get("use_late_fusion"), bool)

    def test_per_encoder_overrides_present(self, params):
        for section in ("model_minilm", "model_mpnet", "model_clip",
                        "model_countvectorizer"):
            assert section in params, f"Missing encoder override section: {section}"


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------
class TestDockerCompose:
    @pytest.fixture(scope="class")
    def compose(self):
        return _load_yaml("docker-compose.yml")

    def _services(self, compose):
        return compose.get("services", {})

    def test_required_services_present(self, compose):
        required = {"train-api", "predict-api", "gate-api",
                    "minilm-encoder", "clip-encoder", "airflow",
                    "streamlit", "postgres", "prometheus",
                    "grafana", "alertmanager", "pushgateway", "minio"}
        present = set(self._services(compose).keys())
        missing = required - present
        assert not missing, f"Missing services: {missing}"

    def test_train_api_port(self, compose):
        svc = self._services(compose)["train-api"]
        ports = str(svc.get("ports", ""))
        assert "5002" in ports, "train-api should expose port 5002"

    def test_predict_api_port(self, compose):
        svc = self._services(compose)["predict-api"]
        ports = str(svc.get("ports", ""))
        assert "5003" in ports, "predict-api should expose port 5003"

    def test_airflow_port(self, compose):
        svc = self._services(compose)["airflow"]
        ports = str(svc.get("ports", ""))
        assert "8080" in ports, "airflow should expose port 8080"

    def test_train_api_has_volume_for_artifacts(self, compose):
        svc = self._services(compose)["train-api"]
        volumes = str(svc.get("volumes", ""))
        assert "artifacts" in volumes, "train-api must mount data/artifacts volume"

    def test_train_api_has_params_yaml_mount(self, compose):
        svc = self._services(compose)["train-api"]
        volumes = str(svc.get("volumes", ""))
        assert "params.yaml" in volumes, "train-api must mount params.yaml"


# ---------------------------------------------------------------------------
# monitoring/prometheus.yml
# ---------------------------------------------------------------------------
class TestPrometheusConfig:
    @pytest.fixture(scope="class")
    def prom(self):
        return _load_yaml("monitoring/prometheus.yml")

    def _job_names(self, prom):
        return {sc["job_name"] for sc in prom.get("scrape_configs", [])}

    def test_required_scrape_targets(self, prom):
        required = {"train-api", "predict-api", "gate-api",
                    "minilm-encoder", "clip-encoder",
                    "prometheus", "pushgateway", "grafana"}
        missing = required - self._job_names(prom)
        assert not missing, f"Missing scrape targets: {missing}"

    def test_scrape_interval_reasonable(self, prom):
        interval = prom.get("global", {}).get("scrape_interval", "999s")
        seconds = int(interval.replace("s", ""))
        assert seconds <= 60, f"scrape_interval={interval} too long (max 60s)"


# ---------------------------------------------------------------------------
# monitoring/alert-rules.yml
# ---------------------------------------------------------------------------
class TestAlertRules:
    @pytest.fixture(scope="class")
    def rules(self):
        return _load_yaml("monitoring/alert-rules.yml")

    def _alert_names(self, rules):
        names = set()
        for group in rules.get("groups", []):
            for rule in group.get("rules", []):
                if "alert" in rule:
                    names.add(rule["alert"])
        return names

    def test_all_4_encoder_accuracy_alerts(self, rules):
        names = self._alert_names(rules)
        required = {
            "CVModelValAccuracyLow",
            "CLIPModelValAccuracyLow",
            "MiniLMModelValAccuracyLow",
            "mpnetModelValAccuracyLow",
        }
        missing = required - names
        assert not missing, f"Missing encoder accuracy alerts: {missing}"

    def test_service_down_alerts(self, rules):
        names = self._alert_names(rules)
        for alert in ("TrainAPIDown", "PredictAPIDown", "GateAPIDown"):
            assert alert in names, f"Missing alert: {alert}"

    def test_drift_alerts_present(self, rules):
        names = self._alert_names(rules)
        assert "PredictionConfidenceDrift" in names
        assert "PredictionEntropyHigh" in names

    def test_resource_alerts_present(self, rules):
        names = self._alert_names(rules)
        assert "DiskSpaceLow" in names
        assert "HighMemoryUsage" in names

    def test_clip_threshold_is_080(self, rules):
        """CLIP best val_acc is 0.85 — floor should be 0.80, not 0.70."""
        for group in rules.get("groups", []):
            for rule in group.get("rules", []):
                if rule.get("alert") == "CLIPModelValAccuracyLow":
                    expr = rule.get("expr", "")
                    assert "0.80" in expr, \
                        f"CLIPModelValAccuracyLow threshold should be 0.80, got: {expr}"


# ---------------------------------------------------------------------------
# .env.template
# ---------------------------------------------------------------------------
class TestEnvTemplate:
    @pytest.fixture(scope="class")
    def template_content(self):
        path = PROJECT_ROOT / ".env.template"
        assert path.exists(), ".env.template not found"
        return path.read_text()

    def test_required_vars_documented(self, template_content):
        required = [
            "DAGSHUB_USER",
            "DAGSHUB_TOKEN",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT_NAME",
        ]
        for var in required:
            assert var in template_content, \
                f"Required variable {var!r} missing from .env.template"

    def test_no_real_credentials_in_template(self, template_content):
        """Template must not contain real tokens/passwords."""
        suspicious = ["ghp_", "sk-", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"]
        for token_prefix in suspicious:
            assert token_prefix not in template_content, \
                f"Possible real credential found in .env.template: {token_prefix!r}"
