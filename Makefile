# MLOps Observability Stack

COMPOSE=docker compose
GRAFANA_CONTAINER=grafana
DASHBOARD_FILE=./grafana/mlops_dashboard.json

.PHONY: observability up down logs restart grafana-dash rebuild rebuild-security

# Lancer uniquement la stack observabilité
observability:
	$(COMPOSE) up -d prometheus grafana pushgateway

# Stopper
down:
	$(COMPOSE) down

# Voir logs
logs:
	$(COMPOSE) logs -f

# Restart rapide
restart:
	$(COMPOSE) down && $(COMPOSE) up -d

# Importer automatiquement le dashboard Grafana
grafana-dash:
	@echo "Importing Grafana dashboard..."
	docker cp $(DASHBOARD_FILE) $$(docker ps -qf "name=$(GRAFANA_CONTAINER)"):/var/lib/grafana/dashboards/mlops_dashboard.json
	docker exec -it $$(docker ps -qf "name=$(GRAFANA_CONTAINER)") grafana-cli admin reset-admin-password admin
	@echo "Dashboard imported. Connect to Grafana at http://localhost:3000 (user: admin / pass: admin)"

# Rebuild ordonné après les corrections de sécurité
# Ordre : gate-api → train-api → predict-api → streamlit
# Les autres services (postgres, minio, airflow, prometheus, grafana…)
# n'ont pas changé et restent actifs pendant le rebuild.
rebuild-security:
	@echo ">>> [1/4] Rebuild gate-api..."
	$(COMPOSE) build gate-api
	$(COMPOSE) up -d --no-deps gate-api
	@echo ">>> Attente gate-api healthy..."
	@until $$(curl -sf http://localhost:5004/health > /dev/null); do sleep 2; done
	@echo ">>> gate-api OK"

	@echo ">>> [2/4] Rebuild train-api..."
	$(COMPOSE) build train-api
	$(COMPOSE) up -d --no-deps train-api
	@echo ">>> Attente train-api healthy..."
	@until $$(curl -sf http://localhost:5002/health > /dev/null); do sleep 5; done
	@echo ">>> train-api OK"

	@echo ">>> [3/4] Rebuild predict-api..."
	$(COMPOSE) build predict-api
	$(COMPOSE) up -d --no-deps predict-api
	@echo ">>> Attente predict-api healthy..."
	@until $$(curl -sf http://localhost:5003/health > /dev/null); do sleep 5; done
	@echo ">>> predict-api OK"

	@echo ">>> [4/4] Rebuild streamlit..."
	$(COMPOSE) build streamlit
	$(COMPOSE) up -d --no-deps streamlit
	@echo ">>> Attente streamlit healthy..."
	@until $$(curl -sf http://localhost:8501 > /dev/null); do sleep 3; done
	@echo ">>> streamlit OK"

	@echo ""
	@echo "=== Rebuild terminé. Vérification finale ==="
	$(COMPOSE) ps gate-api train-api predict-api streamlit
