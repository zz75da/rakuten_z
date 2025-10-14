# MLOps Observability Stack

COMPOSE=docker compose
GRAFANA_CONTAINER=grafana
DASHBOARD_FILE=./grafana/mlops_dashboard.json

.PHONY: observability up down logs restart grafana-dash

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
