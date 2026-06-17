# Kubernetes manifests — predict-api (proof of concept)

Manifests demonstrating horizontal scalability for `predict-api`, deployed
and tested locally on Docker Desktop's built-in Kubernetes cluster
(`docker-desktop` context). Not used in the docker-compose stack — this is
an additional deployment target showing how the service could be scaled
behind an orchestrator.

## Contents

- `namespace.yaml` — `rakuten-mlops` namespace
- `predict-api-configmap.yaml` — non-secret env vars (model names, MLflow
  experiment, artifacts path...), mirrors `docker-compose.yml`'s
  `predict-api.environment` block
- `predict-api-deployment.yaml` — Deployment (1 replica by default),
  resource requests/limits, startup/readiness/liveness probes on `/health`,
  hostPath volumes for `params.yaml`, `data/artifacts`, `data/hf_cache`
- `predict-api-service.yaml` — ClusterIP service, port 5003
- `predict-api-hpa.yaml` — HorizontalPodAutoscaler (CPU-based, 1-2 replicas,
  70% target). Requires `metrics-server`.

## Prerequisites

- Docker Desktop with Kubernetes enabled (context `docker-desktop`)
- The `rakuten_mlops_services-predict-api:latest` image already built
  (`docker compose build predict-api`) — reused directly, no registry push
- `metrics-server` installed (required for the HPA):
  ```bash
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  kubectl patch deployment metrics-server -n kube-system --type='json' \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
  ```
  (the `--kubelet-insecure-tls` patch is only needed for local clusters with
  self-signed kubelet certs)

## Deploy

```bash
kubectl apply -f k8s/namespace.yaml -f k8s/predict-api-configmap.yaml

# Secret: same variables predict-api gets via env_file: .env in docker-compose
# (AWS/DagsHub/MLflow creds + JWT_SECRET_KEY). Not committed — recreate locally:
kubectl create secret generic predict-api-secrets -n rakuten-mlops --from-env-file=.env

kubectl apply -f k8s/predict-api-deployment.yaml -f k8s/predict-api-service.yaml -f k8s/predict-api-hpa.yaml
```

## Verify

```bash
kubectl get pods -n rakuten-mlops -w
kubectl logs -n rakuten-mlops deploy/predict-api -f
kubectl get hpa -n rakuten-mlops
```

Model loading (CV/CLIP/MiniLM/mpnet from `data/artifacts` + `data/hf_cache`
via hostPath) takes a few minutes on first start — the startup probe allows
up to ~10 minutes (matches docker-compose's healthcheck).

## Demo: scaling

```bash
# Manual scale
kubectl scale deployment predict-api -n rakuten-mlops --replicas=2

# Or generate load to trigger the HPA (from inside the cluster, hitting the
# ClusterIP service on port 5003)
kubectl run load-test --rm -it --image=busybox -n rakuten-mlops -- \
  /bin/sh -c "while true; do wget -q -O- http://predict-api:5003/health; done"
```

## hostPath note

`hostPath` volumes use Docker Desktop's Windows filesystem bind
(`/run/desktop/mnt/host/c/...`), so paths are hardcoded to this machine's
project location. For a portable/production setup these would become
PVCs (e.g. backed by an S3/MinIO-mounted volume or an init container that
pulls artifacts from MLflow/DagsHub on startup) and the local image would
be pushed to a registry instead of relying on the local Docker image cache.

## Teardown

```bash
kubectl delete namespace rakuten-mlops
```
