# Mode opératoire — démo Kubernetes (predict-api)

État par défaut : le namespace `rakuten-mlops` existe (Deployment, Service,
ConfigMap, Secret) mais le Deployment est **scalé à 0** et le **HPA est
supprimé**, pour ne pas consommer ~2 Go de RAM en permanence à chaque
démarrage du PC (le Deployment serait recréé automatiquement par
Kubernetes au boot sinon).

## Avant la démo (à faire ~5-10 min avant)

```bash
cd c:\Users\zobir\DScientest\rakuten_mlops_services

# 1. Remonter le pod predict-api (chargement des 4 modèles : ~2-3 min)
kubectl scale deployment predict-api -n rakuten-mlops --replicas=1

# 2. Recréer le HPA (supprimé pour éviter qu'il remonte le pod au boot)
kubectl apply -f k8s/predict-api-hpa.yaml

# 3. Attendre que le pod soit Ready
kubectl get pods -n rakuten-mlops -w
```

Attendre `1/1 Running`. Vérifier :

```bash
kubectl get hpa -n rakuten-mlops
# doit afficher : cpu: x%/70%   MINPODS 1   MAXPODS 2

kubectl port-forward -n rakuten-mlops svc/predict-api 18503:5003
# puis dans un autre terminal :
curl http://localhost:18503/health
# {"status":"ok","cv_model_loaded":true,"minilm_model_loaded":true,"mpnet_model_loaded":true,"clip_model_loaded":true}
```

## Pendant la démo — montrer le scaling horizontal

Le HPA ne montera pas à 2 répliques sous charge réelle faible. Pour une
démo fiable et rapide, forcer temporairement `minReplicas: 2` :

```bash
kubectl patch hpa predict-api -n rakuten-mlops --type='merge' -p '{"spec":{"minReplicas":2}}'

# observer le 2e pod se créer et charger ses modèles (~2-3 min)
kubectl get pods -n rakuten-mlops -w
kubectl get hpa -n rakuten-mlops
```

Une fois les 2 pods `1/1 Running`, c'est démontré : 2 réplicas indépendantes,
chacune avec ses 4 modèles chargés, derrière le même Service.

## Après la démo — tout remettre à 0

```bash
# Revenir à minReplicas: 1 (le 2e pod se termine automatiquement
# après la fenêtre de stabilisation HPA, ~5 min)
kubectl patch hpa predict-api -n rakuten-mlops --type='merge' -p '{"spec":{"minReplicas":1}}'

# Puis scaler le Deployment à 0 et supprimer le HPA (sinon il
# remonterait le pod au prochain redémarrage du PC)
kubectl scale deployment predict-api -n rakuten-mlops --replicas=0
kubectl delete hpa predict-api -n rakuten-mlops
```

## Suppression complète (si le PoC n'est plus nécessaire)

```bash
kubectl delete namespace rakuten-mlops
```

Pour tout recréer depuis zéro, voir [`README.md`](README.md) (section
"Deploy") — nécessite de recréer le Secret depuis `.env` :

```bash
kubectl create secret generic predict-api-secrets -n rakuten-mlops --from-env-file=.env
```
