# kubeflow_demo

## Commands

[Installation guidelines](https://www.kubeflow.org/docs/components/pipelines/v1/installation/localcluster-deployment/)

### How to create the local K8S dev cluster

```bash
# To create
kind create cluster

# To delete
kind delete cluster
```

### How to deploy Kubeflow Pipelines

```bash
export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# Wait a couple of minutes.

kubectl config set-context --current --namespace=kubeflow

# Port forwading for the UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Kubeflow UI will be at [http://localhost:8080](http://localhost:8080/)

```bash
export PIPELINE_VERSION=1.8.5
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
```

### How to setup local env for developing

```bash
python3 -m venv/venv
source venv/bin/activate
pip install -r requirements.txt
```

### How to compile the pipeline

```bash
# From the venv
dsl-compile --py 2_demo_pipeline.py --output 2_demo_pipeline.yml
```
