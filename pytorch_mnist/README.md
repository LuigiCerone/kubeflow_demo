## Setup local cluster

For the local cluster setup please follow [README.md](../README.md) instructions. 

## Setup registry for image pulling

```bash
kubectl -n kubeflow create secret docker-registry registry-secret \
--docker-server=https://index.docker.io/v1/ \
--docker-username=<username> \
--docker-password=<access-key> \
--docker-email=<email>
```

