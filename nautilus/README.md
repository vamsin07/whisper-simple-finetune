# Running on Nautilus

This directory contains everything needed to run vanilla Whisper fine-tuning on
the [Pacific Research Platform Nautilus](https://nautilus.optiputer.net/)
Kubernetes cluster.

## One-time setup

### 1. Get a Nautilus account + namespace
Follow the official onboarding at <https://docs.nautilus.optiputer.net/userdocs/start/get-access/>.
Once you have a namespace, set it as your kubectl default:
```bash
kubectl config set-context --current --namespace=<your-namespace>
```

### 2. Create PVCs (one shared HF cache, one personal output)
```bash
# Shared HF cache — many pods can mount read-only; first one populates it.
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: hf-cache-shared}
spec:
  accessModes: [ReadWriteMany]
  resources: {requests: {storage: 100Gi}}
  storageClassName: rook-cephfs   # or whatever ReadWriteMany class your namespace has
EOF

# Personal output PVC — checkpoints + JSON results land here.
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: vamsin07-output}      # rename to your username
spec:
  accessModes: [ReadWriteOnce]
  resources: {requests: {storage: 200Gi}}
  storageClassName: rook-ceph-block
EOF
```

### 3. Add secrets for W&B and (optionally) HuggingFace
```bash
# Get your W&B key at https://wandb.ai/settings → API keys
kubectl create secret generic wandb-credentials \
  --from-literal=api-key=YOUR_WANDB_KEY

# Only needed if you'll pull a gated model from HF (e.g. Llama, gated Whisper variants)
kubectl create secret generic hf-credentials \
  --from-literal=token=hf_xxxxxxxxxx
```

### 4. Build and push the container image
```bash
# From the repo root
docker build -t gitlab-registry.nautilus.optiputer.net/YOUR-GROUP/whisper-vanilla:latest \
  -f nautilus/Dockerfile .

docker push gitlab-registry.nautilus.optiputer.net/YOUR-GROUP/whisper-vanilla:latest
```
(See <https://docs.nautilus.optiputer.net/userdocs/development/docker/> for
how to authenticate to the gitlab registry the first time.)

## Single fine-tune

Edit the placeholders in `train-job.yaml` (search for `EDIT ME` — at minimum
the image path, the PVC names, and the `--lang` / `--fleurs-code` args), then:

```bash
kubectl apply -f nautilus/train-job.yaml
kubectl logs -f job/whisper-vft-spanish     # watch the run
kubectl delete job whisper-vft-spanish      # clean up when done
```

To retrieve the checkpoint when the job finishes, start a short-lived pod that
mounts the output PVC and copy locally:
```bash
kubectl run grab --rm -it --restart=Never \
  --image=busybox --overrides='{
    "spec":{"containers":[{"name":"grab","image":"busybox","command":["sh"],
            "volumeMounts":[{"name":"o","mountPath":"/o"}]}],
            "volumes":[{"name":"o","persistentVolumeClaim":{"claimName":"vamsin07-output"}}]}}'
# inside the pod:  ls /o/spanish-base/best/   # checkpoint.pt + training_config.json
```
Or use `kubectl cp` against a running pod that mounts the same PVC.

## W&B hyperparameter sweep

```bash
# 1) On your laptop (NOT on Nautilus): register the sweep
wandb sweep configs/sweep.yaml
# → prints "wandb agent <user>/<project>/<sweep-id>"
SWEEP_ID=<user>/<project>/<sweep-id>

# 2) Launch N agent pods (one per parallel worker / GPU you want to use)
for i in 1 2 3 4; do
  sed "s|SWEEP_ID_PLACEHOLDER|$SWEEP_ID|; s|agent-NAME|agent-$i|" \
    nautilus/sweep-agent-job.yaml | kubectl apply -f -
done

# 3) Watch on wandb.ai — the sweep page shows all agents + runs in real time
# 4) When done, clean up
kubectl delete jobs -l app=whisper-vft-sweep
```

## Tips

- **Pod stuck `Pending`**: usually no GPU is available — check `kubectl describe job/<name>`. For Whisper-base any GPU works; for Whisper-large-v3 add a `nodeSelector` for A100/H100.
- **First run is slow**: Whisper checkpoint (~300MB for base, ~6GB for large-v3) + FLEURS audio (~10GB per lang) are downloaded into the shared HF cache PVC. Second+ runs reuse the cache and start in ~30s.
- **Cost**: vanilla Whisper-base fine-tunes finish in ~30-60 min on one consumer GPU. Whisper-large-v3 in 2-4h on A100. Sweep agents finish whenever the sweep's `metric` plateaus.
