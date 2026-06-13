# Configs

## `spanish.yaml`

Example training config. **It is NOT directly consumed by `train.py`** — `train.py`
takes its config from CLI flags only, to stay simple and Nautilus-friendly. The YAML
is a structured place to record what flags one run uses. Copy values into a CLI
invocation, or write a small wrapper that reads the YAML.

## `sweep.yaml`

W&B sweep definition. This IS consumed by `wandb sweep` to register a sweep on
your W&B project, and then by `wandb agent <sweep-id>` to actually run the
configs. See the comments inside `sweep.yaml` for the two-step launch.

A typical sweep on 4 GPUs:
```bash
export WANDB_API_KEY=...   # https://wandb.ai/settings → API keys
SWEEP_ID=$(wandb sweep configs/sweep.yaml | grep -oP 'agent \S+/\S+/\S+' | awk '{print $2}')

# Launch 4 agents in parallel (one per GPU)
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu wandb agent $SWEEP_ID &
done
wait
```

On Nautilus, run each agent in its own Job pod (see `nautilus/sweep-job.yaml`).
