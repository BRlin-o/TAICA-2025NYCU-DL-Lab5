# lab5-1811232011-CHL

## Scripts

### Normal

```bash
python dqn-3.py --wandb-run-name pong-mps-3-3 --save-dir ./runs/result-pong-mps-3-3 --episodes 5000 --memory-size 500000
```

### Continue Training

```bash
python dqn-2.py --wandb-run-name pong-continued --load-ckpt ./results/ckpt_ep900.pt --batch-size 32 --memory-size 200000
```

### 3.2V

```bash
python dqn-3-2.py --wandb-run-name pong-vector4-cuda-fast --save-dir ./runs/result-pong-vector4-cuda-fast --episodes 5000 --batch-size 128 --train-per-step 8 --target-update-frequency 50000
```