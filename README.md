# lab5-1811232011-CHL

## Scripts

### Normal

python dqn-2.py --wandb-run-name pong-mps-3-1 --save-dir ./runs/result-pong-mps-3-1 --episodes 20000

### Continue Training

```bash
python dqn-2.py --wandb-run-name pong-continued --load-ckpt ./results/ckpt_ep900.pt --batch-size 32 --memory-size 200000
```