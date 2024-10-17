import os

WANDB_REFRESH_INTERVAL = int(os.getenv("WANDB_REFRESH_INTERVAL", str(7200)))
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "w-ai-wombo")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "rule-30")
WANDB_ENABLED = os.getenv("WANDB_ENABLED", str(True)).lower() == str(True).lower()
