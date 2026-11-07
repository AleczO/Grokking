"""Turning CLI arguments into a run config."""
from .checkpoints import NextRunDir

# Training hyperparameters taken from the CLI and stored in config["training"].
TRAINING_KEYS = ["lr", "weight_decay", "epochs", "train_frac", "seed"]


def list_experiments(experiments):
    print("Available experiments:")
    for name in sorted(experiments):
        print(f"  {name}")


def build_config(args, experiments):
    """Build the run config from CLI args. Returns None if nothing should be trained."""
    if args.list or not args.experiment:
        list_experiments(experiments)
        return None

    if args.N is None:
        raise SystemExit("--N is required to run an experiment")

    config = experiments[args.experiment](args.N)
    config["training"] = {key: getattr(args, key) for key in TRAINING_KEYS}
    config["experiment"]["save_path"] = NextRunDir(config["experiment"]["save_path"])
    return config