import argparse
from pathlib import Path

import wandb


def download_artifacts(run_path: str) -> None:
    """Download model checkpoint."""
    output_dir = Path(__file__).parents[1].resolve() / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    wandb_run = api.run(f"{run_path}")

    artifacts = {
        "tokenizer.json": "Tokenizer",
        "lit_model.ckpt": "Model checkpoint",
    }

    for file in wandb_run.files():
        if file.name in artifacts:
            file.download(root=output_dir, replace=True)
            print(f"{file.name} downloaded to {str(output_dir / file.name)}.")
            artifacts.pop(file.name)

    for artifact in artifacts:
        print(f"{artifact} not found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str)
    args = parser.parse_args()
    download_artifacts(args.run_path)


if __name__ == "__main__":
    main()
