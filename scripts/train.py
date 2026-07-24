"""Run the minimal end-to-end Model B baseline from YAML configuration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.config import apply_runtime_overrides, load_config
from gnn_siamese.training import fit_model_b_baseline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--device", default=None, help="Override training.device.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the reduced smoke-test setup.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    config["__config_path__"] = str(Path(args.config).resolve())
    config = apply_runtime_overrides(config, device=args.device, smoke_test=args.smoke_test)
    pipeline = None
    try:
        pipeline = build_training_pipeline(config)
        output = fit_model_b_baseline(
            pipeline.model,
            train_dataloader=pipeline.dataloaders.train_loader,
            validation_dataloader=pipeline.dataloaders.validation_loader,
            optimizer=pipeline.optimizer,
            loss_fn=pipeline.total_loss_assembler,
            epochs=int(pipeline.config["training"]["epochs"]),
            device=pipeline.device,
            augmenter=pipeline.augmenter,
        )

        if not args.smoke_test:
            print(
                "split_path=",
                pipeline.split_bundle.split_path,
                "created=",
                pipeline.split_bundle.created,
                "device=",
                output.device,
            )
            print(
                "train_loss=",
                f"{output.final_train_loss:.6f}",
                "validation_loss=",
                f"{output.final_validation_loss:.6f}",
                "epochs_completed=",
                output.epochs_completed,
            )
            return 0

        if not (math.isfinite(output.final_train_loss) and math.isfinite(output.final_validation_loss)):
            raise RuntimeError("Smoke test finished with non-finite losses.")

        smoke_source = "configured_paths"
        if pipeline.smoke_data is not None:
            smoke_source = pipeline.smoke_data.source

        print(f"smoke_dataset_source={smoke_source}")
        print(f"train_examples={len(pipeline.dataloaders.train_dataset)}")
        print(f"validation_examples={len(pipeline.dataloaders.validation_dataset)}")
        print(f"node_dim={pipeline.dataset.node_input_dim}")
        print(f"epochs_completed={output.epochs_completed}")
        print(f"train_loss={output.final_train_loss:.6f}")
        print(f"validation_loss={output.final_validation_loss:.6f}")
        for metric_name, metric_value in sorted(output.final_validation_metrics.items()):
            if isinstance(metric_value, bool):
                print(f"{metric_name}={str(metric_value).lower()}")
            elif isinstance(metric_value, str):
                print(f"{metric_name}={metric_value}")
            elif isinstance(metric_value, (int, float)):
                print(f"{metric_name}={metric_value:.6f}")
        print(f"device={output.device}")
        return 0
    finally:
        if pipeline is not None and pipeline.smoke_data is not None:
            pipeline.smoke_data.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
