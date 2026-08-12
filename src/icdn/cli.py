"""Command line interface: ``icdn fit | predict | elasticities``."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .api import ICDNModel
from .config import ICDNConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="icdn", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    fit = subcommands.add_parser("fit", help="train a model and save it")
    fit.add_argument("--data", required=True, help="panel in csv or parquet format")
    fit.add_argument("--config", help="yaml configuration file")
    fit.add_argument("--out", required=True, help="destination of the trained model")

    for name, help_text in [
        ("predict", "predict demand with a trained model"),
        ("elasticities", "estimate elasticities with a trained model"),
    ]:
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("--model", required=True, help="path to a saved model")
        command.add_argument("--data", required=True, help="panel in csv or parquet format")
        command.add_argument("--out", required=True, help="destination csv or parquet file")

    args = parser.parse_args(argv)
    panel = read_table(args.data)

    if args.command == "fit":
        config = ICDNConfig.from_yaml(args.config) if args.config else ICDNConfig()
        model = ICDNModel(config).fit(panel)
        path = model.save(args.out)
        print(f"model saved to {path}")
        return 0

    model = ICDNModel.load(args.model)
    result = model.predict(panel) if args.command == "predict" else model.elasticities(panel)
    write_table(result, args.out)
    print(f"{len(result)} rows written to {args.out}")
    return 0


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format '{suffix}', use csv or parquet")


def write_table(frame: pd.DataFrame, path: str) -> None:
    if Path(path).suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


if __name__ == "__main__":
    sys.exit(main())
