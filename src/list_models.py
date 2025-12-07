"""
List all registered model versions.
Usage: python -m src.list_models
"""

import argparse

from src.model_registry import get_version_info, list_model_versions

try:
    from tabulate import tabulate  # type: ignore[import-untyped]

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def main():
    parser = argparse.ArgumentParser(description="List all registered model versions")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Show details for a specific version",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json", "csv"],
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    # Get all versions
    all_versions = list_model_versions()

    if not all_versions:
        print("📭 No model versions found.")
        return

    # Show specific version details
    if args.version:
        try:
            metadata = get_version_info(args.version)
            print(f"\n📋 Model Version v{args.version} Details:")
            print("=" * 70)

            if args.format == "json":
                import json

                print(json.dumps(metadata, indent=2))
            else:
                print(f"Model Type:      {metadata.get('model_type', 'N/A')}")
                print(f"RMSE:            {metadata.get('rmse', 0):.2f}")
                print(f"MAE:             {metadata.get('mae', 0):.2f}")
                print(f"Sequence Length: {metadata.get('sequence_length', 'N/A')}")
                print(f"Input Features:  {metadata.get('input_features', 'N/A')}")
                print(f"Timestamp:       {metadata.get('timestamp', 'N/A')}")

                if "epochs" in metadata:
                    print(f"Epochs:          {metadata['epochs']}")
                if "batch_size" in metadata:
                    print(f"Batch Size:      {metadata['batch_size']}")
                if "learning_rate" in metadata:
                    print(f"Learning Rate:   {metadata['learning_rate']}")

            print("=" * 70)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
        return

    # Show all versions
    print(f"\n📚 Found {len(all_versions)} model version(s):\n")

    if args.format == "json":
        import json

        print(json.dumps(all_versions, indent=2))
    elif args.format == "csv":
        import csv
        import sys

        if all_versions:
            fieldnames = ["version", "model_type", "rmse", "mae", "timestamp"]
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for v in all_versions:
                writer.writerow(
                    {
                        "version": v.get("version", ""),
                        "model_type": v.get("model_type", ""),
                        "rmse": v.get("rmse", 0),
                        "mae": v.get("mae", 0),
                        "timestamp": v.get("timestamp", "")[:19] if v.get("timestamp") else "",
                    }
                )
    else:
        # Table format
        table_data = []
        for meta in all_versions:
            version = meta.get("version", "N/A")
            model_type = meta.get("model_type", "N/A")
            rmse = meta.get("rmse", 0)
            mae = meta.get("mae", 0)
            timestamp = meta.get("timestamp", "N/A")
            if isinstance(timestamp, str) and len(timestamp) > 19:
                timestamp = timestamp[:19]  # Truncate to date+time

            table_data.append(
                [
                    f"v{version}",
                    model_type.upper(),
                    f"{rmse:.2f}",
                    f"{mae:.2f}",
                    timestamp,
                ]
            )

        headers = ["Version", "Model Type", "RMSE", "MAE", "Timestamp"]
        if HAS_TABULATE:
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            # Fallback to simple table format
            print(f"{'Version':<10} {'Model Type':<15} {'RMSE':<10} {'MAE':<10} {'Timestamp':<20}")
            print("-" * 70)
            for row in table_data:
                print(f"{row[0]:<10} {row[1]:<15} {row[2]:<10} {row[3]:<10} {row[4]:<20}")

        # Find best model
        if all_versions:
            best_rmse = min(all_versions, key=lambda x: x.get("rmse", float("inf")))
            best_mae = min(all_versions, key=lambda x: x.get("mae", float("inf")))

            print(
                f"\n🏆 Best RMSE: v{best_rmse.get('version')} ({best_rmse.get('model_type', '').upper()}) - {best_rmse.get('rmse', 0):.2f}"
            )
            print(
                f"🏆 Best MAE:  v{best_mae.get('version')} ({best_mae.get('model_type', '').upper()}) - {best_mae.get('mae', 0):.2f}"
            )


if __name__ == "__main__":
    main()
