"""
Train all models sequentially (LSTM, BiLSTM, GRU, Transformer).
Usage: python -m src.train_all --epochs 20
"""

import argparse
import subprocess
import sys

from src.config import BATCH_SIZE, EPOCHS, LR, SEQ_LENGTH


def main():
    parser = argparse.ArgumentParser(description="Train all RUL prediction models")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate (default: {LR})",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=SEQ_LENGTH,
        help=f"Sequence length (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["lstm", "bilstm", "gru", "transformer"],
        choices=["lstm", "bilstm", "gru", "transformer"],
        help="Models to train (default: all)",
    )

    args = parser.parse_args()

    models_to_train = args.models
    print("=" * 70)
    print("🚀 Training All Models")
    print("=" * 70)
    print(f"Models to train: {', '.join([m.upper() for m in models_to_train])}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 70)
    print()

    results = {}

    for i, model_name in enumerate(models_to_train, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(models_to_train)}] Training {model_name.upper()} model")
        print("=" * 70)

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seq-length",
            str(args.seq_length),
        ]

        # Run training
        try:
            subprocess.run(cmd, check=True)
            results[model_name] = "✅ Success"
            print(f"\n✅ {model_name.upper()} training completed successfully!")
        except subprocess.CalledProcessError as e:
            results[model_name] = f"❌ Failed (exit code: {e.returncode})"
            print(f"\n❌ {model_name.upper()} training failed!")
            print(f"   Error: {e}")
        except Exception as e:
            results[model_name] = f"❌ Failed: {str(e)}"
            print(f"\n❌ {model_name.upper()} training failed!")
            print(f"   Error: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Training Summary")
    print("=" * 70)
    for model_name, status in results.items():
        print(f"{model_name.upper():<15} {status}")
    print("=" * 70)

    # Check if all succeeded
    all_success = all("✅" in status for status in results.values())
    if all_success:
        print("\n🎉 All models trained successfully!")
        print("\nNext steps:")
        print("  1. Evaluate models: python -m src.evaluate --model <model_name>")
        print("  2. Compare all models: python -m src.compare_models")
    else:
        print("\n⚠️  Some models failed to train. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
