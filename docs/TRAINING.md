# Training Guide

## Training Options

### Option 1: Train with Automatic Versioning (Recommended)

```bash
python -m src.train_with_versioning --model lstm --epochs 20
```

This automatically:
- Trains the model
- Evaluates on test set
- Saves to versioned directory (v1, v2, etc.)

### Option 2: Train Without Versioning

```bash
python -m src.train --model lstm --epochs 20
```

Then register manually:
```bash
python -m src.register_model --model-path models/rul_lstm.pth --model-type lstm --rmse 24.04 --mae 16.81
```

### Option 3: Train All Models

```bash
python -m src.train_all --epochs 20
```

## Model Types

- `lstm` - Standard LSTM
- `bilstm` - Bidirectional LSTM
- `gru` - GRU
- `transformer` - Transformer encoder

## Hyperparameters

All configurable via command-line arguments:
- `--epochs` - Number of training epochs
- `--batch-size` - Batch size
- `--lr` - Learning rate
- `--seq-length` - Sequence length

See `python -m src.train --help` for all options.

