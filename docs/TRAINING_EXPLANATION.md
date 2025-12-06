# Training Explanation: Single Model vs All Models

## ❓ Question: Does `python -m src.train --model lstm` train all models?

**Answer: NO** - It only trains **ONE model** (LSTM in this case).

## 📝 How the Training Script Works

### Single Model Training

The `src/train.py` script trains **ONE model at a time** based on the `--model` argument:

```bash
# Train ONLY LSTM
python -m src.train --model lstm

# Train ONLY BiLSTM
python -m src.train --model bilstm

# Train ONLY GRU
python -m src.train --model gru

# Train ONLY Transformer
python -m src.train --model transformer
```

**Each command trains only the specified model.**

## 🚀 How to Train All Models

You have **two options**:

### Option 1: Train Each Model Separately (Manual)

Run the training command 4 times:

```bash
# Train LSTM
python -m src.train --model lstm --epochs 20

# Train BiLSTM
python -m src.train --model bilstm --epochs 20

# Train GRU
python -m src.train --model gru --epochs 20

# Train Transformer
python -m src.train --model transformer --epochs 20
```

**Pros:**
- Full control over each training run
- Can use different hyperparameters for each model
- Can stop/resume individual models

**Cons:**
- Manual, repetitive
- Need to run 4 separate commands

### Option 2: Train All Models Automatically (Recommended)

Use the new `train_all.py` script to train all models sequentially:

```bash
# Train ALL models with default settings
python -m src.train_all

# Train ALL models with custom epochs
python -m src.train_all --epochs 30

# Train ALL models with custom hyperparameters
python -m src.train_all --epochs 20 --batch-size 128 --lr 0.0005

# Train specific models only
python -m src.train_all --models lstm bilstm
```

**Pros:**
- One command trains all models
- Automatic sequential training
- Summary report at the end
- Can specify which models to train

**Cons:**
- All models use the same hyperparameters
- Takes longer (trains sequentially)

## 📊 Comparison

| Method | Command | Trains | Use Case |
|--------|---------|--------|----------|
| **Single Model** | `python -m src.train --model lstm` | 1 model | Testing, fine-tuning specific model |
| **All Models (Manual)** | 4 separate commands | 4 models | Full control, different hyperparameters |
| **All Models (Auto)** | `python -m src.train_all` | 4 models | Quick training of all models |

## 🎯 Recommended Workflow

### For First-Time Training:

```bash
# Step 1: Train all models
python -m src.train_all --epochs 20

# Step 2: Compare all models
python -m src.compare_models
```

### For Fine-Tuning:

```bash
# Train specific model with custom hyperparameters
python -m src.train --model lstm --epochs 30 --batch-size 128 --lr 0.0005

# Evaluate
python -m src.evaluate --model lstm
```

## 📋 What Happens During Training

When you run `python -m src.train --model lstm`:

1. ✅ Loads training data from `data/raw/train_FD001.txt`
2. ✅ Calculates RUL for each engine
3. ✅ Fits StandardScaler on training features (saves to `models/scaler.pkl`)
4. ✅ Scales training features
5. ✅ Generates sequences (sliding windows)
6. ✅ Creates LSTM model architecture
7. ✅ Trains the model for specified epochs
8. ✅ Saves model to `models/rul_lstm.pth`

**Only the LSTM model is trained and saved.**

## 🔍 Verify What Was Trained

Check the `models/` directory:

```bash
# List trained models
ls models/*.pth

# Should see:
# - rul_lstm.pth (if LSTM was trained)
# - bi_lstm.pth (if BiLSTM was trained)
# - gru.pth (if GRU was trained)
# - transformer_rul.pth (if Transformer was trained)
```

## 💡 Summary

- **`python -m src.train --model lstm`** → Trains **ONLY LSTM**
- **`python -m src.train_all`** → Trains **ALL models** (LSTM, BiLSTM, GRU, Transformer)

Choose the method that fits your workflow!

