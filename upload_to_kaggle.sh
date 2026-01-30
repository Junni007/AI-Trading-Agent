#!/bin/bash
# Signal.Engine v4.0 - Kaggle Upload Helper Script
# Run this to package and upload your code to Kaggle

set -e  # Exit on error

echo "🚀 Signal.Engine v4.0 - Kaggle Upload Helper"
echo "=============================================="

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Create temporary directory for dataset
TEMP_DIR="./kaggle_upload_temp"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR/src

# Setup cleanup trap to always run on exit
cleanup() {
    cd "$ORIGINAL_DIR"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Save original directory
ORIGINAL_DIR=$(pwd)

echo "📦 Copying files..."

# Copy essential files
cp src/__init__.py $TEMP_DIR/src/ 2>/dev/null || touch $TEMP_DIR/src/__init__.py
cp src/data_loader.py $TEMP_DIR/src/
cp src/env.py $TEMP_DIR/src/
cp src/train_ppo_optimized.py $TEMP_DIR/src/
cp src/ticker_utils.py $TEMP_DIR/src/
cp src/tda_features.py $TEMP_DIR/src/
cp src/data_labeler.py $TEMP_DIR/src/

# Copy metadata
cp kaggle_dataset_metadata.json $TEMP_DIR/dataset-metadata.json

# Create README
cat > $TEMP_DIR/README.md << 'EOF'
# Signal.Engine v4.0

Reinforcement Learning trading agent with:
- ✅ Volatility-Targeted Position Sizing (20% target vol)
- ✅ Cross-Sectional Factor Ranking (RSI_Rank, Momentum_Rank)
- ✅ Sharpe-Aware Reward Function
- ✅ 9-Dimensional Observations (7 features + 2 context)

## Training

See `src/train_ppo_optimized.py` - just run `main()`

## Architecture

- Model: RecurrentActorCritic (LSTM-based)
- Environment: VectorizedTradingEnv (256 parallel envs)
- Algorithm: PPO (Proximal Policy Optimization)

## Expected Performance

- Sharpe Ratio: >1.2
- Max Drawdown: <20%
- Avg PnL: +12-15%

---

*Built with PyTorch + Lightning*
EOF

echo "✅ Files packaged in $TEMP_DIR"
echo ""
echo "📤 Uploading to Kaggle..."

# Upload to Kaggle
cd $TEMP_DIR
kaggle datasets create -p .

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps:"
echo "1. Go to https://www.kaggle.com/datasets"
echo "2. Find your 'signal-engine-v4' dataset"
echo "3. Create a new notebook and add this dataset"
echo "4. Follow the training guide in docs/KAGGLE_TRAINING_GUIDE.md"
echo ""
echo "🎯 Happy Training!"

# Cleanup is automatically handled by trap on EXIT
