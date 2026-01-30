# Signal.Engine v4.0 - Kaggle Upload Helper (Windows PowerShell)
# Run this to package and upload your code to Kaggle

Write-Host "🚀 Signal.Engine v4.0 - Kaggle Upload Helper" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Check if kaggle CLI is installed
$kaggleInstalled = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $kaggleInstalled) {
    Write-Host "❌ Kaggle CLI not found. Installing..." -ForegroundColor Red
    pip install kaggle
}

# Create temporary directory for dataset
$tempDir = ".\kaggle_upload_temp"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}
New-Item -ItemType Directory -Path "$tempDir\src" | Out-Null

try {
    Write-Host "📦 Copying files..." -ForegroundColor Yellow

    # Copy essential files
    $files = @(
        "src\data_loader.py",
        "src\env.py",
        "src\train_ppo_optimized.py",
        "src\ticker_utils.py",
        "src\tda_features.py",
        "src\data_labeler.py"
    )

    foreach ($file in $files) {
        Copy-Item $file "$tempDir\$file" -ErrorAction Stop
    }

    # Create __init__.py if it doesn't exist
    if (-not (Test-Path "src\__init__.py")) {
        New-Item -ItemType File -Path "$tempDir\src\__init__.py" | Out-Null
    } else {
        Copy-Item "src\__init__.py" "$tempDir\src\__init__.py"
    }

    # Copy metadata
    Copy-Item "kaggle_dataset_metadata.json" "$tempDir\dataset-metadata.json"

    # Create README
    $readme = @"
# Signal.Engine v4.0

Reinforcement Learning trading agent with:
- ✅ Volatility-Targeted Position Sizing (20% target vol)
- ✅ Cross-Sectional Factor Ranking (RSI_Rank, Momentum_Rank)
- ✅ Sharpe-Aware Reward Function
- ✅ 9-Dimensional Observations (7 features + 2 context)

## Training

See ``src/train_ppo_optimized.py`` - just run ``main()``

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
"@

    Set-Content -Path "$tempDir\README.md" -Value $readme

    Write-Host "✅ Files packaged in $tempDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "📤 Uploading to Kaggle..." -ForegroundColor Yellow

    # Upload to Kaggle
    Push-Location $tempDir
    try {
        kaggle datasets create -p .
        Write-Host ""
        Write-Host "✅ Upload complete!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Upload failed: $_" -ForegroundColor Red
        Write-Host "Make sure you have configured Kaggle API credentials"
        Write-Host "See: https://github.com/Kaggle/kaggle-api#api-credentials"
        throw
    } finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Go to https://www.kaggle.com/datasets"
    Write-Host "2. Find your 'signal-engine-v4' dataset"
    Write-Host "3. Create a new notebook and add this dataset"
    Write-Host "4. Follow the training guide in docs/KAGGLE_TRAINING_GUIDE.md"
    Write-Host ""
    Write-Host "🎯 Happy Training!" -ForegroundColor Green

} finally {
    # Cleanup always runs, even if errors occurred
    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir
    }
}
