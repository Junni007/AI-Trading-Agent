# Signal.Engine v4.0: Bug Fixes Summary

**Date**: 2026-01-30  
**Status**: ✅ All Issues Resolved

---

## Issues Fixed

### 1. **Column Name Mismatch** (`src/data_loader.py`)
- **Lines**: 325-326, 340
- **Issue**: Code checked for 'Returns' but feature_engineering() produces 'Log_Return'
- **Fix**: Updated `compute_cross_sectional_features()` to use 'Log_Return' in checks and rank computation
- **Impact**: Cross-sectional ranking now processes rows correctly

### 2. **ATR Double-Normalization** (`src/env.py`)
- **Lines**: 169-174
- **Issue**: ATR already normalized by price in data_loader, dividing again was incorrect
- **Fix**: Changed `current_vol = atr / (current_price + 1e-8)` to `current_vol = atr`
- **Impact**: Proper volatility targeting without double-normalization

### 3. **Test Fixture Column Name** (`test_v4_features.py`)
- **Lines**: 66-79
- **Issue**: Test data used 'Returns' but pipeline expects 'Log_Return'
- **Fix**: Updated all test DataFrames to use 'Log_Return' column
- **Impact**: Tests now match production code

### 4. **Duplicate Assignment** (`src/train_ppo_optimized.py`)
- **Lines**: 320-322
- **Issue**: `self.ppo_epochs` assigned twice in constructor
- **Fix**: Removed duplicate assignment
- **Impact**: Cleaner code, no functional change

### 5. **Comment Mismatch** (`src/train_ppo_optimized.py`)
- **Lines**: 559-561
- **Issue**: Comment said "64 environments" but N_ENVS = 256
- **Fix**: Updated comment to "Run 256 environments in parallel"
- **Impact**: Documentation now matches code

### 6. **Insecure torch.load** (`src/train_ppo_optimized.py`)
- **Lines**: 594-603
- **Issue**: Missing `weights_only=True` and poor error logging
- **Fix**: Added `weights_only=True` and changed `logger.error` to `logger.exception`
- **Impact**: Secure deserialization + full stack traces on failure

### 7. **env.step() Unpacking** (`test_v4_integration.py`)
- **Lines**: 47-56
- **Issue**: Test expected 4 return values but VectorizedTradingEnv.step() returns 3
- **Fix**: Changed `new_obs, rewards, dones, info = env.step()` to `new_obs, rewards, dones = env.step()`
- **Impact**: Test now matches API correctly

### 8. **Unconditional Success Banner** (`test_v4_integration.py`)
- **Lines**: 98-110
- **Issue**: Summary always printed success even if tests failed
- **Fix**: Added `test_results` tracking dict, conditional success/failure reporting
- **Impact**: Accurate test reporting

---

## Verification Results

### test_v4_features.py
```
✅ Volatility Targeting: WORKING
✅ Cross-Sectional Ranking: WORKING
```

### test_v4_integration.py
```
✅ Observation dimension: 9 (7 features + 2 context)
✅ Reset successful. Obs shape: torch.Size([4, 50, 9])
✅ Step successful
✅ Extended simulation successful (20 steps)
✅ Model created. Input dim: 9, Output dim: 3
✅ Forward pass successful
```

**Final Status**: All v4.0 components verified and ready for training! 🚀

---

## Files Modified

| File | LOC Changed | Description |
| :--- | :--- | :--- |
| `src/data_loader.py` | ~5 | Fixed Returns→Log_Return |
| `src/env.py` | ~2 | Fixed ATR normalization |
| `test_v4_features.py` | ~6 | Fixed test column names |
| `src/train_ppo_optimized.py` | ~4 | Removed duplicate, fixed comment, secure load |
| `test_v4_integration.py` | ~15 | Fixed unpacking + test tracking |

**Total**: ~32 lines changed across 5 files
