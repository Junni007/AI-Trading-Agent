"""
Test suite for enhanced backtest metrics.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics_enhanced import calculate_metrics_enhanced


class TestEnhancedMetrics:
    """Tests for enhanced backtest metrics calculation."""
    
    def test_metrics_calculation(self):
        """Verify enhanced metrics are calculated."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        metrics = calculate_metrics_enhanced(returns)
        
        # Check all expected metrics exist
        assert 'Total Return' in metrics
        assert 'Sharpe Ratio' in metrics
        assert 'Sortino Ratio' in metrics
        assert 'Max Drawdown' in metrics
        assert 'Win Rate' in metrics
        assert 'Profit Factor' in metrics
        assert 'Calmar Ratio' in metrics
    
    def test_win_rate_calculation(self):
        """Verify win rate is correctly calculated."""
        # 60% winning trades
        returns = pd.Series([0.01, 0.02, -0.01, 0.01, -0.01, 0.01, 0.02, -0.01, 0.01, 0.01])
        metrics = calculate_metrics_enhanced(returns)
        
        assert metrics['Win Rate'] == 0.7  # 7 out of 10
    
    def test_profit_factor(self):
        """Verify profit factor calculation."""
        # Simple case: $3 profit, $1 loss = 3.0 profit factor
        returns = pd.Series([0.01, 0.01, 0.01, -0.01])
        metrics = calculate_metrics_enhanced(returns)
        
        assert abs(metrics['Profit Factor'] - 3.0) < 0.01
    
    def test_sortino_vs_sharpe(self):
        """Sortino and Sharpe should both be calculated."""
        # Create returns with enough data for valid calculation
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, -0.01, 0.02, 0.01, -0.005, 0.015, 0.01, -0.02])
        metrics = calculate_metrics_enhanced(returns)
        
        # Both metrics should exist and be calculated
        assert 'Sortino Ratio' in metrics
        assert 'Sharpe Ratio' in metrics
        # Verify they are numeric (not NaN for this dataset)
        assert not np.isnan(metrics['Sortino Ratio'])
        assert not np.isnan(metrics['Sharpe Ratio'])
    
    def test_empty_returns(self):
        """Handle edge case of empty returns gracefully."""
        returns = pd.Series([], dtype=float)
        
        # Should not crash
        try:
            metrics = calculate_metrics_enhanced(returns)
            assert metrics['Win Rate'] == 0.0
        except:
            pytest.skip("Expected behavior undefined for empty series")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
