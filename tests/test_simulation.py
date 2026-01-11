import pytest
import numpy as np
from src.simulation.engine import SimulationEngine
from src.config import settings

# Use a temporary file for testing
settings.STATE_FILE = settings.BASE_DIR / "test_simulation_state.json"

@pytest.fixture
def sim_engine():
    engine = SimulationEngine()
    engine.reset() # Ensure clean state
    return engine

def test_initial_state(sim_engine):
    state = sim_engine.get_portfolio()
    assert state['balance'] == 10000.0
    assert state['cash'] == 10000.0
    assert len(state['positions']) == 0

def test_buy_logic(sim_engine):
    # Mock Market Data with High Confidence Buy
    market_data = [{
        'Ticker': 'TEST', 
        'Price': 100.0, 
        'Signal': 'BUY', 
        'Confidence': 0.95,
        'Reason': 'High Volume'
    }]
    
    logs = sim_engine.process_tick(market_data)
    
    state = sim_engine.get_portfolio()
    assert 'TEST' in state['positions']
    assert state['positions']['TEST']['qty'] > 0
    assert "BOUGHT TEST" in logs[0]

def test_stats_calculation(sim_engine):
    # Simulate an equity curve
    sim_engine.state['equity_curve'] = [10000, 10200, 10100, 10300, 10500]
    sim_engine.calculate_stats()
    
    state = sim_engine.get_portfolio()
    assert state['max_drawdown'] > 0 # Should have some DD from 10200 -> 10100
    assert state['sharpe_ratio'] > 0 # Positive trend

def test_regime_impact(sim_engine):
    # Test High Volatility Regime (Should increase Profit Target)
    # 1. Buy
    sim_engine.state['cash'] = 10000
    sim_engine.state['positions'] = {'TEST': {'qty': 10, 'avg_price': 100}}
    sim_engine.state['balance'] = 10000
    
    # 2. Price moves up 1.5% (Normal Target 1.0% would sell, but High Vol Target 2.0% should hold)
    market_data = [{
        'Ticker': 'TEST', 
        'Price': 101.5, 
        'Confidence': 0.0
    }]
    
    # Process with HIGH_VOLATILITY regime
    logs = sim_engine.process_tick(market_data, regime="HIGH_VOLATILITY")
    
    # Check: Should NOT have sold
    state = sim_engine.get_portfolio()
    assert 'TEST' in state['positions']
