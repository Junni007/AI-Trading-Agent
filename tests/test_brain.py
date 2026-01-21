"""
Unit tests for brain experts.
Tests the HybridBrain's thinking process and decision output structure.
"""
import pytest
from src.brain.hybrid import HybridBrain


@pytest.fixture
def brain():
    """Create a HybridBrain instance for testing."""
    return HybridBrain()


def test_brain_think_returns_list(brain):
    """Test that brain.think() returns a list."""
    result = brain.think()
    assert isinstance(result, list), "Brain.think() should return a list"


def test_brain_decision_structure(brain):
    """Test that each decision has required fields."""
    decisions = brain.think()
    
    # Skip if no decisions (market closed, etc.)
    if not decisions:
        pytest.skip("No decisions returned (market may be closed)")
    
    required_fields = ['Ticker', 'Action', 'Confidence', 'Rational']
    
    for decision in decisions:
        for field in required_fields:
            assert field in decision, f"Decision missing required field: {field}"


def test_confidence_range(brain):
    """Test that confidence values are in valid range [0, 1]."""
    decisions = brain.think()
    
    if not decisions:
        pytest.skip("No decisions returned (market may be closed)")
    
    for decision in decisions:
        conf = decision.get('Confidence', 0)
        assert 0 <= conf <= 1, f"Confidence {conf} out of range [0, 1]"


def test_rational_is_list(brain):
    """Test that Rational field is a list of strings."""
    decisions = brain.think()
    
    if not decisions:
        pytest.skip("No decisions returned (market may be closed)")
    
    for decision in decisions:
        rational = decision.get('Rational', [])
        assert isinstance(rational, list), "Rational should be a list"
        
        for item in rational:
            assert isinstance(item, str), "Each rational item should be a string"


def test_ticker_format(brain):
    """Test that ticker symbols have expected format."""
    decisions = brain.think()
    
    if not decisions:
        pytest.skip("No decisions returned (market may be closed)")
    
    for decision in decisions:
        ticker = decision.get('Ticker', '')
        assert len(ticker) > 0, "Ticker should not be empty"
        # NSE tickers typically end with .NS
        assert '.NS' in ticker or len(ticker) <= 10, f"Unexpected ticker format: {ticker}"


def test_action_is_string(brain):
    """Test that Action field is a non-empty string."""
    decisions = brain.think()
    
    if not decisions:
        pytest.skip("No decisions returned (market may be closed)")
    
    for decision in decisions:
        action = decision.get('Action', '')
        assert isinstance(action, str), "Action should be a string"
        assert len(action) > 0, "Action should not be empty"
