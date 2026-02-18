"""
Test configuration — patches heavy-weight objects to keep tests fast.
"""
from unittest.mock import patch, MagicMock

# ─── Mock brain.think() globally ────────────────────────────────────────────
# brain.think() invokes the full ML model pipeline (~9 min).
# We patch it at the module level so ALL tests get a fast, deterministic mock.

MOCK_DECISIONS = [
    {
        "Ticker": "TEST",
        "Action": "Hold",
        "Confidence": 0.5,
        "Rational": ["Mock test signal"],
    }
]

# Patch before any test module imports src.api.main
_brain_patch = patch("src.brain.hybrid.HybridBrain.think", return_value=MOCK_DECISIONS)
_brain_patch.start()

# Don't stop — let it live for the entire test session
