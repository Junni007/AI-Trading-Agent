import logging
from src.brain.hybrid import HybridBrain

logging.basicConfig(level=logging.INFO)

def test_brain():
    print("Initializing Hybrid Brain...")
    brain = HybridBrain()
    
    print("Running think cycle (this may take a few seconds due to yfinance)...")
    decisions = brain.think()
    
    print("\n========= RESULTS =========")
    print(f"Total Decisions Generated: {len(decisions)}")
    
    if len(decisions) == 0:
        print("ERROR: No decisions generated. Pipeline is returning empty.")
    else:
        for i, dec in enumerate(decisions[:5]):
            print(f"{i+1}. {dec['Ticker']} | Action: {dec['Action']} | Conf: {dec['Confidence']}")
            
if __name__ == "__main__":
    test_brain()
