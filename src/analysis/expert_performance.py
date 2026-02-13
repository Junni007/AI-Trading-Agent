"""
Expert Performance Analysis

Compares accuracy and confidence of all 4 experts (Sniper, Volatility, RL, Quant)
to identify which expert is most reliable and active.
"""

from .base import Analysis
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

class ExpertPerformanceAnalysis(Analysis):
    name = "expert_performance"
    description = "Compare accuracy and confidence of all experts"
    
    def run(self):
        # Load historical scan results if available
        # For MVP, we use mock data - replace with actual history
        experts = ['Sniper', 'Volatility', 'RL', 'Quant']
        avg_confidence = [0.75, 0.82, 0.65, 0.88]
        signal_count = [120, 45, 180, 30]
        colors = ['#fbbf24', '#3b82f6', '#10b981', '#8b5cf6']
        
        # Create dual-panel visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#1a1a1a')
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1a1a1a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666')
            ax.spines['bottom'].set_color('#666')
            ax.tick_params(colors='#ccc')
            ax.xaxis.label.set_color('#ccc')
            ax.yaxis.label.set_color('#ccc')
        
        # Panel 1: Confidence comparison
        bars1 = ax1.bar(experts, avg_confidence, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax1.set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
        ax1.set_title('Expert Confidence Levels', fontsize=14, fontweight='bold', color='#fbbf24', pad=15)
        ax1.set_ylim([0, 1])
        ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (50%)')
        ax1.legend(facecolor='#2a2a2a', edgecolor='#666', labelcolor='#ccc')
        
        # Add value labels on bars
        for bar, val in zip(bars1, avg_confidence):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.0%}', ha='center', va='bottom', color='white', fontsize=10)
        
        # Panel 2: Signal volume
        bars2 = ax2.bar(experts, signal_count, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.set_ylabel('Signals Generated', fontsize=12, fontweight='bold')
        ax2.set_title('Expert Activity', fontsize=14, fontweight='bold', color='#fbbf24', pad=15)
        
        # Add value labels
        for bar, val in zip(bars2, signal_count):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val}', ha='center', va='bottom', color='white', fontsize=10)
        
        plt.tight_layout()
        
        return fig, {
            'experts': experts,
            'avg_confidence': avg_confidence,
            'signal_count': signal_count,
            'colors': colors,
            'summary': {
                'most_confident': experts[np.argmax(avg_confidence)],
                'most_active': experts[np.argmax(signal_count)],
                'total_signals': sum(signal_count)
            }
        }
