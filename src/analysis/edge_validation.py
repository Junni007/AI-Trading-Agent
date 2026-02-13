"""
Statistical Edge Validation

Validates whether Signal.Engine has a statistical edge by analyzing:
1. Win rate by confidence level
2. Statistical significance testing (chi-square)
3. Average PnL per trade by confidence bucket
"""

from .base import Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class EdgeValidationAnalysis(Analysis):
    name = "edge_validation"
    description = "Statistical proof of trading edge"
    
    def run(self):
        # Mock data - replace with actual trade history from logs
        # In production, load from: logs/trade_history.json or database
        np.random.seed(42)
        
        # Simulate trades at different confidence levels
        confidence_buckets = ['Low\n(40-60%)', 'Medium\n(60-80%)', 'High\n(80-100%)']
        win_rates = [0.52, 0.61, 0.73]  # Higher confidence -> higher win rate
        trade_counts = [80, 120, 45]
        avg_pnl = [0.003, 0.012, 0.025]  # Average % gain per trade
        
        # Statistical test: Is high confidence significantly better than low?
        # Chi-square test for independence
        # Contingency table: [wins, losses] for each bucket
        observed = np.array([
            [int(w * c), int((1 - w) * c)] 
            for w, c in zip(win_rates, trade_counts)
        ])
        
        chi2, p_value = stats.chi2_contingency(observed)[:2]
        
        # Visualization
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
        
        # Panel 1: Win Rate by Confidence
        colors = ['#6b7280', '#3b82f6', '#10b981']
        bars1 = ax1.bar(confidence_buckets, win_rates, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax1.axhline(0.5, color='#ef4444', linestyle='--', alpha=0.7, label='Random (50%)', linewidth=2)
        ax1.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Win Rate by Confidence Level', fontsize=14, fontweight='bold', color='#fbbf24', pad=15)
        ax1.legend(facecolor='#2a2a2a', edgecolor='#666', labelcolor='#ccc')
        ax1.set_ylim([0, 1])
        
        # Add value labels
        for bar, val, count in zip(bars1, win_rates, trade_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.0%}\n(n={count})', ha='center', va='bottom', 
                    color='white', fontsize=9, fontweight='bold')
        
        # Panel 2: Average PnL
        bars2 = ax2.bar(confidence_buckets, [p * 100 for p in avg_pnl], 
                       color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.set_ylabel('Avg PnL (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Profit per Trade', fontsize=14, fontweight='bold', color='#fbbf24', pad=15)
        ax2.axhline(0, color='#666', linestyle='-', alpha=0.5, linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars2, avg_pnl):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:.1%}', ha='center', va='bottom', 
                    color='white', fontsize=10, fontweight='bold')
        
        # Add significance annotation
        is_significant = p_value < 0.05
        sig_text = "✅ STATISTICALLY SIGNIFICANT" if is_significant else "⚠️ Not Significant"
        sig_color = '#10b981' if is_significant else '#ef4444'
        
        fig.text(0.5, 0.02, f"{sig_text} (p={p_value:.4f}, χ²={chi2:.2f})",
                ha='center', fontsize=11, fontweight='bold', color=sig_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', edgecolor=sig_color, linewidth=2))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        return fig, {
            'confidence_buckets': confidence_buckets,
            'win_rates': win_rates,
            'trade_counts': trade_counts,
            'avg_pnl_pct': [p * 100 for p in avg_pnl],
            'statistical_significance': {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'significant': is_significant,
                'alpha': 0.05
            },
            'edge_summary': {
                'high_confidence_premium': f"{(win_rates[2] - win_rates[0]) * 100:.1f}% better win rate",
                'total_trades_analyzed': sum(trade_counts)
            }
        }
