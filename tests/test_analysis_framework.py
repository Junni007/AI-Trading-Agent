"""
Test suite for the analysis framework.

Tests the base Analysis class and concrete implementations.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.base import Analysis
from src.analysis.expert_performance import ExpertPerformanceAnalysis
from src.analysis.edge_validation import EdgeValidationAnalysis


class MockAnalysis(Analysis):
    """Simple mock analysis for testing."""
    name = "test_analysis"
    description = "Test analysis for unit tests"
    
    def run(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        data = {"test": "passed", "value": 123}
        return fig, data


class TestBaseAnalysis:
    """Tests for Analysis base class."""
    
    def test_analysis_creation(self):
        """Test that analysis can be instantiated."""
        analysis = MockAnalysis()
        assert analysis.name == "test_analysis"
        assert analysis.description == "Test analysis for unit tests"
    
    def test_run_returns_tuple(self):
        """Test that run() returns (Figure, dict)."""
        analysis = MockAnalysis()
        fig, data = analysis.run()
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(data, dict)
        assert "test" in data
        plt.close(fig)
    
    def test_save_creates_files(self, tmp_path):
        """Test that save() creates PNG and JSON files."""
        analysis = MockAnalysis()
        output_dir = str(tmp_path / "test_output")
        
        analysis.save(output_dir=output_dir)
        
        # Check files exist
        png_file = tmp_path / "test_output" / "test_analysis.png"
        json_file = tmp_path / "test_output" / "test_analysis.json"
        
        assert png_file.exists()
        assert json_file.exists()
        
        # Check JSON content
        with open(json_file) as f:
            data = json.load(f)
        assert data["test"] == "passed"
        assert data["value"] == 123


class TestExpertPerformanceAnalysis:
    """Tests for expert performance analysis."""
    
    def test_expert_analysis_runs(self):
        """Verify analysis executes without errors."""
        analysis = ExpertPerformanceAnalysis()
        fig, data = analysis.run()
        
        assert fig is not None
        assert 'experts' in data
        assert len(data['experts']) == 4
        assert 'summary' in data
        plt.close(fig)
    
    def test_expert_analysis_output_structure(self):
        """Verify output has expected structure."""
        analysis = ExpertPerformanceAnalysis()
        fig, data = analysis.run()
        
        # Check all expected fields
        assert 'experts' in data
        assert 'avg_confidence' in data
        assert 'signal_count' in data
        assert 'colors' in data
        assert 'summary' in data
        
        # Check summary fields
        summary = data['summary']
        assert 'most_confident' in summary
        assert 'most_active' in summary
        assert 'total_signals' in summary
        
        plt.close(fig)


class TestEdgeValidationAnalysis:
    """Tests for edge validation analysis."""
    
    def test_edge_validation_runs(self):
        """Verify edge validation executes."""
        analysis = EdgeValidationAnalysis()
        fig, data = analysis.run()
        
        assert fig is not None
        assert 'statistical_significance' in data
        plt.close(fig)
    
    def test_statistical_significance_structure(self):
        """Verify statistical test results."""
        analysis = EdgeValidationAnalysis()
        fig, data = analysis.run()
        
        sig = data['statistical_significance']
        assert 'chi_square' in sig
        assert 'p_value' in sig
        assert 'significant' in sig
        # Accept both Python bool and numpy bool_
        assert isinstance(sig['significant'], (bool, np.bool_))
        
        plt.close(fig)
    
    def test_edge_summary_exists(self):
        """Verify edge summary is generated."""
        analysis = EdgeValidationAnalysis()
        fig, data = analysis.run()
        
        assert 'edge_summary' in data
        assert 'total_trades_analyzed' in data['edge_summary']
        
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
