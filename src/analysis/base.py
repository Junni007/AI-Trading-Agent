"""
Base class for all analysis scripts.

Provides a consistent interface for creating modular analyses
with automatic multi-format output (PNG, JSON).
"""

from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import json

class Analysis(ABC):
    """
    Base class for all analysis scripts.
    
    Subclasses must implement:
    - name: str (unique identifier)
    - description: str (what this analysis does)
    - run(): tuple[Figure, Dict] (execute analysis)
    
    Example:
        class MyAnalysis(Analysis):
            name = "my_analysis"
            description = "Analyzes XYZ"
            
            def run(self):
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3])
                data = {"result": 123}
                return fig, data
    """
    
    name: str = "base_analysis"
    description: str = "Base analysis class"
    
    def __init__(self):
        self.logger = logging.getLogger(f"Analysis.{self.name}")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    @abstractmethod
    def run(self) -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Execute the analysis.
        
        Returns:
            tuple: (matplotlib.Figure, dict)
                - Figure: Visualization to save as PNG
                - dict: Results data to save as JSON
        """
        pass
    
    def save(self, output_dir: str = "output") -> None:
        """
        Auto-save analysis outputs to multiple formats.
        
        Args:
            output_dir: Directory to save outputs (default: "output/")
            
        Creates:
            - {name}.png: High-resolution chart (300 DPI)
            - {name}.json: Raw data for further analysis
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"▶ Running {self.name}...")
        
        try:
            fig, data = self.run()
            
            # Save PNG (high DPI for presentations)
            png_path = output_path / f"{self.name}.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
            self.logger.info(f"  ✅ Saved chart: {png_path}")
            
            # Save data as JSON
            json_path = output_path / f"{self.name}.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"  ✅ Saved data: {json_path}")
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}")
            raise
            
    def progress(self, message: str):
        """
        Context manager for showing progress during long operations.
        
        Usage:
            with self.progress("Loading data"):
                df = load_large_dataset()
        """
        class ProgressContext:
            def __init__(self, logger, msg):
                self.logger = logger
                self.msg = msg
                
            def __enter__(self):
                self.logger.info(f"  ⏳ {self.msg}...")
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    self.logger.info(f"  ✓ {self.msg} complete")
                return False
                
        return ProgressContext(self.logger, message)
