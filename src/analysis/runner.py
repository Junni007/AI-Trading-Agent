"""
Analysis Runner - Interactive CLI

Usage: python -m src.analysis.runner

Auto-discovers and runs all Analysis subclasses in this module.
"""

import importlib
import pkgutil
from pathlib import Path
from . import base
import sys

def discover_analyses():
    """Auto-discover all Analysis subclasses in this package."""
    analyses = []
    analysis_dir = Path(__file__).parent
    
    for _, name, _ in pkgutil.iter_modules([str(analysis_dir)]):
        # Skip base and runner modules
        if name in ['base', 'runner', '__init__']:
            continue
            
        try:
            module = importlib.import_module(f'.{name}', package='src.analysis')
            
            # Find all Analysis subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, base.Analysis) and 
                    attr is not base.Analysis):
                    analyses.append(attr())
        except ImportError as e:
            print(f"⚠️  Could not load {name}: {e}")
    
    return analyses

def print_banner():
    """Print fancy banner."""
    print("\n" + "=" * 60)
    print("📊  SIGNAL.ENGINE ANALYSIS FRAMEWORK  📊")
    print("=" * 60 + "\n")

def main():
    print_banner()
    
    analyses = discover_analyses()
    
    if not analyses:
        print("❌ No analyses found. Create analysis scripts in src/analysis/")
        sys.exit(1)
    
    print(f"Found {len(analyses)} analysis module(s):\n")
    for i, analysis in enumerate(analyses, 1):
        print(f"  {i}. {analysis.name}")
        print(f"     └─ {analysis.description}\n")
    
    print(f"  0. Run all analyses\n")
    
    try:
        choice = input("Select analysis to run [0-{}]: ".format(len(analyses)))
        
        if choice == '0':
            print("\n▶ Running all analyses...")
            for analysis in analyses:
                analysis.save()
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(analyses):
                analyses[idx].save()
            else:
                print("❌ Invalid selection")
                sys.exit(1)
                
        print("\n" + "=" * 60)
        print("✅ Analysis complete! Check the 'output/' directory")
        print("=" * 60 + "\n")
        
    except (ValueError, IndexError, KeyboardInterrupt):
        print("\n❌ Cancelled or invalid input")
        sys.exit(1)

if __name__ == '__main__':
    main()
