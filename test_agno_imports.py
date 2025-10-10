#!/usr/bin/env python3
"""
Test script to check agno package imports and structure.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_agno_imports():
    """Test different agno import patterns."""
    print("Testing agno package imports...")
    print("=" * 50)
    
    try:
        import agno
        print(f"✓ agno package imported successfully")
        print(f"  agno version: {getattr(agno, '__version__', 'unknown')}")
        print(f"  agno location: {agno.__file__}")
        print(f"  agno attributes: {dir(agno)}")
    except ImportError as e:
        print(f"✗ Failed to import agno: {e}")
        return False
    
    # Test different import patterns for agno 2.1.1
    import_patterns = [
        ("from agno.agent import Agent", "from agno.agent import Agent"),
        ("from agno.agent import Message", "from agno.agent import Message"),
        ("from agno.agent import Function", "from agno.agent import Function"),
        ("from agno.agent import Toolkit", "from agno.agent import Toolkit"),
        ("from agno.tools import Function", "from agno.tools import Function"),
        ("from agno.tools import FunctionCall", "from agno.tools import FunctionCall"),
        ("from agno.tools import Toolkit", "from agno.tools import Toolkit"),
        ("from agno.models.message import Message", "from agno.models.message import Message"),
        ("from agno.run.agent import RunEvent", "from agno.run.agent import RunEvent"),
        ("from agno.run.agent import RunOutput", "from agno.run.agent import RunOutput"),
    ]
    
    for pattern_name, pattern in import_patterns:
        try:
            exec(pattern)
            print(f"✓ {pattern_name}")
        except ImportError as e:
            print(f"✗ {pattern_name}: {e}")
        except Exception as e:
            print(f"✗ {pattern_name}: {e}")
    
    return True

def explore_agno_structure():
    """Explore the agno package structure to find correct imports."""
    print("\nExploring agno package structure...")
    print("=" * 50)
    
    try:
        import agno
        import pkgutil
        
        # List all submodules
        print("Available submodules:")
        for importer, modname, ispkg in pkgutil.walk_packages(agno.__path__, agno.__name__ + "."):
            print(f"  {modname} {'(package)' if ispkg else '(module)'}")
        
        # Try to import each submodule and see what's available
        submodules = []
        for importer, modname, ispkg in pkgutil.walk_packages(agno.__path__, agno.__name__ + "."):
            if not ispkg:  # Only modules, not packages
                submodules.append(modname)
        
        print(f"\nTesting {len(submodules)} modules:")
        for module_name in submodules:
            try:
                module = __import__(module_name, fromlist=[''])
                attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                print(f"  {module_name}: {attrs}")
            except Exception as e:
                print(f"  {module_name}: Error - {e}")
                
    except Exception as e:
        print(f"Error exploring agno structure: {e}")

if __name__ == "__main__":
    test_agno_imports()
    explore_agno_structure()
