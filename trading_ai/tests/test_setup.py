"""
Test Phase 1 Setup

Verifies that all Phase 1 files and directories are correctly created.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_directory_structure():
    """Test that all required directories exist"""
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/cache",
        "models/xgboost",
        "models/lstm",
        "modules",
        "config",
        "outputs/signals",
        "tests",
        "logs"
    ]
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        assert full_path.exists(), f"Directory {dir_path} does not exist"
        assert full_path.is_dir(), f"{dir_path} is not a directory"
    
    print("✓ All directories exist")


def test_required_files():
    """Test that all required files exist"""
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "config/config.yaml",
        "README.md",
        "main.py",
        "dashboard.py",
        "modules/__init__.py",
        "tests/__init__.py"
    ]
    
    for file_path in required_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"File {file_path} does not exist"
        assert full_path.is_file(), f"{file_path} is not a file"
    
    print("✓ All required files exist")


def test_gitkeep_files():
    """Test that .gitkeep files exist to preserve directory structure"""
    base_dir = Path(__file__).parent.parent
    
    gitkeep_files = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "data/cache/.gitkeep",
        "models/xgboost/.gitkeep",
        "models/lstm/.gitkeep",
        "outputs/signals/.gitkeep",
        "logs/.gitkeep"
    ]
    
    for file_path in gitkeep_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Gitkeep file {file_path} does not exist"
    
    print("✓ All .gitkeep files exist")


def test_config_yaml_valid():
    """Test that config.yaml is valid YAML"""
    try:
        import yaml
    except ImportError:
        print("✓ config.yaml exists (pyyaml not installed yet - install with: pip install -r requirements.txt)")
        return
    
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    required_sections = [
        "watchlist",
        "prediction_horizon_days",
        "confidence_threshold",
        "options",
        "risk_management",
        "llm",
        "api_rate_limits",
        "logging"
    ]
    
    for section in required_sections:
        assert section in config, f"Config missing required section: {section}"
    
    # Check watchlist has tickers
    assert len(config["watchlist"]) > 0, "Watchlist is empty"
    
    print(f"✓ config.yaml is valid with {len(config['watchlist'])} tickers")


def test_requirements_txt_valid():
    """Test that requirements.txt contains key packages"""
    base_dir = Path(__file__).parent.parent
    req_path = base_dir / "requirements.txt"
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "anthropic",
        "yfinance",
        "streamlit"
    ]
    
    for package in required_packages:
        assert package in content, f"requirements.txt missing package: {package}"
    
    print(f"✓ requirements.txt contains all required packages")


def test_env_example_has_keys():
    """Test that .env.example has all required API key placeholders"""
    base_dir = Path(__file__).parent.parent
    env_path = base_dir / ".env.example"
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    required_keys = [
        "ANTHROPIC_API_KEY",
        "ALPHA_VANTAGE_KEY",
        "FRED_API_KEY",
        "NEWS_API_KEY",
        "FINNHUB_API_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY"
    ]
    
    for key in required_keys:
        assert key in content, f".env.example missing key: {key}"
    
    print("✓ .env.example has all required API keys")


def test_main_executable():
    """Test that main.py is executable"""
    base_dir = Path(__file__).parent.parent
    main_path = base_dir / "main.py"
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    assert "def main()" in content, "main.py missing main() function"
    assert "argparse" in content, "main.py missing argparse for CLI"
    
    print("✓ main.py is properly structured")


def test_gitignore_comprehensive():
    """Test that .gitignore has critical exclusions"""
    base_dir = Path(__file__).parent.parent
    gitignore_path = base_dir / ".gitignore"
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    critical_exclusions = [
        ".env",
        "data/raw/",
        "data/processed/",
        "models/",
        "logs/",
        "__pycache__"
    ]
    
    for exclusion in critical_exclusions:
        assert exclusion in content, f".gitignore missing: {exclusion}"
    
    print("✓ .gitignore has critical exclusions")


def run_all_tests():
    """Run all Phase 1 tests"""
    print("=" * 70)
    print("PHASE 1 SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Required Files", test_required_files),
        ("Gitkeep Files", test_gitkeep_files),
        ("Config YAML", test_config_yaml_valid),
        ("Requirements.txt", test_requirements_txt_valid),
        (".env.example", test_env_example_has_keys),
        ("Main Entry Point", test_main_executable),
        (".gitignore", test_gitignore_comprehensive)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: Unexpected error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print()
        print("🎉 PHASE 1 SETUP COMPLETE!")
        print()
        print("Next steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Confirm with user before proceeding to Phase 2")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
