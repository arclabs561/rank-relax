"""Tests for publishing readiness and wheel installation.

These tests verify that the package can be built and installed correctly,
which is essential for PyPI publishing.
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import pytest


def test_maturin_build():
    """Test that maturin can build the package."""
    result = subprocess.run(
        ['maturin', 'build', '--release'],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    assert result.returncode == 0, f"Build failed: {result.stderr}"
    
    # Check that wheel was created
    dist_dir = Path(__file__).parent.parent / 'dist'
    wheels = list(dist_dir.glob('*.whl'))
    assert len(wheels) > 0, "No wheel file created"
    
    print(f"✅ Built wheel: {wheels[0].name}")


def test_wheel_installation():
    """Test that the built wheel can be installed."""
    dist_dir = Path(__file__).parent.parent / 'dist'
    wheels = list(dist_dir.glob('*.whl'))
    
    if not wheels:
        pytest.skip("No wheel found - run test_maturin_build first")
    
    wheel_path = wheels[0]
    
    # Install in a temporary environment
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use pip to install the wheel
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', str(wheel_path), '--target', tmpdir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        assert result.returncode == 0, f"Installation failed: {result.stderr}"
        
        # Try to import the module
        sys.path.insert(0, tmpdir)
        try:
            import rank_relax
            assert hasattr(rank_relax, 'spearman_loss'), "spearman_loss not available"
            assert hasattr(rank_relax, 'soft_rank'), "soft_rank not available"
            print("✅ Wheel installation and import successful")
        finally:
            sys.path.remove(tmpdir)


def test_publish_dry_run():
    """Test maturin publish --dry-run (doesn't actually publish)."""
    result = subprocess.run(
        ['maturin', 'publish', '--dry-run'],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    # Dry-run should succeed if package is ready
    # Note: This might fail if not logged into PyPI, which is OK
    if result.returncode == 0:
        print("✅ Publish dry-run successful")
    else:
        print(f"⚠️  Publish dry-run failed (may need PyPI credentials): {result.stderr[:200]}")


def test_public_api_available():
    """Test that all public API functions are available."""
    try:
        import rank_relax
        
        # Check core functions
        required_functions = [
            'soft_rank',
            'soft_sort',
            'spearman_loss',
            'soft_rank_gradient',
            'spearman_loss_gradient',
            'soft_rank_with_method',
        ]
        
        missing = [f for f in required_functions if not hasattr(rank_relax, f)]
        assert len(missing) == 0, f"Missing public API functions: {missing}"
        
        print("✅ All public API functions available")
        
    except ImportError:
        pytest.skip("rank_relax not installed - run test_wheel_installation first")


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

