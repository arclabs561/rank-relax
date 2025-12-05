#!/usr/bin/env python3
"""
Test runner for rank-relax Python bindings.

This script runs all tests manually since pytest has discovery issues.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all test modules."""
    print("=" * 70)
    print("Running rank-relax Python bindings tests")
    print("=" * 70)
    
    results = {"passed": 0, "failed": 0, "errors": []}
    
    # Test basic functionality
    print("\n[1/4] Testing basic functionality...")
    try:
        from tests import test_relax
        test_relax.test_soft_rank()
        test_relax.test_soft_sort()
        test_relax.test_spearman_loss()
        test_relax.test_spearman_loss_perfect_correlation()
        test_relax.test_empty_input()
        print("  ✓ Basic functionality tests passed")
        results["passed"] += 5
    except Exception as e:
        print(f"  ✗ Basic functionality tests failed: {e}")
        results["failed"] += 1
        results["errors"].append(("basic", str(e)))
    
    # Test numerical stability
    print("\n[2/4] Testing numerical stability...")
    try:
        from tests import test_numerical_stability
        test_numerical_stability.test_stability_across_regularization_range()
        test_numerical_stability.test_stability_with_extreme_values()
        test_numerical_stability.test_gradient_stability()
        test_numerical_stability.test_stability_with_ties()
        test_numerical_stability.test_stability_with_large_inputs()
        test_numerical_stability.test_spearman_loss_stability()
        test_numerical_stability.test_warning_for_extreme_regularization()
        print("  ✓ Numerical stability tests passed")
        results["passed"] += 7
    except Exception as e:
        print(f"  ✗ Numerical stability tests failed: {e}")
        results["failed"] += 1
        results["errors"].append(("numerical_stability", str(e)))
    
    # Test gradient correctness
    print("\n[3/4] Testing gradient correctness...")
    try:
        from tests import test_gradient_correctness
        test_gradient_correctness.test_soft_rank_gradient_correctness()
        test_gradient_correctness.test_spearman_loss_gradient_correctness()
        test_gradient_correctness.test_gradient_stability_across_regularization()
        test_gradient_correctness.test_gradient_edge_cases()
        print("  ✓ Gradient correctness tests passed")
        results["passed"] += 4
    except Exception as e:
        print(f"  ✗ Gradient correctness tests failed: {e}")
        results["failed"] += 1
        results["errors"].append(("gradient_correctness", str(e)))
    
    # Test integration
    print("\n[4/4] Testing integration...")
    try:
        from tests import test_integration
        test_integration.test_soft_rank_basic()
        test_integration.test_soft_rank_methods()
        test_integration.test_spearman_loss()
        test_integration.test_gradient_computation()
        test_integration.test_spearman_loss_gradient()
        test_integration.test_edge_cases()
        test_integration.test_regularization_effects()
        print("  ✓ Integration tests passed")
        results["passed"] += 7
    except Exception as e:
        print(f"  ✗ Integration tests failed: {e}")
        results["failed"] += 1
        results["errors"].append(("integration", str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Test Results: {results['passed']} passed, {results['failed']} failed")
    print("=" * 70)
    
    if results["errors"]:
        print("\nErrors:")
        for test_type, error in results["errors"]:
            print(f"  {test_type}: {error}")
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

