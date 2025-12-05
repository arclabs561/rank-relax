# Next Steps: Prioritized Implementation Plan

Based on MCP research into production-ready differentiable ranking libraries and analysis of the current codebase, here are the critical next steps.

## Critical Blockers (Do First)

### 1. Fix Python Dependency Conflicts ⚠️ **BLOCKER**

**Status**: ✅ **RESOLVED**

**Why**: Blocks all Python usage, which is the primary integration target.

**Issue**: `pyo3-tch = "^0.3"` doesn't exist (only 0.22.0, 0.20.0, etc. available)

**Action Taken**:
- ✅ Python bindings build successfully with `uv`
- ✅ Fixed `__init__.py` to properly export all functions
- ✅ All core functions verified working:
  - `soft_rank`, `soft_sort`, `spearman_loss`
  - `soft_rank_gradient`, `spearman_loss_gradient`
  - `soft_rank_with_method`
- ✅ Note: `autograd_pytorch.rs` uses `pyo3-tch` but isn't compiled (not in `lib.rs`), so no conflict

**Impact**: Python users can now use the library

**Verification**:
```python
import rank_relax
ranks = rank_relax.soft_rank([5.0, 1.0, 2.0, 4.0, 3.0], 1.0)
# Works! ✅
```

---

### 2. Validate Framework Integrations Work End-to-End 🧪 **CRITICAL**

**Why**: MCP research shows that autograd integration is the #1 production blocker. Examples exist but may not actually work.

**Issues to verify**:
- PyTorch autograd functions (`pytorch_autograd.py`) - do gradients actually flow?
- JAX primitives (`jax_primitive.py`) - do JVP/transpose rules work?
- Numerical gradient checks - compare analytical vs finite differences

**Action**:
- Create integration tests that verify gradient flow
- Test with `torch.autograd.gradcheck` for PyTorch
- Test with `jax.grad` and `jax.jit` for JAX
- Add gradient correctness tests (analytical vs numerical)

**Impact**: Ensures the library actually works for training, not just examples

**Evidence from MCP**: "Autograd and non-differentiable edges" are the biggest blockers - any break in gradient flow defeats the purpose

---

### 3. Add Numerical Stability Tests and Warnings 🛡️ **CRITICAL**

**Why**: MCP research identifies numerical stability as the #1 production issue. Temperature/regularization tuning is fragile.

**Issues to address**:
- Gradient explosion/vanishing with wrong regularization_strength
- NaN handling in edge cases
- Mixed precision compatibility
- Score scale sensitivity

**Action**:
- Add tests for gradient stability across regularization_strength range
- Add warnings when regularization_strength is too low/high
- Document safe parameter ranges
- Add numerical stability checks in gradient computation

**Impact**: Prevents silent failures and training instability

**Evidence from MCP**: "Training becomes unstable after scaling up" due to badly tuned temperature - this is the most common failure mode

---

## High Priority (Do Next)

### 4. Implement True Differentiable `soft_sort` 🔧 **MAJOR FEATURE**

**Why**: Currently a placeholder. This is a core feature users expect.

**Current state**: Uses hard `std::sort` - no gradients

**Options** (from `src/sort.rs` TODO):
1. **Permutahedron projection** (O(n log n), exact gradients) - Best for performance
2. **Optimal transport** (O(n²) per Sinkhorn iteration) - More flexible
3. **Sorting networks** (O(n log² n)) - Good balance

**Recommendation**: Start with permutahedron projection (Blondel et al., 2020) - it's the most efficient and has exact gradients.

**Impact**: Completes core API, enables sorting-based losses

---

### 5. Add Performance Benchmarks to Validate Claims 📊 **VALIDATION**

**Why**: README claims "< 1ms for n=1000" but no benchmarks exist to validate.

**Action**:
- Create benchmark suite comparing all methods
- Benchmark forward/backward passes
- Compare against paper implementations (torchsort, diffsort)
- Document actual performance characteristics

**Impact**: Validates performance claims, identifies bottlenecks

**Evidence from MCP**: Production users need "good performance in full models, not just microbenchmarks"

---

### 6. Fix Missing Feature Flags 🔧 **BUILD ISSUE**

**Why**: `cargo check` warns about `parallel` feature not being declared.

**Action**:
- Add `parallel` feature to `Cargo.toml`
- Make parallel processing optional (requires `rayon`)
- Document feature flags

**Impact**: Fixes build warnings, enables optional parallel processing

---

## Medium Priority (Important but Not Blocking)

### 7. Add Batch Processing Optimizations ⚡ **PERFORMANCE**

**Why**: MCP research emphasizes scalability to realistic batch sizes.

**Current**: Basic batch processing exists but not optimized

**Action**:
- Optimize batch processing (avoid repeated allocations)
- Add parallel batch processing (when `parallel` feature enabled)
- Benchmark batch performance vs single-item processing

**Impact**: Better performance for real training workloads

---

### 8. Add Comprehensive Edge Case Handling 🛡️ **ROBUSTNESS**

**Why**: MCP research shows edge cases break production workflows.

**Issues to handle**:
- Variable-length batches (padding/masking)
- Ties in values
- Extreme values (very large/small)
- Empty inputs
- Single-element inputs

**Action**:
- Add tests for all edge cases
- Document behavior for each case
- Add warnings for problematic inputs

**Impact**: Prevents silent failures in production

---

### 9. Create Paper Reproduction Benchmarks 📚 **VALIDATION**

**Why**: README claims paper reproduction but none exists.

**Action**:
- Implement SoftRank reproduction (LETOR dataset, NDCG)
- Implement NeuralSort reproduction (sorting benchmarks)
- Compare results with paper-reported values
- Document any discrepancies

**Impact**: Validates correctness, builds credibility

---

## Lower Priority (Nice to Have)

### 10. Add More Ranking Methods 🔬 **FEATURE EXPANSION**

**Current**: 4 methods (Sigmoid, NeuralSort, Probabilistic, SmoothI)

**Could add**:
- SoftSort (optimal transport-based)
- Differentiable Top-K
- ListNet/ListMLE

**Impact**: More options for users, but current methods cover most use cases

---

### 11. Add SIMD Optimizations ⚡ **PERFORMANCE**

**Why**: Could improve performance for large inputs

**Action**:
- Use `std::arch` for SIMD sigmoid computations
- Benchmark SIMD vs scalar performance

**Impact**: Better performance, but current O(n²) complexity is the bigger issue

---

## Summary: Top 3 Immediate Actions

1. **Fix Python dependencies** - Unblocks Python users
2. **Validate framework integrations** - Ensures library actually works
3. **Add numerical stability tests** - Prevents production failures

These three address the critical blockers identified by MCP research:
- Dependency issues block usage
- Autograd integration is the #1 production concern
- Numerical stability is the #1 failure mode

---

## Success Metrics

After completing critical blockers:
- ✅ Python bindings install and work without errors
- ✅ PyTorch/JAX integrations pass gradient correctness tests
- ✅ Numerical stability tests pass across parameter ranges
- ✅ Performance benchmarks validate README claims
- ✅ `soft_sort` provides actual gradients

---

## References

- MCP Research: Production blockers for differentiable ranking
- IMPLEMENTATION_PLAN.md: Comprehensive roadmap
- BENCHMARKING.md: Paper reproduction plan
- Codebase analysis: Current state and gaps

