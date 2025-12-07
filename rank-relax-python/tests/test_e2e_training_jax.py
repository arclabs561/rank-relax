"""
E2E test: Install from PyPI and run actual JAX training.

This test verifies that the published package can be used in real JAX training loops.
"""

import pytest


def test_e2e_jax_training_installed():
    """Test that we can train a model using the installed rank-relax package with JAX."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import core
        from jax.interpreters import ad
        import rank_relax
    except ImportError as e:
        pytest.skip(f"rank-relax or JAX not installed: {e}")
    
    # Inline JAX primitive (examples may not be packaged)
    spearman_loss_p = core.Primitive("spearman_loss_rust")
    
    def spearman_loss_jax(predictions, targets, regularization_strength=1.0):
        return spearman_loss_p.bind(predictions, targets, regularization_strength=regularization_strength)
    
    def spearman_loss_impl(predictions, targets, regularization_strength):
        pred_list = predictions.tolist()
        targ_list = targets.tolist()
        loss = rank_relax.spearman_loss(pred_list, targ_list, regularization_strength)
        return jnp.array(loss)
    
    spearman_loss_p.def_impl(spearman_loss_impl)
    
    def spearman_loss_abstract_eval(pred_aval, target_aval, reg_aval):
        return core.ShapedArray((), pred_aval.dtype)
    
    spearman_loss_p.def_abstract_eval(spearman_loss_abstract_eval)
    
    def spearman_loss_jvp(primals, tangents, *, regularization_strength):
        predictions, targets = primals
        pred_dot, target_dot = tangents
        y = spearman_loss_jax(predictions, targets, regularization_strength)
        pred_list = predictions.tolist()
        targ_list = targets.tolist()
        pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
        target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
        grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
        y_dot = jnp.sum(jnp.array(grad) * pred_dot)
        return y, y_dot
    
    ad.primitive_jvps[spearman_loss_p] = spearman_loss_jvp
    
    def spearman_loss_transpose(ct, predictions, targets, *, regularization_strength):
        if predictions is None:
            return None, None, None
        pred_list = predictions.tolist()
        targ_list = targets.tolist()
        pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
        target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
        grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
        predictions_bar = jnp.array(grad) * ct
        return predictions_bar, None, None
    
    ad.primitive_transposes[spearman_loss_p] = spearman_loss_transpose
    
    # Simple model parameters
    key = jax.random.PRNGKey(42)
    params = {
        'w': jax.random.normal(key, (128, 10)),
        'b': jax.random.normal(key, (10,))
    }
    
    def model(params, x):
        """Simple linear model."""
        return jnp.dot(x, params['w']) + params['b']
    
    def loss_fn(params, x, targets):
        """Loss function using rank-relax."""
        predictions = model(params, x)
        return spearman_loss_jax(predictions, targets, regularization_strength=1.0)
    
    # Synthetic data
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (4, 128))
    targets = jax.random.normal(key, (10,))
    
    # Compute loss
    loss = loss_fn(params, inputs, targets)
    assert loss > 0, "Loss should be positive"
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=0)
    grads = grad_fn(params, inputs, targets)
    
    assert grads is not None, "Gradients must be computed"
    assert 'w' in grads, "Gradients for weights must exist"
    assert 'b' in grads, "Gradients for bias must exist"
    
    # Training step
    learning_rate = 0.01
    new_params = {
        'w': params['w'] - learning_rate * grads['w'],
        'b': params['b'] - learning_rate * grads['b']
    }
    
    # Verify loss decreases (or at least changes)
    new_loss = loss_fn(new_params, inputs, targets)
    assert new_loss != loss, "Loss should change after update"
    
    # Test JIT compilation
    loss_jit = jax.jit(loss_fn)
    jit_loss = loss_jit(params, inputs, targets)
    assert jit_loss > 0, "JIT-compiled loss should work"
    
    print("✅ E2E JAX training test passed - package works in real training loop")


def test_e2e_jax_gradient_flow():
    """Test that gradients actually flow through rank-relax operations in JAX."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import core
        from jax.interpreters import ad
        import rank_relax
    except ImportError as e:
        pytest.skip(f"rank-relax or JAX not installed: {e}")
    
    # Inline JAX primitive (same as above)
    spearman_loss_p = core.Primitive("spearman_loss_rust")
    
    def spearman_loss_jax(predictions, targets, regularization_strength=1.0):
        return spearman_loss_p.bind(predictions, targets, regularization_strength=regularization_strength)
    
    def spearman_loss_impl(predictions, targets, regularization_strength):
        pred_list = predictions.tolist()
        targ_list = targets.tolist()
        loss = rank_relax.spearman_loss(pred_list, targ_list, regularization_strength)
        return jnp.array(loss)
    
    spearman_loss_p.def_impl(spearman_loss_impl)
    spearman_loss_p.def_abstract_eval(lambda p, t, r: core.ShapedArray((), p.dtype))
    
    def spearman_loss_jvp(primals, tangents, *, regularization_strength):
        predictions, targets = primals
        pred_dot, target_dot = tangents
        y = spearman_loss_jax(predictions, targets, regularization_strength)
        pred_list = predictions.tolist()
        targ_list = targets.tolist()
        pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
        target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
        grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
        y_dot = jnp.sum(jnp.array(grad) * pred_dot)
        return y, y_dot
    
    ad.primitive_jvps[spearman_loss_p] = spearman_loss_jvp
    
    def spearman_loss_transpose(ct, predictions, targets, *, regularization_strength):
        if predictions is None:
            return None, None, None
        pred_list = predictions.tolist()
        targ_list = targets.tolist()
        pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
        target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
        grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
        return jnp.array(grad) * ct, None, None
    
    ad.primitive_transposes[spearman_loss_p] = spearman_loss_transpose
    
    predictions = jnp.array([0.1, 0.9, 0.3, 0.7, 0.5])
    targets = jnp.array([0.0, 1.0, 0.2, 0.8, 0.4])
    
    grad_fn = jax.grad(lambda p: spearman_loss_jax(p, targets, 1.0))
    grads = grad_fn(predictions)
    
    assert grads is not None, "Gradients must be computed"
    assert jnp.all(jnp.isfinite(grads)), "Gradients must be finite"
    assert jnp.linalg.norm(grads) > 0, "Gradients must be non-zero"
    
    print("✅ JAX gradient flow test passed")

