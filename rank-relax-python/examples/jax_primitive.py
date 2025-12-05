"""
JAX custom primitive with analytical JVP and transpose rules.

This implements proper gradient flow through rank-relax operations in JAX.
"""

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import ad
import rank_relax


# Define primitives (JAX v0.4+ API)
soft_rank_p = core.Primitive("soft_rank_rust")
spearman_loss_p = core.Primitive("spearman_loss_rust")


def soft_rank_jax(values, regularization_strength=1.0, method="sigmoid"):
    """User-facing function for soft_rank in JAX."""
    return soft_rank_p.bind(values, regularization_strength=regularization_strength, method=method)


def spearman_loss_jax(predictions, targets, regularization_strength=1.0, method="sigmoid"):
    """User-facing function for spearman_loss in JAX."""
    return spearman_loss_p.bind(
        predictions, targets,
        regularization_strength=regularization_strength,
        method=method
    )


# Concrete implementations
def soft_rank_impl(values, regularization_strength, method):
    """Concrete implementation calling Rust."""
    val_list = values.tolist()
    ranks = rank_relax.soft_rank_with_method(val_list, regularization_strength, method)
    return jnp.array(ranks)


def spearman_loss_impl(predictions, targets, regularization_strength, method):
    """Concrete implementation calling Rust."""
    pred_list = predictions.tolist()
    targ_list = targets.tolist()
    loss = rank_relax.spearman_loss(pred_list, targ_list, regularization_strength)
    return jnp.array(loss)


# Register implementations
soft_rank_p.def_impl(soft_rank_impl)
spearman_loss_p.def_impl(spearman_loss_impl)


# Abstract evaluation
def soft_rank_abstract_eval(values_aval, regularization_strength, method):
    """Shape/dtype inference for soft_rank."""
    return core.ShapedArray(values_aval.shape, values_aval.dtype)


def spearman_loss_abstract_eval(pred_aval, target_aval, regularization_strength, method):
    """Shape/dtype inference for spearman_loss."""
    return core.ShapedArray((), pred_aval.dtype)  # Scalar output


soft_rank_p.def_abstract_eval(soft_rank_abstract_eval)
spearman_loss_p.def_abstract_eval(spearman_loss_abstract_eval)


# JVP rules (forward-mode AD)
def soft_rank_jvp(primals, tangents, *, regularization_strength, method):
    """Forward-mode AD for soft_rank."""
    values, = primals
    values_dot, = tangents
    
    # Primal output
    y = soft_rank_jax(values, regularization_strength, method)
    
    # Tangent output: use analytical gradient
    val_list = values.tolist()
    ranks = rank_relax.soft_rank_with_method(val_list, regularization_strength, method)
    grad = rank_relax.soft_rank_gradient(val_list, ranks, regularization_strength)
    
    # y_dot = grad_matrix @ values_dot
    y_dot = jnp.zeros_like(y)
    for i in range(len(val_list)):
        for j in range(len(val_list)):
            y_dot = y_dot.at[i].add(grad[i][j] * values_dot[j])
    
    return y, y_dot


def spearman_loss_jvp(primals, tangents, *, regularization_strength, method):
    """Forward-mode AD for spearman_loss."""
    predictions, targets = primals
    pred_dot, target_dot = tangents
    
    # Primal output
    y = spearman_loss_jax(predictions, targets, regularization_strength, method)
    
    # Tangent output: use analytical gradient
    pred_list = predictions.tolist()
    targ_list = targets.tolist()
    
    pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
    target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
    
    grad = rank_relax.spearman_loss_gradient(
        pred_list, targ_list, pred_ranks, target_ranks, regularization_strength
    )
    
    # y_dot = grad @ pred_dot (no gradient for targets in loss)
    y_dot = jnp.sum(jnp.array(grad) * pred_dot)
    
    return y, y_dot


ad.primitive_jvps[soft_rank_p] = soft_rank_jvp
ad.primitive_jvps[spearman_loss_p] = spearman_loss_jvp


# Transpose rules (reverse-mode AD)
def soft_rank_transpose(ct, values, *, regularization_strength, method):
    """Reverse-mode AD for soft_rank."""
    if values is None:
        return None, None
    
    # Compute gradient matrix
    val_list = values.tolist()
    ranks = rank_relax.soft_rank_with_method(val_list, regularization_strength, method)
    grad = rank_relax.soft_rank_gradient(val_list, ranks, regularization_strength)
    
    # values_bar = grad_matrix^T @ ct
    values_bar = jnp.zeros_like(values)
    for i in range(len(val_list)):
        for j in range(len(val_list)):
            values_bar = values_bar.at[j].add(grad[i][j] * ct[i])
    
    return values_bar, None, None


def spearman_loss_transpose(ct, predictions, targets, *, regularization_strength, method):
    """Reverse-mode AD for spearman_loss."""
    if predictions is None:
        return None, None, None, None
    
    # Compute gradient
    pred_list = predictions.tolist()
    targ_list = targets.tolist()
    
    pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
    target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
    
    grad = rank_relax.spearman_loss_gradient(
        pred_list, targ_list, pred_ranks, target_ranks, regularization_strength
    )
    
    # predictions_bar = grad * ct (scalar ct)
    predictions_bar = jnp.array(grad) * ct
    
    return predictions_bar, None, None, None


ad.primitive_transposes[soft_rank_p] = soft_rank_transpose
ad.primitive_transposes[spearman_loss_p] = spearman_loss_transpose


# Example usage
if __name__ == "__main__":
    # Test
    key = jax.random.PRNGKey(42)
    
    predictions = jax.random.normal(key, (10,))
    targets = jax.random.normal(key, (10,))
    
    # Forward pass
    loss = spearman_loss_jax(predictions, targets, regularization_strength=1.0)
    print(f"Loss: {loss:.4f}")
    
    # Test gradients
    grad_fn = jax.grad(lambda p: spearman_loss_jax(p, targets, 1.0))
    grads = grad_fn(predictions)
    print(f"Gradients computed: {grads is not None}")
    print(f"Gradient norm: {jnp.linalg.norm(grads):.4f}")
    
    # Test JIT
    loss_jit = jax.jit(
        lambda p, t: spearman_loss_jax(p, t, 1.0)
    )(predictions, targets)
    print(f"JIT loss: {loss_jit:.4f}")

