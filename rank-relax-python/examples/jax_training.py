"""
Example: Using rank-relax in JAX training loop.

This shows how to integrate the optimized Rust code into a JAX training workflow.
"""

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import ad
import rank_relax


# Define JAX primitive for Spearman loss
spearman_loss_p = core.Primitive("spearman_loss_rust")


def spearman_loss_jax(predictions, targets, regularization_strength=1.0):
    """
    User-facing function for Spearman loss in JAX.
    
    Args:
        predictions: JAX array [n_items] or [batch_size, n_items]
        targets: JAX array [n_items] or [batch_size, n_items]
        regularization_strength: float
    
    Returns:
        loss: JAX array (scalar or [batch_size])
    """
    return spearman_loss_p.bind(predictions, targets, regularization_strength)


# Concrete implementation
def spearman_loss_impl(predictions, targets, regularization_strength):
    """Call Rust implementation."""
    # Convert to Python lists
    pred_list = predictions.tolist()
    targ_list = targets.tolist()
    
    # Handle batched case
    if predictions.ndim == 2:
        batch_size = predictions.shape[0]
        losses = []
        for i in range(batch_size):
            loss_val = rank_relax.spearman_loss(
                pred_list[i], targ_list[i], regularization_strength
            )
            losses.append(loss_val)
        return jnp.array(losses)
    else:
        loss_val = rank_relax.spearman_loss(
            pred_list, targ_list, regularization_strength
        )
        return jnp.array(loss_val)


spearman_loss_p.def_impl(spearman_loss_impl)


# Abstract evaluation
def spearman_loss_abstract_eval(pred_aval, target_aval, reg_aval):
    """Return output shape/dtype."""
    if pred_aval.ndim == 2:
        # Batched: return [batch_size]
        return core.ShapedArray((pred_aval.shape[0],), pred_aval.dtype)
    else:
        # Single: return scalar
        return core.ShapedArray((), pred_aval.dtype)


spearman_loss_p.def_abstract_eval(spearman_loss_abstract_eval)


# JVP rule (forward-mode AD)
def spearman_loss_jvp(primals, tangents, *, regularization_strength):
    """Forward-mode automatic differentiation."""
    predictions, targets = primals
    pred_tangent, target_tangent = tangents
    
    # Primal output
    y = spearman_loss_jax(predictions, targets, regularization_strength)
    
    # Compute gradients using JAX's autodiff
    def loss_fn(p, t):
        return spearman_loss_jax(p, t, regularization_strength)
    
    # Get gradients
    pred_grad_fn = jax.grad(loss_fn, argnums=0)
    target_grad_fn = jax.grad(loss_fn, argnums=1)
    
    pred_grad = pred_grad_fn(predictions, targets)
    target_grad = target_grad_fn(predictions, targets)
    
    # Tangent = gradient · tangent_input
    if predictions.ndim == 2:
        # Batched: sum over items dimension
        y_dot = jnp.sum(pred_grad * pred_tangent, axis=-1) + jnp.sum(
            target_grad * target_tangent, axis=-1
        )
    else:
        y_dot = jnp.sum(pred_grad * pred_tangent) + jnp.sum(
            target_grad * target_tangent
        )
    
    return y, y_dot


ad.primitive_jvps[spearman_loss_p] = spearman_loss_jvp


# Transpose rule (reverse-mode AD)
def spearman_loss_transpose(ct, predictions, targets, *, regularization_strength):
    """Reverse-mode automatic differentiation."""
    # Compute gradients
    def loss_fn(p, t):
        return spearman_loss_jax(p, t, regularization_strength)
    
    pred_grad = jax.grad(loss_fn, argnums=0)(predictions, targets)
    target_grad = jax.grad(loss_fn, argnums=1)(predictions, targets)
    
    # Cotangents = gradient * output_cotangent
    if predictions is not None:
        if predictions.ndim == 2:
            # Batched: broadcast ct to match
            pred_bar = pred_grad * jnp.expand_dims(ct, axis=-1)
        else:
            pred_bar = pred_grad * ct
    else:
        pred_bar = None
    
    if targets is not None:
        if targets.ndim == 2:
            target_bar = target_grad * jnp.expand_dims(ct, axis=-1)
        else:
            target_bar = target_grad * ct
    else:
        target_bar = None
    
    return pred_bar, target_bar, None


ad.primitive_transposes[spearman_loss_p] = spearman_loss_transpose


# Example: Training with Flax
def training_step(state, batch):
    """Single training step using rank-relax."""
    predictions = state.apply_fn(state.params, batch.inputs)
    targets = batch.targets
    
    # Compute loss
    loss = spearman_loss_jax(predictions, targets, regularization_strength=1.0)
    loss = jnp.mean(loss)  # Average over batch
    
    # Compute gradients
    def loss_fn(params):
        pred = state.apply_fn(params, batch.inputs)
        return jnp.mean(spearman_loss_jax(pred, targets, 1.0))
    
    grads = jax.grad(loss_fn)(state.params)
    
    # Update state
    state = state.apply_gradients(grads=grads)
    
    return state, loss


# JIT-compiled version
@jax.jit
def training_step_jit(state, batch):
    """JIT-compiled training step."""
    return training_step(state, batch)


# Example usage
if __name__ == "__main__":
    # Test
    key = jax.random.PRNGKey(42)
    
    n_items = 10
    predictions = jax.random.normal(key, (n_items,))
    targets = jax.random.normal(key, (n_items,))
    
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

