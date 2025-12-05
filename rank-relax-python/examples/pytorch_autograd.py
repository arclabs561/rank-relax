"""
PyTorch autograd function with analytical gradients.

This implements proper gradient flow through rank-relax operations.
"""

import torch
import torch.autograd as autograd
import rank_relax


class SoftRankAutograd(autograd.Function):
    """PyTorch autograd function for soft_rank with analytical gradients."""
    
    @staticmethod
    def forward(ctx, values, regularization_strength, method="sigmoid"):
        """
        Forward pass: compute soft ranks.
        
        Args:
            values: Tensor [n] or [batch_size, n]
            regularization_strength: float
            method: str, one of ["sigmoid", "neural_sort", "probabilistic", "smooth_i"]
        
        Returns:
            ranks: Tensor with same shape as values
        """
        ctx.save_for_backward(values)
        ctx.regularization_strength = regularization_strength
        ctx.method = method
        
        # Handle batched case
        if values.dim() == 2:
            batch_size, n = values.shape
            ranks_list = []
            
            for i in range(batch_size):
                val = values[i].detach().cpu().numpy()
                ranks = rank_relax.soft_rank_with_method(
                    val.tolist(),
                    regularization_strength,
                    method
                )
                ranks_list.append(ranks)
            
            ranks_tensor = torch.tensor(
                ranks_list,
                device=values.device,
                dtype=values.dtype,
                requires_grad=True
            )
        else:
            val = values.detach().cpu().numpy()
            ranks = rank_relax.soft_rank_with_method(
                val.tolist(),
                regularization_strength,
                method
            )
            ranks_tensor = torch.tensor(
                ranks,
                device=values.device,
                dtype=values.dtype,
                requires_grad=True
            )
        
        return ranks_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute analytical gradients.
        
        Uses the analytical gradient formula from rank-relax.
        """
        values, = ctx.saved_tensors
        regularization_strength = ctx.regularization_strength
        method = ctx.method
        
        # Convert to numpy for gradient computation
        val = values.detach().cpu().numpy()
        grad_out = grad_output.detach().cpu().numpy()
        
        # Get ranks for gradient computation
        ranks = rank_relax.soft_rank_with_method(
            val.tolist(),
            regularization_strength,
            method
        )
        
        # Compute analytical gradient using rank-relax
        # This uses the efficient Rust implementation
        grad = rank_relax.soft_rank_gradient(
            val.tolist(),
            ranks,
            regularization_strength
        )
        
        # Apply chain rule: grad_input = grad_output * grad_matrix
        n = len(val)
        grad_input = torch.zeros(n, device=values.device, dtype=values.dtype)
        
        for i in range(n):
            for j in range(n):
                grad_input[j] += grad_output[i] * grad[i][j]
        
        return grad_input, None, None


class SpearmanLossAutograd(autograd.Function):
    """PyTorch autograd function for spearman_loss with analytical gradients."""
    
    @staticmethod
    def forward(ctx, predictions, targets, regularization_strength, method="sigmoid"):
        """
        Forward pass: compute Spearman loss.
        
        Args:
            predictions: Tensor [n] or [batch_size, n]
            targets: Tensor [n] or [batch_size, n]
            regularization_strength: float
            method: str, ranking method to use
        
        Returns:
            loss: Tensor scalar or [batch_size]
        """
        ctx.save_for_backward(predictions, targets)
        ctx.regularization_strength = regularization_strength
        ctx.method = method
        
        # Handle batched case
        if predictions.dim() == 2:
            batch_size = predictions.shape[0]
            losses = []
            
            for i in range(batch_size):
                pred = predictions[i].detach().cpu().numpy()
                targ = targets[i].detach().cpu().numpy()
                
                loss_val = rank_relax.spearman_loss(
                    pred.tolist(),
                    targ.tolist(),
                    regularization_strength
                )
                losses.append(loss_val)
            
            loss_tensor = torch.tensor(
                losses,
                device=predictions.device,
                dtype=predictions.dtype,
                requires_grad=True
            )
        else:
            pred = predictions.detach().cpu().numpy()
            targ = targets.detach().cpu().numpy()
            
            loss_val = rank_relax.spearman_loss(
                pred.tolist(),
                targ.tolist(),
                regularization_strength
            )
            
            loss_tensor = torch.tensor(
                loss_val,
                device=predictions.device,
                dtype=predictions.dtype,
                requires_grad=True
            )
        
        return loss_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute analytical gradients.
        
        Uses spearman_loss_gradient from rank-relax.
        """
        predictions, targets = ctx.saved_tensors
        regularization_strength = ctx.regularization_strength
        method = ctx.method
        
        # Convert to numpy
        pred = predictions.detach().cpu().numpy()
        targ = targets.detach().cpu().numpy()
        grad_out = grad_output.detach().cpu().numpy()
        
        # Get ranks
        pred_ranks = rank_relax.soft_rank_with_method(
            pred.tolist(),
            regularization_strength,
            method
        )
        target_ranks = rank_relax.soft_rank_with_method(
            targ.tolist(),
            regularization_strength,
            method
        )
        
        # Compute analytical gradient
        grad = rank_relax.spearman_loss_gradient(
            pred.tolist(),
            targ.tolist(),
            pred_ranks,
            target_ranks,
            regularization_strength
        )
        
        # Scale by grad_output
        if predictions.dim() == 2:
            batch_size = predictions.shape[0]
            grad_tensor = torch.zeros_like(predictions)
            
            for i in range(batch_size):
                grad_scaled = [g * grad_out[i] for g in grad]
                grad_tensor[i] = torch.tensor(
                    grad_scaled,
                    device=predictions.device,
                    dtype=predictions.dtype
                )
        else:
            grad_scaled = [g * grad_out.item() for g in grad]
            grad_tensor = torch.tensor(
                grad_scaled,
                device=predictions.device,
                dtype=predictions.dtype
            )
        
        # No gradient for targets
        return grad_tensor, None, None, None


# User-friendly wrappers
def soft_rank_pytorch(values, regularization_strength=1.0, method="sigmoid"):
    """Compute soft ranks with gradient support."""
    return SoftRankAutograd.apply(values, regularization_strength, method)


def spearman_loss_pytorch(predictions, targets, regularization_strength=1.0, method="sigmoid"):
    """Compute Spearman loss with gradient support."""
    return SpearmanLossAutograd.apply(
        predictions, targets, regularization_strength, method
    )


# Example usage
if __name__ == "__main__":
    # Test gradient flow
    predictions = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.2, 0.8, 0.4])
    
    # Compute loss with gradients
    loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print(f"Gradients: {predictions.grad}")
    print(f"Gradient norm: {predictions.grad.norm().item():.4f}")

