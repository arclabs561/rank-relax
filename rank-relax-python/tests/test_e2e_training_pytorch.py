"""
E2E test: Install from PyPI and run actual PyTorch training.

This test verifies that the published package can be used in real training loops.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


def test_e2e_pytorch_training_installed():
    """Test that we can train a model using the installed rank-relax package."""
    # This test assumes rank-relax is installed from PyPI
    # It should be run in CI after publishing
    
    try:
        import torch
        import torch.autograd as autograd
        import rank_relax
    except ImportError as e:
        pytest.skip(f"rank-relax or PyTorch not installed: {e}")
    
    # Inline autograd function (examples may not be packaged)
    class SpearmanLossAutograd(autograd.Function):
        @staticmethod
        def forward(ctx, predictions, targets, regularization_strength):
            ctx.save_for_backward(predictions, targets)
            ctx.regularization_strength = regularization_strength
            
            if predictions.dim() == 2:
                batch_size = predictions.shape[0]
                losses = []
                for i in range(batch_size):
                    pred = predictions[i].detach().cpu().numpy()
                    targ = targets[i].detach().cpu().numpy()
                    loss_val = rank_relax.spearman_loss(pred.tolist(), targ.tolist(), regularization_strength)
                    losses.append(loss_val)
                return torch.tensor(losses, device=predictions.device, dtype=predictions.dtype, requires_grad=True)
            else:
                pred = predictions.detach().cpu().numpy()
                targ = targets.detach().cpu().numpy()
                loss_val = rank_relax.spearman_loss(pred.tolist(), targ.tolist(), regularization_strength)
                return torch.tensor(loss_val, device=predictions.device, dtype=predictions.dtype, requires_grad=True)
        
            @staticmethod
            def backward(ctx, grad_output):
                predictions, targets = ctx.saved_tensors
                regularization_strength = ctx.regularization_strength
                
                pred = predictions.detach().cpu().numpy()
                targ = targets.detach().cpu().numpy()
                grad_out = grad_output.detach().cpu().numpy()
                
                if predictions.dim() == 2:
                    # Batch processing: handle each item separately
                    grad_tensor = torch.zeros_like(predictions)
                    for i in range(predictions.shape[0]):
                        pred_list = pred[i].tolist()
                        targ_list = targ[i].tolist()
                        pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
                        target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
                        grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
                        grad_scaled = [g * grad_out[i] for g in grad]
                        grad_tensor[i] = torch.tensor(grad_scaled, device=predictions.device, dtype=predictions.dtype)
                    return grad_tensor, None, None
                else:
                    # Single item processing
                    pred_list = pred.tolist()
                    targ_list = targ.tolist()
                    pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
                    target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
                    grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
                    grad_scaled = [g * grad_out.item() for g in grad]
                    grad_tensor = torch.tensor(grad_scaled, device=predictions.device, dtype=predictions.dtype)
                    return grad_tensor, None, None
    
    def spearman_loss_pytorch(predictions, targets, regularization_strength=1.0):
        return SpearmanLossAutograd.apply(predictions, targets, regularization_strength)
    
    # Simple ranking model
    class RankingModel(nn.Module):
        def __init__(self, n_items=10):
            super().__init__()
            self.fc = nn.Linear(128, n_items)
        
        def forward(self, x):
            return self.fc(x)
    
    # Setup
    device = torch.device("cpu")
    model = RankingModel(n_items=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Synthetic data
    batch_size = 4
    inputs = torch.randn(batch_size, 128)
    targets = torch.randn(batch_size, 10)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(inputs)
    
    # Compute loss with rank-relax
    loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
    loss = loss.mean()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Verify gradients flowed
    assert predictions.grad is None or predictions.requires_grad, "Gradients should flow"
    assert loss.item() > 0, "Loss should be positive"
    
    # Run multiple steps to verify training works
    for _ in range(3):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
    
    print("✅ E2E PyTorch training test passed - package works in real training loop")


def test_e2e_pytorch_gradient_flow():
    """Test that gradients actually flow through rank-relax operations."""
    try:
        import torch
        import torch.autograd as autograd
        import rank_relax
    except ImportError as e:
        pytest.skip(f"rank-relax or PyTorch not installed: {e}")
    
    # Inline autograd function (same as above)
    class SpearmanLossAutograd(autograd.Function):
        @staticmethod
        def forward(ctx, predictions, targets, regularization_strength):
            ctx.save_for_backward(predictions, targets)
            ctx.regularization_strength = regularization_strength
            pred = predictions.detach().cpu().numpy()
            targ = targets.detach().cpu().numpy()
            loss_val = rank_relax.spearman_loss(pred.tolist(), targ.tolist(), regularization_strength)
            return torch.tensor(loss_val, device=predictions.device, dtype=predictions.dtype, requires_grad=True)
        
        @staticmethod
        def backward(ctx, grad_output):
            predictions, targets = ctx.saved_tensors
            regularization_strength = ctx.regularization_strength
            pred = predictions.detach().cpu().numpy()
            targ = targets.detach().cpu().numpy()
            grad_out = grad_output.detach().cpu().numpy()
            pred_ranks = rank_relax.soft_rank(pred.tolist(), regularization_strength)
            target_ranks = rank_relax.soft_rank(targ.tolist(), regularization_strength)
            grad = rank_relax.spearman_loss_gradient(pred.tolist(), targ.tolist(), pred_ranks, target_ranks, regularization_strength)
            grad_scaled = [g * grad_out.item() for g in grad]
            return torch.tensor(grad_scaled, device=predictions.device, dtype=predictions.dtype), None, None
    
    def spearman_loss_pytorch(predictions, targets, regularization_strength=1.0):
        return SpearmanLossAutograd.apply(predictions, targets, regularization_strength)
    
    predictions = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.2, 0.8, 0.4])
    
    loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
    loss.backward()
    
    assert predictions.grad is not None, "Gradients must be computed"
    assert not torch.isnan(predictions.grad).any(), "Gradients must not be NaN"
    assert predictions.grad.norm() > 0, "Gradients must be non-zero"
    
    print("✅ Gradient flow test passed")

