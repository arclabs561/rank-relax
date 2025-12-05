"""
Complete training example using rank-relax with analytical gradients.

This demonstrates how to use rank-relax in a real PyTorch training loop
with proper gradient flow and multiple ranking methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import rank_relax


class SpearmanLossAutograd(torch.autograd.Function):
    """
    PyTorch autograd function with analytical gradients from rank-relax.
    
    This preserves gradient flow and uses efficient Rust-implemented gradients.
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, regularization_strength, method="sigmoid"):
        """Forward pass: compute Spearman loss."""
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
                    regularization_strength,
                )
                losses.append(loss_val)
            
            return torch.tensor(
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
                regularization_strength,
            )
            
            return torch.tensor(
                loss_val,
                device=predictions.device,
                dtype=predictions.dtype,
                requires_grad=True
            )
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: use analytical gradients from rank-relax."""
        predictions, targets = ctx.saved_tensors
        regularization_strength = ctx.regularization_strength
        method = ctx.method
        
        # Convert to numpy
        pred = predictions.detach().cpu().numpy()
        targ = targets.detach().cpu().numpy()
        grad_out = grad_output.detach().cpu().numpy()
        
        # Get ranks for gradient computation
        pred_ranks = rank_relax.soft_rank(
            pred.tolist(),
            regularization_strength,
            method=method
        )
        target_ranks = rank_relax.soft_rank(
            targ.tolist(),
            regularization_strength,
            method=method
        )
        
        # Compute analytical gradient using Rust implementation
        grad = rank_relax.spearman_loss_gradient(
            pred.tolist(),
            targ.tolist(),
            pred_ranks,
            target_ranks,
            regularization_strength,
        )
        
        # Scale by grad_output and convert back to tensor
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
        
        return grad_tensor, None, None, None


def spearman_loss_pytorch(predictions, targets, regularization_strength=1.0, method="sigmoid"):
    """User-friendly wrapper for Spearman loss with gradients."""
    return SpearmanLossAutograd.apply(
        predictions, targets, regularization_strength, method
    )


# Example: Training a ranking model
class RankingModel(nn.Module):
    """Simple model that outputs relevance scores for ranking."""
    
    def __init__(self, input_dim, hidden_dim, n_items):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_items)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """x: [batch_size, input_dim] -> [batch_size, n_items]"""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def training_loop(model, train_loader, optimizer, device, epochs=10, method="sigmoid"):
    """Training loop using rank-relax Spearman loss."""
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)  # [batch_size, n_items]
            
            # Forward pass
            predictions = model(inputs)  # [batch_size, n_items]
            
            # Compute Spearman loss with gradients preserved
            loss = spearman_loss_pytorch(
                predictions, targets, regularization_strength=1.0, method=method
            )
            
            # Average over batch
            loss = loss.mean()
            
            # Backward pass (gradients flow through Rust implementation!)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} average loss: {total_loss / len(train_loader):.4f}")


# Example: Compare different ranking methods
def compare_methods(predictions, targets):
    """Compare different ranking methods on the same data."""
    methods = ["sigmoid", "neural_sort", "probabilistic", "smooth_i"]
    
    print("Comparing ranking methods:")
    print("-" * 60)
    
    for method in methods:
        loss = spearman_loss_pytorch(
            predictions, targets, regularization_strength=1.0, method=method
        )
        print(f"{method:15s}: Loss = {loss.item():.6f}")
    
    print("-" * 60)


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test gradient flow
    print("Testing gradient flow...")
    predictions = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5], requires_grad=True, device=device)
    targets = torch.tensor([0.0, 1.0, 0.2, 0.8, 0.4], device=device)
    
    loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    print(f"Gradients computed: {predictions.grad is not None}")
    print(f"Gradient norm: {predictions.grad.norm().item():.4f}")
    print()
    
    # Compare methods
    print("Comparing ranking methods:")
    compare_methods(predictions, targets)
    
    # Example training (commented out - requires data loader)
    # model = RankingModel(input_dim=128, hidden_dim=64, n_items=100).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # training_loop(model, train_loader, optimizer, device, epochs=10)

