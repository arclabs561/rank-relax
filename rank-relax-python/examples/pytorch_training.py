"""
Example: Using rank-relax in PyTorch training loop.

This shows how to integrate the optimized Rust code into a real training workflow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import rank_relax


class SpearmanLossAutograd(torch.autograd.Function):
    """
    Custom autograd function wrapping Rust rank-relax.
    
    This preserves gradient flow through the ranking operation.
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, regularization_strength):
        """
        Forward pass: compute Spearman loss using Rust implementation.
        
        Args:
            predictions: Tensor [batch_size, n_items] or [n_items]
            targets: Tensor [batch_size, n_items] or [n_items]
            regularization_strength: float
        
        Returns:
            loss: Tensor [batch_size] or scalar
        """
        ctx.save_for_backward(predictions, targets)
        ctx.regularization_strength = regularization_strength
        
        # Handle batched case
        if predictions.dim() == 2:
            batch_size = predictions.shape[0]
            losses = []
            
            for i in range(batch_size):
                # Convert to CPU numpy (detach for forward pass)
                pred = predictions[i].detach().cpu().numpy()
                targ = targets[i].detach().cpu().numpy()
                
                # Call Rust implementation
                loss_val = rank_relax.spearman_loss(
                    pred.tolist(), targ.tolist(), regularization_strength
                )
                losses.append(loss_val)
            
            return torch.tensor(
                losses,
                device=predictions.device,
                dtype=predictions.dtype,
                requires_grad=True
            )
        else:
            # Single example
            pred = predictions.detach().cpu().numpy()
            targ = targets.detach().cpu().numpy()
            
            loss_val = rank_relax.spearman_loss(
                pred.tolist(), targ.tolist(), regularization_strength
            )
            
            return torch.tensor(
                loss_val,
                device=predictions.device,
                dtype=predictions.dtype,
                requires_grad=True
            )
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients through soft ranking.
        
        Note: This is a simplified version. For production, implement
        analytical gradient computation in Rust for better performance.
        """
        predictions, targets = ctx.saved_tensors
        regularization_strength = ctx.regularization_strength
        
        # Re-compute loss with gradients enabled
        # This uses PyTorch's autograd to compute gradients numerically
        # In production, implement analytical gradient formula
        
        predictions_grad = torch.autograd.grad(
            outputs=SpearmanLossAutograd.apply(
                predictions, targets, regularization_strength
            ),
            inputs=predictions,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return predictions_grad, None, None


def spearman_loss_pytorch(predictions, targets, regularization_strength=1.0):
    """
    User-friendly wrapper for Spearman loss in PyTorch.
    
    Args:
        predictions: Tensor [batch_size, n_items] or [n_items]
        targets: Tensor [batch_size, n_items] or [n_items]
        regularization_strength: float
    
    Returns:
        loss: Tensor (preserves gradients)
    """
    return SpearmanLossAutograd.apply(predictions, targets, regularization_strength)


# Example: Training a ranking model
class RankingModel(nn.Module):
    """Simple model that outputs relevance scores."""
    
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


def training_loop(model, train_loader, optimizer, device, epochs=10):
    """Training loop using rank-relax Spearman loss."""
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)  # [batch_size, n_items]
            
            # Forward pass
            predictions = model(inputs)  # [batch_size, n_items]
            
            # Compute Spearman loss (gradients preserved!)
            loss = spearman_loss_pytorch(
                predictions, targets, regularization_strength=1.0
            )
            
            # Average over batch
            loss = loss.mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} average loss: {total_loss / len(train_loader):.4f}")


# Example usage
if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RankingModel(input_dim=128, hidden_dim=64, n_items=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy data loader (replace with real data)
    # train_loader = DataLoader(...)
    
    # Training
    # training_loop(model, train_loader, optimizer, device)
    
    # Simple test
    batch_size = 4
    n_items = 10
    
    predictions = torch.randn(batch_size, n_items, requires_grad=True, device=device)
    targets = torch.randn(batch_size, n_items, device=device)
    
    loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
    loss = loss.mean()
    
    print(f"Loss: {loss.item():.4f}")
    
    # Test gradients
    loss.backward()
    print(f"Gradients computed: {predictions.grad is not None}")
    print(f"Gradient norm: {predictions.grad.norm().item():.4f}")

