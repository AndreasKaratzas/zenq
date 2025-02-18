import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class CubicSpline(nn.Module):
    def __init__(self, num_segments, enforce_continuity=True):
        super().__init__()
        self.num_segments = num_segments
        self.enforce_continuity = enforce_continuity
        
        # Initialize knot points (segment boundaries)
        self.register_buffer('knots', torch.linspace(-2, 2, num_segments + 1))
        
        if enforce_continuity:
            # For n segments, we need n+1 points for position
            # Remaining coefficients will be computed from continuity
            self.control_points = nn.Parameter(torch.randn(num_segments + 1, 1) * 0.1)
            self.derivatives = nn.Parameter(torch.randn(num_segments + 1, 1) * 0.1)
        else:
            # Direct coefficient learning (4 coefficients per segment)
            self.coefficients = nn.Parameter(torch.randn(num_segments, 4) * 0.1)
    
    def compute_coefficients(self):
        if not self.enforce_continuity:
            return self.coefficients
        
        coefficients = torch.zeros(self.num_segments, 4, device=self.knots.device)
        
        for i in range(self.num_segments):
            # Get segment information
            x0 = self.knots[i]
            x1 = self.knots[i + 1]
            h = x1 - x0
            
            # Get control points and derivatives for this segment
            y0 = self.control_points[i]
            y1 = self.control_points[i + 1]
            dy0 = self.derivatives[i]
            dy1 = self.derivatives[i + 1]
            
            # Compute cubic Hermite spline coefficients
            # f(x) = a + b(x-x₀) + c(x-x₀)² + d(x-x₀)³
            a = y0
            b = dy0
            c = (3*(y1 - y0)/h - 2*dy0 - dy1)/h
            d = (2*(y0 - y1)/h + dy0 + dy1)/h**2
            
            coefficients[i] = torch.cat([a, b, c, d])
        
        return coefficients
    
    def forward(self, x):
        # Ensure x is 2D
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # Get coefficients
        coeffs = self.compute_coefficients() if self.enforce_continuity else self.coefficients
        
        # Find which segment each x belongs to
        segment_idx = torch.sum(x >= self.knots.unsqueeze(0), dim=1) - 1
        segment_idx = torch.clamp(segment_idx, 0, self.num_segments - 1)
        
        # Get segment start points
        x0 = self.knots[segment_idx]
        
        # Compute local coordinates
        x_local = x - x0.unsqueeze(-1)
        
        # Build power matrix
        x_powers = torch.cat([
            torch.ones_like(x_local),
            x_local,
            x_local**2,
            x_local**3
        ], dim=-1)
        
        # Get coefficients for each point's segment
        batch_coeffs = coeffs[segment_idx]
        
        # Compute values
        y = torch.sum(batch_coeffs * x_powers, dim=-1, keepdim=True)
        
        return y

def generate_data(num_points=200):
    """Generate synthetic data with clear structure"""
    x = np.linspace(-2, 2, num_points)
    y = np.sin(2 * x) + 0.5 * x**2
    y += np.random.normal(0, 0.1, x.shape)  # Add some noise
    return x, y

def train_spline(model, x, y, num_epochs=1000, batch_size=32, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    x_tensor = torch.FloatTensor(x.reshape(-1, 1))
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))
    dataset_size = len(x)
    
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, dataset_size, batch_size):
            batch_x = x_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            # Forward pass
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses

def plot_results(model, x, y, losses):
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    x_sorted = np.sort(x)
    x_tensor = torch.FloatTensor(x_sorted.reshape(-1, 1))
    
    with torch.no_grad():
        y_pred = model(x_tensor).numpy()
    
    plt.scatter(x, y, color='blue', alpha=0.5, label='Data')
    plt.plot(x_sorted, y_pred, 'r-', label='Spline Fit')
    
    # Plot knot points
    knots = model.knots.cpu().numpy()
    if model.enforce_continuity:
        knot_values = model.control_points.detach().cpu().numpy()
        plt.scatter(knots, knot_values, color='green', label='Knot Points')
    
    plt.title('Spline Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate data
    x, y = generate_data(num_points=200)
    
    # Create and train model
    model = CubicSpline(num_segments=8, enforce_continuity=True)
    losses = train_spline(model, x, y, num_epochs=1000, batch_size=32, learning_rate=0.01)
    
    # Plot results
    plot_results(model, x, y, losses)

if __name__ == "__main__":
    main()