import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PolynomialNet(nn.Module):
    def __init__(self):
        super(PolynomialNet, self).__init__()
        # Initialize the 5 learnable coefficients
        self.coefficients = nn.Parameter(torch.randn(5))
    
    def forward(self, x):
        # Create powers of x: x^4, x^3, x^2, x^1, x^0
        x_powers = torch.stack([
            x.squeeze()**4,
            x.squeeze()**3,
            x.squeeze()**2,
            x.squeeze()**1,
            torch.ones_like(x.squeeze())
        ], dim=1)  # Shape: [batch_size, 5]
        
        # Multiply by coefficients and sum
        return torch.matmul(x_powers, self.coefficients.unsqueeze(1))  # Shape: [batch_size, 1]

# Generate training data
def generate_polynomial_data(true_coefficients, num_points=100, noise=0.1):
    x = np.linspace(-2, 2, num_points)
    powers = np.stack([x**4, x**3, x**2, x**1, np.ones_like(x)], axis=1)
    y = np.sum(true_coefficients * powers, axis=1)
    y += np.random.normal(0, noise, y.shape)
    return x, y

# Set true coefficients [a, b, c, d, e]
true_coefficients = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

# Training parameters
learning_rate = 0.01
epochs = 1000

# Generate data
x, y = generate_polynomial_data(true_coefficients)
x_tensor = torch.FloatTensor(x.reshape(-1, 1))
y_tensor = torch.FloatTensor(y.reshape(-1, 1))

# Initialize model, loss function, and optimizer
model = PolynomialNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
learned_coefficients = []

print("True coefficients:", true_coefficients)
print("Initial learned coefficients:", model.coefficients.data.numpy())

for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Store loss and coefficients
    losses.append(loss.item())
    learned_coefficients.append(model.coefficients.data.numpy().copy())
    
    # Print progress
    if (epoch + 1) % 200 == 0:
        print(f'\nEpoch [{epoch+1}/{epochs}]')
        print(f'Loss: {loss.item():.4f}')
        print(f'Learned coefficients: {model.coefficients.data.numpy()}')

# Final results
print("\nFinal Results:")
print("True coefficients:", true_coefficients)
print("Learned coefficients:", model.coefficients.data.numpy())

# Plotting
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

# Plot coefficient convergence
learned_coefficients = np.array(learned_coefficients)
plt.subplot(1, 3, 2)
for i in range(5):
    plt.plot(learned_coefficients[:, i], label=f'Coefficient {i}')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Coefficient Convergence')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()

# Plot polynomial fit
plt.subplot(1, 3, 3)
with torch.no_grad():
    y_pred = model(x_tensor).numpy()
plt.scatter(x, y, label='Data', alpha=0.5)
plt.plot(x, y_pred.squeeze(), 'r', label='Learned Polynomial')
true_y = np.sum(true_coefficients * np.stack([x**4, x**3, x**2, x, np.ones_like(x)], axis=1), axis=1)
plt.plot(x, true_y, 'g--', label='True Polynomial')
plt.title('Polynomial Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.savefig('polynomial_fit.png', dpi=400)
plt.close()
