import torch
import matplotlib.pyplot as plt


def generate(training_size=200, test_size=100, seed=None,input_dim=1,dtype=torch.float64):
    if seed is not None:
        torch.manual_seed(seed)

    # Generate training data
    xtr = torch.rand(training_size, input_dim, dtype=dtype)
    ytr = sum(((6 * xtr[:, i:i+1] - 2) ** 2) * torch.sin(12 * xtr[:, i:i+1] - 4) for i in range(input_dim)) + torch.randn(training_size, 1)

    # Generate test data
    xte = torch.linspace(0, 1, test_size,dtype=dtype).view(-1, 1).repeat(1, input_dim)
    yte = sum(((6 * xte[:, i:i+1] - 2) ** 2) * torch.sin(12 * xte[:, i:i+1] - 4) for i in range(input_dim))

    return xtr, ytr, xte, yte


# Usage example:
def generate_complex_data(num=400, seed=None):
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x-2))

    def sin(x):
        return torch.sin(0.5*torch.pi*x)
    # Define the segments
    if seed is not None:
        torch.manual_seed(seed)
    num1 = int(num * 0.45)
    num2 = int(num * 0.1)
    num3 = num - num1 - num2
    x1 = torch.linspace(-15, -7, num1)
    x2 = torch.linspace(-7, 3, num2)
    x3 = torch.linspace(3, 15, num3)

    # Compute the function values for each segment
    y1 = 0.3*sin(x1)-0.2933
    y2 = sigmoid(x2)
    y3 = 0.3*sin(x3)+1.2933

    # Concatenate the segments
    x_combined = torch.cat((x1, x2, x3))
    y_combined = torch.cat((y1, y2, y3))

    # Create random permutation of indices for splitting
    indices = torch.randperm(len(x_combined))
    split_index = int(0.5 * len(x_combined))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Split into training and test datasets
    xtr = x_combined[train_indices].view(-1,1)
    ytr = y_combined[train_indices].view(-1,1)+ torch.randn(len(train_indices), 1) * 0.1
    xte = x_combined[test_indices].view(-1,1)
    yte = y_combined[test_indices].view(-1,1)

    return xtr, ytr, xte, yte,x_combined, y_combined
def plot(xtr, ytr,x_combined, y_combined,figsize=(10, 6),show=True):
    if xtr is None or ytr is None or x_combined is None or y_combined is None:
        print("Data not generated. Please call generate_data() first.")
        return

    # Plot the data
    plt.figure(figsize=figsize)
    plt.plot(xtr.numpy(), ytr.numpy(), 'b+', label='Training data')
    plt.plot(x_combined.numpy(), y_combined.numpy(), 'r-', alpha=0.5, label='Latent function')
    plt.legend()
    if show:
        plt.show()

if __name__ == "__main__":
    print('testing')

    xtr, ytr, xte, yte,x_combined, y_combined = generate_complex_data(seed=42)
    plot(xtr, ytr, x_combined, y_combined)

    xtr, ytr, xte, yte = generate(seed=42)
    plot(xtr, ytr, xte, yte)