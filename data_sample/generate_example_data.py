import torch
import matplotlib.pyplot as plt


def generate(training_size=200, test_size=100, seed=None,input_dim=1):
    if seed is not None:
        torch.manual_seed(seed)

    # Generate training data
    xtr = torch.rand(training_size, input_dim)
    ytr = sum(((6 * xtr[:, i:i+1] - 2) ** 2) * torch.sin(12 * xtr[:, i:i+1] - 4) for i in range(input_dim)) + torch.randn(training_size, 1)

    # Generate test data
    xte = torch.linspace(0, 1, test_size).view(-1, 1).repeat(1, input_dim)
    yte = sum(((6 * xte[:, i:i+1] - 2) ** 2) * torch.sin(12 * xte[:, i:i+1] - 4) for i in range(input_dim))

    return xtr, ytr, xte, yte


def plot(xtr, ytr, xte, yte):
    if xtr is None or ytr is None or xte is None or yte is None:
        print("Data not generated. Please call generate_data() first.")
        return

    # Plot the data
    plt.plot(xtr.numpy(), ytr.numpy(), 'b+', label='Training data')
    plt.plot(xte.numpy(), yte.numpy(), 'r-', alpha=0.5, label='Test data')
    plt.legend()
    plt.show()


# Usage example:

if __name__ == "__main__":
    print('testing')

    xtr, ytr, xte, yte = generate(seed=42)
    plot(xtr, ytr, xte, yte)