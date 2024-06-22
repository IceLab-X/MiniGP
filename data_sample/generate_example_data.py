import torch
import matplotlib.pyplot as plt


def generate(training_size=200, test_size=100, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Generate training data
    xtr = torch.rand(training_size, 1)
    ytr = ((6 * xtr - 2) ** 2) * torch.sin(12 * xtr - 4) + torch.randn(training_size, 1)

    # Generate test data
    xte = torch.linspace(0, 1, test_size).view(-1, 1)
    yte = ((6 * xte - 2) ** 2) * torch.sin(12 * xte - 4)

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