import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.autoGP import autoGP
from core.sgpr import vsgp
from core.kernel import ARDKernel, NeuralKernel

torch.set_default_dtype(torch.float64)


class DataGenerator:
    def __init__(self, n_datasets=100, n_samples=1000, x_range=10, x_dim=1):
        self.n_datasets = n_datasets
        self.n_samples = n_samples
        self.x_range = x_range
        self.x_dim = x_dim
        self.datasets = []

    def generate_polynomial_data(self):
        for _ in range(self.n_datasets):
            degree = np.random.randint(2, 5)
            coefficients = torch.FloatTensor(degree + 1).uniform_(-1, 1).tolist()

            X = torch.linspace(0, self.x_range, self.n_samples * self.x_dim).reshape(self.n_samples, self.x_dim)

            y_poly = sum(c * X ** i for i, c in enumerate(coefficients)).sum(dim=1) + \
                     torch.normal(mean=0, std=0.1, size=(self.n_samples,))

            self.datasets.append((X, y_poly))

    def generate_warped_data(self):
        for _ in range(self.n_datasets):
            scale = torch.FloatTensor(1).uniform_(0.5, 2.0).item()
            shift = torch.FloatTensor(1).uniform_(-2.0, 2.0).item()

            X = torch.linspace(0, self.x_range, self.n_samples * self.x_dim).reshape(self.n_samples, self.x_dim)

            X_warped = scale * torch.exp(X + shift) / (1 + torch.exp(X + shift))
            y_warped = 2 * X_warped.sum(dim=1) + torch.normal(mean=0, std=0.1, size=(self.n_samples,))

            self.datasets.append((X, y_warped))

    def generate_periodic_data(self):
        for _ in range(self.n_datasets):
            frequency = torch.FloatTensor(1).uniform_(0.5, 1.0).item()
            amplitude = torch.FloatTensor(1).uniform_(0.5, 2.0).item()
            phase = torch.FloatTensor(1).uniform_(0, np.pi).item()

            X = torch.linspace(0, self.x_range, self.n_samples * self.x_dim).reshape(self.n_samples, self.x_dim)

            y_periodic = amplitude * torch.sin(frequency * X + phase).sum(dim=1) + \
                         torch.normal(mean=0, std=0.1, size=(self.n_samples,))

            self.datasets.append((X, y_periodic))

    def plot_datasets(self, num_plots=5):
        for i in range(num_plots):
            X, y = self.datasets[i]
            plt.figure()
            if self.x_dim == 1:
                plt.scatter(X.numpy(), y.numpy(), s=1)
                plt.xlabel('Feature')
            else:
                plt.scatter(X[:, 0].numpy(), y.numpy(), s=1)
                plt.xlabel('First Feature')
            plt.ylabel('Target')
            plt.title(f'Dataset {i + 1}')
            plt.show()


# Example usage:

torch.manual_seed(1)
np.random.seed(0)

data_gen = DataGenerator(n_datasets=20, n_samples=300, x_range=20, x_dim=1)
data_gen.generate_polynomial_data()
data_gen.generate_warped_data()
data_gen.generate_periodic_data()
#data_gen.plot_datasets(num_plots=5)

model1_better_count = 0
model2_better_count = 0
similar_performance_count = 0
fail_model1_count = 0
fail_model2_count = 0

r2_scores_model1 = []
r2_scores_model2 = []

number_of_datasets = 0
for X, y in data_gen.datasets:
    number_of_datasets += 1
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float64).view(-1, 1)

    # Model 1: autoGP
    model1 = autoGP(X_train, y_train, normal_method="standard", kernel=ARDKernel, inputwarp=False, deepkernel=False)
    model1.train_auto()
    y_pred_model1, _ = model1.forward(X_test)
    r2_model1 = r2_score(y_test.detach(), y_pred_model1.detach())
    mse1 = torch.mean((y_pred_model1 - y_test) ** 2)
    print("AutoGP R^2:", r2_model1, mse1)

    # Model 2: vsgp
    model2 = vsgp(X_train, y_train, 5)
    model2.train_adam(100, 0.1)
    y_pred_model2, _ = model2.forward(X_test)
    r2_model2 = r2_score(y_test.detach(), y_pred_model2.detach())
    mse2 = torch.mean((y_pred_model2 - y_test) ** 2)
    print("SGP R^2:", r2_model2, mse2)

    # Check for model failure
    model1_failed = r2_model1 < 0.3
    model2_failed = r2_model2 < 0.3

    if model1_failed:
        fail_model1_count += 1
        print(f"Dataset {number_of_datasets}: autoGP fails to capture the structure of the data.")

    if model2_failed:
        fail_model2_count += 1
        print(f"Dataset {number_of_datasets}: vsgp fails to capture the structure of the data.")

    # Skip performance comparison if either model fails
    if model1_failed and model2_failed:
        continue

    # Store R² scores
    r2_scores_model1.append(r2_model1)
    r2_scores_model2.append(r2_model2)

    # Compare R² scores
    if abs(r2_model1 - r2_model2) < 0.01:
        similar_performance_count += 1
        print(f"Dataset {number_of_datasets}: Models have similar performance.")
    elif r2_model1 > r2_model2:
        model1_better_count += 1
    else:
        model2_better_count += 1
        print(f"Dataset {number_of_datasets}: vsgp performs better.")

# Print the number of times each model performed better
print(f'Model 1 (autoGP) performed better {model1_better_count} times.')
print(f'Model 2 (vsgp) performed better {model2_better_count} times.')
print(f'Models had similar performance {similar_performance_count} times.')
print(f'autoGP failed {fail_model1_count} times.')
print(f'vsgp failed {fail_model2_count} times.')


# Plot the R² scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(r2_scores_model1) + 1), r2_scores_model1, '^-', label='autoGP R²')
plt.plot(range(1, len(r2_scores_model2) + 1), r2_scores_model2, 'o-', label='vsgp R²')
plt.xlabel('Dataset Number', fontsize=16)
plt.ylabel('R² Score', fontsize=16)
plt.title('R² Scores of autoGP and vsgp across Different Datasets', fontsize=18)
plt.legend(fontsize=14)

# Optional: Increase tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# Save the plot as an image file
plt.savefig('Model_comparison_autoGP.png')
plt.show()
