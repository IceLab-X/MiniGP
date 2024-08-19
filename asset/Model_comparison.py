# A Python script that compares the performance of different GP models on various synthetic datasets, including periodic, warped, and polynomial. The default models are set as autoGP and its base model vsgp.
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from core.autoGP import autoGP
from core.sgpr import vsgp
from core.kernel import ARDKernel, NeuralKernel
import core.GP_CommonCalculation as GP
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
            y_warped = 2 * X_warped.sum(dim=1) + 0.5*torch.normal(mean=0, std=0.1, size=(self.n_samples,))

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

data_gen = DataGenerator(n_datasets=24, n_samples=800, x_range=20, x_dim=1)
# data_gen.generate_polynomial_data()
data_gen.generate_warped_data()
# data_gen.generate_periodic_data()
#data_gen.plot_datasets(num_plots=5)
device = 'cpu'
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
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.5, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float64)
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float64).view(-1, 1)

    normalizer = GP.DataNormalization(method='min_max')
    normalizer.fit(X_train, 'x')
    normalizer.fit(y_train, 'y')
    normalizer.fit(X_test, 'xte')
    X_train_normalized = normalizer.normalize(X_train, 'x')
    y_train_normalized = normalizer.normalize(y_train, 'y')
    X_test_normalized = normalizer.normalize(X_test, 'xte')
    # Model 1: autoGP
    model1 = autoGP(input_dim=1,kernel=NeuralKernel, inputwarp=False, deepkernel=False, device=device)
    model1.train_auto(X_train_normalized, y_train_normalized)
    y_pred_model1, _ = model1.forward(X_train_normalized,y_train_normalized,X_test_normalized)
    y_pred_model1 = normalizer.denormalize(y_pred_model1, 'y')
    r2_model1 = r2_score(y_test.detach(), y_pred_model1.detach())
    mse1 = torch.mean((y_pred_model1 - y_test) ** 2)
    print("AutoGP R^2:", r2_model1, mse1)

    # Model 2: vsgp
    model2 = autoGP(input_dim=1,kernel=NeuralKernel, inputwarp=True, deepkernel=False, device=device)
    model2.train_auto(X_train_normalized, y_train_normalized)
    y_pred_model2, _ = model1.forward(X_train_normalized,y_train_normalized,X_test_normalized)
    y_pred_model2 = normalizer.denormalize(y_pred_model2, 'y')
    r2_model2 = r2_score(y_test.detach(), y_pred_model2.detach())
    mse2 = torch.mean((y_pred_model2 - y_test) ** 2)
    print("SGP R^2:", r2_model2, mse2)
    # Store R² scores
    r2_scores_model1.append(r2_model1)
    r2_scores_model2.append(r2_model2)
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



    # Compare R² scores
    if abs(r2_model1 - r2_model2) < 0.00005:
        similar_performance_count += 1
        print(f"Dataset {number_of_datasets}: Models have similar performance.")
    elif r2_model1 > r2_model2:
        model1_better_count += 1
    else:
        model2_better_count += 1
        print(f"Dataset {number_of_datasets}: vsgp performs better.")

# Print the number of times each model performed better
print(f'Model 1  performed better {model1_better_count} times.')
print(f'Model 2  performed better {model2_better_count} times.')
print(f'Models had similar performance {similar_performance_count} times.')
print(f'Model 1 failed {fail_model1_count} times.')
print(f'Model 2 failed {fail_model2_count} times.')

# Create an array for the x-axis
x = np.arange(1, len(r2_scores_model1) + 1)

# Set the width of the bars
width = 0.35

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the R² scores as bars
rects1 = ax.bar(x - width / 2, r2_scores_model1, width, label='without inputWarp')
rects2 = ax.bar(x + width / 2, r2_scores_model2, width, label='with inputWarp')

# Add labels, title and legend
ax.set_xlabel('Dataset Number', fontsize=18)
ax.set_ylabel('R² Score', fontsize=18)
ax.set_title('R² Scores of autoGP with vs. without input warp across warped datasets.', fontsize=20)
ax.legend(fontsize=16)

# Optional: Increase tick label size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Set the y-axis limits
plt.ylim(-0.05, 1.25)
plt.xlim(0,25)
# Set the y-axis ticks to show only from 0 to 1.0
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Save the plot as an image file
#plt.savefig('Model_comparison_warped.png')

# Show the plot
plt.show()
