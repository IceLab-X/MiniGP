# A Python script that tests the accuracy and training speed on different sizes of training sets. The results are stored in result1.csv and result2.csv.
import os
import time
import torch
import torch.optim as optim
import csv
from matplotlib import pyplot as plt
import data_sample.generate_example_data as data
from core.parametricGP import parametricGP
from core.svgp import svgp
from core.cigp import cigp
from core.sgpr import vsgp
from torch.utils.data import DataLoader, TensorDataset
import core.GP_CommonCalculation as GP
from core.kernel import ARDKernel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
torch.manual_seed(4)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate data
training_sizes = [10, 100, 1000, 2000, 3000]  # Included 10 for calculation
learning_rate = 0.1
num_epochs = [200, 200, 250, 250]

# num_inducings =[50,100,150,200,250,300]
all_results = []
for training_size in training_sizes:
    xtr, ytr, xte, yte = data.generate(training_size, 500, seed=2)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xte = xte.to(device)
    yte = yte.to(device)
    normalizer = GP.DataNormalization(method='standard')
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'x')
    # Dynamically adjust num_inducing and batchsize based on training size
    num_inducing = training_size // 40
    num_inducing_for_vsgp = training_size // 10
    batchsize = training_size // 4
    # for num_inducing in num_inducings:
    #     # Dynamically adjust num_inducing and batchsize based on training size
    #     batchsize = num_inducing*2

    dataset = TensorDataset(xtr_normalized, ytr_normalized)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    models = {
        "CIGP": cigp().to(device),
        "VSGP": vsgp(kernel=ARDKernel(1), num_inducing=num_inducing_for_vsgp, input_dim=xtr_normalized.size(1)).to(
            device),
        "SVIGP": svgp(num_inducing=num_inducing, input_dim=xtr_normalized.size(1), num_data=training_size).to(device),
        "ParametricGP": parametricGP(kernel=ARDKernel(1), input_dim=1, num_inducing=num_inducing, device=device)
    }

    results = {}
    num_epochs_index = -1
    for model_name, model in models.items():
        num_epochs_index += 1
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        mse_values = []

        start_time = time.time()
        if isinstance(num_epochs, list):
            num_epochs_current = num_epochs[
                num_epochs_index]  # num_epochs2 is the number of epochs for the current model
        else:
            num_epochs_current = num_epochs
        for i in range(num_epochs_current):

            optimizer.zero_grad()
            if model_name in ["ParametricGP", "SVIGP"]:
                for X_batch, Y_batch in dataloader:
                    loss = model.loss_function(X_batch, Y_batch)
            elif model_name in ["VSGP", "CIGP"]:
                loss = model.negative_log_likelihood(xtr_normalized, ytr_normalized)

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'{model_name} - Training Size: {training_size} - Epoch: {i} - Loss: {loss.item()}')
        end_time = time.time()
        if model_name in ["ParametricGP", "SVIGP"]:
            output = model.forward(xte_normalized)
        elif model_name in ["VSGP", "CIGP"]:
            output = model.forward(xtr_normalized, ytr_normalized, xte_normalized)
        mse = torch.mean((yte - output[0]).pow(2))
        mse_values.append(mse.item())
        iteration_time = (end_time - start_time) * 1000  # in milliseconds
        average_iteration_time = iteration_time / num_epochs_current
        results[model_name] = {
            "mse_values": mse_values,
            "average_iteration_time": average_iteration_time
        }
        print(
            f'{model_name} - Training Size: {training_size} -Number of Inducing point:{num_inducing}- Average iteration time: {average_iteration_time:.5f} ms')

    # Only store results if training_size is not 50
    if training_size != 10:
        for model_name, result in results.items():
            for iteration, mse in enumerate(result["mse_values"]):
                all_results.append({
                    'Model': model_name,
                    'Training Size': training_size,
                    'Batch Size': batchsize,
                    'Number of Inducing Points': num_inducing,
                    'Iteration': iteration,
                    'Training Time per Iteration (MilliSeconds)': result["average_iteration_time"],
                    'MSE': mse
                })

# Save all results to CSV
from datetime import datetime
import pandas as pd
import numpy as np

# Generate a unique filename with a timestamp, this will save all results to a new CSV file without overwriting the previous ones
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'results_{timestamp}.csv'

# Save all results to CSV
filename = '../asset/results1.csv'
with open(filename, 'w', newline='') as csvfile:
    fieldnames = ['Model', 'Training Size', 'Batch Size', 'Number of Inducing Points', 'Iteration',
                  'Training Time per Iteration (MilliSeconds)', 'MSE']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in all_results:
        writer.writerow(result)

# Plotting the results
# Read the CSV file
df = pd.read_csv(filename)
markers = {'CIGP': 'o', 'VSGP': '^', 'SVIGP': 's', 'ParametricGP': 'd'}

plt.figure(figsize=(10, 6))
for model_name in df['Model'].unique():
    model_df = df[df['Model'] == model_name]
    plt.plot(model_df['Training Size'], model_df['MSE'], marker=markers[model_name], linewidth=2.5,
             label=f'{model_name}')

plt.xlabel('Training Size', fontsize=18)
plt.xticks(sorted(df['Training Size'].unique()), fontsize=16)
plt.ylim(0, 0.2)
plt.yticks([i * 0.02 for i in range(11)], fontsize=16)
plt.ylabel('MSE', fontsize=18)
plt.title('Model Comparison -- Accuracy', fontsize=20)
plt.legend(fontsize=16)
plt.savefig('Model_comparison Accuracy.PNG')
plt.show()

# Plot 2: Training Time per Iteration vs. Training Size
plt.figure(figsize=(10, 6))
for model_name in df['Model'].unique():
    model_df = df[df['Model'] == model_name]
    Training_Time = model_df['Training Time per Iteration (MilliSeconds)']
    log_Training_Time = np.log(Training_Time)
    plt.plot(model_df['Training Size'], log_Training_Time, marker=markers[model_name], linewidth=2.5,
             label=f'{model_name}')

plt.xlabel('Training Size', fontsize=18)
plt.xticks(sorted(df['Training Size'].unique()), fontsize=16)
plt.ylabel('Average Training Time per Iteration (log, ms)', fontsize=18)
plt.title('Model Comparison -- Speed', fontsize=20)
plt.legend(fontsize=16)
plt.show()
