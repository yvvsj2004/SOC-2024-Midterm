# 1) Design model(input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#    - Forward pass
#    - Backward pass, Calculate gradients
#    - Weights update

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
# print(X)
y = y.view(y.shape[0],1)
# print("----------------------------------------")
# print(y)
n_samples, n_features = X.shape
# 1) MODEL
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
# 2) LOSS AND OPTIMIZER
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 3) TRAINING LOOP
num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X)
    l = criterion(y_pred,y)
    # backward pass
    l.backward()
    # weight update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%50 == 0:
        print(f"Epoch {epoch+1}: Loss = {l.item():8f}")

# PLOT
predicted = model(X).detach().numpy() # detach() is used to remove the required_grads=True part
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()