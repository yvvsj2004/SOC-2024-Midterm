# 0) Prepare the data
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# preparing the data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# model
# y = wx+b, apply a sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)

# loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimzer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
n_epochs = 500
for epoch in range(n_epochs):
    # forward pass
    y_pred = model(X_train)
    # backward pass
    loss = criterion(y_pred, y_train)
    loss.backward()
    # calculating gradients and updating
    optimzer.step()
    optimzer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_pred_class = y_predicted.round()
    accuracy = (y_pred_class.eq(y_test)).sum()/float(y_test.shape[0])
    print(f"Accuracy = {accuracy:.4f}")