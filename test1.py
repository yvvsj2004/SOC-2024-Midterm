import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)

X_test = torch.tensor([5.0], dtype=torch.float32)

print(f"Prediction before training f(5) = {model(X_test).item():.3f}")

learn_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

for epoch in range(n_iters):
    y_pred = model(X)
    # print(y_pred)
    l = loss(Y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch%50 == 0:
        [w, b] = model.parameters()
        # print(w)
        print(f"round {epoch} w = {w.item():.3f} loss = {l:.8f}")

print(f"Prediction before training f(5) = {model(X_test).item():.3f}")
    