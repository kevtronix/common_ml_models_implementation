'''
Linear Regression example using PyTorch
'''
import torch 
import torch.nn as nn
import torch.optim as optim 


# Sample Data 
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], requires_grad=False)

# Model 
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()


# Loss and optimizer 
criteron = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')