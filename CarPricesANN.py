import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class CurveData(Dataset):
    def __init__(self):
        df = pd.read_csv("car_price_dataset.csv")  # Load dataset
        df = df.dropna().copy()
        df = pd.get_dummies(df, columns=["Brand", "Model", "Fuel_Type", "Transmission"], drop_first=True)
        df[df.select_dtypes("bool").columns] = df[df.select_dtypes("bool").columns].astype(float)

        y = df.iloc[:,-1].to_numpy()
        self.y = torch.tensor(y, dtype=torch.float32)

        self.X = torch.tensor(df.iloc[:, :-1].copy().to_numpy(), dtype=torch.float)
        self.X -= torch.mean(self.X, dim=0)
        self.X /= torch.std(self.X, dim=0)

        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X), np.array(self.y)


class CurveFit(nn.Module):
    def __init__(self):
        super(CurveFit, self).__init__()

        self.in_to_h1 = nn.Linear(48, 64)  # First hidden layer
        self.h1_to_h2 = nn.Linear(64, 32)  # Second hidden layer
        self.h2_to_out = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        x = self.h2_to_out(x)
        return x


def trainNN(epochs=100, batch_size=16, lr=0.001, epoch_display=25):
    cd = CurveData()
    curve_loader = DataLoader(cd, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create my neural network
    curve_network = CurveFit()
    # Mean square error (RSS)
    mse_loss = nn.MSELoss(reduction='sum')
    # Select the optimizer
    optimizer = torch.optim.Adam(curve_network.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        for _, data in enumerate(curve_loader, 0):
            x, y = data
            optimizer.zero_grad() # resets gradients to zero
            output = curve_network(x)  # evaluate the neural network on x
            loss = mse_loss(output.view(-1), y)  # compare to the actual label value
            loss.backward()  # perform back propagation
            optimizer.step()  # perform gradient descent with an Adam optimizer
            running_loss += loss.item() # update the total loss

        if epoch % epoch_display == epoch_display - 1:
            print(f"Epoch {epoch + 1} / {epochs} Average loss: {running_loss / (len(cd) * epoch_display):.6f}")
            running_loss = 0.0
    return curve_network, cd


#curve network cn, and curve dataset cd
cn, cd = trainNN(epochs=100)

#done training, use neural network
with torch.no_grad():
    y_pred = cn(cd.X).view(-1)

X_numpy, y_numpy = cd.to_numpy()

print(f"MSE (fully trained): {np.average((y_numpy - np.array(y_pred)) ** 2)}")

#Find the unnormalized MSE
df = pd.read_csv("car_price_dataset.csv")
df = df.dropna().copy()
y = df.iloc[:,-1]
y_pred_original = y_pred * y.std() + y.mean()
# Compute MSE in the original scale
mse_original_scale = np.mean((y_numpy * y.std() + y.mean() - np.array(y_pred_original)) ** 2)
print(f"Unnormalized MSE: {mse_original_scale:.2f}")
