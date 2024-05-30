import sys
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("../src")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Working on device: {device}')

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

file_path = "../data/EDD_isoprenol_production.csv"
INPUT_VARS = ["ACCOAC", "MDH", "PTAr", "CS", "ACACT1r", "PPC", "PPCK", "PFL"]
RESPONSE_VARS = ["Value"]

# *** Read the data ***
df = pd.read_csv(file_path, index_col=0)
df = df[INPUT_VARS + RESPONSE_VARS]
df[INPUT_VARS] = df[INPUT_VARS].astype(int)
X_train = df[INPUT_VARS]
y_train = df[RESPONSE_VARS].values.ravel()
print(f"Shape of the data: {df.shape}")

class Network(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.encode1 = nn.Linear(input_shape, 500)
        self.encode2 = nn.Linear(500, 250)
        self.encode3 = nn.Linear(250, 50)
        
        self.decode1 = nn.Linear(50, 250)
        self.decode2 = nn.Linear(250, 500)
        self.decode3 = nn.Linear(500, input_shape)   
      
    def encode(self, x: torch.Tensor):
        x = F.relu(self.encode1(x))
        x = F.relu(self.encode2(x))
        x = F.relu(self.encode3(x))
        return x   
  
    def decode(self, x: torch.Tensor):
        x = F.relu(self.decode1(x))
        x = F.relu(self.decode2(x))
        x = F.relu(self.decode3(x))
        return x   
  
    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        x = self.decode(x)
        return x

net = Network(input_shape=df.shape[1])
optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
losses = []   

df = torch.tensor(df.values, dtype=torch.float32)

for epoch in range(1000):
    optimizer.zero_grad()
    output = net(df)
    loss = F.mse_loss(output, df)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save trained torch model
torch.save(net.state_dict(), "../models/autoencoder.pth")

# Load trained torch model
net = Network(input_shape=df.shape[1])
net.load_state_dict(torch.load("../models/autoencoder.pth"))