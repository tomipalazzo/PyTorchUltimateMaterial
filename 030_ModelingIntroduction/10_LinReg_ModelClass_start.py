#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns








#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch,self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def foward(self, x):
        out = self.linear(x)
        return out
    
INPUT_DIM = 1
OUTPUT_DIM = model = LinearRegressionTorch(INPUT_DIM, OUTPUT_DIM)

#%% Loss Funtion 
loss_fun = nn.MSELoss()

#%%  Optimizer
LR = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


#%%
losses, slope, bias = [], [], []

NUM_EPOCHS = 10000
for epoch in range(NUM_EPOCHS):
    # Set the gradients to zero
    optimizer.zero_grad()
    # Foward Pass
    y_pred = model.foward(X)
    # Compute loss
    loss = loss_fun(y_pred, y_true)
    # Backward Pass
    loss.backward()
    # Update Weights
    optimizer.step()
    # Get Parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])
    
    # store the losses
    losses.append(float(loss.data))

    # Print the losses
    if epoch % 100 == 0:
        print('Epoch: {}, Loss: {:4f}'.format(epoch, loss.data)) 

# %%
sns.scatterplot(x=range(NUM_EPOCHS), y=losses)
# %%
sns.scatterplot(x=range(NUM_EPOCHS), y=bias)

# %%
sns.scatterplot(x=range(NUM_EPOCHS), y=slope)

# %%
y_pred = model.foward(X).data.numpy().reshape(-1)
sns.scatterplot(x=X_list, y=y_list)
sns.scatterplot(x=X_list,y=y_pred,c='red')
# %%

