
#%% packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv('heart.csv')
df.head()

#%% separate independent / dependent features
X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

#%% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#%% network class
class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w = np.random.random(X_train.shape[1])
        self.b = np.random.randn()
        self.LR      = LR
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        self.L_train = []
        self.L_test  = []

    def activation(self, x):
        # Sigmoid
        return 1/(1+np.exp(-x))
    
    def derivative_activation(self,x):
        # Derivative of Sigmoid
        sigmoid = self.activation(x)
        return sigmoid * (1 - sigmoid)
    
    def forward(self, X):
        hidden_1 = np.dot(X, self.w) + self.b
        activate_1 = self.activation(hidden_1)
        return activate_1
    
    def backward(self,X, y_true):
        # Calculate Gradients
        hidden_1    = np.dot(X, self.w) + self.b
        y_pred      = self.forward(X)
        dL_dpred    = 2*(y_pred - y_true)
        d_pred_dH1  = self.derivative_activation(hidden_1)
        d_H1_b      = 1
        d_H1_dw     = X

        dL_db       = dL_dpred * d_pred_dH1 * d_H1_b
        dL_dw       = dL_dpred * d_pred_dH1 * d_H1_dw

        return dL_db, dL_dw
    
    def optimizer(self, dL_db, dL_dw):
        # Update the weights
        self.b = self.b - dL_db * self.LR
        self.w = self.w - dL_dw * self.LR

    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            # Random Position
            random_pos = np.random.randint(len(self.X_train))

            # Foward Pass
            y_train_true = self.y_train
            y_train_pred = self.forward(X_train[random_pos])

            # Calculate the Losses

            L = np.square(y_train_true - y_train_pred)
            self.L_train.append(L)

            # Calculate the Gradients
            dL_db, dL_dw = self.backward(self.X_train[random_pos]
                                         , self.y_train[random_pos])

            # Update Weights 
            self.optimizer(dL_db, dL_dw)

            # Calculate the error for test Data 
            L_sum = 0 
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])

                L_sum += np.square(y_true-y_pred)
            self.L_test.append(L_sum)
        
        return 'Training Succesful'

#%% Hyper parameters
LR = 0.01
ITERATIONS = 10000

#%% model instance and training
nn = NeuralNetworkFromScratch(LR=LR, X_train=X_train_scale, y_train=y_train,X_test=X_test_scale, y_test=y_test)
nn.train(ITERATIONS=ITERATIONS)

# %% check losses
results = pd.DataFrame({'x' : np.array(range(len(nn.L_test))),
                         'y':np.array(nn.L_test)})

plt.plot(np.array(results['x']), np.array(results['y']))


 # %% iterate over test data
total = X_test_scale.shape[0]
correct = 0
y_preds = []

for i in range(total):
    y_true = y_test[i] 
    y_pred = np.round(nn.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true==y_pred else 0


# %% Calculate Accuracy
correct / total
# %% Baseline Classifier
from collections import Counter
Counter(y_test)
31/61

# %% Confusion Matrix
confusion_matrix(y_true=y_test,y_pred=y_preds)
# %%
