import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

# Define the model
# Define the ANN model

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim=1, output_dim=1):
        torch.manual_seed(42)
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

# Define the main class
class ICICIPredictor():
    def __init__(self, ma_windows=None, lags=None, date=None, print_=True,data=None):
        self.print_ = print_
        self.date = date or datetime.today().date()
        self.ma_windows = ma_windows or [44, 29, 11, 3]
        self.lags = lags or [1,2,3]
        self.scaler = StandardScaler()
        self.data = data if data is not None else self.data_get()
        self.y, self.x, self.x_final = None, None, None
        self.model = None
        self.loss = None
        self.pred=None
    def data_get(self):
        if isinstance(self.date, str):
            self.date = datetime.strptime(self.date, '%Y-%m-%d')
        if self.print_:
            print('Downloading data...')
        data = yf.download('ICICIBANK.NS', start=self.date - timedelta(days=365), end=self.date,progress=True)
        if self.print_:
           print('Data downloaded')
        data.columns = ['close','High','Low','Open','Volume']
        data = data[['close']]
        return data

    def data_preprocessing(self):
        # Generate moving averages
        data = self.data.copy()
        for window in self.ma_windows:
            data[f'ma{window}'] = data['close'].rolling(window=window).mean()
        
        # Generate lag features
        for lag in self.lags:
            data[f'F{lag}'] = (data['close'].shift(lag-1) - data['close'].shift(lag))

        data['close'] = data['close'].shift(-1)
        x_final = data.iloc[[-1]].drop('close',axis=1)  # Feature vector for prediction
        data.dropna(inplace=True)
        y = data['close']
        x = data.drop(['close'], axis=1)
        return y, x, x_final

    def fit(self):
        self.y, self.x, self.x_final = self.data_preprocessing()
        self.model = ANN(input_dim=self.x.shape[1], hidden_dim=1).double()
        if self.print_:
            print('Model training in process...')
        X = self.x.values
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
        y_tensor = torch.tensor(self.y.values, dtype=torch.float64).view(-1, 1)
        learning_rate = 0.2
        epochs = 500
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.model.train()
            
            # Forward pass
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.print_:
                if (epoch+1) % 50 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        self.loss = loss.item()
        if self.print_:
            print('model has trained')

    def predict_next_day(self):
        X = self.x_final
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
        with torch.no_grad():
            self.pred = self.model.forward(X_tensor)
            self.pred=round(self.pred.numpy()[0][0], 2)
        return self.pred
    def save_pred(self):
        pred=pd.read_csv(r'D:\Etharia quants\ICICI bank\ICICI\ICICI_PRED.csv')
        pred.loc[len(pred)]=[self.date.strftime('%Y-%m-%d'),self.pred,None]
        pred.loc[len(pred)-2,'close']=round(self.data.iloc[-1]['close'],2)
        pred.to_csv(r'D:\Etharia quants\ICICI bank\ICICI\ICICI_PRED.csv',index=False)

