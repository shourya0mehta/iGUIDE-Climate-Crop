import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLS = ['avg_temp', 'total_precip', 'avg_humidity', 'avg_radiation', 'avg_vpd']
TARGET_COL = 'yield_bu_acre'
REGION_COL = 'region'


class MLP(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return round(r2, 3), round(rmse, 2)


def run_ridge_loco(df):
    results = []
    for held_out in df[REGION_COL].unique():
        train = df[df[REGION_COL] != held_out]
        test = df[df[REGION_COL] == held_out]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS].values)
        X_test = scaler.transform(test[FEATURE_COLS].values)
        model = Ridge(alpha=1.0)
        model.fit(X_train, train[TARGET_COL].values)
        y_pred = model.predict(X_test)
        r2, rmse = evaluate(test[TARGET_COL].values, y_pred)
        results.append({'held_out_region': held_out, 'R2': r2, 'RMSE': rmse})
        print(f"Ridge | {held_out}: R²={r2}, RMSE={rmse}")
    return pd.DataFrame(results)


def run_mlp_loco(df, epochs=100, lr=1e-3):
    results = []
    for held_out in df[REGION_COL].unique():
        train = df[df[REGION_COL] != held_out]
        test = df[df[REGION_COL] == held_out]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS].values).astype(np.float32)
        X_test = scaler.transform(test[FEATURE_COLS].values).astype(np.float32)
        y_train = train[TARGET_COL].values.astype(np.float32)
        y_test = test[TARGET_COL].values.astype(np.float32)
        loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test)).numpy()
        r2, rmse = evaluate(y_test, y_pred)
        results.append({'held_out_region': held_out, 'R2': r2, 'RMSE': rmse})
        print(f"MLP   | {held_out}: R²={r2}, RMSE={rmse}")
    return pd.DataFrame(results)


if __name__ == '__main__':
    df = pd.read_csv('./master_dataset.csv')
    print("=== Ridge Regression LOCO ===")
    ridge_results = run_ridge_loco(df)
    ridge_results.to_csv('./results/ridge_loco_results.csv', index=False)
    print("\n=== MLP LOCO ===")
    mlp_results = run_mlp_loco(df)
    mlp_results.to_csv('./results/mlp_loco_results.csv', index=False)