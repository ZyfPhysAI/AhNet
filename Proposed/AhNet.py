import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# ====================== Config ======================
CFG = {
    "look_back": 60,
    "predict_steps": 800,
    "train_ratio": 0.8,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_size_1": 128,
    "hidden_size_2": 64,
    "temp_coeff": 15.0
}


# ====================== Tools ======================
def compute_momentum(x):
    """(p = dq/dt)"""
    momentum = x[:, 1:, :] - x[:, :-1, :]
    return torch.cat([torch.zeros((x.shape[0], 1, x.shape[2])).to(CFG["device"]), momentum], dim=1)

def compute_2nd_momentum(x):
    """(dp/dt = d²q/dt²)"""
    p1 = compute_momentum(x)
    p2 = compute_momentum(p1)
    return p1, p2

def build_symplectic_matrix(d):
    """J (2d×2d)"""
    I = torch.eye(d, device=CFG["device"])
    O = torch.zeros_like(I)
    return torch.cat([torch.cat([O, I], 1), torch.cat([-I, O], 1)], 0)

# ====================== 3. AhNet Model ======================
class HamiltonianMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, 1))

    def forward(self, x):
        return self.mlp(x)

class HNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, CFG["hidden_size_1"]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(CFG["hidden_size_1"], CFG["hidden_size_2"]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(CFG["hidden_size_2"], dim)
        )

    def forward(self, x):
        return self.fc(x)

class AhNet_MLP(nn.Module):
    def __init__(self, feat_dim, look_back):
        super().__init__()
        self.d = feat_dim
        self.look_back = look_back
        self.J = build_symplectic_matrix(self.d)

        self.q_mlp = nn.Sequential(
            nn.Linear(look_back * feat_dim, CFG["hidden_size_1"]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(CFG["hidden_size_1"], CFG["hidden_size_2"])
        )

        self.hamiltonian_mlp = HamiltonianMLP(2 * CFG["hidden_size_2"])
        self.fusion = nn.Linear(2 * CFG["hidden_size_2"], 2 * self.d)
        self.hnn = HNN(2 * self.d)
        self.predictor = nn.Sequential(
            nn.Linear(4 * self.d, CFG["hidden_size_2"]),
            nn.ReLU(),
            nn.Linear(CFG["hidden_size_2"], self.d)
        )

    def forward(self, q):
        p = compute_momentum(q)
        batch_size = q.shape[0]
        q_flat = q.reshape(batch_size, -1)
        p_flat = p.reshape(batch_size, -1)
        C_q = self.q_mlp(q_flat)
        C_p = self.q_mlp(p_flat)
        H = self.hamiltonian_mlp(torch.cat([C_q, C_p], dim=1))
        x = self.fusion(torch.cat([C_q, C_p], dim=1))
        grad_H = self.hnn(x)
        x_dot = torch.matmul(grad_H, self.J.T)
        pred = self.predictor(torch.cat([x, x_dot], dim=1))
        return pred, x, grad_H, x_dot, H

# ====================== 4. Loss ======================
def ahnet_loss(model, q, y_true, curr_H, prev_H):
    p1, p2 = compute_2nd_momentum(q)
    pred, _, _, x_dot_pred, _ = model(q)

    loss_data = nn.MSELoss()(pred, y_true)
    dot_q = p1[:, -1, :]
    dot_p = p2[:, -1, :]
    x_dot_true = torch.cat([dot_q, dot_p], 1)
    loss_phy = nn.MSELoss()(x_dot_pred, x_dot_true)

    if prev_H is None:
        delta_H = torch.zeros(curr_H.shape[0], device=CFG["device"])
    else:
        min_bs = min(curr_H.shape[0], prev_H.shape[0])

        delta_H = curr_H[:min_bs].squeeze() - prev_H[:min_bs].squeeze()


    alpha_phy = torch.exp(-torch.abs(delta_H.detach()) * CFG["temp_coeff"])
    alpha_phy = torch.clamp(alpha_phy, 0.01, 0.99)
    alpha_data = 1.0 - alpha_phy

    total_loss = (alpha_phy * loss_phy + alpha_data * loss_data).mean()
    return total_loss, loss_data, loss_phy, alpha_phy.mean(), alpha_data.mean(), delta_H.abs().mean()

# ====================== 5. Data ======================
class TSDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def get_data():
    from dysts.flows import Lorenz, Rossler, \
        Chen, DoublePendulum, Chua, HyperLorenz, RabinovichFabrikant

    model = Chua()
    data = model.make_trajectory(5000)
    feat_dim = data.shape[1]
    scaler = MinMaxScaler((0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(CFG["look_back"], len(data_scaled)):
        X.append(data_scaled[i - CFG["look_back"]:i, :])
        y.append(data_scaled[i, :])
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * CFG["train_ratio"])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"===== Preliminary Info =====")
    print(f"Feature Dim：{feat_dim}")
    print(f"TrainSet：X={X_train.shape}, y={y_train.shape}")
    print(f"TestSet：X={X_test.shape}, y={y_test.shape}")
    print("============================")
    return X_train, X_test, y_train, y_test, scaler, data_scaled, feat_dim

# ====================== 6. Inference ======================
def predict(model, X_test, scaler, data_scaled, feat_dim):
    """Prediction"""
    model.eval()
    look_back = CFG["look_back"]
    start_idx = len(data_scaled) - CFG["predict_steps"]
    input_window = data_scaled[start_idx - look_back:start_idx, :].reshape(1, look_back, feat_dim)

    y_pred = []
    with torch.no_grad():
        for _ in range(CFG["predict_steps"]):
            x_tensor = torch.tensor(input_window, dtype=torch.float32).to(CFG["device"])
            pred, _, _, _, _ = model(x_tensor)
            pred_np = pred.cpu().numpy()[0]
            y_pred.append(pred_np)
            input_window = np.concatenate([input_window[:, 1:, :], pred_np.reshape(1, 1, feat_dim)], axis=1)

    y_pred = scaler.inverse_transform(np.array(y_pred))
    y_true = scaler.inverse_transform(data_scaled[start_idx:, :])


    print("\n===== Evaluation Metrics (Per Dimension) =====")
    for dim in range(feat_dim):
        mae = mean_absolute_error(y_true[:, dim], y_pred[:, dim])
        rmse = np.sqrt(mean_squared_error(y_true[:, dim], y_pred[:, dim]))
        print(f"Dimension {dim + 1} | MAE：{mae:.4f} | RMSE：{rmse:.4f}")
    print("==============================================")


    fig, axes = plt.subplots(nrows=feat_dim, ncols=1, figsize=(8, 2.5 * feat_dim))

    time_steps = np.arange(CFG["predict_steps"])
    for dim, ax in enumerate(axes):
        ax.plot(time_steps, y_true[:, dim], label='Ground-truth' if dim == 0 else "",
                color='dimgray', linewidth=2, alpha=0.8)
        ax.plot(time_steps, y_pred[:, dim], label='Prediction' if dim == 0 else "",
                color='#FF6B6B', linewidth=2, alpha=0.9)

        #        min_val = np.min(y_true_real[:, dim])
        #        max_val = np.max(y_true_real[:, dim])
        #        range_val = max_val - min_val
        #        ax.set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)

        # 子图样式设置
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, CFG["predict_steps"] - 1)

        if dim == 0:
            ax.legend(fontsize=9, loc='upper right')
        # 只在最后一个子图显示x轴标签
        if dim == feat_dim - 1:
            ax.set_xlabel("Time Steps", fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, data_scaled, feat_dim = get_data()
    train_loader = DataLoader(TSDataSet(X_train, y_train), batch_size=CFG["batch_size"], shuffle=False)
    val_loader = DataLoader(TSDataSet(X_test, y_test), batch_size=CFG["batch_size"], shuffle=False)

    model = AhNet_MLP(feat_dim, CFG["look_back"]).to(CFG["device"])
    predict(model, X_test, scaler, data_scaled, feat_dim)