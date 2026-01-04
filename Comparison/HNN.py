import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ================= 配置 =================
CFG = {
    "look_back": 60,
    "predict_steps": 800,
    "train_ratio": 0.8,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-3,
    "hidden_size": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ================= HNN 模型 (保持不变，自动适配维度) =================
def build_symplectic_matrix(dim, device):
    split = dim // 2
    O = torch.zeros(split, split, device=device)
    I = torch.eye(split, device=device)
    top = torch.cat([O, I], dim=1)
    bot = torch.cat([-I, O], dim=1)
    J = torch.cat([top, bot], dim=0)
    return J


class HNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.h_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.J = build_symplectic_matrix(output_dim, CFG["device"])

    def forward(self, x):

        with torch.enable_grad():
            batch_size = x.shape[0]
            x_reshaped = x.reshape(batch_size, -1)

            if not x_reshaped.requires_grad:
                x_reshaped.requires_grad_(True)

            z = self.encoder(x_reshaped)

            H = self.h_net(z)
            grads = torch.autograd.grad(H, z, grad_outputs=torch.ones_like(H), create_graph=True)[0]

        return grads @ self.J.T


# ================= 工具函数 =================
class TSDataSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def create_loader(data, look_back, batch_size):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return DataLoader(TSDataSet(np.array(X), np.array(y)), batch_size=batch_size, shuffle=True)


def train_hnn(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience = 5
    counter = 0

    print("\n===== Start Training =====")
    for epoch in range(CFG["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(CFG["device"]), batch_y.to(CFG["device"])
            last_val = batch_x[:, -1, :]
            true_delta = batch_y - last_val
            optimizer.zero_grad()
            pred_delta = model(batch_x)
            loss = criterion(pred_delta, true_delta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        # Val
        model.eval()
        val_loss = 0.0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(CFG["device"]), batch_y.to(CFG["device"])
            last_val = batch_x[:, -1, :]
            true_delta = batch_y - last_val
            pred_delta = model(batch_x)
            val_loss += criterion(pred_delta, true_delta).item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{CFG['epochs']}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop at epoch {epoch + 1}")
                break
    print("===== Training Finished =====")
    return model


# ================= 主程序 =================
if __name__ == "__main__":

    from dysts.flows import Lorenz, Rossler, \
        Chen, DoublePendulum, Chua, HyperLorenz, RabinovichFabrikant

    model_sys = RabinovichFabrikant()
    raw_data = model_sys.make_trajectory(5000)

    original_dim = raw_data.shape[1]

    scaler = MinMaxScaler((-1, 1))
    data_pos_scaled = scaler.fit_transform(raw_data)

    data_diff = np.zeros_like(data_pos_scaled)
    data_diff[1:] = data_pos_scaled[1:] - data_pos_scaled[:-1]
    data_diff[0] = data_diff[1]  # 填充首位

    data_augmented = np.concatenate([data_pos_scaled, data_diff], axis=1)
    aug_dim = data_augmented.shape[1]

    # 划分训练/验证集
    train_len = int(len(data_augmented) * CFG["train_ratio"])
    train_data = data_augmented[:train_len]
    val_data = data_augmented[train_len:]

    train_loader = create_loader(train_data, CFG["look_back"], CFG["batch_size"])
    val_loader = create_loader(val_data, CFG["look_back"], CFG["batch_size"])

    print(f"===== Preliminary Info =====")
    print(f"Original Dim: {original_dim} | Augmented HNN Dim: {aug_dim}")
    print(f"Device: {CFG['device']}")
    print("============================")

    # 3. 初始化与训练
    # input_dim = look_back * aug_dim (展平)
    # output_dim = aug_dim (单步预测的所有状态)
    model = HNN_Model(CFG["look_back"] * aug_dim, aug_dim, CFG["hidden_size"]).to(CFG["device"])
    model = train_hnn(model, train_loader, val_loader)

    # 4. 滚动预测
    predict_steps = CFG["predict_steps"]
    start_idx = len(data_augmented) - predict_steps - CFG["look_back"]

    # 输入窗口包含位置和速度
    input_window = data_augmented[start_idx: start_idx + CFG["look_back"]].reshape(1, CFG["look_back"], aug_dim)

    preds_aug_scaled = []  # 存储拼接后的预测结果

    print("\nStarting Rolling Prediction...")
    with torch.no_grad():
        for _ in range(predict_steps):
            x_tensor = torch.tensor(input_window, dtype=torch.float32).to(CFG["device"])
            pred_delta = model(x_tensor).cpu().numpy()  # [1, aug_dim]

            last_val = input_window[:, -1, :]
            pred_next = last_val + pred_delta

            preds_aug_scaled.append(pred_next[0])
            input_window = np.concatenate([input_window[:, 1:, :], pred_next.reshape(1, 1, aug_dim)], axis=1)

    preds_aug_scaled = np.array(preds_aug_scaled)  # [Predict_Steps, 2*original_dim]

    # === 关键步骤：后处理 ===
    # 我们只关心前半部分（位置），用于反归一化和绘图
    preds_pos_scaled = preds_aug_scaled[:, :original_dim]
    truth_pos_scaled = data_pos_scaled[start_idx + CFG["look_back"]: start_idx + CFG["look_back"] + predict_steps]

    # 反归一化 (使用只fit过位置数据的scaler)
    preds_real = scaler.inverse_transform(preds_pos_scaled)
    truth_real = scaler.inverse_transform(truth_pos_scaled)

    # 5. 指标计算 (完全对齐 FreTS)
    dim_mae, dim_rmse = [], []
    for dim in range(original_dim):
        mae = mean_absolute_error(truth_real[:, dim], preds_real[:, dim])
        rmse = np.sqrt(mean_squared_error(truth_real[:, dim], preds_real[:, dim]))
        dim_mae.append(mae)
        dim_rmse.append(rmse)

    overall_mae = mean_absolute_error(truth_real, preds_real)
    overall_rmse = np.sqrt(mean_squared_error(truth_real, preds_real))

    print("\n===== Evaluation Metrics (Per Dimension) =====")
    for dim in range(original_dim):
        print(f"Dimension {dim + 1} | MAE: {dim_mae[dim]:.4f} | RMSE: {dim_rmse[dim]:.4f}")
    print(f"All Samples+Dims | MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f}")
    print("==============================================")

    # 6. 绘图 (完全对齐 FreTS)
    fig, axes = plt.subplots(nrows=original_dim, ncols=1, figsize=(8, 2.5 * original_dim))
    if original_dim == 1: axes = [axes]

    time_steps = np.arange(predict_steps)
    for dim, ax in enumerate(axes):
        ax.plot(time_steps, truth_real[:, dim], label='Ground-truth' if dim == 0 else "",
                color='dimgray', linewidth=2, alpha=0.8)
        ax.plot(time_steps, preds_real[:, dim], label='Prediction' if dim == 0 else "",
                color='#FF6B6B', linewidth=2, alpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, predict_steps - 1)
        if dim == 0: ax.legend(fontsize=9, loc='upper right')
        if dim == original_dim - 1: ax.set_xlabel("Time Steps", fontsize=11)

    # min_val = np.min(truth_real[:, dim])
    # max_val = np.max(truth_real[:, dim])
    # range_val = max_val - min_val
    # ax.set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()