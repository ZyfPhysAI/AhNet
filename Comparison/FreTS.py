import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===================== 1. 配置参数 =====================
FRAMEWORK_CONFIG = {
    # 训练相关
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    # 模型/滚动预测相关
    "seq_len": 60,  # 输入序列长度
    "pred_len": 1,  # 单次预测步长=1
    "predict_steps": 800,  # 总滚动预测步数
    "enc_in": 3,  # 输入通道数（特征数）
    "task_name": "long_term_forecast",
    "channel_independence": "0",

    "look_back": 60,
    "train_ratio": 0.8,
}


# ===================== 2. 模型定义 =====================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs["task_name"]
        self.pred_len = configs["pred_len"]  # 单次预测1步
        self.embed_size = 128
        self.hidden_size = 256
        self.feature_size = configs["enc_in"]
        self.seq_len = configs["seq_len"]
        self.channel_independence = configs["channel_independence"]
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        # 输出层适配单次预测1步
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)  # 输出1步
        )

    def tokenEmb(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    def MLP_temporal(self, x, B, N, L):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    def MLP_channel(self, x, B, N, L):
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        return x

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forecast(self, x_enc):
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)
        bias = x
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :].squeeze(1)  # [B, enc_in]
        else:
            raise ValueError('Only forecast tasks implemented yet')

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, train_loader, val_loader, config=FRAMEWORK_CONFIG):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    print("\n===== Start Training =====")
    for epoch in range(config["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(config["device"]), batch_y.to(config["device"])
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss_avg = train_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(config["device"]), batch_y.to(config["device"])
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
        val_loss_avg = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{config['epochs']}] | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")

        # 早停逻辑
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop triggered (no improvement for {patience} epochs)")
                break

    print("===== Training Finished =====")
    return model


def rolling_predict(model, scaler, data_scaled, feature_dim, config=FRAMEWORK_CONFIG):
    model.eval()
    predict_steps = config["predict_steps"]
    look_back = config["look_back"]

    # 选测试集中最后一个可预测样本
    start_idx = len(data_scaled) - predict_steps
    input_start_idx = start_idx - look_back
    current_input = data_scaled[input_start_idx:input_start_idx + look_back, :].reshape(1, look_back, feature_dim)

    # 存储预测结果和真实值
    y_pred_scaled = []
    y_true_scaled = data_scaled[start_idx : start_idx + predict_steps, :]

    # 滚动预测循环
    with torch.no_grad():
        for _ in range(predict_steps):
            input_tensor = torch.tensor(current_input, dtype=torch.float32).to(config["device"])
            step_pred = model(input_tensor)
            step_pred_np = step_pred.cpu().numpy()
            y_pred_scaled.append(step_pred_np[0])

            # 更新输入序列（滑动窗口）
            current_input = np.concatenate([
                current_input[:, 1:, :],
                step_pred_np.reshape(1, 1, feature_dim)
            ], axis=1)

    # 转换为数组并反归一化
    y_pred_scaled = np.array(y_pred_scaled)
    y_true_scaled = np.array(y_true_scaled)
    y_pred_real = scaler.inverse_transform(y_pred_scaled)
    y_true_real = scaler.inverse_transform(y_true_scaled)

    # 计算评估指标
    dim_mae = []
    dim_rmse = []

    for dim in range(feature_dim):
        # 计算每个维度的MAE和RMSE
        mae_dim = mean_absolute_error(y_true_real[:, dim], y_pred_real[:, dim])
        rmse_dim = np.sqrt(mean_squared_error(y_true_real[:, dim], y_pred_real[:, dim]))
        dim_mae.append(mae_dim)
        dim_rmse.append(rmse_dim)

    overall_mae = mean_absolute_error(y_true_real, y_pred_real)  # 所有样本+维度合并计算
    overall_rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))

    # 打印结果
    print("\n===== Evaluation Metrics (Per Dimension) =====")
    for dim in range(feature_dim):
        print(f"Dimension {dim + 1} | MAE: {dim_mae[dim]:.4f} | RMSE: {dim_rmse[dim]:.4f}")
    print(f"All Samples+Dims | MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f}")
    print("==============================================")

    # 绘图
    fig, axes = plt.subplots(nrows=feature_dim, ncols=1, figsize=(8, 2.5 * feature_dim))

    time_steps = np.arange(predict_steps)
    for dim, ax in enumerate(axes):
        ax.plot(time_steps, y_true_real[:, dim], label='Ground-truth' if dim == 0 else "",
                color='dimgray', linewidth=2, alpha=0.8)
        ax.plot(time_steps, y_pred_real[:, dim], label='Prediction' if dim == 0 else "",
                color='#FF6B6B', linewidth=2, alpha=0.9)

 #       min_val = np.min(y_true_real[:, dim])
 #       max_val = np.max(y_true_real[:, dim])
 #       range_val = max_val - min_val
 #       ax.set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)

        # 子图样式设置
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, predict_steps - 1)
        # 只在第一个子图显示图例
        if dim == 0:
            ax.legend(fontsize=9, loc='upper right')
        # 只在最后一个子图显示x轴标签
        if dim == feature_dim - 1:
            ax.set_xlabel("Time Steps", fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    return

def generate_data():
    """数据生成模块（可替换为其他数据集/系统）"""
    from dysts.flows import Lorenz, Rossler, \
        Chen, DoublePendulum, Chua, HyperLorenz, RabinovichFabrikant

    model = Rossler()
    sol = model.make_trajectory(5000)
    feature_dim = sol.shape[1]
    data = sol
    return data, feature_dim


def preprocess_data(config=FRAMEWORK_CONFIG):
    data, feature_dim = generate_data()

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(config["look_back"], len(data_scaled)):
        X.append(data_scaled[i - config["look_back"]:i, :])
        y.append(data_scaled[i, :])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * config["train_ratio"])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"===== Preliminary Info =====")
    print(f"Feature Dim：{feature_dim}")
    print(f"TrainSet：X={X_train.shape}, y={y_train.shape}")
    print(f"TestSet：X={X_test.shape}, y={y_test.shape}")
    print(f"Device：{config['device']}")
    print("============================")
    return X_train, X_test, y_train, y_test, scaler, data_scaled, feature_dim



if __name__ == "__main__":

    X_train, X_test, y_train, y_test, scaler, data_scaled, feature_dim = preprocess_data(FRAMEWORK_CONFIG)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=FRAMEWORK_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=FRAMEWORK_CONFIG["batch_size"], shuffle=False)

    model = Model(FRAMEWORK_CONFIG).to(FRAMEWORK_CONFIG["device"])
    model = train_model(model, train_loader, val_loader, FRAMEWORK_CONFIG)
    rolling_predict(model, scaler, data_scaled, feature_dim, FRAMEWORK_CONFIG)
