import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

FRAMEWORK_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    # 模型/滚动预测相关
    "seq_len": 60,  # 输入序列长度
    "pred_len": 1,  # 单次预测步长=1
    "predict_steps": 800,  # 总滚动预测步数
    "enc_in": 3,  # 输入通道数（特征数）
    "task_name": "short_term_forecast",

    "d_model": 64,
    "embed": "timeF",
    "freq": "h",
    "dropout": 0.1,
    "factor": 3,
    "n_heads": 4,
    "d_ff": 32,
    "activation": "gelu",
    "e_layers": 1,

    "look_back": 60,
    "train_ratio": 0.8,
}


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    适配单次预测1步的修改版
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs["task_name"]
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]  # 单次预测1步，pred_len=1

        self.enc_embedding = DataEmbedding_inverted(
            configs["seq_len"], configs["d_model"], configs["embed"],
            configs["freq"], configs["dropout"]
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs["factor"], attention_dropout=configs["dropout"],
                                      output_attention=False), configs["d_model"], configs["n_heads"]),
                    configs["d_model"],
                    configs["d_ff"],
                    dropout=configs["dropout"],
                    activation=configs["activation"]
                ) for l in range(configs["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(configs["d_model"])
        )
        # Decoder（适配单次预测1步，统一projection输出维度为1）
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs["d_model"], 1, bias=True)  # 强制输出1步
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs["d_model"], configs["seq_len"], bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs["d_model"], configs["seq_len"], bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs["dropout"])
            self.projection = nn.Linear(configs["d_model"] * configs["enc_in"], configs["num_class"])

    def forecast(self, x_enc):
        # 简化接口：移除x_mark_enc/x_dec/x_mark_dec（仅保留x_enc，对齐之前单次预测1步的接口）
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding（移除x_mark_enc，适配简化接口）
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 适配单次预测1步：projection输出1步，permute后维度为[B,1,N]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer（适配1步输出）
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, 1, 1))  # 1步，repeat(1,1,1)
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, 1, 1))   # 1步，repeat(1,1,1)
        return dec_out

    def forward(self, x_enc, mask=None):
        # 简化forward接口：仅保留x_enc和可选的mask，移除x_mark_enc/x_dec/x_mark_dec
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            # 适配单次预测1步：squeeze(1)去掉pred_len维度，输出[B, enc_in]
            return dec_out[:, -1:, :].squeeze(1)  # [B, N] （N=enc_in）
        return None

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


        # min_val = np.min(y_true_real[:, dim])
        # max_val = np.max(y_true_real[:, dim])
        # range_val = max_val - min_val
        # ax.set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)
        #
        # # 子图样式设置
        # ax.grid(alpha=0.3, linestyle='--')
        # ax.set_xlim(0, predict_steps - 1)
        # # 只在第一个子图显示图例
        # if dim == 0:
        #     ax.legend(fontsize=9, loc='upper right')
        # # 只在最后一个子图显示x轴标签
        # if dim == feature_dim - 1:
        #     ax.set_xlabel("Time Steps", fontsize=11)


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

    return

def generate_data():
    """数据生成模块（可替换为其他数据集/系统）"""
    from dysts.flows import Lorenz, Rossler, \
        Chen, DoublePendulum, Chua, HyperLorenz, RabinovichFabrikant

    model = Chua()
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
