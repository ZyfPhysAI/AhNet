import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
    "c_out": 3,
    "task_name": "short_term_forecast",

    "d_model": 64,
    "d_ff": 32,
    "n_heads": 4,
    "e_layers": 1,

    "dropout": 0.1,
    "look_back": 60,
    "train_ratio": 0.8,
}


from layers.Autoformer_EncDec import series_decomp


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    适配单次预测1步：无修改（基础模块）
    """

    def __init__(self, top_k: int = 5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    适配单次预测1步：配置读取改为字典方式
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs["seq_len"] // (configs["down_sampling_window"] ** i),
                        configs["seq_len"] // (configs["down_sampling_window"] ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs["seq_len"] // (configs["down_sampling_window"] ** (i + 1)),
                        configs["seq_len"] // (configs["down_sampling_window"] ** (i + 1)),
                    ),

                )
                for i in range(configs["down_sampling_layers"])
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    适配单次预测1步：配置读取改为字典方式
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs["seq_len"] // (configs["down_sampling_window"] ** (i + 1)),
                        configs["seq_len"] // (configs["down_sampling_window"] ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs["seq_len"] // (configs["down_sampling_window"] ** i),
                        configs["seq_len"] // (configs["down_sampling_window"] ** i),
                    ),
                )
                for i in reversed(range(configs["down_sampling_layers"]))
            ])

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """
    适配单次预测1步：移除模型内归一化，配置读取改为字典方式
    """
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]  # 单次预测1步，pred_len=1
        self.down_sampling_window = configs["down_sampling_window"]

        self.layer_norm = nn.LayerNorm(configs["d_model"])
        self.dropout = nn.Dropout(configs["dropout"])
        self.channel_independence = configs["channel_independence"]

        if configs["decomp_method"] == 'moving_avg':
            self.decompsition = series_decomp(configs["moving_avg"])
        elif configs["decomp_method"] == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs["top_k"])
        else:
            raise ValueError('decompsition is error')

        if not configs["channel_independence"]:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs["d_model"], out_features=configs["d_ff"]),
                nn.GELU(),
                nn.Linear(in_features=configs["d_ff"], out_features=configs["d_model"]),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs["d_model"], out_features=configs["d_ff"]),
            nn.GELU(),
            nn.Linear(in_features=configs["d_ff"], out_features=configs["d_model"]),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


import torch
import torch.nn as nn
import math
from einops import rearrange

from layers.SelfAttention_Family import AttentionLayer, FullAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class Encoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            mha: AttentionLayer,
            d_hidden: int,
            dropout: float = 0,
            channel_wise=False,
    ):
        super(Encoder, self).__init__()

        self.channel_wise = channel_wise
        if self.channel_wise:
            self.conv = torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="reflect",
            )
        self.MHA = mha
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        q = residual
        if self.channel_wise:
            x_r = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
            k = x_r
            v = x_r
        else:
            k = residual
            v = residual
        x, score = self.MHA(q, k, v, attn_mask=None)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(residual)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs  # 改为字典读取
        self.task_name = configs["task_name"]
        self.seq_len = configs["seq_len"]
        self.pred_len = 1  # 强制单次预测1步
        self.d_channel = configs["enc_in"]
        self.N = configs["e_layers"]
        # Embedding
        self.d_model = configs["d_model"]
        self.d_hidden = configs["d_ff"]
        self.n_heads = configs["n_heads"]
        self.mask = True
        self.dropout = configs["dropout"]

        self.stride1 = 8
        self.patch_len1 = 8
        self.stride2 = 8
        self.patch_len2 = 16
        self.stride3 = 7
        self.patch_len3 = 24
        self.stride4 = 6
        self.patch_len4 = 32
        # 修复patch_num计算（避免0维）
        self.patch_num1 = max(int((self.seq_len - self.patch_len2) // self.stride2) + 1, 1)
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))

        self.shared_MHA = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.shared_MHA_ch = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.encoder_list = nn.ModuleList(
            [
                Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA[ll],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=False,
                )
                for ll in range(self.N)
            ]
        )

        self.encoder_list_ch = nn.ModuleList(
            [
                Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA_ch[0],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=True,
                )
                for ll in range(self.N)
            ]
        )

        # 位置编码（适配patch_num1）
        pe = torch.zeros(self.patch_num1, self.d_model)
        for pos in range(self.patch_num1):
            for i in range(0, self.d_model, 2):
                wavelength = 10000 ** ((2 * i) / self.d_model)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0)  # add a batch dimention to your pe matrix
        self.register_buffer("pe", pe)

        self.embedding_channel = nn.Conv1d(
            in_channels=self.d_model * self.patch_num1,
            out_channels=self.d_model,
            kernel_size=1,
        )

        self.embedding_patch_1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len1,
            stride=self.stride1,
        )
        self.embedding_patch_2 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len2,
            stride=self.stride2,
        )
        self.embedding_patch_3 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len3,
            stride=self.stride3,
        )
        self.embedding_patch_4 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len4,
            stride=self.stride4,
        )

        # ========== 核心修改：适配单次1步预测 ==========
        # 替换原8段式输出为直接输出1步，避免维度计算错误
        self.out_linear = torch.nn.Linear(self.d_model, self.pred_len)
        # 移除原8个out_linear层

        self.remap = torch.nn.Linear(self.d_model, self.seq_len)

    def forecast(self, x_enc):
        """
        简化接口：仅保留x_enc，移除模型内归一化
        """

        # Multi-scale embedding
        x_i = x_enc.permute(0, 2, 1)  # [B, C, L]

        x_i_p1 = x_i
        x_i_p2 = self.padding_patch_layer2(x_i)
        x_i_p3 = self.padding_patch_layer3(x_i)
        x_i_p4 = self.padding_patch_layer4(x_i)

        # 修复einops维度变换（避免维度混乱）
        b, c, l = x_i_p1.shape
        encoding_patch1 = self.embedding_patch_1(
            rearrange(x_i_p1, "b c l -> (b c) 1 l")  # 改为[B*C, 1, L]，适配Conv1d输入
        ).permute(0, 2, 1)
        encoding_patch2 = self.embedding_patch_2(
            rearrange(x_i_p2, "b c l -> (b c) 1 l")
        ).permute(0, 2, 1)
        encoding_patch3 = self.embedding_patch_3(
            rearrange(x_i_p3, "b c l -> (b c) 1 l")
        ).permute(0, 2, 1)
        encoding_patch4 = self.embedding_patch_4(
            rearrange(x_i_p4, "b c l -> (b c) 1 l")
        ).permute(0, 2, 1)

        # 获取所有patch的长度，取最小值作为目标长度
        patch_lengths = [
            encoding_patch1.shape[1],
            encoding_patch2.shape[1],
            encoding_patch3.shape[1],
            encoding_patch4.shape[1]
        ]
        target_len = min(patch_lengths)

        # 裁剪所有patch到相同长度
        encoding_patch1 = encoding_patch1[:, :target_len, :]
        encoding_patch2 = encoding_patch2[:, :target_len, :]
        encoding_patch3 = encoding_patch3[:, :target_len, :]
        encoding_patch4 = encoding_patch4[:, :target_len, :]
        # ==============================================

        # 拼接多尺度特征 + 位置编码
        encoding_patch = torch.cat(
            (encoding_patch1, encoding_patch2, encoding_patch3, encoding_patch4),
            dim=-1,
        )
        # 适配位置编码维度（避免维度不匹配）
        if encoding_patch.shape[1] > self.pe.shape[1]:
            pe = self.pe.repeat(1, encoding_patch.shape[1] // self.pe.shape[1] + 1, 1)
            pe = pe[:, :encoding_patch.shape[1], :]
        else:
            pe = self.pe[:, :encoding_patch.shape[1], :]
        encoding_patch = encoding_patch + pe

        # Temporal encoding
        for i in range(self.N):
            encoding_patch = self.encoder_list[i](encoding_patch)[0]

        # Channel-wise encoding
        x_patch_c = rearrange(
            encoding_patch, "(b c) p d -> b c (p d)", b=x_enc.shape[0], c=self.d_channel
        )
        x_ch = self.embedding_channel(x_patch_c.permute(0, 2, 1)).transpose(
            1, 2
        )  # [b, c, d]

        encoding_1_ch = self.encoder_list_ch[0](x_ch)[0]  # [B, C, D]

        # ========== 核心修改：单次1步预测 ==========
        # 直接输出1步，替换原8段式自回归逻辑
        final_forecast = self.out_linear(encoding_1_ch)  # [B, C, 1]
        final_forecast = final_forecast.permute(0, 2, 1)  # [B, 1, C]



        return final_forecast

    def forward(self, x_enc, mask=None):
        """
        简化接口：仅保留x_enc，输出压缩为[B, C]（单次1步）
        """
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            dec_out = self.forecast(x_enc)  # [B, 1, C]
            return dec_out.squeeze(1)  # 压缩为[B, C]
        if self.task_name == "imputation":
            raise NotImplementedError("Task imputation is temporarily not supported")
        if self.task_name == "anomaly_detection":
            raise NotImplementedError("Task anomaly_detection is temporarily not supported")
        if self.task_name == "classification":
            raise NotImplementedError("Task classification is temporarily not supported")
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
    # 添加梯度裁剪（解决发散问题）
    max_grad_norm = 1.0

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
            # 梯度裁剪（核心：防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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

    # 转换为数组并反归一化（仅外部反归一化，模型内无操作）
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
    from dysts.flows import Lorenz, Rossler, Chen, DoublePendulum,\
        Chua, HyperLorenz, RabinovichFabrikant

    model = RabinovichFabrikant()
    sol = model.make_trajectory(5000)
    feature_dim = sol.shape[1]
    data = sol
    return data, feature_dim


def preprocess_data(config=FRAMEWORK_CONFIG):
    data, feature_dim = generate_data()

    # 仅外部归一化（核心：模型内无额外归一化）
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