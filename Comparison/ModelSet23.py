import torch
import torch.nn as nn
from Train import FRAMEWORK_CONFIG
import torch.nn.functional as F


MODEL_CONFIGS = {
    "LSTM": {
        "hidden_dim": 64,
        "dropout_rate": 0.2
    },
    "GRU": {
        "hidden_dim": 64,
        "dropout_rate": 0.2
    },
    "CNN": {
        "cnn_channels": [64, 32],
        "kernel_size": 3,
        "pool_size": 2,
        "dropout_rate": 0.2
    },
    "Transformer": {
        "hidden_dim": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout_rate": 0.2
    },
    "MLP": {
        "hidden_layers": [128, 64],  # 隐藏层维度列表
        "dropout_rate": 0.2
    },
    "TCN": {
        "num_channels": [64, 64, 32],  # 每层通道数
        "kernel_size": 3,
        "dropout_rate": 0.2
    },
    "ESN": {
        "reservoir_size": 200,  # 储备池大小
        "spectral_radius": 0.95,
        "leaking_rate": 0.3,
        "connectivity": 0.1,
        "noise": 0.01
    },
    "WaveNet": {
        "num_blocks": 2,  # 残差块数量
        "num_layers": 3,  # 每个块的扩张卷积层数
        "num_channels": 64,
        "kernel_size": 2,
        "dropout_rate": 0.2
    }
}


class Chomp1d(nn.Module):
    """TCN因果卷积的Chomp操作"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN残差块"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class WaveNetBlock(nn.Module):
    """WaveNet残差块（因果扩张卷积）"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.dilated_conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size,
                                      padding=dilation * (kernel_size - 1), dilation=dilation)
        self.chomp = Chomp1d(dilation * (kernel_size - 1))
        self.gate_act = nn.Sigmoid()
        self.out_act = nn.Tanh()
        self.res_conv = nn.Conv1d(out_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dilated_conv(x)
        out = self.chomp(out)
        gate, out = torch.chunk(out, 2, dim=1)
        out = self.gate_act(gate) * self.out_act(out)
        out = self.dropout(out)

        skip = self.skip_conv(out)
        out = self.res_conv(out) + residual
        return out, skip


class LSTMPredictor(nn.Module):
    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["LSTM"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=model_config["hidden_dim"],
            num_layers=1,
            batch_first=True,
            dropout=model_config["dropout_rate"] if model_config["hidden_dim"] > 1 else 0
        )
        self.dropout = nn.Dropout(model_config["dropout_rate"])
        self.fc = nn.Linear(model_config["hidden_dim"], feature_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


class GRUPredictor(nn.Module):
    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["GRU"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=model_config["hidden_dim"],
            num_layers=1,
            batch_first=True,
            dropout=model_config["dropout_rate"] if model_config["hidden_dim"] > 1 else 0
        )
        self.dropout = nn.Dropout(model_config["dropout_rate"])
        self.fc = nn.Linear(model_config["hidden_dim"], feature_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


class CNNSNPredictor(nn.Module):
    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["CNN"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=feature_dim,
                out_channels=model_config["cnn_channels"][0],
                kernel_size=model_config["kernel_size"],
                padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=model_config["pool_size"]),
            nn.Conv1d(
                in_channels=model_config["cnn_channels"][0],
                out_channels=model_config["cnn_channels"][1],
                kernel_size=model_config["kernel_size"],
                padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=model_config["pool_size"])
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, feature_dim, framework_config["look_back"])
            conv_out = self.conv_layers(dummy_input)
            fc_input_dim = conv_out.size(1) * conv_out.size(2)

        self.dropout = nn.Dropout(model_config["dropout_rate"])
        self.fc = nn.Linear(fc_input_dim, feature_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_out = self.conv_layers(x)
        flatten_out = torch.flatten(conv_out, 1)
        flatten_out = self.dropout(flatten_out)
        return self.fc(flatten_out)


class TransformerPredictor(nn.Module):
    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["Transformer"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, model_config["hidden_dim"])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_config["hidden_dim"],
                nhead=model_config["nhead"],
                dropout=model_config["dropout_rate"],
                batch_first=True
            ),
            num_layers=model_config["num_layers"]
        )
        self.fc = nn.Linear(model_config["hidden_dim"], feature_dim)

    def forward(self, x):
        x_embed = self.embedding(x)
        trans_out = self.transformer(x_embed)
        last_hidden = trans_out[:, -1, :]
        return self.fc(last_hidden)


class MLPPredictor(nn.Module):
    """多层感知机"""

    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["MLP"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        input_dim = framework_config["look_back"] * feature_dim  # 展平后的输入维度
        hidden_layers = model_config["hidden_layers"]

        layers = []
        prev_dim = input_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(model_config["dropout_rate"]))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, feature_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x_flat = torch.flatten(x, 1)  # [batch, look_back, feat] → [batch, look_back*feat]
        return self.mlp(x_flat)


class TCNPredictor(nn.Module):
    """时间卷积网络"""

    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["TCN"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        num_channels = model_config["num_channels"]
        kernel_size = model_config["kernel_size"]
        dropout = model_config["dropout_rate"]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = feature_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout))

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], feature_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, look_back, feat] → [batch, feat, look_back]
        tcn_out = self.tcn(x)
        last_step = tcn_out[:, :, -1]  # 取最后一个时间步
        return self.fc(last_step)


class ESNPredictor(nn.Module):
    """回声状态网络"""

    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["ESN"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        self.reservoir_size = model_config["reservoir_size"]
        self.spectral_radius = model_config["spectral_radius"]
        self.leaking_rate = model_config["leaking_rate"]
        self.noise = model_config["noise"]

        # 初始化储备池权重
        W_in = torch.randn(feature_dim, self.reservoir_size) * 0.1
        self.register_buffer("W_in", W_in)

        W_res = torch.randn(self.reservoir_size, self.reservoir_size)
        # 稀疏连接
        mask = torch.rand(self.reservoir_size, self.reservoir_size) < model_config["connectivity"]
        W_res[~mask] = 0
        # 调整谱半径
        eigvals = torch.linalg.eigvals(W_res)
        W_res = W_res / torch.max(torch.abs(eigvals)) * self.spectral_radius
        self.register_buffer("W_res", W_res)

        # 输出层（可训练）
        self.W_out = nn.Linear(self.reservoir_size, feature_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        states = torch.zeros(batch_size, self.reservoir_size, device=x.device)

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            # 状态更新
            pre_state = torch.matmul(x_t, self.W_in) + torch.matmul(states, self.W_res)
            pre_state = torch.tanh(pre_state)
            # 加入噪声
            pre_state += self.noise * torch.randn_like(pre_state)
            # 泄漏积分
            states = (1 - self.leaking_rate) * states + self.leaking_rate * pre_state

        return self.W_out(states)


class WaveNetPredictor(nn.Module):
    """WaveNet模型"""

    def __init__(self, feature_dim, model_config=MODEL_CONFIGS["WaveNet"], framework_config=FRAMEWORK_CONFIG):
        super().__init__()
        num_blocks = model_config["num_blocks"]
        num_layers = model_config["num_layers"]
        num_channels = model_config["num_channels"]
        kernel_size = model_config["kernel_size"]

        self.init_conv = nn.Conv1d(feature_dim, num_channels, 1)
        self.blocks = nn.ModuleList()

        for b in range(num_blocks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.blocks.append(WaveNetBlock(num_channels, num_channels, kernel_size,
                                                dilation, model_config["dropout_rate"]))

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, 1),
            nn.ReLU(),
            nn.Conv1d(num_channels, feature_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, look_back, feat] → [batch, feat, look_back]
        x = self.init_conv(x)

        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        out = sum(skip_connections)
        out = self.final_conv(out)
        return out[:, :, -1].squeeze(-1)  # 取最后一个时间步


def build_model(model_type, feature_dim):

    device = FRAMEWORK_CONFIG["device"]
    if model_type == "LSTM":
        model = LSTMPredictor(feature_dim).to(device)
    elif model_type == "GRU":
        model = GRUPredictor(feature_dim).to(device)
    elif model_type == "CNN":
        model = CNNSNPredictor(feature_dim).to(device)
    elif model_type == "Transformer":
        model = TransformerPredictor(feature_dim).to(device)
    elif model_type == "MLP":
        model = MLPPredictor(feature_dim).to(device)
    elif model_type == "TCN":
        model = TCNPredictor(feature_dim).to(device)
    elif model_type == "ESN":
        model = ESNPredictor(feature_dim).to(device)
    elif model_type == "WaveNet":
        model = WaveNetPredictor(feature_dim).to(device)
    elif model_type == "FreMLP":
        model = FreMLPPredictor(feature_dim).to(device)

    print(f"\n===== Model Info: {model_type} =====")
    print(model)
    print("====================================")
    return model