import torch
import torch.nn as nn
import optuna

class Net(nn.Module):
    def __init__(self, trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1):
        super(Net, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # 假设输入图像通道数为 3

        # 根据建议的卷积层数量和滤波器数量构建卷积层
        for i in range(num_conv_layers):
            out_channels = num_filters[i]
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels

        # 进一步使用 trial 来决定是否添加批归一化层
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(num_conv_layers):
                self.batch_norm_layers.append(nn.BatchNorm2d(num_filters[i]))

        # 构建全连接层
        self.fc1 = nn.Linear(num_filters[-1] * 8 * 8, num_neurons)
        self.dropout_fc1 = nn.Dropout(drop_fc1)
        self.fc2 = nn.Linear(num_neurons, 10)

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if hasattr(self, 'batch_norm_layers'):
                x = self.batch_norm_layers[i](x)
            x = torch.relu(x)

        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)
        x = self.dropout_fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
