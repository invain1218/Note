import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm # 对权重进行归一化
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
import random
import os

# 右边的信息（未来信息）切掉，保证因果卷积（只使用过去数据）
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() # contiguous保证连续存储


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        # out_put是卷积核数量，如果是[B,T,C]需要转置变成[B,C,T]
        # 每个卷积核都会在时间维上滑动并产生一条输出序列
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights() # 初始化参数

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
        # resblock H(x) = x + f(x)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super().__init__()
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
#         self.linear = nn.Linear(num_channels[-1], output_size) # 换一下head就行

#     def forward(self, x):
#         # x 需要转置变成 [B,C,T],因为需要Conv在时间维上对特征进行处理
#         output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
#         # 然后再回来
#         output = self.linear(output)
        
#         return output

# 这个是sequence-to-sequence，X_t->Y_t
# 要变成X_(t-seq_len) -> Y_(t+pred_len),输入pred_len，head维度变成output_size*pred_len然后reshape
# 注意这里和rnn，lstm一样，只用最后的时间步（最新信息）来预测未来multi-step


class TCN(nn.Module):
    def __init__(self, input_size, output_size,pred_len, num_channels, kernel_size, dropout):
        super().__init__()
        self.output_size = output_size
        self.pred_len = pred_len
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size*pred_len) # 换一下head就行

    def forward(self, x):
        B,_,_ = x.shape
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        y_hat = output[:,-1,:]
        y_hat = self.linear(y_hat)
        y_hat = y_hat.view(B, self.pred_len, self.output_size)
        
        return output, y_hat


class CustomDataLoader:
    """Generate data loader from raw data."""
    def __init__(self, data, batch_size, seq_len, pred_len, feature_type, target=None, scale = True):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len # T_in
        self.pred_len = pred_len # T_out
        self.feature_type = feature_type 
        # 'S': 单变量输入 → 单变量输出（只用一列）
        # 'M': 多变量输入 → 多变量输出（所有列）
        # 'MS': 多变量输入 → 单变量输出（只预测 target 列）
        self.target = target
        self.target_slice = slice(0, None)
        self.scale = scale
        
        self._split_data()

    def _split_data(self):
        df_raw = self.data
        df = df_raw.set_index('date')
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1) # 保留target列

        # split train/valid/test
        n = len(df)
        train_end = int(n * 0.7)
        val_end = n - int(n * 0.2)
        test_end = n
        train_df = df[:train_end] # 70%
        val_df = df[train_end - self.seq_len : val_end] # 10%
        test_df = df[val_end - self.seq_len : test_end] # 20%

        # standardize by training set
        if self.scale:
            self.scaler = StandardScaler() 
            self.scaler.fit(train_df.values) # 注意是用训练集的mean和sd
            def scale_df(df, scaler):
                data = scaler.transform(df.values)
                return pd.DataFrame(data, index=df.index, columns=df.columns)

            self.train_df = scale_df(train_df, self.scaler)
            self.val_df = scale_df(val_df, self.scaler)
            self.test_df = scale_df(test_df, self.scaler)
            self.n_feature = self.train_df.shape[-1]
        else:
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            self.n_feature = self.train_df.shape[-1]

    def _make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)

        data_x = torch.tensor(data, dtype=torch.float32)
        data_y = torch.tensor(data[:, self.target_slice], dtype=torch.float32)
        
        return DataLoader(
            torch.utils.data.Subset(
                CustomDataset(data_x, data_y, self.seq_len, self.pred_len),
                range(len(data_x) - self.seq_len - self.pred_len + 1)
            ), # Subset(dataset, indices) 
            # 用 Subset 把合法的索引限制在range(len(data_x) - self.seq_len - self.pred_len + 1)
            batch_size=self.batch_size, 
            shuffle=shuffle
        )

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        else:
            return data

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False)
    
class CustomDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, pred_len):
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y
        
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        return self.data_x[idx : idx + self.seq_len], self.data_y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        # 输入[seq_len, C]
        # 标签[pred_len, C_target]

def gen_ts_data(n_points=500, feature_size=1, start_time = None, frequency = "H"):
    """
    生成时间序列 shape = [n_points,feature_size]
    """
    t = torch.linspace(0, 8 * math.pi, steps=n_points)
    series_list = []
    
    if feature_size > 0:
        for i in range(feature_size):
            freq   = 1.0 + 0.2 * i
            phase  = 0.5 * i
            amp    = 1.0 + 0.3 * i

            base = amp * torch.sin(freq * t + phase) 
            noise = 0.1 * torch.randn_like(base)
            series_i = base + noise                        # [n_points]
            series_list.append(series_i.unsqueeze(-1))     # [n_points, 1]

    y_freq = 1.0 + 0.2 * feature_size
    y_phase = 0.5 * feature_size
    y_amp = 1.0 + 0.3 * feature_size
    y_base = y_amp * torch.sin(y_freq * t + y_phase)
    y_noise = 0.1 * torch.randn_like(y_base)
    y_series = y_base + y_noise                    # [n_points]
    series_list.append(y_series.unsqueeze(-1)) 
    
    series = torch.cat(series_list, dim=-1) # 根据最后一个维度进行拼接 cbind
    
    # 生成date
    if start_time is None:
        start_dt = datetime.now()
    elif isinstance(start_time, str):
        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    elif isinstance(start_time, datetime):
        start_dt = start_time
    else:
        raise ValueError("start_time必须是datetime对象或'YYYY-MM-DD HH:MM:SS'格式的字符串！")
    date_series = pd.date_range(
        start=start_dt,
        periods=n_points,
        freq=frequency
    )
    date = pd.DataFrame(date_series, columns=['date'])
    
    if feature_size == 0:
        col_names = ["Y"]  # 无特征列，仅保留Y
    else:
        col_names = [f"X{i}" for i in range(feature_size)] + ["Y"]
    
    ts_data = pd.DataFrame(data = series.numpy(), columns=col_names)
    final_data = pd.concat([date, ts_data], axis=1) # cbind
    
    return final_data

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

def train_model(model, train_loader,val_loader, save_directory, num_epochs=50, lr = 1e-2, max_patience = 10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }
    
    os.makedirs(save_directory, exist_ok=True)  # 修复：使用makedirs
    best_loss = best_loss = torch.tensor(float('inf'), device=device)
    patience = 0
    best_model = None

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        # train =====
        model.train()
        train_loss = 0.0
        print(('\n' + '%-10s' * 2) % ('Epoch', 'Train loss'))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (X_batch, y_batch) in pbar:
            X_batch, y_batch  = X_batch.to(device), y_batch.to(device)
            
            _, y_pred = model(X_batch)
            optimizer.zero_grad()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            loss_value = loss.item() # 因为后面set_description里的'%g' 这种格式化需要的是 float
            train_loss = (train_loss * i + loss_value) / (i+1)
            # tqdm 上显示当前 batch 的 loss
            pbar.set_description(('%-10s' * 1 + '%-10.4g' * 1) %
                                (f'{epoch+1}/{num_epochs}', train_loss))
        pbar.close()
        
        # validation ====
        model.eval()
        val_loss = 0.0
        
        print(('%-10s' * 2) % ('', 'Val loss'))
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        with torch.no_grad(): 
            for i, (X_batch, y_batch) in pbar:
                X_batch, y_batch  = X_batch.to(device), y_batch.to(device)
                _, y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss_value = loss.item()
                val_loss = (val_loss * i + loss_value) / (i + 1)
                pbar.set_description(('%-10s' * 1 + '%-10.4g') %
                                        (f'', val_loss))
        pbar.close()
        
        # 记录历史
        train_history["epoch"].append(epoch + 1)
        train_history["train_loss"].append(train_loss)
        train_history["val_loss"].append(val_loss)
        
        # 早停
        if val_loss < best_loss: 
            best_loss = val_loss
            best_model = deepcopy(model.state_dict()) # 保存参数
            torch.save(best_model, os.path.join(save_directory, "best.pt"))
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print("早停触发，停止训练。")
                break 
    
    # 绘制图片
    plt.figure(figsize = (12,6))
    plt.plot(train_history["epoch"], train_history["train_loss"], label="Train Loss")
    plt.plot(train_history["epoch"], train_history["val_loss"],   label="Val Loss")    
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    # 保存到和 best.pt 同一个目录
    fig_path = os.path.join(save_directory, "loss_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print("Loss 曲线已保存到:", fig_path)
    
    # 最终设置是评估模式
    if best_model is not None:
        model.load_state_dict(best_model)
    model.eval()

    return model, train_history["train_loss"][-1], train_history["val_loss"][-1]

def evaluate_model(model, test_loader, criterion, device = None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    test_loss = torch.zeros(1, device=device)
    print(('\n' + '%-10s' * 1) % ('Test loss'))
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            _, y_pred = model(batch_x)       
            loss = criterion(y_pred, batch_y)  
            test_loss = (test_loss * i + loss.detach()) / (i + 1)
            pbar.set_description(('%-10.4g' * 1) % (test_loss))

    print("MSE:", test_loss.item())
    print("RMSE:", test_loss.sqrt().item())


num_channels = [32,16] 
kernel_size = 2
dropout = 0.7
seq_len =256
pred_len = 96 # pred_len
batch_size = 32
feature_type = "MS" # S, M, MS
output_size = 1
target = "OT"
ROOT = os.getcwd()

set_seed(123456)
data = pd.read_csv("/workspace/Data/all_six_datasets/electricity/electricity.csv")
data_loader = CustomDataLoader(data,batch_size,seq_len,pred_len,feature_type,target)
train_data = data_loader.get_train()
val_data = data_loader.get_val()
test_data = data_loader.get_test()

x0, y0 = train_data.dataset.dataset[0]
print("x0 shape:", x0.shape)  
print("y0 shape:", y0.shape)
# 重点是使用过去预测未来！

model = TCN(input_size=data_loader.n_feature, output_size=output_size, pred_len=pred_len, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
batch_x, batch_y = next(iter(train_data))
print(batch_x.shape, batch_y.shape)
model(batch_x) # [B, T, C]
model, last_tr_loss, last_val_loss = train_model(model, train_loader=train_data, val_loader=val_data, 
                    save_directory="/workspace/Code/05_TCN/Save_TCN", num_epochs=100, lr=1e-3)

evaluate_model(model=model, test_loader=test_data, criterion=nn.MSELoss())
