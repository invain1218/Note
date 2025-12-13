import math
import torch
from torch import nn
from torchviz import make_dot
from torchinfo import summary 
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
import random
import os



'''
RNN是 h_t = tanh(W_xh*x_t+W_hh*h_(t-1)+b_h); y_t = tanh(W_hy*h_t+b_y)

LSTM增加三个门控解决RNN长期依赖导致的梯度消失问题:
forget gate: f_t = sigmoid(U_f*x_t+V_f*h_(t-1)+b_f) 范围在0-1
Input gate: i_t = sigmoid(U_i*x_t+V_i*h_(t-1)+b_i) 
Output gate: o_t = sigmoid(U_o*x_t+V_o*h_(t-1)+b_o)

candidate cell: g_t = tanh(U_g*x_t+V_g*h_(t-1)+b_g) a.k.a \tilde c_t

c_t = f_t \circ c_(t-1) + i_t \circ g_t # 遗忘门调节过去cell state + 输入门调节现在的 candidate cell
h_t = o_t \circ tanh(c_t) # 输出门调节过去cell state


遗忘门对长期记忆c_(t-1)进行筛选

LSTM增加的peephole：
让 LSTM 的门控单元输出门能直接看到cell state，而非仅依赖hidden state

遗忘门 + peephole:
f_t = sigmoid(U_f*x_t + V_f*h_(t-1) + W_cf*c_(t-1) + b_f)
输入门 + peephole:  
i_t = sigmoid(U_i*x_t + V_i*h_(t-1) + W_ci*c_(t-1) + b_i)
候选值（不变）:
g_t = tanh(U_g*x_t + V_g*h_(t-1) + b_g)
更新cell state:
c_t = f_t ∘ c_(t-1) + i_t ∘ g_t
输出门 + peephole:
o_t = sigmoid(U_o*x_t + V_o*h_(t-1) + W_co*c_t + b_o)
最终输出:
h_t = o_t ∘ tanh(c_t)


DeepAR是在LSTM架构上把预测head换成了输出概率分布参数的。
例如，假设序列服从正态分布，那么就输出mu和sigma。

sigma通过softplus

'''

# class NavieCustomLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
        
#         # f_t
#         self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
#         # i_t
#         self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
#         # g_t or c_t
#         self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        
#         # o_t
#         self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.V_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         self.b_o = nn.Parameter(torch.Tensor(hidden_size))

#         self.init_weights() # 调用初始参数函数
    
#     def init_weights(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv) # 所有可被更新的参数都设置均匀初始值
    
#     def forward(self, x, init_states = None):
#         bs, seq_sz, c = x.size() # batch_size, seq_len, feature_size
#         hidden_seq = []
#         if init_states is None:
#             h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
#                         torch.zeros(bs, self.hidden_size).to(x.device))
#         else:
#             h_t, c_t = init_states
        
#         hidden_seq = self.hidde_size
#         for t in range(seq_sz):
#             x_t = x[:, t, :] # batch里所有第t个时间步的所有特征
            
#             i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
#             f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
#             g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
#             o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            
#             # 更新c_t h_t
#             c_t = f_t * c_t + i_t * g_t
#             h_t = o_t * torch.tanh(c_t)
            
#             hidden_seq.append(h_t.unsqueeze(0))
            
#         #reshape hidden_seq p/ retornar
#         hidden_seq = torch.cat(hidden_seq, dim=0) # rbind 此时是[T,B,H]
#         hidden_seq = hidden_seq.transpose(0, 1).contiguous() 
#         # 先转置0，1交换然后contiguous整理
#         # 最终是[B,T,H]
#         return hidden_seq, (h_t, c_t)

# # 矩阵乘法
# class CustomLSTM(nn.Module):
#     def __init__(self, input_sz, hidden_sz):
#         super().__init__()
#         self.input_sz = input_sz
#         self.hidden_size = hidden_sz
#         self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
#         self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
#         self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
#         self.init_weights()
        
#     def init_weights(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
        
#     def forward(self, x,
#                 init_states=None):
#         """Assumes x is of shape (batch, sequence, feature)"""
#         bs, seq_sz, _ = x.size()
#         hidden_seq = []
#         if init_states is None:
#             h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
#                         torch.zeros(bs, self.hidden_size).to(x.device))
#         else:
#             h_t, c_t = init_states
        
#         HS = self.hidden_size
#         for t in range(seq_sz):
#             x_t = x[:, t, :] # x_t是[B, C] h_t是[B, H]
#             # batch the computations into a single matrix multiplication
#             gates = x_t @ self.W + h_t @ self.U + self.bias # [2, 8]
#             i_t, f_t, g_t, o_t = (
#                 torch.sigmoid(gates[:, :HS]), # input 
#                 torch.sigmoid(gates[:, HS:HS*2]), # forget
#                 torch.tanh(gates[:, HS*2:HS*3]),
#                 torch.sigmoid(gates[:, HS*3:]), # output
#             )
#             c_t = f_t * c_t + i_t * g_t
#             h_t = o_t * torch.tanh(c_t)
#             hidden_seq.append(h_t.unsqueeze(0))
#         hidden_seq = torch.cat(hidden_seq, dim=0)
#         # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
#         hidden_seq = hidden_seq.transpose(0, 1).contiguous()
#         return hidden_seq, (h_t, c_t)

class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, peephole=False):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        if self.peephole:
            self.W_fc = nn.Parameter(torch.Tensor(hidden_sz))
            self.W_ic = nn.Parameter(torch.Tensor(hidden_sz))
            self.W_oc = nn.Parameter(torch.Tensor(hidden_sz))
            
        self.init_weights()
        
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.shape
        device = x.device
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(device),
                        torch.zeros(bs, self.hidden_size).to(device))
        else:
            h_t, c_t = init_states
            h_t = h_t.to(device)
            c_t = c_t.to(device)
        
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :] # 当前时刻输入，shape=(batch_size, input_sz)
            # batch the computations into a single matrix multiplication
            # - x_t@W: (batch, input_sz) @ (input_sz, 4HS) → (batch, 4HS)
            # - h_t@U: (batch, HS) @ (HS, 4HS) → (batch, 4HS)
            # gates = x_t @ self.W + h_t @ self.U + self.bias
            gates = torch.matmul(x_t, self.W) + torch.matmul(h_t, self.U) + self.bias.to(device)
            
            if self.peephole:
                # 遗忘门 += 前一时刻细胞状态 * W_fc（按元素乘）
                f_t = torch.sigmoid(gates[:, HS:HS*2] + c_t * self.W_fc)
                # 输入门 += 前一时刻细胞状态 * W_ic（按元素乘）
                i_t = torch.sigmoid(gates[:, :HS] + c_t * self.W_ic)
                # 先更新细胞状态（用于输出门的peephole）
                c_t = f_t * c_t + i_t * torch.tanh(gates[:, HS*2:HS*3])
                # 输出门 += 当前时刻细胞状态 * W_oc（按元素乘）
                o_t = torch.sigmoid(gates[:, 3*HS:4*HS] + c_t * self.W_oc)
            else:
                # LSTM 
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]), # input 
                    torch.sigmoid(gates[:, HS:HS*2]), # forget
                    torch.tanh(gates[:, HS*2:HS*3]),
                    torch.sigmoid(gates[:, HS*3:HS*4]), # output
                )
                c_t = f_t * c_t + i_t * g_t
            
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
            
        hidden_seq = torch.cat(hidden_seq, dim=0) # rbind
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        return hidden_seq, (h_t, c_t)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size: list, output_size, pred_len, peephole, dropout):
        super().__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_size  
        self.output_size = output_size
        self.pred_len = pred_len
        self.peephole = peephole
        self.dropout = dropout
        
        self.batch_norm = nn.BatchNorm1d(input_size) 
        self.lstm_layers = nn.ModuleList()
        prev_size = input_size
        for hs in self.hidden_sizes:
            lstm_layer = CustomLSTM(input_sz=prev_size,
                                    hidden_sz=hs,
                                    peephole=peephole)
            self.lstm_layers.append(lstm_layer)
            prev_size = hs
        
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(len(self.lstm_layers))
        ])
        
        self.head = nn.Linear(self.hidden_sizes[-1],output_size*pred_len)
    
    def forward(self, x):
        B, T, C = x.shape
        current_x = x
        hidden_state = []
        
        for idx, lstm_layer in  enumerate(self.lstm_layers):
            hidden_seq, (h_t, c_t) = lstm_layer(current_x)
            hidden_state.append((h_t,c_t)) # 最后时刻的h和c
            
            if idx < len(self.lstm_layers)-1:
                current_x = self.dropout_layers[idx](hidden_seq)
            else:
                current_x = hidden_seq # 最后一个lstm不增加dropout
        
        final_h = hidden_state[-1][0] # hidden_seq
        y_hat = self.head(final_h)
        y_hat = y_hat.view(B, self.pred_len, self.output_size) # [B,T,C]
        
        return hidden_state, y_hat

class OfficeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size: list, output_size, pred_len, dropout):
        super().__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_size  
        self.output_size = output_size
        self.pred_len = pred_len
        self.dropout = dropout
        
        self.batch_norm = nn.BatchNorm1d(input_size) 
        self.lstm_layers = nn.ModuleList()
        prev_size = input_size
        for i, hs in enumerate(self.hidden_sizes):
            dropout_rate = dropout if i < len(hidden_size) - 1 else 0 # 最后一层不用dropout
            lstm = nn.LSTM(
                input_size=prev_size,
                hidden_size=hs,
                batch_first=True,
                dropout=dropout_rate 
            )
            self.lstm_layers.append(lstm)
            prev_size = hs
        
        self.head = nn.Linear(self.hidden_sizes[-1],output_size*pred_len)
    
    def forward(self, x):
        B, T, C = x.shape
        current_x = x
        hidden_state = []
        
        for lstm_layer in  self.lstm_layers:
            hidden_seq, (h_t, c_t) = lstm_layer(current_x)
            hidden_state.append((h_t,c_t)) # 最后时刻的h和c
            current_x = hidden_seq
        
        final_h = hidden_state[-1][0] # hidden_seq
        y_hat = self.head(final_h)
        y_hat = y_hat.view(B, self.pred_len, self.output_size) # [B,T,C]
        
        return hidden_state, y_hat

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


# input_size = 10
# hidden_size = 20
# # lstm = CustomLSTM(input_size, hidden_size, peephole=True)
# lstm = CustomLSTM(input_size, hidden_size, peephole=False)
# # batch_size=32, sequence_len=15, input_size=10
# x = torch.randn(32, 15, 10)
# hidden_seq, (h_final, c_final) = lstm(x)
# hidden_seq.shape  # (32, 15, 20)
# h_final.shape  # (32, 20)
# c_final.shape  # (32, 20)


feature_size = 3
hidden_size = [32, 16] 
seq_len =256
pred_len = 96 # pred_len
batch_size = 32
feature_type = "M" # S, M, MS
output_size = 1
target = "OT"
ROOT = os.getcwd()

set_seed(123456)
# nsample = 5000
# data = gen_ts_data(n_points=nsample, feature_size=feature_size, start_time="2020-12-12 14:30:00",frequency="H") 
# data.head()
data = pd.read_csv("/workspace/Data/all_six_datasets/electricity/electricity.csv")
data_loader = CustomDataLoader(data,batch_size,seq_len,pred_len,feature_type,target)
train_data = data_loader.get_train()
val_data = data_loader.get_val()
test_data = data_loader.get_test()

input_size = data_loader.n_feature

x0, y0 = train_data.dataset.dataset[0]
print("x0 shape:", x0.shape)  
print("y0 shape:", y0.shape)
# 重点是使用过去预测未mylstn = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, pred_len=pred_len, peephole=True, dropout=0.1)
# batch_x, batch_y = next(iter(train_data))
# print(batch_x.shape, batch_y.shape)
# model_c(batch_x) # [B, T, C]
m_lstm, last_tr_loss_mlstn, last_val_loss_mlstn = train_model(mylstn, train_loader=train_data, val_loader=val_data, 
                    save_directory="/workspace/Code/04_DeepAR/Save_MyLSTM", num_epochs=100, lr=1e-2)

lstn = OfficeLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, pred_len=pred_len, dropout=0.1)
o_lstm, last_tr_loss_olstn, last_val_loss_olstn = train_model(lstn, train_loader=train_data, val_loader=val_data, 
                    save_directory="/workspace/Code/04_DeepAR/Save_OfficeLSTM", num_epochs=100, lr=1e-2)


evaluate_model(model=m_lstm, test_loader=test_data, criterion=nn.MSELoss())
evaluate_model(model=o_lstm, test_loader=test_data, criterion=nn.MSELoss())来！




# DeepAR 
# Gaussian distribution
# Negative Binomial distribution
# softplus
class DeepAR(nn.modules):
    def __init__(self, input_size, output_size, pred_len, hidden_size:list, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.dropout = dropout
        
        self.batch_norm = nn.BatchNorm1d(input_size) 
        self.lstm_layers = nn.ModuleList()
        prev_size = input_size
        for i, hs in enumerate(self.hidden_sizes):
            dropout_rate = dropout if i < len(hidden_size) - 1 else 0 # 最后一层不用dropout
            lstm = nn.LSTM(
                input_size=prev_size,
                hidden_size=hs,
                batch_first=True,
                dropout=dropout_rate 
            )
            self.lstm_layers.append(lstm)
            prev_size = hs
        
        self.head = nn.Linear(self.hidden_sizes[-1],output_size*pred_len)
    
    def forward(self, x):
        B, T, C = x.shape
        current_x = x
        hidden_state = []
        
        for lstm_layer in  self.lstm_layers:
            hidden_seq, (h_t, c_t) = lstm_layer(current_x)
            hidden_state.append((h_t,c_t)) # 最后时刻的h和c
            current_x = hidden_seq
        
        final_h = hidden_state[-1][0] # hidden_seq
        y_hat = self.head(final_h)
        y_hat = y_hat.view(B, self.pred_len, self.output_size) # [B,T,C]
        
        return hidden_state, y_hat


import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_time_token

'''
We define a recurrent network that predicts the future values of a time-dependent variable based on
past inputs and covariates.
'''
class Deepar(nn.Module):
    def __init__(self,seq_len, pred_len, label_len, input_size, hidden_size d_feature, d_mark,lstm_layers=3,dropout=0.2):
        super().__init__()
        self.seq_len = seq_len # Known length of time series
        self.label_len = label_len
        self.pred_len = pred_len
        self.input_size = input_size
        self.hiiden_size = hidden_size
        self.d_feature = d_feature
        self.d_mark = d_mark
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
        # d_model -> input_size
        # d_ff -> hidden_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=self.dropout)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        # input：(batch_size,hidden_size*num_layers)-->output：（batch_size,d_model）
        self.distribution_mu = nn.Linear(self.d_ff * self.lstm_layers, self.d_model)
        # input：(batch_size,hidden_size*num_layers)-->output：（batch_size,d_model）
        self.distribution_presigma = nn.Linear(self.d_ff * self.lstm_layers, self.d_model)
        # Use the softplus activation function to make sigma greater than 0.
        self.distribution_sigma = nn.Softplus()
        # input：(batch_size,d_model)-->output：(batch_size,d_feature)
        self.mu_outfc = nn.Linear(self.d_model,self.d_feature)
        # input：(batch_size,d_model)-->output：(batch_size,d_feature)
        self.sigma_outfc = nn.Linear(self.d_model, self.d_feature)
        # input：（batch_size,pred_len,d_model）-->output：（batch_size,pred_len,d_feature）
        self.pred_outfc = nn.Linear(self.d_model, self.d_feature)

        self.embedding = DataEmbedding_time_token(self.d_feature, self.d_mark, self.d_model)

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)



    def pred_onestep(self,x,hidden,cell):

        #x：（seq_len,batch_size,self.d_feature）-->output (seq_len,batch_size,hidden_size)
        # hidden and cell (num_layers,batch_size,hidden_size)
        output, (hidden, cell) = self.lstm(x, (hidden, cell)) # (96,32,64)-->(96,32,128)
        # use h from all three layers to calculate mu and sigma
        # hidden_permute (batch_size,hidden_size*num_layers)
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma) # Use the softplus activation function to make sigma greater than zero.

        gaussian = torch.distributions.normal.Normal(mu, sigma) # Construct a Gaussian distribution using mu and sigma
        pred = gaussian.sample()  # Predicted values are sampled from the Gaussian distribution just constructed
        # pred and mu and sigma are (batch_size,d_model)
        return pred,mu,sigma


    def forward(self, enc_x, enc_mark, y, y_mark,mode):
        loss = torch.zeros(1, device=enc_x.device)
        B = enc_x.shape[0] # batch_size
        x_embed = self.embedding(enc_x, enc_mark)
        y_embed = self.embedding(y,y_mark)

        #  (batch_size,pred_len,d_model)
        pred_zero = torch.zeros_like(y_embed[:, -self.pred_len:, :]).float()
        input_zero = torch.zeros_like(y_embed[:, -self.pred_len:, :]).float()

        x_cat_pred = torch.cat([x_embed[:, :self.seq_len, :], pred_zero], dim=1).float().to(enc_x.device) # 把初始化的预测值和原本的时序数据拼接起来，用来装预测值
        x_cat_input = torch.cat([x_embed[:, :self.seq_len, :], input_zero], dim=1).float().to(enc_x.device) # 打算在训练的时候每一次输入LSTM的数据都是真实的数据，而不是上一个时间步预测值

        hidden = torch.zeros(self.lstm_layers,B , self.d_ff, device=enc_x.device)
        cell = torch.zeros(self.lstm_layers,B, self.d_ff, device=enc_x.device)


        for i in range(self.pred_len):
            if mode == 'train':
                lstm_input = x_cat_input[:, i:i + self.seq_len, :].permute(1, 0, 2).clone()
            else:
                lstm_input = x_cat_pred[:, i:i + self.seq_len, :].permute(1,0,2).clone()
            pred,mu,sigma = self.pred_onestep(lstm_input ,hidden,cell)
            # input：（batch_size,d_model）-->(batch_size,d_feature)
            out_mu = self.mu_outfc(mu)
            out_sigma = self.distribution_sigma(self.sigma_outfc(sigma))
            loss += self.loss_fn(out_mu,out_sigma,y[:,self.label_len+i,:])
            x_cat_pred[:, self.seq_len + i, :] = pred
            x_cat_input[:, self.seq_len + i, :] = y_embed[:, self.label_len + i,:]

        return self.pred_outfc(x_cat_pred[:,-self.pred_len:,:]),loss

    def loss_fn(self,mu, sigma,labels): # Customise a loss function with negative logarithmic loss
        '''
        Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
        Args:
            mu: (Variable) dimension [batch_size] - estimated mean at time step t
            sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
            labels: (Variable) dimension [batch_size] z_t
        Returns:
            loss: (Variable) average log-likelihood loss across the batch
        '''
        distribution = torch.distributions.normal.Normal(mu, sigma) # Reconstructing a Gaussian distribution using mu,sigma
        likelihood = distribution.log_prob(labels) # Negative log likelihood
        return -torch.mean(likelihood)





