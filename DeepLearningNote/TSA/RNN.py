#### 使用nn.Linear实现RNN ####
'''
RNN本质上就是一堆Linear+非线性+时间循环结构的堆叠
公式是
h_t = f(W_ih*x_t + W_hh*h_(t-1) + b_hh)
y_t = W_ho*h_t + b_ho

__init__：搭骨架、创建层 & 参数
forward：规定数据怎么从输入流到输出（前向计算过程）
'''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
from tqdm import tqdm

class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入到隐藏层的线性变换 W_ih*x_t
        self.i2h = nn.Linear(input_size, hidden_size, bias = False)
        # 隐藏层到隐藏层的线性变换 W_hh*h_(t-1)+b_hh
        self.h2h = nn.Linear(hidden_size, hidden_size, bias = True)
        
    def forward(self, x_t, h_prev):
        # h_t = f(W_ih*x_t + W_hh*h_(t-1) + b_hh)
        h_t = self.i2h(x_t) + self.h2h(h_prev)
        h_t = torch.tanh(h_t)
        return h_t

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = MyRNNCell(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0=None):
        B, T, _ = x.size() # [B, T, C]
        if h0 is None:
            h_t = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_t = h0
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            h_t = self.cell(x_t, h_t)
            outputs.append(h_t.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1) # outputs是记录每个时间步的隐藏状态 [B, T, H]
        y_hat = self.head(h_t) # 只用最后一个时间步的隐藏状态预测输出
        return outputs,y_hat

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0):
        rnn_out, h_n = self.rnn(x, h0) 
        y_hat = self.head(h_n.squeeze(0))
        return rnn_out, y_hat

def copy_rnn_weights(official_rnn: nn.RNN, custom_rnn: MyRNN):
    """
    让自定义 MyRNN 和官方 nn.RNN 有一模一样的参数
    """
    with torch.no_grad():
        W_ih = official_rnn.weight_ih_l0      # [H, C]
        W_hh = official_rnn.weight_hh_l0      # [H, H]
        b_ih = official_rnn.bias_ih_l0        # [H]
        b_hh = official_rnn.bias_hh_l0        # [H]

        # 对应到我们自定义的 cell
        custom_rnn.cell.i2h.weight.copy_(W_ih)
        custom_rnn.cell.h2h.weight.copy_(W_hh)
        custom_rnn.cell.h2h.bias.copy_(b_ih + b_hh)
        
        custom_rnn.head.weight.copy_(official_rnn._modules['head'].weight)
        custom_rnn.head.bias.copy_(official_rnn._modules['head'].bias)

def generate_series(n_points=300, feature_size=1):
    """
    生成时间序列 shape = [n_points,feature_size]
    """
    t = torch.linspace(0, 8 * math.pi, steps=n_points)
    series_list = []
    for i in range(feature_size):
        freq   = 1.0 + 0.2 * i
        phase  = 0.5 * i
        amp    = 1.0 + 0.3 * i

        base = amp * torch.sin(freq * t + phase) 
        noise = 0.1 * torch.randn_like(base)
        series_i = base + noise                        # [n_points]
        series_list.append(series_i.unsqueeze(-1))     # [n_points, 1]

    series = torch.cat(series_list, dim=-1)            # [n_points, feature_size]
    return series

def make_dataset(series, window_size, horizon):
    """
    series: [n_points, C]
    window_size: T
    horizon: 预测步数
    X: [N, T, C] 
    y: [N, horizon]
    """
    X, y = [], []
    n_points, _ = series.shape
    for i in range(n_points - window_size - horizon + 1):
        window = series[i : i + window_size]                     # [T, C]
        target = series[i + window_size : i + window_size + horizon]  # [horizon, C]
        X.append(window)
        y.append(target)

    X = torch.stack(X)  # [N, T, C]
    y = torch.stack(y)  # [N, horizon, C]
    return X, y

def train_model(model, dataloader, device, num_epochs=50, lr = 1e-2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    # 表头
    print(('\n' + '%-10s' * 2) % ('Epoch', 'loss'))
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_samples = 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            _, y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)
            
            # tqdm 上显示当前 batch 的 loss
            pbar.set_description(f"Epoch {epoch:03d}")
            pbar.set_postfix(batch_loss=f"{loss.item():.6f}")
            
        epoch_loss /= n_samples
        if (epoch + 1) % 10 == 0:
            print(f"{epoch:<10d}{epoch_loss:<10.6f}")
        
    return model


# 测试代码 =========================
feature_size = 1
hidden_size = 8
output_size = 1
series = generate_series(n_points=300, feature_size=feature_size)  # [300, 1]
window_size = 20
horizon = 3
X, y = make_dataset(series, window_size, horizon) 
y = y.squeeze(-1)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MyRNN(input_size=feature_size,
              hidden_size=hidden_size,
              output_size=output_size)
model = train_model(model, loader, num_epochs=30, lr=1e-2, device='gpu' if torch.cuda.is_available() else 'cpu')

B, T, C = 16, 20, 4 # batch size, time steps, input dimension (feature size)
H = 8 # 自定义了主要是看设置需要多少个cell
O = 10 # 输出维度

my_rnn = MyRNN(input_size=C, hidden_size=H, output_size=O)

# 把 nn.RNN 的初始化参数拷贝到 MyRNN 里
with torch.no_grad():
    # RNN 的参数
    W_ih = rnn.weight_ih_l0      # [H, C]
    W_hh = rnn.weight_hh_l0      # [H, H]
    b_ih = rnn.bias_ih_l0        # [H]
    b_hh = rnn.bias_hh_l0        # [H]

    # 拷贝到我们的 cell 里
    my_rnn.cell.x2h.weight.copy_(W_ih)
    my_rnn.cell.h2h.weight.copy_(W_hh)
    my_rnn.cell.h2h.bias.copy_(b_ih + b_hh)  # 合并两个 bias


