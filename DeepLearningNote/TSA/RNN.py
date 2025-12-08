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
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from copy import deepcopy
import os

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe

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
        self.head = nn.Linear(hidden_size, output_size*input_size)  # [B, horizon]
        
    def forward(self, x, h0=None):
        B, T, C = x.size() # [B, T, C]
        if h0 is None:
            h_t = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_t = h0
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            h_t = self.cell(x_t, h_t)
            outputs.append(h_t.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1) # outputs是记录每个时间步的隐藏状态 
        y_hat = self.head(h_t) # 只用最后一个时间步的隐藏状态预测输出 [B, horizon*C]
        y_hat = y_hat.view(B, self.output_size, C)  # [B, horizon, C]
        return outputs,y_hat

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size*input_size)
        
    def forward(self, x, h0=None):
        if h0 is None:
            rnn_out, h_n = self.rnn(x)      # [B, T, H], [1, B, H]
        else:
            rnn_out, h_n = self.rnn(x, h0)
        y_hat = self.head(h_n.squeeze(0)) 
        B = y_hat.size(0)
        y_hat = y_hat.view(B, self.output_size, self.input_size)  # [B, horizon, C]
        return rnn_out, y_hat

def copy_rnn_weights(official_rnn: RNN, custom_rnn: MyRNN):
    """
    让自定义 MyRNN 和官方封装 RNN 有一样的参数
    """
    with torch.no_grad():
        W_ih = official_rnn.rnn.weight_ih_l0      # [H, C]
        W_hh = official_rnn.rnn.weight_hh_l0      # [H, H]
        b_ih = official_rnn.rnn.bias_ih_l0        # [H]
        b_hh = official_rnn.rnn.bias_hh_l0        # [H]

        # 对应到我们自定义的 cell
        custom_rnn.cell.i2h.weight.copy_(W_ih)
        custom_rnn.cell.h2h.weight.copy_(W_hh)
        custom_rnn.cell.h2h.bias.copy_(b_ih + b_hh)

        # 复制 head
        custom_rnn.head.weight.copy_(official_rnn.head.weight)
        custom_rnn.head.bias.copy_(official_rnn.head.bias)


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
    一维数据
    series: [n_points]
    window_size: T
    horizon: 预测步数
    X: [N, T, 1] 
    y: [N, horizon, 1]
    
    二维数据
    series: [n_points, C]
    window_size: T
    horizon: 预测步数
    X: [N, T, C] 
    y: [N, horizon, C]
    """
    X, y = [], []
    if series.dim() == 1:
        series = series.unsqueeze(-1)  # [n_points] -> [n_points, 1]
    
    n_points, n_features = series.shape
    for i in range(n_points - window_size - horizon + 1):
        window = series[i : i + window_size]                     # [T, C]
        target = series[i + window_size : i + window_size + horizon]  # [horizon, C]
        X.append(window)
        y.append(target)

    X = torch.stack(X)  # [N, T, C]
    y = torch.stack(y)  # [N, horizon, C]
    
    return X, y

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


# 测试代码 =========================
# set_seed(123456)
feature_size = 1
hidden_size = 8
series = generate_series(n_points=300, feature_size=feature_size)  # [300, 1]
window_size = 20
horizon = 3
X, y = make_dataset(series, window_size, horizon) 
print(f"X.shape: {X.shape}, y.shape: {y.shape}")
output_size = y.shape[1]
X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=0.2,shuffle=False)
tr_dataset = TensorDataset(X_tr, y_tr)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(tr_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 官方RNN
rnn = RNN(input_size=feature_size, hidden_size=hidden_size, output_size=output_size)
h, y_pred = rnn(X[:5])  # 测试前5个样本
print(f"h.shape: {h.shape}, y_pred.shape: {y_pred.shape}")
m_rnn, last_tr_loss_rnn, last_val_loss_rnn = train_model(rnn, train_loader=train_loader, val_loader=val_loader, 
                    save_directory="Save_RNN", num_epochs=100, lr=1e-2)

# 手写RNN
myrnn = MyRNN(input_size=feature_size, hidden_size=hidden_size, output_size=output_size)
h, y_pred = myrnn(X[:5])  # 测试前5个样本
print(f"h.shape: {h.shape}, y_pred.shape: {y_pred.shape}")

copy_rnn_weights(rnn, myrnn)

m_myrnn, last_tr_loss_myrnn, last_val_loss_myrnn = train_model(myrnn, train_loader=train_loader, val_loader=val_loader,
                    save_directory="Save_MyRNN",num_epochs=100, lr=1e-2)

# 比较 和官方RNN结果相似，误差在0.01内
print(f"Train: {last_tr_loss_rnn:.4f}, Val: {last_val_loss_rnn:.4f}, Diff: {abs(last_tr_loss_rnn-last_val_loss_rnn):.4f}")
print(f"Train: {last_tr_loss_myrnn:.4f}, Val: {last_val_loss_myrnn:.4f}, Diff: {abs(last_tr_loss_myrnn-last_val_loss_myrnn):.4f}")
