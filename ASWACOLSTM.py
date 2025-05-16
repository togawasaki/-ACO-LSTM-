import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    try:
        # 读取Excel文件
        raw_df = pd.read_excel(filepath, header=None)
        print("原始数据列数:", raw_df.shape[1])

        # 检查数据格式
        if raw_df.shape[1] < 8:
            raise ValueError(f"需要至少8列数据，当前得到{raw_df.shape[1]}列")
        
        # 提取各列数据
        data = {
            'year': raw_df[6].astype(int),
            'month': raw_df[4].astype(int),
            'day': raw_df[5].astype(int),
            'temp_F': pd.to_numeric(raw_df[7], errors='coerce')  # 处理非数值数据
        }
        
        # 创建DataFrame并删除无效数据
        df = pd.DataFrame(data).dropna()
        
        # 构建日期列并设为索引
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.set_index('date', inplace=True)
        
        # 检查并处理重复索引
        if df.index.duplicated().any():
            print("警告: 发现重复的时间戳，将对这些时间点的温度取平均值")
            df = df.groupby(df.index).mean()  # 对重复索引的值取平均
        
        # 异常值处理：当temp_F小于0时
        for idx in df[df['temp_F'] < 0].index:
            try:
                # 获取异常值的位置，处理可能返回slice的情况
                pos = df.index.get_loc(idx)
                
                if isinstance(pos, slice):
                    # 如果是slice，取第一个位置
                    pos = pos.start if pos.start is not None else 0
                elif isinstance(pos, np.ndarray):
                    # 如果是数组，取第一个True的位置
                    pos = np.argmax(pos)
                
                # 寻找前一个有效值
                prev_val = None
                prev_pos = pos - 1
                while prev_pos >= 0 and prev_val is None:
                    if df.iloc[prev_pos]['temp_F'] != -99:
                        prev_val = df.iloc[prev_pos]['temp_F']
                    prev_pos -= 1
                
                # 寻找后一个有效值
                next_val = None
                next_pos = pos + 1
                while next_pos < len(df) and next_val is None:
                    if df.iloc[next_pos]['temp_F'] != -99:
                        next_val = df.iloc[next_pos]['temp_F']
                    next_pos += 1
                
                # 根据找到的有效值进行替换
                if prev_val is not None and next_val is not None:
                    # 前后都有有效值，取平均值
                    df.at[idx, 'temp_F'] = (prev_val + next_val) / 2
                elif prev_val is not None:
                    # 只有前一个有效值
                    df.at[idx, 'temp_F'] = prev_val
                elif next_val is not None:
                    # 只有后一个有效值
                    df.at[idx, 'temp_F'] = next_val
                # 如果前后都没有有效值，则保持不变
                
            except Exception as e:
                print(f"处理索引 {idx} 时出错: {e}")
                continue
        
        print(f"\n成功加载 {len(df)} 条记录")
        print("时间范围:", df.index.min(), "至", df.index.max())

        return df
        
    except Exception as e:
        print(f"加载数据出错: {e}")
        raise

def check_stationarity(series):
    if len(series) < 5:  # ADF检验至少需要5个样本
        raise ValueError(f"有效数据量不足({len(series)})，至少需要5个观测值")
    
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]:.3f}')
    print(f'p-value: {result[1]:.3f}')
    return result[1] > 0.05  # 返回是否非平稳

def prepare_data(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back)])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

class NumpyLSTM:
    def __init__(self, input_size, hidden_size):
        # 初始化参数 - 使用更小的随机值
        combined_size = input_size + hidden_size
        scale = 0.01  # 减小初始化规模
        
        self.Wf = np.random.randn(hidden_size, combined_size) * scale
        self.Wi = np.random.randn(hidden_size, combined_size) * scale
        self.Wo = np.random.randn(hidden_size, combined_size) * scale
        self.Wc = np.random.randn(hidden_size, combined_size) * scale
        self.Wy = np.random.randn(1, hidden_size) * scale
        
        # 初始化偏置项为小的正值
        self.bf = np.ones((hidden_size, 1)) * 0.1
        self.bi = np.ones((hidden_size, 1)) * 0.1
        self.bo = np.ones((hidden_size, 1)) * 0.1
        self.bc = np.zeros((hidden_size, 1))
        self.by = np.zeros((1, 1))
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.caches = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        # 确保输入形状正确
        x = x.reshape(-1, 1)
        h_prev = h_prev.reshape(-1, 1)
        
        # 拼接输入和前一个隐藏状态
        concat = np.vstack((h_prev, x))
        
        # LSTM门计算
        ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        cct = self.tanh(np.dot(self.Wc, concat) + self.bc)
        ct = ft * c_prev + it * cct
        ht = ot * self.tanh(ct)
        yt = np.dot(self.Wy, ht) + self.by
        
        # 保存中间结果用于反向传播
        cache = (x, h_prev, c_prev, ft, it, ot, cct, ct, ht, concat)
        self.caches.append(cache)
        
        return yt, ht, ct
    
    def backward(self, dyt, learning_rate=0.01):
        if not self.caches:
            return 0, 0, 0
            
        cache = self.caches[-1]
        x, h_prev, c_prev, ft, it, ot, cct, ct, ht, concat = cache
        
        # 输出层梯度
        dWy = np.dot(dyt, ht.T)
        dby = dyt
        dht = np.dot(self.Wy.T, dyt)
        
        # 输出门梯度
        dot = dht * self.tanh(ct)
        dot = dot * ot * (1 - ot)
        dWo = np.dot(dot, concat.T)
        dbo = dot
        
        # 细胞状态梯度
        dct = dht * ot * (1 - self.tanh(ct)**2)
        
        # 候选记忆梯度
        dcct = dct * it * (1 - cct**2)
        dWc = np.dot(dcct, concat.T)
        dbc = dcct
        
        # 输入门梯度
        dit = dct * cct * it * (1 - it)
        dWi = np.dot(dit, concat.T)
        dbi = dit
        
        # 遗忘门梯度
        dft = dct * c_prev * ft * (1 - ft)
        dWf = np.dot(dft, concat.T)
        dbf = dft
        
        # 更新参数
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby
        self.Wo -= learning_rate * dWo
        self.bo -= learning_rate * dbo
        self.Wc -= learning_rate * dWc
        self.bc -= learning_rate * dbc
        self.Wi -= learning_rate * dWi
        self.bi -= learning_rate * dbi
        self.Wf -= learning_rate * dWf
        self.bf -= learning_rate * dbf
        
        # 返回对前一个隐藏状态和细胞状态的梯度
        dconcat = (np.dot(self.Wf.T, dft) + np.dot(self.Wi.T, dit) + 
                  np.dot(self.Wo.T, dot) + np.dot(self.Wc.T, dcct))
        dh_prev = dconcat[:self.hidden_size, :]
        dx = dconcat[self.hidden_size:, :]
        
        return dx, dh_prev, dct * ft
    
    def train(self, X, Y, epochs=30, learning_rate=0.01):
        """训练方法 - 确保学习率是标量值"""
        for epoch in range(epochs):
            total_loss = 0
            h_prev = np.zeros((self.hidden_size, 1))
            c_prev = np.zeros((self.hidden_size, 1))
            self.caches = []
            
            # 确保learning_rate是标量
            if isinstance(learning_rate, (np.ndarray, list)):
                lr = float(learning_rate[0])  # 取第一个元素转为浮点数
            else:
                lr = float(learning_rate)  # 直接转为浮点数
            
            for x, y_true in zip(X, Y):
                x = x.reshape(-1, 1)
                y_true = y_true.reshape(-1, 1)
                
                # 前向传播
                y_pred, h_prev, c_prev = self.forward(x, h_prev, c_prev)
                
                # 计算损失
                loss = np.mean((y_pred - y_true)**2)
                total_loss += loss
                
                # 反向传播
                dyt = 2 * (y_pred - y_true) / y_true.size
                dyt = np.clip(dyt, -1, 1)  # 梯度裁剪
                self.backward(dyt, lr)  # 使用标量lr
            
            # 打印训练信息 - 确保所有格式化值都是标量
            if epoch % 10 == 0:
                avg_loss = float(total_loss) / len(X)  # 确保是标量
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {lr:.5f}")

    def predict(self, X):
        predictions = []
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        
        for x in X:
            x = x.reshape(-1, 1)
            y_pred, h_prev, c_prev = self.forward(x, h_prev, c_prev)
            predictions.append(y_pred[0, 0])
        
        return np.array(predictions)

class BaseLSTM:
    """基准LSTM模型"""
    def __init__(self, look_back=3, hidden_size=30):
        self.look_back = look_back
        self.hidden_size = hidden_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def train(self, train_data):
        # 数据归一化
        train_data = self.scaler.fit_transform(train_data)
        
        # 准备训练数据
        X_train, y_train = prepare_data(train_data, self.look_back)
        
        # 创建并训练LSTM
        self.model = NumpyLSTM(input_size=self.look_back, hidden_size=self.hidden_size)
        self.model.train(X_train, y_train, epochs=10, learning_rate=0.01)
    
    def predict(self, test_data):
        # 数据归一化
        test_data = self.scaler.transform(test_data)
        
        # 准备测试数据
        X_test, y_test = prepare_data(test_data, self.look_back)
        
        # 预测
        predictions = self.model.predict(X_test)
        
        # 反归一化
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # 确保长度一致
        min_length = min(len(predictions), len(y_test))
        return predictions[:min_length], y_test[:min_length]

class ARIMAModel:
    def __init__(self, order, seasonal_order=None, test_seasonality=True):

        self.order = order
        self.seasonal_order = seasonal_order
        self.test_seasonality = test_seasonality
        self.model = None
        self.season_length = None
        self.is_seasonal = False
        
    def train(self, train_data):
        if train_data.ndim > 1:
            train_data = train_data.flatten()
       # 拟合模型

        print(f"\n拟合季节性SARIMA模型")
        
        self.model = ARIMA(train_data, order=(1,0,1),seasonal_order=(0,1,1,7))
        self.model = self.model.fit()
    
    def predict(self, test_data, steps=None):
        if steps is None:
            steps = len(test_data)
        steps = min(steps, len(test_data))
        
        # 获取预测结果
        predictions = self.model.forecast(steps=steps)
        
        if hasattr(predictions, 'values'):  # 如果是pandas Series
            predictions = predictions.values
        predictions = np.array(predictions).reshape(-1, 1)
        
        # 准备真实值并确保长度匹配
        y_test = test_data[:steps].reshape(-1, 1)
        
        return predictions, y_test

class ASWACOLSTM:
    def __init__(self, look_back=3, n_ants=50, n_iter=30, 
                 rho=0.2, q=1.0, alpha=1, beta=3,
                 min_window=12, max_window=16,
                 warm_start=False):
        self.look_back = look_back
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.rho = rho
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.min_window = min_window
        self.max_window = max_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.best_hidden_size = min_window
        self.best_window_size = min_window
        self.best_mse = float('inf')
        self.model = None
        self.adaptive_weights = []
        self.warm_start = warm_start
        self.pheromone_units = None
        self.pheromone_window = None
        self.best_solution_history = []
    
    def get_state(self):
        #热启动
        return {
            'best_hidden_size': self.best_hidden_size,
            'best_window_size': self.best_window_size,
            'best_mse': self.best_mse,
            'pheromone_units': self.pheromone_units.copy() if self.pheromone_units is not None else None,
            'pheromone_window': self.pheromone_window.copy() if self.pheromone_window is not None else None,
            'adaptive_weights': self.adaptive_weights.copy(),
            'best_solution_history': self.best_solution_history.copy(),
        }
    def set_state(self, state):
        #热启动
        self.best_hidden_size = state['best_hidden_size']
        self.best_window_size = state['best_window_size']
        self.best_mse = state['best_mse']
        self.pheromone_units = state['pheromone_units'].copy() if state['pheromone_units'] is not None else None
        self.pheromone_window = state['pheromone_window'].copy() if state['pheromone_window'] is not None else None
        self.best_solution_history = state['best_solution_history'].copy()
        
        if self.model is None:
                self.model = NumpyLSTM(input_size=self.look_back, hidden_size=self.best_hidden_size)
    
    def initialize_pheromones(self, min_units, max_units, min_win, max_win):
        # 判断热启动或是冷启动并执行
        if self.warm_start and self.pheromone_units is not None and self.pheromone_window is not None:
            # 热启动时调整信息素矩阵大小（如果参数范围变化）
            old_units_size = len(self.pheromone_units)
            new_units_size = max_units - min_units + 1
            if old_units_size != new_units_size:
                # 线性插值调整信息素矩阵大小
                x_old = np.linspace(0, 1, old_units_size)
                x_new = np.linspace(0, 1, new_units_size)
                self.pheromone_units = np.interp(x_new, x_old, self.pheromone_units)
            old_window_size = len(self.pheromone_window)
            new_window_size = max_win - min_win + 1
            if old_window_size != new_window_size:
                x_old = np.linspace(0, 1, old_window_size)
                x_new = np.linspace(0, 1, new_window_size)
                self.pheromone_window = np.interp(x_new, x_old, self.pheromone_window)
        else:
            # 冷启动初始化信息素
            self.pheromone_units = np.ones(max_units - min_units + 1)
            self.pheromone_window = np.ones(max_win - min_win + 1)
    def calculate_adaptive_weights(self, data_window):
        """计算自适应权重"""
        mean_val = np.mean(data_window)
        std_val = np.std(data_window)
        trend = (data_window[-1] - data_window[0]) / len(data_window)
        
        volatility_weight = min(1.0, std_val / (mean_val + 1e-10))
        trend_weight = abs(trend) * 10
        recency_weight = 1.2
        
        total = volatility_weight + trend_weight + recency_weight
        weights = {
            'volatility': volatility_weight / total,
            'trend': trend_weight / total,
            'recency': recency_weight / total
        }
        
        self.adaptive_weights.append(weights)
        return weights
    
    def evaluate_ant(self, hidden_size, window_size, train_data):
        try:
            # 1. 参数有效性检查
            if window_size < self.look_back + 2 or window_size < self.min_window:
                return float('inf')
                
            # 2. 数据预处理（保持scaler一致性）
            scaled_data = self.scaler.transform(train_data.reshape(-1, 1)).flatten()
            total_samples = len(scaled_data)
            
            # 3. 窗口大小自适应调整
            effective_window = min(window_size, total_samples - self.look_back - 1)
            if effective_window < self.min_window:
                return float('inf')
            
            # 4. 模型初始化（增加隐藏层维度检查）
            hidden_size = max(4, min(hidden_size, 256))  # 限制隐藏层大小范围
            model = NumpyLSTM(input_size=self.look_back, 
                            hidden_size=hidden_size)
            
            # 5. 多指标评估
            total_mse = 0
            total_mae = 0
            processed_samples = 0
            best_window_loss = float('inf')
            
            # 6. 改进的滑动窗口训练
            stride = max(1, effective_window // 4)  # 动态步长
            for start_idx in range(0, total_samples - effective_window, stride):
                end_idx = start_idx + effective_window
                window_data = scaled_data[start_idx:end_idx]
                
                # 准备数据
                X, y = self.prepare_data(window_data, self.look_back)
                if len(X) < 10:  # 最小样本量要求
                    continue
                    
                # 计算自适应参数
                weights = self.calculate_adaptive_weights(window_data)
                lr = 0.01 * (1 + weights['trend'])
                
                # 训练与验证拆分（防止过拟合）
                split_idx = int(0.8 * len(X))
                X_train, y_train = X[:split_idx], y[:split_idx]
                X_val, y_val = X[split_idx:], y[split_idx:]
                
                # 模型训练（增加梯度裁剪）
                model.train(X_train, y_train, 
                        epochs=8,  # 适当增加epoch
                        learning_rate=lr,
                        max_grad_norm=1.0)  # 新增梯度裁剪
                
                # 验证集评估
                y_pred = np.array([model.predict(x.reshape(-1,1), 
                                            np.zeros((hidden_size,1)), 
                                            np.zeros((hidden_size,1)))[0][0,0] 
                                for x in X_val])
                
                # 多指标计算
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                # 动态加权损失
                loss = 0.6*mse + 0.4*mae  # 综合指标
                loss *= (1 + weights['volatility'])
                # 更新总损失
                total_mse += mse * len(y_val)
                total_mae += mae * len(y_val)
                processed_samples += len(y_val)
                
                # 早停机制（基于窗口性能）
                if loss < best_window_loss * 0.98:
                    best_window_loss = loss
                else:
                    break  # 当前窗口性能下降则跳过后续
                
            # 7. 返回综合评估指标（优先MSE，兼顾MAE）
            if processed_samples == 0:
                return float('inf')
                
            avg_mse = total_mse / processed_samples
            avg_mae = total_mae / processed_samples
            
            # 最终得分（可调整权重）
            final_score = 0.7*avg_mse + 0.3*avg_mae
            return final_score
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return float('inf')
    
    def train(self, train_data):
        # 参数搜索范围
        min_units, max_units = 10, 100
        min_win, max_win = self.min_window, self.max_window
        
        # 初始化或热启动信息素矩阵
        self.initialize_pheromones(min_units, max_units, min_win, max_win)
        
        # 如果热启动且有历史最佳解，调整初始信息素
        if self.warm_start and self.best_solution_history:
            last_best = self.best_solution_history[-1]
            u_idx = last_best['hidden_size'] - min_units
            w_idx = last_best['window_size'] - min_win
            
            if 0 <= u_idx < len(self.pheromone_units):
                self.pheromone_units[u_idx] *= 1.5  # 增强上次最佳解的信息素
            
            if 0 <= w_idx < len(self.pheromone_window):
                self.pheromone_window[w_idx] *= 1.3
        
        for iteration in range(self.n_iter):
            ants_solutions = []
            ants_mse = []
            
            for _ in range(self.n_ants):
                # 基于信息素选择window_size
                prob_window = (self.pheromone_window ** self.alpha) * ((1 / (np.arange(len(self.pheromone_window)) + 1)) ** self.beta)
                prob_window /= prob_window.sum()
                win_idx = np.random.choice(range(len(self.pheromone_window)), p=prob_window)
                window = win_idx + min_win
                window = max(self.min_window, min(self.max_window, window))

                # 基于信息素选择hidden_size
                prob_units = (self.pheromone_units ** self.alpha) * ((1 / (np.arange(len(self.pheromone_units)) + 1)) ** self.beta)
                prob_units /= prob_units.sum()
                units_idx = np.random.choice(range(len(self.pheromone_units)), p=prob_units)
                units = units_idx + min_units

                # 评估蚂蚁选择的解
                mse = self.evaluate_ant(units, window, train_data)
                ants_solutions.append((units_idx, win_idx))
                ants_mse.append(mse)
                # 更新全局最优
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_hidden_size = units
                    self.best_window_size = window
                    self.best_solution_history.append({
                        'iteration': iteration,
                        'hidden_size': units,
                        'window_size': window,
                        'mse': mse
                    })
            
            # 更新信息素
            self.pheromone_units *= (1 - self.rho)
            self.pheromone_window *= (1 - self.rho)
            
            # 信息素增强
            for (u_idx, w_idx), mse in zip(ants_solutions, ants_mse):
                reinforcement = self.q / (mse + 1e-10)
                self.pheromone_units[u_idx] += reinforcement
                self.pheromone_window[w_idx] += reinforcement * 0.8
            
            print(f"Iteration {iteration+1}, Best Units: {self.best_hidden_size}, "
                  f"Best Window: {self.best_window_size}, Best MSE: {self.best_mse:.6f}")
        
        # 用最佳参数训练最终模型
        self.train_final_model(train_data)
    
    def train_final_model(self, train_data):
        """使用最佳参数训练最终模型（支持热启动）"""
        scaled_data = self.scaler.fit_transform(train_data)
        total_samples = len(scaled_data)
        
        # 确保window_size合理
        window_size = min(self.best_window_size, total_samples - self.look_back - 1)
        window_size = max(window_size, self.min_window)
        
        # 初始化模型（热启动时重用已有模型）
        if self.model is None:
            self.model = NumpyLSTM(input_size=self.look_back, hidden_size=self.best_hidden_size)
        elif self.model.hidden_size != self.best_hidden_size:
            # 如果最佳hidden_size变化，重新初始化模型
            self.model = NumpyLSTM(input_size=self.look_back, hidden_size=self.best_hidden_size)
        
        for start_idx in range(0, total_samples - window_size, max(1, window_size // 2)):
            end_idx = start_idx + window_size
            if end_idx > total_samples:
                end_idx = total_samples
            
            window_data = scaled_data[start_idx:end_idx]
            X_train, y_train = prepare_data(window_data, self.look_back)
            
            if len(X_train) == 0:
                continue
            
            window_weights = self.calculate_adaptive_weights(window_data)
            adaptive_lr = float(0.01 * (1 + window_weights['trend']))
            
            # 如果是热启动且不是第一次训练，使用更小的学习率
            if self.warm_start and self.best_solution_history:
                adaptive_lr *= 0.5
            
            self.model.train(X_train, y_train, epochs=10, learning_rate=adaptive_lr)
    
    def predict(self, test_data):
        """预测方法"""
        test_data = self.scaler.transform(test_data)
        predictions = []
        h_prev = np.zeros((self.best_hidden_size, 1))
        c_prev = np.zeros((self.best_hidden_size, 1))
        
        # 使用滑动窗口进行预测
        for i in range(len(test_data) - self.look_back):
            x = test_data[i:i+self.look_back]
            if len(x) < self.look_back:
                x = np.pad(x, (self.look_back - len(x), 0), 'constant')
            
            x = x.reshape(-1, 1)
            y_pred, h_prev, c_prev = self.model.forward(x, h_prev, c_prev)
            predictions.append(y_pred[0, 0])
        
        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test = test_data[self.look_back:].reshape(-1, 1)
        y_test = self.scaler.inverse_transform(y_test)
        
        return predictions, y_test

def evaluate_model(y_true, y_pred, model_name, time_taken):
    """评估模型性能"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} 性能评估:")
    print(f"RMSE: {rmse:.4f}") #均方根误差
    print(f"MAE: {mae:.4f}") #平方误差
    print(f"R²: {r2:.4f}") #R方误差
    print(f"预测耗时: {time_taken:.2f}秒")
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Time': time_taken
    }

def plot_predictions(y_true, lstm_pred, arima_pred, aco_pred):
    plt.figure(figsize=(14, 7))
    
    # 绘制真实值
    plt.plot(y_true, label='真实值', color='black', linewidth=2, alpha=0.7)
    
    # 绘制各模型预测结果
    plt.plot(lstm_pred, label='基准LSTM', linestyle='--', color='blue')
    plt.plot(arima_pred, label='ARIMA', linestyle='-.', color='green')
    plt.plot(aco_pred, label='ASW-ACO-LSTM', linestyle=':', color='red')
    
    # 添加图例和标签
    plt.legend(fontsize=12)
    plt.title('模型预测结果对比', fontsize=16)
    plt.xlabel('时间步', fontsize=14)
    plt.ylabel('值', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加预测区间
    if len(y_true) == len(arima_pred):
        plt.fill_between(
            range(len(y_true)),
            arima_pred.flatten() * 0.95,
            arima_pred.flatten() * 1.05,
            color='green', alpha=0.1
        )
    
    plt.tight_layout()
    plt.show()



def main():
        # 加载数据
        df = load_data('Guangzhou.xlsx')
        
        # 可视化温度序列
        plt.figure(figsize=(12,6))
        plt.plot(df['temp_F'])
        plt.title('Temperature Time Series')
        plt.show()
        
        # 平稳性检测
        if check_stationarity(df['temp_F']):
            print("数据非平稳，进行差分处理...")
            df['temp_diff'] = df['temp_F'].diff().dropna()
            target = df['temp_diff'].values.reshape(-1, 1)
        else:
            print("数据是平稳的")
            target = df['temp_F'].values.reshape(-1, 1)
        
        # 数据标准化和分割
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(target)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        # 存储结果
        results = []

        #ARIMA模型
        print("\n训练ARIMA模型...")
        arima = ARIMAModel(order=(1,0,3))
        arima.train(train_data.flatten())
        steps = len(test_data) - 3  # 确保不超过长度
        arima_pred, arima_true = arima.predict(test_data, steps=steps)
        results.append(evaluate_model(arima_true, arima_pred, "ARIMA", 0))

         # 基础LSTM模型
        print("\n训练基准LSTM模型...")
        lstm = BaseLSTM(look_back=3, hidden_size=10)
        lstm.train(train_data)
        lstm_pred, lstm_true = lstm.predict(test_data)
        results.append(evaluate_model(lstm_true, lstm_pred, "Baseline LSTM", 0))

        # 运用滑动窗口的蚁群优化LSTM模型
        # 第一次训练（冷启动）
        print("\n训练ASW-ACO-LSTM模型...")
        model = ASWACOLSTM(look_back=3, n_ants=50, n_iter=30, 
                             min_window=170, max_window=190,warm_start=False)
        model.train(train_data)
        # 保存状态
        state = model.get_state()
        # 第二次训练（热启动）
        aco_lstm = ASWACOLSTM(look_back=3, n_ants=50, n_iter=30, 
                             min_window=15, max_window=30,warm_start=True)
        aco_lstm.set_state(state)           
        aco_lstm.train(train_data)
        aco_pred, aco_true = aco_lstm.predict(test_data)
        results.append(evaluate_model(aco_true, aco_pred, "ASW-ACO-LSTM", 0))
        

        # 显示所有结果
        print("\n模型性能对比:")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        plot_predictions(
            y_true=test_data,
            lstm_pred=lstm_pred,
            arima_pred=arima_pred,
            aco_pred=aco_pred
        )
        

if __name__ == "__main__":
    main()