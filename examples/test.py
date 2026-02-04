import numpy as np

def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    """恒等函数 (用于输出层，直接返回输入值)"""
    return x

def init_network():
    """
    初始化网络权重和偏置
    网络结构: 
        输入层 (2节点) -> 隐藏层1 (3节点) -> 隐藏层2 (2节点) -> 输出层 (2节点)
    """
    network = {}
    
    # --- 第1层: 输入 2 -> 输出 3 ---
    # 权重 W1 形状: (2, 3)
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    # 偏置 b1 形状: (3,)
    network['b1'] = np.array([0.1, 0.2, 0.3])
    
    # --- 第2层: 输入 3 -> 输出 2 ---
    # 权重 W2 形状: (3, 2)
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    # 偏置 b2 形状: (2,)
    network['b2'] = np.array([0.1, 0.2])
    
    # --- 第3层 (输出层): 输入 2 -> 输出 2 ---
    # 权重 W3 形状: (2, 2)
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    # 偏置 b3 形状: (2,)
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    """
    前向传播函数
    :param network: 包含权重和偏置的字典
    :param x: 输入数据
    :return: 网络输出
    """
    # 获取权重和偏置
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # --- 第1层计算 ---
    # 点积运算: (输入 x 权重) + 偏置
    a1 = np.dot(x, W1) + b1 
    # 激活函数处理 (Sigmoid)
    z1 = sigmoid(a1)        
    
    # --- 第2层计算 ---
    # 这里以上一层的输出 z1 作为输入
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    # --- 第3层 (输出层) 计算 ---
    a3 = np.dot(z2, W3) + b3
    # 输出层通常使用不同的激活函数 (这里是恒等函数，常用于回归问题)
    y = identity_function(a3)
    
    return y

# --- 主程序执行 ---

# 1. 初始化网络
network = init_network()

# 2. 定义输入数据 (形状: (2,))
x = np.array([1.0, 0.5])

# 3. 执行前向传播
y = forward(network, x)

# 4. 打印结果
print(y) 
# 输出结果应为: [0.31682708 0.69627909]