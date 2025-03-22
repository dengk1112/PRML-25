import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 读取数据
# 读取Excel文件（确保文件在当前文件夹）
train_data = pd.read_excel('Data4Regression.xlsx', sheet_name='Training Data')
test_data = pd.read_excel('Data4Regression.xlsx', sheet_name='Test Data')

# 提取训练数据
x_train = train_data['x'].values.reshape(-1, 1)  # 变成二维数组
y_train = train_data['y_complex'].values

# 提取测试数据
x_test = test_data['x_new'].values.reshape(-1, 1)
y_test = test_data['y_new_complex'].values

# 给数据添加截距项
x_train_with_bias = np.c_[x_train, np.ones((len(x_train), 1))]
x_test_with_bias = np.c_[x_test, np.ones((len(x_test), 1))]

#最小二乘法
# 公式：theta = (X^T X)^-1 X^T y
xtx = np.dot(x_train_with_bias.T, x_train_with_bias)  # 计算X转置乘X
xtx_inv = np.linalg.inv(xtx)  # 求逆矩阵
xty = np.dot(x_train_with_bias.T, y_train)  # X转置乘y
theta_ols = np.dot(xtx_inv, xty)  # 最终参数

# 梯度下降法
theta_gd = np.zeros(2)  # 初始化参数 [斜率, 截距]
learning_rate = 0.02  # 学习率
n_iterations = 1000  # 迭代次数

for i in range(n_iterations):
    # 计算预测值和误差
    predictions = np.dot(x_train_with_bias, theta_gd)
    errors = predictions - y_train

    # 计算梯度（均方误差的导数）
    gradients = 2 / len(x_train) * np.dot(x_train_with_bias.T, errors)

    # 更新参数
    theta_gd -= learning_rate * gradients

# 牛顿法（线性回归特例）
hessian = 2 / len(x_train) * np.dot(x_train_with_bias.T, x_train_with_bias)
gradient = 2 / len(x_train) * np.dot(x_train_with_bias.T, (np.dot(x_train_with_bias, theta_ols) - y_train))
theta_newton = theta_ols - np.dot(np.linalg.inv(hessian), gradient)

# 计算预测结果
# 训练集的预测值
train_pred_ols = np.dot(x_train_with_bias, theta_ols)
train_pred_gd = np.dot(x_train_with_bias, theta_gd)
train_pred_newton = np.dot(x_train_with_bias, theta_newton)

# 测试集的预测值
test_pred_ols = np.dot(x_test_with_bias, theta_ols)
test_pred_gd = np.dot(x_test_with_bias, theta_gd)
test_pred_newton = np.dot(x_test_with_bias, theta_newton)


# 计算误差
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


print("【训练误差】")
print(f"最小二乘法: {mse(y_train, train_pred_ols):.4f}")
print(f"梯度下降法: {mse(y_train, train_pred_gd):.4f}")
print(f"牛顿法:    {mse(y_train, train_pred_newton):.4f}\n")

print("【测试误差】")
print(f"最小二乘法: {mse(y_test, test_pred_ols):.4f}")
print(f"梯度下降法: {mse(y_test, test_pred_gd):.4f}")
print(f"牛顿法:    {mse(y_test, test_pred_newton):.4f}")


plt.figure(figsize=(10, 6))

# 绘制数据点
plt.scatter(x_train, y_train, color='blue', label='训练数据')
plt.scatter(x_test, y_test, color='red', marker='x', label='测试数据')

# 生成拟合线数据
x_line = np.linspace(min(x_train.min(), x_test.min()),
                     max(x_train.max(), x_test.max()), 100)
x_line_with_bias = np.c_[x_line.reshape(-1, 1), np.ones((100, 1))]

# 绘制三条拟合线
plt.plot(x_line, np.dot(x_line_with_bias, theta_ols),
         color='green', label='最小二乘法')
plt.plot(x_line, np.dot(x_line_with_bias, theta_gd),
         color='orange', linestyle='--', label='梯度下降')
plt.plot(x_line, np.dot(x_line_with_bias, theta_newton),
         color='purple', linestyle=':', linewidth=3, label='牛顿法')

plt.xlabel('x ')
plt.ylabel('y ')
plt.title('三种线性回归方法对比')
plt.legend()
plt.grid(True)
plt.show()