import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 读取数据
train_data = pd.read_excel('Data4Regression.xlsx', sheet_name='Training Data')
test_data = pd.read_excel('Data4Regression.xlsx', sheet_name='Test Data')

x_train = train_data['x'].values.reshape(-1, 1)
y_train = train_data['y_complex'].values
x_test = test_data['x_new'].values.reshape(-1, 1)
y_test = test_data['y_new_complex'].values


# 生成多项式特征
def make_poly_features(x, degree):

    return np.hstack([x ** i for i in range(1, degree + 1)])


# 测试不同多项式次数（2次到5次）
degrees = [6, 7, 8, 9]
results = []

# 训练和评估不同次数的模型
for d in degrees:
    # 生成多项式特征
    train_poly = make_poly_features(x_train, d)
    test_poly = make_poly_features(x_test, d)

    # 添加截距项
    train_poly = np.c_[train_poly, np.ones(len(train_poly))]
    test_poly = np.c_[test_poly, np.ones(len(test_poly))]

    # 用最小二乘法训练
    xtx = np.dot(train_poly.T, train_poly)
    xtx_inv = np.linalg.inv(xtx)
    xty = np.dot(train_poly.T, y_train)
    theta = np.dot(xtx_inv, xty)

    # 计算误差
    train_pred = np.dot(train_poly, theta)
    test_pred = np.dot(test_poly, theta)
    train_error = np.mean((y_train - train_pred) ** 2)
    test_error = np.mean((y_test - test_pred) ** 2)

    results.append((d, train_error, test_error))

# 可视化结果
plt.figure(figsize=(12, 6))

# 绘制原始数据
plt.scatter(x_train, y_train, c='blue', label='训练数据')
plt.scatter(x_test, y_test, c='red', marker='x', label='测试数据')

# 为每个次数绘制拟合曲线
x_plot = np.linspace(x_train.min(), x_train.max(), 100).reshape(-1, 1)
for d in degrees:
    # 生成多项式特征
    poly_features = make_poly_features(x_plot, d)
    poly_features = np.c_[poly_features, np.ones(len(poly_features))]

    # 训练该次数的模型
    theta = np.dot(np.linalg.inv(np.dot(poly_features.T, poly_features)),
                   np.dot(poly_features.T, y_train))

    plt.plot(x_plot, np.dot(poly_features, theta),
             label=f'{d}次多项式')

plt.title('不同次数多项式拟合对比')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 输出结果分析
print("\n不同次数模型表现：")
print("次数 | 训练误差 | 测试误差")
for d, train_err, test_err in results:
    print(f"{d}次 | {train_err:.4f} | {test_err:.4f}")

