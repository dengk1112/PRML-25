# -*- coding: utf-8 -*-
"""
刚学机器学习的小白代码，包含大量注释和分步说明
"""

# 导入需要的库（如果报错就pip install安装）
import numpy as np  # 数据处理
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split  # 数据划分（虽然题目要求固定数据，但这里保留备用）
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.svm import SVC  # SVM
from sklearn.metrics import accuracy_score  # 准确率计算

try:
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 生成数据
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # 第三维是sin(2t)

    X = np.vstack([np.column_stack([x, y, z]),
                   np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# 生成训练数据：1000个样本（每个类500个）
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)  # 500 * 2=1000

# 生成测试数据：500个样本（每个类250个）
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)  # 250 * 2=500

# 数据预处理
# 创建一个标准化器
scaler = StandardScaler()
# 用训练数据拟合标准化器（计算均值和标准差）
scaler.fit(X_train)
# 对训练数据和测试数据进行标准化转换
X_train_scaled = scaler.transform(X_train)  # 训练集标准化
X_test_scaled = scaler.transform(X_test)     # 测试集用相同的参数标准化

# 训练模型
# 模型1：决策树
dt_model = DecisionTreeClassifier(random_state=42)#数据可重复
dt_model.fit(X_train_scaled, y_train)  # 喂入训练数据

# 模型2：AdaBoost+决策树
base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
ada_model = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=50,  # 使用50个弱分类器
    learning_rate=0.5,  # 学习率，控制每个模型贡献的权重
    random_state=42
)
ada_model.fit(X_train_scaled, y_train)

# 模型3：SVM（尝试三种核函数）
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)

svm_rbf = SVC(kernel='rbf', random_state=42)  # 默认使用RBF核
svm_rbf.fit(X_train_scaled, y_train)

svm_poly = SVC(kernel='poly', degree=3, random_state=42)  # 3次多项式
svm_poly.fit(X_train_scaled, y_train)

# 评估模型（用测试集计算准确率）
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} 准确率：{acc:.4f}")
    return acc

print("\n=== 模型评估结果 ===")
# 评估所有模型
models = [dt_model, ada_model, svm_linear, svm_rbf, svm_poly]
accuracies = []
for model in models:
    acc = evaluate(model, X_test_scaled, y_test)
    accuracies.append(acc)

# 简单结果对比
print("\n=== 各模型准确率对比 ===")
print(f"决策树: {accuracies[0]:.4f}")
print(f"AdaBoost+决策树: {accuracies[1]:.4f}")
print(f"SVM-线性核: {accuracies[2]:.4f}")
print(f"SVM-RBF核: {accuracies[3]:.4f}")
print(f"SVM-多项式核: {accuracies[4]:.4f}")

#分类结果可视化
def plot_3d_predictions_with_accuracy(model, X, y_true, title_prefix):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, cmap='viridis', marker='o', s=20)

    legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
    ax.add_artist(legend1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title_prefix} 分类结果\n准确率: {acc:.4f}")
    plt.tight_layout()
    plt.show()


# 模型及名称
model_names = ["决策树", "AdaBoost+决策树", "SVM-线性核", "SVM-RBF核", "SVM-多项式核"]
for model, name in zip(models, model_names):
    plot_3d_predictions_with_accuracy(model, X_test_scaled, y_test, name)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], X_test_scaled[:, 2], c=y_test, cmap='viridis', marker='o', s=20)
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('测试集分类')
plt.show()