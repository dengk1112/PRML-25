# 1. 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os


# 2. 数据加载与编码
def load_data(filename):
    """加载并预处理原始数据"""
    try:
        # 检查文件存在性
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件 {os.path.abspath(filename)} 不存在")

        # 读取数据
        data = pd.read_csv(filename, parse_dates=['date'])
        #print(data.head(3))

        # 设置时间索引
        data.set_index('date', inplace=True)

        # 处理特殊风向值
        data['wnd_dir'] = data['wnd_dir'].replace('cv', 'CALM')

        # 标签编码风向列
        encoder = LabelEncoder()
        data['wnd_dir'] = encoder.fit_transform(data['wnd_dir'].astype(str))

        # 验证编码结果
        """print("=> 风向编码映射表：")
        for i, cat in enumerate(encoder.classes_):
            print(f"  {cat} => {i}")"""

        return data, encoder

    except Exception as e:
        print(f"\n[错误] 数据加载失败：{str(e)}")
        return None, None


# 3. 数据预处理
def preprocess_data(raw_data, look_back):
    """数据标准化和序列生成"""
    try:

        # 验证输入数据
        if raw_data is None:
            raise ValueError("输入数据为空")

        # 选择特征列
        features = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        data = raw_data[features].copy()

        # 检查数据类型
        # print(data.dtypes)

        # 数据标准化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        #print(scaled_data[:2])

        # 创建时间序列数据集
        #print(f"\n=> 正在生成时间序列（窗口大小：{look_back}）...")
        X, y = [], []
        for i in range(len(scaled_data) - look_back - 1):
            X.append(scaled_data[i:i + look_back, :])  # 输入序列
            y.append(scaled_data[i + look_back, 0])  # 输出值（PM2.5）

        # 转换为Numpy数组
        X = np.array(X)
        y = np.array(y)

        # 数据集分割
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        #print(f"训练集形状：X{X_train.shape} y{y_train.shape}")
        #print(f"测试集形状：X{X_test.shape} y{y_test.shape}")

        return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        print(f"\n[错误] 预处理失败：{str(e)}")
        return None, None, None, None, None


# 4. 模型构建
def create_model(input_shape):
    """构建LSTM模型"""
    model = Sequential()

    # 第一LSTM层
    model.add(LSTM(
        units=128,
        activation='relu',
        input_shape=input_shape,
        return_sequences=True  # 保留序列给下一层
    ))
    model.add(Dropout(0.2))

    # 第二LSTM层
    model.add(LSTM(
        units=64,
        activation='relu',
        return_sequences=False  # 最后一层LSTM
    ))
    model.add(Dropout(0.2))

    # 全连接层
    model.add(Dense(32, activation='relu'))

    # 输出层
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    #model.summary()
    return model


# 5. 主程序
if __name__ == "__main__":
    # 参数配置
    DATA_FILE = "LSTM-Multivariate_pollution.csv"
    TIME_STEPS = 24  # 时间窗口
    EPOCHS = 50  # 训练轮数
    BATCH_SIZE = 32  # 批大小

    # 执行流程
    try:
        # 加载数据
        data, encoder = load_data(DATA_FILE)
        if data is None:
            exit()

        # 预处理
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data, TIME_STEPS)
        if X_train is None:
            exit()

        # 构建模型
        model = create_model((X_train.shape[1], X_train.shape[2]))

        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            verbose=1
        )

        # 评估模型
        print("\n 评估模型...")
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"测试集损失值：{test_loss:.4f}")

        # 预测
        test_pred = model.predict(X_test)


        # 反归一化
        def inverse_transform(data):
            dummy = np.zeros((len(data), len(scaler.feature_names_in_)))
            dummy[:, 0] = data
            return scaler.inverse_transform(dummy)[:, 0]


        y_test_real = inverse_transform(y_test)
        y_pred_real = inverse_transform(test_pred.flatten())

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        print(f"\nRMSE：{rmse:.2f}")

        # 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_real, label="Actual Values")
        plt.plot(y_pred_real, label="Predicted Values", alpha=0.7)
        plt.title("PM2.5 Concentration Prediction Comparison")
        plt.xlabel("Time Step")
        plt.ylabel("Concentration (μg/m³)")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\n[致命错误] 程序异常终止：{str(e)}")