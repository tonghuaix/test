import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# 特征描述
feature_description = {
    'features': tf.io.FixedLenFeature([700], tf.float32),  # 展平后的特征数据，长度为 700
    'label': tf.io.FixedLenFeature([1], tf.int64)         # 标签数据
}

# 解析函数
def _parse_function(proto):
    # 解析 TFRecord 文件
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    
    # 取出特征和标签
    features = parsed_features['features']
    label = parsed_features['label']
    
    # 恢复特征为 (70, 10)
    features = tf.reshape(features, (70, 10))
    
    # 将标签转换为独热编码，8 类分类
    label = tf.one_hot(label, 8)
    
    return features, label

# 读取 TFRecord 文件并加载数据
def load_data(data_dir):
    # 创建 Dataset
    dataset = tf.data.TFRecordDataset([data_dir])
    
    # 使用 map 函数应用解析操作
    dataset = dataset.map(_parse_function)
    
    # 将数据收集成 numpy 数组
    x_data = []
    y_data = []
    
    # 遍历 dataset 获取数据
    for features, label in dataset:
        x_data.append(features.numpy())
        y_data.append(label.numpy())
    
    # 转换为 numpy 数组
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # 使用 squeeze 去除 y_data 中的冗余维度
    y_data = np.squeeze(y_data, axis=1)  # 去除多余的 1 维度
    
    return x_data, y_data

def validate_lstm_model():
# 数据文件路径
    arr=['NoAttack','Flush+Flush','Prime+Probe','Spectrev2','Spectrev4','Flush+Reload','Meltdown','Spectrev1']
    data_dir = './tfRecord/newAllData63000.tfrecord'

# 加载数据
    x_data, y_data = load_data(data_dir)

# 打印数据形状，确保加载正确
    print("x_data shape:", x_data.shape)  # 期望形状 (n, 70, 10)
    print("y_data shape:", y_data.shape)  # 期望形状 (n, 8)
    #x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=1.0, random_state=42)

    # 加载现有的模型
    
    model = load_model('./model/cnn3.keras')
    print("Loaded existing model.")

    start_time = time.time()

    y_pred_probs = model.predict(x_data, verbose=1)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Average Time per Sample: {inference_time / len(x_data):.4f} s")
    print(f"Average Time per Sample: {(inference_time / len(x_data))* 1000:.4f} ms")
    #print(f"x_data Length:{len(x_data)}")
    y_pred = np.argmax(y_pred_probs, axis=1)  # 将概率转换为类别索引
    y_true = np.argmax(y_data, axis=1)  # 假设 y_val 是 one-hot 编码，需要转换为类别索引

    # 生成分类报告
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(len(np.unique(y_true)))], digits=4)
    print(report)

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 逐类别计算 False Positive 和 False Negative
    print("Detailed Metrics per Class:")
    num_classes = conf_matrix.shape[0]
    for i in range(num_classes):
        tp = conf_matrix[i, i]  # True Positive
        fn = np.sum(conf_matrix[i, :]) - tp  # False Negative
        fp = np.sum(conf_matrix[:, i]) - tp  # False Positive
        tn = np.sum(conf_matrix) - (tp + fp + fn)  # True Negative
    
    # Accuracy, Precision, Recall
        accuracy = (tp + tn) / np.sum(conf_matrix) * 100
        precision = tp / (tp + fp) * 100 if tp + fp > 0 else 0
        recall = tp / (tp + fn) * 100 if tp + fn > 0 else 0
        false_positive_rate = fp / (fp + tn) * 100 if fp + tn > 0 else 0
        false_negative_rate = fn / (fn + tp) * 100 if fn + tp > 0 else 0
    
        print(f"{arr[i]}:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.2f}%")
        print(f"  Recall: {recall:.2f}%")
        print(f"  False Positive Rate: {false_positive_rate:.2f}%")
        print(f"  False Negative Rate: {false_negative_rate:.2f}%")
        print()
    
# 仅验证模型
if __name__ == '__main__':
    validate_lstm_model()