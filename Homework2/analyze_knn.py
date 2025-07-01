#!/usr/bin/env python3
"""
分析 kNN 分类器在不同 k 值下准确率不变的原因
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载 Iris 数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

print("=== Iris 数据集信息 ===")
print(f"总样本数: {len(X)}")
print(f"特征数: {X.shape[1]}")
print(f"类别数: {len(np.unique(y))}")
print(f"各类别样本数: {[list(y).count(i) for i in range(3)]}")
print()

# 使用与原代码相同的随机种子进行分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print("=== 数据分割信息 ===")
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print(f"测试集各类别分布: {[list(y_test).count(i) for i in range(3)]}")
print(f"测试集标签: {y_test}")
print()

# 测试不同的 k 值
k_values = [1, 3, 5, 7]
results = []

print("=== 不同 k 值的结果 ===")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 计算错误分类的样本
    errors = np.sum(y_test != y_pred)

    results.append({
        'k': k,
        'accuracy': accuracy,
        'errors': errors,
        'predictions': y_pred.copy()
    })

    print(f"k={k}: 准确率={accuracy:.4f}, 错误数={errors}")
    print(f"  预测结果: {y_pred}")
    print()

print("=== 分析结果 ===")
print(f"真实标签:   {y_test}")

# 检查所有预测结果是否相同
all_predictions = [r['predictions'] for r in results]
predictions_equal = all(np.array_equal(all_predictions[0], pred) for pred in all_predictions[1:])

if predictions_equal:
    print("⚠️  所有 k 值产生了相同的预测结果！")
    print("这解释了为什么准确率相同。")
else:
    print("✓ 不同 k 值产生了不同的预测结果。")

print()
print("=== 可能的原因 ===")
print("1. 测试集太小（只有38个样本）")
print("2. Iris数据集类别分离度高，容易分类")
print("3. 固定的random_state可能导致特定的数据分布")
print("4. 选择的k值范围可能都落在相同的决策边界内")

# 建议的改进方案
print()
print("=== 建议的改进方案 ===")
print("1. 使用更多不同的random_state值进行多次实验")
print("2. 尝试更大范围的k值（如1,5,10,15,20）")
print("3. 使用交叉验证而不是单次分割")
print("4. 分析具体的分类边界和距离")
