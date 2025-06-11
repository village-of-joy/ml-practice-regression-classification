import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# データの準備
# 勉強時間（時間）
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)  # 2次元配列にする(縦のベクトルの形)
# テストの点数
test_scores = np.array([60, 65, 75, 70, 80, 85, 90, 95])

# 散布図をプロット
plt.scatter(study_hours, test_scores)
plt.title("Study Hours vs Test Scores")
plt.xlabel("Study Hours")
plt.ylabel("Test Scores")
plt.grid(True)
plt.show()


# 線形回帰モデルの作成
model = LinearRegression()
model.fit(study_hours, test_scores)


# 傾きと切片の確認
print(f"傾き（係数）: {model.coef_[0]}")
print(f"切片: {model.intercept_}")

# 予測線の追加
prediction_x = np.array([0, 10]).reshape(-1, 1)
prediction_y = model.predict(prediction_x)

plt.scatter(study_hours, test_scores)
plt.plot(prediction_x, prediction_y, color='red', linewidth=2)
plt.title("Study Hours vs Test Scores with Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Test Scores")
plt.grid(True)
plt.show()

# 新しいデータで予測
new_hours = np.array([3.5, 9]).reshape(-1, 1)
predicted_scores = model.predict(new_hours)
print(f"3.5時間勉強した場合の予測点数: {predicted_scores[0]}")
print(f"9時間勉強した場合の予測点数: {predicted_scores[1]}")