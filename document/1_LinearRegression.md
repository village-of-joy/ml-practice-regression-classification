### 1. 必要なライブラリのインポート
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
```
- `numpy`：数値計算ライブラリ。配列操作に使用。
- `matplotlib.pyplot`:データの可視化（グラフ作成）に使用。
- `sklearn.linear_model.LinearRegression`：線形回帰モデルの作成に使用。
- `pandas`：データ分析用ライブラリ（今回は使用していませんが、インポート済み）。


### 2. データの準備
```py
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
test_scores = np.array([60, 65, 75, 70, 80, 85, 90, 95])
```
- `study_hours`：勉強時間（x軸のデータ）。`.reshape(-1, 1)`で2次元配列（縦ベクトル）に変換。
- `test_scores`：対応するテストの点数（y軸のデータ）。

### 3. 散布図のプロット
```py
plt.scatter(study_hours, test_scores)
plt.title("Study Hours vs Test Scores")
plt.xlabel("Study Hours")
plt.ylabel("Test Scores")
plt.grid(True)
plt.show()
```
- 勉強時間と点数の関係を可視化。
- 散布図（scatter plot）で相関を視覚的に確認。


### 4. 線形回帰モデルの作成と学習
```py
model = LinearRegression()
model.fit(study_hours, test_scores)
```
- `LinearRegression()`でモデル作成
- `.fit()`メソッドでモデルを学習（=最適な直線を当てはめる）。


### 5. 回帰直線の傾きと切片を表示
```py
print(f"傾き（係数）: {model.coef_[0]}")
print(f"切片: {model.intercept_}")
```
- 傾き（係数） → 勉強時間が1時間増えるごとに点数がどれだけ上がるか。
- 切片 → 勉強時間が0時間だったときの予測点数。

### 6. 回帰直線をグラフに追加
```py
prediction_x = np.array([0, 10]).reshape(-1, 1)
prediction_y = model.predict(prediction_x)

plt.scatter(study_hours, test_scores)
plt.plot(prediction_x, prediction_y, color='red', linewidth=2)
plt.title("Study Hours vs Test Scores with Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Test Scores")
plt.grid(True)
plt.show()
```
- `prediction_x`：0時間から10時間までのxの値を使って、予測値を取得。
- `.predict()`でyの予測値を取得し、回帰直線を描画（赤線）。

### 7. 新しいデータで予測
```py
new_hours = np.array([3.5, 9]).reshape(-1, 1)
predicted_scores = model.predict(new_hours)
print(f"3.5時間勉強した場合の予測点数: {predicted_scores[0]}")
print(f"9時間勉強した場合の予測点数: {predicted_scores[1]}")
```
- 3.5時間と9時間勉強した場合の予測点数を出力。
- `.predict()`メソッドで任意の勉強時間に対して点数を予測。