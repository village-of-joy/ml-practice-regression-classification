# 6_SVM_KNN_Classification.md

##  概要
このセクションでは、分類問題における代表的な手法である **サポートベクターマシン（SVM）** と **k-近傍法（k-NN）** を使って、タイタニック号の乗客が生存したかどうかを予測するモデルを構築します。

分類モデルの性能比較や、特徴量の標準化の重要性も学びます。


##  モデル①：サポートベクターマシン（SVM）

###  解説

SVMは、分類境界（超平面）を最大限にマージン（余白）を取って引くことで、データを二値に分けるモデルです。非線形な分類にも対応するために「カーネル関数」を使用できます。

- **特徴**：
  - 高次元の特徴空間でも有効
  - マージン最大化による汎化性能の向上
  - ノイズに敏感（ハイパーパラメータ調整が重要）

###  重要コード

```python
svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
```

### 各パラメータの意味：
- `kernel='rbf'`：非線形の分類に有効なRBF（ガウシアン）カーネルを使用
- `C=1`：誤分類にどれだけペナルティを与えるか（正則化項）
- `gamma='scale'`：カーネル幅の自動調整
- `probability=True`：予測時に確率を出力する（ROC曲線などで使用可能）

## モデル②：k-近傍法（k-NN）

### 解説
k-NNは、分類したいデータ点の「近くにあるk個のデータの多数決」でラベルを決定するシンプルな手法です。

### 特徴：
- 学習自体は不要（インスタンスベース）
- 新しいデータの分類時にすべての訓練データを参照
- 距離尺度と標準化が非常に重要（スケールが大きい特徴量が優先されてしまう）

### 重要コード
```py
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
```

### 各パラメータの意味：
- `n_neighbors=5`：近くの5つの点の多数決でクラスを決定
- デフォルトでユークリッド距離を使用
- 必ず標準化（StandardScaler）とセットで使用すること！

## 評価指標（共通）
```py
print("正解率:", accuracy_score(y_test, y_pred))
print("混同行列:\n", confusion_matrix(y_test, y_pred))
print("分類レポート:\n", classification_report(y_test, y_pred))
```

### 指標の意味
- **正解率（accuracy）**：予測がどれだけ当たったか（全体に対する割合）
- **混同行列（confusion matrix）**：実際と予測の組み合わせを表
- **分類レポート**：precision, recall, F1-scoreなど詳細指標を出力

## まとめ
| モデル  | 長所               | 短所                  |
| ---- | ---------------- | ------------------- |
| SVM  | 高次元・非線形に強い、精度が高い | パラメータ調整が難しい、計算コスト高  |
| k-NN | 実装が簡単、パラメータが少ない  | 計算コスト高い、特徴量スケーリング必須 |
