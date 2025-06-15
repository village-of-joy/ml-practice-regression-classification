# ロジスティック回帰：データ前処理とモデル学習の解説

このドキュメントでは、Titanic データセットにおけるロジスティック回帰モデルの構築に向けた「前処理」と「学習・予測」部分のコードを詳しく解説します。

## 0. テーマ
- タイタニック号の乗客が「生存したかどうか（0=死亡, 1=生存）」を予測


## 1. 欠損値の補完（Age列）

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
```
- `Age`（年齢）には欠損値（NaN）が含まれているため、**中央値**で補完します。
- `inplace=True` により、元の `df` を直接変更しています。
- なぜ中央値か？  
    → 外れ値（非常に高齢 or 若年）に影響されにくく、バランスが取れているため。

##  2. カテゴリ変数「性別」の数値化

```py
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0
```

### 解説
- 性別（'male' / 'female'）は文字列なので、そのままでは学習に使えません。
- `LabelEncoder` を使って数値（0または1）に変換します。
    - `female` → 0、`male` → 1 に変換されます。

##  3. 説明変数と目的変数の定義

```py
x = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']
```

### 解説
- `x`: モデルに与える「説明変数」（特徴量）
    - `Pclass`: チケットのクラス（1等, 2等, 3等）
    - `Sex`: 性別（数値化済）
    - `Age`: 年齢（補完済）
    - `Fare`: チケット料金
- `y`: 予測したい「目的変数」（0 = 死亡、1 = 生存）


## 4. 学習用・テスト用データに分割
```py
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
### 解説
- 全データの **80% を訓練用、20% をテスト用** に分割。
- `random_state=42` を設定することで、同じ分割が再現できます（再現性の確保）。


##  5. ロジスティック回帰モデルの学習と予測
```py
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

### 解説
- `LogisticRegression()` でロジスティック回帰モデルを定義。
- `max_iter=1000`：反復回数の上限。デフォルトより多めに設定して収束しやすくしています。
- `.fit()`：訓練データを使ってモデルを学習。
- `.predict()`：テストデータに対する生存予測を実施し、`y_pred` に格納。

<br>

# ロジスティック回帰：モデル評価の基礎

前のステップで構築したロジスティック回帰モデルに対して、予測性能を評価します。ここでは主に以下の指標を使用します：

- 正解率（accuracy）
- 混同行列（confusion matrix）
- 分類レポート（classification report）

---

## 正解率（Accuracy）
```py
accuracy_score(y_test, y_pred)
```
- **全体の中で予測が正しかった割合**
- 例: 正解率が 0.79 であれば、79% の正解率
> 💡 単純な指標だが、不均衡データ（例：死亡者が圧倒的に多い場合など）では注意が必要

## 混同行列（Confusion Matrix）
```py
confusion_matrix(y_test, y_pred)
```

| | **予測 = 死亡 (0)** | **予測 = 生存 (1)** |
|-|--------------------|---------------------|
|実際 = 死亡 (0) | True Negative | False Positive |
|実際 = 生存 (1) | False Negative | True Positive |

- モデルの**予測の内訳**を表す2×2の行列
- **どこで間違えたのか**がわかる

## 分類レポート（Classification Report）
```py
classification_report(y_test, y_pred)
```
| 指標        | 説明                            |
| --------- | ----------------------------- |
| Precision | 「生存と予測した中で、本当に生存だった割合」        |
| Recall    | 「実際に生存していた人のうち、正しく生存と予測できた割合」 |
| F1-score  | Precision と Recall のバランスの指標   |
| Support   | 各クラスのサンプル数                    |

> F1-score が高いほど、安定して正しく分類できていることを意味します。
