# コードの説明
- `2_StudentScoreRegression.py`の説明

## 1. エンコーディング自動検出によるCSV読み込み

```py
def read_csv_auto_encoding(file_path):
    with open(file_path, 'rb') as f:
    raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    df = pd.read_csv(file_path, encoding=encoding)
    return df
```

この関数は、CSVファイルの文字エンコーディングを自動検出し、`pandas.read_csv`でデータフレームとして読み込みます。これにより、エンコーディングエラーを防ぎます[5][2]。

---

## 2. データの確認と可視化
```py
print("データの先頭5行:\n", df.head())
plt.scatter(df["study_hours"], df["test_score"], label="Study Hours", color="blue")
```
CSVから読み込んだデータの先頭5行を表示し、勉強時間（study_hours）とテスト点数（test_score）の散布図を描画します。

---

## 3. 単回帰（単変数線形回帰）
```py
X_single = df[["study_hours"]]
y = df["test_score"]
model_single = LinearRegression()
model_single.fit(X_single, y)
```
「勉強時間（study_hours）」だけを使って「テスト点数（test_score）」を予測する単回帰モデルを作成します。  
単回帰は、1つの説明変数（ここでは勉強時間）で目的変数（テスト点数）を予測する線形モデルです。  
係数（傾き）と切片を出力し、散布図上に回帰直線を描画します。

---

## 4. 重回帰（多変数線形回帰）

```py
X_multi = df[["study_hours", "sleep_hours"]]
model_multi = LinearRegression()
model_multi.fit(X_multi, y)
y_pred_multi = model_multi.predict(X_multi)

```
「勉強時間（study_hours）」と「睡眠時間（sleep_hours）」の2つの説明変数を使って「テスト点数（test_score）」を予測する重回帰モデルを作成します。  
重回帰は、複数の説明変数で目的変数を予測する線形モデルです。  
各説明変数の係数と切片、MSE（平均二乗誤差）、R²（決定係数）を出力します。

---

## 5. 評価指標
```py
mse = mean_squared_error(y, y_pred_multi)
r2 = r2_score(y, y_pred_mult
```

- **MSE（平均二乗誤差）**：予測値と実測値の差の2乗の平均。値が小さいほど予測精度が高い。
- **R²（決定係数）**：モデルが目的変数の分散をどれだけ説明できているかを示す指標。1に近いほど良い。

---

## まとめ

このコードは、CSVデータの自動読み込み・単回帰・重回帰分析・可視化・評価までを一通り実行します。
