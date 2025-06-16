# ステップ6：特徴量エンジニアリング



## 特徴量エンジニアリングとは？

**特徴量エンジニアリング**とは、モデルが学習しやすいように、元データから**有効な特徴（説明変数）を作成・変換・加工**するプロセスです。

良い特徴量は、精度の高い予測モデルを作るうえで非常に重要です。  
データサイエンティストの間では「ゴミを入れればゴミが出る（Garbage In, Garbage Out）」という言葉があるように、前処理と特徴量設計が成果を左右します。



## 本ステップで扱う重要な処理

この演習では、以下の3つのステップを通じて、実践的な特徴量エンジニアリングを学びます。



### ① 欠損値処理（Missing Value Imputation）

データに欠損（NaN）があると多くの機械学習モデルは動作しません。  
そこで、数値データは**平均値**で、カテゴリデータは**最頻値**で補います。

```python
# 数値特徴量（Age）の欠損値を平均で補完
df['Age'] = SimpleImputer(strategy='mean').fit_transform(df[['Age']]).ravel()

# カテゴリ特徴量（Embarked）の欠損値を最頻値で補完
df['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Embarked']]).ravel()
```

### ② One-hotエンコーディング（カテゴリ変数の数値化）
文字列のままでは扱えないカテゴリ変数（例：性別や乗船地）を、数値のダミー変数に変換します。
```py
df_encoded = pd.get_dummies(df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']], columns=['Sex', 'Embarked'])
```
> 例：「Sex」→ Sex_male, Sex_female の2列に分割され、0 or 1の数値で表現されます。


### ③ 標準化（Standardization）
Age や Fare のような数値のスケールがバラバラだと、学習が不安定になります。
標準化により、平均0・分散1の正規化スケールに整えます。
```py
scaler = StandardScaler()
df_encoded[['Age', 'Fare']] = scaler.fit_transform(df_encoded[['Age', 'Fare']])
```
>  この処理は、距離を基にするモデル（SVMやKNN）や、ニューラルネットにおいて特に重要です。

## まとめ
| ステップ            | 処理内容             | 使用ツール                  |
| --------------- | ---------------- | ---------------------- |
| 欠損値補完           | 平均値または最頻値で埋める    | `SimpleImputer`        |
| One-hotエンコーディング | カテゴリを0/1の数値に変換   | `pandas.get_dummies()` |
| 標準化             | 数値のスケールを揃える（平均0） | `StandardScaler`       |
