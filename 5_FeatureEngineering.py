import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# データ読み込み
df = pd.read_csv('data/titanic.csv')

# 使う特徴量と目的変数
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

# 欠損値の確認
print("欠損値数:\n", df[features].isnull().sum())

# ====== 1. 欠損値処理 ======
# 数値：Age → _______ で補完
num_imputer = SimpleImputer(strategy='________')   # ←①
df['Age'] = num_imputer.fit_transform(df[['Age']])

# カテゴリ：Embarked → _______ で補完
cat_imputer = SimpleImputer(strategy='________')   # ←②
df['Embarked'] = cat_imputer.fit_transform(df[['Embarked']]).ravel()

# ====== 2. One-hot encoding（Sex, Embarked） ======
# SexとEmbarkedをone-hot化する
df_encoded = pd.get_dummies(df[features], columns=[________])  # ←③

# ====== 3. 標準化（数値特徴量のみ） ======
# AgeとFareを標準化
scaler = StandardScaler()
df_encoded[[________]] = scaler.fit_transform(df_encoded[[________]])  # ←④

# データ分割
X = df_encoded
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=________, random_state=42)  # ←⑤

# 結果確認
print("\n前処理後のデータ（訓練用）:")
print(X_train.head())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# データ読み込み
df = pd.read_csv('data/titanic.csv')

# 使う特徴量と目的変数
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

# 欠損値の確認
print("欠損値数:\n", df[features].isnull().sum())

# ====== 1. 欠損値処理 ======
# 数値：Age → _______ で補完
num_imputer = SimpleImputer(strategy='________')   # ←①
df['Age'] = num_imputer.fit_transform(df[['Age']])

# カテゴリ：Embarked → _______ で補完
cat_imputer = SimpleImputer(strategy='________')   # ←②
df['Embarked'] = cat_imputer.fit_transform(df[['Embarked']]).ravel()

# ====== 2. One-hot encoding（Sex, Embarked） ======
# SexとEmbarkedをone-hot化する
df_encoded = pd.get_dummies(df[features], columns=[________])  # ←③

# ====== 3. 標準化（数値特徴量のみ） ======
# AgeとFareを標準化
scaler = StandardScaler()
df_encoded[[________]] = scaler.fit_transform(df_encoded[[________]])  # ←④

# データ分割
X = df_encoded
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=________, random_state=42)  # ←⑤

# 結果確認
print("\n前処理後のデータ（訓練用）:")
print(X_train.head())
