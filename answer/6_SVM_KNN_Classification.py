import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# データ読み込み
csv_file = "data/titanic.csv"
df = pd.read_csv(csv_file)

# 欠損値補完（Age：中央値、Embarked：最頻値）
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 不要な列を除外（名前・チケット番号など）
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# カテゴリ変数の変換
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 説明変数と目的変数
X = df.drop('Survived', axis=1)
y = df['Survived']

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徴量の標準化（SVMやk-NNに必要）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# モデル①：サポートベクターマシン（SVM）
# ----------------------------------------
svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("=== サポートベクターマシン ===")
print("正解率:", accuracy_score(y_test, y_pred_svm))
print("混同行列:\n", confusion_matrix(y_test, y_pred_svm))
print("分類レポート:\n", classification_report(y_test, y_pred_svm))

# ----------------------------------------
# モデル②：k-近傍法（k-NN）
# ----------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("=== k-近傍法 ===")
print("正解率:", accuracy_score(y_test, y_pred_knn))
print("混同行列:\n", confusion_matrix(y_test, y_pred_knn))
print("分類レポート:\n", classification_report(y_test, y_pred_knn))
