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
df['Age'] = df['Age'].fillna(_____())  # TODO: 年齢の中央値で補完
df['Embarked'] = df['Embarked'].fillna(_____()[0])  # TODO: 最頻値で補完

# 不要な列を除外（名前・チケット番号など）
df.drop([_____, 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)  # TODO: ID列を除外

# カテゴリ変数の変換
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0
df = pd.get_dummies(df, columns=['Embarked'], _____=True)  # TODO: ダミー変数の1つ目を削除

# 説明変数と目的変数
X = df.drop('_____', axis=1)  # TODO: 目的変数の列名を指定
y = df['_____']  # TODO: 同上

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_____, random_state=42)  # TODO: テストサイズを指定

# 特徴量の標準化（SVMやk-NNに必要）
scaler = StandardScaler()
X_train = scaler.fit_transform(_____)  # TODO: 訓練データを標準化
X_test = scaler.transform(_____)       # TODO: テストデータを標準化

# ----------------------------------------
# モデル①：サポートベクターマシン（SVM）
# ----------------------------------------
svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(_____)

print("=== サポートベクターマシン ===")
print("正解率:", accuracy_score(_____, _____))  # TODO: 正解率を表示
print("混同行列:\n", confusion_matrix(_____, _____))  # TODO: 混同行列を表示
print("分類レポート:\n", classification_report(_____, _____))  # TODO: 分類レポートを表示

# ----------------------------------------
# モデル②：k-近傍法（k-NN）
# ----------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(_____, _____)
y_pred_knn = knn_model.predict(_____)

print("=== k-近傍法 ===")
print("正解率:", accuracy_score(_____, _____))
print("混同行列:\n", confusion_matrix(_____, _____))
print("分類レポート:\n", classification_report(_____, _____))
