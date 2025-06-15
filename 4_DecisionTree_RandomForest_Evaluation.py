import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. データ読み込み
df = pd.read_csv('data/titanic.csv')
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()

# 2. 前処理
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0
x = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -----------------------------------------
# 3. モデル①：決定木
# -----------------------------------------
'''
決定木の処理
'''
y_pred_tree = tree.predict(x_test)

print("=== 決定木 ===")
print("正解率:", accuracy_score(y_test, y_pred_tree))
print("混同行列:\n", confusion_matrix(y_test, y_pred_tree))
print("分類レポート:\n", classification_report(y_test, y_pred_tree))

# ROC曲線用スコア
y_prob_tree = tree.predict_proba(x_test)[:, 1]
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
auc_tree = roc_auc_score(y_test, y_prob_tree)

# -----------------------------------------
# 4. モデル②：ランダムフォレスト
# -----------------------------------------
'''
ランダムフォレストの処理
'''
y_pred_rf = forest.predict(x_test)

print("=== ランダムフォレスト ===")
print("正解率:", accuracy_score(y_test, y_pred_rf))
print("混同行列:\n", confusion_matrix(y_test, y_pred_rf))
print("分類レポート:\n", classification_report(y_test, y_pred_rf))

# ROC曲線用スコア
y_prob_rf = forest.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

# -----------------------------------------
# 5. ROC曲線のプロット
# -----------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {auc_tree:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('output/4_roc_curve.png')
plt.show()
