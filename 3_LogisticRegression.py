import pandas as pd
import chardet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


''' エンコーディングに合わせてtitanic.csvファイルを読み込む '''
def read_titanic_csv():
    file_path = 'data/titanic.csv'

    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    df = pd.read_csv(file_path, encoding=encoding) 
    
    # 必要なカラムを抽出
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

    return df


''' 前処理と学習と予測 '''
def logistic_reg(df):
    # 欠損地を補完（簡易的に中央値で）
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # カテゴリ変数(sex)を数値化
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0

    # 説明変数と目的変数

    # データ分割
    x_train, x_test, y_train, y_test = 

    # モデル学習と予測

    return y_test, y_pred # 本当の値（正解）, 予測結果（モデルの答え）


if __name__=="__main__":
    # titanic.csvを読み込み、必要なカラムを抽出
    df = read_titanic_csv()

    # ロジスティック回帰
    y_test, y_pred = logistic_reg(df)

    # 結果評価
    print("===予測結果===")
    print("正解率", accuracy_score(y_test, y_pred))
    print("混同行列:\n", confusion_matrix(y_test, y_pred))
    print("分類レポート:\n", classification_report(y_test, y_pred))


    # グラフに表示
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df[df['Survived']==1], x='Age', fill=True, label='Survived', color='green')
    sns.kdeplot(data=df[df['Survived']==0], x='Age', fill=True, label='Not Survived', color='red')
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/3_age_vs_survival.png')
    plt.show()
    
   

