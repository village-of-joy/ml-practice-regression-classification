# 機械学習 練習用リポジトリ

このリポジトリは、Python・pandas・scikit-learnを使って、機械学習の基本を実践的に学ぶための練習用教材です。  
回帰・分類・前処理の流れに沿って構成されています。



## 線形回帰
### 1_LinearRegression.py
- 勉強時間からテストの点数を予測
- document: [1_LinearRegression.md](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/document/1_LinearRegression.md)



## 単回帰と重回帰
### 2_StudentScoreRegression.py
- 勉強時間、睡眠時間からテストの点数を予測
- document: [2_StudentScoreRegression.md](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/document/2_StudentScoreRegression.md)
- 模範解答: [answer/2_StudentScoreRegression.py](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/answer/2_StudentScoreRegression.py)



## ロジスティック回帰
### 3_LogisticRegression.py
- タイタニック号の乗客が「生存したかどうか（0=死亡, 1=生存）」を予測
- document: [3_LogisticRegression.md](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/document/3_LogisticRegression.md)
- 模範解答: [answer/3_LogisticRegression.py](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/answer/3_LogisticRegression.py)



## 決定木・ランダムフォレスト・ROC曲線
### 4_DecisionTree_RandomForest_Evaluation.py
- タイタニック号の乗客が「生存したかどうか（0=死亡, 1=生存）」を決定木とランダムフォレストで予測し、ROC曲線で性能評価
- document: [4_DecisionTree_RandomForest_Evaluation.md](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/document/4_DecisionTree_RandomForest_Evaluation.md)
- 模範解答: [answer/4_DecisionTree_RandomForest_Evaluation.py](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/answer/4_DecisionTree_RandomForest_Evaluation.py)



## 特徴量エンジニアリング
### 5_FeatureEngineering.py
- 欠損値処理・カテゴリ変数のone-hotエンコーディング・標準化の実践
- document: [5_FeatureEngineering.md](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/document/5_FeatureEngineering.md)
- 模範解答: [answer/5_FeatureEngineering.py](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/answer/5_FeatureEngineering.py)


## 他の分類アルゴリズム（SVM, KNN）
### 6_SVM_KNN_Classification.py
- タイタニック号の乗客データから、SVMとk-NN手法を用いて、乗客が「生存したかどうか」を予測する
- document: [6_SVM_KNN_Classification.md](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/document/6_SVM_KNN_Classification.md)
- 模範解答: [answer/6_SVM_KNN_Classification.py](https://github.com/village-of-joy/ml-practice-regression-classification/blob/main/answer/6_SVM_KNN_Classification.py)