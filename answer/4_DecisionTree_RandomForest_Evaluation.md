# 決定木とランダムフォレストによる分類モデルの構築と評価

## 概要

Titanicデータセットを使い、決定木（Decision Tree）とランダムフォレスト（Random Forest）の2つのモデルを構築し、  
正解率や混同行列、分類レポート、さらにROC曲線とAUCスコアを用いてモデルの性能を比較します。



## モデル①：決定木 (Decision Tree)

### 1. モデルの生成と学習
```py
tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)
```
- `DecisionTreeClassifier` は決定木モデルを作成するクラスです。
- `random_state=42` は再現性を保つための乱数シードです。
- `.fit(x_train, y_train)` で訓練データを使いモデルを学習します。

### 2. テストデータで予測
```py
y_pred_tree = tree.predict(x_test)
```
- 学習したモデルを使って、テストデータ `x_test` のクラス（生存か死亡か）を予測します。
- 予測結果は `y_pred_tree` に格納されます。

### 3. モデルの性能評価（標準指標）
```py
print("正解率:", accuracy_score(y_test, y_pred_tree))
print("混同行列:\n", confusion_matrix(y_test, y_pred_tree))
print("分類レポート:\n", classification_report(y_test, y_pred_tree))
```
- **正解率 (Accuracy)**  
    全体の中で正しく分類できた割合を示します。

- **混同行列 (Confusion Matrix)**  
    真陽性・偽陽性・真陰性・偽陰性の数を表形式で示し、どの誤分類が多いかを把握できます。

- **分類レポート (Classification Report)**  
    Precision（適合率）、Recall（再現率）、F1スコアなど詳細な評価指標をまとめて表示します。

### 4. ROC曲線とAUCスコア用の予測確率取得
```py
y_prob_tree = tree.predict_proba(x_test)[:, 1]
```
- `predict_proba` は各クラスに属する確率を返します。
- `[:, 1]` は「生存（クラス1）」の確率だけを抜き出しています。

### 5. ROC曲線の計算
```py
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
auc_tree = roc_auc_score(y_test, y_prob_tree)
```
- `roc_curve` は偽陽性率 (False Positive Rate) と真陽性率 (True Positive Rate) を計算し、ROC曲線を描くための値を取得します。
- `roc_auc_score` はROC曲線の下の面積（AUC）を計算し、モデルの総合的な識別能力を示します。1に近いほど良いモデル。


## モデル②：ランダムフォレスト (Random Forest)

### 1. モデルの生成と学習
```py
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(x_train, y_train)
```
- `RandomForestClassifier` は多数の決定木を組み合わせたアンサンブル学習のモデルです。
- `n_estimators=100` は作成する決定木の数を示します。
- `random_state=42` は結果の再現性を保つための乱数シードです。
- `.fit(x_train, y_train)` で訓練データを使いモデルを学習します。


### 2. テストデータで予測
```py
y_pred_rf = forest.predict(x_test)
```
- 学習したランダムフォレストモデルを使ってテストデータのクラス（生存・死亡）を予測し、`y_pred_rf`に格納します。

### 3. モデルの性能評価（標準指標）
```py
print("正解率:", accuracy_score(y_test, y_pred_rf))
print("混同行列:\n", confusion_matrix(y_test, y_pred_rf))
print("分類レポート:\n", classification_report(y_test, y_pred_rf))
```
- **正解率 (Accuracy)**  
    全体の中で正しく分類できた割合を示します。

- **混同行列 (Confusion Matrix)**  
    真陽性・偽陽性・真陰性・偽陰性の数を表形式で示し、どの誤分類が多いかを把握できます。

- **分類レポート (Classification Report)**  
    Precision（適合率）、Recall（再現率）、F1スコアなど詳細な評価指標をまとめて表示します。

### 4. ROC曲線とAUCスコア用の予測確率取得
```py
y_prob_rf = forest.predict_proba(x_test)[:, 1]
```
- `predict_proba` は各クラスに属する確率を返します。
- `[:, 1]` は「生存（クラス1）」の確率を抽出しています。


### 5. ROC曲線の計算
```py
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
```
- `roc_curve` で偽陽性率 (False Positive Rate) と真陽性率 (True Positive Rate) を計算し、ROC曲線を描くための値を得ます。
- `roc_auc_score` はROC曲線の下の面積（AUC）を算出し、モデルの識別性能を評価します。

<br>

# ROC曲線とは？


## ROC曲線の概要

ROC曲線（Receiver Operating Characteristic Curve）は、分類モデルの性能を評価するためのグラフです。  
特に、2クラス分類問題において、モデルの識別能力を視覚的に把握するのに使われます。


## ROC曲線の軸

- **横軸 (False Positive Rate; FPR)：偽陽性率**  
  モデルが「陽性」と誤判定した割合（実際は陰性のデータのうち誤って陽性と判定された割合）

- **縦軸 (True Positive Rate; TPR)：真陽性率（感度）**  
  モデルが正しく「陽性」と判定した割合（実際に陽性のデータのうち正しく陽性と判定された割合）

---

## ROC曲線の意味

- 右上に近い曲線は「良いモデル」を示します。  
  TPRが高く、FPRが低いため、陽性を多く見つけつつ誤検出は少ないことを意味します。

- 45度の対角線はランダムな予測を示します。  
  この線より上に曲線があるほど、モデルの性能はランダム以上であると言えます。


## AUCスコア

- ROC曲線の下の面積（Area Under the Curve, AUC）は、モデルの識別能力を数値化したものです。  
- **AUCの値は0.5から1の範囲で、1に近いほど優れたモデル**と評価されます。  
- 0.5はランダム予測と同じレベルを示します。


## まとめ

ROC曲線とAUCは、しきい値を変化させた場合のモデルの性能を総合的に評価できるため、  
単一の正解率では分かりにくい分類モデルの良し悪しを判断する際に非常に有効です。

---

