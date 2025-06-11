import pandas as pd
import chardet
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

''' エンコーディングに合わせてcsvファイルを読み込む '''
def read_csv_auto_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    df = pd.read_csv(file_path, encoding=encoding)
    return df


if __name__=="__main__":

    # CSVの読み込み
    df = read_csv_auto_encoding("./data/student_scores.csv")

    # データの確認
    print("データの先頭5行:\n", df.head())

    # 散布図（勉強時間 vs テスト点数）
    plt.scatter(df["study_hours"], df["test_score"], label="Study Hours", color="blue")
    plt.title("Study Hours vs Test Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Test Score")
    plt.grid(True)
    plt.savefig("output/study_hours_vs_score.png")
    plt.close()

    # 単回帰（勉強時間のみ）
    X_single = df[["study_hours"]]
    y = df["test_score"]

    model_single = LinearRegression()
    model_single.fit(X_single, y)

    print("\n【単回帰モデル】")
    print(f"傾き（係数）: {model_single.coef_[0]:.2f}")
    print(f"切片: {model_single.intercept_:.2f}")

    # 予測線の描画
    x_line = pd.DataFrame({"study_hours": [0, 10]})
    y_pred_line = model_single.predict(x_line)

    plt.scatter(df["study_hours"], df["test_score"], label="real data")
    plt.plot(x_line, y_pred_line, color="red", label="regression line")
    plt.title("simple regression：Study Hours vs Test Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Test Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/linear_regression_single.png")
    plt.close()

    # 重回帰（勉強時間＋睡眠時間）
    X_multi = df[["study_hours", "sleep_hours"]]

    model_multi = LinearRegression()
    model_multi.fit(X_multi, y)

    y_pred_multi = model_multi.predict(X_multi)

    print("\n【重回帰モデル】")
    print(f"係数: {model_multi.coef_}")
    print(f"切片: {model_multi.intercept_:.2f}")

    # 評価指標
    mse = mean_squared_error(y, y_pred_multi)
    r2 = r2_score(y, y_pred_multi)
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.3f}")
