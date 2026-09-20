import os
import glob
import re
import numpy as np
import pandas as pd

# ==========================================
# ⚙️ 設定
# ==========================================
TARGET_DIRS = [
    "r2m_手信号あり/filtered",
    "r2m_手信号なし/filtered",
    "r4m_手信号あり/filtered",
    "r4m_手信号なし/filtered",
]

OUTPUT_FILENAME = "実験1_進行方向変化量_標準偏差集計.csv"


def extract_subject(filename):
    """
    ファイル名から被験者名（アルファベット1文字）を確実に抽出する。
    条件: 先頭が「アルファベット1文字」＋「スペース(任意)」＋「(数字)」で始まること
    （例: "A (1)_tracked.csv" -> "A"）
    """
    match = re.match(r"^([A-Za-z])\s*\(", filename)
    if match:
        return match.group(1).upper()
    return None


def calculate_std_for_file(filepath):
    """
    1つのCSVを読み込み、DC区間の進行方向変化量の標準偏差を計算する。
    戻り値: (標準偏差の値, ステータスメッセージ)
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return np.nan, f"CSV読み込みエラー: {e}"

    # Section列が存在する場合は 'DC' の行のみ抽出
    if "Section" in df.columns:
        df_curve = df[df["Section"].astype(str).str.contains("DC", case=False, na=False)]
    else:
        df_curve = df

    if len(df_curve) < 3:
        return np.nan, f"データ不足 (DC区間が {len(df_curve)} 行しかありません)"

    if "Direction_5F(deg)" not in df_curve.columns:
        return np.nan, "Direction_5F(deg) 列が存在しません"

    dir_5f = df_curve["Direction_5F(deg)"].dropna().values
    if len(dir_5f) < 3:
        return np.nan, "有効な角度データ(NaN以外)が3行未満です"

    # 毎フレームの変化量を計算し、-180〜180度に補正
    diff_dir = np.diff(dir_5f)
    diff_dir_norm = (diff_dir + 180) % 360 - 180

    # 不偏標準偏差を計算 (ddof=1)
    std_val = np.std(diff_dir_norm, ddof=1)
    return std_val, "OK"


def main():
    print("==========================================")
    print(" 📊 進行方向変化量の標準偏差 集計プログラム (Pandas一括集計版)")
    print("==========================================")

    # すべての解析結果をストックするリスト
    raw_data_records = []

    # --------------------------------------------------
    # Step 1: 全ファイルの読み込みと値の計算
    # --------------------------------------------------
    for condition_dir in TARGET_DIRS:
        print(f"\n🔍 検索ディレクトリ: {condition_dir}")

        if not os.path.exists(condition_dir):
            print("  ⚠️ ディレクトリが存在しません。スキップします。")
            continue

        csv_files = glob.glob(os.path.join(condition_dir, "*.csv"))
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            subj = extract_subject(filename)
            
            # 被験者名が抽出できないファイル（summary_*.csvなど）は無視
            if not subj:
                continue

            std_val, status = calculate_std_for_file(filepath)
            
            # 結果をレコードとして保存
            raw_data_records.append({
                "Condition": condition_dir,
                "Subject": subj,
                "Filename": filename,
                "Std": std_val,
                "Status": status
            })

            # ログ出力（成功・失敗の理由を明記）
            if status == "OK":
                print(f"    ✅ {subj} | {filename} -> Std: {std_val:.3f}")
            else:
                print(f"    ⚠️ {subj} | {filename} -> スキップ: {status}")

    # レコードが1つもない場合は終了
    if not raw_data_records:
        print("\n❌ 解析対象のデータが一つも見つかりませんでした。")
        return

    # --------------------------------------------------
    # Step 2: Pandasを利用したクロス集計（Pivot）
    # --------------------------------------------------
    # 生データをDataFrame化
    df_raw = pd.DataFrame(raw_data_records)

    # 正常に計算できたデータ（StdがNaNでないもの）だけを抽出
    df_valid = df_raw.dropna(subset=["Std"]).copy()

    if df_valid.empty:
        print("\n❌ 有効な標準偏差が計算できたファイルがありませんでした。")
        return

    # 実験条件・被験者ごとに複数ファイル（試行）がある場合は平均をとる
    df_grouped = df_valid.groupby(["Condition", "Subject"])["Std"].mean().reset_index()

    # 縦持ちのデータを、被験者を列にした横持ち（クロス集計表）に一発変換
    df_pivot = df_grouped.pivot(index="Condition", columns="Subject", values="Std")

    # 指定した4つのディレクトリ行が必ず存在するようインデックスを再構成（無い場合はNaNで埋まる）
    df_pivot = df_pivot.reindex(TARGET_DIRS)

    # 「全体平均」の列を追加（行ごとの平均値）
    df_pivot["全体平均"] = df_pivot.mean(axis=1)

    # --------------------------------------------------
    # Step 3: 出力フォーマットの整形と保存
    # --------------------------------------------------
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={"Condition": "実験条件"}, inplace=True)

    # カラム名に「標準偏差_」を付与（実験条件と全体平均はそのまま）
    new_cols = []
    for col in df_pivot.columns:
        if col in ["実験条件", "全体平均"]:
            new_cols.append(col)
        else:
            new_cols.append(f"標準偏差_{col}")
    df_pivot.columns = new_cols

    # CSVファイルとして出力
    output_path = os.path.join(os.path.dirname(__file__ if "__file__" in locals() else "."), OUTPUT_FILENAME)
    df_pivot.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n==========================================")
    print(" 🎉 クロス集計とCSV出力が完了しました！")
    print(f" 📄 保存先: {output_path}")
    print("==========================================")
    print("\n【出力データプレビュー】")
    print(df_pivot)


if __name__ == "__main__":
    main()
