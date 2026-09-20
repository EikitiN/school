import os
import glob
import re
import numpy as np
import pandas as pd

# ==========================================
# ⚙️ 設定
# ==========================================
# 入力ファイル
MU_CSV_FILE = "実験1_方向変化集計結果.csv"
SIGMA_CSV_FILE = "実験1_進行方向変化量_標準偏差集計.csv"

# 探索対象の実験2ディレクトリと、それに対応する実験1の条件名
TARGET_DIRS = {
    "障害物_r2m_手信号あり/filtered": "r2m_手信号あり/filtered",
    "障害物_r2m_手信号なし/filtered": "r2m_手信号なし/filtered"
}

# 出力ファイル
OUTPUT_FILENAME = "実験2_回避開始時間.csv"

# 閾値
ANGLE_THRESHOLD = 27.3


def extract_subject(filename):
    """
    ファイル名から被験者名（アルファベット1文字）を抽出
    例: "A (1)_tracked_MedianFilter.csv" -> "A"
    """
    match = re.match(r"^([A-Za-z])\s*\(", filename)
    if match:
        return match.group(1).upper()
    return None


def get_mu_sigma(df_mu, df_sigma, exp1_cond, subject):
    """
    実験1のデータフレームから特定の条件・被験者のμとσを取得する
    """
    mu_col = f"{subject}_方向変化平均"
    sigma_col = f"標準偏差_{subject}"
    
    try:
        mu = df_mu.loc[exp1_cond, mu_col]
        sigma = df_sigma.loc[exp1_cond, sigma_col]
        return float(mu), float(sigma)
    except KeyError:
        return np.nan, np.nan


def analyze_avoidance(filepath, mu, sigma):
    """
    1つのファイルを解析し、回避開始時間、障害物認識時間、その差分を計算する
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return np.nan, np.nan, np.nan, f"読込エラー: {e}"

    # 必須カラムの確認
    required_cols = ["Time(s)", "Angle(deg)", "Direction_5F(deg)"]
    for col in required_cols:
        if col not in df.columns:
            return np.nan, np.nan, np.nan, f"カラム不足: {col}"

    if len(df) < 5:
        return np.nan, np.nan, np.nan, "データ行数が不足"

    # ==========================================
    # 1. 障害物の見え始める時間 (T_visible) の算出
    # ==========================================
    angle_mask = df["Angle(deg)"] > ANGLE_THRESHOLD
    if angle_mask.any():
        # 初めて閾値を超えたインデックスを取得
        idx_visible = angle_mask.idxmax()
        t_visible = df.loc[idx_visible, "Time(s)"]
    else:
        t_visible = np.nan

    # ==========================================
    # 2. 回避動作開始時間 (T_avoid) の算出
    # ==========================================
    # 5フレーム平均進行方向の毎フレームの変化量を計算
    dir_5f = df["Direction_5F(deg)"].values
    diff_dir = np.append([np.nan], np.diff(dir_5f)) # 行数を合わせるため先頭にNaNを挿入
    diff_dir_norm = (diff_dir + 180) % 360 - 180      # 角度のラップアラウンド補正
    
    df["Dir_Change"] = diff_dir_norm

    # μ±2σ の範囲外かどうかの判定 (True/False)
    lower_bound = mu - (2 * sigma)
    upper_bound = mu + (2 * sigma)
    is_outlier = (df["Dir_Change"] < lower_bound) | (df["Dir_Change"] > upper_bound)

    # 3フレーム連続でTrueになっている箇所を探す
    # rolling(3).sum() が 3 になる最初のインデックスを探す
    rolling_sum = is_outlier.rolling(window=3).sum()
    valid_indices = rolling_sum[rolling_sum == 3].index

    if len(valid_indices) > 0:
        # rollingは「最後」のインデックスを返すため、-2 して「最初」のフレームに戻す
        start_idx = valid_indices[0] - 2
        t_avoid = df.loc[start_idx, "Time(s)"]
    else:
        t_avoid = np.nan

    # ==========================================
    # 3. 差分 (Delay) の算出
    # ==========================================
    if pd.notna(t_avoid) and pd.notna(t_visible):
        t_delay = t_avoid - t_visible
    else:
        t_delay = np.nan

    return t_visible, t_avoid, t_delay, "OK"


def main():
    print("==========================================")
    print(" 🚗 実験2 回避開始時間 算出プログラム")
    print("==========================================")

    # --- 実験1の基準データの読み込み ---
    if not os.path.exists(MU_CSV_FILE) or not os.path.exists(SIGMA_CSV_FILE):
        print(f"❌ エラー: 基準データ ({MU_CSV_FILE} または {SIGMA_CSV_FILE}) が見つかりません。")
        return

    df_mu = pd.read_csv(MU_CSV_FILE, index_col="Directory")
    df_sigma = pd.read_csv(SIGMA_CSV_FILE, index_col="実験条件")

    results = []

    # --- ディレクトリごとの処理 ---
    for exp2_dir, exp1_cond in TARGET_DIRS.items():
        print(f"\n🔍 検索ディレクトリ: {exp2_dir} (基準: {exp1_cond})")

        if not os.path.exists(exp2_dir):
            print("  ⚠️ ディレクトリが存在しません。スキップします。")
            continue

        csv_files = glob.glob(os.path.join(exp2_dir, "*.csv"))
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            subj = extract_subject(filename)
            
            if not subj:
                continue
                
            # 実験1の基準値を取得
            mu, sigma = get_mu_sigma(df_mu, df_sigma, exp1_cond, subj)
            
            if pd.isna(mu) or pd.isna(sigma):
                print(f"    ⚠️ {subj} | {filename} -> スキップ: μまたはσのデータがありません")
                continue

            # 解析実行
            t_vis, t_avd, t_delay, status = analyze_avoidance(filepath, mu, sigma)
            
            # 結果を保存
            results.append({
                "実験条件": exp2_dir.split("/")[0], # フォルダ名から条件名だけを抽出
                "被験者": subj,
                "ファイル名": filename,
                "μ(基準)": round(mu, 3),
                "σ(基準)": round(sigma, 3),
                "障害物認識_T_visible(s)": t_vis,
                "回避開始_T_avoid(s)": t_avd,
                "回避遅れ_T_delay(s)": t_delay,
                "備考": status if status != "OK" else ""
            })

            # ログ表示
            if status == "OK":
                print(f"    ✅ {subj} | {filename} -> T_vis: {t_vis:.2f}s, T_avoid: {t_avd:.2f}s, Delay: {t_delay:.2f}s")
            else:
                print(f"    ⚠️ {subj} | {filename} -> スキップ: {status}")

    # --- 結果の保存 ---
    if not results:
        print("\n❌ 解析対象のデータが一つも見つかりませんでした。")
        return

    df_results = pd.DataFrame(results)
    
    # 見やすくするためにソート
    df_results.sort_values(by=["実験条件", "被験者", "ファイル名"], inplace=True)

    output_path = os.path.join(os.path.dirname(__file__ if "__file__" in locals() else "."), OUTPUT_FILENAME)
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n==========================================")
    print(" 🎉 解析が完了しました！")
    print(f" 📄 保存先: {output_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
