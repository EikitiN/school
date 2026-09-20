import os
import glob
import numpy as np
import pandas as pd
import re

# ==========================================
# ⚙️ 設定
# ==========================================
TARGET_DIRS = [
    "r2m_手信号あり/filtered",
    "r2m_手信号なし/filtered",
    "r4m_手信号あり/filtered",
    "r4m_手信号なし/filtered"
]

OUTPUT_FILENAME = "実験1.csv"
'''
# 5度刻みの階級境界 (0~5, 5~10, ..., 175~180)
BIN_EDGES = list(range(0, 185, 5))
BIN_LABELS = [f"進行方向変化率_{BIN_EDGES[i]}-{BIN_EDGES[i+1]}deg" for i in range(len(BIN_EDGES)-1)]
'''
# 2度刻み (0~2, 2~4, ..., 28~30, 30以上) の階級境界設定
BIN_EDGES = list(range(0, 32, 2)) + [180]  # [0, 2, 4, ..., 28, 30, 180]

# ラベルの生成
BIN_LABELS = [
    f"進行方向変化率_{BIN_EDGES[i]}-{BIN_EDGES[i+1]}deg"
    for i in range(len(BIN_EDGES) - 2)
]
BIN_LABELS.append("進行方向変化率_30deg以上")  # 最後の階級（30〜180度）を追加

def get_subject_id(file_path):
    """
    ファイル名から被験者識別子を取得します。
    例: "P01_filtered.csv" -> "P01"
    ※お使いのファイル名規則に合わせて調整可能です。
    """
    base_name = os.path.basename(file_path)
    #clean_name = base_name.replace("_tracked", "").replace("_MedianFilter", "").replace(".csv", "")
    clean_name = re.sub(r" \(\d+\)_tracked(_MedianFilter)?\.csv", "", base_name)
    return clean_name


def process_csv_file(file_path):
    """1つのCSVファイルを解析し、各指標を計算する"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  ⚠️ 読み込み失敗: {file_path} ({e})")
        return None

    # カーブ区間のフィルタリング
    if 'Section' in df.columns:
        # Section列に 'Curve', 'カーブ', '1' などが含まれる行を抽出
        curve_mask = df['Section'].astype(str).str.contains('DC', case=False, na=False)
        df_curve = df[curve_mask].copy()
    else:
        df_curve = df.copy()

    if len(df_curve) < 2:
        return None

    # ------------------------------------------
    # 1. カーブ中の偏差の絶対値の平均
    # ------------------------------------------
    dev_abs_mean = df_curve['Deviation(cm)'].abs().mean()

    # ------------------------------------------
    # 2. カーブ中の速度の平均（Corrected_X/Y, Timeから再計算）
    # ------------------------------------------
    dx = df_curve['Corrected_X(cm)'].diff()
    dy = df_curve['Corrected_Y(cm)'].diff()
    dt = df_curve['Time(s)'].diff()

    # Timeの差分が0より大きい有効なフレーム間のみ計算
    valid = dt > 0
    dist = np.sqrt(dx[valid]**2 + dy[valid]**2)
    speed = dist / dt[valid]
    speed_mean = speed.mean()

    # ------------------------------------------
    # 3. 進行方向の5F平均変化率の絶対値の度数分布 (5度刻み)
    # ------------------------------------------
    dir_5f = df_curve['Direction_5F(deg)'].values
    diff_dir = np.diff(dir_5f)
    # -180 ~ +180 度補正
    diff_dir_norm = (diff_dir + 180) % 360 - 180
    abs_diff_dir = np.abs(diff_dir_norm)

    # ヒストグラム計算
    counts, _ = np.histogram(abs_diff_dir, bins=BIN_EDGES)
    hist_dict = {label: count for label, count in zip(BIN_LABELS, counts)}

    return {
        'dev_abs_mean': dev_abs_mean,
        'speed_mean': speed_mean,
        'hist': hist_dict
    }


def main():
    print("==========================================")
    print(" 📊 実験1 データ自動集計プログラム")
    print("==========================================")

    # 全条件フォルダから全被験者リストを作成
    all_subjects = set()
    dir_files_map = {}

    for d in TARGET_DIRS:
        if not os.path.exists(d):
            print(f"⚠️ ディレクトリが見つかりません: {d}")
            dir_files_map[d] = []
            continue

        files = glob.glob(os.path.join(d, "*.csv"))
        dir_files_map[d] = files

        for f in files:
            subj = get_subject_id(f)
            all_subjects.add(subj)

    sorted_subjects = sorted(list(all_subjects))
    print(f"📂 検出された全被験者数: {len(sorted_subjects)} 名")
    print(f"📂 検出された被験者一覧: {sorted_subjects}\n")

    # 集計結果を入れるリスト
    rows = []

    for d in TARGET_DIRS:
        print(f"🔍 処理中: {d}")
        files = dir_files_map.get(d, [])
        
        # 被験者ごとの結果を保持する辞書
        subj_results = {}
        for f in files:
            subj = get_subject_id(f)
            res = process_csv_file(f)
            if res:
                subj_results[subj] = res

        # 行データの構築（1ディレクトリ＝1行）
        row_data = {'ディレクトリ(条件)': d}

        # --- 1. 偏差絶対値の平均 ---
        dev_values = []
        for subj in sorted_subjects:
            val = subj_results[subj]['dev_abs_mean'] if subj in subj_results else np.nan
            row_data[f"偏差絶対値平均_{subj}"] = val
            if pd.notna(val):
                dev_values.append(val)
        row_data["4_偏差絶対値_全体平均"] = np.mean(dev_values) if dev_values else np.nan

        # --- 2. 速度の平均 ---
        speed_values = []
        for subj in sorted_subjects:
            val = subj_results[subj]['speed_mean'] if subj in subj_results else np.nan
            row_data[f"速度平均_{subj}"] = val
            if pd.notna(val):
                speed_values.append(val)
        row_data["5_速度_全体平均"] = np.mean(speed_values) if speed_values else np.nan

        # --- 3. 進行方向変化率の度数分布表 ---
        for label in BIN_LABELS:
            bin_counts = []
            for subj in sorted_subjects:
                val = subj_results[subj]['hist'][label] if subj in subj_results else np.nan
                row_data[f"{label}_{subj}"] = val
                if pd.notna(val):
                    bin_counts.append(val)
            row_data[f"6_{label}_全体平均"] = np.mean(bin_counts) if bin_counts else np.nan

        rows.append(row_data)

    # DataFrame化と保存
    df_result = pd.DataFrame(rows)
    
    # 保存場所（スクリプトと同じディレクトリ）
    output_path = os.path.join(os.path.dirname(__file__ if '__file__' in locals() else '.'), OUTPUT_FILENAME)
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n==========================================")
    print(f" 🎉 集計が完了しました！")
    print(f" 📄 出力ファイル: {output_path}")
    print("==========================================")


if __name__ == '__main__':
    main()
