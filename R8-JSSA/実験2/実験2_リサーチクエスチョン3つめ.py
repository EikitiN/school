import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 設定（日本語フォント対応など）
# ==========================================
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Takao', 'IPAexGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 入力ファイル
AVOID_TIME_CSV = "実験2_回避開始時間.csv"

# 処理対象ディレクトリと手信号フラグ
TARGET_CONDITIONS = {
    "障害物_r2m_手信号あり/filtered": "手信号あり",
    "障害物_r2m_手信号なし/filtered": "手信号なし"
}


def extract_subject(filename):
    match = re.match(r"^([A-Za-z])\s*\(", filename)
    if match:
        return match.group(1).upper()
    return None


def main():
    print("==========================================")
    print(" 📊 軌跡図 & 速度変化グラフ 自動生成プログラム")
    print("==========================================")

    if not os.path.exists(AVOID_TIME_CSV):
        print(f"❌ エラー: '{AVOID_TIME_CSV}' が見つかりません。先に回避開始時間を算出してください。")
        return

    # 回避開始時間データの読み込み
    df_avoid = pd.read_csv(AVOID_TIME_CSV)

    for target_dir, condition_label in TARGET_CONDITIONS.items():
        print(fn_msg := f"\n🔍 処理中の条件: {condition_label} ({target_dir})")
        
        if not os.path.exists(target_dir):
            print(f"  ⚠️ ディレクトリが存在しません。スキップします。")
            continue

        csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
        if not csv_files:
            print(f"  ⚠️ CSVファイルが見つかりません。")
            continue

        # --------------------------------------------------
        # 1. 軌跡図の作成（位置の重み付き・理想軌道・赤丸プロット）
        # --------------------------------------------------
        plt.figure(figsize=(10, 8))
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            subj = extract_subject(filename)
            if not subj:
                continue

            try:
                df = pd.read_csv(filepath)
            except:
                continue

            if "Corrected_X(cm)" not in df.columns or "Corrected_Y(cm)" not in df.columns:
                continue

            x = df["Corrected_X(cm)"].values
            y = df["Corrected_Y(cm)"].values
            t = df["Time(s)"].values

            # 位置の重み（カラーマップ: 時間の経過に応じて色が変化）
            plt.scatter(x, y, c=t, cmap='Blues', s=10, alpha=0.5, edgecolors='none')
            plt.plot(x, y, alpha=0.3, label=f"被験者 {subj} ({filename})")

            # 当該ファイルの回避開始時間（T_avoid）をCSVから取得
            match_row = df_avoid[(df_avoid["実験条件"] == target_dir.split("/")[0]) & 
                                 (df_avoid["被験者"] == subj) & 
                                 (df_avoid["ファイル名"] == filename)]
            
            if not match_row.empty:
                t_avoid_val = match_row.iloc[0]["回避開始_T_avoid(s)"]
                if pd.notna(t_avoid_val):
                    # 回避開始時間に最も近いフレームの座標を探す
                    idx_closest = (np.abs(t - t_avoid_val)).argmin()
                    plt.scatter(x[idx_closest], y[idx_closest], color='red', s=100, zorder=5, marker='o', 
                                label=f"{subj} 回避開始" if filename == os.path.basename(filepath) else "")

        plt.title(f"軌跡追跡図 - {condition_label} (位置重み付き・回避開始点赤丸)")
        plt.xlabel("Corrected X (cm)")
        plt.ylabel("Corrected Y (cm)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        
        # 凡例が重なりすぎないように調整
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        
        trajectory_img_path = f"実験2_軌跡図_{condition_label}.png"
        plt.savefig(trajectory_img_path, dpi=300)
        plt.close()
        print(f"  ✅ 軌跡図を保存しました: {trajectory_img_path}")

        # --------------------------------------------------
        # 2. 速度変化グラフの作成（位置または時間軸・重み付き・赤丸プロット）
        # --------------------------------------------------
        plt.figure(figsize=(12, 6))

        for filepath in csv_files:
            filename = os.path.basename(filepath)
            subj = extract_subject(filename)
            if not subj:
                continue

            try:
                df = pd.read_csv(filepath)
            except:
                continue

            if "Speed_5F(cm/s)" not in df.columns or "Time(s)" not in df.columns:
                continue

            t = df["Time(s)"].values
            speed = df["Speed_5F(cm/s)"].values

            # 速度変化のプロット
            plt.plot(t, speed, alpha=0.7, label=f"{subj}: {filename}")

            # 回避開始点のプロット
            match_row = df_avoid[(df_avoid["実験条件"] == target_dir.split("/")[0]) & 
                                 (df_avoid["被験者"] == subj) & 
                                 (df_avoid["ファイル名"] == filename)]
            
            if not match_row.empty:
                t_avoid_val = match_row.iloc[0]["回避開始_T_avoid(s)"]
                if pd.notna(t_avoid_val):
                    idx_closest = (np.abs(t - t_avoid_val)).argmin()
                    plt.scatter(t[idx_closest], speed[idx_closest], color='red', s=80, zorder=5, marker='o')

        plt.title(f"速度変化グラフ (Speed_5F) - {condition_label} (回避開始点赤丸)")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed 5F (cm/s)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()

        speed_img_path = f"実験2_速度変化グラフ_{condition_label}.png"
        plt.savefig(speed_img_path, dpi=300)
        plt.close()
        print(f"  ✅ 速度変化グラフを保存しました: {speed_img_path}")

    print("\n==========================================")
    print(" 🎉 すべての図とグラフの生成が完了しました！")
    print("==========================================")


if __name__ == "__main__":
    main()
