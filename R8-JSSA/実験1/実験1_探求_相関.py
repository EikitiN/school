import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 🎨 フォント＆スタイル設定（日本語対応）
# ==========================================
try:
    import japanize_matplotlib
except ImportError:
    plt.rcParams["font.sans-serif"] = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "TakaoPGothic",
        "IPAexGothic",
        "IPAGothic",
        "MS Gothic",
    ]
    plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# ⚙️ 設定（実験条件・マーカー・描画色）
# ==========================================
CONDITIONS = [
    {
        "dir": "r2m_手信号あり/filtered",
        "label": "r2m 手信号あり",
        "marker": "o",  # ●
        "color": "#1f77b4",  # 青
    },
    {
        "dir": "r2m_手信号なし/filtered",
        "label": "r2m 手信号なし",
        "marker": "s",  # ■
        "color": "#ff7f0e",  # オレンジ
    },
    {
        "dir": "r4m_手信号あり/filtered",
        "label": "r4m 手信号あり",
        "marker": "^",  # ▲
        "color": "#2ca02c",  # 緑
    },
    {
        "dir": "r4m_手信号なし/filtered",
        "label": "r4m 手信号なし",
        "marker": "*",  # ★
        "color": "#d62728",  # 赤
    },
]

OUTPUT_FILENAME = "平均速度_絶対偏差_相関散布図.png"


def process_single_csv(file_path):
    """
    1つのCSVファイル（1試行）からDC区間の「速度平均」と「偏差絶対値平均」を計算する
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"⚠️ 読み込み失敗: {file_path} ({e})")
        return None, None

    # カーブ区間（Sectionに 'DC' が含まれる行）の抽出
    if "Section" in df.columns:
        curve_mask = df["Section"].astype(str).str.contains(
            "DC", case=False, na=False
        )
        df_curve = df[curve_mask].copy()
    else:
        df_curve = df.copy()

    if len(df_curve) < 2:
        return None, None

    # 1. 横軸：速度の平均 (cm/s)
    dx = df_curve["Corrected_X(cm)"].diff()
    dy = df_curve["Corrected_Y(cm)"].diff()
    dt = df_curve["Time(s)"].diff()

    valid = dt > 0
    dist = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2)
    speed = dist / dt[valid]
    speed_mean = speed.mean()

    # 2. 縦軸：偏差の絶対値の平均 (cm)
    dev_abs_mean = df_curve["Deviation(cm)"].abs().mean()

    # ※もし縦軸を「進行方向変化率(5F)の絶対値平均」にしたい場合は、上の行をコメントアウトし
    #  以下の3行のコメントアウトを解除してください。
    # dir_5f = df_curve['Direction_5F(deg)'].values
    # diff_dir_norm = (np.diff(dir_5f) + 180) % 360 - 180
    # dev_abs_mean = np.mean(np.abs(diff_dir_norm))

    return speed_mean, dev_abs_mean


def main():
    print("==========================================")
    print(" 📊 平均速度 vs 絶対偏差 相関分析プログラム")
    print("==========================================")

    fig, ax = plt.subplots(figsize=(9, 7))

    summary_info = []

    for cond in CONDITIONS:
        target_dir = cond["dir"]
        label = cond["label"]
        marker = cond["marker"]
        color = cond["color"]

        if not os.path.exists(target_dir):
            print(f"⚠️ ディレクトリが見つかりません: {target_dir}")
            continue

        csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
        x_vals = []
        y_vals = []

        for f in csv_files:
            sp_mean, dev_mean = process_single_csv(f)
            if (
                sp_mean is not None
                and dev_mean is not None
                and not np.isnan(sp_mean)
                and not np.isnan(dev_mean)
            ):
                x_vals.append(sp_mean)
                y_vals.append(dev_mean)

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        if len(x_vals) == 0:
            print(f"⚠️ データが存在しません: {label}")
            continue

        # --- ① 散布図のプロット ---
        ax.scatter(
            x_vals,
            y_vals,
            marker=marker,
            color=color,
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.6,
            label=label,
            zorder=3,
        )

        # --- ② 最小2乗法による回帰直線（細い点線） & 相関係数r の計算 ---
        r_val = np.nan
        if len(x_vals) >= 2:
            # 回帰直線の傾きと切片を計算
            poly = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(min(x_vals), max(x_vals), 100)
            y_line = np.polyval(poly, x_line)

            # 点線で描画
            ax.plot(
                x_line,
                y_line,
                color=color,
                linestyle=":",
                linewidth=1.5,
                zorder=2,
            )

            # 相関係数 r の算出
            r_matrix = np.corrcoef(x_vals, y_vals)
            r_val = r_matrix[0, 1]

        # グラフ下部の集計情報用テキストを準備
        r_str = f"r = {r_val:.3f}" if not np.isnan(r_val) else "r = N/A"
        # ディスプレイ用の記号
        symbol_map = {"o": "●", "s": "■", "^": "▲", "*": "★"}
        disp_symbol = symbol_map.get(marker, marker)

        summary_info.append(
            f"{disp_symbol}  {label}:  {r_str}  (試行数 n={len(x_vals)})"
        )
        print(f"✅ {label}: データ数={len(x_vals)}, 相関係数 {r_str}")

    # --- ③ 軸とタイトルの設定 ---
    ax.set_xlabel("平均速度 [cm/s]", fontsize=12, fontweight="bold")
    ax.set_ylabel("絶対偏差 [cm]", fontsize=12, fontweight="bold")
    ax.set_title(
        "実験条件ごとの平均速度と絶対偏差の相関",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # --- ④ グラフ下の領域に「印・実験条件・相関係数 r」を表示 ---
    if summary_info:
        summary_text = "【各実験条件の相関係数 r】\n" + "   /   ".join(
            summary_info[:2]
        ) + "\n" + "   /   ".join(summary_info[2:])

        # グラフ下にスペースを空けてテキストボックスを配置
        plt.subplots_adjust(bottom=0.22)
        fig.text(
            0.5,
            0.03,
            summary_text,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#F8F9FA",
                edgecolor="gray",
                alpha=0.95,
            ),
        )

    # --- ⑤ 画像の保存 ---
    output_path = os.path.join(
        os.path.dirname(__file__ if "__file__" in locals() else "."),
        OUTPUT_FILENAME,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n==========================================")
    print(f" 🎉 散布図の出力が完了しました！")
    print(f" 📄 保存ファイル: {output_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
