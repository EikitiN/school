import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 🎨 フォント＆スタイル設定（日本語対応）
# ==========================================
try:
    import japanize_matplotlib
except ImportError:
    plt.rcParams['font.sans-serif'] = [
        'Hiragino Sans', 'Yu Gothic', 'Meiryo', 
        'TakaoPGothic', 'IPAexGothic', 'IPAGothic', 'MS Gothic'
    ]
    plt.rcParams['axes.unicode_minus'] = False


def parse_condition_rows(df):
    """CSVの1列目（条件ディレクトリ名）から半径と手信号フラグを判定"""
    cond_col = df.columns[0]
    row_map = {}
    
    for idx, val in enumerate(df[cond_col]):
        val_str = str(val)
        r = 'r2m' if 'r2m' in val_str else 'r4m'
        sig = '手信号あり' if 'あり' in val_str else '手信号なし'
        row_map[(r, sig)] = idx
        
    return row_map


def plot_bar_by_radius(df, value_cols, subjects, title, ylabel, output_filename):
    """
    半径2m、半径4mごとにデータをまとめて並べる棒グラフ描画関数
    - データが存在しない被験者はx軸から自動除外
    - 警告を回避するため facecolor を使用
    """
    row_map = parse_condition_rows(df)

    pair_labels = []
    has_signal_vals = []
    no_signal_vals = []
    radius_group = []

    # 半径ごとにデータをループ（r2m -> r4m の順）
    for r in ['r2m', 'r4m']:
        row_has_idx = row_map.get((r, '手信号あり'))
        row_no_idx = row_map.get((r, '手信号なし'))

        for subj_idx, subj in enumerate(subjects):
            col_name = value_cols[subj_idx]

            val_has = df.loc[row_has_idx, col_name] if row_has_idx is not None else np.nan
            val_no = df.loc[row_no_idx, col_name] if row_no_idx is not None else np.nan

            # どちらもNaN（データが存在しない）場合はグラフ要素から除外
            if pd.isna(val_has) and pd.isna(val_no):
                continue

            pair_labels.append(f"{subj}\n({r})")
            has_signal_vals.append(val_has)
            no_signal_vals.append(val_no)
            radius_group.append(r)

    n_pairs = len(pair_labels)
    if n_pairs == 0:
        print(f"⚠️ 表示できるデータがありません: {title}")
        return

    x = np.arange(n_pairs) * 1.6  # 束同士の間隔
    bar_width = 0.45               # 棒の幅

    fig, ax = plt.subplots(figsize=(max(10, n_pairs * 1.2), 6))

    COLOR_R2M_BG = '#EBF5FB'  # ペールブルー
    COLOR_R4M_BG = '#FEF9E7'  # ペールイエロー

    # 半径ごとに背景色を塗り分け
    for i, r in enumerate(radius_group):
        bg_color = COLOR_R2M_BG if r == 'r2m' else COLOR_R4M_BG
        ax.axvspan(x[i] - 0.7, x[i] + 0.7, color=bg_color, zorder=0, alpha=0.85)

    # 棒グラフの描画
    rects1 = ax.bar(x - bar_width/2, has_signal_vals, bar_width, label='手信号あり', color='#2B5B84', edgecolor='black', zorder=3)
    rects2 = ax.bar(x + bar_width/2, no_signal_vals, bar_width, label='手信号なし', color='#E67E22', edgecolor='black', zorder=3)

    # 棒の上に数値ラベルを表示
    ax.bar_label(rects1, fmt='%.2f', padding=3, fontsize=8, zorder=4)
    ax.bar_label(rects2, fmt='%.2f', padding=3, fontsize=8, zorder=4)

    # 軸・タイトルの設定
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)

    # 凡例の作成（facecolorを使用してWarningを完全回避）
    patch_has = mpatches.Patch(facecolor='#2B5B84', edgecolor='black', label='手信号あり')
    patch_no = mpatches.Patch(facecolor='#E67E22', edgecolor='black', label='手信号なし')
    patch_r2m = mpatches.Patch(facecolor=COLOR_R2M_BG, edgecolor='gray', label='背景: 半径2m (r2m)')
    patch_r4m = mpatches.Patch(facecolor=COLOR_R4M_BG, edgecolor='gray', label='背景: 半径4m (r4m)')

    ax.legend(handles=[patch_has, patch_no, patch_r2m, patch_r4m], loc='upper right', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"✨ [グラフ作成完了] -> {output_filename}")


def plot_frequency_polygon(df, hist_cols, radius, output_filename):
    """
    半径ごとに分けた進行方向変化率(5F)の頻度多角形（折れ線グラフ）を作成
    """
    row_map = parse_condition_rows(df)

    # 階級ラベルの整形 (例: 0-2deg -> 0-2°, 30deg以上 -> 30°以上)
    bin_labels = []
    for c in hist_cols:
        match = re.search(r'(\d+-\d+deg|30deg以上)', c)
        if match:
            lbl = match.group(1).replace('deg', '°')
            bin_labels.append(lbl)
        else:
            bin_labels.append(c)

    # 行インデックスの取得
    row_has = row_map.get((radius, '手信号あり'))
    row_no = row_map.get((radius, '手信号なし'))

    vals_has = df.loc[row_has, hist_cols].values.astype(float) if row_has is not None else None
    vals_no = df.loc[row_no, hist_cols].values.astype(float) if row_no is not None else None

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(bin_labels))

    # 手信号あり: 実線
    if vals_has is not None:
        ax.plot(x, vals_has, label='手信号あり', color='#2B5B84', linestyle='-', linewidth=2, marker='o', markersize=6)

    # 手信号なし: 破線
    if vals_no is not None:
        ax.plot(x, vals_no, label='手信号なし', color='#E67E22', linestyle='--', linewidth=2, marker='s', markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, fontsize=10, fontweight='bold')
    ax.set_xlabel('進行方向変化率の階級', fontsize=11, fontweight='bold')
    ax.set_ylabel('平均度数 (フレーム数)', fontsize=11, fontweight='bold')

    r_label = '半径2m' if radius == 'r2m' else '半径4m'
    ax.set_title(f'進行方向変化率(5F)の頻度多角形 ({r_label}・全体平均)', fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"✨ [グラフ作成完了] -> {output_filename}")


def main():
    csv_file = '実験1.csv'
    if not os.path.exists(csv_file):
        print(f"❌ エラー: '{csv_file}' が見つかりません。")
        return

    df = pd.read_csv(csv_file)
    print("📊 '実験1.csv' を読み込みました。グラフ生成を開始します...\n")

    # 1. 偏差絶対値列の抽出
    dev_cols = [c for c in df.columns if '偏差' in c]
    dev_subjects = ['全体平均' if '全体平均' in c else c.split('_')[-1] for c in dev_cols]

    # 2. 速度列の抽出
    speed_cols = [c for c in df.columns if '速度' in c]
    speed_subjects = ['全体平均' if '全体平均' in c else c.split('_')[-1] for c in speed_cols]

    # 3. 度数分布（全体平均）列の抽出
    hist_cols = [c for c in df.columns if '進行方向変化率' in c and '全体平均' in c]

    # --- グラフ1: カーブ中における絶対偏差 ---
    if dev_cols:
        plot_bar_by_radius(
            df, dev_cols, dev_subjects,
            title='カーブ中における絶対偏差',
            ylabel='偏差の絶対値平均 [cm]',
            output_filename='実験1_偏差絶対値_比較.png'
        )

    # --- グラフ2: カーブ中における走行速度 ---
    if speed_cols:
        plot_bar_by_radius(
            df, speed_cols, speed_subjects,
            title='カーブ中における走行速度',
            ylabel='速度平均 [cm/s]',
            output_filename='実験1_速度_比較.png'
        )

    # --- グラフ3 & 4: 進行方向変化率の頻度多角形（半径2m / 半径4m） ---
    if hist_cols:
        plot_frequency_polygon(
            df, hist_cols, radius='r2m',
            output_filename='実験1_進行方向変化率_r2m.png'
        )
        plot_frequency_polygon(
            df, hist_cols, radius='r4m',
            output_filename='実験1_進行方向変化率_r4m.png'
        )

    print("\n==========================================")
    print(" 🎉 すべての修正版グラフ画像が出力されました！")
    print("==========================================")


if __name__ == '__main__':
    main()
