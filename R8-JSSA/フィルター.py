import os
import glob
import numpy as np
import pandas as pd

def filter_tracking_csv(file_path, output_dir):
    """ 通常のフィルタリング処理（5回ループ補完） """
    print(f"🎬 [通常ルート] フィルタリング処理中: {os.path.basename(file_path)}")
    
    df = pd.read_csv(file_path)
    if len(df) < 5:
        print("  ⚠️ データ数が少なすぎるためスキップします。")
        return

    df['Err_Flag'] = ""

    # 1. 離れ小島フレームの除外
    df['Island_Group'] = (df['Frame'].diff() > 3).cumsum()
    longest_group_id = df['Island_Group'].value_counts().idxmax()
    df = df[df['Island_Group'] == longest_group_id].copy()
    df = df.drop(columns=['Island_Group']).reset_index(drop=True)
    
    target_cols = [
        "Raw_X(cm)", "Raw_Y(cm)", "Corrected_X(cm)", "Corrected_Y(cm)", 
        "Angle(deg)", "Deviation(cm)", "Speed_5F(cm/s)", "Direction_5F(deg)"
    ]

    MAX_ITERATIONS = 5
    for iteration in range(MAX_ITERATIONS):
        error_indices = set()
        
        # 条件2: 偏差(H列)のチェック
        i = 1
        while i < len(df):
            d_dev = df.loc[i, 'Deviation(cm)'] - df.loc[i-1, 'Deviation(cm)']
            if abs(d_dev) >= 20:
                found_reverse = False
                for j in range(i+1, len(df)):
                    d_dev_j = df.loc[j, 'Deviation(cm)'] - df.loc[j-1, 'Deviation(cm)']
                    if abs(d_dev_j) >= 15 and np.sign(d_dev_j) != np.sign(d_dev):
                        found_reverse = True
                        error_len = j - i
                        other_len = len(df) - error_len
                        if error_len < other_len:
                            for k in range(i, j): error_indices.add(k)
                        else:
                            for k in range(0, i): error_indices.add(k)
                            for k in range(j, len(df)): error_indices.add(k)
                        i = j
                        break
                if not found_reverse: i += 1
            else: i += 1

        # 条件3: Angle(G列)の逆行チェック
        angles = df['Angle(deg)'].values
        diff_angles = np.diff(angles)
        diff_angles_normalized = (diff_angles + 180) % 360 - 180
        for idx, d_ang in enumerate(diff_angles_normalized):
            if d_ang > 0:
                error_indices.add(idx + 1)

        if not error_indices:
            break
            
        df.loc[list(error_indices), target_cols] = np.nan
        df.loc[list(error_indices), 'Err_Flag'] = 'Err'
        df[target_cols] = df[target_cols].interpolate(method='linear')
        df[target_cols] = df[target_cols].bfill().ffill()
    else:
        # 5回で収束しない場合は、main側でキャッチさせるためにカスタム例外を投げる
        raise RuntimeError("収束エラー（メディアンフィルター救済ルートへ移行します）")

    # 列順序の整理と保存
    column_order = ["Frame", "Time(s)", "Raw_X(cm)", "Raw_Y(cm)", "Corrected_X(cm)", "Corrected_Y(cm)", "Angle(deg)", "Deviation(cm)", "Speed_5F(cm/s)", "Direction_5F(deg)", "Section", "Err_Flag"]
    df = df[column_order]
    
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_path, index=False)
    print(f"  ✨ [FILTERED SAVED] -> {output_path}\n")


def filter_tracking_with_median(file_path, output_dir):
    """ 救済ルート：偏差フィルター ＋ 正常箇所のメディアンフィルター処理 """
    print(f"🔄 [救済ルート] メディアンフィルターを適用します: {os.path.basename(file_path)}")
    
    # オリジナルCSVを再読み込み
    df = pd.read_csv(file_path)
    df['Err_Flag'] = ""

    # 1. 離れ小島フレームの除外
    df['Island_Group'] = (df['Frame'].diff() > 3).cumsum()
    longest_group_id = df['Island_Group'].value_counts().idxmax()
    df = df[df['Island_Group'] == longest_group_id].copy()
    df = df.drop(columns=['Island_Group']).reset_index(drop=True)
    
    target_cols = [
        "Raw_X(cm)", "Raw_Y(cm)", "Corrected_X(cm)", "Corrected_Y(cm)", 
        "Angle(deg)", "Deviation(cm)", "Speed_5F(cm/s)", "Direction_5F(deg)"
    ]

    # 2. 偏差(H列)の20cm/15cmフィルター（ループなし・1回のみ適用）
    error_indices = set()
    i = 1
    while i < len(df):
        d_dev = df.loc[i, 'Deviation(cm)'] - df.loc[i-1, 'Deviation(cm)']
        if abs(d_dev) >= 20:
            found_reverse = False
            for j in range(i+1, len(df)):
                d_dev_j = df.loc[j, 'Deviation(cm)'] - df.loc[j-1, 'Deviation(cm)']
                if abs(d_dev_j) >= 15 and np.sign(d_dev_j) != np.sign(d_dev):
                    found_reverse = True
                    error_len = j - i
                    other_len = len(df) - error_len
                    if error_len < other_len:
                        for k in range(i, j): error_indices.add(k)
                    else:
                        for k in range(0, i): error_indices.add(k)
                        for k in range(j, len(df)): error_indices.add(k)
                    i = j
                    break
            if not found_reverse: i += 1
        else: i += 1

    # 偏差エラー箇所を一時的に NaN に設定
    df.loc[list(error_indices), target_cols] = np.nan
    df.loc[list(error_indices), 'Err_Flag'] = 'Err'

    # 3. それ以外の部分（正常箇所）を前後5フレームでメディアンフィルター
    # 💡 窓サイズ(window)の設定：
    # 「中心とその前後2フレームずつ（計5フレーム）」にする場合は window=5
    # 「自分を中心に、前に5フレーム・後ろに5フレーム（計11フレーム）」にする場合は window=11 に変更してください。
    WINDOW_SIZE = 5 
    df_median = df[target_cols].rolling(window=WINDOW_SIZE, center=True, min_periods=1).median()
    
    # エラーインデックス以外の「正常部分」だけをメディアンフィルター後のデータで上書き
    normal_indices = df.index.difference(list(error_indices))
    df.loc[normal_indices, target_cols] = df_median.loc[normal_indices]

    # 4. 最後に、偏差エラー（NaN）の部分を周囲の綺麗なデータから線形補完
    df[target_cols] = df[target_cols].interpolate(method='linear')
    df[target_cols] = df[target_cols].bfill().ffill()

    # 列順序の整理
    column_order = ["Frame", "Time(s)", "Raw_X(cm)", "Raw_Y(cm)", "Corrected_X(cm)", "Corrected_Y(cm)", "Angle(deg)", "Deviation(cm)", "Speed_5F(cm/s)", "Direction_5F(deg)", "Section", "Err_Flag"]
    df = df[column_order]
    
    # 💡 ファイル名に「_MedianFilter」を付与して保存
    base_name, ext = os.path.splitext(os.path.basename(file_path))
    output_path = os.path.join(output_dir, f"{base_name}_MedianFilter{ext}")
    
    df.to_csv(output_path, index=False)
    print(f"  🎨 [MEDIAN FILTERED SAVED] -> {output_path}\n")


def main():
    print("==========================================")
    print(" 🧪 トラッキングデータ・2段階パイプライン")
    print("==========================================")
    
    target_dir = input("解析CSVが格納されているディレクトリ名を入力してください: ").strip()
    if not os.path.exists(target_dir):
        print(f"[ERROR] ディレクトリ '{target_dir}' が存在しません。")
        return
        
    output_dir = os.path.join(target_dir, "filtered")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    csv_files = [f for f in csv_files if "filtered" not in f]
    
    if not csv_files:
        print("[ERROR] 対象ディレクトリ内にCSVファイルが見つかりません。")
        return
        
    print(f"📂 全 {len(csv_files)} 本のCSVファイルを検出しました。\n")
    
    for file_path in csv_files:
        try:
            # 🟢 まずは通常ルート（角度チェックあり・5回制限）を試す
            filter_tracking_csv(file_path, output_dir)
        except RuntimeError:
            # 🔵 5回ループで収束しなかった場合、自動でメディアン救済ルートへ
            try:
                filter_tracking_with_median(file_path, output_dir)
            except Exception as e:
                print(f"  ❌ [ERROR] 救済処理中に致命的なエラーが発生しました: {e}\n")
        except Exception as e:
            print(f"  ❌ [ERROR] 予期せぬエラーのためスキップします: {e}\n")
            
    print("==========================================")
    print(" 🎉 すべてのCSVファイルの処理が完了しました！")
    print(f"  👉 結果は '{output_dir}' フォルダを確認してください。")
    print("==========================================")

if __name__ == '__main__':
    main()
