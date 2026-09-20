import glob
import os
import re
import pandas as pd

# 処理対象のディレクトリリスト
DIRECTORIES = [
    "r2m_手信号あり/filtered",
    "r2m_手信号なし/filtered",
    "r4m_手信号あり/filtered",
    "r4m_手信号なし/filtered",
]

# ファイル名パターン: * (d)_tracked.csv (*: 半角英字1文字, d: 半角数字1桁)
FILE_PATTERN = re.compile(r"^([a-zA-Z])\s*\(\d\)_tracked\.csv$")


def process_csv(file_path):
    """単一のCSVファイルを読み込み、条件に合致する変位（絶対値）を計算してリストで返す"""
    try:
        # csvの読み込み (ヘッダーありを想定)
        df = pd.read_csv(file_path)

        # 列名またはインデックスで指定 (10列目: J列, 11列目: K列)
        # 列名が存在する場合は列名で指定し、念のためフォールバックを用意
        if "Section" in df.columns and "Direction_5F(deg)" in df.columns:
            section_col = df["Section"]
            dir_col = df["Direction_5F(deg)"]
        else:
            section_col = df.iloc[:, 10]  # K列 (0始まりで10)
            dir_col = df.iloc[:, 9]  # J列 (0始まりで9)

        # 一つ上の行との差分（変位）を計算
        # ※変位を絶対値とする場合は .abs() を付与してください。現在はそのままの差分としています。
        diff_series = dir_col.diff().abs()

        # K列(Section)が "DC" である行の差分を抽出
        dc_diffs = diff_series[section_col == "DC"].dropna().tolist()

        return dc_diffs
    except Exception as e:
        print(f"エラー: {file_path} の処理中に問題が発生しました: {e}")
        return []


def main():
    # ディレクトリ別の集計データ構造
    # { dir_name: { 'A': [diff1, diff2, ...], 'B': [...], ... } }
    summary_data = {d: {} for d in DIRECTORIES}

    # 全ファイルの変位を保持するリスト
    all_diffs_global = []

    # 各ディレクトリを走査
    for dir_path in DIRECTORIES:
        if not os.path.exists(dir_path):
            print(f"警告: ディレクトリ '{dir_path}' が見つかりません。スキップします。")
            continue

        files = os.listdir(dir_path)
        for file_name in files:
            match = FILE_PATTERN.match(file_name)
            if match:
                prefix = match.group(1).upper()  # * の部分 (例: 'A')
                file_full_path = os.path.join(dir_path, file_name)

                # 変位の抽出
                diffs = process_csv(file_full_path)

                if diffs:
                    if prefix not in summary_data[dir_path]:
                        summary_data[dir_path][prefix] = []
                    summary_data[dir_path][prefix].extend(diffs)
                    all_diffs_global.extend(diffs)

    # 全体のプレフィックス（アルファベット）一覧を取得してソート
    all_prefixes = sorted(
        list(
            set(
                prefix
                for dir_dict in summary_data.values()
                for prefix in dir_dict.keys()
            )
        )
    )

    # 集計用データフレームの構築
    rows = []
    for dir_path in DIRECTORIES:
        row = {"Directory": dir_path}
        dir_all_diffs = []

        # アルファベットごとの平均を計算
        for prefix in all_prefixes:
            col_name = f"{prefix}_方向変化平均"
            diffs = summary_data[dir_path].get(prefix, [])
            if diffs:
                avg = sum(diffs) / len(diffs)
                row[col_name] = avg
                dir_all_diffs.extend(diffs)
            else:
                row[col_name] = None  # 該当データがない場合

        # ディレクトリ全体の平均を計算
        if dir_all_diffs:
            row["ディレクトリ内全体平均"] = sum(dir_all_diffs) / len(
                dir_all_diffs
            )
        else:
            row["ディレクトリ内全体平均"] = None

        rows.append(row)

    # 全体合計行（末尾）
    total_row = {"Directory": "全ディレクトリ合計"}
    for prefix in all_prefixes:
        col_name = f"{prefix}_方向変化平均"
        # 全ディレクトリを通じた該当アルファベットの平均
        prefix_all_diffs = [
            diff
            for dir_dict in summary_data.values()
            for diff in dir_dict.get(prefix, [])
        ]
        total_row[col_name] = (
            sum(prefix_all_diffs) / len(prefix_all_diffs)
            if prefix_all_diffs
            else None
        )

    total_row["ディレクトリ内全体平均"] = (
        sum(all_diffs_global) / len(all_diffs_global)
        if all_diffs_global
        else None
    )
    rows.append(total_row)

    # データフレームに変換してCSV出力
    result_df = pd.DataFrame(rows)

    output_filename = "実験1_方向変化集計結果.csv"
    result_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"処理が完了しました。'{output_filename}' を出力しました。")


if __name__ == "__main__":
    main()
