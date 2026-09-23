import glob
import os
import numpy as np
import pandas as pd

# 対象のディレクトリ一覧
directories = ['r2m_手信号あり', 'r2m_手信号なし', 'r4m_手信号あり', 'r4m_手信号なし']

# 対象の文字一覧
target_chars = ['A', 'I', 'K', 'M', 'S']


def remove_outliers(values):
  """IQR法（1.5 * IQR）に基づいて外れ値を除外する関数"""
  if not values or len(values) < 4:
    # データ数が極端に少ない場合は外れ値除外を行わずそのまま返す
    return values

  q1 = np.percentile(values, 25)
  q3 = np.percentile(values, 75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr

  # 範囲内のデータのみを抽出
  filtered = [v for v in values if lower_bound <= v <= upper_bound]
  return filtered


for dir_name in directories:
  print(f'{dir_name}:')
  dir_all_values = (
      []
  )  # ディレクトリ全体の平均を出すために全数値を集めるリスト

  for char in target_chars:
    pattern = os.path.join(dir_name, f'{char} *_tracked.csv')
    files = glob.glob(pattern)

    if not files:
      continue

    std_list = []
    for file_path in files:
      try:
        df = pd.read_csv(file_path)
        dc_rows = df[df['Section'] == 'DC']

        if not dc_rows.empty:
          std_val = dc_rows['Speed_5F(cm/s)'].std()
          if pd.notna(std_val):
            std_list.append(std_val)
      except Exception as e:
        print(f'  [警告] {file_path} の処理中にエラーが発生しました: {e}')

    if std_list:
      # 外れ値を除外
      filtered_std = remove_outliers(std_list)

      if filtered_std:
        mean_std = np.mean(filtered_std)
        print(f' {char}: {mean_std:.4f}')
        # ディレクトリ全体の平均計算用に蓄積
        dir_all_values.extend(filtered_std)
      else:
        print(f' {char}: 外れ値除外により有効データなし')
    else:
      print(f' {char}: 対象データなし')

  # 各ディレクトリごとの平均値（全文字・全ファイルを統合した値の平均）を表示
  if dir_all_values:
    dir_mean = np.mean(dir_all_values)
    print(f' [ディレクトリ平均]: {dir_mean:.4f}')
  else:
    print(f' [ディレクトリ平均]: 対象データなし')

  print('-' * 30)
