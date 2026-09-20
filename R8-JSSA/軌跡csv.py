import cv2
import numpy as np
import os
import glob
import pandas as pd

# ====================================================
# 💡 1. 空間補正および幾何学計算関数
# ====================================================
def fit_circle_least_squares(pts):
    pts = np.array(pts, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    A = np.vstack([x, y, np.ones_like(x)]).T
    B = x**2 + y**2
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    cx = C[0] / 2.0
    cy = C[1] / 2.0
    r = np.sqrt(C[2] + cx**2 + cy**2)
    return cx, cy, r

def correct_marker_shift(mx, my, shifts_data):
    if len(shifts_data) == 0: 
        return mx, my
    weights, diffs_x, diffs_y = [], [], []
    for row in shifts_data:
        tx, ty, vx, vy = row
        dist = np.hypot(mx - vx, my - vy)
        if dist < 1e-5: 
            return tx, ty
        weight = 1.0 / (dist ** 2)
        weights.append(weight)
        diffs_x.append(tx - vx)
        diffs_y.append(ty - vy)
    weights = np.array(weights)
    sum_w = np.sum(weights)
    if sum_w == 0:
        return mx, my
    return mx + np.sum(weights * np.array(diffs_x)) / sum_w, my + np.sum(weights * np.array(diffs_y)) / sum_w

# ====================================================
# 💡 2. 画面サイズ制限付きウィンドウ生成関数 (1366x768対応)
# ====================================================
def create_fitted_window(win_name, img_w, img_h):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    max_w, max_h = 960, 600
    scale = min(max_w / img_w, max_h / img_h)
    
    if scale < 1.0:
        win_w = int(img_w * scale)
        win_h = int(img_h * scale)
    else:
        win_w = img_w
        win_h = img_h
        
    cv2.resizeWindow(win_name, win_w, win_h)

# ====================================================
# 💡 3. OpenCV マウスコールバック関数
# ====================================================
click_pts = []
click_type = "circle"
temp_pt = None

def mouse_callback_calib(event, x, y, flags, param):
    global click_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pts.append([x, y])
        img_disp = param['img'].copy()
        for p in click_pts:
            cv2.circle(img_disp, (p[0], p[1]), 8, (0, 0, 255), -1)
        cv2.imshow(param['win'], img_disp)

def mouse_callback_circle(event, x, y, flags, param):
    global click_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pts.append([x, y])
        img_disp = param['img'].copy()
        for p in click_pts:
            cv2.circle(img_disp, (p[0], p[1]), 5, (255, 0, 0), -1)
        cv2.imshow(param['win'], img_disp)
        print(f" 🔵 円周上の点 {len(click_pts)}: ({x}, {y})")

def mouse_callback_shift(event, x, y, flags, param):
    global click_pts, click_type, temp_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        img_disp = param['img']
        if click_type == "circle":
            temp_pt = [x, y]
            click_type = "marker"
            cv2.circle(img_disp, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(param['win'], img_disp)
            print(f" 🟢 本来の路面位置を選択: ({x}, {y}) -> 次に【マーカー中心】をクリック")
        elif click_type == "marker":
            click_pts.append([temp_pt[0], temp_pt[1], x, y])
            click_type = "circle"
            cv2.circle(img_disp, (x, y), 5, (0, 0, 255), -1)
            cv2.line(img_disp, (temp_pt[0], temp_pt[1]), (x, y), (0, 255, 255), 2)
            cv2.imshow(param['win'], img_disp)
            print(f" 🔴 マーカー中心を選択: ({x}, {y}) [現在 {len(click_pts)} ペア]")

# ====================================================
# 💡 4. メイン制御フロー
# ====================================================
def main():
    global click_pts, click_type
    
    print("==========================================")
    print(" 📂 カーブ区間判定＆解析順序指定型データ抽出")
    print("==========================================")
    
    target_dir = input("解析対象のディレクトリ名を入力してください: ").strip()
    
    current_width = 520.0
    current_height = 510.0
    
    # 全動画の読み込みと重複排除
    raw_video_files = glob.glob(os.path.join(target_dir, "*.MP4")) + glob.glob(os.path.join(target_dir, "*.mp4"))
    raw_video_files = list({os.path.abspath(f): f for f in raw_video_files}.values())
    raw_video_files.sort()  # 一旦名前順ソート
    
    if not raw_video_files:
        print(f"[ERROR] '{target_dir}' 内に動画が見つかりません。")
        return
        
    # 💡 【新規機能】動画の解析順序を自由に入れ替えるUI
    print("\n📂 検出された動画一覧:")
    for i, f in enumerate(raw_video_files):
        print(f"  [{i+1}] {os.path.basename(f)}")
        
    print("\n💡 解析したい順番に番号をカンマ区切りで入力してください（例: 3,1,2）")
    print("   デフォルトの順序で良い場合は、何も入力せずにそのままEnterを押してください。")
    order_input = input("解析順序を指定: ").strip()
    
    video_files = []
    if order_input:
        try:
            indices = [int(x.strip()) - 1 for x in order_input.split(",") if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(raw_video_files):
                    video_files.append(raw_video_files[idx])
            if not video_files:
                print(" ⚠️ 有効な番号がなかったため、デフォルト順で処理します。")
                video_files = raw_video_files
        except Exception:
            print(" ⚠️ 入力エラーのため、デフォルトの順序で処理します。")
            video_files = raw_video_files
    else:
        video_files = raw_video_files

    print(f"\n 🚀 確定した解析スケジュール (全 {len(video_files)} 本):")
    for i, f in enumerate(video_files):
        print(f"  [{i+1}] {os.path.basename(f)}")

    # メモリ上保持用のキャリブレーション変数
    M = None
    cx, cy, r_cm = 0.0, 0.0, 0.0
    shifts = []
    start_cm = None  # 💡 カーブ開始クリック座標(cm空間)
    end_cm = None    # 💡 カーブ終了クリック座標(cm空間)
    
    PAD_LEFT = 300
    
    for idx, video_path in enumerate(video_files):
        video_filename = os.path.basename(video_path)
        video_basename, _ = os.path.splitext(video_filename)
        csv_output_path = os.path.join(target_dir, f"{video_basename}_tracked.csv")
        
        print(f"\n🎬 [{idx+1}/{len(video_files)}] 処理開始: {video_filename}")
        
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            print(f" [ERROR] 動画フレームを読み込めません。")
            cap.release()
            continue
            
        orig_h, orig_w = first_frame.shape[:2]
        need_calibration = False
        
        if M is None:
            need_calibration = True
        else:
            dst_w, dst_h = int(current_width), int(current_height)
            warped_check = cv2.warpPerspective(first_frame, M, (dst_w, dst_h))
            cv2.circle(warped_check, (int(cx), int(cy)), int(r_cm), (255, 255, 0), 3)
            
            # カーブ区間クリックのインジケータも確認画面に仮表示
            if start_cm is not None and end_cm is not None:
                cv2.circle(warped_check, (int(start_cm[0]), int(start_cm[1])), 8, (0, 255, 0), -1)
                cv2.circle(warped_check, (int(end_cm[0]), int(end_cm[1])), 8, (0, 0, 255), -1)
            
            win_check = "Calibration Check"
            create_fitted_window(win_check, dst_w, dst_h)
            cv2.imshow(win_check, warped_check)
            
            print(f" ❓ この設定で続行しますか？ 【Y】進む / 【N】サイズ入力＆補正をやり直す")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in [ord('y'), ord('Y'), 13]:
                    break
                elif key in [ord('n'), ord('N')]:
                    need_calibration = True
                    break
            cv2.destroyWindow(win_check)
            
        if need_calibration:
            print(f"\n⚙️ 【再設定モード】実験環境の実際のサイズを入力してください。")
            current_width = float(input(f"  実験エリアの実際の横幅 (cm) [現在: {current_width}]: ") or current_width)
            current_height = float(input(f"  実験エリアの実際の縦幅 (cm) [現在: {current_height}]: ") or current_height)
            dst_w, dst_h = int(current_width), int(current_height)
            
            # --- Step 1: 4隅キャリブレーション (左側黒帯300px) ---
            win_calib = "1. Click 4 Corners (Left side padded)"
            img_calib = cv2.copyMakeBorder(first_frame, 0, 0, PAD_LEFT, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            click_pts = []
            create_fitted_window(win_calib, orig_w + PAD_LEFT, orig_h)
            cv2.setMouseCallback(win_calib, mouse_callback_calib, {'win': win_calib, 'img': img_calib})
            cv2.imshow(win_calib, img_calib)
            
            print(f" 📍 【左上 ➡️ 右上 ➡️ 右下 ➡️ 左下】の順に4点クリックしてください。")
            while len(click_pts) < 4:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit()
            cv2.destroyWindow(win_calib)
            
            corrected_pts = []
            for p in click_pts[:4]:
                corrected_pts.append([p[0] - PAD_LEFT, p[1]])
                
            pts_src = np.array(corrected_pts, dtype=np.float32)
            pts_dst = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            
            # --- Step 2: 円周フィッティング (カーブ開始・終了も兼ねる) ---
            base_warped = cv2.warpPerspective(first_frame, M, (dst_w, dst_h))
            win_circle = "2. Click 3+ points on Road Circle"
            img_circle = base_warped.copy()
            click_pts = []
            
            create_fitted_window(win_circle, dst_w, dst_h)
            cv2.setMouseCallback(win_circle, mouse_callback_circle, {'win': win_circle, 'img': img_circle})
            cv2.imshow(win_circle, img_circle)
            
            print(" 🔵 円周ラインを【3箇所以上】クリックし、【Enter】を押してください。")
            print("    💡 【重要】1番目にクリックした点を『カーブ開始位置』、最後の点を『カーブ終了位置』として自動記録します！")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 13:
                    if len(click_pts) >= 3:
                        cx, cy, r_cm = fit_circle_least_squares(click_pts)
                        # 💡 1px=1cm空間なので、クリック座標をそのままカーブの開始・終了マークとする
                        start_cm = click_pts[0]
                        end_cm = click_pts[-1]
                        print(f" 🎯 カーブ区間設定: 開始点={start_cm}, 終了点={end_cm}")
                        break
            cv2.destroyWindow(win_circle)
            
            # --- Step 3: 立体ズレ(シフト)クリック ---
            win_shift = "3. Shift Calibration"
            click_pts = []
            click_type = "circle"
            create_fitted_window(win_shift, dst_w, dst_h)
            
            print(" 🕹️ 立体ズレ補正を設定します。【Space】で5コマ進む。終わったら【Enter】。")
            f_count = 1
            current_warped = base_warped.copy()
            
            while True:
                img_shift_disp = current_warped.copy()
                for p in click_pts:
                    cv2.circle(img_shift_disp, (p[0], p[1]), 4, (0, 255, 0), -1)
                    cv2.circle(img_shift_disp, (p[2], p[3]), 4, (0, 0, 255), -1)
                    cv2.line(img_shift_disp, (p[0], p[1]), (p[2], p[3]), (0, 255, 255), 2)
                
                cv2.putText(img_shift_disp, f"Pairs: {len(click_pts)}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(win_shift, img_shift_disp)
                cv2.setMouseCallback(win_shift, mouse_callback_shift, {'win': win_shift, 'img': img_shift_disp})
                
                key = cv2.waitKey(0) & 0xFF
                if key == 32:
                    click_type = "circle"
                    for _ in range(5):
                        ret, frame_next = cap.read()
                        f_count += 1
                        if not ret: break
                    if not ret: break
                    current_warped = cv2.warpPerspective(frame_next, M, (dst_w, dst_h))
                elif key == 13:
                    if len(click_pts) >= 3:
                        shifts = np.array(click_pts)
                        break
            cv2.destroyWindow(win_shift)

        # ====================================================
        # 💡 5. 自動データ抽出処理
        # ====================================================
        print(" 🚀 自動データ抽出を実行中...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        LOWER_PINK = np.array([145, 70, 80])
        UPPER_PINK = np.array([170, 255, 255])
        MIN_AREA_PX = 10
        
        raw_tracking_data = []
        f_idx = 0
        
        win_run = "Data Extracting Pipeline..."
        create_fitted_window(win_run, dst_w, dst_h)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            f_idx += 1
            
            warped = cv2.warpPerspective(frame, M, (dst_w, dst_h))
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_PINK, UPPER_PINK)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_candidates = []
            for cnt in contours:
                if cv2.contourArea(cnt) >= MIN_AREA_PX:
                    M_mom = cv2.moments(cnt)
                    if M_mom["m00"] != 0:
                        valid_candidates.append((int(M_mom["m10"]/M_mom["m00"]), int(M_mom["m01"]/M_mom["m00"])))
                        
            detected = False
            raw_mx, raw_my = 0.0, 0.0
            if len(valid_candidates) >= 1:
                raw_mx = np.mean([p[0] for p in valid_candidates])
                raw_my = np.mean([p[1] for p in valid_candidates])
                detected = True
                
            if detected:
                corrected_x, corrected_y = correct_marker_shift(raw_mx, raw_my, shifts)
                current_angle = np.arctan2(corrected_y - cy, corrected_x - cx)
                actual_dist = np.hypot(corrected_x - cx, corrected_y - cy)
                orbit_deviation = actual_dist - r_cm
                time_sec = f_idx / fps
                
                raw_tracking_data.append({
                    "Frame": f_idx,
                    "Time(s)": round(time_sec, 3),
                    "Raw_X(cm)": round(raw_mx, 2),
                    "Raw_Y(cm)": round(raw_my, 2),
                    "Corrected_X(cm)": round(corrected_x, 2),
                    "Corrected_Y(cm)": round(corrected_y, 2),
                    "Angle(deg)": round(np.degrees(current_angle), 2),
                    "Deviation(cm)": round(orbit_deviation, 2)
                })
                
                cv2.circle(warped, (int(raw_mx), int(raw_my)), 4, (0, 0, 255), -1)
                cv2.circle(warped, (int(corrected_x), int(corrected_y)), 4, (0, 255, 0), -1)
                
            cv2.putText(warped, f"Processing: {f_idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow(win_run, warped)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyWindow(win_run)
        
        # 💡 5フレーム平均処理 ＆ 【カーブ区間判定】 ＆ CSV保存
        if len(raw_tracking_data) >= 5:
            df = pd.DataFrame(raw_tracking_data)
            v_list = [None] * len(df)
            dir_list = [None] * len(df)
            
            for i in range(2, len(df) - 2):
                x1, y1 = df.loc[i - 2, "Corrected_X(cm)"], df.loc[i - 2, "Corrected_Y(cm)"]
                x2, y2 = df.loc[i + 2, "Corrected_X(cm)"], df.loc[i + 2, "Corrected_Y(cm)"]
                t1, t2 = df.loc[i - 2, "Time(s)"], df.loc[i + 2, "Time(s)"]
                
                dist = np.hypot(x2 - x1, y2 - y1)
                dt = t2 - t1
                if dt > 0:
                    v_list[i] = round(dist / dt, 2)
                dir_list[i] = round(np.degrees(np.arctan2(y2 - y1, x2 - x1)), 1)
                
            df["Speed_5F(cm/s)"] = v_list
            df["Direction_5F(deg)"] = dir_list
            
            # 💡 【新規ロジック】開始クリック点と終了クリック点に最も近づいたフレームを割り出す
            dists_to_start = [np.hypot(r["Corrected_X(cm)"] - start_cm[0], r["Corrected_Y(cm)"] - start_cm[1]) for _, r in df.iterrows()]
            dists_to_end = [np.hypot(r["Corrected_X(cm)"] - end_cm[0], r["Corrected_Y(cm)"] - end_cm[1]) for _, r in df.iterrows()]
            
            t_idx1 = np.argmin(dists_to_start)
            t_idx2 = np.argmin(dists_to_end)
            
            # 走行方向やクリック順を自動補正（時系列で先に通った方をカーブの入り口にする）
            idx_start = min(t_idx1, t_idx2)
            idx_end = max(t_idx1, t_idx2)
            
            sections = []
            for i in range(len(df)):
                if i < idx_start:
                    sections.append("BC")
                elif idx_start <= i <= idx_end:
                    sections.append("DC")
                else:
                    sections.append("AC")
            df["Section"] = sections
            
            # 💡 CSVのK列目（11番目の列）にピッタリ固定するための並び替え
            column_order = [
                "Frame", "Time(s)", "Raw_X(cm)", "Raw_Y(cm)", 
                "Corrected_X(cm)", "Corrected_Y(cm)", "Angle(deg)", 
                "Deviation(cm)", "Speed_5F(cm/s)", "Direction_5F(deg)", "Section"
            ]
            df = df[column_order]
            
            df.to_csv(csv_output_path, index=False)
            print(f" ✨ [SAVED] -> {csv_output_path} (K列にBC/DC/ACを付与)")
        else:
            print(" ⚠️ データ不足のためCSVは生成されませんでした。")

    cv2.destroyAllWindows()
    print("\n[COMPLETE] 指定されたすべての動画のデータ一括抽出が完了しました！")

if __name__ == '__main__':
    main()
