import cv2
import numpy as np
import glob
import re
import IPython
from IPython.display import display, HTML
import base64

image_dir = '/content/uma_images'

def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

image_paths = sorted(glob.glob(f"{image_dir}/*.png"), key=natural_keys)

if len(image_paths) < 2:
    print("画像が2枚以上ありません。上のステップで画像をペーストしてください。")
else:
    print(f"ペーストされた {len(image_paths)} 枚の画像を結合処理中...\n")

    # 1. 基準となる1枚目の画像を読み込む
    base_img = cv2.imread(image_paths[0])
    base_H, base_W = base_img.shape[:2]

    # ========================================================
    # 【最重要修正】固定UIのカット範囲とテンプレートサイズ
    # ========================================================
    # タブ（スキル・継承など）のすぐ下を正確に狙う (約33%)
    header_ratio = 0.33 
    # 「閉じる」ボタンと下部のグラデーションをカット (約13%)
    footer_ratio = 0.13 
    
    header_h = int(base_H * header_ratio)
    footer_h = int(base_H * footer_ratio)

    final_header = base_img[:header_h, :]
    final_footer = base_img[base_H - footer_h:, :]
    
    # 切り出したリスト部分
    base_list = base_img[header_h : base_H - footer_h, :]

    # 重なり判定用：1行分（約5%）だけをピンポイントで抜き出す
    template_h = int(base_H * 0.05)
    
    # 左右のノイズ（顔アイコンやスクロールバー）を完全に無視する範囲
    x_start = int(base_W * 0.25)
    x_end = int(base_W * 0.85)

    warning_flag = False

    for i in range(1, len(image_paths)):
        img_next = cv2.imread(image_paths[i])
        next_H, next_W = img_next.shape[:2]
        
        # ウインドウサイズが途中で変わっていても、横幅と比率を1枚目にピタッと合わせる
        if next_W != base_W:
            scale = base_W / next_W
            new_H = int(next_H * scale)
            img_next = cv2.resize(img_next, (base_W, new_H))
            
        curr_H = img_next.shape[0]
        curr_header_h = int(curr_H * header_ratio)
        curr_footer_h = int(curr_H * footer_ratio)
        
        # 次の画像のリスト部分を抽出
        next_list = img_next[curr_header_h : curr_H - curr_footer_h, :]

        # ベース画像の一番下から「1行分」を切り取って型にする
        template = base_list[-template_h:, x_start:x_end]
        
        # 次の画像のリスト全体から、型と完全に一致する場所を探す
        search_area = next_list[:, x_start:x_end]
        
        # OpenCVのAIによる重なり検知
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        match_y = max_loc[1]

        # 0.75以上なら「ほぼ完全一致」とみなす
        if max_val < 0.75:
            print(f"⚠️ 【警告】{i}枚目と{i+1}枚目で重なりが足りないか、ズレています (一致スコア: {max_val:.2f})")
            warning_flag = True
            base_list = np.vstack((base_list, next_list)) 
        else:
            # ピタリと重なった場所の「直下」から新しい部分だけを切り出し、結合する
            new_part = next_list[match_y + template_h:, :]
            base_list = np.vstack((base_list, new_part))

    # 3. 上部ヘッダー ＋ 完成した巨大リスト ＋ 下部フッター を合体
    final_img = np.vstack((final_header, base_list, final_footer))

    if not warning_flag:
        print("🎉 結合が完了しました！完璧に繋がりました。")
    else:
        print("※一部で警告が出ました。画像を確認してください。")

    print("\n=========================================")
    print("【完成画像】")
    print("以下の画像を 右クリック して「画像をコピー」 してください。")
    print("=========================================\n")

    _, buffer = cv2.imencode('.png', final_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    display(HTML(f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">'))