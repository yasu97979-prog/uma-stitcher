import streamlit as st
import cv2
import numpy as np
import re
import base64

st.set_page_config(page_title="ウマ娘 因子スクロール結合", layout="centered")

st.title("🐴 ウマ娘 画像結合ツール")
st.write("枠内をクリックして **Ctrl+V (Cmd+V)** でスクショを何枚でも連続ペーストできます！")

# --- 画像を軽量化してWeb表示用（Base64）に変換する関数 ---
def image_to_base64(img):
    h, w = img.shape[:2]
    new_h = 250  # サムネイルの高さを250pxに縮小して軽くする
    new_w = int(w * (new_h / h))
    thumb = cv2.resize(img, (new_w, new_h))
    is_success, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")

def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

# --- データの保存場所を準備 ---
if 'image_list' not in st.session_state:
    st.session_state.image_list = []
if 'processed_hashes' not in st.session_state:
    st.session_state.processed_hashes = set()

# --- アップロード枠（複数ファイル・連続ペースト対応） ---
uploaded_files = st.file_uploader(
    "ここをクリックして Ctrl+V でスクショを連続ペースト（またはファイルを選択）", 
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ペーストされた画像を重複しないように追加していく
if uploaded_files:
    new_added = False
    for f in uploaded_files:
        bytes_data = f.read()
        file_hash = hash(bytes_data)
        # まだ追加されていない新しい画像ならリストに加える
        if file_hash not in st.session_state.processed_hashes:
            st.session_state.processed_hashes.add(file_hash)
            img_array = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                st.session_state.image_list.append({
                    "hash": file_hash,
                    "img": img
                })
                new_added = True
    if new_added:
        st.rerun()

# --- プレビュー・個別削除エリア（横スクロール対応） ---
if st.session_state.image_list:
    st.write("---")
    st.subheader(f"📸 読み込み済みの画像 ({len(st.session_state.image_list)}枚)")
    
    # 横スクロール用のカスタムデザイン（HTML/CSS）
    html = """
    <style>
    .h-scroll {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 15px 10px;
        border: 2px dashed #ddd;
        border-radius: 8px;
        background: #f9f9f9;
    }
    .h-scroll::-webkit-scrollbar { height: 12px; }
    .h-scroll::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 8px; }
    .h-scroll::-webkit-scrollbar-thumb { background: #ccc; border-radius: 8px; }
    .h-scroll::-webkit-scrollbar-thumb:hover { background: #aaa; }
    .img-card { position: relative; flex: 0 0 auto; }
    .img-card img { height: 250px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .img-badge {
        position: absolute; top: -10px; left: -10px;
        background: #ff4b4b; color: white; width: 28px; height: 28px;
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        z-index: 10;
    }
    </style>
    <div class="h-scroll">
    """
    for i, item in enumerate(st.session_state.image_list):
        b64 = image_to_base64(item["img"])
        html += f'''
        <div class="img-card">
            <div class="img-badge">{i+1}</div>
            <img src="data:image/jpeg;base64,{b64}">
        </div>
        '''
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # 削除とリセット用のボタン配置
    st.write("")
    col_del1, col_del2, col_del3 = st.columns([2, 2, 3])
    with col_del1:
        del_idx = st.number_input("削除する番号", min_value=1, max_value=len(st.session_state.image_list), value=len(st.session_state.image_list))
    with col_del2:
        st.write("") # 高さ合わせ
        if st.button("✖️ 指定番号を削除"):
            removed = st.session_state.image_list.pop(del_idx - 1)
            st.session_state.processed_hashes.remove(removed["hash"])
            st.rerun()
    with col_del3:
        st.write("")
        if st.button("🗑️ すべてリセット", type="secondary"):
            st.session_state.image_list = []
            st.session_state.processed_hashes = set()
            st.rerun()

    st.write("---")
    
    # --- 結合処理スタート ---
    if len(st.session_state.image_list) >= 2:
        if st.button("✨ 結合スタート！", type="primary"):
            with st.spinner("画像を結合しています... 少々お待ちください。"):
                # 画像データだけを抽出
                images = [item["img"] for item in st.session_state.image_list]
                
                base_img = images[0]
                base_H, base_W = base_img.shape[:2]

                header_ratio = 0.33 
                footer_ratio = 0.13 
                header_h = int(base_H * header_ratio)
                footer_h = int(base_H * footer_ratio)

                final_header = base_img[:header_h, :]
                final_footer = base_img[base_H - footer_h:, :]
                base_list = base_img[header_h : base_H - footer_h, :]

                template_h = int(base_H * 0.05)
                x_start = int(base_W * 0.25)
                x_end = int(base_W * 0.85)

                warning_flag = False

                for i in range(1, len(images)):
                    img_next = images[i]
                    next_H, next_W = img_next.shape[:2]
                    
                    if next_W != base_W:
                        scale = base_W / next_W
                        new_H = int(next_H * scale)
                        img_next = cv2.resize(img_next, (base_W, new_H))
                        
                    curr_H = img_next.shape[0]
                    curr_header_h = int(curr_H * header_ratio)
                    curr_footer_h = int(curr_H * footer_ratio)
                    
                    next_list = img_next[curr_header_h : curr_H - curr_footer_h, :]
                    template = base_list[-template_h:, x_start:x_end]
                    search_area = next_list[:, x_start:x_end]
                    
                    res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    match_y = max_loc[1]

                    if max_val < 0.75:
                        st.warning(f"⚠️ {i}枚目と{i+1}枚目で重なりが足りないか、ズレている可能性があります")
                        warning_flag = True
                        base_list = np.vstack((base_list, next_list)) 
                    else:
                        new_part = next_list[match_y + template_h:, :]
                        base_list = np.vstack((base_list, new_part))

                final_img = np.vstack((final_header, base_list, final_footer))
                
                if not warning_flag:
                    st.success("🎉 結合が完了しました！")
                
                final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                st.image(final_img_rgb, caption="完成画像", use_container_width=True)
                
                is_success, buffer = cv2.imencode(".png", final_img)
                if is_success:
                    st.download_button(
                        label="📥 完成画像をダウンロード",
                        data=buffer.tobytes(),
                        file_name="uma_stitched_result.png",
                        mime="image/png"
                    )
    else:
        st.info("💡 結合には2枚以上の画像が必要です。続けてスクショをペーストしてください。")
