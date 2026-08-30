import streamlit as st
import cv2
import numpy as np

# 画面を広く使う設定
st.set_page_config(page_title="ウマ娘 因子スクロール結合", layout="wide")

# ========================================================
# 【超重要】お客様のご要望通り、邪魔なUpload枠を完全に消去する魔法のCSS
# ========================================================
st.markdown("""
<style>
/* アップロード枠とUploadボタンを画面から完全に消し去る（機能だけ裏で残す） */
[data-testid="stFileUploader"] {
    position: absolute !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    z-index: -999 !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🐴 ウマ娘 画像結合ツール")
st.info("💡 **【完全ペースト専用版】** 邪魔な「Upload」ボタンを完全に消去しました！\nこの画面の**どこでもいいので一度クリック**し、そのまま **Ctrl+V (Cmd+V)** を連打するだけで画像が次々と追加されます。")

if 'image_list' not in st.session_state:
    st.session_state.image_list = []
if 'processed_file_ids' not in st.session_state:
    st.session_state.processed_file_ids = set()

# --- 見えない画像ペースト受け皿 ---
# 入力枠をリセットしないことで、2枚目以降のペーストも途切れず受け付ける
uploaded_files = st.file_uploader(
    "hidden_uploader", 
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="hidden"
)

if uploaded_files:
    new_added = False
    for f in uploaded_files:
        if f.file_id not in st.session_state.processed_file_ids:
            st.session_state.processed_file_ids.add(f.file_id)
            file_bytes = f.read()
            img_array = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                st.session_state.image_list.append(img)
            new_added = True
    
    if new_added:
        st.rerun()

# --- プレビュー・結合エリア ---
if st.session_state.image_list:
    st.write("---")
    st.subheader(f"📸 読み込み済みの画像 ({len(st.session_state.image_list)}枚)")
    
    cols = st.columns(5)
    for idx, img in enumerate(st.session_state.image_list):
        with cols[idx % 5]:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"{idx+1}枚目", use_container_width=True)
            if st.button(f"❌ {idx+1}を削除", key=f"del_btn_{idx}"):
                st.session_state.image_list.pop(idx)
                st.rerun()
                
    st.write("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🗑️ すべてリセット", type="secondary"):
            st.session_state.image_list = []
            st.session_state.processed_file_ids = set()
            st.rerun()
            
    with col2:
        if len(st.session_state.image_list) >= 2:
            if st.button("✨ 結合スタート！", type="primary"):
                with st.spinner("画像を結合しています... 少々お待ちください。"):
                    images = st.session_state.image_list
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
            st.info("💡 結合には2枚以上の画像が必要です。Ctrl+Vで画像を追加してください。")
