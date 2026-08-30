import streamlit as st
import cv2
import numpy as np
import hashlib
from streamlit_paste_button import paste_image_button as pbutton

# 画面を広く使う設定
st.set_page_config(page_title="ウマ娘 因子スクロール結合", layout="wide")

st.title("🐴 ウマ娘 因子スクロール自動結合ツール")

st.markdown("""
下の「📋 画像を貼り付け」ボタンをクリックすると、クリップボードにコピーされている
スクリーンショットがそのまま取り込まれます（ファイル選択ダイアログは開きません）。
1枚コピー→ボタンをクリック、を繰り返して必要な枚数を追加してください。
""")

# --- データの保持 ---
if 'image_list' not in st.session_state:
    st.session_state.image_list = []
if 'last_pasted_hash' not in st.session_state:
    st.session_state.last_pasted_hash = None
if 'freed_slots' not in st.session_state:
    # 削除された枠の番号（インデックス）を保持しておき、
    # 次に貼り付けた画像を末尾ではなくこの位置に差し込むために使う
    st.session_state.freed_slots = []

# --- 貼り付けボタン ---
# ※ブラウザのClipboard APIを使うため、初回はブラウザ側の許可ダイアログが出ることがあります。
#   Chrome / Edge / Safari で動作確認済み。Firefoxは非対応です。
paste_result = pbutton(
    label="📋 画像を貼り付け",
    text_color="#ffffff",
    background_color="#4285f4",
    hover_background_color="#3367d6",
    key="paste_button",
)

if paste_result.image_data is not None:
    # PIL Image -> OpenCV(BGR) に変換
    pil_img = paste_result.image_data.convert("RGB")
    img_array = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 同じ貼り付けイベントを再実行（rerun）のたびに重複追加しないよう、
    # 画像内容のハッシュで直前の貼り付けと同一かどうかを判定する
    img_hash = hashlib.md5(img_bgr.tobytes()).hexdigest()
    if img_hash != st.session_state.last_pasted_hash:
        if st.session_state.freed_slots:
            # 空いている枠があれば、一番若い番号の枠に差し込む
            st.session_state.freed_slots.sort()
            slot = st.session_state.freed_slots.pop(0)
            slot = min(slot, len(st.session_state.image_list))  # 念のため範囲を安全に丸める
            st.session_state.image_list.insert(slot, img_bgr)
        else:
            # 空き枠が無ければ今まで通り末尾に追加
            st.session_state.image_list.append(img_bgr)
        st.session_state.last_pasted_hash = img_hash
        st.rerun()

# --- プレビュー・結合エリア ---
if st.session_state.image_list:
    st.write("---")
    st.subheader(f"📸 読み込み済みの画像 ({len(st.session_state.image_list)}枚)")

    # 5列に分割してサムネイルを並べる
    cols = st.columns(5)
    for idx, img in enumerate(st.session_state.image_list):
        with cols[idx % 5]:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"{idx+1}枚目", use_container_width=True)
            # 個別削除ボタン
            if st.button(f"❌ {idx+1}を削除", key=f"del_btn_{idx}"):
                st.session_state.image_list.pop(idx)
                st.session_state.freed_slots.append(idx)
                st.rerun()

    st.write("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🗑️ すべてリセット", type="secondary"):
            st.session_state.image_list = []
            st.session_state.last_pasted_hash = None
            st.session_state.freed_slots = []
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
                    base_list = base_img[header_h: base_H - footer_h, :]

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

                        next_list = img_next[curr_header_h: curr_H - curr_footer_h, :]
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
                            mime="image/png",
                        )
        else:
            st.info("💡 結合には2枚以上の画像が必要です。上のボタンから画像を貼り付けてください。")
