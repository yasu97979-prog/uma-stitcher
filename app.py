import streamlit as st
import cv2
import numpy as np
import hashlib
import base64
import streamlit.components.v1 as components
from streamlit_paste_button import paste_image_button as pbutton


class StitchError(Exception):
    """画像結合に失敗した際に、問題のあった画像番号を保持して投げる例外"""
    def __init__(self, index_a, index_b):
        self.index_a = index_a
        self.index_b = index_b
        super().__init__(f"{index_a}枚目と{index_b}枚目の間で十分な重なりが検出できませんでした")


def stitch_images(images, header_ratio=0.33, footer_ratio=0.13, threshold=0.75, search_ratio=0.85):
    """
    画像リストを縦方向に結合する。
    重なりが不十分な箇所があれば StitchError を投げて処理を中断する。

    search_ratio: 次の画像のうち、上から何割の範囲までをマッチング候補として探すか。
    因子リストには似たようなアイコン・行が多く並ぶため、探索範囲を全体にすると
    離れた場所にある別の行を誤って「重なり」として検出してしまうことがある。
    ある程度上側に制限しつつ、正しい重なり位置を取りこぼさない範囲に留める。
    """
    base_img = images[0]
    base_H, base_W = base_img.shape[:2]

    header_h = int(base_H * header_ratio)
    footer_h = int(base_H * footer_ratio)

    final_header = base_img[:header_h, :]
    final_footer = base_img[base_H - footer_h:, :]
    base_list = base_img[header_h: base_H - footer_h, :]

    template_h = int(base_H * 0.05)
    # base_listの一番下ぎりぎりは、ウィンドウ下端で行が途中で切れている場合があり
    # テンプレートとして不安定になりやすいため、少し上（margin分）にずらして採る
    margin = max(1, int(base_H * 0.015))
    x_start = int(base_W * 0.25)
    x_end = int(base_W * 0.85)

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
        template = base_list[-(template_h + margin):-margin, x_start:x_end]

        # 探索範囲を上側寄りに限定し、離れた場所での誤マッチを防ぐ
        max_search_h = max(template_h * 4, int(next_list.shape[0] * search_ratio))
        search_area = next_list[:max_search_h, x_start:x_end]

        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        match_y = max_loc[1]

        if max_val < threshold:
            # i番目（0-indexed）は表示上「i+1枚目」なので、直前の画像は「i枚目」
            raise StitchError(index_a=i, index_b=i + 1)

        # マッチ位置はbase_list末尾からmargin分手前を指しているので、
        # 実際の継ぎ目位置はそこにmarginを足した位置になる
        new_part = next_list[match_y + template_h + margin:, :]
        base_list = np.vstack((base_list, new_part))

    return np.vstack((final_header, base_list, final_footer))


def render_copy_button(png_bytes):
    """結合済み画像をクリップボードにコピーするボタンをHTML/JSで描画する"""
    img_b64 = base64.b64encode(png_bytes).decode()
    copy_html = f"""
    <div>
      <button id="copyImgBtn" style="background-color:#34a853;color:white;border:none;
        padding:8px 18px;border-radius:6px;font-size:15px;cursor:pointer;">
        📋 画像をコピー
      </button>
      <span id="copyImgStatus" style="margin-left:10px;font-size:14px;"></span>
    </div>
    <script>
    const btn = document.getElementById('copyImgBtn');
    btn.addEventListener('click', async () => {{
        const statusEl = document.getElementById('copyImgStatus');
        try {{
            const base64Data = "{img_b64}";
            const byteCharacters = atob(base64Data);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {{
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }}
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {{ type: 'image/png' }});
            await navigator.clipboard.write([new ClipboardItem({{ 'image/png': blob }})]);
            statusEl.innerText = '✅ コピーしました！';
        }} catch (err) {{
            statusEl.innerText = '❌ コピーに失敗しました（' + err + '）';
        }}
    }});
    </script>
    """
    components.html(copy_html, height=45)


# 画面を広く使う設定
st.set_page_config(page_title="ウマ娘 因子スクロール結合", layout="wide")

st.title("🐴 ウマ娘 因子スクロール自動結合ツール")

st.markdown("""
PCの方は「📋 画像を貼り付け」ボタンでクリップボードの画像をそのまま追加できます
（フォルダ選択は開きません）。スマホの方は「📱 スマホの方はこちら」から
写真フォルダを開いて複数枚まとめて選択してください。
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
if 'processed_file_ids' not in st.session_state:
    st.session_state.processed_file_ids = set()


def add_image(img_bgr):
    """画像1枚をimage_listに追加する（空き枠があればそこに、無ければ末尾に）"""
    if st.session_state.freed_slots:
        st.session_state.freed_slots.sort()
        slot = st.session_state.freed_slots.pop(0)
        slot = min(slot, len(st.session_state.image_list))  # 念のため範囲を安全に丸める
        st.session_state.image_list.insert(slot, img_bgr)
    else:
        st.session_state.image_list.append(img_bgr)


# --- 画像の追加方法（PC / スマホで使い分け） ---
paste_col, upload_col = st.columns(2)

with paste_col:
    st.caption("🖥️ PCの方はこちら（フォルダは開きません）")
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
            add_image(img_bgr)
            st.session_state.last_pasted_hash = img_hash
            st.rerun()

with upload_col:
    st.caption("📱 スマホの方はこちら（複数選択可）")
    uploaded_files = st.file_uploader(
        "画像を選択",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="mobile_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_added = False
        for f in uploaded_files:
            # まだ読み込んでいない新しいファイルだけを処理する
            if f.file_id not in st.session_state.processed_file_ids:
                st.session_state.processed_file_ids.add(f.file_id)
                file_bytes = f.read()
                img_array = np.frombuffer(file_bytes, np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    add_image(img_bgr)
                new_added = True
        if new_added:
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
            st.session_state.processed_file_ids = set()
            st.rerun()

    with col2:
        if len(st.session_state.image_list) >= 2:
            if st.button("✨ 結合スタート！", type="primary"):
                with st.spinner("画像を結合しています... 少々お待ちください。"):
                    try:
                        final_img = stitch_images(st.session_state.image_list)
                    except StitchError as e:
                        st.error(
                            f"❌ 結合に失敗しました。{e.index_a}枚目と{e.index_b}枚目の"
                            f"重なりが不足しているため、正しく結合できません。\n\n"
                            f"{e.index_a}枚目・{e.index_b}枚目のスクリーンショットを撮り直し、"
                            f"重なりを大きくしてから再度お試しください。"
                        )
                    else:
                        st.success("🎉 結合が完了しました！")

                        is_success, buffer = cv2.imencode(".png", final_img)
                        if is_success:
                            png_bytes = buffer.tobytes()
                            dl_col, copy_col = st.columns([1, 1])
                            with dl_col:
                                st.download_button(
                                    label="📥 完成画像をダウンロード",
                                    data=png_bytes,
                                    file_name="uma_stitched_result.png",
                                    mime="image/png",
                                )
                            with copy_col:
                                render_copy_button(png_bytes)

                        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                        st.image(final_img_rgb, caption="完成画像", use_container_width=True)
        else:
            st.info("💡 結合には2枚以上の画像が必要です。上のボタンから画像を貼り付けてください。")
