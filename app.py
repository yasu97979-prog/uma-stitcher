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


def detect_header_footer_ratio(img_a, img_b, diff_threshold=15, min_run=6, x_frac=(0.25, 0.85)):
    """
    同じウィンドウ・同じキャラをスクロールしただけの2枚（img_a, img_b）を比較し、
    「毎回まったく同じ絵になっている＝スクロールしない部分（ヘッダー／フッター）」を自動検出する。

    やり方：中央の帯（x_frac範囲）で1行ごとの画素差分を計算し、
    上から見て差分が続けて大きくなり始める位置＝ヘッダーの終わり、
    下から見て同様の位置＝フッターの始まり、とみなす。

    2枚の画像の縦幅が異なる場合（表示されているリストの行数が違う場合）でも、
    ヘッダーは「両画像の上端」、フッターは「両画像の下端」をそれぞれ基準に
    比較することで正しく検出できるようにしている。
    """
    Ha, Wa = img_a.shape[:2]
    Hb, Wb = img_b.shape[:2]
    W = min(Wa, Wb)
    x_start, x_end = int(W * x_frac[0]), int(W * x_frac[1])

    # ヘッダー：両画像の「上端」を基準に比較する
    H_top = min(Ha, Hb)
    a_top = img_a[:H_top, x_start:x_end].astype(np.int16)
    b_top = img_b[:H_top, x_start:x_end].astype(np.int16)
    diff_top = np.mean(np.abs(a_top - b_top), axis=(1, 2))
    header_h = 0
    for y in range(H_top - min_run):
        if diff_top[y:y + min_run].mean() > diff_threshold:
            header_h = y
            break

    # フッター：両画像の「下端」を基準に比較する（縦幅が違っても正しく揃う）
    H_bot = min(Ha, Hb)
    a_bot = img_a[Ha - H_bot:, x_start:x_end].astype(np.int16)
    b_bot = img_b[Hb - H_bot:, x_start:x_end].astype(np.int16)
    diff_bot = np.mean(np.abs(a_bot - b_bot), axis=(1, 2))
    footer_h = 0
    for y in range(H_bot - 1, min_run, -1):
        if diff_bot[y - min_run:y].mean() > diff_threshold:
            footer_h = H_bot - y
            break

    header_ratio = header_h / Ha
    footer_ratio = footer_h / Ha

    # 明らかにおかしい結果（検出失敗）の場合はNoneを返し、呼び出し側でデフォルト値にフォールバックさせる
    if header_ratio <= 0 or header_ratio >= 0.7 or footer_ratio >= 0.5 or (header_ratio + footer_ratio) >= 0.85:
        return None

    return header_ratio, footer_ratio


def estimate_row_period(img_list, x_start, x_end, min_period=15, max_period=150):
    """
    因子・スキルの各行が繰り返される「周期（1行の高さ）」を自己相関から推定する。
    テンプレートのサイズをこれより小さくすることで、重なりがわずか1〜2行しか
    無い場合でも、テンプレートが行の境界をまたいで不安定になるのを防ぐ。
    """
    gray = cv2.cvtColor(img_list[:, x_start:x_end], cv2.COLOR_BGR2GRAY).astype(np.float64)
    signal = gray.mean(axis=1)
    signal = signal - signal.mean()
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')[n - 1:]
    autocorr /= (autocorr[0] + 1e-9)

    best_p, best_v = None, -1.0
    upper = min(max_period, n - 1)
    for p in range(min_period, upper):
        if autocorr[p] > autocorr[p - 1] and autocorr[p] > autocorr[p + 1] and autocorr[p] > best_v:
            best_v = autocorr[p]
            best_p = p
    return best_p


def stitch_images(images, manual_ratio=None, threshold=0.75, search_ratio=0.85):
    """
    画像リストを縦方向に結合する。
    重なりが不十分な箇所があれば StitchError を投げて処理を中断する。

    manual_ratio: (header_ratio, footer_ratio) を指定すると自動判定の代わりに使う。
                  Noneなら各ペアごとに自動判定する（機種・キャラが混在していても安定するように）。

    search_ratio: 次の画像のうち、上から何割の範囲までをマッチング候補として探すか。
    因子リストには似たようなアイコン・行が多く並ぶため、探索範囲を全体にすると
    離れた場所にある別の行を誤って「重なり」として検出してしまうことがある。
    ある程度上側に制限しつつ、正しい重なり位置を取りこぼさない範囲に留める。
    """
    base_img = images[0]
    base_H, base_W = base_img.shape[:2]

    if manual_ratio is not None:
        header_ratio, footer_ratio = manual_ratio
    else:
        detected = detect_header_footer_ratio(images[0], images[1]) if len(images) > 1 else None
        header_ratio, footer_ratio = detected if detected else (0.33, 0.13)

    header_h = int(base_H * header_ratio)
    footer_h = int(base_H * footer_ratio)

    final_header = base_img[:header_h, :]
    final_footer = base_img[base_H - footer_h:, :]
    base_list = base_img[header_h: base_H - footer_h, :]

    x_start = int(base_W * 0.25)
    x_end = int(base_W * 0.85)

    for i in range(1, len(images)):
        img_next = images[i]
        next_H, next_W = img_next.shape[:2]

        if next_W != base_W:
            scale = base_W / next_W
            new_H = int(next_H * scale)
            img_next = cv2.resize(img_next, (base_W, new_H))

        # このペア専用のヘッダー・フッター比率を判定する（機種・キャラが途中で変わっても
        # 前後のペアの影響を受けないように、毎回そのペアの生画像同士で判定し直す）。
        if manual_ratio is not None:
            pair_header_ratio, pair_footer_ratio = header_ratio, footer_ratio
        else:
            detected_pair = detect_header_footer_ratio(images[i - 1], images[i])
            pair_header_ratio, pair_footer_ratio = detected_pair if detected_pair else (header_ratio, footer_ratio)

        curr_H = img_next.shape[0]
        curr_header_h = int(curr_H * pair_header_ratio)
        curr_footer_h = int(curr_H * pair_footer_ratio)

        next_list = img_next[curr_header_h: curr_H - curr_footer_h, :]

        # 行の高さ（周期）を検出し、テンプレートは「1行より少し小さいサイズ」にする。
        # これにより、重なりが1〜2行しか無いケースでもテンプレートが行の境界を
        # またいで不安定になることを防ぎ、マッチ精度が大幅に上がる。
        row_period = estimate_row_period(base_list, x_start, x_end)
        if row_period is None:
            row_period = max(20, int(base_H * 0.05))
        template_h = max(10, int(row_period * 0.7))
        margin = max(1, int(row_period * 0.1))

        template = base_list[-(template_h + margin):-margin, x_start:x_end]

        # 探索範囲を上側寄りに限定し、離れた場所での誤マッチを防ぐ
        max_search_h = max(template_h * 4, int(next_list.shape[0] * search_ratio))
        search_area = next_list[:max_search_h, x_start:x_end]

        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        scores = res[:, 0]

        # 「○○レース場」のように非常によく似た行が並ぶ場合、1行分ズレた位置の方が
        # わずかに高いスコアになってしまうことがある。そこで単純な最高スコアではなく、
        # 「閾値を超える最初の（＝一番早い）候補」を採用する。重なりが本当に存在するなら、
        # それは必ず一番浅い位置で最初に見つかるはずなので、この方が誤マッチに強い。
        match_y, max_val = None, -1.0
        in_cluster = False
        for y in range(len(scores)):
            if scores[y] >= threshold:
                in_cluster = True
                if scores[y] > max_val:
                    max_val = scores[y]
                    match_y = y
            elif in_cluster:
                break
        if match_y is None:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            match_y = max_loc[1]

        if max_val < threshold:
            # i番目（0-indexed）は表示上「i+1枚目」なので、直前の画像は「i枚目」
            raise StitchError(index_a=i, index_b=i + 1)

        # base_listはテンプレートの開始位置（マッチした行の途中）までで打ち切り、
        # それより先はnext_list側のデータで丸ごと置き換える。
        # base_listの末尾は、ウィンドウ端で行が部分的にしか見えていないことがあり、
        # そのまま残すと「見切れた行」の直後に次の画像の「完全な同じ行」が
        # 続いてしまい、行が潰れて見える原因になる。マッチ位置から先を
        # まるごとnext_list側に差し替えることで、この問題を避けられる。
        cut_in_base = base_list.shape[0] - template_h - margin
        base_list = base_list[:max(0, cut_in_base)]
        new_part = next_list[match_y:, :]
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
if 'widget_version' not in st.session_state:
    # リセット時にこれを増やし、貼り付けボタン／アップローダーのkeyを変えることで
    # ウィジェット自体が保持している「選択済みの画像」を強制的にクリアする
    st.session_state.widget_version = 0


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
wv = st.session_state.widget_version

with paste_col:
    st.caption("🖥️ PCの方はこちら（フォルダは開きません）")
    # ※ブラウザのClipboard APIを使うため、初回はブラウザ側の許可ダイアログが出ることがあります。
    #   Chrome / Edge / Safari で動作確認済み。Firefoxは非対応です。
    paste_result = pbutton(
        label="📋 画像を貼り付け",
        text_color="#ffffff",
        background_color="#4285f4",
        hover_background_color="#3367d6",
        key=f"paste_button_{wv}",
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
        key=f"mobile_uploader_{wv}",
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
            st.session_state.widget_version += 1  # ウィジェットのkeyを変え、選択状態を強制クリア
            st.rerun()

    with col2:
        if len(st.session_state.image_list) >= 2:
            if st.button("✨ 結合スタート！", type="primary"):
                with st.spinner("画像を結合しています... 少々お待ちください。"):
                    try:
                        final_img = stitch_images(
                            st.session_state.image_list,
                        )
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
