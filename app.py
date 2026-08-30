import streamlit as st
import cv2
import numpy as np
import hashlib
import base64
import glob
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from streamlit_paste_button import paste_image_button as pbutton


def find_japanese_font():
    """
    日本語フォントを探す。デプロイ環境によって入っているフォントが異なるため、
    候補をいくつか順に探し、見つからなければNoneを返す（呼び出し側でフォールバックする）。
    """
    candidates = [
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    ]
    for path in candidates:
        if glob.glob(path):
            return glob.glob(path)[0]
    # 上記で見つからない場合は、システム内を広く検索する
    for pattern in ["/usr/share/fonts/**/*CJK*.tt*", "/usr/share/fonts/**/*japanese*.tt*",
                    "/usr/share/fonts/**/*Noto*JP*.tt*"]:
        found = glob.glob(pattern, recursive=True)
        if found:
            return found[0]
    return None


JP_FONT_PATH = find_japanese_font()


class StitchError(Exception):
    """画像結合に失敗した際に、問題のあった画像番号を保持して投げる例外"""
    def __init__(self, index_a, index_b):
        self.index_a = index_a
        self.index_b = index_b
        super().__init__(f"{index_a}枚目と{index_b}枚目の間で十分な重なりが検出できませんでした")


def detect_tab_and_list_start(img, green=(8, 220, 158), tol=40):
    """
    画面上部の「スキル／継承／育成情報」タブのうち、緑色でハイライトされている
    位置からアクティブなタブの種類を判定し、リスト本体が始まるY座標を返す。
    """
    H, W = img.shape[:2]
    target = np.array(green)
    for y in range(int(H * 0.05), int(H * 0.35)):
        row = img[y, :int(W * 0.95)].astype(int)
        dist = np.sqrt(((row - target) ** 2).sum(axis=1))
        green_mask = dist < tol
        if green_mask.sum() > W * 0.15:
            green_xs = np.where(green_mask)[0]
            center_x = green_xs.mean()
            if center_x < W * 0.4:
                tab = "スキル"
            elif center_x < W * 0.7:
                tab = "継承"
            else:
                tab = "育成情報"
            yy = y
            while yy < H and np.sqrt(((img[yy, :int(W * 0.95)].astype(int) - target) ** 2).sum(axis=1)).min() < tol + 20:
                yy += 1
            return tab, yy
    return None, None


def find_avatar_section_starts(img, x_range=(40, 85), bg_thresh=225, frac_thresh=0.5,
                                min_run=30, min_gap=200, gap_tol=8):
    """
    継承タブで、親・祖父母のアバターサムネイルが表示されている位置（＝各因子セクションの
    開始位置）を検出する。画面一番上のアバターは除外し、それ以降に現れるものだけを返す。

    キャラクターの髪・服の色によっては、アバター領域の判定が数px単位で途切れることがあるため、
    gap_tol分までの短い途切れは同じアバターとして許容し、判定漏れを防ぐ。
    """
    H = img.shape[0]
    band = img[:, x_range[0]:x_range[1]].astype(int)
    is_bg = (band[:, :, 0] > bg_thresh) & (band[:, :, 1] > bg_thresh) & (band[:, :, 2] > bg_thresh)
    non_bg_frac = 1 - is_bg.mean(axis=1)
    is_avatar_row = non_bg_frac > frac_thresh
    starts = []
    y = 0
    while y < H:
        if is_avatar_row[y]:
            run_start = y
            last_true = y
            y += 1
            while y < H:
                if is_avatar_row[y]:
                    last_true = y
                    y += 1
                elif y - last_true <= gap_tol:
                    y += 1
                else:
                    break
            if last_true - run_start >= min_run:
                if not starts or run_start - starts[-1] >= min_gap:
                    starts.append(run_start)
        else:
            y += 1
    return starts[1:] if len(starts) > 1 else []


def find_list_true_end(img, list_start, row_period, upper_bound=None, x_frac=(0.03, 0.94),
                        bg_thresh=240, density_thresh=0.05):
    """
    リスト本体が実際に終わる位置（「閉じる」ボタン手前や、次のセクションの見出しテキスト手前）を探す。
    行と行の間の小さな隙間は行の高さ（row_period）よりずっと短いのに対し、
    リストが終わった後は「内容の無い行」がまとまって続くことを利用して区別する。
    単純な均一性（分散）だけで判定すると、通常の行間ギャップの長さが画像ごとにばらつき
    誤判定しやすいため、行の高さに対する相対的な長さ（row_period基準）で判定する。

    upper_bound：これ以上は絶対に探索しない上限（次のセクションの開始位置など）。
    セクションの間に十分な空白が無い場合、探索がその先の別セクションの内容まで
    入り込んでしまうことがあるため、安全のため上限を設けられるようにしている。
    見つからなければupper_bound（無ければ画像の高さ）を返す。
    """
    H, W = img.shape[:2]
    limit = min(H, upper_bound) if upper_bound is not None else H
    x0, x1 = int(W * x_frac[0]), int(W * x_frac[1])
    band = img[:, x0:x1].astype(int)
    is_bg = (band[:, :, 0] > bg_thresh) & (band[:, :, 1] > bg_thresh) & (band[:, :, 2] > bg_thresh)
    non_bg_density = 1 - is_bg.mean(axis=1)
    min_blank_run = max(6, int(row_period * 0.35))
    y = list_start + int(row_period * 0.5)  # 最低半行分は必ず内容があるはずなので、そこから探索開始
    while y < limit:
        if non_bg_density[y] < density_thresh:
            run_start = y
            while y < limit and non_bg_density[y] < density_thresh:
                y += 1
            if y - run_start >= min_blank_run:
                return run_start
        else:
            y += 1
    return limit


def count_boxes_in_range(img, y_start, y_end, row_period, x_pairs=((0.03, 0.48), (0.52, 0.94)),
                          bg_thresh=240, std_thresh=5):
    """
    指定した縦範囲内で、行の周期ごとに左右のセルに「箱（因子/スキル）」があるかを判定して数える。

    右端ギリギリ（0.94超）にはスクロールバーの細い線が写り込むことがあり、これを内容と
    誤認識しないよう右セルの範囲を少し内側に絞っている。また、行数で単純に整数の周期を
    掛け算すると、割り切れない端数が行を重ねるごとに蓄積して最後の方の行位置がずれるため、
    範囲全体を実際の行数で割った「正確な周期」を使って各行の位置を計算し直している。

    判定はセルの上半分だけを見る。1個だけの単独行（右セルが本来空）の場合、
    左の箱の角丸の影がわずかに右セル側のセル下端に薄くはみ出すことがあり、
    セル全体で判定すると誤って「内容あり」と数えてしまうことがあるため。
    """
    W = img.shape[1]
    n_rows = max(0, int(round((y_end - y_start) / row_period)))
    if n_rows == 0:
        return 0
    precise_period = (y_end - y_start) / n_rows
    count = 0
    for r in range(n_rows):
        ry0 = y_start + int(round(r * precise_period)) + 4
        ry1 = y_start + int(round((r + 1) * precise_period)) - 4
        if ry1 <= ry0 or ry1 > y_end:
            continue
        ry_mid = ry0 + (ry1 - ry0) // 2
        for xs, xe in x_pairs:
            x0, x1 = int(W * xs), int(W * xe)
            cell = img[ry0:ry_mid, x0:x1]
            if cell.size == 0:
                continue
            mean = cell.reshape(-1, 3).mean(axis=0)
            std = cell.reshape(-1, 3).std()
            is_bg = all(c > bg_thresh for c in mean) and std < std_thresh
            if not is_bg:
                count += 1
    return count


def count_skill_total(img):
    """スキルタブ：リスト内の全スキル数を数える"""
    tab, list_start = detect_tab_and_list_start(img)
    if list_start is None:
        return None
    H, W = img.shape[:2]
    row_period = estimate_row_period(img[list_start:min(list_start + 1500, H)], int(W * 0.03), int(W * 0.95))
    if row_period is None:
        return None
    list_end = find_list_true_end(img, list_start, row_period)
    total = count_boxes_in_range(img, list_start, list_end, row_period)
    return {"スキル合計": total}


def count_inheritance_factors(img):
    """
    継承タブ：親・祖父母1・祖父母2ごとの因子数と合計を数える。
    各セクションの先頭3つ（青・ピンク・黄緑のカテゴリタグ）は因子として数えない。

    セクションの終わりは、次のセクションの開始位置ではなく、常にそのセクション自身の
    内容が実際に終わる位置（find_list_true_end）を使う。次のセクション開始位置を
    そのまま使うと、間にある「継承元」という見出しテキストまで因子として数えて
    しまうことがあるため。
    """
    tab, _ = detect_tab_and_list_start(img)
    section_starts = find_avatar_section_starts(img)
    if not section_starts:
        return None
    H, W = img.shape[:2]
    labels = ["親の因子", "祖父母1の因子", "祖父母2の因子"]
    x_start, x_end = int(W * 0.25), int(W * 0.85)
    results = {}
    total = 0
    for i, (label, s) in enumerate(zip(labels, section_starts)):
        row_period = estimate_row_period(img[s:min(s + 800, H)], x_start, x_end)
        if row_period is None:
            continue
        # 次のセクションが存在する場合は、そこを超えて探索しないよう上限として渡す
        upper_bound = section_starts[i + 1] if i < len(section_starts) - 1 else None
        e = find_list_true_end(img, s, row_period, upper_bound=upper_bound)
        all_boxes = count_boxes_in_range(img, s, e, row_period)
        factor_count = max(0, all_boxes - 3)
        results[label] = factor_count
        total += factor_count
    results["合計"] = total
    return results


def render_count_overlay(img, count_dict, position="top"):
    """カウント結果をテキストとして画像の上部または下部に描画する"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    W = pil_img.width
    font_size = max(20, W // 22)
    font = None
    if JP_FONT_PATH:
        try:
            font = ImageFont.truetype(JP_FONT_PATH, font_size)
        except Exception:
            font = None
    if font is None:
        # 日本語フォントが見つからない場合、文字が表示されない（豆腐化）ので、
        # ラベルをローマ字に変換してでも読める形にフォールバックする
        romanize = {
            "スキル合計": "Skills total", "親の因子": "Parent factors",
            "祖父母1の因子": "Grandparent1 factors", "祖父母2の因子": "Grandparent2 factors",
            "合計": "Total",
        }
        count_dict = {romanize.get(k, k): v for k, v in count_dict.items()}
        font = ImageFont.load_default()

    lines = [f"{k}：{v}個" if JP_FONT_PATH else f"{k}: {v}" for k, v in count_dict.items()]
    line_height = int(font_size * 1.4)
    pad = int(font_size * 0.6)
    panel_h = line_height * len(lines) + pad * 2

    panel = Image.new("RGB", (W, panel_h), (255, 250, 230))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, W - 1, panel_h - 1], outline=(230, 180, 60), width=3)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_height), line, fill=(90, 60, 10), font=font)

    if position == "top":
        combined = Image.new("RGB", (W, pil_img.height + panel_h), (255, 255, 255))
        combined.paste(panel, (0, 0))
        combined.paste(pil_img, (0, panel_h))
    else:
        combined = Image.new("RGB", (W, pil_img.height + panel_h), (255, 255, 255))
        combined.paste(pil_img, (0, 0))
        combined.paste(panel, (0, pil_img.height))

    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


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

    count_enabled = st.checkbox(
        "🔢 結合後にスキル／因子の数を自動で数えて画像に記載する（自動判定・多少の誤差が出る場合があります）",
        value=False,
    )

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

                        if count_enabled:
                            tab, _ = detect_tab_and_list_start(final_img)
                            counts = None
                            if tab == "スキル":
                                counts = count_skill_total(final_img)
                            elif tab == "継承":
                                counts = count_inheritance_factors(final_img)

                            if counts:
                                st.info(
                                    "📊 " + " ／ ".join(f"{k}：{v}個" for k, v in counts.items())
                                    + "\n\n（自動判定のため、多少の誤差が出ることがあります）"
                                )
                                final_img = render_count_overlay(final_img, counts, position="top")
                            else:
                                st.warning(
                                    "⚠️ タブの種類（スキル／継承）を自動判定できなかったため、"
                                    "数のカウントはスキップされました。"
                                )

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
