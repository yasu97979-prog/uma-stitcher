import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mss
import io
import ctypes
from ctypes import wintypes
from PIL import Image
import keyboard

# ==============================================================================
# Claude様 作成の高度な画像結合ロジック
# ==============================================================================
class StitchError(Exception):
    def __init__(self, index_a, index_b):
        self.index_a = index_a
        self.index_b = index_b
        super().__init__(f"{index_a}枚目と{index_b}枚目の間で十分な重なりが検出できませんでした")

def detect_header_footer_ratio(img_a, img_b, diff_threshold=15, min_run=6, x_frac=(0.25, 0.85)):
    Ha, Wa = img_a.shape[:2]
    Hb, Wb = img_b.shape[:2]
    W = min(Wa, Wb)
    x_start, x_end = int(W * x_frac[0]), int(W * x_frac[1])

    H_top = min(Ha, Hb)
    a_top = img_a[:H_top, x_start:x_end].astype(np.int16)
    b_top = img_b[:H_top, x_start:x_end].astype(np.int16)
    diff_top = np.mean(np.abs(a_top - b_top), axis=(1, 2))
    header_h = 0
    for y in range(H_top - min_run):
        if diff_top[y:y + min_run].mean() > diff_threshold:
            header_h = y
            break

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

    if header_ratio <= 0 or header_ratio >= 0.7 or footer_ratio >= 0.5 or (header_ratio + footer_ratio) >= 0.85:
        return None
    return header_ratio, footer_ratio

def estimate_row_period(img_list, x_start, x_end, min_period=15, max_period=150):
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

        if manual_ratio is not None:
            pair_header_ratio, pair_footer_ratio = header_ratio, footer_ratio
        else:
            detected_pair = detect_header_footer_ratio(images[i - 1], images[i])
            pair_header_ratio, pair_footer_ratio = detected_pair if detected_pair else (header_ratio, footer_ratio)

        curr_H = img_next.shape[0]
        curr_header_h = int(curr_H * pair_header_ratio)
        curr_footer_h = int(curr_H * pair_footer_ratio)

        next_list = img_next[curr_header_h: curr_H - curr_footer_h, :]

        row_period = estimate_row_period(base_list, x_start, x_end)
        if row_period is None:
            row_period = max(20, int(base_H * 0.05))
        template_h = max(10, int(row_period * 0.7))
        margin = max(1, int(row_period * 0.1))

        template = base_list[-(template_h + margin):-margin, x_start:x_end]
        max_search_h = max(template_h * 4, int(next_list.shape[0] * search_ratio))
        search_area = next_list[:max_search_h, x_start:x_end]

        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        match_y = max_loc[1]

        if max_val < threshold:
            raise StitchError(index_a=i, index_b=i + 1)

        cut_in_base = base_list.shape[0] - template_h - margin
        base_list = base_list[:max(0, cut_in_base)]
        new_part = next_list[match_y:, :]
        base_list = np.vstack((base_list, new_part))

    return np.vstack((final_header, base_list, final_footer))

# ==============================================================================
# クリップボードへ画像をコピーする機能
# ==============================================================================
def copy_image_to_clipboard(cv_img):
    output = io.BytesIO()
    image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:] 

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE

    user32.OpenClipboard(0)
    user32.EmptyClipboard()

    hGlobalMem = kernel32.GlobalAlloc(0x0042, len(data))
    pGlobalMem = kernel32.GlobalLock(hGlobalMem)
    ctypes.memmove(pGlobalMem, data, len(data))
    kernel32.GlobalUnlock(hGlobalMem)

    user32.SetClipboardData(8, hGlobalMem)
    user32.CloseClipboard()

# ==============================================================================
# 手動キャプチャ型 デスクトップアプリGUI
# ==============================================================================
class ManualStitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐴 手動スクロール結合ツール")
        self.root.geometry("450x800")
        self.root.attributes("-topmost", True)

        try:
            self.root.attributes("-transparentcolor", "magenta")
        except Exception:
            pass

        self.captured_images = []
        self.border_width = 4 

        # --- コントロールパネル ---
        self.ctrl_frame = tk.Frame(root, bg="#f8f9fa", bd=1, relief=tk.RAISED)
        self.ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        self.capture_btn = tk.Button(self.ctrl_frame, text="📸 キャプチャ(X) [0枚]", command=self.capture_frame, 
                                     bg="#34a853", fg="white", font=("Arial", 11, "bold"))
        self.capture_btn.pack(side=tk.LEFT, padx=3, pady=5)

        # ★ 新規追加: 1枚戻す(Undo)ボタン
        self.undo_btn = tk.Button(self.ctrl_frame, text="↩️ 戻す(C)", command=self.undo_capture, 
                                  bg="#fbbc04", fg="black", font=("Arial", 10, "bold"))
        self.undo_btn.pack(side=tk.LEFT, padx=3, pady=5)

        self.stitch_btn = tk.Button(self.ctrl_frame, text="✨ 結合", command=self.process_and_copy, 
                                    bg="#4285f4", fg="white", font=("Arial", 11, "bold"))
        self.stitch_btn.pack(side=tk.LEFT, padx=3, pady=5)

        self.reset_btn = tk.Button(self.ctrl_frame, text="🗑️ リセット", command=self.reset_images, 
                                   bg="#ea4335", fg="white", font=("Arial", 9))
        self.reset_btn.pack(side=tk.RIGHT, padx=3, pady=5)

        self.frame_canvas = tk.Canvas(root, bg="magenta", highlightthickness=self.border_width, highlightbackground="red")
        self.frame_canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        try:
            keyboard.add_hotkey('x', self.trigger_capture)
            keyboard.add_hotkey('c', self.trigger_undo) # ★ Cキーの監視を追加
        except Exception as e:
            print(f"ホットキーの登録に失敗しました: {e}")

    def trigger_capture(self):
        self.root.after(0, self.capture_frame)

    def trigger_undo(self):
        self.root.after(0, self.undo_capture)

    def capture_frame(self):
        x = self.frame_canvas.winfo_rootx() + self.border_width
        y = self.frame_canvas.winfo_rooty() + self.border_width
        w = self.frame_canvas.winfo_width() - (self.border_width * 2)
        h = self.frame_canvas.winfo_height() - (self.border_width * 2)
        
        monitor = {"top": y, "left": x, "width": w, "height": h}

        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            self.captured_images.append(img)
            
        count = len(self.captured_images)
        self.capture_btn.config(text=f"📸 キャプチャ(X) [{count}枚]")
        
        self.frame_canvas.config(highlightbackground="blue")
        self.root.after(150, lambda: self.frame_canvas.config(highlightbackground="red"))

    def undo_capture(self):
        """直前にキャプチャした画像を1枚削除する"""
        if len(self.captured_images) > 0:
            self.captured_images.pop() # リストの末尾（最後）の画像を削除
            count = len(self.captured_images)
            self.capture_btn.config(text=f"📸 キャプチャ(X) [{count}枚]")
            
            # 戻したことが視覚的にわかるように枠を「黄色」に光らせる
            self.frame_canvas.config(highlightbackground="yellow")
            self.root.after(150, lambda: self.frame_canvas.config(highlightbackground="red"))

    def reset_images(self):
        self.captured_images = []
        self.capture_btn.config(text="📸 キャプチャ(X) [0枚]")

    def process_and_copy(self):
        if len(self.captured_images) < 2:
            messagebox.showwarning("警告", "結合には2枚以上のキャプチャが必要です！")
            return

        original_text = self.stitch_btn.cget("text")
        self.stitch_btn.config(text="⚙️ 結合中...", bg="orange")
        self.root.update()

        try:
            final_img = stitch_images(self.captured_images)
            copy_image_to_clipboard(final_img)
            messagebox.showinfo("成功", "🎉 画像の結合とコピーに成功しました！\n\nCtrl+Vでどこにでも貼り付けられます。")
        except StitchError as e:
            messagebox.showerror("結合エラー", f"❌ 結合に失敗しました。\n{e}\n\n「↩️ 戻す(C)」を押して最後の1枚を取り消し、重なる部分を増やしてキャプチャし直してみてください。")
        except Exception as e:
            messagebox.showerror("エラー", f"❌ 予期せぬエラーが発生しました:\n{e}")
        finally:
            self.stitch_btn.config(text=original_text, bg="#4285f4")

if __name__ == "__main__":
    root = tk.Tk()
    app = ManualStitcherApp(root)
    root.mainloop()
