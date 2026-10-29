# YouTubeショート動画生成スクリプト（10秒・1080x1920・BGM付き）
# 使い方: python3 render_short.py data.json out.mp4
# data.json の例:
# {"date":"10/16","headline":"日経 大反発！","headline_value":"+898円","headline_up":true,
#  "indices":[["日経平均","65,324円","+898 (+1.39%)",true], ...4件],
#  "stocks":[["大末建設","1814","上方修正・増配・自社株買い"], ...5件]}
import json, sys, os, subprocess, tempfile, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

data = json.load(open(sys.argv[1], encoding="utf-8")); out = sys.argv[2]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"
def f(s): return ImageFont.truetype(FONT, s, index=0)
W, H, FPS = 1080, 1920, 30
BG = (12, 16, 28); UP = (255, 80, 90); DOWN = (80, 170, 255); GOLD = (255, 200, 60); WH = (240, 240, 245); GR = (150, 155, 170)

def base(d):
    d.rectangle([0, 0, W, 180], fill=(20, 26, 44))
    d.text((W//2, 95), f"{data['date']} マーケット速報", font=f(56), fill=GOLD, anchor="mm")
    d.text((W//2, H-90), "※情報提供のみ・投資助言ではありません", font=f(34), fill=GR, anchor="mm")

def scene_title(t):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); base(d)
    s = min(1, t/0.5); c = UP if data.get("headline_up", True) else DOWN
    d.text((W//2, 780), data["headline"], font=f(int(110*(0.6+0.4*s))), fill=c, anchor="mm")
    d.text((W//2, 960), data["headline_value"], font=f(150), fill=WH, anchor="mm")
    d.text((W//2, 1180), "今日の注目株 5選", font=f(90), fill=GOLD, anchor="mm")
    return im

def scene_idx(t):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); base(d)
    d.text((W//2, 330), "主要指数", font=f(80), fill=GOLD, anchor="mm")
    for i, (n, v, c, up) in enumerate(data["indices"][:4]):
        if t < i*0.3: continue
        y = 520 + i*290
        d.rounded_rectangle([60, y-110, W-60, y+110], 30, fill=(26, 34, 56))
        d.text((110, y-45), n, font=f(56), fill=WH, anchor="lm")
        d.text((W-110, y-45), v, font=f(64), fill=WH, anchor="rm")
        d.text((W-110, y+50), ("▲ " if up else "▼ ")+c, font=f(52), fill=UP if up else DOWN, anchor="rm")
    return im

def scene_stock(k):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); base(d)
    d.text((W//2, 330), "注目株 5選", font=f(80), fill=GOLD, anchor="mm")
    for i, (n, c, r) in enumerate(data["stocks"][:5]):
        y = 520 + i*250; on = i <= k
        d.rounded_rectangle([50, y-100, W-50, y+100], 28, fill=(26, 34, 56) if on else (18, 22, 36),
                            outline=GOLD if i == k else None, width=6)
        if not on:
            d.text((W//2, y), "？？？", font=f(60), fill=(70, 75, 90), anchor="mm"); continue
        d.text((95, y), str(i+1), font=f(90), fill=GOLD, anchor="lm")
        name = f"{n}（{c}）"; size = 54
        while size > 36 and f(size).getlength(name) > W-260: size -= 2
        d.text((200, y-38), name, font=f(size), fill=WH, anchor="lm")
        col = UP if r.startswith("+") else (DOWN if r.startswith("-") else GR)
        d.text((200, y+45), r, font=f(42), fill=col, anchor="lm")
    return im

def bgm(path, dur=10.0, sr=44100):
    # 明るいオルゴール調（オリジナル旋律・テンポ132）
    t = np.arange(int(sr*dur))/sr; mix = np.zeros_like(t); beat = 60/132
    def note(st, freq, amp, decay=1.6):
        m = t >= st; x = t[m]-st
        # オルゴールの櫛歯：基音＋非整数倍音（金属的な響き）、速いアタックと長い減衰
        tone = (np.sin(2*np.pi*freq*x)
                + 0.35*np.sin(2*np.pi*freq*2.0*x)*np.exp(-x*4)
                + 0.18*np.sin(2*np.pi*freq*5.4*x)*np.exp(-x*12)
                + 0.08*np.sin(2*np.pi*freq*8.9*x)*np.exp(-x*25))
        mix[m] += amp*tone*np.clip(x/0.002, 0, 1)*np.exp(-x/decay)
    N = {"C5":523.3,"D5":587.3,"E5":659.3,"F5":698.5,"G5":784.0,"A5":880.0,"B5":987.8,"C6":1046.5,"D6":1174.7,"E6":1318.5,"G6":1568.0,
         "C4":261.6,"E4":329.6,"F4":349.2,"G4":392.0,"A3":220.0,"B3":246.9,"D4":293.7}
    # 明るいオルゴール：テンポ132・C-F-G-C の長調進行・高音域で跳ねる旋律
    N.update({"F6":1396.9,"A6":1760.0,"C7":2093.0,"A4":440.0,"B4":493.9})
    melody = [(0,"C6"),(0.5,"E6"),(1,"G6"),(1.5,"E6"),(2,"C7"),(3,"G6"),
              (4,"A6"),(4.5,"F6"),(5,"C6"),(5.5,"F6"),(6,"A6"),(7,"F6"),
              (8,"G6"),(8.5,"D6"),(9,"B5"),(9.5,"D6"),(10,"G6"),(10.5,"A6"),(11,"G6"),
              (12,"E6"),(12.5,"G6"),(13,"C7"),(14,"G6"),(14.5,"E6"),(15,"C6"),
              (16,"D6"),(16.5,"E6"),(17,"F6"),(17.5,"G6"),(18,"A6"),(18.5,"B5"),(19,"C7")]
    for b, nm in melody:
        if b*beat < dur-0.3: note(b*beat, N[nm], 0.30, 1.2)
    # 伴奏：16分の分散和音で軽快に
    arps = [["C5","E5","G5","C6"],["F4","A4","C5","F5"],["G4","B4","D5","G5"],["C5","E5","G5","C6"]]
    for i in range(int(dur/(beat/2))+1):
        bar = int(i*0.5//4) % 4; nm = arps[bar][i % 4]
        note(i*beat/2, N[nm], 0.12, 0.6)
    # きらめき（高音のトリル）を小節頭に
    for b in range(0, 24, 4): note(b*beat+beat*0.75, N["C7"], 0.07, 0.3)
    for nm in ["C5","E5","G5","C6","E6"]: note(19*beat, N[nm], 0.10, 1.3)
    mix *= np.minimum(1, np.minimum(t/0.05, (dur-t)/0.6)); mix = mix/np.max(np.abs(mix))*0.75
    # ほんのり残響
    d1 = int(sr*0.09); d2 = int(sr*0.17); rev = mix.copy()
    rev[d1:] += 0.25*mix[:-d1]; rev[d2:] += 0.15*mix[:-d2]; rev = rev/np.max(np.abs(rev))*0.75
    l = rev; r = np.concatenate([np.zeros(int(sr*0.012)), rev[:-int(sr*0.012)]])
    w = wave.open(path, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(np.stack([(l*32767).astype(np.int16), (r*32767).astype(np.int16)], 1).tobytes()); w.close()

tmp = tempfile.mkdtemp(); n = 0
def emit(im):
    global n; im.save(f"{tmp}/{n:04d}.png"); n += 1
for i in range(60): emit(scene_title(i/FPS))
for i in range(75): emit(scene_idx(i/FPS))
for k in range(5):
    im = scene_stock(k)
    for i in range(30): emit(im)
for i in range(15): emit(scene_stock(4))
# BGM：R&B風オリジナル曲 3(サンセット)と5(シティ・ナイト)を日付で交互に使用
# 第3引数で bgm/xxx.wav を直接指定も可。指定がなければ日付の通し番号が偶数→3、奇数→5
import re, datetime, shutil
here = os.path.dirname(os.path.abspath(__file__)); bdir = os.path.join(here, "..", "bgm")
if len(sys.argv) > 3:
    bgm_file = sys.argv[3]
else:
    mt = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(out))
    day = datetime.date(*map(int, mt.groups())) if mt else datetime.datetime.utcnow().date()
    bgm_file = os.path.join(bdir, "3_sunset_groove.wav" if day.toordinal() % 2 == 0 else "5_city_night.wav")
if os.path.exists(bgm_file):
    shutil.copy(bgm_file, f"{tmp}/bgm.wav"); print("BGM:", os.path.basename(bgm_file))
else:
    bgm(f"{tmp}/bgm.wav"); print("BGM: 合成オルゴール（R&B音源が見つからないため）")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{tmp}/%04d.png",
                "-i", f"{tmp}/bgm.wav", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", out], check=True)
print("OK", out)
