# 耳ざわりのよいR&B風オリジナルBGM（10秒）×5パターン生成
# 使い方: python3 rnb_bgm.py 出力フォルダ
import numpy as np, wave, sys, os
from scipy.signal import butter, sosfilt

SR = 44100; DUR = 10.0; N = int(SR*DUR)
def hz(m): return 440*2**((m-69)/12)
def tt(d): return np.arange(int(SR*d))/SR
def lpf(x, fc, o=2): return sosfilt(butter(o, fc, 'low', fs=SR, output='sos'), x)
def hpf(x, fc, o=2): return sosfilt(butter(o, fc, 'high', fs=SR, output='sos'), x)
def bpf(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)

class Mix:
    def __init__(s): s.L = np.zeros(N); s.R = np.zeros(N)
    def put(s, sig, st, pan=0.0, g=1.0):
        i = int(st*SR)
        if i >= N or i < 0: return
        x = sig[:N-i]*g
        s.L[i:i+len(x)] += x*np.sqrt((1-pan)/2); s.R[i:i+len(x)] += x*np.sqrt((1+pan)/2)

# ---------- 音色 ----------
def rhodes(m, ln, vel=1.0, bright=1.0):
    """FM合成のエレピ（ローズ風）：柔らかい胴鳴り＋アタックの金属的なティン"""
    x = tt(ln+0.6); f = hz(m)
    idx = 1.6*bright*np.exp(-x*3.0)+0.25
    body = np.sin(2*np.pi*f*x + idx*np.sin(2*np.pi*f*x))
    tine = 0.18*bright*np.sin(2*np.pi*f*x + 2.0*np.exp(-x*25)*np.sin(2*np.pi*f*14*x))*np.exp(-x*12)
    env = np.minimum(1, x/0.004)*np.exp(-x*0.9)*np.clip((ln+0.6-x)/0.6, 0, 1)
    return (body+tine)*env*vel

def pad(notes, ln):
    x = tt(ln+0.8); s = 0
    for m in notes:
        for d in (-0.006, 0, 0.0065):
            s = s + (2*((hz(m)*(1+d)*x) % 1)-1)
    s = lpf(s/len(notes)/3, 1400, 2)
    env = np.minimum(1, x/0.6)*np.clip((ln+0.8-x)/0.8, 0, 1)
    return s*env

def sub(m, ln, slide_from=None):
    x = tt(ln+0.05); f = hz(m)
    if slide_from is not None:
        fr = hz(slide_from); fcur = f + (fr-f)*np.exp(-x*25)
        ph = 2*np.pi*np.cumsum(fcur)/SR
    else:
        ph = 2*np.pi*f*x
    s = np.sin(ph)+0.25*np.sin(2*ph)+0.08*np.sin(3*ph)
    env = np.minimum(1, x/0.01)*np.exp(-x*0.6)*np.clip((ln+0.05-x)/0.05, 0, 1)
    return s*env

def pluck(m, ln):
    """ミュートギター風のプラック"""
    x = tt(ln+0.2); f = hz(m)
    s = 2*((f*x) % 1)-1
    s = lpf(s, 2400, 2)*np.exp(-x*9)
    return s*np.minimum(1, x/0.002)

def lead(m, ln, kind="flute"):
    x = tt(ln+0.35); f = hz(m)
    vib = 1+0.005*np.sin(2*np.pi*5.2*x)*np.clip((x-0.18)/0.2, 0, 1)
    ph = 2*np.pi*f*np.cumsum(vib)/SR
    if kind == "flute":
        s = np.sin(ph)+0.12*np.sin(2*ph)+0.03*np.random.default_rng(int(m)).standard_normal(len(x))
        s = lpf(s, 3500, 2)
    elif kind == "bell":
        s = np.sin(ph+0.8*np.exp(-x*4)*np.sin(3.5*ph))
    else:  # "ooh" 声っぽい音色（フォルマント）
        saw = 2*((np.cumsum(f*vib)/SR) % 1)-1
        s = bpf(saw, 300, 900)*1.0+bpf(saw, 700, 1300)*0.6+bpf(saw, 2300, 2900)*0.15
        s = s*2.2
    env = np.minimum(1, x/0.06)*np.clip((ln+0.35-x)/0.35, 0, 1)*(0.85+0.15*np.exp(-x*3))
    return s*env

RNG = np.random.default_rng(3)
def kick(): x = tt(0.5); return np.sin(2*np.pi*(42*x+3.2*(1-np.exp(-x*22))))*np.exp(-x*6.5)
def snare():
    x = tt(0.35); nz = RNG.standard_normal(len(x))
    s = bpf(nz, 900, 6000)*np.exp(-x*16)*0.55 + np.sin(2*np.pi*185*x)*np.exp(-x*28)*0.5
    return lpf(s, 7000)
def rim():
    x = tt(0.08); return (np.sin(2*np.pi*1650*x)*0.6+bpf(RNG.standard_normal(len(x)), 1500, 5000)*0.5)*np.exp(-x*60)
def snap():
    x = tt(0.2); nz = RNG.standard_normal(len(x))
    return bpf(nz, 1200, 5500)*(np.exp(-x*40)+0.5*np.exp(-np.maximum(x-0.008, 0)*35)*(x > 0.008))
def hat(open_=False):
    x = tt(0.3 if open_ else 0.05)
    return hpf(RNG.standard_normal(len(x)), 7000, 2)*np.exp(-x*(10 if open_ else 80))*0.3
def shaker():
    x = tt(0.09); return bpf(RNG.standard_normal(len(x)), 4000, 10000)*np.sin(np.pi*np.minimum(x/0.09, 1))*0.18

# ---------- 共通の組み立て ----------
def build(p, out):
    M = Mix(); beat = 60/p["bpm"]; sw = p.get("swing", 0.6)  # 8分のハネ
    def eighth(i):  # i番目の8分音符の開始時刻（スイング付き）
        b = i//2; return b*beat + (beat*sw if i % 2 else 0)
    def six(i):     # 16分（スイング付き）
        b = i//4; r = i % 4
        pos = [0, 0.25+ (sw-0.5)*0.5, 0.5, 0.75+(sw-0.5)*0.5][r]
        return (b+pos)*beat
    nb = int(DUR/beat)+2
    # ドラム
    K, S, RM, SN, H, HO, SH = kick(), snare(), rim(), snap(), hat(), hat(True), shaker()
    for b in range(nb):
        st = b*beat; bar_pos = b % 4
        for kp in p["kick"]:
            if int(kp) == bar_pos: M.put(K, st+(kp-int(kp))*beat, 0, 0.85)
        if bar_pos in (1, 3):
            back = {"snare": S, "rim": RM, "snap": SN}[p["back"]]
            M.put(back, st+0.012, 0.05, {"snare": 0.5, "rim": 0.35, "snap": 0.45}[p["back"]])
        if p.get("ghost") and bar_pos == 2: M.put(RM, st+beat*0.75, -0.2, 0.12)
    hp = p.get("hats", "8")
    if hp == "8":
        for i in range(nb*2): M.put(H, eighth(i), 0.35, 0.55 if i % 2 == 0 else 0.35)
    elif hp == "16":
        for i in range(nb*4):
            M.put(H, six(i), 0.35, [0.5, 0.2, 0.35, 0.22][i % 4])
    if p.get("shaker"):
        for i in range(nb*4): M.put(SH, six(i), -0.4, [0.6, 0.3, 0.45, 0.3][i % 4])
    if p.get("open_hat"):
        for b in range(nb):
            if b % 4 == 3: M.put(HO, b*beat+beat*sw, 0.35, 0.3)

    # コード・ベース
    cl = p["chord_beats"]*beat
    chords = p["chords"]
    nchord = int(DUR/cl)+1
    prev_root = None
    for c in range(nchord):
        ch = chords[c % len(chords)]; st = c*cl; root, notes = ch
        # エレピのコンピング（パターン：拍位置, 長さ拍）
        for pos, ln in p["comp"]:
            if pos >= p["chord_beats"]: continue
            for j, m in enumerate(notes):
                strum = j*0.012
                M.put(rhodes(m, ln*beat, 0.30, p.get("bright", 1.0)), st+pos*beat+strum,
                      -0.35+0.7*j/max(1, len(notes)-1), p.get("keys_gain", 1.0))
        if p.get("pad"):
            M.put(pad([n+12 for n in notes[:3]], cl), st, 0, 0.16)
        if p.get("pluck"):
            for i, (pos, k) in enumerate(p["pluck"]):
                if pos < p["chord_beats"]:
                    M.put(pluck(notes[k % len(notes)]+12, beat*0.25), st+pos*beat, 0.45, 0.22)
        # ベース
        for pos, ln, off in p["bass"]:
            if pos >= p["chord_beats"]: continue
            m = root+off
            M.put(sub(m, ln*beat, slide_from=(prev_root if pos == 0 and p.get("slide") and prev_root else None)),
                  st+pos*beat, 0, 0.62)
        prev_root = root
    # メロディ
    for b, m, ln in p["melody"]:
        st = b*beat
        if st > DUR-0.5: continue
        s = lead(m, ln*beat, p["lead"])
        M.put(s, st, 0.1, p.get("lead_gain", 0.22))
        M.put(lpf(s, 2500)*0.3, st+beat*0.75, -0.45, p.get("lead_gain", 0.22))  # ディレイ
    # 仕上げ：レコードノイズ、残響、ローパス、ソフトサチュレーション
    L, R = M.L, M.R
    if p.get("vinyl"):
        crack = np.zeros(N); idx = RNG.integers(0, N, 60); crack[idx] = RNG.uniform(-1, 1, 60)
        hiss = lpf(RNG.standard_normal(N), 5000)*0.004
        vin = lpf(crack, 6000)*0.08+hiss; L += vin; R += vin
    def verb(x):
        y = x.copy()
        for d, g in [(0.029, 0.32), (0.037, 0.28), (0.051, 0.24), (0.067, 0.2), (0.089, 0.16), (0.113, 0.12)]:
            k = int(d*SR); y[k:] += g*lpf(x, 4000)[:-k]
        return y
    L, R = verb(L)*0.8+L*0.2, verb(np.roll(R, 37))*0.8+R*0.2
    L, R = lpf(L, p.get("master_lp", 11000)), lpf(R, p.get("master_lp", 11000))
    if p.get("wobble"):  # ローファイの揺れ（音量で軽く）
        t = np.arange(N)/SR; wob = 1+0.03*np.sin(2*np.pi*0.7*t); L *= wob; R *= wob
    t = np.arange(N)/SR; fade = np.minimum(1, np.minimum(t/0.05, (DUR-t)/0.9))
    L *= fade; R *= fade
    pk = max(np.abs(L).max(), np.abs(R).max())
    L = np.tanh(L/pk*1.3)/np.tanh(1.3)*0.88; R = np.tanh(R/pk*1.3)/np.tanh(1.3)*0.88
    d = np.stack([(L*32767).astype(np.int16), (R*32767).astype(np.int16)], 1)
    w = wave.open(out, 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(d.tobytes()); w.close()

# ---------- 5パターン ----------
P = {}
# 1. ミッドナイト・ネオソウル（82BPM, Dm9 - G13 - Cmaj9 - A7）
P["1_midnight_neosoul"] = dict(
    bpm=82, swing=0.62, chord_beats=4, back="snare", ghost=True, hats="16", open_hat=True,
    kick=[0, 1.75, 2.5],
    chords=[(38, [53, 57, 60, 64]), (43, [53, 59, 64, 65]), (36, [52, 55, 59, 62]), (45, [55, 61, 64, 67])],
    comp=[(0, 1.4), (1.75, 0.6), (2.5, 1.2), (3.5, 0.4)],
    bass=[(0, 1.5, 0), (1.75, 0.5, 0), (2.5, 1.0, 7), (3.5, 0.4, 12)], slide=True,
    melody=[(0.5, 74, 0.5), (1, 76, 0.5), (1.5, 77, 1.2), (3, 76, 0.5), (3.5, 74, 0.5),
            (4.5, 74, 0.5), (5, 71, 1.5), (7, 72, 0.5), (7.5, 74, 0.5),
            (8.5, 76, 0.5), (9, 79, 1.0), (10, 76, 0.5), (10.5, 74, 1.5), (12, 73, 0.5), (12.5, 76, 1.5)],
    lead="ooh", lead_gain=0.2)
# 2. シルク・スロウジャム（70BPM, Ebmaj9 - Cm9 - Fm9 - Bb13）
P["2_silk_slowjam"] = dict(
    bpm=70, swing=0.58, chord_beats=2, back="snap", hats="8", pad=True, bright=0.7,
    kick=[0, 2.5, 2.75],
    chords=[(39, [55, 58, 62, 65]), (36, [55, 58, 62, 63]), (41, [56, 60, 63, 67]), (46, [56, 62, 67, 68])],
    comp=[(0, 1.8), (1.5, 0.45)],
    bass=[(0, 1.4, 0), (1.5, 0.45, 12)], slide=True,
    melody=[(0.5, 79, 0.5), (1, 77, 0.5), (1.5, 75, 1.5), (3.5, 74, 0.5), (4, 75, 0.5), (4.5, 77, 1.5),
            (6.5, 79, 0.5), (7, 82, 1.0), (8, 80, 0.5), (8.5, 79, 0.5), (9, 77, 1.5), (10.5, 75, 1.0)],
    lead="flute", lead_gain=0.2)
# 3. サンセット・グルーヴ（92BPM, Fmaj7 - Em7 - Dm7 - Cmaj7）明るめ
P["3_sunset_groove"] = dict(
    bpm=92, swing=0.6, chord_beats=2, back="snare", hats="16", shaker=True, bright=1.2,
    kick=[0, 1.5, 2, 3.5],
    chords=[(41, [57, 60, 64, 65]), (40, [55, 59, 62, 64]), (38, [53, 57, 60, 62]), (36, [52, 55, 59, 62])],
    comp=[(0.5, 0.4), (1.25, 0.6)],
    pluck=[(0, 0), (0.75, 2), (1.5, 1)],
    bass=[(0, 0.7, 0), (0.75, 0.25, 12), (1, 0.5, 0), (1.5, 0.45, 7)], slide=False,
    melody=[(0, 76, 0.5), (0.5, 79, 0.5), (1, 81, 1.0), (2, 79, 0.5), (2.5, 76, 1.0), (4, 74, 0.5), (4.5, 76, 0.5),
            (5, 79, 1.0), (6, 76, 0.5), (6.5, 72, 1.5), (8, 76, 0.5), (8.5, 79, 0.5), (9, 84, 1.0), (10, 83, 0.5),
            (10.5, 81, 0.5), (11, 79, 1.0), (12, 76, 0.5), (12.5, 79, 1.5)],
    lead="bell", lead_gain=0.18)
# 4. ローファイR&B（76BPM, Am9 - Dm9 - Gmaj9 - E7）レコードノイズ入り
P["4_lofi_rnb"] = dict(
    bpm=76, swing=0.64, chord_beats=4, back="rim", ghost=True, hats="8", vinyl=True, wobble=True,
    master_lp=6500, bright=0.8, keys_gain=1.15,
    kick=[0, 0.75, 2.5],
    chords=[(45, [55, 59, 60, 64]), (38, [53, 57, 60, 64]), (43, [54, 57, 59, 62]), (40, [56, 59, 62, 65])],
    comp=[(0, 2.0), (2.5, 1.2)],
    bass=[(0, 2.0, 0), (2.5, 0.8, 0), (3.5, 0.4, 7)], slide=True,
    melody=[(1, 76, 0.5), (1.5, 74, 0.5), (2, 72, 1.5), (4.5, 77, 0.5), (5, 76, 0.5), (5.5, 74, 1.5),
            (8.5, 74, 0.5), (9, 71, 1.0), (10, 72, 1.5), (11.5, 74, 1.0)],
    lead="flute", lead_gain=0.17)
# 5. シティ・ナイト（88BPM, Abmaj9 - Gm7 - Cm9 - Bbm7/Eb9）都会的
P["5_city_night"] = dict(
    bpm=88, swing=0.57, chord_beats=2, back="snap", hats="16", open_hat=True, pad=True,
    kick=[0, 1.75, 2.5],
    chords=[(44, [55, 60, 63, 70]), (43, [53, 58, 62, 65]), (36, [55, 58, 62, 63]), (39, [55, 58, 61, 65])],
    comp=[(0, 0.9), (1, 0.4), (1.5, 0.45)],
    bass=[(0, 0.9, 0), (1, 0.4, 12), (1.5, 0.45, 7)], slide=True,
    melody=[(0, 79, 0.5), (0.5, 80, 0.5), (1, 82, 1.0), (2, 79, 0.5), (2.5, 77, 1.0), (4, 75, 0.5), (4.5, 77, 0.5),
            (5, 79, 1.5), (7, 80, 0.5), (7.5, 82, 0.5), (8, 84, 1.0), (9, 82, 0.5), (9.5, 79, 1.0),
            (11, 77, 0.5), (11.5, 79, 2.0)],
    lead="bell", lead_gain=0.16)

if __name__ == "__main__":
    od = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(od, exist_ok=True)
    for k, p in P.items():
        build(p, f"{od}/{k}.wav"); print("OK", k)
