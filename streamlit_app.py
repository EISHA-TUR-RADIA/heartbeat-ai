# HeartBeat AI — Streamlit version (free hosting on Streamlit Community Cloud)
import os, glob, json, math
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, wfdb

st.set_page_config(page_title="HeartBeat AI", page_icon="❤️", layout="wide")

FS = 360
PRE, POST = int(0.2 * FS), int(0.4 * FS)
NORMAL = {"N", "L", "R", "e", "j"}
ABNORMAL = {"A", "a", "J", "S", "V", "E", "F", "!"}
DEMO_RECORDS = ["100", "103", "105", "111", "113", "117", "200", "202", "210", "212"]  # unseen-by-model patients
DATA_DIR = "mitdb"
WINDOW = 10
BLUE, PINK, INK, BG = "#5b7fe8", "#c2417d", "#2b3a67", "#f1f4fd"

def find_hea(name):
    hits = glob.glob(f"{DATA_DIR}/**/{name}.hea", recursive=True)
    return hits[0][:-4] if hits else None

def load_record(name):
    path = find_hea(name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    if "MLII" not in rec.sig_name:
        return None
    return rec.p_signal[:, rec.sig_name.index("MLII")], ann.sample, ann.symbol

def extract_beats(ecg, samples, symbols):
    X, y, pos = [], [], []
    for s, sym in zip(samples, symbols):
        if sym in NORMAL:
            label = 0
        elif sym in ABNORMAL:
            label = 1
        else:
            continue
        a, b = s - PRE, s + POST
        if a < 0 or b > len(ecg):
            continue
        beat = ecg[a:b]
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
        X.append(beat); y.append(label); pos.append(s)
    return np.array(X), np.array(y), np.array(pos)

@st.cache_resource(show_spinner="Loading the AI model and ECG data (first start takes 1-2 minutes)...")
def load_everything():
    model = joblib.load("heartbeat_model.pkl")
    metrics = json.load(open("metrics.json"))
    missing = [r for r in DEMO_RECORDS if find_hea(r) is None]
    if missing:
        os.makedirs(DATA_DIR, exist_ok=True)
        wfdb.dl_database("mitdb", DATA_DIR, records=missing)
    data = {}
    for r in DEMO_RECORDS:
        d = load_record(r)
        if d is None:
            continue
        ecg, samples, symbols = d
        X, y, pos = extract_beats(ecg, samples, symbols)
        data[r] = {"ecg": ecg, "pos": pos, "truth": y, "prob": model.predict_proba(X)[:, 1]}
    return data, metrics

HEART_SVG = """<svg viewBox="0 0 240 270" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="hg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#bcd8fa"/><stop offset="0.45" stop-color="#6f8fe4"/><stop offset="1" stop-color="#b03a78"/></linearGradient>
<linearGradient id="vg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8f3f9a"/><stop offset="1" stop-color="#c2417d"/></linearGradient></defs>
<rect x="96" y="6" width="30" height="80" rx="15" fill="url(#vg)" transform="rotate(-8 110 46)"/>
<rect x="136" y="20" width="22" height="70" rx="11" fill="#7f95e0" transform="rotate(12 147 55)" opacity=".9"/>
<path d="M120 248 C 38 182 8 128 24 84 C 38 42 92 34 120 78 C 150 34 204 42 217 84 C 232 128 202 182 120 248 Z" fill="url(#hg)"/>
<path d="M120 78 C 112 130 118 190 120 248" stroke="#fff" stroke-opacity=".35" stroke-width="3" fill="none"/>
<path d="M60 92 C 80 120 92 150 100 196 M180 92 C 160 120 150 150 142 196 M70 70 C 100 90 140 90 170 70" stroke="#fff" stroke-opacity=".28" stroke-width="2.5" fill="none"/>
<ellipse cx="78" cy="88" rx="26" ry="14" fill="#fff" opacity=".28" transform="rotate(-30 78 88)"/>
<g fill="#fff"><circle cx="112" cy="24" r="5"/><circle cx="58" cy="118" r="5"/><circle cx="192" cy="110" r="5"/><circle cx="130" cy="150" r="5"/></g>
<g stroke="#fff" stroke-opacity=".7" stroke-width="1.2"><path d="M112 24 L112 46"/><path d="M58 118 L80 118"/><path d="M192 110 L170 110"/><path d="M130 150 L130 130"/></g></svg>"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html,body,.stApp,[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif !important;
 background:radial-gradient(900px 600px at 0% 0%,#a7c4ee 0%,transparent 60%),radial-gradient(900px 700px at 100% 100%,#b6a9e6 0%,transparent 60%),linear-gradient(135deg,#7ea4d8,#a7b0e6 55%,#c3bde9) fixed !important;}
header[data-testid="stHeader"],footer,#MainMenu{display:none !important}
.block-container{max-width:1150px !important;background:#fff;border-radius:34px;padding:26px 28px 18px !important;margin-top:22px;box-shadow:0 30px 80px rgba(60,80,150,.35)}
.stApp,.stApp p,.stApp label,.stApp span,.stApp div{color:#2b3a67}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#f1f4fd;border:none !important;border-radius:26px;padding:6px 10px}
.hero{position:relative;border-radius:28px;padding:26px;min-height:330px;overflow:hidden;display:flex;gap:20px;align-items:center;background:linear-gradient(135deg,#86abd8 0%,#a6bde7 55%,#9db2e6 100%);margin-bottom:8px}
.hero,.hero *{color:#fff !important}
.hero h1{font-weight:300;font-size:34px;line-height:1.05;margin:0 0 16px;padding:0;letter-spacing:-.5px}
.glass{background:rgba(255,255,255,.22);border:1px solid rgba(255,255,255,.4);backdrop-filter:blur(10px);border-radius:22px;padding:18px;max-width:340px}
.glass h3{margin:0 0 10px;font-size:15px;font-weight:500;padding:0}
.glass p{margin:6px 0 0;font-size:12.5px;line-height:1.6}
.chip{display:inline-block;background:rgba(255,255,255,.6);font-size:11px;border-radius:10px;padding:3px 10px;margin:0 6px 10px 0}
.hero .chip,.hero .chip *{color:#4a5a8c !important}
.heartbox{flex:1;display:flex;justify-content:center}
.heartbox svg{height:270px;filter:drop-shadow(0 20px 28px rgba(70,60,140,.4));animation:beat 1.2s infinite;transform-origin:center}
@keyframes beat{0%,100%{transform:scale(1)}15%{transform:scale(1.05)}30%{transform:scale(1)}45%{transform:scale(1.03)}}
.callout{position:absolute;right:24px;bottom:22px;background:rgba(190,176,222,.95);border-radius:16px;padding:12px 18px;font-size:16px;line-height:1.25;max-width:250px}
.callout small{display:block;font-size:11px;opacity:.9;margin-top:5px}
.conf{position:absolute;right:30px;top:24px;background:#fff;border-radius:12px;padding:6px 12px;font-size:12px;box-shadow:0 6px 16px rgba(60,80,150,.2)}
.hero .conf,.hero .conf *{color:#2b3a67 !important}
.hero .conf b{color:#5b7fe8 !important}
.card-title{font-weight:300;font-size:26px;line-height:1.1;margin:4px 0 6px}
.ringcard{text-align:center}.ringcard svg{max-width:200px;width:100%}
.stat{display:flex;justify-content:space-between;background:#fff;border-radius:14px;padding:10px 14px;margin-top:8px;font-size:13px}
.stat b{color:#5b7fe8;font-weight:600}
.disc{font-size:12px;color:#6b7aa8;text-align:center;padding:10px 10px 0}
.stButton>button{background:linear-gradient(135deg,#93abf2,#5b7fe8) !important;border:none !important;border-radius:999px !important;color:#fff !important;height:46px;font-weight:500;box-shadow:0 10px 24px rgba(91,127,232,.35)}
.stButton>button *{color:#fff !important}
div[data-baseweb="select"]>div{background:#fff !important;border-radius:14px !important}
@media(max-width:800px){.hero{flex-direction:column}.callout{position:static;margin-top:10px}.conf{display:none}}
"""

def flat(h):
    return " ".join(l.strip() for l in h.splitlines() if l.strip())

def make_hero(record, t0, t1, n, k, maxp, n_abn, agree, metrics):
    if n == 0:
        callout, sub = "No beats found in this segment.", ""
    elif k == 0:
        callout, sub = "Rhythm looks regular in this segment!", "No irregular beats flagged"
    else:
        callout, sub = f"{k} of {n} beats look irregular!", "Clinical review recommended"
    return flat(f"""
<div class="hero"><div>
<h1>Heart health<br>overview</h1>
<div class="glass"><span class="chip">Record {record}</span><span class="chip">{t0}s – {t1}s</span>
<h3>AI heartbeat screening</h3>
<p>A Random Forest model classified <b>{n}</b> heartbeats in this segment. Cardiologist labels mark <b>{n_abn}</b> as abnormal, and the model agrees on <b>{agree}/{n}</b> beats.</p>
<p>Tested on patients it never saw: accuracy {metrics['accuracy']:.1f}% · abnormal recall {metrics['recall']:.1f}%.</p></div></div>
<div class="heartbox">{HEART_SVG}</div>
<div class="conf">Max abnormal confidence <b>{maxp*100:.0f}%</b></div>
<div class="callout">{callout}<small>{sub}</small></div></div>""")

def make_ring(n, k):
    pct = 100 * (n - k) / n if n else 0
    C = 2 * math.pi * 70
    return flat(f"""
<div class="ringcard"><div class="card-title">Rhythm score</div>
<svg viewBox="0 0 180 180"><defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#c3cdf7"/><stop offset="1" stop-color="#5b7fe8"/></linearGradient></defs>
<circle cx="90" cy="90" r="70" fill="none" stroke="#dde4fa" stroke-width="14"/>
<circle cx="90" cy="90" r="70" fill="none" stroke="url(#rg)" stroke-width="14" stroke-linecap="round" stroke-dasharray="{C*pct/100:.1f} {C:.1f}" transform="rotate(-90 90 90)"/>
<text x="90" y="80" text-anchor="middle" font-size="12" fill="#6b7aa8">Normal beats</text>
<text x="90" y="110" text-anchor="middle" font-size="30" font-weight="500" fill="#4f6fd8">{pct:.0f}%</text></svg>
<div class="stat"><span>Beats analysed</span><b>{n}</b></div>
<div class="stat"><span>Flagged irregular</span><b style="color:#c2417d">{k}</b></div></div>""")

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

try:
    DATA, METRICS = load_everything()
except Exception as e:
    st.error(f"Could not load the model or ECG data: {e}")
    st.stop()

records = list(DATA.keys())
rng = np.random.default_rng()

def abnormal_start(rec):
    d = DATA[rec]
    abn = d["pos"][d["truth"] == 1]
    dur = len(d["ecg"]) / FS
    if len(abn) == 0:
        return 0
    return int(max(0, min(rng.choice(abn) / FS - WINDOW / 2, dur - WINDOW)))

if "start" not in st.session_state:
    st.session_state["start"] = abnormal_start(records[0])

hero_ph = st.empty()
left, right = st.columns([3, 1], gap="medium")

with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">Heartbeat monitor</div>', unsafe_allow_html=True)
        plot_ph = st.empty()
        c1, c2 = st.columns([2, 1])
        record = c1.selectbox("Choose a patient recording (unseen by the model)", records)
        c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if c2.button("🫀 Jump to an abnormal beat", use_container_width=True):
            st.session_state["start"] = abnormal_start(record)
        max_start = int(len(DATA[record]["ecg"]) / FS - WINDOW)
        st.session_state["start"] = int(min(st.session_state["start"], max_start))
        start = st.slider("Start time (seconds)", 0, max_start, key="start")

with right:
    with st.container(border=True):
        ring_ph = st.empty()

d = DATA[record]
a, b = int(start * FS), int((start + WINDOW) * FS)
m = (d["pos"] >= a) & (d["pos"] < b)
pos, prob, truth = d["pos"][m], d["prob"][m], d["truth"][m]
flagged = prob >= 0.5
n, k = len(pos), int(flagged.sum())

fig, ax = plt.subplots(figsize=(9, 3.3))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(np.arange(a, b) / FS, d["ecg"][a:b], color=BLUE, linewidth=1.4)
ax.scatter(pos[~flagged] / FS, d["ecg"][pos[~flagged]], color="#7fb2ea", s=36, zorder=3, label="Normal")
ax.scatter(pos[flagged] / FS, d["ecg"][pos[flagged]], color=PINK, s=80, marker="v", zorder=4, label="Possible arrhythmia")
ax.set_xlabel("Time (seconds)", color=INK); ax.set_ylabel("Amplitude (mV)", color=INK)
ax.tick_params(colors="#6b7aa8"); ax.grid(color="#d5ddf5", linewidth=0.8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#c7d1f0")
ax.legend(loc="upper right", facecolor="white", edgecolor="#dbe3fa", labelcolor=INK)
plt.tight_layout()
plot_ph.pyplot(fig)
plt.close(fig)

agree = int((flagged.astype(int) == truth).sum())
maxp = float(prob.max()) if n else 0.0
hero_ph.markdown(make_hero(record, start, start + WINDOW, n, k, maxp, int(truth.sum()), agree, METRICS), unsafe_allow_html=True)
ring_ph.markdown(make_ring(n, k), unsafe_allow_html=True)

st.markdown('<div class="disc"><b>Disclaimer:</b> Screening-support prototype, not a medical diagnostic device. '
            'ECG shown is real data from the MIT-BIH Arrhythmia Database; beat positions come from the dataset annotations.</div>',
            unsafe_allow_html=True)
