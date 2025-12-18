# app.py — AI Sales Call Assistant (Updated UI + New OpenAI API + Mic animation + Waveform + Dashboard)
import os
import json
import time
import wave
import glob
import html
import math
import streamlit as st
import sounddevice as sd
import wavio
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# New OpenAI SDK import (optional)
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None
# ---------------- LANDING / START SCREEN -------------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:

    # Fullscreen background
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://static.vecteezy.com/system/resources/previews/026/976/089/non_2x/light-blue-tech-abstract-background-ai-generated-free-photo.jpg');
            background-size: cover;
            background-position: center;
        }
        .landing-box {
            text-align: center;
            padding-top: 120px;
            color: white;
        }
        .title {
            font-size: 40px;
            font-weight: 700;
            text-shadow: 0 4px 18px rgba(0,0,128,0.6);

        }
        .subtitle {
            font-size: 25px;
            opacity: 0.85;
            margin-top: 15px;
        }
        .logo {
            width: 160px;
            margin-bottom: 25px;
            filter: drop-shadow(0 6px 15px rgba(0,0,0,0.4));
        }
        .start-btn button {
            background: linear-gradient(90deg, #8b5cf6, #6366f1);
            padding: 14px 28px;
            font-size: 20px;
            border-radius: 12px;
            color: dark blue;
            border: none;
            box-shadow: 0 8px 24px rgba(0,0,0,0.45);
        }
        .start-btn button:hover {
            transform: scale(1.05);
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # UI: logo + title + button
    st.markdown(
        """
        <div class="landing-box">
            <img class="logo" src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png">
            <div class="title">🎧 AI Enabled Real Time AI Sales Call Assistant</div>
            <div class="subtitle">Professional Real-Time Call Analysis with AI Coaching</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Center the Start button
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        start = st.button("🚀 Start", key="start_app", help="Begin using the assistant", use_container_width=True)

    if start:
        st.session_state.started = True
        st.rerun()

    st.stop()   # <-- Prevents rest of app from loading until Start is pressed

# --------------------- CONFIG & DIRECTORIES ---------------------
st.set_page_config(page_title="AI Sales Call Assistant", page_icon="🎧", layout="wide")
st.markdown(
    f'''
    <h1 style="font-size:48px; font-weight:700; color:#000;
               margin-top:40px; 
               margin-bottom:30px;">
        🎧 AI Enabled Real Time AI Sales Call Assistant
    </h1>
    ''',
    unsafe_allow_html=True
)


AUDIO_DIR = "milestone1/dataset/audio"
TRANS_DIR = "milestone1/dataset/transcripts"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)

# --------------------- SIDEBAR: Settings + Theme ---------------------
st.sidebar.header("⚙️ Controls & Settings")

theme = st.sidebar.selectbox("🎨 Select Theme", ["Purple", "Blue", "Emerald"])
theme_colors = {
    "Purple": ["#8b5cf6", "#6366f1"],
    "Blue": ["#3b82f6", "#1d4ed8"],
    "Emerald": ["#10b981", "#059669"],

}
primary, secondary = theme_colors.get(theme, theme_colors["Purple"])

VOSK_PATH_INPUT = st.sidebar.text_input("Vosk model path", value="vosk_models/vosk-model-en-us-0.22")
MIC_SECONDS = st.sidebar.slider("🎙️ Recording Duration (seconds)", min_value=2, max_value=120, value=6, step=1)
ENABLE_LLM = st.sidebar.checkbox("🤖 Enable LLM Coaching (OpenAI)", value=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key (optional)", value="", type="password")
if OPENAI_API_KEY and ENABLE_LLM:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.sidebar.markdown("---")

# --------------------- BEAUTIFUL THEME + MIC + WAVEFORM CSS ---------------------
st.markdown(
    f"""
<style>
:root {{
  --primary: {primary};
  --secondary: {secondary};
  --bg: #0b0f18;
  --glass: rgba(255,255,255,0.045);
  --border: rgba(255,255,255,0.06);
  --muted: #9aa4b2;
  --text: #e6eef8;
}}
<style>
html, body, [class*="css"] {{
  background:
    linear-gradient(135deg, rgba(7,16,41,0.75), rgba(15,23,36,0.75)),
    url("https://wallpapers.com/images/hd/light-color-background-swzb97w5pxngu8yv.jpg") !important;
  background-size: cover !important;
  background-position: center !important;
  background-attachment: fixed !important;
  color: var(--text) !important;
}}

.block-container {{ padding-top: 1.2rem; }}
h1, h2, h3 {{ color: #000 !important; letter-spacing: 0.3px; }}
.sidebar .sidebar-content {{
  background: linear-gradient(180deg, rgba(20,22,30,0.9), rgba(10,11,15,0.9)) !important;
}}
.card {{
  background: var(--glass);
  border-radius: 16px;
  padding: 16px;
  border: 1px solid var(--border);
  box-shadow: 0 6px 24px rgba(2,6,23,0.6);
  margin-bottom: 1rem;
}}
.header-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
.big-title {{ font-size:28px; font-weight:700; margin:0 0 8px 0; }}
.controls {{ display:flex; gap:10px; align-items:center; }}
.stButton>button {{
  background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
  color: white !important;
  border-radius: 12px !important;
  padding: 8px 14px !important;
  font-weight:600;
  border:none !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.45);
}}
.stButton>button:hover {{ transform: scale(1.03); }}
.chat-container {{ padding: 8px; }}
.chat-bubble {{  color: #000;padding: 14px 16px; border-radius: 14px; margin-bottom: 12px; max-width: 84%; line-height: 1.45; backdrop-filter: blur(8px); font-size:15px; }}
.agent {{ background: rgba(255,255,255,0.03); border-left: 4px solid rgba(255,255,255,0.04); }}
.customer {{ background: linear-gradient(90deg, rgba(0,0,0,0.12), rgba(0,0,0,0.04)); border-left: 4px solid var(--primary); margin-left: auto; color: #000; }}
.meta {{ font-size: 13.5px; color: #000 ; margin-top: 8px; padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.02); border: 1px solid var(--border); }}
.llm-tip {{ margin-top:8px; padding: 10px; border-radius: 10px; background: linear-gradient(90deg, rgba(16,185,129,0.12), rgba(16,185,129,0.06)); color: #065f46; border-left: 4px solid #10b981; }}
.mic-wrap {{ display:flex; align-items:center; gap:12px; }}
.mic-pulse {{ width:68px; height:68px; border-radius:50%; background: linear-gradient(135deg, var(--primary), var(--secondary)); box-shadow: 0 8px 30px rgba(0,0,0,0.45), 0 0 28px rgba(139,92,246,0.08); animation: micPulse 1.6s infinite; display:inline-block; }}
@keyframes micPulse {{ 0% {{ transform: scale(1); opacity:0.95; }} 50% {{ transform: scale(1.12); opacity:1; }} 100% {{ transform: scale(1); opacity:0.95; }} }}
.wave {{ display:flex; align-items:flex-end; gap:5px; width:260px; height:40px; }}
.wave .bar {{ width:6px; background: linear-gradient(180deg, var(--primary), var(--secondary)); border-radius:6px; animation: waveAnim 0.8s infinite ease-in-out; }}
.wave .bar:nth-child(1) {{ animation-delay: 0s; }}
.wave .bar:nth-child(2) {{ animation-delay: 0.15s; }}
.wave .bar:nth-child(3) {{ animation-delay: 0.3s; }}
.wave .bar:nth-child(4) {{ animation-delay: 0.45s; }}
.wave .bar:nth-child(5) {{ animation-delay: 0.6s; }}
@keyframes waveAnim {{ 0% {{ height:8px; opacity:0.7; }} 50% {{ height:36px; opacity:1; }} 100% {{ height:8px; opacity:0.7; }} }}
.small-muted {{ color:var(--muted); font-size:13px; }}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------- UTIL: VOSK model load check ---------------------
if not os.path.exists(VOSK_PATH_INPUT):
    st.sidebar.error(f"Vosk model not found at: {VOSK_PATH_INPUT}")
st.sidebar.caption("Set a valid Vosk model path to enable offline ASR.")

@st.cache_resource(show_spinner=True)
def load_vosk_model(path: str):
    return Model(path)

try:
    vosk_model = load_vosk_model(VOSK_PATH_INPUT)
except Exception as e:
    st.error(f"Failed to load Vosk model: {e}")
    st.stop()

# --------------------- UTILS: audio conversion, transcribe ---------------------
def ensure_16k_mono_wav(path_in: str) -> str:
    audio = AudioSegment.from_file(path_in)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(path_in, format="wav")
    return path_in

def transcribe_wav(path_wav: str) -> str:
    wf = wave.open(path_wav, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)
    text_chunks = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            if r.get("text"):
                text_chunks.append(r["text"])
    final = json.loads(rec.FinalResult()).get("text", "")
    if final:
        text_chunks.append(final)
    return " ".join(text_chunks).strip()

# --------------------- AUTO-INCREMENT micN naming ---------------------
def next_mic_index():
    files = glob.glob(os.path.join(AUDIO_DIR, "mic*.wav"))
    nums = []
    for f in files:
        name = os.path.basename(f)
        try:
            n = int(name.replace("mic", "").replace(".wav",""))
            nums.append(n)
        except:
            pass
    return max(nums)+1 if nums else 1

if "mic_count" not in st.session_state:
    st.session_state.mic_count = next_mic_index()

# --------------------- HEURISTICS ---------------------
POS_WORDS = {"good","great","awesome","best","love","like","perfect","sure","yes","interested","affordable","cheap","discount"}
NEG_WORDS = {"bad","slow","not","problem","issue","worst","hate","expensive","too costly","no","can't","cannot","dislike","bother"}

INTENT_KEYWORDS = {
    "Buy": ["buy","purchase","order","need","want","get this","i will take","i'll take","i will go with","i'll go with"],
    "Ask for price": ["price","cost","how much","rate","₹","rs","rupees","amount"],
    "Ask for warranty": ["warranty","guarantee","replacement","coverage"],
    "Ask for discount": ["discount","offer","deal","cashback","sale"],
    "Complain": ["slow","not working","problem","issue","broken","damage","complain"],
    "Compare products": ["compare","difference","which is better","vs","versus","better than"],
    "General inquiry": ["who are you","your name","company","details about you","what do you do","demo"]
}

PRODUCTS = ["phone","mobile","laptop","tv","television","headphones","earbuds","speaker","watch","smartwatch","tablet","food","camera"]
BRANDS = ["samsung","apple","oneplus","sony","mi","redmi","noise","boat","dell","hp","asus","lenovo"]
PERSON_NAMES = ["rahul","anjali","neha","aman","arjun","ravi","priya","kiran","sneha","pooja","rohit","mike","john","david","emma","alex","anita"]

def heuristic_sentiment(text: str):
    t = text.lower()
    pos = sum(t.count(w) for w in POS_WORDS)
    neg = sum(t.count(w) for w in NEG_WORDS)
    if pos > neg:
        score = min(0.99, 0.5 + 0.1*pos)
        return f"POSITIVE ({score:.2f})"
    if neg > pos:
        score = min(0.99, 0.5 + 0.1*neg)
        return f"NEGATIVE ({score:.2f})"
    return f"NEUTRAL (0.85)"

def heuristic_intent(text: str):
    t = text.lower()
    for intent, keys in INTENT_KEYWORDS.items():
        for k in keys:
            if k in t:
                return intent
    if any(w in t for w in ["price","cost","how much"]):
        return "Ask for price"
    if any(w in t for w in ["buy","need","want"]):
        return "Buy"
    return "General inquiry"

def heuristic_entities(text: str):
    twords = text.lower().split()
    found = []
    for p in PRODUCTS:
        if p in twords:
            found.append({"entity":"PRODUCT","word":p})
    for b in BRANDS:
        if b in twords:
            found.append({"Brand":b})
    for name in PERSON_NAMES:
        if name in twords:
            found.append({"entity":"PERSON","word":name.capitalize()})
    return found

# --------------------- OPENAI (NEW SDK) ---------------------
USE_OPENAI = False
openai_client = None
if ENABLE_LLM and (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")) and OpenAIClient is not None:
    try:
        key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        openai_client = OpenAIClient(api_key=key)
        USE_OPENAI = True
    except Exception as e:
        st.sidebar.warning(f"OpenAI client init failed: {e}. Falling back to heuristics.")
        USE_OPENAI = False

def call_openai_nlp(text: str):
    if not USE_OPENAI or openai_client is None:
        raise RuntimeError("OpenAI client not available")
    prompt = (
        "You are a compact NLP extractor. Given the customer's sentence, "
        "return ONLY a JSON object with keys: sentiment, sentiment_score, intent, entities.\n"
        "entities is a list of objects with keys entity (PRODUCT/BRAND/PERSON/etc) and word.\n\n"
        f"Text: '''{text}'''\n\n"
        "Example output:\n"
        '{{\"sentiment\":\"POSITIVE\",\"sentiment_score\":0.95,\"intent\":\"Buy\",\"entities\":[{{\"entity\":\"PRODUCT\",\"word\":\"phone\"}},{{\"entity\":\"BRAND\",\"word\":\"samsung\"}}]}}'
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai_client.chat.completions.create.__doc__ else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        choice = resp.choices[0]
        content = getattr(choice.message, "content", None) or (choice["message"]["content"] if isinstance(choice, dict) else "")
        txt = content.strip()
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1:
            raise RuntimeError(f"No JSON found in model response: {txt[:200]}")
        data = json.loads(txt[start:end+1])
        sentiment = f"{data.get('sentiment','NEUTRAL')} ({data.get('sentiment_score',0.0):.2f})"
        intent = data.get("intent","General inquiry")
        entities = data.get("entities", [])
        return sentiment, intent, entities
    except Exception as e:
        raise RuntimeError(f"OpenAI NLP failed: {e}")

def call_openai_tip(customer_text: str, sentiment: str, intent: str, entities):
    if not USE_OPENAI or openai_client is None:
        raise RuntimeError("OpenAI client not available")
    prompt = (
        "You are an expert sales coach. Given the customer's sentence, the sentiment, intent, and entities, "
        "return 2-3 short actionable questions the agent can ask next to advance the sale. "
        "Return as a JSON list of strings only.\n\n"
        f"Customer: {customer_text}\nSentiment: {sentiment}\nIntent: {intent}\nEntities: {entities}\n\nReply:"
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content": prompt}],
            temperature=0.4,
            max_tokens=150
        )
        choice = resp.choices[0]
        txt = getattr(choice.message, "content", None) or (choice["message"]["content"] if isinstance(choice, dict) else "")
        # try parsing as JSON list
        try:
            questions = json.loads(txt)
            if isinstance(questions, list):
                return questions
        except:
            pass
        # fallback: split lines
        return [line.strip() for line in txt.splitlines() if line.strip()]
    except Exception as e:
        return [f"(LLM tip unavailable: {e})"]

# --------------------- CORE ANALYSIS & TIP ---------------------
def analyze_customer_text(text: str):
    if USE_OPENAI:
        try:
            return call_openai_nlp(text)
        except Exception as e:
            st.sidebar.warning(f"OpenAI NLP failed, using heuristics: {str(e)[:200]}")
    return heuristic_sentiment(text), heuristic_intent(text), heuristic_entities(text)

def generate_agent_tip(text: str, sentiment: str, intent: str, entities):
    if USE_OPENAI:
        try:
            return call_openai_tip(text, sentiment, intent, entities)
        except Exception as e:
            st.sidebar.warning(f"LLM tip generation failed: {e}")
    text_l = text.lower()
    if "price" in text_l or "how much" in text_l:
        return [
            "Ask what their budget is.",
            "Confirm the exact product they are interested in.",
            "Offer details about any ongoing discounts or EMI options."
        ]
    if intent.lower().startswith("buy"):
        return [
            "Confirm the model/variant they want.",
            "Ask if they prefer any specific color or configuration.",
            "Offer payment/checkout options."
        ]
    if intent.lower().startswith("compare"):
        return [
            "Ask what features matter most to them.",
            "Highlight 2-3 differentiating features of products.",
            "Recommend one product based on their needs."
        ]
    if "warranty" in text_l:
        return [
            "Explain the warranty coverage.",
            "Ask if they want an extended warranty.",
            "Provide any relevant terms and conditions."
        ]
    return [
        "Ask a clarifying question to move the sale forward (budget, timeline, or preferred brand).",
        "Confirm if they need further assistance.",
        "Offer product recommendations based on their interest."
    ]

# --------------------- UI MODE ---------------------
mode = st.radio("Choose Input Method", ["📂 Upload Call Recording", "🎤 Live Recording Mode"], index=1, horizontal=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "mic_count" not in st.session_state:
    st.session_state.mic_count = next_mic_index()

# --------------------- Utility: render chat + dashboard ---------------------
def render_chat_output(items):
    html_blocks = '<div class="chat-container card">'
    for it in items:
        role = it.get("role", "customer")
        text = html.escape(it.get("text", ""))
        meta = it.get("meta", "")
        tip = it.get("tip", None)
        if role == "agent":
            html_blocks += f'<div class="chat-bubble agent"><strong>Sales Agent:</strong> {text}</div>'
        else:
            html_blocks += f'<div class="chat-bubble customer"><strong>Customer:</strong> {text}</div>'
        if meta:
            meta_html = html.escape(meta).replace("\n", "<br>")
            html_blocks += f'<div class="meta">{meta_html}</div>'
        if tip:
            if isinstance(tip, list):
                tip_str = "<br>".join([f"{i+1}. {html.escape(q)}" for i, q in enumerate(tip)])
            else:
                tip_str = html.escape(tip)
            html_blocks += f'<div class="llm-tip">{tip_str}</div>'
    html_blocks += "</div>"
    st.markdown(html_blocks, unsafe_allow_html=True)

def dashboard_cards_from_history(items):
    total = len(items)
    pos = neg = neu = 0
    intent_counts = {}
    last_entities = []
    for it in items:
        if it.get("meta"):
            try:
                lines = it["meta"].splitlines()
                sent_line = next((l for l in lines if l.lower().startswith("sentiment:")), None)
                intent_line = next((l for l in lines if l.lower().startswith("intent:")), None)
                entities_line = next((l for l in lines if l.lower().startswith("entities:")), None)
                if sent_line:
                    s = sent_line.split(":",1)[1].strip().lower()
                    if "positive" in s:
                        pos += 1
                    elif "negative" in s:
                        neg += 1
                    else:
                        neu += 1
                if intent_line:
                    intent = intent_line.split(":",1)[1].strip()
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
                if entities_line:
                    last_entities.append(entities_line.split(":",1)[1].strip())
            except Exception:
                pass
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "intents": intent_counts,
        "entities_sample": last_entities[-3:] if last_entities else []
    }

# --------------------- UPLOAD MODE ---------------------
if mode == "📂 Upload Call Recording":
    st.header("📂 Upload Call Recording")
    uploaded = st.file_uploader("Upload a WAV audio file (optional transcript must be in transcripts/ with same base name)", type=["wav"], accept_multiple_files=False)
    if uploaded:
        fname = uploaded.name
        base = os.path.splitext(fname)[0]
        audio_path = os.path.join(AUDIO_DIR, fname)
        with open(audio_path, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            ensure_16k_mono_wav(audio_path)
        except Exception as e:
            st.warning(f"Audio conversion warning: {e}")
        st.audio(audio_path)

        transcript_path = os.path.join(TRANS_DIR, f"{base}.txt")
        if not os.path.exists(transcript_path):
            st.warning(f"Transcript not found at {transcript_path}. You can create {base}.txt in {TRANS_DIR}.")
        else:
            with open(transcript_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            chat_items = []
            for line in lines:
                if ":" not in line:
                    continue
                speaker, text = line.split(":",1)
                speaker = speaker.strip().lower()
                text = text.strip()
                if speaker in ("agent","sales agent","sales"):
                    chat_items.append({"role":"agent","text":text})
                else:
                    sentiment, intent, entities = analyze_customer_text(text)
                    meta_lines = f"Sentiment: {sentiment}\nIntent: {intent}\nEntities: {entities}"
                    tip = generate_agent_tip(text, sentiment, intent, entities)
                    chat_items.append({"role":"customer","text":text,"meta":meta_lines,"tip":tip})
            st.subheader("Conversation (with analysis)")
            render_chat_output(chat_items)

# --------------------- LIVE RECORDING MODE (MAIN) ---------------------
if mode == "🎤 Live Recording Mode":
    st.header("🎤 Live Microphone (Live)")

    # ❌ REMOVED: TOP CIRCLED AREA (MIC CARD + STAT MINI-CARDS)

    st.markdown("")  # spacer

    # Speaker selector
    speaker_choice = st.radio("Who is speaking now?", ["Customer", "Sales Agent"], horizontal=True)

    # Layout after removing the top section
    left_col, right_col = st.columns([1, 1.7])

    with left_col:
        # Recorder UI only
        st.markdown(
            """
            <div class="card">
                <div style="font-weight:600; margin-bottom:8px;">Recorder</div>
                <div style="display:flex;gap:10px;align-items:center;">
                    <div>
                        <div class="wave" aria-hidden="true">
                            <div class="bar"></div>
                            <div class="bar"></div>
                            <div class="bar"></div>
                            <div class="bar"></div>
                            <div class="bar"></div>
                        </div>
                        <div style="margin-top:8px;" class="small-muted">Live waveform (visual)</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🎙️ Record & Analyze"):
            st.info(f"Recording for {MIC_SECONDS} seconds...")
            try:
                audio = sd.rec(int(MIC_SECONDS * 16000), samplerate=16000, channels=1, dtype="int16")
                sd.wait()
                samples = audio.flatten()
            except Exception as e:
                st.error(f"Microphone recording failed: {e}")
                samples = None

            if samples is not None:
                idx = st.session_state.mic_count
                filename = f"mic{idx}.wav"
                path = os.path.join(AUDIO_DIR, filename)

                try:
                    wavio.write(path, samples.reshape(-1,1), 16000, sampwidth=2)
                except Exception as e:
                    st.error(f"Failed to write WAV: {e}")

                try:
                    ensure_16k_mono_wav(path)
                except Exception:
                    pass

                try:
                    text = transcribe_wav(path)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    text = ""

                txtfile = os.path.join(TRANS_DIR, f"mic{idx}.txt")
                try:
                    with open(txtfile, "a", encoding="utf-8") as tf:
                        tf.write(f"{speaker_choice.lower()}: {text}\n")
                except:
                    pass

                st.session_state.mic_count += 1

                # analysis section
                if speaker_choice == "Customer":
                    sentiment, intent, entities = analyze_customer_text(text)
                    meta_lines = f"Sentiment: {sentiment}\nIntent: {intent}\nEntities: {entities}"
                    tip = generate_agent_tip(text, sentiment, intent, entities)
                    st.session_state.history.append(
                        {"role": "customer", "text": text, "meta": meta_lines, "tip": tip}
                    )
                else:
                    st.session_state.history.append({"role": "agent", "text": text})

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.success("Cleared history.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div style="display:flex;flex-direction:column;gap:10px;">', unsafe_allow_html=True)
        st.subheader("Live Conversation")

        if st.session_state.history:
            render_chat_output(st.session_state.history)
        else:
            st.info("No live conversation recorded yet. Use the Record button to begin.")

        st.markdown("</div>", unsafe_allow_html=True)


