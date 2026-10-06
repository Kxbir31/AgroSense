import os
import re
import time
import tempfile
from bisect import bisect_right
from difflib import get_close_matches

import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gnani.stt import GnaniSTTClient

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GNANI_API_KEY = os.getenv("GNANI_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-4-31B-it")

PDF_FILES = ["India_District_Agri_Master_RAG.pdf"]
DB_DIR = "./chroma_db"
COLLECTION = "agriculture_rag_v2"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ---------------- load pdf ----------------
pages = []
for file in PDF_FILES:
    for page in PyPDFLoader(file).load():
        page.metadata["source_file"] = file
        pages.append(page)

print(f"Total pages loaded: {len(pages)}")


# ---------------- chunking ----------------
# The pdf is one paragraph per district ("Sehore district, Madhya Pradesh. Soil: ...").
# Page breaks cut some of them in half, so join all pages and split on the district heading.
RECORD_START = re.compile(
    r"(?:\A|(?<=\. ))(?=[A-Z][A-Za-z.&'()\- ]{1,45}? district, [A-Za-z&() ]{2,45}?\. Soil:)"
)
RECORD_HEAD = re.compile(r"^(.+?) district, (.+?)\. Soil:")


def build_chunks(pages):
    text, page_starts = "", []
    for p in pages:
        page_starts.append(len(text))
        text += re.sub(r"\s+", " ", p.page_content).strip() + " "

    starts = [m.start() for m in RECORD_START.finditer(text)]
    ends = starts[1:] + [len(text)]

    chunks = []
    for s, e in zip(starts, ends):
        record = text[s:e].strip()
        head = RECORD_HEAD.match(record)
        if not head:
            continue
        district, state = head.group(1), head.group(2)
        meta = {
            "district": district,
            "state": state,
            "page": bisect_right(page_starts, s),   # page number starting from 1
            "source_file": pages[0].metadata.get("source_file", ""),
        }

        # part 1: soil + climate + suited crops, part 2: state top-4 crop table
        cut = record.find(" State top-4 crops")
        if cut == -1:
            parts = [("profile", record)]
        else:
            crop_text = f"{district} district, {state} (state-level crop data). " + record[cut:].strip()
            parts = [("profile", record[:cut].strip()), ("crops", crop_text)]

        for name, content in parts:
            chunks.append(Document(page_content=content, metadata={**meta, "part": name}))

    for i, c in enumerate(chunks):
        c.metadata["id"] = i
    return chunks


chunks = build_chunks(pages)
print(f"Total chunks created: {len(chunks)}")

# (district, state) -> chunk ids
records = {}
for c in chunks:
    key = (c.metadata["district"].lower(), c.metadata["state"].lower())
    records.setdefault(key, []).append(c.metadata["id"])


# ---------------- embeddings + chromadb ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def open_vectorstore():
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
    # already built for this pdf -> reuse, else rebuild (stops duplicates on every run)
    if store._collection.count() == len(chunks):
        return store
    store.delete_collection()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=[str(c.metadata["id"]) for c in chunks],
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
    )


vectorstore = open_vectorstore()

# ---------------- bm25 ----------------
bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])


# ---------------- llm setup ----------------
llm_client = InferenceClient(token=HF_TOKEN)
MODELS = [HF_MODEL, "deepseek-ai/DeepSeek-V4.1-Flash", "meta-llama/Llama-3.1-8B-Instruct", "openai/gpt-oss-20b"]
working_model = None


# ---------------- router ----------------
# weather words
WEATHER_KW = {
    "weather", "rain", "rainy", "raining", "rainfall", "shower", "drizzle",
    "temperature", "temp", "humidity", "humid", "forecast", "wind", "windy",
    "climate", "hot", "cold", "heat", "heatwave", "frost", "fog", "cloud",
    "cloudy", "storm", "thunderstorm", "lightning", "hailstorm", "hail",
    "flood", "drought", "dry", "sunny", "sunlight", "monsoon", "cyclone",
    "today", "tomorrow", "tonight", "now", "current", "currently", "latest",
    "week", "outside", "condition", "conditions", "degree", "celsius",
    "uv", "dew", "mist", "pressure", "visibility",
    # farming actions that depend on live weather
    "spray", "spraying", "sow", "sowing", "harvest", "harvesting",
    "irrigate", "irrigating", "water", "watering", "drying",
    "transplant", "transplanting", "plough", "ploughing", "plow", "plowing",
    "thresh", "threshing",
    # hindi / hinglish
    "barish", "baarish", "mausam", "garmi", "sardi", "thand", "dhoop",
    "hawa", "aandhi", "toofan", "kohra", "olavrishti", "sukha", "badal",
    "aaj", "kal", "abhi", "tapman", "nami", "paani", "pani", "sinchai",
    "kataai", "buwai", "bovai", "chidkav", "chhidkav",
}

# pdf / district knowledge words
RETRIEVAL_KW = {
    # soil and land
    "soil", "soils", "mitti", "land", "zameen", "jameen", "terrain", "texture",
    "ph", "organic", "carbon", "nutrient", "nutrients", "nitrogen", "phosphorus",
    "potash", "npk", "alluvial", "black", "red", "laterite", "loamy", "clay",
    "sandy", "saline", "alkaline", "acidic", "fertility", "topography",
    # crops
    "crop", "crops", "fasal", "fasalein", "kheti", "farming", "farm", "farmer",
    "kisan", "agriculture", "agricultural", "cultivation", "cultivate", "grow",
    "grown", "growing", "plant", "planting", "variety", "varieties", "hybrid",
    "seed", "seeds", "beej", "wheat", "gehu", "gehun", "rice", "dhan", "paddy",
    "soybean", "soyabean", "cotton", "kapas", "maize", "makka", "corn",
    "pulses", "dal", "gram", "chana", "lentil", "masoor", "tur", "arhar",
    "moong", "urad", "mustard", "sarson", "groundnut", "sugarcane", "ganna",
    "potato", "aloo", "onion", "pyaz", "tomato", "garlic", "lahsun", "millet",
    "bajra", "jowar", "sorghum", "ragi", "barley", "jau", "vegetable",
    "vegetables", "sabzi", "fruit", "fruits", "orchard", "banana", "mango",
    "orange", "pomegranate", "chilli", "chili", "turmeric", "coriander",
    "sunflower", "sesame", "til", "linseed", "oilseed", "oilseeds", "cereal",
    "cereals", "horticulture", "fodder", "forage", "tea", "coffee", "jute",
    # seasons
    "season", "seasons", "kharif", "rabi", "zaid", "summer", "winter",
    # inputs and protection
    "fertilizer", "fertiliser", "fertilizers", "khad", "urea", "dap", "manure",
    "compost", "pesticide", "insecticide", "fungicide", "herbicide", "pest",
    "pests", "disease", "diseases", "rog", "keet", "weed", "weeds", "dawa",
    # water and irrigation
    "irrigation", "canal", "well", "borewell", "tubewell", "tank", "pond",
    "river", "dam", "groundwater", "watershed", "drip", "sprinkler",
    # statistics and economics
    "yield", "production", "produce", "area", "hectare", "hectares", "ha",
    "acre", "acres", "tonnes", "tons", "productivity", "output", "average",
    "total", "percent", "percentage", "statistics", "data", "report",
    "mandi", "market", "msp", "price", "prices", "rate", "rates", "subsidy",
    "scheme", "schemes", "insurance", "loan", "kcc",
    # district / general info
    "district", "zila", "jila", "state", "block", "tehsil", "village", "gaon",
    "region", "agroclimatic", "zone", "major", "main", "top", "type", "types",
    "kind", "which", "what", "list", "tell", "about", "information", "details",
    "livestock", "cattle", "dairy", "poultry", "fishery", "fish",
}

# advice style questions need both weather and district knowledge
ADVICE_KW = {
    "should", "can", "could", "advice", "advise", "suggest", "suggestion",
    "recommend", "recommendation", "best", "when", "plan", "planning",
    "ideal", "suitable", "right", "safe", "good", "worth", "ok", "okay",
    "sahi", "sakte", "sakta", "chahiye", "kab", "kya", "kaise", "salah",
}


def route_query(query, district):
    words = set(tokenize(query))
    need_weather = bool(words & WEATHER_KW)
    need_retrieval = bool(words & RETRIEVAL_KW) or district.lower() in query.lower()

    # word forms like irrigating / fertilizers
    if any(w.startswith(("weath", "rain", "temper", "forecast")) for w in words):
        need_weather = True
    if any(w.startswith(("irrigat", "fertili", "cultivat", "agri")) for w in words):
        need_retrieval = True

    # advice question about a farming topic -> use both
    if words & ADVICE_KW and (need_weather or need_retrieval):
        need_weather = need_retrieval = True

    if need_weather and need_retrieval:
        return "BOTH"
    if need_weather:
        return "WEATHER"
    if need_retrieval:
        return "RETRIEVAL"

    # nothing matched -> skip the pdf only for tiny messages like "hi"
    if len(words) <= 2:
        return "DIRECT"
    return "RETRIEVAL"


# ---------------- weather ----------------
_weather_cache = {}
WEATHER_TTL = 600  # seconds


def get_weather_data(district, state):
    if not OPENWEATHER_API_KEY:
        print("Weather error: OPENWEATHER_API_KEY missing")
        return None

    key = (district.lower(), state.lower())
    cached = _weather_cache.get(key)
    if cached and time.time() - cached[0] < WEATHER_TTL:
        return cached[1]

    geo_url = "https://api.openweathermap.org/geo/1.0/direct"
    try:
        geo = requests.get(
            geo_url,
            params={"q": f"{district},{state},IN", "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=5,
        ).json()
        if not geo:
            geo = requests.get(
                geo_url,
                params={"q": f"{district},IN", "limit": 1, "appid": OPENWEATHER_API_KEY},
                timeout=5,
            ).json()
        if not geo or isinstance(geo, dict):
            print("Weather error:", geo)
            return None

        w = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": geo[0]["lat"], "lon": geo[0]["lon"], "units": "metric", "appid": OPENWEATHER_API_KEY},
            timeout=5,
        ).json()
        if "main" not in w:
            print("Weather error:", w)
            return None

        data = {
            "temp": w["main"]["temp"],
            "feels_like": w["main"]["feels_like"],
            "humidity": w["main"]["humidity"],
            "condition": w["weather"][0]["description"],
            "wind": w["wind"]["speed"],
            "rain_1h": w.get("rain", {}).get("1h", 0),
        }
        _weather_cache[key] = (time.time(), data)
        return data
    except Exception as e:
        print("Weather fetch failed:", repr(e))
        return None


def get_weather(district, state):
    d = get_weather_data(district, state)
    if not d:
        return None
    return (
        f"Temperature: {d['temp']}°C (feels like {d['feels_like']}°C), "
        f"Humidity: {d['humidity']}%, Condition: {d['condition']}, "
        f"Wind: {d['wind']} m/s, Rain (last 1h): {d['rain_1h']} mm"
    )


# ---------------- retrieval ----------------
def find_district(district, state):
    """Chunk ids of the user's own district (state name can be 'Telangana' etc.)."""
    d, s = district.lower().strip(), state.lower().strip()
    pool = [k for k in records if s and (s in k[1] or k[1] in s)] or list(records)
    match = get_close_matches(d, [k[0] for k in pool], n=1, cutoff=0.8)
    if not match:
        return []
    key = next(k for k in pool if k[0] == match[0])
    return records[key]


def districts_in_query(query, state, skip, limit=2):
    """Other districts the user names in the question, e.g. 'compare with Indore'."""
    q = query.lower()
    found = [k for k in records if k[0] != skip and re.search(rf"\b{re.escape(k[0])}\b", q)]
    s = state.lower().strip()
    found.sort(key=lambda k: not (s and (s in k[1] or k[1] in s)))  # same state first
    return found[:limit]


def hybrid_search(q, state, k):
    """bm25 + vector, merged with reciprocal rank fusion."""
    scores = bm25.get_scores(tokenize(q))
    bm_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    vec_top = [d.metadata["id"] for d in vectorstore.similarity_search(q, k=5)]

    fused = {}
    for ranked in (bm_top, vec_top):
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)

    # small boost for chunks from the user's state
    s = state.lower().strip()
    for idx in fused:
        if s and s in chunks[idx].metadata["state"].lower():
            fused[idx] += 0.05

    return sorted(fused, key=fused.get, reverse=True)[:k]


def retrieve(query, district, state, k=3):
    # 1. the user's own district (soil + climate + crops), always first
    ids = list(find_district(district, state))
    own_name = chunks[ids[0]].metadata["district"].lower() if ids else ""

    # 2. any other district mentioned in the question
    for key in districts_in_query(query, state, skip=own_name):
        for i in records[key]:
            if i not in ids:
                ids.append(i)

    # 3. district not in the pdf -> fall back to hybrid search
    if not ids:
        ids = hybrid_search(f"{query} {district} {state}", state, k)
        found = [f"[Page {chunks[i].metadata['page']}] {chunks[i].page_content}" for i in ids]
        note = (f"[Note] '{district}' is not in the dataset. The records below belong to "
                f"other districts and must not be treated as {district} data.")
        return [note] + found

    return [f"[Page {chunks[i].metadata['page']}] {chunks[i].page_content}" for i in ids]


# ---------------- llm ----------------
SYSTEM_PROMPT = """
You are AgroAI, a trustworthy agricultural assistant for Indian farmers.

Use retrieved agricultural documents, user information, and live weather data when available.

RULES:
1. Evidence first. Prefer retrieved sources and preserve their location, year, season, metric and units.
2. NEVER confuse geographic levels. State-level data cannot prove district-level facts; a soil sample/study cannot represent an entire district unless explicitly stated.
3. Never invent statistics, soil values, weather, prices, yields, profitability, dosages, schemes or citations.
4. If the retrieved evidence directly answers the question, give the verified answer.
5. If evidence is incomplete, do NOT stop at "cannot be determined." Give:
   - "Verified from provided data:" what the evidence supports.
   - "Suggestive information:" a clearly labelled general/external indication that may help the user, but is NOT verified by the retrieved documents.
6. Never call a crop "most profitable", "best", or "most grown" unless the evidence supports that exact claim.
7. For profitability, distinguish price from profit. Profit requires factors such as yield, selling price and cultivation cost.
8. If external/web information is available, identify it as external information and never present it as retrieved government data. If web information is unavailable, use only clearly labelled general agricultural knowledge.
9. Distinguish current weather, forecast, historical weather and climate averages.
10. For chemical/fertilizer/pesticide advice, never invent doses or safety instructions.
11. Cite supplied filename/page when available. Never invent citations.
12. Ask follow-up questions only when essential.

Before answering, check:
LOCATION → YEAR → SEASON → METRIC → UNITS → SOURCE → EVIDENCE.


Priority:
ACCURACY > EVIDENCE > TRANSPARENCY > HELPFULNESS.

In last translate the whole message in hindi with a heading of Hindi .
"""


def call_llm(messages, max_tokens, temperature, log=False):
    """Try the model that worked last time, then the others one by one."""
    global working_model

    order = MODELS
    if working_model:
        order = [working_model] + [m for m in MODELS if m != working_model]

    for model in order:
        try:
            response = llm_client.chat_completion(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            )
            text = response.choices[0].message.content
            if not text:
                print(f"{model} gave an empty answer, trying next")
                continue
            working_model = model
            if log:
                print(f"[model] {model}")
            return text
        except Exception as e:
            print(f"{model} failed ({repr(e)[:80]}), trying next")
            if model == working_model:
                working_model = None
    return None


def ask_llm(query, district, state, route, weather, context, memory=None):
    parts = [f"Location: {district}, {state}, India"]
    if memory:
        parts.append(f"Summary of user's previous messages:\n{memory}")
    if weather:
        parts.append(f"Live weather:\n{weather}")
    if context:
        parts.append("District knowledge:\n" + "\n---\n".join(context))
    parts.append(f"Question: {query}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return call_llm(messages, max_tokens=1000, temperature=0.3, log=True)


# ---------------- memory (summary of last 5 user messages) ----------------
MEMORY_SIZE = 5
user_memory = {}  # one entry per session, so users on the server don't mix up


def get_memory(session_id="default"):
    if session_id not in user_memory:
        user_memory[session_id] = {"messages": [], "summary": ""}
    return user_memory[session_id]


def summarize_messages(messages):
    if not messages:
        return ""

    numbered = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(messages))
    prompt = [
        {
            "role": "system",
            "content": (
                "You summarise a farmer's recent messages to an agriculture assistant. "
                "Write 2-3 short sentences covering the crops, problems, farming actions "
                "and any details (like land size or season) the farmer mentioned. "
                "Only use what the farmer said. Do not add advice."
            ),
        },
        {"role": "user", "content": f"Farmer's recent messages:\n{numbered}"},
    ]
    text = call_llm(prompt, max_tokens=200, temperature=0.2)
    if text:
        return text.strip()

    # llm not reachable -> keep the raw messages so memory is not lost
    return " | ".join(messages)


def update_memory(session_id, query):
    mem = get_memory(session_id)
    mem["messages"] = (mem["messages"] + [query])[-MEMORY_SIZE:]
    mem["summary"] = summarize_messages(mem["messages"])


# ---------------- main query flow ----------------
THINK_HINT = "\n\n(Think carefully and give a detailed, well-reasoned, step-by-step answer.)"


def process_query(query, district, state, think=False, session_id="default"):
    route = route_query(query, district)
    weather, context = None, []

    if route in ("WEATHER", "BOTH"):
        weather = get_weather(district, state)
        if weather is None:
            print("[warn] weather unavailable, continuing without it")
    if route in ("RETRIEVAL", "BOTH"):
        context = retrieve(query, district, state)

    llm_query = query + THINK_HINT if think else query
    memory = get_memory(session_id)["summary"]
    answer = ask_llm(llm_query, district, state, route, weather, context, memory)

    # save after answering, so memory holds the previous messages only
    update_memory(session_id, query)
    return {"answer": answer, "route": route}


# ---------------- terminal agent ----------------
def run_agent():
    print("\n=== Agri Agent ===")
    state = input("Enter your state: ").strip()
    district = input("Enter your district: ").strip()
    print(f"Location set to {district}, {state}. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        t0 = time.time()
        out = process_query(query, district, state)
        print(f"[router] -> {out['route']}")
        print(f"\nAgent: {out['answer']}")
        print(f"[time: {time.time() - t0:.2f}s]\n")


# ---------------- speech_API ----------------
_stt_client = None


def _get_stt_client():
    global _stt_client
    if _stt_client is None:
        _stt_client = GnaniSTTClient(api_key=GNANI_API_KEY)
    return _stt_client


def translate_hi_to_en(text):
    """Hindi -> English with fallbacks, so one blocked service doesn't break voice."""

    # 1. official Google Cloud Translation (only if key is in .env)
    gkey = os.getenv("GOOGLE_TRANSLATE_API_KEY")
    if gkey:
        try:
            r = requests.post(
                "https://translation.googleapis.com/language/translate/v2",
                params={"key": gkey},
                json={"q": text, "source": "hi", "target": "en", "format": "text"},
                timeout=8,
            )
            return r.json()["data"]["translations"][0]["translatedText"]
        except Exception as e:
            print("[translate] Google Cloud failed:", repr(e))

    # 2. free google endpoint (one retry)
    for _ in range(2):
        try:
            return GoogleTranslator(source="hi", target="en").translate(text)
        except Exception as e:
            print("[translate] free Google failed:", repr(e))
            time.sleep(1.5)

    # 3. MyMemory (free, no key)
    try:
        return MyMemoryTranslator(source="hi-IN", target="en-GB").translate(text)
    except Exception as e:
        print("[translate] MyMemory failed:", repr(e))

    # 4. the HF llm as last resort
    out = call_llm(
        [
            {"role": "system", "content": "Translate the Hindi text to English. Output only the translation, nothing else."},
            {"role": "user", "content": text},
        ],
        max_tokens=300, temperature=0,
    )
    if out:
        return out.strip()

    raise RuntimeError("All translation methods failed")


def voice_to_english(audio_bytes, language_code="hi-IN"):
    """WAV bytes -> (hindi_text, english_text)"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        result = _get_stt_client().transcribe(path, language_code=language_code)
    finally:
        os.remove(path)

    hindi = (result.get("transcript") or "").strip()
    if not hindi:
        return "", ""
    return hindi, translate_hi_to_en(hindi)


def process_voice_query(audio_bytes, district, state, think=False, session_id="default"):
    hindi, english = voice_to_english(audio_bytes)
    if not english:
        return {"error": "Could not understand the audio. Please try again."}

    out = process_query(english, district, state, think=think, session_id=session_id)
    out["transcript"] = hindi
    out["translated"] = english
    return out


if __name__ == "__main__":
    run_agent()

# uvicorn server:app --host 127.0.0.1 --port 8000