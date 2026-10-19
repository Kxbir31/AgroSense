import os
import re
import time
import tempfile
from bisect import bisect_right
from difflib import get_close_matches

import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gnani.stt import GnaniSTTClient



# Hugging Face
# from huggingface_hub import InferenceClient

# Google AI Studio
from google import genai
from google.genai import types

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

# LLM provider keys. Keep only one block active at a time.

# Hugging Face
# HF_TOKEN = os.getenv("HF_TOKEN")
# HF_MODEL = os.getenv("HF_MODEL", "google/gemma-4-31B-it")

# Google AI Studio
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")

# Required regardless of providerr
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GNANI_API_KEY = os.getenv("GNANI_API_KEY")

PDF_FILES = ["India_District_Agri_Master_RAG.pdf"]
DB_DIR = "./chroma_db"
COLLECTION = "agriculture_rag_v2"


# Lowercase word tokens, shared by the router & BM25
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# Load the PDF
pages = []
for file in PDF_FILES:
    for page in PyPDFLoader(file).load():
        page.metadata["source_file"] = file
        pages.append(page)

print(f"Total pages loaded: {len(pages)}")


# Chunking
# The PDF has one paragraph per district, starting like " Sehore district, Madhya Pradesh. Soil:".
# Page breaks can cut a paragraph in half, so all pages are joined first
# and the text is then split at every district heading.
RECORD_START = re.compile(
    r"(?:\A|(?<=\. ))(?=[A-Z][A-Za-z.&'()\- ]{1,45}? district, [A-Za-z&() ]{2,45}?\. Soil:)"
)
RECORD_HEAD = re.compile(r"^(.+?) district, (.+?)\. Soil:")


def build_chunks(pages):
    # Join all pages into one text and remember where each page starts
    text, page_starts = "", []
    for p in pages:
        page_starts.append(len(text))
        text += re.sub(r"\s+", " ", p.page_content).strip() + " "

    # Each district record runs until the start of the next one
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
            "page": bisect_right(page_starts, s),   # 1-based page number
            "source_file": pages[0].metadata.get("source_file", ""),
        }

        # Each record has two parts: the district profile (soil, climate, suited crops)
        # and the state's top four crops
        cut = record.find(" State top-4 crops")
        if cut == -1:
            parts = [("profile", record)]
        else:
            crop_text = f"{district} district, {state} (state-level crop data). " + record[cut:].strip()
            parts = [("profile", record[:cut].strip()), ("crops", crop_text)]

        for name, content in parts:
            chunks.append(Document(page_content=content, metadata={**meta, "part": name}))

    # Sequential chunk ids
    for i, c in enumerate(chunks):
        c.metadata["id"] = i
    return chunks


chunks = build_chunks(pages)
print(f"Total chunks created: {len(chunks)}")

# Lookup from (district, state) to chunk ids
records = {}
for c in chunks:
    key = (c.metadata["district"].lower(), c.metadata["state"].lower())
    records.setdefault(key, []).append(c.metadata["id"])


# Embeddings and Chroma database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def open_vectorstore():
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
    # Reuse the existing collection if it matches the current chunks,
    # otherwise rebuild it. This also prevents duplicates across runs.
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

# BM25 index for keyword search
bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])


# LLM setup. Keep only one of the two blocks active at a time.

# Hugging Face
# llm_client = InferenceClient(token=HF_TOKEN)
# MODELS = [HF_MODEL, "deepseek-ai/DeepSeek-V4.1-Flash", "meta-llama/Llama-3.1-8B-Instruct", "openai/gpt-oss-20b"]

# Google AI Studio
llm_client = genai.Client(api_key=GOOGLE_API_KEY)
MODELS = [GOOGLE_MODEL, "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
MODELS = list(dict.fromkeys(MODELS))   # drop duplicates, keep order

# Last model that worked, tried first on the next call
working_model = None


# Router
# Keywords indicating the question needs live weather
WEATHER_KW = {
    "weather", "rain", "rainy", "raining", "rainfall", "shower", "drizzle",
    "temperature", "temp", "humidity", "humid", "forecast", "wind", "windy",
    "climate", "hot", "cold", "heat", "heatwave", "frost", "fog", "cloud",
    "cloudy", "storm", "thunderstorm", "lightning", "hailstorm", "hail",
    "flood", "drought", "dry", "sunny", "sunlight", "monsoon", "cyclone",
    "today", "tomorrow", "tonight", "now", "current", "currently", "latest",
    "week", "outside", "condition", "conditions", "degree", "celsius",
    "uv", "dew", "mist", "pressure", "visibility",
    # weather-dependent farm work
    "spray", "spraying", "sow", "sowing", "harvest", "harvesting",
    "irrigate", "irrigating", "water", "watering", "drying",
    "transplant", "transplanting", "plough", "ploughing", "plow", "plowing",
    "thresh", "threshing",
    # Hindi and Hinglish
    "barish", "baarish", "mausam", "garmi", "sardi", "thand", "dhoop",
    "hawa", "aandhi", "toofan", "kohra", "olavrishti", "sukha", "badal",
    "aaj", "kal", "abhi", "tapman", "nami", "paani", "pani", "sinchai",
    "kataai", "buwai", "bovai", "chidkav", "chhidkav",
}

# Keywords indicating the question needs the PDF's district knowledge
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
    # inputs and crop protection
    "fertilizer", "fertiliser", "fertilizers", "khad", "urea", "dap", "manure",
    "compost", "pesticide", "insecticide", "fungicide", "herbicide", "pest",
    "pests", "disease", "diseases", "rog", "keet", "weed", "weeds", "dawa",
    # water and irrigation
    "irrigation", "canal", "well", "borewell", "tubewell", "tank", "pond",
    "river", "dam", "groundwater", "watershed", "drip", "sprinkler",
    # statistics and markets
    "yield", "production", "produce", "area", "hectare", "hectares", "ha",
    "acre", "acres", "tonnes", "tons", "productivity", "output", "average",
    "total", "percent", "percentage", "statistics", "data", "report",
    "mandi", "market", "msp", "price", "prices", "rate", "rates", "subsidy",
    "scheme", "schemes", "insurance", "loan", "kcc",
    # district and general information
    "district", "zila", "jila", "state", "block", "tehsil", "village", "gaon",
    "region", "agroclimatic", "zone", "major", "main", "top", "type", "types",
    "kind", "which", "what", "list", "tell", "about", "information", "details",
    "livestock", "cattle", "dairy", "poultry", "fishery", "fish",
}

# Advice-style questions need both weather and district knowledge
ADVICE_KW = {
    "should", "can", "could", "advice", "advise", "suggest", "suggestion",
    "recommend", "recommendation", "best", "when", "plan", "planning",
    "ideal", "suitable", "right", "safe", "good", "worth", "ok", "okay",
    "sahi", "sakte", "sakta", "chahiye", "kab", "kya", "kaise", "salah",
}


# Decides whether a question needs weather, PDF knowledge, both, or neither
def route_query(query, district):
    words = set(tokenize(query))
    need_weather = bool(words & WEATHER_KW)
    need_retrieval = bool(words & RETRIEVAL_KW) or district.lower() in query.lower()

    # Catch other word forms such as irrigating or fertilizers
    if any(w.startswith(("weath", "rain", "temper", "forecast")) for w in words):
        need_weather = True
    if any(w.startswith(("irrigat", "fertili", "cultivat", "agri")) for w in words):
        need_retrieval = True

    # Advice questions about farming need both sources
    if words & ADVICE_KW and (need_weather or need_retrieval):
        need_weather = need_retrieval = True

    if need_weather and need_retrieval:
        return "BOTH"
    if need_weather:
        return "WEATHER"
    if need_retrieval:
        return "RETRIEVAL"

    # No keyword matched: skip retrieval only for very short messages like "hi"
    if len(words) <= 2:
        return "DIRECT"
    return "RETRIEVAL"


# Weather
_weather_cache = {}
WEATHER_TTL = 600  # cache lifetime in seconds


# Fetches live weather from OpenWeather, with a short-lived cache to avoid repeat calls
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
        # Geocode the district first
        geo = requests.get(
            geo_url,
            params={"q": f"{district},{state},IN", "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=5,
        ).json()
        # Retry without the state name
        if not geo:
            geo = requests.get(
                geo_url,
                params={"q": f"{district},IN", "limit": 1, "appid": OPENWEATHER_API_KEY},
                timeout=5,
            ).json()
        # An error response comes back as a dict rather than a list
        if not geo or isinstance(geo, dict):
            print("Weather error:", geo)
            return None

        # Current weather for the resolved coordinates
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


# Formats the weather data as one readable line for the LLM
def get_weather(district, state):
    d = get_weather_data(district, state)
    if not d:
        return None
    return (
        f"Temperature: {d['temp']}°C (feels like {d['feels_like']}°C), "
        f"Humidity: {d['humidity']}%, Condition: {d['condition']}, "
        f"Wind: {d['wind']} m/s, Rain (last 1h): {d['rain_1h']} mm"
    )


# Retrieval
def find_district(district, state):
    """Chunk ids of the user's own district (state name can be 'Telangana' etc.)."""
    d, s = district.lower().strip(), state.lower().strip()
    # Search within the same state, falling back to all districts
    pool = [k for k in records if s and (s in k[1] or k[1] in s)] or list(records)
    # Fuzzy match to tolerate spelling mistakes
    match = get_close_matches(d, [k[0] for k in pool], n=1, cutoff=0.8)
    if not match:
        return []
    key = next(k for k in pool if k[0] == match[0])
    return records[key]


def districts_in_query(query, state, skip, limit=2):
    """Other districts the user names in the question, for example compare with Indore."""
    q = query.lower()
    found = [k for k in records if k[0] != skip and re.search(rf"\b{re.escape(k[0])}\b", q)]
    s = state.lower().strip()
    found.sort(key=lambda k: not (s and (s in k[1] or k[1] in s)))  # same-state districts first
    return found[:limit]


def hybrid_search(q, state, k):
    """Combines BM25 keyword search and vector search using Reciprocal Rank Fusion."""
    # Keyword search
    scores = bm25.get_scores(tokenize(q))
    bm_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    # Semantic search
    vec_top = [d.metadata["id"] for d in vectorstore.similarity_search(q, k=5)]

    # Reciprocal Rank Fusion: chunks ranked high in both lists score best
    fused = {}
    for ranked in (bm_top, vec_top):
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)

    # Small boost for chunks from the user's state
    s = state.lower().strip()
    for idx in fused:
        if s and s in chunks[idx].metadata["state"].lower():
            fused[idx] += 0.05

    return sorted(fused, key=fused.get, reverse=True)[:k]


def retrieve(query, district, state, k=3):
    # Always start with the user's own district
    ids = list(find_district(district, state))
    own_name = chunks[ids[0]].metadata["district"].lower() if ids else ""

    # Add any other districts named in the question
    for key in districts_in_query(query, state, skip=own_name):
        for i in records[key]:
            if i not in ids:
                ids.append(i)

    # District not in the PDF: fall back to hybrid search
    if not ids:
        ids = hybrid_search(f"{query} {district} {state}", state, k)
        found = [f"[Page {chunks[i].metadata['page']}] {chunks[i].page_content}" for i in ids]
        note = (f"[Note] '{district}' is not in the dataset. The records below belong to "
                f"other districts and must not be treated as {district} data.")
        return [note] + found

    return [f"[Page {chunks[i].metadata['page']}] {chunks[i].page_content}" for i in ids]


# LLM
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
LOCATION → YEAR → SEASON → METRIC → UNITS → SOURCE → EVIDENCE


Priority:
ACCURACY > EVIDENCE > TRANSPARENCY > HELPFULNESS

In last translate the whole message in hindi with a heading of HINDI .
"""


# Hugging Face version of call_llm
# def call_llm(messages, max_tokens, temperature, log=False):
#     """Try the model that worked last time, then the others one by one."""
#     global working_model
#
#     order = MODELS
#     if working_model:
#         order = [working_model] + [m for m in MODELS if m != working_model]
#
#     for model in order:
#         try:
#             response = llm_client.chat_completion(
#                 model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
#             )
#             text = response.choices[0].message.content
#             if not text:
#                 print(f"{model} gave an empty answer, trying next")
#                 continue
#             working_model = model
#             if log:
#                 print(f"[model] {model}")
#             return text
#         except Exception as e:
#             print(f"{model} failed ({repr(e)[:80]}), trying next")
#             if model == working_model:
#                 working_model = None
#     return None


# Google AI Studio version of call_llm

# Splits out the system message and maps the assistant role to Gemini's "model"
def _to_gemini(messages):
    system_parts, contents = [], []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return ("\n\n".join(system_parts) or None), contents


def call_llm(messages, max_tokens, temperature, log=False):
    """Try the model that worked last time, then the others one by one."""
    global working_model

    # Try the last working model first
    order = MODELS
    if working_model:
        order = [working_model] + [m for m in MODELS if m != working_model]

    system_instruction, contents = _to_gemini(messages)

    for model in order:
        try:
            cfg = dict(
                system_instruction=system_instruction,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            # Gemini models spend part of the token limit on thinking, which can
            # leave the answer empty or truncated. 2.5 flash allows disabling
            # thinking; newer models do not, so they get extra token headroom.
            if model.startswith("gemini-2.5-flash"):
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            else:
                cfg["max_output_tokens"] = max_tokens + 2000

            response = llm_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg),
            )
            text = response.text
            if not text:
                print(f"{model} gave an empty answer, trying next")
                continue
            working_model = model
            if log:
                print(f"[model] {model}")
            return text
        except Exception as e:
            print(f"{model} failed ({repr(e)[:300]}), trying next")
            if model == working_model:
                working_model = None
    return None


# Builds the final prompt from location, memory, weather and PDF context, then calls the LLM
def ask_llm(query, district, state, route, weather, context, memory=None):
    parts = [f"Location: {district}, {state}, India"]
    if memory:
        parts.append(f"Summary of user's previous messages :\n{memory}")
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


# Memory: short summary of the last 5 user messages
MEMORY_SIZE = 5
user_memory = {}  # keyed by session so users do not get mixed up


def get_memory(session_id="default"):
    if session_id not in user_memory:
        user_memory[session_id] = {"messages": [], "summary": ""}
    return user_memory[session_id]


# Asks the LLM for a short summary of the farmer's recent messages
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
                "Only use what the farmer said. Do not add advices"
            ),
        },
        {"role": "user", "content": f"Farmer's recent messages:\n{numbered}"},
    ]
    text = call_llm(prompt, max_tokens=200, temperature=0.2)
    if text:
        return text.strip()

    # LLM unavailable: keep the raw messages so the memory is not lost
    return " | ".join(messages)


def update_memory(session_id, query):
    mem = get_memory(session_id)
    mem["messages"] = (mem["messages"] + [query])[-MEMORY_SIZE:]
    mem["summary"] = summarize_messages(mem["messages"])


# Main query flow
THINK_HINT = "\n\n(Think carefully and give a detailed, well-reasoned, step-by-step answer.)"


def process_query(query, district, state, think=False, session_id="default"):
    route = route_query(query, district)
    weather, context = None, []

    # Fetch only what the route requires
    if route in ("WEATHER", "BOTH"):
        weather = get_weather(district, state)
        if weather is None:
            print("[warn] weather unavailable, continuing without it")
    if route in ("RETRIEVAL", "BOTH"):
        context = retrieve(query, district, state)

    llm_query = query + THINK_HINT if think else query
    memory = get_memory(session_id)["summary"]
    answer = ask_llm(llm_query, district, state, route, weather, context, memory)

    # Update memory after answering so the summary only covers earlier messages
    update_memory(session_id, query)
    return {"answer": answer, "route": route}


# Terminal agent for command-line testing
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


# Speech API
_stt_client = None


# Creates the speech client once and reuses it
def _get_stt_client():
    global _stt_client
    if _stt_client is None:
        _stt_client = GnaniSTTClient(api_key=GNANI_API_KEY)
    return _stt_client


def translate_hi_to_en(text):
    """Hindi to English with backups, so one blocked service does not break voice."""

    # 1. Official Google Cloud Translation, only if the key is set
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

    # 2. Free Google service, two attempts
    for _ in range(2):
        try:
            return GoogleTranslator(source="hi", target="en").translate(text)
        except Exception as e:
            print("[translate] free Google failed:", repr(e))
            time.sleep(1.5)

    # 3. MyMemory, free and keyless
    try:
        return MyMemoryTranslator(source="hi-IN", target="en-GB").translate(text)
    except Exception as e:
        print("[translate] MyMemory failed:", repr(e))

    # 4. Last resort: the LLM
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
    """Takes wav bytes and gives back the hindi text and the english text."""
    # The speech client requires a file path, so write the audio to a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        result = _get_stt_client().transcribe(path, language_code=language_code)
    finally:
        # Always remove the temp file, even on failure
        os.remove(path)

    hindi = (result.get("transcript") or "").strip()
    if not hindi:
        return "", ""
    return hindi, translate_hi_to_en(hindi)


def process_voice_query(audio_bytes, district, state, think=False, session_id="default"):
    hindi, english = voice_to_english(audio_bytes)
    if not english:
        return {"error" : "Could not understand the audio. Please try again ."}

    out = process_query(english, district, state, think=think, session_id=session_id)
    out["transcript"] = hindi
    out["translated"] = english
    return out


if __name__ == "__main__":
    run_agent()



# uvicorn server:app --host 127.0.0.1 --port 8000