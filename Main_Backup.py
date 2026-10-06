import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rank_bm25 import BM25Okapi

# LOAD ENVIRONMENT VARIABLES


load_dotenv()

# LOAD PDF DOCUMENTS

files = ['India_District_Agri_Master_RAG.pdf'
         ]

docs = []

for file in files:
    loader = PyPDFLoader(file)
    file_docs = loader.load()

    # Store source filename in metadata
    for doc in file_docs:
        doc.metadata["source_file"] = file

    docs.extend(file_docs)

print(f"Total pages loaded: {len(docs)}")

# SPLIT DOCUMENTS INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

# CREATE EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STORE CHUNKS IN CHROMADB
import os

# db already saved -> just open it, else build it (stops duplicates on every run)
if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
    vectorstore = Chroma(
        collection_name="agriculture_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="agriculture_rag",
        persist_directory="./chroma_db"
    )


# CREATE BM25 INDEX
def tokenize(text):
    """
    Convert text into lowercase words.
    Handles punctuation better than simple .split()
    """
    return re.findall(r"\b\w+\b", text.lower())


tokenized_chunks = [
    tokenize(doc.page_content)
    for doc in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

# AGENT: ROUTER + WEATHER + HYBRID RETRIEVAL + HF LLM
import time
import requests
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-4-31B-it")

# no model fixed here, ask_llm tries the models below one by one
llm_client = InferenceClient(token=HF_TOKEN)
MODELS = [HF_MODEL, "deepseek-ai/DeepSeek-V4.1-Flash", "meta-llama/Llama-3.1-8B-Instruct", "openai/gpt-oss-20b"]
working_model = None

# Map chunk text -> index, so vector hits can be matched with BM25 indices
chunk_index = {c.page_content: i for i, c in enumerate(chunks)}

# ---------------- 1. ROUTER (keyword based, no LLM call) ----------------
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
    # Farming actions that depend on live weather
    "spray", "spraying", "sow", "sowing", "harvest", "harvesting",
    "irrigate", "irrigating", "water", "watering", "dry", "drying",
    "transplant", "transplanting", "plough", "ploughing", "plow", "plowing",
    "thresh", "threshing",
    # Hindi / Hinglish
    "barish", "baarish", "mausam", "garmi", "sardi", "thand", "dhoop",
    "hawa", "aandhi", "toofan", "kohra", "olavrishti", "sukha", "badal",
    "aaj", "kal", "abhi", "tapman", "nami", "paani", "pani", "sinchai",
    "kataai", "buwai", "bovai", "chidkav", "chhidkav",
}

# pdf / district knowledge words
RETRIEVAL_KW = {
    # Soil and land
    "soil", "soils", "mitti", "land", "zameen", "jameen", "terrain", "texture",
    "ph", "organic", "carbon", "nutrient", "nutrients", "nitrogen", "phosphorus",
    "potash", "npk", "alluvial", "black", "red", "laterite", "loamy", "clay",
    "sandy", "saline", "alkaline", "acidic", "fertility", "topography",
    # Crops
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
    # Seasons
    "season", "seasons", "kharif", "rabi", "zaid", "summer", "winter",
    # Inputs and protection
    "fertilizer", "fertiliser", "fertilizers", "khad", "urea", "dap", "manure",
    "compost", "pesticide", "insecticide", "fungicide", "herbicide", "pest",
    "pests", "disease", "diseases", "rog", "keet", "weed", "weeds", "dawa",
    # Water and irrigation infrastructure
    "irrigation", "canal", "well", "borewell", "tubewell", "tank", "pond",
    "river", "dam", "groundwater", "watershed", "drip", "sprinkler",
    # Statistics and economics
    "yield", "production", "produce", "area", "hectare", "hectares", "ha",
    "acre", "acres", "tonnes", "tons", "productivity", "output", "average",
    "total", "percent", "percentage", "statistics", "data", "report",
    "mandi", "market", "msp", "price", "prices", "rate", "rates", "subsidy",
    "scheme", "schemes", "insurance", "loan", "kcc",
    # District / general location info
    "district", "zila", "jila", "state", "block", "tehsil", "village", "gaon",
    "region", "agroclimatic", "zone", "major", "main", "top", "type", "types",
    "kind", "which", "what", "list", "tell", "about", "information", "details",
    "livestock", "cattle", "dairy", "poultry", "fishery", "fish",
}

# Advice-style questions need BOTH weather and district knowledge
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

    # advice question mentioning a farming action -> use both sources
    if words & ADVICE_KW and (need_weather or need_retrieval):
        need_weather = need_retrieval = True

    if need_weather and need_retrieval:
        return "BOTH"
    if need_weather:
        return "WEATHER"
    if need_retrieval:
        return "RETRIEVAL"

    # nothing matched -> only skip the pdf for tiny messages like "hi"
    if len(words) <= 2:
        return "DIRECT"
    return "RETRIEVAL"


# ---------------- 2. WEATHER (OpenWeatherMap + 10 min cache) ----------------
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
    try:
        geo = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": f"{district},{state},IN", "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=5,
        ).json()
        if not geo:
            geo = requests.get(
                "https://api.openweathermap.org/geo/1.0/direct",
                params={"q": f"{district},IN", "limit": 1, "appid": OPENWEATHER_API_KEY},
                timeout=5,
            ).json()
        if not geo or isinstance(geo, dict):
            print("Weather error:", geo)
            return None
        lat, lon = geo[0]["lat"], geo[0]["lon"]
        w = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "units": "metric", "appid": OPENWEATHER_API_KEY},
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


# ---------------- 3. HYBRID RETRIEVAL (BM25 + Vector, merged with RRF) ----------------
def retrieve(query, district, state, k=3):
    q = f"{query} {district} {state}"

    # BM25 top 10
    bm_scores = bm25.get_scores(tokenize(q))
    bm_top = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:10]

    # Vector top 1
    vec_docs = vectorstore.similarity_search(q, k=1)
    vec_top = [chunk_index[d.page_content] for d in vec_docs if d.page_content in chunk_index]

    # Reciprocal Rank Fusion
    fused = {}
    for ranked in (bm_top, vec_top):
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)

    # Boost chunks that actually mention the user's district
    for idx in fused:
        if district.lower() in chunks[idx].page_content.lower():
            fused[idx] += 0.05

    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [
        f"[Page {chunks[i].metadata.get('page')}] {chunks[i].page_content}"
        for i in best
    ]


# ---------------- 4. LLM (Hugging Face Inference API) ----------------
SYSTEM_PROMPT = (
    """

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
)


def ask_llm(query, district, state, route, weather, context, memory=None):
    global working_model

    parts = [f"Location: {district}, {state}, India"]
    # summary of the user's last few messages (memory)
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

    # use the model that worked last time, else try them one by one
    to_try = [working_model] if working_model else MODELS

    for model in to_try:
        try:
            response = llm_client.chat_completion(
                model=model, messages=messages, max_tokens=1000, temperature=0.3,
            )
            text = response.choices[0].message.content
            if not text:
                print(f"{model} gave an empty answer, trying next")
                continue
            working_model = model
            print(f"[model] {model}")
            return text
        except Exception:
            print(f"{model} failed, trying next")
            if model == working_model:
                working_model = None
    return None


# ---------------- 4.5 MEMORY (summary of the last 5 user messages) ----------------
MEMORY_SIZE = 5
# one entry per session, so different users on the server don't mix up
user_memory = {}


def get_memory(session_id="default"):
    if session_id not in user_memory:
        user_memory[session_id] = {"messages": [], "summary": ""}
    return user_memory[session_id]


def summarize_messages(messages):
    """Ask the LLM to squeeze the last few user messages into a short summary."""
    global working_model

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

    to_try = [working_model] if working_model else MODELS
    for model in to_try:
        try:
            response = llm_client.chat_completion(
                model=model, messages=prompt, max_tokens=200, temperature=0.2,
            )
            text = response.choices[0].message.content
            if text:
                working_model = model
                return text.strip()
        except Exception:
            print(f"[memory] {model} failed, trying next")
            if model == working_model:
                working_model = None

    # llm not reachable -> just keep the raw messages so memory is not lost
    return " | ".join(messages)


def update_memory(session_id, query):
    mem = get_memory(session_id)
    mem["messages"].append(query)
    mem["messages"] = mem["messages"][-MEMORY_SIZE:]  # keep only last 5
    mem["summary"] = summarize_messages(mem["messages"])


THINK_HINT = "\n\n(Think carefully and give a detailed, well-reasoned, step-by-step answer.)"


def process_query(query, district, state, think=False, session_id="default"):
    route = route_query(query, district)
    weather, context = None, []
    if route in ("WEATHER", "BOTH"):
        weather = get_weather(district, state)
    if route in ("RETRIEVAL", "BOTH"):
        context = retrieve(query, district, state)
    llm_query = query + THINK_HINT if think else query
    memory = get_memory(session_id)["summary"]
    answer = ask_llm(llm_query, district, state, route, weather, context, memory)
    # save this message only after answering, so memory holds the *previous* messages
    update_memory(session_id, query)
    return {"answer": answer, "route": route}


# ---------------- 5. AGENT LOOP ----------------

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
        route = route_query(query, district)
        print(f"[router] -> {route}")

        weather, context = None, []

        # weather
        if route in ("WEATHER", "BOTH"):
            weather = get_weather(district, state)
            if weather is None:
                print("[warn] weather unavailable, continuing without it")

        # retrieval
        if route in ("RETRIEVAL", "BOTH"):
            context = retrieve(query, district, state)

        # llm (with summary of the last 5 user messages)
        memory = get_memory()["summary"]
        answer = ask_llm(query, district, state, route, weather, context, memory)
        update_memory("default", query)

        print(f"\nAgent: {answer}")
        print(f"[time: {time.time() - t0:.2f}s]\n")


# VOICE INPUT
import tempfile
from gnani.stt import GnaniSTTClient
from deep_translator import GoogleTranslator

GNANI_API_KEY = os.getenv("GNANI_API_KEY")
_stt_client = None


def _get_stt_client():
    global _stt_client
    if _stt_client is None:
        _stt_client = GnaniSTTClient(api_key=GNANI_API_KEY)
    return _stt_client


from deep_translator import MyMemoryTranslator


def translate_hi_to_en(text):
    """Hindi -> English with fallbacks, so one blocked service doesn't break voice."""
    global working_model

    # 1. Official Google Cloud Translation (only if you set GOOGLE_TRANSLATE_API_KEY in .env)
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

    # 2. Free Google web endpoint (one retry)
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

    # 4. Your own Hugging Face LLM as a last resort
    for model in ([working_model] if working_model else MODELS):
        try:
            resp = llm_client.chat_completion(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "Translate the Hindi text to English. Output only the translation, nothing else."},
                    {"role": "user", "content": text},
                ],
                max_tokens=300, temperature=0,
            )
            out = resp.choices[0].message.content
            if out:
                working_model = model
                return out.strip()
        except Exception as e:
            print(f"[translate] {model} failed:", repr(e))
            if model == working_model:
                working_model = None

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
    english = translate_hi_to_en(hindi)
    return hindi, english


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