# AgroSense 🌾

> **Your intelligent agriculture assistant for India**

AgroSense is a location-aware AI agriculture assistant that combines **Hybrid RAG, BM25 retrieval, vector search, live weather data, deterministic query routing, and LLM-based response generation** to provide practical agricultural insights for Indian users.

The project has two major layers:

- **Frontend:** A responsive, glassmorphism-style web interface built with HTML, CSS, and JavaScript.
- **AI Backend:** A Python-based RAG and agent pipeline that routes queries, retrieves agricultural knowledge, fetches live weather, and generates grounded responses using Hugging Face models.

---

## 📸 Overview

AgroSense starts by asking the user for their:

```text
State
District
```

The selected location becomes the context for the entire session.

The user can then ask questions such as:

```text
What crops are suitable for my district?
What's the weather today?
Which crops have the highest profitability?
What is the soil condition in my district?
Should I irrigate today?
```

AgroSense determines which information sources are required and generates the final answer accordingly.

---

# ✨ Features

## 📍 Location Personalization

The frontend opens with a location-selection modal where the user selects:

- State
- District

The district list dynamically changes according to the selected state.

The selected location is displayed in the top-right corner and is sent with every chat request.

The frontend obtains the available state/district data from:

```text
GET /api/locations
```

---

## 🌦️ Live Weather

Users can open the weather panel from the top-right weather button.

The weather card displays:

- Current temperature
- Feels-like temperature
- Weather condition
- Humidity
- Wind speed
- Rainfall in the last hour

The frontend requests weather data through:

```text
GET /api/weather?state=<state>&district=<district>
```

The frontend also caches weather responses for **10 minutes** to avoid unnecessary repeated requests.

The Python backend independently uses OpenWeatherMap and also maintains a 10-minute weather cache.

---

## 🤖 AI Agriculture Assistant

The main chat interface allows users to ask agricultural questions in natural language.

Each request sends:

```json
{
  "query": "User question",
  "state": "State",
  "district": "District",
  "think": false
}
```

to:

```text
POST /api/chat
```

The backend processes the query and returns the generated answer.

---

## 🧠 Think Mode

AgroSense includes a **Think** toggle in the chat interface.

When enabled:

```text
think = true
```

The backend adds a reasoning hint to the query, requesting a more detailed and carefully reasoned answer.

This allows the user to choose between a normal response and a more detailed response.

---

## 🎤 Voice Input

The frontend supports browser-based speech recognition using:

```javascript
SpeechRecognition
webkitSpeechRecognition
```

The configured language is:

```text
en-IN
```

If the browser does not support speech recognition, the microphone button is automatically hidden.

---

## 💬 Suggested Queries

The landing screen provides ready-to-use prompts:

- What crops are suitable for my district?
- What's the weather today?
- Which crops have the highest profitability?
- What is the soil condition in my district?

Clicking a suggestion directly sends it to the AI assistant.

---

## 📝 Markdown Responses

AI responses are rendered as Markdown using:

```text
Marked.js
```

The rendered HTML is sanitized using:

```text
DOMPurify
```

This allows responses to contain:

- Headings
- Lists
- Tables
- Code blocks
- Inline code
- Paragraphs

while sanitizing generated HTML before inserting it into the page.

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      AgroSense UI    │
                         │ HTML + CSS + JS      │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼──────────────────┐
                  │                 │                  │
                  ▼                 ▼                  ▼
           /api/locations     /api/weather        /api/chat
                  │                 │                  │
                  │                 ▼                  │
                  │          OpenWeatherMap            │
                  │                                    │
                  │                                    ▼
                  │                          ┌──────────────────┐
                  │                          │ Python AI Agent │
                  │                          └────────┬─────────┘
                  │                                   │
                  │                         ┌─────────▼─────────┐
                  │                         │ Deterministic     │
                  │                         │ Query Router      │
                  │                         └─────────┬─────────┘
                  │                                   │
                  │                    ┌──────────────┼──────────────┐
                  │                    │              │              │
                  │                    ▼              ▼              ▼
                  │                 WEATHER       RETRIEVAL        BOTH
                  │                    │              │              │
                  │                    │              ▼              │
                  │                    │       ┌─────────────┐       │
                  │                    │       │ BM25        │       │
                  │                    │       │ + Vector    │       │
                  │                    │       │ Retrieval   │       │
                  │                    │       └──────┬──────┘       │
                  │                    │              │              │
                  │                    │              ▼              │
                  │                    │             RRF             │
                  │                    │              │              │
                  │                    └──────────────┼──────────────┘
                  │                                   ▼
                  │                          ┌─────────────────┐
                  │                          │ Context Builder │
                  │                          └────────┬────────┘
                  │                                   │
                  │                                   ▼
                  │                          ┌─────────────────┐
                  │                          │ Hugging Face    │
                  │                          │ LLM             │
                  │                          └────────┬────────┘
                  │                                   │
                  └───────────────────────────────────┤
                                                      ▼
                                             AI Response → UI
```

---

# 🔄 Complete Request Flow

## 1. Application Startup

The frontend requests:

```text
GET /api/locations
```

The server returns the available state/district mapping.

The location modal is then displayed.

---

## 2. User Selects Location

The user selects:

```text
State → District
```

The frontend stores the selection in:

```javascript
LOC = {
    state: "...",
    district: "..."
}
```

The main AgroSense interface is then activated.

---

## 3. User Sends Query

For example:

```text
Should I irrigate my crop today?
```

The frontend sends:

```text
POST /api/chat
```

with:

```json
{
  "query": "Should I irrigate my crop today?",
  "state": "Bihar",
  "district": "Katihar",
  "think": false
}
```

---

## 4. Query Routing

The backend examines the query using a deterministic keyword-based router.

Possible routes:

```text
WEATHER
RETRIEVAL
BOTH
DIRECT
```

For example:

```text
What's the weather today?
        ↓
WEATHER
```

```text
What is the soil type in my district?
        ↓
RETRIEVAL
```

```text
Should I irrigate today?
        ↓
BOTH
```

The router does not require a separate LLM call.

---

# 🔎 Hybrid RAG Pipeline

AgroSense uses a hybrid retrieval approach combining lexical and semantic retrieval.

```text
User Query
    │
    ├───────────────► BM25 Search
    │                     │
    │                     ▼
    │                 Top 10
    │
    └───────────────► Vector Search
                          │
                          ▼
                       Top 10
                          │
                          ▼
                 Reciprocal Rank
                    Fusion (RRF)
                          │
                          ▼
                  District Boost
                          │
                          ▼
                     Top 3 Chunks
```

---

## 📚 Document Loading

The current RAG pipeline loads:

```text
India_District_Agri_Master_RAG.pdf
```

using:

```python
PyPDFLoader
```

Each loaded page receives source metadata.

---

## ✂️ Document Chunking

Documents are split using:

```python
RecursiveCharacterTextSplitter
```

Current configuration:

```text
chunk_size = 750
chunk_overlap = 150
```

---

## 🧬 Embeddings

AgroSense uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through:

```text
HuggingFaceEmbeddings
```

---

## 🗄️ ChromaDB

The generated embeddings are stored in:

```text
ChromaDB
```

with a persistent directory:

```text
./chroma_db
```

If the database already exists, it is reopened instead of unnecessarily rebuilding it.

---

# 🔤 BM25 Retrieval

AgroSense uses:

```text
rank_bm25
```

with:

```text
BM25Okapi
```

for lexical retrieval.

This is useful when a query contains exact terms such as:

- District names
- Crop names
- Soil types
- Fertilizers
- Agricultural schemes
- MSP
- Irrigation terms

The system retrieves the top 10 BM25 results.

---

# 🔬 Vector Retrieval

The same query is sent to ChromaDB for semantic similarity search.

The system retrieves:

```text
Top 10 vector results
```

This helps retrieve relevant information even when the user's wording differs from the wording in the source document.

---

# 🔀 Reciprocal Rank Fusion

BM25 and vector results are combined using Reciprocal Rank Fusion.

The implementation uses:

```text
RRF Score = Σ 1 / (60 + rank)
```

The highest-ranked results from both retrieval methods receive stronger scores.

AgroSense then gives an additional boost to retrieved chunks that explicitly mention the user's district.

Finally, the best:

```text
3 chunks
```

are passed to the LLM as context.

---

# 🌦️ Weather Architecture

Weather requests follow:

```text
State + District
       │
       ▼
OpenWeatherMap Geocoding API
       │
       ▼
Latitude + Longitude
       │
       ▼
OpenWeatherMap Current Weather API
       │
       ▼
Weather Data
```

The backend extracts:

```text
Temperature
Feels-like temperature
Humidity
Weather condition
Wind speed
Rainfall in last hour
```

The frontend displays the same information through the weather popup.

---

# 🧠 LLM Layer

The backend uses the:

```text
Hugging Face Inference API
```

for response generation.

The configured model is controlled using:

```env
HF_MODEL
```

The backend also supports fallback models.

Configured candidates include:

```text
deepseek-ai/DeepSeek-V4.1-Flash
meta-llama/Llama-3.1-8B-Instruct
Qwen/Qwen2.5-72B-Instruct
openai/gpt-oss-20b
```

The backend remembers the last working model during runtime and attempts to reuse it.

---

# 🛡️ Grounded Response Generation

The LLM receives:

```text
Location
+
Live Weather (when relevant)
+
Retrieved Agricultural Knowledge (when relevant)
+
User Question
```

The response policy prioritizes:

- Retrieved evidence
- State/district relevance
- Soil information
- Crop information
- Season information
- Live weather when relevant

The system is instructed not to fabricate:

```text
Government data
Soil measurements
Weather forecasts
Fertilizer dosages
Pesticide instructions
Prices
Scheme eligibility
```

If information is incomplete, the assistant should provide the supported information and communicate uncertainty.

---

# 🖥️ Frontend

The AgroSense frontend is implemented as a standalone HTML interface with embedded CSS and JavaScript.

The interface uses:

```text
HTML5
CSS3
Vanilla JavaScript
```

External frontend libraries loaded through CDN:

```text
Marked.js
DOMPurify
Google Fonts
```

The frontend source explicitly loads Inter and Outfit fonts and uses Marked.js and DOMPurify for response rendering and sanitization. fileciteturn1file0L5-L11

---

## 🎨 UI Design

The interface uses:

- Full-screen agricultural background
- Dark translucent glass panels
- Blur effects
- Rounded UI components
- Minimal typography
- Responsive layout
- Animated message appearance
- Floating chat input
- Weather popup
- Location selector modal

The background image is loaded from:

```text
/static/background.jpg
```

---

## 🧭 Location Modal

The startup modal provides:

```text
State
District
```

with searchable comboboxes.

The district input remains disabled until a state is selected.

Users can change their location later using the location button.

---

## 💬 Chat Interface

The chat interface supports:

- User messages
- AI messages
- Markdown
- Tables
- Code blocks
- Loading animation
- Error states
- Auto-scrolling
- Multiline input
- Enter-to-send
- Shift+Enter for a new line
- New chat button

---

## 🎤 Voice Interaction

Speech recognition is automatically enabled when supported by the browser.

The frontend uses:

```javascript
window.SpeechRecognition ||
window.webkitSpeechRecognition
```

and configures:

```text
en-IN
```

The recognized text is inserted into the chat input.

---

## 🔐 Response Sanitization

AI Markdown is processed through:

```text
marked.parse()
        ↓
DOMPurify.sanitize()
        ↓
Rendered in UI
```

This provides Markdown formatting while sanitizing the generated HTML before rendering.

---

# 🛠️ Tech Stack

## Frontend

| Technology | Purpose |
|---|---|
| **HTML5** | Frontend structure |
| **CSS3** | Styling, glassmorphism, responsive UI |
| **Vanilla JavaScript** | Frontend logic and API communication |
| **Marked.js** | Markdown rendering |
| **DOMPurify** | HTML sanitization |
| **Google Fonts** | Inter + Outfit typography |
| **Web Speech API** | Voice input |

## Backend / AI

| Technology | Purpose |
|---|---|
| **Python** | AI backend |
| **LangChain** | RAG/document processing components |
| **PyPDFLoader** | PDF ingestion |
| **RecursiveCharacterTextSplitter** | Text chunking |
| **HuggingFaceEmbeddings** | Embeddings |
| **Sentence Transformers** | Semantic embeddings |
| **ChromaDB** | Vector database |
| **rank_bm25** | BM25 lexical retrieval |
| **RRF** | Hybrid retrieval fusion |
| **Hugging Face Inference API** | LLM inference |
| **Requests** | External API requests |
| **python-dotenv** | Environment variables |

## External Services

| Service | Purpose |
|---|---|
| **OpenWeatherMap** | Geocoding + live weather |
| **Hugging Face** | LLM inference + model access |

---

# 📁 Recommended Project Structure

```text
AgroSense/
│
├── main.py
├── app.py
│
├── India_District_Agri_Master_RAG.pdf
│
├── chroma_db/
│
├── static/
│   └── background.jpg
│
├── templates/
│   └── index.html
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> The exact backend entry-point filename depends on how the API routes are currently implemented. The frontend requires the server to expose `/api/locations`, `/api/weather`, and `/api/chat`.

---

# 🔌 Frontend API Contract

The frontend expects three backend endpoints.

## 1. Locations

### Request

```http
GET /api/locations
```

### Expected response

```json
{
  "Bihar": [
    "Katihar",
    "Patna",
    "Gaya"
  ],
  "Madhya Pradesh": [
    "Bhopal",
    "Indore"
  ]
}
```

The exact locations depend on the dataset returned by the backend.

---

## 2. Weather

### Request

```http
GET /api/weather?state=Bihar&district=Katihar
```

### Expected response

```json
{
  "temp": 28.5,
  "feels_like": 30.1,
  "humidity": 72,
  "condition": "overcast clouds",
  "wind": 3.2,
  "rain_1h": 0
}
```

---

## 3. Chat

### Request

```http
POST /api/chat
Content-Type: application/json
```

### Body

```json
{
  "query": "What crops are suitable for my district?",
  "state": "Bihar",
  "district": "Katihar",
  "think": false
}
```

### Expected response

```json
{
  "answer": "Generated agricultural response..."
}
```

The frontend renders the `answer` field as sanitized Markdown.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AgroSense.git
cd AgroSense
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Core backend dependencies include:

```text
python-dotenv
requests
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-chroma
chromadb
rank-bm25
huggingface-hub
sentence-transformers
pypdf
```

---

# 🔐 Environment Variables

Create:

```text
.env
```

in the project root.

Add:

```env
HF_TOKEN=your_huggingface_token
OPENWEATHER_API_KEY=your_openweathermap_api_key
HF_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
```

Never commit `.env` to GitHub.

Add:

```gitignore
.env
venv/
__pycache__/
*.pyc
chroma_db/
```

---

# ▶️ Running the Application

Start the backend/API server that serves the frontend and exposes:

```text
/api/locations
/api/weather
/api/chat
```

Then open the application in the browser.

The frontend automatically requests the location data on startup.

If the API server is unavailable, the UI displays:

```text
Could not load locations. Is the server running?
```

---

# 🧪 Example User Queries

## Weather

```text
What's the weather today?
```

Route:

```text
WEATHER
```

---

## Soil

```text
What is the soil condition in my district?
```

Route:

```text
RETRIEVAL
```

---

## Crop Suitability

```text
What crops are suitable for my district?
```

Route:

```text
RETRIEVAL
```

---

## Weather + Agriculture

```text
Should I irrigate today?
```

Route:

```text
BOTH
```

---

## Spraying

```text
Can I spray pesticides today?
```

Route:

```text
BOTH
```

---

## Hindi / Hinglish

```text
Aaj mausam kaisa hai?
```

```text
Kya aaj sinchai karni chahiye?
```

---

# 🚦 Query Routing

| Query | Route | Information Used |
|---|---|---|
| What's the weather today? | `WEATHER` | Live weather |
| What is my soil type? | `RETRIEVAL` | Agricultural RAG |
| Which crops grow here? | `RETRIEVAL` | Agricultural RAG |
| Should I irrigate today? | `BOTH` | Weather + RAG |
| Can I spray today? | `BOTH` | Weather + RAG |
| Hi | `DIRECT` | LLM |
| General agricultural question | `RETRIEVAL` | Agricultural RAG |

---

# ⚡ Why Deterministic Routing?

A separate LLM could be used to classify every query before deciding which tools to call.

AgroSense instead uses keyword-based routing.

Benefits:

- Lower latency
- Lower inference cost
- Predictable behavior
- Easier debugging
- No additional routing LLM call
- Explicit control over agricultural keywords
- Supports Hindi/Hinglish keyword matching

This allows the LLM to focus primarily on generating the final answer.

---

# 📊 Performance-Oriented Design

AgroSense includes several mechanisms designed to reduce unnecessary work.

### Weather caching

```text
10-minute TTL
```

### Persistent vector database

```text
./chroma_db
```

prevents unnecessary embedding/database rebuilding when the database already exists.

### Model fallback

If the currently selected Hugging Face model fails, the system attempts another configured model.

### Deterministic routing

Weather and retrieval operations are only executed when the query requires them.

---

# 🛡️ Reliability & Safety

AgroSense is designed to prioritize evidence and transparency.

The system prompt instructs the LLM to avoid fabricating:

```text
Government statistics
Soil measurements
Weather forecasts
Fertilizer dosages
Pesticide instructions
Prices
Scheme eligibility
```

When context is incomplete, the assistant should provide what is supported and clearly indicate uncertainty.

For high-impact agricultural decisions, users should verify critical information with qualified agricultural professionals or official local agricultural sources.

---

# 📚 Data & Knowledge Sources

The current RAG pipeline uses:

```text
India_District_Agri_Master_RAG.pdf
```

The system preserves:

```text
Source filename
Page number
Document content
```

for retrieved context.

If more datasets are added, their:

- Source
- License
- Date
- Coverage
- Update frequency

should be documented in this section.

---

# 🧪 Evaluation

A future production evaluation system can measure:

## Retrieval

```text
Recall@K
Precision@K
MRR
NDCG
District relevance
RRF improvement
```

## LLM

```text
Faithfulness
Answer relevance
Context relevance
Citation accuracy
Hallucination rate
```

## System

```text
Average response latency
Retrieval latency
Weather API latency
LLM latency
API failure rate
Model fallback rate
```

---

# 🚧 Future Improvements

## Frontend

- Streaming AI responses
- Chat history
- Persistent conversations
- Mobile-first refinements
- Better weather visualization
- Source/citation cards
- Agricultural dashboards
- Crop cards
- Interactive crop profitability charts
- Better multilingual voice support

## AI / RAG

- Metadata filtering by state/district
- Cross-encoder reranking
- Query rewriting
- Parent-child retrieval
- More government agricultural documents
- Multilingual embeddings
- Better source attribution
- RAG evaluation pipeline

## Agent

- Structured tool calling
- More robust intent classification
- Multi-step workflows
- Conversation memory
- Additional agricultural APIs
- Agricultural scheme lookup
- Market-price lookup
- Crop calendar tools

## ML

A future version can integrate dedicated ML models for:

```text
Crop profitability prediction
Yield prediction
Crop disease detection
Market forecasting
Soil analysis
```

These models can complement the LLM rather than replacing the knowledge-retrieval layer.

## Production

- FastAPI backend
- Docker
- Cloud deployment
- Authentication
- Rate limiting
- Observability
- Centralized logging
- Automated evaluation
- CI/CD
- Production vector database

---

# 🔭 Potential Production Architecture

```text
                         ┌──────────────────────┐
                         │      AgroSense UI    │
                         │ HTML/CSS/JavaScript  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    FastAPI Backend   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Query Router        │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
              Weather             RAG              Direct
                  │                 │                 │
                  ▼                 │                 │
          OpenWeatherMap            │                 │
                                    ▼                 │
                           ┌────────────────┐         │
                           │ BM25 + Vector  │         │
                           └───────┬────────┘         │
                                   ▼                  │
                                  RRF                 │
                                   │                  │
                                   ▼                  │
                              Reranking               │
                                   │                  │
                  ┌────────────────┴──────────────────┘
                  ▼
                         Context Builder
                                │
                                ▼
                         Hugging Face LLM
                                │
                                ▼
                         Grounded Response
                                │
                                ▼
                            AgroSense UI
```

---

# 🎯 Project Goals

AgroSense is designed around three principles:

### 1. Relevant

Use the user's state and district to provide geographically relevant agricultural information.

### 2. Grounded

Use retrieved agricultural knowledge and live weather information rather than relying entirely on generic LLM knowledge.

### 3. Efficient

Avoid unnecessary LLM routing calls and use caching, persistent retrieval, and model fallback strategies.

---

# 💡 What Makes AgroSense Different?

AgroSense is not simply a chatbot with an agricultural prompt.

It combines:

```text
Location Awareness
        +
Deterministic Routing
        +
Hybrid RAG
        +
BM25
        +
Vector Search
        +
Reciprocal Rank Fusion
        +
District Boosting
        +
Live Weather
        +
LLM Generation
        +
Fallback Models
        +
Interactive Web UI
```

The core principle is:

> **Use the right information source for the right agricultural question, then use the LLM to turn that information into a clear and useful response.**

---

# 📌 Project Highlights

```text
✓ Location-aware agriculture assistant
✓ State + district personalization
✓ Hybrid RAG
✓ BM25 lexical retrieval
✓ Semantic vector retrieval
✓ Reciprocal Rank Fusion
✓ District-specific relevance boosting
✓ Persistent ChromaDB
✓ Live OpenWeatherMap integration
✓ 10-minute weather caching
✓ Deterministic query routing
✓ Hugging Face LLM inference
✓ Model fallback strategy
✓ Hindi/Hinglish routing keywords
✓ Think mode
✓ Voice input
✓ Markdown response rendering
✓ DOMPurify response sanitization
✓ Responsive glassmorphism UI
✓ Suggested agricultural queries
```

---

# 🗂️ GitHub Topics

Recommended repository topics:

```text
ai
artificial-intelligence
agriculture
agritech
generative-ai
llm
rag
retrieval-augmented-generation
agentic-ai
hybrid-search
bm25
vector-search
chromadb
langchain
huggingface
openweathermap
python
javascript
html
css
```

---

# 🤝 Contributing

Contributions are welcome.

Create a feature branch:

```bash
git checkout -b feature/your-feature
```

Make your changes and test them.

Then:

```bash
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
```

Open a Pull Request on GitHub.

---

# 📄 License

Choose the license appropriate for your project.

For example:

```text
MIT License
```

If external agricultural datasets or documents are redistributed with the project, their individual licenses and redistribution requirements must also be respected.

---

# 👨‍💻 Author

**Kabir Bhardwaj**

BTech CSE  
AI/ML • Generative AI • RAG • Agentic AI

---
