import logging
import re

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import main  # your existing backend: loads PDF, Chroma, BM25 ONCE at startup
from locations import LOCATIONS

log = logging.getLogger("agrosense")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

NAME_OK = re.compile(r"^[A-Za-z0-9 .,'()/&-]{1,80}$")


class ChatIn(BaseModel):
    query: str
    state: str
    district: str
    think: bool = False


def bad(msg, code=400):
    return JSONResponse({"error": msg}, status_code=code)


def valid_location(state, district):
    return state in LOCATIONS and bool(NAME_OK.match(district or ""))


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/locations")
def locations():
    return LOCATIONS


# Plain `def` endpoints run in FastAPI's threadpool, so slow LLM calls don't block the server.
@app.post("/api/chat")
def chat(body: ChatIn):
    q = body.query.strip()
    if not q:
        return bad("Please type a question.")
    if len(q) > 2000:
        return bad("Your question is too long. Please shorten it.")
    if not valid_location(body.state, body.district.strip()):
        return bad("Please choose a valid state and district.")
    try:
        result = main.process_query(q, body.district.strip(), body.state, body.think)
    except Exception:
        log.exception("agent failure")
        return bad("Something went wrong while answering. Please try again.", 502)
    if not result.get("answer"):
        return bad("The AI service is busy right now. Please try again in a moment.", 503)
    return {"answer": result["answer"], "route": result["route"]}


@app.get("/api/weather")
def weather(state: str, district: str):
    if not valid_location(state, district.strip()):
        return bad("Invalid location.")
    try:
        data = main.get_weather_data(district.strip(), state)  # your weather logic
    except Exception:
        log.exception("weather failure")
        data = None
    if not data:
        return bad("Weather is unavailable right now.", 503)
    return data
