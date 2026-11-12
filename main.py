from contextlib import asynccontextmanager
import logging
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ARTIFACTS" / "bidirectional_model.keras"
TOKENIZER_PATH = BASE_DIR / "ARTIFACTS" / "tokenizer (1).pkl"
STATIC_DIR = BASE_DIR / "static"
MAX_SEQUENCE_LENGTH = 50
EMOTION_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def validate_artifacts(model: Any, tokenizer: Any) -> None:
    input_shape = model.input_shape
    output_shape = model.output_shape
    if (
        not isinstance(input_shape, tuple)
        or len(input_shape) != 2
        or input_shape[-1] != MAX_SEQUENCE_LENGTH
    ):
        raise RuntimeError(
            f"Model input shape {input_shape!r} is incompatible with "
            f"sequence length {MAX_SEQUENCE_LENGTH}."
        )
    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or output_shape[-1] != len(EMOTION_LABELS)
    ):
        raise RuntimeError(
            f"Model output shape {output_shape!r} is incompatible with "
            f"{len(EMOTION_LABELS)} emotion labels."
        )
    if not callable(getattr(tokenizer, "texts_to_sequences", None)):
        raise RuntimeError("The tokenizer artifact does not provide texts_to_sequences().")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading sentiment model and tokenizer.")
    model = load_model(MODEL_PATH, compile=False)
    with TOKENIZER_PATH.open("rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    validate_artifacts(model, tokenizer)

    app.state.model = model
    app.state.tokenizer = tokenizer
    logger.info("Sentiment model and tokenizer loaded.")
    try:
        yield
    finally:
        app.state.model = None
        app.state.tokenizer = None


app = FastAPI(title="Moodline Sentiment API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def server_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = (
        getattr(app.state, "model", None) is not None
        and getattr(app.state, "tokenizer", None) is not None
    )
    return HealthResponse(
        status="ready" if model_loaded else "loading",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput) -> PredictionResponse:
    model = getattr(app.state, "model", None)
    tokenizer = getattr(app.state, "tokenizer", None)
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The sentiment model is not ready. Please try again later.",
        )

    cleaned_text = preprocess_text(text_input.text)
    if not cleaned_text:
        raise HTTPException(
            status_code=422,
            detail="Text must contain at least one letter or number.",
        )

    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
    probabilities = np.asarray(model(padded_sequence, training=False))[0]

    if probabilities.shape != (len(EMOTION_LABELS),) or not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError("The sentiment model returned invalid probabilities.")

    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(probability)
        for label, probability in zip(EMOTION_LABELS, probabilities)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=EMOTION_LABELS[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities,
    )
