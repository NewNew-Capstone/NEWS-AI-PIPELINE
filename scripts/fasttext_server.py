import os

import fasttext
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
_model: fasttext.FastText._FastText | None = None


class EmbedRequest(BaseModel):
    words: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


@app.on_event("startup")
def load() -> None:
    global _model
    path = os.getenv("FASTTEXT_MODEL_PATH", "analysis_bc/data/cc.ko.300.bin")
    _model = fasttext.load_model(path)


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    vectors = [
        _model.get_word_vector(word).astype(float).tolist()
        for word in req.words
    ]
    return EmbedResponse(vectors=vectors)
