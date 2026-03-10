import base64
import json
import os
import traceback
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def extract_text_output(resp: Any) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    try:
        parts = []
        for item in resp.output:
            content = getattr(item, "content", None) or []
            for c in content:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", ""))
        return "\n".join(p for p in parts if p)
    except Exception:
        return ""


@app.get("/")
def root():
    return {"message": "server running"}


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty image")

b64 = base64.b64encode(raw).decode("utf-8")

mime = image.content_type or "image/jpeg"
if mime not in ["image/jpeg", "image/png", "image/webp"]:
    mime = "image/jpeg"


        prompt = """
あなたは、変体仮名・くずし字の候補提示補助です。
入力画像は1文字だけです。
現代かなでの読み候補を最大3件、JSONのみで返してください。
必ず次の形式にしてください。

{
  "candidates": [
    {"reading": "か", "confidence": 0.82, "characterGuess": "可"},
    {"reading": "が", "confidence": 0.10, "characterGuess": "加"},
    {"reading": "や", "confidence": 0.04, "characterGuess": null}
  ]
}
"""

        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime};base64,{b64}"
                        }
                    ]
                }
            ],
        )

        text = extract_text_output(resp).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(status_code=502, detail=f"model returned non-json: {text}")

        data = json.loads(text[start:end + 1])
        return data

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
