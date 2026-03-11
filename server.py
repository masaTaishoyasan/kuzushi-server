from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
from io import BytesIO
from openai import OpenAI
import os
import json
import base64
import re

# =========================
# 設定
# =========================

app = FastAPI(title="KuzushiReader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 本番では必要に応じて絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY が設定されていません。")

client = OpenAI(api_key=api_key)

# =========================
# ユーティリティ
# =========================

def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_json_block(text: str) -> dict:
    """
    応答文字列からJSONをできるだけ安全に抜き出す
    """
    text = text.strip()

    # ```json ... ``` を優先して抽出
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # 普通の {...} を抽出
    plain = re.search(r"(\{.*\})", text, re.DOTALL)
    if plain:
        return json.loads(plain.group(1))

    raise ValueError("JSONを抽出できませんでした。")


#ここから
def normalize_image(img: Image.Image) -> Image.Image:
    """
    認識前の軽い前処理
    - EXIF補正
    - グレースケール
    - 余白付き正方形化
    - リサイズ
    - コントラスト強化
    - 二値化
    """

    img = ImageOps.exif_transpose(img)

    # グレースケール
    img = img.convert("L")

    # コントラスト自動調整
    img = ImageOps.autocontrast(img)

    # 元画像の周囲に少し余白を加える
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 正方形キャンバスへ中央配置
    w, h = img.size
    side = max(w, h) + 40
    canvas = Image.new("L", (side, side), color=255)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(img, (x, y))

    # サイズ統一
    canvas = canvas.resize((512, 512))

    # 二値化（重要）
    canvas = canvas.point(lambda x: 0 if x < 140 else 255)

    return canvas
    #ここまで


#ここから
def build_prompt(input_mode: str) -> str:
    """
    input_mode:
      - photo
      - drawing
    """
    common_rule = """
あなたは、くずし字・変体仮名の判読補助AIです。
画像から単独文字を推定し、候補を返してください。
不明な場合は不明と書いてください。
JSONのみを返してください。説明文は不要です。

候補は最大7個まで返してください。
各候補について、次の3つを対応する順番で返してください。
1. 読み候補
2. 元になった漢字（元字候補）
3. 字形の簡単な説明

現代かなの見た目だけでなく、変体仮名・草書・くずし字として自然な元字を優先してください。
元字候補は、変体仮名として実際に使われやすいものを優先してください。
配列の長さはそろえてください。

返却形式:
{
  "readings": ["候補1", "候補2", "候補3", "候補4", "候補5", "候補6", "候補7"],
  "source_kanji": ["元字候補1", "元字候補2", "元字候補3", "元字候補4", "元字候補5", "元字候補6", "元字候補7"],
  "shapes": ["字形説明1", "字形説明2", "字形説明3", "字形説明4", "字形説明5", "字形説明6", "字形説明7"],
  "note": "短い補足"
}
""".strip()

    if input_mode == "drawing":
        extra = """
これはユーザーが指やペンで手描きした単独文字です。
写真の文字認識ではなく、形の類似から推定してください。
線が単純化されている可能性があります。
変体仮名・くずし字としてあり得る候補を優先してください。
現代ひらがなの見た目に引っ張られすぎないでください。
"""
    else:
        extra = """
これは写真または切り出し画像です。
紙の質感、かすれ、にじみ、背景ノイズが含まれる可能性があります。
変体仮名・くずし字として自然な候補を優先してください。
現代ひらがなの見た目だけでなく、元字となる漢字の草書体も考慮してください。
"""

    return common_rule + "\n\n" + extra.strip()
    #ここまで


def call_openai_with_image(img: Image.Image, input_mode: str) -> dict:
    prompt = build_prompt(input_mode)
    b64 = image_to_base64(img)

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}"
                    }
                ]
            }
        ]
    )

    text = response.output_text
    return extract_json_block(text)

# =========================
# ルート
# =========================

@app.get("/")
def root():
    return {"message": "KuzushiReader API is running"}


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    input_mode: str = Form("photo")  # "photo" or "drawing"
):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))

        normalized = normalize_image(img)
        result = call_openai_with_image(normalized, input_mode)

        return JSONResponse(
            content={
                "success": True,
                "input_mode": input_mode,
                "result": result
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"JSON解析エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"認識処理エラー: {str(e)}")
