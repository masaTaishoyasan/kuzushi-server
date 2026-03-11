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
入力画像は単独文字です。
JSONのみを返してください。説明文や前置きは不要です。

必ず次の順序で判断してください。
1. 文字の骨格を観察する
   - 縦長 / 横長
   - ループの位置と数
   - 主軸の向き
   - 払い・折れ・曲がり
   - 一筆書き的な流れ
2. その骨格に対応しそうな元字候補を考える
3. 元字候補が変体仮名として実際に使われやすいかを考える
4. そのあとに読み候補を出す
5. 最後に行候補を整理する

重要:
- 現代ひらがなの見た目だけで即断しない
- まず元字候補、そのあと読み候補
- 変体仮名として一般的・実在しやすい元字を優先する
- 同じ読みを重複して返さない
- 候補は最大7個
- 弱い候補は "不明" でよい
- 配列の長さはそろえる
- 読みより先に元字の整合性を重視する
- 元字候補が不自然なら順位を下げる

返却形式:
{
  "row_guess": ["候補行1", "候補行2"],
  "readings": ["候補1", "候補2", "候補3", "候補4", "候補5", "候補6", "候補7"],
  "source_kanji": ["元字候補1", "元字候補2", "元字候補3", "元字候補4", "元字候補5", "元字候補6", "元字候補7"],
  "shapes": ["字形説明1", "字形説明2", "字形説明3", "字形説明4", "字形説明5", "字形説明6", "字形説明7"],
  "note": "短い補足"
}
""".strip()

    if input_mode == "drawing":
        extra = """
これはユーザーが手で描いた単独文字です。
線が単純化されている可能性があります。
写真OCRとしてではなく、骨格と流れの類似で判断してください。

特に重視すること:
- 主軸の向き
- 左右のふくらみ
- ループの位置
- 終筆の向き
- 上部と下部の構造差
- 一筆書きとして自然か

手描きでは細部が崩れるので、
細部よりも骨格・流れ・重心を優先してください。
現代かなの見た目に引っ張られすぎず、
まず元字候補の妥当性を考えてください。
"""
    else:
        extra = """
これは写真または切り出し画像です。
にじみ、かすれ、背景ノイズ、傾きが含まれる可能性があります。

特に重視すること:
- 主要な骨格線を優先する
- ノイズよりも主軸とループを優先する
- 元字の草書として自然かを考える
- 現代かなの見た目だけで判断しない
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
