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
    allow_origins=["*"],
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

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    plain = re.search(r"(\{.*\})", text, re.DOTALL)
    if plain:
        return json.loads(plain.group(1))

    raise ValueError("JSONを抽出できませんでした。")


def normalize_image(img: Image.Image) -> Image.Image:
    """
    認識前の軽い前処理
    - EXIF補正
    - グレースケール
    - 自動コントラスト
    - 余白付き正方形化
    - リサイズ
    - 二値化
    """
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    w, h = img.size
    side = max(w, h) + 40
    canvas = Image.new("L", (side, side), color=255)

    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(img, (x, y))

    canvas = canvas.resize((512, 512))
    canvas = canvas.point(lambda x: 0 if x < 140 else 255)

    return canvas


def build_prompt(input_mode: str) -> str:
    common_rule = """
あなたは、くずし字・変体仮名の判読補助AIです。
入力は単独の文字画像です。
必ずJSONのみを返してください。説明文・前置き・コードブロックは不要です。

推論は必ず次の順序で行ってください。

1. 字形の骨格を観察する
- 縦長 / 横長
- 主軸
- ループ
- 折れ
- 払い
- 一筆書きの流れ

2. 草書の漢字として近い骨格を考える

3. 変体仮名として使われやすい元字を優先する

4. 元字から読みを推定する

重要:
- 現代ひらがなの見た目だけで即断しない
- まず元字候補を考え、そのあと読み候補を出す
- 変体仮名として実在しやすい元字を優先する
- 同じ読みを重複して返さない
- 候補は最大7個
- 弱い候補は "不明" でよい
- 配列の長さはそろえる
- 元字候補が不自然なら順位を下げる

変体仮名で頻出する元字の例:
あ: 安
い: 以
う: 宇
え: 衣
お: 於

か: 加
き: 喜
く: 久
け: 計
こ: 己

さ: 左
し: 之
す: 寸
せ: 世
そ: 曾

た: 多
ち: 千
つ: 津
て: 天
と: 徒

な: 奈
に: 仁
ぬ: 奴
ね: 祢
の: 乃

は: 者
ひ: 比
ふ: 不
へ: 部
ほ: 保

ま: 万
み: 美
む: 武
め: 女
も: 毛

や: 也
ゆ: 由
よ: 与

ら: 良
り: 利
る: 留
れ: 礼
ろ: 呂

わ: 和
を: 遠
ん: 无

キー名は必ず次の英語キーを使ってください:
row_guess
readings
source_kanji
shapes
note

日本語のキー名（例: 元字候補、推定される読み、備考）は使わないでください。

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
これはユーザーが手で描いた文字です。
細部は崩れる可能性があります。

重視するポイント:
- 主軸
- ループ位置
- 終筆方向
- 全体の流れ

細部より骨格を優先してください。
"""
    else:
        extra = """
これは写真または切り出し画像です。
にじみやかすれがある可能性があります。

重視するポイント:
- 主要な骨格
- 主軸
- ループ
- 払い

ノイズより骨格を優先してください。
"""

    return common_rule + "\n\n" + extra.strip()


def sanitize_result(result: dict) -> dict:
    """
    AIの返答揺れを吸収して、必ずSwift側が読める形に整える
    """
    row_guess = result.get("row_guess", [])
    readings = result.get("readings", [])
    source_kanji = result.get("source_kanji", [])
    shapes = result.get("shapes", [])
    note = result.get("note", "")

    if not isinstance(row_guess, list):
        row_guess = []
    if not isinstance(readings, list):
        readings = []
    if not isinstance(source_kanji, list):
        source_kanji = []
    if not isinstance(shapes, list):
        shapes = []
    if not isinstance(note, str):
        note = str(note)

    max_len = max(len(readings), len(source_kanji), len(shapes), 1)

    def pad_list(lst, fill="不明"):
        lst = [str(x) for x in lst]
        if len(lst) < max_len:
            lst += [fill] * (max_len - len(lst))
        return lst[:max_len]

    readings = pad_list(readings, "不明")
    source_kanji = pad_list(source_kanji, "不明")
    shapes = pad_list(shapes, "不明")

    row_guess = [str(x) for x in row_guess][:2]

    return {
        "row_guess": row_guess,
        "readings": readings,
        "source_kanji": source_kanji,
        "shapes": shapes,
        "note": note
    }


def call_openai_with_image(img: Image.Image, input_mode: str) -> dict:
    prompt = build_prompt(input_mode)
    b64 = image_to_base64(img)

    response = client.responses.create(
        model="gpt-5-mini",
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
    parsed = extract_json_block(text)
    return sanitize_result(parsed)

# =========================
# ルート
# =========================

@app.get("/")
def root():
    return {"message": "KuzushiReader API is running"}


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    input_mode: str = Form("photo")
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
