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
    common_rule = """
あなたは、くずし字・変体仮名の判読補助AIです。
入力は単独の文字画像です。
必ずJSONのみを返してください。

推論は次の順序で行ってください。

1. 字形の骨格を観察する
・縦長 / 横長
・主軸
・ループ
・折れ
・払い
・一筆書きの流れ

2. 草書の漢字として近い骨格を考える

3. 変体仮名として使われやすい元字を優先する

4. 元字から読みを推定する

重要:
・現代ひらがなの見た目だけで判断しない
・まず元字候補を考える
・変体仮名として実在しやすい元字を優先する
・候補は最大7個
・不明な場合は "不明"

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

この対応はヒントであり、字形が一致する場合は優先してください。
"""

    if input_mode == "drawing":
        extra = """
これはユーザーが手で描いた文字です。
細部は崩れる可能性があります。

重視するポイント:
・主軸
・ループ位置
・終筆方向
・全体の流れ

細部より骨格を優先してください。
"""
    else:
        extra = """
これは写真から切り出した文字です。
にじみやかすれがある可能性があります。

重視するポイント:
・主要な骨格
・主軸
・ループ
・払い

ノイズより骨格を優先してください。
"""

    return common_rule + "\n" + extra
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
