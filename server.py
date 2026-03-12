from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageEnhance
from io import BytesIO
from openai import OpenAI
import os
import base64
import re
import json

with open("kana_dictionary.json", "r", encoding="utf-8") as f:
    KANA_DICT = json.load(f)

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

def dictionary_candidates_from_readings(readings: list[str]) -> list[list[str]]:
    results = []
    for r in readings:
        if r in KANA_DICT:
            results.append(KANA_DICT[r])
        else:
            results.append([])
    return results


def build_prompt(input_mode: str) -> str:
    common_rule = """
あなたは、くずし字・変体仮名の判読補助AIです。
入力は単独文字の画像です。
必ず JSON のみを返してください。説明文、前置き、コードブロックは不要です。

この課題では、現代ひらがなの見た目に直接飛びつかず、
必ず「骨格 → 元字候補 → 読み候補」の順で推論してください。

推論手順:
1. まず字形の骨格を観察する
   - 縦長 / 横長
   - 主軸の位置
   - ループの有無と位置
   - 払い、折れ、曲がり
   - 上部 / 中央 / 下部の構造
   - 一筆書きの流れ
2. 次に、その骨格に近い草書・崩しの漢字（元字候補）を複数考える
3. その元字候補が、変体仮名として使われる可能性を検討する
4. 最後に、元字候補から読み候補を出す

重要:
- 現代かなの形だけで即断しない
- まず元字候補、そのあと読み候補
- 頻出元字は参考にしてよいが、骨格一致より優先しない
- 骨格が合わないなら、頻出元字でも順位を下げる
- 同じ読みを重複して返さない
- 候補は最大7個
- 弱い候補や自信の低い候補は "不明" でもよい
- 読み候補、元字候補、字形説明の配列長はそろえる
- 正解を1つに決め打ちせず、骨格が近い候補を幅広く残す
- 特に「之・者・於・計・奴・徒・喜・津・遠・与」のような変体仮名由来の可能性を常に意識する

変体仮名で比較的よく現れる元字の参考:
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

ただし、この表は参考情報であり、
骨格が一致しないのに無理に当てはめてはいけません。

キー名は必ず次の英語キーを使ってください:
row_guess
readings
source_kanji
shapes
note

日本語のキー名は使わないでください。

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
手描きなので細部は崩れている可能性があります。
細部の小さな違いより、骨格・主軸・ループ・終筆方向・全体の流れを重視してください。

手描きで特に重視すること:
- 線の本数そのものより骨格
- どこに重心があるか
- 右へ流れるか、下へ落ちるか
- ループが閉じるか、開くか
- 上部 / 中央 / 下部の配置バランス
"""
    else:
        extra = """
これは写真または切り出し画像です。
紙の質感、にじみ、かすれ、背景ノイズ、傾きが含まれる可能性があります。
ノイズやかすれに引きずられず、主要な骨格線を優先してください。

写真で特に重視すること:
- 主軸
- ループ
- 横画の位置
- 主要な払い
- 背景ノイズと骨格線の区別
"""
    return common_rule + "\\n\\n" + extra.strip()


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

        #ここから
normalized = normalize_image(img)
result = call_openai_with_image(normalized, input_mode)

dictionary_hits = dictionary_candidates_from_readings(result["readings"])
result["dictionary_candidates"] = dictionary_hits

return JSONResponse(
    content={
        "success": True,
        "input_mode": input_mode,
        "result": result
    }
)
#ここまで
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"JSON解析エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"認識処理エラー: {str(e)}")
