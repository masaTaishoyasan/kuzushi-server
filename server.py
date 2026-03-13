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
あなたは、くずし字・変体仮名の候補生成AIです。
入力は単独文字の画像です。
この課題では、正解を1つに当てることよりも、候補集合の中に正解を含めることを優先してください。

必ず JSON のみを返してください。
説明文、前置き、コードブロックは不要です。

この課題の最重要方針:
- 正解を1つに決め打ちしない
- 骨格が近い候補は広く残す
- 珍しい変体仮名や頻度の低い元字も、字形が近ければ落とさない
- 頻出元字だけに引っ張られない
- 厳密な一点当てよりも、候補集合に正解を含めることを優先する

推論手順:
1. まず字形の骨格を観察する
   - 縦長 / 横長
   - 主軸の位置
   - ループの有無と位置
   - 払い、折れ、曲がり
   - 上部 / 中央 / 下部の構造
   - 一筆書きの流れ
2. 次に、骨格が近い読み候補を広く考える
3. その後、各読み候補に対応しうる元字候補を広く考える
4. 候補を狭めすぎずに返す

重要:
- 現代かなの見た目だけで即断しない
- 元字候補と読み候補は、骨格が近いものを広めに残す
- 同じ読みは重複して返さない
- 候補は最大8個まで出してよい
- 弱い候補は "不明" でもよい
- ただし候補を狭めすぎない
- 読み候補、元字候補、字形説明の配列長はそろえる
- 特に、戸籍や古文書で出やすい変体仮名を落とさない
- AIが最も自信のある1個に寄せるより、2位3位4位の候補を広く残すこと

参考となる変体仮名の元字例:
あ: 安 阿 愛 悪 亜
い: 以 伊 意 移 異
う: 宇 羽 有 雲
え: 衣 江 恵 要 盈
お: 於 尾 小 意 隠

か: 加 可 嘉 賀 閑 我 歌 家
き: 喜 幾 支 希 起 木 貴
く: 久 九 供 具 倶
け: 計 介 希 遣 氣 祁 気
こ: 己 古 許 故

さ: 左 散 佐
し: 之 志 四 春 新
す: 寸 寿 須 春
せ: 世 勢 瀬 聲
そ: 曾 楚 所 曽 處

た: 多 堂 當 太
ち: 千 知 遅 地
つ: 津 徒 都 川
て: 天 帝 亭 手
と: 徒 止 登 東 度 斗 土

な: 奈 那 南 名 菜
に: 仁 爾 丹 耳
ぬ: 奴 怒 努 駑
ね: 祢 年 熱 念 根 音 子
の: 乃 能 農

は: 者 八 半 波 盤 葉
ひ: 比 悲 飛 日 非
ふ: 不 布 婦 風
へ: 部 遍 弊 倍
ほ: 保 本 奉 報

ま: 万 末 満 眞 麻
み: 美 身 三 見 微
む: 武 無 牟 无
め: 女 免 面 馬
も: 毛 母 裳 茂

や: 也 夜 屋 耶
ゆ: 由 遊 湯
よ: 与 餘 余 世 夜

ら: 良 羅 等
り: 利 里 李 理 梨
る: 留 流 累 類
れ: 礼 連 麗 禮
ろ: 呂 路 楼

わ: 和 王 輪
ゐ: 為 井 居
ゑ: 恵 江 衛 慧
を: 遠 乎 越 袁
ん: 无 无二 尓

この表は候補を広げるための参考情報であり、
骨格と全く合わないものを無理に当てはめてはいけません。
ただし、字形が少しでも近ければ安易に捨てないでください。

キー名は必ず次の英語キーを使ってください:
row_guess
readings
source_kanji
shapes
note

日本語のキー名は使わないでください。

返却形式:
{
  "row_guess": ["候補行1", "候補行2", "候補行3"],
  "readings": ["候補1", "候補2", "候補3", "候補4", "候補5", "候補6", "候補7", "候補8"],
  "source_kanji": ["元字候補1", "元字候補2", "元字候補3", "元字候補4", "元字候補5", "元字候補6", "元字候補7", "元字候補8"],
  "shapes": ["字形説明1", "字形説明2", "字形説明3", "字形説明4", "字形説明5", "字形説明6", "字形説明7", "字形説明8"],
  "note": "候補を広めに取った短い補足"
}
""".strip()

    if input_mode == "drawing":
        extra = """
これはユーザーが手で描いた単独文字です。
手描きなので細部は崩れている可能性があります。
細部の正確さより、骨格・主軸・ループ・終筆方向・全体の流れを重視してください。

手描きで特に重視すること:
- 線の本数を数えすぎない
- 形の大づかみを優先する
- 少し似ている候補も残す
- 閉じたループ、開いたループ、右下への流れ、左払いを重視する
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
- 少しでも似ている候補を安易に捨てない
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

    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"JSON解析エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"認識処理エラー: {str(e)}")

