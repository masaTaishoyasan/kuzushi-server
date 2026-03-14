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


def build_prompt() -> str:
    return """
あなたは、戸籍・史料中のくずし字・変体仮名の判読支援AIです。
入力は単独文字の画像です。

この段階の目的は最終確定ではありません。
人間が50音順の対応表を優先的に確認できるよう、
「先に見るべき読み候補」を5件前後返してください。

この課題では、
元漢字の一点当てよりも、
「優先的に照合すべき読み候補の順序づけ」
を最優先してください。

【最重要ルール】

・正解を1つに断定しない
・まず読み候補を返す
・元漢字は補助情報として少数だけ添える
・候補集合に正解を含めることを優先する
・候補を狭めすぎない
・現代ひらがなの見た目だけで判断しない
・行分類（か行、さ行等）は主判断に使わない
・珍しい変体仮名も安易に除外しない
・ただし無秩序に広げすぎず、優先順位を付ける

【観察の優先順位】

次の順で観察してください。

1. 運筆の流れ
   - どこから書き始めて、どこへ流れているか
   - 一筆書きの連続性
   - 上から下か、左から右か、回り込みか

2. 骨格
   - 主軸の位置
   - 縦長か横長か
   - 上中下の構造
   - 左右の張り出し
   - 閉じたループか、開いたループか

3. 終筆
   - 右下へ払う
   - 下で巻く
   - 左へ返る
   - 短く止まる

4. 定着した崩し字形との近さ
   - 元漢字の草書・くずしとして広く使われる形か

【強い禁止事項】

・「現代かなで○○っぽく見えるから」という理由だけで決めない
・主軸や運筆が合わない候補を一位にしない
・元漢字の知識だけで無理に読みを決めない
・画像ノイズやかすれを主要骨格と誤認しない

【推論手順】

1. 画像の主要骨格と運筆を短く整理する
2. その骨格に近い読み候補を広く考える
3. ただし、優先して確認すべき順に5件前後へ絞る
4. 各読み候補に対して、参考元漢字を1〜3個だけ添える
5. 各候補について、なぜ先に確認すべきかを短く書く

【出力方針】

・読み候補が主役
・候補数は基本5件
・本当に必要なときだけ最大7件
・各候補の元漢字ヒントは1〜3個まで
・shape_reasons は短く具体的に
・note も短く

【参考となる変体仮名の元字】

あ: 安 阿 愛 悪 亜
い: 以 伊 意 移 異
う: 宇 羽 有 雲
え: 衣 江 恵 要 盈
お: 於 尾 小 意 隠

か: 加 可 嘉 賀 閑 我 歌 家
き: 喜 幾 支 希 起 木 貴
く: 久 九 供 具 倶
け: 計 介 希 遣 氣 祁 気
こ: 己 古 許 故 期 胡 子 興

さ: 左 散 佐 差 斜
し: 之 志 四 春 新 師 思 事 斯
す: 寸 寿 須 春 数 爪
せ: 世 勢 瀬 聲 声
そ: 曾 楚 所 曽 處

た: 多 堂 當 太 当 田
ち: 千 知 遅 地 智 致
つ: 津 徒 都 川
て: 天 帝 亭 手 伝 低 豆 而
と: 徒 止 登 東 度 斗 土 戸 等 砥

な: 奈 那 南 名 菜 難
に: 仁 爾 丹 耳 尼 児 二 而
ぬ: 奴 怒 努 駑
ね: 祢 年 熱 念 根 音 子
の: 乃 能 農 濃 遁 野

は: 者 八 半 波 盤 葉 芳 破 羽 婆
ひ: 比 悲 飛 日 非 肥 避
ふ: 不 布 婦 風
へ: 部 遍 弊 倍 辺
ほ: 保 本 奉 報

ま: 万 末 満 眞 麻 馬 真
み: 美 身 三 見 微 民
む: 武 無 牟 无 舞 夢
め: 女 免 面 馬 綿
も: 毛 母 裳 茂 藻

や: 也 夜 屋 耶
ゆ: 由 遊 湯 揺
よ: 与 餘 余 世 夜 代

ら: 良 羅 等
り: 利 里 李 理 梨
る: 留 流 累 類
れ: 礼 連 麗 禮
ろ: 呂 路 楼 露 樓

わ: 和 王 輪
ゐ: 為 井 居 遺 委
ゑ: 恵 江 衛 慧
を: 遠 乎 越 袁 尾
ん: 无 尓

この表は補助情報です。
この表にあるからといって、骨格が遠い候補を上位にしないでください。

【出力形式】

必ずJSONのみを返してください。
説明文、前置き、コードブロックは禁止です。

{
  "reading_guess": ["候補1","候補2","候補3","候補4","候補5"],
  "source_kanji_hint": [
    ["元字候補1a","元字候補1b"],
    ["元字候補2a","元字候補2b"],
    ["元字候補3a","元字候補3b"],
    ["元字候補4a","元字候補4b"],
    ["元字候補5a","元字候補5b"]
  ],
  "shape_reasons": [
    "理由1",
    "理由2",
    "理由3",
    "理由4",
    "理由5"
  ],
  "note": "優先確認候補を提示"
}
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


def call_openai_with_image(img: Image.Image) -> dict:
    prompt = build_prompt()
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
        result = call_openai_with_image(normalized)

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

