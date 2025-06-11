import os
import time
import re
import google.generativeai as genai
from typing import Optional

# Gemini APIの設定
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("環境変数 'GOOGLE_API_KEY' が設定されていません")

genai.configure(api_key=GOOGLE_API_KEY)

# 生成パラメータの設定
generation_config = {
    "temperature": 0.7,           # 創造性の度合い (0.0-1.0)
    "top_p": 0.8,                # 出力の多様性
    "top_k": 40,                 # 考慮する次のトークンの数
    "max_output_tokens": 2048,    # 最大出力トークン数
}

# 安全性の設定
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

print("利用可能なモデル:")
for m in genai.list_models():
    if 'gemini' in m.name.lower():
        print(f"- {m.name}")

# モデルの初期化
try:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",  # パラメータ名をmodel_nameに修正
        generation_config=generation_config,
        safety_settings=safety_settings
    )
except Exception as e:
    print(f"モデル初期化エラー: {e}")
    print("利用可能なモデルの一覧を確認してください。")
    raise

def get_retry_delay(error_message: str) -> int:
    """エラーメッセージから待機時間を抽出"""
    if match := re.search(r'retry_delay.*?seconds: (\d+)', str(error_message)):
        return int(match.group(1))
    return 60  # デフォルトの待機時間

def verify_api_key() -> bool:
    """APIキーの有効性を確認"""
    try:
        # 簡単なテストプロンプトで確認
        test_prompt = "Hello"
        response = model.generate_content(test_prompt)
        return True
    except Exception as e:
        print(f"APIキー検証エラー: {e}")
        return False

def generate_with_retry(prompt: str, max_retries: int = 5, base_delay: int = 120) -> Optional[str]:
    """リトライ機能付きでGemini APIを実行"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                contents=prompt,
                stream=False  # ストリーミングを無効化
            )
            
            if response.prompt_feedback.block_reason:
                print(f"ブロックされました: {response.prompt_feedback.block_reason}")
                return None
                
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:  # レート制限エラー
                delay = max(get_retry_delay(error_msg), base_delay * (attempt + 1))
                print(f"\nAPIレート制限。{delay}秒待機します... (試行 {attempt + 1}/{max_retries})")
                print("※ 無料枠の制限に達した可能性があります。")
                time.sleep(delay)
                continue
            else:
                print(f"エラー: {e}")
                return None
    print("\n最大リトライ回数に達しました。")
    print("24時間後に再試行してください。")
    return None

def create_watering_prompt(plant_name: str) -> str:
    """植物の水やり情報を問い合わせるプロンプトを生成"""
    return f"""
あなたは植物の専門家です。以下の形式で回答してください。

{plant_name}の水やり頻度について、各季節ごとに具体的に説明してください。
{plant_name}の種類や生育環境によって異なる場合は、一般的な指針を示してください。

出力形式:
- 春・夏期の水やり方法: [具体的な頻度、土の状態、水やりのタイミング]
- 秋期の水やり方法: [具体的な頻度、土の状態、水やりのタイミング]
- 冬期の水やり方法: [具体的な頻度、土の状態、水やりのタイミング]
- 注意事項: [特に気をつけるべきポイント]
"""

def get_watering_info(plant_name: str) -> Optional[str]:
    """指定された植物の水やり情報を取得"""
    prompt = create_watering_prompt(plant_name)
    try:
        if result := generate_with_retry(prompt):
            print(f"\n=== {plant_name}の水やり情報 ===")
            return result
    except KeyboardInterrupt:
        print("\n処理を中断しました。")
    return None

# テスト実行部分
if __name__ == '__main__':
    # テストする植物名のリスト
    test_plants = ["エケベリア", "サボテン", "アロエ"]
    
    for plant in test_plants:
        if info := get_watering_info(plant):
            print(info)
            print("=" * 80 + "\n")
        time.sleep(2)  # API制限を考慮して待機
