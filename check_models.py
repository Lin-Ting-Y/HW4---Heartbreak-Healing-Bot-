import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. 載入 .env 裡的 Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ 找不到 Key，請檢查 .env 檔案")
    exit()

print(f"🔑 正在使用 Key: {api_key[:5]}... 查詢可用模型中...\n")

# 2. 設定 Google SDK
genai.configure(api_key=api_key)

# 3. 列出所有模型
try:
    found = False
    for m in genai.list_models():
        # 我們只找可以 "generateContent" (生成文字) 的模型
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ 可用模型: {m.name}")
            found = True
    
    if not found:
        print("⚠️ 連線成功，但沒有找到支援 generateContent 的模型。")

except Exception as e:
    print(f"❌ 查詢失敗，錯誤原因:\n{e}")