# Heartbreak Healing Bot

一位好友最近歷經失戀。因此寫下這個 Heartbreak Healing Bot——一個懂得先接住情緒、再給出小建議的 AI 夥伴，讓她慢慢回到自己的步調，重新找到力量。

這個專案強調「陪伴」和「可共感的語氣」，結合 RAG（Retrieval Augmented Generation）與 Gemini 2.0，擷取自製的療癒文章與失戀陪伴素材，盡力讓回覆溫暖、真誠。希望未來不只幫助她，也能陪伴更多被失戀傷到的人。

## 功能亮點
- Streamlit 介面，簡潔、可直接部署。
- FAISS 向量資料庫搭配 HuggingFace Embeddings，檢索書籍與療癒文章。
- Gemini 2.0（flash / pro-exp）負責生成回覆，維持溫柔同理語氣。
- 溫暖的預設人格提示，先同理再提供 1–3 個可行的小步驟。

## Setup

1. Create and activate the conda environment:

```powershell
conda create -n aiot-hw4 python=3.11 -y
conda activate aiot-hw4
```

2. Install dependencies:

```powershell
cd "C:\Users\linty\OneDrive\class\AIoT\HW4"
pip install -r requirements.txt
```

3. Configure environment:

- Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

4. Add knowledge files:

- Place `.txt` files in `books/` directory.

## Run

```powershell
streamlit run app.py
```

執行後，畫面左上是「Heartbreak Healing Bot」，底下的副標「失戀陣線聯盟關心你」提醒她永遠有人在意。輸入框適合寫下任何可對話的心情，模型會根據資料庫內容與人格提示回覆。

## Notes
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: Local FAISS built from `books/*.txt`
- LLM: `gemini-2.0-flash` 或 `gemini-2.0-pro-exp`

## 未來方向
- 針對沒有 `books/` 檔案時的提示與引導（避免初次啟動失敗）。
- 更細緻的對話流程與情緒追蹤，例如記錄使用者心情變化。
- 部署到 Streamlit Community Cloud 或其他平台，讓朋友更容易連線使用。