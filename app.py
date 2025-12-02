import os
from pathlib import Path
from shutil import rmtree
import streamlit as st
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_LC_GOOGLE = True
    _LC_GOOGLE_ERR = None
except Exception as _e:
    _HAS_LC_GOOGLE = False
    _LC_GOOGLE_ERR = str(_e)
    ChatGoogleGenerativeAI = None  # type: ignore
    import google.generativeai as genai

# 1. 環境變數載入與檢查
def load_env() -> str:
    load_dotenv()
    # 優先從環境變數讀取，其次從 Streamlit Secrets (雲端部署用)
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.warning("⚠️ 缺少 GOOGLE_API_KEY，請在 .env 或 Streamlit secrets 中設定。", icon="⚠️")
    return api_key

# 2. 建立向量資料庫 (強制使用 CPU 版 HuggingFace，穩定且免費)
def get_vector_store(books_dir: str = "books", cache_dir: str = ".faiss_index") -> FAISS:
    base_path = Path(books_dir)
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)

    # 嘗試讀取快取
    cache_path = Path(cache_dir)
    if cache_path.exists():
        try:
            # 強制指定 device="cpu"，避免在雲端找不到 GPU 而報錯
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            return FAISS.load_local(str(cache_path), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass # 讀取失敗就重新建立

    # 讀取書籍檔案
    loader = DirectoryLoader(
        str(base_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, # 確保中文正常
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        return None

    # 切分與向量化
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # 建立新索引
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 儲存快取
    cache_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(cache_path))
    return vector_store

# 3. 建立 AI 人設 Prompt
def build_persona_prompt(context: str) -> str:
    persona = (
        "你是一位『暖心療癒師』— 一位溫暖、善於傾聽的好朋友。"
        "你的主要任務是陪伴剛失戀或心情低落的使用者，但請根據使用者的對話內容調整回應。"
    )
    instructions = (
        "回答指引：\n"
        "1. **判斷意圖**：\n"
        "   - **如果是打招呼**（如「你好」、「早安」）：請親切回應並簡單介紹自己，**請勿**預設對方已經失戀。\n"
        "   - **如果是傾訴煩惱**：才開始運用下方的【參考資料】進行同理與建議。\n"
        "2. **回應原則**：\n"
        "   - 先肯定並接納使用者的情緒。\n"
        "   - 引用參考資料中的建議時，請自然融入對話。\n"
        "   - 保持簡潔、溫柔且帶有希望。\n"
    )
    return f"{persona}\n\n【參考資料 (Context)】:\n{context}\n\n{instructions}"


def main():
    st.set_page_config(page_title="暖心療癒 Agent", page_icon="❤️‍🩹")
    st.title("Heartbreak Healing Bot")
    st.subheader("失戀陣線聯盟關心你 拒絕戀愛腦大作戰")

    api_key = load_env()

    # 初始化資料庫
    if "vector_store" not in st.session_state:
        # 如果 books 資料夾存在且有檔案，才建立
        if Path("books").exists() and list(Path("books").glob("*.txt")):
             with st.spinner("正在閱讀療癒書籍..."):
                st.session_state.vector_store = get_vector_store("books")
        else:
            st.session_state.vector_store = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 側邊欄控制 ---
    with st.sidebar:
        st.header("設定")
        
        # 模型選擇 (包含你帳號可用的 2.0 版本)
        model_name = st.selectbox(
            "Gemini 模型",
            options=["gemini-2.0-flash", "gemini-2.0-pro-exp", "gemini-1.5-pro"],
            index=0,
            help="Flash 速度快，Pro 邏輯強。",
        )
        
        temperature = st.slider(
            "感性程度 (Temperature)",
            0.0, 1.0, 0.7, 0.05,
            help="調高會更溫暖感性，調低會更理性。"
        )
        st.caption("💡 提示：數值越高，回應越溫暖感性。")
        
        st.divider()
        if st.button("重建知識庫 (Rebuild)"):
            with st.spinner("正在重新閱讀並整理記憶..."):
                try:
                    if Path(".faiss_index").exists():
                        rmtree(Path(".faiss_index"))
                except Exception:
                    pass
                st.session_state.vector_store = get_vector_store("books")
            st.success("知識庫更新完成！")

    # 檢查資料庫狀態
    if not st.session_state.vector_store:
        st.info("👈 請在 `books` 資料夾放入 .txt 文章，並點擊側邊欄的「重建知識庫」。")
        return

    # --- 聊天視窗 ---
    user_input = st.chat_input("想說什麼都可以，我在這裡陪你...")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 檢索
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        with st.spinner("正在尋找溫暖的建議..."):
            docs = retriever.invoke(user_input)
        
        context_text = "\n\n".join(d.page_content for d in docs)
        
        # 整理資料來源 (去除重複)
        sources = sorted(set(
            (d.metadata.get("source") or "未知來源").split("\\")[-1].split("/")[-1] 
            for d in docs
        ))

        system_prompt = build_persona_prompt(context_text)

        if not api_key:
            st.error("請設定 API Key。")
            return

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        # 生成
        with st.chat_message("assistant"):
            with st.spinner("正在用心撰寫回應..."):
                try:
                    if _HAS_LC_GOOGLE:
                        llm = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=api_key,
                            temperature=temperature,
                        )
                        response = llm.invoke(messages)
                        reply_text = getattr(response, "content", str(response))
                    else:
                        genai.configure(api_key=api_key)
                        gmodel = genai.GenerativeModel(model_name)
                        fallback_prompt = system_prompt + "\n\n使用者：\n" + user_input
                        response = gmodel.generate_content(
                            fallback_prompt,
                            generation_config={"temperature": temperature},
                        )
                        reply_text = getattr(response, "text", str(response))
                    
                    # 顯示資料來源
                    if sources:
                        reply_text += "\n\n---\n📚 **參考資料**: " + ", ".join(sources)
                    
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                
                except Exception as e:
                    err_msg = str(e)
                    if not _HAS_LC_GOOGLE and _LC_GOOGLE_ERR:
                        err_msg += f"\nImport error: {_LC_GOOGLE_ERR}"
                    st.error(f"發生錯誤: {err_msg}")

if __name__ == "__main__":
    main()