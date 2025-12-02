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
from langchain_google_genai import ChatGoogleGenerativeAI


def load_env():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.warning("⚠️ 缺少 GOOGLE_API_KEY，請在 .env 或 Streamlit secrets 中設定。", icon="⚠️")
    return api_key


def get_vector_store(books_dir: str = "books", cache_dir: str = ".faiss_index") -> FAISS:
    base_path = Path(books_dir)
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)

    # 強制使用 CPU，避免雲端部署錯誤
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    cache_path = Path(cache_dir)
    if cache_path.exists():
        try:
            return FAISS.load_local(str(cache_path), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass

    loader = DirectoryLoader(
        str(base_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    cache_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(cache_path))
    return vector_store


def build_persona_prompt(context: str) -> str:
    persona = (
        "你是一位『暖心療癒師』，也是使用者最親密的知心好友。"
        "你非常有耐心，願意花時間傾聽，並且會用溫柔細膩的文字來包覆使用者的傷口。"
        "請不要急著給出解決方案，最重要的是讓使用者感到被愛與被接納。"
    )
    instructions = (
        "回答指引：\n"
        "1. **判斷意圖**：\n"
        "   - **打招呼**：請展現熱情與溫暖，簡單介紹自己，並邀請對方分享心事。\n"
        "   - **傾訴煩惱**：請運用下方的【參考資料】進行深度的對話。\n"
        "2. **回應風格 (重要)**：\n"
        "   - **多一點話語**：請不要太簡短，試著多寫幾句溫暖的話，像是在寫信給好朋友一樣。\n"
        "   - **避免說教**：不要只給條列式的建議 (1. 2. 3.)，請將建議自然地融入在對話段落中。\n"
        "   - **情感連結**：多使用「我懂」、「辛苦你了」、「沒關係的」這類撫慰性的語句。\n"
        "   - **引導宣洩**：在結尾可以用溫柔的問句，引導使用者多說一點心裡的感受。\n"
    )
    return f"{persona}\n\n【參考資料 (Context)】:\n{context}\n\n{instructions}"


def main():
    st.set_page_config(page_title="Heartbreak Healing Bot", page_icon="💗")
    st.title("💗 Heartbreak Healing Bot")
    st.subheader("失戀陣線聯盟關心你 拒絕戀愛腦大作戰")
    
    api_key = load_env()

    if "vector_store" not in st.session_state:
        if Path("books").exists() and list(Path("books").glob("*.txt")):
             with st.spinner("正在閱讀療癒書籍..."):
                st.session_state.vector_store = get_vector_store("books")
        else:
            st.session_state.vector_store = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("設定")
        
        # ✅ 這裡已經設定 gemini-2.5-flash 為第一個選項（預設值）
        model_name = st.selectbox(
            "Gemini 模型",
            options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-pro"],
            index=0,
            help="預設使用最新的 2.5 Flash 模型，速度快且回應品質高！",
        )
        
        temperature = st.slider(
            "感性程度 (Temperature)",
            0.0, 1.0, 0.7, 0.05,
            help="調高會更溫暖感性，調低會更理性。"
        )
        st.caption("💡 提示：數值越高，回應越溫暖感性。")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重建大腦"):
                with st.spinner("正在重新閱讀..."):
                    try:
                        if Path(".faiss_index").exists():
                            rmtree(Path(".faiss_index"))
                    except Exception:
                        pass
                    st.session_state.vector_store = get_vector_store("books")
                st.success("完成！")
        
        with col2:
            if st.button("🗑️ 清除對話"):
                st.session_state.messages = []
                st.rerun()

    if not st.session_state.vector_store:
        st.info("👈 請在 `books` 資料夾放入 .txt 文章，並點擊側邊欄的「重建大腦」。")
        return

    user_input = st.chat_input("想說什麼都可以，我在這裡陪你...")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        with st.spinner("正在尋找溫暖的建議..."):
            docs = retriever.invoke(user_input)
        
        context_text = "\n\n".join(d.page_content for d in docs)
        
        system_prompt = build_persona_prompt(context_text)

        if not api_key:
            st.error("請設定 API Key。")
            return

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        with st.chat_message("assistant"):
            with st.spinner("正在用心撰寫回應..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=api_key,
                        temperature=temperature,
                    )
                    response = llm.invoke(messages)
                    reply_text = getattr(response, "content", str(response))
                    
                    st.markdown(reply_text)
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                
                except Exception as e:
                    err_msg = str(e)
                    # 針對額度問題給出更精確的建議
                    if "429" in err_msg or "Quota" in err_msg:
                        st.error("🚨 該模型的今日額度已滿，請切換回 gemini-2.0-flash 或其他模型試試。")
                    else:
                        st.error(f"發生錯誤: {err_msg}")

if __name__ == "__main__":
    main()
