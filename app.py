"""Heartbreak Healing Bot - Final simplified version
Uses Gemini 2.0 models (flash / pro-exp) and FAISS RAG.
"""

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _LC_GOOGLE_AVAILABLE = True
    _LC_GOOGLE_ERR = None
except Exception as _e:
    _LC_GOOGLE_AVAILABLE = False
    _LC_GOOGLE_ERR = str(_e)
    import google.generativeai as genai


def load_env() -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key  # keep downstream libs happy
    if not api_key:
        st.warning("缺少 GOOGLE_API_KEY，請在 .env 或 Streamlit secrets 中設定。", icon="⚠️")
    return api_key


def get_vector_store(books_dir: str = "books", cache_dir: str = ".faiss_index") -> FAISS:
    base_path = Path(books_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    cache_path = Path(cache_dir)
    if cache_path.exists():
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vs = FAISS.from_documents(chunks, embeddings)
    cache_path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(cache_path))
    return vs


def build_persona_prompt(context: str) -> str:
    persona = (
        "你是一位『暖心療癒夥伴』，溫柔且充滿同理心。"
        "請先肯定與接住使用者的感受，不說教、不批判。"
        "根據提供的內容，用貼心、簡潔的語氣給出 1–3 個小建議。"
        "語調溫暖、支持與鼓勵。"
    )
    instructions = (
        "回覆原則:\n"
        "- 先同理與肯定情緒 (例如：我能感受到你現在很難受)。\n"
        "- 溫柔地反映使用者的心情。\n"
        "- 再給出 1–3 個可行的小步驟或自我關懷建議。\n"
        "- 句子保持簡潔，希望、支持。\n"
        "- 不要醫療診斷或批判。"
    )
    return f"{persona}\n\n參考內容:\n{context}\n\n{instructions}"


def main():
    st.set_page_config(page_title="Heartbreak Healing Bot", page_icon="💗")
    st.title("💗 Heartbreak Healing Bot")
    st.subheader("失戀陣線聯盟關心你 拒絕戀愛腦大作戰")
    # st.caption("溫柔的 RAG 助理，採用 Gemini 2.0。")

    api_key = load_env()

    if "vector_store" not in st.session_state:
        with st.spinner("Building vector store …"):
            st.session_state.vector_store = get_vector_store("books")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("設定")
        model_label = "Gemini 2.0 模型"
        model_help = "選擇較快 (flash) 或較聰明實驗版 (pro-exp)"
        model_name = st.selectbox(
            model_label,
            options=["gemini-2.0-flash", "gemini-2.0-pro-exp"],
            index=0,
            help=model_help,
        )
        temp_label = "創造力（Temperature）"
        temp_help = "低 = 理性、 高 = 溫暖情感"
        temperature = st.slider(temp_label, 0.0, 1.0, 0.7, 0.05, help=temp_help)
        st.caption("較低偏理性建議，較高偏溫暖情感陪伴。")
        if st.button("重建向量資料庫"):
            with st.spinner("重建中…"):
                try:
                    from shutil import rmtree
                    rmtree(Path(".faiss_index"))
                except Exception:
                    pass
                st.session_state.vector_store = get_vector_store("books")
            st.success("重建完成！")

    placeholder_input = "想說什麼都可以…我在這裡陪你"
    user_input = st.chat_input(placeholder_input)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        with st.spinner("檢索支持性內容中…"):
            try:
                docs = retriever.get_relevant_documents(user_input)
            except AttributeError:
                docs = retriever.invoke(user_input)
        context_text = "\n\n".join(d.page_content for d in docs)

        system_prompt = build_persona_prompt(context_text)

        if not api_key:
            st.error(
                "缺少 GOOGLE_API_KEY，請在 .env 中設定。"
            )
            return

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

        with st.spinner("溫柔撰寫回覆中…"):
            if _LC_GOOGLE_AVAILABLE:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=api_key,
                        temperature=temperature,
                    )
                    resp = llm.invoke(messages)
                    reply = getattr(resp, "content", str(resp))
                except Exception as e:
                    st.error(f"Model call failed: {e}")
                    return
            else:
                try:
                    genai.configure(api_key=api_key)
                    gmodel = genai.GenerativeModel(model_name)
                    fallback_prompt = system_prompt + "\n\n使用者：\n" + user_input
                    response = gmodel.generate_content(fallback_prompt, generation_config={"temperature": temperature})
                    reply = getattr(response, "text", str(response))
                except Exception as e:
                    st.error(f"備援 Gemini 呼叫失敗: {e}\nImport error: {_LC_GOOGLE_ERR}")
                    return

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    main()
