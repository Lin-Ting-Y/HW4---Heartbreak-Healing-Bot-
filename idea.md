# Project Name: Heartbreak Healing Bot (暖心療癒 Agent)

## 1. Project Background & Motivation (專案背景與動機)
**The Problem:**
My inspiration comes from a close friend who recently went through a painful breakup. While friends want to support them, we cannot be available 24/7 to listen to their repetitive sorrow. Furthermore, general AI models (like standard ChatGPT) often give generic, overly logical, or "preachy" advice that lacks emotional warmth.

**The Solution:**
I want to build a **RAG-based (Retrieval-Augmented Generation) Chatbot** specifically designed for emotional healing.
- It acts not as a doctor, but as a **supportive, non-judgmental best friend**.
- It uses a specific knowledge base (articles on psychology, letting go, and self-worth) to provide grounded wisdom, not just hallucinations.
- It is available anytime to listen and gently guide the user out of the "loop" of sadness.

## 2. Technical Workflow (系統流程)
The system will be built using **Streamlit** (UI), **LangChain** (Logic), **Google Gemini** (LLM), and **HuggingFace** (Embeddings).

**Step 1: Knowledge Ingestion (知識庫建立)**
- **Input:** A collection of `.txt` files in a `books/` directory (content covers: breakup recovery, self-care, emotional validation).
- **Process:**
  1. Load documents using `DirectoryLoader`.
  2. Split text into chunks (e.g., size=500, overlap=100) to maintain context.
  3. Convert chunks into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` (Local & Free).
  4. Store vectors in a local `FAISS` vector database.

**Step 2: Retrieval & Augmentation (檢索與增強)**
- **User Action:** User types a message (e.g., "I miss him so much, I can't sleep.").
- **Retrieval:** The system searches the `FAISS` database for the top 3-4 text chunks most relevant to the user's emotion.
- **Prompt Engineering:**
  - Combine: `User Query` + `Retrieved Context` + `System Persona`.
  - **Persona Definition:** "You are a warm, empathetic listener. Validate feelings first, then offer gentle advice based on the context. Do not lecture."

**Step 3: Generation & Response (生成與回應)**
- **Model:** Send the constructed prompt to **Google Gemini 1.5 Flash**.
- **Output:** Stream the warm, context-aware response back to the Streamlit UI.
- **UI Experience:** Display chat history to simulate a continuous conversation.

## 3. Key Features (核心功能)
1.  **Custom Knowledge Base:** Only answers based on selected healing literature (reduces generic AI responses).
2.  **Empathetic Persona:** The system prompt enforces a "warm friend" tone.
3.  **Memory:** The chat interface retains the current session's conversation history.