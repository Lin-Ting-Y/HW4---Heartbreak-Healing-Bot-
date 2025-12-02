# Heartbreak Healing Bot Update Report

**Date:** 2025-11-29

## 專案初衷
好友最近經歷失戀，夜深時常被不安驚醒，也不知道可以向誰傾訴。為了讓她在孤單時至少能有溫柔的陪伴，我決定打造 Heartbreak Healing Bot。靈感主要來自蔡炎龍老師的 GitHub 範例《demo6 RAG 打造向量、系統、打造心靈處方籤機器人》，我沿用其中的結構觀念與實作技巧，再加入自己整理的療癒文本，希望讓這個小小的 AI 夥伴成為她的臨時避風港。

## Overview
- Conducted targeted review and made incremental updates to `app.py` for the Heartbreak Healing Bot Streamlit application.
- All UI copy is now fixed in Traditional Chinese; persona prompt and helper texts were aligned.
- Recorded execution attempts for later verification.
- Curated RAG knowledge base from自製失戀療癒文章與朋友的心情筆記，提升回覆的共感度。

## Review Findings
- Identified risk: rebuilding the FAISS cache with no source documents raises a `ValueError`; recommend guarding against empty `books/` directories，讓首次啟動更順暢。
- Identified security concern: `FAISS.load_local(..., allow_dangerous_deserialization=True)` allows arbitrary pickle deserialization; suggest rebuilding indexes instead of enabling the dangerous flag. (No changes applied yet; decision pending.)

## Implementation Notes
1. `app.py`
   - Added supportive subheader `失戀陣線聯盟關心你` under the main title，提醒朋友不孤單。
   - Removed citation footer appended to assistant replies to hide document source filenames，保持談話自然。
   - Eliminated language toggle; UI and prompts now default to Traditional Chinese, including sidebar labels, error messages, and spinner texts。
   - Simplified `build_persona_prompt` to the Chinese persona and guidelines only，明確要求 AI 先同理再給建議。
   - Updated captions and placeholders to consistent Chinese phrasing，讓操作體驗更貼近使用情境。

## Execution Attempts
- Attempted to launch the app via `streamlit run app.py` using the `aiot-hw4` conda environment; run command was initiated twice through the assistant but cancelled before completion (no confirmation of a successful launch)。
- Latest manual run on local PowerShell confirmed the Streamlit UI renders successfully and回覆語氣溫柔，同時顯示副標與自訂提示。

## Outstanding Items & Next Steps
- [ ] Decide on handling empty `books/` directories to prevent FAISS initialization failures。
- [ ] Revisit FAISS cache loading strategy to avoid `allow_dangerous_deserialization=True` or justify/mitigate its use。
- [ ] Manually run `streamlit run app.py` to verify the latest UI changes once ready。
- [ ] Explore hosting options (Streamlit Community Cloud, Render) and同步設置 `GOOGLE_API_KEY` secrets，準備部署給朋友日常使用。
