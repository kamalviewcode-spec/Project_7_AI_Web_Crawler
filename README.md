# Kamal AI Web Crawler & Question Answering

Scrape any website → Index its content → Ask questions using AI.

**GitHub:** https://github.com/kamalviewcode-spec/Project_7_AI_Web_Crawler

---

## Setup

```bash
uv sync
```

Add your API keys to `.env`:

```env
GROQ_API_KEY=
DEEPSEEK_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

---

## Run

```bash
python webpage_query_groq_ui.py       # http://localhost:7860
python webpage_query_deepseek_ui.py   # http://localhost:7861
python webpage_query_claude_ui.py     # http://localhost:7862
python webpage_query_openai_ui.py     # http://localhost:7863
python webpage_query_gemini_ui.py     # http://localhost:7864
```

---

## How It Works

1. Enter a URL → Discover links → Select pages to crawl
2. Crawl & Index — content is embedded into a FAISS vector store
3. Ask a question — AI retrieves relevant chunks and answers

---

## Tech Stack

`Gradio` · `LangChain` · `FAISS` · `HuggingFace Embeddings` · `BeautifulSoup4` · `Groq` · `DeepSeek` · `Claude` · `OpenAI` · `Gemini`
