# =====================================================================
# FILE: webpage_query_deepseek_ui.py
# =====================================================================
# Kamal AI Web Crawler & Question Answering — DeepSeek Edition
# =====================================================================
#
# PURPOSE:
#   Same workflow as webpage_query_ui.py, but uses the DeepSeek LLM
#   instead of Groq. DeepSeek provides an OpenAI-compatible API, so
#   we use ChatOpenAI from langchain_openai with a custom base_url.
#
# WORKFLOW:
#   1. User enters a website URL in the "Scrape & Index" tab
#   2. Click "Discover Links" → UI finds all internal links on that page
#   3. User selects which links to crawl using the multiselect dropdown
#   4. Click "Crawl & Index" → real-time scraping with live log output
#   5. Scraped content is cleaned, chunked, embedded → FAISS vector store
#   6. Switch to "Ask Questions" tab → type a question → get AI answer
#
# TECH STACK:
#   - UI:           Gradio 6.x (streaming generators for real-time progress)
#   - LLM:          DeepSeek API (deepseek-chat / deepseek-reasoner)
#   - Embeddings:   HuggingFace sentence-transformers (runs locally)
#   - Vector Store: FAISS (in-memory, fast similarity search)
#   - RAG:          LangChain LCEL (4 chain types: stuff/map_reduce/refine/map_rerank)
#
# DEEPSEEK API:
#   - Compatible with the OpenAI SDK interface
#   - Requires DEEPSEEK_API_KEY in your .env file
#   - Base URL: https://api.deepseek.com
#   - Models: deepseek-chat (V3, general), deepseek-reasoner (R1, reasoning)
#
# USAGE:
#   python webpage_query_deepseek_ui.py
#   Then open http://localhost:7861 in your browser
#   (Port 7861 avoids conflict with webpage_query_ui.py on port 7860)
# =====================================================================


# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────

import gradio as gr                         # Web UI framework
import requests                             # HTTP requests for web scraping
from bs4 import BeautifulSoup               # HTML parser
import tldextract                           # Extract domain from URL
from urllib.parse import urljoin, urlparse  # URL manipulation utilities
import html2text                            # Convert HTML to Markdown
import re                                   # Regular expressions for text cleaning
import os                                   # OS utilities (env vars, file paths)
from typing import Generator               # Type hint for generator functions

# LangChain vector store + embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain core: prompts and LCEL building blocks
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   # Converts LLM output → plain string
from langchain_core.runnables import RunnablePassthrough    # Passes input through unchanged

# ── DeepSeek LLM integration ──────────────────────────────────────────
# DeepSeek exposes an OpenAI-compatible REST API.
# We use ChatOpenAI and override base_url + api_key to point at DeepSeek.
from langchain_openai import ChatOpenAI

# Load DEEPSEEK_API_KEY and other secrets from the .env file
from dotenv import load_dotenv
load_dotenv()

# DeepSeek API endpoint (OpenAI-compatible)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Read the DeepSeek API key from the environment
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")


# =====================================================================
# SECTION 1: WEB SCRAPING FUNCTIONS
# =====================================================================
# Identical to webpage_query_ui.py — scraping logic is LLM-independent.


def discover_links(base_url: str) -> tuple[list[str], str]:
    """
    Scrape only the base URL and collect all internal hyperlinks found on it.
    This is the "discovery" step — we do NOT crawl linked pages yet.
    The user can then choose which of the discovered links to crawl.

    Args:
        base_url (str): The starting URL to inspect for links.

    Returns:
        links (list[str]): Unique internal URLs found on the page.
        log   (str):       Status message shown in the UI.
    """
    links = []
    log   = ""

    try:
        # Fetch the HTML of the base URL with a 10-second timeout
        response = requests.get(base_url.strip(), timeout=10)
        response.raise_for_status()  # Raise an exception for 4xx / 5xx HTTP errors

        soup        = BeautifulSoup(response.text, "html.parser")
        base_domain = tldextract.extract(base_url).domain  # e.g. "python" from "python.langchain.com"

        # Loop through every anchor tag that has an href attribute
        for link_tag in soup.find_all("a", href=True):
            href     = link_tag["href"]
            full_url = urljoin(base_url, href)   # Convert relative URL → absolute
            parsed   = urlparse(full_url)

            # Keep only internal links that haven't been seen yet
            if (base_domain in parsed.netloc
                    and full_url != base_url
                    and "#" not in full_url
                    and full_url not in links):
                links.append(full_url)

        log = (
            f"✅ Base URL scraped successfully.\n"
            f"🔗 Found {len(links)} internal link(s) on this page."
        )

    except Exception as e:
        log = f"❌ Failed to scrape base URL: {e}"

    return links, log


def clean_scraped_text(text: str) -> str:
    """
    Normalize raw scraped text by collapsing redundant whitespace.

    Transformations:
      1. Multiple spaces/tabs        → single space
      2. Multiple consecutive blanks → one blank line
      3. Trailing whitespace on lines → removed
      4. Leading whitespace on lines  → removed

    Args:
        text (str): Raw scraped text.

    Returns:
        str: Cleaned, normalized text.
    """
    text = re.sub(r"[ \t]+",    " ",    text)   # Step 1
    text = re.sub(r"\n\s*\n+", "\n\n", text)   # Step 2
    text = re.sub(r"[ \t]+\n",  "\n",   text)   # Step 3
    text = re.sub(r"\n[ \t]+",  "\n",   text)   # Step 4
    return text.strip()


def scrape_urls_streaming(urls: list[str]) -> Generator:
    """
    Scrape a list of URLs and yield real-time progress tuples.

    This is a Gradio-compatible streaming generator:
      - Each `yield` sends an immediate UI update (log, status).
      - The final `yield` includes the fully extracted Markdown text.

    Only meaningful HTML tags (p, h1–h3, code) are extracted;
    navigation, sidebars, and footers are skipped.

    Args:
        urls (list[str]): Ordered list of URLs to scrape.

    Yields:
        Tuple[str, str, str]: (log_text, status_label, cleaned_markdown)
          - log_text:        Full running log (appended on each yield)
          - status_label:    Short status string for the status indicator
          - cleaned_markdown: Non-empty only on the final yield
    """
    # Configure html2text: convert HTML → Markdown
    h              = html2text.HTML2Text()
    h.ignore_links = False   # Keep hyperlinks
    h.body_width   = 0       # No line-wrapping

    # Tags to extract — skips nav bars, scripts, ads
    content_tags = ["p", "h1", "h2", "h3", "code"]

    visited            = set()
    extracted_markdown = ""
    log                = ""

    for current_url in urls:
        if current_url in visited:
            log += f"⏭️  Already visited: {current_url}\n"
            yield log, "⏳ Scraping...", ""
            continue

        try:
            log += f"\n🔎 Scraping: {current_url}\n"
            yield log, "⏳ Scraping...", ""   # Emit BEFORE the request (shows URL immediately)

            response = requests.get(current_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            visited.add(current_url)

            # Build focused HTML: only desired tags, skip nav/sidebar parents
            isolated_html = ""
            for tag in soup.find_all(content_tags):
                if tag.find_parent(
                    class_=lambda c: c and any(
                        x in c.lower() for x in ["menu", "sidebar", "nav", "footer"]
                    )
                ):
                    continue
                isolated_html += tag.prettify()

            # Convert extracted HTML fragment to clean Markdown
            if isolated_html.strip():
                markdown_text       = h.handle(isolated_html)
                extracted_markdown += f"\n\n--- Page: {current_url} ---\n\n"
                extracted_markdown += markdown_text
                log += f"   ✅ Content extracted ({len(markdown_text)} chars)\n"
            else:
                log += f"   ⚠️  No extractable content found.\n"

        except Exception as e:
            log += f"   ❌ Failed: {e}\n"

        yield log, "⏳ Scraping...", ""

    # ── Post-scraping: clean and save ────────────────────────────────
    log += "\n🧹 Cleaning extracted text...\n"
    yield log, "⏳ Cleaning...", ""

    cleaned_text = clean_scraped_text(extracted_markdown)

    # Save to a separate file so it doesn't overwrite the Groq version
    with open("scraped_data_deepseek.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    log += "💾 Saved to scraped_data_deepseek.txt\n"

    # Final yield — pass the full text back via the 3rd output slot
    yield log, "✅ Scraping complete!", cleaned_text


# =====================================================================
# SECTION 2: VECTOR STORE BUILDER
# =====================================================================
# Identical to webpage_query_ui.py — embeddings are LLM-independent.


def build_vector_store(text: str, chunk_size: int = 1500, chunk_overlap: int = 200):
    """
    Convert a large text string into a searchable FAISS vector store.

    Steps:
      1. Split text into overlapping chunks (RecursiveCharacterTextSplitter).
      2. Embed each chunk using HuggingFace all-MiniLM-L6-v2 (local model).
      3. Build a FAISS index for fast similarity search.

    Args:
        text          (str): Full cleaned text to index.
        chunk_size    (int): Max characters per chunk (default: 1500).
        chunk_overlap (int): Characters shared between adjacent chunks (default: 200).

    Returns:
        vectorstore (FAISS): Ready for similarity search.
        num_chunks  (int):   Number of chunks created.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_text(text)

    # all-MiniLM-L6-v2: fast, 80 MB, excellent semantic similarity
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embedding_model)
    return vectorstore, len(chunks)


# =====================================================================
# SECTION 3: PROMPT TEMPLATES
# =====================================================================
# Same prompts as webpage_query_ui.py — prompts are LLM-agnostic.


# ── Stuff: all retrieved chunks in one prompt ──────────────────────
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following extracted content to answer the question.
Answer in a clear, factual, and concise way. If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ── Map-Reduce MAP step: applied to each chunk individually ────────
MAP_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question:

{context}

Question: {question}
Answer:
"""
)

# ── Map-Reduce COMBINE step: merges all chunk-level answers ────────
COMBINE_PROMPT = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
The following are answers extracted from different document sections:
{summaries}

Given the above, provide a final, concise answer to the question:

Question: {question}
Answer:
"""
)

# ── Refine INITIAL step: generates answer from the first chunk ─────
QUESTION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are given a document and a question. Use the document to answer.

Document:
{context}

Question: {question}
Answer:
"""
)

# ── Refine REFINEMENT step: updates answer using next chunk ─────────
REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "context", "question"],
    template="""
We have an existing answer: {existing_answer}

Here is another document section that may help refine it:
{context}

Question: {question}

Update the answer if this document provides new useful information.
If not, keep the original answer unchanged.

Refined Answer:
"""
)

# ── Map-Rerank: scores each chunk, returns the best answer ─────────
RERANK_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are given a document and a question.

Document:
{context}

Question: {question}

Provide:
1. An answer to the question (if the document is relevant).
2. A relevance score between 0 and 10 (higher means more relevant).

Format:
Answer: <your answer here>
Score: <number between 0 and 10>
"""
)


# =====================================================================
# SECTION 4: QA CHAIN BUILDER  (LCEL — DeepSeek edition)
# =====================================================================
# Same LCEL logic as webpage_query_ui.py.
# The ONLY difference: LLM is ChatOpenAI pointed at DeepSeek's API
# instead of ChatGroq.


def _format_docs(docs: list) -> str:
    """Merge a list of Document objects into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _run_stuff_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    STUFF strategy — concatenate all retrieved chunks into a single
    context block and send everything to DeepSeek in one prompt.

    Best for: short pages, quick answers, few retrieved chunks.
    """
    docs  = retriever.invoke(question)
    chain = CUSTOM_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context":  _format_docs(docs),
        "question": question
    })
    return answer, docs


def _run_map_reduce_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    MAP-REDUCE strategy — each chunk is answered independently (map),
    then all partial answers are combined into one final answer (reduce).

    Best for: many pages / large volumes of retrieved text.
    """
    docs = retriever.invoke(question)

    # MAP: ask DeepSeek about each chunk individually
    map_chain       = MAP_PROMPT | llm | StrOutputParser()
    partial_answers = [
        map_chain.invoke({"context": doc.page_content, "question": question})
        for doc in docs
    ]

    # REDUCE: combine partial answers → one final answer
    combine_chain = COMBINE_PROMPT | llm | StrOutputParser()
    answer = combine_chain.invoke({
        "summaries": "\n\n".join(partial_answers),
        "question":  question
    })
    return answer, docs


def _run_refine_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    REFINE strategy — first chunk produces an initial answer; each
    subsequent chunk may improve it if it contains new information.

    Best for: high-quality, thorough answers over long documents.
    """
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant documents found.", []

    # Initial answer from the first chunk
    init_chain = QUESTION_PROMPT | llm | StrOutputParser()
    answer = init_chain.invoke({
        "context":  docs[0].page_content,
        "question": question
    })

    # Refine using each subsequent chunk
    refine_chain = REFINE_PROMPT | llm | StrOutputParser()
    for doc in docs[1:]:
        answer = refine_chain.invoke({
            "existing_answer": answer,
            "context":         doc.page_content,
            "question":        question
        })
    return answer, docs


def _run_map_rerank_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    MAP-RERANK strategy — DeepSeek scores each chunk independently (0–10).
    The answer from the highest-scoring chunk is returned.

    Best for: finding the single most relevant document chunk.
    """
    docs         = retriever.invoke(question)
    rerank_chain = RERANK_PROMPT | llm | StrOutputParser()
    best_answer  = "No relevant document found."
    best_score   = -1

    for doc in docs:
        raw = rerank_chain.invoke({
            "context":  doc.page_content,
            "question": question
        })

        # Parse structured "Answer: ... Score: ..." response
        score_match  = re.search(r"Score:\s*(\d+)",              raw)
        answer_match = re.search(r"Answer:\s*(.+?)(?=Score:|$)", raw, re.DOTALL)

        score  = int(score_match.group(1))      if score_match  else 0
        answer = answer_match.group(1).strip()  if answer_match else raw.strip()

        if score > best_score:
            best_score  = score
            best_answer = answer

    return best_answer, docs


def run_qa(vectorstore, model_name: str, chain_type: str, question: str) -> tuple[str, list]:
    """
    Dispatcher: initialises the DeepSeek LLM, selects the correct LCEL
    chain strategy, and returns the answer with its source documents.

    ── DeepSeek LLM setup ───────────────────────────────────────────
    DeepSeek's API is OpenAI-compatible, so we use ChatOpenAI and
    override two parameters:
      - api_key  → DEEPSEEK_API_KEY from .env
      - base_url → https://api.deepseek.com

    Available models:
      deepseek-chat      DeepSeek V3 — fast, general-purpose (recommended)
      deepseek-reasoner  DeepSeek R1 — chain-of-thought reasoning, slower
    ─────────────────────────────────────────────────────────────────

    Chain Types:
      "stuff"       All chunks in one prompt (fastest)
      "map_reduce"  Per-chunk answers combined (handles more text)
      "refine"      Iteratively improved answer (highest quality)
      "map_rerank"  Best-scored chunk wins (most targeted)

    Args:
        vectorstore (FAISS): Indexed vector store to search.
        model_name  (str):   DeepSeek model name.
        chain_type  (str):   RAG strategy.
        question    (str):   User's question.

    Returns:
        answer      (str):  DeepSeek's answer.
        source_docs (list): Retrieved Document objects used as context.
    """
    # ── Initialise DeepSeek via the OpenAI-compatible interface ──────
    llm = ChatOpenAI(
        model       = model_name,
        api_key     = DEEPSEEK_API_KEY,       # From DEEPSEEK_API_KEY in .env
        base_url    = DEEPSEEK_BASE_URL,      # https://api.deepseek.com
        temperature = 0                        # Deterministic output
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    dispatch = {
        "stuff":      _run_stuff_chain,
        "map_reduce": _run_map_reduce_chain,
        "refine":     _run_refine_chain,
        "map_rerank": _run_map_rerank_chain,
    }
    if chain_type not in dispatch:
        raise ValueError(f"Unknown chain_type '{chain_type}'. Choose: {list(dispatch)}")

    return dispatch[chain_type](retriever, llm, question)


# =====================================================================
# SECTION 5: GRADIO EVENT HANDLERS
# =====================================================================


def on_discover_links(url: str):
    """
    Event handler for the 'Discover Links' button.

    Scrapes the base URL and returns discovered links as Dropdown choices
    (all pre-selected by default).

    Args:
        url (str): URL entered by the user.

    Returns:
        Tuple[gr.update, str]: Updated Dropdown + status message.
    """
    if not url.strip():
        return gr.update(choices=[], value=[]), "⚠️ Please enter a valid URL first."

    links, log = discover_links(url.strip())
    return gr.update(choices=links, value=links), log


def on_scrape_and_index(url: str, selected_links: list, max_pages: int):
    """
    Streaming event handler for the 'Crawl & Index' button.

    Yields real-time log updates to the UI as each page is scraped,
    then builds the FAISS vector store and stores it in Gradio State.

    Args:
        url            (str):  Base URL (always included).
        selected_links (list): Links chosen in the multiselect dropdown.
        max_pages      (int):  Maximum pages to crawl.

    Yields:
        Tuple[str, str, object, int]:
            (scrape_log, status_text, vectorstore_or_None, chunks_count)
    """
    if not url.strip():
        yield "⚠️ No URL provided. Please enter a URL and discover links first.", \
              "❌ No URL", None, 0
        return

    # Deduplicate and cap at max_pages
    all_urls = list(dict.fromkeys([url.strip()] + (selected_links or [])))[:max_pages]

    log = f"📋 Will crawl {len(all_urls)} URL(s):\n"
    for u in all_urls:
        log += f"   • {u}\n"
    log += "\n"

    yield log, "⏳ Starting scrape...", None, 0

    # ── Relay the streaming scraper's progress to the UI ─────────────
    cleaned_text = ""
    for scrape_log_update, scrape_status, final_text in scrape_urls_streaming(all_urls):
        combined_log = log + scrape_log_update
        yield combined_log, scrape_status, None, 0
        if final_text:
            cleaned_text = final_text

    if not cleaned_text.strip():
        yield combined_log + "\n❌ No content extracted. Check the URL and try again.", \
              "❌ Nothing scraped", None, 0
        return

    # ── Build FAISS vector store from scraped text ────────────────────
    combined_log += "\n🧠 Building vector store — embedding text chunks...\n"
    yield combined_log, "⏳ Building index...", None, 0

    try:
        vectorstore, num_chunks = build_vector_store(cleaned_text)
        combined_log += (
            f"✅ Vector store ready! {num_chunks} chunks indexed.\n\n"
            f"🎉 All done! Switch to the '💬 Ask Questions' tab to query the content."
        )
        yield combined_log, f"✅ Indexed {num_chunks} chunks — Ready!", vectorstore, num_chunks

    except Exception as e:
        combined_log += f"\n❌ Indexing error: {e}"
        yield combined_log, f"❌ Error: {e}", None, 0


def on_ask_question(question: str, model_name: str, chain_type: str, vectorstore):
    """
    Event handler for the 'Get Answer' button.

    Runs the user's question through the DeepSeek-powered RAG chain
    and returns the answer plus the retrieved source document chunks.

    Args:
        question    (str):   The user's question.
        model_name  (str):   DeepSeek model to use.
        chain_type  (str):   RAG chain strategy.
        vectorstore (FAISS): Indexed vector store from Gradio State.

    Returns:
        Tuple[str, str]: (answer_text, formatted_source_chunks)
    """
    if vectorstore is None:
        return (
            "⚠️ No content indexed yet. Please scrape and index a website first.",
            ""
        )
    if not question.strip():
        return "⚠️ Please enter a question.", ""

    try:
        answer, source_docs = run_qa(vectorstore, model_name, chain_type, question.strip())

        if source_docs:
            sources_text = ""
            for i, doc in enumerate(source_docs, start=1):
                preview = doc.page_content[:400].strip()
                sources_text += f"─── Source {i} ───\n{preview}\n\n"
        else:
            sources_text = "No source documents returned."

        return answer, sources_text

    except Exception as e:
        return f"❌ Error running QA chain: {e}", ""


# =====================================================================
# SECTION 6: GRADIO UI LAYOUT
# =====================================================================


def build_ui():
    """
    Construct and return the Gradio Blocks application.

    Identical layout to webpage_query_ui.py with these visual differences:
      - Header colour: teal/deep-green accent to distinguish from the Groq UI
      - Badge mentions DeepSeek instead of Groq
      - Model dropdown lists DeepSeek models

    Returns:
        gr.Blocks: The fully configured Gradio application.
    """

    # ── Professional CSS — White / Black / Teal-Blue (DeepSeek edition) ──
    PRO_CSS = """

    /* ── Base: clean white page ── */
    body, .gradio-container {
        background: #f8fafc !important;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        color: #111827 !important;
    }

    /* ── Hero header — teal accent to distinguish from the Groq UI ── */
    .app-header {
        background: linear-gradient(135deg, #0f4c6e 0%, #065f46 100%);
        border-radius: 10px;
        padding: 26px 32px;
        margin-bottom: 4px;
        box-shadow: 0 2px 12px rgba(6,95,70,0.18);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin: 0 0 5px 0 !important;
        letter-spacing: -0.2px;
    }
    .app-header p {
        color: #a7f3d0 !important;
        font-size: 0.92rem !important;
        margin: 0 0 12px 0 !important;
    }
    .badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.73rem;
        color: #d1fae5;
        font-weight: 500;
    }

    /* ── Page background & card panels ── */
    .gradio-container .prose,
    .block, fieldset {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    }

    /* ── Tab bar ── */
    .tabs > .tab-nav {
        background: #ffffff !important;
        border-bottom: 2px solid #6ee7b7 !important;
        padding: 0 8px !important;
    }
    .tabs > .tab-nav button {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 10px 22px !important;
        border: none !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -2px !important;
        transition: color 0.15s !important;
    }
    .tabs > .tab-nav button.selected {
        color: #065f46 !important;
        border-bottom: 2px solid #059669 !important;
        background: transparent !important;
    }
    .tabs > .tab-nav button:hover:not(.selected) {
        color: #059669 !important;
        background: #ecfdf5 !important;
    }

    /* ── Section labels ── */
    .section-label {
        color: #065f46 !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        border-left: 3px solid #059669 !important;
        padding-left: 9px !important;
        margin: 16px 0 6px 0 !important;
        background: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }

    /* ── Labels & info text ── */
    label span {
        color: #374151 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    .info { color: #94a3b8 !important; font-size: 0.77rem !important; }

    /* ── Inputs, textareas, selects ── */
    input, textarea, select {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }
    input:focus, textarea:focus {
        border-color: #059669 !important;
        box-shadow: 0 0 0 3px rgba(5,150,105,0.12) !important;
        outline: none !important;
    }

    /* ── Slider ── */
    input[type=range] {
        accent-color: #059669 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Multiselect dropdown ── */
    .multiselect, .multiselect > div {
        background: #ffffff !important;
        border: 1.5px solid #6ee7b7 !important;
        border-radius: 8px !important;
        min-height: 52px !important;
    }
    .multiselect:focus-within {
        border-color: #059669 !important;
        box-shadow: 0 0 0 3px rgba(5,150,105,0.12) !important;
    }
    /* Selected link chips */
    .multiselect .token, .multiselect [data-token] {
        background: #059669 !important;
        color: #ffffff !important;
        border-radius: 5px !important;
        padding: 3px 10px !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 5px !important;
        margin: 2px !important;
    }
    .multiselect .token button, .multiselect [data-token] button {
        background: transparent !important;
        color: #a7f3d0 !important;
        border: none !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        padding: 0 2px !important;
    }
    .multiselect .token button:hover { color: #ffffff !important; }
    /* Dropdown list */
    .multiselect .dropdown, .multiselect ul {
        background: #ffffff !important;
        border: 1px solid #6ee7b7 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
        max-height: 260px !important;
        overflow-y: auto !important;
    }
    .multiselect li, .multiselect .option {
        color: #065f46 !important;
        font-size: 0.83rem !important;
        padding: 8px 14px !important;
        cursor: pointer !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }
    .multiselect li:hover, .multiselect .option:hover {
        background: #ecfdf5 !important;
        color: #059669 !important;
    }
    .multiselect li.selected, .multiselect .option.selected {
        background: #d1fae5 !important;
        color: #065f46 !important;
        font-weight: 600 !important;
    }

    /* ── Primary button ── */
    button.primary {
        background: #059669 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 10px 24px !important;
        box-shadow: 0 1px 6px rgba(5,150,105,0.3) !important;
        transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    }
    button.primary:hover {
        background: #047857 !important;
        box-shadow: 0 3px 12px rgba(4,120,87,0.35) !important;
        transform: translateY(-1px) !important;
    }
    button.primary:active { transform: translateY(0) !important; }

    /* ── Secondary button ── */
    button.secondary {
        background: #ffffff !important;
        color: #059669 !important;
        border: 1.5px solid #059669 !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: background 0.15s, box-shadow 0.15s !important;
    }
    button.secondary:hover {
        background: #ecfdf5 !important;
        border-color: #047857 !important;
        box-shadow: 0 1px 6px rgba(5,150,105,0.15) !important;
    }

    /* ── Live log box (dark terminal) ── */
    .log-box textarea {
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 0.82rem !important;
        background: #022c22 !important;
        color: #6ee7b7 !important;
        border: 1px solid #6ee7b7 !important;
        border-radius: 6px !important;
        line-height: 1.65 !important;
    }

    /* ── Answer box ── */
    .answer-box textarea {
        background: #ecfdf5 !important;
        color: #111827 !important;
        font-size: 0.93rem !important;
        line-height: 1.75 !important;
        border: 1px solid #6ee7b7 !important;
        border-radius: 6px !important;
    }

    /* ── Status box ── */
    .status-box textarea, .status-box input {
        background: #f0fdf4 !important;
        color: #166534 !important;
        border: 1px solid #bbf7d0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* ── Chunks badge ── */
    .chunks-badge input {
        background: #ecfdf5 !important;
        color: #065f46 !important;
        border: 1px solid #6ee7b7 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-align: center !important;
    }

    /* ── Accordion ── */
    .accordion {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .accordion .label-wrap span {
        color: #059669 !important;
        font-weight: 600 !important;
    }
    """

    with gr.Blocks(title="Kamal DeepSeek Web Crawler & Q&A", css=PRO_CSS) as demo:

        # ── State components (invisible, persist across button clicks) ─
        vectorstore_state = gr.State(None)   # Holds the FAISS index object
        num_chunks_state  = gr.State(0)      # Number of indexed chunks

        # ── Hero Header ───────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>🤖 Kamal AI Web Crawler &amp; Question Answering</h1>
            <p>Scrape any website &rarr; Index its content &rarr; Ask questions using DeepSeek AI</p>
            <div class="badge-row">
                <span class="badge">🔵 DeepSeek LLM</span>
                <span class="badge">🔗 LangChain RAG</span>
                <span class="badge">🗄️ FAISS Vector Search</span>
                <span class="badge">🤗 HuggingFace Embeddings</span>
            </div>
        </div>
        """)

        # ─────────────────────────────────────────────────────────────
        # TAB 1: SCRAPE & INDEX
        # ─────────────────────────────────────────────────────────────
        with gr.Tab("🌐 Scrape & Index"):

            gr.HTML('<p class="section-label">Step 1 — Enter a URL and discover its links</p>')

            with gr.Row():
                url_input = gr.Textbox(
                    label       = "Website URL",
                    placeholder = "https://docs.deepseek.com/",
                    scale       = 4,
                    info        = "Enter the starting URL. Links from this page will be discovered."
                )
                max_pages_slider = gr.Slider(
                    minimum = 1,
                    maximum = 20,
                    value   = 5,
                    step    = 1,
                    label   = "Max Pages to Crawl",
                    scale   = 1,
                    info    = "Limit the total number of pages scraped."
                )

            with gr.Row():
                discover_btn = gr.Button(
                    "🔍 Discover Links",
                    variant = "secondary",
                    scale   = 1
                )
                discover_status = gr.Textbox(
                    label       = "Discovery Status",
                    interactive = False,
                    scale       = 3,
                    lines       = 2
                )

            gr.HTML('<p class="section-label">Step 2 — Select links to crawl</p>')

            # Multiselect dropdown: selected links appear as teal chips
            links_selector = gr.Dropdown(
                label       = "Discovered Links",
                info        = "All links selected by default. Remove any you want to skip, or use the buttons below.",
                choices     = [],
                value       = [],
                multiselect = True,
                interactive = True
            )

            with gr.Row():
                select_all_btn   = gr.Button("✅ Select All",   variant="secondary", scale=1)
                deselect_all_btn = gr.Button("❌ Deselect All", variant="secondary", scale=1)

            gr.HTML('<p class="section-label">Step 3 — Crawl & build the search index</p>')

            crawl_btn = gr.Button(
                "🕷️ Crawl & Index Selected Pages",
                variant = "primary"
            )

            with gr.Row():
                index_status = gr.Textbox(
                    label        = "Status",
                    interactive  = False,
                    scale        = 1,
                    lines        = 1,
                    elem_classes = ["status-box"]
                )
                chunks_display = gr.Number(
                    label        = "Chunks Indexed",
                    value        = 0,
                    interactive  = False,
                    scale        = 1,
                    elem_classes = ["chunks-badge"]
                )

            # Live scraping log — teal text on dark background
            scrape_log = gr.Textbox(
                label        = "Live Scraping Log",
                interactive  = False,
                lines        = 18,
                max_lines    = 100,
                autoscroll   = True,
                elem_classes = ["log-box"]
            )

        # ─────────────────────────────────────────────────────────────
        # TAB 2: ASK QUESTIONS
        # ─────────────────────────────────────────────────────────────
        with gr.Tab("💬 Ask Questions"):

            gr.HTML(
                '<p class="section-label">Ask anything about the scraped website content</p>'
                '<p style="color:#64748b;font-size:0.83rem;margin:0 0 12px 0;">'
                'Make sure you have indexed a website first in the Scrape &amp; Index tab.</p>'
            )

            question_input = gr.Textbox(
                label       = "Your Question",
                placeholder = "e.g. What are DeepSeek's main capabilities?",
                lines       = 3,
                info        = "Type a question about the content you scraped and indexed."
            )

            with gr.Row():
                # ── DeepSeek model selector ──────────────────────────
                # deepseek-chat     → DeepSeek V3 (fast, general purpose, recommended)
                # deepseek-reasoner → DeepSeek R1 (chain-of-thought, slower but smarter)
                model_dropdown = gr.Dropdown(
                    label   = "DeepSeek Model",
                    choices = [
                        "deepseek-chat",      # DeepSeek V3 — fast, general purpose (recommended)
                        "deepseek-reasoner",  # DeepSeek R1 — advanced chain-of-thought reasoning
                    ],
                    value = "deepseek-chat",
                    info  = (
                        "deepseek-chat = fast & general | "
                        "deepseek-reasoner = slower but deeper reasoning"
                    )
                )

                # ── RAG chain type selector ───────────────────────────
                chain_type_dropdown = gr.Dropdown(
                    label   = "RAG Chain Type",
                    choices = [
                        "stuff",        # All chunks in one prompt (fastest)
                        "map_reduce",   # Per-chunk answers combined
                        "refine",       # Answer improved across chunks
                        "map_rerank",   # Best-scored chunk wins
                    ],
                    value = "stuff",
                    info  = (
                        "stuff = fast/simple | "
                        "map_reduce = handles more text | "
                        "refine = most thorough | "
                        "map_rerank = most relevant"
                    )
                )

            ask_btn = gr.Button("💬 Get Answer", variant="primary")

            answer_output = gr.Textbox(
                label        = "Answer",
                interactive  = False,
                lines        = 10,
                elem_classes = ["answer-box"]
            )

            with gr.Accordion("📚 Retrieved Source Documents", open=False):
                gr.Markdown(
                    "_These are the text chunks retrieved from the vector store "
                    "and passed to DeepSeek as context._"
                )
                sources_output = gr.Textbox(
                    label       = "Source Chunks",
                    interactive = False,
                    lines       = 15,
                    max_lines   = 50
                )

        # ─────────────────────────────────────────────────────────────
        # EVENT WIRING
        # ─────────────────────────────────────────────────────────────

        # Discover Links → populate multiselect dropdown
        discover_btn.click(
            fn      = on_discover_links,
            inputs  = [url_input],
            outputs = [links_selector, discover_status]
        )

        # Select All → set dropdown value to all available choices
        select_all_btn.click(
            fn      = lambda choices: gr.update(value=choices),
            inputs  = [links_selector],
            outputs = [links_selector]
        )

        # Deselect All → clear all selections
        deselect_all_btn.click(
            fn      = lambda _: gr.update(value=[]),
            inputs  = [links_selector],
            outputs = [links_selector]
        )

        # Crawl & Index → streaming scrape + FAISS build
        crawl_btn.click(
            fn      = on_scrape_and_index,
            inputs  = [url_input, links_selector, max_pages_slider],
            outputs = [scrape_log, index_status, vectorstore_state, num_chunks_state]
        )

        # Keep the chunks number display in sync with state
        num_chunks_state.change(
            fn      = lambda n: n,
            inputs  = [num_chunks_state],
            outputs = [chunks_display]
        )

        # Ask button → run DeepSeek RAG chain
        ask_btn.click(
            fn      = on_ask_question,
            inputs  = [question_input, model_dropdown, chain_type_dropdown, vectorstore_state],
            outputs = [answer_output, sources_output]
        )

        # Enter key in question box also triggers the chain
        question_input.submit(
            fn      = on_ask_question,
            inputs  = [question_input, model_dropdown, chain_type_dropdown, vectorstore_state],
            outputs = [answer_output, sources_output]
        )

    return demo


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    """
    Launch the DeepSeek-powered Gradio web application.

    Access the UI at: http://localhost:7861
    Port 7861 is used so this can run alongside webpage_query_ui.py (port 7860).
    """
    app = build_ui()
    app.launch(
        server_name = "0.0.0.0",     # Listen on all network interfaces
        server_port = 7861,           # Different port from the Groq UI (7860)
        share       = False,          # Set True for a public gradio.live link
        inbrowser   = True,           # Auto-open browser on launch
        theme       = gr.themes.Default(),  # Clean white base — CSS handles the rest
    )
