# =====================================================================
# FILE: webpage_query_groq_ui.py
# =====================================================================
# AI Web Crawler & Question Answering — Real-Time Gradio Interface
# =====================================================================
#
# PURPOSE:
#   Converts the Jupyter notebook (webpage_query_modular.ipynb) into a
#   fully interactive, real-time web UI using Gradio 5.x.
#
# WORKFLOW:
#   1. User enters a website URL in the "Scrape & Index" tab
#   2. Click "Discover Links" → UI finds all internal links on that page
#   3. User selects which links to crawl using checkboxes
#   4. Click "Crawl & Index" → real-time scraping with live log output
#   5. Scraped content is cleaned, chunked, embedded → FAISS vector store
#   6. Switch to "Ask Questions" tab → type a question → get AI answer
#
# TECH STACK:
#   - UI:           Gradio 5.x (streaming generators for real-time progress)
#   - LLM:          Groq API  (llama/gemma models, fast inference)
#   - Embeddings:   HuggingFace sentence-transformers (runs locally)
#   - Vector Store: FAISS (in-memory, fast similarity search)
#   - RAG:          LangChain RetrievalQA (4 chain types supported)
#
# USAGE:
#   python webpage_query_ui.py
#   Then open http://localhost:7860 in your browser
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
import os                                   # OS utilities (file paths, env vars)
from typing import Generator               # Type hint for generator functions

# LangChain vector store + embeddings (from langchain_community)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain core: prompts and LCEL (LangChain Expression Language) building blocks
# Note: RetrievalQA was removed in LangChain 1.x — we use LCEL chains instead
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   # Converts LLM output to plain string
from langchain_core.runnables import RunnablePassthrough    # Passes input through unchanged

# Groq LLM integration
from langchain_groq import ChatGroq

# Load environment variables from .env file
# This reads GROQ_API_KEY, OPENAI_API_KEY, etc. from the .env file
from dotenv import load_dotenv
load_dotenv()


# =====================================================================
# SECTION 1: WEB SCRAPING FUNCTIONS
# =====================================================================


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
            full_url = urljoin(base_url, href)   # Convert relative URL (e.g. "/docs") → absolute
            parsed   = urlparse(full_url)

            # Keep the link only if:
            #  - It belongs to the same domain as the base URL
            #  - It is not the base URL itself
            #  - It is not a fragment-only anchor (e.g. "#section")
            #  - It has not already been added to the list
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
    Normalize raw scraped text to remove noise introduced during HTML-to-text conversion.

    Transformations applied:
      1. Collapse multiple spaces/tabs → single space (preserves newlines)
      2. Collapse multiple consecutive blank lines → one blank line
      3. Remove trailing whitespace at end of each line
      4. Remove leading whitespace at start of each line

    Args:
        text (str): Raw text (possibly with redundant whitespace).

    Returns:
        str: Cleaned, normalized text.
    """
    text = re.sub(r"[ \t]+",    " ",    text)   # Step 1: multi-space → single space
    text = re.sub(r"\n\s*\n+", "\n\n", text)   # Step 2: multi-blank-lines → one blank line
    text = re.sub(r"[ \t]+\n",  "\n",   text)   # Step 3: trailing spaces before newline
    text = re.sub(r"\n[ \t]+",  "\n",   text)   # Step 4: leading spaces after newline
    return text.strip()


def scrape_urls_streaming(urls: list[str]) -> Generator:
    """
    Scrape a list of URLs and yield real-time progress tuples.

    This is a Gradio-compatible **streaming generator**:
      - Each `yield` sends a progress update to the UI immediately.
      - The final `yield` includes the fully extracted Markdown text.

    HTML content is converted to Markdown using html2text.
    Only meaningful tags (p, h1-h3, code) are extracted — navigation,
    sidebars, and footers are excluded to keep the content focused.

    Args:
        urls (list[str]): Ordered list of URLs to scrape.

    Yields:
        Tuple[str, str, str]: (log_text, status_label, accumulated_markdown)
          - log_text: Full running log of scraping activity (appended to each yield)
          - status_label: Short status string shown in the status indicator
          - accumulated_markdown: Full extracted text so far (empty until last yield)
    """
    # Configure html2text converter
    h             = html2text.HTML2Text()
    h.ignore_links = False   # Keep hyperlinks in the Markdown output
    h.body_width   = 0       # Disable line-wrapping

    # HTML tags to extract (excludes nav bars, footers, scripts, ads)
    content_tags = ["p", "h1", "h2", "h3", "code"]

    visited            = set()          # Avoid revisiting URLs
    extracted_markdown = ""             # Accumulated full-text content
    log                = ""             # Running log displayed in the UI

    for current_url in urls:
        # Skip URLs already visited in this run
        if current_url in visited:
            log += f"⏭️  Already visited: {current_url}\n"
            yield log, "⏳ Scraping...", ""
            continue

        try:
            log += f"\n🔎 Scraping: {current_url}\n"
            yield log, "⏳ Scraping...", ""   # Emit progress BEFORE the actual request

            response = requests.get(current_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            visited.add(current_url)

            # ── Focused extraction ──────────────────────────────────────
            # Build a string of only the desired HTML tags,
            # skipping any tag that lives inside a navigation or sidebar container.
            isolated_html = ""
            for tag in soup.find_all(content_tags):
                # Check if this tag is a descendant of a nav/sidebar/menu element
                if tag.find_parent(
                    class_=lambda c: c and any(
                        x in c.lower() for x in ["menu", "sidebar", "nav", "footer"]
                    )
                ):
                    continue  # Skip navigation content
                isolated_html += tag.prettify()  # Preserve tag structure for html2text

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

        # Emit updated log after each URL (no markdown yet — will send at the end)
        yield log, "⏳ Scraping...", ""

    # ── Post-scraping cleanup and final yield ────────────────────────────
    log += "\n🧹 Cleaning extracted text...\n"
    yield log, "⏳ Cleaning...", ""

    cleaned_text = clean_scraped_text(extracted_markdown)

    # Persist scraped content to disk for inspection / debugging
    with open("scraped_data_groq.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    log += "💾 Saved to scraped_data_groq.txt\n"

    # Final yield: pass the full cleaned text back via the 3rd output slot
    yield log, "✅ Scraping complete!", cleaned_text


# =====================================================================
# SECTION 2: VECTOR STORE BUILDER
# =====================================================================


def build_vector_store(text: str, chunk_size: int = 1500, chunk_overlap: int = 200):
    """
    Convert a large text string into a searchable FAISS vector store.

    Steps:
      1. Split the text into overlapping chunks using RecursiveCharacterTextSplitter.
         Overlap ensures that context spanning chunk boundaries is not lost.
      2. Load a lightweight local embedding model (all-MiniLM-L6-v2).
         This converts each text chunk into a dense vector (384 dimensions).
      3. Build a FAISS index from the chunk vectors.
         FAISS enables fast approximate nearest-neighbour search.

    Args:
        text          (str): Full cleaned text to index.
        chunk_size    (int): Maximum characters per chunk (default: 1500).
        chunk_overlap (int): Characters shared between adjacent chunks (default: 200).

    Returns:
        vectorstore: FAISS vector store (ready for similarity search).
        num_chunks (int): Number of text chunks created.
    """
    # Split text into overlapping chunks for better retrieval coverage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = splitter.split_text(text)

    # Load sentence-transformer embedding model (downloads automatically on first use)
    # "all-MiniLM-L6-v2" is fast, small (80 MB), and excellent for semantic similarity
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build the FAISS index: embed all chunks and store them for retrieval
    vectorstore = FAISS.from_texts(chunks, embedding_model)

    return vectorstore, len(chunks)


# =====================================================================
# SECTION 3: PROMPT TEMPLATES
# =====================================================================
# LangChain's RetrievalQA uses prompt templates to structure the context
# and question sent to the LLM. Different chain types require different
# template variables.


# ── Stuff chain prompt ─────────────────────────────────────────────
# Used when all retrieved chunks fit into a single prompt.
# Simple and fast; works best for short documents or small k values.
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

# ── Map-Reduce: MAP step prompt ────────────────────────────────────
# Applied independently to EACH retrieved chunk to generate a partial answer.
# These partial answers are then combined by the COMBINE prompt below.
MAP_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question:

{context}

Question: {question}
Answer:
"""
)

# ── Map-Reduce: COMBINE (reduce) step prompt ───────────────────────
# Merges all partial answers from the MAP step into one final answer.
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

# ── Refine: INITIAL question prompt ───────────────────────────────
# The first chunk is passed to this prompt to generate an initial answer.
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

# ── Refine: REFINEMENT prompt ──────────────────────────────────────
# Each subsequent chunk is passed here along with the existing answer.
# The LLM decides whether the new chunk improves the answer.
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

# ── Map-Rerank prompt ──────────────────────────────────────────────
# Each chunk is scored independently for relevance.
# The answer from the highest-scoring chunk is returned.
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
# SECTION 4: QA CHAIN BUILDER  (LCEL — LangChain Expression Language)
# =====================================================================
# RetrievalQA was removed in LangChain 1.x.
# We now build each chain manually using LCEL pipe ( | ) syntax:
#   retriever → prompt → llm → output_parser
#
# Each chain function returns: (answer: str, source_docs: list)


def _format_docs(docs: list) -> str:
    """Join a list of Document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _run_stuff_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    STUFF strategy — all retrieved chunks are concatenated into one context block
    and sent to the LLM in a single prompt.

    Best for: short content, quick answers, small number of retrieved chunks.
    """
    docs = retriever.invoke(question)   # Fetch top-k most relevant chunks from FAISS

    # Build LCEL chain: fill prompt → call LLM → parse to string
    chain = CUSTOM_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context":  _format_docs(docs),   # All chunks merged into one context string
        "question": question
    })
    return answer, docs


def _run_map_reduce_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    MAP-REDUCE strategy — each chunk is answered independently (map), then
    all partial answers are merged into one final answer (reduce).

    Best for: large volumes of retrieved text that won't fit in one prompt.
    """
    docs = retriever.invoke(question)

    # ── MAP step: ask the LLM about each chunk individually ───────────
    map_chain = MAP_PROMPT | llm | StrOutputParser()
    partial_answers = []
    for doc in docs:
        partial = map_chain.invoke({
            "context":  doc.page_content,
            "question": question
        })
        partial_answers.append(partial)

    # ── REDUCE step: combine all partial answers into one final answer ─
    combined_summaries = "\n\n".join(partial_answers)
    combine_chain = COMBINE_PROMPT | llm | StrOutputParser()
    answer = combine_chain.invoke({
        "summaries": combined_summaries,
        "question":  question
    })
    return answer, docs


def _run_refine_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    REFINE strategy — starts with the first chunk to produce an initial answer,
    then iterates over remaining chunks, refining the answer each time new
    relevant information is found.

    Best for: comprehensive, high-quality answers over long documents.
    """
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant documents found.", []

    # ── Initial answer from the first chunk ───────────────────────────
    init_chain = QUESTION_PROMPT | llm | StrOutputParser()
    answer = init_chain.invoke({
        "context":  docs[0].page_content,
        "question": question
    })

    # ── Refine with each subsequent chunk ─────────────────────────────
    refine_chain = REFINE_PROMPT | llm | StrOutputParser()
    for doc in docs[1:]:
        answer = refine_chain.invoke({
            "existing_answer": answer,          # Carry the running answer forward
            "context":         doc.page_content,
            "question":        question
        })
    return answer, docs


def _run_map_rerank_chain(retriever, llm, question: str) -> tuple[str, list]:
    """
    MAP-RERANK strategy — each chunk is independently answered and scored
    for relevance (0–10). The answer with the highest score wins.

    Best for: pinpointing the single most relevant document chunk.
    """
    docs = retriever.invoke(question)

    rerank_chain = RERANK_PROMPT | llm | StrOutputParser()
    best_answer = "No relevant document found."
    best_score  = -1

    for doc in docs:
        raw = rerank_chain.invoke({
            "context":  doc.page_content,
            "question": question
        })

        # Parse the structured "Answer: ... Score: ..." output from the LLM
        score_match  = re.search(r"Score:\s*(\d+)",                  raw)
        answer_match = re.search(r"Answer:\s*(.+?)(?=Score:|$)", raw, re.DOTALL)

        score  = int(score_match.group(1))        if score_match  else 0
        answer = answer_match.group(1).strip()    if answer_match else raw.strip()

        if score > best_score:
            best_score  = score
            best_answer = answer

    return best_answer, docs


def run_qa(vectorstore, model_name: str, chain_type: str, question: str) -> tuple[str, list]:
    """
    Dispatcher: selects the correct LCEL chain implementation based on chain_type,
    runs the query, and returns the answer with its source documents.

    Chain Types:
    ─────────────────────────────────────────────────────────────────
    "stuff"       Simple concat — all chunks in one prompt (fastest)
    "map_reduce"  Per-chunk answers combined into final answer
    "refine"      Answer iteratively improved across chunks (highest quality)
    "map_rerank"  Best-scored chunk answer wins
    ─────────────────────────────────────────────────────────────────

    Args:
        vectorstore (FAISS): Indexed vector store to search.
        model_name  (str):   Groq model (e.g. "llama-3.1-8b-instant").
        chain_type  (str):   Strategy name.
        question    (str):   User's question.

    Returns:
        answer      (str):  The LLM's answer.
        source_docs (list): Retrieved Document objects used as context.
    """
    # Initialize Groq LLM — reads GROQ_API_KEY from .env automatically
    llm       = ChatGroq(model=model_name, temperature=0)
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
# These functions are wired to Gradio UI buttons and dropdowns.
# They translate user actions into calls to the scraping/indexing/QA logic.


def on_discover_links(url: str):
    """
    Event handler for the 'Discover Links' button.

    Fetches the base URL and extracts all internal links.
    Returns the links as Dropdown choices (all pre-selected by default).

    Args:
        url (str): URL entered by the user.

    Returns:
        Tuple[gr.update, str]: Updated Dropdown choices + status message.
    """
    if not url.strip():
        return gr.update(choices=[], value=[]), "⚠️ Please enter a valid URL first."

    links, log = discover_links(url.strip())

    # Pre-select all discovered links in the CheckboxGroup
    return gr.update(choices=links, value=links), log


def on_scrape_and_index(url: str, selected_links: list, max_pages: int):
    """
    Streaming event handler for the 'Crawl & Index' button.

    This is a Gradio **generator function** — it yields UI updates in real time
    as each URL is scraped. The final yield stores the FAISS vector store in
    Gradio State so the 'Ask Questions' tab can use it.

    Args:
        url            (str):  The base URL to always include in scraping.
        selected_links (list): Links the user checked in the CheckboxGroup.
        max_pages      (int):  Maximum total pages to scrape.

    Yields:
        Tuple[str, str, object, int]:
            (scrape_log, status_text, vectorstore_or_None, chunks_count)
    """
    if not url.strip():
        yield "⚠️ No URL provided. Please enter a URL and discover links first.", \
              "❌ No URL", None, 0
        return

    # Build the final URL list: base URL first, then selected links, up to max_pages
    # dict.fromkeys preserves order while deduplicating
    all_urls = list(dict.fromkeys([url.strip()] + (selected_links or [])))[:max_pages]

    log = f"📋 Will crawl {len(all_urls)} URL(s):\n"
    for u in all_urls:
        log += f"   • {u}\n"
    log += "\n"

    yield log, "⏳ Starting scrape...", None, 0

    # ── Streaming scrape ──────────────────────────────────────────────
    # scrape_urls_streaming is itself a generator.
    # We relay its yields to Gradio, then capture the final text.
    cleaned_text = ""
    for scrape_log_update, scrape_status, final_text in scrape_urls_streaming(all_urls):
        combined_log = log + scrape_log_update   # Prepend our header to the scraping log
        yield combined_log, scrape_status, None, 0
        # On the last yield from the scraper, final_text will be non-empty
        if final_text:
            cleaned_text = final_text

    if not cleaned_text.strip():
        log_final = combined_log + "\n❌ No content extracted. Check the URL and try again."
        yield log_final, "❌ Nothing scraped", None, 0
        return

    # ── Build vector store ────────────────────────────────────────────
    combined_log += "\n🧠 Building vector store — embedding text chunks...\n"
    yield combined_log, "⏳ Building index...", None, 0

    try:
        vectorstore, num_chunks = build_vector_store(cleaned_text)
        combined_log += (
            f"✅ Vector store ready! {num_chunks} chunks indexed.\n\n"
            f"🎉 All done! Switch to the '💬 Ask Questions' tab to query the content."
        )
        # Final yield: pass vectorstore into Gradio State
        yield combined_log, f"✅ Indexed {num_chunks} chunks — Ready!", vectorstore, num_chunks

    except Exception as e:
        combined_log += f"\n❌ Indexing error: {e}"
        yield combined_log, f"❌ Error: {e}", None, 0


def on_ask_question(question: str, model_name: str, chain_type: str, vectorstore):
    """
    Event handler for the 'Ask' button.

    Builds a fresh RetrievalQA chain, runs the user's question against
    the FAISS vector store, and returns the answer + source documents.

    Args:
        question    (str):   The user's question.
        model_name  (str):   Groq model to use.
        chain_type  (str):   RAG chain strategy.
        vectorstore (FAISS): The indexed vector store from Gradio State.

    Returns:
        Tuple[str, str]: (answer_text, formatted_source_documents)
    """
    # Guard: vector store must be built before answering questions
    if vectorstore is None:
        return (
            "⚠️ No content indexed yet. Please scrape and index a website first.",
            ""
        )

    if not question.strip():
        return "⚠️ Please enter a question.", ""

    try:
        # Run the LCEL-based QA chain (replaces the deprecated RetrievalQA)
        answer, source_docs = run_qa(vectorstore, model_name, chain_type, question.strip())

        # Format the retrieved source documents for display
        if source_docs:
            sources_text = ""
            for i, doc in enumerate(source_docs, start=1):
                # Show a preview of each retrieved chunk (first 400 chars)
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

    Layout:
      - gr.State:  Hidden components that persist data between interactions
      - Tab 1:     Scrape & Index  (URL input → link discovery → crawl → index)
      - Tab 2:     Ask Questions   (question input → model/chain selection → answer)

    Returns:
        gr.Blocks: The fully configured Gradio application.
    """

    # ── Professional CSS — White / Black / Light Blue palette ────────
    PRO_CSS = """

    /* ── Base: clean white page ── */
    body, .gradio-container {
        background: #f8fafc !important;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        color: #111827 !important;
    }

    /* ── Hero header ── */
    .app-header {
        background: #1e3a8a;
        border-radius: 10px;
        padding: 26px 32px;
        margin-bottom: 4px;
        box-shadow: 0 2px 12px rgba(30,58,138,0.15);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin: 0 0 5px 0 !important;
        letter-spacing: -0.2px;
    }
    .app-header p {
        color: #bfdbfe !important;
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
        color: #dbeafe;
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
        border-bottom: 2px solid #bfdbfe !important;
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
        color: #1d4ed8 !important;
        border-bottom: 2px solid #1d4ed8 !important;
        background: transparent !important;
    }
    .tabs > .tab-nav button:hover:not(.selected) {
        color: #2563eb !important;
        background: #eff6ff !important;
    }

    /* ── Section labels ── */
    .section-label {
        color: #1e3a8a !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        border-left: 3px solid #3b82f6 !important;
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
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
        outline: none !important;
    }

    /* ── Slider ── */
    input[type=range] {
        accent-color: #2563eb !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Multiselect Dropdown — outer container ── */
    .multiselect, .multiselect > div {
        background: #ffffff !important;
        border: 1.5px solid #bfdbfe !important;
        border-radius: 8px !important;
        min-height: 52px !important;
    }
    .multiselect:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }

    /* ── Selected link chips (the blue tags shown after selection) ── */
    .multiselect .token, .multiselect [data-token] {
        background: #1d4ed8 !important;
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
    /* The ✕ remove button on each chip */
    .multiselect .token button, .multiselect [data-token] button {
        background: transparent !important;
        color: #bfdbfe !important;
        border: none !important;
        font-size: 0.9rem !important;
        cursor: pointer !important;
        padding: 0 2px !important;
    }
    .multiselect .token button:hover {
        color: #ffffff !important;
    }

    /* ── Dropdown list options ── */
    .multiselect .dropdown, .multiselect ul {
        background: #ffffff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
        max-height: 260px !important;
        overflow-y: auto !important;
    }
    .multiselect li, .multiselect .option {
        color: #1e3a8a !important;
        font-size: 0.83rem !important;
        padding: 8px 14px !important;
        cursor: pointer !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }
    .multiselect li:hover, .multiselect .option:hover {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
    }
    .multiselect li.selected, .multiselect .option.selected {
        background: #dbeafe !important;
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    /* ── Primary button ── */
    button.primary {
        background: #1d4ed8 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 10px 24px !important;
        box-shadow: 0 1px 6px rgba(29,78,216,0.3) !important;
        transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
    }
    button.primary:hover {
        background: #2563eb !important;
        box-shadow: 0 3px 12px rgba(37,99,235,0.35) !important;
        transform: translateY(-1px) !important;
    }
    button.primary:active { transform: translateY(0) !important; }

    /* ── Secondary button ── */
    button.secondary {
        background: #ffffff !important;
        color: #1d4ed8 !important;
        border: 1.5px solid #3b82f6 !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: background 0.15s, box-shadow 0.15s !important;
    }
    button.secondary:hover {
        background: #eff6ff !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 1px 6px rgba(29,78,216,0.12) !important;
    }

    /* ── Live log box ── */
    .log-box textarea {
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 0.82rem !important;
        background: #0f172a !important;
        color: #93c5fd !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
        line-height: 1.65 !important;
    }

    /* ── Answer box ── */
    .answer-box textarea {
        background: #f0f7ff !important;
        color: #111827 !important;
        font-size: 0.93rem !important;
        line-height: 1.75 !important;
        border: 1px solid #bfdbfe !important;
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
        background: #eff6ff !important;
        color: #1d4ed8 !important;
        border: 1px solid #bfdbfe !important;
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
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }
    """

    with gr.Blocks(title="Kamal AI Web Crawler & Q&A", css=PRO_CSS) as demo:

        # ── State components ──────────────────────────────────────────
        # These are invisible but store Python objects between button clicks.
        vectorstore_state  = gr.State(None)   # Holds the FAISS index object
        num_chunks_state   = gr.State(0)      # Tracks how many chunks were indexed

        # ── Hero Header ───────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>🕷️ Kamal AI Web Crawler &amp; Question Answering</h1>
            <p>Scrape any website &rarr; Index its content &rarr; Ask questions using AI</p>
            <div class="badge-row">
                <span class="badge">⚡ Groq LLM</span>
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
                    placeholder = "https://python.langchain.com/docs/introduction/",
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
                    variant = "secondary",   # Grey button (less prominent)
                    scale   = 1
                )
                discover_status = gr.Textbox(
                    label      = "Discovery Status",
                    interactive = False,
                    scale      = 3,
                    lines      = 2
                )

            gr.HTML('<p class="section-label">Step 2 — Select links to crawl</p>')

            # Multiselect dropdown — each selected link appears as a visible blue chip.
            # Much clearer than a CheckboxGroup: selected = blue tag, removed = click ✕
            links_selector = gr.Dropdown(
                label       = "Discovered Links",
                info        = "All links are selected by default. Click a link to remove it, or use the buttons below.",
                choices     = [],       # Populated dynamically after "Discover Links"
                value       = [],
                multiselect = True,     # Allow selecting multiple links
                interactive = True
            )

            # Quick-select helper buttons
            with gr.Row():
                select_all_btn   = gr.Button("✅ Select All",   variant="secondary", scale=1)
                deselect_all_btn = gr.Button("❌ Deselect All", variant="secondary", scale=1)

            gr.HTML('<p class="section-label">Step 3 — Crawl & build the search index</p>')

            crawl_btn = gr.Button(
                "🕷️ Crawl & Index Selected Pages",
                variant = "primary"    # Blue/accent button (main action)
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

            # Live log output — updates in real time as each URL is scraped
            scrape_log = gr.Textbox(
                label       = "Live Scraping Log",
                interactive = False,
                lines       = 18,
                max_lines   = 100,
                autoscroll  = True,
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

            # Question input takes up full width
            question_input = gr.Textbox(
                label       = "Your Question",
                placeholder = "e.g. What is LangChain used for?",
                lines       = 3,
                info        = "Type a question about the content you scraped and indexed."
            )

            with gr.Row():
                # Model dropdown — choose which Groq LLM to use
                model_dropdown = gr.Dropdown(
                    label   = "Groq Model",
                    choices = [
                        "llama-3.1-8b-instant",      # Fast, lightweight — good default
                        "llama-3.3-70b-versatile",   # More capable, slightly slower
                        "gemma2-9b-it",              # Google's Gemma 2 (9B)
                        "mixtral-8x7b-32768",        # Mixtral MoE with 32k context
                    ],
                    value = "llama-3.1-8b-instant",
                    info  = "Larger models give better answers but are slower."
                )

                # Chain type dropdown — controls how retrieved chunks are used
                chain_type_dropdown = gr.Dropdown(
                    label   = "RAG Chain Type",
                    choices = [
                        "stuff",        # Simple: all chunks in one prompt (fastest)
                        "map_reduce",   # Each chunk answered separately, then combined
                        "refine",       # Answer iteratively refined across chunks
                        "map_rerank",   # Best-scored chunk answer wins
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

            # Answer display — where the LLM's response appears
            answer_output = gr.Textbox(
                label        = "Answer",
                interactive  = False,
                lines        = 10,
                elem_classes = ["answer-box"]
            )

            # Collapsible panel showing the raw retrieved document chunks
            # Useful for understanding WHY the model gave a particular answer
            with gr.Accordion("📚 Retrieved Source Documents", open=False):
                gr.Markdown(
                    "_These are the text chunks retrieved from the vector store "
                    "and passed to the LLM as context._"
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
        # Connect UI components to their handler functions.

        # "Discover Links" button → fetch base URL, populate CheckboxGroup
        discover_btn.click(
            fn      = on_discover_links,
            inputs  = [url_input],
            outputs = [links_selector, discover_status]
        )

        # "Select All" → tick every discovered link
        select_all_btn.click(
            fn      = lambda choices: gr.update(value=choices),
            inputs  = [links_selector],
            outputs = [links_selector]
        )

        # "Deselect All" → untick every discovered link
        deselect_all_btn.click(
            fn      = lambda _: gr.update(value=[]),
            inputs  = [links_selector],
            outputs = [links_selector]
        )

        # "Crawl & Index" button → stream scraping progress, build FAISS index
        # This uses Gradio's streaming: the generator yields to multiple outputs
        crawl_btn.click(
            fn      = on_scrape_and_index,
            inputs  = [url_input, links_selector, max_pages_slider],
            outputs = [scrape_log, index_status, vectorstore_state, num_chunks_state]
        )

        # Update the chunks display whenever num_chunks_state changes
        num_chunks_state.change(
            fn      = lambda n: n,
            inputs  = [num_chunks_state],
            outputs = [chunks_display]
        )

        # "Ask" button → run RAG chain, display answer + sources
        ask_btn.click(
            fn      = on_ask_question,
            inputs  = [question_input, model_dropdown, chain_type_dropdown, vectorstore_state],
            outputs = [answer_output, sources_output]
        )

        # Allow pressing Enter in the question box to trigger the ask
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
    Launch the Gradio web application.

    Access the UI at: http://localhost:7860
    Share publicly (temporary link): set share=True below.
    """
    app = build_ui()
    app.launch(
        server_name = "0.0.0.0",    # Listen on all network interfaces
        server_port = 7860,          # Default Gradio port
        share       = False,         # Set True to get a public gradio.live link
        inbrowser   = True,          # Automatically open browser on launch
        theme       = gr.themes.Default(),  # Clean white base — CSS handles the rest
    )
