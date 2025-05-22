import os
from dotenv import load_dotenv

load_dotenv()  # Load .env if present (optional but recommended)

from typing import Optional, List, Tuple
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streamlit.streamlit_callback_handler import (
    StreamlitCallbackHandler,
)
from langchain_core.prompts import MessagesPlaceholder

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi           # pip install rank_bm25
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
import numpy as np


def create_groq_llama(
        model: str = "llama3-8b-8192",         # any model Groq hosts
        temperature: float = 0.0,
        top_p: float = 0.0,
        seed: int = 42,
        streaming: bool = False,
):
    """
    Return a ChatOpenAI instance pointed at Groq’s API.
    • Make sure GROQ_API_KEY is defined in your .env (or env vars).
    • Groq’s OpenAI-compatible endpoint is https://api.groq.com/openai/v1
    """
    return ChatOpenAI(
        model          = model,
        openai_api_key = os.getenv("GROQ_API_KEY"),
        base_url       = "https://api.groq.com/openai/v1",
        temperature    = temperature,
        model_kwargs   = {
            "top_p": top_p,   # 0 ⇒ greedy
            "seed" : seed
        },
        streaming      = streaming,
    )

def bm25_rerank(docs, query, keep=8):
    bm = BM25Okapi([d.page_content.split() for d in docs])
    scores = bm.get_scores(query.split())
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best]


def compress_chunk(doc, query, top_n=3, max_chars=400):
    """
    Keep the top‑n sentences in `doc.page_content` most relevant to `query`
    using BM25; also hard‑cap to `max_chars`.
    """
    sentences = sent_tokenize(doc.page_content)
    if len(sentences) > top_n:
        bm25   = BM25Okapi([s.split() for s in sentences])
        scores = bm25.get_scores(query.split())
        top_ids = sorted(range(len(scores)),
                         key=scores.__getitem__,
                         reverse=True)[:top_n]
        keep = " ".join(sentences[i] for i in sorted(top_ids))
    else:
        keep = " ".join(sentences)

    doc.page_content = keep[:max_chars]   # overwrite chunk text
    return doc                             # ← MUST return the Document



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def truncate_context_to_fit(context: str, max_tokens=7500):
    tokens = tokenizer(context)["input_ids"]
    if len(tokens) <= max_tokens:
        return context
    return tokenizer.decode(tokens[:max_tokens])



# ─── Dedup encoder loads only once and is reused ────────────────────
dedup_encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)  # ~70 MB, fast

def deduplicate_chunks(docs, cosine_thresh: float = 0.90):
    """
    Remove near-duplicate passages: keep a chunk only if its cosine
    similarity (MiniLM embeddings) to *every* previously kept chunk is < thresh.
    """
    if len(docs) <= 1:
        return docs

    texts = [d.page_content for d in docs]
    embs  = dedup_encoder.encode(texts, normalize_embeddings=True)

    keep, keep_idx = [], []
    for i, (doc, emb) in enumerate(zip(docs, embs)):
        if all(np.dot(emb, embs[j]) < cosine_thresh for j in keep_idx):
            keep.append(doc)
            keep_idx.append(i)
    return keep






from collections import defaultdict

def weighted_rrf(results, weights=None, k: int = 60):
    """
    Reciprocal-Rank Fusion with optional per-list weights.

    • `results`  – list[ list[Document] ]
    • `weights`  – list[float] or None
        If None, every list gets weight 1.0
        If shorter than `results`, the extras default to 1.0
    • `k`        – RRF constant (higher ⇒ flatter weighting)
    """
    if weights is None:
        weights = [1.0] * len(results)
    elif len(weights) < len(results):
        # pad with 1.0 for any additional result lists
        weights = list(weights) + [1.0] * (len(results) - len(weights))

    fused_scores = defaultdict(float)
    key2doc      = {}

    for i, docs in enumerate(results):
        w = weights[i]                        # safe: list is now long enough
        for rank, doc in enumerate(docs):
            key = doc.page_content            # stable text key
            fused_scores[key] += w / (rank + k)
            key2doc[key] = doc

    fused = [(key2doc[key], score)
             for key, score in fused_scores.items()]
    return sorted(fused, key=lambda x: x[1], reverse=True)




# ----------------------------------------------------------------------
# Hybrid retrieval helpers
# ----------------------------------------------------------------------
from rank_bm25 import BM25Okapi   # already imported earlier

def bm25_candidates(vs, query, k=32):
    """
    Return top-k BM25 Documents from the entire FAISS docstore.
    """
    # (id, Document) pairs so we can map index → Document
    corpus_items = list(vs.docstore._dict.items())          # [(id, Document), …]
    corpus_tokens = [d.page_content.split() for _, d in corpus_items]

    bm25   = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query.split())                 # 1 score per doc
    top_ix = np.argsort(scores)[::-1][:k]                   # indices of best docs

    return [corpus_items[i][1] for i in top_ix]             # list[Document]


def hybrid_retrieve(vs, query,
                    k_dense=8, k_bm25=32, final_k=12, rrf_k=60):
    """
    1. dense FAISS   -> k_dense docs
    2. BM25 keyword  -> k_bm25 docs
    3. fuse with RRF (k parameter = rrf_k)
    4. return top final_k fused docs
    """
    dense_docs = vs.similarity_search(query, k=k_dense)
    bm25_docs  = bm25_candidates(vs, query, k=k_bm25)
    fused_docs = reciprocal_rank_fusion([dense_docs, bm25_docs], k=rrf_k)
    top_docs   = [doc for doc,_ in fused_docs][:64]     # limit for speed
    return rerank_top(top_docs, query, k=final_k)





from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

def rerank_top(docs, query, k=8):
    pairs  = [[query, d.page_content[:512]] for d in docs]   # truncate long docs
    scores = reranker.predict(pairs, batch_size=32)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d,_ in ranked[:k]]

















def format_docs(docs):
    res = ""
    for doc in docs:
        escaped_page_content = doc.page_content.replace("\n", "\\n")
        res += "<doc>\n"
        res += f"  <content>{escaped_page_content}</content>\n"
        for m in doc.metadata:
            res += f"  <{m}>{doc.metadata[m]}</{m}>\n"
        res += "</doc>\n"
    return res


def convert_message(m):
    if m["role"] == "user":
        return HumanMessage(content=m["content"])
    elif m["role"] == "assistant":
        return AIMessage(content=m["content"])
    elif m["role"] == "system":
        return SystemMessage(content=m["content"])
    else:
        raise ValueError(f"Unknown role {m['role']}")


_condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)

_rag_template = """
You are a helpful research assistant.  Use **only** the context below to answer the question.

Answering rules (read carefully):
1. Reply with the exact phrase, number, or token(s) that answer the question—nothing else.
2. Do not add explanations, bullet points, or quotes.
3. If the correct answer is “no difference / unchanged / not significant”, reply exactly: No change

Context:
{context}

Question: {question}

Answer:
"""









ANSWER_PROMPT = ChatPromptTemplate.from_template(_rag_template)


def _format_chat_history(chat_history):
    def format_single_chat_message(m):
        if isinstance(m, HumanMessage):
            return "Human: " + m.content
        elif isinstance(m, AIMessage):
            return "Assistant: " + m.content
        elif isinstance(m, SystemMessage):
            return "System: " + m.content
        else:
            raise ValueError("Unknown message type")

    return "\n".join([format_single_chat_message(m) for m in chat_history])


# -------------------------------------------------------------------------
# Vectorstore loading (unchanged from your original code)
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_search_index(file_names: List[str], index_folder: str = "index") -> List[FAISS]:
    model_name = os.getenv("EMBEDDING_MODEL", "abhinand/MedEmbed-large-v0.1")  # keep one source of truth
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    search_indexes = []
    for file_name in file_names:
        search_index = FAISS.load_local(
            folder_path=index_folder,
            index_name=file_name + ".index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        search_indexes.append(search_index)
    return search_indexes



# -------------------------------------------------------------------------
# Basic RAG chain, replacing ChatOllama with OpenRouter


def get_standalone_question_from_chat_history_chain():
    """
    Rewrites the user question as a standalone query
    using the condense prompt chain + OpenRouter Llama.
    """
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | create_groq_llama()  # Replaces ChatOllama
        | StrOutputParser(),
    )
    return _inputs


def get_rag_chain(
    file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", retrieval_cb=None
):
    """
    Basic RAG chain that does:
      - condense question
      - retrieve docs from vectorstore
      - feed docs + question to LLM
    """
    vectorstore = get_search_index([file_name], index_folder)[0]
    retriever = vectorstore.as_retriever( search_kwargs={"k": 10} )

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    def context_update_fn(q):
        retrieval_cb([q])
        return q

    # 1) Condense user Q using OpenRouter Llama
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | create_groq_llama()
        | StrOutputParser(),
    )

    # 2) Retrieve docs & format
    _context = {
        "context": (
            itemgetter("standalone_question")
            | RunnablePassthrough(context_update_fn)
            | retriever               # vectorstore.as_retriever(...)
            | format_docs             # turns list[Document] into one string
        ),
        "question": lambda x: x["standalone_question"],
    }



    # 3) Final answer step
    conversational_qa_chain = (
        _inputs
        | _context
        | ANSWER_PROMPT
        | create_groq_llama()  # Replaces ChatOllama
    )

    return conversational_qa_chain


# -------------------------------------------------------------------------
# RAG Fusion Helpers
def reciprocal_rank_fusion(results: List[List], k=60):
    from langchain.load import dumps, loads

    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def get_search_query_generation_chain():
    """
    Generate multiple subqueries for RAG Fusion, using OpenRouter Llama.
    Cleans and filters the output for better retrieval quality.
    """
    from langchain.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    prompt = ChatPromptTemplate(
        input_variables=["original_query"],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template="You are a helpful assistant that generates multiple search queries based on a single input query."
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["original_query"],
                    template=    "Given the user query: {original_query}\n"
                                  "Generate exactly 6 standalone search queries.\n"
                                  "Only return the queries, numbered 1 to 6. No explanation."
                )
            ),
        ],
    )

    def clean_queries(lines):
        return [
            line.strip().lstrip("1234. ").strip('"')
            for line in lines
            if line.strip()
            and not line.lower().startswith("here are")
            and "search queries" not in line.lower()
        ]

    generate_queries = (
        prompt
        | create_groq_llama()
        | StrOutputParser()
        | (lambda x: clean_queries(x.split("\n")))
    )

    return generate_queries



def get_rag_fusion_chain(
    file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", retrieval_cb=None
):
    """
    RAG Fusion approach:
      - condense Q
      - generate multiple subqueries
      - retrieve docs for each subquery
      - fuse results
      - final LLM answer
    """
    vectorstore = get_search_index([file_name], index_folder)[0]
    retriever = vectorstore.as_retriever( search_kwargs={"k": 10} )
    query_generation_chain = get_search_query_generation_chain()

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    # Step 1: condense
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | create_groq_llama()
        | StrOutputParser(),
    )

    # Step 2: retrieve docs for each subquery, then fuse
    _context = {
        "context": RunnablePassthrough.assign(
            original_query=lambda x: x["standalone_question"]
        )
        | query_generation_chain
        | retrieval_cb
        | retriever.map()
        | reciprocal_rank_fusion
        | (lambda x: [item[0] for item in x])
        | format_docs,
        "question": lambda x: x["standalone_question"],
    }

    # Step 3: final LLM answer
    conversational_qa_chain = (
        _inputs | _context | ANSWER_PROMPT | create_groq_llama()
    )
    return conversational_qa_chain


# -------------------------------------------------------------------------
# Tools + Agents
def get_search_tool_from_indexes(
    search_indexes, st_cb: Optional[StreamlitCallbackHandler] = None
):
    from langchain.agents import tool
    from agent_helper import retry_and_streamlit_callback

    @tool
    @retry_and_streamlit_callback(st_cb=st_cb, tool_name="Content Search Tool")
    def search(query: str) -> str:
        """Search the contents of the source documents for the queries."""
        docs = []
        for index in search_indexes:
            docs += index.similarity_search(query, k=5)
        return format_docs(docs)

    return search


def get_lc_oai_tools(
    file_names: List[str],
    index_folder: str = "index",
    st_cb: Optional[StreamlitCallbackHandler] = None,
):
    from langchain.tools.render import format_tool_to_openai_tool

    search_indexes = get_search_index(file_names, index_folder)
    lc_tool = get_search_tool_from_indexes(search_indexes, st_cb=st_cb)
    oai_tool = format_tool_to_openai_tool(lc_tool)
    return [lc_tool], [oai_tool]


def get_agent_chain(
    file_names: List[str],
    index_folder="index",
    callbacks=None,
    st_cb: Optional[StreamlitCallbackHandler] = None,
):
    """
    Example of an Agent approach that uses a 'search' tool.
    Replaces ChatOllama with OpenRouter Llama as the LLM behind the agent.
    """
    if callbacks is None:
        callbacks = []

    lc_tools, oai_tools = get_lc_oai_tools(file_names, index_folder, st_cb)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant, use the search tool to answer the user's question and cite only the page number when you use information coming (like [p1]) from the source document.\nchat history: {chat_history}",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = create_groq_llama()

    # Format the chain as an agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        }
        | prompt
        | llm.bind(tools=oai_tools)  # pass the entire list of tools if multiple
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=lc_tools, verbose=True, callbacks=callbacks
    )
    return agent_executor


# -------------------------------------------------------------------------
# Additional multi-file RAG for your system
def get_rag_chain_files(
    file_names: List[str], index_folder: str = "index", retrieval_cb=None
):
    vectorstores = get_search_index(file_names, index_folder)
    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | create_groq_llama()
        | StrOutputParser(),
    )

    def multi_retriever_fusion(query: str) -> str:
        """
        Retrieve from each vectorstore (k=8), compress each chunk to the most relevant
        3 sentences (~≤400 chars), then concatenate with format_docs.
        """
        docs = []
        for vs in vectorstores:
            # smaller k because we merge across files
            retrieved = hybrid_retrieve(vs, query,
                            k_dense=8, k_bm25=32, final_k=8)

            # retrieved = bm25_rerank(retrieved, query, keep=10) # ← new line
            # sentence‑level compression
            retrieved = [compress_chunk(d, query, top_n=3, max_chars=400)
                        for d in retrieved]
            docs += retrieved

        # finally turn list[Document] → long string for the LLM
        return format_docs(docs)


    _context = {
        "context": (
            itemgetter("standalone_question")
            | RunnablePassthrough(retrieval_cb)
            | multi_retriever_fusion          # returns a *formatted* string already
        ),
        "question": lambda x: x["standalone_question"],
    }




    return _inputs | _context | ANSWER_PROMPT | create_groq_llama()


def get_rag_fusion_chain_files(
    file_names: List[str], index_folder: str = "index", retrieval_cb=None
):
    vectorstores = get_search_index(file_names, index_folder)
    query_generation_chain = get_search_query_generation_chain()

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | create_groq_llama()
        | StrOutputParser(),
    )

    def retrieve_and_fuse_queries(queries):
        """
        • Use only the first 3 sub‑queries (plenty for RAG‑Fusion).  
        • For each query, fetch k=5 chunks per vectorstore.  
        • Compress each chunk to its top‑3 sentences.  
        • Fuse with reciprocal rank fusion, return list[Document].
        """
        queries = queries[:4]                 # limit to 3 sub‑queries
        all_docs = []

        for q in queries:
            docs_q = []
            for vs in vectorstores:
                retrieved = hybrid_retrieve(vs, q,
                            k_dense=8, k_bm25=32, final_k=8) # lower k
                # retrieved = bm25_rerank(retrieved, q, keep=7)     # keep fewer per sub‑query

                # compress each chunk
                retrieved = [compress_chunk(d, q, top_n=3, max_chars=450)
                            for d in retrieved]
                docs_q += retrieved
            all_docs.append(docs_q)

        fused_docs  = weighted_rrf(all_docs, weights=[1.5, 1.0])
        docs_ranked = [doc for doc, _ in fused_docs[:20]]          # take 20 before dedup
        docs_unique = deduplicate_chunks(docs_ranked, cosine_thresh=0.90)
        return docs_unique[:10]                                    # final unique top-10

        # unpack (doc, score) tuples → doc list
        return [doc for doc, _ in fused_docs[:10]]  # Keep only top 10 docs


    _context = {
        "context": RunnablePassthrough.assign(
            original_query=lambda x: x["standalone_question"]
        )
        | query_generation_chain
        | retrieval_cb
        | (lambda queries: retrieve_and_fuse_queries(queries))
        | format_docs
        | truncate_context_to_fit,
        "question": lambda x: x["standalone_question"],
    }

    return _inputs | _context | ANSWER_PROMPT | create_groq_llama()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 200)
    print("RAG Chain with OpenRouter Integration")
