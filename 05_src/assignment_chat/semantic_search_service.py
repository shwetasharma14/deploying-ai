from langchain.tools import tool
import re
from pathlib import Path

import pandas as pd
import chromadb
from dotenv import load_dotenv
import os
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

import traceback

env_dir = Path(__file__).resolve().parent
load_dotenv(env_dir / ".env")
load_dotenv(env_dir / ".secrets")

api_gateway_key = os.getenv("API_GATEWAY_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",   # or any embedding model your gateway supports
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key=api_gateway_key,
)



# ==================== SERVICE 2: SEMANTIC SEARCH ====================
# Implementation follows the class lab pattern from 02_4_embeddings_api, 02_5_vectordb, and 02_6_embeddings_at_scale
_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_data"
_KNOWLEDGE_CSV = _CHROMA_DIR / "knowledge_base.csv"
_COLLECTION_NAME = "ai_knowledge_base"
_semantic_client = None
_semantic_collection = None


def _ensure_chroma_dir():
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def _load_knowledge_base():
    _ensure_chroma_dir()
    if not _KNOWLEDGE_CSV.exists():
        raise FileNotFoundError(
            f"Knowledge base CSV not found at {_KNOWLEDGE_CSV}. "
            "Please make sure the file exists before running the app."
        )
    df = pd.read_csv(_KNOWLEDGE_CSV)
    return df


def _tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def _lexical_query_candidates(query: str, knowledge_df: pd.DataFrame, top_n: int = 5):
    query_tokens = set(_tokenize(query))
    candidates = []
    for _, row in knowledge_df.iterrows():
        text = f"{row['title']} {row['topic']} {row['content']}"
        row_tokens = set(_tokenize(text))
        overlap = len(query_tokens & row_tokens)
        if overlap > 0:
            candidates.append((str(row["id"]), overlap))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in candidates[:top_n]]


def _format_search_results(documents, metadatas, distances):
    response = "📚 Semantic Search Results:\n\n"
    for i, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        score = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
        response += (
            f"{i}. {meta.get('title', 'Untitled')} ({meta.get('topic', 'General')})\n"
            f"   {doc}\n"
            f"   Relevance score: {score:.0f}%\n\n"
        )
    return response.strip()


def _lexical_fallback(query: str) -> str:
    print("DEBUG: Loading knowledge base...")
    knowledge_df = _load_knowledge_base()
    query_tokens = set(_tokenize(query))
    fallback = []
    for _, row in knowledge_df.iterrows():
        text = f"{row['title']} {row['topic']} {row['content']}"
        row_tokens = set(_tokenize(text))
        overlap = len(query_tokens & row_tokens)
        if overlap > 0:
            fallback.append((row['title'], row['topic'], row['content'], overlap))

    if not fallback:
        return "No relevant information found in the knowledge base for your query."

    fallback.sort(key=lambda item: item[3], reverse=True)
    response = "📚 Knowledge Base Results (lexical fallback):\n\n"
    for i, (title, topic, content, _) in enumerate(fallback[:3], 1):
        response += f"{i}. {title} ({topic})\n   {content}\n\n"
    return response.strip()


from chromadb.utils import embedding_functions
import traceback

def _get_semantic_collection():
    global _semantic_client, _semantic_collection

    try:
        print("DEBUG: Entering _get_semantic_collection()")

        _ensure_chroma_dir()
        _semantic_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

        print("DEBUG: Initializing Chroma OpenAIEmbeddingFunction...")
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_gateway_key,
            api_base="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
            model_name="text-embedding-3-small",
        )

        print("DEBUG: Listing existing collections...")
        existing = [c.name for c in _semantic_client.list_collections()]
        print("DEBUG: Existing collections:", existing)

        if _COLLECTION_NAME in existing:
            print("DEBUG: Collection exists, loading with embedding_fn...")
            _semantic_collection = _semantic_client.get_collection(
                name=_COLLECTION_NAME,
                embedding_function=embedding_fn,
            )
            return _semantic_collection

        print("DEBUG: Loading knowledge base CSV...")
        knowledge_df = _load_knowledge_base()

        print("DEBUG: Creating Chroma collection with embedding_fn...")
        _semantic_collection = _semantic_client.create_collection(
            name=_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"source": "assignment_chat"},
        )

        print("DEBUG: Adding documents to Chroma collection...")
        _semantic_collection.add(
            ids=knowledge_df["id"].astype(str).tolist(),
            documents=knowledge_df["content"].tolist(),
            metadatas=knowledge_df[["title", "topic"]].to_dict(orient="records"),
        )

        print("DEBUG: Collection created successfully.")
        return _semantic_collection

    except Exception:
        print("🔥 ERROR IN _get_semantic_collection() 🔥")
        print(traceback.format_exc())
        raise


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base using a persistent Chroma semantic index.
    
    Args:
        query: Search query (e.g., "machine learning", "embeddings", "RAG")
    
    Returns:
        Top 3 most relevant results from the knowledge base
    """
    if not query or not query.strip():
        return "Please enter a search query to look up in the knowledge base."

    try:
        print("DEBUG: Loading knowledge base...")
        knowledge_df = _load_knowledge_base()
        print("DEBUG: Getting semantic collection...")
        collection = _get_semantic_collection()
        print("DEBUG: Running lexical candidate search...")
        candidate_ids = _lexical_query_candidates(query, knowledge_df, top_n=6)

        print("DEBUG: Querying Chroma now...")

        try:
            if candidate_ids:
                result = collection.query(
                    query_texts=[query],
                    ids=candidate_ids,
                    n_results=3,
                    include=["documents", "metadatas", "distances"],
                )
            else:
                result = collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["documents", "metadatas", "distances"],
                )

            print("DEBUG: Chroma query result:", result)

        except Exception as e:
            print("🔥 ERROR DURING CHROMA QUERY 🔥")
            print(traceback.format_exc())
            return "ERROR during Chroma query:\n" + traceback.format_exc()


        documents = result.documents[0] if getattr(result, 'documents', None) else []
        metadatas = result.metadatas[0] if getattr(result, 'metadatas', None) else []
        distances = result.distances[0] if getattr(result, 'distances', None) else []

        if not documents:
            return _lexical_fallback(query)

        return _format_search_results(documents, metadatas, distances)
    except Exception as e:
        print("🔥 FULL ERROR TRACEBACK 🔥")
        print(traceback.format_exc())
        return "Error:\n" + traceback.format_exc()
