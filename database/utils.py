import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import utils
from chromadb.config import Settings

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _get_config_value(name: str):
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def create_chroma_client(persist_directory: str = "./chroma_db"):
    """Create the Chroma client used by the Streamlit app.
    """
    api_key = _get_config_value("CHROMA_API_KEY")
    tenant = _get_config_value("CHROMA_TENANT")
    database = _get_config_value("CHROMA_DATABASE")

    if api_key and tenant and database:
        logging.info("Using Chroma Cloud database '%s' for tenant '%s'.", database, tenant)
        return utils.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )

    if os.getenv("STREAMLIT_SERVER_PORT") or os.getenv("RENDER"):
        raise RuntimeError(
            "Chroma Cloud is not configured. Set CHROMA_API_KEY, CHROMA_TENANT, "
            "and CHROMA_DATABASE in your Streamlit/Render environment."
        )

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    logging.info("Using local Chroma persistence at '%s'.", persist_path)
    return utils.Client(Settings(persist_directory=str(persist_path), is_persistent=True))


def build_chroma_collection(
        chunks: List[Dict[str, Any]],
        embedding_function: Callable[[List[str]], List[List[float]]],
        collection_name: str = "document_chunks",
        persist_directory: str = "./chroma_db",
):
    """Builds or opens a Chroma collection and stores chunk embeddings.

    Args:
        chunks (List[Dict[str, Any]]): A list of chunk dictionaries containing text and metadata.
        embedding_function (Callable[[List[str]], List[List[float]]]): Function to convert texts into embeddings.
        collection_name (str): Chroma collection name.
        persist_directory (str): Directory to persist the collection.

    Returns:
        chromadb.api.models.Collection.Collection: A Chroma collection with stored embeddings.
    """
    client = create_chroma_client(persist_directory)
    collection = client.get_or_create_collection(name=collection_name)

    ids = [str(index) for index, _ in enumerate(chunks)]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk.get("metadata", {}) for chunk in chunks]
    embeddings = embedding_function(documents)

    if collection.count() > 0:
        client.delete_collection(name=collection_name)
        collection = client.get_or_create_collection(name=collection_name)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    logging.info(f"Chroma collection '{collection_name}' built with {len(chunks)} documents.")
    return collection


def query_chroma_collection(
    collection: Any,
    query: str,
    embedding_function: Callable[[List[str]], List[List[float]]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    
    """Queries a Chroma collection and returns scored results."""
    query_embedding = embedding_function([query])
    search = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances", "ids"],
    )

    results: List[Dict[str, Any]] = []
    for doc, metadata, distance, idx in zip(
        search["documents"][0],
        search["metadatas"][0],
        search["distances"][0],
        search["ids"][0],
    ):
        results.append(
            {
                "id": int(idx),
                "score": float(distance),
                "text": doc,
                "metadata": metadata,
            }
        )

    return results




