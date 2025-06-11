import os
import json
import uuid
from typing import TypedDict, Optional, List, Dict, Any

from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest

# --- Configuration ---
project_id = ""
location = ""

# --- Initialize Vertex AI SDK ---
aiplatform.init(project=project_id, location=location)

# --- Qdrant Client for schema retrieval ---
qdrant_schema = QdrantClient(host="localhost", port=0000)

# --- Step 1: Embed a query using Gemini embedding model ---
def embed_query(text):
    model = TextEmbeddingModel.from_pretrained("")
    embeddings = model.get_embeddings([text])
    return embeddings[0].values

# --- Step 2: Retrieve top schema chunks from Qdrant ---
def retrieve_top_schema_chunks(query, top_k=15):
    embedding = embed_query(query)
    results = qdrant_schema.search(
        collection_name="",
        query_vector=embedding,
        limit=top_k
    )
    return [hit.payload["text"] for hit in results]

# --- Step 3: Build prompt for the Gemini model ---
def build_prompt(user_query, schema_str):
    context = f"""You're a Cypher expert for a FHIR-based medical knowledge graph built from the Synthea dataset in Neo4j.

Schema:
{schema_str}

User Question: "{user_query}"

Return a Cypher query that answers the question.
No explanation, just the Cypher code."""
    return context

# --- Step 4: Run one query and return result ---
def run_retrieval_process(user_query) -> str:
    try:
        model = GenerativeModel("")
        schema_chunks = retrieve_top_schema_chunks(user_query, top_k=15)
        schema_context = "\n---\n".join(schema_chunks)
        prompt = build_prompt(user_query, schema_context)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- Step 5: Read queries from file and write results to another file ---
def process_queries(input_path="", output_path=""):
    with open(input_path, "r", encoding="utf-8") as infile:
        content = infile.read()
        queries = [q.strip() for q in content.split(",") if q.strip()]


    with open(output_path, "w", encoding="utf-8") as outfile:
        for i, query in enumerate(queries, start=1):
            print(f"Processing Query {i}: {query}")
            result = run_retrieval_process(query)
            outfile.write(f"Query {i}: {query}\n")
            outfile.write("Cypher Query:\n")
            outfile.write(result + "\n")
            outfile.write("=" * 80 + "\n")

# Run the process
process_queries()






