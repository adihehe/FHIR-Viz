from typing import TypedDict, Optional, List, Dict, Any
from neo4j import GraphDatabase
import openai
from qdrant_client import QdrantClient
import os
import json
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
import uuid  # Required for store_query_and_response()
from qdrant_client.http.models import SearchRequest, PointStruct  # Needed for vector search and upload


# ---- Agent State ---- #
class AgentState(TypedDict, total=False):
    query: str
    cypher: Optional[str]
    kg_data: Optional[List[dict]]
    code: Optional[str]
    output: Optional[str]
    success: Optional[bool]
    error: Optional[str]
    debug_attempts: Optional[int]

# ---- Config ---- #
NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = ""
QDRANT_COLLECTION = ""
OPENAI_KEY = ""  # Add your OpenAI API key here

openai.api_key = OPENAI_KEY
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
qdrant_client = QdrantClient("http://localhost:6333")

client = openai.OpenAI(api_key=OPENAI_KEY)

qdrant_queries = QdrantClient(host="localhost", port=6333)  # For query-response
qdrant_schema = QdrantClient(host="localhost", port=7000)   # For schema chunks


def embed_query(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]  # Must be a list for batching
    )
    return response.data[0].embedding 


def search_qdrant(collection: str, embedding: list, top_k: int = 1) -> str:
    qdrant = qdrant_queries #if collection == QDRANT_COLLECTION else qdrant_schema
    results = qdrant.search(
        collection_name=collection,
        search_request=SearchRequest(
            vector=embedding,
            limit=top_k
        )
    )
    if results:
        return results[0].payload.get("text")
    return None


def retrieve_top_schema_chunks(query: str, top_k: int = 5) -> list:
    embedding = embed_query(query)
    results = qdrant_schema.search(
        collection_name="schema",
        search_request=SearchRequest(
            vector=embedding,
            limit=top_k
        )
    )
    return [hit.payload["text"] for hit in results]


def store_query_and_response(query: str, response: str):
    embedding = embed_query(query)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": response, "query": query}
    )
    qdrant_queries.upload_points(collection_name=QDRANT_COLLECTION, points=[point])


# ---- Prompt Builder ---- #
def build_prompt(user_query, schema_str) -> List[Dict[str, str]]:
    context = f"""You're a Cypher expert for a FHIR-based medical knowledge graph built from the Synthea dataset in Neo4j.

Schema:
{schema_str}

User Question: "{user_query}"

Return a Cypher query that answers the question. 
No explanation, just the Cypher code.
"""
    return [{"role": "user", "content": context}]

# 

# ---- Cypher Executor ---- #
def execute_cypher(driver, cypher_query: str) -> List[dict]:
    with driver.session() as session:
        result = session.run(cypher_query)
        return [dict(record) for record in result]

# ---- Retrieval Node ---- #
def run_retrieval_node(state: AgentState) -> AgentState:
    user_query = state["query"]
    try:
        # STEP 1: Try retrieving from query-response Qdrant (DB 1)
        embedding = embed_query(user_query)
        result = search_qdrant(collection=QDRANT_COLLECTION, embedding=embedding)

        if result:
            print("\n--- RETRIEVAL RESULT (from Qdrant: queries DB) ---")
            print(result)
            approved = input("Approve this result? (y/n): ").strip().lower()
            if approved == "y":
                state.update({
                    "output": result,
                    "success": True
                })
                return state

        # STEP 2: Fall back to LLM with context from schema Qdrant (DB 2)
        print("⚠️ Query not found. Falling back to LLM...")

        schema_chunks = retrieve_top_schema_chunks(user_query, top_k=10)
        schema_context = "\n---\n".join(schema_chunks)
        prompt = build_prompt(user_query, schema_context)

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0,
        )
        cypher_query = completion.choices[0].message.content.strip()



        print("\n--- GENERATED CYPHER QUERY ---")
        print(cypher_query)
        approved = input("Approve this Cypher query? (y/n): ").strip().lower()
        if approved == "y":
            final = input("Edit the final Cypher (or press Enter to keep): ").strip()
            if final:
                cypher_query = final

            kg_data = execute_cypher(neo4j_driver, cypher_query)

            # Optional: store new query-response in Qdrant
            # store_query_and_response(user_query, cypher_query)

            state.update({
                "cypher": cypher_query,
                "kg_data": kg_data,
                "success": True
            })
        else:
            state.update({
                "error": "User rejected Cypher query.",
                "success": False
            })

    except Exception as e:
        state.update({
            "error": str(e),
            "success": False
        })

    return state


# ---- Coding Node ---- #

def run_coding_node(state: AgentState) -> AgentState:
    if not state.get("kg_data"):
        state.update({
            "error": "No data to generate code.",
            "success": False
        })
        return state

    prompt = [
    {"role": "system", "content": "You're a Python expert in medical data visualization."},
    {"role": "user", "content": f"""Given this Neo4j query result (FHIR data from Synthea):

{state['query']}
{state['kg_data']}

Generate a Python script using pandas and matplotlib/seaborn to visualize it.

- Use plt.savefig("chart.png") at the end.
- Output only valid Python code.
- No explanations or markdown."""}
]


    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0.2,
        )
        code = response.choices[0].message.content.strip()

        print("\n--- GENERATED CODE ---")
        print(code)
        approved = input("Approve this code? (y/n): ").strip().lower()
        if approved == "y":
            final = input("Edit the final code (or press Enter to keep): ").strip()
            if final:
                code = final
            state.update({"code": code, "success": True})
        else:
            state.update({"error": "Code rejected", "success": False})

    except Exception as e:
        state.update({"error": str(e), "success": False})

    return state



# ---- Execution Node ---- #
def run_executor_node(state: AgentState) -> AgentState:
    try:
        exec(state["code"], globals(), locals())
        if os.path.exists("chart.png"):
            state.update({"output": "chart.png", "success": True})
        else:
            state.update({"error": "chart.png not created", "success": False})
    except Exception as e:
        state.update({"error": f"Execution failed: {str(e)}", "success": False})

    return state

# ---- Debugging Node ---- #
def run_debugging_node(state: AgentState) -> AgentState:
    if state.get("debug_attempts", 0) >= 3:
        state.update({"error": "Too many debugging attempts.", "success": False})
        return state

    prompt = [
    {"role": "system", "content": "You are a Python code fixer."},
    {"role": "user", "content": f"""Fix the following visualization script.

Error:
{state['error']}

Code:
{state['code']}

Only return the corrected Python code. No extra text."""}
]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            temperature=0,
        )
        fixed_code = response.choices[0].message.content.strip()
        state.update({
            "code": fixed_code,
            "error": None,
            "success": True,
            "debug_attempts": state.get("debug_attempts", 0) + 1
        })
    except Exception as e:
        state.update({
            "error": f"Debugging failed: {str(e)}",
            "success": False,
            "debug_attempts": state.get("debug_attempts", 0) + 1
        })

    return state


# ---- LangGraph ---- #
graph = StateGraph(state_schema=AgentState)

graph.add_node("retrieval", RunnableLambda(run_retrieval_node))
graph.add_node("coding", RunnableLambda(run_coding_node))
graph.add_node("execution", RunnableLambda(run_executor_node))
graph.add_node("debugging", RunnableLambda(run_debugging_node))
graph.add_node("end", lambda x: x)

graph.set_entry_point("retrieval")
graph.add_edge("retrieval", "coding")
graph.add_edge("coding", "execution")
graph.add_edge("debugging", "execution")

def execution_transition_condition(state: AgentState):
    return "end" if state.get("success") else "debugging"

graph.add_conditional_edges("execution", execution_transition_condition, {
    "end": "end",
    "debugging": "debugging"
})

fsm = graph.compile()


if __name__ == "__main__":
    # Test Retrieval Node Only
    test_state = AgentState(query=input("Enter query: ").strip())
    result_state = run_retrieval_node(test_state)

    print("\n--- FINAL RETRIEVAL STATE ---")
    print(json.dumps(result_state, indent=2))













































# ---- Load Schema ---- #
# def get_schema_str_from_file(path="C:/Users/adity/OneDrive/Desktop/fhirVIZ/schema.json") -> str:
#     with open(path, "r", encoding="utf-8-sig") as f:
#         content = f.read()
#         data = json.loads(content)

#     # Extract the first item which has nodes and relationships
#     schema = data[0]  # List with a single dict
#     nodes = schema.get("nodes", [])
#     relationships = schema.get("relationships", [])

#     # Convert to strings (you can format this better if needed)
#     nodes_str = "\n".join([str(node) for node in nodes])
#     rels_str = "\n".join([str(rel) for rel in relationships])

#     return f"Nodes:\n{nodes_str}\n\nRelationships:\n{rels_str}"



# ---- Entry ---- #
# if __name__ == "__main__":
#     query = input("Query: ").strip()
#     result = fsm.invoke({"query": query})
#     print(result.get("output") or result.get("error"))