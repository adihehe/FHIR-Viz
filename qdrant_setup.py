# # from qdrant_client import QdrantClient
# # from qdrant_client.http.models import Distance, VectorParams

# # # Connect to Qdrant
# # client = QdrantClient(host="localhost", port=7000)

# # # Collection name
# # collection_name = "schema"

# # # Delete if exists
# # if client.collection_exists(collection_name=collection_name):
# #     client.delete_collection(collection_name=collection_name)



# # client.create_collection(
# #     collection_name="schema",
# #     vectors_config=VectorParams(size=768, distance=Distance.COSINE)
# # )







from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import json
import uuid
from qdrant_client.models import PointStruct


from vertexai.preview.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

PORT = ""
qdrant = QdrantClient("http://localhost", port=PORT)
COLLECTION_NAME = "schema"



def create_qdrant_collection():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=3072,  # for `text-embedding-3-small`
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists.")


# # Step 2: Load and chunk schema
# def chunk_schema(path="schema.json"):
#     with open(path, "r", encoding="utf-8-sig") as f:
#         content = f.read()
#         data = json.loads(content)

#     chunks = []
#     for section in data:
#         if "nodes" in section:
#             for node in section["nodes"]:
#                 chunks.append(f"NODE:\n{json.dumps(node, indent=2)}")
#         if "relationships" in section:
#             for rel in section["relationships"]:
#                 chunks.append(f"RELATIONSHIP:\n{json.dumps(rel, indent=2)}")

#     return chunks

def chunk_schema(path="properties_values.json", max_distinct=5):
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()
        data = json.loads(content)

    chunks = []
    for entry in data:
        label = entry.get("label", "UnknownLabel")
        prop = entry.get("property", "UnknownProperty")
        values = entry.get("distinctValues", [])
        limited_values = values[:max_distinct]

        # Create a formatted chunk string
        chunk_text = (
            f"LABEL: {label}\n"
            f"PROPERTY: {prop}\n"
            f"DISTINCT_VALUES (up to {max_distinct}):\n"
            f"{json.dumps(limited_values, indent=2)}"
        )
        chunks.append(chunk_text)

    return chunks





# Step 3: Embed a chunk using Gemini
def embed_text(text):
    embeddings = model.get_embeddings([text])
    # The get_embeddings method returns a list of TextEmbedding objects.
    # Each TextEmbedding object has an 'values' attribute which is the actual embedding vector.
    return embeddings[0].values # Extract the actual vector values


# Step 4: Upload to Qdrant
def upload_chunks_to_qdrant(chunks):
    points = []
    for chunk in chunks:
        vector = embed_text(chunk) # This now returns a list of floats
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk}
        ))

    # You need to initialize the Qdrant client before calling upload_points
    # For the purpose of this snippet, assuming 'client' is already defined globally as in your original code.
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points) # Use upsert for clarity and common practice
    print(f"✅ Uploaded {len(points)} schema chunks to Qdrant in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    create_qdrant_collection()
    chunks = chunk_schema("")  #add file path of schema.json or properties_values.json.
    upload_chunks_to_qdrant(chunks)




# from qdrant_client import QdrantClient
# qdrant_schema = QdrantClient(host="localhost", port=7000)
# try:
#     collections = qdrant_schema.get_collections()
#     print("Successfully connected to Qdrant. Collections:", collections)
# except Exception as e:
#     print(f"Failed to connect to Qdrant: {e}")
