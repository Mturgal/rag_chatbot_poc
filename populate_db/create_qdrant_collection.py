from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="novartis_collection",
    vectors_config=VectorParams(size=1024, distance=Distance.DOT),
)
