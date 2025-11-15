from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

URL = "https://068d2af6-e623-468d-bb4e-05dfdc33efae.us-east4-0.gcp.cloud.qdrant.io:6333"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzcwOTkwODA2fQ.bU0nwx791EZRnBigfP56idMXtFZUcj-P_S5iUrANyOI"
COLLECTION = "experiences"


def main() -> None:
    client = QdrantClient(url=URL, api_key=API_KEY)
    print(f"Deleting collection '{COLLECTION}' if it exists...")
    client.delete_collection(collection_name=COLLECTION)
    print("Creating collection with dimension 2560 and cosine distance...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=2560, distance=Distance.COSINE),
    )
    print("Collection recreated with dim 2560.")


if __name__ == "__main__":
    main()
