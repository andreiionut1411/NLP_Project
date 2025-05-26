from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import DeletePayload

# === Config ===
collection_name = "nq_docs"
host = "localhost"
port = 6333

# === Connect to Qdrant ===
client = QdrantClient(host=host, port=port)

# === Define filter to match poison == True inside metadata ===
poison_filter = Filter(
    must=[
        FieldCondition(
            key="metadata.poison",  # Use dot notation to access nested field
            match=MatchValue(value=True)
        )
    ]
)

# === Search to find poisoned point IDs ===
search_result = client.scroll(
    collection_name=collection_name,
    scroll_filter=poison_filter,
    with_payload=True,
    limit=10000  # You can increase this if needed
)

points = search_result[0]

if not points:
    print("⚠️ No poisoned points found.")
else:
    poisoned_ids = [point.id for point in points]

    print(f"🧹 Found {len(poisoned_ids)} poisoned points. Deleting...")

    # Delete points by ID
    client.delete(
        collection_name=collection_name,
        points_selector=poisoned_ids
    )

    print("✅ Poisoned chunks deleted.")
