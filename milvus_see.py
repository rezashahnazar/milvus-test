from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")

# List all collections
collections = client.list_collections()
print("Collections:", collections)

# Get collection statistics
collection_name = "semantic_search_demo"
stats = client.get_collection_stats(collection_name)
print("\nCollection stats:", stats)

# Query all records (limit to first 10)
results = client.query(
    collection_name=collection_name,
    filter="",  # empty filter means get all
    output_fields=["id", "text"],
    limit=10,
)
print("\nSample records:")
for record in results:
    print(f"ID: {record['id']}, Text: {record['text']}")

# Get collection schema
schema = client.describe_collection(collection_name)
print("\nCollection schema:", schema)
