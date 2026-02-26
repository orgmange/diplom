import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host="localhost", port=6333)
collection_name = "documents"

try:
    count_res = client.count(collection_name=collection_name, exact=True)
    print(f"Total points: {count_res.count}")
    
    # Try with count_filter
    examples_count = client.count(
        collection_name=collection_name,
        exact=True,
        count_filter=models.Filter(must=[models.FieldCondition(key="is_example", match=models.MatchValue(value=True))])
    )
    print(f"Example points: {examples_count.count}")
    
    res = client.scroll(collection_name=collection_name, limit=100)
    for p in res[0]:
        print(f"ID: {p.id}, is_example: {p.payload.get('is_example')}, filename: {p.payload.get('filename')}")
except Exception as e:
    print(f"Error: {e}")
