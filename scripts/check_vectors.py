import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host="localhost", port=6333)
collection_name = "documents"

try:
    count_res = client.count(collection_name=collection_name, exact=True)
    print(f"Total points: {count_res.count}")
    
    res = client.scroll(collection_name=collection_name, limit=100)
    for p in res[0]:
        print(f"ID: {p.id}, filename: {p.payload.get('filename')}")
except Exception as e:
    print(f"Error: {e}")
