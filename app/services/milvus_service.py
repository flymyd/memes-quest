from pymilvus import MilvusClient
from app.core.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    MILVUS_URI
)
import numpy as np

class MilvusService:
    def __init__(self):
        try:
            self.client = MilvusClient(uri=MILVUS_URI)
            print(f"Successfully connected to Milvus Lite using MilvusClient at {MILVUS_URI}")
        except Exception as e:
            print(f"Failed to connect to Milvus Lite using MilvusClient: {e}")
            raise

        self.collection_name = COLLECTION_NAME
        self.dimension = EMBEDDING_DIM
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        if not self.client.has_collection(self.collection_name):
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT", # Or "HNSW", "AUTOINDEX"
                metric_type="L2",
                params={"nlist": 128} 
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="L2",
                auto_id=True,
                primary_field_name="id",
                vector_field_name="embedding",
                index_params=index_params,
                overwrite=False
            )
            print(f"Collection '{self.collection_name}' created with IVF_FLAT index on 'embedding' field.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")
            

    def insert_vectors(self, vectors: list[list[float]], file_paths: list[str]) -> list[int]:
        if not vectors or not file_paths or len(vectors) != len(file_paths):
            raise ValueError("Vectors and file_paths must be non-empty and of the same length.")

        data_to_insert = [
            {"embedding": vec, "file_path": path} for vec, path in zip(vectors, file_paths)
        ]
        try:
            insert_result = self.client.insert(collection_name=self.collection_name, data=data_to_insert)
            print(f"Inserted {len(vectors)} vectors. IDs: {insert_result['ids']}")
            return insert_result['ids']
        except Exception as e:
            print(f"Error inserting vectors using MilvusClient: {e}")
            raise

    def search_vectors(self, query_vector: list[float], n: int) -> list[dict]:
        if not query_vector:
            raise ValueError("Query vector cannot be empty.")
        
        search_params = {"nprobe": 10}

        try:

            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="embedding",
                search_params=search_params,
                limit=n,
                output_fields=['file_path', 'id'],
            )
            
            hits_data = []
            for hits_for_one_query in results:
                for hit in hits_for_one_query:
                    entity_data = {}
                    if hasattr(hit, 'entity'): 
                        entity_data = hit.entity.to_dict() if hasattr(hit.entity, 'to_dict') else vars(hit.entity)
                    elif isinstance(hit, dict) and 'entity' in hit: 
                         entity_data = hit['entity']
                    elif isinstance(hit, dict): 
                         entity_data = hit

                    hits_data.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "file_path": entity_data.get('file_path')
                    })
            print(f"Search completed. Found {len(hits_data)} results.")
            return hits_data
        except Exception as e:
            print(f"Error searching vectors using MilvusClient: {e}")
            raise

    def count_entities(self) -> int:
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return stats['row_count']
        except Exception as e:
            print(f"Error counting entities using MilvusClient: {e}")
            return 0
            
    def drop_collection(self):
        try:
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' dropped using MilvusClient.")
        except Exception as e:
            print(f"Error dropping collection using MilvusClient: {e}")

# if __name__ == '__main__':
#     import sys
#     import os
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#     from app.core.config import EMBEDDING_DIM, COLLECTION_NAME, MILVUS_URI 
#     print("Testing MilvusService with MilvusClient...")
#     try:
#         milvus_service = MilvusService() 
#         print(f"Collection '{milvus_service.collection_name}' initial count: {milvus_service.count_entities()} entities.")

#         if milvus_service.count_entities() == 0:
#             print("Inserting dummy data...")
#             dummy_vectors = [[np.random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)] for _ in range(2)]
#             dummy_paths = ["test/image1.jpg", "example/image2.png"]
#             milvus_service.insert_vectors(dummy_vectors, dummy_paths)
#             print(f"After insert, collection has {milvus_service.count_entities()} entities.")

#         if milvus_service.count_entities() > 0:
#             print("Searching...")
#             query_vec = [np.random.uniform(-0.5, 0.5) for _ in range(EMBEDDING_DIM)]
#             search_results = milvus_service.search_vectors(query_vec, n=1)
#             print("Search results raw:", search_results)
#             if search_results:
#                 print(f"Found: {search_results[0]['file_path']} with distance {search_results[0]['distance']}")
#     except Exception as e:
#         print(f"Error during MilvusService (MilvusClient) test: {e}")
#     finally:
#         print("MilvusService (MilvusClient) test finished.") 