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
            

    def _get_all_entities_iter(self, output_fields: list[str], batch_size: int = 1000):
        """
        Helper to iterate over all entities in the collection.
        Not directly using iteration_expr because we want to fetch all.
        """
        total_entities = self.count_entities()
        fetched_count = 0
        pk_field_name = "id" 

        all_entities = []

        if total_entities == 0:
            return []

        
        offset = 0
        limit = batch_size 
        while fetched_count < total_entities:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="", # No filter, get all
                output_fields=output_fields,
                limit=limit,
                offset=offset
            )
            if not results:
                break
            all_entities.extend(results)
            fetched_count += len(results)
            offset += len(results)
            if len(results) < limit: 
                break
        return all_entities


    def get_all_file_paths(self, category_prefix: str = None) -> set[str]:
        file_paths = set()
        try:
            entities = self._get_all_entities_iter(output_fields=["file_path"])
            
            for entity in entities:
                file_path = entity.get('file_path')
                if file_path:
                    if category_prefix:
                        if file_path.startswith(category_prefix):
                            file_paths.add(file_path)
                    else:
                        file_paths.add(file_path)
            print(f"Retrieved {len(file_paths)} file paths. Category prefix: '{category_prefix}'.")
        except Exception as e:
            print(f"Error retrieving file paths: {e}")
        return file_paths

    def delete_vectors_by_file_paths(self, file_paths: list[str]) -> int:
        if not file_paths:
            return 0

        deleted_count_total = 0
        batch_size = 500
        
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            
            sanitized_expr_paths = []
            for p in batch_paths:
                sanitized_p = p.replace("'", "''") 
                sanitized_expr_paths.append(f"'{sanitized_p}'")
            
            file_path_expr_str = ", ".join(sanitized_expr_paths)
            query_expr_for_ids = f"file_path in [{file_path_expr_str}]"

            print(f"DELETE_WORKAROUND: For batch starting at index {i}, attempting to find IDs with query_expr: {query_expr_for_ids}")
            ids_to_delete = []
            try:
                query_results = self.client.query(
                    collection_name=self.collection_name,
                    filter=query_expr_for_ids, 
                    output_fields=['id'], 
                    limit=len(batch_paths) + 5
                )
                if query_results:
                    for entity in query_results:
                        if entity.get('id'):
                            ids_to_delete.append(entity['id'])
                    print(f"DELETE_WORKAROUND: Found {len(ids_to_delete)} IDs to delete: {ids_to_delete}")
                else:
                    print(f"DELETE_WORKAROUND: Query with filter '{query_expr_for_ids}' found NO entities to delete.")
            except Exception as e_query:
                print(f"DELETE_WORKAROUND: ERROR during query to find IDs with filter '{query_expr_for_ids}': {e_query}")
                continue 
            
            if not ids_to_delete:
                print(f"DELETE_WORKAROUND: No IDs found for batch (paths: {batch_paths}), skipping delete call for this batch.")
                continue
            
            try:
                print(f"MILVUS_DELETE (by direct IDs): Attempting to delete with IDs: {ids_to_delete}")
                delete_result = self.client.delete(collection_name=self.collection_name, ids=ids_to_delete)
                
                num_deleted_in_batch = len(delete_result) if isinstance(delete_result, list) else 0
                
                deleted_count_total += num_deleted_in_batch
                print(f"MILVUS_DELETE (by direct IDs): Successfully called delete for IDs: {ids_to_delete}. Resulting deleted PKs in batch: {delete_result}, Count: {num_deleted_in_batch}")

            except Exception as e:
                print(f"MILVUS_DELETE (by direct IDs): ERROR deleting vectors with IDs {ids_to_delete}: {e}")
        
        print(f"Total deleted vectors by file_paths (via ID lookup and direct ID deletion): {deleted_count_total}")
        return deleted_count_total

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