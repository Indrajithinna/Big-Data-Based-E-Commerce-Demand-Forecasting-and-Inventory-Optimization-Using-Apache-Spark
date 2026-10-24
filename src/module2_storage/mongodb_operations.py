# =============================================================================
# Module 2: Big Data Storage
# Script: mongodb_operations.py
# =============================================================================
# PURPOSE:
#   Provides MongoDB CRUD operations for storing and retrieving processed
#   data, analytics results, forecasts, and inventory recommendations.
#
#   MongoDB is used as the OPERATIONAL data store for:
#   - Cleaned sales records
#   - Aggregated analytics & KPI results
#   - Demand forecast outputs
#   - Inventory optimization recommendations
#
#   This demonstrates NoSQL storage concepts from the BIS701 syllabus.
#
# BIS701 RELEVANCE:
#   - MongoDB → NoSQL document-oriented database
#   - Demonstrates schema-flexible storage for semi-structured data
#   - Shows how Big Data results are persisted for application consumption
#
# NOTE: MongoDB is OPTIONAL. If not installed, the system falls back to
#       CSV/Parquet file storage. Set mongodb.enabled=false in config.
# =============================================================================

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config_loader import load_config


class MongoDBHandler:
    """
    Handler for MongoDB operations — stores and retrieves project data.
    
    Falls back gracefully if MongoDB is not available (mongodb.enabled=false).
    """
    
    def __init__(self, config=None):
        """
        Initialize MongoDB connection.
        
        Args:
            config: Configuration dictionary. Loaded from file if None.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.mongo_config = config.get("mongodb", {})
        self.enabled = self.mongo_config.get("enabled", False)
        self.client = None
        self.db = None
        
        if self.enabled:
            self._connect()
    
    def _connect(self):
        """Establish MongoDB connection."""
        try:
            from pymongo import MongoClient
            
            conn_str = self.mongo_config["connection_string"]
            db_name = self.mongo_config["database"]
            
            self.client = MongoClient(conn_str, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            self.db = self.client[db_name]
            print(f"[MongoDB] Connected to {conn_str}, database: {db_name}")
        except ImportError:
            print("[MongoDB] pymongo not installed. Install with: pip install pymongo")
            self.enabled = False
        except Exception as e:
            print(f"[MongoDB] Connection failed: {e}")
            print("[MongoDB] Falling back to file-based storage.")
            self.enabled = False
    
    def store_dataframe(self, df_pandas, collection_name, drop_existing=True):
        """
        Store a Pandas DataFrame into a MongoDB collection.
        
        NOTE: This expects a SMALL result DataFrame (aggregated results),
        not the full Big Data dataset. Large datasets stay in Spark/HDFS.
        
        Args:
            df_pandas: Pandas DataFrame to store.
            collection_name: Target MongoDB collection name.
            drop_existing: If True, drops existing collection first.
        
        Returns:
            int: Number of documents inserted, or 0 if MongoDB is disabled.
        """
        if not self.enabled:
            return 0
        
        collection = self.db[collection_name]
        
        if drop_existing:
            collection.drop()
        
        # Convert DataFrame to list of dictionaries
        records = df_pandas.to_dict("records")
        
        # Handle datetime serialization
        for record in records:
            for key, value in record.items():
                if hasattr(value, "isoformat"):
                    record[key] = value.isoformat()
        
        if records:
            result = collection.insert_many(records)
            count = len(result.inserted_ids)
            print(f"[MongoDB] Inserted {count:,} documents into '{collection_name}'")
            return count
        return 0
    
    def retrieve_collection(self, collection_name, query=None, projection=None):
        """
        Retrieve documents from a MongoDB collection.
        
        Args:
            collection_name: Collection to query.
            query: MongoDB query filter (dict). Default: all documents.
            projection: Fields to include/exclude.
        
        Returns:
            list: List of documents (dicts), or empty list if disabled.
        """
        if not self.enabled:
            return []
        
        collection = self.db[collection_name]
        cursor = collection.find(query or {}, projection or {})
        documents = list(cursor)
        
        # Remove MongoDB _id field for cleaner output
        for doc in documents:
            doc.pop("_id", None)
        
        return documents
    
    def store_kpis(self, kpi_dict):
        """
        Store KPI metrics with timestamp.
        
        Args:
            kpi_dict: Dictionary of KPI name → value.
        """
        if not self.enabled:
            return
        
        collection_name = self.mongo_config["collections"]["kpis"]
        collection = self.db[collection_name]
        
        kpi_dict["timestamp"] = datetime.now().isoformat()
        collection.insert_one(kpi_dict)
        print(f"[MongoDB] Stored KPI metrics to '{collection_name}'")
    
    def get_collection_stats(self):
        """Print statistics for all project collections."""
        if not self.enabled:
            print("[MongoDB] Disabled. No stats available.")
            return
        
        collections = self.mongo_config.get("collections", {})
        print("\n[MongoDB] Collection Statistics:")
        print("-" * 40)
        for name, coll_name in collections.items():
            try:
                count = self.db[coll_name].count_documents({})
                print(f"  {name}: {count:,} documents")
            except Exception:
                print(f"  {name}: (not found)")
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("[MongoDB] Connection closed.")


class FileStorageHandler:
    """
    Fallback file-based storage when MongoDB is not available.
    Stores results as CSV files in the configured results directory.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        self.results_dir = config["paths"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)
    
    def store_dataframe(self, df_pandas, name, format="csv"):
        """
        Save a Pandas DataFrame to file.
        
        Args:
            df_pandas: Pandas DataFrame.
            name: File name (without extension).
            format: 'csv' or 'json'.
        """
        if format == "csv":
            path = os.path.join(self.results_dir, f"{name}.csv")
            df_pandas.to_csv(path, index=False)
        elif format == "json":
            path = os.path.join(self.results_dir, f"{name}.json")
            df_pandas.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"[FileStorage] Saved {len(df_pandas):,} rows to {path}")
        return path


def get_storage_handler(config=None):
    """
    Factory function: returns MongoDB handler if enabled, else file-based.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        MongoDBHandler or FileStorageHandler instance.
    """
    if config is None:
        config = load_config()
    
    if config.get("mongodb", {}).get("enabled", False):
        handler = MongoDBHandler(config)
        if handler.enabled:
            return handler
    
    return FileStorageHandler(config)


if __name__ == "__main__":
    print("=" * 60)
    print("Module 2: Storage Layer Test")
    print("=" * 60)
    
    handler = get_storage_handler()
    print(f"Storage handler: {type(handler).__name__}")
    
    if isinstance(handler, MongoDBHandler):
        handler.get_collection_stats()
        handler.close()
    else:
        print(f"File storage directory: {handler.results_dir}")
