import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ============ CONFIGURATION ============
# ต้องตรงกับที่ตั้งไว้ใน build_vectorstore.py
VECTOR_DB_PATH = Path("npa_vectorstore")
COLLECTION_NAME = "npa_assets_v2"
EMB_MODEL_NAME = "BAAI/bge-m3"

def main():
    # 1. โหลด Model
    print(f"⏳ Loading embedding model: {EMB_MODEL_NAME}...")
    model = SentenceTransformer(EMB_MODEL_NAME)
    print("✅ Model loaded.")

    # 2. เชื่อมต่อ Database
    if not VECTOR_DB_PATH.exists():
        print(f"❌ Error: Database path '{VECTOR_DB_PATH}' not found. Please run build_vectorstore.py first.")
        return

    print(f"⏳ Connecting to ChromaDB at: {VECTOR_DB_PATH}...")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Connected to collection: '{COLLECTION_NAME}' ({collection.count()} items)")
    except Exception as e:
        print(f"❌ Error getting collection: {e}")
        return

    # ============ TEST LOGIC ============
    
    # ลองเปลี่ยนคำค้นหาตรงนี้ได้ครับ
    test_queries = [
        "บ้านหรู", 
        "บ้านราคาประหยัด", 
        "คอนโดติดรถไฟฟ้า"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        # แปลงข้อความ เป็น Vector
        query_vec = model.encode([query]).tolist()
        
        # ค้นหาใน DB
        results = collection.query(
            query_embeddings=query_vec,
            n_results=3, # เอามาดู 3 อันดับแรก
            include=["metadatas", "distances"]
        )
        
        # แสดงผล
        for i, meta in enumerate(results['metadatas'][0]):
            price = float(meta.get('asset_details_selling_price', 0))
            name = meta.get('name_th', 'N/A')
            # ตัดคำบรรยายให้สั้นลงหน่อยจะได้อ่านง่าย
            desc = str(meta.get('asset_details_description_th', ''))[:60].replace('\n', ' ') + "..."
            
            print(f"#{i+1} [Price: {price:,.0f}] {name}")
            print(f"   Desc: {desc}")
        print("-" * 50)

if __name__ == "__main__":
    main()