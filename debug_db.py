import chromadb
from pathlib import Path
import pandas as pd

# 1. เชื่อมต่อ DB
db_path = Path("npa_vectorstore")
client = chromadb.PersistentClient(path=str(db_path))
collection = client.get_collection(name="npa_assets_v2")

# 2. ดึงข้อมูลทั้งหมดออกมาดู
results = collection.get(include=["metadatas"])

print(f"📦 มีข้อมูลทั้งหมด: {len(results['ids'])} รายการ\n")
print(f"{'ID':<5} | {'TYPE_ID':<8} | {'PRICE':<12} | {'NAME'}")
print("-" * 60)

# 3. ไล่ดูทีละรายการ
found_target = False
for i, meta in enumerate(results['metadatas']):
    price = meta.get('asset_details_selling_price', 0)
    name = meta.get('name_th', 'N/A')
    type_id = meta.get('asset_type_id', 'N/A')
    
    # แปลงราคาเป็นล้านเพื่อให้ดูง่าย
    price_mb = float(price) / 1_000_000 if price else 0
    
    print(f"{i+1:<5} | {type_id:<8} | {price_mb:,.1f}M      | {name[:40]}")

    # เช็คบ้านเป้าหมาย (64 ล้าน)
    if 64 < price_mb < 65:
        found_target = True
        print(f"   >>> 🎯 เจอบ้านเป้าหมายแล้ว! ID ของมันคือ: {type_id} <<<")

print("-" * 60)
if not found_target:
    print("❌ ไม่พบบ้านราคา ~64 ล้าน ใน Database เลย! (ข้อมูลอาจตกหล่น)")