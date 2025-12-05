import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import argparse
import logging
import math
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_vectorstore")

# Asset type mapping
ASSET_TYPE_MAPPING = {
    "คอนโด": "อาคารชุด",
    "บ้าน": "บ้าน",
    "ทาวน์โฮม": "ทาวน์เฮ้าส์/ทาวน์โฮม",
    "อาคารพาณิชย์": "อาคารพาณิชย์"
}

# ✅ POI Config
POI_CONFIG = {
    "bts_station": {"radius": 3000, "weight": 1.2, "curve": "exponential"}, 
    "mrt": {"radius": 3000, "weight": 1.2, "curve": "exponential"},
    "train_station": {"radius": 2000, "weight": 0.5, "curve": "exponential"},
    "bus_station": {"radius": 2000, "weight": 0.5, "curve": "exponential"},
    "convenience_store": {"radius": 1000, "weight": 0.5, "curve": "exponential"},
    "market": {"radius": 1500, "weight": 0.4, "curve": "linear"},
    "supermarket": {"radius": 2000, "weight": 0.5, "curve": "linear"},
    "shopping_mall": {"radius": 3000, "weight": 1.1, "curve": "linear"},
    "community_mall": {"radius": 2000, "weight": 0.7, "curve": "linear"},
    "restaurant": {"radius": 1000, "weight": 0.4, "curve": "linear"},
    "cafe": {"radius": 1000, "weight": 0.4, "curve": "linear"},
    "hospital": {"radius": 3000, "weight": 0.7, "curve": "linear"},
    "park": {"radius": 3000, "weight": 0.6, "curve": "linear"}, 
    "gym": {"radius": 2000, "weight": 0.5, "curve": "linear"},
    "spa": {"radius": 2000, "weight": 0.2, "curve": "linear"},
    "veterinary": {"radius": 2000, "weight": 0.5, "curve": "linear"},
    "school": {"radius": 3000, "weight": 0.5, "curve": "linear"},
    "university": {"radius": 3000, "weight": 0.3, "curve": "linear"},
    "river": {"radius": 1500, "weight": 0.4, "curve": "linear"}, 
    "beach": {"radius": 3000, "weight": 0.0, "curve": "linear"}, 
    "viewpoint": {"radius": 3000, "weight": 0.2, "curve": "linear"},
    "temple": {"radius": 1500, "weight": 0.1, "curve": "linear"},
    "museum": {"radius": 5000, "weight": 0.1, "curve": "linear"},
    "tourist_attraction": {"radius": 3000, "weight": 0.2, "curve": "linear"},
    "hotel": {"radius": 2000, "weight": 0.1, "curve": "linear"},
    "golf_course": {"radius": 5000, "weight": 0.2, "curve": "linear"},
}

def fix_asset_type(row):
    """Fix asset type text based on name and description"""
    name = str(row.get('name_th', '')).lower()
    desc = str(row.get('asset_details_description_th', '')).lower()
    eng_name = str(row.get('name_en', '')).lower()
    current_type = str(row.get('fixed_type', '')).strip() 
    
    if current_type and current_type != 'nan':
        return current_type

    if any(x in eng_name or x in name or x in desc for x in ["condominium", "คอนโด", "อาคารชุด", "ห้องชุด"]):
        return ASSET_TYPE_MAPPING["คอนโด"]
    if any(x in eng_name or x in name for x in ["townhouse", "townhome", "ทาวน์เฮ้าส์", "ทาวน์โฮม"]):
        return ASSET_TYPE_MAPPING["ทาวน์โฮม"]
    if any(x in eng_name or x in name for x in ["commercial", "shophouse", "อาคารพาณิชย์", "ตึกแถว"]):
        return ASSET_TYPE_MAPPING["อาคารพาณิชย์"]

    return ASSET_TYPE_MAPPING["บ้าน"]

def compute_poi_percentiles(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    logger.info("Calculating POI percentiles...")
    percentiles_data = {}
    found_cols = 0
    for col_name in POI_CONFIG.keys():
        if col_name in df.columns:
            distances = df[col_name].dropna()
            distances = pd.to_numeric(distances, errors='coerce').dropna()
            if not distances.empty:
                percentiles_data[col_name] = {
                    "p10": np.percentile(distances, 10),
                    "p50": np.percentile(distances, 50),
                    "p90": np.percentile(distances, 90),
                }
                found_cols += 1
    return percentiles_data

def compute_lifestyle_score(row: pd.Series, percentiles: Dict[str, Dict[str, float]]) -> float:
    total_score = 0
    for col_name, config in POI_CONFIG.items():
        if col_name in row and pd.notna(row[col_name]):
            try:
                distance = float(row[col_name])
            except:
                continue 
            radius = config["radius"]
            weight = config["weight"]
            if distance <= radius:
                score = 0
                if config.get("curve") == "exponential":
                    score = (1 - (distance / radius)) ** 2
                else:
                    score = 1 - (distance / radius)
                total_score += max(0, score * weight)
    total_weight = sum(c['weight'] for c in POI_CONFIG.values())
    if total_weight == 0: return 0.0
    return min(10, (total_score / total_weight) * 10)

def extract_features(row: pd.Series) -> Dict[str, Any]:
    features = {}
    features['bedroom'] = row.get('asset_details_number_of_bedrooms', 'N/A')
    features['bathroom'] = row.get('asset_details_number_of_bathrooms', 'N/A')
    desc_th = str(row.get('asset_details_description_th', '')).lower()
    desc_en = str(row.get('asset_details_description_en', '')).lower()
    
    if "สัตว์เลี้ยง" in desc_th or "pet-friendly" in desc_en or "pet friendly" in desc_en:
        features['pet_friendly'] = True
    else:
        features['pet_friendly'] = None 
    return features

# ✅ ฟังก์ชันใหม่: แปลงสีผังเมืองโดยใช้ "รหัสไปรษณีย์" เช็คโซน EEC
def get_area_color_meaning(color_text, postal_code):
    text = str(color_text).strip()
    # แปลงรหัสไปรษณีย์เป็น string แล้วดึง 2 ตัวแรก
    p_code = str(postal_code).strip()[:2] 
    
    # 🔍 Logic: เช็คจังหวัดจากรหัสไปรษณีย์
    # 20 = ชลบุรี, 21 = ระยอง, 24 = ฉะเชิงเทรา
    is_eec = p_code in ["20", "21", "24"]

    if "จุด" in text or "ขาว" in text or "ลาย" in text:
        if is_eec:
            return "พื้นที่ EEC รองรับการพัฒนาเมือง ทำเลศักยภาพสูง อนาคตไกล"
        else:
            return "ย่านอนุรักษ์ศิลปวัฒนธรรม เมืองเก่า บรรยากาศคลาสสิคและดั้งเดิม"

    elif "เหลือง" in text:
        return "ย่านที่อยู่อาศัยหนาแน่นน้อย บรรยากาศเงียบสงบ เป็นส่วนตัว"
    elif "ส้ม" in text:
        return "ย่านที่อยู่อาศัยปานกลาง เดินทางสะดวก ใกล้แหล่งความเจริญ"
    elif "แดง" in text:
        return "ย่านพาณิชยกรรม ศูนย์กลางธุรกิจ คึกคัก พลุกพล่าน"
    elif "ม่วง" in text:
        if is_eec: return "เขตส่งเสริมอุตสาหกรรมพิเศษ EEC"
        return "ย่านอุตสาหกรรมและคลังสินค้า ใกล้แหล่งงาน"
    elif "เขียว" in text:
        return "พื้นที่ชนบทและเกษตรกรรม บรรยากาศธรรมชาติ อากาศดี"
    
    return ""

def main(csv_path: str, db_path: str, model_name: str, collection_name: str):
    logger.info(f"🚀 Starting vector store build from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"❌ Error loading CSV: {e}")
        return
    logger.info(f"Loaded {len(df)} rows from CSV.")
    
    logger.info("⚙️ Processing data...")
    df['asset_type_fixed'] = df.apply(fix_asset_type, axis=1)
    percentiles = compute_poi_percentiles(df)
    df['lifestyle_score'] = df.apply(lambda row: compute_lifestyle_score(row, percentiles), axis=1)
    df_features = df.apply(extract_features, axis=1)
    df = pd.concat([df, pd.json_normalize(df_features)], axis=1)
    logger.info("✅ Processing complete.")

    logger.info("Embedding text...")
    
    df['price_text'] = df['asset_details_selling_price'].fillna(0).astype(float).apply(
        lambda x: f"ราคา {x:,.0f} บาท" if x > 0 else ""
    )
    
    # สร้าง Location Text
    df['location_text'] = df.apply(
        lambda row: " ".join([
            str(row['location_village_th']) if pd.notna(row.get('location_village_th')) else "",
            str(row['location_road_th']) if pd.notna(row.get('location_road_th')) else "",
            str(row['location_postal_code']) if pd.notna(row.get('location_postal_code')) else ""
        ]).strip(), 
        axis=1
    )

    # ⚠️ FIX: ส่ง Postal Code ไปเช็ค EEC แทนการเดาชื่อ
    if 'asset_details_area_color' in df.columns:
        df['zone_desc'] = df.apply(
            lambda x: get_area_color_meaning(x['asset_details_area_color'], x.get('location_postal_code', '')), 
            axis=1
        )
    else:
        df['zone_desc'] = ""

    # รวมร่าง Text
    df['text_for_embedding'] = df['name_th'].fillna('') + " | " + \
                               df['asset_type_fixed'].fillna('') + " | " + \
                               df['price_text'] + " | " + \
                               df['location_text'] + " | " + \
                               df['asset_details_description_th'].fillna('') + " | " + \
                               df['zone_desc'] 
                               
    texts = df['text_for_embedding'].tolist()
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    logger.info("✅ Embeddings generated.")

    logger.info("Preparing metadata for ChromaDB...")
    
    metadata_cols = [
            'id', 'name_th', 'name_en', 'asset_type_fixed', 
            'location_road_th', 'location_village_th', 'location_postal_code', 
            'asset_details_selling_price', 'location_latitude', 'location_longitude',
            'asset_details_description_th', 'asset_details_description_en',
            'bedroom', 'bathroom', 'pet_friendly', 
            'lifestyle_score',
            'asset_type_id',
            'zone_desc' 
        ]
    
    for poi_key in POI_CONFIG.keys():
        metadata_cols.append(poi_key)
        metadata_cols.append(f"{poi_key}_name")

    final_metadata_cols = [col for col in metadata_cols if col in df.columns]
    df_metadata = df[final_metadata_cols].copy()
    
    for col in df_metadata.columns:
        if df_metadata[col].dtype == 'object':
            df_metadata[col] = df_metadata[col].fillna('N/A')
        elif np.issubdtype(df_metadata[col].dtype, np.number):
            if col in POI_CONFIG: 
                df_metadata[col] = df_metadata[col].fillna(99999.0)
            elif col == 'asset_type_id': 
                df_metadata[col] = df_metadata[col].fillna(0).astype(int)
            elif col == 'pet_friendly':
                pass 
            else:
                df_metadata[col] = df_metadata[col].fillna(0.0)

    metadatas = df_metadata.to_dict(orient="records")
    for m in metadatas:
        if m.get('pet_friendly') is None: m['pet_friendly'] = False 

    ids_list = df["id"].astype(str).tolist()

    logger.info(f"Setting up ChromaDB client at path: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        if collection_name in [c.name for c in client.list_collections()]:
            logger.warning(f"Collection '{collection_name}' already exists. Deleting and rebuilding.")
            client.delete_collection(name=collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)

    logger.info(f"Adding {len(df)} documents...")
    BATCH_SIZE = 1000
    for i in range(0, len(ids_list), BATCH_SIZE):
        try:
            collection.add(
                ids=ids_list[i:i+BATCH_SIZE],
                embeddings=embeddings[i:i+BATCH_SIZE].tolist(),
                documents=texts[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE]
            )
            logger.info(f"Added batch {i // BATCH_SIZE + 1} / {math.ceil(len(ids_list) / BATCH_SIZE)}")
        except Exception as e:
            logger.error(f"❌ Error adding batch: {e}")

    print("\n" + "="*80)
    print(f"✅ DONE: Vector DB built at {db_path}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--db_path", type=str, default="npa_vectorstore")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--collection", type=str, default="npa_assets_v2")
    args = parser.parse_args()
    main(csv_path=args.csv_path, db_path=args.db_path, model_name=args.model, collection_name=args.collection)