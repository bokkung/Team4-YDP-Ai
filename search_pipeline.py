import os
from dotenv import load_dotenv 
load_dotenv() 
import time
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import requests # <--- ต้องมีตัวนี้
from sentence_transformers import SentenceTransformer
import chromadb

# ============ CONFIGURATION ============
VECTOR_DB_PATH = Path("npa_vectorstore") 
COLLECTION_NAME = "npa_assets_v2" 

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY is not set in .env")

EMB_MODEL_NAME = "BAAI/bge-m3"
TOP_K_RESULTS = 100 
FINAL_TOP_N = 5 
LLM_MODEL = "openai/gpt-4o-mini" 

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("search_pipeline")

# ============ PROMPT ENGINEERING ============

ENHANCED_INTENT_DETECTION_PROMPT = """
คุณคือผู้เชี่ยวชาญด้านอสังหาริมทรัพย์ในไทย หน้าที่ของคุณคือวิเคราะห์คำค้นหา (Query) ที่ผู้ใช้ป้อนมา และแปลงมันเป็น JSON structure ที่ชัดเจน

(ไม่ต้องใส่ Query: "{query}" ตรงนี้)

จงวิเคราะห์ Query ที่ผู้ใช้ป้อนมา และตอบกลับเป็น JSON object เท่านั้น โดยมีโครงสร้างดังนี้:
{{
  "asset_types": ["ประเภท1", "ประเภท2", ...],
  "must_have": ["poi1", "poi2", ...],
  "nice_to_have": ["poi1", "poi2", ...],
  "avoid_poi": ["poi1", "poi2", ...],
  "pet_friendly": true/false/null,
  "price_range": {{
    "min": null_or_number,
    "max": null_or_number
  }}
}}

คำอธิบาย Field:
1.  "asset_types":
    * ประเภทของอสังหาฯ ที่ผู้ใช้มองหา (ระบุให้ชัดเจนที่สุด)
    * ตัวเลือก: ["คอนโด", "บ้านเดี่ยว", "บ้านแฝด", "ทาวน์โฮม", "อาคารพาณิชย์", "ที่ดิน"]
    * ถ้าบอกรวมๆ ว่า "บ้าน" ให้ใส่: ["บ้านเดี่ยว", "บ้านแฝด"]
    * ถ้าไม่ระบุ ให้เป็น: []
2.  "must_have":
    * POI ที่ผู้ใช้ "ต้องมี" (ใช้ POI key มาตรฐาน)
    * ถ้าไม่ระบุ ให้เป็น: []
3.  "nice_to_have":
    * POI ที่ผู้ใช้ "อยากได้" (ใช้ POI key มาตรฐาน)
    * ถ้าไม่ระบุ ให้เป็น: []
4.  "pet_friendly":
    * `true` (ถ้า "เลี้ยงสัตว์"), `false` (ถ้า "ไม่เลี้ยงสัตว์"), `null` (ถ้า "ไม่พูดถึง")
5.  "price_range":
    * ช่วงงบประมาณของผู้ใช้ (แปลงเป็นตัวเลขเท่านั้น)
    * "5 ล้าน" -> 5000000, "10m" -> 10000000, "2.5 ล." -> 2500000
    * "ไม่เกิน 5 ล้าน" -> {{ "min": null, "max": 5000000 }}
    * "3-5 ล้าน" -> {{ "min": 3000000, "max": 5000000 }}
    * ถ้าไม่พูดถึงราคา ให้เป็น: {{ "min": null, "max": null }}
6.  "avoid_poi":
    * POI ที่ผู้ใช้ "ไม่ต้องการ", "ไม่อยากอยู่ใกล้", "หนีห่าง" (ใช้ POI key มาตรฐาน)
    * เช่น "ไม่เอาใกล้โรงเรียน", "หนีความวุ่นวาย (market/mall)"
    * ถ้าไม่ระบุ ให้เป็น: []

[กฎ POI key มาตรฐาน]
* "bts", "รถไฟฟ้า", "บีทีเอส", "skytrain" -> "bts_station"
* "เซเว่น", "7-11", "ร้านสะดวกซื้อ" -> "convenience_store"
* "mrt", "รถไฟฟ้าใต้ดิน" -> "mrt"
* "รถไฟ", "สถานีรถไฟ" -> "train_station"
ห้ามเหมา "รถไฟฟ้า" เป็น "train_station" เด็ดขาด! (คนละอย่างกัน)
* "ห้าง", "สรรพสินค้า" -> "shopping_mall"
* "โรงเรียน", "มหาลัย" -> "school" (หรือ "university")
* "โรงพยาบาล", "คลินิก" -> "hospital"
* "สวนสาธารณะ" -> "park"
* "ตลาด" -> "market"
* "ร้านอาหาร" -> "restaurant"
* "คาเฟ่" -> "cafe"

ตอบกลับเป็น JSON เท่านั้น:
"""

RAG_SYSTEM_PROMPT = """
คุณคือ "Mercil" ผู้เชี่ยวชาญด้านการวิเคราะห์อสังหาริมทรัพย์ AI หน้าที่ของคุณคือ "อธิบายเหตุผล" (Explainability) ว่าทำไมทรัพย์สินนี้ถึงตรงหรือไม่ตรงกับความต้องการของผู้ใช้

[Input Data]
คุณจะได้รับ:
1. User Query: สิ่งที่ลูกค้าตามหา
2. Verified Data: ข้อมูลจริงของทรัพย์สิน
3. Analysis Result: ผลการคำนวณระยะทางและเงื่อนไขจากระบบ (ผ่าน/ไม่ผ่าน)

⚠️ **สิ่งที่ต้องจำ:**
1. **BTS/MRT**: ระบบขนส่งรวดเร็ว (Rapid Transit) = ถนนสาย Skytrain/Subway
2. **train_station**: รถไฟแบบดั้งเดิม = State Railway (ช้า, ไม่บ่อย)
3. ห้ามสับสน! ถ้า POI ชื่อ "train_station" ก็ไม่ใช่ BTS/MRT

[งานของคุณ]
จงเขียนคำอธิบายแบบ XAI (Explainable AI) ให้ลูกค้าเข้าใจเหตุผลเบื้องหลัง โดยใช้หลักการ "Chain of Thought":
1.  **เชื่อมโยง (Connect):** เชื่อมโยงความต้องการ (Query) เข้ากับข้อมูลจริง (Data)
    * *ตัวอย่าง:* "ที่คุณต้องการคอนโดเลี้ยงสัตว์ได้..."
2.  **หลักฐาน (Evidence):** อ้างอิงตัวเลขจากผลวิเคราะห์ (Analysis Result) เพื่อยืนยัน
    * *ตัวอย่าง:* "...ทรัพย์สินนี้ตอบโจทย์เพราะอยู่ห่างจากสวนสันติภาพเพียง 439 เมตร ซึ่งอยู่ในระยะเดินถึง"
3.  **ข้อดี/ข้อเสีย (Trade-off):** ถ้ามีจุดที่ไม่ตรง ให้ชี้แจงเหตุผลอย่างตรงไปตรงมา
    * *ตัวอย่าง:* "แม้ทำเลจะดีมาก แต่ระบบตัดคะแนนในส่วนนี้เพราะเป็น 'บ้านเดี่ยว' ซึ่งไม่ตรงกับที่คุณมองหา 'คอนโด' ครับ"

[ไกด์ไลน์การแปลความหมายระยะทาง (Contextual Distance)]

🔴 **กลุ่ม 1: สำหรับ รถไฟฟ้า (BTS/MRT) เท่านั้น** (เน้นการเดินทางไปทำงาน)
* **0 - 500 ม.:** "ทำเลทอง (Prime Location) ครับ เดินไปสถานีได้สบายๆ เลย"
* **500 - 800 ม.:** "ยังอยู่ในระยะเดินไหวครับ ถือว่าได้ออกกำลังกาย หรือจะเรียกพี่วินก็แป๊บเดียว"
* **800 ม. - 1.5 กม.:** "ระยะนี้เดินเหนื่อยครับ แนะนำให้นั่งพี่วิน (Motorcycle Taxi) ออกมาปากซอยจะสะดวกที่สุดครับ"
* **1.5 กม. - 5 กม.:** "ระยะนี้ต้องอาศัยรถส่วนตัว หรือขับไปจอด (Park & Ride) ที่สถานีครับ"

🔵 **กลุ่ม 2: สำหรับ ร้านสะดวกซื้อ (7-11/Family Mart)**
* **< 800 ม.:** "เดินไปซื้อของกินของใช้ได้สะดวกเลยครับ"
* **> 800 ม.:** "อาจจะต้องขี่มอเตอร์ไซค์ไปหน่อยนะครับ"

🟢 **กลุ่ม 3: สำหรับ สถานที่อื่นๆ (ห้าง/โรงพยาบาล/โรงเรียน)** (เน้นขับรถ)
* **< 2 กม.:** "อยู่ใกล้มากครับ ขับรถแป๊บเดียวถึง"
* **2 - 5 กม.:** "อยู่ในระยะขับรถที่สะดวกครับ ไม่ไกลเกินไป"
* **> 5 กม.:** "ระยะนี้ถือว่าค่อนข้างไกลจาก [สถานที่] พอสมควรครับ อาจจะต้องเผื่อเวลาเดินทางหน่อยนะครับ"

[ข้อห้ามเด็ดขาด (Strict Rules)]
1. ห้ามมั่วข้อมูล: ถ้าใน [Verified Data] ไม่มีข้อมูลรถไฟฟ้า "ห้าม" บอกว่าเดินทางด้วยรถไฟฟ้าสะดวก
2. ห้ามแถ: ถ้า User หา "รถไฟฟ้า" แต่ถ้าไม่ ให้ตอบตรงๆ ว่า "ไม่อยู่ใกล้รถไฟฟ้า "
3. แยกแยะรถไฟ: "สถานีรถไฟ(ธรรมดา)" ไม่ใช่ "รถไฟฟ้า BTS/MRT" ห้ามเหมารวม

[ถ้ามี SYSTEM NOTE]
⚠️ SYSTEM NOTE: ไม่พบสถานี BTS/MRT ในระยะที่เหมาะ (แต่มี train_station = State Railway)
ให้ระบุให้ชัดว่า "มีสถานีรถไฟเท่านั้น ไม่ใช่รถไฟฟ้า"

[Tone & Style]
* เป็นมืออาชีพแต่น่าเชื่อถือ (Professional & Trustworthy)
* ใช้ภาษาที่เป็นธรรมชาติ ไม่เหมือนหุ่นยนต์
* **ต้องระบุตัวเลขระยะทาง หรือชื่อสถานที่เสมอ** เพื่อความโปร่งใส
"""


def create_rag_user_content(query: str, meta: Dict, reasons: List[str], penalties: List[str]) -> str:
    """
    สร้าง User Content สำหรับ RAG Prompt
    - ใช้ display_name จาก POI_CONFIG (Single Source of Truth)
    - แยก BTS/MRT (rapid_transit) จาก train_station
    - เพิ่ม SYSTEM NOTE ให้ชัดเจน
    """
    
    # ============================================================================
    # 1. DYNAMIC EXTRACTION: ดึง POI จาก POI_CONFIG พร้อม display_name
    # ============================================================================
    poi_context = []
    found_keys = set()  # เก็บ key ที่เจอจริง (สำหรับ Trap Logic)
    
    # Loop ผ่าน POI_CONFIG ทั้งหมด
    for key in POI_CONFIG.keys():
        dist = meta.get(key)
        
        # ตรวจสอบว่ามีข้อมูลระยะห่างและอยู่ในช่วง 0-10km
        if dist is not None and isinstance(dist, (int, float)) and 0 <= dist < 10000:
            
            # 1. ดึง display_name จาก POI_CONFIG (ไม่ใช่ hardcoded mapping!)
            poi_config = POI_CONFIG[key]
            label = poi_config.get("display_name", key)  # fallback to key ถ้าไม่มี
            
            # 2. ดึงชื่อเฉพาะสถาน (เช่น "BTS ลาดพร้าว")
            specific_name = meta.get(f"{key}_name", "-")
            
            # 3. จัดเก็บข้อมูล
            poi_context.append(f"- {label}: ชื่อ '{specific_name}' ห่าง {dist:,.0f} เมตร")
            
            # เก็บ key ไว้สำหรับ Trap Logic
            found_keys.add(key)

    # ============================================================================
    # 2. HALLUCINATION TRAP: ตรวจสอบความไม่สมดุลระหว่าง Query กับ Data
    # ============================================================================
    q_lower = query.lower()

    # [Trap 1: รถไฟฟ้า (BTS/MRT)] ✅ FIXED - แยก rapid_transit จาก train
    need_rapid_transit = any(k in q_lower for k in ["รถไฟฟ้า", "bts", "mrt", "skytrain", "ใกล้ระบบขนส่ง"])
    has_bts = "bts_station" in found_keys
    has_mrt = "mrt" in found_keys
    has_rapid_transit = has_bts or has_mrt
    has_state_train = "train_station" in found_keys  # State Railway (อื่น)
    
    if need_rapid_transit and not has_rapid_transit:
        if has_state_train:
            poi_context.append(
                "\n⚠️ **SYSTEM NOTE: พบสถานีรถไฟแบบดั้งเดิม (State Railway) แต่ไม่มีรถไฟฟ้า BTS/MRT ในระยะ**"
            )
        else:
            poi_context.append(
                "\n⚠️ **SYSTEM NOTE: ไม่พบสถานีรถไฟฟ้า BTS/MRT ในระยะ (ไม่มี rapid transit accessibility)**"
            )

    # [Trap 2: โรงพยาบาล]
    if any(k in q_lower for k in ["โรงพยาบาล", "หมอ", "ทำฟัน"]) and "hospital" not in found_keys:
        poi_context.append(
            "\n⚠️ **SYSTEM NOTE: ไม่พบโรงพยาบาลในระยะที่เหมาะสม**"
        )
        
    # [Trap 3: โรงเรียน]
    if any(k in q_lower for k in ["โรงเรียน", "ลูก", "เรียน"]) and "school" not in found_keys:
        poi_context.append(
            "\n⚠️ **SYSTEM NOTE: ไม่พบโรงเรียนในระยะที่เหมาะสม**"
        )

    # [Trap 4: ตลาด]
    if any(k in q_lower for k in ["ตลาด", "ตลาดสด", "ซื้อของสด"]) and "market" not in found_keys:
        if "convenience_store" in found_keys or "supermarket" in found_keys:
            poi_context.append(
                "\n⚠️ **SYSTEM NOTE: มีซูเปอร์/สะดวกซื้อ แต่ไม่มีตลาดสดในระยะ**"
            )
        else:
            poi_context.append(
                "\n⚠️ **SYSTEM NOTE: ไม่พบตลาดสดในระยะที่เหมาะสม**"
            )

    # [Trap 5: สัตวแพทย์ (สำหรับคนเลี้ยงสัตว์)]
    if any(k in q_lower for k in ["เลี้ยงสัตว์", "หมา", "แมว", "pet"]) and "veterinary" not in found_keys:
        poi_context.append(
            "\n⚠️ **SYSTEM NOTE: ไม่พบคลินิกสัตวแพทย์/Pet Hospital ในระยะ**"
        )

    # สรุปข้อมูล POI
    poi_text = "\n".join(poi_context) if poi_context else "- ไม่พบสถานที่สำคัญในระยะ 10 กม."

    # ============================================================================
    # 3. PREPARE REASONS & PENALTIES (เฉพาะของสินค้านี้)
    # ============================================================================
    clean_reasons = []
    if reasons:
        for r in reasons:
            # Clean up dummy values
            clean_r = r.replace("99999.0", "ระยะไกลมาก").replace("99999", "ระยะไกลมาก")
            clean_reasons.append(clean_r)

    # ข้อมูล Zone/Area Description
    zone_desc = meta.get("zone_desc", "")
    zone_info = f"- ผังเมือง/บริวาร: {zone_desc}" if zone_desc else ""

    # ============================================================================
    # 4. BUILD FINAL USER CONTENT
    # ============================================================================
    user_content = f"""
[บริบทการวิเคราะห์]
- สิ่งที่ลูกค้าต้องการ (Query): "{query}"
- ประเภททรัพย์สินจริง: {meta.get("asset_type_fixed", "N/A")}
- ราคาขาย: {float(meta.get("asset_details_selling_price", 0)):,.0f} บาท
- รายละเอียดจาก AI ก่อนหน้า: {str(meta.get("asset_details_description_th", "N/A"))[:400]}...
- ทำเลตั้งจริง: {meta.get("location_village_th", "")} {meta.get("location_road_th", "")}
{zone_info}

[ระยะทางสถานที่สำคัญ (Verified Data)]
{poi_text}

[ผลลัพธ์การวิเคราะห์ (Execution Trace)]
✅ ปัจจัยบวก:
{chr(10).join("- " + r for r in clean_reasons) if clean_reasons else "- ไม่มี"}

⚠️ ข้อควรระวัง:
{chr(10).join("- " + p for p in penalties) if penalties else "- ไม่มี"}

[คำสั่ง]
โปรดสรุปทรัพย์สินนี้ โดยต้องอ้างอิงข้อมูลจาก [Verified Data] เท่านั้น หากมี SYSTEM NOTE แจ้งเตือน ให้ปกติบัตติตามอย่างเคร่งครัด
"""
    
    return user_content

# ✅ POI Config (Final Version - Park & Ride Logic + Research Backed)
# ✅ POI Config (Final Version - with display_name)
POI_CONFIG = {
    # === 🚆 TRANSPORTATION ===
    "bts_station": {
        "radius": 3000,
        "weight": 1.2,
        "curve": "exponential",
        "display_name": "สถานี BTS (รถไฟฟ้า)",
        "poi_type": "rapid_transit"
    },
    "mrt": {
        "radius": 3000,
        "weight": 1.2,
        "curve": "exponential",
        "display_name": "สถานี MRT (รถไฟฟ้าใต้ดิน)",
        "poi_type": "rapid_transit"
    },
    "train_station": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "exponential",
        "display_name": "สถานีรถไฟ (การรถไฟแห่งประเทศไทย)",
        "poi_type": "train"
    },
    "bus_station": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "exponential",
        "display_name": "สถานีขนส่งบัสและสถานีรถ",
        "poi_type": "bus"
    },

    # === 🏪 CONVENIENCE ===
    "convenience_store": {
        "radius": 1000,
        "weight": 0.5,
        "curve": "exponential",
        "display_name": "ร้านสะดวกซื้อ (7-11 / Family Mart)",
        "poi_type": "convenience"
    },
    "market": {
        "radius": 1500,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "ตลาด / ตลาดสด",
        "poi_type": "market"
    },
    "supermarket": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "ซูเปอร์มาร์เก็ต",
        "poi_type": "convenience"
    },

    # === 🛍️ LIFESTYLE ===
    "shopping_mall": {
        "radius": 3000,
        "weight": 1.1,
        "curve": "linear",
        "display_name": "ห้างสรรพสินค้า / ShoppingMall",
        "poi_type": "lifestyle"
    },
    "community_mall": {
        "radius": 2000,
        "weight": 0.7,
        "curve": "linear",
        "display_name": "คอมมูนิตี้มอลล์",
        "poi_type": "lifestyle"
    },
    "restaurant": {
        "radius": 1000,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "ร้านอาหาร",
        "poi_type": "dining"
    },
    "cafe": {
        "radius": 1000,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "คาเฟ่",
        "poi_type": "dining"
    },

    # === 🏥 HEALTH & WELLNESS ===
    "hospital": {
        "radius": 3000,
        "weight": 0.7,
        "curve": "linear",
        "display_name": "โรงพยาบาล",
        "poi_type": "health"
    },
    "park": {
        "radius": 3000,
        "weight": 0.6,
        "curve": "linear",
        "display_name": "สวนสาธารณะ / สวนเฉพาะ",
        "poi_type": "recreation"
    },
    "gym": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "ห้องออกกำลังกาย / Fitness Center",
        "poi_type": "health"
    },
    "spa": {
        "radius": 2000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "สปา / นวดไทย",
        "poi_type": "wellness"
    },

    # === 🐶 PET FRIENDLY ===
    "veterinary": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "คลินิกสัตวแพทย์ / Pet Hospital",
        "poi_type": "pet"
    },

    # === 🏫 EDUCATION & CULTURE ===
    "school": {
        "radius": 3000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "โรงเรียน / สถาบันการศึกษา",
        "poi_type": "education"
    },
    "university": {
        "radius": 3000,
        "weight": 0.3,
        "curve": "linear",
        "display_name": "มหาวิทยาลัย",
        "poi_type": "education"
    },
    "temple": {
        "radius": 1500,
        "weight": 0.1,
        "curve": "linear",
        "display_name": "วัด / สถานที่ศักดิ์สิทธิ์",
        "poi_type": "culture"
    },
    "museum": {
        "radius": 5000,
        "weight": 0.1,
        "curve": "linear",
        "display_name": "พิพิธภัณฑ์",
        "poi_type": "culture"
    },

    # === 🌳 OUTDOOR & NATURE ===
    "river": {
        "radius": 1500,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "แม่น้ำ / ชุมชนริมน้ำ",
        "poi_type": "nature"
    },
    "beach": {
        "radius": 3000,
        "weight": 0.0,
        "curve": "linear",
        "display_name": "ทะเล / หาด",
        "poi_type": "nature"
    },
    "viewpoint": {
        "radius": 3000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "จุดชมวิวเมืองและสถานที่ท่องเที่ยว",
        "poi_type": "attraction"
    },

    # === 🏨 TRAVEL & LEISURE ===
    "tourist_attraction": {
        "radius": 3000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "สถานที่ท่องเที่ยว / Landmark",
        "poi_type": "attraction"
    },
    "hotel": {
        "radius": 2000,
        "weight": 0.1,
        "curve": "linear",
        "display_name": "โรงแรม / ที่พักแรม",
        "poi_type": "accommodation"
    },
    "golf_course": {
        "radius": 5000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "สนามกอล์ฟ",
        "poi_type": "recreation"
    },
}


# ✅ ASSET ID MAPPING (Verified with asset_type_rows.json)
ASSET_ID_MAPPING = {
    # === 🏠 กลุ่มที่อยู่อาศัย (Living) ===
    "คอนโด": [3, 12],           # ห้องชุดพักอาศัย(3), อาคารชุดพักอาศัย(12)
    "ห้องชุด": [3, 11, 16],     # พักอาศัย(3), สนง.(11), พาณิชย์(16)
    "บ้าน": [4, 15],            # บ้านเดี่ยว(4), บ้านแฝด(15)
    "บ้านเดี่ยว": [4],
    "บ้านแฝด": [15],            
    "ทาวน์โฮม": [1],            # ทาวน์เฮ้าส์(1)
    "ทาวน์เฮ้าส์": [1],
    "อพาร์ทเมนท์": [17, 30],    # อพาร์ทเมนท์(17), อาคารพักอาศัย(30)
    "หอพัก": [30],              # (อาคารพักอาศัย มักเป็นหอพัก/ตึกแถวอยู่อาศัย)

    # === 🏢 กลุ่มพาณิชย์ (Commercial) ===
    "อาคารพาณิชย์": [5],
    "ตึกแถว": [5, 30],          # ส่วนใหญ่เป็น 5 แต่บางทีเป็น 30
    "โฮมออฟฟิศ": [9],           # (ID 9 โดยเฉพาะ)
    "สำนักงาน": [11, 13],       # ห้องชุดสำนักงาน(11), อาคารสำนักงาน(13)
    "ออฟฟิศ": [9, 11, 13],
    "โชว์รูม": [8],
    "ห้าง": [22],               # ห้างสรรพสินค้า(22)
    "ร้านอาหาร": [35],
    "ตลาด": [25],
    "ปั๊มน้ำมัน": [14],

    # === 🏭 กลุ่มอุตสาหกรรม/ที่ดิน (Industrial/Land) ===
    "ที่ดิน": [2],
    "ที่ดินเปล่า": [2],
    "โรงงาน": [6, 36],          # โรงงาน/โกดัง(6), มินิแฟคตอรี่(36)
    "โกดัง": [6, 34],           # โรงงาน/โกดัง(6), ศูนย์จำหน่ายสินค้า(34)
    "คลังสินค้า": [6, 34],

    # === 🏨 ธุรกิจท่องเที่ยว/อื่นๆ ===
    "โรงแรม": [10],
    "รีสอร์ท": [10],
    "โรงเรียน": [29],
    "โรงพยาบาล": [18, 19],      # (18=สิทธิการเช่า/พื้นที่พาณิชย์ เผื่อเป็นคลินิก, 19=รพ.)
    "สนามกอล์ฟ": [21]
}

# ============ SERVICE FUNCTIONS ============

def get_embedding_model(model_name: str) -> SentenceTransformer:
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logger.info("✅ Embedding model loaded.")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise

def get_chroma_collection(db_path: Path, collection_name: str) -> chromadb.Collection:
    if not db_path.exists():
        logger.error(f"❌ Vector DB path not found: {db_path}")
        raise FileNotFoundError(f"Vector DB path not found: {db_path}")
    logger.info(f"Connecting to ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"✅ Connected to collection '{collection_name}' ({collection.count()} documents)")
        return collection
    except Exception as e:
        logger.error(f"❌ Failed to connect to collection '{collection_name}'.")
        raise e

# ============ LLM CALLING FUNCTION (FINAL ROBUST VERSION) ============

def call_openrouter(system_prompt: str, user_content: str, model: str = LLM_MODEL, retries: int = 3) -> Optional[str]:
    """
    ฟังก์ชันยิง API ไปหา OpenRouter พร้อมระบบ Retry และดักจับ Error แบบละเอียด
    Returns: str (เนื้อหาตอบกลับ) หรือ None (ถ้าเกิด Error จนครบจำนวนครั้ง)
    """
    
    # 1. เช็ค Key ก่อนยิง
    if not OPENROUTER_API_KEY:
        logger.error("❌ Error: OPENROUTER_API_KEY is missing in .env")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Mercil Real Estate AI",
    }
    
    # Payload
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7, 
        "max_tokens": 1000, 
    }

    # 2. เริ่ม Loop การ Retry
    for attempt in range(retries):
        try:
            # ยิง Request (เพิ่ม Timeout เป็น 45 วินาที)
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload, 
                timeout=45 
            )
            
            # ✅ กรณีสำเร็จ (200 OK)
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    logger.warning(f"⚠️ API Response format unexpected: {result}")
                    return None

            # ⚠️ กรณีติด Rate Limit (429) -> ให้รอแล้วยิงใหม่
            elif response.status_code == 429:
                logger.warning(f"⚠️ Rate Limit hit (Attempt {attempt+1}/{retries}). Retrying in 2s...")
                time.sleep(2)
                continue # ข้ามไปรอบถัดไป
            
            # ❌ กรณี Error อื่นๆ (4xx, 5xx) -> ปริ้นท์แล้วจบเลย (ส่วนใหญ่เป็นที่ Payload ผิด ไม่ต้อง Retry)
            else:
                print(f"\n❌ OPENROUTER API ERROR: {response.status_code}")
                print(f"👉 Error Details: {response.text}") 
                return None

        except Exception as e:
            # 🔌 กรณีเน็ตหลุด หรือ Timeout -> ให้รอแล้วยิงใหม่
            logger.error(f"❌ Connection Error (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(1)
            continue

    # ถ้าวนลูปครบ 3 รอบแล้วยังไม่ได้
    logger.error("❌ Failed to call OpenRouter after multiple attempts.")
    return None

# ============ SEARCH PIPELINE FUNCTIONS ============

def enhanced_intent_detection(query: str) -> Dict[str, Any]:
    system_prompt = ENHANCED_INTENT_DETECTION_PROMPT
    user_content = query
    logger.info("Detecting intent...")
    
    # ✅ FIX: เรียกใช้ call_openrouter ให้ถูก (ไม่ต้องใส่ model ซ้ำก็ได้เพราะมี default)
    raw_response = call_openrouter(system_prompt, user_content)
    
    # Fallback ถ้า API ตาย
    default_intent = { "asset_types": [], "must_have": [], "nice_to_have": [], "avoid_poi": [], "pet_friendly": None, "price_range": {"min": None, "max": None} }
    
    if not raw_response:
        return default_intent

    try:
        match = re.search(r'```json\n(.*?)\n```', raw_response, re.DOTALL)
        if match: json_str = match.group(1)
        else:
            json_str = raw_response.strip()
            if not json_str.startswith("{"):
                 start = json_str.find("{")
                 if start != -1: json_str = json_str[start:]
        
        intent_json = json.loads(json_str)
        validated_intent = {
                "asset_types": intent_json.get("asset_types", []),
                "must_have": intent_json.get("must_have", []),
                "nice_to_have": intent_json.get("nice_to_have", []),
                "avoid_poi": intent_json.get("avoid_poi", []),
                "pet_friendly": intent_json.get("pet_friendly", None),
                "price_range": intent_json.get("price_range", {"min": None, "max": None})
            }
        logger.info(f"Intent detected: {validated_intent}")
        return validated_intent
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM response: {raw_response}")
        return default_intent

def chroma_query(collection: chromadb.Collection, embed_model: SentenceTransformer, query: str, k: int, filters: Dict = {}) -> List[Dict[str, Any]]:
    logger.info("Performing semantic search...")
    query_embedding = embed_model.encode([query]).tolist()
    chroma_filter = None 
    if filters:
        filter_list = []
        if "max_price" in filters and filters["max_price"] > 0:
            filter_list.append({"asset_details_selling_price": {"$lte": filters["max_price"]}})
        if filter_list:
            chroma_filter = {"$and": filter_list} if len(filter_list) > 1 else filter_list[0]
    try:
        results = collection.query(query_embeddings=query_embedding, n_results=k, where=chroma_filter, include=["metadatas", "distances"])
        processed_results = []
        if 'ids' not in results or not results['ids']:
            logger.warning("ChromaDB query returned no results.")
            return []
        for i, dist in enumerate(results['distances'][0]):
            meta = results['metadatas'][0][i]
            semantic_score = max(0, 1 - (dist / 2.0))
            processed_results.append({"id": results['ids'][0][i], "semantic_score": semantic_score, "metadata": meta})
        return processed_results
    except Exception as e:
        logger.error(f"❌ Error during Chroma query: {e}", exc_info=True)
        return []

def apply_filters(results: List[Dict], filters_cli: Dict, intent: Dict) -> List[Dict]:
    if not filters_cli and not intent.get("price_range"): return results 
    filtered_results = []
    price_range = intent.get("price_range", {})
    final_max_price = filters_cli.get("max_price") if filters_cli.get("max_price") is not None else price_range.get("max")
    final_min_price = price_range.get("min")
    
    for r in results:
        meta = r.get("metadata", {})
        keep = True
        price = float(meta.get("asset_details_selling_price", 0))
        if final_max_price is not None and price > final_max_price: keep = False
        if final_min_price is not None and price < final_min_price: keep = False

        if keep: filtered_results.append(r)
    return filtered_results

def compute_intent_match_score(metadata: Dict[str, Any], intent: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """
    ✅ FIXED VERSION - BTS/Train differentiation + Proper Penalty System
    
    Returns:
        - score: Final intent match score (can be negative!)
        - reasons: List of positive matching reasons
        - penalties: List of warnings/negative factors
    """
    score = 0.0
    reasons = []
    penalties = []

    # =========================================================
    # 1. Asset Type Matching (ตรวจสอบประเภททรัพย์สินตรงใจ)
    # =========================================================
    intent_types = intent.get("asset_types", [])
    if intent_types:
        asset_id = int(metadata.get("asset_type_id", 0))
        asset_type_name = metadata.get("asset_type_fixed", "ทรัพย์สินอื่น")
        
        # ดึง ID ที่ยอมรับได้จาก ASSET_ID_MAPPING
        accepted_ids = []
        for t in intent_types:
            accepted_ids.extend(ASSET_ID_MAPPING.get(t, []))
            
        if asset_id in accepted_ids:
            score += 1.0
            reasons.append(f"✅ ตรงประเภททรัพย์สิน ({asset_type_name})")
        else:
            score -= 10.0
            penalties.append(f"❌ ไม่ตรงประเภท (ต้องการ {', '.join(intent_types)} แต่พบ {asset_type_name})")

    # =========================================================
    # 2. Pet-Friendly Matching (ตรวจสอบเลี้ยงสัตว์ได้หรือไม่)
    # =========================================================
    intent_pet = intent.get("pet_friendly")
    if intent_pet is True:  # ต้องการเลี้ยงสัตว์ได้
        meta_pet_explicit = metadata.get("pet_friendly")  # True/False/None
        asset_id = int(metadata.get("asset_type_id", 0))
        
        # 2.1: ถ้าระบุชัดเจนว่าเลี้ยงได้
        if meta_pet_explicit is True:
            score += 1.5
            reasons.append("✅ อนุญาตให้เลี้ยงสัตว์ (ระบุชัดเจน)")
            
        # 2.2: ถ้าไม่ระบุหรือระบุว่าไม่ได้ → soft logic
        elif meta_pet_explicit is None or meta_pet_explicit is False:
            if asset_id == 3:  # คอนโด
                score -= 10.0  # คอนโดส่วนใหญ่ห้ามเลี้ยง
                penalties.append("❌ เลี้ยงสัตว์ไม่ได้ (คอนโดห้ามเลี้ยง)")
            elif asset_id in [4, 15, 1]:  # บ้านเดี่ยว, บ้านแถว, ทาวน์โฮม
                score += 0.5
                reasons.append("✅ น่าจะเลี้ยงสัตว์ได้ (เป็นบ้านแนวรา)")
            else:
                score -= 5.0
                penalties.append("⚠️ ไม่ระบุเรื่องเลี้ยงสัตว์ (ต้องยืนยัน)")
                
        # Bonus: ใกล้คลินิกสัตวแพทย์
        vet_dist = float(metadata.get("veterinary", 99999))
        if vet_dist <= 2000:
            score += 0.25
            reasons.append(f"✅ ใกล้คลินิกสัตวแพทย์ ({vet_dist:.0f} ม.)")
            
    elif intent_pet is False:  # ไม่ต้องการเลี้ยง
        if metadata.get("pet_friendly") is True:
            score -= 2.0
            penalties.append("⚠️ เป็นสถานที่ Pet Friendly (อาจมีเสียงรบกวน)")

    # =========================================================
    # 3. Must-Have POI with Proper Penalty System (ต้องมี POI)
    # =========================================================
    must_haves = intent.get("must_have", [])
    
    # ✅ SPECIAL CHECK: ถ้า intent ต้อง BTS/MRT ต้องแยกจาก train_station
    if "bts_station" in must_haves or "mrt" in must_haves:
        has_bts = metadata.get("bts_station", 99999) < 3000
        has_mrt = metadata.get("mrt", 99999) < 3000
        has_rapid_transit = has_bts or has_mrt
        has_state_train = metadata.get("train_station", 99999) < 2500
        
        # ถ้าต้องการ rapid transit แต่เจอแค่ State Railway → หนักโทษ!
        if not has_rapid_transit and has_state_train:
            score -= 20.0  # ← MAJOR PENALTY: ผิดประเภท transport
            bts_dist = float(metadata.get("bts_station", 99999))
            mrt_dist = float(metadata.get("mrt", 99999))
            train_dist = float(metadata.get("train_station", 99999))
            penalties.append(
                f"❌ ต้องการ BTS/MRT แต่มี State Railway เท่านั้น "
                f"(BTS: {bts_dist:,.0f}ม., MRT: {mrt_dist:,.0f}ม., Train: {train_dist:,.0f}ม.)"
            )

    # Loop through must_have POI
    for poi_key in must_haves:
        if poi_key in POI_CONFIG:
            raw_dist = metadata.get(poi_key, 99999)
            distance = float(raw_dist) if raw_dist is not None else 99999
            
            poi_config = POI_CONFIG[poi_key]
            limit_radius = poi_config.get("radius", 3000)
            poi_display_name = poi_config.get("display_name", poi_key)
            
            # ดึงชื่อเฉพาะสถาน
            specific_name = metadata.get(f"{poi_key}_name", poi_display_name)
            
            if distance <= limit_radius:
                # ✅ POI เจอในระยะ → ให้ score
                if poi_config.get("curve") == "exponential":
                    match_score = (1 - (distance / limit_radius)) ** 2
                else:
                    match_score = 1 - (distance / limit_radius)
                
                final_match_score = max(0.1, match_score)
                score += (final_match_score * 1.5)
                
                reasons.append(f"✅ ใกล้ {poi_display_name} '{specific_name}' ({distance:,.0f} ม.)")
            else:
                # ❌ POI ต้องการแต่ห่าง → MAJOR PENALTY!
                # (เปลี่ยนจาก -1.0 เป็น -15.0)
                score -= 15.0
                
                penalties.append(
                    f"❌ ต้องการ {poi_display_name} แต่ห่าง {distance:,.0f} ม. (เกินระยะ {limit_radius:,.0f} ม.)"
                )

    # =========================================================
    # 4. Nice-to-Have POI (อยากได้แต่ไม่บังคับ)
    # =========================================================
    nice_to_haves = intent.get("nice_to_have", [])
    for poi_key in nice_to_haves:
        if poi_key in POI_CONFIG:
            distance = metadata.get(poi_key, 99999)
            poi_config = POI_CONFIG[poi_key]
            limit_radius = poi_config.get("radius", 2000)
            poi_display_name = poi_config.get("display_name", poi_key)
            
            specific_name = metadata.get(f"{poi_key}_name", poi_display_name)
            
            if distance <= limit_radius:
                # ✅ Nice to have เจอ → bonus score (ไม่หลัก)
                score += 0.25
                reasons.append(f"➕ มี {poi_display_name} '{specific_name}' ({distance:.0f} ม.) [Bonus]")

    # =========================================================
    # 5. Avoid POI (ต้องหลีกเลี่ยง)
    # =========================================================
    avoid_pois = intent.get("avoid_poi", [])
    for poi_key in avoid_pois:
        if poi_key in POI_CONFIG:
            distance = metadata.get(poi_key, 99999)
            poi_config = POI_CONFIG[poi_key]
            
            # ระยะ avoid ส่วนใหญ่น้อยกว่า must_have
            avoid_radius = poi_config.get("radius", 1000) * 0.6  # ลดระยะ 40%
            poi_display_name = poi_config.get("display_name", poi_key)
            
            if distance <= avoid_radius:
                # ❌ เจอ avoid POI ในระยะ → ลบมาก
                score -= 5.0
                penalties.append(
                    f"❌ อยู่ใกล้ {poi_display_name} (ต้องหลีกเลี่ยง) - ห่าง {distance:,.0f} ม."
                )
            else:
                # ✅ หลีกเลี่ยง avoid POI ได้ → เล็กน้อย
                score += 0.5
                reasons.append(f"✅ หลีกเลี่ยง {poi_display_name} ได้ (ห่าง {distance:,.0f} ม.)")

    # =========================================================
    # 6. Price Range Matching (ตรวจสอบราคา)
    # =========================================================
    price_range = intent.get("price_range", {})
    min_price = price_range.get("min")
    max_price = price_range.get("max")
    asset_price = float(metadata.get("asset_details_selling_price", 0))
    
    if min_price is not None and asset_price < min_price:
        score -= 5.0
        penalties.append(f"⚠️ ราคาต่ำกว่าที่ต้องการ ({asset_price:,.0f} < {min_price:,.0f} บาท)")
    elif max_price is not None and asset_price > max_price:
        score -= 5.0
        penalties.append(f"⚠️ ราคาสูงกว่าที่ต้องการ ({asset_price:,.0f} > {max_price:,.0f} บาท)")
    else:
        if min_price is not None or max_price is not None:
            score += 0.5
            reasons.append(f"✅ ราคาตรงในช่วงที่ต้องการ ({asset_price:,.0f} บาท)")

    return score, reasons, penalties

def apply_nice_to_have_boost(metadata: Dict[str, Any], intent: Dict[str, Any]) -> Tuple[float, List[str]]:
    nice_boost = 0.0
    nice_reasons = []
    nice_to_haves = intent.get("nice_to_have", [])
    for poi_key in nice_to_haves:
        if poi_key in POI_CONFIG:
            distance = metadata.get(poi_key, 99999)
            poi_name = metadata.get(f"{poi_key}_name", poi_key)
            limit_radius = POI_CONFIG[poi_key].get("radius", 2000)
            
            if distance <= limit_radius: 
                nice_boost += 0.25 
                nice_reasons.append(f"มี {poi_name} ใกล้ๆ ({distance:.0f} ม.)")
    return nice_boost, nice_reasons

def rag_explain_single_item(query: str, intent: Dict, result: Dict, reasons: List[str], penalties: List[str]) -> str:
    """ฟังก์ชันหลักที่เรียกใช้ LLM"""
    
    # 1. เตรียมข้อมูล
    meta = result.get("metadata", {})
    user_content = create_rag_user_content(query, meta, reasons, penalties)
    
    # 2. เรียกฟังก์ชันยิง API (ที่แก้แล้ว)
    explanation = call_openrouter(RAG_SYSTEM_PROMPT, user_content)
    
    # 3. Clean ข้อมูลก่อนส่งกลับ
    if not explanation:
        return "ขออภัยครับ ระบบไม่สามารถสร้างคำอธิบายได้ในขณะนี้ แต่ทรัพย์สินนี้ตรงกับเงื่อนไขที่คุณค้นหาครับ (System Busy)"
        
    return explanation.strip().replace('"', '')

def execute_search(query: str, filters: Dict, embed_model: SentenceTransformer, collection: chromadb.Collection) -> Dict[str, Any]:
    query_intent = enhanced_intent_detection(query)
    results = chroma_query(collection, embed_model, query, TOP_K_RESULTS, filters)
    if not results:
        return { "query": query, "intent_detected": query_intent, "results": [], "message": f"🤷 ไม่พบผลลัพธ์ที่ตรงกับคำค้นหา: \"{query}\"" }
    
    filtered_results = apply_filters(results, filters, query_intent)
    logger.info("Re-ranking results...")
    ranked_results = []
    for r in filtered_results:
        meta = r.get("metadata", {})
        lifestyle_score = float(meta.get("lifestyle_score", 0))
        intent_score, reasons, penalties = compute_intent_match_score(meta, query_intent)
        nice_boost, nice_reasons = apply_nice_to_have_boost(meta, query_intent)
        r["intent_reasons"] = reasons + nice_reasons
        r["intent_penalties"] = penalties
        final_score = ((intent_score * 0.7) + (r["semantic_score"] * 0.2) + (lifestyle_score * 0.05) + (nice_boost * 0.05))
        r["final_score"] = final_score
        r["intent_score"] = intent_score
        r["lifestyle_score"] = lifestyle_score 
        ranked_results.append(r)

    ranked_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # ✅ [QUALITY GATE]
    if not ranked_results or ranked_results[0]['final_score'] < 0.35:
        return {
            "query": query,
            "intent_detected": query_intent,
            "results": [],
            "message": "🤔 ไม่พบทรัพย์สินที่ตรงกับความต้องการ หรือคำค้นหาอาจไม่ชัดเจนครับ (Low Matching Score)"
        }
    
    final_results_list = []
    for r in ranked_results[:FINAL_TOP_N]:
        meta = r.get("metadata", {})
        summary_text = rag_explain_single_item(query, query_intent, r, r.get('intent_reasons', []), r.get('intent_penalties', []))
        final_results_list.append({
            "id": r['id'],
            "final_score": round(r['final_score'], 2),
            "intent_score": round(r['intent_score'], 2),
            "summary": summary_text,
            "reasons": r.get('intent_reasons', []),
            "penalties": r.get('intent_penalties', []),
            "asset_details": {
                "name": meta.get('name_th', 'N/A'),
                "price": float(meta.get('asset_details_selling_price', 0)),
                "location": f"{meta.get('location_village_th', '')} {meta.get('location_road_th', '')}".strip() or "ไม่ระบุทำเล",
                "bedroom": meta.get('bedroom', 'N/A'),
                "bathroom": meta.get('bathroom', 'N/A'),
                "type_id": meta.get('asset_type_id', 'N/A') 
            }
        })
    
    return { "query": query, "intent_detected": query_intent, "results": final_results_list, "message": "Search completed successfully." }