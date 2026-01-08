"""
Search Pipeline Configuration
=============================
Externalized configuration for the search pipeline.
All magic numbers and tunable parameters in one place.

Design decisions for the open questions:
1. Semantic fallback: Return semantic-only results WITH warning when intent parsing fails
2. Missing data: Include but rank lower with explicit "unverified" warning (lenient)
3. Threshold: Minimum quality gate - must pass all hard constraints to be included
"""

from typing import Dict, Any, List
from pathlib import Path

# ============ PATHS ============
VECTOR_DB_PATH = Path("npa_vectorstore")
COLLECTION_NAME = "npa_assets_v2"

# ============ MODEL CONFIG ============
EMB_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL = "openai/gpt-4o-mini"

# ============ RETRIEVAL CONFIG ============
RETRIEVAL_CONFIG = {
    "top_k_candidates": 100,    # Semantic retrieval pool size
    "final_top_n": 5,           # Final results to return
}

# ============ SCORING WEIGHTS ============
# All adjustable from here without code changes
SCORING_WEIGHTS = {
    # Positive signals
    "asset_type_match": 2.0,
    "must_have_poi_base": 1.5,      # Multiplied by distance factor
    "nice_to_have_poi": 0.25,
    "pet_friendly_explicit": 1.5,
    "pet_friendly_inferred": 0.5,   # For บ้านเดี่ยว/ทาวน์โฮม
    "price_in_range": 0.5,
    "avoid_poi_success": 0.3,       # Successfully avoided
    "near_vet_bonus": 0.25,         # For pet-friendly searches
    
    # Negative signals (soft - still included in results)
    "price_out_of_range": -3.0,
    "avoid_poi_failure": -5.0,
    "pet_not_allowed_condo": -8.0,
    "pet_status_unknown": -2.0,
    
    # Target Location Signals (Geocoding)
    "location_very_close": 3.0,     # < 2km
    "location_close": 1.5,          # 2km - 5km
    "location_far": -2.0,           # > 10km
    
    # Avoid Location Signals
    "avoid_location_hit_hard": -5.0,    # < 2km (Too close to what user wants to avoid)
    "avoid_location_hit_soft": -2.0,    # 2-5km
    "avoid_location_success": 0.5,      # > 5km (Successfully avoided)
}

# ============ HARD CONSTRAINTS ============
# These cause immediate disqualification (score doesn't matter)
HARD_CONSTRAINT_CONFIG = {
    "wrong_asset_type": True,           # Disqualify if type doesn't match
    "must_have_poi_too_far": True,      # Disqualify if must-have POI verified but far
    "wrong_transport_type": True,       # Disqualify if user wants BTS but only train_station
    "target_location_too_far": True,    # Disqualify if target location is too far
    "avoid_poi_too_close": True,        # Disqualify if too close to POI that should be avoided
}

# ============ DATA QUALITY CONFIG ============
DATA_QUALITY_CONFIG = {
    # Value that indicates missing data (to detect legacy 99999 usage)
    "missing_data_sentinels": [99999, 99999.0, None],
    
    # Minimum data quality to include in results (0.0 to 1.0)
    "min_quality_for_inclusion": 0.0,   # Lenient: include all, but note quality
    
    # POIs that MUST have data for strict scoring (optional stricter mode)
    "critical_pois": [],  # Empty = lenient mode
}

# ============ TARGET LOCATION CONFIG ============
TARGET_LOCATION_CONFIG = {
    "radius_very_close": 2000,   # meters
    "radius_close": 5000,        # meters
    "radius_far_limit": 10000,   # meters (beyond this = penalty)
}

# ============ POI CONFIGURATION ============
# Migrated from search_pipeline.py with added metadata
POI_CONFIG: Dict[str, Dict[str, Any]] = {
    # === TRANSPORTATION (Rapid Transit) ===
    "bts_station": {
        "radius": 3000,
        "weight": 1.2,
        "curve": "exponential",
        "display_name": "สถานี BTS (รถไฟฟ้า)",
        "poi_type": "rapid_transit",
        "category": "transportation",
    },
    "mrt": {
        "radius": 3000,
        "weight": 1.2,
        "curve": "exponential",
        "display_name": "สถานี MRT (รถไฟฟ้าใต้ดิน)",
        "poi_type": "rapid_transit",
        "category": "transportation",
    },
    "train_station": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "exponential",
        "display_name": "สถานีรถไฟ (การรถไฟแห่งประเทศไทย)",
        "poi_type": "state_railway",  # Explicitly NOT rapid_transit
        "category": "transportation",
    },
    "bus_station": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "exponential",
        "display_name": "สถานีขนส่งบัสและสถานีรถ",
        "poi_type": "bus",
        "category": "transportation",
    },

    # === CONVENIENCE ===
    "convenience_store": {
        "radius": 3000,
        "weight": 0.5,
        "curve": "exponential",
        "display_name": "ร้านสะดวกซื้อ (7-11 / Family Mart)",
        "poi_type": "convenience",
        "category": "shopping",
    },
    "market": {
        "radius": 1500,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "ตลาด / ตลาดสด",
        "poi_type": "market",
        "category": "shopping",
    },
    "supermarket": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "ซูเปอร์มาร์เก็ต",
        "poi_type": "convenience",
        "category": "shopping",
    },

    # === LIFESTYLE ===
    "shopping_mall": {
        "radius": 3000,
        "weight": 1.1,
        "curve": "linear",
        "display_name": "ห้างสรรพสินค้า / ShoppingMall",
        "poi_type": "lifestyle",
        "category": "shopping",
    },
    "community_mall": {
        "radius": 2000,
        "weight": 0.7,
        "curve": "linear",
        "display_name": "คอมมูนิตี้มอลล์",
        "poi_type": "lifestyle",
        "category": "shopping",
    },
    "restaurant": {
        "radius": 1000,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "ร้านอาหาร",
        "poi_type": "dining",
        "category": "dining",
    },
    "cafe": {
        "radius": 1000,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "คาเฟ่",
        "poi_type": "dining",
        "category": "dining",
    },

    # === HEALTH & WELLNESS ===
    "hospital": {
        "radius": 3000,
        "weight": 0.7,
        "curve": "linear",
        "display_name": "โรงพยาบาล",
        "poi_type": "health",
        "category": "health",
    },
    "park": {
        "radius": 3000,
        "weight": 0.6,
        "curve": "linear",
        "display_name": "สวนสาธารณะ / สวนเฉพาะ",
        "poi_type": "recreation",
        "category": "recreation",
    },
    "gym": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "ห้องออกกำลังกาย / Fitness Center",
        "poi_type": "health",
        "category": "health",
    },
    "spa": {
        "radius": 2000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "สปา / นวดไทย",
        "poi_type": "wellness",
        "category": "health",
    },

    # === PET FRIENDLY ===
    "veterinary": {
        "radius": 2000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "คลินิกสัตวแพทย์ / Pet Hospital",
        "poi_type": "pet",
        "category": "pet",
    },

    # === EDUCATION ===
    "school": {
        "radius": 3000,
        "weight": 0.5,
        "curve": "linear",
        "display_name": "โรงเรียน / สถาบันการศึกษา",
        "poi_type": "education",
        "category": "education",
    },
    "university": {
        "radius": 3000,
        "weight": 0.3,
        "curve": "linear",
        "display_name": "มหาวิทยาลัย",
        "poi_type": "education",
        "category": "education",
    },
    "temple": {
        "radius": 1500,
        "weight": 0.1,
        "curve": "linear",
        "display_name": "วัด / สถานที่ศักดิ์สิทธิ์",
        "poi_type": "culture",
        "category": "culture",
    },
    "museum": {
        "radius": 5000,
        "weight": 0.1,
        "curve": "linear",
        "display_name": "พิพิธภัณฑ์",
        "poi_type": "culture",
        "category": "culture",
    },

    # === NATURE ===
    "river": {
        "radius": 1500,
        "weight": 0.4,
        "curve": "linear",
        "display_name": "แม่น้ำ / ชุมชนริมน้ำ",
        "poi_type": "nature",
        "category": "nature",
    },
    "beach": {
        "radius": 3000,
        "weight": 0.0,
        "curve": "linear",
        "display_name": "ทะเล / หาด",
        "poi_type": "nature",
        "category": "nature",
    },
    "viewpoint": {
        "radius": 3000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "จุดชมวิวเมืองและสถานที่ท่องเที่ยว",
        "poi_type": "attraction",
        "category": "tourism",
    },

    # === TRAVEL ===
    "tourist_attraction": {
        "radius": 3000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "สถานที่ท่องเที่ยว / Landmark",
        "poi_type": "attraction",
        "category": "tourism",
    },
    "hotel": {
        "radius": 2000,
        "weight": 0.1,
        "curve": "linear",
        "display_name": "โรงแรม / ที่พักแรม",
        "poi_type": "accommodation",
        "category": "tourism",
    },
    "golf_course": {
        "radius": 5000,
        "weight": 0.2,
        "curve": "linear",
        "display_name": "สนามกอล์ฟ",
        "poi_type": "recreation",
        "category": "recreation",
    },
}

# ============ ASSET TYPE MAPPING ============
# Maps Thai asset type names to database IDs
ASSET_ID_MAPPING: Dict[str, List[int]] = {
    # === Living ===
    "คอนโด": [3, 12],
    "ห้องชุด": [3, 11, 16],
    "บ้าน": [4, 15],
    "บ้านเดี่ยว": [4],
    "บ้านแฝด": [15],
    "ทาวน์โฮม": [1],
    "ทาวน์เฮ้าส์": [1],
    "อพาร์ทเมนท์": [17, 30],
    "หอพัก": [30],

    # === Commercial ===
    "อาคารพาณิชย์": [5],
    "ตึกแถว": [5, 30],
    "โฮมออฟฟิศ": [9],
    "สำนักงาน": [11, 13],
    "ออฟฟิศ": [9, 11, 13],
    "โชว์รูม": [8],
    "ห้าง": [22],
    "ร้านอาหาร": [35],
    "ตลาด": [25],
    "ปั๊มน้ำมัน": [14],

    # === Industrial/Land ===
    "ที่ดิน": [2],
    "ที่ดินเปล่า": [2],
    "โรงงาน": [6, 36],
    "โกดัง": [6, 34],
    "คลังสินค้า": [6, 34],

    # === Tourism ===
    "โรงแรม": [10],
    "รีสอร์ท": [10],
    "โรงเรียน": [29],
    "โรงพยาบาล": [18, 19],
    "สนามกอล์ฟ": [21],
}

# Asset IDs that typically allow pets (บ้านแนวราบ)
PET_FRIENDLY_ASSET_IDS = [4, 15, 1]  # บ้านเดี่ยว, บ้านแฝด, ทาวน์โฮม

# Condo IDs (typically don't allow pets unless explicitly stated)
CONDO_ASSET_IDS = [3, 12]


def get_poi_display_name(poi_key: str) -> str:
    """Get display name for a POI key."""
    return POI_CONFIG.get(poi_key, {}).get("display_name", poi_key)


def get_poi_radius(poi_key: str) -> int:
    """Get radius threshold for a POI key."""
    return POI_CONFIG.get(poi_key, {}).get("radius", 3000)


def is_rapid_transit(poi_key: str) -> bool:
    """Check if a POI is rapid transit (BTS/MRT) vs regular train."""
    return POI_CONFIG.get(poi_key, {}).get("poi_type") == "rapid_transit"
