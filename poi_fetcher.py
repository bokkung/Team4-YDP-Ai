"""
POI Fetcher using Google Maps API - ENHANCED VERSION
Includes tourism, landmarks, lifestyle, and beach/water locations
COMPLETE AND READY TO RUN
"""
import os
import json
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import googlemaps
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# ============ CONFIGURATION ============
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
CSV_PATH = Path(r"C:\Users\bokthaiban\Desktop\mercilnew\data\assets_rows.csv")
OUTPUT_PATH = Path(r"C:\Users\bokthaiban\Desktop\mercilnew\poi_results_enhanced.csv")
CACHE_FILE = Path("poi_cache_enhanced.json")

#  ENHANCED POI TYPES - Include tourism, landmarks, and Thai lifestyle
POI_TYPES = {
    # === ESSENTIAL SERVICES ===
    "school": "school",
    "park": "park",
    "hospital": "hospital",
    "veterinary": "veterinary_care",

    # === SHOPPING & CONVENIENCE ===
    "convenience_store": "convenience_store",
    "shopping_mall": "shopping_mall",
    "market": "market",
    "supermarket": "supermarket",
    "community_mall": "shopping_mall",  

    # === TRANSPORTATION ===
    "bts_station": "subway_station",
    "mrt": "subway_station",          
    "train_station": "train_station",
    "bus_station": "bus_station",
    
    # === TOURISM & LANDMARKS ===
    "beach": "point_of_interest",  
    "temple": "place_of_worship",  
    "museum": "museum",
    "tourist_attraction": "tourist_attraction",
    "viewpoint": "point_of_interest",  
    "river": "natural_feature",
    "golf_course": "golf_course",
    
    # === LIFESTYLE & WELLNESS ===
    "restaurant": "restaurant",
    "cafe": "cafe",
    "gym": "gym",
    "sports_club": "sports_club",
    "spa": "spa",
    "hotel": "hotel",
    "university": "university",
}

SEARCH_RADIUS = 3000  # meters

print("🔑 Google Maps API Configuration")
if not GOOGLE_MAPS_API_KEY:
    print("❌ GOOGLE_MAPS_API_KEY not found!")
    print("   Set it: set GOOGLE_MAPS_API_KEY=your_api_key")
    exit(1)

print(f" API Key found: {GOOGLE_MAPS_API_KEY[:20]}...")

# ============ HELPERS ============
def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

def find_nearest_poi(gmaps, lat: float, lon: float, poi_type: str, location_name: str = "") -> Optional[dict]:
    """
    Find nearest POI using Google Maps Nearby Search
    Returns dict with {'distance': meters, 'name': poi_name}, or None if not found
    """
    try:
        cache_key = f"{lat:.6f}_{lon:.6f}_{poi_type}"
        cache = load_cache()
        
        if cache_key in cache:
            print(f"      [CACHE HIT]", end=" ")
            return cache[cache_key]
        
        print(f"\n      [API CALL]", end=" ")
        
        # Nearby search
        try:
            print(f"places_nearby(location=({lat:.4f},{lon:.4f}), type={poi_type})...", end=" ")
            results = gmaps.places_nearby(
                location=(lat, lon),
                radius=SEARCH_RADIUS,
                type=poi_type,
                language="th"
            )
            print(f"OK", end=" ")
        except Exception as e:
            print(f"❌ API ERROR: {str(e)[:80]}")
            return None
        
        print(f"Found {len(results.get('results', []))} results", end=" ")
        
        if results["results"]:
            nearest = results["results"][0]
            lat_dest = nearest["geometry"]["location"]["lat"]
            lon_dest = nearest["geometry"]["location"]["lng"]
            name = nearest.get("name", "Unknown")
            
            print(f"| Nearest: {name[:30]}", end=" ")
            
            # Calculate distance
            try:
                distance_result = gmaps.distance_matrix(
                    origins=(lat, lon),
                    destinations=(lat_dest, lon_dest),
                    mode="driving",
                    language="th"
                )
            except Exception as e:
                print(f"❌ distance_matrix ERROR: {str(e)[:60]}")
                return None
            
            status = distance_result["rows"][0]["elements"][0]["status"]
            print(f"| distance_status: {status}", end=" ")
            
            if status == "OK":
                dist_meters = distance_result["rows"][0]["elements"][0]["distance"]["value"]
                
                print(f"| {dist_meters}m → RETURN ")
                
                result_dict = {
                    "distance": dist_meters,
                    "name": name,
                    "place_id": nearest.get("place_id", "")
                }
                cache[cache_key] = result_dict
                save_cache(cache)
                return result_dict
            else:
                print(f"❌ Not OK status")
        else:
            print(f"❌ No results")
        
        cache[cache_key] = None
        save_cache(cache)
        return None
        
    except Exception as e:
        print(f"❌ Exception: {str(e)[:80]}")
        return None

def main():
    print("\n📍 Loading property data...")
    if not CSV_PATH.exists():
        print(f"❌ CSV not found: {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH).fillna("")
    print(f" Loaded {len(df)} properties")
    
    # Validate location data
    valid_locations = df[df['location_latitude'].notna() & df['location_longitude'].notna()]
    missing_locations = len(df) - len(valid_locations)
    
    print(f"\n Location Data Quality:")
    print(f"   Total properties: {len(df)}")
    print(f"   Valid locations: {len(valid_locations)}")
    print(f"   Missing locations: {missing_locations}")
    print(f"   Location coverage: {(len(valid_locations)/len(df)*100):.1f}%")
    
    if missing_locations > 0:
        print("\n⚠️ Warning: Some properties are missing location data")
    
    # Initialize Google Maps client
    print("\n Initializing Google Maps client...")
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    
    # Check if lat/lon exist
    if "location_latitude" not in df.columns or "location_longitude" not in df.columns:
        print("❌ location_latitude or location_longitude columns not found!")
        return
    
    # Create POI results dataframe
    poi_results = pd.DataFrame()
    poi_results["id"] = df["id"]
    
    # Initialize POI columns (distance + name)
    for poi_key in POI_TYPES.keys():
        poi_results[poi_key] = None  # distance
        poi_results[f"{poi_key}_name"] = None
    
    print("\n🔍 Fetching POI data from Google Maps...")
    print(f"   POI types: {list(POI_TYPES.keys())}")
    print(f"   Total POI types: {len(POI_TYPES)}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prop_id = row.get("id")
        lat = pd.to_numeric(row.get("location_latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("location_longitude"), errors="coerce")
        name = row.get("name_th", "")
        
        print(f"\n📍 Property #{idx}: {name}")
        print(f"   Location: ({lat:.4f}, {lon:.4f})")
        
        if pd.isna(lat) or pd.isna(lon):
            print(f"   ❌ Invalid coordinates - skipping")
            continue
        
        # Fetch each POI type
        for poi_key, poi_type in POI_TYPES.items():
            print(f"   🔍 Finding {poi_key} (type: {poi_type})...", end=" ")
            result = find_nearest_poi(gmaps, lat, lon, poi_type, name)
            
            if result and isinstance(result, dict):  #  Check if dict
                distance = result.get("distance", None)
                poi_name = result.get("name", "Unknown")
                print(f" {poi_name} ({distance}m)")
                poi_results.at[idx, poi_key] = distance
                poi_results.at[idx, f"{poi_key}_name"] = poi_name
            else:
                print(f"❌ Not found")
                poi_results.at[idx, poi_key] = None
                poi_results.at[idx, f"{poi_key}_name"] = None
        
        # Rate limiting (Google Maps API)
        time.sleep(0.1)
    
    # Save results
    print(f"\n Saving POI results to {OUTPUT_PATH}...")
    poi_results.to_csv(OUTPUT_PATH, index=False)
    print(f" Saved {len(poi_results)} records")
    
    # Display summary
    print("\n POI Summary:")
    print("\n=== ESSENTIAL SERVICES ===")
    for poi_key in ["school", "park", "hospital", "veterinary"]:
        if poi_key in poi_results.columns:
            found = poi_results[poi_key].notna().sum()
            valid_data = poi_results[poi_results[poi_key] != float('inf')][poi_key]
            avg_dist = valid_data.mean() if len(valid_data) > 0 else 0
            print(f"   {poi_key}: {found}/{len(poi_results)} found | Avg distance: {avg_dist:.0f}m")
    
    print("\n=== SHOPPING & CONVENIENCE ===")
    for poi_key in ["convenience_store", "shopping_mall", "market", "supermarket"]:
        if poi_key in poi_results.columns:
            found = poi_results[poi_key].notna().sum()
            valid_data = poi_results[poi_results[poi_key] != float('inf')][poi_key]
            avg_dist = valid_data.mean() if len(valid_data) > 0 else 0
            print(f"   {poi_key}: {found}/{len(poi_results)} found | Avg distance: {avg_dist:.0f}m")
    
    print("\n=== TRANSPORTATION ===")
    for poi_key in ["bts_station", "train_station", "bus_station"]:
        if poi_key in poi_results.columns:
            found = poi_results[poi_key].notna().sum()
            valid_data = poi_results[poi_results[poi_key] != float('inf')][poi_key]
            avg_dist = valid_data.mean() if len(valid_data) > 0 else 0
            print(f"   {poi_key}: {found}/{len(poi_results)} found | Avg distance: {avg_dist:.0f}m")
    
    print("\n=== TOURISM & LANDMARKS ===")
    for poi_key in ["beach", "temple", "museum", "tourist_attraction", "viewpoint", "river", "golf_course"]:
        if poi_key in poi_results.columns:
            found = poi_results[poi_key].notna().sum()
            valid_data = poi_results[poi_results[poi_key] != float('inf')][poi_key]
            avg_dist = valid_data.mean() if len(valid_data) > 0 else 0
            print(f"   {poi_key}: {found}/{len(poi_results)} found | Avg distance: {avg_dist:.0f}m")
    
    print("\n=== LIFESTYLE & WELLNESS ===")
    for poi_key in ["restaurant", "cafe", "gym", "spa", "hotel", "university"]:
        if poi_key in poi_results.columns:
            found = poi_results[poi_key].notna().sum()
            valid_data = poi_results[poi_results[poi_key] != float('inf')][poi_key]
            avg_dist = valid_data.mean() if len(valid_data) > 0 else 0
            print(f"   {poi_key}: {found}/{len(poi_results)} found | Avg distance: {avg_dist:.0f}m")

if __name__ == "__main__":
    main()