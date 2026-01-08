"""
Geocoding Service
=================
Handles conversion of location names to coordinates using Google Maps API.
Includes caching and distance calculation utilities.
"""

import os
import logging
import math
from typing import Optional, Tuple, Dict
from functools import lru_cache

try:
    import googlemaps
except ImportError:
    googlemaps = None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("geocoding_service")

# Global client instance
_gmaps_client = None

def get_google_maps_client():
    """Get or create Google Maps client."""
    global _gmaps_client
    if _gmaps_client is None:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if api_key and googlemaps:
            try:
                _gmaps_client = googlemaps.Client(key=api_key)
                logger.info("✅ Google Maps client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Google Maps client: {e}")
                _gmaps_client = None
        else:
            if not api_key:
                logger.warning("⚠️ GOOGLE_MAPS_API_KEY not found in .env")
            if not googlemaps:
                logger.error("❌ googlemaps library not installed")
    return _gmaps_client

@lru_cache(maxsize=100)
def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Convert a location name to (latitude, longitude).
    Cached to minimize API costs.
    
    Args:
        location_name: Name of the location (e.g., "Suankularb Rangsit School")
        
    Returns:
        Tuple of (lat, lng) or None if not found/error.
    """
    if not location_name:
        return None
        
    client = get_google_maps_client()
    if not client:
        logger.warning(f"⚠️ Cannot geocode '{location_name}': API client not available")
        return None
        
    try:
        # Geocode the location
        # Restrict to Thailand for better relevance (optional but recommended)
        result = client.geocode(location_name, region='th', language='th')
        
        if result and len(result) > 0:
            location = result[0]['geometry']['location']
            lat = location['lat']
            lng = location['lng']
            logger.info(f"📍 Geocoded '{location_name}' -> ({lat}, {lng})")
            return lat, lng
        else:
            logger.warning(f"⚠️ Location not found: '{location_name}'")
            return None
            
    except Exception as e:
        logger.error(f"❌ Geocoding error for '{location_name}': {e}")
        return None

def calculate_haversine_distance(
    lat1: float, lon1: float, 
    lat2: float, lon2: float
) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Result in meters.
    """
    try:
        if None in (lat1, lon1, lat2, lon2):
            return 999999.0  # Return distinct "far" value if coords missing
            
        # Convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        r = 6371 * 1000  # Radius of earth in meters
        return c * r
        
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return 999999.0
