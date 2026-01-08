"""
Data Quality Module
===================
Handles missing/invalid data detection and quality assessment.
Replaces the broken 99999 magic number pattern.

Key principle: Missing data ≠ Penalty
Instead, we track what's known vs unknown and score accordingly.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Any, List, Optional
import logging

from search_config import DATA_QUALITY_CONFIG, POI_CONFIG

logger = logging.getLogger("data_quality")


@dataclass
class DataQualityReport:
    """
    Comprehensive report on data availability for an asset.
    This replaces the implicit 99999 = "far" assumption.
    """
    asset_id: str
    
    # POI data availability
    available_poi_keys: Set[str] = field(default_factory=set)
    missing_poi_keys: Set[str] = field(default_factory=set)
    
    # Core data validity
    has_valid_price: bool = False
    has_valid_asset_type: bool = False
    has_valid_location: bool = False
    
    # Overall quality score (0.0 to 1.0)
    quality_score: float = 0.0
    
    # Warnings for user
    warnings: List[str] = field(default_factory=list)
    
    def is_poi_available(self, poi_key: str) -> bool:
        """Check if we have verified data for a specific POI."""
        return poi_key in self.available_poi_keys
    
    def is_poi_missing(self, poi_key: str) -> bool:
        """Check if POI data is missing (not just far)."""
        return poi_key in self.missing_poi_keys
    
    def get_missing_must_haves(self, must_have_pois: List[str]) -> List[str]:
        """Get list of must-have POIs that have no data."""
        return [poi for poi in must_have_pois if poi in self.missing_poi_keys]


def is_missing_value(value: Any) -> bool:
    """
    Check if a value represents missing data.
    Handles legacy 99999 sentinel values.
    """
    if value is None:
        return True
    
    sentinels = DATA_QUALITY_CONFIG.get("missing_data_sentinels", [99999, 99999.0, None])
    
    if value in sentinels:
        return True
    
    # Check for very large numbers that likely indicate missing data
    if isinstance(value, (int, float)) and value >= 90000:
        return True
    
    return False


def assess_data_quality(
    metadata: Dict[str, Any],
    required_pois: List[str],
    nice_to_have_pois: Optional[List[str]] = None
) -> DataQualityReport:
    """
    Assess the data quality for an asset's metadata.
    
    Args:
        metadata: Asset metadata dictionary
        required_pois: List of POI keys that are required (must-have)
        nice_to_have_pois: Optional list of nice-to-have POI keys
    
    Returns:
        DataQualityReport with detailed availability information
    """
    asset_id = str(metadata.get("id", metadata.get("asset_id", "unknown")))
    
    available_pois: Set[str] = set()
    missing_pois: Set[str] = set()
    warnings: List[str] = []
    
    # Check all POIs we might care about
    all_pois_to_check = set(required_pois or [])
    if nice_to_have_pois:
        all_pois_to_check.update(nice_to_have_pois)
    
    for poi_key in all_pois_to_check:
        value = metadata.get(poi_key)
        
        if is_missing_value(value):
            missing_pois.add(poi_key)
            
            # Add warning if this is a required POI
            if poi_key in (required_pois or []):
                display_name = POI_CONFIG.get(poi_key, {}).get("display_name", poi_key)
                warnings.append(f"⚠️ ไม่มีข้อมูล {display_name} (cannot verify)")
        else:
            # Valid data exists
            if isinstance(value, (int, float)) and value >= 0:
                available_pois.add(poi_key)
    
    # Check core data validity
    has_valid_price = not is_missing_value(metadata.get("asset_details_selling_price"))
    has_valid_asset_type = metadata.get("asset_type_id") is not None
    
    # Location check
    has_valid_location = bool(
        metadata.get("location_village_th") or 
        metadata.get("location_road_th") or
        ((metadata.get("latitude") or metadata.get("location_latitude")) and 
         (metadata.get("longitude") or metadata.get("location_longitude")))
    )
    
    # Calculate quality score
    total_checked = len(all_pois_to_check) if all_pois_to_check else 1
    poi_completeness = len(available_pois) / total_checked if total_checked > 0 else 1.0
    
    # Weight core data more heavily
    core_score = (
        (0.3 if has_valid_price else 0) +
        (0.2 if has_valid_asset_type else 0) +
        (0.1 if has_valid_location else 0)
    )
    
    quality_score = (poi_completeness * 0.4) + core_score
    
    return DataQualityReport(
        asset_id=asset_id,
        available_poi_keys=available_pois,
        missing_poi_keys=missing_pois,
        has_valid_price=has_valid_price,
        has_valid_asset_type=has_valid_asset_type,
        has_valid_location=has_valid_location,
        quality_score=quality_score,
        warnings=warnings,
    )


def get_verified_distance(metadata: Dict[str, Any], poi_key: str) -> Optional[float]:
    """
    Get POI distance only if the data is verified present.
    Returns None if data is missing (instead of 99999).
    
    This is the proper replacement for:
        distance = metadata.get(poi_key, 99999)  # WRONG
    """
    value = metadata.get(poi_key)
    
    if is_missing_value(value):
        return None
    
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def batch_assess_quality(
    results: List[Dict[str, Any]],
    required_pois: List[str],
    nice_to_have_pois: Optional[List[str]] = None
) -> Dict[str, DataQualityReport]:
    """
    Assess data quality for multiple results at once.
    Returns a mapping of asset_id -> DataQualityReport.
    """
    reports = {}
    
    for result in results:
        metadata = result.get("metadata", result)
        report = assess_data_quality(metadata, required_pois, nice_to_have_pois)
        reports[report.asset_id] = report
    
    return reports
