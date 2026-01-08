"""
Structured Scorer Module
========================
Pure structured constraint matching with NO semantic score mixing.
Uses multiplicative gating: hard constraint failures = disqualification.

Key design principles:
1. Semantic score is NOT used for ranking (only retrieval)
2. Wrong asset type = immediate disqualification
3. Missing data ≠ penalty, tracked separately
4. All scoring decisions are explainable
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import logging

from search_config import (
    POI_CONFIG,
    ASSET_ID_MAPPING,
    SCORING_WEIGHTS,
    HARD_CONSTRAINT_CONFIG,
    PET_FRIENDLY_ASSET_IDS,
    CONDO_ASSET_IDS,
    TARGET_LOCATION_CONFIG,
)
from data_quality import DataQualityReport, get_verified_distance
import geocoding_service

logger = logging.getLogger("structured_scorer")


@dataclass
class ScoringResult:
    """
    Explicit scoring result with disqualification flag.
    All scoring decisions are captured for explainability.
    """
    score: float
    is_disqualified: bool
    disqualification_reason: Optional[str]
    positive_signals: List[str] = field(default_factory=list)
    negative_signals: List[str] = field(default_factory=list)
    data_quality: Optional[DataQualityReport] = None
    
    # Breakdown for debugging
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def add_positive(self, signal: str, score_delta: float = 0.0):
        """Add a positive signal with optional score contribution."""
        self.positive_signals.append(signal)
        if score_delta:
            self.score_breakdown[signal[:50]] = score_delta
    
    def add_negative(self, signal: str, score_delta: float = 0.0):
        """Add a negative signal with optional score penalty."""
        self.negative_signals.append(signal)
        if score_delta:
            self.score_breakdown[signal[:50]] = score_delta


class StructuredScorer:
    """
    Scores assets based ONLY on structured constraint matching.
    
    Architecture:
    1. Hard constraint gates (any failure = disqualified)
    2. Soft scoring signals (additive, cannot rescue disqualification)
    3. All decisions logged for explainability
    """
    
    def __init__(self):
        self.poi_config = POI_CONFIG
        self.asset_mapping = ASSET_ID_MAPPING
        self.weights = SCORING_WEIGHTS
        self.hard_constraints = HARD_CONSTRAINT_CONFIG
    
    def score(
        self,
        metadata: Dict[str, Any],
        intent: Dict[str, Any],
        quality: DataQualityReport,
        target_location_coords: Optional[Tuple[float, float]] = None,
        avoid_location_coords: Optional[Tuple[float, float]] = None
    ) -> ScoringResult:
        """
        Main scoring function.
        
        Args:
            metadata: Asset metadata dictionary
            intent: Parsed user intent
            quality: Data quality report for this asset
        
        Returns:
            ScoringResult with score, disqualification status, and explanations
        """
        result = ScoringResult(
            score=0.0,
            is_disqualified=False,
            disqualification_reason=None,
            data_quality=quality,
        )
        
        # ===== GATE 1: Asset Type (Hard Constraint) =====
        type_check = self._check_asset_type(metadata, intent)
        if type_check.is_disqualified:
            return type_check
        result.score += type_check.score
        result.positive_signals.extend(type_check.positive_signals)
        result.score_breakdown.update(type_check.score_breakdown)
        
        # ===== GATE 2: Transport Type Mismatch (Hard Constraint) =====
        transport_check = self._check_transport_type_mismatch(metadata, intent, quality)
        if transport_check.is_disqualified:
            return transport_check
        result.negative_signals.extend(transport_check.negative_signals)
        
        # ===== SCORE: Rapid Transit (BTS/MRT) if in must_have =====
        rapid_score, rapid_signals = self._score_rapid_transit(metadata, intent, quality)
        result.score += rapid_score
        result.positive_signals.extend(rapid_signals)
        if rapid_score != 0:
            result.score_breakdown["rapid_transit"] = rapid_score
        
        # ===== GATE 3: Must-Have POIs (Hard Constraint) =====
        poi_check = self._check_must_have_pois(metadata, intent, quality)
        if poi_check.is_disqualified:
            return poi_check
        result.score += poi_check.score
        result.positive_signals.extend(poi_check.positive_signals)
        result.negative_signals.extend(poi_check.negative_signals)
        result.score_breakdown.update(poi_check.score_breakdown)
        
        # ===== SOFT SIGNALS (Cannot rescue hard failures) =====
        
        # Pet-friendly scoring
        pet_score, pet_signals = self._score_pet_friendly(metadata, intent)
        result.score += pet_score
        result.positive_signals.extend([s for s in pet_signals if s.startswith("✅")])
        result.negative_signals.extend([s for s in pet_signals if not s.startswith("✅")])
        if pet_score != 0:
            result.score_breakdown["pet_friendly"] = pet_score
        
        # Nice-to-have POIs
        nice_score, nice_signals = self._score_nice_to_have(metadata, intent, quality)
        result.score += nice_score
        result.positive_signals.extend(nice_signals)
        if nice_score != 0:
            result.score_breakdown["nice_to_have"] = nice_score
        
        # Avoid POIs (HARD CONSTRAINT - can disqualify)
        avoid_check = self._check_avoid_pois(metadata, intent, quality)
        if avoid_check.is_disqualified:
            return avoid_check
        result.score += avoid_check.score
        for s in avoid_check.positive_signals:
            result.positive_signals.append(s)
        for s in avoid_check.negative_signals:
            result.negative_signals.append(s)
        if avoid_check.score != 0:
            result.score_breakdown["avoid_pois"] = avoid_check.score
        
        # Price range
        price_score, price_signals = self._score_price_range(metadata, intent)
        result.score += price_score
        result.positive_signals.extend([s for s in price_signals if s.startswith("✅")])
        result.negative_signals.extend([s for s in price_signals if not s.startswith("✅")])
        if price_score != 0:
            result.score_breakdown["price_range"] = price_score
        
        # Add data quality warnings
        result.negative_signals.extend(quality.warnings)
        
        # ===== GEOCODING: Target Location Proximity =====
        if target_location_coords:
            loc_check = self._score_target_location_proximity(metadata, target_location_coords)
            if loc_check.is_disqualified:
                return loc_check
            result.score += loc_check.score
            result.positive_signals.extend(loc_check.positive_signals)
            result.negative_signals.extend(loc_check.negative_signals)
            result.score_breakdown.update(loc_check.score_breakdown)

        # ===== GEOCODING: Avoid Location Proximity =====
        if avoid_location_coords:
            avoid_score, avoid_signals = self._score_avoid_location_proximity(metadata, avoid_location_coords)
            result.score += avoid_score
            result.positive_signals.extend([s for s in avoid_signals if s.startswith("✅")])
            result.negative_signals.extend([s for s in avoid_signals if not s.startswith("✅")])
            if avoid_score != 0:
                result.score_breakdown["avoid_location"] = avoid_score
        
        return result
    
    def _check_asset_type(self, metadata: Dict, intent: Dict) -> ScoringResult:
        """
        Hard gate: wrong asset type = disqualified.
        """
        intent_types = intent.get("asset_types", [])
        
        if not intent_types:
            # No type constraint specified
            return ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
        
        asset_id = int(metadata.get("asset_type_id", 0))
        asset_type_name = metadata.get("asset_type_fixed", "ทรัพย์สินอื่น")
        
        # Gather all accepted IDs
        accepted_ids = []
        for t in intent_types:
            accepted_ids.extend(self.asset_mapping.get(t, []))
        
        if asset_id in accepted_ids:
            result = ScoringResult(
                score=self.weights["asset_type_match"],
                is_disqualified=False,
                disqualification_reason=None,
            )
            result.add_positive(f"✅ ตรงประเภททรัพย์สิน ({asset_type_name})", self.weights["asset_type_match"])
            return result
        else:
            # HARD DISQUALIFICATION
            if self.hard_constraints.get("wrong_asset_type", True):
                return ScoringResult(
                    score=0,
                    is_disqualified=True,
                    disqualification_reason=f"ประเภทไม่ตรง: ต้องการ {', '.join(intent_types)} แต่พบ {asset_type_name}",
                )
            else:
                # Soft mode: heavy penalty but not disqualified
                result = ScoringResult(score=-10.0, is_disqualified=False, disqualification_reason=None)
                result.add_negative(f"❌ ไม่ตรงประเภท (ต้องการ {', '.join(intent_types)} แต่พบ {asset_type_name})", -10.0)
                return result
    
    def _check_transport_type_mismatch(
        self,
        metadata: Dict,
        intent: Dict,
        quality: DataQualityReport
    ) -> ScoringResult:
        """
        Hard gate: User wants BTS/MRT but only State Railway available.
        This is a semantic trap that the old code didn't handle well.
        """
        must_haves = intent.get("must_have", [])
        
        wants_rapid_transit = "bts_station" in must_haves or "mrt" in must_haves
        if not wants_rapid_transit:
            return ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
        
        # Check if we have rapid transit data
        bts_dist = get_verified_distance(metadata, "bts_station")
        mrt_dist = get_verified_distance(metadata, "mrt")
        train_dist = get_verified_distance(metadata, "train_station")
        
        has_rapid_transit = (
            (bts_dist is not None and bts_dist < 3000) or
            (mrt_dist is not None and mrt_dist < 3000)
        )
        has_state_train = train_dist is not None and train_dist < 2500
        
        if not has_rapid_transit and has_state_train:
            # User wants BTS/MRT but only State Railway is nearby
            if self.hard_constraints.get("wrong_transport_type", True):
                return ScoringResult(
                    score=0,
                    is_disqualified=True,
                    disqualification_reason=(
                        f"ต้องการรถไฟฟ้า BTS/MRT แต่มีเพียงสถานีรถไฟธรรมดา "
                        f"(BTS: {'ไม่มีข้อมูล' if bts_dist is None else f'{bts_dist:,.0f}ม.'}, "
                        f"MRT: {'ไม่มีข้อมูล' if mrt_dist is None else f'{mrt_dist:,.0f}ม.'}, "
                        f"รถไฟ: {train_dist:,.0f}ม.)"
                    ),
                )
            else:
                result = ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
                result.add_negative(
                    f"❌ ต้องการ BTS/MRT แต่มี State Railway เท่านั้น",
                    -20.0
                )
                return result
        
        return ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
    
    def _score_rapid_transit(
        self,
        metadata: Dict,
        intent: Dict,
        quality: DataQualityReport
    ) -> Tuple[float, List[str]]:
        """
        Score BTS/MRT proximity if they are in must_have.
        This is separate from _check_must_have_pois which skips rapid transit.
        """
        must_haves = intent.get("must_have", [])
        
        score = 0.0
        signals = []
        
        for poi_key in ["bts_station", "mrt"]:
            if poi_key not in must_haves:
                continue
            
            distance = get_verified_distance(metadata, poi_key)
            
            if distance is None:
                # Missing data - warning only
                display_name = self.poi_config[poi_key].get("display_name", poi_key)
                signals.append(f"⚠️ ไม่มีข้อมูล {display_name}")
                continue
            
            poi_cfg = self.poi_config[poi_key]
            limit_radius = poi_cfg.get("radius", 3000)
            display_name = poi_cfg.get("display_name", poi_key)
            specific_name = metadata.get(f"{poi_key}_name", display_name)
            
            if distance <= limit_radius:
                # Calculate score with exponential curve
                match_factor = (1 - (distance / limit_radius)) ** 2
                score_delta = self.weights["must_have_poi_base"] * max(0.1, match_factor)
                score += score_delta
                signals.append(f"✅ ใกล้ {display_name} '{specific_name}' ({distance:,.0f} ม.)")
        
        return score, signals
    
    def _check_must_have_pois(
        self,
        metadata: Dict,
        intent: Dict,
        quality: DataQualityReport
    ) -> ScoringResult:
        """
        Hard constraint: Must-have POIs must be within range.
        
        Key difference from old code:
        - Missing data = warning, not penalty
        - Verified far = disqualification
        """
        must_haves = intent.get("must_have", [])
        
        if not must_haves:
            return ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
        
        result = ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
        
        for poi_key in must_haves:
            if poi_key not in self.poi_config:
                continue
            
            # Skip rapid transit check here (handled by _check_transport_type_mismatch)
            if poi_key in ["bts_station", "mrt"]:
                continue
            
            poi_cfg = self.poi_config[poi_key]
            display_name = poi_cfg.get("display_name", poi_key)
            limit_radius = poi_cfg.get("radius", 3000)
            
            # Get verified distance (None if missing)
            distance = get_verified_distance(metadata, poi_key)
            
            if distance is None:
                # DATA MISSING - do not penalize, but note it
                result.add_negative(f"⚠️ ไม่มีข้อมูล {display_name} (cannot verify)")
                continue
            
            if distance <= limit_radius:
                # POI is within range - calculate score
                curve = poi_cfg.get("curve", "linear")
                if curve == "exponential":
                    match_factor = (1 - (distance / limit_radius)) ** 2
                else:
                    match_factor = 1 - (distance / limit_radius)
                
                score_delta = self.weights["must_have_poi_base"] * max(0.1, match_factor)
                result.score += score_delta
                
                specific_name = metadata.get(f"{poi_key}_name", display_name)
                result.add_positive(
                    f"✅ ใกล้ {display_name} '{specific_name}' ({distance:,.0f} ม.)",
                    score_delta
                )
            else:
                # POI exists but too far = DISQUALIFY
                if self.hard_constraints.get("missing_must_have_poi", True):
                    return ScoringResult(
                        score=0,
                        is_disqualified=True,
                        disqualification_reason=(
                            f"ต้องการ {display_name} แต่ห่าง {distance:,.0f} ม. "
                            f"(เกินระยะ {limit_radius:,.0f} ม.)"
                        ),
                        positive_signals=result.positive_signals,
                        negative_signals=result.negative_signals,
                    )
                else:
                    result.score -= 15.0
                    result.add_negative(
                        f"❌ ต้องการ {display_name} แต่ห่าง {distance:,.0f} ม. (เกินระยะ)",
                        -15.0
                    )
        
        return result
    
    def _score_pet_friendly(self, metadata: Dict, intent: Dict) -> Tuple[float, List[str]]:
        """
        Score pet-friendliness based on explicit data and asset type inference.
        """
        intent_pet = intent.get("pet_friendly")
        
        if intent_pet is None:
            return 0.0, []
        
        signals = []
        score = 0.0
        
        meta_pet_explicit = metadata.get("pet_friendly")
        asset_id = int(metadata.get("asset_type_id", 0))
        
        if intent_pet is True:  # User wants pet-friendly
            if meta_pet_explicit is True:
                score = self.weights["pet_friendly_explicit"]
                signals.append("✅ อนุญาตให้เลี้ยงสัตว์ (ระบุชัดเจน)")
                
            elif meta_pet_explicit is False:
                score = self.weights["pet_not_allowed_condo"]
                signals.append("❌ ไม่อนุญาตให้เลี้ยงสัตว์ (ระบุชัดเจน)")
                
            elif meta_pet_explicit is None:
                # Infer from asset type
                if asset_id in CONDO_ASSET_IDS:
                    score = self.weights["pet_not_allowed_condo"]
                    signals.append("❌ น่าจะเลี้ยงสัตว์ไม่ได้ (คอนโดส่วนใหญ่ห้ามเลี้ยง)")
                    
                elif asset_id in PET_FRIENDLY_ASSET_IDS:
                    score = self.weights["pet_friendly_inferred"]
                    signals.append("✅ น่าจะเลี้ยงสัตว์ได้ (เป็นบ้านแนวราบ)")
                    
                else:
                    score = self.weights["pet_status_unknown"]
                    signals.append("⚠️ ไม่ระบุเรื่องเลี้ยงสัตว์ (ต้องยืนยัน)")
            
            # Bonus for nearby vet
            vet_dist = get_verified_distance(metadata, "veterinary")
            if vet_dist is not None and vet_dist <= 2000:
                score += self.weights["near_vet_bonus"]
                signals.append(f"✅ ใกล้คลินิกสัตวแพทย์ ({vet_dist:.0f} ม.)")
                
        elif intent_pet is False:  # User doesn't want pet-friendly
            if meta_pet_explicit is True:
                score = -2.0
                signals.append("⚠️ เป็นสถานที่ Pet Friendly (อาจมีเสียงรบกวน)")
        
        return score, signals
    
    def _score_nice_to_have(
        self,
        metadata: Dict,
        intent: Dict,
        quality: DataQualityReport
    ) -> Tuple[float, List[str]]:
        """
        Score nice-to-have POIs (bonus only, no penalty).
        """
        nice_to_haves = intent.get("nice_to_have", [])
        
        if not nice_to_haves:
            return 0.0, []
        
        score = 0.0
        signals = []
        
        for poi_key in nice_to_haves:
            if poi_key not in self.poi_config:
                continue
            
            distance = get_verified_distance(metadata, poi_key)
            
            if distance is None:
                continue  # Missing data = no bonus, no penalty
            
            poi_cfg = self.poi_config[poi_key]
            limit_radius = poi_cfg.get("radius", 2000)
            display_name = poi_cfg.get("display_name", poi_key)
            
            if distance <= limit_radius:
                score += self.weights["nice_to_have_poi"]
                specific_name = metadata.get(f"{poi_key}_name", display_name)
                signals.append(f"➕ มี {display_name} '{specific_name}' ({distance:.0f} ม.) [Bonus]")
        
        return score, signals
    
    def _check_avoid_pois(
        self,
        metadata: Dict,
        intent: Dict,
        quality: DataQualityReport
    ) -> ScoringResult:
        """
        HARD CONSTRAINT: Avoid POIs must not be too close.
        
        Key fix: If property is too close to avoided POI = DISQUALIFY.
        Missing data = no penalty (cannot verify).
        """
        avoid_pois = intent.get("avoid_poi", [])
        
        if not avoid_pois:
            return ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
        
        result = ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
        
        for poi_key in avoid_pois:
            if poi_key not in self.poi_config:
                continue
            
            distance = get_verified_distance(metadata, poi_key)
            
            if distance is None:
                # Missing data = cannot verify avoidance
                # NO PENALTY (fixes old bug)
                continue
            
            poi_cfg = self.poi_config[poi_key]
            # ใช้ 60% ของ radius เป็น avoid threshold
            avoid_radius = poi_cfg.get("radius", 1000) * 0.6
            display_name = poi_cfg.get("display_name", poi_key)
            
            if distance <= avoid_radius:
                # Too close to avoided POI = DISQUALIFY
                if self.hard_constraints.get("avoid_poi_too_close", True):
                    return ScoringResult(
                        score=0,
                        is_disqualified=True,
                        disqualification_reason=(
                            f"ต้องหลีกเลี่ยง {display_name} แต่ห่างเพียง {distance:,.0f} ม. "
                            f"(ต้องห่างอย่างน้อย {avoid_radius:,.0f} ม.)"
                        ),
                        positive_signals=result.positive_signals,
                        negative_signals=result.negative_signals,
                    )
                else:
                    # Soft mode: heavy penalty but not disqualified
                    result.score += self.weights["avoid_poi_failure"]
                    result.add_negative(f"❌ อยู่ใกล้ {display_name} (ต้องหลีกเลี่ยง) - ห่าง {distance:,.0f} ม.")
            else:
                # Successfully avoided (verified far)
                result.score += self.weights["avoid_poi_success"]
                result.add_positive(f"✅ หลีกเลี่ยง {display_name} ได้ (ห่าง {distance:,.0f} ม.)")
        
        return result
    
    def _score_target_location_proximity(
        self,
        metadata: Dict,
        target_coords: Tuple[float, float]
    ) -> ScoringResult:
        """
        Score based on proximity to a specific target location (from Geocoding).
        Hard Constraint: If distance > radius_far_limit, disqualify.
        """
        asset_lat = metadata.get("latitude") or metadata.get("location_latitude")
        asset_lng = metadata.get("longitude") or metadata.get("location_longitude")
        
        # Check if asset has valid coordinates
        if not asset_lat or not asset_lng:
            try:
                asset_lat = float(asset_lat)
                asset_lng = float(asset_lng)
            except (ValueError, TypeError):
                # Cannot verify - returning neutral score but with warning
                res = ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
                res.add_negative("⚠️ ไม่มีพิกัดทรัพย์สิน (คำนวณระยะห่างไม่ได้)")
                return res
        
        try:
            asset_lat = float(asset_lat)
            asset_lng = float(asset_lng)
        except (ValueError, TypeError):
             res = ScoringResult(score=0, is_disqualified=False, disqualification_reason=None)
             res.add_negative("⚠️ พิกัดทรัพย์สินไม่ถูกต้อง")
             return res

        target_lat, target_lng = target_coords
        
        distance = geocoding_service.calculate_haversine_distance(
            asset_lat, asset_lng, target_lat, target_lng
        )
        
        radius_very_close = TARGET_LOCATION_CONFIG["radius_very_close"]
        radius_close = TARGET_LOCATION_CONFIG["radius_close"]
        radius_far_limit = TARGET_LOCATION_CONFIG["radius_far_limit"]
        
        # Scoring Logic
        res = ScoringResult(score=0.0, is_disqualified=False, disqualification_reason=None)
        
        if distance <= radius_very_close:
            res.score = self.weights["location_very_close"]
            res.add_positive(f"✅ อยู่ในระยะใกล้มาก ({distance/1000:.1f} กม.)", self.weights["location_very_close"])
            
        elif distance <= radius_close:
            res.score = self.weights["location_close"]
            res.add_positive(f"✅ อยู่ในระยะเดินทางสะดวก ({distance/1000:.1f} กม.)", self.weights["location_close"])
            
        elif distance > radius_far_limit:
            # HARD DISQUALIFICATION CHECK
            if self.hard_constraints.get("target_location_too_far", True):
                return ScoringResult(
                    score=0, 
                    is_disqualified=True, 
                    disqualification_reason=f"ไกลเกินไป: ห่างจากจุดเป้าหมาย {distance/1000:.1f} กม. (เกิน {radius_far_limit/1000:.0f} กม.)"
                )
            else:
                res.score = self.weights["location_far"]
                res.add_negative(f"❌ ไกลจากจุดที่ค้นหา ({distance/1000:.1f} กม.)", self.weights["location_far"])
        else:
            # Between close and far limit (Neutral zone, maybe small negative or 0)
            res.add_negative(f"⚠️ อยู่ในระยะปานกลาง ({distance/1000:.1f} กม.)")
            
        return res

    def _score_avoid_location_proximity(
        self,
        metadata: Dict,
        avoid_coords: Tuple[float, float]
    ) -> Tuple[float, List[str]]:
        """
        Score based on proximity to a location to AVOID.
        """
        asset_lat = metadata.get("latitude") or metadata.get("location_latitude")
        asset_lng = metadata.get("longitude") or metadata.get("location_longitude")
        
        # Check if asset has valid coordinates
        if not asset_lat or not asset_lng:
             # If we can't verify location, we can't confirm avoidance.
             # Neutral score, but warn.
             return 0.0, ["⚠️ ไม่มีพิกัดทรัพย์สิน (ตรวจสอบระยะห่างที่ต้องหลีกเลี่ยงไม่ได้)"]

        try:
            asset_lat = float(asset_lat)
            asset_lng = float(asset_lng)
        except (ValueError, TypeError):
             return 0.0, ["⚠️ พิกัดทรัพย์สินไม่ถูกต้อง"]

        target_lat, target_lng = avoid_coords
        
        distance = geocoding_service.calculate_haversine_distance(
            asset_lat, asset_lng, target_lat, target_lng
        )
        
        score = 0.0
        signals = []
        
        # Logic: Closer = Worse
        if distance <= 2000:
            score = self.weights["avoid_location_hit_hard"]
            signals.append(f"❌ อยู่ใกล้จุดที่ต้องการเลี่ยงมาก ({distance/1000:.1f} กม.)")
        elif distance <= 5000:
            score = self.weights["avoid_location_hit_soft"]
            signals.append(f"⚠️ อยู่ในรัศมีที่ควรเลี่ยง ({distance/1000:.1f} กม.)")
        else:
            score = self.weights["avoid_location_success"]
            signals.append(f"✅ ห่างจากจุดที่ต้องการเลี่ยง ({distance/1000:.1f} กม.)")
            
        return score, signals

    def _score_price_range(self, metadata: Dict, intent: Dict) -> Tuple[float, List[str]]:
        """
        Score price range matching.
        """
        price_range = intent.get("price_range", {})
        min_price = price_range.get("min")
        max_price = price_range.get("max")
        
        if min_price is None and max_price is None:
            return 0.0, []
        
        asset_price = float(metadata.get("asset_details_selling_price", 0))
        
        if asset_price == 0:
            return 0.0, ["⚠️ ไม่มีข้อมูลราคา"]
        
        signals = []
        score = 0.0
        
        if min_price is not None and asset_price < min_price:
            score = self.weights["price_out_of_range"]
            signals.append(f"⚠️ ราคาต่ำกว่าที่ต้องการ ({asset_price:,.0f} < {min_price:,.0f} บาท)")
            
        elif max_price is not None and asset_price > max_price:
            score = self.weights["price_out_of_range"]
            signals.append(f"⚠️ ราคาสูงกว่าที่ต้องการ ({asset_price:,.0f} > {max_price:,.0f} บาท)")
            
        else:
            score = self.weights["price_in_range"]
            signals.append(f"✅ ราคาตรงในช่วงที่ต้องการ ({asset_price:,.0f} บาท)")
        
        return score, signals


# Singleton instance for convenience
_scorer_instance = None

def get_scorer() -> StructuredScorer:
    """Get or create the singleton scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = StructuredScorer()
    return _scorer_instance
