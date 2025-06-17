import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ZoneType(Enum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    PSYCHOLOGICAL = "psychological"

class MarketRegime(Enum):
    RANGE = "range"
    TREND = "trend"

@dataclass
class Zone:
    """Represents a trading zone with all its properties"""
    zone_type: ZoneType
    price: float
    timestamp: pd.Timestamp
    strength: float = 0.0
    bounce_count: int = 0
    last_bounce_time: Optional[pd.Timestamp] = None
    volume_on_bounces: List[float] = None
    rsi_on_bounces: List[float] = None
    bb_touches: int = 0
    last_break_time: Optional[pd.Timestamp] = None
    score: float = 0.0
    is_valid: bool = True
    
    def __post_init__(self):
        if self.volume_on_bounces is None:
            self.volume_on_bounces = []
        if self.rsi_on_bounces is None:
            self.rsi_on_bounces = []

class ZoneDetector:
    """
    Phase 1 Zone Detection System
    Detects swing highs/lows, psychological levels, and calculates ML features
    """
    
    def __init__(self, 
                 window_size: int = 5,
                 min_amplitude_pct: float = 0.25,
                 bounce_lookback: int = 50,
                 validation_lookback: int = 20,
                 confluence_threshold: int = 2):
        
        self.window_size = window_size
        self.min_amplitude_pct = min_amplitude_pct
        self.bounce_lookback = bounce_lookback
        self.validation_lookback = validation_lookback
        self.confluence_threshold = confluence_threshold
        
        # Zone storage
        self.zones: List[Zone] = []
        
        # Market regime detection
        self.ma50_period = 50
        self.adx_period = 14
        self.adx_threshold = 20
        
    def detect_swing_points(self, df: pd.DataFrame) -> List[Zone]:
        """
        Detect swing high and low points in the data
        """
        zones = []
        min_amplitude = df['close'].iloc[-1] * (self.min_amplitude_pct / 100)
        
        # Detect swing highs
        for i in range(self.window_size, len(df) - self.window_size):
            current_price = df.iloc[i]['high']
            current_time = df.index[i]
            
            # Check if current point is higher than surrounding points
            is_swing_high = True
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i:
                    try:
                        if df.iloc[j]['high'] >= current_price:
                            is_swing_high = False
                            break
                    except (IndexError, KeyError):
                        continue
            
            if is_swing_high:
                # Check amplitude filter
                try:
                    surrounding_min = min(df.iloc[i-self.window_size:i+self.window_size+1]['low'])
                    amplitude = current_price - surrounding_min
                    
                    if amplitude >= min_amplitude:
                        zone = Zone(
                            zone_type=ZoneType.SWING_HIGH,
                            price=current_price,
                            timestamp=current_time
                        )
                        zones.append(zone)
                except (IndexError, KeyError):
                    continue
        
        # Detect swing lows
        for i in range(self.window_size, len(df) - self.window_size):
            current_price = df.iloc[i]['low']
            current_time = df.index[i]
            
            # Check if current point is lower than surrounding points
            is_swing_low = True
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i:
                    try:
                        if df.iloc[j]['low'] <= current_price:
                            is_swing_low = False
                            break
                    except (IndexError, KeyError):
                        continue
            
            if is_swing_low:
                # Check amplitude filter
                try:
                    surrounding_max = max(df.iloc[i-self.window_size:i+self.window_size+1]['high'])
                    amplitude = surrounding_max - current_price
                    
                    if amplitude >= min_amplitude:
                        zone = Zone(
                            zone_type=ZoneType.SWING_LOW,
                            price=current_price,
                            timestamp=current_time
                        )
                        zones.append(zone)
                except (IndexError, KeyError):
                    continue
        
        return zones
    
    def detect_psychological_levels(self, df: pd.DataFrame) -> List[Zone]:
        """
        Detect major psychological levels (00, 50)
        """
        zones = []
        current_price = df['close'].iloc[-1]
        
        # Find nearest psychological levels
        price_rounded = round(current_price, -2)  # Round to nearest 100
        levels = [
            price_rounded,  # 00 level
            price_rounded + 50,  # 50 level
            price_rounded - 50,  # 50 level below
            price_rounded + 100,  # Next 00 level
            price_rounded - 100   # Previous 00 level
        ]
        
        for level in levels:
            if level > 0:  # Ensure positive price
                zone = Zone(
                    zone_type=ZoneType.PSYCHOLOGICAL,
                    price=level,
                    timestamp=df.index[-1]
                )
                zones.append(zone)
        
        return zones
    
    def calculate_zone_strength(self, zone: Zone, df: pd.DataFrame) -> float:
        """
        Calculate zone strength with weighted components:
        - Bounce count (40%)
        - Time since creation (20%)
        - Average volume on bounces (40%)
        """
        # Count bounces in lookback period
        bounce_count = self._count_bounces(zone, df)
        
        # Time since creation (normalized to 0-1, newer = higher)
        time_since_creation = (df.index[-1] - zone.timestamp).total_seconds() / 3600  # hours
        time_score = max(0, 1 - (time_since_creation / 168))  # Decay over 1 week
        
        # Average volume on bounces
        avg_volume = np.mean(zone.volume_on_bounces) if zone.volume_on_bounces else 0
        volume_score = min(1, avg_volume / df['volume'].mean()) if df['volume'].mean() > 0 else 0
        
        # Weighted strength calculation
        strength = (0.4 * min(1, bounce_count / 5) +  # Normalize bounce count
                   0.2 * time_score +
                   0.4 * volume_score)
        
        return strength
    
    def _count_bounces(self, zone: Zone, df: pd.DataFrame) -> int:
        """
        Count how many times price bounced off this zone
        """
        if len(df) < 10:
            return 0
        
        bounce_count = 0
        lookback_df = df.tail(self.bounce_lookback)
        tolerance = df['close'].iloc[-1] * 0.001  # 0.1% tolerance
        
        # Use iterrows() for safer iteration
        for idx, row in lookback_df.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            volume = row['volume']
            
            # Check if price touched the zone
            if zone.zone_type == ZoneType.SWING_HIGH:
                if high >= zone.price - tolerance and high <= zone.price + tolerance:
                    # Check if it bounced (closed below the zone)
                    if close < zone.price - tolerance:
                        bounce_count += 1
                        zone.volume_on_bounces.append(volume)
                        zone.last_bounce_time = idx
            
            elif zone.zone_type == ZoneType.SWING_LOW:
                if low >= zone.price - tolerance and low <= zone.price + tolerance:
                    # Check if it bounced (closed above the zone)
                    if close > zone.price + tolerance:
                        bounce_count += 1
                        zone.volume_on_bounces.append(volume)
                        zone.last_bounce_time = idx
            
            elif zone.zone_type == ZoneType.PSYCHOLOGICAL:
                if low <= zone.price + tolerance and high >= zone.price - tolerance:
                    # For psychological levels, any touch counts as potential bounce
                    bounce_count += 1
                    zone.volume_on_bounces.append(volume)
                    zone.last_bounce_time = idx
        
        zone.bounce_count = bounce_count
        return bounce_count
    
    def calculate_zone_score(self, zone: Zone, df: pd.DataFrame) -> float:
        """
        Calculate zone score with weighted criteria:
        - Volume (40%)
        - RSI (20%)
        - Bollinger Bands (20%)
        - Time since last break (20%)
        """
        # Volume score
        volume_score = 0
        if zone.volume_on_bounces:
            avg_volume = np.mean(zone.volume_on_bounces)
            volume_score = min(1, avg_volume / df['volume'].mean()) if df['volume'].mean() > 0 else 0
        
        # RSI score (check if RSI was extreme during bounces)
        rsi_score = 0
        if 'rsi' in df.columns and zone.rsi_on_bounces:
            extreme_rsi_count = sum(1 for rsi in zone.rsi_on_bounces if rsi < 30 or rsi > 70)
            rsi_score = extreme_rsi_count / len(zone.rsi_on_bounces) if zone.rsi_on_bounces else 0
        
        # Bollinger Bands score
        bb_score = 0
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_touches = 0
            total_touches = 0
            
            lookback_df = df.tail(self.bounce_lookback)
            tolerance = df['close'].iloc[-1] * 0.001
            
            for idx, row in lookback_df.iterrows():
                high = row['high']
                low = row['low']
                bb_upper = row['bb_upper']
                bb_lower = row['bb_lower']
                
                # Check if price touched the zone and BB bands
                if zone.zone_type == ZoneType.SWING_HIGH:
                    if (high >= zone.price - tolerance and high <= zone.price + tolerance and
                        high >= bb_upper - tolerance):
                        bb_touches += 1
                    if high >= zone.price - tolerance and high <= zone.price + tolerance:
                        total_touches += 1
                
                elif zone.zone_type == ZoneType.SWING_LOW:
                    if (low >= zone.price - tolerance and low <= zone.price + tolerance and
                        low <= bb_lower + tolerance):
                        bb_touches += 1
                    if low >= zone.price - tolerance and low <= zone.price + tolerance:
                        total_touches += 1
            
            bb_score = bb_touches / total_touches if total_touches > 0 else 0
        
        # Time since last break score
        time_score = 1.0
        if zone.last_break_time:
            time_since_break = (df.index[-1] - zone.last_break_time).total_seconds() / 3600
            time_score = max(0, 1 - (time_since_break / 168))  # Decay over 1 week
        
        # Weighted score
        score = (0.4 * volume_score + 
                0.2 * rsi_score + 
                0.2 * bb_score + 
                0.2 * time_score)
        
        return score
    
    def validate_zone(self, zone: Zone, df: pd.DataFrame) -> bool:
        """
        Validate zone based on recent bounces and confluence
        """
        # Check for recent bounce
        recent_bounce = False
        if zone.last_bounce_time:
            time_since_bounce = (df.index[-1] - zone.last_bounce_time).total_seconds() / 900  # M15 periods
            recent_bounce = time_since_bounce <= self.validation_lookback
        
        # Check confluence (at least 2 indicators aligned)
        confluence_count = 0
        
        # Volume confluence
        if zone.volume_on_bounces and np.mean(zone.volume_on_bounces) > df['volume'].mean():
            confluence_count += 1
        
        # RSI confluence
        if 'rsi' in df.columns and zone.rsi_on_bounces:
            extreme_rsi_count = sum(1 for rsi in zone.rsi_on_bounces if rsi < 30 or rsi > 70)
            if extreme_rsi_count > 0:
                confluence_count += 1
        
        # Bollinger Bands confluence
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            lookback_df = df.tail(self.bounce_lookback)
            bb_touches = 0
            total_touches = 0
            
            tolerance = df['close'].iloc[-1] * 0.001
            for idx, row in lookback_df.iterrows():
                high = row['high']
                low = row['low']
                bb_upper = row['bb_upper']
                bb_lower = row['bb_lower']
                
                if zone.zone_type == ZoneType.SWING_HIGH:
                    if high >= zone.price - tolerance and high <= zone.price + tolerance:
                        total_touches += 1
                        if high >= bb_upper - tolerance:
                            bb_touches += 1
                
                elif zone.zone_type == ZoneType.SWING_LOW:
                    if low >= zone.price - tolerance and low <= zone.price + tolerance:
                        total_touches += 1
                        if low <= bb_lower + tolerance:
                            bb_touches += 1
            
            if total_touches > 0 and bb_touches / total_touches > 0.3:  # 30% of touches
                confluence_count += 1
        
        # Zone is valid if recent bounce and sufficient confluence
        is_valid = recent_bounce and confluence_count >= self.confluence_threshold
        
        return is_valid
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect market regime: Range vs Trend
        """
        if len(df) < self.ma50_period:
            return MarketRegime.RANGE
        
        # Calculate MA50
        ma50 = df['close'].rolling(window=self.ma50_period).mean()
        current_ma50 = ma50.iloc[-1]
        prev_ma50 = ma50.iloc[-5] if len(ma50) > 5 else current_ma50
        
        # Calculate ADX
        if 'adx' in df.columns:
            current_adx = df['adx'].iloc[-1]
        else:
            # Simple trend strength calculation
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            current_adx = abs(price_change) * 100  # Simplified ADX
        
        # Determine regime
        ma_direction = abs(current_ma50 - prev_ma50) / prev_ma50
        
        if current_adx > self.adx_threshold and ma_direction > 0.001:  # 0.1% MA movement
            return MarketRegime.TREND
        else:
            return MarketRegime.RANGE
    
    def generate_ml_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate ML features for current price relative to zones
        """
        current_price = df['close'].iloc[-1]
        features = {}
        
        # Find nearest zones
        nearest_zones = self._find_nearest_zones(current_price, max_distance_pct=2.0)
        
        # Proximity features (normalized)
        for i, zone in enumerate(nearest_zones[:3]):  # Top 3 nearest zones
            distance_pct = abs(current_price - zone.price) / current_price
            features[f'zone_{i+1}_proximity'] = distance_pct
            features[f'zone_{i+1}_strength'] = zone.strength
            features[f'zone_{i+1}_score'] = zone.score
            features[f'zone_{i+1}_bounce_count'] = zone.bounce_count
            features[f'zone_{i+1}_type'] = zone.zone_type.value
        
        # Fill missing features with 0
        for i in range(3):
            for feature in ['proximity', 'strength', 'score', 'bounce_count']:
                key = f'zone_{i+1}_{feature}'
                if key not in features:
                    features[key] = 0.0
        
        # Market regime feature
        regime = self.detect_market_regime(df)
        features['market_regime'] = 1.0 if regime == MarketRegime.TREND else 0.0
        
        return features
    
    def _find_nearest_zones(self, price: float, max_distance_pct: float = 2.0) -> List[Zone]:
        """
        Find zones within maximum distance percentage
        """
        max_distance = price * (max_distance_pct / 100)
        nearby_zones = []
        
        for zone in self.zones:
            if zone.is_valid:
                distance = abs(price - zone.price)
                if distance <= max_distance:
                    nearby_zones.append((zone, distance))
        
        # Sort by distance and return zones
        nearby_zones.sort(key=lambda x: x[1])
        return [zone for zone, _ in nearby_zones]
    
    def update_zones(self, df: pd.DataFrame) -> None:
        """
        Update all zones with current market data
        """
        # Detect new zones
        swing_zones = self.detect_swing_points(df)
        psych_zones = self.detect_psychological_levels(df)
        
        # Merge with existing zones (avoid duplicates)
        all_new_zones = swing_zones + psych_zones
        existing_prices = {zone.price for zone in self.zones}
        
        for zone in all_new_zones:
            # Check if zone already exists (within tolerance)
            tolerance = df['close'].iloc[-1] * 0.001
            zone_exists = any(abs(zone.price - existing_price) <= tolerance 
                            for existing_price in existing_prices)
            
            if not zone_exists:
                self.zones.append(zone)
        
        # Update existing zones
        for zone in self.zones:
            # Calculate strength and score
            zone.strength = self.calculate_zone_strength(zone, df)
            zone.score = self.calculate_zone_score(zone, df)
            
            # Validate zone
            zone.is_valid = self.validate_zone(zone, df)
            
            # Check for breaks
            self._check_for_breaks(zone, df)
        
        # Remove old invalid zones (older than 1 month)
        current_time = df.index[-1]
        self.zones = [zone for zone in self.zones 
                     if (current_time - zone.timestamp).days < 30 or zone.is_valid]
    
    def _check_for_breaks(self, zone: Zone, df: pd.DataFrame) -> None:
        """
        Check if zone has been broken recently
        """
        tolerance = df['close'].iloc[-1] * 0.001
        
        # Check last few candles for breaks
        recent_df = df.tail(5)
        
        for i in range(len(recent_df)):
            close = recent_df.iloc[i]['close']
            
            if zone.zone_type == ZoneType.SWING_HIGH:
                if close > zone.price + tolerance:
                    # Check if followed by another candle in same direction
                    if i < len(recent_df) - 1:
                        next_close = recent_df.iloc[i+1]['close']
                        if next_close > close:
                            zone.last_break_time = recent_df.index[i]
                            break
            
            elif zone.zone_type == ZoneType.SWING_LOW:
                if close < zone.price - tolerance:
                    # Check if followed by another candle in same direction
                    if i < len(recent_df) - 1:
                        next_close = recent_df.iloc[i+1]['close']
                        if next_close < close:
                            zone.last_break_time = recent_df.index[i]
                            break
    
    def get_zone_summary(self) -> Dict:
        """
        Get summary of all zones for logging/monitoring
        """
        summary = {
            'total_zones': len(self.zones),
            'valid_zones': len([z for z in self.zones if z.is_valid]),
            'swing_highs': len([z for z in self.zones if z.zone_type == ZoneType.SWING_HIGH]),
            'swing_lows': len([z for z in self.zones if z.zone_type == ZoneType.SWING_LOW]),
            'psychological': len([z for z in self.zones if z.zone_type == ZoneType.PSYCHOLOGICAL]),
            'avg_strength': np.mean([z.strength for z in self.zones]) if self.zones else 0,
            'avg_score': np.mean([z.score for z in self.zones]) if self.zones else 0
        }
        
        return summary 