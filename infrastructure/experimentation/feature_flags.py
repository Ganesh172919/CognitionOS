"""Feature Flag System"""
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class FeatureFlag:
    flag_key: str
    name: str
    enabled: bool

class FeatureFlagSystem:
    """Feature flag system"""
    def __init__(self, session):
        self.session = session
        self.flags = {}
    
    async def is_enabled(self, flag_key: str, user_id: str) -> bool:
        flag = self.flags.get(flag_key)
        return flag.enabled if flag else False
