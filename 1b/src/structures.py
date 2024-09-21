from dataclasses import dataclass
from typing import Optional

@dataclass
class Inform:
    """Class for keeping track of an item in inventory."""
    type: Optional[str] = None
    pricerange: Optional[str] = None
    area: Optional[str] = None
    food: Optional[str] = None
