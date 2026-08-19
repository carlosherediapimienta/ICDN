"""ICDN: demand and price elasticities from retail panels.

Typical use::

    from icdn import ICDNModel, ICDNConfig, PanelSchema

    model = ICDNModel(ICDNConfig(n_products=5))
    model.fit(panel)
    elasticities = model.elasticities()
"""

from .api import ICDNModel
from .config import ICDNConfig, PanelSchema

__version__ = "0.1.1"
__all__ = ["ICDNConfig", "ICDNModel", "PanelSchema", "__version__"]
