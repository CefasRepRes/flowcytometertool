# functions file written whilst writing flow_cytometer_tool.py
"""
Compatibility facade for the historical flowcytometer_tool.misc.functions module.

The implementation has been split across focused modules, but legacy code still
imports from flowcytometer_tool.misc.functions. Keep this file intentionally
small: it imports and re-exports the public API from the split modules.
"""

from flowcytometer_tool.misc.functions_model_selection import *
from flowcytometer_tool.misc.functions_training import *
from flowcytometer_tool.misc.functions_runtime import *
from flowcytometer_tool.misc.functions_visualisation import *
from flowcytometer_tool.misc.functions_blob_and_utils import *

from flowcytometer_tool.misc.functions_model_selection import __all__ as _ms_all
from flowcytometer_tool.misc.functions_training import __all__ as _tr_all
from flowcytometer_tool.misc.functions_runtime import __all__ as _rt_all
from flowcytometer_tool.misc.functions_visualisation import __all__ as _vis_all
from flowcytometer_tool.misc.functions_blob_and_utils import __all__ as _blob_all

__all__ = list(dict.fromkeys(
    list(_ms_all)
    + list(_tr_all)
    + list(_rt_all)
    + list(_vis_all)
    + list(_blob_all)
))
