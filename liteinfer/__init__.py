"""liteinfer: pure-Python reference kernels for efficient LLM inference.

Three self-contained building blocks for studying and prototyping efficient
inference, all runnable offline on plain NumPy:

* ``liteinfer.quant`` — post-training INT8 / INT4 quantization
* ``liteinfer.kvcache`` — paged KV-cache and paged attention
* ``liteinfer.specdec`` — speculative decoding with the exact-sampling correction
"""

from liteinfer._version import __version__
from liteinfer.config import QuantConfig, SpecDecodeConfig
from liteinfer.kvcache.attention import paged_attention
from liteinfer.kvcache.blocks import BlockAllocator
from liteinfer.kvcache.manager import BlockSpaceManager
from liteinfer.kvcache.paged_cache import PagedKVStore
from liteinfer.quant.int4 import dequantize_group, quantize_group
from liteinfer.quant.int8 import dequantize, quantize_symmetric
from liteinfer.quant.linear import QuantLinear
from liteinfer.specdec.speculative import SpecDecodeResult, speculative_decode

__all__ = [
    "__version__",
    "QuantConfig",
    "SpecDecodeConfig",
    "QuantLinear",
    "quantize_symmetric",
    "dequantize",
    "quantize_group",
    "dequantize_group",
    "PagedKVStore",
    "BlockAllocator",
    "BlockSpaceManager",
    "paged_attention",
    "speculative_decode",
    "SpecDecodeResult",
]
