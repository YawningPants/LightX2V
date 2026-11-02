import functools
import math

import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


SUPPORTED_PRECISIONS = ("fp8", "mxfp8", "mxfp4")
# Keep the migrated rotation identical to the reference implementation so a
# checkpoint evaluated there keeps the same attention numerics here.
ROTATION_SEED = 425500


def _load_quant_attention():
    try:
        from mindiesd import quant_attention
    except ImportError as exc:
        raise ImportError(
            "npu_quant_flash_attn requires a MindIE-SD build that exports quant_attention. "
            "Install a compatible MindIE-SD, or select attn_type='npu_flash_attn' for the unquantized path. "
            "Automatic precision fallback is not performed."
        ) from exc

    return quant_attention


@functools.lru_cache(maxsize=8)
def get_quant_attention_rotation(device, dtype, head_dim):
    """Build the fixed sign-flipped Walsh-Hadamard rotation for one head size.

    The rotation is applied to Q and K before block quantization.  It is drawn
    from a local generator so it never disturbs the global RNG stream, and it
    is cached per (device, dtype, head_dim) because every block of every
    denoising step reuses the same matrix.
    """
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("Quantized attention rotations require a power-of-two head dimension.")
    generator = torch.Generator(device="cpu").manual_seed(ROTATION_SEED)
    signs = torch.randint(0, 2, (head_dim,), generator=generator, dtype=torch.int64)
    signs = signs.to(dtype=torch.float32).mul_(2).sub_(1)
    hadamard = torch.ones(1, 1)
    while hadamard.shape[0] < head_dim:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), dim=1), torch.cat((hadamard, -hadamard), dim=1)), dim=0)
    rotation = signs[:, None] * hadamard / math.sqrt(head_dim)
    return rotation.to(device=device, dtype=dtype).contiguous()


@PLATFORM_ATTN_WEIGHT_REGISTER("npu_quant_flash_attn")
class NpuQuantFlashAttnWeight(AttnWeightTemplate):
    """Dense self-attention with quantized Q/K/V compute on Ascend NPU.

    Q/K/V stay floating point in the model; MindIE-SD ``quant_attention``
    quantizes them inside the attention call and returns floating-point output.
    Nothing is cached in a quantized form, so this only changes the precision of
    the attention computation.

    Masked, causal, varlen and packed-sequence calls are rejected instead of
    silently falling back: a fallback would make a misconfiguration look like a
    quality difference.  Use ``npu_flash_attn`` or ``torch_sdpa`` for those.
    """

    def __init__(self):
        self.config = {}
        self.precision = SUPPORTED_PRECISIONS[0]
        self.use_rotation = True
        assert torch_npu is not None, "torch_npu is not installed."

    def set_config(self, config=None):
        if config is None:
            return
        self.config = config
        precision = config.get("npu_attn_quant_precision", self.precision)
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"npu_attn_quant_precision must be one of {SUPPORTED_PRECISIONS}, but got {precision!r}.")
        self.precision = precision
        self.use_rotation = bool(config.get("npu_attn_quant_rotation", True))

    def _validate(self, q, k, v, **kwds):
        if q.ndim not in (3, 4):
            raise ValueError(f"npu_quant_flash_attn expects 3D or 4D q/k/v tensors, but q is {q.ndim}D.")
        if k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError(f"q, k, and v must have the same rank, but got {q.ndim}D, {k.ndim}D, and {v.ndim}D.")
        if kwds.get("attn_mask", None) is not None:
            raise ValueError("npu_quant_flash_attn does not support attn_mask; keep masked attention on torch_sdpa.")
        if kwds.get("causal", False):
            raise ValueError("npu_quant_flash_attn does not support causal attention.")
        if kwds.get("cu_seqlens_q", None) is not None or kwds.get("cu_seqlens_kv", None) is not None:
            raise ValueError("npu_quant_flash_attn requires dense inputs without cu_seqlens metadata.")
        dropout = kwds.get("dropout", kwds.get("drop_rate", None))
        if dropout not in (None, 0.0):
            raise ValueError("npu_quant_flash_attn does not support dropout.")
        head_dim = q.shape[-1]
        if self.use_rotation and (head_dim <= 0 or head_dim & (head_dim - 1)):
            raise ValueError(f"Generated attention rotations require a power-of-two head dimension, but got {head_dim}.")

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwds):
        self._validate(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, **kwds)

        batch = 1 if q.ndim == 3 else q.shape[0]
        if batch != 1:
            raise ValueError(f"npu_quant_flash_attn requires batch size 1, but got {batch}.")

        # MindIE-SD takes four-dimensional BSND inputs only; the model feeds
        # token-major (S, H, D) tensors on both the prefill and denoise paths.
        q4 = q.unsqueeze(0) if q.ndim == 3 else q
        k4 = k.unsqueeze(0) if k.ndim == 3 else k
        v4 = v.unsqueeze(0) if v.ndim == 3 else v
        token_count = q4.shape[1]

        scale = kwds.get("softmax_scale", None)
        if scale is None:
            scale = 1.0 / math.sqrt(q4.shape[-1])

        runtime = _load_quant_attention()
        kwargs = {"precision": self.precision, "layout": "BSND", "scale": scale}
        if self.use_rotation:
            rotation = get_quant_attention_rotation(q4.device, q4.dtype, q4.shape[-1])
            kwargs["q_rot"] = rotation
            kwargs["k_rot"] = rotation

        output = runtime(q4, k4, v4, **kwargs)
        return output.reshape(token_count, -1)
