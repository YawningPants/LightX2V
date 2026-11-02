from types import SimpleNamespace

import pytest
import torch

from lightx2v_platform.ops.attn.ascend_npu import npu_quant_flash_attn as module
from lightx2v_platform.ops.attn.ascend_npu.npu_quant_flash_attn import (
    NpuQuantFlashAttnWeight,
    get_quant_attention_rotation,
)


class _Runtime:
    """Records the MindIE-SD call and returns a correctly shaped result."""

    def __init__(self):
        self.calls = []
        self.failure = None

    def __call__(self, q, k, v, **kwargs):
        self.calls.append({"q": q, "k": k, "v": v, "kwargs": kwargs})
        if self.failure is not None:
            raise self.failure
        return torch.zeros_like(q)


@pytest.fixture(autouse=True)
def fake_npu(monkeypatch):
    monkeypatch.setattr(module, "torch_npu", SimpleNamespace())
    runtime = _Runtime()
    monkeypatch.setattr(module, "_load_quant_attention", lambda: runtime)
    get_quant_attention_rotation.cache_clear()
    yield runtime
    get_quant_attention_rotation.cache_clear()


def attention(**config):
    weight = NpuQuantFlashAttnWeight()
    if config:
        weight.set_config(config)
    return weight


def qkv(sequence=8, heads=2, head_dim=64, dtype=torch.bfloat16):
    q = torch.randn(sequence, heads, head_dim, dtype=dtype)
    return q, torch.randn_like(q), torch.randn_like(q)


def test_three_dimensional_input_is_promoted_to_bsnd(fake_npu):
    q, k, v = qkv()
    weight = attention()

    output = weight.apply(q, k, v)

    call = fake_npu.calls[0]
    assert call["q"].shape == (1, 8, 2, 64)
    assert call["k"].shape == (1, 8, 2, 64)
    assert call["kwargs"]["layout"] == "BSND"
    assert call["kwargs"]["precision"] == "fp8"
    assert output.shape == (8, 128)


def test_default_scale_matches_head_dim(fake_npu):
    q, k, v = qkv(head_dim=64)
    attention().apply(q, k, v)

    assert fake_npu.calls[0]["kwargs"]["scale"] == pytest.approx(64**-0.5)


def test_explicit_softmax_scale_is_forwarded(fake_npu):
    q, k, v = qkv()
    attention().apply(q, k, v, softmax_scale=0.37)

    assert fake_npu.calls[0]["kwargs"]["scale"] == pytest.approx(0.37)


@pytest.mark.parametrize("precision", ["fp8", "mxfp8", "mxfp4"])
def test_precision_from_config_is_forwarded(fake_npu, precision):
    q, k, v = qkv()
    attention(npu_attn_quant_precision=precision).apply(q, k, v)

    assert fake_npu.calls[0]["kwargs"]["precision"] == precision


def test_unsupported_precision_is_rejected():
    with pytest.raises(ValueError, match="npu_attn_quant_precision"):
        attention(npu_attn_quant_precision="int8")


def test_rotation_is_deterministic_and_power_of_two_only():
    first = get_quant_attention_rotation(torch.device("cpu"), torch.bfloat16, 64)
    second = get_quant_attention_rotation(torch.device("cpu"), torch.bfloat16, 64)

    assert first is second
    assert first.shape == (64, 64)
    # Orthogonality is the property the rotation relies on.
    identity = first.to(torch.float32) @ first.to(torch.float32).t()
    torch.testing.assert_close(identity, torch.eye(64), atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="power-of-two"):
        get_quant_attention_rotation(torch.device("cpu"), torch.bfloat16, 48)


def test_rotation_does_not_consume_global_rng():
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    get_quant_attention_rotation(torch.device("cpu"), torch.bfloat16, 32)
    actual = torch.randn(4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rotation_is_passed_to_both_q_and_k(fake_npu):
    q, k, v = qkv()
    attention().apply(q, k, v)

    kwargs = fake_npu.calls[0]["kwargs"]
    assert kwargs["q_rot"].shape == (64, 64)
    assert torch.equal(kwargs["q_rot"], kwargs["k_rot"])


def test_rotation_can_be_disabled(fake_npu):
    q, k, v = qkv()
    attention(npu_attn_quant_rotation=False).apply(q, k, v)

    kwargs = fake_npu.calls[0]["kwargs"]
    assert "q_rot" not in kwargs and "k_rot" not in kwargs


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"attn_mask": torch.zeros(1, 1)}, "attn_mask"),
        ({"causal": True}, "causal"),
        ({"cu_seqlens_q": torch.tensor([0, 8])}, "cu_seqlens"),
        ({"dropout": 0.1}, "dropout"),
    ],
)
def test_unsupported_call_shapes_are_rejected(fake_npu, overrides, match):
    q, k, v = qkv()
    with pytest.raises(ValueError, match=match):
        attention().apply(q, k, v, **overrides)
    assert fake_npu.calls == []


def test_batch_larger_than_one_is_rejected(fake_npu):
    q = torch.randn(2, 8, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="batch size 1"):
        attention().apply(q, q, q)


def test_non_power_of_two_head_dim_is_rejected(fake_npu):
    q, k, v = qkv(head_dim=48)
    with pytest.raises(ValueError, match="power-of-two"):
        attention().apply(q, k, v)


def test_runtime_errors_propagate_without_retry(fake_npu):
    q, k, v = qkv()
    fake_npu.failure = RuntimeError("operator rejected the input")

    with pytest.raises(RuntimeError, match="operator rejected"):
        attention().apply(q, k, v)
    assert len(fake_npu.calls) == 1
