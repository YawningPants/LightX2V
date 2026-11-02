# Qwen-Image-2.1 W8A8F8 验证指南

## 1. 接手信息

- 仓库：`LightX2V`
- 分支：`feat/qwen21-npu-mxfp8-w8a8`
- 代码提交：`1d51dc63`
- 目标：在 Ascend NPU 上验证 Qwen-Image-2.1 DiT 的在线 MXFP8 W8A8；当前尚未进行 NPU 运行验证。
- 实现：`lightx2v/common/ops/mm/mm_weight.py` 中的 `MMWeightWmxfp8Amxfp8dynamicNpu`，注册名为 `mxfp8-npu`。
- runner：`lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py`。

量化契约：权重和激活为 E4M3，scale 为 E8M0，K 维 group size 为 32；权重布局为 `(K, N)`，权重 scale 为 `(S/2, N, 2)`；GEMM 使用 `npu_dynamic_mx_quant` 和 `npu_quant_matmul(group_sizes=[1, 1, 32])`，输出保持输入 BF16/FP16 dtype。

仅量化每个 transformer block 的 Q/K/V/out 和 `img_mlp` 三个 Linear；32 层模型应有 224 个量化 Linear。pre/post、text encoder、VAE、scheduler 和 Attention 不量化。当前仅支持单卡常驻，不支持 LoRA、TP/SP、CPU offload、离线量化 checkpoint 或量化 Attention 组合。

## 2. Smoke 配置

将以下内容保存为 `mxfp8_npu_smoke.json`。使用同一个 BF16 Qwen-Image-2.1 checkpoint，不设置 `dit_quantized_ckpt` 或 `weight_auto_quant`，不配置 LoRA。

```json
{
    "infer_steps": 2,
    "resolution": 1024,
    "size": [1024, 1024],
    "attn_type": "npu_flash_attn",
    "rope_type": "torch_real_rope",
    "rms_norm_type": "fp32_variance",
    "layer_norm_type": "torch",
    "modulate_type": "torch",
    "fused_qk_rms_norm": false,
    "fused_block_ops": false,
    "enable_cfg": false,
    "sample_guide_scale": 1.0,
    "warmup": false,
    "cpu_offload": false,
    "tensor_parallel": false,
    "seq_parallel": false,
    "shared_cpu_weights": false,
    "use_compile": false,
    "feature_caching": "NoCaching",
    "dit_quantized": true,
    "dit_quant_scheme": "mxfp8-npu"
}
```

BF16 对照组只改为：

```json
"dit_quantized": false,
"dit_quant_scheme": "Default"
```

## 3. 执行顺序

### A. 单层数值检查

直接实例化 `MM_WEIGHT_REGISTER["mxfp8-npu"]`，用 BF16 `(N, K)` 权重检查：

1. 加载后权重/scale 的 dtype、形状和 contiguous 布局符合上述契约；K=96 时验证奇数组 scale 补齐。
2. 二维和三维输入的输出形状正确，输出 dtype 与输入一致，结果无 NaN/Inf。
3. 同一输入与参考 MXFP8 计算进行数值对比；forward 不再次量化权重。
4. 非 BF16/FP16 输入、错误 K 维和非 NPU 环境应明确报错，不得静默回退 BF16。

### B. 2-step T2I/I2I smoke

在仓库根目录、已配置好的 NPU 环境中执行。先设置 `DTYPE=BF16` 和 `SENSITIVE_LAYER_DTYPE=None`，并替换模型路径。

```bash
python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task t2i \
    --model_path /path/to/Qwen-Image-2.1 \
    --config_json /path/to/mxfp8_npu_smoke.json \
    --prompt "A capybara wearing a wizard hat, oil painting" \
    --size 1024 1024 \
    --seed 142 \
    --save_result_path save_results/mxfp8_npu_t2i.png
```

```bash
python -m lightx2v.infer \
    --model_cls qwen_image_21 \
    --task i2i \
    --model_path /path/to/Qwen-Image-2.1 \
    --config_json /path/to/mxfp8_npu_smoke.json \
    --image_path assets/inputs/imgs/img_0.jpg \
    --prompt "Transform the cat into a photorealistic dog while preserving the composition." \
    --size 1024 1024 \
    --seed 42 \
    --save_result_path save_results/mxfp8_npu_i2i.png
```

通过标准：启动和请求完成、图像可解码、无 NaN/Inf、无静默 BF16 回退。随后用相同输入运行 BF16 对照组并保存输出。

### C. Kernel 与覆盖检查

单独运行短步 profiler，不与正式性能数据混用。确认：

- 加载阶段权重量化只执行一次；默认 32 层时覆盖 224 个目标 Linear。
- trace 命中 `npu_dynamic_mx_quant` 和 MXFP8 `npu_quant_matmul`。
- pre/post 和其他组件未进入该量化路径，Attention 未使用量化后端。

## 4. 后续实验与归档

Smoke 和 trace 通过后，再进行 50-step BF16/MXFP8 对比：每任务 warmup 1 次、正式至少 5 次，T2I seed=142、I2I seed=42，profiler 与性能运行分离。记录 complete DiT、denoise、prefill、峰值显存和输出质量；不要只比较原生 step 时间。

至少保存：实际命令与配置、分支/commit、NPU 及 torch/torch_npu/CANN 版本、日志、输出图、trace、逐次测量和错误摘要。若失败，保留完整配置与报错，不改用 Wan2.2、普通 FP8 或量化 Attention 替代。
