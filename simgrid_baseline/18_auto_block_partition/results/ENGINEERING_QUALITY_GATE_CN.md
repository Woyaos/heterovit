# 工程质量门禁结果

- 总状态：`pass_with_hardware_warnings`
- 是否可以进入硬件标定阶段：`True`
- 是否可以声称真实硬件加速：`False`

## 结论

当前项目的软件建模、FFN 结构证据、SimGrid 事件回放和硬件测量计划已经可以作为上板前工程依据。但现阶段仍不能声称 RF880-AGX 板卡上已经获得端到端加速，因为关键算子、PCIe/DMA 路径和整模型结果还没有实测。

## 检查项

- `pass` evidence_is_not_labeled_as_hardware_prediction
  证据：structural evidence valid_for_target_prediction=False
  后续动作：Keep sensitivity and target-measured results in separate tables.
- `pass` ffn_regions_are_dependency_certified
  证据：found 12 FFN regions; rejected=0
  后续动作：Fix DAG extraction before using FFN-level conclusions.
- `pass` ffn_fusion_has_structural_transfer_reason
  证据：all FFN regions reduce split FFN boundary bytes by 80%
  后续动作：Recompute the derivation from actual edge sizes.
- `pass` split_linear_offload_is_rejected_by_cost_model
  证据：GPU-only 24.749 ms; split FFN linears 30.411 ms
  后续动作：Do not offload singleton FFN Linears unless measured costs change.
- `pass` full_ffn_is_beneficial_only_under_declared_assumption
  证据：full FFN 17.850 ms; no full-FFN kernel 24.749 ms
  后续动作：Verify the FPGA full-FFN/GELU kernel before claiming speedup.
- `pass` simgrid_replay_matches_expected_task_and_transfer_counts
  证据：split_ffn_linears: tasks=125, transfers=96; fixed_ffn: tasks=101, transfers=48; linear_only_auto: tasks=125, transfers=0
  后续动作：Regenerate event replay before using timeline or latency numbers.
- `pass` measurement_plan_covers_fused_and_original_dags
  证据：tasks=137, fpga_rows=36, activation_bytes=2
  后续动作：Regenerate the union measurement plan for the exact model precision.
- `pass` measurement_plan_includes_key_ffn_payload_sizes
  证据：302592 B boundary activations and 1210368 B expanded FFN activations are planned
  后续动作：Add missing FFN payload sizes to transfer calibration.
- `warn` heldout_end_to_end_validation_is_planned_but_not_completed
  证据：measurement plan status=waiting_for_target_hardware; heldout rows=4
  后续动作：Collect held-out board measurements before target-performance claims.

## 不能提前下结论的原因

- FPGA full-FFN/GELU kernel has not been measured on RF880.
- Jetson GPU-only baseline has not been measured under the final runtime and precision.
- PCIe/DMA path, fixed latency, bandwidth, and overlap have not been calibrated on the board.
- Held-out end-to-end hardware validation has not been collected.
