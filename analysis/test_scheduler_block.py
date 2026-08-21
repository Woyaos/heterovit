from analysis.build_blocks import blocks
from runtime.cost_model.block_cost import BlockCostModel
from runtime.scheduler.block_scheduler import BlockScheduler



cost=BlockCostModel()

scheduler=BlockScheduler(cost)



for b in blocks:


    gpu_time=scheduler.gpu_only(b)


    fpga_time=scheduler.ffn_fpga(b)


    print("================")

    print(b.name)

    print(
        "GPU only:",
        gpu_time
    )

    print(
        "FFN FPGA:",
        fpga_time
    )