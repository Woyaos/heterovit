from models.vit import get_model


from analysis.build_blocks import build_blocks


from runtime.executor.vit_executor import ViTExecutor


from runtime.cost_model.gpu import GPUModel
from runtime.cost_model.fpga import FPGAModel
from runtime.cost_model.communication import CommunicationModel



# ==========================
# build model
# ==========================

model = get_model()


blocks = build_blocks(model)



# ==========================
# cost model
# ==========================

gpu = GPUModel()

fpga = FPGAModel()

comm = CommunicationModel(
    bandwidth=16e9
)



# ==========================
# executor
# ==========================

executor = ViTExecutor(
    blocks,
    gpu,
    fpga,
    comm
)



# ==========================
# Test
# ==========================


gpu_only = executor.run_gpu_only()


fpga_offload = executor.run_block_offload()



print("================")

print(
    "GPU only:",
    gpu_only
)


print(
    "FFN FPGA offload:",
    fpga_offload
)