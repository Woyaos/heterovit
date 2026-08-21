from models.vit import get_model


from analysis.build_blocks import build_blocks


from runtime.executor.vit_executor import ViTExecutor


from runtime.cost_model.gpu import GPUModel
from runtime.cost_model.fpga import FPGAModel
from runtime.cost_model.communication import CommunicationModel



model=get_model()



blocks=build_blocks(model)



executor=ViTExecutor(
    blocks,
    GPUModel(),
    FPGAModel(),
    CommunicationModel()
)



latency = executor.run_gpu_only()


print("================")
print(
    "ViT GPU only:",
    latency
)