import torch

from fvcore.nn import FlopCountAnalysis

from models.vit import get_model



model=get_model()

model.eval()



x=torch.randn(
    1,
    3,
    224,
    224
)



flops=FlopCountAnalysis(
    model,
    x
)


print(
    "Total FLOPs:",
    flops.total()
)


print(
    flops.by_operator()
)
