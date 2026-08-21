import torch
import timm

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False
)

model.eval()

dummy = torch.randn(1,3,224,224)

torch.onnx.export(
    model,
    dummy,
    "vit.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0:"batch"},
        "output": {0:"batch"}
    }
)