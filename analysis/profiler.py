import torch.nn as nn

from models.vit import get_model



model=get_model()



total_params=0


for name,module in model.named_modules():


    if isinstance(module,nn.Linear):

        params=(
            module.in_features*
            module.out_features
        )


        total_params+=params


        print(
            name,
            module.in_features,
            module.out_features,
            params
        )


print("================")

print(
    "Linear params:",
    total_params
)
