import torch.nn as nn

from models.vit import get_model

from runtime.operator import LinearOperator

model=get_model()


operators=[]


for name,module in model.named_modules():


    if isinstance(module,nn.Linear):


        op=LinearOperator(

            name,

            module.in_features,

            module.out_features

        )


        operators.append(op)


for op in operators:

    print(op)


print(
    "Total:",
    len(operators)
)