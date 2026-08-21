from models.vit import get_model
import torch.nn as nn

from runtime.operator import LinearOperator
from runtime.scheduler import Scheduler



model=get_model()


operators=[]


for name,module in model.named_modules():

    if isinstance(module,nn.Linear):

        operators.append(

            LinearOperator(
                name,
                module.in_features,
                module.out_features
            )

        )


scheduler=Scheduler()



for op in operators:

    scheduler.decide(op)


for op in operators:

    print(op)