import torch

from models.vit import get_model



model=get_model()


x=torch.randn(
    1,
    3,
    224,
    224
)



with torch.no_grad():

    y=model(x)


print(y.shape)