from models.vit import get_model
from runtime.operator import LinearOperator



model=get_model()


linear_ops=[]



for name,module in model.named_modules():


    if module.__class__.__name__=="Linear":


        op=LinearOperator(
            name,
            module.in_features,
            module.out_features
        )


        linear_ops.append(op)



if __name__=="__main__":

    for op in linear_ops:

        print(
            op.name,
            op.mac
        )