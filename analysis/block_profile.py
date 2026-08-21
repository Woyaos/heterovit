from models.vit import get_model



model=get_model()



blocks=[]


for i,block in enumerate(model.blocks):


    print(
        "Block:",
        i
    )


    print(block)
