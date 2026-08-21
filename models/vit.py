import timm


def get_model():

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True
    )

    model.eval()

    return model