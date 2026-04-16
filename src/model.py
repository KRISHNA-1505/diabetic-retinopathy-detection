import timm

def get_swin_model(num_classes=5):

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes
    )

    return model
