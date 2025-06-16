import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel

def make_cnn_model(
    in_channels: int = 3,
    out_classes: int = 1,
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet"
):
  
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_classes,
        decoder_channels=[256, 128, 64, 32, 16]
    )
    return model


def make_transformer_model(
    pretrained_backbone: str = "nvidia/mit-b2",
    in_channels: int = 3,
    out_classes: int = 1
):
   
    backbone = SegformerModel.from_pretrained(pretrained_backbone)

    config = SegformerConfig.from_pretrained(
        pretrained_backbone,
        num_labels=out_classes,
        id2label={i: f"class_{i}" for i in range(out_classes)},
        label2id={f"class_{i}": i for i in range(out_classes)}
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_backbone,
        config=config,
        ignore_mismatched_sizes=True
    )
    return model
