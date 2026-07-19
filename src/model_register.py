"""Model registry: enums and factory for donor model loading.

Enums are pure named constants. Loading logic lives in `load_model_internals`,
which returns a `ModelInternals` dataclass — modules and optional transform —
in a single call.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import open_clip
import torch.nn as tnn
import torchvision.models as tvm
import torchvision.transforms.v2 as tt2

__all__ = [
    "ALL_MODELS",
    "KnownModel",
    "ModelInternals",
    "load_model_internals",
    "lookup_model",
]


class _TorchvisionModel(Enum):
    ALEXNET = "alexnet"
    VGG_11 = "vgg11"
    VGG_13 = "vgg13"
    VGG_16 = "vgg16"
    VGG_19 = "vgg19"
    VGG_11_BN = "vgg11_bn"
    VGG_13_BN = "vgg13_bn"
    VGG_16_BN = "vgg16_bn"
    VGG_19_BN = "vgg19_bn"
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"
    SQUEEZENET_1_0 = "squeezenet1_0"
    SQUEEZENET_1_1 = "squeezenet1_1"
    DENSENET_121 = "densenet121"
    DENSENET_161 = "densenet161"
    DENSENET_169 = "densenet169"
    DENSENET_201 = "densenet201"
    INCEPTION_V3 = "inception_v3"
    GOOGLENET = "googlenet"
    SHUFFLENET_V2_0_5 = "shufflenet_v2_x0_5"
    SHUFFLENET_V2_1_0 = "shufflenet_v2_x1_0"
    SHUFFLENET_V2_1_5 = "shufflenet_v2_x1_5"
    SHUFFLENET_V2_2_0 = "shufflenet_v2_x2_0"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_L = "mobilenet_v3_large"
    MOBILENET_V3_S = "mobilenet_v3_small"
    RESNEXT_50 = "resnext50_32x4d"
    RESNEXT_101_32 = "resnext101_32x8d"
    RESNEXT_101_64 = "resnext101_64x4d"
    WIDERESNET_50_2 = "wide_resnet50_2"
    WIDERESNET_101_2 = "wide_resnet101_2"
    MNASNET_0_5 = "mnasnet0_5"
    MNASNET_0_75 = "mnasnet0_75"
    MNASNET_1_0 = "mnasnet1_0"
    MNASNET_1_3 = "mnasnet1_3"
    VIT_B_16 = "vit_b_16"
    VIT_B_32 = "vit_b_32"
    VIT_L_16 = "vit_l_16"
    VIT_L_32 = "vit_l_32"
    VIT_H_14 = "vit_h_14"
    SWIN_T = "swin_t"
    SWIN_S = "swin_s"
    SWIN_B = "swin_b"
    SWIN_V2_T = "swin_v2_t"
    SWIN_V2_S = "swin_v2_s"
    SWIN_V2_B = "swin_v2_b"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    CONVNEXT_BASE = "convnext_base"
    CONVNEXT_LARGE = "convnext_large"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"
    EFFICIENTNET_B4 = "efficientnet_b4"
    EFFICIENTNET_B5 = "efficientnet_b5"
    EFFICIENTNET_B6 = "efficientnet_b6"
    EFFICIENTNET_B7 = "efficientnet_b7"


_MODEL_TRANSFORMATIONS = {
    "ALEXNET": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_11": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_13": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_16": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_19": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_11_BN": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_13_BN": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_16_BN": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VGG_19_BN": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNET_18": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNET_34": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNET_50": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNET_101": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNET_152": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SQUEEZENET_1_0": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SQUEEZENET_1_1": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "DENSENET_121": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "DENSENET_161": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "DENSENET_169": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "DENSENET_201": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "INCEPTION_V3": tt2.Compose(
        [
            tt2.Resize((342, 342)),
            tt2.CenterCrop(299),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "GOOGLENET": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SHUFFLENET_V2_0_5": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SHUFFLENET_V2_1_0": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SHUFFLENET_V2_1_5": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SHUFFLENET_V2_2_0": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "MOBILENET_V2": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "MOBILENET_V3_L": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "MOBILENET_V3_S": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNEXT_50": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNEXT_101_32": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "RESNEXT_101_64": tt2.Compose(
        [
            tt2.Resize((232, 232)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "WIDERESNET_50_2": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "WIDERESNET_101_2": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "MNASNET_0_5": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "MNASNET_0_75": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "MNASNET_1_0": tt2.Compose(
        [
            tt2.Resize((256, 256)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "MNASNET_1_3": tt2.Compose(
        [
            tt2.Resize((232, 232)),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VIT_B_16": tt2.Compose(
        [
            tt2.Resize(256),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VIT_B_32": tt2.Compose(
        [
            tt2.Resize(256),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VIT_L_16": tt2.Compose(
        [
            tt2.Resize(242),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VIT_L_32": tt2.Compose(
        [
            tt2.Resize(256),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "VIT_H_14": tt2.Compose(
        [
            tt2.Resize(518, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(518),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SWIN_T": tt2.Compose(
        [
            tt2.Resize(232, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SWIN_S": tt2.Compose(
        [
            tt2.Resize(246, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SWIN_B": tt2.Compose(
        [
            tt2.Resize(238, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SWIN_V2_T": tt2.Compose(
        [
            tt2.Resize(260, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(256),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SWIN_V2_S": tt2.Compose(
        [
            tt2.Resize(260, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(256),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "SWIN_V2_B": tt2.Compose(
        [
            tt2.Resize(272, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(256),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "CONVNEXT_TINY": tt2.Compose(
        [
            tt2.Resize(236),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "CONVNEXT_SMALL": tt2.Compose(
        [
            tt2.Resize(230),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "CONVNEXT_BASE": tt2.Compose(
        [
            tt2.Resize(232),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "CONVNEXT_LARGE": tt2.Compose(
        [
            tt2.Resize(232),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B0": tt2.Compose(
        [
            tt2.Resize(256, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(224),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B1": tt2.Compose(
        [
            tt2.Resize(256, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(240),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B2": tt2.Compose(
        [
            tt2.Resize(288, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(288),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B3": tt2.Compose(
        [
            tt2.Resize(320, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(300),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B4": tt2.Compose(
        [
            tt2.Resize(384, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(380),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B5": tt2.Compose(
        [
            tt2.Resize(456, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(456),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B6": tt2.Compose(
        [
            tt2.Resize(528, tt2.InterpolationMode.BICUBIC),
            tt2.CenterCrop(528),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "EFFICIENTNET_B7": tt2.Compose(
        [
            tt2.Resize(600),
            tt2.CenterCrop(600),
            tt2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
}


class _OpenClipSpec(NamedTuple):
    model_name: str
    pretrained: str


class _OpenClipModel(Enum):
    MOBILECLIP_S1 = _OpenClipSpec("MobileCLIP-S1", "datacompdr")
    MOBILECLIP_B = _OpenClipSpec("MobileCLIP-B", "datacompdr")


KnownModel = _TorchvisionModel | _OpenClipModel


@dataclass(frozen=True)
class ModelInternals:
    modules: list[tnn.Module]
    transform: tt2.Compose
    class_token: tnn.Parameter | None = None


def load_model_internals(model: KnownModel) -> ModelInternals:
    match model:
        case _TorchvisionModel():
            name = model.value
            weights = tvm.get_model_weights(name)["DEFAULT"]
            donor = getattr(tvm, name)(weights=weights)
            return ModelInternals(
                modules=list(donor.children()),
                transform=_MODEL_TRANSFORMATIONS[model.name.upper()],
                class_token=donor.class_token if name.startswith("vit_") else None,
            )
        case _OpenClipModel():
            spec = model.value
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                spec.model_name, pretrained=spec.pretrained
            )
            return ModelInternals(
                modules=list(clip_model.visual.children())[:1],
                transform=preprocess,
            )
        case _:
            raise NotImplementedError(f"Unkown model: {model!r}")


ALL_MODELS: dict[str, KnownModel] = {
    model.name: model for cls in (_TorchvisionModel, _OpenClipModel) for model in cls
}


def lookup_model(name: str) -> KnownModel:
    key = name.upper()
    if key not in ALL_MODELS:
        valid = ", ".join(sorted(ALL_MODELS))
        raise ValueError(f"Unknown model {name!r}. Valid names: {valid}")
    return ALL_MODELS[key]
