# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DCU_WHITELIST = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "PP-LCNet_x1_0",
    "PP-HGNetV2-B0_ML",
    "PP-HGNetV2-B4_ML",
    "PP-HGNetV2-B6_ML",
    "CLIP_vit_base_patch16_224_ML",
    "PP-ShiTuV2_rec_CLIP_vit_base",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "RT-DETR-R18",
    "PicoDet-L",
    "PicoDet-M",
    "PicoDet-S",
    "PicoDet-XS",
    "FCOS-ResNet50",
    "YOLOX-N",
    "FasterRCNN-ResNet34-FPN",
    "YOLOv3-DarkNet53",
    "Cascade-FasterRCNN-ResNet50-FPN",
    "PP-YOLOE_plus_SOD-S",
    "PP-YOLOE_plus_SOD-L",
    "PP-YOLOE_plus_SOD-largesize-L",
    "STFPM",
    "Deeplabv3_Plus-R50",
    "Deeplabv3_Plus-R101",
    "PP-LiteSeg-T",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "DLinear",
    "RLinear",
    "NLinear",
    "PicoDet_LCNet_x2_5_face",
]

MLU_WHITELIST = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet18_vd",
    "ResNet34_vd",
    "ResNet50_vd",
    "ResNet101_vd",
    "ResNet152_vd",
    "ResNet200_vd",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_0",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_5",
    "PP-LCNet_x2_0",
    "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5",
    "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0",
    "MobileNetV3_large_x1_25",
    "MobileNetV3_small_x0_35",
    "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75",
    "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25",
    "PP-HGNet_small",
    "PP-HGNet_tiny",
    "PP-HGNet_base",
    "PP-ShiTuV2_rec_CLIP_vit_base",
    "PP-ShiTuV2_rec_CLIP_vit_large",
    "PP-YOLOE_plus-X",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "PicoDet-L",
    "PicoDet-M",
    "PicoDet-S",
    "PicoDet-XS",
    "STFPM",
    "PP-LiteSeg-T",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "PicoDet_layout_1x",
    "DLinear",
    "RLinear",
    "NLinear",
    "PicoDet_LCNet_x2_5_face",
]

NPU_WHITELIST = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet18_vd",
    "ResNet34_vd",
    "ResNet50_vd",
    "ResNet101_vd",
    "ResNet152_vd",
    "ResNet200_vd",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_0",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_5",
    "PP-LCNet_x2_0",
    "PP-LCNetV2_small",
    "PP-LCNetV2_base",
    "PP-LCNetV2_large",
    "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5",
    "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0",
    "MobileNetV3_large_x1_25",
    "MobileNetV3_small_x0_35",
    "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75",
    "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25",
    "ConvNeXt_tiny",
    "ConvNeXt_small",
    "ConvNeXt_base_224",
    "ConvNeXt_base_384",
    "ConvNeXt_large_224",
    "ConvNeXt_large_384",
    "MobileNetV1_x0_25",
    "MobileNetV1_x0_5",
    "MobileNetV1_x0_75",
    "MobileNetV1_x1_0",
    "MobileNetV2_x0_25",
    "MobileNetV2_x0_5",
    "MobileNetV2_x1_0",
    "MobileNetV2_x1_5",
    "MobileNetV2_x2_0",
    "SwinTransformer_tiny_patch4_window7_224",
    "SwinTransformer_small_patch4_window7_224",
    "SwinTransformer_base_patch4_window7_224",
    "SwinTransformer_base_patch4_window12_384",
    "SwinTransformer_large_patch4_window7_224",
    "SwinTransformer_large_patch4_window12_384",
    "PP-HGNet_small",
    "PP-HGNet_tiny",
    "PP-HGNet_base",
    "PP-HGNetV2-B0",
    "PP-HGNetV2-B1",
    "PP-HGNetV2-B2",
    "PP-HGNetV2-B3",
    "PP-HGNetV2-B4",
    "PP-HGNetV2-B5",
    "PP-HGNetV2-B6",
    "CLIP_vit_base_patch16_224",
    "CLIP_vit_large_patch14_224",
    "MobileNetV4_conv_small",
    "MobileNetV4_conv_medium",
    "MobileNetV4_hybrid_medium",
    "MobileNetV4_conv_large",
    "MobileNetV4_hybrid_large",
    "StarNet-S1 ",
    "StarNet-S2",
    "StarNet-S3",
    "StarNet-S4",
    "FasterNet-T0",
    "FasterNet-T1",
    "FasterNet-T2",
    "FasterNet-S",
    "FasterNet-M",
    "FasterNet-L",
    "PP-LCNet_x1_0_ML",
    "ResNet50_ML",
    "PP-HGNetV2-B0_ML",
    "PP-HGNetV2-B4_ML",
    "PP-HGNetV2-B6_ML",
    "CLIP_vit_base_patch16_224_ML",
    "PPLCNet_x1_0_pedestrian_attribute",
    "PP-LCNet_x1_0_vehicle_attribute",
    "PP-ShiTuV2_rec",
    "PP-ShiTuV2_rec_CLIP_vit_base",
    "PP-ShiTuV2_rec_CLIP_vit_large",
    "PP-YOLOE_plus-X",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "RT-DETR-L",
    "RT-DETR-H",
    "RT-DETR-X",
    "RT-DETR-R18",
    "RT-DETR-R50",
    "PicoDet-L",
    "PicoDet-M",
    "PicoDet-S",
    "PicoDet-XS",
    "CenterNet-DLA-34",
    "CenterNet-ResNet50",
    "FCOS-ResNet50",
    "DETR-R50",
    "YOLOX-N",
    "YOLOX-T",
    "YOLOX-S",
    "YOLOX-M",
    "FasterRCNN-ResNet34-FPN",
    "FasterRCNN-ResNet50",
    "FasterRCNN-ResNet50-FPN",
    "FasterRCNN-ResNet50-vd-FPN",
    "FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "FasterRCNN-ResNet101",
    "FasterRCNN-ResNet101-FPN",
    "FasterRCNN-ResNeXt101-vd-FPN",
    "FasterRCNN-Swin-Tiny-FPN",
    "YOLOv3-DarkNet53",
    "YOLOv3-ResNet50_vd_DCN",
    "YOLOv3-MobileNetV3",
    "Cascade-FasterRCNN-ResNet50-FPN",
    "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "PP-ShiTuV2_det",
    "PP-YOLOE-S_human",
    "PP-YOLOE-L_human",
    "PP-YOLOE-L_vehicle",
    "PP-YOLOE-S_vehicle",
    "PP-YOLOE_plus_SOD-S",
    "PP-YOLOE_plus_SOD-L",
    "PP-YOLOE_plus_SOD-largesize-L",
    "STFPM",
    "Deeplabv3-R50",
    "Deeplabv3-R101",
    "Deeplabv3_Plus-R50",
    "Deeplabv3_Plus-R101",
    "PP-LiteSeg-T",
    "OCRNet_HRNet-W18",
    "OCRNet_HRNet-W48",
    "SeaFormer-tiny",
    "SeaFormer-small",
    "SeaFormer-base",
    "SeaFormer-large",
    "SegFormer-B0",
    "SegFormer-B1",
    "SegFormer-B2",
    "SegFormer-B3",
    "SegFormer-B4",
    "SegFormer-B5",
    "PP-YOLOE_seg-S",
    "SOLOv2",
    "MaskRCNN-ResNet50",
    "MaskRCNN-ResNet50-FPN",
    "MaskRCNN-ResNet50-vd-FPN",
    "MaskRCNN-ResNet101-FPN",
    "MaskRCNN-ResNet101-vd-FPN",
    "MaskRCNN-ResNeXt101-vd-FPN",
    "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN",
    "Cascade-MaskRCNN-ResNet50-FPN",
    "Mask-RT-DETR-H",
    "Mask-RT-DETR-L",
    "Mask-RT-DETR-X",
    "Mask-RT-DETR-M",
    "Mask-RT-DETR-S",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "ch_RepSVTR_rec",
    "ch_SVTRv2_rec",
    "PicoDet_layout_1x",
    "PicoDet-L_layout_3cls",
    "RT-DETR-H_layout_3cls",
    "RT-DETR-H_layout_17cls",
    "SLANet",
    "SLANet_plus",
    "DLinear",
    "DLinear_ad",
    "RLinear",
    "NLinear",
    "PatchTST",
    "PatchTST_ad",
    "TimesNet",
    "TimesNet_cls",
    "TimesNet_ad",
    "TiDE",
    "Nonstationary",
    "Nonstationary_ad",
    "AutoEncoder_ad",
    "PicoDet_LCNet_x2_5_face",
    "UVDoc",
    "PP-OCRv4_mobile_seal_det",
    "PP-OCRv4_server_seal_det",
    "PP-LCNet_x1_0_doc_ori",
]

XPU_WHITELIST = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet18_vd",
    "ResNet34_vd",
    "ResNet50_vd",
    "ResNet101_vd",
    "ResNet152_vd",
    "ResNet200_vd",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_0",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_5",
    "PP-LCNet_x2_0",
    "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5",
    "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0",
    "MobileNetV3_large_x1_25",
    "MobileNetV3_small_x0_35",
    "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75",
    "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25",
    "PP-HGNet_small",
    "PP-HGNet_tiny",
    "PP-HGNet_base",
    "PP-YOLOE_plus-X",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "PicoDet-L",
    "PicoDet-M",
    "PicoDet-S",
    "PicoDet-XS",
    "STFPM",
    "PP-LiteSeg-T",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "PicoDet_layout_1x",
    "DLinear",
    "RLinear",
    "NLinear",
    "PicoDet_LCNet_x2_5_face",
]

GCU_WHITELIST = [
    "ConvNeXt_base_224",
    "ConvNeXt_base_384",
    "ConvNeXt_large_224",
    "ConvNeXt_large_384",
    "ConvNeXt_small",
    "ConvNeXt_tiny",
    "FasterNet-L",
    "FasterNet-M",
    "FasterNet-S",
    "FasterNet-T0",
    "FasterNet-T1",
    "FasterNet-T2",
    "MobileNetV1_x0_25",
    "MobileNetV1_x0_5",
    "MobileNetV1_x0_75",
    "MobileNetV1_x1_0",
    "MobileNetV2_x0_25",
    "MobileNetV2_x0_5",
    "MobileNetV2_x1_0",
    "MobileNetV2_x1_5",
    "MobileNetV2_x2_0",
    "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5",
    "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0",
    "MobileNetV3_large_x1_25",
    "MobileNetV3_small_x0_35",
    "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75",
    "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25",
    "MobileNetV4_conv_large",
    "MobileNetV4_conv_medium",
    "MobileNetV4_conv_small",
    "PP-HGNet_base",
    "PP-HGNet_small",
    "PP-HGNet_tiny",
    "PP-HGNetV2-B0",
    "PP-HGNetV2-B1",
    "PP-HGNetV2-B2",
    "PP-HGNetV2-B3",
    "PP-HGNetV2-B4",
    "PP-HGNetV2-B5",
    "PP-HGNetV2-B6",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_0",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_0",
    "PP-LCNet_x2_5",
    "PP-LCNetV2_base",
    "PP-LCNetV2_large",
    "PP-LCNetV2_small",
    "ResNet18_vd",
    "ResNet18",
    "ResNet34_vd",
    "ResNet34",
    "ResNet50_vd",
    "ResNet50",
    "ResNet101_vd",
    "ResNet101",
    "ResNet152_vd",
    "ResNet152",
    "ResNet200_vd",
    "StarNet-S1",
    "StarNet-S2",
    "StarNet-S3",
    "StarNet-S4",
    "FCOS-ResNet50",
    "PicoDet-L",
    "PicoDet-M",
    "PicoDet-S",
    "PicoDet-XS",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "PP-YOLOE_plus-X",
    "RT-DETR-H",
    "RT-DETR-L",
    "RT-DETR-R18",
    "RT-DETR-R50",
    "RT-DETR-X",
    "PP-YOLOE-L_human",
    "PP-YOLOE-S_human",
    "PP-OCRv4_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_rec",
]
