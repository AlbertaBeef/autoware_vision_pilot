#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import inspect
import torch
import onnx
from argparse import ArgumentParser
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# The `external_data` kwarg was added in torch ~2.6. Older torches reject it.
# Detect once at import time so the call sites below stay clean.
_EXPORT_KWARGS = {}
if "external_data" in inspect.signature(torch.onnx.export).parameters:
    _EXPORT_KWARGS["external_data"] = False
from Models.model_components.scene_seg_network import SceneSegNetwork
from Models.model_components.scene_3d_network import Scene3DNetwork
from Models.model_components.domain_seg_network import DomainSegNetwork
from Models.model_components.auto_speed.auto_speed_network import AutoSpeedNetwork
from Models.model_components.ego_lanes_network import EgoLanesNetwork
from Models.model_components.auto_steer.auto_steer_network import AutoSteerNetwork
from Models.model_components.autodrive.autodrive_network import AutoDrive
from Models.model_components.lite_models.DeepLabv3Plus import DeepLabV3Plus
from Models.exports.lite_models.helpers import SCENESEGLITE_DEFAULT_CONFIG
def main():

    # Argument parser for data root path and save path
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="network_name", required=True, \
                        help="specify the name of the network which will be benchmarked")

    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", required=True, \
                        help="path to pytorch checkpoint file to load model dict")

    parser.add_argument("-o", "--onnx_model_path", dest="onnx_model_path", required=True, \
                        help="path to converted ONNX model, must include output file name with .onnx extension")

    parser.add_argument("--input-shape", dest="input_shape", default=None,
                        help="Override default input shape as comma-separated NCHW "
                             "(e.g. '1,3,256,512'). Useful for backends like MemryX that "
                             "may need a smaller spatial resolution to fit the chip budget. "
                             "Not used for AutoDrive (two-input model).")

    parser.add_argument("--opset", dest="opset", type=int, default=18,
                        help="ONNX opset version (default: 18). Use 17 when running on older "
                             "PyTorch (e.g. 2.3.x in the MemryX venv) where opset-18 emits "
                             "ReduceMean with the legacy attribute form that the ONNX checker "
                             "rejects.")

    args = parser.parse_args()

    # Get input arguments
    model_name = args.network_name
    model_checkpoint_path = args.model_checkpoint_path
    onnx_model_path = args.onnx_model_path

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')


    # Instantiating Model and setting to evaluation mode
    model = 0
    if(model_name == 'SceneSeg'):
        print('Processing SceneSeg Network')
        model = SceneSegNetwork()
    elif (model_name == 'Scene3D'):
        print('Processing Scene3D Network')
        sceneSegNetwork = SceneSegNetwork()
        model = Scene3DNetwork(sceneSegNetwork)
    elif (model_name == 'DomainSeg'):
        print('Processing DomainSeg Network')
        sceneSegNetwork = SceneSegNetwork()
        model = DomainSegNetwork(sceneSegNetwork)
    elif (model_name == 'SceneSegLite'):
        print('Processing SceneSegLite Network')
        cfg = SCENESEGLITE_DEFAULT_CONFIG
        backbone_cfg = cfg["network"]["backbone"]
        decoder_cfg = cfg["network"]["decoder"]
        head_cfg = cfg["network"]["head"]
        # Match lite_trainer_base._build_encoder_decoder: SMP encoder names are
        # 'timm-efficientnet-b1', not 'efficientnet_b1'. Apply the same remap.
        encoder_name = backbone_cfg["type"]
        if "timm" not in encoder_name:
            encoder_name = "timm-" + encoder_name.replace("_", "-")
        model = DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_output_stride=backbone_cfg.get("output_stride", 16),
            encoder_depth=backbone_cfg.get("encoder_depth", 5),
            decoder_atrous_rates=tuple(decoder_cfg.get("aspp_dilations", [12, 24, 36])),
            decoder_channels=decoder_cfg.get("deeplabv3plus_decoder_channels", 256),
            output_channels=cfg["network"].get("output_channels", 19),
            head_depth=head_cfg.get("head_depth", 1),
            head_mid_channels=head_cfg.get("head_mid_channels", None),
            head_activation=head_cfg.get("head_activation", None),
            head_upsampling=head_cfg.get("head_upsampling", 4),
            head_kernel_size=head_cfg.get("head_kernel_size", 1),
        )
    elif (model_name == 'AutoSpeed'):
        print('Processing AutoSpeed Network')
        model = AutoSpeedNetwork().build_model(version='n', num_classes=4)
    elif (model_name == 'EgoLanes'):
        print('Processing EgoLanes Network')
        model = EgoLanesNetwork()
    elif (model_name == 'AutoSteer'):
        print('Processing AutoSteer Network')
        model = AutoSteerNetwork().build_model(version='n')
    elif (model_name == 'AutoDrive'):
        print('Processing AutoDrive Network')
        model = AutoDrive()
    else:
        raise Exception("Model name not specified correctly, please check")

    # Loading Pytorch checkpoint
    print('Loading Network')
    if(len(model_checkpoint_path) > 0):
            checkpoint = torch.load(model_checkpoint_path, weights_only=False, map_location=device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # If checkpoint['model'] is a model object, extract its state_dict
                if hasattr(checkpoint['model'], 'state_dict'):
                    model.load_state_dict(checkpoint['model'].state_dict())
                else:
                    model.load_state_dict(checkpoint['model'])
            elif isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                # Lite trainer checkpoint format (lite_trainer_base.py saves under 'model_state')
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
    else:
        raise ValueError('No path to checkpiont file provided in class initialization')
    model = model.to(device)
    model = model.eval()

    # Fake input data
    if args.input_shape is not None and model_name != 'AutoDrive':
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
        print(f'Using user-supplied input shape: {input_shape}')
    elif model_name == 'AutoSpeed':
        input_shape=(1, 3, 512, 1024)
    elif model_name == 'AutoSteer':
        input_shape=(1, 3, 512, 1024)
    elif model_name == 'AutoDrive':
        input_shape=(1, 3, 512, 1024)
    else:
        input_shape=(1, 3, 320, 640)
    input_data = torch.randn(input_shape).to(device)
    input_data_prev = torch.randn(input_shape).to(device)

    # Test inference
    print('Testing inference')
    if model_name == 'AutoDrive':
        _ = model(input_data_prev, input_data)
    else:
        _ = model(input_data)

    # Export FP32 model to onnx
    print('Converting model to ONNX at FP32 and exporting')
    if model_name == 'AutoDrive':
        torch.onnx.export(model,                                          # model
                        (input_data_prev, input_data),                    # model input tuple
                        onnx_model_path,                                  # path
                        export_params=True,                               # store the trained parameter weights inside the model file
                        opset_version=args.opset,                         # the ONNX version to export the model to
                        do_constant_folding=True,                         # constant folding for optimization
                        input_names=['image_prev', 'image_curr'],         # input names
                        output_names=['distance', 'curvature', 'flag_logit'], # output names
                        dynamic_axes={
                            'image_prev': {0: 'batch_size'},
                            'image_curr': {0: 'batch_size'},
                            'distance': {0: 'batch_size'},
                            'curvature': {0: 'batch_size'},
                            'flag_logit': {0: 'batch_size'},
                        },
                        **_EXPORT_KWARGS)
    elif model_name == 'AutoSteer':
        # AutoSteer 2.0 returns a 2-tuple (lane_value, height)
        torch.onnx.export(model,
                        input_data,
                        onnx_model_path,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['lane_value', 'height'],
                        dynamic_axes={
                            'input': {0: 'batch_size'},
                            'lane_value': {0: 'batch_size'},
                            'height': {0: 'batch_size'},
                        },
                        **_EXPORT_KWARGS)
    else:
        torch.onnx.export(model,                                          # model
                        input_data,                                       # model input
                        onnx_model_path,                                  # path
                        export_params=True,                               # store the trained parameter weights inside the model file
                        opset_version=args.opset,                         # the ONNX version to export the model to
                        do_constant_folding=True,                         # constant folding for optimization
                        input_names = ['input'],                          # input names
                        output_names = ['output'],                        # output names
                        dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                        'output' : {0 : 'batch_size'}},
                        **_EXPORT_KWARGS)

    # Run checks on exported FP32 ONNX network
    ONNX_network = onnx.load(onnx_model_path)
    onnx.checker.check_model(ONNX_network)
    print('Checks passed - export complete')

if __name__ == '__main__':
  main()
# %%