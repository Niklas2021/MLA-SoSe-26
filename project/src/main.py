import torch
import sys
import autotuner.device_props as device_props


if torch.cuda.is_available():
    Properties = device_props.get_device_properties()
else:
    sys.exit("No CUDA available")