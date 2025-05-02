# hardware.py
import platform
import os
import numpy as np
import warnings


def detect_hardware():
    """
    Detect available hardware accelerators and return configuration.

    Returns:
        dict: Hardware configuration with acceleration capabilities
    """
    hardware_config = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": False,
        "mps_available": False,  # Apple Metal Performance Shaders
        "recommended_backend": "cpu",
        "num_cores": os.cpu_count() or 1
    }

    # Check for CUDA
    try:
        import torch
        hardware_config["cuda_available"] = torch.cuda.is_available()
        if hardware_config["cuda_available"]:
            hardware_config["cuda_devices"] = torch.cuda.device_count()
            hardware_config["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in
                                                    range(hardware_config["cuda_devices"])]
            hardware_config["recommended_backend"] = "cuda"
    except ImportError:
        pass

    # Check for Apple Silicon / Metal
    if hardware_config["platform"] == "Darwin" and hardware_config["architecture"] == "arm64":
        hardware_config["is_apple_silicon"] = True
        try:
            import torch
            hardware_config["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if hardware_config["mps_available"]:
                hardware_config["recommended_backend"] = "mps"
        except ImportError:
            # Check for other MPS-enabled libraries
            try:
                import tensorflow as tf
                hardware_config["tf_metal_available"] = tf.config.list_physical_devices('GPU')
                if hardware_config["tf_metal_available"]:
                    hardware_config["recommended_backend"] = "tensorflow"
            except ImportError:
                pass
    else:
        hardware_config["is_apple_silicon"] = False

    return hardware_config