import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np


TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002
MODEL_NAME = "fr_model"
MODEL_VERSION = "1"
MODEL_INPUT_NAME = "input.1"
MODEL_OUTPUT_NAME = "516"
MODEL_IMAGE_SIZE = (112, 112)


def prepare_model_repository(model_repo: Path) -> None:
    fr_dir = model_repo / "fr_model" / "1"
    fr_dir.mkdir(parents=True, exist_ok=True)
    fr_config_path = model_repo / "fr_model" / "config.pbtxt"
    
    fr_config = textwrap.dedent(f"""
        name: "fr_model"
        platform: "onnxruntime_onnx"
        max_batch_size: 0
        input [ {{ name: "input.1", data_type: TYPE_FP32, dims: [1, 3, 112, 112] }} ]
        output [ {{ name: "516", data_type: TYPE_FP32, dims: [1, 512] }} ]
        instance_group [ {{ kind: KIND_CPU }} ]
    """).strip()
    fr_config_path.write_text(fr_config)

    det_dir = model_repo / "face_detector" / "1"
    det_dir.mkdir(parents=True, exist_ok=True)
    det_config_path = model_repo / "face_detector" / "config.pbtxt"
    
    det_config = textwrap.dedent(f"""
        name: "face_detector"
        platform: "onnxruntime_onnx"
        max_batch_size: 0
        input [ {{ name: "input.1", data_type: TYPE_FP32, dims: [1, 3, 640, 640] }} ]
        output [ {{ name: "443", data_type: TYPE_FP32, dims: [12800, 1] }} ]
        instance_group [ {{ kind: KIND_CPU }} ]
    """).strip()
    det_config_path.write_text(det_config)
    
    print("[triton] Both model configs prepared!")


def start_triton_server(model_repo: Path) -> Any:
    """
    Launch Triton Inference Server (CPU) pointing at model_repo and return a handle/process.
    """
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH. Is Triton installed?")

    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server with command: {' '.join(cmd)}")
    time.sleep(3)  # Give the server a moment to load the model
    return process


def stop_triton_server(server_handle: Any) -> None:
    """
    Cleanly stop the Triton server started in start_triton_server.
    """
    if server_handle is None:
        return

    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton server killed after timeout.")


def create_triton_client(url: str) -> Any:
    """
    Initialize a Triton HTTP client for the FR model endpoint.
    """
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("tritonclient[http] is required; install from requirements.txt") from exc

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client


def run_inference(client: Any, image_bytes: bytes, model_name: str = "fr_model") -> Any:
    try:
        from io import BytesIO
        from PIL import Image
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("Pillow, numpy, and tritonclient[http] are required.") from exc

    if model_name == "face_detector":
        input_name = "input.1"
        output_name = "443"
        image_size = (640, 640)
    else: 
        input_name = "input.1"
        output_name = "516"
        image_size = (112, 112)

    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize(image_size) 
        np_img = np.asarray(img, dtype=np.float32) / 255.0

    np_img = np.transpose(np_img, (2, 0, 1)) 
    batch = np.expand_dims(np_img, axis=0)

    infer_input = httpclient.InferInput(input_name, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    infer_output = httpclient.InferRequestedOutput(output_name)
    
    response = client.infer(model_name=model_name, inputs=[infer_input], outputs=[infer_output])
    
    return response.as_numpy(output_name)
