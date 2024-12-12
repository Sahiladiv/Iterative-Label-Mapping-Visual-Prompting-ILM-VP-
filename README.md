# Modified ILM-VP Repository

This repository is a modified version of the original repository present at [ILM-VP](https://github.com/optml-group/ilm-vp).

The research paper associated with this work is [Understanding and Improving Visual Prompting: A Label-Mapping Perspective](https://arxiv.org/pdf/2211.11635).

## Installation

1. Run the following command to install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
2. Configure paths in `cfg.py` as needed:
   - Update the `data_path` variable to set the dataset path.

## Datasets Used

The authors of the original paper tested models on various datasets:
- Abide
- Flowers102
- DTD
- UCF101
- Food101
- EuroSAT
- OxfordPets
- StanfordCars
- SUN397

For simplicity and easier computation, this repository focuses on two datasets:
- **Abide**
- **DTD**

## Models Used

The original authors utilized the following models:
- **ResNet-18**
- **ResNet-50**
- **ResNeXt-101-32x8d**

In this repository, two additional models have been added to evaluate their efficiency:
- **VGG-16**
- **VGG-19**

## Generate Prompts

### VP on CNN

#### ILM-VP:
```bash
python experiments/cnn/ilm_vp_updated.py --network model_name --dataset dataset_name --epoch num_of_epochs
```

#### FLM-VP:
```bash
python experiments/cnn/flm_vp_updated.py --network model_name --dataset dataset_name --epoch num_of_epochs
```

#### RLM-VP:
```bash
python experiments/cnn/rlm_vp_updated.py --network model_name --dataset dataset_name --epoch num_of_epochs
```

### TP on CLIP

#### ILM-TP-VP:
```bash
python experiments/clip/ilm_tp_vp_updated.py --dataset dataset_name --epoch num_of_epochs
```

#### SINGLE-TP-VP:
```bash
python experiments/clip/single_tp_vp_updated.py --dataset dataset_name
```

### Parameters

- **`model_name`**: Options include `resnet18`, `resnet50`, `instagram`, `vgg16`, `vgg19`
- **`dataset_name`**: Options include `abide`, `dtd`
- **`num_of_epochs`**: Default is `200`. You can set it to `100` or other values as needed.

---

