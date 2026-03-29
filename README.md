```markdown
# LV-BLiF

<div align="center">

<h3>LV-BLiF: Harnessing Language-Vision Representation Learning for Blind Light Field Image Quality Assessment</h3>

[![Paper](https://img.shields.io/badge/Journal-IEEE_TBC-blue.svg)](https://doi.org/10.1109/TBC.2026.3668512)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTBC.2026.3668512-darkred.svg)](https://doi.org/10.1109/TBC.2026.3668512)
[![Dataset](https://img.shields.io/badge/Dataset-Quark_NetDisk-green.svg)](#3-data-preparation)

*Official PyTorch Implementation for No-Reference Light Field Image Quality Assessment*

</div>

---

## 📖 Introduction

This repository provides the official implementation of **LV-BLiF**, a no-reference light field image quality assessment (NR-LFIQA) framework based on **language–vision representation learning**. 

LV-BLiF elegantly integrates a **textual prompts-assisted semantic branch** with a **subspace cues-assisted visual branch** to achieve content-aware and spatio-angular consistent quality assessment for light field images.

---

## 🛠️ Environment Setup

All required Python dependencies are listed in the `requirements.txt` file. Please create a virtual environment and install the dependencies via:

```bash
# Clone the repository
git clone [https://github.com/shuaibilx/LV-BLiF.git](https://github.com/shuaibilx/LV-BLiF.git)
cd LV-BLiF

# Install dependencies
pip install -r requirements.txt
```

> 💡 **Note on Large Multimodal Model:**
> The large multimodal model **mPLUG-Owl2** is **not trained online** in this project to save computational resources. Semantic features are **pre-extracted offline** and provided via cloud storage (see Section 3).

---

## 📂 Code Structure

```text
LV-BLiF/
├── configs/                  # Configuration files
│   └── combined.yaml
├── data/                     # Pre-extracted semantic features (Need to be downloaded)
├── mplug_owl2/               # mPLUG-Owl2 related code (for offline extraction)
├── arg.py                    # Argument parser
├── data_splits.py            # Dataset splitting logic
├── dataset.py                # PyTorch Dataset definitions
├── metrics.py                # Evaluation metrics (PLCC, SROCC, etc.)
├── utils.py                  # Utility functions
└── main.py                   # Main training and evaluation script
```

---

## 📦 Data Preparation

### 3.1 MATLAB-Processed Datasets
The light field datasets preprocessed using MATLAB are provided via Quark NetDisk:
- 📥 **[Download MATLAB-processed datasets](https://pan.quark.cn/s/9b8361eb5785)**

After downloading, please organize the dataset according to the directory structure expected by `dataset.py`.

### 3.2 Pre-extracted Semantic Features (mPLUG-Owl2)
To significantly reduce computational cost and improve reproducibility, semantic features extracted by **mPLUG-Owl2** are provided offline:
- 📥 **[Download Pre-extracted semantic features](https://pan.quark.cn/s/49e6e75d9321)**

Please place the downloaded semantic feature files directly into the following directory:
```bash
LV-BLiF/data/
```

> ⚠️ **Important:**
> mPLUG-Owl2 is **used only for offline feature extraction**. During the actual training and testing phases, LV-BLiF directly loads these pre-extracted semantic features from the `data/` folder.

---

## 🚀 Running the Code

All experiments are controlled by a unified configuration file located at `configs/combined.yaml`. You can run the model on different datasets using the following commands:

### NBU-LF1.0 Dataset
```bash
python main.py --config configs/combined.yaml --active_dataset NBU
```

### SHU Dataset
```bash
python main.py --config configs/combined.yaml --active_dataset SHU
```

### Win5-LID Dataset
```bash
python main.py --config configs/combined.yaml --active_dataset Win5LID
```

---

## 📝 Citation

If this project or the provided resources are helpful to your research, please cite our IEEE Transactions on Broadcasting paper:

```bibtex
@ARTICLE{11420244,
  author={Liao, Xin and Chai, Xiongli and Chen, Hangwei and Jing, Weiyi and Shao, Feng and Jiang, Qiuping},
  journal={IEEE Transactions on Broadcasting}, 
  title={LV-BLiF: Harnessing Language-Vision Representation Learning for Blind Light Field Image Quality Assessment}, 
  year={2026},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TBC.2026.3668512}
}
```

---

## 📧 Contact

If you have any questions regarding the code, datasets, or implementation details, please feel free to contact:

- **Email**: [2871474054@qq.com](mailto:2871474054@qq.com)
```
