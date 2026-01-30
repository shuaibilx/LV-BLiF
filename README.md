### LV-BLiF

This repository provides the official implementation of **LV-BLiF**, a no-reference light field image quality assessment (NR-LFIQA) framework based on **language–vision representation learning**.

LV-BLiF integrates a **textual prompts–assisted semantic branch** with a **subspace cues–assisted visual branch** to achieve content-aware and spatio-angular consistent quality assessment for light field images.

* * *

1. Environment Setup

* * *

All required Python dependencies are listed in `requirements.txt`.

Please create a virtual environment and install dependencies via: pip install -r requirements.txt

> **Note**The large multimodal model **mPLUG-Owl2** is **not trained online** in this project.Semantic features are **pre-extracted offline** and provided via cloud storage (see Section 3).

* * *

2. Code Structure

* * *

    LV-BLiF/
    ├── configs/                  # Configuration files
    │   └── combined.yaml
    ├── data/                     # Pre-extracted semantic features
    ├── mplug_owl2/               # mPLUG-Owl2 related code
    ├── arg.py                    
    ├── data_splits.py            
    ├── dataset.py                
    ├── metrics.py                
    ├── utils.py                  
    └── main.py                   

* * *

3. Data Preparation

* * *

### 3.1 MATLAB-Processed Datasets

The light field datasets preprocessed using MATLAB are provided via Quark NetDisk:

* **MATLAB-processed datasets**:[夸克网盘分享](https://pan.quark.cn/s/9b8361eb5785)

After downloading, please organize the dataset according to the structure expected by `dataset.py`.

* * *

### 3.2 Pre-extracted Semantic Features (mPLUG-Owl2)

To reduce computational cost and improve reproducibility, semantic features extracted by **mPLUG-Owl2** are provided offline:

* **Pre-extracted semantic features**:[夸克网盘分享](https://pan.quark.cn/s/49e6e75d9321)

Please place the downloaded semantic feature files into the following directory: LV-BLiF/data/

> **Important**mPLUG-Owl2 is **used only for offline feature extraction**.During training and testing, LV-BLiF directly loads these pre-extracted semantic features.

* * *

4. Running the Code

* * *

All experiments are controlled by a unified configuration file: configs/combined.yaml

You can run the model on different datasets using the following commands.

### 4.1 NBU-LF1.0 Dataset

    python main.py --config configs/combined.yaml --active_dataset NBU

### 4.2 SHU Dataset

    python main.py --config configs/combined.yaml --active_dataset SHU

### 4.3 Win5-LID Dataset

    python main.py --config configs/combined.yaml --active_dataset Win5LID

* * *

5. Citation

* * *

If this project or the provided resources are helpful to your research, please cite our paper:

> **LV-BLiF: Harnessing Language-Vision Representation Learning for Blind Light Field Image Quality Assessment**

(The BibTeX entry will be updated after publication.)

* * *

6. Contact

* * *

If you have any questions regarding the code, datasets, or implementation details, please feel free to contact:

📧 **Email**: [2871474054@qq.com](mailto:2871474054@qq.com)

* * *
