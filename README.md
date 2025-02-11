# LungSeg2D_Unet_ResNet

This repository provides a 2D U-Net segmentation pipeline (with a ResNet-50 backbone) for neonatal lung MRI data.

## Getting Started

### 1. Clone this Repository

```bash
git clone https://github.com/Dhyey-Patel28/LungSeg2D_Unet_ResNet.git
cd LungSeg2D_Unet_ResNet
```

### 2. Create and Activate a Virtual Environment

#### Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\activate
```

#### macOS / Linux (Bash):

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

Once the environment is active, install dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

### 4. Set Up Your Data Path

Open `config.py` and modify the following line to point to your data folder (where `proton.nii` and `mask.nii` live):

```
DATA_PATH = 'Neonatal Test Data - August 1 2024'
```

Replace `Neonatal Test Data - August 1 2024` with the path to your own data directory.

### 5. Run the Main Script

Run `Main_Resnet_Xe.py` to start training:

```
python Main_Resnet_Xe.py
```

## Stopping The Model

To interrupt training or the infinite loop, press `Ctrl + C` in the terminal or command prompt window.

## Repository Structure

- `Main_Resnet_Xe.py`
  The main script that sets up the data generators, model, and training loop.
- `HelpFunctions.py`
Utility functions for data loading, plotting, and evaluation metrics.
- `config.py`
Configuration script with hyperparameters, data paths, and augmentation settings.
- `requirements.txt`
Python libraries needed to run the segmentation pipeline.
- `UnitTest_Model.py`
Example of a test script or placeholder for unit tests.
- `Test_Model.py`
Additional script for testing or example usage.

## Contributing

Feel free to open an issue or submit a pull request if you have improvements or find bugs.
