# classify_skin_cancer_images

This repository contains a machine learning pipeline that processes image data, trains multiple models one after the other (VGG16, ResNet50, InceptionV3, EfficientNetB0, and DenseNet121), and performs post-processing to evaluate the results.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Scripts Overview](#scripts-overview)
- [Link to dataset](#Link-to-dataset)


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/TimGoebel/classify_skin_cancer_images
.git
    cd classify_skin_cancer_images
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the pipeline, you need to have a directory structure where the image data is stored in a folder named `data` within the current working directory. The pipeline will preprocess the data, train the models, and then apply post-processing.

### Running the Pipeline

Execute the following command to start the pipeline:

```bash
python main.py
```

### Arguments

- `directory`: The directory containing the `data` folder. The script will automatically use the current working directory if not specified.
- `target_name`: The name of the folder containing the image data. Default is `data`.

## Directory Structure

The expected directory structure is as follows:
create folders:
data
models
plots
reports

```
classify_skin_cancer_images/
│
├── data/
│   ├── class1/
│   ├── class2/
│   └── ...
├── models/
│   └── ...
├── plots/
│   └── ...
├── reports/
│   └── ...
├── main.py
├── preprocessing.py
├── model_VGG.py
├── model_resnet50.py
├── model_inceptionV3.py
├── model_efficientnet.py
├── model_densenet.py
├── post_processing.py
└── requirements.txt
```

## Scripts Overview

- **`main.py`**: The main script that orchestrates the entire pipeline trains all the models.
- **`preprocessing.py`**: Handles data preprocessing, including resizing images.
- **`model_VGG.py`**: Contains the function to train a VGG16 model and plots.
- **`model_resnet50.py`**: Contains the function to train a ResNet50 model and plots.
- **`model_inceptionV3.py`**: Contains the function to train an InceptionV3 model and plots.
- **`model_efficientnet.py`**: Contains the function to train an EfficientNetB0 model and plots.
- **`model_densenet.py`**: Contains the function to train a DenseNet model and plots.
- **`post_processing.py`**: Handles the post-processing steps, such as generating confusion matrices and classification reports and ensembling models 2 at a time and runs throughs all of them at once.

### Link to dataset

https://drive.google.com/file/d/1yc3LSiviVfrnyluHClZfmo4tjFGCxe_E/view?usp=sharing 

