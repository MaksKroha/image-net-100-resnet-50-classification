# ImageNet-100 ResNet-50 Classification

This repository contains the code and resources for an image classification task on the ImageNet-100 dataset using a pre-trained ResNet-50 model. The project demonstrates the power of transfer learning for building highly accurate image classifiers with less training time.

## Project Overview

The goal of this project is to fine-tune a ResNet-50 model, pre-trained on the full ImageNet dataset, to classify images from the ImageNet-100 subset. This approach, known as transfer learning, leverages the knowledge learned from a larger dataset to improve performance on a smaller, related task.

## Key Features

* **Transfer Learning:** Utilizes a pre-trained ResNet-50 model to achieve high accuracy with minimal training.
* **ResNet-50 Architecture:** Employs a state-of-the-art deep residual network known for its effectiveness in computer vision tasks.
* **ImageNet-100 Dataset:** Uses a manageable subset of the famous ImageNet dataset, making it accessible for users without extensive computational resources.
* **Modular Code:** The code is structured to be easily understandable and adaptable for other datasets or models.

## Dataset

This project uses the **ImageNet-100** dataset, which is a subset of the larger ILSVRC (ImageNet Large Scale Visual Recognition Challenge) dataset. It contains:
* **100 object categories.**
* A total of **~130,000 images.**
* Each category has approximately **1,300 training images** and **50 validation images.**

This smaller dataset is ideal for faster training and experimentation while still providing a robust challenge for image classification models.

## Model Architecture

The core of this project is the **ResNet-50** model, a 50-layer deep convolutional neural network. Its key innovation is the use of "residual blocks" with skip connections, which help to mitigate the vanishing gradient problem in very deep networks. This allows for the training of much deeper and more accurate models.

We use a model pre-trained on the full ImageNet (1000 classes) and fine-tune it on the ImageNet-100 dataset.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python 3.7+ installed. The project relies on several Python libraries. The main ones are:

* `pytorch`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `opencv-python`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MaksKroha/image-net-100-resnet-50-classification.git](https://github.com/MaksKroha/image-net-100-resnet-50-classification.git)
    cd image-net-100-resnet-50-classification
    ```

2.  **Install the required packages:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Download the Dataset:**
    You will need to download the ImageNet-100 dataset. You can often find it through academic sources or torrents. Once downloaded, place it in a `data/` directory or update the path in the configuration files.

## Usage

The following sections describe how to train, evaluate, and use the model.

### Training

To train the model, run the `train.py` script. You can customize the training process with command-line arguments.

```bash
python train.py --dataset_path /path/to/your/imagenet-100 --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Evaluation
To evaluate the performance of a trained model on the validation or test set, use the evaluate.py script.

```Bash
python evaluate.py --dataset_path /path/to/your/imagenet-100 --model_path /path/to/your/trained_model.h5
```

### Prediction
To make a prediction on a single image, you can use the predict.py script.

```Bash
python predict.py --image_path /path/to/your/image.jpg --model_path /path/to/your/trained_model.h5
```
###Results
After training, the model should achieve a high accuracy on the validation set. Here's an example of expected results:

- Top-1 Accuracy: ~90-95%

- Top-5 Accuracy: ~98-99%

You can include charts for training/validation accuracy and loss here:

### Contributing
Contributions are welcome! If you have suggestions for improving this project, please feel free to open an issue or submit a pull request.

1. Fork the Project

2. Create your Feature Branch (git checkout -b feature/AmazingFeature)

3. Commit your Changes (git commit -m 'Add some AmazingFeature')

4. Push to the Branch (git push origin feature/AmazingFeature)

5. Open a Pull Request

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Acknowledgments
1. The creators of the ResNet architecture.

2. The curators of the ImageNet dataset.

3. The open-source community for providing the tools and libraries that made this project possible.
