# wcd_project_template
This repository acts as the Computer Vision Project Template for WeCloudData.

The goal of this template is to make the Capstone Project implementation easy for the students.

### The Template can be used for 4 different problems
- Image Classification
- Image Segmentation
- Object Detection
- Object Tracking

### There are 5 key components of any ML Project
- Problem Statement
- Dataset
- Model
- Model Training
- Model Evaluation

The Template provides wrappers (Data and Model classes) and an API (ModelEngine) that works hand-in-hand with a Configuration. Together, they form a complete Training API.

## Installation
```
git clone https://github.com/jjaskirat/wcd_project_template.git
cd wcd_project_template
python3 -m venv wcd-venv
source wcd-venv/bin/activate
python3 setup.py install
```

### Video Demo Link!!!:
https://www.dropbox.com/scl/fo/ti6gyhk68w6ihlr7lpj7r/AApRblvm3K4sPU34PREC0pU?rlkey=2t45bew7erkt6q6hfmbiujhhg&st=gjtxymoa&dl=0

## Problem Statement / Task
The task is a elements of machine learning, acting as a guiding beacon for the entire ML process. It outlines the exact problem to be solved, as well as the model's objectives and aims. 

From data collection and preprocessing through model selection and model validation, every choice in the ML pipeline is inextricably related to the nature of the task at hand. 

The task specifies the sort of data needed, the features to build, and the metrics to measure success. It influences algorithm selection and hyperparameter tweaking, ensuring that the model accurately matches the demands of the task. 

In the end, the task decides how the machine learning model is deployed and used in practical applications, giving it the foundation for success.

### Template Usage:
The Problem Statement has to be defined by the student entirely. The instructors and TA's will help the students form their problem statement.

## Data
Data is first among the key elements of machine learning, an indispensable ingredient that fuels the algorithms and models that make this technology possible. In the realm of machine learning, data serves as both the raw material and the compass. It provides the necessary information for algorithms to learn patterns, make predictions, and drive decision-making processes. 

The quality, quantity, and relevance of the data directly impact the performance and accuracy of machine learning systems. Through data, machines can recognize trends, identify anomalies, and adapt to changing circumstances. 

Moreover, data is not a static component but an ever-evolving entity that requires constant curation and refinement to ensure the continued efficacy of machine learning models. In essence, data is the lifeblood of machine learning, the crucial key that unlocks its potential to transform industries, solve complex problems, and enhance our understanding of the world.

### Template Usage:
The template provides a wrapper on top of a torch Dataset. It provides a very basic data interface.

The user is responsible to create their own dataset and load it in PyTorch.

**Wrapper**

The `Data` class inherits from the `torch.utils.data.Dataset` class. It implements the `__len__` and the `__getitem__` functions. The user must implement the `get_input`, `get_label` and `apply_transform` functions, which define how the input and the labels can be obtained from a `pd.core.frame.DataFrame`

**Config**

The `Data` configuration consists of the `root_dir` to specify where the images are present. Then it consists of a `dataloader` where you can specify the kwargs for `torch.utils.data.DataLoader`

## Model
A machine learning model is an object (stored locally in a file) that has been trained to recognize certain types of patterns. You train a model over a set of data, providing it an algorithm that it can use to reason over and learn from those data.

Once you have trained the model, you can use it to reason over data that it hasn't seen before, and make predictions about those data. For example, let's say you want to build an application that can recognize a user's emotions based on their facial expressions. You can train a model by providing it with images of faces that are each tagged with a certain emotion, and then you can use that model in an application that can recognize any user's emotion.

### Template usage
The template provides a wrapper on top of a torch model. The template provides the usage of 3 different libraries: 
- timm
    - https://github.com/huggingface/pytorch-image-models/tree/main
    - comprehensive list of encoders or classification models,
    - Ex: Resnet, ViT, Mobilenet, etc
- segmentation_models_pytorch 
    - https://github.com/qubvel-org/segmentation_models.pytorch
    - comprehensive list of segmentation models
    - Ex: DeeplabV3, FCN, LinkNet, etc
- detectron2
    - https://github.com/facebookresearch/detectron2/tree/main
    - comprehensive list of object detection models.
    - Ex: FasterRCNN, MaskRCNN, Keypoint RCNN, etc
- Custom Models
    - The template also supports custom PyTorch models that are created by the students.

The Model can be loaded from the respective library, or created custom in the notebook.

**Config**

The `Model` configuration consists of the `model_hyperparameters` to specify the parameters of the model. Then it consists of a `weights_path` where you can load pretrained model_weights from. Finally a `freeze_layer` which specifies the layers you want to freeze (perform no backpropagation on)

## Model Training
After a data scientist has preprocessed the collected data and split it into three subsets, he or she can proceed with a model training. This process entails “feeding” the algorithm with training data. An algorithm will process data and output a model that is able to find a target value (attribute) in new data — an answer you want to get with predictive analysis. The purpose of model training is to develop a model.

Two model training styles are most common — supervised and unsupervised learning. The choice of each style depends on whether you must forecast specific attributes or group data objects by similarities.

**Supervised learning.** Supervised learning allows for processing data with target attributes or labeled data. These attributes are mapped in historical data before the training begins. With supervised learning, a data scientist can solve classification and regression problems.

**Unsupervised learning.** During this training style, an algorithm analyzes unlabeled data. The goal of model training is to find hidden interconnections between data objects and structure objects by similarities or differences. Unsupervised learning aims at solving such problems as clustering, association rule learning, and dimensionality reduction. For instance, it can be applied at the data preprocessing stage to reduce data complexity.

### Template Usage
The template provides you with a `ModelEngine` class. This class is responsible for supervised model training and evaluation. The code for training or evaluating the model is generally the same for every model. The components that change are the `Loss`, `Metric` and `Optimizer`.
- Loss
    - The loss function can be imported from any library
    - It can also be a custom loss function as seen [here](https://discuss.pytorch.org/t/custom-loss-functions/29387/2)
- Metric
    - Any metric can be used from the library `torchmetrics`.
    - Options for a custom metric are also provided [here](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html)
- Optimizer
    - The optimizer must be from the `torch.optim` module.
    - Any optimizer can be selected
    - Options for a custom optimizer are also provided on the docs

## Model Evaluation
Evaluation is a inherent part of the key elements of machine learning, acting as the yardstick by which models' effectiveness and performance are judged. It is crucial to carefully assess how well models generalize from training data to new or upcoming data in the quest to create reliable and accurate models. 

Various metrics and approaches are used in this evaluation, depending on the particular issue and the type of data. Accuracy, precision, recall, F1-score, and mean squared error are examples of common evaluation measures. 

These metrics give data scientists and machine learning professionals a measurable way to assess a model's performance, enabling them to compare various algorithms, hone hyperparameters, and make sure that models satisfy the required standards for success.

Furthermore, evaluation is a continuous process that includes testing models against actual data, keeping tabs on how they perform in use, and adapting them to changing conditions. Furthermore, it aids in the detection and mitigation of problems with overfitting, underfitting, and bias in models, assuring their fairness and dependability.

### Template Usage
Same as Model Training


# Template Usage


