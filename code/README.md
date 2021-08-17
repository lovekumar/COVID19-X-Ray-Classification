
## Experimental Setup

Our code is structured in a pySpark based project for model training using DeepImageFeaturizer libraries from Databricks. 
We are using the anaconda and [environment.yml](./environment.yml) has all the dependencies listed for creating python(=3.7) conda environments needed for the model training. 
The pipeline is tested locally on Macbook (2.4 GHz, 8-Core Intel Core i9, 16 GB RAM) using spark in local mode. 
We also tested the job against a local standalone spark cluster. 
The application uses [X-ray Image data](./dataset_resized/) stored on a local filesystem to train and test a deep learning model for both binary and multi-class classification of COVID conditions.

[wrapper.sh](./wrapper.sh) helps in setting up the environment as well as running the model training jobs

### 1. Setup Conda environment
First step is to setup conda environment.

You can setup conda environment either using
```
conda env create -f environment.yml
```
or using
```
sh wrapper.sh env-setup
```
<br>

### 2. Activate Conda environment
Next step is to activate the conda environment. 
In [environment.yml](./environment.yml), we are using **team11-pyspark** as conda environment name, so
```
conda activate team11-pyspark
```
<br>

### 3. Run Job

We are using [Deep Learning Pipeline - v1.5.0](https://github.com/databricks/spark-deep-learning/tree/v1.5.0) for model training.

Our model training pipeline looks like this 
```
# Pipeline [DeepImageFeaturizer, LogisticRegression]
pipeline = Pipeline(stages=[dif, lr])
``` 

DeepImageFeaturizer supports the following models from Keras:
* InceptionV3
* Xception
* ResNet50
* VGG16
* VGG19

For model training, instead of passing too many parameters on command line, we are reading them from [config.yaml](./config.yaml)

#### Model Training : Binary Classification
```
# Using Model: InceptionV3 
sh wrapper.sh run InceptionV3_binary_classification

# Using Model: ResNet50
sh wrapper.sh run ResNet50_binary_classification

# Using Model: VGG16
sh wrapper.sh run VGG16_binary_classification

# Using Model: VGG19
sh wrapper.sh run VGG19_binary_classification

# Using Model: Xception
sh wrapper.sh run Xception_binary_classification
```

#### Model Training : Multi-class Classification
```
# Using Model: InceptionV3 
sh wrapper.sh run InceptionV3_multiclass_classification

# Using Model: ResNet50
sh wrapper.sh run ResNet50_multiclass_classification

# Using Model: VGG16
sh wrapper.sh run VGG16_multiclass_classification

# Using Model: VGG19
sh wrapper.sh run VGG19_multiclass_classification

# Using Model: Xception
sh wrapper.sh run Xception_multiclass_classification
```

#### Resize Images 
```
# You may need to change the output location in config.yaml
sh wrapper.sh run dataset_resize
```

