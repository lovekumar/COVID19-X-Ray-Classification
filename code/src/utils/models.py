from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from sparkdl import DeepImageFeaturizer


def lr_model(dataset, feature_model):
    '''
    This function is used for model training. DeepImageFeaturizer extracts the features using pretrained model.
    LogisticRegression is doing binary/multiclass classification.

    :param dataset: input data set
    :param feature_model: Name of pretrained model to be used.
                          Possible Values - [InceptionV3, ResNet50, VGG16, VGG19, Xception]
    :return: Trained Model
    '''

    # extracting feature from images
    dif = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName=feature_model)

    # LogisticRegression model
    lr = LogisticRegression(labelCol="label", featuresCol="features")

    # Pipeline [DeepImageFeaturizer, LogisticRegression]
    pipeline = Pipeline(stages=[dif, lr])

    # Create ParamGrid for Cross Validation
    param_grid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.03, 0.5])
                 .addGrid(lr.elasticNetParam, [0.5])
                 .addGrid(lr.maxIter, [5])
                 .build())

    # MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator()

    # Create 3-fold CrossValidator
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=param_grid,
                        evaluator=evaluator,
                        numFolds=3,
                        seed=6250,
                        parallelism=3)

    # Run cross validations
    spark_model = cv.fit(dataset)

    return spark_model
