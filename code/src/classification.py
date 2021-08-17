
from utils.plots import plot_confusion_matrix, evaluate_model
from utils.datasets import load_dataset
from utils.models import lr_model


def run(spark, job_config_map):
    '''
    This is common function for classification (both binary as well as multi-class classification)

    :param spark: SparkSession object
    :param job_config_map: map for passing multiple parameters to the job
        It contains
            dataset_path     : location of the xray images

            folders          : List of names of folders to be loaded for classification.
                               If there are 2 folders, then it does binary classification
                               If there are more than 2 folders, then it does multi-class classification

            confusion_matrix : output location for saving confusion_matrix

            feature_model    : Name of pretrained model to be used.
                               Possible Values - [InceptionV3, ResNet50, VGG16, VGG19, Xception]

    :return: None
    '''

    dataset_path = job_config_map['dataset_path']
    class_names = job_config_map['folders']
    output_png = job_config_map['confusion_matrix']
    feature_model = job_config_map['feature_model']

    # Load Data
    folder = {i: class_names[i] for i in range(0, len(class_names))}
    dataframe = load_dataset(spark, dataset_path, folder)

    # Split the input data to train_df and test_df
    train_df, test_df = dataframe.randomSplit([0.8, 0.2], seed=6250)

    # Print the stats for the input data
    for key, value in folder.items():
        print("{} ---> {}".format(key,value))
    print('Train Events : ')
    print(train_df.select('label').groupBy("label").count().show())
    print('Test Events : ')
    print(test_df.select('label').groupBy("label").count().show())

    # Get Trained Model
    spark_model = lr_model(train_df, feature_model)

    # predict Test data
    pred_df = spark_model.transform(test_df).select("label", "prediction")

    pred = pred_df.toPandas()
    test_results = pred.apply(tuple, axis=1)

    # evaluate the model with test set
    acc, f1, recall, precision = evaluate_model(test_results)
    print('output_png: {}, accuracy: {}, f1: {}, recall: {}, precision:{}'
          .format(output_png, acc, f1, recall, precision))

    # Save Confusion Matrix
    plot_confusion_matrix(test_results, class_names, output_png)
