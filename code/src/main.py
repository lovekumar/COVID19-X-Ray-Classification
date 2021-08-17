import argparse
import os
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import importlib
import sys
import yaml

if os.path.exists('src.zip'):
    sys.path.insert(0, 'src.zip')

'''
This is the main class to start the model training. 
Input Parameters
    --job : The name of the spark job you want to run

Based upon the Input Parameters, it reads the configs from config.yaml
    "python_class" : Name of the .py file which will be run by pyspark Job
    "job_config_map": 
            dataset_path     : location of the xray images
            folders          : Array of folders to be used
            confusion_matrix : output location for confusion_matrix
            feature_model    : Name of pretrained model to be used.
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='pySpark Job Arguments')

    parser.add_argument('--job', type=str, required=True, dest='model_name',
                        help='The name of the spark job you want to run')

    args = parser.parse_args()

    # read config.yaml
    with open('config.yaml', 'r') as f:
        all_config = yaml.safe_load(f)

    # spark parameters
    master_url = all_config['SPARK_MASTER']['master.url']
    spark_configs = all_config['SPARK_CONFIG']
    spark_log_level = all_config['SPARK_LOG_LEVEL']

    # read configs for the job
    config = all_config[args.model_name]
    python_class = config['python_class']
    job_config_map = config['job_config_map']

    # Start Spark Job
    conf = SparkConf().setMaster(master_url).setAppName(args.model_name)
    spark_builder = SparkSession.builder.config(conf=conf)
    for key, val in spark_configs.items():
        spark_builder.config(key, val)

    spark = spark_builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_log_level)

    # Run the model training
    job_module = importlib.import_module('%s' % python_class)
    res = job_module.run(spark, job_config_map)

