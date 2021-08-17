from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType


def load_dataset(spark, base_path, folder):
    '''
    Reads the directory of images from the local or remote source.

    :param spark: SparkSession object
    :param base_path: base location of the xray images
    :param folder:  List of names of the folders to be loaded for classification.
    :return: `DataFrame` with columns of ("images", label)
    '''

    df = spark.createDataFrame(spark.sparkContext.emptyRDD(), StructType([]))

    # For each label and folder name in list of folders, read the images
    for key, value in folder.items():
        l_path = base_path + '/' + value
        df1 = ImageSchema.readImages(l_path, recursive=False, seed=6250).withColumn("label", lit(key))
        if df.count() == 0:
            df = df1
        else:
            df = df.union(df1)

    # repartition
    dataframe = df.repartition(8)

    return dataframe
