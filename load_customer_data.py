from pyspark.sql import SparkSession

def load_data(file_path):
    spark = SparkSession.builder.appName("CustomerSegmentationDataLoad").getOrCreate()
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    return data
