'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 6  Scaling Up Prediction to Terabyte Click Logs
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

from pyspark.sql import SparkSession


spark = SparkSession\
    .builder\
    .appName("CTR")\
    .getOrCreate()



from pyspark.sql.types import StructField, StringType, StructType, IntegerType

schema = StructType([
    StructField("id", StringType(), True),
    StructField("click", IntegerType(), True),
    StructField("hour", IntegerType(), True),
    StructField("C1", StringType(), True),
    StructField("banner_pos", StringType(), True),
    StructField("site_id", StringType(), True),
    StructField("site_domain", StringType(), True),
    StructField("site_category", StringType(), True),
    StructField("app_id", StringType(), True),
    StructField("app_domain", StringType(), True),
    StructField("app_category", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("device_ip", StringType(), True),
    StructField("device_model", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("device_conn_type", StringType(), True),
    StructField("C14", StringType(), True),
    StructField("C15", StringType(), True),
    StructField("C16", StringType(), True),
    StructField("C17", StringType(), True),
    StructField("C18", StringType(), True),
    StructField("C19", StringType(), True),
    StructField("C20", StringType(), True),
    StructField("C21", StringType(), True),
])



df = spark.read.csv("file://path_to_file/train", schema=schema, header=True)


df.printSchema()

df.count()

df = df.drop('id').drop('hour').drop('device_id').drop('device_ip')

df = df.withColumnRenamed("click", "label")

df.columns


df_train, df_test = df.randomSplit([0.7, 0.3], 42)

df_train.cache()
df_train.count()

df_test.cache()
df_test.count()



categorical = df_train.columns
categorical.remove('label')
print(categorical)





from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator




indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)).setHandleInvalid("keep")
    for c in categorical
]

encoder = OneHotEncoderEstimator(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=[
        "{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]
)

assembler = VectorAssembler(
    inputCols=encoder.getOutputCols(),
    outputCol="features"
)

stages = indexers + [encoder, assembler]

from pyspark.ml import Pipeline


pipeline = Pipeline(stages=stages)


one_hot_encoder = pipeline.fit(df_train)


df_train_encoded = one_hot_encoder.transform(df_train)


df_train_encoded.show()

df_train_encoded = df_train_encoded.select(["label", "features"])

df_train_encoded.show()

df_train_encoded.cache()

df_train.unpersist()



df_test_encoded = one_hot_encoder.transform(df_test)



df_test_encoded = df_test_encoded.select(["label", "features"])

df_test_encoded.show()

df_test_encoded.cache()

df_test.unpersist()



from pyspark.ml.classification import LogisticRegression

classifier = LogisticRegression(maxIter=20, regParam=0.000, elasticNetParam=0.000)

lr_model = classifier.fit(df_train_encoded)


df_train_encoded.unpersist()

predictions = lr_model.transform(df_test_encoded)

df_test_encoded.unpersist()

predictions.cache()

predictions.show()


from pyspark.ml.evaluation import BinaryClassificationEvaluator

ev = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", metricName = "areaUnderROC")
print(ev.evaluate(predictions))


spark.stop()
