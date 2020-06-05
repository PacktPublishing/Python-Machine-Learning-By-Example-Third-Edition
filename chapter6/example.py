'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 6  Scaling Up Prediction to Terabyte Click Logs
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


from pyspark.sql import SparkSession
spark = SparkSession \
            .builder \
            .appName("test") \
            .getOrCreate()


df = spark.read.csv("examples/src/main/resources/people.csv", header=True, sep=';')
df.show()

df.count()

df.printSchema()

df.select("name").show()
df.select(["name", "job"]).show()


df.filter(df['age'] > 31).show()



from pyspark.sql.functions import monotonically_increasing_id
df.withColumn('index', monotonically_increasing_id())
