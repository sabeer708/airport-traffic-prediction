# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# -------------------------------
# START SPARK SESSION
# -------------------------------
spark = SparkSession.builder.appName("Airport Traffic Prediction").getOrCreate()

print("\n===== AIRPORT TRAFFIC PREDICTION =====\n")

# -------------------------------
# LOAD DATASET
# -------------------------------
df = spark.read.csv("air_traffic_data.csv", header=True, inferSchema=True)

print("Dataset Columns:")
print(df.columns)

print("\nSample Data:")
df.show(5, truncate=False)

# -------------------------------
# DATA CLEANING
# -------------------------------
df = df.dropna(subset=["Passenger Count"])
df = df.withColumn("Passenger Count", col("Passenger Count").cast("int"))
df = df.withColumn("Year", col("Year").cast("int"))

print("\nData after cleaning:")
df.select("Passenger Count", "Year", "Month").show(5)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df = df.withColumn("Month_num",
    when(col("Month")=="January",1)
    .when(col("Month")=="February",2)
    .when(col("Month")=="March",3)
    .when(col("Month")=="April",4)
    .when(col("Month")=="May",5)
    .when(col("Month")=="June",6)
    .when(col("Month")=="July",7)
    .when(col("Month")=="August",8)
    .when(col("Month")=="September",9)
    .when(col("Month")=="October",10)
    .when(col("Month")=="November",11)
    .when(col("Month")=="December",12)
)

# Encode Airline
indexer = StringIndexer(inputCol="Operating Airline", outputCol="airline_index")
df = indexer.fit(df).transform(df)

print("\nFeature Engineered Data:")
df.select("Year", "Month_num", "airline_index").show(5)

# -------------------------------
# FEATURE VECTOR
# -------------------------------
assembler = VectorAssembler(
    inputCols=["Year", "Month_num", "airline_index"],
    outputCol="features"
)

data = assembler.transform(df)

# -------------------------------
# TRAIN MODEL
# -------------------------------
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(
    featuresCol="features",
    labelCol="Passenger Count"
)

model = lr.fit(train_data)

print("\nModel Training Completed")

# -------------------------------
# PREDICTIONS
# -------------------------------
predictions = model.transform(test_data)

print("\n===== PREDICTION RESULTS =====")

# Clean table
predictions.select(
    col("Passenger Count").alias("Actual"),
    col("prediction").alias("Predicted")
).show(20, truncate=False)

# -------------------------------
# EVALUATION
# -------------------------------
evaluator_rmse = RegressionEvaluator(
    labelCol="Passenger Count",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="Passenger Count",
    predictionCol="prediction",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n===== MODEL PERFORMANCE =====")
print("RMSE (Error):", round(rmse, 2))
print("R2 Score (Accuracy):", round(r2, 4))

# -------------------------------
# VISUALIZATION
# -------------------------------
pdf_pred = predictions.select("Passenger Count", "prediction").toPandas()

print("\nSample Prediction Table:")
print(pdf_pred.head(10))

# Add error column
pdf_pred["Error"] = pdf_pred["Passenger Count"] - pdf_pred["prediction"]

print("\nWith Error:")
print(pdf_pred.head(10))

# Plot Actual vs Predicted
plt.figure()
plt.plot(pdf_pred["Passenger Count"].values[:50], label="Actual")
plt.plot(pdf_pred["prediction"].values[:50], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Passenger Traffic")
plt.xlabel("Samples")
plt.ylabel("Passengers")
plt.show()
