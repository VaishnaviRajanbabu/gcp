# Databricks notebook source
# MAGIC
%pip install faker

# COMMAND ----------

import random

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, rand, when
from faker import Faker

# Initialize Spark session
spark = SparkSession.builder.appName("HealthAnalysis").getOrCreate()

# Initialize Faker for generating random data
fake = Faker()

# Generate 10000 rows of data
data = []
for _ in range(10000):
    timestamp = fake.date_time_this_year()
    patient_id = fake.unique.random_number(digits=6)
    gender = random.choice(["Male", "Female", "Other"])
    weight = round(random.uniform(40, 120), 2)
    height = round(random.uniform(140, 200), 2)
    heart_rate = random.choice([random.randint(50, 100), None])
    data.append((timestamp, patient_id, gender, weight, height, heart_rate))

# Add duplicate rows
duplicate_rows = random.choices(data, k=50)
data.extend(duplicate_rows)

# Create a DataFrame from the data
columns = ["timestamp", "patient_id", "gender", "weight", "height", "heart_rate"]
df = spark.createDataFrame(data, columns)

# Save the DataFrame to a CSV file
df.coalesce(1).write.csv("dbfs:/FileStore/health_analysis.csv", header=True, mode="overwrite")

print("health_analysis.csv with 10000 lines of data has been created successfully.")

# COMMAND ----------

df.show()
df.display()

# COMMAND ----------

df.count()

# COMMAND ----------

#drop rows with null values in heart_rate column
df_cleaned = df.na.drop(subset=["heart_rate"]).show()

# COMMAND ----------

# Remove noise words from the 'gender' column
df_cleaned1 = df.withColumn("gender", when(col("gender") == "Other", None).otherwise(col("gender")))
df_cleaned1.show()

# COMMAND ----------

#This function will remove rows with any null values in any column.
df_cleaned2 = df_cleaned1.na.drop(how='any').show()

# COMMAND ----------

# Remove rows with "male" and "female" values in the "gender" column
df_o = df.withColumn("gender", when(col("gender") != "Other", None).otherwise(col("gender")))
df_o.show()

# COMMAND ----------

#This function Will contain only rows where at least one column has a non-null value
df_other=df_o.na.drop(how='all').show()

# COMMAND ----------

# Remove rows with null values in the "gender" column for "Other"
df_o1 = df_o.filter(col("gender").isNotNull())
# Show the first few rows of the filtered DataFrame
df_o1.show()

# COMMAND ----------

# Fill missing heart rate values with the mean heart rate
mean_heart_rate = df.select("heart_rate").agg({"heart_rate": "mean"}).collect()[0][0]
df_mean_heart_rate =df.fillna(mean_heart_rate, subset=["heart_rate"])

# COMMAND ----------

df_mean_heart_rate.show()

# COMMAND ----------

# Remove rows with null values in the "gender" column
df_filtered = df_cleaned1.filter(col("gender").isNotNull())
# Show the first few rows of the filtered DataFrame
df_filtered.show()

# COMMAND ----------

# Calculate BMI (Body Mass Index) using height and weight columns
df_with_bmi = df_filtered.withColumn("bmi", (col("weight") / (col("height") / 100) ** 2))

# COMMAND ----------

# Show the first few rows of the DataFrame with the new "bmi" column
df_with_bmi.show()

# COMMAND ----------

# Round off the "bmi" column values to 2 decimal places
from pyspark.sql.functions import round
df_with_bmi_rounded = df_with_bmi.withColumn("bmi", round(col("bmi"), 2))
df_with_bmi_rounded.show()

# COMMAND ----------

# Define BMI thresholds for weight status
underweight_threshold = 18.5
normal_weight_threshold = 24.9
overweight_threshold = 29.9

# COMMAND ----------

# Categorize weight status based on BMI
df_with_weight_status = df_with_bmi_rounded.withColumn(
    "weight_status",
    when(col("bmi") < underweight_threshold, "Underweight")
    .when((col("bmi") >= underweight_threshold) & (col("bmi") <= normal_weight_threshold), "Normal weight")
    .when((col("bmi") > normal_weight_threshold) & (col("bmi") <= overweight_threshold), "Overweight")
    .otherwise("Obesity")
)

# COMMAND ----------

# Show the first few rows of the DataFrame with weight status
df_with_weight_status.select("patient_id","weight","height","bmi","weight_status").show()

# COMMAND ----------

# Generate random blood pressure values for the DataFrame
df_with_blood_pressure = df_with_weight_status.withColumn("blood_pressure", rand() * 50 + 80)

# COMMAND ----------

df_with_blood_pressure.show()

# COMMAND ----------

# Convert decimal blood pressure values to integers
df_with_bp_integer = df_with_blood_pressure.withColumn("blood_pressure_int", col("blood_pressure").cast("int"))
# Show the first few rows of the DataFrame with integer blood pressure values
df_with_bp_integer.show()

# COMMAND ----------

# Drop the 'blood_pressure' column
df_with_bp_integer1 = df_with_bp_integer.drop("blood_pressure")
# Show the first few rows of the DataFrame without the dropped column
df_with_bp_integer1.show()

# COMMAND ----------

# Define blood pressure thresholds for categorization
normal_bp_threshold = 80
elevated_bp_threshold = 120
high_bp_stage1_threshold = 130
high_bp_stage2_threshold = 140

# COMMAND ----------

# Categorize blood pressure based on the thresholds
df_with_bp_category = df_with_bp_integer1.withColumn(
    "bp_category",
    when(col("blood_pressure_int") < normal_bp_threshold, "Low")
    .when((col("blood_pressure_int") >= normal_bp_threshold) & (col("blood_pressure_int") < elevated_bp_threshold), "Normal")
    .when((col("blood_pressure_int") >= elevated_bp_threshold) & (col("blood_pressure_int") < high_bp_stage1_threshold), "Elevated")
    .when((col("blood_pressure_int") >= high_bp_stage1_threshold) & (col("blood_pressure_int") < high_bp_stage2_threshold), "High (Stage 1)")
    .otherwise("High (Stage 2 or higher)")
)

# COMMAND ----------

# Show the first few rows of the DataFrame with bp_category column
df_with_bp_category.select("patient_id","blood_pressure_int","bp_category").show()

# COMMAND ----------

# Define heart rate thresholds for categorization
low_hr_threshold = 60
normal_hr_threshold = 100

# COMMAND ----------

# Categorize heart rate based on the thresholds
df_with_hr_category = df_with_bp_category.withColumn(
    "heart_rate_category",
    when(col("heart_rate").isNull(), "Unknown")
    .when(col("heart_rate") < low_hr_threshold, "Low")
    .when((col("heart_rate") >= low_hr_threshold) & (col("heart_rate") <= normal_hr_threshold), "Normal")
    .otherwise("Elevated")
)

# COMMAND ----------

# Show the first few rows of the DataFrame with heart rate category
df_with_hr_category.select("patient_id","heart_rate","heart_rate_category").show()

# COMMAND ----------

#Column has to have at least 10 non null values otherwise it will get dropped
df_thresh = df_with_hr_category.na.drop(thresh=10).show()

# COMMAND ----------

#removes duplicate rows based on all columns
df_f = df_with_hr_category.dropDuplicates().show()

# COMMAND ----------

#Result
df_final=df_with_hr_category.na.drop(how='any').show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# COMMAND ----------

df = spark.read.csv("/health_analysis.csv", header=True, inferSchema=True)

# COMMAND ----------

# Create a Pandas DataFrame for visualization
pandas_df = df_with_bp_category.select("weight", "height", "bmi", "weight_status").toPandas()

# Bar plot of weight status
plt.figure(figsize=(8, 6))
sns.countplot(x="weight_status", data=pandas_df, order=["Underweight", "Normal", "Overweight", "Obese"])
plt.xlabel("Weight Status")
plt.ylabel("Count")
plt.title("Distribution of Weight Status")
plt.show()

# COMMAND ----------

# Create a Pandas DataFrame for visualization
pandas_bp_df = df_with_bp_category.select("bp_category").toPandas()

# Bar plot of blood pressure category
plt.figure(figsize=(8, 6))
sns.countplot(x="bp_category", data=pandas_bp_df, order=["Low", "Normal", "Elevated", "High(Stage1)", "High(Stage2 or higher)"])
plt.xlabel("Blood Pressure Category")
plt.ylabel("Count")
plt.title("Distribution of Blood Pressure Categories")
plt.show()

# COMMAND ----------

# Create a Pandas DataFrame for visualization
pandas_hr_df = df_with_hr_category.select("heart_rate", "heart_rate_category").toPandas()

# Bar plot of heart rate category
plt.figure(figsize=(8, 6))
sns.countplot(x="heart_rate_category", data=pandas_hr_df, order=["Low","Normal", "Elevated"])
plt.xlabel("Heart Rate Category")
plt.ylabel("Count")
plt.title("Distribution of Heart Rate Categories")
plt.show()
