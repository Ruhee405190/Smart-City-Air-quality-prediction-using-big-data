from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString, StringIndexerModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Start Spark session
spark = SparkSession.builder.appName('AirQualityPrediction').getOrCreate()

# Load the saved model
model = PipelineModel.load("model/air_quality_pipeline_model")

# Load the AQI_Bucket StringIndexerModel
aqi_indexer_model = StringIndexerModel.load("model/aqi_indexer_model")

# Read data from CSV
csv_file_path = "dataset.csv"  # Specify the correct path to your CSV file
df = pd.read_csv(csv_file_path)

# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Example: Define the single input for prediction
single_input = {
    "City": "Hyderabad",
    "PM2_5": 10.5,
    "PM10": 32.75,
    "NO": 2.83,
    "NO2": 12.8,
    "NOx": 9.11,
    "NH3": 5.74,
    "CO": 0.26,
    "SO2": 4.14,
    "O3": 17.09,
    "Benzene": 0.03,
    "Toluene": 0.31,
    "Xylene": 0
}

# Convert the dictionary to a Spark DataFrame for single input prediction
single_input_df = spark.createDataFrame([single_input])

# Make prediction for single input
predicted_single_input = model.transform(single_input_df)

# Create IndexToString to decode the prediction for single input
index_to_string_single_input = IndexToString(inputCol="prediction", outputCol="AQI_Bucket_Predicted", labels=aqi_indexer_model.labels)

# Decode the prediction
decoded_single_input = index_to_string_single_input.transform(predicted_single_input)

# Extract the predicted label for single input
predicted_single_label = decoded_single_input.select("AQI_Bucket_Predicted").first()["AQI_Bucket_Predicted"]

# Print the result for single input
print(f"The predicted AQI Bucket for the single input is: {predicted_single_label}")

# Now, make predictions for all rows in the dataset
predicted_output = model.transform(spark_df)

# Create IndexToString to decode the prediction for the entire dataset
index_to_string = IndexToString(inputCol="prediction", outputCol="AQI_Bucket_Predicted", labels=aqi_indexer_model.labels)

# Decode the prediction
decoded_output = index_to_string.transform(predicted_output)

# Extract the predicted labels as a list
predicted_labels = [row["AQI_Bucket_Predicted"] for row in decoded_output.select("AQI_Bucket_Predicted").collect()]

# Extract the actual labels from the dataframe
actual_labels = df['AQI_Bucket'].tolist()

# Ensure both predicted and actual labels have the same length
assert len(actual_labels) == len(predicted_labels), "Actual and predicted labels do not have the same length!"

# Compute the confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=df['AQI_Bucket'].unique())

# Display the confusion matrix
cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=df['AQI_Bucket'].unique())
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix: Actual vs Predicted AQI Buckets')
plt.show()

# Now, plot using the CSV data

# Plotting AQI Bucket Distribution (Bar Chart)
plt.figure(figsize=(8, 6))
sns.countplot(x='AQI_Bucket', data=df, palette="Set2")
plt.title('AQI Bucket Distribution')
plt.xlabel('AQI Bucket')
plt.ylabel('Count')
plt.show()

# Scatter Plot: PM2_5 vs AQI Bucket
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PM2_5', y='AQI_Bucket', data=df, hue='AQI_Bucket', palette='Set1')
plt.title('PM2_5 vs AQI Bucket')
plt.xlabel('PM2_5 Concentration')
plt.ylabel('AQI Bucket')
plt.show()

# Box Plot: PM2_5 vs AQI Bucket
plt.figure(figsize=(8, 6))
sns.boxplot(x='AQI_Bucket', y='PM2_5', data=df, palette='Set3')
plt.title('PM2_5 Concentration Distribution by AQI Bucket')
plt.xlabel('AQI Bucket')
plt.ylabel('PM2_5 Concentration')
plt.show()

# Correlation Matrix: Show correlations between pollutants
plt.figure(figsize=(10, 8))
corr = df[['PM2_5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Pollutants')
plt.show()

# Pollutant Contribution to AQI Bucket (Bar Chart)
plt.figure(figsize=(8, 6))
sns.barplot(x=['PM2_5', 'NO', 'NOx', 'CO', 'SO2'], y=[10.5, 2.83, 9.11, 0.26, 4.14], palette='viridis')
plt.title('Pollutant Contribution to AQI Bucket')
plt.xlabel('Pollutant')
plt.ylabel('Concentration')
plt.show()
