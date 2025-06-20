# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Start Spark session
spark = SparkSession.builder.appName('AirQualityPrediction').getOrCreate()

# Load the dataset
data = spark.read.csv('dataset.csv', header=True, inferSchema=True)

# Drop the 'Datetime' column
data = data.drop('Datetime')

# Label encoding for 'City' column using StringIndexer
city_indexer = StringIndexer(inputCol="City", outputCol="City_Index")

# Label encoding for 'AQI_Bucket' column using StringIndexer (target column)
aqi_indexer = StringIndexer(inputCol="AQI_Bucket", outputCol="AQI_Bucket_Index")

# Assemble features (except the target and encoded 'City')
feature_columns = [col for col in data.columns if col not in ['City', 'AQI_Bucket', 'AQI_Bucket_Index']]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# RandomForestClassifier
rf = RandomForestClassifier(labelCol="AQI_Bucket_Index", featuresCol="features")

# Create pipeline with indexers, assembler, and classifier
pipeline = Pipeline(stages=[city_indexer, aqi_indexer, assembler, rf])

# Split the data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Save the entire model pipeline
model.save("model/air_quality_pipeline_model2")

# Save individual encoders (StringIndexerModels)
city_indexer_model = model.stages[0]  # City StringIndexerModel
city_indexer_model.save("model/city_indexer_model2")

aqi_indexer_model = model.stages[1]  # AQI_Bucket StringIndexerModel
aqi_indexer_model.save("model/aqi_indexer_model2")

print("Model and encoders saved successfully.")

# Make predictions
predictions = model.transform(test_data)

# Convert predictions to Pandas DataFrame for visualization
predictions_pd = predictions.select("City", "AQI_Bucket", "prediction").toPandas()
data_pd = data.toPandas()

# Evaluate the model using accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="AQI_Bucket_Index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Visualization 1: AQI Bucket Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=data_pd, x="AQI_Bucket", order=data_pd["AQI_Bucket"].value_counts().index)
plt.title("AQI Bucket Distribution")
plt.xticks(rotation=45)
plt.show()

# Visualization 2: Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Extract feature importance from the trained model
rf_model = model.stages[-1]  # Last stage of pipeline is the RandomForestClassifier
feature_importance = rf_model.featureImportances

# Convert feature importance to a list
feature_importance = feature_importance.toArray()

# Ensure feature_columns and feature_importance are of the same length
if len(feature_columns) != len(feature_importance):
    print("Mismatch in feature importance and feature columns length")
    print(f"Feature Columns: {len(feature_columns)}, Feature Importance: {len(feature_importance)}")

# Convert to DataFrame for Seaborn
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Check for invalid types
print(feature_importance_df.dtypes)  # Debugging step

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance in RandomForest Model")
plt.show()


# Visualization 3: City-wise Average AQI
city_avg_aqi = data_pd.groupby("City")["PM2_5"].mean().sort_values()
plt.figure(figsize=(12,6))
city_avg_aqi.plot(kind='bar', color='b')
plt.title("City-wise Average PM2.5 Levels")
plt.show()

# Visualization 4: AQI vs PM2.5
plt.figure(figsize=(8,5))
sns.scatterplot(x=data_pd["PM2_5"], y=data_pd["AQI_Bucket"], alpha=0.7)
plt.title("AQI vs PM2.5")
plt.show()

# Visualization 5: AQI vs NO2
plt.figure(figsize=(8,5))
sns.scatterplot(x=data_pd["NO2"], y=data_pd["AQI_Bucket"], alpha=0.7)
plt.title("AQI vs NO2")
plt.show()

# Visualization 6: Confusion Matrix
from pyspark.ml.feature import IndexToString
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Convert predictions back to original labels
label_converter = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=aqi_indexer_model.labels)
predictions = label_converter.transform(predictions)

# Convert to Pandas DataFrame
predictions_pd = predictions.select("AQI_Bucket", "predicted_label").toPandas()


# Compute confusion matrix
cm = confusion_matrix(predictions_pd["AQI_Bucket"], predictions_pd["predicted_label"])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=aqi_indexer_model.labels, yticklabels=aqi_indexer_model.labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Visualization 7: Feature Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas
data_pd = data.toPandas()

# Drop categorical columns before computing correlation
non_numeric_columns = ["City", "AQI_Bucket"]  # Add more if needed
numeric_data = data_pd.drop(columns=non_numeric_columns, errors="ignore")

# Compute and plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

