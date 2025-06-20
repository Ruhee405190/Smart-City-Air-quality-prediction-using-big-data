from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString, StringIndexerModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Initialize Flask app
app = Flask(__name__)

# Start Spark session
spark = SparkSession.builder.appName('AirQualityPrediction').getOrCreate()

# Load the saved model
model = PipelineModel.load("model/air_quality_pipeline_model")

# Load the AQI_Bucket StringIndexerModel
aqi_indexer_model = StringIndexerModel.load("model/aqi_indexer_model")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from form
    input_data = {
        "City": request.form['city'],
        "PM2_5": float(request.form['pm25']),
        "PM10": float(request.form['pm10']),
        "NO": float(request.form['no']),
        "NO2": float(request.form['no2']),
        "NOx": float(request.form['nox']),
        "NH3": float(request.form['nh3']),
        "CO": float(request.form['co']),
        "SO2": float(request.form['so2']),
        "O3": float(request.form['o3']),
        "Benzene": float(request.form['benzene']),
        "Toluene": float(request.form['toluene']),
        "Xylene": float(request.form['xylene'])
    }

    # Convert input data to Spark DataFrame
    single_input_df = spark.createDataFrame([input_data])

    # Predict AQI bucket
    predicted_single_input = model.transform(single_input_df)

    # Decode the prediction
    index_to_string_single_input = IndexToString(inputCol="prediction", outputCol="AQI_Bucket_Predicted", labels=aqi_indexer_model.labels)
    decoded_single_input = index_to_string_single_input.transform(predicted_single_input)
    predicted_single_label = decoded_single_input.select("AQI_Bucket_Predicted").first()["AQI_Bucket_Predicted"]

    # Load dataset
    df = pd.read_csv("dataset.csv")

    # Convert dataset to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # Predict for entire dataset
    predicted_output = model.transform(spark_df)
    index_to_string = IndexToString(inputCol="prediction", outputCol="AQI_Bucket_Predicted", labels=aqi_indexer_model.labels)
    decoded_output = index_to_string.transform(predicted_output)
    
    # Extract predictions
    predicted_labels = [row["AQI_Bucket_Predicted"] for row in decoded_output.select("AQI_Bucket_Predicted").collect()]
    actual_labels = df['AQI_Bucket'].tolist()

    # Confusion Matrix
    conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=df['AQI_Bucket'].unique())
    cm_display = ConfusionMatrixDisplay(conf_matrix, display_labels=df['AQI_Bucket'].unique())
    cm_display.plot(cmap='Blues')
    plt.title('Confusion Matrix: Actual vs Predicted AQI Buckets')
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # AQI Bucket Distribution (Bar Chart)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='AQI_Bucket', data=df, palette="Set2")
    plt.title('AQI Bucket Distribution')
    plt.xlabel('AQI Bucket')
    plt.ylabel('Count')
    plt.savefig("static/aqi_distribution.png")
    plt.close()

    # Scatter Plot: PM2_5 vs AQI Bucket
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PM2_5', y='AQI_Bucket', data=df, hue='AQI_Bucket', palette='Set1')
    plt.title('PM2_5 vs AQI Bucket')
    plt.xlabel('PM2_5 Concentration')
    plt.ylabel('AQI Bucket')
    plt.savefig("static/pm25_vs_aqi.png")
    plt.close()

    # Box Plot: PM2_5 vs AQI Bucket
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='AQI_Bucket', y='PM2_5', data=df, palette='Set3')
    plt.title('PM2_5 Concentration Distribution by AQI Bucket')
    plt.xlabel('AQI Bucket')
    plt.ylabel('PM2_5 Concentration')
    plt.savefig("static/pm25_boxplot.png")
    plt.close()

    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = df[['PM2_5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Pollutants')
    plt.savefig("static/correlation_matrix.png")
    plt.close()

    # Pollutant Contribution to AQI Bucket (Bar Chart)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['PM2_5', 'NO', 'NOx', 'CO', 'SO2'], y=[10.5, 2.83, 9.11, 0.26, 4.14], palette='viridis')
    plt.title('Pollutant Contribution to AQI Bucket')
    plt.xlabel('Pollutant')
    plt.ylabel('Concentration')
    plt.savefig("static/pollutant_contribution.png")
    plt.close()

    return render_template('prediction.html', prediction=predicted_single_label)

if __name__ == '__main__':
    app.run(debug=True)
