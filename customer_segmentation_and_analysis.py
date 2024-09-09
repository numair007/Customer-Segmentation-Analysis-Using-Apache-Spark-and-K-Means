from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import load_data

data = load_data.load_data("path_to_mall_customers_data.csv")

data.show(5)

selected_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
assembled_data = assembler.transform(data)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaled_data = scaler.fit(assembled_data).transform(assembled_data)

kmeans = KMeans(featuresCol="scaledFeatures", k=3)
model = kmeans.fit(scaled_data)

predictions = model.transform(scaled_data)

evaluator = ClusteringEvaluator(featuresCol="scaledFeatures", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette_score = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette_score}")

predictions.select("CustomerID", "Age", "Annual Income (k$)", "Spending Score (1-100)", "prediction").show()

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

model.save("kmeans_customer_segmentation_model")
