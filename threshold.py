from DataProcesing import *

print("Hello World")

clusters = 9

data_dir = "Final_images_dataset"
data_preprocessor = DataPreprocessing(data_dir, num_clusters=clusters, blue_threshold=0.5, green_threshold=0.25, red_threshold=0.4)
data_preprocessor.configure_gpu()
preprocessed_images = data_preprocessor.load_images()

# Detect outliers
outlier_indices = data_preprocessor.detect_outliers(preprocessed_images)

# Generate true labels for evaluation (replace this with your actual ground truth labels)
true_labels = np.random.randint(0, 3, len(preprocessed_images))

# Exclude outliers from clustering
valid_images = [image for i, image in enumerate(preprocessed_images) if i not in outlier_indices]
extracted_features = data_preprocessor.extract_features(valid_images)
cluster_labels = data_preprocessor.categorize_images(extracted_features)


# Evaluate categorization
evaluation_metrics = data_preprocessor.evaluate_categorization(true_labels, cluster_labels)
print("Evaluation Metrics:")
for metric, value in evaluation_metrics.items():
   print(f"{metric}: {value:.4f}")



