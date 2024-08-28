import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tf_keras.src.applications import VGG16
from tf_keras.src.applications.vgg16 import preprocess_input


class DataPreprocessing:
    def __init__(self, data_dir, num_clusters=5, blue_threshold=0.2):
        self.data_dir = data_dir
        self.num_clusters = num_clusters
        self.blue_threshold = blue_threshold
        self.feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.cluster_centers = None

    def load_images(self):
        images = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.data_dir, filename)
                image = cv2.imread(image_path)
                preprocessed_image = self._preprocess_image(image)
                images.append(preprocessed_image)

        return images

    def _preprocess_image(self, image):
        resized_image = cv2.resize(image, (224, 224))
        normalized_image = preprocess_input(resized_image)
        return normalized_image

    def extract_features(self, images):
        features = []
        for image in images:
            # Assuming a global average pooling layer is used for feature extraction
            feature_vector = self.feature_extractor.predict(np.expand_dims(image, axis=0))
            features.append(feature_vector.flatten())

        return np.array(features)

    def categorize_images(self, features):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        self.cluster_centers = kmeans.cluster_centers_
        return labels

    def detect_outliers(self, images):

        outlier_indices = []
        for i, image in enumerate(images):
            blue_channel_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
            blue_percentage = blue_channel_histogram[200:].sum() / blue_channel_histogram.sum()

            if blue_percentage > self.blue_threshold:
                outlier_indices.append(i)

        return outlier_indices

    def evaluate_categorization(self, true_labels, predicted_labels):

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        evaluation_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return evaluation_metrics

    def visualize_clusters(self, images, labels, images_per_row=3):
        unique_labels = np.unique(labels)

        for cluster_label in unique_labels:
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_images = [images[i] for i in cluster_indices]

            num_images = len(cluster_images)
            num_rows = (num_images + images_per_row - 1) // images_per_row

            plt.figure(figsize=(15, 8))

            # Find the reference image (e.g., the first image in the cluster)
            reference_image = cluster_images[0]
            reference_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256])

            # Find the image with the least matching color histogram
            least_matching_image = None
            min_hist_distance = float('inf')

            for i, image in enumerate(cluster_images):
                current_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                hist_distance = cv2.compareHist(reference_hist, current_hist, cv2.HISTCMP_CORREL)

                if hist_distance < min_hist_distance:
                    min_hist_distance = hist_distance
                    least_matching_image = image

            # Display the images in a grid
            for i, image in enumerate(cluster_images):
                plt.subplot(num_rows + 1, images_per_row, i + 1)
                plt.title(f'Cluster {cluster_label} - Image {i + 1}')
                plt.axis('off')

                # Display the image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image_rgb.astype('uint8'))

            # Display the image with the least matching color histogram separately
            plt.subplot(num_rows + 1, images_per_row, num_images + 1)
            plt.title(f'Cluster {cluster_label} - Least Matching Image')
            plt.axis('off')
            least_matching_image_rgb = cv2.cvtColor(least_matching_image, cv2.COLOR_BGR2RGB)
            plt.imshow(least_matching_image_rgb.astype('uint8'))

            plt.tight_layout()
            plt.show()

    def visualize_cluster_centers(self):
        if self.cluster_centers is not None:
            n_clusters, feature_dim = self.cluster_centers.shape
            center_shape = self.feature_extractor.output_shape[1:]

            plt.figure(figsize=(10, 5))
            for i, center in enumerate(self.cluster_centers):
                center_reshaped = center.reshape(center_shape)

                if len(center_shape) == 1:  # 1D vector
                    plt.subplot(1, n_clusters, i + 1)
                    plt.plot(center_reshaped, marker='o')
                    plt.title(f'Cluster Center {i}')
                elif len(center_shape) == 2:  # 2D array
                    plt.subplot(1, n_clusters, i + 1)
                    plt.imshow(center_reshaped, cmap='viridis')
                    plt.axis('off')
                    plt.title(f'Cluster Center {i}')

            plt.show()
        else:
            print("Cluster centers not available. Run categorization first.")

    def visualize_histograms(self, images, labels):
        unique_labels = np.unique(labels)
        for cluster_label in unique_labels:
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_images = [images[i] for i in cluster_indices]

            plt.figure(figsize=(12, 4))

            # Plot histograms
            plt.subplot(1, 2, 1)
            for image in cluster_images:
                blue_channel_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
                plt.plot(blue_channel_histogram, color='blue', alpha=0.2)
            plt.title(f'Cluster {cluster_label} - Blue Channel Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')

            # Plot cluster centers
            plt.subplot(1, 2, 2)
            center_reshaped = self.cluster_centers[cluster_label].reshape(-1)
            plt.plot(center_reshaped, marker='o')
            plt.title(f'Cluster {cluster_label} - Cluster Center')

            plt.tight_layout()
            plt.show()
