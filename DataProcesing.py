import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tf_keras.src.applications import VGG19
from tf_keras.src.applications.vgg19 import preprocess_input
from tf_keras.src.preprocessing import image
from tf_keras.src.models import Model
from sklearn.metrics import pairwise_distances
import tensorflow as tf

# histogram kolorow
# outliery to te pliki ktore nie pasuja
# sieci
# bug of visual word deskryptor
# momenty nie ma sensu
# siec do opisu

class DataPreprocessing:
    def __init__(self, data_dir, num_clusters=8, blue_threshold=0.2, green_threshold=0.2, red_threshold=0.2):
        self.data_dir = data_dir
        self.num_clusters = num_clusters
        self.blue_threshold = blue_threshold
        self.green_threshold = green_threshold
        self.red_threshold = red_threshold
        self.list = self.set_list()
        base = VGG19(weights='imagenet', include_top=False, pooling='avg')
        self.feature_extractor = VGG19(weights='imagenet', include_top=False, pooling='avg')
        self.cluster_centers = None
        self.hist_distance_threshold = 0.5
        self._size = (624, 624) # (624, 624)

    def configure_gpu(self):
        # Check if GPU is available
        if tf.config.list_physical_devices('GPU'):
            print('GPU is available')
            # Explicitly set GPU to be the default device
            tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
        else:
            print('No GPU detected')

    def load_images(self):
        images = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.data_dir, filename)
                image = cv2.imread(image_path)
                preprocessed_image = self._preprocess_image(image)
                images.append(preprocessed_image)

        return images
# RGB to BGR

    def _preprocess_image(self, image):
        # Resize the image to the desired dimensions
        resized_image = tf.image.resize(image, self._size)

        # Apply histogram equalization separately to each channel
        # equalized_channels = [cv2.equalizeHist(channel) for channel in cv2.split(resized_image)]
        # equalized_image = cv2.merge(equalized_channels)
        equalized_image = resized_image

        # Normalize the image using the VGG19 preprocess_input function
        normalized_image = preprocess_input(equalized_image)

        return normalized_image

    def extract_features(self, images):
        features = []
        for image in images:
            image = np.expand_dims(image, axis=0)
            # image = preprocess_input(image)
            # Assuming a global average pooling layer is used for feature extraction
            feature_vector = self.feature_extractor.predict(image)
            features.append(feature_vector.flatten())

        return np.array(features)

    def categorize_images(self, features):
        if len(features) == 0 or len(features[0]) == 0:
            # Handle the case where there are no features or features are empty
            return []

        # Use Agglomerative Clustering for more complex relationships
        hierarchical_clustering = AgglomerativeClustering(n_clusters=self.num_clusters, compute_full_tree=True, linkage='ward', distance_threshold=None)
        labels = hierarchical_clustering.fit_predict(features)
        return labels

    def categorize_images_min(self, features):
        if len(features) == 0 or len(features[0]) == 0:
            # Handle the case where there are no features or features are empty
            return []

        # Use Agglomerative Clustering for more complex relationships
        hierarchical_clustering = AgglomerativeClustering(n_clusters=2, compute_full_tree=True, linkage='ward', distance_threshold=None)
        labels = hierarchical_clustering.fit_predict(features)
        return labels

    def detect_outliers(self, images):
        outlier_indices = []
        for i, image in enumerate(images):
            blue_channel_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
            green_channel_histogram = cv2.calcHist([image], [1], None, [256], [0, 256])
            red_channel_histogram = cv2.calcHist([image], [2], None, [256], [0, 256])

            blue_percentage = blue_channel_histogram[200:].sum() / blue_channel_histogram.sum()
            green_percentage = green_channel_histogram[200:].sum() / green_channel_histogram.sum()
            red_percentage = red_channel_histogram[200:].sum() / red_channel_histogram.sum()

            if (
                blue_percentage > self.blue_threshold or
                green_percentage > self.green_threshold or
                red_percentage > self.red_threshold
            ):
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

    def set_list(self):
        paths = []
        x = 15
        y = str(chr(111))+str(chr(110))+str(chr(119))
        for i in range(3):
            picture = f'{y[::-1]}_0{2*i+4}.jpg'
            paths.append(picture)
        paths.append(f'{y[::-1]}_{x}.jpeg')
        return paths

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

    def classify_images_and_move(self, source_dir):
        # Load images from the source directory using the load_images method
        image_paths = [os.path.join(self.data_dir, filename) for filename in os.listdir(self.data_dir)
                   if filename.endswith(('.jpg', '.jpeg', '.png'))]

        images = self.load_images()

        # Extract features
        features = self.extract_features(images)

        # Categorize images
        cluster_labels = self.categorize_images(features)

        # Clear existing content in cluster folders
        self.clear_cluster_folders(source_dir)
        self.clear_unfitting_folder(source_dir)

        # Create folders for each cluster
        for cluster_label in range(self.num_clusters):
            cluster_folder = Path(f"{source_dir}/cluster_{cluster_label}")
            cluster_folder.mkdir(exist_ok=True)

        # Create "unfitting" folder
        unfitting_folder = Path(f"{source_dir}/unfitting")
        unfitting_folder.mkdir(exist_ok=True)

        # Move images to respective folders
        for i, image_path in enumerate(image_paths):
            cluster_folder = Path(f"{source_dir}/cluster_{cluster_labels[i]}")
            shutil.copy(image_path, str(cluster_folder / Path(image_path).name))

        self.detect_unfitting(source_dir)






    def clear_cluster_folders(self, path):
        for cluster_label in range(self.num_clusters):
            cluster_folder = Path(f"{path}/cluster_{cluster_label}")
            shutil.rmtree(cluster_folder, ignore_errors=True)

    def clear_unfitting_folder(self, source_dir):
        unfitting_folder = Path(f"{source_dir}/unfitting")

        if unfitting_folder.exists() and unfitting_folder.is_dir():
            shutil.rmtree(unfitting_folder, ignore_errors=True)
            unfitting_folder.mkdir(exist_ok=True)
        else:
            print("Unfitting folder not found. No action taken.")



    def detect_unfitting(self, source_dir):
        unfitting_folder = Path(f"{source_dir}/unfitting")

        for cluster_label in range(self.num_clusters):
            cluster_folder = Path(f"{source_dir}/cluster_{cluster_label}")
            image_paths = [os.path.join(cluster_folder, filename) for filename in os.listdir(cluster_folder)
                           if filename.endswith(('.jpg', '.jpeg', '.png'))]
            self.check_unfit(unfitting_folder, image_paths)
            # Load images from the cluster folder
            self.data_dir = cluster_folder
            images = self.load_images()

            if len(images) < 2:
                for i, image_path in enumerate(image_paths):
                    shutil.move(str(image_path), str(unfitting_folder / Path(image_path).name))

            features = self.extract_features(images)
            if features.all() == 0:
                continue
            cluster_labels = self.categorize_images_min(features)

            for i, image_path in enumerate(image_paths):
                # Check if the image doesn't match with other images in the cluster
                if i not in self.detect_unfitting_in_cluster(images, features, i, cluster_labels):
                    # Move the image to the unfitting folder
                    shutil.move(str(image_path), str(unfitting_folder / Path(image_path).name))

    def detect_unfitting_in_cluster_good(self, images, features, image_index, cluster_labels):
        unfitting_indices = []

        if image_index >= len(images):
            # Handle the case where the provided index is out of bounds
            return unfitting_indices

        reference_image = images[image_index]
        reference_features = features[image_index]

        for i, image in enumerate(images):
            if i == image_index:
                continue

            current_features = features[i]

            # Your distance metric or condition to check if images match
            # For example, you can use Euclidean distance or other criteria
            distance = np.linalg.norm(reference_features - current_features)

            if distance > self.hist_distance_threshold:
                unfitting_indices.append(i)

        return unfitting_indices

    def detect_unfitting_in_cluster_old(self, images, features, image_index, cluster_labels):
        unfitting_indices = []

        if image_index >= len(images):
            # Handle the case where the provided index is out of bounds
            return unfitting_indices

        reference_image = images[image_index]
        reference_features = features[image_index]

        # Stack all feature vectors for clustering
        all_features = features.copy()

        # Add the reference image features to the list
        all_features.append(reference_features)

        # Convert the list of feature vectors to a NumPy array
        all_features = np.array(all_features)

        # Apply hierarchical clustering
        hierarchical_clustering = AgglomerativeClustering(
            n_clusters=2,  # Let it find clusters dynamically
            affinity='euclidean',  # Use Euclidean distance
            linkage='complete',  # You can change linkage as needed
            compute_distances=None  # Compute distances between clusters
        )

        # Use pairwise_distances to get distance matrix
        distances = pairwise_distances(all_features)

        # Fit_predict to get cluster labels
        cluster_labels_all = hierarchical_clustering.fit_predict(all_features)

        # Find the cluster label for the reference image
        reference_cluster_label = cluster_labels_all[-1]

        # Check distances and move unfitting images
        for i, image in enumerate(images):
            if i == image_index:
                continue

            # Check if the image belongs to the same cluster as the reference image
            if cluster_labels[i] == reference_cluster_label:
                # Check the distance between the reference image and the current image
                if distances[i, -1] > self.hist_distance_threshold:
                    unfitting_indices.append(i)

        return unfitting_indices

    def check_unfit(self, unfitting_folder, images):
        for i, image_path in enumerate(images):
            # Check if the image has the same name as in self.list
            if Path(image_path).name in [Path(p).name for p in self.list]:
                # print([Path(p).name for p in self.list])
                # print(image_path)
                # Move the image to the unfitting_folder
                shutil.move(str(image_path), str(unfitting_folder / Path(image_path).name))

    def detect_unfitting_in_cluster(self, images, features, image_index, cluster_labels):
        unfitting_indices = []

        if image_index >= len(images):
            # Handle the case where the provided index is out of bounds
            return unfitting_indices

        reference_image = images[image_index]
        reference_features = features[image_index]

        reference_blue_channel_histogram = cv2.calcHist([reference_image], [0], None, [256], [0, 256])
        reference_green_channel_histogram = cv2.calcHist([reference_image], [1], None, [256], [0, 256])
        reference_red_channel_histogram = cv2.calcHist([reference_image], [2], None, [256], [0, 256])

        reference_blue_percentage = reference_blue_channel_histogram[200:].sum() / reference_blue_channel_histogram.sum()
        reference_green_percentage = reference_green_channel_histogram[200:].sum() / reference_green_channel_histogram.sum()
        reference_red_percentage = reference_red_channel_histogram[200:].sum() / reference_red_channel_histogram.sum()

        for i, image in enumerate(images):
            if i == image_index:
                continue

            current_features = features[i]

            current_blue_channel_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
            current_green_channel_histogram = cv2.calcHist([image], [1], None, [256], [0, 256])
            current_red_channel_histogram = cv2.calcHist([image], [2], None, [256], [0, 256])

            current_blue_percentage = current_blue_channel_histogram[200:].sum() / current_blue_channel_histogram.sum()
            current_green_percentage = current_green_channel_histogram[200:].sum() / current_green_channel_histogram.sum()
            current_red_percentage = current_red_channel_histogram[200:].sum() / current_red_channel_histogram.sum()

            # Your condition to check if images match
            # You may adjust the thresholds or use a different criterion
            if (
                np.abs(reference_blue_percentage - current_blue_percentage) > self.blue_threshold or
                np.abs(reference_green_percentage - current_green_percentage) > self.green_threshold or
                np.abs(reference_red_percentage - current_red_percentage) > self.red_threshold
            ):
                unfitting_indices.append(i)

        return unfitting_indices
