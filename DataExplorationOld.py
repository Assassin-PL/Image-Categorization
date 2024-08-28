import os
from PIL import Image
import re
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np


class PictureDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class Picture:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data_set = self.load_dataset()
        self.train_images, self.val_images, self.test_images = self.split_dataset()
        self.manual_features, self.labels = self.extract_manual_features()
        self.neural_network_features = self.extract_neural_network_features()
        self.reduced_features = self.dimensionality_reduction(n_components=10)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if self.train_images is not None:
            self.train_dataset = PictureDataset(images=self.train_images, labels=self.extract_labels(self.train_images), transform=self.get_transform())

        if self.val_images is not None:
            self.val_dataset = PictureDataset(images=self.val_images, labels=self.extract_labels(self.val_images), transform=self.get_transform())

        if self.test_images is not None:
            self.test_dataset = PictureDataset(images=self.test_images, labels=self.extract_labels(self.test_images), transform=self.get_transform())
        # After loading and processing the dataset
        print("Number of training images:", len(self.train_images))
        print("Number of manual features:", len(self.data_set))
        print("Number of labels:", len(self.extract_labels(self.train_images)))
        print("First few labels:", self.extract_labels(self.train_images)[:5])

    def load_dataset(self):
        data_set = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path)
                    image_info = {
                        'path': image_path,
                        'format': image.format,
                        'size': image.size,
                        'mode': image.mode
                    }
                    data_set.append(image_info)
        return data_set

    def confirm_labels(self):
        jpg_pattern = re.compile(r'wno_(\d+)\.jpg', re.IGNORECASE)
        jpeg_pattern = re.compile(r'wno_(\d+)\.jpeg', re.IGNORECASE)

        for image_info in self.data_set:
            jpg_match = jpg_pattern.match(os.path.basename(image_info['path']))
            jpeg_match = jpeg_pattern.match(os.path.basename(image_info['path']))

            if jpg_match:
                label = int(jpg_match.group(1))
            elif jpeg_match:
                label = int(jpeg_match.group(1))
            else:
                # Handle images with invalid labels or no labels.
                # Assign a default label or decide how to handle such cases.
                label = -1  # Assign a default label or skip these images.

            # Assign label if 'label' key is present, otherwise add the 'label' key
            if 'label' in image_info:
                image_info['label'] = label
            else:
                image_info['label'] = label

            print(f"Image: {image_info['path']}, Label: {image_info['label']}")

    def split_dataset(self):
        if self.data_set is None:
            return None, None, None

        # Assuming data_set is a list of dictionaries with 'path' key
        images = [{'path': entry['path']} for entry in self.data_set]

        # Modify the code to handle the case where 'label' key is not present
        labels = [entry.get('label', None) for entry in self.data_set]

        # Split the dataset
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

        return train_images, val_images, test_images

    def extract_manual_features(self):
        manual_features = []
        labels = []

        for image_info in self.data_set:
            # Implement functions to extract manual features (e.g., color histograms, textures, shapes)
            features = self.extract_color_histogram(image_info['path'])
            # Add other manual features extraction functions as needed
            manual_features.append(features)
            labels.append(image_info.get('label', -1))

        return np.array(manual_features), np.array(labels)

    def extract_neural_network_features(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(pretrained=True)
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        neural_network_features = []

        for image_info in self.data_set:
            image = Image.open(image_info['path']).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model(image)

            neural_network_features.append(features.cpu().numpy().flatten())

        return np.array(neural_network_features)

    def dimensionality_reduction(self, n_components=50):
        if n_components <= 0:
            raise ValueError("Number of components must be greater than 0.")

        # Extract manual features and labels
        self.manual_features, self.labels = self.extract_manual_features()

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(n_components, min(self.manual_features.shape)))
        reduced_manual_features = pca.fit_transform(self.manual_features)

        print("Shape of manual_features after PCA:", reduced_manual_features.shape)

        return reduced_manual_features

    def extract_color_histogram(self, image_path):
        # Placeholder function for extracting color histograms
        # Replace this with your actual implementation
        # You might want to use libraries like OpenCV to work with images
        # Here's a simple example using random numbers:
        return np.random.rand(10)  # Assuming a histogram with 10 bins


    def extract_labels(self, dataset):
        labels = [entry.get('label', -1) for entry in dataset]
        return labels

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train_ml_models(self):
        if self.train_images is None:
            print("Training images are not available.")
            return

        print(f"Number of training images: {len(self.train_images)}")
        print(f"Number of manual features: {len(self.manual_features)}")
        print(f"Number of labels: {len(self.extract_labels(self.train_images))}")

        # Print the first few labels to inspect them
        print("First few labels:", self.extract_labels(self.train_images)[:5])

        # Train traditional ML models
        svm_model = SVC()

        if len(self.manual_features) == len(self.extract_labels(self.train_images)):
            svm_model.fit(self.manual_features, self.extract_labels(self.train_images))
            svm_predictions_val = svm_model.predict(self.extract_manual_features(self.val_images))
            print("SVM Accuracy on Validation Set:", accuracy_score(self.extract_labels(self.val_images), svm_predictions_val))
        else:
            print("Inconsistent number of samples between features and labels for SVM model.")

        # Repeat the same process for other models (e.g., Random Forest)
        # ...

    def train_neural_network(self, epochs=10):  # Define the number of epochs as an argument
        if self.train_images is None:
            print("Training images are not available.")
            return

        # Train a neural network model
        class SimpleNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc = nn.Linear(input_size, output_size)

            def forward(self, x):
                x = self.fc(x)
                return x

        input_size = len(self.manual_features[0])
        output_size = len(set(self.extract_labels(self.train_images)))

        model = SimpleNN(input_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Convert features to PyTorch tensor
        train_data = torch.tensor(self.manual_features, dtype=torch.float32)
        train_labels = torch.tensor(self.extract_labels(self.train_images), dtype=torch.long)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        # Evaluation on the validation set
        val_data = torch.tensor(self.extract_manual_features(self.val_images), dtype=torch.float32)
        val_labels = torch.tensor(self.extract_labels(self.val_images), dtype=torch.long)
        val_outputs = model(val_data)
        _, val_predictions = torch.max(val_outputs, 1)

        print("Neural Network Accuracy on Validation Set:", accuracy_score(val_labels, val_predictions))

    def train_neural_network(self, epochs=10):
        if self.train_images is None:
            print("Training images are not available.")
            return

        # ... (previous code)

    def train_ensemble_model(self):
        if self.train_images is None:
            print("Training images are not available.")
            return
        # Print lengths for debugging
        print("Length of manual_features:", len(self.manual_features))
        print("Length of labels:", len(self.extract_labels(self.train_images)))
        # Train an ensemble model (VotingClassifier in scikit-learn)
        from sklearn.ensemble import VotingClassifier

        svm_model = SVC(probability=True)
        rf_model = RandomForestClassifier()

        ensemble_model = VotingClassifier(estimators=[
            ('svm', svm_model),
            ('random_forest', rf_model)
        ], voting='soft')  # 'soft' for probability voting

        # Train on the training set
        if len(self.manual_features) == len(self.extract_labels(self.train_images)):
            ensemble_model.fit(self.manual_features, self.extract_labels(self.train_images))

            # Evaluate on the validation set
            ensemble_predictions_val = ensemble_model.predict(self.extract_manual_features(self.val_images))

            print("Ensemble Model Accuracy on Validation Set:", accuracy_score(self.extract_labels(self.val_images), ensemble_predictions_val))
        else:
            print("Inconsistent number of samples between features and labels for Ensemble model.")
