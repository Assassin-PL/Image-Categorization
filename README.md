# Image Categorization and Outlier Detection

This project focuses on analyzing and categorizing images into predefined categories using neural networks and image processing techniques. Additionally, it identifies unique images that do not fit into any category, labeling them as outliers.

## Project Overview

The primary objective of this project is to categorize images from a dataset into distinct groups and identify images that are outliers—those that are unique or do not belong to any category.

### Categories

The images in the dataset are classified into the following categories:

- **Sports**
- **Monuments**
- **Landmarks**
- **Sharks**
- **Vegetables**
- **Rubbish**
- **Food**
- **Glass**

### Outliers

Certain images are identified as outliers due to their unique characteristics. Examples include:

- `wno_04.jpg`: The only stadium image.
- `wno_06.jpg`: The only grayscale image.
- `wno_08.jpg`: The only image of a person with a weapon.
- `wno_09.jpg`: The only sunset image.
- `wno_87.jpg`: The only image of shoes.
- `wno_88.jpg`: The only anime image.
- `wno_15.jpeg`: The only Picasso painting in the set.
- `wno_89.jpg`: The only oil painting.

**Note:** Only one image can be considered a valid outlier.

## Project Structure

The project is organized as follows:

- **DataExplorationOld.py**: Contains classes and methods for loading, splitting, and analyzing the dataset, extracting features using both manual methods and neural networks, and training machine learning models.
- **DataProcessing.py**: Implements the main image preprocessing pipeline, feature extraction, image categorization using clustering, and outlier detection.
- **main.py**: The main entry point for running the preprocessing and categorization pipeline.
- **threshold.py**: Focuses on clustering and evaluating the categorization, with specific emphasis on outlier detection.
- **preprocess.py**: Handles preprocessing tasks such as image resizing, normalization, feature extraction using pre-trained neural networks, and visualization of clusters and histograms.
- **test.py**: Includes basic setup for testing the pipeline and categorization logic.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Scikit-learn
- OpenCV
- Matplotlib
- NumPy
- PIL (Python Imaging Library)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
### Installation

1. Clone the repository:

     ```bash
     git clone https://github.com/yourusername/your-repo-name.git
2.   Install the required dependencies:
     ```bash
       pip install -r requirements.txt
### Running the Project

1. **Preprocess and Categorize Images**:
   Run the preprocessing and categorization pipeline by executing:

   ```bash
   python main.py
2. Detect Outliers: Outliers can be detected by running:
   ```bash
   python threshold.py
3. Test the Pipeline: Basic tests can be conducted by running:
   ```bash
   python test.py
### Configuration

- **Data Directory**: Update the `data_dir` variable in the scripts to point to your dataset directory.
- **Number of Clusters**: Adjust the `clusters` variable to define the number of categories for clustering.
- **Color Thresholds**: Modify the `blue_threshold`, `green_threshold`, and `red_threshold` parameters to fine-tune the outlier detection based on color histograms.

### Visualization

The project includes methods for visualizing the clusters, histograms, and cluster centers. These visualizations help in understanding the distribution of images across categories and identifying outliers.

#### Cluster Visualization

The images are displayed in a grid, grouped by their respective clusters, with an additional visualization for the least matching image in each cluster.

#### Histogram Visualization

Histograms of color channels (blue, green, red) are plotted for images within each cluster, alongside the cluster centers, to visualize the distribution of colors.

### Evaluation

The categorization performance is evaluated using metrics such as Accuracy, Precision, Recall, and F1 Score. These metrics help in assessing the effectiveness of the clustering and outlier detection.

### Contributing

Feel free to contribute to this project by submitting issues or pull requests. Please ensure that you follow the contribution guidelines.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
