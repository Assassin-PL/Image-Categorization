from DataProcesing import *

print("Hello World")

clusters = 8

data_dir = "Final_images_dataset"
data_preprocessor = DataPreprocessing(data_dir, num_clusters=clusters, blue_threshold=0.5, green_threshold=0.25, red_threshold=0.4)


