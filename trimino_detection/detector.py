import data_preprocess
import image_analyzer
import blocks_reader

from PIL import Image


def detector(img_name):
    path = "данные/"
    print("Data preprocessing")
    base_img = Image.open(path + img_name)
    nimg = data_preprocess.normalized_img(base_img)

    pattern = image_analyzer.create_pattern()
    metric = image_analyzer.TriangleMetric(pattern, 28)

    print("Analyzing image")
    img = image_analyzer.ImageReader(nimg)
    img.find_triangles(metric, base_img)
    
    img.visualize_centroids(save_file="output/trimino_centers.png")
    img.visualize_mask(save_file="output/trimino_mask.png")
    
    print("Blocks detection")
    reader = blocks_reader.BlocksReader(img)
    blocks_reader.format_output(img, reader)
    return reader


s = str(input())
reader = detector(s)