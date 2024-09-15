import os
import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure
from sklearn.cluster import KMeans
import scipy.fftpack as fftpack
import numpy as np

def remove_noise(image):

    if image.shape[0] != 512 or image.shape[1] != 512:

        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    img1 = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img2 = clahe.apply(img1.astype("uint8"))
    central_area = img2[100:400, 100:400]
    kmeans = KMeans(n_clusters=2).fit(np.reshape(central_area, [np.prod(central_area.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    ret, thres_img = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY_INV)
    img4 = cv2.erode(thres_img, kernel=np.ones([4,4]))
    img5 = cv2.dilate(img4, kernel=np.ones([13, 13]))
    img6 = cv2.erode(img5, kernel=np.ones([8, 8]))
    labels = measure.label(img6)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0.] < 500 and B[3] - B[1] < 490 and B[0] > 17 and B[2] < 495:
            good_labels.append(prop.label)

    mask = np.zeros_like(labels)
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)


    contours, hirearchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    internal_contours = np.zeros(mask.shape)
    external_contours = np.zeros(mask.shape)
    for i in range(len(contours)):
        if hirearchy[0][i][3] == -1:
           area = cv2.contourArea(contours[i])
           if area > 518.0:
             cv2.drawContours(external_contours, contours, i, (1, 1, 1), -1)
    external_contours = img = cv2.dilate(external_contours, kernel=np.ones([4, 4]))

    mask = cv2.bitwise_not(external_contours.astype(np.uint8))
    mask = cv2.erode(mask, kernel=np.ones((7, 7)))
    mask = cv2.bitwise_not(mask)
    mask = cv2.dilate(mask, kernel=np.ones((12, 12)))
    mask = cv2.erode(mask, kernel=np.ones((12, 12)))

    img7 = img1.astype(np.uint8)
    mask = mask.astype(np.uint8)
    seg = cv2.bitwise_and(img7, img7, mask=mask)
    return seg

def process_directory(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


            cleaned_image = remove_noise(image)


            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, cleaned_image)

            print(f"Processed {filename} cleaned image as {output_path}")


# Example usage:
healthy_dir = r'C:\Users\Marcus Connelly\Downloads\archive\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\Normal cases'
malignant_dir = r'C:\Users\Marcus Connelly\Downloads\archive\The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\Malignant cases'
sample_dir = r'C:\Users\Marcus Connelly\Downloads\ALCTID\ALCTID\healthy\images'
# Define output directories
healthy_output_dir = r'C:\Users\Marcus Connelly\Downloads\noiseremovedhealthy'
malignant_output_dir = r'C:\Users\Marcus Connelly\Downloads\noiseremovedmalignant'
sample = r'C:\Users\Marcus Connelly\Downloads\sample'
# Process all images in each directory and save them in the output directories
process_directory(healthy_dir, healthy_output_dir)
process_directory(malignant_dir, malignant_output_dir)
process_directory(sample_dir, sample)