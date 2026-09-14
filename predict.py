import cv2
import numpy as np

def process_fingerprint(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Step 1: Apply Histogram Equalization
    equalized = cv2.equalizeHist(image)

    # Step 2: Normalize the image
    normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)

    # Step 3: Apply Gabor Filter
    def build_gabor_kernels(ksize=31, sigma=4.0, lambd=10.0, gamma=0.5):
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            kernels.append(kernel)
        return kernels

    def apply_gabor_filter(img, kernels):
        accum = np.zeros_like(img)
        for kernel in kernels:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
            np.maximum(accum, fimg, accum)
        return accum

    kernels = build_gabor_kernels()
    gabor_filtered = apply_gabor_filter(normalized, kernels)

    # Step 4: Binarize the image
    _, binary = cv2.threshold(gabor_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Apply Morphological Transformations
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 6: Skeletonize the image
    def skeletonize(img):
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            done = cv2.countNonZero(img) == 0
        return skel

    skeleton = skeletonize(morphed)

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Gabor Filtered Image', gabor_filtered)
    cv2.imshow('Binary Image', binary)
    cv2.imshow('Skeletonized Image', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the processed images
    cv2.imwrite('gabor_filtered.jpg', gabor_filtered)
    cv2.imwrite('binary_image.jpg', binary)
    cv2.imwrite('skeletonized_image.jpg', skeleton)

# Example usage
process_fingerprint("E:\\Blood Group Detection using Fingerprint\\Blood Group Detection\\niREP.png")
