import cv2
from skimage import color, filters, exposure
from skimage.filters import unsharp_mask
import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "result_images"
os.makedirs(output_dir, exist_ok=True)

image1 = cv2.imread('input/pic1.png')
image2 = cv2.imread('input/pic2.png')
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


def convert_color_spaces(image):
    image_hsv = color.rgb2hsv(image)
    image_lab = color.rgb2lab(image)

    plt.imsave(f"{output_dir}/image_rgb.png", image)

    plt.imsave(f"{output_dir}/image_hsv.png", image_hsv)

    image_lab_normalized = ((
                            image_lab - image_lab.min()) /
                            (image_lab.max() - image_lab.min()))
    plt.imsave(f"{output_dir}/image_lab.png", image_lab_normalized)


def smooth_image_componentwise(image):
    smoothed_rgb = np.zeros_like(image, dtype=float)
    for i in range(3):
        smoothed_rgb[:, :, i] = filters.gaussian(image[:, :, i], sigma=2)
    smoothed_rgb = ((
                    smoothed_rgb - smoothed_rgb.min()) /
                    (smoothed_rgb.max() - smoothed_rgb.min()))
    plt.imsave(f"{output_dir}/smoothed_rgb_componentwise.png", smoothed_rgb)


def smooth_image_in_color_spaces(image):
    smoothed_hsv = color.rgb2hsv(image)
    smoothed_hsv[:, :, 2] = filters.gaussian(smoothed_hsv[:, :, 2], sigma=2)
    smoothed_lab = color.rgb2lab(image)
    smoothed_lab[:, :, 0] = filters.gaussian(smoothed_lab[:, :, 0], sigma=2)
    plt.imsave(f"{output_dir}/smoothed_hsv.png", color.hsv2rgb(smoothed_hsv))
    plt.imsave(f"{output_dir}/smoothed_lab.png", color.lab2rgb(smoothed_lab))


def sharpen_image_componentwise(image):
    sharpened_rgb = np.zeros_like(image, dtype=float)
    for i in range(3):
        sharpened_rgb[:, :, i] = (
            unsharp_mask(image[:, :, i], radius=1, amount=1.5)
        )
    plt.imsave(f"{output_dir}/sharpened_rgb_componentwise.png", sharpened_rgb)


def sharpen_image_in_color_spaces(image):
    sharp_hsv = color.rgb2hsv(image)
    sharp_hsv[:, :, 2] = (
        filters.unsharp_mask(sharp_hsv[:, :, 2], radius=1, amount=0.5)
    )

    sharp_lab = color.rgb2lab(image)
    sharp_lab[:, :, 0] = (
        filters.unsharp_mask(sharp_lab[:, :, 0], radius=1, amount=0.5)
    )

    sharp_lab = np.clip(sharp_lab, 0, 100)

    sharp_lab_rgb = color.lab2rgb(sharp_lab)
    sharp_lab_rgb = np.clip(sharp_lab_rgb, 0, 1)

    plt.imsave(f"{output_dir}/sharpened_hsv.png", color.hsv2rgb(sharp_hsv))
    plt.imsave(f"{output_dir}/sharpened_lab.png", sharp_lab_rgb)


def equalize_histogram(image):
    equalized_rgb = np.zeros_like(image, dtype=float)
    for i in range(3):
        equalized_rgb[:, :, i] = exposure.equalize_hist(image[:, :, i])
    equalized_rgb = (
            (equalized_rgb - equalized_rgb.min()) /
            (equalized_rgb.max() - equalized_rgb.min())
    )
    plt.imsave(f"{output_dir}/equalized_histogram.png", equalized_rgb)


def detect_edges(image):
    edges_rgb = (
        np.sqrt(sum(filters.sobel(image[:, :, i]) ** 2 for i in range(3)))
    )

    edges_vector = filters.sobel(color.rgb2gray(image))

    plt.imsave(
        f"{output_dir}/edges_rgb_componentwise.png",
        edges_rgb,
        cmap="gray"
    )
    plt.imsave(
        f"{output_dir}/edges_vector_function.png",
        edges_vector,
        cmap="gray"
    )

    difference = np.abs(edges_rgb - edges_vector)

    plt.imsave(f"{output_dir}/edges_difference.png", difference, cmap="gray")


if __name__ == "__main__":
    print("1. Перетворення зображення у різні простори кольору")
    convert_color_spaces(image1_rgb)

    print("2. Згладжування зображення покомпонентно")
    smooth_image_componentwise(image2_rgb)

    print("3. Згладжування зображення в HSV та L*a*b* просторах")
    smooth_image_in_color_spaces(image2_rgb)

    print("4. Підвищення різкості покомпонентно")
    sharpen_image_componentwise(image2_rgb)

    print("5. Підвищення різкості в HSV та L*a*b* просторах")
    sharpen_image_in_color_spaces(image2_rgb)

    print("6. Еквалізація гістограм")
    equalize_histogram(image2_rgb)

    print("7. Виявлення контурів")
    detect_edges(image2_rgb)

    print(f"Всі результати збережено у папці '{output_dir}'.")
