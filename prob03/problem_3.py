import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

excel1_file_path = "prob3_im1.xlsx"
excel2_file_path = "prob3_im2.xlsx"

csv1_file_path = "prob3_im1.csv" 
csv2_file_path = "prob3_im2.csv" 

def convert_to_csv():
    """
    Convert the excel files into CSV files to be viewed on VS Code
    """
    excel1_df = pd.read_excel(excel1_file_path, header=None)
    excel2_df = pd.read_excel(excel2_file_path, header=None)

    excel1_df.to_csv(csv1_file_path, index=False, header=None)
    excel2_df.to_csv(csv2_file_path, index=False, header=None)
    print("Converted to CSV successfully.")

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    return df.to_numpy()

def view_original():
    """
    To view the original images from the CSV data
    """
    im1 = load_data(csv1_file_path)
    im2 = load_data(csv2_file_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(im1, cmap='jet') # jet gives the color gradient like the PDF
    plt.title("Original Image 1")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(im2, cmap='jet')
    plt.title("Original Image 2")
    plt.axis('off')
    plt.show()

def plot(img1, img2, title1="Image 1", title2="Image 2"):
    """
    Plot the resulting images
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='jet') 
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='jet')
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot2(img1, img2, title1="Image 1", title2="Image 2"):
    """
    Adjusted the constrast to make the images more similar to the ones in the PDF
    """
    plt.figure(figsize=(12, 6))

    vmin1, vmax1 = np.percentile(img1, [2, 98]) # to get more red and blue, instead of green
    vmin2, vmax2 = np.percentile(img2, [2, 98])

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='jet', vmin=vmin1, vmax=vmax1) # vmin and vmax to stretch the contrast properly
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='jet', vmin=vmin2, vmax=vmax2)
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def solve_art():
    im1 = load_data('prob3_im1.csv')
    im2 = load_data('prob3_im2.csv')

    fft1 = np.fft.fft2(im1)
    fft2 = np.fft.fft2(im2)

    mag1 = np.abs(fft1)
    phase1 = np.angle(fft1)

    mag2 = np.abs(fft2)
    phase2 = np.angle(fft2)

    new_fft1 = mag1 * np.exp(1j * phase2) # magniture of image 1 with phase of image 2
    new_fft2 = mag2 * np.exp(1j * phase1) # magnitude of image 2 with phase of image 1

    new_im1 = np.abs(np.fft.ifft2(new_fft1))
    new_im2 = np.abs(np.fft.ifft2(new_fft2))

    plot2(new_im1, new_im2)

if __name__ == "__main__":
    solve_art()