import cv2
import numpy as np
import matplotlib.pyplot as plt

eiffel = cv2.imread('prob04_im01.jpg')
fuji = cv2.imread('prob04_im02.jpg')

eiffel_rgb = cv2.cvtColor(eiffel, cv2.COLOR_BGR2RGB)
fuji_rgb = cv2.cvtColor(fuji, cv2.COLOR_BGR2RGB)

def gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel for filtering
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel) #normalize 
    return kernel

def apply_lowpass(image, kernel_size=31, sigma=10):
    """
    Apply Gaussian kernel as low-pass filter to extract low frequencies
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    
    #apply filter to each channel
    filtered = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
        filtered[:, :, i] = cv2.filter2D(image[:, :, i].astype(np.float32), -1, kernel)
    
    return filtered

def apply_highpass(image, kernel_size=31, sigma=10):
    """
    Apply high-pass filter by subtracting low-pass from original
    """
    lowpass = apply_lowpass(image, kernel_size, sigma)
    
    #high-pass = original - low-pass
    highpass = image.astype(np.float32) - lowpass
    
    return highpass

def create_hybrid_image(kernel_size=31, sigma_low=10, sigma_high=5):
    """
    Create hybrid image by combining:
    - Low frequencies from image1 (Eiffel Tower)
    - High frequencies from image2 (Mount Fuji)
    """

    img1_float = fuji_rgb.astype(np.float32)
    img2_float = eiffel_rgb.astype(np.float32)
    
    #extract low frequencies from Eiffel Tower
    low_freq = apply_lowpass(img1_float, kernel_size, sigma_low)
    
    #extract high frequencies from Mt. Fuji
    high_freq = apply_highpass(img2_float, kernel_size, sigma_high)
    
    #combine low and high frequencies
    hybrid = low_freq + high_freq
    
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)
    
    return hybrid, low_freq, high_freq
    

if __name__ == "__main__":
    hybrid, low_freq, high_freq = create_hybrid_image(
        kernel_size=31, 
        sigma_low=10,  #controls low-pass
        sigma_high=5   #controls high-pass
    )

    display_low = np.clip(low_freq, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(display_low)
    plt.title("Low pass Mount Fuji")
    plt.axis('off')
    plt.show()

    display_high = high_freq + 128  # Shift to positive range
    display_high = np.clip(display_high, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 8))
    plt.imshow(display_high)
    plt.title("High pass Eiffel Tower")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(hybrid)
    plt.title("Hybrid Image")
    plt.axis('off')
    plt.show()

    #save final hybrid image
    hybrid_bgr = cv2.cvtColor(hybrid, cv2.COLOR_RGB2BGR)
    cv2.imwrite('prob4_hybrid_image.jpg', hybrid_bgr)