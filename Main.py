import cv2
import numpy as np
from numpy import ones, uint8

def region_growing(image, seed, threshold=50):
    rows, cols = image.shape 
    segmented = np.zeros_like(image)
    visited = np.zeros_like(image)

    seed_value = image[seed]

    region = [seed]
    segmented[seed] = 255
    visited[seed] = 1

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while region:
        x, y = region.pop(0)
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                diff = abs(int(image[nx, ny]) - int(seed_value))
                
                if diff < threshold:
                    region.append((nx, ny))
                    segmented[nx, ny] = 255
                    visited[nx, ny] = 1

    return segmented

def noise_removal(image):
    kernel = np.ones((3, 3), np.uint8)  
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = ones((2,2), uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def increase_contrast_histogram(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_image)

def identify_image(image):
    image = cv2.imread(image)
    new_size = (800, 600)


    seed = (100, 100) 

    alpha = 1.5 
    beta = 0

    image = cv2.resize(image, new_size)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    segmented_image = region_growing(image_gray, seed)

    image_no_noise = noise_removal(segmented_image)

    image_thickned = thick_font(image_no_noise)

    cv2.imshow('Imagem Original', image)
    cv2.imshow('Imagem Modificada', image_thickned)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

baiacu = "images/baiacu.jpg"
tucunare = "images/tucunare.jpg"
tubarao = "images/tubarao.jpg"

identify_image(baiacu)
identify_image(tucunare)
identify_image(tubarao)