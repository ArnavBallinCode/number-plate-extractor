import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

# Set Tesseract executable path (adjust the path for your environment)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Function to extract text from the number plate
def extract_number_plate_text(image):
    # Convert to OpenCV format
    image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur (to reduce noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edged = cv2.Canny(blurred, 30, 200)
    
    # Find contours based on edges detected
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours based on area and keep the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    screenCnt = None
    # Loop over contours to find the best possible rectangle
    for contour in contours:
        # Approximate the contour
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the contour has four points, we assume it is a rectangle
        if len(approx) == 4:
            screenCnt = approx
            break
    
    if screenCnt is None:
        return "No number plate detected", None
    
    # Create a mask for the number plate
    mask = cv2.drawContours(np.zeros_like(gray), [screenCnt], -1, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Crop the image to the bounding box of the number plate
    x, y, w, h = cv2.boundingRect(screenCnt)
    cropped = masked[y:y+h, x:x+w]
    
    # Apply binary thresholding to make the image black and white
    _, thresholded = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use Tesseract to extract text
    text = pytesseract.image_to_string(thresholded, config='--psm 8')
    
    return text.strip(), thresholded

# Streamlit app interface
st.title("Number Plate Text Extractor")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a number plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Extract number plate text
    with st.spinner("Processing..."):
        text, processed_image = extract_number_plate_text(image)
    
    # Display extracted text
    if processed_image is not None:
        st.success("Extraction Complete!")
        st.write(f"**Extracted Text:** {text}")
        st.image(processed_image, caption="Processed Image", use_column_width=True)
    else:
        st.error(text)

# Footer
st.markdown("Developed by [Your Name](https://github.com/yourusername)")
