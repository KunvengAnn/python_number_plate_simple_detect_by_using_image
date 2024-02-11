import cv2
from ultralytics import YOLO
import os
import numpy as np
import easyocr

# Load a YOLO model
model = YOLO(r"D:\python\Licence_Plate\detect\train2\weights\best.pt")  # Load a pretrained YOLOv8 model
results = model(r"D:\images\xe_co.jpg", show=False)  # Perform object detection on the input image

# Assuming there is only one image in the batch
image_index = 0
image_info = results[image_index]

# Retrieve bounding boxes from the detected results
boxes = image_info.boxes.data.cpu().numpy()

# Use EasyOCR for text recognition
reader = easyocr.Reader(['en'])

# Load the input image
image_path = r"D:\images\xe_co.jpg"
image = cv2.imread(image_path)

# Create a folder to save cropped images
output_folder = "cropped_images"
os.makedirs(output_folder, exist_ok=True)

# Process each bounding box
for i, box in enumerate(boxes):
    x_min, y_min, x_max, y_max = map(int, box[:4])

    # Crop the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]
    # Display the cropped image
    cv2.imshow(f"Cropped Image {i + 1}", cropped_image)

    # Save the cropped image
    output_path = os.path.join(output_folder, f"cropped_image_{i + 1}.png")
    cv2.imwrite(output_path, cropped_image)

    # Perform OCR on the cropped image using EasyOCR
    results = reader.readtext(cropped_image)

    # Process each recognized text
    text0 = ""
    text1 = ""
    for j, (bbox, text, prob) in enumerate(results):
        print(f"Index: {j}, Text: {text}")
        if j == 0:
            text0 = text
        elif j == 1:
            text1 = text

    # Concatenate the texts
    textResult = text0 + text1
    print(f"Result: {textResult}")
    # Draw bounding box for all, but put text on the original image only for index 1
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, textResult, (x_min - 100, y_min - 50), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# Display the original image with bounding boxes and text
cv2.imshow("Detected License Plate", image)

# Wait for user input to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
