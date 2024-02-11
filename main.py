import cv2
from ultralytics import YOLO
import os
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract\tesseract.exe'


# Load a model
model = YOLO(r"D:\python\Licence_Plate\detect\train2\weights\best.pt")  # Load a pretrained model
results = model(r"D:\images\pp_plate.jpg",  show=False)  # Set show=False to suppress YOLOv8 training visualization

# Assuming there is only one image in the batch
image_index = 0
image_info = results[image_index]

# Retrieve bounding boxes
boxes = image_info.boxes.data.cpu().numpy()

# Load the image
image_path = r"D:\images\pp_plate.jpg"
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

    # Convert the cropped image to grayscale
    img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded_img = cv2.threshold(img_gray, 67, 255, cv2.THRESH_BINARY)

    # Perform OCR on the cropped image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresholded_img, config=custom_config)
    # Split into three parts (30F, 883, 68)
    part1 = text[:4]
    part2 = text[4:11]
    part3 = text[11:]

    print("Part 1:", part1)
    print("Part 2:", part2)
    print("Part 3:", part3)

    # Draw bounding box and put text on the original image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, part2, (x_min - 100, y_min - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the recognized text
    print(f"Recognized text in Cropped Image {i + 1}: {text}")

# Display the original image with bounding boxes and text
cv2.imshow("Detected License Plate", image)

# Wait for user input to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
