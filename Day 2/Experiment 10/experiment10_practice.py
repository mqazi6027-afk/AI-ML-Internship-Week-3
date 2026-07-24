
import cv2

print("=" * 65)
print("      M-Tech AI/ML Internship - Experiment 10")
print(" Computer Vision - Image Processing using OpenCV")
print("=" * 65)

image = cv2.imread("sample_image.jpg")

if image is None:
    print("\nError: sample_image.jpg not found.")
    print("Place the image in the same folder as this program.")
    exit()

print("\nImage Loaded Successfully")

height, width, channels = image.shape

print("Image Width  :", width)
print("Image Height :", height)
print("Channels     :", channels)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

resized = cv2.resize(image, (300, 300))

blur = cv2.GaussianBlur(image, (7,7), 0)

edges = cv2.Canny(gray,100,200)

cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("resized.jpg", resized)
cv2.imwrite("blur.jpg", blur)
cv2.imwrite("edges.jpg", edges)

print("\nProcessed Images Saved Successfully")

print("\nGenerated Files")
print("-------------------------")
print("grayscale.jpg")
print("resized.jpg")
print("blur.jpg")
print("edges.jpg")

print("\nExperiment 10 Completed Successfully")
print("=" * 65)