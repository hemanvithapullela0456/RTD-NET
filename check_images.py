import os
from PIL import Image, ImageFile

# Helps detect truncated images too
ImageFile.LOAD_TRUNCATED_IMAGES = False

# CHANGE THIS PATH
dataset_path = r"E:\RTD-Net\data\aid"

corrupted = []
total = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
            total += 1
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()   # verify header
                with Image.open(path) as img:
                    img.load()     # fully load image data
            except Exception as e:
                corrupted.append((path, str(e)))
                print(f"[CORRUPTED] {path}")
                print(f"   Error: {e}")

print("\n========== SUMMARY ==========")
print(f"Total checked: {total}")
print(f"Corrupted found: {len(corrupted)}")

# Save report
with open("corrupted_images_report.txt", "w", encoding="utf-8") as f:
    for path, err in corrupted:
        f.write(f"{path}\n{err}\n\n")

print("\nReport saved as corrupted_images_report.txt")