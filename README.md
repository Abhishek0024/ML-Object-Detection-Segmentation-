# 🧠 Object Detection and Segmentation with Mask R-CNN

This project performs **object detection** and **instance segmentation** on images using a pre-trained **Mask R-CNN** model from PyTorch's `torchvision` library.

---

## 🔍 What It Does

- Detects objects in an image.
- Draws bounding boxes around each object.
- Applies colored masks over the detected objects for segmentation.

---

## 📁 Project Structure

```
📦 MaskRCNN-Segmentation
├── segmentation_detection.py       # Main Python script
├── Image_brain.jpg                 # Sample image 1
└── Image_dogs.jpg                  # Sample image 2
```

---

## 🧰 Requirements

Make sure you have the following Python packages installed:

- `torch`
- `torchvision`
- `opencv-python`
- `matplotlib`
- `Pillow`
- `numpy`

You can install them all at once using:

```bash
pip install torch torchvision opencv-python matplotlib pillow numpy
```

---

## ▶️ How to Run

1. Place your test image(s) in the same folder as `segmentation_detection.py`.
2. Open the script and update the `image_path` to the filename you want to test:
   ```python
   image_path = "Image_dogs.jpg"  # or "Image_brain.jpg"
   ```
3. Run the script:

```bash
python segmentation_detection.py
```

4. A window will appear displaying:
   - Bounding boxes around detected objects
   - Colored masks over segmented regions

---

## 📸 Sample Output

> 🐶 **Dogs Example Output:**  
> Displays detected dogs with green overlays for segmentation.

> 🧠 **Brain Example Output:**  
> (If used with MRI-type images) Shows segmentation over detected brain areas — for demo use only.

---

## 🧠 Model Used

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) with ResNet-50 FPN backbone
- Pretrained on **COCO dataset** — 80 common object categories

---

## 📌 Notes

- This is for educational/demo purposes.
- Results depend on image quality and if objects are present in COCO dataset categories.
- You can easily swap the image or modify the confidence threshold (`threshold = 0.5`) in the script.

---

## 📜 License

This project is under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Abhishek Rana – [@Abhishek0024](https://github.com/Abhishek0024)
