# Advanced Python Assignment – NumPy & Image Processing

This repository contains a high-level homework assignment that demonstrates advanced use of **NumPy** for data analysis and **image processing techniques** using Gaussian and Laplacian pyramids.

## 📁 Contents

- `Hw.py`: The full implementation, split into:
  - **Part 1**: Weight-loss participant analysis using NumPy.
  - **Part 2**: Image blending using pyramids, with grayscale and RGB support.
- `ex6.pdf`: The original assignment description (in Hebrew).

---

## 🔬 Part 1 – NumPy Data Analysis

Analyzes monthly weight data of participants in a clinical trial.

### Key functions:
- `get_highest_weight_loss_participant(...)`: Find participant with the most weight loss.
- `get_diff_data(...)`: Compute month-to-month weight differences.
- `get_highest_change_month(...)`: Find the month with the highest total weight change.
- `get_inconsistent_participants(...)`: Identify participants who didn’t consistently lose weight.

---

## 🖼️ Part 2 – Image Blending with Pyramids

Implements image blending using Gaussian and Laplacian pyramids.

### Key features:
- `blur_and_downsample(...)` / `upsample_and_blur(...)`: Preprocessing functions.
- `build_gaussian_pyramid(...)` / `build_laplacian_pyramid(...)`: Pyramid construction.
- `laplacian_pyramid_to_image(...)`: Reconstruction of the original image.
- `pyramid_blending(...)`: Grayscale image blending.
- `pyramid_blending_RGB_image(...)`: Bonus – color image blending using all RGB channels.

---

## ✅ Tests

The script includes built-in validation and visualization:
- Output comparisons for all parts.
- Toggle image display with `plot_flag`.

---

## 🛠️ Technologies

- Python 3
- NumPy
- Matplotlib
- imageio

---

## 📸 Example Result

Blending an apple and an orange using a binary mask:

<img src="https://github.com/GalAngel15/numpy-Image-Processing/blob/main/mask.png" alt="Mask" style="height:300px;"/>

<div class="row">
<img src="https://github.com/GalAngel15/numpy-Image-Processing/blob/main/orange.png" alt="Orange" style="height:300px;"/>
<img src="https://github.com/GalAngel15/numpy-Image-Processing/blob/main/apple.png" alt="Apple" style="height:300px;"/>
</div>

<img src="https://github.com/GalAngel15/numpy-Image-Processing/blob/gh-pages/orapple_naive.png" alt="Answer" style="height:300px;"/>

---

## 📝 Notes

- No external packages beyond NumPy and Matplotlib.
- Written as part of a university-level software engineering course.

---

## 🧠 Author

Gal Angel – Software Engineering Student | Puzzle Solver | Tech Enthusiast
