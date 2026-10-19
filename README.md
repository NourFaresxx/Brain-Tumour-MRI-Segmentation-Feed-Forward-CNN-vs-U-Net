# Brain Tumour MRI Segmentation: Feed-Forward CNN vs U-Net

Pixel-level segmentation of low-grade glioma tumours in brain MRI scans. The project compares a simple encoder–decoder CNN against a U-Net with skip connections, trained and evaluated under identical conditions.

The U-Net reaches a **0.66 Dice score on tumour slices, 32% higher than the baseline (0.50)**.

![Predictions](images/predictions.png)
*Rows 1–2: the U-Net follows the tumour's irregular boundary, while the baseline predicts a rough blob. Row 3: on a slice with no tumour, the baseline produces false positives and the U-Net correctly predicts nothing.*

## Results

Evaluated on a held-out test set of 561 slices from 17 patients the models never saw during training.

| Model | Parameters | Dice (tumour slices) | Dice (all slices) |
|---|---|---|---|
| Feed-forward CNN | 168K | 0.498 | 0.650 |
| **U-Net** | 1.87M | **0.657** | **0.795** |

**Which score to look at:** about two-thirds of slices contain no tumour, and an empty prediction on an empty slice counts as a perfect score. This inflates the "all slices" Dice, so the models are compared on **tumour slices only**.

![Training curves](images/training_curves.png)

## Dataset

[LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Kaggle): **3,929 brain MRI slices from 110 patients**, each with a manual tumour mask. 33% of slices contain a tumour.

![Sample MRI and mask](images/sample_mri.png)

## Method

**Patient-level split.** Patients are split 77 / 16 / 17 into train / validation / test, so slices from one patient never appear in both training and testing. Splitting by slice instead would leak near-identical images into the test set and inflate the results.

**Models**
- **Feed-forward CNN (baseline):** a 4-block encoder–decoder with no skip connections.
- **U-Net:** a 2-level encoder–decoder with skip connections that pass fine spatial detail from the encoder to the decoder. This is what lets it trace tumour boundaries accurately.

Both use 3×3 convolutions with batch normalization and He initialization.

**Training** (the same for both models)
- **Loss:** binary cross-entropy + Dice loss. Using Dice loss alone made the models collapse to predicting only background; adding BCE fixed this.
- **Augmentation:** random horizontal and vertical flips, applied identically to each image and its mask.
- **Optimization:** Adam (learning rate 1e-4), halved when validation Dice plateaus.
- **Early stopping:** up to 50 epochs, stopping after 8 epochs without improvement and restoring the best weights. The U-Net's best epoch was 37; the baseline's was 27.
- **Input:** 128×128 RGB, batch size 8.

## Running it

The notebook is designed for **Google Colab with a GPU runtime**.

1. Download the dataset zip from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
2. Open `brain_tumour_segmentation.ipynb` in Colab and set **Runtime → Change runtime type → GPU**.
3. Run all cells and upload the zip when prompted.

A full run takes about 30 minutes on a Colab T4 GPU.

**Built with:** Python, TensorFlow/Keras, OpenCV, NumPy, scikit-learn, Matplotlib

