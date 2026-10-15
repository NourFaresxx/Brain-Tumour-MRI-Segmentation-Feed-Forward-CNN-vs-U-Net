Brain Tumour MRI Segmentation: Feed-Forward CNN vs U-Net
This project compares two convolutional networks on segmenting lower-grade glioma (brain tumour) regions in MRI slices:

Feed-Forward CNN: a simple encoder-decoder with no skip connections (the baseline).
U-Net: an encoder-decoder with skip connections between matching encoder and decoder levels.
Both models are trained and evaluated under identical conditions, so the difference in results comes from the architecture.

Results
Test set: 17 held-out patients (561 slices).

Model	Parameters	Test loss	Soft Dice	Hard Dice (tumour slices)
Feed-Forward CNN	168 K	0.679	0.411	0.399
U-Net	1.87 M	0.455	0.624	0.564
The U-Net reaches a Dice of 0.56 on slices that contain a tumour, compared with 0.40 for the baseline (about 41% better relative to the baseline). Its skip connections carry fine spatial detail straight from the encoder to the decoder, which gives sharper tumour boundaries and fewer missed small lesions.

Metrics

Soft Dice: Dice computed on the raw sigmoid outputs, as reported by model.evaluate.
Hard Dice (tumour slices): predictions are thresholded at 0.5, Dice is computed for each image, and the average is taken only over slices whose ground truth contains a tumour. This is the main comparison metric, because about two thirds of the slices are empty, and an empty prediction on an empty slice scores a perfect 1.
Dataset
LGG MRI Segmentation (Kaggle, Buda et al.)

3,929 brain MRI slices from 110 patients with lower-grade glioma
Each slice <name>.tif has a hand-drawn tumour mask <name>_mask.tif
About one third of the slices contain a tumour
The dataset is not included in this repository. Download it from Kaggle as a zip.

Method
Preprocessing

Images resized to 128×128 and scaled to [0, 1]. Masks resized with nearest-neighbour interpolation and binarised.
The data is split by patient into 70/15/15 train/validation/test sets (77 / 16 / 17 patients). Neighbouring slices from the same patient are nearly identical, so splitting by slice would leak near-duplicate images into the test set.
Training setup (shared by both models)

Setting	Value
Loss	Binary cross-entropy + (1 − Dice)
Optimizer	Adam, learning rate 1e-4
Epochs	15
Batch size	8
Random seed	42
The combined BCE + Dice loss matters here. An earlier version trained the U-Net with pure Dice loss and no BatchNorm, and it collapsed to predicting background everywhere (Dice ≈ 0).

Architectures

Feed-Forward CNN: Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Upsample → Conv(64) → Upsample → 1×1 Conv (sigmoid). Each conv is followed by BatchNorm and ReLU.
U-Net: two encoder levels (64, 128 filters), a 256-filter bottleneck, and two decoder levels that use transposed convolutions and are concatenated with the matching encoder features. Each block is two 3×3 convs, each followed by BatchNorm and ReLU, with He initialisation.
How to run
The notebook is designed for Google Colab.

Open acl2_5_final.ipynb in Colab.
Go to Runtime → Change runtime type → GPU (T4 is enough).
Run the upload cell and select the Kaggle zip. For large files, uploading through the Files panel or Google Drive is more reliable.
Set zip_path in the unzip cell to the uploaded file's name.
Choose Runtime → Run all.
On a T4 GPU, training takes about 2 minutes for the Feed-Forward CNN and about 10 minutes for the U-Net.

Requirements: Python 3, TensorFlow 2.x, NumPy, OpenCV, Pillow, scikit-learn, matplotlib, tqdm. These are all preinstalled on Colab.
