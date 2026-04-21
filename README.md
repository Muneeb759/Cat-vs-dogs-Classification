# Cat vs Dogs Classification (Deep Learning)

This project builds and evaluates image classifiers for the Kaggle Dogs vs Cats dataset using TensorFlow/Keras in a Jupyter notebook.

## Project Files

- `cat_dog.ipynb`: Full notebook (data prep, model training, evaluation)
- `convnet_from_scratch.keras`: Saved best model checkpoint
- `sampleSubmission.csv`: Kaggle sample submission file
- `cats_vs_dogs_small/`: Prepared train/validation/test and training/testing folders

## Tools and Libraries Used

- Python 3
- Jupyter Notebook
- Kaggle API (`kaggle` CLI)
- TensorFlow / Keras
- scikit-learn
- NumPy
- Matplotlib
- Standard library modules: `os`, `shutil`, `pathlib`, `zipfile`

## Dataset and Data Handling Techniques

- Downloaded data from Kaggle using the command line API.
- Extracted zipped dataset archives with `zipfile`.
- Built custom folder-based subsets using a helper function (`make_subset`) and file copy operations.
- Used two directory-based input pipelines:
  - `image_dataset_from_directory` (TensorFlow `tf.data` style)
  - `ImageDataGenerator.flow_from_directory` (Keras generator style)
- Applied image rescaling (`1./255`) for normalization.
- Applied data augmentation (shear, zoom, horizontal flip) in generator-based training.

## Modeling Techniques Used

### 1) CNN from Scratch

- Functional API model with:
  - `Rescaling`
  - Multiple `Conv2D + MaxPooling2D` blocks
  - `Flatten`
  - `Dense(1, sigmoid)` output for binary classification
- Compiled with:
  - Loss: `binary_crossentropy`
  - Optimizer: `rmsprop`
  - Metric: `accuracy`

### 2) Transfer Learning (VGG16)

- Loaded pretrained `VGG16` (`weights='imagenet'`, `include_top=False`).
- Added custom classifier head:
  - `Flatten`
  - `Dense(256, relu)`
  - `Dense(1, sigmoid)`
- Froze convolutional base (`conv_base.trainable = False`).
- Compiled with optimizer `adam` and trained with generators.

## Training and Validation Techniques

- Used model checkpointing:
  - `ModelCheckpoint(filepath='convnet_from_scratch.keras', save_best_only=True, monitor='val_loss')`
- Trained for multiple epochs and tracked train/validation metrics.
- Visualized training curves for:
  - Accuracy vs validation accuracy
  - Loss vs validation loss

## Testing and Evaluation Techniques

- Loaded saved best model and evaluated on held-out test data.
- Reported:
  - Test loss
  - Test accuracy
- Generated prediction probabilities and thresholded outputs (`>= 0.5`) for class labels.
- Used scikit-learn metrics:
  - Confusion matrix (`confusion_matrix`)
  - Precision/recall/F1 report (`classification_report`)

## Workflow Summary

1. Download and extract Kaggle dataset.
2. Organize images into structured subsets.
3. Train baseline CNN from scratch.
4. Train a transfer learning model with VGG16.
5. Save best model checkpoints.
6. Plot training history.
7. Evaluate on unseen test data with accuracy and classification metrics.

## Notes

- The notebook contains both a `train/validation/test` split and a separate `training/testing` split for different experiments.
- For final reporting, always keep a true unseen test set and avoid tuning on test data.
