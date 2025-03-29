# Dental Caries Augmented Dataset V4

This is the augmented version of the dental caries dataset, specifically designed to address class imbalance issues through various data augmentation techniques.

## Dataset Structure

```
dental_ai_dataset_v4_augmented/
├── binary_classification/
│   └── train/
│       ├── normal/    (1290 images)
│       └── caries/    (258 images)
├── three_level_classification/
│   └── train/
│       ├── normal/        (15 images)
│       ├── superficial/   (204 images)
│       ├── medium/        (204 images)
│       └── deep/          (258 images)
├── binary_segmentation/
│   └── train/
│       ├── images/
│       └── masks/
└── detailed_segmentation/
    └── train/
        ├── images/    (344 images)
        └── masks/     (344 masks)
```

## Class Distribution After Augmentation

### Binary Classification
- Total Images: 1,548
- Distribution:
  * Normal: 1,290 images (83.3%)
  * Caries: 258 images (16.7%)

### Three-Level Classification
- Total Images: 681
- Distribution:
  * Normal: 15 images (2.2%)
  * Superficial: 204 images (30.0%)
  * Medium: 204 images (30.0%)
  * Deep: 258 images (37.9%)

### Detailed Segmentation
- Total Images: 344
- Total Regions: 856
- Region Distribution:
  * Superficial: 316 regions (36.9%)
  * Medium: 282 regions (32.9%)
  * Deep: 258 regions (30.1%)

## Augmentation Techniques Applied

### For Classification Tasks
1. Geometric Transformations:
   - Random rotation (±20 degrees)
   - Random horizontal/vertical flips
   - Random translations (±10%)
   - Grid distortion (p=0.2)

2. Intensity Transformations:
   - Random brightness/contrast adjustments
   - Random gamma correction
   - Gaussian noise (σ=0.01)

### For Segmentation Tasks
1. Synchronized Image-Mask Transformations:
   - Random rotation
   - Random flips
   - Controlled elastic deformation
   - Grid distortion

2. Mask Value Preservation:
   - Background: 0
   - Superficial caries: 102
   - Medium caries: 153
   - Deep caries: 255

## File Naming Convention
- Original files: `[id].png`
- Augmented files: `[id]_aug_[number].png`
- All augmented images maintain correspondence between:
  * Original image and its augmentations
  * Images and their corresponding masks (for segmentation)

## Notes on Augmented Data
1. Augmentation was applied only to the training set
2. Validation and test sets remain unchanged
3. All augmented images maintain the same resolution (512x512)
4. Pixel value integrity is preserved in segmentation masks
5. Class balance has been significantly improved

## Usage Guidelines

### For Classification
- Use the entire augmented dataset for training
- Maintain the original validation and test sets
- Consider using weighted sampling if needed

### For Segmentation
- Images and masks are paired by filename
- All transformations are applied consistently to both
- Mask values are preserved for severity levels

## Validation
The augmented dataset has been validated for:
1. Pixel value integrity in masks
2. Image-mask correspondence
3. Proper class distribution
4. Image quality preservation

## Original Dataset Reference
This augmented dataset is derived from dental_ai_dataset_v4. For original dataset details and citations, please refer to the main dataset documentation. 