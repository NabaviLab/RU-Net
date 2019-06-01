# Residual Deep Learning System for Mass Segmentation and Classification in Mammography

# Input

-Create folder  "All_large_patches/Test_dataset/" under the main folder.

Test_dataset containes two folder

           1-labels 2-patches
           
labels folder has ground truth maps

patches folder has mammogram images


# Output

use Test_semantic_segmentation() to find the accuracy in terms of IOU and Dice index of the trained deep learning model.

# Training models

-train residual attention UNET model use Train_residual_attention_UNET()

-train residual attention Segnet model use Train_residual_attention_Segnet()

-train residual attention FCN model useTrain_residual_attention_FCN()

-train residual UNET model useTrain_residual_UNET()

-train basic UNET model use Train_basic_UNET()

-train rbasic Segnet model use  Train_basic_Segnet()

-train basic FCN model use Train_basic_FCN()

-train basic dilution-CNN model use Train_basic_dilution()

