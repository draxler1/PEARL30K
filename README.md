# CLIP_PEARL_30K
[[Paper]](https://openaccess.thecvf.com/content/WACV2025/papers/Vijay_CLIPping_Imbalances_A_Novel_Evaluation_Baseline_and_PEARL_Dataset_for_WACV_2025_paper.pdf) [[Additional Document]](https://openaccess.thecvf.com/content/WACV2025/supplemental/Vijay_CLIPping_Imbalances_A_WACV_2025_supplemental.pdf)

Official repository for "CLIPping Imbalances: A Novel Evaluation Baseline and PEARL Dataset for Pedestrian Attribute Recognition", WACV-24"

## Introduction

The PEARL dataset comprises with 30K pedestrian images, each annotated with 25 attribute categories, spanning over 146 sub-attributes. We have collected images from outdoor surveillance that reflect practical applications and challenges. We comprehensively cover nearly all critical attributes relevant to security surveillance applications, comprising aspects such as body posture, accessories, bag types, clothing styles, colors, and activities. To diversify, we have extracted images from twelve countries that covers seven distinct public locations including streets, parks, airports, stations, college campuses, beaches, and marketplaces. Additionally, we have incorporated four distinct weather conditions: sunny, night-time, rainy, and snow.

![figure1](assests/Attribute_wise_dataset_sample.png)

Below are attributes covered in PEARL30K:

```python
# Attribute List:
PEARL30K = {
    'Gender': ['Male', 'Female', 'Other', 'NO#'],
    'Age': ['Adult', 'Young', 'Teenager', 'Old', 'NO#'],
    'Hair': ['Short', 'Long', 'Bald', 'NO#'],
    'HColor': ['Black', 'Blonde', 'White', 'NA', 'Other', 'NO#'],
    'SLength': ['Short', 'Long', 'No Sleeve', 'NA', 'NO#'],
    'Viewpoint': ['Front', 'Back', 'Left', 'Right', 'NO#'],
    'IsOccluded': ['No', 'Yes'],
    'Activity': ['Standing', 'Walking', 'Running', 'Seating', 'Cellphoning', 'Cycling', 'Pooling', 'Talking', 'NO#'],
    'BodyPos': ['Standing', 'Seating'],
    'WhichOccluded': ['NA', 'Head-Shoulder', 'Lower-body', 'Upper-Body'],
    'Hat': ['No', 'Yes', 'NO#'],
    'Glasses': ['No', 'Yes', 'NO#'],
    'Bag': ['No', 'Yes', 'NO#'],
    'BagType': ['NA', 'Handbag', 'Backpack', 'Plastic Bag', 'Suitcase', 'Trolly', 'Shoulder-Bag', 'Other', 'NO#'],
    'BodyShape': ['Thin', 'Normal', 'Fat', 'NO#'],
    'Boots': ['Yes', 'No', 'Other', 'NO#'],
    'BootsColor': ['Black', 'White', 'Blue', 'Gray', 'Other', 'NA', 'NO#'],
    'FaceMask': ['No', 'Yes', 'NO#'],
    'Weather': ['Day', 'Night', 'Rain', 'Snow', 'NO#'],
    'Height': ['Normal', 'Short', 'Tall', 'NO#'],
    'Accessories': ['Nothing', 'Cellphone', 'Trolly', 'Umbrella', 'BiCycle', 'Bike', 'Other', 'NO#'],
    'UpperBody': ['T-shirt', 'Shirt', 'Dress', 'Jacket', 'Suit', 'Coat', 'Sweater', 'Formal', 'No Cloth', 'Burqa', 'Saree', 'Other', 'NO#'],
    'UpperColor': ['Black', 'White', 'Blue', 'Red', 'Green', 'Brown', 'Grey', 'Orange', 'Pink', 'Purple', 'Yellow', 'NA', 'Other'],
    'LowerBody': ['Shorts', 'Jeans', 'Pants', 'Skirt', 'Burqa', 'Saree', 'Other', 'NO#'],
    'LowerColor': ['Black', 'White', 'Blue', 'Red', 'Green', 'Brown', 'Grey', 'Orange', 'Pink', 'Purple', 'Yellow', 'NA', 'Other']
}
```

## Sample Inference Script

STEP 1: Please install below requirements:

```bash
$ pip install git+https://github.com/openai/CLIP.git
$ pip install pillow numpy pandas
$ pip install torch
```

STEP 2: Download mini-sample model and set appropriate paths

```python
saved_model = '../saved_models/clip_top3_50_VT32.pt'
image_folder = "../sample images/"
caption_csv = "first_three_annotation.csv"
```

STEP 3: Run the `inference_clip_PEARL.py` script:

```bash
$ python inference_clip_PEARL.py
```

Expected Output:

```python
```

