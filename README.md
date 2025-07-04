# CLIP_PEARL_30K
[[Paper]](https://openaccess.thecvf.com/content/WACV2025/papers/Vijay_CLIPping_Imbalances_A_Novel_Evaluation_Baseline_and_PEARL_Dataset_for_WACV_2025_paper.pdf) [[Additional Document]](https://openaccess.thecvf.com/content/WACV2025/supplemental/Vijay_CLIPping_Imbalances_A_WACV_2025_supplemental.pdf)

Official repository for "CLIPping Imbalances: A Novel Evaluation Baseline and PEARL Dataset for Pedestrian Attribute Recognition", WACV-24"

## Introduction

The PEARL dataset comprises with 30K pedestrian images, each annotated with 25 attribute categories, spanning over 146 sub-attributes. We have collected images from outdoor surveillance that reflect practical applications and challenges. We comprehensively cover nearly all critical attributes relevant to security surveillance applications, comprising aspects such as body posture, accessories, bag types, clothing styles, colors, and activities. To diversify, we have extracted images from twelve countries that covers seven distinct public locations including streets, parks, airports, stations, college campuses, beaches, and marketplaces. Additionally, we have incorporated four distinct weather conditions: sunny, night-time, rainy, and snow.

```python
# Attribute List:
PEARL30K = {
    'Gender': ['Male', 'Female', 'Other', 'NO#'],
    'Age': ['Adult', 'Young', 'Teenager', 'Old', 'NO#'],
    'Hair': ['Short', 'Long', 'Bald', 'NO#'],
    'HColor': ['Black', 'Blonde', 'White', 'NA', 'Other', 'NO#'],
    'SLength': ['Short', 'Long', 'No Sleeve', 'NA', 'NO#'],
    'Viewpoint': ['Front', 'Back', 'Left', 'Right', 'NO#'],
    'IsOccluded': ['No'],
    'IsOccluded': ['Yes'],  # Note: 'Is Occluded' and 'IsOccluded' may need normalization
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
