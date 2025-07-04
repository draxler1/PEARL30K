import os
import random
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import clip

# ------------------ Paths -------------------
saved_model = '../saved_models/clip_top3_50_VT32.pt'
image_folder = "../sample images/"
caption_csv = "first_three_annotation.csv"  # Updated CSV with gender, age, hair

# ------------------ Device & Model -------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load(saved_model)
model.load_state_dict(checkpoint)
model.eval()

# ------------------ Attribute Options -------------------
ATTRIBUTE_OPTIONS = {
    "Gender": ["Male", "Female", "NO#"],
    "Age": ["Adult", "Young", "Teenager", "Old", "NO#"],
    "Hair": ["Short", "Long", "Bald", "NO#"]
}

# ------------------ Functions -------------------
# N control number of distractor prompts

def generate_distractors(true_values, n=1):
    distractors = []
    for _ in range(n):
        d = {}
        for attr, true_val in true_values.items():
            options = [v for v in ATTRIBUTE_OPTIONS[attr] if v != true_val]
            d[attr] = random.choice(options)
        distractors.append(d)
    return distractors

def make_caption(attr_values):
    parts = [f"{attr} {val}" for attr, val in attr_values.items()]
    if len(parts) > 1:
        return "A photo of a person with " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."
    elif parts:
        return "A photo of a person with " + parts[0] + "."
    else:
        return "A photo of a person."

def extract_attributes_from_caption(caption):
    attr_values = {}
    caption = caption.strip(".").replace("A photo of a person with ", "")
    items = [x.strip() for x in caption.replace(" and ", ", ").split(",")]
    for item in items:
        if ' ' in item:
            attr, val = item.split(' ', 1)
            attr_values[attr.lower()] = val
    return attr_values

def print_probabilities(probabilities, captions):
    for i, prob in enumerate(probabilities[0]):
        print(f"{captions[i]}: {prob:.4f}")

# ------------------ Load Attributes CSV -------------------
df = pd.read_csv(caption_csv)
filenames = df["Filename"].tolist()

# ------------------ Random Sample -------------------
selected_idx = random.randint(0, len(filenames) - 1)
filename = df.loc[selected_idx, "Filename"]

# Get true attribute values
true_attr_values = {
    "Gender": df.loc[selected_idx, "Gender"],
    "Age": df.loc[selected_idx, "Age"],
    "Hair": df.loc[selected_idx, "Hair"]
}

# Construct ground-truth caption
ground_truth_caption = make_caption(true_attr_values)

image_path = os.path.join(image_folder, filename)
print(f"Selected Image: {filename}")
print(f"Ground Truth Caption: {ground_truth_caption}")

# ------------------ Create Prompts -------------------
distractor_attrs = generate_distractors(true_attr_values, n=1)
distractor_captions = [make_caption(attrs) for attrs in distractor_attrs]

all_captions = [ground_truth_caption] + distractor_captions

# ------------------ Run Inference -------------------
image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
text = clip.tokenize(all_captions).to(device)

with torch.no_grad():
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# ------------------ Display Results -------------------
print("\nPrediction Probabilities:")
print_probabilities(probs, all_captions)

predicted_caption = all_captions[np.argmax(probs)]
predicted_attrs = extract_attributes_from_caption(predicted_caption)

print("\nPredicted Attributes:")
for k, v in predicted_attrs.items():
    print(f"  {k.capitalize()}: {v}")

print("\nGround Truth Attributes:")
for k, v in true_attr_values.items():
    print(f"  {k.capitalize()}: {v}")

