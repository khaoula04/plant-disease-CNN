# Plant Disease Detection with Deep Learning

> A CNN-based web app that identifies plant diseases from leaf images, built with transfer learning on MobileNetV2.

---

## Overview
This project classifies leaf images into 38 disease categories across 14 crop species, trained on the New Plant Diseases Dataset (~87K images). The model was trained in two phases : 
- first training a custom classification head on frozen MobileNetV2 weights,
-  then fine-tuning the base model with a lower learning rate, reaching 98.98% validation accuracy.

-> A Flask web app lets users upload a leaf photo and get an instant prediction with a confidence score. Predictions below a confidence threshold are flagged as "unrecognized" rather than forced into a wrong category.

---

**Tech stack:** Python, TensorFlow/Keras, Flask, scikit-

---

## Limitations
The model performs very well on lab-style images (plain background, single leaf) but is less reliable on real-world field photos, due to a distribution shift between the training data and natural conditions, this is a known limitation of this dataset, confirmed through testing on external images.

## Run it locally

```
bash
pip install -r requirements.txt
python app.py
```

runs on `http://127.0.0.1:5000`
