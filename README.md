# 🏡 Multi-Modal Real Estate Valuation Using Satellite Imagery and Tabular Data

## Project Report

# 1. Core Problem

The primary objective of this project is to move beyond standard, one-dimensional house-price modeling by systematically capturing the "environmental context" of a property.

Traditional real estate valuation models rely heavily on structural attributes (e.g., square footage, bedroom count), often missing the intangible geospatial elements that drive market desirability—such as neighborhood green cover, road density, and immediate curb appeal.

This project solves this by designing a multimodal regression pipeline that programmatically acquires localized satellite imagery and fuses these unstructured visual signals with structured historical housing data into a unified predictive framework. 🌍

---

# 2. About Dataset

The project utilizes a hybrid dataset consisting of both historical tabular records and programmatically fetched high-resolution satellite images.

### 📊 Tabular Data

The structured dataset includes **16,209 records** containing core property attributes such as:
price, bedrooms, bathrooms, sqft_living, grade, condition, yr_built, yr_renovated, zipcode, latitude, longitude etc

### 🛰️ Visual Data

Using the spatial coordinates provided in the tabular dataset, a custom image-fetching pipeline was engineered to download **16,209 corresponding training images**.

These were fetched at a high-resolution **Zoom Level of 18** using the **Esri World Imagery (ArcGIS MapServer)** infrastructure.

---

# 3. My Strategy

The modeling approach hinges on a **Late-Fusion Multimodal Architecture powered by XGBoost**.

The strategy is divided into three distinct phases:

1. **Tabular Feature Engineering**
   Extracting deeper economic and temporal signals from raw timestamps and location tags.

2. **Visual Signal Extraction**
   Leveraging Transfer Learning via a pre-trained Convolutional Neural Network (CNN) to process satellite imagery without needing a massive, manually labeled dataset.

3. **Dimensionality Reduction & Fusion**
   Applying Principal Component Analysis (PCA) to compress high-dimensional image embeddings to prevent over-fitting, merging them with tabular data, and optimizing the final XGBoost regression model using Optuna.

---

# 4. Exploratory Data Analysis (EDA)

Initial data exploration revealed key statistical and geospatial insights that dictated the preprocessing pipeline:

#### 📈 Target Variable Distribution

The target variable (**price**) exhibits a strong right-skewed distribution.

While log transformation (`np.log1p`) can be applied to normalize the variance, tree-based models like XGBoost are generally robust to monotonic transformations and handle this skew natively.

#### 🌊 Waterfront Proximity

Properties possessing a waterfront score of **1** commanded a distinct, quantifiable price premium across the dataset.

#### 🏘️ Neighborhood Density Influence

Features such as **sqft_living15** confirmed that a property's market value is heavily anchored by the scale and quality of its immediate neighbors, validating the hypothesis that local spatial context is a critical predictor.

---
<img width="1127" height="533" alt="image" src="https://github.com/user-attachments/assets/75981ef3-5a30-47d6-9464-ce96fb95876c" />
<img width="1122" height="532" alt="image" src="https://github.com/user-attachments/assets/b3a8f716-9e9f-49d3-8913-7bbf833ea9c4" />


---

# 5. Valuation Using Tabular Data

Before introducing imagery, a robust tabular baseline was established.

Raw features were mathematically transformed to expose underlying economic drivers:

#### 📅 Temporal Decomposition

The raw date string was broken down into cyclical and linear numeric components:

* sale_year
* sale_month
* day_of_month
* day_of_week

This allows the model to map inflation adjustments and seasonal market trends rather than treating the sale date as a static string.

#### 📍 Zipcode Target Encoding

Treating zip codes as high-cardinality categorical variables often leads to sparse tree splits.

Instead, target encoding mapped each zip code to its median training property price, generating a continuous **"Location Index"** that directly quantifies a neighborhood's economic baseline.

#### 🏠 Effective Built Year & House Age

A structure's age isn't strictly defined by its original construction.

An **effective_built_year** was calculated:

* If `yr_renovated > 0`, the renovation year superseded the build year.

The final **house_age** feature accurately reflected modern wear-and-tear.

An XGBoost regressor, heavily tuned via an Optuna hyperparameter search, was trained on this engineered tabular set.

The pure tabular model achieved a baseline:

| Model                   | RMSE       | R²   |
| ----------------------- | ---------- | ---- |
| Baseline (Tabular Only) | 109,575.09 | 0.88 |

---

# 6. Valuation Using Multi-Modal Output

To inject the "environmental context" into the valuation, the visual data pipeline was constructed with significant mathematical and architectural rigor.

### 🗺️ Geospatial Mapping & Acquisition

Raw latitude and longitude coordinates were mapped to Tile Coordinates via the Mercator Projection.

By mathematically converting degrees to radians and applying logarithmic scaling mapping, exact image tiles representing the property's roof and immediate yard were downloaded programmatically.

### 🧠 High-Dimensional Embedding Extraction

Images were resized to **380 × 380** and passed through an **EfficientNet-B4** architecture.

The network's final classification layer was stripped, turning the model into a pure feature extractor.

By tapping into the **global_average_pooling** layer, the model output a raw vector of **1,792 dimensions** per property.

This allowed the pipeline to inherit the network's visual common sense—recognizing shapes, textures, road density, and green cover.

### 📉 Visual Signal Compression (PCA)

Directly feeding 1,792 visual features alongside a handful of tabular features would trigger the **Curse of Dimensionality**, overwhelming the XGBoost model with noise.

To compress the signal:

* Embeddings were standardized.
* PCA was applied.
* Variance retention threshold was targeted at **40%**.

The process successfully condensed the **1,792 features down to the 12 most significant principal components**.

This achieved a **99% reduction in feature space** while safely retaining the core visual markers.

### ⚙️ Late-Fusion & Tuning

The 12 continuous PCA features were horizontally concatenated with the engineered tabular dataset.

A final extensive Optuna sweep over **2,500 estimators** stabilized the complex multimodal architecture, settling on highly regularized parameters such as:

* L1 regularization (`reg_alpha = 19.11`)
* Depthwise tree growth
* Learning rate ≈ 0.051

to control the fused dataset.

---

# 7. Performance and Conclusion

The comparative cross-validation performance of the two architectures yielded the following metrics:

## 📊 Improvement of Multimodal Model over Baseline

| Metric         | Improvement      |
| -------------- | ---------------- |
| RMSE Reduction | 1,838.70 (1.68%) |
| R² Increase    | 0.01 (1.14%)     |

### Final Results

| Model                                       | RMSE       | R²   |
| ------------------------------------------- | ---------- | ---- |
| Baseline (Tabular Only)                     | 109,575.09 | 0.88 |
| Multimodal (Tabular + PCA Image Embeddings) | 107,736.39 | 0.89 |

## 🎯 Conclusion

Integrating satellite imagery effectively engineers an automated, programmatic framework for holistic property valuation.

However, the multimodal pipeline recorded a very slight decrease in RMSE and increase in R² score compared to the tabular baseline.

This outcome reveals two critical insights:

1. The heavily engineered tabular features (specifically the median-encoded Location Index) already encapsulate the vast majority of the pricing signal.

2. The visual embeddings extracted from EfficientNet—while highly dimensional—likely introduced variance noise because the backbone was pre-trained on ImageNet (natural images) rather than top-down satellite topography.

### 🚀 Future Work

Future iterations will replace the generalized EfficientNet weights by mathematically fine-tuning the CNN backbone specifically on aerial and satellite-view datasets (such as ResNet or Vision Transformers pre-trained on remote sensing tasks).

This should dramatically improve the granularity and economic relevance of the extracted visual features.
