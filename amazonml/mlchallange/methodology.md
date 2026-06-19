# Amazon ML Challenge 2025: Smart Product Pricing

**Team Name:** [Your Team Name]
**Participant(s):** [Your Name(s)]

---

## 1. Overall Methodology

Our approach was to build a **multimodal regression model** that leverages features from product text, engineered numerical data, and image content to predict prices.

We utilized a **hybrid workflow**:
- **Kaggle (GPU Environment):** Used for the computationally intensive, one-time tasks of downloading the 75,000 product images and extracting deep learning features using a pre-trained model.
- **Local VS Code Environment (CPU):** Used for all other tasks, including data cleaning, feature engineering, and the rapid training and iteration of our final prediction model.

---

## 2. Feature Engineering Techniques

The core of our solution involved robust feature engineering from the raw `catalog_content`.

- **Custom Numerical Features:** We wrote custom Python functions using regular expressions to extract two high-value numerical features:
    - **Item Pack Quantity (IPQ):** Identified patterns like "(Pack of 6)" or "12 count" to extract the number of items.
    - **Normalized Weight/Volume:** Extracted weight and volume information (e.g., "16 oz", "5 LB") and normalized it to a consistent unit (ounces) to make it comparable across products.

- **Text Cleaning and Vectorization:**
    - The raw text was cleaned by removing HTML tags, converting to lowercase, and stripping punctuation.
    - The cleaned text was then converted into a numerical format using `TfidfVectorizer` on a vocabulary of the 5,000 most important words.

---

## 3. Model Architecture and Algorithms

Our solution employed two distinct models for different tasks:

- **Image Feature Extractor: `EfficientNetB0`**
    - We used a state-of-the-art, pre-trained Convolutional Neural Network (CNN) to convert each product image into a 1280-dimension numerical vector (embedding). This allowed us to capture the rich visual information of the products.

- **Final Price Predictor: `LightGBM`**
    - We chose a Gradient Boosting Machine (LightGBM) for the final prediction task due to its high performance on tabular data, speed, and excellent handling of sparse feature matrices.
    - To optimize for the competition's **SMAPE** metric, the model was trained to predict the logarithm of the price (`np.log1p(price)`), which naturally focuses the model on relative (percentage) errors. The final predictions were then converted back to the original scale using `np.expm1()`.

---

## 4. Conclusion

By combining carefully engineered numerical and text features with deep learning-based image features, our LightGBM model was able to effectively learn the complex relationships between product attributes and pricing. This multimodal approach provided a robust solution to the Smart Product Pricing Challenge.