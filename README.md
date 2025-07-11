E-Commerce Product Recommendation System
Description :
This project is a content-based recommendation system for e-commerce products. It was developed as part of an internship project at Next24tech Technology & Services to demonstrate the use of natural language processing (NLP), machine learning, and data visualization to enhance the product recommendation experience. The system reads product data from a CSV file and uses the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique along with cosine similarity to recommend similar products based on their textual descriptions.

The goal of this system is to help users discover relevant products that match their interests by analyzing the product descriptions and calculating the similarity between them. This method is particularly useful when user interaction data is limited or unavailable.

This project demonstrates how simple yet powerful natural language processing techniques can be applied to build a recommendation engine for products. By analyzing product descriptions, we can derive meaningful relationships between items and offer users relevant suggestions. This system lays the groundwork for more advanced models and applications in both academic and commercial setting

Dataset Description
The dataset used in this project is stored in a file named products.csv. Each row in the dataset represents a unique product and contains two primary fields:

title: The name of the product.
description: A textual description of the product, which provides features, specifications, and keywords.
The system depends on meaningful and consistent product descriptions to function accurately. Descriptions with missing or noisy data can negatively impact the recommendation quality.

Key Features
TF-IDF Vectorization: Transforms product descriptions into numerical vectors while filtering out common stop words.
Cosine Similarity: Calculates pairwise similarity scores between all product vectors to find products with similar textual content.
Multiple Product Recommendation: The script recommends top 5 similar products for any number of product indices specified.
Visualizations:
A word cloud that visualizes frequent terms in product descriptions.
A heatmap showing similarity scores among the top 10 products.
Bar charts for top recommendations of each selected product.
Folder Structures :
├── ecommerce_recommendation_system.py

├── data/

  └── products.csv
├── Outputs

└── README.md

How It Works
Data Loading: The script loads the CSV file into a pandas DataFrame.
Cleaning: Rows with missing product descriptions are removed to ensure reliable recommendations.
TF-IDF Vectorization: Product descriptions are converted into TF-IDF vectors to emphasize important terms.
Similarity Matrix: Cosine similarity is computed between every pair of products.
Recommendation Logic: For each selected product index, the top 5 most similar products are displayed along with their similarity scores.
Visualization:
A word cloud visualizes common keywords in product descriptions.
A similarity heatmap compares top products.
Recommendation scores are shown in horizontal bar plots.
How to Run
Load the data set 'products.cssv'
Make sure the required libraries are installed
Run the file 'ecommerce_recommendation_system.py'
Sample Output :
Top 5 recommendations for 'Wireless Mouse':

Smartphone Gimbal (score: 0.263)
Laptop Cooling Pad (score: 0.225)
Wireless Gaming Mouse (score: 0.152)
Bluetooth Headphones (score: 0.149)