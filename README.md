FeatureComparison Application

Overview
The FeatureComparison application demonstrates how to perform face detection and feature extraction using deep learning models, and compute the similarity between facial features extracted from two images. This is useful for tasks like face recognition and verification.


Features
Face detection using RetinaFace model.
Facial feature extraction.
Calculation of similarity between two sets of facial features.


Prerequisites
Before running the application, ensure you have the following installed:

Java Development Kit (JDK) version 8 or higher
Apache Maven for dependency management (optional but recommended)


Output
The application logs the similarity between the two detected faces as a percentage.

Acknowledgments
This application utilizes the Deep Java Library (DJL) for deep learning inference.
Face detection is powered by the RetinaFace model.
Facial feature extraction is performed using a feature extraction module.

cropFace Method
The cropFace method extracts a cropped image of a detected face from the original image based on the bounding box coordinates.

Parameters:
img1: Original Image object containing the face.
box: Detected face object with bounding box information.
Returns: Cropped Image object representing the face region.
calculateSimilarity Method
The calculateSimilarity method computes the similarity percentage between two sets of feature vectors using cosine similarity.

Parameters:
feature1: Feature vector of the first image.
feature2: Feature vector of the second image.
Returns: Similarity percentage between the feature vectors.
