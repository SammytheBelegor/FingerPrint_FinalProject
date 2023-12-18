Fingerprint Matching using OpenCV and SOCOFing Dataset
Introduction
This project demonstrates fingerprint matching using OpenCV, a popular computer vision library, and the SOCOFing dataset, which is available on Kaggle (SOCOFing Dataset). The code uses the Scale-Invariant Feature Transform (SIFT) algorithm to detect and match keypoints between a reference fingerprint image and a set of real fingerprint images.

Requirements
Before running the code, make sure you have the following dependencies installed:

Python 3.x
OpenCV (pip install opencv-python)
Usage
Download the SOCOFing dataset from Kaggle (SOCOFing Dataset).

Extract the dataset and place the script in the same directory as the "SOCOFing" folder.

Uncomment one of the sample lines in the script to choose a reference fingerprint image. You can choose an altered image with varying difficulty levels or a real image for comparison.

Run the script. It will iterate over the first 6000 real fingerprint images, comparing each with the selected reference image.

The script will display the best match along with a score indicating the similarity percentage.

Understanding the Code
The code utilizes the SIFT algorithm to extract keypoints and descriptors from the reference and real fingerprint images.
Keypoints are then matched using the FlannBasedMatcher, and matches are filtered based on distance ratios.
The best match is determined by calculating the percentage of matched keypoints relative to the total keypoints.
File Descriptions
fingerprint_matching.py: The main Python script for fingerprint matching.
SOCOFing/: The dataset folder containing real and altered fingerprint images.
Important Notes
Ensure that the dataset folder ("SOCOFing") is in the same directory as the script.
Experiment with different reference images by uncommenting the corresponding lines in the script.
Acknowledgments
SOCOFing dataset: SOCOFing Dataset
OpenCV: OpenCV
