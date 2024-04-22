
# Facial Emotion Recognition Project

## Overview
This project involves building and training a convolutional neural network (CNN) for the task of facial emotion recognition. It includes steps for data preprocessing, image enhancement, data augmentation, model training, and evaluation.

## Requirements
- Python 3.8 or later
- TensorFlow 2.x
- OpenCV
- Pandas
- Matplotlib
- Seaborn
- PySpark
- NumPy
- MongoDB and PyMongo for database operations

## Steps to Run the Project

1. **Setup Environment**: 
   - Ensure Python 3.8 or Google Colab later is installed.
   - Installing required Python packages using `pip install tensorflow opencv-python pandas matplotlib seaborn pyspark pymongo`.

2. **Data Preparation**:
   - We placed our training and testing data in respective directories. Structured them with subdirectories for each class/emotion.
      - paths 
        train_dir ='/content/drive/MyDrive/DATA 603/TRAIN'
        test_dir = '/content/drive/MyDrive/DATA 603/TEST'

3. **Count Classes in Directories**:
   - Run the `count_classes` function to ensure data is correctly placed and get an overview of class distribution and to ensure that all files loaded successfully

4. **Image Enhancement**:
   - Use the `enhanced_save_image` function to enhance images in the dataset for better model training.
      Enhancement:
      input_parent_directory = "/content/drive/MyDrive/DATA 603/TRAIN".
      Enhanced images are stored in below path.These images can be fed as input for further processing
      output_parent_directory = "/content/drive/MyDrive/DATA 603/ENHANCED_TRAIN"


5. **Data Augmentation**:
   - Run the data augmentation script to generate augmented images, which will help improve model generalization.
     Augmentation:
     input_parent_directory = "/content/drive/MyDrive/DATA 603/ENHANCED_TRAIN"
     Augmented images are stored in below path.These images can be fed as input for further processing
     output_parent_directory = "/content/drive/MyDrive/DATA 603/AUGMENTATED_TRAIN"


6. **Model Training**:
   - Use the `custom_facial_CNN_Model` function to create a CNN model.
   - Train the model using the training dataset.

7. **Model Evaluation**:
   - Evaluate the model's performance using the test dataset.
   - Generate and examine confusion matrices and other evaluation metrics.

8. **Save and Export Predictions**:
   - Use the model to predict on new data.
   - Save these predictions to a CSV file for further analysis.

9. **Database Integration**:
   - To store predictions we used MongoDB.
   - Use the script section for MongoDB integration to store prediction results.

10. **Visualization**:
    - Utilize matplotlib , seaborn,tableau for visualizing data distributions and model performance metrics.






## Note
- paths 
train_dir ='/content/drive/MyDrive/DATA 603/TRAIN'
test_dir = '/content/drive/MyDrive/DATA 603/TEST'

Enhancement:
    input_parent_directory = "/content/drive/MyDrive/DATA 603/TRAIN"
    output_parent_directory = "/content/drive/MyDrive/DATA 603/ENHANCED_TRAIN"

Augmentation:
   input_parent_directory = "/content/drive/MyDrive/DATA 603/ENHANCED_TRAIN"
    output_parent_directory = "/content/drive/MyDrive/DATA 603/AUGMENTATED_TRAIN"

