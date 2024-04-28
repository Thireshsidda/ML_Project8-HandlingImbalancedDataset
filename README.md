# ML_Project8-HandlingImbalancedDataset

This project tackles the challenge of imbalanced datasets in machine learning, specifically focusing on credit card fraud detection. Here, we explore various techniques to address the issue of a highly skewed class distribution, where fraudulent transactions (the minority class) are significantly outnumbered by normal transactions (the majority class). Our goal is to improve model performance in identifying fraudulent transactions, which is crucial for preventing financial losses.

### 1. Introduction

Imbalanced datasets pose a significant hurdle in machine learning classification tasks. When a dataset has a large imbalance between classes, models tend to prioritize the majority class during training. This can lead to models that perform well on the majority class but poorly on the minority class, failing to detect fraudulent transactions effectively.

### 2. Dataset and Exploration

The project utilizes the creditcard.csv dataset containing features related to credit card transactions and a binary class label indicating fraudulent (1) or normal transactions (0).
Pandas is used for data manipulation and exploration to visualize the class distribution and understand the characteristics of each class.

### 3. Techniques Explored

1)Undersampling: This technique reduces the number of samples from the majority class to achieve a more balanced distribution. We employ NearMiss from the imblearn library, which removes majority class samples carefully to minimize information loss.

2)Oversampling: This technique increases the number of samples from the minority class to create a more balanced distribution. RandomOverSampler from imblearn is used to duplicate existing minority class samples.

3) SMOTETomek (Synthetic Minority Oversampling Technique): This technique combines oversampling with the Tomek Links method to create synthetic minority class samples based on existing data while avoiding overfitting.

### 4. Model Training and Evaluation

We employ a Random Forest Classifier model due to its robustness to imbalanced datasets.
The data is split into training and testing sets using train_test_split.
GridSearchCV is used to find the optimal hyperparameters for the Random Forest model.
For each resampling technique (original, undersampled, oversampled, SMOTETomek), the model is trained on the resampled training set and evaluated on the testing set using metrics like accuracy score, confusion matrix, and classification report.

### 5. Results and Discussion

The expected output includes confusion matrices and classification reports for each resampling approach.
We compare the performance of the model on the original imbalanced dataset and the resampled datasets to identify the most effective technique for improving fraud detection accuracy, particularly for the minority class (fraudulent transactions).

### 6. Conclusion

By addressing imbalanced data using techniques like undersampling, oversampling, and SMOTETomek, we can enhance the performance of machine learning models in detecting fraudulent transactions. This project provides a foundation for understanding and applying these techniques to improve the effectiveness of fraud detection systems.

### 7. Further Exploration

Experiment with different machine learning algorithms that may be more suitable for imbalanced datasets.
Explore advanced techniques like cost-sensitive learning and ensemble methods specifically designed for imbalanced data.
Consider using cross-validation for a more robust evaluation of model generalizability.
Visualize feature importance to gain insights into how the model identifies fraudulent transactions.

### 8. References

Scikit-learn documentation: https://scikit-learn.org/
Imbalanced-learn library: https://imbalanced-learn.org/
Handling Imbalanced Data Sets: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

### Note:
The provided code snippet demonstrates the application of undersampling, oversampling, and SMOTETomek. You might replace Random Forest with a different model based on your experimentation.
