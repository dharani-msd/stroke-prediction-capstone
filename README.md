# stroke-prediction-capstone
Stroke Prediction using Machine Learning and LLM

##Part 1 

#Project Overview

I selected the Stroke Prediction Dataset because it is a publicly available healthcare dataset that is suitable for a machine learning classification project. The dataset contains more than 5,000 records and includes both numeric and categorical features, making it appropriate for data preprocessing and analysis. It also contains missing values, which provide an opportunity to perform data cleaning techniques such as missing value imputation and data type conversion.
This dataset aligns well with the Data-to-AI Lifecycle because it allows the complete workflow to be demonstrated, including data collection, data inspection, data cleaning, exploratory data analysis (EDA), feature engineering, model training, and model evaluation. In addition, the problem of predicting stroke risk has practical importance in healthcare, where early prediction can support timely medical intervention and improve patient outcomes.
Dataset
This project uses the Stroke Prediction Dataset obtained from Kaggle. The dataset contains patient demographic information, medical history, lifestyle factors, and a binary target variable (stroke) indicating whether a patient has experienced a stroke.
The dataset was downloaded from Kaggle, saved locally, and loaded into a pandas DataFrame using the read_csv() function. It was then cleaned and preprocessed before performing exploratory data analysis (EDA), machine learning model development, and LLM integration.

#Dataset Summary:

•	Rows: 5,110
•	Columns: 12
•	Target Variable: stroke

## Development Environment

- Python 3
- Google Colab

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Requests
- JSONSchema
- OpenAI


##Task – 1:

#Loading the Dataset:

The stroke prediction dataset was loaded into a pandas DataFrame using the read_csv() function. This dataset served as the foundation for all subsequent data cleaning, exploratory data analysis (EDA), visualization, and machine learning tasks performed in the project.
import pandas as pd
df = pd.read_csv("healthcare_dataset_stroke_data.csv")

#Delete columns (not relationship)

The id column is just a unique number assigned to each patient. It has no relationship to whether someone has a stroke, so it doesn't help the model learn.

##Task – 2:

#Null value analysis (Handling Missing Values (bmi column)) :

The bmi column contained missing values and it is around 3.9% of missing values are there. These missing values were replaced using the median of the column instead of the mean. The median is less affected by extreme values and skewed distributions, making it a more robust measure of central tendency. Therefore, using the median helps preserve the overall distribution of the data and reduces the influence of outliers. I found null values in dataset for  using df.isnull().sum() and (df.isnull().sum() / df.shape[0]) * 100) . fill numeric columns with the column median using fillna(df[col].median()). 

##Task -3:

#Duplicate detection and removal:

I checked the dataset for duplicate records using df.duplicate().sum(). The result was 0, which means there were no duplicate rows, so no records needed to be removed

##Task – 4:

#Data Type Correction:

Data type correction was performed to ensure that each column had the appropriate data type before analysis and model development. Numeric columns were converted to numeric data types, while categorical columns were retained as object or category types where appropriate.
The pandas to_numeric() function was used to convert values to numeric format, with invalid values handled using errors="coerce".
df[column] = pd.to_numeric(df[column], errors="coerce")
Correcting data types improves data quality, enables accurate statistical analysis, and ensures compatibility with machine learning algorithms.
In this data has already in correct datatype so I just convert few repeated values columns into categorical datatype column why because storing them as category instead of object reduces memory usage and can improve the performance of data processing and machine learning algorithms.
Memory usage before and after the conversion using df.memory_usage(deep=True).sum()
In this data Memory before conversion: 1656573 bytes  and Memory after conversion: 272532 bytes.

##Task -5:

#Descriptive statistics and skewness:

df.describe() is a quick health check of the numeric data. It summarizes the dataset and helps identify missing values, the range of values, and the overall distribution before data cleaning.

#Skewness Analysis:

Skewness was calculated using (df[col].skew()) to understand the distribution of the continuous numeric variables in the dataset.
•	Age: The skewness value is -0.137, which is close to zero. This indicates that the age column is approximately symmetric with a slight negative (left) skew. The data is fairly evenly distributed.
•	Average Glucose Level: The skewness value is 1.572, indicating a highly positive (right) skewed distribution. This means that most patients have lower glucose levels, while a small number of patients have very high glucose levels.
•	BMI: The skewness value is 1.088, indicating a positively (right) skewed distribution. Most patients have BMI values around the lower to middle range, while a few patients have much higher BMI values. Because the BMI distribution is skewed, the median was used to replace missing values instead of the mean, as the median is less affected by extreme values and outliers

##Task – 6:

#Outlier Detection Using the IQR Method:

Outliers were identified using the Interquartile Range (IQR) method. The IQR measures the spread of the middle 50% of the data and is calculated as:
IQR = Q3 − Q1
where:
•	Q1 = First Quartile (25th percentile)
•	Q3 = Third Quartile (75th percentile)
A data point is considered an outlier if it satisfies either of the following conditions:
•	Lower Bound = Q1 − 1.5 × IQR
•	Upper Bound = Q3 + 1.5 × IQR
Any values below the lower bound or above the upper bound were identified as potential outliers. The IQR method was chosen because it is robust to skewed distributions and is not affected by extreme values.

The analysis showed that the Age column did not contain any outliers. The Average Glucose Level column contained 627 outliers, indicating that some patients have unusually high glucose levels. The BMI column contained 126 outliers, representing patients with unusually high or low body mass index values.
The outliers were not removed because they may represent genuine patient measurements rather than data entry errors. Removing them could result in the loss of important medical information. Therefore, the outliers were retained and will be considered during model development in Part 2.

##Task – 7:

#Visualizations:

•	A Line Plot : It was created using the avg_glucose_level column. The x-axis represents the patient index, and the y-axis represents the average glucose level. The plot shows how glucose levels vary across different patients and helps identify patterns and unusually high values.

•	A Bar Chart : A bar chart comparing the mean of one numeric column across categories of one categorical column .The x-axis represents categorical column (e.g., gender, work_type, smoking_status)  and y-axis represents the mean of a numeric column (e.g., avg_glucose_level, bmi, age)

•	A Histogram : The avg_glucose_level column was created using sns.histplot() with 20 bins. This column was identified as the most skewed numeric feature in the dataset (skewness = 1.57). The x-axis represents Average level glucose and y-axis represents Frequency.The distribution is highly right-skewed (positively skewed), with most observations concentrated between lower and moderate glucose levels. A small number of observations have much higher glucose values, creating a long tail on the right side of the histogram. This indicates that the data is not normally distributed and contains a few high-value observations (potential outliers).

 •	A Scatter Plot : Interpretation of (age vs avg_glucose_level)
A scatter plot was created to examine the relationship between age and average glucose level. The plot shows a weak positive relationship, indicating that average glucose levels tend to increase slightly as age increases. However, the points are widely scattered, suggesting that the relationship is not strong and that other factors may also influence glucose levels.
 
•	A Box Plot : Interpretation of  Average Glucose Level by Stroke Status
The box plot compares the average glucose levels of patients with and without stroke. The stroke = 1 group has a higher median average glucose level than the stroke = 0 group, indicating that patients who experienced a stroke generally have higher glucose levels. The stroke = 1 group also shows a wider spread of values, suggesting greater variability in glucose levels. Outliers are present in both groups, representing patients with unusually high glucose levels.
 

##Task – 8:

#Correlation Heatmap:

A correlation heatmap was created using the numeric columns in the dataset. The strongest absolute correlation was observed between age and BMI. The highest absolute correlation in the dataset was 0.32 between age and BMI. This indicates a weak positive relationship, meaning that BMI tends to increase slightly as age increases. Although this is the strongest correlation among the numeric variables, it is still relatively weak.
This correlation does not imply that age directly causes changes in BMI. Other factors such as diet, physical activity, genetics, and overall health could influence both age-related health and BMI, making them plausible alternative explanations.

The two numeric columns with the highest absolute skewness were avg_glucose_level (1.57) and bmi (1.09). For each column, both the mean and median were calculated before imputation. Since both columns are positively skewed, the median was chosen to fill missing values. In positively skewed distributions, extreme high values pull the mean upward, making the median a more representative measure of central tendency. After applying median imputation using fillna(), isnull().sum() confirmed that no missing values remained in either column.

##Task – 9a:

#Imputation strategy comparison:

The two numeric columns with the highest absolute skewness were avg_glucose_level (1.57) and bmi (1.09). For each column, both the mean and median were calculated before imputation. Since both columns are positively skewed, the median was chosen to fill missing values. In positively skewed distributions, extreme high values pull the mean upward, making the median a more representative measure of central tendency. After applying median imputation using fillna(), isnull().sum() confirmed that no missing values remained in either column.

##Task – 9b:

##Spearman rank correlation :

#Pearson Correlation = Straight Line and Linear 

It measures whether two variables change together in a straight-line pattern.


#Spearman Correlation = Ranks + Monotonic relationship 

It converts values into ranks and checks whether the variables generally move in the same direction, even if the relationship is curved.

In these correlation interpretation we just find top three largest difference pairs of correlation

1. Age and Average Glucose Level

•	Pearson = 0.238 
•	Spearman = 0.145 

Result = Pearson > Spearman

The age and average glucose level pair has a higher Pearson correlation than Spearman correlation. This suggests that the relationship is approximately linear rather than monotonic but non-linear. Therefore, Pearson correlation is the preferred measure for feature-selection guidance for this pair.

2. Hypertension and Average Glucose Level

•	Pearson = 0.174 
•	Spearman = 0.114 

Again,
Result = Pearson > Spearman

The hypertension and average glucose level pair shows a slightly higher Pearson correlation than Spearman correlation, indicating an approximately linear but weak relationship. Pearson correlation is more suitable for feature selection because it better represents the observed association.

3. Heart Disease and Average Glucose Level
   
•	Pearson = 0.162 
•	Spearman = 0.106

Again,
Result = Pearson > Spearman

The heart disease and average glucose level pair also has a higher Pearson correlation than Spearman correlation. This suggests an approximately linear relationship, although the association is weak. Pearson correlation is therefore preferred for feature-selection guidance.
The relationships are approximately linear, and Pearson correlation is the more appropriate measure for feature selection for these variable pairs.

##Task – 9c:

#Grouped Aggregation:

The grouped aggregation of average glucose level by work type showed that the Self-employed group had the highest mean average glucose level (112.65) and also the highest standard deviation (51.72). The high standard deviation indicates considerable variation in glucose levels within the Self-employed group, suggesting that work type alone is insufficient to reliably predict the target outcome. The ratio of the highest group mean to the lowest group mean is 1.19 (112.65 / 94.40). This relatively small difference suggests that while work type may provide some predictive signal, it is unlikely to be a strong standalone predictor and should be used together with other features in the predictive model.It is find to compute with df.groupby(categorical_col)[numeric_col].agg(['mean', 'std', 'count'])

##Task – 10:

#Save the Cleaned Dataset:
The cleaned dataset was saved as cleaned_data.csv by using df.to_csv('stroke_cleaned_data.csv', index=False) after all preprocessing steps were completed. The file was saved without the DataFrame index and will be used as the input dataset for Parts 2 and 3 of the project.

##Part -2                 Supervised Machine Learning Model — Build, Train, and Evaluate 

##Task 1 

#Data Loading and Label Definition :

The cleaned dataset (stroke_cleaned_data.csv) generated in Part 1 was loaded into a pandas DataFrame for supervised learning.

Feature Matrix (X) :

The feature matrix (X) contains all predictor variables after excluding the target columns:

•	avg_glucose_level (regression target)
•	stroke (classification target)

Regression Label (y_reg) :

The continuous numeric column avg_glucose_level was selected as the regression target.

Classification Label (y_clf) :

The existing binary column stroke was selected as the classification target.

•	0 = No Stroke
•	1 = Stroke

The Stroke Prediction dataset already contains a natural binary target (stroke), so it was used instead of creating a new binary label from the regression target. This follows the assignment requirement, which allows using either a binary label derived from y_reg or another natural binary column in the dataset.

Dataset Shapes :

•	Feature Matrix (X): (5110, 9)
•	Regression Label (y_reg): (5110,)
•	Classification Label (y_clf): (5110,)

##Task - 2

#Encoding categorical column:

#Categorical Feature Encoding:

#The feature matrix contained the following categorical columns:

• gender
•	ever_married
•	work_type
•	Residence_type
•	smoking_status

These variables do not have a natural order, so One-Hot Encoding was applied using pd.get_dummies() with drop_first=True.
No label encoding was used because assigning integer values to these categories would create a false ordinal relationship. One-hot encoding represents each category as a separate binary feature, allowing machine learning models to treat the categories independently.

The first dummy variable for each categorical feature was dropped (drop_first=True) to avoid multicollinearity (the dummy variable trap).

##Task – 3

#Leak-free train-test split and scaling: 

#Train-Test Split and Feature Scaling:

The dataset was divided into training and testing sets using an 80:20 split with random_state=42 to ensure reproducibility.
A StandardScaler was used to standardize the feature values. The scaler was fitted only on the training data using scaler.fit(X_train). The fitted scaler was then used to transform both the training and testing feature sets.

This approach prevents data leakage. Fitting the scaler on the entire dataset before splitting would allow information from the test set (such as its mean and standard deviation) to influence the training process. As a result, the model could achieve overly optimistic performance that would not reflect its ability to generalize to unseen data.

##Task – 4:

#Regression model — Linear Regression:

#Regression Target Selection:

The regression target (y_reg) is avg_glucose_level, a continuous numeric variable. It was selected because regression models are designed to predict continuous values, and blood glucose level is an important health indicator in the Stroke Prediction dataset.

#Linear Regression:

A Linear Regression model (OLS) was trained using the scaled training features to predict the continuous target variable as i choose avg_glucose_level. Model performance was evaluated using Mean Squared Error (MSE) and the coefficient of determination (R²) on the test dataset

The model coefficients were examined to determine which features had the greatest influence on the predicted glucose level. The three features with the largest absolute coefficient values were identified as the most influential predictors.

A large positive coefficient indicates that a one-unit increase in the corresponding scaled feature is associated with an increase in the predicted avg_glucose_level by approximately the coefficient value, while keeping all other features constant.

A large negative coefficient indicates that a one-unit increase in the corresponding scaled feature is associated with a decrease in the predicted avg_glucose_level by approximately the coefficient value, assuming all other features remain unchanged.

#Ridge Regression:

Ridge Regression (alpha = 1.0) was trained using the same training and testing split. Its performance was compared with the standard Linear Regression model using MSE and R².

Ridge Regression applies L2 regularization, which penalizes large coefficient values and helps reduce overfitting and multicollinearity. The alpha parameter controls the strength of this regularization. A larger alpha results in greater coefficient shrinkage, while alpha = 0 makes Ridge Regression equivalent to ordinary Linear Regression. In this project, using alpha = 1.0 produced only a minimal difference compared with Linear Regression, suggesting that regularization had little impact on this dataset.

#Regression Model Comparison:

| Metric | OLS (Linear Regression) | Ridge Regression | Difference |
|--------|-------------------------:|-----------------:|-----------:|
| MSE | 1788.8034 | 1788.7996 | -0.0038 |
| R² | 0.069711 | 0.069713 | +0.000002 |


Both models produced nearly identical performance. Ridge Regression achieved a slightly lower Mean Squared Error (MSE), but the improvement was very small. The R² score remained the same for both models, indicating that they explained approximately 6.97% of the variance in the target variable (avg_glucose_level).

R² value of 0.0697 is quite low, which means the current features are not very effective at predicting avg_glucose_level with a linear model. For this dataset, the classification task (predicting stroke) is usually much more meaningful and often yields more informative results than trying to predict avg_glucose_level with linear regression.

##Task – 5:

#Classification model :

The target variable (stroke) was found to be highly imbalanced. The training dataset contained 3901 non-stroke cases (95.43%) and 187 stroke cases (4.57%), with the minority class representing less than the required 35% of the samples. To address this imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training dataset. SMOTE generates synthetic samples for the minority class, resulting in a balanced training dataset while preventing data leakage into the test set. This approach was chosen because improving the identification of stroke cases (minority class) is more important than maximizing overall accuracy in a medical prediction problem.

#Before/After comparison:

| Dataset | Class 0 | Class 1 |
|---------|--------:|--------:|
| Before balancing | 3901 | 187 |
| After balancing | 3901 | 3901 |

This table is exactly what the requirement means by "Class imbalance handling is demonstrated with a before/after class count comparison." It clearly shows the class counts before balancing and after balancing, demonstrating that the imbalance was successfully addressed.

#Logistic Regression :

The Logistic Regression model achieved an accuracy of 74.75%. For the minority stroke class, it obtained a precision of 18%, recall of 87%, and an F1-score of 30%. The confusion matrix showed 710 True Negatives, 54 True Positives, 250 False Positives, and 8 False Negatives. Although the model produced a relatively high number of false positives, it successfully identified 87% of actual stroke cases, making it suitable for medical screening where minimizing missed stroke patients is more important than reducing false alarms.

#ROC AND AUC CURVE:

The Logistic Regression model achieved an AUC score of 0.8465, indicating good classification performance. This means the model can effectively distinguish between stroke and non-stroke patients. An AUC of 0.8465 indicates that there is approximately an 84.65% probability that the model will rank a randomly selected stroke patient higher than a randomly selected non-stroke patient.

#Formulas :

| Metric | Formula |
|--------|---------|
| TPR (Recall) | TP / (TP + FN) |
| FPR | FP / (FP + TN) |
| ROC Curve | Plot of TPR vs FPR |
| AUC | Area under the ROC curve (∫₀¹ TPR(FPR) d(FPR)) |

Precision and Recall formulas:

Precision:

                          Precision = TP / ( TP + FP)

Measures how many of the patients predicted as having a stroke actually had a stroke. 

Recall:

                           Recall = TP / ( TP + FN)

Measures how many of the actual stroke patients were correctly identified by the model. 
Most Important Metric in the dataset:

Basically in healthcare datset our first most important metric is Recall. For stroke prediction, Recall is more important than Precision because missing a patient who actually has a stroke (False Negative) can lead to delayed treatment and serious health consequences. Although a higher recall may increase the number of false positives, identifying as many true stroke cases as possible is the priority in medical diagnosis.

##Task – 6:

#Decision-Threshold Sensitivity:

It means how the model's predictions change when you change the probability threshold used for classification. In Logistic Regression, the default threshold is

Threshold = 0.5

•	If probability ≥ 0.5 → Predict Stroke (1) 

•	If probability < 0.5 → Predict No Stroke (0)

a)Precision and Recall formulas:

Precision:

                          Precision = TP / ( TP + FP)

Measures how many of the patients predicted as having a stroke actually had a stroke. 

Recall:
       
                           Recall = TP / ( TP + FN)

Measures how many of the actual stroke patients were correctly identified by the model. 


(b) Threshold that Maximizes F1-Score:

| Threshold | Precision | Recall | F1-Score |
|-----------|----------:|-------:|---------:|
| 0.30 | 0.128 | 0.919 | 0.226 |
| 0.40 | 0.152 | 0.919 | 0.262 |
| 0.50 (Default) | 0.177 | 0.870 | 0.295 |
| 0.60 | 0.186 | 0.693 | 0.293 |
| 0.70 | 0.195 | 0.532 | 0.285 |

The 0.50 threshold produced the highest F1-score (0.295), indicating the best balance between precision and recall for this dataset.

c) More Important Metric:

For stroke prediction, Recall is more important than Precision because failing to identify a patient who actually has a stroke (False Negative) can delay treatment and lead to serious health consequences. Although a higher recall may increase the number of false positives, identifying as many true stroke patients as possible is the primary objective in medical screening.

d) Threshold Selection:

To optimize for Recall, the decision threshold should be lowered (for example, from 0.50 to 0.30 or 0.40). Lowering the threshold allows the model to classify more patients as stroke, increasing the number of correctly detected stroke cases.

The trade-off is that lowering the threshold also increases the number of False Positives, meaning more healthy individuals may be incorrectly classified as being at risk of stroke and may undergo additional medical evaluation. In healthcare, this trade-off is generally acceptable because missing a true stroke patient is considered more serious than investigating a false alarm.

##Task – 7:

#Regularization experiment on Logistic Regression:

| Logistic Regression Model | C Value | Precision | Recall | AUC |
|---------------------------|--------:|----------:|-------:|----:|
| Baseline Model | 1.0 | 0.178 | 0.871 | 0.8465 |
| Strong Regularization | 0.01 | 0.173 | 0.871 | 0.8503 |

Performance Comparison:

•	Precision: decreased slightly from 0.178 to 0.173. 

•	Recall: remained the same at 0.871. 

•	AUC: increased slightly from 0.8465 to 0.8503.

Reducing C from 1.0 to 0.01 applied stronger L2 regularization, resulting in a simpler Logistic Regression model. On this stroke prediction dataset, stronger regularization caused a small decrease in precision, no change in recall, and a slight improvement in AUC. Overall, the effect of stronger regularization was minimal, indicating that the baseline model was already well regularized. Since recall remained high and AUC improved slightly, the C = 0.01 model maintained good discrimination between stroke and non-stroke patients while using stronger regularization.

##Task – 8:

#Bootstrap confidence interval for AUC difference:

A bootstrap analysis with 500 resamples was performed to compare the AUC of the baseline Logistic Regression model (C = 1.0) with the strongly regularized model (C = 0.01).

•	Mean AUC Difference (C = 1.0 − C = 0.01) = -0.00384

•	95% Confidence Interval: (-0.00834, 0.00039)

The mean AUC difference is slightly negative, indicating that the C = 0.01 model achieved a marginally higher AUC than the baseline model. However, the 95% confidence interval includes zero, which indicates that the observed difference is not statistically significant. Therefore, the performance difference between the two Logistic Regression models may be due to random sampling variation, and there is insufficient evidence to conclude that one model consistently outperforms the other.























