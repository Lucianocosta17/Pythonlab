# Our Approach and Project Objectives

This project is an academic exercise from a Master's program in Data Science, aimed at developing a method for automatically predicting English essay scores. The goal was to create a model capable of accurately assessing and predicting these scores. We began by collecting a dataset containing English essays and their corresponding scores. After preprocessing the text data to remove special characters, punctuation, and irrelevant words, we tokenized the essays to prepare them for analysis. Subsequently, we crafted features such as word count, lexical diversity, and average sentence length to capture various linguistic aspects of the essays. Additionally, we utilized machine learning techniques, including linear regression, random forest, and support vector regression, to build predictive models based on these features. 

## Challenges and Limitations

Despite our efforts, achieving a high level of accuracy in predicting essay scores proved to be challenging, partly due to computational limitations and the complexity of the task. For example, our linear regression models showed limitations in capturing the nuances of essay quality, as evidenced by relatively low R-squared values. Similarly, the random forest and support vector regression models struggled to accurately predict scores across all essay sets, indicating the need for further refinement.

## The Development Steps and Results

### Data Validation and Reading

- **Action:** Checked for the existence of a specific excel file and loaded the data into a Pandas DataFrame.
- **Results:** Successful data reading provided an initial overview, descriptive statistics, identified missing values, etc.

### Text Cleaning and Preparation

- **Action:** Cleaned the 'essay' column by removing strings containing '@' and excessive spaces, followed by saving it in a new excel file.
- **Results:** Created a dataset with cleaned essays, ready for subsequent analysis.

### Data Integration

- **Action:** Merged the original dataset with the cleaned essays, reorganizing columns to facilitate comparison and analysis.
- **Results:** A combined dataset with original and cleaned essays was saved for future reference.

### Natural Language Processing 

- **Action:** Tokenized the cleaned essays using the pre-trained spaCy model and saved the tokens in a new excel file.
- **Results:** Enriched the DataFrame with a new column containing the extracted tokens from the essays.

### Score Normalization

- **Action:** Normalized the essay scores based on predefined score ranges for each essay set.
- **Results:** Generated normalized scores within a 0 to 1 range, enabling fair comparisons between essays from different sets.

### Word Count, Lexical Diversity, and Average Sentence Length

- **Action:** Calculated the word count, lexical diversity, and average sentence length for each essay, adding these metrics as new columns.
- **Results:** Updated the dataset with comprehensive quantitative information about the essays, including details on overall length, vocabulary variance, and sentence structure.

### Grammatical Complexity Analysis

- **Action:** Calculated metrics such as the average number of nouns, verbs, and adjectives per sentence.
- **Results:** Enhanced the dataset with new columns reflecting grammatical complexity, providing deeper insights into the linguistic structure of essays. Saved the enriched dataset for further analysis and model training.

### Spelling Error Analysis

- **Action:** Used a spell checker to identify and count spelling errors in each essay.
- **Results:** Included a new column indicating the number of spelling errors, which can be indicative of the essay's quality.

### Visualization of Error Distribution

- **Action:** Created histograms to visualize the distribution of spelling errors in the essays.
- **Results:** The majority of the bars are concentrated at the beginning of the x-axis, indicating a smaller number of errors. This suggests that most of the essays contain few spelling mistakes, with a concentration up to 12 errors.

### Data Consolidation

- **Action:** Extracted and combined columns of interest from various files into a single DataFrame.
- **Results:** Created a final dataset containing a wide range of features for analysis.

### Machine Learning Model Evaluation

- **Action:** Used a Random Forest Regressor to predict normalized scores based on various essay features. The model was trained on the training set, and the importance of features was ranked and visualized in a bar chart.
- **Results:** 'Word_count' emerged as the predominant feature with moderate importance (0.46 out of 1), while other features exhibited relatively low significance, reflecting in varied accuracy levels across essay groups.

### Interaction Feature Calculation

- **Action:** Developed new interaction features like lexical density, noun-verb ratio, adjective-noun ratio, and spelling error density to delve deeper into textual characteristics.
- **Results:** Enhanced the analytical depth of the dataset by integrating these interaction features, providing a more nuanced understanding of the essays beyond basic metrics.

### Correlation Analysis

- **Action:** Computed Pearson correlation coefficients between essay features and the final normalized scores.
- **Results:** The analysis reveals a moderate positive correlation for lexical density and word count. However, most features show weak correlations with the final scores, suggesting that the key determinants of essay quality might not be fully captured by these metrics alone.

### Feature Analysis Across Essay Sets

- **Action:** Analyzed essay features across different essay sets, calculating their mean and standard deviation, and conducted an ANOVA test for each feature to examine variability between groups.
- **Results:** Identified significant differences in essay characteristics across sets, as evidenced by varying means, standard deviations, and ANOVA test outcomes, indicating distinct linguistic and structural profiles per essay set.

### Impact of Essay Characteristics on Scoring

- **Action:** Processed and arranged the correlation coefficients between a range of essay characteristics, including interactive features, and the essays' normalized scores across different essay sets. Subsequently presented these correlations for each group.
- **Results:** The correlation analysis suggests the quality of an essay, as reflected in the normalized final score, is strongly associated with its length, vocabulary richness, and expressiveness. In contrast, spelling accuracy and grammatical complexity have a lesser impact. Well-developed essays with good lexical density and word count tend to be rated higher.

### A Linear Regression Analysis

- **Action:** Performed linear regression analysis on essay data grouped by essay set using statsmodels, with various writing characteristics as independent variables.
- **Results:** The regression analysis revealed varied model fits, with R-squared values between 0.506 and 0.726. Lexical density and word count significantly influence scores in many groups, while the effects of features like spelling errors and sentence length differ by group. Some features showed varied significance, hinting at a context-dependent impact on scores. High condition numbers in some models suggest potential multicollinearity, which might impact coefficient reliability.

### Predictive Analysis of Essay Scores by Group

- **Action:** Conducted linear regression analysis for specified essay groups from a dataset, utilizing statsmodels. For each group, segregated features and the normalized score, split the data into training and testing sets, fit the model on the training set, and predicted scores on the test set. Calculated and displayed the Mean Squared Error (MSE) and R-squared (R^2) values to assess model performance for each group.
- **Results:** Significant variations in both MSE and R^2 across essay groups, indicating differences in prediction accuracy and the model's ability to explain the variance in normalized scores. Group 1 showed the best performance with an R^2 of 0.740, indicating strong predictive capability, while Group 3 had the weakest performance with an R^2 of 0.446, reflecting the varying complexity of essay features and their relationship with scores across different datasets.

### GBM Model: Optimization & Evaluation Approach

- **Action:** Optimized hyperparameters for a Gradient Boosting Machine (GBM) model via GridSearchCV with 5-fold cross-validation, aiming to identify the best parameter combination based on negative mean squared error. Post-optimization, evaluated the model's performance on a test set, calculating MSE and R^2 to quantify its predictive accuracy.
- **Results:** The GBM model was fine-tuned with optimized hyperparameters, yielding a MSE of 0.0120 on the test set, indicating a moderate level of prediction error. The R^2 value of 0.4739 suggests that less than half of the variability in essay scores is explained by the model, indicating potential for improvement.

### Neural Network Progress: 5-Epoch Essay Scoring

- **Action:** Designed to train a neural network model with PyTorch to predict normalized essay scores from their tokenized text representations, using a 5-epoch training process. Tokenized the essay texts, created a vocabulary for token-to-index mapping, and split the data into training and testing sets. The model aimed to minimize Mean Squared Error (MSE) over these epochs to improve its predictive accuracy for essay scores.
- **Results:** Across 5 training epochs, the model demonstrated an initial decrease in loss from 0.0635 to 0.0412, indicating early signs of effective learning. Despite fluctuations, the loss ultimately improved, closing at 0.0507 in the final epoch. This reduction signifies a commendable achievement in predictive accuracy, indicating that the model effectively adapted to better predict essay scores. The final loss reflects substantial progress in the model's learning and prediction capabilities over the training period.

### Essay Group Scoring: Neural Network Efficacy Comparison

- **Action:** Trained a neural network model using PyTorch for each essay group, split data into training and testing sets, and evaluated each model's performance with Cohen's Kappa Score after 5 epochs.
- **Results:** Training outcomes varied across eight essay groups with differing loss trends and Cohen's Kappa Scores. Group 1 showed significant improvement, while Group 8 had minimal loss but a Kappa Score of 0.0, indicating no agreement beyond chance. Results reflect the models' varying success in accurately predicting essay scores.

### Essay Score Prediction with Random Forest across Eight Sets

- **Action:** Employed a Random Forest Regressor to predict normalized essay scores using CountVectorizer for feature extraction. Predefined parameters optimized the Random Forest, with performance evaluated using Mean Squared Error (MSE) for each essay group.
- **Results:** Varied MSE across eight essay sets, with Set 8 showing the highest accuracy and Set 3 the highest MSE. These outcomes demonstrate the Random Forest's capacity to handle the unique characteristics of each essay set.

### SVR Essay Score Prediction across Eight Sets

- **Action:** Applied a Support Vector Regression (SVR) model with CountVectorizer for modeling. The SVR was trained on text features, with performance evaluated by calculating MSE for each essay set.
- **Results:** A spectrum of accuracies was observed, with Sets 1 and 8 showcasing the highest precision. The results highlight SVR's varied ability to predict essay scores, with notable successes and challenges across sets.

### MSE Comparison: Chart Visualizes Random Forest vs. SVR

- **Action:** Generated a bar chart comparing MSE results between Random Forest and SVR models across eight essay sets. The chart allows for a direct visual comparison of performance.
- **Results:** The bar chart presents a close comparison, with SVR slightly outperforming Random Forest in most sets, especially Set 8. Both models are comparably effective for this prediction task across various essay sets.

### Refining LSTM Essay Scoring with Pre-trained Word2Vec

- **Action:** Implemented an essay scoring model using an LSTM network and pre-trained Word2Vec for word representations. The model was trained for 20 epochs with the goal of minimizing prediction errors.
- **Results:** Consistent loss decrease over the training epochs indicates the model's improvement. The training adjustments demonstrate the effectiveness of the process in refining the model's predictive capabilities.

### Enhancing LSTM Essay Scoring: Changes and Performance Evaluation

- **Action:** Made several alterations to the LSTM model configuration to observe how different configurations affect the model's learning and generalization ability.
- **Results:** The training showed a declining loss, suggesting progressive learning. Despite a plateau in loss reduction, the changes provided valuable insights into enhancing model performance.

### Conclusion

While significant strides were made in developing a predictive model for English essay scores, the complexities and challenges inherent in such tasks were also highlighted. Future efforts will focus on exploring more sophisticated modeling approaches and incorporating additional features to enhance predictive model performance.

