Slide 1: Student Cover Sheet 

No speaker notes

Slide 2: Intro:

Welcome to this data analytics presentation, undertaken in conjunction with BPP Level 6 Data Science Integrated Apprenticeship - Module 2 - Data Analytics 

First off; a disclaimer, a synthetic dataset has been used to act as true employee data. This dataset in no way represents of the true personal data held on employees working at Company X. 

Slide 3: Executive Summary: 

The aim of this project was to develop a predictive model using logistic regression to identify employees at risk of leaving the company & 
to understand the key contributing factors to employee turnover. 

....add in rest at the end of the project... 

Slide 4: Project Background:

Company-X are a leading UK data consultancy, operating predominantly in the defence sector. Having conducted their annual employee survey & end of year reviews
they'd like to make use of this internal data to highlight contributing factors to employee turnover & gain insight into preventative measures they could potentially
put in place to increase employee retention & increase their happiness at work. 

Historically, Company-X has manually reviewed this data by speaking with low scoring employees individually, however, they've opted to apply statistical analysis & predictive models to gain a deeper insight into the dataset as a whole to identify less obvious contributing factors. The People Team recognise the recent trend in prospective new-starters & potential-leavers looking further than the salary when considering a role. They would like to understand their areas for improvement. 

The scope of the project aims to apply a statistical model to the dataset they have provided me, in the hope to predict a. which employees are at risk of leaving the company AND b. to identify what the contributing factors are and advise them of corrective measures they can implement across the company to raise employee satisfaction. 

Slide 5: Employee Turnover - Section A - Methodology:

Section A of this presentation will cover the methodology we plan to follow 

Slide 6: Hypothesis & Objectives: 

We aim to provide Company-X with with their desired data insights by firstly, preprocessing the data. This requires... & the benefits/reasons are... - IN BRIEF 

Secondly we plan to apply a predictive model, known as Logistic Regression, to the processed data. Logistic regression is what's known as a binary classification model whereby it will predict a 1 or 0 for each employee, 1 pertaining to a 'yes' they are at risk of leaving and 0 pertaining to 'no' they aren't at risk of leaving. 

Lastly, we will provide Company-X with insights on which employee features/attributes contribute to an employees likelihood of them leaving the company & advise them of 
corrective measure they can implement to higher employee retention & raise job satisfaction. 

To formulate the basis of this study in a true evidence driven approach, we must formulate our hypothesis to test. First our Null Hypothesis, known as H0, is “there is no significant relationship between employee attributes and the likelihood of employee turnover”. Secondly, our Alternative Hypothesis (H1): “there is a significant relationship between certain employee attributes and the likelihood of employee turnover”. 

We aim to definitively accept or reject the null hypothesis. If the statistical analysis doesn't provide sufficient evidence to reject the null hypothesis, we fail to reject it. This implies that the observed data does not provide strong support for the alternative hypothesis, and we cannot conclude that there is a significant relationship between employee attributes and turnover. On the other hand, if the statistical analysis yields evidence that the observed data significantly deviates from what would be expected under the assumption of the null hypothesis, we reject the null hypothesis in favour of the alternative hypothesis. This suggests that there is a significant relationship between certain employee attributes and the likelihood of turnover. 

Slide 7: Data Source & Overview 

We have one simple dataset available to us for this study, known as company-x employee satisfaction dataset, it's a csv file type.  

The dataset contains 1,470 datapoints, 1,203 of which are current employees & 267 are ex-employees - These figures depict a highly imbalanced dataset. During 
model training we will need to appropriately balance the data to avoid bias towards the majority class, I will go into further later on. 

We have a range of employee attributes to feed into our model, there are 20 numerical attributes & 8 categorical attributes, 28 in total ranging from personal circumstances such as age & marital status, to job role and educational attribute, then moving to their survey scores along with some data on their employment such as how long they've been with the company or in their current position for example. The numerical attributes are either int64 or float64 format which is as we would like them. The categorical attributes are in object format, interpreted as string type which works well for this study. 

Notably, the dataset exhibits no null or duplicate datapoints, indicating thorough & concise data collection practices by Company-X. Their absence mitigates the need to employ statistical imputation or deletion methods. Should null/duplicate values have been present, decisions regarding handling strategies such as imputation or deletion, would have been contingent upon the specific attributes' relevance to the overarching goals of the study. 

Slide 8: Exploratory Data Analysis: 

These are the exploratory data analytics techniques used to prepare the dataset for the predictive model. It's important to prepare the data to ensure that it is clean, consistent, and in a format that the model can effectively learn from. Proper data preparation enhances the model's accuracy, reduces bias, and improves its ability to generalise to new, unseen data.

Data cleaning - attribute headers were converted to snake-case using regex for uniformity. As previously mentioned, we checked for null value and duplicate data in the dataset, as well as checking the attribute data types were as we wanted them. We checked the categorical attributes unique value count to see if we needed to potentially group or bin the values. The only attribute with numerous values were 'role' and 'education_field' with 9 & 6 respectively. Grouping/binning is only of benefit to combine similar or infrequent categories but that wasn't the case here. There appeared to be discrepancies between the years_at_company & the years_in_current_role - one can't be more than the other so we had to assume & impute data here with the logic if years in current role is more than years at company then years at company = years at current role. Similarly, years in current role & years since last promotion should sum to years at company so again, we've imputed data here - these are listed in the project limitations as an assumption that we've made.  

Summary statistics - Looking at the summary statistics we can see that 

Data Distribution - 

Correlation analysis - 

Feature engineering - Standarisation / Scaling

Logistic regression, for instance, does not inherently require features to be on the same scale, but scaling can still be beneficial in certain cases, such as when there are significant differences in the magnitude of the features. In machine learning models like logistic regression, each feature contributes to the prediction of the target variable (in this case, employee turnover) based on its magnitude and relationship with the target. When features have different scales or units, their magnitudes can disproportionately influence the model's decision-making process. By scaling the features, we ensure that each feature contributes proportionally to the model's prediction, regardless of its original scale. This helps prevent features with larger magnitudes from dominating the model and ensures that all features are treated equally in terms of their influence on the prediction. For example, consider a dataset with features for both Age (ranging from 20 to 50) and YearsAtCompany (ranging from 1 to 20). Without scaling, the Age feature, which has a larger numerical range, may have a stronger influence on the model compared to YearsAtCompany. This could lead to biased predictions, as the model may prioritize Age over YearsAtCompany when determining employee turnover. Scaling the features to a similar range allows the logistic regression model to weigh each feature appropriately, leading to more reliable and interpretable predictions. It helps ensure that the model learns meaningful patterns from all features, regardless of their original scales, thus improving its performance and generalization to unseen data. To do this we apply the StandardScaler from sklearn libriary to the attribute (X) matrix. 

This is done after the test/train split but is applied to both test and train - perform feature scaling after splitting your data into training and testing sets. This is because you want to prevent any information leakage from the testing set to the training set, which could lead to overly optimistic performance estimates.

Next, fit the scaler (e.g., StandardScaler) to the training data only. This calculates the mean and standard deviation of each feature in the training set.

After fitting the scaler, transform both the training and testing data using the scaler. This applies the same scaling transformation learned from the training data to both sets, ensuring consistency in scaling.





Outlier detection - done with box plots, observe but don't do anything until we've seen how the model performs. 

Visualisation 

Data imbalance - 1,470 datapoints, 1,203 of which are current employees & 267 are ex-employees, as previously mentioned suggests a highly imbalanced dataset. During 
model training we will need to appropriately balance the data to avoid bias towards the majority class (current employees), meaning we will ensure the ratio of current & ex employees is maintained in both the test & train data split. We used stratify to ensure the class distribution on they y target variable due to the data imbalance. Stratify will ensure that the proportion of classes 1/0, yes/no for turnover is maintained as in the original entire dataset - 5:1 ratio which will lead to more reliable model evaluation. 

One-hot encoding - Also, important to note that certain numeric columns, such as the survey scores, actually denote categorical levels of satisfaction rather than quantitative counts. Despite being represented by numeric values ranging from 1 to 4, they signify different levels of satisfaction rather than numerical quantities. Had they been in string format for low, medium, high for example, we would have employed ordinal encoding to assign a hierarchical numeric value to each category. 

--------------------------------------------------

Summary Statistics:

Provide summary statistics for numerical variables, including measures like mean, median, minimum, maximum, and standard deviation.
Summarize categorical variables with counts and frequencies of each category.
Highlight any notable patterns or outliers revealed by the summary statistics.

Data Distribution:

Visualize the distribution of numerical variables using histograms, kernel density plots, or box plots.
Examine the shape of the distributions and look for skewness or multimodality.
For categorical variables, create bar plots to display the frequency of each category.
Discuss any insights gained from analyzing the data distributions.

Correlation Analysis:

Compute correlation coefficients (e.g., Pearson, Spearman) between numerical variables.
Visualize correlations using heatmaps or scatter plots.
Interpret the strength and direction of correlations and identify any significant relationships between variables.

Feature Engineering:

Identify opportunities to create new features from existing ones, such as combining or transforming variables.
Explain the rationale behind feature engineering decisions and how they may improve model performance.

Outlier Detection:

Use statistical methods or visualization techniques (e.g., box plots, scatter plots) to identify outliers.
Discuss the potential impact of outliers on your analysis and consider strategies for handling them (e.g., removal, transformation).

Visualization:

Present visualizations of key relationships and patterns in the data.
Use various types of plots (e.g., scatter plots, bar charts, line plots) to effectively communicate insights.
Ensure that visualizations are clear, informative, and relevant to your analysis goals.


One-hot Encoding:

Explain the process of one-hot encoding for categorical variables.
Discuss when and why one-hot encoding is necessary, particularly for algorithms like logistic regression.
Provide examples of how categorical variables are encoded before and after the transformation.

--------------------------------------


Binary encoding and one-hot encoding are both techniques used to convert categorical data into numerical format, but they differ in their approach and the resulting representation:

1. **Binary Encoding:**
   - Binary encoding represents each category as a binary number.
   - It first converts categories into numerical labels using an ordinal encoding scheme.
   - Then, it converts these numerical labels into binary digits and encodes them as separate columns.
   - For example, if there are 5 categories, each category will be represented by 3 binary digits (2^3 = 8 > 5).
   - Binary encoding is more space-efficient compared to one-hot encoding, especially when dealing with high-cardinality categorical features.
   - However, it may not capture the inherent relationship between categories as effectively as one-hot encoding.
   - Binary encoding is suitable for categorical features with many unique categories.

2. **One-Hot Encoding:**

do i need to onehote encode - it depends on the specific context and the algorithm you're using. Some machine learning algorithms can handle categorical variables directly without one-hot encoding (e.g., decision trees). However, for algorithms like logistic regression, it's usually better to one-hot encode categorical variables to ensure the model can properly interpret them.

   - One-hot encoding represents each category as a binary vector.
   - It creates a new binary column for each unique category, and sets the corresponding column value to 1 if the category is present, and 0 otherwise.
   - One-hot encoding results in a sparse matrix with many zero values, which can increase memory usage and computational overhead.
   - It preserves all the information about the categories but may lead to the "curse of dimensionality" when dealing with high-cardinality categorical features.
   - One-hot encoding is suitable for categorical features with a small number of unique categories or when the relationships between categories are important to preserve.

In summary, binary encoding is more compact and memory-efficient, whereas one-hot encoding is more intuitive and straightforward, especially for small to moderate-sized categorical features. The choice between them depends on factors such as the cardinality of the categorical feature, memory constraints, and the specific requirements of the machine learning model.












Slide 9: Exploratory Data Analysis: 

Show the attributes that do have a correlation and the ones that don't - add the diagrams

Slide 10: Methodology: 

We used a 70/30 test train spit ratio. This is to allow the model enough data to learn from whilst keeping some back to test it on. 
we used random_state parameters to shuffle the data before splitting it 
We used stratify to ensure the class distribution on they y target variable due to the data imbalance. Stratify will ensure that the proportion of classes 1/0, yes/no for turnover is maintained as in the original entire dataset - 5:1 ratio which will lead to more reliable model evaluation. 

Slide 11: Employee Turnover - Section B - Results: 

Section B of this presentation with cover the results of the study 

Slide 12: Data Insights 




Other notes: 

12 minute maximum recording 

Sections & Talk about: 

Executive Summary
Project Background 
Methodology 
Results
Impact & Conclusion 
Reference List

    
The Problem 

    Company introduction
    The task at hand - the survey and what they want to know from it 
    Describe the dataset - Show the sample data and briefly explain the attributes - or show the column headers included 

    EDA - discuss briefly what I found about the dataset during my EDA notebook 

        need for one hot encoding for non categorical attributes 
        Explain difference between one hot and binary encoding 

    

Objectives

    Take an evidence driven approach 

    Hypothesis:

        LOGISTIC - Likelihood of someone to leave 
        Does Age, Sex, Working Away, Distance from Office
        Which departments are least happy
        Which departments has worst management score
        Does overtime affect their satisfaction
        Are male or female more likely to have a better pay rise 
        Overtime and work life balance score related?

        LINEAR - Years at Company Prediction: You could predict the number of years an employee will stay at the company based on factors like age, job satisfaction scores, relationship satisfaction scores, etc.- this would highlight when employees are at risk of leaving due to their attributes & could benefit preventative measures.  

        - explain aoubt testing and training data 
        - explain that we use 'stratify' to ensure that the split between current and ex employees is equal betweeen both test and train sets otherwise 

        K-MEANS CLUSTERING - Work-Life Balance Analysis: You could use K-means clustering to analyse work-life balance among employees by clustering them based on attributes such as hours worked, overtime hours, work-life balance scores, etc. This could help identify groups of employees who may be experiencing different levels of work-life balance.

What I've Done - 
    Justify why I've chosen this statistical method - LOGISTIC REGRESSION 
    Why is it the best one for the job over others
    Explain the statistical methods cons/weaknesses
        have the math formula on the slide & explain how it's calculated
    Discuss potential other techniques that could be used & why 

    Discuss feature engineering I've done to give maximum accuracy in predicting the turnover rate. - think this is just the EDA and removing outliers etc. 
    Discuss test and train divide and if keeping some back for the other reason - look up 
    Find out the accuracy of the model using testing data 
    Test the model using a confusion matrix, accuracy, precision, recall, f1_score? 

The Business Recommendations 
    Explain the commercial insight from the project 
    Explain the factors the model shows are contributing to employee turnover or dissatisfaction 
    Recommend what actions they can take to improve employee satisfaction 

    Next steps - give ideas for other insights that could be gleaned from the dataset if a second phase was initiated 
    Suggest what data the company could obtain to enhance to study 
    Suggest looking at the gender imbalance and give a reference to Code First Girls study gender in tech 
    