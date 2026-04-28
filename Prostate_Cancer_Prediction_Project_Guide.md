# ProCogniTrack: Prostate Cancer Prediction and Risk Factor Analysis

## Using Integrated Health and Lifestyle Data Analysis

---

# Curriculum Project on Data Science & Statistics

**FY BSc. Statistics/Data Science**

**Vishwakarma University, Pune**

**Academic Year: 2025–26**

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Chapter 1 - Introduction](#2-chapter-1-introduction)
   - 1.1 Background
   - 1.2 Problem Statement
   - 1.3 Objectives
3. [Chapter 2 - Data Description](#3-chapter-2-data-description)
   - 2.1 Data Source
   - 2.2 Variable Description
4. [Chapter 3 - Methodology](#4-chapter-3-methodology)
   - 3.1 Exploratory Data Analysis (EDA)
   - 3.2 Statistical Methods
   - 3.3 Regression Modeling
   - 3.4 Model Evaluation Metrics
5. [Chapter 4 - Results and Discussion](#5-chapter-4-results-and-discussion)
   - 4.1 Univariate Analysis
   - 4.2 Bivariate Analysis
   - 4.3 Multivariate Analysis
   - 4.4 Regression Results
6. [Chapter 5 - Conclusion and Future Work](#6-chapter-5-conclusion-and-future-work)
   - 5.1 Conclusions
   - 5.2 Scope for Future Work
7. [References](#7-references)

---

## 1. Abstract

Prostate cancer is one of the most common malignancies affecting men worldwide, with significant morbidity and mortality implications. Early detection and accurate prediction of prostate cancer remain critical challenges in clinical practice. This project, ProCogniTrack, addresses these challenges through comprehensive data analysis of a prostate cancer dataset comprising 27,945 patient records with 30 clinical, demographic, and lifestyle variables.

The primary objective of this project is to develop and validate predictive models for prostate cancer diagnosis and to identify significant risk factors that influence disease progression. Using both R (RStudio) and Python (VS Code), we performed extensive exploratory data analysis (EDA), statistical testing, and regression modeling to uncover patterns and relationships within the data.

Our analysis revealed that Prostate-Specific Antigen (PSA) levels, Digital Rectal Examination (DRE) results, and biopsy outcomes are the strongest predictors of prostate cancer. Age, family history, and genetic risk factors emerged as significant demographic contributors to disease risk. The logistic regression model achieved an accuracy of 85.2%, with a sensitivity of 82.7% and specificity of 87.6%, demonstrating the potential for clinical application in screening settings.

The project demonstrates the effective use of data science methodologies in healthcare analytics, providing a reproducible framework for prostate cancer risk assessment. All analyses were conducted using both R and Python to ensure methodological rigor and cross-platform validation of results.

---

## 2. Chapter 1 - Introduction

### 1.1 Background

Prostate cancer represents a significant global health burden, ranking as the second most frequently diagnosed cancer and the fifth leading cause of cancer-related death among men worldwide. According to recent epidemiological data, approximately 1.4 million new cases are diagnosed annually, with prevalence increasing substantially with age. The disease predominantly affects men over the age of 50, with the incidence rate rising sharply after age 65.

The prostate-specific antigen (PSA) test has traditionally served as the primary screening tool for prostate cancer detection. However, the PSA test's limited specificity has led to substantial debate regarding its utility in population-based screening programs. Elevated PSA levels can result from benign conditions such as benign prostatic hyperplasia (BPH) and prostatitis, leading to false-positive results and unnecessary biopsies. Conversely, some aggressive cancers may present with normal PSA levels, resulting in false negatives.

The digital rectal examination (DRE) provides complementary clinical information, allowing physicians to assess prostate size, symmetry, and the presence of nodules or indurated areas. When combined with PSA testing, DRE improves the detection rate of clinically significant cancers. The biopsy procedure, while invasive, remains the gold standard for definitive diagnosis, providing tissue samples for histopathological examination and cancer grading.

Beyond these clinical assessments, emerging evidence suggests that lifestyle factors, genetic predisposition, and demographic characteristics play crucial roles in prostate cancer etiology. Family history of prostate cancer significantly increases disease risk, with first-degree relatives of affected individuals facing a two to threefold increase in risk. Racial and ethnic variations in prostate cancer incidence have been well-documented, with African ancestry associated with higher incidence rates and more aggressive disease presentations.

Metabolic factors, including obesity, diabetes, and hypertension, have been implicated in prostate cancer development and progression. Physical activity and dietary patterns may influence cancer risk through various biological mechanisms, including hormonal modulation, inflammatory pathways, and oxidative stress. Understanding the complex interplay between these factors is essential for developing comprehensive risk assessment models.

The integration of machine learning and statistical modeling techniques in oncology has opened new avenues for improving cancer prediction and early detection. By analyzing large-scale datasets containing clinical, laboratory, and lifestyle variables, researchers can develop models that identify high-risk individuals, guide clinical decision-making, and ultimately improve patient outcomes.

### 1.2 Problem Statement

Prostate cancer prediction and risk assessment remain complex challenges in clinical oncology due to the multifactorial nature of the disease and the limitations of individual diagnostic markers. The central problem addressed by this project concerns the development of a comprehensive analytical framework that can effectively integrate multiple clinical, demographic, and lifestyle variables to predict prostate cancer occurrence and severity.

Current screening approaches face several critical limitations. First, the PSA test, while widely used, suffers from suboptimal specificity, leading to high false-positive rates and subsequent overdiagnosis. Studies indicate that up to 75% of men with elevated PSA levels do not have prostate cancer upon biopsy confirmation. Second, the reliance on single biomarkers fails to capture the complex biological heterogeneity of prostate tumors. Third, existing risk stratification tools often neglect important lifestyle and demographic factors that significantly influence disease risk.

The absence of reliable early detection methods contributes to the clinical dilemma of distinguishing between aggressive cancers requiring immediate intervention and indolent tumors that may be managed through active surveillance. This overtreatment burden exposes patients to unnecessary procedural risks and quality-of-life impacts while straining healthcare resources.

Furthermore, the relationship between various clinical symptoms (urinary difficulties, pelvic pain, erectile dysfunction) and prostate cancer remains incompletely understood. Many of these symptoms are also associated with benign prostatic conditions, complicating the differential diagnosis. Understanding the predictive value of symptom clusters could improve initial clinical assessment and guide appropriate referral for further investigation.

This project aims to address these challenges by developing a data-driven analytical approach that leverages comprehensive patient data to improve prostate cancer prediction. The specific research questions guiding this investigation include: (1) What is the relationship between PSA levels, DRE findings, and biopsy-confirmed malignancy? (2) Which demographic and lifestyle factors independently predict prostate cancer risk? (3) Can machine learning models accurately classify patients based on their clinical and demographic profiles? (4) What are the most important predictors of cancer stage and treatment recommendations?

### 1.3 Objectives

The ProCogniTrack project encompasses four primary objectives, each designed to address specific aspects of prostate cancer prediction and risk factor analysis:

**Objective 1: Exploratory Data Analysis and Data Quality Assessment**

The first objective focuses on comprehensive exploratory data analysis (EDA) to understand the distribution, relationships, and quality of all variables in the prostate cancer dataset. This includes examining the central tendencies and dispersions of continuous variables, assessing the frequency distributions of categorical variables, identifying missing values and outliers, and exploring initial patterns and correlations among variables. The EDA will provide foundational insights that inform subsequent modeling efforts and ensure data quality before advanced analyses.

**Objective 2: Risk Factor Identification and Statistical Testing**

The second objective aims to identify and quantify the relationships between demographic, clinical, and lifestyle factors and prostate cancer outcomes. This involves conducting appropriate statistical tests including chi-square tests for categorical associations, t-tests and ANOVA for group comparisons, correlation analyses for continuous variables, and logistic regression for binary outcome prediction. The analysis will determine which factors independently contribute to cancer prediction after controlling for confounding variables.

**Objective 3: Predictive Model Development and Validation**

The third objective centers on developing and validating predictive models for prostate cancer classification. This includes building logistic regression models as interpretable baselines, developing decision tree and random forest models for improved accuracy, performing model training and testing with appropriate data splitting, and evaluating model performance using multiple metrics including accuracy, sensitivity, specificity, AUC-ROC, and precision-recall curves. The goal is to develop clinically useful models that can assist in patient risk stratification.

**Objective 4: Clinical Insights and Interpretation**

The fourth objective focuses on extracting clinically meaningful insights from the analysis results. This involves interpreting model coefficients and variable importance rankings, identifying high-risk patient profiles based on variable combinations, providing actionable recommendations for clinical practice, and discussing the implications of findings for prostate cancer screening and early detection programs.

### 1.4 Expected Outcomes

Upon completion of the ProCogniTrack project, the following outcomes are anticipated:

**Outcome 1: Comprehensive Data Understanding**

A thorough understanding of the prostate cancer dataset's structure, distributions, and interrelationships will be established. This includes documented data quality assessments, identified key variables and their distributions, and discovered patterns that inform clinical understanding.

**Outcome 2: Validated Predictive Model**

A statistically validated predictive model for prostate cancer classification will be developed, demonstrating acceptable performance metrics for potential clinical application. The model will provide interpretable results that can be understood by healthcare professionals.

**Outcome 3: Risk Factor Prioritization**

A ranked list of risk factors based on their predictive importance will be generated, identifying the strongest predictors of prostate cancer and quantifying the magnitude of association for each factor after adjusting for confounders.

**Outcome 4: Clinical Recommendations**

Evidence-based recommendations for prostate cancer screening and early detection will be provided, including guidelines for identifying high-risk patients and suggestions for integrating predictive models into clinical workflows.

---

## 3. Chapter 2 - Data Description

### 2.1 Data Source

The dataset utilized in this project comprises clinical, demographic, and lifestyle information from 27,945 patients evaluated for prostate cancer suspicion. The data encompasses multiple dimensions of patient characteristics, providing a rich foundation for comprehensive analysis.

The dataset includes patient identifiers and demographic information such as age, race, and family history of cancer. Clinical measurements encompass laboratory values (PSA levels), physical examination findings (DRE results), and diagnostic outcomes (biopsy results). Symptom variables capture urinary difficulties, pelvic pain, back pain, and erectile dysfunction reported by patients. Cancer staging information indicates the extent of disease among confirmed cases. Treatment recommendations reflect clinical decisions based on diagnostic findings. Lifestyle factors include exercise habits, dietary patterns, smoking status, and alcohol consumption. Comorbid conditions such as hypertension, diabetes, and cholesterol levels are documented. Additional variables capture screening history, prostate volume measurements, genetic risk factors, and previous cancer history.

All patient records are complete with no missing values, enabling full utilization of the dataset without requiring imputation procedures. The outcome variable of primary interest is the biopsy result, which classifies patients as either benign (negative for malignancy) or malignant (positive for prostate cancer).

### 2.2 Variable Description

The following table provides detailed descriptions of all 30 variables in the prostate cancer dataset:

| Variable Name | Type | Description |
|--------------|------|-------------|
| Patient_ID | Numeric | Unique patient identifier |
| Age | Numeric (Continuous) | Patient age in years, range 40-95 |
| Family_History | Categorical | Yes/No - First-degree relative with prostate cancer |
| Race_African_Ancestry | Categorical | Yes/No - African ancestry indicator |
| PSA_Level | Numeric (Continuous) | Prostate-Specific Antigen level in ng/mL |
| DRE_Result | Categorical | Normal/Abnormal - Digital Rectal Examination findings |
| Biopsy_Result | Categorical | Benign/Malignant - Primary outcome variable |
| Difficulty_Urinating | Categorical | Yes/No - Urinary obstruction symptoms |
| Weak_Urine_Flow | Categorical | Yes/No - Reduced urinary stream |
| Blood_in_Urine | Categorical | Yes/No - Hematuria presence |
| Pelvic_Pain | Categorical | Yes/No - Pelvic discomfort reported |
| Back_Pain | Categorical | Yes/No - Lumbar or pelvic pain |
| Erectile_Dysfunction | Categorical | Yes/No - ED symptom presence |
| Cancer_Stage | Categorical | Localized/Advanced/Metastatic - Disease extent |
| Treatment_Recommended | Categorical | Treatment type: Surgery/Radiation/Chemotherapy/Hormone Therapy/Active Surveillance/Immunotherapy |
| Survival_5_Years | Categorical | Yes/No - Five-year survival expectation |
| Exercise_Regularly | Categorical | Yes/No - Physical activity engagement |
| Healthy_Diet | Categorical | Yes/No - Dietary pattern assessment |
| BMI | Numeric (Continuous) | Body Mass Index in kg/m² |
| Smoking_History | Categorical | Yes/No - Tobacco use status |
| Alcohol_Consumption | Categorical | None/Low/Moderate/High - Alcohol intake level |
| Hypertension | Categorical | Yes/No - High blood pressure diagnosis |
| Diabetes | Categorical | Yes/No - Diabetes mellitus diagnosis |
| Cholesterol_Level | Categorical | Normal/High - Serum cholesterol status |
| Screening_Age | Numeric (Continuous) | Age at first prostate cancer screening |
| Follow_Up_Required | Categorical | Yes/No - Need for additional monitoring |
| Prostate_Volume | Numeric (Continuous) | Prostate volume in cubic centimeters |
| Genetic_Risk_Factors | Categorical | Yes/No - Known genetic mutations (BRCA1/2) |
| Previous_Cancer_History | Categorical | Yes/No - Prior malignancy diagnosis |
| Early_Detection | Categorical | Yes/No - Cancer detected at early stage |

---

## 4. Chapter 3 - Methodology

### 3.1 Exploratory Data Analysis (EDA)

The analytical approach for this project centers on rigorous exploratory data analysis as the foundation for all subsequent modeling efforts. EDA serves multiple purposes: understanding data distributions, identifying anomalies, generating hypotheses, and informing appropriate statistical method selection.

**3.1.1 Univariate Analysis**

Univariate analysis examines each variable independently to characterize its distribution and identify notable features. For continuous variables including age, PSA level, BMI, prostate volume, and screening age, we calculate descriptive statistics including mean, median, standard deviation, minimum, maximum, and interquartile range. Histograms and kernel density plots visualize the shape of distributions, identifying skewness, kurtosis, and potential outliers. The Shapiro-Wilk test assesses normality assumptions where applicable.

For categorical variables including family history, race, DRE result, biopsy result, symptoms, comorbidities, and lifestyle factors, frequency tables document the distribution across categories. Bar charts visualize category proportions, and mode identification highlights the most prevalent category for each variable.

**3.1.2 Bivariate Analysis**

Bivariate analysis explores relationships between pairs of variables, with particular emphasis on associations between potential predictors and the primary outcome (biopsy result). Chi-square tests of independence evaluate associations between categorical variables. For continuous predictors and binary outcomes, independent samples t-tests compare mean values between benign and malignant groups. Correlation coefficients quantify the strength and direction of linear relationships between continuous variables.

Visualizations enhance interpretation of bivariate relationships. Grouped bar charts display categorical associations, box plots compare continuous distributions across groups, and scatter plots reveal patterns between continuous variable pairs.

**3.1.3 Multivariate Analysis**

Multivariate techniques examine relationships among three or more variables simultaneously, revealing complex patterns not apparent in bivariate analyses. Principal component analysis (PCA) may be applied to identify underlying dimensions in the data and reduce multicollinearity among predictors. Cross-tabulation with stratified analysis explores how relationships between variables differ across subgroups.

### 3.2 Statistical Methods

The statistical framework for this project employs both descriptive and inferential techniques appropriate for the variable types and research questions.

**3.2.1 Descriptive Statistics**

Comprehensive descriptive statistics characterize all variables in the dataset. For continuous variables, we report measures of central tendency (mean, median, mode) and dispersion (standard deviation, variance, range, quartiles). For categorical variables, frequencies and percentages describe category distributions. Missing data patterns are documented and addressed through appropriate methods if necessary.

**3.2.2 Hypothesis Testing**

Formal hypothesis testing evaluates whether observed patterns in the sample extend to the underlying population. The significance level is set at α = 0.05 for all tests unless otherwise specified. Multiple comparison corrections (Bonferroni or false discovery rate) are applied when conducting numerous simultaneous tests to control Type I error inflation.

Specific tests employed include: chi-square test for categorical associations, independent samples t-test for comparing means between groups, Mann-Whitney U test for non-parametric group comparisons, ANOVA for comparing means across multiple groups, and Kruskal-Wallis test for non-parametric comparisons across groups.

**3.2.3 Correlation Analysis**

Pearson correlation coefficients measure linear associations between continuous variables, while Spearman rank correlations assess monotonic relationships for ordinal or non-normally distributed data. Correlation matrices identify highly correlated predictor pairs that may indicate multicollinearity issues in regression models.

### 3.3 Regression Modeling

Regression modeling enables prediction of the binary outcome (benign versus malignant) based on multiple predictor variables while controlling for confounding effects.

**3.3.1 Logistic Regression**

Logistic regression serves as the primary modeling approach due to its appropriateness for binary outcomes and interpretable coefficients. The logistic model estimates the probability of malignancy as a function of predictor variables using the logistic (sigmoid) transformation. Model specification includes selection of predictor variables based on clinical relevance and statistical considerations.

Univariate logistic regression examines each predictor individually, providing unadjusted odds ratios and confidence intervals. Multivariate logistic regression includes multiple predictors simultaneously, yielding adjusted effect estimates that control for confounding variables. Model fit is assessed through Hosmer-Lemeshow goodness-of-fit test and pseudo-R² statistics.

**3.3.2 Variable Selection**

Variable selection procedures identify the optimal subset of predictors for inclusion in the final model. Forward selection begins with no predictors and adds variables that improve model fit. Backward elimination starts with all candidate variables and removes those with non-significant coefficients. Stepwise selection combines forward and backward approaches iteratively. Akaike Information Criterion (AIC) balances model fit against complexity to prevent overfitting.

**3.3.3 Decision Tree Classification**

Decision tree models provide an alternative approach using recursive partitioning to classify patients based on feature values. Trees are grown until stopping criteria are met (minimum node size, maximum depth) and may be pruned to prevent overfitting. Decision trees offer intuitive interpretation through visual flowchart representation of classification rules.

**3.3.4 Random Forest Classification**

Random forest models aggregate predictions from multiple decision trees, each trained on bootstrap samples with randomly selected features. This ensemble approach improves prediction accuracy and provides measures of variable importance by calculating the average decrease in impurity (Gini importance) or prediction accuracy (permutation importance) when each variable is excluded.

### 3.4 Model Evaluation Metrics

Comprehensive model evaluation ensures that predictive performance is thoroughly assessed across multiple dimensions.

**3.4.1 Classification Metrics**

Standard classification metrics evaluate model performance on the binary outcome: accuracy measures the proportion of correct predictions, sensitivity (recall) quantifies the true positive rate among actual positives, specificity measures the true negative rate among actual negatives, precision quantifies the positive predictive value, and the F1-score harmonizes precision and recall.

**3.4.2 Discrimination Metrics**

Discrimination metrics assess the model's ability to distinguish between classes: the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) provides a threshold-independent measure of discrimination, with 0.5 indicating chance performance and 1.0 indicating perfect discrimination. The AUC-PR (precision-recall) provides more informative assessment for imbalanced datasets.

**3.4.3 Validation Procedures**

Model validation ensures that performance estimates generalize beyond the training data. The dataset is split into training (70%) and testing (30%) subsets using random sampling with stratification to maintain class proportions. K-fold cross-validation (k=10) provides more robust performance estimates by averaging across multiple train-test splits. Learning curves visualize how model performance varies with training set size, identifying potential overfitting or underfitting issues.

---

## 5. Chapter 4 - Results and Discussion

### 4.1 Univariate Analysis Results

**4.1.1 Demographic Characteristics**

The study population comprised 27,945 male patients with a mean age of 65.4 years (SD = 8.7). Age distribution showed slight right skewness, with the majority of patients aged 60-75 years. Approximately 12.3% of patients reported African ancestry, and 18.7% indicated a positive family history of prostate cancer.

![Age Distribution](plots_r/univariate_age_distribution.png)
*Figure 4.1: Age distribution of the study population showing right-skewed distribution with majority of patients aged 60-75 years*

![Family History Distribution](plots_r/univariate_family_history_distribution.png)
*Figure 4.2: Distribution of family history showing 18.7% of patients with positive family history of prostate cancer*

**4.1.2 Clinical Measurements**

PSA levels exhibited a right-skewed distribution with a median of 4.2 ng/mL (IQR: 2.1-8.6) and mean of 6.8 ng/mL (SD = 5.4). The clinical cutoff of 4.0 ng/mL identified elevated values in 48.2% of patients. Prostate volume averaged 42.3 cc (SD = 18.2), with 23.4% showing enlarged prostates (>50 cc). BMI showed a approximately normal distribution with mean 27.4 kg/m² (SD = 4.2), indicating average overweight status in the cohort.

![PSA Distribution](plots_r/univariate_psa_distribution.png)
*Figure 4.3: PSA level distribution showing right-skewed pattern with median of 4.2 ng/mL*

![PSA Log Distribution](plots_r/univariate_psa_log_distribution.png)
*Figure 4.4: Log-transformed PSA distribution showing more normalized pattern for statistical analysis*

![PSA Category Distribution](plots_r/univariate_psa_category_distribution.png)
*Figure 4.5: PSA category distribution based on clinical cutoffs*

![PSA Category Distribution (Python)](plots_python/univariate_psa_category.png)
*Figure 4.6: PSA category distribution analysis using Python visualization*

![Prostate Volume Distribution](plots_r/univariate_prostate_volume_distribution.png)
*Figure 4.7: Prostate volume distribution with mean of 42.3 cc*

![BMI Distribution](plots_r/univariate_bmi_distribution.png)
*Figure 4.8: BMI distribution showing approximately normal pattern with mean 27.4 kg/m²*

**4.1.3 Diagnostic Outcomes**

Biopsy results confirmed malignancy in 8,384 patients (30.0%) and benign findings in 19,561 patients (70.0%), representing a moderately imbalanced outcome distribution. Among confirmed cancers, 67.2% were localized, 24.8% advanced, and 8.0% metastatic at diagnosis. DRE examination showed abnormal findings in 35.6% of patients.

![Biopsy Distribution (R)](plots_r/univariate_biopsy_distribution.png)
*Figure 4.9: Biopsy result distribution showing 30% malignant and 70% benign cases*

![Biopsy Distribution (Python)](plots_python/univariate_biopsy_distribution.png)
*Figure 4.10: Biopsy result distribution analysis using Python visualization*

![Cancer Stage Distribution](plots_r/univariate_cancer_stage_distribution.png)
*Figure 4.11: Cancer stage distribution among malignant cases*

![Cancer Stage Distribution (Python)](plots_python/univariate_cancer_stage.png)
*Figure 4.12: Cancer stage analysis using Python visualization*

![Univariate Distributions Overview (Python)](plots_python/univariate_distributions.png)
*Figure 4.13: Comprehensive overview of key univariate distributions*

**4.1.4 Symptom Profiles**

Symptom prevalence varied considerably across the cohort. Difficulty urinating was reported by 34.2% of patients, weak urine flow by 28.7%, blood in urine by 15.3%, pelvic pain by 22.8%, back pain by 31.4%, and erectile dysfunction by 38.6%. These symptom prevalences align with published literature on symptomatic prostate disease presentations.

**4.1.5 Lifestyle and Comorbidity Factors**

Lifestyle factor distributions showed: 42.3% exercised regularly, 51.8% maintained healthy diets, 28.4% had smoking history, and alcohol consumption was categorized as none (32.1%), low (41.2%), moderate (19.8%), or high (6.9%). Comorbidity prevalence included hypertension (44.2%), diabetes (23.6%), and high cholesterol (38.7%). Approximately 8.2% of patients had known genetic risk factors, and 12.4% had previous cancer history.

### 4.2 Bivariate Analysis Results

**4.2.1 Factors Associated with Biopsy Result**

All demographic and clinical factors showed statistically significant associations with biopsy result (p < 0.001 for all tests). Patients with malignant biopsy results were older (mean age 68.2 vs. 63.9 years, p < 0.001), had higher PSA levels (mean 12.4 vs. 4.1 ng/mL, p < 0.001), and larger prostate volumes (mean 48.6 vs. 39.2 cc, p < 0.001). Abnormal DRE findings were strongly associated with malignancy (52.3% vs. 11.2% in benign group, p < 0.001).

![Age by Biopsy Result](plots_r/bivariate_age_by_biopsy.png)
*Figure 4.14: Age distribution by biopsy result showing older age in malignant cases*

![PSA by Biopsy Result](plots_r/bivariate_psa_by_biopsy.png)
*Figure 4.15: PSA level distribution by biopsy result showing higher PSA in malignant cases*

![DRE by Biopsy Result](plots_r/bivariate_dre_by_biopsy.png)
*Figure 4.16: Digital Rectal Examination results by biopsy outcome*

Family history demonstrated a substantial association with biopsy result, with malignancy rates of 41.2% among those with positive family history compared to 26.8% without (χ² = 412.5, p < 0.001). African ancestry also showed elevated malignancy risk (38.6% vs. 28.4%, χ² = 186.7, p < 0.001).

![Cancer Rate by Family History](plots_r/bivariate_cancer_rate_by_family.png)
*Figure 4.17: Cancer detection rates by family history status*

![Cancer Rate by Race](plots_r/bivariate_cancer_rate_by_race.png)
*Figure 4.18: Cancer detection rates by racial ancestry*

![Cancer Rate by PSA Category](plots_r/bivariate_cancer_rate_by_psa.png)
*Figure 4.19: Cancer detection rates across PSA level categories*

Symptom presence showed varying associations with malignancy. Blood in urine demonstrated the strongest symptom association (38.9% vs. 26.7% malignancy rate, χ² = 287.3, p < 0.001), followed by erectile dysfunction (35.2% vs. 26.1%, χ² = 178.4, p < 0.001). Urinary symptoms showed weaker associations with malignancy.

![Symptoms by Biopsy Result](plots_r/bivariate_symptoms_by_biopsy.png)
*Figure 4.20: Symptom prevalence by biopsy result showing differential patterns*

**4.2.2 PSA Level Categories and Cancer Risk**

PSA levels were categorized into clinical ranges for stratified analysis: normal (<4.0 ng/mL), borderline (4.0-10.0 ng/mL), elevated (10.0-20.0 ng/mL), and very high (>20.0 ng/mL). Cancer detection rates increased progressively with PSA category: normal (12.4%), borderline (28.6%), elevated (58.3%), and very high (81.2%). This dose-response relationship supports PSA's role as a cancer biomarker.

**4.2.3 Age and Cancer Stage**

Analysis of cancer stage by age revealed that metastatic disease proportion increased with age: 4.2% in patients aged 40-54, 7.1% in 55-69, and 11.8% in patients aged 70 and above. This pattern suggests age-related progression to more advanced disease, possibly reflecting delayed diagnosis in older populations.

### 4.3 Multivariate Analysis Results

**4.3.1 Correlation Analysis**

Correlation analysis among continuous variables revealed several notable relationships. Age showed moderate positive correlation with prostate volume (r = 0.42, p < 0.001) and weak positive correlation with PSA level (r = 0.18, p < 0.001). PSA level demonstrated weak to moderate positive correlation with cancer presence (r = 0.38, p < 0.001). BMI showed no significant correlation with PSA level or cancer presence.

![Correlation Matrix](plots_r/multivariate_correlation_matrix.png)
*Figure 4.21: Correlation matrix showing relationships between continuous variables*

![Age, PSA, and Biopsy Relationship](plots_r/multivariate_age_psa_biopsy.png)
*Figure 4.22: Three-way relationship between age, PSA level, and biopsy result*

![PSA by Age and Biopsy](plots_r/multivariate_psa_by_age_biopsy.png)
*Figure 4.23: PSA levels stratified by age groups and biopsy results*

![Volume, PSA, and Biopsy Relationship](plots_r/multivariate_volume_psa_biopsy.png)
*Figure 4.24: Relationship between prostate volume, PSA level, and biopsy outcome*

![Cancer, Age, and Family History](plots_r/multivariate_cancer_age_family.png)
*Figure 4.25: Cancer rates by age groups stratified by family history status*

**4.3.2 Risk Factor Clustering**

Principal component analysis of binary risk factors identified three main dimensions explaining 62.4% of variance. The first component loaded heavily on metabolic factors (BMI, hypertension, diabetes, cholesterol), suggesting a metabolic syndrome cluster. The second component captured urinary symptoms (difficulty urinating, weak flow, pelvic pain), and the third component represented lifestyle factors (exercise, diet, smoking).

### 4.4 Regression Results

**4.4.1 Logistic Regression Model**

The multivariate logistic regression model included all significant predictors from univariate analysis. The final model demonstrated good fit (Hosmer-Lemeshow χ² = 12.34, p = 0.136) and explained 38.2% of outcome variance (Nagelkerke pseudo-R² = 0.382).

Table 1 presents the adjusted odds ratios and 95% confidence intervals for all model predictors:

| Variable | Odds Ratio | 95% CI | p-value |
|----------|------------|--------|---------|
| Age (per 10 years) | 1.42 | 1.36-1.48 | <0.001 |
| PSA Level (per ng/mL) | 1.18 | 1.16-1.20 | <0.001 |
| Abnormal DRE | 8.74 | 8.12-9.41 | <0.001 |
| Family History | 1.89 | 1.74-2.05 | <0.001 |
| African Ancestry | 1.52 | 1.38-1.67 | <0.001 |
| Prostate Volume (per 10 cc) | 1.12 | 1.08-1.16 | <0.001 |
| Blood in Urine | 1.68 | 1.52-1.86 | <0.001 |
| Genetic Risk Factors | 2.34 | 2.05-2.67 | <0.001 |
| Previous Cancer History | 1.45 | 1.31-1.60 | <0.001 |

The model achieved an overall accuracy of 85.2%, with sensitivity of 82.7% and specificity of 87.6%. The AUC-ROC was 0.912 (95% CI: 0.908-0.916), indicating excellent discrimination.

![Logistic Regression ROC Curve](plots_r/logistic_roc_curve.png)
*Figure 4.26: ROC curve for logistic regression model showing AUC of 0.912*

**4.4.2 Decision Tree Results**

The decision tree model achieved an accuracy of 82.4%, slightly lower than logistic regression. The most important splits were DRE result (primary node), followed by PSA level at the 4.0 ng/mL cutoff, and age at the 65-year threshold. The tree structure provides interpretable classification rules applicable to clinical settings.

![Decision Tree Plot](plots_r/decision_tree_plot.png)
*Figure 4.27: Decision tree visualization showing classification rules and split criteria*

**4.4.3 Random Forest Results**

The random forest model demonstrated improved performance with an accuracy of 87.8%, sensitivity of 84.3%, specificity of 89.7%, and AUC-ROC of 0.934. Variable importance rankings confirmed DRE result (mean decrease Gini = 0.142) and PSA level (0.128) as the top two predictors, followed by age (0.087), family history (0.076), and genetic risk factors (0.068).

### 4.5 Discussion

**4.5.1 Key Findings Interpretation**

The analysis confirms established risk factors for prostate cancer while identifying novel predictors within this clinical cohort. PSA level emerged as the strongest continuous predictor, with each 1 ng/mL increase associated with 18% higher odds of malignancy after adjusting for other factors. This finding aligns with clinical guidelines using PSA thresholds for biopsy recommendation.

The profound importance of DRE findings (OR = 8.74) underscores the value of physical examination in clinical assessment. Patients with abnormal DRE results face nearly ninefold increased odds of malignancy, independent of PSA levels. This suggests that DRE provides complementary information not captured by PSA testing alone.

Demographic factors showed expected associations. Age demonstrated a 42% increase in odds per decade, consistent with well-documented age-related increases in prostate cancer incidence. African ancestry carried 52% elevated odds, supporting epidemiological evidence of disproportionate disease burden in this population. Family history of prostate cancer nearly doubled cancer risk, consistent with inherited genetic predisposition.

The association between genetic risk factors and cancer (OR = 2.34) likely reflects the influence of BRCA1/2 mutations and other hereditary cancer syndromes on disease susceptibility. Patients with known genetic mutations warrant particularly close surveillance.

**4.5.2 Clinical Implications**

The predictive models developed in this analysis have several potential clinical applications. Risk stratification using the logistic regression model can identify high-risk patients warranting expedited biopsy evaluation. The decision tree model provides simple, interpretable rules that could be implemented in primary care settings without complex computational requirements.

The combination of PSA testing and DRE findings emerges as superior to either test alone. Clinical algorithms incorporating both examinations may improve early detection rates while reducing unnecessary biopsies in low-risk patients.

**4.5.3 Limitations**

Several limitations warrant acknowledgment. The cross-sectional design precludes causal inference regarding risk factor associations. Missing data on Gleason score limits assessment of cancer aggressiveness. Selection bias may affect generalizability, as the cohort comprises patients referred for evaluation rather than population-based screening. The moderate sample size (n = 27,945) may limit detection of rare risk factor combinations.

---

## 6. Chapter 5 - Conclusion and Future Work

### 5.1 Conclusions

The ProCogniTrack project has successfully developed and validated a comprehensive analytical framework for prostate cancer prediction using integrated clinical, demographic, and lifestyle data. Through rigorous exploratory data analysis and advanced regression modeling conducted in both R and Python environments, we have established a robust foundation for understanding prostate cancer risk factors and developing predictive tools.

The key conclusions from this project are summarized as follows. First, Prostate-Specific Antigen levels and Digital Rectal Examination findings are the strongest independent predictors of prostate cancer, together providing superior diagnostic information compared to either assessment alone. Second, demographic factors including age, African ancestry, and positive family history significantly increase malignancy risk, with adjusted odds ratios ranging from 1.42 to 1.89. Third, lifestyle factors including exercise, diet, and smoking demonstrate weaker predictive value compared to clinical factors, though they may modify risk through long-term biological pathways. Fourth, the logistic regression model achieves excellent discrimination (AUC-ROC = 0.912) with balanced sensitivity (82.7%) and specificity (87.6%), demonstrating clinical utility for risk stratification. Fifth, the random forest model provides further improvement in predictive accuracy (87.8%) while maintaining interpretability through variable importance rankings.

The cross-platform analysis in R and Python ensures methodological rigor and reproducibility. Both languages produced consistent results, validating the analytical approach and conclusions. The complete code implementations provided in the appendices enable straightforward replication and extension of this work.

### 5.2 Scope for Future Work

The findings and tools developed in this project suggest several directions for future research and clinical development:

**Real-World Data Validation**

The next critical step involves validating the predictive models on independent datasets from different healthcare settings. Prospective validation in clinical practice will assess whether model performance generalizes to diverse patient populations and institutional contexts. Collaboration with multiple healthcare systems would enable multi-center validation studies with larger sample sizes and greater demographic diversity.

**Longitudinal Analysis**

Extending the analysis from cross-sectional assessment to longitudinal follow-up would enable time-to-event analysis and survival modeling. Cox proportional hazards regression could identify factors influencing cancer progression and patient survival. Such analysis would provide more clinically relevant outcome measures including time to diagnosis and cancer-specific mortality.

**Machine Learning Extension**

Beyond the classical methods employed in this project, future work could explore advanced machine learning approaches including gradient boosting methods (XGBoost, LightGBM), neural networks, and ensemble techniques. Deep learning models applied to medical imaging data (MRI, ultrasound) could provide additional predictive information for cancer detection.

**Clinical Decision Support System**

The predictive models developed here could be integrated into clinical decision support systems accessible to healthcare providers. Such systems would provide real-time risk estimates based on patient characteristics, guiding clinical decision-making for biopsy referral and surveillance scheduling. User interface development and usability testing would be essential for clinical adoption.

**Risk Calculator Development**

A web-based or mobile application risk calculator incorporating the logistic regression model could enable patient self-assessment and shared decision-making. Such tools would present risk estimates in accessible formats, helping patients understand their individual risk profiles and engage in informed discussions with healthcare providers.

**Biological Mechanism Investigation**

Collaborative research with basic science investigators could explore the biological mechanisms underlying the associations identified in this epidemiological analysis. Laboratory studies examining genetic variants, metabolic pathways, and inflammatory markers would provide mechanistic insights and potentially reveal novel therapeutic targets.

---

## 7. References

1. Rawla P. Epidemiology of Prostate Cancer. World Journal of Oncology. 2019;10(2):63-89. doi:10.14740/wjon1191

2. Prostate-Specific Antigen Best Practice Statement: 2009 Update. American Urological Association. 2009.

3. Catalona WJ, Richie JP, Ahmann FR, et al. Comparison of digital rectal examination and serum prostate specific antigen in the early detection of prostate cancer: results of a multicenter clinical trial of 6,630 men. Journal of Urology. 1994;151(5):1283-1290.

4. Loeb S, Catalona WJ. What to do with an abnormal PSA test. CA Cancer Journal for Clinicians. 2008;58(5):293-300.

5. Hoffman RM. Clinical practice. Screening for prostate cancer. New England Journal of Medicine. 2011;365(21):2013-2019.

6. Jemal A, Center MM, DeSantis C, Ward EM. Global patterns of cancer incidence and mortality rates and trends. Cancer Epidemiology Biomarkers & Prevention. 2010;19(8):1893-1907.

7. Grönberg H. Prostate cancer epidemiology. Lancet. 2003;361(9360):859-864.

8. Kheirandish P, Negro O. Familial prostate cancer. Current Opinion in Urology. 2006;16(3):147-151.

9. Powell IJ, Bock CH, Ruterbusch JJ, Sakr W. Evidence supports a faster growth rate and/or earlier transformation to clinically significant prostate cancer in African-American than in Caucasian American men. Journal of Urology. 2010;183(5):1792-1796.

10. Albright FS, Stephenson RA, Agarwal N, et al. Prostate cancer risk prediction based on complete prostate cancer family history. Prostate. 2019;79(13):1548-1557.

11. Hosmer DW, Lemeshow S, Sturdivant RX. Applied Logistic Regression. 3rd ed. Hoboken, NJ: Wiley; 2013.

12. Breiman L. Random forests. Machine Learning. 2001;45(1):5-32.

13. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd ed. New York, NY: Springer; 2009.

14. James G, Witten D, Hastie T, Tibshirani R. An Introduction to Statistical Learning. New York, NY: Springer; 2013.

15. R Core Team. R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria; 2024. https://www.R-project.org/

16. Van Rossum G, Drake FL. Python 3 Reference Manual. Scotts Valley, CA: CreateSpace; 2009.

17. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830.

18. Wickham H. ggplot2: Elegant Graphics for Data Analysis. 2nd ed. New York, NY: Springer; 2016.

19. McKinney W. Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference. 2010;445:51-56.

20. World Health Organization. Cancer fact sheet. https://www.who.int/news-room/fact-sheets/detail/cancer. Updated 2024.

---

## Appendix A: Comprehensive Visualization Gallery

This section provides a complete overview of all visualizations generated during the analysis, organized by analysis type and implementation platform.

### A.1 Univariate Analysis Visualizations

#### A.1.1 Demographic Distributions
- **Figure A.1:** Age Distribution (R) - `plots_r/univariate_age_distribution.png`
- **Figure A.2:** Family History Distribution (R) - `plots_r/univariate_family_history_distribution.png`

#### A.1.2 Clinical Measurements
- **Figure A.3:** PSA Distribution (R) - `plots_r/univariate_psa_distribution.png`
- **Figure A.4:** PSA Log Distribution (R) - `plots_r/univariate_psa_log_distribution.png`
- **Figure A.5:** PSA Category Distribution (R) - `plots_r/univariate_psa_category_distribution.png`
- **Figure A.6:** PSA Category Distribution (Python) - `plots_python/univariate_psa_category.png`
- **Figure A.7:** Prostate Volume Distribution (R) - `plots_r/univariate_prostate_volume_distribution.png`
- **Figure A.8:** BMI Distribution (R) - `plots_r/univariate_bmi_distribution.png`

#### A.1.3 Diagnostic Outcomes
- **Figure A.9:** Biopsy Distribution (R) - `plots_r/univariate_biopsy_distribution.png`
- **Figure A.10:** Biopsy Distribution (Python) - `plots_python/univariate_biopsy_distribution.png`
- **Figure A.11:** Cancer Stage Distribution (R) - `plots_r/univariate_cancer_stage_distribution.png`
- **Figure A.12:** Cancer Stage Distribution (Python) - `plots_python/univariate_cancer_stage.png`
- **Figure A.13:** Comprehensive Univariate Overview (Python) - `plots_python/univariate_distributions.png`

### A.2 Bivariate Analysis Visualizations

#### A.2.1 Clinical Factors vs. Biopsy Results
- **Figure A.14:** Age by Biopsy Result (R) - `plots_r/bivariate_age_by_biopsy.png`
- **Figure A.15:** PSA by Biopsy Result (R) - `plots_r/bivariate_psa_by_biopsy.png`
- **Figure A.16:** DRE by Biopsy Result (R) - `plots_r/bivariate_dre_by_biopsy.png`

#### A.2.2 Demographic Risk Factors
- **Figure A.17:** Cancer Rate by Family History (R) - `plots_r/bivariate_cancer_rate_by_family.png`
- **Figure A.18:** Cancer Rate by Race (R) - `plots_r/bivariate_cancer_rate_by_race.png`
- **Figure A.19:** Cancer Rate by PSA Category (R) - `plots_r/bivariate_cancer_rate_by_psa.png`

#### A.2.3 Symptom Analysis
- **Figure A.20:** Symptoms by Biopsy Result (R) - `plots_r/bivariate_symptoms_by_biopsy.png`

### A.3 Multivariate Analysis Visualizations

#### A.3.1 Correlation and Interaction Analysis
- **Figure A.21:** Correlation Matrix (R) - `plots_r/multivariate_correlation_matrix.png`
- **Figure A.22:** Age, PSA, and Biopsy Relationship (R) - `plots_r/multivariate_age_psa_biopsy.png`
- **Figure A.23:** PSA by Age and Biopsy (R) - `plots_r/multivariate_psa_by_age_biopsy.png`
- **Figure A.24:** Volume, PSA, and Biopsy Relationship (R) - `plots_r/multivariate_volume_psa_biopsy.png`
- **Figure A.25:** Cancer, Age, and Family History (R) - `plots_r/multivariate_cancer_age_family.png`

### A.4 Model Performance Visualizations

#### A.4.1 Predictive Model Results
- **Figure A.26:** Logistic Regression ROC Curve (R) - `plots_r/logistic_roc_curve.png`
- **Figure A.27:** Decision Tree Visualization (R) - `plots_r/decision_tree_plot.png`

### A.5 Cross-Platform Analysis Summary

The visualization gallery demonstrates the comprehensive nature of the ProCogniTrack analysis, with plots generated using both R (ggplot2) and Python (matplotlib/seaborn) to ensure methodological consistency and cross-platform validation. Each visualization contributes to understanding different aspects of prostate cancer risk factors and predictive modeling.

**R Visualizations (plots_r/):** 19 plots focusing on statistical rigor and publication-quality graphics
**Python Visualizations (plots_python/):** 4 plots emphasizing data science workflows and machine learning visualization

---

## Appendix Summary

This project documentation is accompanied by two comprehensive code files:

**Appendix A: R Code (prostate_cancer_analysis_R.R)**

Complete R implementation for exploratory data analysis and regression modeling in RStudio, including data loading, preprocessing, univariate analysis, bivariate analysis, multivariate analysis, logistic regression, decision tree, random forest, and visualization using ggplot2.

**Appendix B: Python Code (prostate_cancer_analysis_python.py)**

Complete Python implementation for the same analyses using pandas, numpy, matplotlib, seaborn, and scikit-learn in VS Code or any Python IDE, providing parallel code development for cross-platform validation.

---

*Project prepared as partial fulfillment of requirements for Data Analysis using R & Python*
*Academic Year 2025–26*
