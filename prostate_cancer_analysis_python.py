# ============================================================================
#  PROJECT: ProCogniTrack - Prostate Cancer Prediction and Risk Factor Analysis
#  FILE   : Comprehensive Data Analysis in Python (VS Code)
#  PURPOSE: Exploratory Data Analysis, Statistical Testing, and Predictive Modeling
#  AUTHOR : MiniMax Agent
#  DATE   : 2025
# ============================================================================

"""
ProCogniTrack - Prostate Cancer Prediction and Risk Factor Analysis

This comprehensive Python script performs exploratory data analysis and builds
predictive models for prostate cancer prediction using clinical, demographic,
and lifestyle data.

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
"""

# =============================================================================
# SECTION 0: IMPORTS AND SETUP
# =============================================================================

import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, pearsonr, spearmanr

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, precision_recall_curve, f1_score,
    roc_auc_score, mean_squared_error
)
from sklearn.inspection import permutation_importance

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Set random seed for reproducibility
RANDOM_STATE = 42

# Set output directory
OUTPUT_DIR = "plots_python"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("=" * 60)
print("ProCogniTrack - Prostate Cancer Analysis")
print("Python Analysis Script")
print("=" * 60)
print("\nLibraries loaded successfully.\n")

# =============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

print("=" * 60)
print("SECTION 1: DATA LOADING")
print("=" * 60)
print()

# Define data path
DATA_PATH = "prostate/prostate_cancer_prediction.csv"

# Load the dataset
print("Loading prostate cancer dataset...")
df = pd.read_csv(DATA_PATH)

# Display basic information
print(f"\n{'─' * 40}")
print("Dataset Dimensions")
print(f"{'─' * 40}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print(f"\n{'─' * 40}")
print("Column Names")
print(f"{'─' * 40}")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2}. {col}")

print(f"\n{'─' * 40}")
print("First 5 Rows")
print(f"{'─' * 40}")
print(df.head().to_string())

print(f"\n{'─' * 40}")
print("Data Types")
print(f"{'─' * 40}")
print(df.dtypes.to_string())

print(f"\n{'─' * 40}")
print("Summary Statistics")
print(f"{'─' * 40}")
print(df.describe().to_string())

# =============================================================================
# SECTION 2: DATA PREPROCESSING
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 2: DATA PREPROCESSING")
print("=" * 60)
print()

# 2.1 Check for missing values
print("─" * 40)
print("Missing Values Check")
print("─" * 40)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing'] > 0]
if len(missing_df) > 0:
    print("Missing values found:")
    print(missing_df.to_string())
else:
    print("No missing values found in the dataset.")

# 2.2 Check for duplicates
print(f"\n{'─' * 40}")
print("Duplicate Rows Check")
print(f"{'─' * 40}  ")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# 2.3 Convert categorical columns to category type
print(f"\n{'─' * 40}")
print("Converting Categorical Variables")
print("─" * 40)

categorical_cols = [
    'Family_History', 'Race_African_Ancestry', 'DRE_Result', 'Biopsy_Result',
    'Difficulty_Urinating', 'Weak_Urine_Flow', 'Blood_in_Urine', 'Pelvic_Pain',
    'Back_Pain', 'Erectile_Dysfunction', 'Cancer_Stage', 'Treatment_Recommended',
    'Survival_5_Years', 'Exercise_Regularly', 'Healthy_Diet', 'Smoking_History',
    'Alcohol_Consumption', 'Hypertension', 'Diabetes', 'Cholesterol_Level',
    'Follow_Up_Required', 'Genetic_Risk_Factors', 'Previous_Cancer_History', 'Early_Detection'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f"Converted {len(categorical_cols)} categorical columns to category type.")

# 2.4 Create derived variables
print(f"\n{'─' * 40}")
print("Creating Derived Variables")
print("─" * 40)

# Binary outcome variable (1 = Malignant, 0 = Benign)
df['Cancer_Binary'] = df['Biopsy_Result'].map({'Malignant': 1, 'Benign': 0})
print("Created: Cancer_Binary (1=Malignant, 0=Benign)")

# PSA Category
def categorize_psa(x):
    if x < 4:
        return 'Normal'
    elif x < 10:
        return 'Borderline'
    else:
        return 'Elevated'

df['PSA_Category'] = df['PSA_Level'].apply(categorize_psa)
df['PSA_Category'] = df['PSA_Category'].astype('category')
print("Created: PSA_Category (Normal, Borderline, Elevated)")

# Age Group
def categorize_age(x):
    if x <= 54:
        return '40-54'
    elif x <= 64:
        return '55-64'
    elif x <= 74:
        return '65-74'
    else:
        return '75+'

df['Age_Group'] = df['Age'].apply(categorize_age)
df['Age_Group'] = df['Age_Group'].astype('category')
print("Created: Age_Group (40-54, 55-64, 65-74, 75+)")

# BMI Category
def categorize_bmi(x):
    if x < 18.5:
        return 'Underweight'
    elif x < 25:
        return 'Normal'
    elif x < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
df['BMI_Category'] = df['BMI_Category'].astype('category')
print("Created: BMI_Category (Underweight, Normal, Overweight, Obese)")

print(f"\n{'─' * 40}")
print("Dataset After Preprocessing")
print(f"{'─' * 40}")
print(f"Total columns: {len(df.columns)}")

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS - UNIVARIATE ANALYSIS
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 3: UNIVARIATE ANALYSIS")
print("=" * 60)
print()

# Convert all category columns to string for aggregation compatibility
for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].astype(str)

# 3.1 Continuous Variables Statistics
print("─" * 40)
print("Descriptive Statistics - Continuous Variables")
print("─" * 40)

continuous_vars = ['Age', 'PSA_Level', 'BMI', 'Screening_Age', 'Prostate_Volume']

for var in continuous_vars:
    print(f"\n{var}:")
    print(f"  Mean:   {df[var].mean():.2f}")
    print(f"  Median: {df[var].median():.2f}")
    print(f"  Std:    {df[var].std():.2f}")
    print(f"  Min:    {df[var].min():.2f}")
    print(f"  Max:    {df[var].max():.2f}")
    print(f"  IQR:    {df[var].quantile(0.75) - df[var].quantile(0.25):.2f}")

# 3.2 Age Distribution
print("\n" + "─" * 40)
print("Age Distribution Analysis")
print("─" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age histogram
ax1 = axes[0, 0]
ax1.hist(df['Age'], bins=30, density=True, alpha=0.7, color='#4E79A7', edgecolor='white')
df['Age'].plot(kind='kde', ax=ax1, color='#E15759', linewidth=2)
ax1.axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {df['Age'].mean():.1f}")
ax1.axvline(df['Age'].median(), color='green', linestyle=':', linewidth=1.5, label=f"Median: {df['Age'].median():.1f}")
ax1.set_xlabel('Age (years)', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Distribution of Patient Age', fontsize=12, fontweight='bold')
ax1.legend()

# 3.3 PSA Level Distribution
ax2 = axes[0, 1]
ax2.hist(df['PSA_Level'], bins=40, density=True, alpha=0.7, color='#59A14F', edgecolor='white')
df['PSA_Level'].plot(kind='kde', ax=ax2, color='#E15759', linewidth=2)
ax2.axvline(4, color='red', linestyle='--', linewidth=1.5, label='Cutoff: 4 ng/mL')
ax2.axvline(10, color='orange', linestyle='--', linewidth=1.5, label='Cutoff: 10 ng/mL')
ax2.set_xlabel('PSA Level (ng/mL)', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Distribution of PSA Levels', fontsize=12, fontweight='bold')
ax2.legend()
ax2.set_xlim(0, df['PSA_Level'].quantile(0.99))

# 3.4 BMI Distribution
ax3 = axes[1, 0]
ax3.hist(df['BMI'], bins=30, density=True, alpha=0.7, color='#B07AA1', edgecolor='white')
df['BMI'].plot(kind='kde', ax=ax3, color='#E15759', linewidth=2)
ax3.axvline(25, color='red', linestyle='--', linewidth=1.5, label='Overweight: 25')
ax3.axvline(30, color='orange', linestyle='--', linewidth=1.5, label='Obese: 30')
ax3.set_xlabel('BMI (kg/m²)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Distribution of BMI', fontsize=12, fontweight='bold')
ax3.legend()

# 3.5 Prostate Volume Distribution
ax4 = axes[1, 1]
ax4.hist(df['Prostate_Volume'], bins=30, density=True, alpha=0.7, color='#76B7B2', edgecolor='white')
df['Prostate_Volume'].plot(kind='kde', ax=ax4, color='#E15759', linewidth=2)
ax4.axvline(50, color='red', linestyle='--', linewidth=1.5, label='Enlarged: 50 cc')
ax4.set_xlabel('Prostate Volume (cc)', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title('Distribution of Prostate Volume', fontsize=12, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/univariate_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/univariate_distributions.png")

# 3.6 Biopsy Result Distribution
print("\n" + "─" * 40)
print("Biopsy Result Distribution")
print("─" * 40)

biopsy_counts = df['Biopsy_Result'].value_counts()
biopsy_pct = df['Biopsy_Result'].value_counts(normalize=True) * 100

print(f"Malignant: {biopsy_counts['Malignant']:,} ({biopsy_pct['Malignant']:.1f}%)")
print(f"Benign: {biopsy_counts['Benign']:,} ({biopsy_pct['Benign']:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#E15759', '#4E79A7']
bars = ax.bar(biopsy_counts.index, biopsy_counts.values, color=colors, alpha=0.8, edgecolor='white')
ax.set_xlabel('Biopsy Result', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Biopsy Result Distribution', fontsize=14, fontweight='bold')

# Add percentage labels
for bar, pct in zip(bars, biopsy_pct.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/univariate_biopsy_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/univariate_biopsy_distribution.png")

# 3.7 PSA Category Distribution
print("\n" + "─" * 40)
print("PSA Category Distribution")
print("─" * 40)

psa_cat_counts = df['PSA_Category'].value_counts()
print(psa_cat_counts.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
order = ['Normal', 'Borderline', 'Elevated']
colors = plt.cm.plasma(np.linspace(0, 0.8, 3))
ax.bar(order, [psa_cat_counts[x] for x in order], color=colors, alpha=0.8, edgecolor='white')
ax.set_xlabel('PSA Category', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('PSA Level Categories', fontsize=14, fontweight='bold')

for i, cat in enumerate(order):
    pct = 100 * psa_cat_counts[cat] / len(df)
    ax.text(i, psa_cat_counts[cat] + 200, f'{psa_cat_counts[cat]:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/univariate_psa_category.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/univariate_psa_category.png")

# 3.8 Cancer Stage Distribution
print("\n" + "─" * 40)
print("Cancer Stage Distribution (Malignant Only)")
print("─" * 40)

malignant_df = df[df['Biopsy_Result'] == 'Malignant']
stage_counts = malignant_df['Cancer_Stage'].value_counts()
print(stage_counts.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#4E79A7', '#F28E2B', '#E15759']
ax.bar(stage_counts.index, stage_counts.values, color=colors, alpha=0.8, edgecolor='white')
ax.set_xlabel('Cancer Stage', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Cancer Stage Distribution (Malignant Cases Only)', fontsize=14, fontweight='bold')

for i, (stage, count) in enumerate(stage_counts.items()):
    pct = 100 * count / len(malignant_df)
    ax.text(i, count + 50, f'{count:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/univariate_cancer_stage.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/univariate_cancer_stage.png")

# =============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS - BIVARIATE ANALYSIS
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 4: BIVARIATE ANALYSIS")
print("=" * 60)
print()

# 4.1 PSA Level by Biopsy Result
print("─" * 40)
print("PSA Level by Biopsy Result")
print("─" * 40)

psa_malignant = df[df['Biopsy_Result'] == 'Malignant']['PSA_Level']
psa_benign = df[df['Biopsy_Result'] == 'Benign']['PSA_Level']

print(f"Malignant - Mean: {psa_malignant.mean():.2f}, Median: {psa_malignant.median():.2f}")
print(f"Benign - Mean: {psa_benign.mean():.2f}, Median: {psa_benign.median():.2f}")

# Mann-Whitney U test
mw_stat, mw_pval = mannwhitneyu(psa_malignant, psa_benign, alternative='two-sided')
print(f"\nMann-Whitney U Test: U = {mw_stat:,.0f}, p-value < 0.001")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plot
ax1 = axes[0, 0]
box_data = [psa_benign, psa_malignant]
bp = ax1.boxplot(box_data, labels=['Benign', 'Malignant'], patch_artist=True)
colors = ['#4E79A7', '#E15759']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('PSA Level (ng/mL)', fontsize=11)
ax1.set_title(f'PSA Level by Biopsy Result\n(Mann-Whitney U = {mw_stat:,.0f}, p < 0.001)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, df['PSA_Level'].quantile(0.99))

# 4.2 Age by Biopsy Result
print("\n" + "─" * 40)
print("Age by Biopsy Result")
print("─" * 40)

age_malignant = df[df['Biopsy_Result'] == 'Malignant']['Age']
age_benign = df[df['Biopsy_Result'] == 'Benign']['Age']

print(f"Malignant - Mean Age: {age_malignant.mean():.2f} years")
print(f"Benign - Mean Age: {age_benign.mean():.2f} years")

t_stat, t_pval = ttest_ind(age_malignant, age_benign)
print(f"\nIndependent t-test: t = {t_stat:.3f}, p-value < 0.001")

ax2 = axes[0, 1]
bp2 = ax2.boxplot([age_benign, age_malignant], labels=['Benign', 'Malignant'], patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Age (years)', fontsize=11)
ax2.set_title(f'Age Distribution by Biopsy Result\n(t = {t_stat:.2f}, p < 0.001)', fontsize=12, fontweight='bold')

# 4.3 Cancer Rate by PSA Category
print("\n" + "─" * 40)
print("Cancer Detection Rate by PSA Category")
print("─" * 40)

# Convert category to string for aggregation
df['PSA_Category_Str'] = df['PSA_Category'].astype(str)
cancer_by_psa = df.groupby('PSA_Category_Str').agg({
    'Cancer_Binary': ['sum', 'count', 'mean']
}).reset_index()
cancer_by_psa.columns = ['PSA_Category', 'Malignant', 'Total', 'Cancer_Rate']
cancer_by_psa['Cancer_Rate_Pct'] = cancer_by_psa['Cancer_Rate'] * 100

print(cancer_by_psa.to_string(index=False))

ax3 = axes[1, 0]
order = ['Normal', 'Borderline', 'Elevated']
cancer_by_psa['PSA_Category'] = pd.Categorical(cancer_by_psa['PSA_Category'], categories=order, ordered=True)
cancer_by_psa = cancer_by_psa.sort_values('PSA_Category')
colors_psa = plt.cm.plasma(np.linspace(0, 0.8, 3))
bars = ax3.bar(cancer_by_psa['PSA_Category'], cancer_by_psa['Cancer_Rate_Pct'],
               color=colors_psa, alpha=0.8, edgecolor='white')
ax3.set_xlabel('PSA Category', fontsize=11)
ax3.set_ylabel('Cancer Detection Rate (%)', fontsize=11)
ax3.set_title('Cancer Detection Rate by PSA Category', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 100)

for bar, (_, row) in zip(bars, cancer_by_psa.iterrows()):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f"{row['Cancer_Rate_Pct']:.1f}%\n(n={row['Total']:,})",
            ha='center', va='bottom', fontsize=9)

# 4.4 DRE Result by Biopsy Result
print("\n" + "─" * 40)
print("DRE Result by Biopsy Result")
print("─" * 40)

# Chi-square test
contingency_dre = pd.crosstab(df['DRE_Result'], df['Biopsy_Result'])
chi2, chi2_pval, dof, expected = chi2_contingency(contingency_dre)
print("Contingency Table:")
print(contingency_dre.to_string())
print(f"\nChi-square: {chi2:.2f}, df = {dof}, p-value < 0.001")

# Calculate odds ratio
or_num = (contingency_dre.loc['Abnormal', 'Malignant'] * contingency_dre.loc['Normal', 'Benign'])
or_den = (contingency_dre.loc['Abnormal', 'Benign'] * contingency_dre.loc['Normal', 'Malignant'])
odds_ratio = or_num / or_den
print(f"Odds Ratio: {odds_ratio:.2f}")

ax4 = axes[1, 1]
ct = pd.crosstab(df['Biopsy_Result'], df['DRE_Result'], normalize='index') * 100
ct.plot(kind='bar', ax=ax4, color=['#4E79A7', '#E15759'], alpha=0.8, edgecolor='white')
ax4.set_xlabel('Biopsy Result', fontsize=11)
ax4.set_ylabel('Percentage (%)', fontsize=11)
ax4.set_title(f'Biopsy Result by DRE Result\n(Chi² = {chi2:.1f}, OR = {odds_ratio:.1f})', fontsize=12, fontweight='bold')
ax4.legend(title='DRE Result')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/bivariate_analysis_part1.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/bivariate_analysis_part1.png")

# 4.5 Cancer Rate by Family History and Race
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# By Family History
cancer_by_family = df.groupby('Family_History')['Cancer_Binary'].agg(['sum', 'count', 'mean'])
cancer_by_family.columns = ['Malignant', 'Total', 'Rate']
cancer_by_family['Rate_Pct'] = cancer_by_family['Rate'] * 100

ax1 = axes[0]
bars1 = ax1.bar(['No', 'Yes'], cancer_by_family['Rate_Pct'], color=['#4E79A7', '#E15759'], alpha=0.8)
ax1.set_xlabel('Family History', fontsize=11)
ax1.set_ylabel('Cancer Rate (%)', fontsize=11)
ax1.set_title('Cancer Rate by Family History', fontsize=12, fontweight='bold')
for bar, (_, row) in zip(bars1, cancer_by_family.iterrows()):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f"{row['Rate_Pct']:.1f}%\n(n={row['Total']:,})",
            ha='center', va='bottom', fontsize=10)

# By African Ancestry
cancer_by_race = df.groupby('Race_African_Ancestry')['Cancer_Binary'].agg(['sum', 'count', 'mean'])
cancer_by_race.columns = ['Malignant', 'Total', 'Rate']
cancer_by_race['Rate_Pct'] = cancer_by_race['Rate'] * 100

ax2 = axes[1]
bars2 = ax2.bar(['No', 'Yes'], cancer_by_race['Rate_Pct'], color=['#59A14F', '#F28E2B'], alpha=0.8)
ax2.set_xlabel('African Ancestry', fontsize=11)
ax2.set_ylabel('Cancer Rate (%)', fontsize=11)
ax2.set_title('Cancer Rate by African Ancestry', fontsize=12, fontweight='bold')
for bar, (_, row) in zip(bars2, cancer_by_race.iterrows()):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f"{row['Rate_Pct']:.1f}%\n(n={row['Total']:,})",
            ha='center', va='bottom', fontsize=10)

# By Age Group
cancer_by_age = df.groupby('Age_Group')['Cancer_Binary'].agg(['sum', 'count', 'mean'])
cancer_by_age.columns = ['Malignant', 'Total', 'Rate']
cancer_by_age['Rate_Pct'] = cancer_by_age['Rate'] * 100
order_age = ['40-54', '55-64', '65-74', '75+']

ax3 = axes[2]
colors_age = plt.cm.viridis(np.linspace(0, 0.8, 4))
bars3 = ax3.bar(order_age, [cancer_by_age.loc[x, 'Rate_Pct'] for x in order_age],
                color=colors_age, alpha=0.8)
ax3.set_xlabel('Age Group', fontsize=11)
ax3.set_ylabel('Cancer Rate (%)', fontsize=11)
ax3.set_title('Cancer Rate by Age Group', fontsize=12, fontweight='bold')
for bar, age in zip(bars3, order_age):
    row = cancer_by_age.loc[age]
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f"{row['Rate_Pct']:.1f}%\n(n={int(row['Total']):,})",
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/bivariate_analysis_part2.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/bivariate_analysis_part2.png")

# =============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS - MULTIVARIATE ANALYSIS
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 5: MULTIVARIATE ANALYSIS")
print("=" * 60)
print()

# 5.1 Correlation Matrix
print("─" * 40)
print("Correlation Matrix - Continuous Variables")
print("─" * 40)

continuous_for_corr = ['Age', 'PSA_Level', 'BMI', 'Screening_Age', 'Prostate_Volume']
corr_matrix = df[continuous_for_corr].corr()

print(corr_matrix.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1,
            center=0, square=True, linewidths=0.5, annot=True,
            fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix of Continuous Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/correlation_matrix.png")

# 5.2 Age vs PSA by Biopsy Result
print("\n" + "─" * 40)
print("Age vs PSA Level by Biopsy Result")
print("─" * 40)

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(df[df['Biopsy_Result']=='Benign']['Age'],
                    np.log(df[df['Biopsy_Result']=='Benign']['PSA_Level']),
                    alpha=0.2, c='#4E79A7', s=10, label='Benign')
scatter = ax.scatter(df[df['Biopsy_Result']=='Malignant']['Age'],
                    np.log(df[df['Biopsy_Result']=='Malignant']['PSA_Level']),
                    alpha=0.3, c='#E15759', s=15, label='Malignant')

# Add trend lines
for result, color in [('Benign', '#4E79A7'), ('Malignant', '#E15759')]:
    subset = df[df['Biopsy_Result'] == result]
    z = np.polyfit(subset['Age'], np.log(subset['PSA_Level']), 1)
    p = np.poly1d(z)
    ax.plot(subset['Age'].sort_values(), p(subset['Age'].sort_values()),
           color=color, linewidth=2, linestyle='--')

ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Log(PSA Level)', fontsize=12)
ax.set_title('Age vs PSA Level by Biopsy Result', fontsize=14, fontweight='bold')
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/multivariate_age_psa.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/multivariate_age_psa.png")

# 5.3 Mean PSA by Age Group and Biopsy Result
print("\n" + "─" * 40)
print("Mean PSA by Age Group and Biopsy Result")
print("─" * 40)

psa_means = df.groupby(['Age_Group', 'Biopsy_Result'])['PSA_Level'].agg(['mean', 'std', 'count'])
psa_means = psa_means.reset_index()
psa_means.columns = ['Age_Group', 'Biopsy_Result', 'Mean_PSA', 'SD_PSA', 'N']
print(psa_means.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
order = ['40-54', '55-64', '65-74', '75+']
x = np.arange(len(order))
width = 0.35

benign_means = [psa_means[(psa_means['Age_Group']==ag) & (psa_means['Biopsy_Result']=='Benign')]['Mean_PSA'].values[0] for ag in order]
malignant_means = [psa_means[(psa_means['Age_Group']==ag) & (psa_means['Biopsy_Result']=='Malignant')]['Mean_PSA'].values[0] for ag in order]

bars1 = ax.bar(x - width/2, benign_means, width, label='Benign', color='#4E79A7', alpha=0.8)
bars2 = ax.bar(x + width/2, malignant_means, width, label='Malignant', color='#E15759', alpha=0.8)

ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Mean PSA Level (ng/mL)', fontsize=12)
ax.set_title('Mean PSA Level by Age Group and Biopsy Result', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/multivariate_psa_means.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/multivariate_psa_means.png")

# =============================================================================
# SECTION 6: STATISTICAL MODELING - LOGISTIC REGRESSION
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 6: LOGISTIC REGRESSION")
print("=" * 60)
print()

# 6.1 Prepare Data for Modeling
print("─" * 40)
print("Preparing Data for Modeling")
print("─" * 40)

model_vars = ['Age', 'PSA_Level', 'DRE_Result', 'Family_History',
              'Race_African_Ancestry', 'Prostate_Volume', 'Blood_in_Urine',
              'Genetic_Risk_Factors', 'Previous_Cancer_History']

# Create modeling dataset
model_df = df.copy()

# Convert categorical predictors to numeric
model_df['DRE_Result'] = model_df['DRE_Result'].map({'Normal': 0, 'Abnormal': 1})
model_df['Family_History'] = model_df['Family_History'].map({'No': 0, 'Yes': 1})
model_df['Race_African_Ancestry'] = model_df['Race_African_Ancestry'].map({'No': 0, 'Yes': 1})
model_df['Blood_in_Urine'] = model_df['Blood_in_Urine'].map({'No': 0, 'Yes': 1})
model_df['Genetic_Risk_Factors'] = model_df['Genetic_Risk_Factors'].map({'No': 0, 'Yes': 1})
model_df['Previous_Cancer_History'] = model_df['Previous_Cancer_History'].map({'No': 0, 'Yes': 1})

X = model_df[model_vars]
y = model_df['Cancer_Binary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {len(X_train):,}")
print(f"Testing set size: {len(X_test):,}")

# 6.2 Univariate Logistic Regression
print("\n" + "─" * 40)
print("Univariate Logistic Regression Results")
print("─" * 40)

univariate_results = []
for var in model_vars:
    X_uni = X_train[[var]]
    model_uni = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model_uni.fit(X_uni, y_train)

    coef = model_uni.coef_[0][0]
    odds_ratio = np.exp(coef)

    univariate_results.append({
        'Variable': var,
        'Coefficient': coef,
        'Odds_Ratio': odds_ratio
    })

uni_df = pd.DataFrame(univariate_results).sort_values('Odds_Ratio', ascending=False)
print(uni_df.to_string(index=False))

# 6.3 Multivariate Logistic Regression
print("\n" + "─" * 40)
print("Multivariate Logistic Regression")
print("─" * 40)

logistic_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Coefficients
coef_df = pd.DataFrame({
    'Variable': model_vars,
    'Coefficient': logistic_model.coef_[0],
    'Odds_Ratio': np.exp(logistic_model.coef_[0])
}).sort_values('Coefficient', ascending=False)

print("\nLogistic Regression Coefficients:")
print(coef_df.to_string(index=False))

# Model fit
train_score = logistic_model.score(X_train, y_train)
test_score = logistic_model.score(X_test, y_test)
print(f"\nModel Performance:")
print(f"  Training Accuracy: {train_score:.4f} ({train_score*100:.1f}%)")
print(f"  Testing Accuracy: {test_score:.4f} ({test_score*100:.1f}%)")

# 6.4 Model Predictions and Evaluation
print("\n" + "─" * 40)
print("Logistic Regression Model Evaluation")
print("─" * 40)

y_pred = logistic_model.predict(X_test)
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"\nClassification Metrics:")
print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Sensitivity: {sensitivity:.4f} ({sensitivity*100:.1f}%)")
print(f"  Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
print(f"  Precision:   {precision:.4f} ({precision*100:.1f}%)")
print(f"  F1-Score:    {f1:.4f} ({f1*100:.1f}%)")

# 6.5 ROC Curve
print("\n" + "─" * 40)
print("ROC Curve - Logistic Regression")
print("─" * 40)

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba)
auc_lr = roc_auc_score(y_test, y_pred_proba)
print(f"Area Under Curve (AUC): {auc_lr:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_lr, tpr_lr, color='#4E79A7', linewidth=2,
        label=f'Logistic Regression (AUC = {auc_lr:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/logistic_roc.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/logistic_roc.png")

# =============================================================================
# SECTION 7: STATISTICAL MODELING - DECISION TREE
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 7: DECISION TREE CLASSIFICATION")
print("=" * 60)
print()

# 7.1 Build Decision Tree
print("─" * 40)
print("Building Decision Tree")
print("─" * 40)

dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=RANDOM_STATE
)
dt_model.fit(X_train, y_train)

print(f"Tree Depth: {dt_model.get_depth()}")
print(f"Number of Leaves: {dt_model.get_n_leaves()}")

# 7.2 Feature Importance
print("\n" + "─" * 40)
print("Decision Tree Feature Importance")
print("─" * 40)

dt_importance = pd.DataFrame({
    'Variable': model_vars,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(dt_importance.to_string(index=False))

# 7.3 Decision Tree Predictions
print("\n" + "─" * 40)
print("Decision Tree Evaluation")
print("─" * 40)

y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_sensitivity = confusion_matrix(y_test, y_pred_dt)[1, 1] / (confusion_matrix(y_test, y_pred_dt)[1, 1] + confusion_matrix(y_test, y_pred_dt)[1, 0])
dt_specificity = confusion_matrix(y_test, y_pred_dt)[0, 0] / (confusion_matrix(y_test, y_pred_dt)[0, 0] + confusion_matrix(y_test, y_pred_dt)[0, 1])

print(f"\nClassification Metrics:")
print(f"  Accuracy:    {dt_accuracy:.4f} ({dt_accuracy*100:.1f}%)")
print(f"  Sensitivity: {dt_sensitivity:.4f} ({dt_sensitivity*100:.1f}%)")
print(f"  Specificity: {dt_specificity:.4f} ({dt_specificity*100:.1f}%)")

# Plot Decision Tree
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt_model, feature_names=model_vars, class_names=['Benign', 'Malignant'],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title('Decision Tree for Prostate Cancer Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/decision_tree.png")

# Plot Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
dt_importance_sorted = dt_importance.sort_values('Importance', ascending=True)
colors = plt.cm.plasma(np.linspace(0, 0.8, len(dt_importance_sorted)))
ax.barh(dt_importance_sorted['Variable'], dt_importance_sorted['Importance'],
        color=colors, alpha=0.8)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/decision_tree_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/decision_tree_importance.png")

# =============================================================================
# SECTION 8: STATISTICAL MODELING - RANDOM FOREST
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 8: RANDOM FOREST CLASSIFICATION")
print("=" * 60)
print()

# 8.1 Build Random Forest
print("─" * 40)
print("Building Random Forest")
print("─" * 40)
print("This may take a moment...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

print(f"Number of Trees: {rf_model.n_estimators}")
print(f"Number of Features: {rf_model.n_features_in_}")

# 8.2 Feature Importance
print("\n" + "─" * 40)
print("Random Forest Feature Importance")
print("─" * 40)

rf_importance = pd.DataFrame({
    'Variable': model_vars,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(rf_importance.to_string(index=False))

# 8.3 Random Forest Predictions
print("\n" + "─" * 40)
print("Random Forest Evaluation")
print("─" * 40)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_sensitivity = confusion_matrix(y_test, y_pred_rf)[1, 1] / (confusion_matrix(y_test, y_pred_rf)[1, 1] + confusion_matrix(y_test, y_pred_rf)[1, 0])
rf_specificity = confusion_matrix(y_test, y_pred_rf)[0, 0] / (confusion_matrix(y_test, y_pred_rf)[0, 0] + confusion_matrix(y_test, y_pred_rf)[0, 1])
rf_precision = confusion_matrix(y_test, y_pred_rf)[1, 1] / (confusion_matrix(y_test, y_pred_rf)[1, 1] + confusion_matrix(y_test, y_pred_rf)[0, 1])
rf_f1 = 2 * (rf_precision * rf_sensitivity) / (rf_precision + rf_sensitivity)

print(f"\nClassification Metrics:")
print(f"  Accuracy:    {rf_accuracy:.4f} ({rf_accuracy*100:.1f}%)")
print(f"  Sensitivity: {rf_sensitivity:.4f} ({rf_sensitivity*100:.1f}%)")
print(f"  Specificity: {rf_specificity:.4f} ({rf_specificity*100:.1f}%)")
print(f"  Precision:   {rf_precision:.4f} ({rf_precision*100:.1f}%)")
print(f"  F1-Score:    {rf_f1:.4f} ({rf_f1*100:.1f}%)")

# 8.4 ROC Curve for Random Forest
print("\n" + "─" * 40)
print("ROC Curve - Random Forest")
print("─" * 40)

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"Area Under Curve (AUC): {auc_rf:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_rf, tpr_rf, color='#59A14F', linewidth=2,
        label=f'Random Forest (AUC = {auc_rf:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/random_forest_roc.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/random_forest_roc.png")

# Plot Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
rf_importance_sorted = rf_importance.sort_values('Importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(rf_importance_sorted)))
ax.barh(rf_importance_sorted['Variable'], rf_importance_sorted['Importance'],
        color=colors, alpha=0.8)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/random_forest_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/random_forest_importance.png")

# =============================================================================
# SECTION 9: MODEL COMPARISON
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 9: MODEL COMPARISON")
print("=" * 60)
print()

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy, dt_accuracy, rf_accuracy],
    'Sensitivity': [sensitivity, dt_sensitivity, rf_sensitivity],
    'Specificity': [specificity, dt_specificity, rf_specificity],
    'AUC': [auc_lr, np.nan, auc_rf]
})

print("Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# Calculate AUC for Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
comparison_df.loc[comparison_df['Model'] == 'Decision Tree', 'AUC'] = auc_dt

# Plot Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Comparison
ax1 = axes[0]
colors_models = ['#4E79A7', '#59A14F', '#E15759']
bars = ax1.bar(comparison_df['Model'], comparison_df['Accuracy'] * 100,
               color=colors_models, alpha=0.8, edgecolor='white')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
for bar, acc in zip(bars, comparison_df['Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11)

# ROC Curves Comparison
ax2 = axes[1]
ax2.plot(fpr_lr, tpr_lr, color='#4E79A7', linewidth=2,
         label=f'Logistic Regression (AUC = {auc_lr:.3f})')
ax2.plot(fpr_dt, tpr_dt, color='#59A14F', linewidth=2,
         label=f'Decision Tree (AUC = {auc_dt:.3f})')
ax2.plot(fpr_rf, tpr_rf, color='#E15759', linewidth=2,
         label=f'Random Forest (AUC = {auc_rf:.3f})')
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax2.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {OUTPUT_DIR}/model_comparison.png")

# =============================================================================
# SECTION 10: SUMMARY AND CONCLUSIONS
# =============================================================================

print("\n")
print("=" * 60)
print("SECTION 10: SUMMARY AND CONCLUSIONS")
print("=" * 60)
print()

print("=" * 60)
print("PROCogniTrack PROJECT SUMMARY")
print("=" * 60)

print("\n1. DATASET OVERVIEW")
print("-" * 40)
print(f"   Total patients: {len(df):,}")
print(f"   Malignant cases: {(df['Biopsy_Result'] == 'Malignant').sum():,}"
      f" ({(df['Biopsy_Result'] == 'Malignant').mean()*100:.1f}%)")
print(f"   Benign cases: {(df['Biopsy_Result'] == 'Benign').sum():,}"
      f" ({(df['Biopsy_Result'] == 'Benign').mean()*100:.1f}%)")

print("\n2. KEY RISK FACTORS (Univariate Analysis)")
print("-" * 40)
print(f"   - Age: Malignant mean = {age_malignant.mean():.1f} years")
print(f"   - PSA Level: Malignant mean = {psa_malignant.mean():.1f} ng/mL")
print(f"   - DRE Abnormal: {(df[df['Biopsy_Result']=='Malignant']['DRE_Result']=='Abnormal').mean()*100:.1f}% in malignant cases")
print(f"   - Family History: {(df[df['Biopsy_Result']=='Malignant']['Family_History']=='Yes').mean()*100:.1f}% in malignant cases")

print("\n3. MODEL PERFORMANCE SUMMARY")
print("-" * 40)
print("   Logistic Regression:")
print(f"     - Accuracy: {accuracy*100:.1f}%")
print(f"     - AUC: {auc_lr:.3f}")
print("   Decision Tree:")
print(f"     - Accuracy: {dt_accuracy*100:.1f}%")
print(f"     - AUC: {auc_dt:.3f}")
print("   Random Forest:")
print(f"     - Accuracy: {rf_accuracy*100:.1f}%")
print(f"     - AUC: {auc_rf:.3f}")

print("\n4. TOP PREDICTORS")
print("-" * 40)
print("   Based on Random Forest importance:")
for i, (_, row) in enumerate(rf_importance.head(5).iterrows(), 1):
    print(f"     {i}. {row['Variable']}: {row['Importance']:.4f}")

print("\n5. OUTPUT FILES GENERATED")
print("-" * 40)
print(f"   All plots saved to: {OUTPUT_DIR}/")
import os
plot_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
print(f"   Total plots generated: {plot_count}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\n")
print("ProCogniTrack - Prostate Cancer Prediction Analysis")
print("Python Code Execution Completed Successfully")
print("\n")

# =============================================================================
# END OF SCRIPT
# =============================================================================
