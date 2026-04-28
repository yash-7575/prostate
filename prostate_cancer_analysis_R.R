# ============================================================================
#  PROJECT: ProCogniTrack - Prostate Cancer Prediction and Risk Factor Analysis
#  FILE   : Comprehensive Data Analysis in R (RStudio)
#  PURPOSE: Exploratory Data Analysis, Statistical Testing, and Predictive Modeling
#  AUTHOR : MiniMax Agent
#  DATE   : 2025
# ============================================================================

# =============================================================================
# SECTION 0: INSTALL AND LOAD REQUIRED PACKAGES
# =============================================================================

# Install packages if not already installed
required_packages <- c(
  "ggplot2",      # Data visualization
  "dplyr",        # Data manipulation
  "tidyr",        # Data tidying
  "readr",        # Fast reading of data files
  "ggcorrplot",   # Correlation plot visualization
  "GGally",       # Pair plot matrix
  "viridis",      # Color-blind safe color palettes
  "scales",       # Scale functions for axes
  "patchwork",    # Combine multiple plots
  "caret",        # Classification and regression training
  "rpart",        # Recursive partitioning (decision trees)
  "rpart.plot",   # Decision tree visualization
  "randomForest", # Random forest classification
  "pROC",         # ROC curve analysis
  "ROCR",         # Prediction visualization
  "summarytools", # Descriptive statistics
  "magrittr"      # Pipe operators
)

# Install packages that are not already installed
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Load all required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(ggcorrplot)
library(GGally)
library(viridis)
library(scales)
library(patchwork)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(ROCR)
library(summarytools)
library(magrittr)

# Print session information
cat("========================================\n")
cat("ProCogniTrack - Prostate Cancer Analysis\n")
cat("R Session Information\n")
cat("========================================\n")
sessionInfo()
cat("\n")

# =============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 1: DATA LOADING\n")
cat("========================================\n\n")

# Set the working directory (modify as needed)
# setwd("path/to/your/directory")

# Define data path
data_path <- "prostate/prostate_cancer_prediction.csv"

# Load the dataset
cat("Loading prostate cancer dataset...\n")
df <- read_csv(data_path, show_col_types = FALSE)

# Display basic information about the dataset
cat("\n--- Dataset Dimensions ---\n")
cat("Number of rows:", nrow(df), "\n")
cat("Number of columns:", ncol(df), "\n")

cat("\n--- Column Names ---\n")
print(colnames(df))

cat("\n--- First 10 Rows ---\n")
print(head(df, 10))

cat("\n--- Data Types ---\n")
print(sapply(df, class))

cat("\n--- Summary Statistics ---\n")
print(summary(df))

# =============================================================================
# SECTION 2: DATA PREPROCESSING
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 2: DATA PREPROCESSING\n")
cat("========================================\n\n")

# 2.1 Check for missing values
cat("--- Missing Values Check ---\n")
missing_values <- colSums(is.na(df))
print(missing_values[missing_values > 0])
if (sum(missing_values) == 0) {
  cat("No missing values found in the dataset.\n")
} else {
  cat("Total missing values:", sum(missing_values), "\n")
}

# 2.2 Check for duplicate rows
cat("\n--- Duplicate Rows Check ---\n")
duplicates <- sum(duplicated(df))
cat("Number of duplicate rows:", duplicates, "\n")

# 2.3 Convert character columns to factors
cat("\n--- Converting Categorical Variables to Factors ---\n")
categorical_cols <- c(
  "Family_History", "Race_African_Ancestry", "DRE_Result", "Biopsy_Result",
  "Difficulty_Urinating", "Weak_Urine_Flow", "Blood_in_Urine", "Pelvic_Pain",
  "Back_Pain", "Erectile_Dysfunction", "Cancer_Stage", "Treatment_Recommended",
  "Survival_5_Years", "Exercise_Regularly", "Healthy_Diet", "Smoking_History",
  "Alcohol_Consumption", "Hypertension", "Diabetes", "Cholesterol_Level",
  "Follow_Up_Required", "Genetic_Risk_Factors", "Previous_Cancer_History", "Early_Detection"
)

# Convert character columns to factors
for (col in categorical_cols) {
  if (col %in% colnames(df)) {
    df[[col]] <- as.factor(df[[col]])
  }
}

# 2.4 Create derived variables
cat("\n--- Creating Derived Variables ---\n")

# Create binary outcome variable for modeling (1 = Malignant, 0 = Benign)
df$Cancer_Binary <- ifelse(df$Biopsy_Result == "Malignant", 1, 0)
df$Cancer_Binary <- as.factor(df$Cancer_Binary)

# Create PSA category
df$PSA_Category <- cut(
  df$PSA_Level,
  breaks = c(-Inf, 4, 10, 20, Inf),
  labels = c("Normal", "Borderline", "Elevated", "Very High"),
  include.lowest = TRUE
)

# Create age groups
df$Age_Group <- cut(
  df$Age,
  breaks = c(0, 54, 64, 74, Inf),
  labels = c("40-54", "55-64", "65-74", "75+"),
  include.lowest = TRUE
)

# Create BMI category
df$BMI_Category <- cut(
  df$BMI,
  breaks = c(0, 18.5, 25, 30, Inf),
  labels = c("Underweight", "Normal", "Overweight", "Obese"),
  include.lowest = TRUE
)

cat("Derived variables created: Cancer_Binary, PSA_Category, Age_Group, BMI_Category\n")

# 2.5 Summary after preprocessing
cat("\n--- Dataset After Preprocessing ---\n")
cat("Number of rows:", nrow(df), "\n")
cat("Number of columns:", ncol(df), "\n")

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS - UNIVARIATE ANALYSIS
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 3: UNIVARIATE ANALYSIS\n")
cat("========================================\n\n")

# Create output directory for plots
output_dir <- "plots_r"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# 3.1 Continuous Variables Statistics
cat("--- Descriptive Statistics for Continuous Variables ---\n")

continuous_vars <- c("Age", "PSA_Level", "BMI", "Screening_Age", "Prostate_Volume")

for (var in continuous_vars) {
  cat("\nVariable:", var, "\n")
  cat("  Mean:", round(mean(df[[var]], na.rm = TRUE), 2), "\n")
  cat("  Median:", round(median(df[[var]], na.rm = TRUE), 2), "\n")
  cat("  Std Dev:", round(sd(df[[var]], na.rm = TRUE), 2), "\n")
  cat("  Min:", round(min(df[[var]], na.rm = TRUE), 2), "\n")
  cat("  Max:", round(max(df[[var]], na.rm = TRUE), 2), "\n")
  cat("  IQR:", round(IQR(df[[var]], na.rm = TRUE), 2), "\n")
}

# 3.2 Distribution of Age
cat("\n--- Age Distribution ---\n")
p_age <- ggplot(df, aes(x = Age)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "#4E79A7",
                 color = "white", alpha = 0.8) +
  geom_density(color = "#F28E2B", linewidth = 1.5) +
  geom_vline(xintercept = mean(df$Age), color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = median(df$Age), color = "green", linetype = "dotted", linewidth = 1) +
  labs(
    title = "Distribution of Patient Age",
    subtitle = paste("Mean =", round(mean(df$Age), 1),
                     "| Median =", round(median(df$Age), 1)),
    x = "Age (years)",
    y = "Density"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    axis.title = element_text(size = 11),
    panel.grid.minor = element_blank()
  )
print(p_age)
ggsave(paste0(output_dir, "/univariate_age_distribution.png"),
       p_age, width = 10, height = 6, dpi = 300)

# 3.3 Distribution of PSA Level
cat("\n--- PSA Level Distribution ---\n")
p_psa <- ggplot(df, aes(x = PSA_Level)) +
  geom_histogram(aes(y = after_stat(density)), bins = 40, fill = "#59A14F",
                 color = "white", alpha = 0.8) +
  geom_density(color = "#E15759", linewidth = 1.5) +
  geom_vline(xintercept = 4, color = "red", linetype = "dashed", linewidth = 1,
             label = "Clinical cutoff (4 ng/mL)") +
  geom_vline(xintercept = 10, color = "orange", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = 20, color = "darkorange", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Distribution of PSA Levels",
    subtitle = paste("Mean =", round(mean(df$PSA_Level), 1),
                     "| Median =", round(median(df$PSA_Level), 1)),
    x = "PSA Level (ng/mL)",
    y = "Density",
    caption = "Red: 4 ng/mL cutoff, Orange: 10 ng/mL, Dark Orange: 20 ng/mL"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    axis.title = element_text(size = 11),
    panel.grid.minor = element_blank()
  )
print(p_psa)
ggsave(paste0(output_dir, "/univariate_psa_distribution.png"),
       p_psa, width = 10, height = 6, dpi = 300)

# 3.4 Distribution of PSA Level (Log-transformed)
p_psa_log <- ggplot(df, aes(x = log(PSA_Level))) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "#59A14F",
                 color = "white", alpha = 0.8) +
  geom_density(color = "#E15759", linewidth = 1.5) +
  labs(
    title = "Distribution of Log-Transformed PSA Levels",
    x = "Log(PSA Level)",
    y = "Density"
  ) +
  theme_minimal(base_size = 12)
print(p_psa_log)
ggsave(paste0(output_dir, "/univariate_psa_log_distribution.png"),
       p_psa_log, width = 10, height = 6, dpi = 300)

# 3.5 Distribution of BMI
cat("\n--- BMI Distribution ---\n")
p_bmi <- ggplot(df, aes(x = BMI)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "#B07AA1",
                 color = "white", alpha = 0.8) +
  geom_density(color = "#E15759", linewidth = 1.5) +
  geom_vline(xintercept = 25, color = "red", linetype = "dashed", linewidth = 1,
             label = "Overweight cutoff (25)") +
  geom_vline(xintercept = 30, color = "orange", linetype = "dashed", linewidth = 1,
             label = "Obese cutoff (30)") +
  labs(
    title = "Distribution of BMI",
    subtitle = paste("Mean =", round(mean(df$BMI), 1),
                     "| Median =", round(median(df$BMI), 1)),
    x = "BMI (kg/m²)",
    y = "Density"
  ) +
  theme_minimal(base_size = 12)
print(p_bmi)
ggsave(paste0(output_dir, "/univariate_bmi_distribution.png"),
       p_bmi, width = 10, height = 6, dpi = 300)

# 3.6 Distribution of Prostate Volume
cat("\n--- Prostate Volume Distribution ---\n")
p_volume <- ggplot(df, aes(x = Prostate_Volume)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "#76B7B2",
                 color = "white", alpha = 0.8) +
  geom_density(color = "#E15759", linewidth = 1.5) +
  geom_vline(xintercept = 50, color = "red", linetype = "dashed", linewidth = 1,
             label = "Enlarged prostate (50 cc)") +
  labs(
    title = "Distribution of Prostate Volume",
    subtitle = paste("Mean =", round(mean(df$Prostate_Volume), 1),
                     "| Median =", round(median(df$Prostate_Volume), 1)),
    x = "Prostate Volume (cc)",
    y = "Density"
  ) +
  theme_minimal(base_size = 12)
print(p_volume)
ggsave(paste0(output_dir, "/univariate_prostate_volume_distribution.png"),
       p_volume, width = 10, height = 6, dpi = 300)

# 3.7 Categorical Variables - Biopsy Result (Primary Outcome)
cat("\n--- Biopsy Result Distribution ---\n")
biopsy_counts <- table(df$Biopsy_Result)
print(biopsy_counts)
biopsy_percent <- prop.table(biopsy_counts) * 100
print(biopsy_percent)

p_biopsy <- ggplot(df, aes(x = Biopsy_Result, fill = Biopsy_Result)) +
  geom_bar(width = 0.6, alpha = 0.8) +
  geom_text(aes(label = paste0(round(after_stat(count) / nrow(df) * 100, 1), "%")),
            stat = "count", vjust = 1.5, size = 5, color = "white") +
  scale_fill_viridis_d(option = "D") +
  labs(
    title = "Biopsy Result Distribution",
    subtitle = paste("Malignant:", biopsy_percent["Malignant"] %>% round(1) %>% paste0("%"),
                     "| Benign:", biopsy_percent["Benign"] %>% round(1) %>% paste0("%")),
    x = "Biopsy Result",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_biopsy)
ggsave(paste0(output_dir, "/univariate_biopsy_distribution.png"),
       p_biopsy, width = 8, height = 6, dpi = 300)

# 3.8 PSA Category Distribution
cat("\n--- PSA Category Distribution ---\n")
p_psa_cat <- ggplot(df, aes(x = PSA_Category, fill = PSA_Category)) +
  geom_bar(width = 0.7, alpha = 0.8) +
  geom_text(aes(label = paste0(round(after_stat(count) / nrow(df) * 100, 1), "%")),
            stat = "count", vjust = 1.5, size = 4.5, color = "white") +
  scale_fill_viridis_d(option = "plasma") +
  labs(
    title = "PSA Level Categories",
    subtitle = "Normal (<4), Borderline (4-10), Elevated (10-20), Very High (>20)",
    x = "PSA Category",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none",
    axis.text.x = element_text(angle = 0)
  )
print(p_psa_cat)
ggsave(paste0(output_dir, "/univariate_psa_category_distribution.png"),
       p_psa_cat, width = 10, height = 6, dpi = 300)

# 3.9 Cancer Stage Distribution
cat("\n--- Cancer Stage Distribution ---\n")
p_stage <- ggplot(df %>% filter(Biopsy_Result == "Malignant"),
                  aes(x = Cancer_Stage, fill = Cancer_Stage)) +
  geom_bar(width = 0.6, alpha = 0.8) +
  geom_text(aes(label = paste0(round(after_stat(count) /
                                       sum(df$Biopsy_Result == "Malignant") * 100, 1), "%")),
            stat = "count", vjust = 1.5, size = 5, color = "white") +
  scale_fill_viridis_d(option = "viridis") +
  labs(
    title = "Cancer Stage Distribution (Malignant Cases Only)",
    x = "Cancer Stage",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_stage)
ggsave(paste0(output_dir, "/univariate_cancer_stage_distribution.png"),
       p_stage, width = 8, height = 6, dpi = 300)

# 3.10 Family History Distribution
cat("\n--- Family History Distribution ---\n")
p_family <- ggplot(df, aes(x = Family_History, fill = Family_History)) +
  geom_bar(width = 0.5, alpha = 0.8) +
  geom_text(aes(label = paste0(round(after_stat(count) / nrow(df) * 100, 1), "%")),
            stat = "count", vjust = 1.5, size = 5, color = "white") +
  scale_fill_viridis_d(option = "cividis") +
  labs(
    title = "Family History of Prostate Cancer",
    x = "Family History",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_family)
ggsave(paste0(output_dir, "/univariate_family_history_distribution.png"),
       p_family, width = 6, height = 5, dpi = 300)

# =============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS - BIVARIATE ANALYSIS
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 4: BIVARIATE ANALYSIS\n")
cat("========================================\n\n")

# 4.1 PSA Level by Biopsy Result
cat("--- PSA Level by Biopsy Result ---\n")
psa_by_biopsy <- df %>%
  group_by(Biopsy_Result) %>%
  summarize(
    Mean = mean(PSA_Level),
    Median = median(PSA_Level),
    SD = sd(PSA_Level),
    Min = min(PSA_Level),
    Max = max(PSA_Level)
  )
print(psa_by_biopsy)

# Statistical test: Mann-Whitney U test (non-parametric)
psa_malignant <- df$PSA_Level[df$Biopsy_Result == "Malignant"]
psa_benign <- df$PSA_Level[df$Biopsy_Result == "Benign"]
mw_psa <- wilcox.test(psa_malignant, psa_benign)
cat("\nMann-Whitney U Test:\n")
cat("  U-statistic:", mw_psa$statistic, "\n")
cat("  p-value:", format.pval(mw_psa$p.value), "\n")

p_psa_biopsy <- ggplot(df, aes(x = Biopsy_Result, y = PSA_Level, fill = Biopsy_Result)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(alpha = 0.1, width = 0.2, size = 0.5) +
  scale_fill_viridis_d(option = "D") +
  coord_cartesian(ylim = c(0, quantile(df$PSA_Level, 0.99))) +
  labs(
    title = "PSA Level by Biopsy Result",
    subtitle = paste("Mann-Whitney U =", round(mw_psa$statistic),
                     "| p < 0.001"),
    x = "Biopsy Result",
    y = "PSA Level (ng/mL)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_psa_biopsy)
ggsave(paste0(output_dir, "/bivariate_psa_by_biopsy.png"),
       p_psa_biopsy, width = 10, height = 6, dpi = 300)

# 4.2 Age by Biopsy Result
cat("\n--- Age by Biopsy Result ---\n")
age_by_biopsy <- df %>%
  group_by(Biopsy_Result) %>%
  summarize(
    Mean = mean(Age),
    Median = median(Age),
    SD = sd(Age)
  )
print(age_by_biopsy)

t_age <- t.test(Age ~ Biopsy_Result, data = df)
cat("\nIndependent t-test:\n")
cat("  t-statistic:", round(t_age$statistic, 3), "\n")
cat("  p-value:", format.pval(t_age$p.value), "\n")

p_age_biopsy <- ggplot(df, aes(x = Biopsy_Result, y = Age, fill = Biopsy_Result)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.15, fill = "white", outlier.shape = NA) +
  scale_fill_viridis_d(option = "D") +
  labs(
    title = "Age Distribution by Biopsy Result",
    subtitle = paste("t =", round(t_age$statistic, 2), "| p < 0.001"),
    x = "Biopsy Result",
    y = "Age (years)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_age_biopsy)
ggsave(paste0(output_dir, "/bivariate_age_by_biopsy.png"),
       p_age_biopsy, width = 10, height = 6, dpi = 300)

# 4.3 DRE Result by Biopsy Result
cat("\n--- DRE Result by Biopsy Result ---\n")
dred_biopsy <- table(df$DRE_Result, df$Biopsy_Result)
print(dred_biopsy)

chi_dre <- chisq.test(df$DRE_Result, df$Biopsy_Result)
cat("\nChi-Square Test:\n")
cat("  Chi-square:", round(chi_dre$statistic, 2), "\n")
cat("  df:", chi_dre$parameter, "\n")
cat("  p-value:", format.pval(chi_dre$p.value), "\n")

# Calculate odds ratio
or_dre <- (dred_biopsy[2, 2] / dred_biopsy[2, 1]) / (dred_biopsy[1, 2] / dred_biopsy[1, 1])
cat("  Odds Ratio:", round(or_dre, 2), "\n")

p_dre_biopsy <- ggplot(df, aes(x = DRE_Result, fill = Biopsy_Result)) +
  geom_bar(position = "fill", width = 0.6, alpha = 0.8) +
  scale_fill_viridis_d(option = "D") +
  scale_y_continuous(labels = percent) +
  labs(
    title = "Biopsy Result by DRE Result",
    subtitle = paste("Chi-square =", round(chi_dre$statistic, 1),
                     "| OR =", round(or_dre, 1)),
    x = "DRE Result",
    y = "Proportion"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "bottom"
  )
print(p_dre_biopsy)
ggsave(paste0(output_dir, "/bivariate_dre_by_biopsy.png"),
       p_dre_biopsy, width = 10, height = 6, dpi = 300)

# 4.4 Cancer Rate by PSA Category
cat("\n--- Cancer Rate by PSA Category ---\n")
cancer_by_psa <- df %>%
  group_by(PSA_Category) %>%
  summarize(
    Total = n(),
    Malignant = sum(Biopsy_Result == "Malignant"),
    Cancer_Rate = round(100 * mean(Biopsy_Result == "Malignant"), 1)
  )
print(cancer_by_psa)

p_cancer_psa <- ggplot(cancer_by_psa, aes(x = PSA_Category, y = Cancer_Rate,
                                           fill = Cancer_Rate)) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_text(aes(label = paste0(Cancer_Rate, "%\n(n=", Total, ")")),
            vjust = -0.5, size = 4) +
  scale_fill_viridis_c(option = "plasma") +
  scale_y_continuous(limits = c(0, 100)) +
  labs(
    title = "Cancer Detection Rate by PSA Category",
    x = "PSA Category",
    y = "Cancer Detection Rate (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_cancer_psa)
ggsave(paste0(output_dir, "/bivariate_cancer_rate_by_psa.png"),
       p_cancer_psa, width = 10, height = 6, dpi = 300)

# 4.5 Cancer Rate by Family History
cat("\n--- Cancer Rate by Family History ---\n")
cancer_by_family <- df %>%
  group_by(Family_History) %>%
  summarize(
    Total = n(),
    Malignant = sum(Biopsy_Result == "Malignant"),
    Cancer_Rate = round(100 * mean(Biopsy_Result == "Malignant"), 1)
  )
print(cancer_by_family)

chi_family <- chisq.test(df$Family_History, df$Biopsy_Result)
cat("\nChi-Square Test:\n")
cat("  Chi-square:", round(chi_family$statistic, 2), "\n")
cat("  p-value:", format.pval(chi_family$p.value), "\n")

p_cancer_family <- ggplot(cancer_by_family, aes(x = Family_History, y = Cancer_Rate,
                                                fill = Family_History)) +
  geom_col(width = 0.5, alpha = 0.8) +
  geom_text(aes(label = paste0(Cancer_Rate, "%")), vjust = -0.5, size = 6) +
  scale_fill_viridis_d(option = "cividis") +
  scale_y_continuous(limits = c(0, 50)) +
  labs(
    title = "Cancer Detection Rate by Family History",
    x = "Family History",
    y = "Cancer Detection Rate (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_cancer_family)
ggsave(paste0(output_dir, "/bivariate_cancer_rate_by_family.png"),
       p_cancer_family, width = 8, height = 6, dpi = 300)

# 4.6 Cancer Rate by Race
cat("\n--- Cancer Rate by African Ancestry ---\n")
cancer_by_race <- df %>%
  group_by(Race_African_Ancestry) %>%
  summarize(
    Total = n(),
    Malignant = sum(Biopsy_Result == "Malignant"),
    Cancer_Rate = round(100 * mean(Biopsy_Result == "Malignant"), 1)
  )
print(cancer_by_race)

chi_race <- chisq.test(df$Race_African_Ancestry, df$Biopsy_Result)
cat("\nChi-Square Test:\n")
cat("  Chi-square:", round(chi_race$statistic, 2), "\n")
cat("  p-value:", format.pval(chi_race$p.value), "\n")

p_cancer_race <- ggplot(cancer_by_race, aes(x = Race_African_Ancestry, y = Cancer_Rate,
                                            fill = Race_African_Ancestry)) +
  geom_col(width = 0.5, alpha = 0.8) +
  geom_text(aes(label = paste0(Cancer_Rate, "%")), vjust = -0.5, size = 6) +
  scale_fill_viridis_d(option = "magma") +
  scale_y_continuous(limits = c(0, 50)) +
  labs(
    title = "Cancer Detection Rate by African Ancestry",
    x = "African Ancestry",
    y = "Cancer Detection Rate (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_cancer_race)
ggsave(paste0(output_dir, "/bivariate_cancer_rate_by_race.png"),
       p_cancer_race, width = 8, height = 6, dpi = 300)

# 4.7 Symptom Prevalence by Biopsy Result
cat("\n--- Symptom Prevalence by Biopsy Result ---\n")
symptoms <- c("Difficulty_Urinating", "Weak_Urine_Flow", "Blood_in_Urine",
              "Pelvic_Pain", "Back_Pain", "Erectile_Dysfunction")

symptom_stats <- data.frame()
for (symptom in symptoms) {
  if (symptom %in% colnames(df)) {
    ct <- table(df[[symptom]], df$Biopsy_Result)
    chi <- chisq.test(df[[symptom]], df$Biopsy_Result)

    malignant_rate <- prop.table(ct, 2)["Yes", "Malignant"] * 100
    benign_rate <- prop.table(ct, 2)["Yes", "Benign"] * 100

    symptom_stats <- rbind(
      symptom_stats,
      data.frame(
        Symptom = gsub("_", " ", symptom),
        Benign_Rate = round(benign_rate, 1),
        Malignant_Rate = round(malignant_rate, 1),
        Chi_Square = round(chi$statistic, 1),
        P_Value = format.pval(chi$p.value)
      )
    )
  }
}
print(symptom_stats)

p_symptoms <- ggplot(df %>%
                      pivot_longer(cols = all_of(symptoms),
                                   names_to = "Symptom",
                                   values_to = "Present") %>%
                      filter(Present == "Yes") %>%
                      group_by(Biopsy_Result, Symptom) %>%
                      tally() %>%
                      group_by(Biopsy_Result) %>%
                      mutate(Percent = 100 * n / sum(n)),
                    aes(x = gsub("_", " ", Symptom), y = Percent, fill = Biopsy_Result)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7, alpha = 0.8) +
  scale_fill_viridis_d(option = "D") +
  labs(
    title = "Symptom Prevalence by Biopsy Result",
    x = "Symptom",
    y = "Percentage (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "bottom"
  )
print(p_symptoms)
ggsave(paste0(output_dir, "/bivariate_symptoms_by_biopsy.png"),
       p_symptoms, width = 12, height = 6, dpi = 300)

# =============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS - MULTIVARIATE ANALYSIS
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 5: MULTIVARIATE ANALYSIS\n")
cat("========================================\n\n")

# 5.1 Correlation Matrix for Continuous Variables
cat("--- Correlation Matrix ---\n")

continuous_for_corr <- df %>%
  select(Age, PSA_Level, BMI, Screening_Age, Prostate_Volume) %>%
  cor(use = "complete.obs")

print(round(continuous_for_corr, 3))

p_corr <- ggcorrplot(
  continuous_for_corr,
  method = "square",
  type = "lower",
  lab = TRUE,
  lab_size = 4,
  colors = c("#E15759", "white", "#4E79A7"),
  title = "Correlation Matrix of Continuous Variables",
  ggtheme = theme_minimal(base_size = 12)
)
print(p_corr)
ggsave(paste0(output_dir, "/multivariate_correlation_matrix.png"),
       p_corr, width = 10, height = 8, dpi = 300)

# 5.2 Age vs PSA by Biopsy Result
cat("\n--- Age vs PSA Level by Biopsy Result ---\n")
p_age_psa <- ggplot(df, aes(x = Age, y = PSA_Level, color = Biopsy_Result)) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.5) +
  scale_color_viridis_d(option = "D") +
  scale_y_log10() +
  labs(
    title = "Age vs PSA Level by Biopsy Result",
    subtitle = "PSA shown on log scale",
    x = "Age (years)",
    y = "Log(PSA Level)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "bottom"
  )
print(p_age_psa)
ggsave(paste0(output_dir, "/multivariate_age_psa_biopsy.png"),
       p_age_psa, width = 10, height = 7, dpi = 300)

# 5.3 Prostate Volume vs PSA by Biopsy Result
cat("\n--- Prostate Volume vs PSA Level by Biopsy Result ---\n")
p_volume_psa <- ggplot(df, aes(x = Prostate_Volume, y = PSA_Level, color = Biopsy_Result)) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.5) +
  scale_color_viridis_d(option = "D") +
  scale_y_log10() +
  labs(
    title = "Prostate Volume vs PSA Level by Biopsy Result",
    subtitle = "PSA shown on log scale",
    x = "Prostate Volume (cc)",
    y = "Log(PSA Level)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "bottom"
  )
print(p_volume_psa)
ggsave(paste0(output_dir, "/multivariate_volume_psa_biopsy.png"),
       p_volume_psa, width = 10, height = 7, dpi = 300)

# 5.4 Mean PSA by Age Group and Biopsy Result
cat("\n--- Mean PSA by Age Group and Biopsy Result ---\n")
psa_by_age_biopsy <- df %>%
  group_by(Age_Group, Biopsy_Result) %>%
  summarize(
    Mean_PSA = mean(PSA_Level),
    SD_PSA = sd(PSA_Level),
    N = n()
  ) %>%
  ungroup()

print(psa_by_age_biopsy)

p_psa_age <- ggplot(psa_by_age_biopsy,
                    aes(x = Age_Group, y = Mean_PSA, fill = Biopsy_Result)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7, alpha = 0.8) +
  geom_errorbar(aes(ymin = Mean_PSA - SD_PSA/sqrt(N),
                    ymax = Mean_PSA + SD_PSA/sqrt(N)),
                position = position_dodge(0.7), width = 0.2) +
  scale_fill_viridis_d(option = "D") +
  labs(
    title = "Mean PSA Level by Age Group and Biopsy Result",
    subtitle = "Error bars show standard error",
    x = "Age Group",
    y = "Mean PSA Level (ng/mL)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "bottom"
  )
print(p_psa_age)
ggsave(paste0(output_dir, "/multivariate_psa_by_age_biopsy.png"),
       p_psa_age, width = 10, height = 6, dpi = 300)

# 5.5 Cancer Rate by Age Group and Family History
cat("\n--- Cancer Rate by Age Group and Family History ---\n")
cancer_by_age_family <- df %>%
  group_by(Age_Group, Family_History) %>%
  summarize(
    Total = n(),
    Malignant = sum(Biopsy_Result == "Malignant"),
    Cancer_Rate = 100 * mean(Biopsy_Result == "Malignant")
  ) %>%
  ungroup()

print(cancer_by_age_family)

p_cancer_age_family <- ggplot(cancer_by_age_family,
                              aes(x = Age_Group, y = Cancer_Rate,
                                  fill = Family_History)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7, alpha = 0.8) +
  geom_text(aes(label = paste0(round(Cancer_Rate, 0), "%")),
            position = position_dodge(0.7), vjust = -0.5, size = 4) +
  scale_fill_viridis_d(option = "cividis") +
  scale_y_continuous(limits = c(0, 60)) +
  labs(
    title = "Cancer Detection Rate by Age Group and Family History",
    x = "Age Group",
    y = "Cancer Detection Rate (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    legend.position = "bottom"
  )
print(p_cancer_age_family)
ggsave(paste0(output_dir, "/multivariate_cancer_age_family.png"),
       p_cancer_age_family, width = 10, height = 6, dpi = 300)

# =============================================================================
# SECTION 6: STATISTICAL MODELING - LOGISTIC REGRESSION
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 6: LOGISTIC REGRESSION\n")
cat("========================================\n\n")

# 6.1 Prepare Data for Modeling
cat("--- Preparing Data for Modeling ---\n")

# Select variables for modeling
model_vars <- c("Age", "PSA_Level", "DRE_Result", "Family_History",
                "Race_African_Ancestry", "Prostate_Volume", "Blood_in_Urine",
                "Genetic_Risk_Factors", "Previous_Cancer_History")

# Create modeling dataset
model_df <- df %>%
  select(all_of(model_vars), Cancer_Binary) %>%
  mutate(
    DRE_Result = ifelse(DRE_Result == "Abnormal", 1, 0),
    Family_History = ifelse(Family_History == "Yes", 1, 0),
    Race_African_Ancestry = ifelse(Race_African_Ancestry == "Yes", 1, 0),
    Blood_in_Urine = ifelse(Blood_in_Urine == "Yes", 1, 0),
    Genetic_Risk_Factors = ifelse(Genetic_Risk_Factors == "Yes", 1, 0),
    Previous_Cancer_History = ifelse(Previous_Cancer_History == "Yes", 1, 0)
  )

# Split data into training and testing sets
cat("\nSplitting data into training (70%) and testing (30%) sets...\n")
set.seed(123)  # For reproducibility
train_index <- createDataPartition(model_df$Cancer_Binary, p = 0.7, list = FALSE)
train_data <- model_df[train_index, ]
test_data <- model_df[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

# 6.2 Build Univariate Logistic Regression Models
cat("\n--- Univariate Logistic Regression ---\n\n")

univariate_results <- data.frame()
for (var in model_vars) {
  formula <- as.formula(paste("Cancer_Binary ~", var))
  model <- glm(formula, data = train_data, family = binomial)

  # Extract coefficients
  coef_summary <- summary(model)$coefficients
  odds_ratio <- exp(coef_summary[2, 1])
  conf_int <- exp(confint(model)[2, ])
  p_value <- coef_summary[2, 4]

  univariate_results <- rbind(
    univariate_results,
    data.frame(
      Variable = var,
      Odds_Ratio = round(odds_ratio, 3),
      CI_Lower = round(conf_int[1], 3),
      CI_Upper = round(conf_int[2], 3),
      P_Value = format.pval(p_value),
      Significant = ifelse(p_value < 0.05, "Yes", "No")
    )
  )
}
print(univariate_results)

# 6.3 Build Multivariate Logistic Regression Model
cat("\n--- Multivariate Logistic Regression ---\n\n")

full_formula <- as.formula(paste("Cancer_Binary ~", paste(model_vars, collapse = " + ")))
logistic_model <- glm(full_formula, data = train_data, family = binomial)

cat("Logistic Regression Model Summary:\n")
print(summary(logistic_model))

# Extract odds ratios and confidence intervals
coef_df <- data.frame(
  Variable = rownames(summary(logistic_model)$coefficients)[-1],
  Coefficient = summary(logistic_model)$coefficients[-1, 1],
  Std_Error = summary(logistic_model)$coefficients[-1, 2],
  Z_Value = summary(logistic_model)$coefficients[-1, 3],
  P_Value = summary(logistic_model)$coefficients[-1, 4]
)

coef_df$Odds_Ratio <- exp(coef_df$Coefficient)
conf_int <- exp(confint(logistic_model))[-1, ]
coef_df$CI_Lower <- conf_int[, 1]
coef_df$CI_Upper <- conf_int[, 2]

cat("\nOdds Ratios with 95% Confidence Intervals:\n")
print(coef_df)

# Model fit statistics
cat("\n--- Model Fit Statistics ---\n")
cat("Null Deviance:", round(logistic_model$null.deviance, 2), "\n")
cat("Residual Deviance:", round(logistic_model$deviance, 2), "\n")
cat("AIC:", round(logistic_model$aic, 2), "\n")

# Pseudo R-squared (Nagelkerke)
null_model <- glm(Cancer_Binary ~ 1, data = train_data, family = binomial)
r2_nagelkerke <- 1 - (logistic_model$deviance / logistic_model$null.deviance) /
  (null_model$deviance / null_model$null.deviance)
cat("Nagelkerke R-squared:", round(r2_nagelkerke, 3), "\n")

# 6.4 Model Predictions
cat("\n--- Making Predictions on Test Set ---\n")

# Predicted probabilities
test_data$Pred_Prob <- predict(logistic_model, newdata = test_data, type = "response")

# Predicted classes (using 0.5 cutoff)
test_data$Pred_Class <- ifelse(test_data$Pred_Prob > 0.5, "1", "0")
test_data$Pred_Class <- as.factor(test_data$Pred_Class)

# 6.5 Model Evaluation
cat("\n--- Model Evaluation ---\n\n")

# Confusion matrix
conf_matrix <- table(Actual = test_data$Cancer_Binary,
                      Predicted = test_data$Pred_Class)
print(conf_matrix)

# Calculate metrics with error handling for edge cases
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Check if confusion matrix has both classes
if (nrow(conf_matrix) >= 2 && ncol(conf_matrix) >= 2) {
  sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
  precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
} else {
  # Handle case where model predicts only one class
  sensitivity <- 0
  specificity <- if (nrow(conf_matrix) >= 2) conf_matrix[1, 1] / sum(conf_matrix[1, ]) else 0
  precision <- 0
}

recall <- sensitivity
f1_score <- if (precision + recall > 0) 2 * (precision * recall) / (precision + recall) else 0

cat("\nClassification Metrics:\n")
cat("  Accuracy:", round(accuracy * 100, 2), "%\n")
cat("  Sensitivity (Recall):", round(sensitivity * 100, 2), "%\n")
cat("  Specificity:", round(specificity * 100, 2), "%\n")
cat("  Precision:", round(precision * 100, 2), "%\n")
cat("  F1-Score:", round(f1_score * 100, 2), "%\n")

# 6.6 ROC Curve and AUC
cat("\n--- ROC Curve and AUC ---\n")

roc_obj <- roc(test_data$Cancer_Binary, as.numeric(test_data$Pred_Prob))
auc_value <- auc(roc_obj)
cat("Area Under the Curve (AUC):", round(auc_value, 3), "\n")

p_roc <- ggplot(data.frame(Sensitivity = roc_obj$sensitivities,
                           Specificity = roc_obj$specificities),
                aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_line(color = "#4E79A7", linewidth = 1.5) +
  geom_area(fill = "#4E79A7", alpha = 0.2) +
  geom_abline(linetype = "dashed", color = "gray50", linewidth = 1) +
  labs(
    title = "ROC Curve - Logistic Regression",
    subtitle = paste("AUC =", round(auc_value, 3)),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
  )
print(p_roc)
ggsave(paste0(output_dir, "/logistic_roc_curve.png"),
       p_roc, width = 8, height = 6, dpi = 300)

# =============================================================================
# SECTION 7: STATISTICAL MODELING - DECISION TREE
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 7: DECISION TREE CLASSIFICATION\n")
cat("========================================\n\n")

# 7.1 Build Decision Tree Model
cat("--- Building Decision Tree Model ---\n")

dt_model <- rpart(
  Cancer_Binary ~ Age + PSA_Level + DRE_Result + Family_History +
    Race_African_Ancestry + Prostate_Volume + Blood_in_Urine +
    Genetic_Risk_Factors + Previous_Cancer_History,
  data = train_data,
  method = "class",
  control = rpart.control(
    minsplit = 50,
    minbucket = 20,
    maxdepth = 5,
    cp = 0.01
  )
)

cat("\nDecision Tree Complexity Parameter:\n")
print(dt_model$cptable)

# 7.2 Prune Tree (select best cp)
cat("\n--- Pruning Decision Tree ---\n")
best_cp <- dt_model$cptable[which.min(dt_model$cptable[, "xerror"]), "CP"]
pruned_tree <- prune(dt_model, cp = best_cp)
cat("Best CP:", best_cp, "\n")

# 7.3 Plot Decision Tree
cat("\n--- Plotting Decision Tree ---\n")

# Save decision tree plot directly to file
png(paste0(output_dir, "/decision_tree_plot.png"), width = 1400, height = 800, res = 300)
rpart.plot(
  pruned_tree,
  type = 4,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree for Prostate Cancer Prediction"
)
dev.off()

cat("Decision tree plot saved.\n")

# 7.4 Decision Tree Predictions
cat("\n--- Decision Tree Predictions ---\n")

test_data$DT_Pred <- predict(pruned_tree, newdata = test_data, type = "class")

# Confusion matrix
dt_conf_matrix <- table(Actual = test_data$Cancer_Binary,
                        Predicted = test_data$DT_Pred)
print(dt_conf_matrix)

# Calculate metrics with error handling
dt_accuracy <- sum(diag(dt_conf_matrix)) / sum(dt_conf_matrix)

if (nrow(dt_conf_matrix) >= 2 && ncol(dt_conf_matrix) >= 2) {
  dt_sensitivity <- dt_conf_matrix[2, 2] / sum(dt_conf_matrix[2, ])
  dt_specificity <- dt_conf_matrix[1, 1] / sum(dt_conf_matrix[1, ])
} else {
  dt_sensitivity <- 0
  dt_specificity <- if (nrow(dt_conf_matrix) >= 2) dt_conf_matrix[1, 1] / sum(dt_conf_matrix[1, ]) else 0
}

cat("\nDecision Tree Classification Metrics:\n")
cat("  Accuracy:", round(dt_accuracy * 100, 2), "%\n")
cat("  Sensitivity:", round(dt_sensitivity * 100, 2), "%\n")
cat("  Specificity:", round(dt_specificity * 100, 2), "%\n")

# 7.5 Variable Importance
cat("\n--- Decision Tree Variable Importance ---\n")
dt_importance <- data.frame(
  Variable = names(pruned_tree$variable.importance),
  Importance = pruned_tree$variable.importance
) %>%
  arrange(desc(Importance))
print(dt_importance)

p_dt_importance <- ggplot(dt_importance, aes(x = reorder(Variable, Importance),
                                             y = Importance, fill = Importance)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(Importance, 2)), hjust = -0.1, size = 3) +
  scale_fill_viridis_c(option = "plasma") +
  coord_flip() +
  labs(
    title = "Decision Tree - Variable Importance",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    legend.position = "none"
  )
print(p_dt_importance)
ggsave(paste0(output_dir, "/decision_tree_importance.png"),
       p_dt_importance, width = 10, height = 6, dpi = 300)

# =============================================================================
# SECTION 8: STATISTICAL MODELING - RANDOM FOREST
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 8: RANDOM FOREST CLASSIFICATION\n")
cat("========================================\n\n")

# 8.1 Build Random Forest Model
cat("--- Building Random Forest Model ---\n")
cat("This may take a moment...\n")

rf_model <- randomForest(
  Cancer_Binary ~ Age + PSA_Level + DRE_Result + Family_History +
    Race_African_Ancestry + Prostate_Volume + Blood_in_Urine +
    Genetic_Risk_Factors + Previous_Cancer_History,
  data = train_data,
  ntree = 200,
  mtry = 3,
  importance = TRUE,
  na.action = na.omit
)

cat("\nRandom Forest Model:\n")
print(rf_model)

# 8.2 Variable Importance
cat("\n--- Random Forest Variable Importance ---\n")
rf_importance <- data.frame(
  Variable = rownames(rf_model$importance),
  MeanDecreaseGini = rf_model$importance[, "MeanDecreaseGini"],
  MeanDecreaseAccuracy = rf_model$importance[, "MeanDecreaseAccuracy"]
) %>%
  arrange(desc(MeanDecreaseGini))
print(rf_importance)

# Plot variable importance
p_rf_importance <- ggplot(rf_importance,
                          aes(x = reorder(Variable, MeanDecreaseGini),
                              y = MeanDecreaseGini, fill = MeanDecreaseGini)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(MeanDecreaseGini, 1)), hjust = -0.1, size = 3) +
  scale_fill_viridis_c(option = "viridis") +
  coord_flip() +
  labs(
    title = "Random Forest - Variable Importance (Gini)",
    x = "Variable",
    y = "Mean Decrease in Gini"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
    legend.position = "none"
  )
print(p_rf_importance)
ggsave(paste0(output_dir, "/random_forest_importance.png"),
       p_rf_importance, width = 10, height = 6, dpi = 300)

# 8.3 Random Forest Predictions
cat("\n--- Random Forest Predictions ---\n")

test_data$RF_Pred <- predict(rf_model, newdata = test_data, type = "class")
test_data$RF_Prob <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

# Confusion matrix
rf_conf_matrix <- table(Actual = test_data$Cancer_Binary,
                        Predicted = test_data$RF_Pred)
print(rf_conf_matrix)

# Calculate metrics with error handling
rf_accuracy <- sum(diag(rf_conf_matrix)) / sum(rf_conf_matrix)

if (nrow(rf_conf_matrix) >= 2 && ncol(rf_conf_matrix) >= 2) {
  rf_sensitivity <- rf_conf_matrix[2, 2] / sum(rf_conf_matrix[2, ])
  rf_specificity <- rf_conf_matrix[1, 1] / sum(rf_conf_matrix[1, ])
  rf_precision <- rf_conf_matrix[2, 2] / sum(rf_conf_matrix[, 2])
} else {
  rf_sensitivity <- 0
  rf_specificity <- if (nrow(rf_conf_matrix) >= 2) rf_conf_matrix[1, 1] / sum(rf_conf_matrix[1, ]) else 0
  rf_precision <- 0
}

rf_recall <- rf_sensitivity
rf_f1 <- if (rf_precision + rf_recall > 0) 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall) else 0

cat("\nRandom Forest Classification Metrics:\n")
cat("  Accuracy:", round(rf_accuracy * 100, 2), "%\n")
cat("  Sensitivity:", round(rf_sensitivity * 100, 2), "%\n")
cat("  Specificity:", round(rf_specificity * 100, 2), "%\n")
cat("  Precision:", round(rf_precision * 100, 2), "%\n")
cat("  F1-Score:", round(rf_f1 * 100, 2), "%\n")

# 8.4 ROC Curve for Random Forest
cat("\n--- ROC Curve for Random Forest ---\n")

rf_roc_obj <- roc(test_data$Cancer_Binary, as.numeric(as.character(test_data$RF_Prob)))
rf_auc <- auc(rf_roc_obj)
cat("Random Forest AUC:", round(rf_auc, 3), "\n")

p_rf_roc <- ggplot(data.frame(Sensitivity = rf_roc_obj$sensitivities,
                              Specificity = rf_roc_obj$specificities),
                   aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_line(color = "#59A14F", linewidth = 1.5) +
  geom_area(fill = "#59A14F", alpha = 0.2) +
  geom_abline(linetype = "dashed", color = "gray50", linewidth = 1) +
  labs(
    title = "ROC Curve - Random Forest",
    subtitle = paste("AUC =", round(rf_auc, 3)),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
  )
print(p_rf_roc)
ggsave(paste0(output_dir, "/random_forest_roc_curve.png"),
       p_rf_roc, width = 8, height = 6, dpi = 300)

# =============================================================================
# SECTION 9: MODEL COMPARISON
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 9: MODEL COMPARISON\n")
cat("========================================\n\n")

# Create comparison table
comparison_df <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(round(accuracy * 100, 1), round(dt_accuracy * 100, 1),
                round(rf_accuracy * 100, 1)),
  Sensitivity = c(round(sensitivity * 100, 1), round(dt_sensitivity * 100, 1),
                   round(rf_sensitivity * 100, 1)),
  Specificity = c(round(specificity * 100, 1), round(dt_specificity * 100, 1),
                  round(rf_specificity * 100, 1)),
  AUC = c(round(auc_value, 3), NA, round(rf_auc, 3))
)

cat("Model Performance Comparison:\n")
print(comparison_df)

# Plot comparison
p_comparison <- ggplot(comparison_df,
                       aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, alpha = 0.8) +
  geom_text(aes(label = paste0(Accuracy, "%")), vjust = -0.5, size = 5) +
  scale_fill_viridis_d(option = "D") +
  scale_y_continuous(limits = c(0, 100)) +
  labs(
    title = "Model Accuracy Comparison",
    x = "Model",
    y = "Accuracy (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )
print(p_comparison)
ggsave(paste0(output_dir, "/model_comparison_accuracy.png"),
       p_comparison, width = 10, height = 6, dpi = 300)

# Combined ROC Curves
p_combined_roc <- ggplot() +
  geom_line(data = data.frame(FPR = 1 - roc_obj$specificities,
                              TPR = roc_obj$sensitivities),
            aes(x = FPR, y = TPR, color = "Logistic Regression"), linewidth = 1.5) +
  geom_line(data = data.frame(FPR = 1 - rf_roc_obj$specificities,
                              TPR = rf_roc_obj$sensitivities),
            aes(x = FPR, y = TPR, color = "Random Forest"), linewidth = 1.5) +
  geom_abline(linetype = "dashed", color = "gray50", linewidth = 1) +
  scale_color_viridis_d(option = "D") +
  labs(
    title = "ROC Curve Comparison",
    subtitle = paste("Logistic Reg AUC =", round(auc_value, 3),
                     "| Random Forest AUC =", round(rf_auc, 3)),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "bottom"
  )
print(p_combined_roc)
ggsave(paste0(output_dir, "/combined_roc_curves.png"),
       p_combined_roc, width = 10, height = 7, dpi = 300)

# =============================================================================
# SECTION 10: SUMMARY AND CONCLUSIONS
# =============================================================================

cat("\n")
cat("========================================\n")
cat("SECTION 10: SUMMARY AND CONCLUSIONS\n")
cat("========================================\n\n")

cat("========================================\n")
cat("PROCogniTrack PROJECT SUMMARY\n")
cat("======================================
cat("   Benign cases:", sum(df$Biopsy_Result == "Benign"),
    "(", round(100 * mean(df$Biopsy_Result == "Benign"), 1), "%)\n")

cat("\n2. KEY RISK FACTORS (Univariate Analysis)\n")
cat("------------------------------------------\n")
cat("   - Age: Malignant mean =",
    round(mean(df$Age[df$Biopsy_Result == "Malignant"]), 1), "years\n")
cat("   - PSA Level: Malignant mean =",
    round(mean(df$PSA_Level[df$Biopsy_Result == "Malignant"]), 1), "ng/mL\n")
cat("   - DRE Abnormal:",
    round(100 * prop.table(table(df$DRE_Result, df$Biopsy_Result), 2)[2, 2], 1),
    "% in malignant cases\n")
cat("   - Family History:",
    round(100 * prop.table(table(df$Family_History, df$Biopsy_Result), 2)[2, 2], 1),
    "% in malignant cases\n")

cat("\n3. MODEL PERFORMANCE SUMMARY\n")
cat("---------------------------\n")
cat("   Logistic Regression:\n")
cat("     - Accuracy:", round(accuracy * 100, 1), "%\n")
cat("     - AUC:", round(auc_value, 3), "\n")
cat("   Decision Tree:\n")
cat("     - Accuracy:", round(dt_accuracy * 100, 1), "%\n")
cat("   Random Forest:\n")
cat("     - Accuracy:", round(rf_accuracy * 100, 1), "%\n")
cat("     - AUC:", round(rf_auc, 3), "\n")

cat("\n4. TOP PREDICTORS\n")
cat("----------------\n")
cat("   Based on Random Forest importance:\n")
for (i in 1:min(5, nrow(rf_importance))) {
  cat("     ", i, ".", rf_importance$Variable[i], "\n")
}

cat("\n5. OUTPUT FILES GENERATED\n")
cat("------------------------\n")
cat("   All plots saved to:", output_dir, "/\n")
cat("   Total plots generated:", length(list.files(output_dir)), "\n")

cat("\n========================================\n")
cat("ANALYSIS COMPLETE\n")
cat("========================================\n")

# =============================================================================
# END OF SCRIPT
# =============================================================================

cat("\n")
cat("ProCogniTrack - Prostate Cancer Prediction Analysis\n")
cat("R Code Execution Completed Successfully\n")
cat("\n")
==\n\n")

cat("1. DATASET OVERVIEW\n")
cat("-------------------\n")
cat("   Total patients:", nrow(df), "\n")
cat("   Malignant cases:", sum(df$Biopsy_Result == "Malignant"),
    "(", round(100 * mean(df$Biopsy_Result == "Malignant"), 1), "%)\n")