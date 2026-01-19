# Bivariate-Descriptive-Analysis-of-Fitness-Engagement-Data

## Executive Summary
This analysis examines user engagement patterns on a subscription-based fitness platform. Results indicate that **workout completion** is the primary driver of engagement, while commonly used metrics such as **video views** and **user tenure** are unreliable indicators of true engagement.

A strong positive correlation (**r = 0.70**) exists between **CompletedWorkouts** and **EngagementScore**, whereas the relationship between **VideoViews** and **EngagementScore** is negligible (**r = 0.03**). These findings suggest a strategic shift away from promoting video starts toward encouraging workout completion.

---

## Recommendations
- **Prioritize workout completion** by analyzing drop-off points and A/B testing shorter workouts (15–20 minutes).
- **Redefine churn risk** using behavioral health scores based on recent trends in CompletedWorkouts and EngagementScore rather than tenure.
- **Target re-engagement campaigns** toward users with high VideoViews but low CompletedWorkouts using reminders, tips, or shorter routines.
- **Move beyond New vs. Returning users** by applying clustering techniques to identify behavioral segments such as:
  - Consistent Finishers  
  - Occasional Viewers  
  - High Starters, Low Finishers  

These segments enable more precise and effective marketing and retention strategies.

---

## Part I: Business Understanding
The organization operates a subscription-based fitness platform focused on improving user engagement and retention. The primary business objectives are to:
- Reduce churn  
- Personalize the user experience  
- Support sustainable growth  

Key stakeholders include:
- **Executive Leadership** (Retention, Lifetime Value)
- **Product, Marketing, and Content Teams** (Operational and strategic decisions)

---

## Part II: Data Understanding
The dataset contains **1,000 users** with the following key variables:
- UserID  
- SubscriptionDate  
- UserType (New vs. Returning)  
- WorkoutType  
- VideoViews  
- CompletedWorkouts  
- EngagementScore  

Key observations:
- Returning users represent **73.3%** of the user base.
- CompletedWorkouts is positively skewed, indicating a small group of highly active users.
- No significant data quality issues were identified.

---

## Part III: Data Preparation
- SubscriptionDate was converted into a valid datetime format.
- A new feature, **TenureDays**, was created to measure subscription length.
- Data was cleaned and validated to ensure analytical consistency.

---

## Part IV: Modeling (Descriptive Analytics)

### A. Bivariate Analysis

#### 1. Workout Type vs. Engagement Score
Box plots indicate similar median engagement scores across workout types (Yoga, Dance, HIIT, etc.). While Dance and Yoga show slightly higher averages, no workout type is significantly more engaging, emphasizing the need for diverse content.

#### 2. User Type vs. Completed Workouts
New users show slightly higher median completed workouts, likely due to early motivation. Returning users display greater variability and high-end outliers, indicating the presence of long-term power users.

#### 3. Video Views vs. Engagement Score
Scatter plot analysis shows no meaningful relationship. The near-zero correlation (**r = 0.03**) confirms that video views function as a vanity metric rather than a true engagement indicator.

---

### B. Hypothesis Testing

**Hypothesis 1:** Engagement differs between New and Returning users  
- Test: Two-sample t-test  
- p-value: 0.9416  
- Conclusion: No statistically significant difference

**Hypothesis 2:** Workout type affects completed workouts  
- Test: One-way ANOVA  
- p-value: 0.3162  
- Conclusion: No statistically significant effect

---

### C. Multivariate Analysis

#### 1. Correlation Matrix
- Strong positive correlation between CompletedWorkouts and EngagementScore (**r = 0.70**)
- TenureDays shows minimal correlation with engagement metrics

#### 2. Pair Plot by User Type
Both New and Returning users exhibit the same positive relationship between CompletedWorkouts and EngagementScore, reinforcing that engagement drivers are consistent across the user lifecycle.

---

## Part V: Evaluation

### Key Findings
1. **Workout completion is the definitive engagement metric**
2. **User tenure does not predict engagement**
3. **New and Returning users follow similar engagement patterns**

### Limitations
- Correlation does not imply causation
- Important variables (workout duration, difficulty, instructor) are omitted
- Results are based on a sample of 1,000 users and require further validation

---

```python
# =========================================
# 1) Install necessary packages (Colab only)
# =========================================
# !pip install pandas numpy matplotlib seaborn scipy


# =====================
# 2) Import libraries
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt


# =========================================
# 3) Upload file (Google Colab)
# =========================================
from google.colab import files
uploaded = files.upload()


# ======================
# 4) Load the dataset
# ======================
file_path = list(uploaded.keys())[0]

# If the uploaded file is CSV:
df = pd.read_csv(file_path)

# If the uploaded file is Excel instead, use:
# df = pd.read_excel(file_path)

print("✅ Data loaded successfully!")
print("Shape:", df.shape)
df.head()


# =========================================
# 5) Basic data checks
# =========================================
print("\n--- Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Summary Stats ---")
df.describe()


# =========================================
# 6) Data Preparation / Feature Engineering
# =========================================
# Convert SubscriptionDate to datetime
df["SubscriptionDate"] = pd.to_datetime(df["SubscriptionDate"], errors="coerce")

# Create TenureDays using a reference "today" date from the dataset
today = df["SubscriptionDate"].max() + dt.timedelta(days=1)
df["TenureDays"] = (today - df["SubscriptionDate"]).dt.days

print("✅ TenureDays created.")
df[["SubscriptionDate", "TenureDays"]].head()


# =========================================
# 7) Bivariate Analysis (Visualizations)
# =========================================
sns.set(style="whitegrid")

# --- (A) WorkoutType vs EngagementScore (Boxplot) ---
plt.figure(figsize=(12, 6))
order = df.groupby("WorkoutType")["EngagementScore"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="WorkoutType", y="EngagementScore", order=order)
plt.title("Engagement Score Distribution by Workout Type")
plt.xlabel("Workout Type")
plt.ylabel("Engagement Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- (B) UserType vs CompletedWorkouts (Boxplot) ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="UserType", y="CompletedWorkouts")
plt.title("Completed Workouts: New vs. Returning Users")
plt.xlabel("User Type")
plt.ylabel("Completed Workouts")
plt.tight_layout()
plt.show()

# --- (C) VideoViews vs EngagementScore (Scatter + Regression) ---
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="VideoViews", y="EngagementScore",
            scatter_kws={"alpha": 0.4}, line_kws={"linestyle": "--"})
plt.title("Video Views vs. Engagement Score")
plt.xlabel("Video Views")
plt.ylabel("Engagement Score")
plt.tight_layout()
plt.show()

# Pearson correlation for VideoViews vs EngagementScore
corr_vv_es, p_vv_es = stats.pearsonr(df["VideoViews"].dropna(), df["EngagementScore"].dropna())
print(f"📌 Pearson Corr(VideoViews, EngagementScore) = {corr_vv_es:.4f} (p={p_vv_es:.4f})")


# =========================================
# 8) Hypothesis Testing
# =========================================
alpha = 0.05

# --- Hypothesis 1: EngagementScore differs by UserType (t-test) ---
new_scores = df.loc[df["UserType"] == "New", "EngagementScore"].dropna()
ret_scores = df.loc[df["UserType"] == "Returning", "EngagementScore"].dropna()

t_stat, p_val = stats.ttest_ind(new_scores, ret_scores, equal_var=False)  # Welch t-test safer
print("\n--- Hypothesis 1: EngagementScore (New vs Returning) ---")
print(f"T-stat: {t_stat:.4f}, p-value: {p_val:.4f}")
if p_val < alpha:
    print("✅ Reject H0: Significant difference in engagement scores.")
else:
    print("❌ Fail to reject H0: No significant difference in engagement scores.")

# --- Hypothesis 2: CompletedWorkouts differs by WorkoutType (ANOVA) ---
groups = [df.loc[df["WorkoutType"] == wt, "CompletedWorkouts"].dropna()
          for wt in df["WorkoutType"].dropna().unique()]

f_stat, p_anova = stats.f_oneway(*groups)
print("\n--- Hypothesis 2: CompletedWorkouts by WorkoutType (ANOVA) ---")
print(f"F-stat: {f_stat:.4f}, p-value: {p_anova:.4f}")
if p_anova < alpha:
    print("✅ Reject H0: WorkoutType affects completed workouts.")
else:
    print("❌ Fail to reject H0: No significant effect of WorkoutType on completed workouts.")


# =========================================
# 9) Multivariate: Correlation Heatmap
# =========================================
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()


# =========================================
# 10) Pair Plot (optional - may be slow)
# =========================================
# sns.pairplot(df[["VideoViews", "CompletedWorkouts", "EngagementScore", "TenureDays", "UserType"]],
#              hue="UserType", corner=True)
# plt.show()
