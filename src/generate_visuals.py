import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

train = pd.read_csv("data/raw/train.csv")

# Figure 1: Survival Distribution Across Class and Gender
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
sns.barplot(
    data=train,
    x="Pclass",
    y="Survived",
    hue="Sex",
    palette=["#3182CE", "#E53E3E"],
    errorbar=None,
    ax=ax
)
ax.set_title("Empirical Survival Rate by Passenger Class & Sex", fontsize=14, weight="bold", pad=15)
ax.set_xlabel("Passenger Class (Pclass)", fontsize=12, weight="bold")
ax.set_ylabel("Observed Survival Rate", fontsize=12, weight="bold")
ax.set_ylim(0, 1.05)
ax.axhline(0.3838, color="#718096", linestyle="--", linewidth=1.2, label="Base Survival Prior (38.4%)")
ax.legend(title="Sex", frameon=True, loc="upper right")
plt.tight_layout()
plt.savefig("figures/survival_distribution_by_demographic.png")
plt.close()

# Figure 2: Feature Correlation Heatmap
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr_df = train[corr_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
sns.heatmap(
    corr_df,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    square=True,
    linewidths=0.5,
    ax=ax
)
ax.set_title("Feature Correlation Matrix (Raw Numerical Covariates)", fontsize=14, weight="bold", pad=15)
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png")
plt.close()
