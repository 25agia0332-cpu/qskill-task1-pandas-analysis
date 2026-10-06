import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "sales_data.csv", parse_dates=["Date"])
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistics:\n", df.describe(numeric_only=True))
print("\nAverage sales:", round(df["Sales"].mean(), 2))
print("Total sales:", round(df["Sales"].sum(), 2))
print("Total units:", int(df["Units_Sold"].sum()))

category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
print("\nSales by category:\n", category_sales)
print("\nSales by region:\n", region_sales)

plt.figure(figsize=(8,5))
category_sales.plot(kind="bar")
plt.title("Total Sales by Category")
plt.xlabel("Category"); plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(OUT/"bar_chart_category_sales.png", dpi=150)
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(df["Units_Sold"], df["Sales"], alpha=0.7)
plt.title("Units Sold vs Sales")
plt.xlabel("Units Sold"); plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(OUT/"scatter_units_vs_sales.png", dpi=150)
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT/"correlation_heatmap.png", dpi=150)
plt.close()

print("\nInsights:")
print("1. The highest-selling category is:", category_sales.index[0])
print("2. The highest-selling region is:", region_sales.index[0])
print("3. The scatter plot shows the relationship between units sold and sales.")
print("4. The heatmap shows correlations among numerical variables.")
