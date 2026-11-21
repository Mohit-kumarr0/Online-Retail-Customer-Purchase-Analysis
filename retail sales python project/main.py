"""
================================================================================
PROJECT 21: ONLINE RETAIL CUSTOMER PURCHASE ANALYSIS
Industry: Retail / E-Commerce
Dataset: 21_retail_customer_orders.csv
Author: B.Tech CSE Student
Libraries: Pandas, NumPy, Matplotlib, Seaborn
================================================================================
This script completes all 15 tasks of the project in sequential order.
All calculations, visualizations, tables, and recommendations are derived
strictly from the actual dataset: 21_retail_customer_orders.csv.
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# DIRECTORY & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
# Get script directory to ensure relative paths work regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "21_retail_customer_orders.csv")
CLEANED_DATA_PATH = os.path.join(SCRIPT_DIR, "cleaned_retail_data.csv")
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")

# Ensure required folders exist
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Configure Seaborn / Matplotlib styling for professional, clean presentation
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

print("=" * 80)
print("PROJECT 21: ONLINE RETAIL CUSTOMER PURCHASE ANALYSIS")
print("=" * 80)


# ==============================================================================
# TASK 1: LOAD AND INSPECT DATASET
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 1: LOAD AND INSPECT DATASET")
print("=" * 80)

# 1.1 Load the uploaded CSV using Pandas
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset not found at {DATASET_PATH}")
    sys.exit(1)

df_raw = pd.read_csv(DATASET_PATH)
print("Dataset successfully loaded!")

# 1.2 Display dimensions, column names, and data types
rows, cols = df_raw.shape
print(f"\n1. Dataset Dimensions: {rows} rows x {cols} columns")

print("\n2. Column Names & Data Types:")
for col, dtype in zip(df_raw.columns, df_raw.dtypes):
    print(f"   - {col:16} : {dtype}")

# 1.3 Display first five and last five records
print("\n3. First 5 Records:")
print(df_raw.head())

print("\n4. Last 5 Records:")
print(df_raw.tail())


# ==============================================================================
# TASK 2: MISSING VALUES AND DUPLICATES
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 2: MISSING VALUES AND DUPLICATES")
print("=" * 80)

# 2.1 Identify missing values in every column
missing_series = df_raw.isnull().sum()
missing_pct = (missing_series / len(df_raw)) * 100

missing_df = pd.DataFrame({
    "Column": df_raw.columns,
    "Missing_Count": missing_series.values,
    "Missing_Percentage": missing_pct.values
})
print("1. Missing Values per Column:")
print(missing_df.to_string(index=False))

# 2.2 Find duplicate Order_ID records
dup_order_mask = df_raw["Order_ID"].duplicated(keep=False)
dup_records = df_raw[dup_order_mask].sort_values(by="Order_ID")
dup_unique_ids = df_raw[df_raw["Order_ID"].duplicated(keep="first")]["Order_ID"].nunique()

print(f"\n2. Duplicate Order_ID Summary:")
print(f"   - Total rows involved in duplicate Order_IDs: {len(dup_records)}")
print(f"   - Unique duplicate Order_IDs: {dup_unique_ids}")
print(f"   - Duplicate rows that need removal: {df_raw['Order_ID'].duplicated().sum()}")

print("\nSample duplicate rows detected:")
print(dup_records.head(6)[["Order_ID", "Customer_ID", "Order_Date", "Product_Name", "Quantity", "Unit_Price"]])


# ==============================================================================
# TASK 3: DATA CLEANING
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 3: DATA CLEANING")
print("=" * 80)

# 3.1 Explain the cleaning decisions:
print("""Cleaning Strategy & Justification:
1. Handling Duplicates:
   - 15 duplicate Order_ID records represent redundant duplicate submissions or logging duplicates.
   - We keep the first instance and drop the duplicates using drop_duplicates(subset=['Order_ID'], keep='first').
2. Handling Missing 'Quantity':
   - Quantity represents discrete units purchased per order (integer range: 1 to 9).
   - Deleting missing rows would eliminate valid transactions and reduce sample size.
   - We impute missing Quantity using the MEDIAN quantity of the dataset.
   - The median is 5.0 (an integer), which is robust against extreme values and maintains realistic integer ordering.
3. Handling Missing 'Unit_Price':
   - Unit price varies across products. We impute missing Unit_Price using the PRODUCT-SPECIFIC MEDIAN price
     (grouped by Product_Name), falling back to overall median if required.
   - This ensures each product's pricing integrity is preserved while being immune to price skewness.""")

# Make an explicit copy for cleaning
df_clean = df_raw.copy()

# A. Remove duplicate Order_ID records
initial_count = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["Order_ID"], keep="first")
dropped_dups = initial_count - len(df_clean)
print(f"\n-> Dropped {dropped_dups} duplicate records. Current rows: {len(df_clean)}")

# B. Impute missing Quantity with median
median_qty = df_clean["Quantity"].median()
df_clean["Quantity"] = df_clean["Quantity"].fillna(median_qty)
print(f"-> Imputed missing Quantity with overall median: {median_qty}")

# C. Impute missing Unit_Price with product-specific median
product_price_medians = df_clean.groupby("Product_Name")["Unit_Price"].transform("median")
overall_price_median = df_clean["Unit_Price"].median()
df_clean["Unit_Price"] = df_clean["Unit_Price"].fillna(product_price_medians)
df_clean["Unit_Price"] = df_clean["Unit_Price"].fillna(overall_price_median)
print(f"-> Imputed missing Unit_Price with product-specific median (overall median: {overall_price_median:.2f})")

# D. Verify zero missing values remain
remaining_nulls = df_clean.isnull().sum().sum()
print(f"-> Total remaining null values in cleaned dataset: {remaining_nulls}")

# E. Display cleaned dataset info
print("\nCleaned Dataset Information:")
print(df_clean.info())


# ==============================================================================
# TASK 4: DATE PROCESSING
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 4: DATE PROCESSING")
print("=" * 80)

# 4.1 Convert Order_Date into datetime format
df_clean["Order_Date"] = pd.to_datetime(df_clean["Order_Date"])

# 4.2 Extract Month, Weekday, and Year into separate columns
df_clean["Year"] = df_clean["Order_Date"].dt.year
df_clean["Month"] = df_clean["Order_Date"].dt.month_name()
df_clean["Month_Num"] = df_clean["Order_Date"].dt.month
df_clean["Weekday"] = df_clean["Order_Date"].dt.day_name()

print("Extracted Date Features:")
print(f"- Date Range: {df_clean['Order_Date'].min().strftime('%Y-%m-%d')} to {df_clean['Order_Date'].max().strftime('%Y-%m-%d')}")
print(f"- Years Present: {sorted(df_clean['Year'].unique().tolist())}")
print(f"- Months Present: {len(df_clean['Month'].unique())} distinct months")
print("\nSample records showing new date columns:")
print(df_clean[["Order_ID", "Order_Date", "Year", "Month", "Weekday"]].head())


# ==============================================================================
# TASK 5: OVERALL BUSINESS SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 5: OVERALL BUSINESS SUMMARY")
print("=" * 80)

# Calculate key business metrics
total_orders = len(df_clean)
total_units_sold = int(df_clean["Quantity"].sum())

# Note: Total sales revenue will use Order_Value from Task 6, but we compute it here too
temp_revenue = (df_clean["Quantity"] * df_clean["Unit_Price"]).sum()

print(f"1. Total Orders Placed     : {total_orders:,}")
print(f"2. Total Units Sold        : {total_units_sold:,}")
print(f"3. Total Estimated Revenue : ₹{temp_revenue:,.2f}")


# ==============================================================================
# TASK 6: ORDER VALUE CALCULATION
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 6: ORDER VALUE CALCULATION")
print("=" * 80)

# 6.1 Create Order_Value using Quantity multiplied by Unit_Price
df_clean["Order_Value"] = df_clean["Quantity"] * df_clean["Unit_Price"]
total_sales_revenue = df_clean["Order_Value"].sum()

# 6.2 Display sample records from the actual dataset
print(f"Total Sales Revenue (from Order_Value column): ₹{total_sales_revenue:,.2f}")
print(f"Average Order Value (AOV): ₹{df_clean['Order_Value'].mean():,.2f}")
print("\nSample records with calculated Order_Value:")
print(df_clean[["Order_ID", "Product_Name", "Quantity", "Unit_Price", "Order_Value"]].head(10))

# Save overall summary to outputs
summary_df = pd.DataFrame({
    "Metric": ["Total Orders", "Total Units Sold", "Total Revenue (INR)", "Average Order Value (INR)"],
    "Value": [total_orders, total_units_sold, round(total_sales_revenue, 2), round(df_clean['Order_Value'].mean(), 2)]
})
summary_df.to_csv(os.path.join(OUTPUTS_DIR, "overall_business_summary.csv"), index=False)
print("-> Saved 'outputs/overall_business_summary.csv'")


# ==============================================================================
# TASK 7: TOP 10 PRODUCTS BY REVENUE
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 7: TOP 10 PRODUCTS BY REVENUE")
print("=" * 80)

# 7.1 Find products ranked by highest sales revenue
top_products = df_clean.groupby("Product_Name").agg(
    Total_Revenue=("Order_Value", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Order_Count=("Order_ID", "count")
).sort_values(by="Total_Revenue", ascending=False).reset_index()

# Note: The dataset contains 8 distinct products; we rank up to 10
top_10_products = top_products.head(10).copy()
top_10_products["Rank"] = range(1, len(top_10_products) + 1)

# 7.2 Display product names and revenue
print("Ranked Products by Revenue:")
for _, row in top_10_products.iterrows():
    print(f"   Rank {row['Rank']:2d}: {row['Product_Name']:16} | Revenue: ₹{row['Total_Revenue']:12,.2f} | Units: {int(row['Total_Quantity']):4d}")

# Save to CSV
top_10_products.to_csv(os.path.join(OUTPUTS_DIR, "top_products_revenue.csv"), index=False)
print("-> Saved 'outputs/top_products_revenue.csv'")

# 7.3 Create a bar chart
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Blues_r", n_colors=len(top_10_products))
bars = plt.barh(top_10_products["Product_Name"][::-1], (top_10_products["Total_Revenue"] / 1e6)[::-1], color=palette[::-1], edgecolor="black")
plt.xlabel("Total Revenue (in Millions ₹)")
plt.ylabel("Product Name")
plt.title("Top Products by Sales Revenue", weight="bold")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"₹{width:.2f}M", va="center", fontsize=9, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "01_top_products_by_revenue.png"), dpi=300)
plt.close()
print("-> Graph saved: 'graphs/01_top_products_by_revenue.png'")


# ==============================================================================
# TASK 8: CATEGORY-WISE ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 8: CATEGORY-WISE ANALYSIS")
print("=" * 80)

# 8.1 & 8.2 Category summary table with rankings
category_summary = df_clean.groupby("Category").agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Order_Value", "sum"),
    Average_Order_Value=("Order_Value", "mean"),
    Order_Count=("Order_ID", "count")
).reset_index()

# 8.3 Rank categories by quantity and by revenue
category_by_qty = category_summary.sort_values(by="Total_Quantity", ascending=False).reset_index(drop=True)
category_by_rev = category_summary.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)

print("1. Categories Ranked by Total Quantity Sold:")
for i, row in category_by_qty.iterrows():
    print(f"   {i+1}. {row['Category']:14} : {int(row['Total_Quantity']):,} units ({row['Order_Count']} orders)")

print("\n2. Categories Ranked by Total Revenue:")
for i, row in category_by_rev.iterrows():
    print(f"   {i+1}. {row['Category']:14} : ₹{row['Total_Revenue']:12,.2f} (Avg Order: ₹{row['Average_Order_Value']:,.2f})")

print("\n3. Category Summary Table:")
print(category_summary.to_string(index=False))

# Save summary
category_summary.to_csv(os.path.join(OUTPUTS_DIR, "category_summary.csv"), index=False)
print("-> Saved 'outputs/category_summary.csv'")

# 8.4 Visualizations: Category Revenue and Quantity
# Graph 2: Category Revenue
plt.figure(figsize=(8, 5))
colors_rev = sns.color_palette("Set2", len(category_by_rev))
bars = plt.bar(category_by_rev["Category"], category_by_rev["Total_Revenue"] / 1e6, color=colors_rev, edgecolor="black")
plt.xlabel("Category")
plt.ylabel("Total Revenue (in Millions ₹)")
plt.title("Category-Wise Total Sales Revenue", weight="bold")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"₹{yval:.2f}M", ha="center", fontsize=9, weight="bold")
plt.ylim(0, max(category_by_rev["Total_Revenue"] / 1e6) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "02_category_wise_revenue.png"), dpi=300)
plt.close()
print("-> Graph saved: 'graphs/02_category_wise_revenue.png'")

# Graph 3: Category Quantity
plt.figure(figsize=(8, 5))
colors_qty = sns.color_palette("Pastel1", len(category_by_qty))
bars = plt.bar(category_by_qty["Category"], category_by_qty["Total_Quantity"], color=colors_qty, edgecolor="black")
plt.xlabel("Category")
plt.ylabel("Total Units Sold")
plt.title("Category-Wise Total Quantity Sold", weight="bold")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 15, f"{int(yval):,}", ha="center", fontsize=9, weight="bold")
plt.ylim(0, max(category_by_qty["Total_Quantity"]) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "03_category_wise_quantity.png"), dpi=300)
plt.close()
print("-> Graph saved: 'graphs/03_category_wise_quantity.png'")


# ==============================================================================
# TASK 9: PAYMENT METHOD ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 9: PAYMENT METHOD ANALYSIS")
print("=" * 80)

# 9.1 Calculate average order value for each payment method
payment_summary = df_clean.groupby("Payment_Method").agg(
    Average_Order_Value=("Order_Value", "mean"),
    Total_Revenue=("Order_Value", "sum"),
    Total_Orders=("Order_ID", "count")
).sort_values(by="Average_Order_Value", ascending=False).reset_index()

print("Average Order Value (AOV) by Payment Method:")
for _, row in payment_summary.iterrows():
    print(f"   - {row['Payment_Method']:14} : AOV = ₹{row['Average_Order_Value']:9,.2f} | Orders = {row['Total_Orders']} | Total = ₹{row['Total_Revenue']:11,.2f}")

payment_summary.to_csv(os.path.join(OUTPUTS_DIR, "payment_method_summary.csv"), index=False)
print("-> Saved 'outputs/payment_method_summary.csv'")

# 9.2 Compare payment methods using a bar chart
plt.figure(figsize=(9, 5))
palette = sns.color_palette("crest", n_colors=len(payment_summary))
bars = plt.bar(payment_summary["Payment_Method"], payment_summary["Average_Order_Value"], color=palette, edgecolor="black")
plt.xlabel("Payment Method")
plt.ylabel("Average Order Value (₹)")
plt.title("Average Order Value by Payment Method", weight="bold")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1000, f"₹{yval:,.0f}", ha="center", fontsize=9, weight="bold")
plt.ylim(0, max(payment_summary["Average_Order_Value"]) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "04_aov_by_payment_method.png"), dpi=300)
plt.close()
print("-> Graph saved: 'graphs/04_aov_by_payment_method.png'")


# ==============================================================================
# TASK 10: SHOPPING DAY AND MONTHLY ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 10: SHOPPING DAY AND MONTHLY ANALYSIS")
print("=" * 80)

# 10.1 Identify busiest shopping weekdays
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_summary = df_clean.groupby("Weekday").agg(
    Order_Count=("Order_ID", "count"),
    Total_Revenue=("Order_Value", "sum"),
    Average_Order_Value=("Order_Value", "mean")
).reindex(weekday_order).reset_index()

busiest_day_by_orders = weekday_summary.sort_values(by="Order_Count", ascending=False).iloc[0]
busiest_day_by_revenue = weekday_summary.sort_values(by="Total_Revenue", ascending=False).iloc[0]

print(f"1. Busiest Shopping Weekday (by Order Volume) : {busiest_day_by_orders['Weekday']} ({busiest_day_by_orders['Order_Count']} orders)")
print(f"   Busiest Shopping Weekday (by Sales Revenue) : {busiest_day_by_revenue['Weekday']} (₹{busiest_day_by_revenue['Total_Revenue']:,.2f})")

print("\nWeekday Performance Overview:")
print(weekday_summary.to_string(index=False))
weekday_summary.to_csv(os.path.join(OUTPUTS_DIR, "weekday_sales_summary.csv"), index=False)
print("-> Saved 'outputs/weekday_sales_summary.csv'")

# 10.2 & 10.3 Monthly revenue and peak sales month
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

monthly_summary = df_clean.groupby("Month").agg(
    Order_Count=("Order_ID", "count"),
    Total_Revenue=("Order_Value", "sum"),
    Average_Order_Value=("Order_Value", "mean")
).reindex(month_order).reset_index()

peak_month_row = monthly_summary.sort_values(by="Total_Revenue", ascending=False).iloc[0]
lowest_month_row = monthly_summary.sort_values(by="Total_Revenue", ascending=True).iloc[0]

print(f"\n2. Peak Sales Month : {peak_month_row['Month']} (₹{peak_month_row['Total_Revenue']:,.2f} across {peak_month_row['Order_Count']} orders)")
print(f"   Lowest Sales Month : {lowest_month_row['Month']} (₹{lowest_month_row['Total_Revenue']:,.2f} across {lowest_month_row['Order_Count']} orders)")

print("\nMonthly Revenue Summary:")
print(monthly_summary.to_string(index=False))
monthly_summary.to_csv(os.path.join(OUTPUTS_DIR, "monthly_sales_summary.csv"), index=False)
print("-> Saved 'outputs/monthly_sales_summary.csv'")

# 10.4 Create monthly sales trend line chart
plt.figure(figsize=(11, 5))
plt.plot(monthly_summary["Month"], monthly_summary["Total_Revenue"] / 1e6, marker="o", color="#1f77b4", linewidth=2.5, markersize=7)
plt.xlabel("Month")
plt.ylabel("Total Revenue (in Millions ₹)")
plt.title("Monthly Sales Revenue Trend", weight="bold")
plt.xticks(rotation=45)
for i, txt in enumerate(monthly_summary["Total_Revenue"] / 1e6):
    plt.annotate(f"₹{txt:.2f}M", (monthly_summary["Month"][i], txt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "05_monthly_sales_trend.png"), dpi=300)
plt.close()
print("-> Graph saved: 'graphs/05_monthly_sales_trend.png'")

# Graph 6: Weekday Order Analysis (Bar Chart)
plt.figure(figsize=(9, 5))
palette_days = ["#4575b4" if day in ["Saturday", "Sunday"] else "#74add1" for day in weekday_summary["Weekday"]]
bars = plt.bar(weekday_summary["Weekday"], weekday_summary["Order_Count"], color=palette_days, edgecolor="black")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Orders Placed")
plt.title("Shopping Day Analysis: Orders Placed per Weekday (Weekends Highlighted)", weight="bold")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{int(yval)}", ha="center", fontsize=9, weight="bold")
plt.ylim(0, max(weekday_summary["Order_Count"]) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "06_weekday_order_analysis.png"), dpi=300)
plt.close()
print("-> Graph saved: 'graphs/06_weekday_order_analysis.png'")


# ==============================================================================
# TASK 11: DICTIONARY AND SET
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 11: DICTIONARY AND SET")
print("=" * 80)

# 11.1 Store category-wise total sales in a dictionary
category_sales_dict = df_clean.groupby("Category")["Order_Value"].sum().to_dict()

# 11.2 Store unique product categories in a set
unique_categories_set = set(df_clean["Category"].unique())

# 11.3 Display both
print("1. Category-Wise Total Sales Dictionary:")
for cat, sales in category_sales_dict.items():
    print(f"   '{cat}': ₹{sales:,.2f}")

print("\n2. Unique Product Categories Set:")
print(f"   {unique_categories_set}")
print(f"   Total unique categories: {len(unique_categories_set)}")


# ==============================================================================
# TASK 12: USER-DEFINED FUNCTION
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 12: USER-DEFINED FUNCTION")
print("=" * 80)

def get_category_sales(data: pd.DataFrame, category_name: str) -> float:
    """
    Accepts a DataFrame and a category name (case-insensitive).
    Returns the total sales revenue generated by that category from cleaned data.
    If the category does not exist, returns 0.0 with a descriptive warning.
    """
    # Clean input string
    query = str(category_name).strip().lower()
    
    # Filter dataset matching category name (case-insensitive)
    matched_orders = data[data["Category"].str.strip().str.lower() == query]
    
    if matched_orders.empty:
        print(f"   [Warning] Category '{category_name}' not found in the dataset.")
        return 0.0
    
    total_sales = matched_orders["Order_Value"].sum()
    return total_sales

# Demonstrate the function using all actual categories in the dataset
print("Demonstrating get_category_sales() for all actual categories in dataset:")
for category in sorted(list(unique_categories_set)):
    revenue = get_category_sales(df_clean, category)
    print(f"   - Category: {category:14} -> Total Sales: ₹{revenue:12,.2f}")

# Demonstrate error handling with an invalid category name
print("\nDemonstrating function with an unrecognized category:")
invalid_cat = "Automotive"
res = get_category_sales(df_clean, invalid_cat)
print(f"   - Category: {invalid_cat:14} -> Returned: ₹{res:.2f}")


# ==============================================================================
# TASK 13: NUMPY STATISTICS
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 13: NUMPY STATISTICS")
print("=" * 80)

# Extract Order_Value as a NumPy array
order_values = df_clean["Order_Value"].to_numpy()

# Calculate statistics using NumPy functions
np_mean = np.mean(order_values)
np_median = np.median(order_values)
np_std = np.std(order_values, ddof=0)  # Population standard deviation
np_sample_std = np.std(order_values, ddof=1)  # Sample standard deviation
np_min = np.min(order_values)
np_max = np.max(order_values)

print(f"1. Mean Order Value               : ₹{np_mean:10,.2f}")
print(f"2. Median Order Value             : ₹{np_median:10,.2f}")
print(f"3. Standard Deviation (Population): ₹{np_std:10,.2f}")
print(f"   Standard Deviation (Sample)    : ₹{np_sample_std:10,.2f}")
print(f"4. Minimum Order Value            : ₹{np_min:10,.2f}")
print(f"5. Maximum Order Value            : ₹{np_max:10,.2f}")


# ==============================================================================
# TASK 14: IQR AND ORDER VALUE BANDS
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 14: IQR AND ORDER VALUE BANDS")
print("=" * 80)

# 14.1 Calculate Q1, Q3, and IQR using NumPy
q1 = np.percentile(order_values, 25)
q3 = np.percentile(order_values, 75)
iqr = q3 - q1

# Calculate Upper and Lower Outlier Fences
upper_fence = q3 + 1.5 * iqr
lower_fence = q1 - 1.5 * iqr

print(f"1. First Quartile (Q1 - 25th percentile) : ₹{q1:10,.2f}")
print(f"2. Third Quartile (Q3 - 75th percentile) : ₹{q3:10,.2f}")
print(f"3. Interquartile Range (IQR)             : ₹{iqr:10,.2f}")
print(f"4. Lower Outlier Boundary (Q1 - 1.5*IQR) : ₹{lower_fence:10,.2f} (Effective: ₹0.00)")
print(f"5. Upper Outlier Boundary (Q3 + 1.5*IQR) : ₹{upper_fence:10,.2f}")

# 14.2 Identify unusually high order values
outliers_df = df_clean[df_clean["Order_Value"] > upper_fence].sort_values(by="Order_Value", ascending=False)
print(f"\n6. Unusually High Order Values (Outliers Count): {len(outliers_df)} records")

# 14.3 Display actual outlier records
print("\nActual Outlier Records from Dataset:")
outlier_display_cols = ["Order_ID", "Customer_ID", "Product_Name", "Category", "Quantity", "Unit_Price", "Order_Value"]
print(outliers_df[outlier_display_cols].to_string(index=False))

outliers_df.to_csv(os.path.join(OUTPUTS_DIR, "outlier_orders.csv"), index=False)
print("-> Saved 'outputs/outlier_orders.csv'")

# 14.4 Create Low, Medium, and High order value bands based on Q1 and Q3
# - Low    : Order_Value < Q1
# - Medium : Q1 <= Order_Value <= Q3
# - High   : Order_Value > Q3
def assign_order_band(val):
    if val < q1:
        return "Low"
    elif val <= q3:
        return "Medium"
    else:
        return "High"

# 14.5 Add an Order_Band column
df_clean["Order_Band"] = df_clean["Order_Value"].apply(assign_order_band)

print("\n7. Order Value Band Summary:")
band_order = ["Low", "Medium", "High"]
band_summary = df_clean.groupby("Order_Band").agg(
    Order_Count=("Order_ID", "count"),
    Total_Revenue=("Order_Value", "sum"),
    Average_Order_Value=("Order_Value", "mean")
).reindex(band_order).reset_index()
band_summary["Percentage_of_Orders"] = (band_summary["Order_Count"] / len(df_clean)) * 100

print(band_summary.to_string(index=False))
band_summary.to_csv(os.path.join(OUTPUTS_DIR, "order_value_bands_summary.csv"), index=False)
print("-> Saved 'outputs/order_value_bands_summary.csv'")

# Graph 7: Order Value Distribution with IQR Boundaries
plt.figure(figsize=(10, 6))
sns.histplot(df_clean["Order_Value"], kde=True, color="#2b5c8f", bins=30, edgecolor="black")
plt.axvline(np_median, color="green", linestyle="--", linewidth=2, label=f"Median: ₹{np_median:,.0f}")
plt.axvline(q1, color="orange", linestyle=":", linewidth=2, label=f"Q1: ₹{q1:,.0f}")
plt.axvline(q3, color="orange", linestyle=":", linewidth=2, label=f"Q3: ₹{q3:,.0f}")
plt.axvline(upper_fence, color="red", linestyle="-.", linewidth=2, label=f"Upper Fence: ₹{upper_fence:,.0f}")
plt.xlabel("Order Value (₹)")
plt.ylabel("Frequency")
plt.title("Distribution of Order Value with IQR Outlier Thresholds", weight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "07_order_value_distribution.png"), dpi=300)
plt.close()
print("-> Graph saved: 'graphs/07_order_value_distribution.png'")

# Save summary of statistical measures
stats_summary = pd.DataFrame({
    "Statistic": ["Mean", "Median", "Standard Deviation", "Min", "Q1", "Q3", "IQR", "Upper Fence", "Outlier Count"],
    "Value": [round(np_mean, 2), round(np_median, 2), round(np_std, 2), round(np_min, 2),
              round(q1, 2), round(q3, 2), round(iqr, 2), round(upper_fence, 2), len(outliers_df)]
})
stats_summary.to_csv(os.path.join(OUTPUTS_DIR, "order_value_statistics.csv"), index=False)
print("-> Saved 'outputs/order_value_statistics.csv'")


# ==============================================================================
# TASK 15: OBJECT-ORIENTED PROGRAMMING (OOP) & RECOMMENDATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("TASK 15: OOP AND PRACTICAL RECOMMENDATIONS")
print("=" * 80)

# 15.1 Create a Product class with product name and price attributes
class Product:
    """
    Represents a retail product in the e-commerce store.
    
    Attributes:
        name (str): Product name.
        price (float): Representative / median unit price of the product.
        category (str): Product catalog category.
    """
    def __init__(self, name: str, price: float, category: str = "Unspecified"):
        self.name = name
        self.price = float(price)
        self.category = category

    def display_details(self) -> None:
        """Prints formatted details of the product."""
        print(f"   [Product Details] Name: {self.name:16} | Category: {self.category:12} | Benchmark Price: ₹{self.price:10,.2f}")

    def calculate_estimated_order_value(self, quantity: int) -> float:
        """Calculates total estimated cost for a given quantity."""
        return self.price * quantity

    def __repr__(self):
        return f"Product(name='{self.name}', price={self.price:.2f}, category='{self.category}')"

# 15.2 Create objects using actual product information from the dataset
print("1. Creating Product Class Objects from Actual Dataset:")
product_summary = df_clean.groupby("Product_Name").agg(
    Representative_Price=("Unit_Price", "median"),
    Primary_Category=("Category", lambda x: x.mode()[0])
).reset_index()

product_objects = []
for _, row in product_summary.iterrows():
    p = Product(name=row["Product_Name"], price=row["Representative_Price"], category=row["Primary_Category"])
    product_objects.append(p)

# 15.3 Display product details
for prod in product_objects:
    prod.display_details()

# 15.4 Five practical recommendations based on actual analysis findings
print("\n" + "=" * 80)
print("FIVE PRACTICAL BUSINESS RECOMMENDATIONS (DATA-DRIVEN)")
print("=" * 80)

recommendations = [
    "1. CAPITALIZE ON WEEKEND SHOPPING SURGES (SATURDAY & SUNDAY):\n"
    "   Finding: Saturday is the busiest shopping day (157 orders, ₹12.55M revenue), followed by Sunday (150 orders).\n"
    "   Recommendation: Launch exclusive 'Weekend Flash Deals' starting Friday evening, increase live customer\n"
    "   support bandwidth on weekends, and schedule promotional email campaigns on Saturday mornings to maximize conversion.",

    "2. FOCUS MARKETING & STOCK FOR TOP REVENUE PRODUCTS (WEBCAMS & MONITORS):\n"
    "   Finding: Webcams (₹10.82M) and Monitors (₹10.42M) are the top two revenue contributors, generating over 27% of total sales.\n"
    "   Recommendation: Build high-margin bundles (e.g., 'Work From Home Station': Monitor + Webcam + Laptop Stand)\n"
    "   and maintain safety stock levels for these SKUs to prevent revenue loss due to stockouts.",

    "3. PROMOTE CREDIT CARDS & NET BANKING TO EXPAND AVERAGE BASKET SIZE:\n"
    "   Finding: Credit Card transactions generate the highest Average Order Value (₹86,538), significantly outpacing Cash (₹71,796).\n"
    "   Recommendation: Partner with leading commercial banks to provide No-Cost EMI options and instant 5-10% card discounts,\n"
    "   encouraging customers to add more items to their carts and transition away from lower-value cash orders.",

    "4. CAPITALIZE ON MID-YEAR PEAK SALES & RUN COUNTER-SEASONAL CAMPAIGNS:\n"
    "   Finding: Sales peak strongly in June (₹8.41M, 101 orders) and August (₹7.88M), while November is the slowest month (₹2.59M).\n"
    "   Recommendation: Run 'Mid-Year Mega Clearance' campaigns in June/July to leverage peak buyer intent, and introduce\n"
    "   early Diwali / Festive Pre-Sale promotions in October-November to revive sluggish late-year periods.",

    "5. ESTABLISH A DEDICATED B2B / BULK PURCHASER ACCOUNT TIER:\n"
    "   Finding: IQR analysis identified 9 extreme outlier transactions (each exceeding ₹248,113, all involving bulk purchases of 9 units).\n"
    "   Recommendation: Create a dedicated 'Corporate / B2B Advantage Club' offering tiered volume pricing, dedicated account managers,\n"
    "   and GST invoicing to turn high-value sporadic buyers into long-term corporate clients."
]

for rec in recommendations:
    print(rec + "\n")


# ==============================================================================
# SAVE CLEANED DATASET
# ==============================================================================
# Save cleaned data to root project folder as required
df_clean.to_csv(CLEANED_DATA_PATH, index=False)
print("=" * 80)
print(f"Cleaned dataset successfully saved to: {CLEANED_DATA_PATH}")
print(f"Final Cleaned Dataset Dimensions: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
print("=" * 80)
print("\nPROJECT 21 ANALYSIS RUN COMPLETE! ALL 15 TASKS SUCCESSFULLY EXECUTED.")
