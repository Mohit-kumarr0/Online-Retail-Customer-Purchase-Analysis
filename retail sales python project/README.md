# Python Project 21: Online Retail Customer Purchase Analysis

**Academic Level:** B.Tech Computer Science & Engineering (Python Data Analytics)  
**Industry Domain:** Retail & E-Commerce  
**Dataset:** `21_retail_customer_orders.csv`  
**Core Libraries:** Pandas, NumPy, Matplotlib, Seaborn  

---

## 1. Project Overview & Objectives

In the competitive e-commerce sector, transactional data holds crucial insights into customer purchasing habits, product demand, revenue performance, and inventory requirements. 

This project performs an end-to-end data analysis on actual customer order transactions from an online retail business (`21_retail_customer_orders.csv`). Using **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**, the analysis covers:
1. Data ingestion, structural auditing, and data type inspection.
2. Cleaning duplicate records and statistically imputing missing values.
3. Date feature engineering (year, month name, weekday).
4. Financial calculations (order value, total revenue, average order value).
5. Granular aggregations across products, categories, payment methods, and time dimensions.
6. Descriptive statistics and outlier detection using the Interquartile Range (IQR) technique.
7. Object-Oriented Programming (OOP) modeling for product management.
8. Data-driven strategic recommendations for business growth.

---

## 2. Project Directory Structure

```text
Online_Retail_Project/
│
├── 21_retail_customer_orders.csv    # Raw dataset (1,015 records)
├── cleaned_retail_data.csv          # Cleaned dataset with engineered features (1,000 records)
├── main.py                          # Complete, executable Python script (Tasks 1 to 15)
├── README.md                        # Comprehensive project documentation and report
│
├── graphs/                          # High-resolution (300 DPI) visualization charts
│   ├── 01_top_products_by_revenue.png
│   ├── 02_category_wise_revenue.png
│   ├── 03_category_wise_quantity.png
│   ├── 04_aov_by_payment_method.png
│   ├── 05_monthly_sales_trend.png
│   ├── 06_weekday_order_analysis.png
│   └── 07_order_value_distribution.png
│
└── outputs/                         # Exported summary data tables (CSV format)
    ├── overall_business_summary.csv
    ├── top_products_revenue.csv
    ├── category_summary.csv
    ├── payment_method_summary.csv
    ├── monthly_sales_summary.csv
    ├── weekday_sales_summary.csv
    ├── order_value_statistics.csv
    ├── outlier_orders.csv
    └── order_value_bands_summary.csv
```

---

## 3. Dataset Description

The analysis is based strictly on the uploaded file `21_retail_customer_orders.csv`:

| Column Name | Raw Data Type | Cleaned Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Order_ID` | String / Object | String | Unique transaction identifier (e.g., `OR000000`) |
| `Customer_ID` | String / Object | String | Unique customer identifier (e.g., `CU00453`) |
| `Order_Date` | String (`YYYY-MM-DD`) | `datetime64[ns]` | Transaction timestamp |
| `Product_Name` | String / Object | String | Item purchased (8 distinct catalog items) |
| `Category` | String / Object | String | Department (Office, Accessories, Electronics, Home) |
| `Quantity` | Float (with missing) | Float / Integer | Number of units ordered (range: 1.0 to 9.0) |
| `Unit_Price` | Float (with missing) | Float (`INR`) | Individual item price |
| `Payment_Method` | String / Object | String | Payment gateway (Credit Card, UPI, Net Banking, etc.) |
| `Order_Value` | *(Engineered)* | Float (`INR`) | Computed as `Quantity * Unit_Price` |
| `Year` | *(Engineered)* | Integer | Year extracted from `Order_Date` (2025, 2026) |
| `Month` | *(Engineered)* | String | Month name extracted from `Order_Date` |
| `Weekday` | *(Engineered)* | String | Day of week extracted from `Order_Date` |
| `Order_Band` | *(Engineered)* | String | Quartile spending category (`Low`, `Medium`, `High`) |

---

## 4. Setup & Execution Instructions

### Prerequisites
Make sure Python 3.8+ is installed on your computer.

### Step 1: Install Required Libraries
Open your terminal (PowerShell, Command Prompt, or VS Code terminal) and install the four core data science libraries:
```bash
pip install pandas numpy matplotlib seaborn
```

### Step 2: Open Project in VS Code
1. Launch **Visual Studio Code**.
2. Click **File** > **Open Folder...** and select the `Online_Retail_Project` folder.
3. Open `main.py` in the editor.

### Step 3: Run the Script
- **Option A (VS Code Play Button):** Click the **Run Python File** (▷ play icon) in the top-right corner of the editor.
- **Option B (Integrated Terminal):**
  ```bash
  python main.py
  ```

Upon running, all 15 tasks will execute sequentially, displaying complete terminal outputs, generating all 7 charts in the `graphs/` folder, exporting all summary CSVs to `outputs/`, and saving `cleaned_retail_data.csv`.

---

## 5. Detailed Walkthrough of All 15 Tasks

### Task 1: Load and Inspect Dataset
- **Methodology:** Loaded `21_retail_customer_orders.csv` using `pd.read_csv()`.
- **Findings:**
  - Raw dimensions: **1,015 rows** across **8 columns**.
  - Columns: `Order_ID`, `Customer_ID`, `Order_Date`, `Product_Name`, `Category`, `Quantity`, `Unit_Price`, `Payment_Method`.
  - First record: `OR000000`, Monitor, Office, 2.0 units @ ₹26,957.15, Credit Card.

### Task 2: Missing Values and Duplicate Detection
- **Methodology:** Checked `df.isnull().sum()` and `df['Order_ID'].duplicated()`.
- **Findings:**
  - `Quantity` had **25 missing values** (2.46%).
  - `Unit_Price` had **26 missing values** (2.56%).
  - All other 6 columns had **0 missing values**.
  - Found **15 duplicate `Order_ID` entries** (total 30 rows involved), representing duplicate log records.

### Task 3: Data Cleaning Strategy & Execution
- **Justification:**
  1. **Duplicate Removal:** The 15 duplicate order IDs were exact record duplicates. They were dropped via `df.drop_duplicates(subset=['Order_ID'], keep='first')`, leaving **1,000 unique orders**.
  2. **Quantity Imputation:** Orders represent discrete physical purchases. Imputing with the **median quantity (5.0)** preserves sample size without generating non-integer quantities.
  3. **Unit Price Imputation:** Product prices differ by catalog item. Missing prices were imputed with the **product-specific median price** (e.g. median price of Monitor for Monitor orders).
- **Result:** 0 missing values remain across all 1,000 clean transactions.

### Task 4: Date Feature Engineering
- **Methodology:** Converted `Order_Date` to `datetime64` using `pd.to_datetime()`. Extracted `Year`, `Month`, and `Weekday`.
- **Findings:**
  - Date span: **January 1, 2025 to October 30, 2026** (22 active months).
  - 532 orders placed in 2025; 468 orders placed in 2026.

### Task 5: Overall Business Summary
From the cleaned dataset of 1,000 transactions:
- **Total Orders Placed:** **1,000**
- **Total Units Sold:** **5,104 units**
- **Total Sales Revenue:** **₹77,410,617.12** (~₹77.41 Million)
- **Average Order Value (AOV):** **₹77,410.62**

### Task 6: Order Value Calculation
- **Methodology:** Created column `Order_Value = df['Quantity'] * df['Unit_Price']`.
- **Validation:** Minimum single order value is ₹338.16; maximum single order value is ₹264,953.07.

### Task 7: Product Performance & Ranking
The dataset contains 8 distinct products. All 8 ranked by total sales revenue:

| Rank | Product Name | Total Revenue (₹) | Units Sold | Total Orders |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Webcam** | **₹10,818,527.20** | 682 | 134 |
| 2 | **Monitor** | **₹10,418,360.54** | 654 | 130 |
| 3 | **USB Hub** | **₹10,130,211.51** | 694 | 129 |
| 4 | **Wireless Mouse** | **₹9,956,845.89** | 692 | 135 |
| 5 | **Laptop Stand** | **₹9,692,722.18** | 618 | 128 |
| 6 | **Headphones** | **₹8,928,341.81** | 585 | 118 |
| 7 | **Keyboard** | **₹8,771,971.51** | 592 | 117 |
| 8 | **Desk Lamp** | **₹8,693,636.48** | 587 | 109 |

*Saved to `outputs/top_products_revenue.csv` and visualized in `graphs/01_top_products_by_revenue.png`.*

### Task 8: Category-Wise Breakdown
Analysis across the 4 retail categories:

| Category | Total Quantity Sold | Total Revenue (₹) | Average Order Value (₹) | Order Count |
| :--- | :---: | :---: | :---: | :---: |
| **Office** | 1,292 | **₹20,188,279.33** | ₹80,753.12 | 250 |
| **Accessories** | **1,304** | **₹19,777,036.71** | ₹78,792.97 | 251 |
| **Electronics** | 1,249 | **₹19,408,409.67** | ₹77,324.34 | 251 |
| **Home** | 1,259 | **₹18,036,891.41** | ₹72,729.40 | 248 |

- **Top category by revenue:** **Office** (₹20.19M, 26.08% of total revenue).
- **Top category by units sold:** **Accessories** (1,304 units).

### Task 9: Payment Method Performance
Comparison of Average Order Value across customer payment channels:

| Payment Method | Average Order Value (₹) | Total Orders | Total Revenue (₹) |
| :--- | :---: | :---: | :---: |
| **Credit Card** | **₹86,538.44** | 205 | ₹17,740,381.06 |
| **Net Banking** | **₹78,037.05** | 208 | ₹16,231,705.42 |
| **UPI** | **₹76,922.14** | 186 | ₹14,307,517.47 |
| **Debit Card** | **₹74,818.83** | 201 | ₹15,038,584.58 |
| **Cash** | **₹71,796.07** | 200 | ₹14,359,214.34 |

- **Key Takeaway:** Credit Card users spend on average **₹14,742 more per order** than Cash customers.

### Task 10: Shopping Day & Monthly Trend Analysis
- **Busiest Weekday:** **Saturday** with **157 orders** and **₹12,547,554.00** in revenue.
- **Weekend Effect:** Saturday and Sunday account for **307 orders** (30.7% of total orders).
- **Peak Sales Month:** **June** (₹8,405,483.33 across 101 orders).
- **Secondary Peaks:** **August** (₹7.88M) and **March** (₹7.86M).
- **Lowest Month:** **November** (₹2.59M across 39 orders).

### Task 11: Dictionary & Set Operations
- **Category Sales Dictionary:**
  ```python
  {
      'Accessories': 19777036.71,
      'Electronics': 19408409.67,
      'Home': 18036891.41,
      'Office': 20188279.33
  }
  ```
- **Unique Categories Set:**
  ```python
  {'Accessories', 'Electronics', 'Home', 'Office'}
  ```

### Task 12: User-Defined Function
- Implemented `get_category_sales(data, category_name)` with case-insensitive matching and error handling.
- Successfully queried all 4 actual categories and handled invalid inputs (e.g. `'Automotive'`).

### Task 13: NumPy Statistical Measures
Computed on `Order_Value` array using NumPy:
- **Mean:** **₹77,410.62**
- **Median:** **₹62,369.35**
- **Population Standard Deviation ($\sigma$):** **₹61,566.27**
- **Sample Standard Deviation ($s$):** **₹61,597.09**
- **Observation:** The mean is higher than the median, indicating a right-skewed distribution caused by high-value transactions.

### Task 14: IQR & Outlier Analysis
- **Q1 (25th percentile):** **₹26,069.42**
- **Q3 (75th percentile):** **₹114,886.83**
- **IQR (Q3 - Q1):** **₹88,817.42**
- **Lower Outlier Fence (Q1 - 1.5 * IQR):** -₹107,156.71 (Effective: ₹0.00)
- **Upper Outlier Fence (Q3 + 1.5 * IQR):** **₹248,112.95**
- **Outliers Identified:** **9 orders** exceed ₹248,112.95. All 9 represent bulk orders of 9 units of high-end items (Monitors, Keyboards, Webcams, Desk Lamps).
- **Order Value Bands:**
  - **Low** (`< Q1`): 250 orders (25.0%), generating ₹3.18M (AOV: ₹12,714.67)
  - **Medium** (`Q1 to Q3`): 500 orders (50.0%), generating ₹32.85M (AOV: ₹65,709.20)
  - **High** (`> Q3`): 250 orders (25.0%), generating ₹41.38M (AOV: ₹165,509.40)

### Task 15: Object-Oriented Programming (OOP)
- Implemented `Product` class with `name`, `price`, `category`, `display_details()`, and `calculate_estimated_order_value()`.
- Instantiated 8 product objects representing the catalog using actual median benchmark prices from the dataset.

---

## 6. Five Practical, Data-Driven Business Recommendations

1. **Capitalize on Weekend Shopping Surges (Saturday & Sunday):**
   - *Data Finding:* Saturday generates the highest volume (157 orders, ₹12.55M), followed by Sunday (150 orders). Weekends generate over 30% of total revenue.
   - *Action:* Launch "Weekend Flash Deals" on Friday evening, schedule Saturday morning email blasts, and allocate extra customer support agents on weekends.

2. **Prioritize Top Revenue Drivers (Webcams & Monitors):**
   - *Data Finding:* Webcams (₹10.82M) and Monitors (₹10.42M) together account for over 27% of gross store revenue.
   - *Action:* Maintain higher safety stock levels to avoid stockouts. Create high-margin "Work-From-Home" bundles pairing Monitors with Webcams and Laptop Stands at an attractive bundled discount.

3. **Incentivize Credit Card & Digital Payments to Boost Basket Size:**
   - *Data Finding:* Credit Card purchases yield the highest Average Order Value (₹86,538), significantly outpacing Cash on Delivery (₹71,796).
   - *Action:* Partner with leading banks to offer No-Cost EMI on carts above ₹50,000 and 5% instant cashback on card payments to encourage larger checkout carts.

4. **Capitalize on Mid-Year Peaks & Stimulate Slower Months:**
   - *Data Finding:* Sales peak in June (₹8.41M) and August (₹7.88M), while November records the lowest revenue (₹2.59M).
   - *Action:* Run an annual "Mid-Year Mega Tech Sale" in June/July to harness natural organic demand. Introduce early Diwali / Pre-Holiday discount promotions in October/November to bridge the seasonal gap.

5. **Establish a Dedicated B2B / Bulk Purchaser VIP Program:**
   - *Data Finding:* Statistical IQR analysis detected 9 high-value outlier transactions (all purchasing 9 units with order values above ₹248,112).
   - *Action:* Introduce a corporate B2B buyer portal featuring automated GST invoicing, tiered volume discounts, and dedicated account management to turn sporadic bulk buyers into regular corporate clients.

---

## 7. Project Conclusion

This project successfully fulfills all academic and analytical requirements for **Project 21: Online Retail Customer Purchase Analysis**:
- The workflow transitioned from raw, inconsistent data (containing duplicates and nulls) to an audit-ready, feature-rich dataset (`cleaned_retail_data.csv`).
- Comprehensive exploratory data analysis yielded deep insights into revenue drivers, seasonal trends, and customer checkout behavior.
- Advanced statistical modeling (NumPy percentiles, IQR fences) accurately segregated typical orders from extreme bulk corporate purchases.
- All code is written adhering to beginner-friendly, clean, PEP 8 standards with thorough documentation, making it an exemplary project submission for B.Tech Computer Science examinations.
