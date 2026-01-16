import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl # เพิ่มตัวนี้เข้ามา


# เลือกฟอนต์ที่มีในเครื่อง Windows (Tahoma หรือ Leelawadee UI หรือ Angsana New)
mpl.rc('font', family='Angsana New') 

# แก้ปัญหาเครื่องหมายลบ (-) กลายเป็นสี่เหลี่ยม
mpl.rcParams['axes.unicode_minus'] = False 

# ตั้งค่า Theme กราฟ
sns.set_theme(style="whitegrid", font='Angsana New')
# ---------------------------------------------------------

# 1. โหลดข้อมูล
filename = "Bangkok Housing Condo Apartment Prices.csv"
df = pd.read_csv(filename)

# ---------------------------------------------------------
# กราฟที่ 1: การกระจายตัวของราคา
plt.figure(figsize=(10, 6))
sns.histplot(df['Price (THB)'], kde=True, color='blue')
plt.title('การกระจายตัวของราคาที่พักอาศัย (Price Distribution)', fontsize=16)
plt.xlabel('ราคา (บาท)', fontsize=12)
plt.ylabel('จำนวน', fontsize=12)
plt.show()

# ---------------------------------------------------------
# กราฟที่ 2: พื้นที่ vs ราคา
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Area (sq. ft.)', y='Price (THB)', hue='Property Type', data=df, alpha=0.7)
plt.title('ความสัมพันธ์ระหว่างพื้นที่ใช้สอยกับราคา', fontsize=16)
plt.xlabel('พื้นที่ใช้สอย (sq. ft.)', fontsize=12)
plt.ylabel('ราคา (บาท)', fontsize=12)
plt.show()

# ---------------------------------------------------------
# กราฟที่ 3: ราคาแยกตามทำเล
plt.figure(figsize=(12, 6))
sns.boxplot(x='Location', y='Price (THB)', data=df, palette="Set3")
plt.title('ช่วงราคาของแต่ละทำเล (Price by Location)', fontsize=16)
plt.xticks(rotation=45) 
plt.ylabel('ราคา (บาท)', fontsize=12)
plt.show()

# ---------------------------------------------------------
# กราฟที่ 4: Correlation Matrix
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('ตารางความสัมพันธ์ของตัวแปร', fontsize=16)
plt.show()