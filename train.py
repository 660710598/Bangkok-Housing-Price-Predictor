import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# 1. โหลดข้อมูล
df = pd.read_csv('Bangkok Housing Condo Apartment Prices.csv')

print("--- ตัวอย่างข้อมูล ---")
print(df.head())

# 2. แยก Features (X) และ Target (y)
# เลือกคอลัมน์ที่จะใช้ทำนาย
X = df[['Property Type', 'Location', 'Area (sq. ft.)', 'Bedrooms', 'Bathrooms']]
y = df['Price (THB)']

# 3. สร้างตัวแปลงข้อมูล (Preprocessor)
# บอก AI ว่า: คอลัมน์ไหนเป็นตัวเลข (ปล่อยไว้) คอลัมน์ไหนเป็นตัวหนังสือ (ให้แปลงเป็นเลข)
categorical_features = ['Property Type', 'Location']
numerical_features = ['Area (sq. ft.)', 'Bedrooms', 'Bathrooms']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. สร้างท่อ Pipeline (Preprocessor + Model)
# ข้อมูลเข้ามา -> แปลงเป็นเลข -> ส่งให้ Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. แบ่งข้อมูลและเทรน
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nกำลังเทรนโมเดล... ⏳")
model.fit(X_train, y_train)

# 6. วัดผล (Evaluation)
predictions = model.predict(X_test)

# คำนวณค่าต่างๆ
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse) # RMSE คือถอดรากที่สองของ MSE
r2 = r2_score(y_test, predictions)

print("-" * 30)
print("📊 สรุปผลการสอบของ AI")
print("-" * 30)
print(f"1. ความแม่นยำ (R2 Score):\t {r2:.2%} ")
print(f"2. ผิดพลาดเฉลี่ย (MAE):\t {mae:,.2f} บาท ")
print(f"3. ผิดพลาดรุนแรง (RMSE):\t {rmse:,.2f} บาท")
print("-" * 30)

# 7. บันทึกโมเดล
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("บันทึกโมเดลเรียบร้อย: house_model.pkl ✅")