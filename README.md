# 🏙️ Bangkok Housing Price Predictor (AI)

โปรเจกต์ Machine Learning สำหรับทำนายราคาที่พักอาศัย (คอนโด, บ้าน, อพาร์ทเม้นท์) ในเขตกรุงเทพมหานคร โดยวิเคราะห์จากทำเล พื้นที่ใช้สอย และจำนวนห้อง
A Machine Learning project to estimate housing prices in Bangkok based on location, area, and property type.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

## 🎯 Features (ความสามารถของระบบ)
- **Price Prediction:** ทำนายราคาขายโดยใช้โมเดล Random Forest Regression
- **Interactive Web App:** ใช้งานง่ายผ่านหน้าเว็บ (พัฒนาด้วย Streamlit)
- **Data Visualization:** กราฟวิเคราะห์ความสัมพันธ์ของราคาและทำเลต่างๆ

## 🛠️ Tech Stack (เครื่องมือที่ใช้)
- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Random Forest, Pipeline, One-Hot Encoding)
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

## 📊 Model Performance (ผลลัพธ์โมเดล)
จากการเทรนโมเดลด้วยข้อมูลราคาที่พักอาศัยในกรุงเทพฯ ผลลัพธ์ที่ได้คือ:
- **R² Score:** 99.41% (Very High Accuracy)
- **MAE (Mean Absolute Error):** ~53,145 THB
- **RMSE:** ~205,826 THB

## 🚀 How to Run (วิธีใช้งาน)

1. **Clone this repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Bangkok-Housing-Price-Predictor.git](https://github.com/YOUR_USERNAME/Bangkok-Housing-Price-Predictor.git)
   cd Bangkok-Housing-Price-Predictor

```

2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Train the model (Optional)**
หากต้องการเทรนโมเดลใหม่ด้วยข้อมูลล่าสุด
```bash
python train.py

```


4. **Run the Web App**
เปิดหน้าเว็บสำหรับทำนายราคา
```bash
streamlit run app.py

```


5. **View Analytics**
ดู Data Visualization
```bash
python visualize.py

```



## 📂 Project Structure

```text
├── Bangkok Housing...csv   # Dataset (Raw Data)
├── train.py                # Script for training the ML model
├── app.py                  # Streamlit Web Application
├── visualize.py            # Script for data visualization
├── house_model.pkl         # Trained Model (Saved Artifact)
├── requirements.txt        # List of dependencies
└── README.md               # Project Documentation

```

---

Developed by [Your Name]

```

---

### ✅ Checklist สุดท้ายก่อนอัปโหลด

1.  **อย่าอัปโหลดโฟลเดอร์ `.venv`:** มันใหญ่และรกครับ ให้สร้างไฟล์ชื่อ `.gitignore` แล้วพิมพ์คำว่า `.venv` ใส่ลงไป (รวมถึง `__pycache__` ด้วย)
2.  **Dataset:** ไฟล์ CSV ของคุณขนาดเล็ก อัปโหลดขึ้นไปได้เลยครับ คนอื่นจะได้ลองเล่นได้
3.  **Screenshot (ทางเลือก):** ถ้าขยัน ให้แคปหน้าจอตอนรันเว็บ `app.py` สวยๆ แล้วเอาไปแปะใน README จะเรียกแขกได้ดีมากครับ

ยินดีด้วยครับ! คุณมีโปรเจกต์ Data Science ครบวงจร (End-to-End) ชิ้นแรกเป็นของตัวเองแล้ว 🎉

```