# 🏙️ Bangkok Housing Price Predictor (BaanAI)

> **Machine Learning Web Application for Real Estate Valuation in Bangkok** > เว็บแอปพลิเคชัน AI สำหรับประเมินราคาที่พักอาศัยในกรุงเทพฯ (คอนโด, บ้าน, อพาร์ทเม้นท์)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)

## 📖 Overview (ภาพรวมโปรเจกต์)
โปรเจกต์นี้พัฒนาขึ้นเพื่อศึกษาและประยุกต์ใช้ **Machine Learning (Supervised Learning)** ในการทำนายราคาอสังหาริมทรัพย์ โดยโมเดลเรียนรู้จากข้อมูลจริงในกรุงเทพมหานคร ผู้ใช้งานสามารถระบุทำเล (Location), ขนาดพื้นที่ (Area), และจำนวนห้อง เพื่อให้ AI ประเมินราคาขายที่เหมาะสมได้ทันที

This project utilizes a **Random Forest Regression** model to predict housing prices in Bangkok based on key features like location, property type, and size.

## ✨ Features (ฟีเจอร์หลัก)
* **Price Prediction AI:** ระบบทำนายราคาที่มีความแม่นยำสูง (R² Score > 99%)
* **Interactive Dashboard:** ใช้งานง่ายผ่านหน้าเว็บด้วย **Streamlit**
* **Data Visualization:** กราฟวิเคราะห์แนวโน้มราคาตลาดในแต่ละทำเล
* **Real-time Processing:** ประมวลผลและแปลงข้อมูล (One-Hot Encoding) ผ่าน Pipeline อัตโนมัติ

## 🛠️ Tech Stack (เครื่องมือที่ใช้)
* **Language:** Python
* **Core Library:** Pandas, NumPy (Data Manipulation)
* **Machine Learning:** Scikit-learn (Random Forest, Pipeline)
* **Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit

## 📊 Model Performance (ผลการทดสอบ)
จากการเทรนโมเดลด้วยข้อมูลที่พักอาศัยในกรุงเทพฯ ผลลัพธ์ที่ได้คือ:
| Metric | Value | Description |
| :--- | :--- | :--- |
| **R² Score** | **99.41%** | ความแม่นยำของโมเดล (สูงมาก) |
| **MAE** | ~53,145 THB | ความคลาดเคลื่อนเฉลี่ย (Mean Absolute Error) |
| **RMSE** | ~205,826 THB | ความคลาดเคลื่อนเมื่อเจอบ้านราคาสูงผิดปกติ |

## 🚀 Installation & Usage (วิธีติดตั้งและใช้งาน)

### 1. Clone Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Bangkok-Housing-Price-Predictor.git](https://github.com/YOUR_USERNAME/Bangkok-Housing-Price-Predictor.git)
cd Bangkok-Housing-Price-Predictor

```

### 2. Install Dependencies

ติดตั้ง Library ที่จำเป็นทั้งหมด

```bash
pip install -r requirements.txt

```

### 3. Run the Application

เปิดหน้าเว็บเพื่อเริ่มใช้งาน

```bash
streamlit run app.py

```

### 4. (Optional) Retrain Model

หากต้องการเทรนโมเดลใหม่ด้วยข้อมูลล่าสุด

```bash
python train.py

```

### 5. (Optional) View Analytics

ดู Data Visualization กราฟวิเคราะห์ข้อมูล

```bash
python visualize.py

```

## 📂 Project Structure

```text
├── Bangkok Housing...csv   # Raw Dataset
├── train.py                # Script for training ML model (Creates .pkl)
├── app.py                  # Streamlit Web Application (Frontend)
├── visualize.py            # Script for generating graphs/charts
├── house_model.pkl         # Trained Model (Saved Artifact)
├── requirements.txt        # Python dependencies
└── README.md               # Project Documentation

```

---

**Developed by:** [Dechatorn Laikhain](www.linkedin.com/in/dechatorn-laikhain)

*Data Source:* [Kaggle (Bangkok Housing Condo Apartment Prices](https://www.kaggle.com/datasets/varintornsithisint/bangkok-housing-condo-apartment-prices)
