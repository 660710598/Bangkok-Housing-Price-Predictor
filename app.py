import streamlit as st
import pandas as pd
import pickle

# 1. โหลดโมเดล
try:
    with open('house_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ house_model.pkl กรุณารัน train.py ก่อนนะครับ")
    st.stop()

st.title("🏙️ Bangkok Housing Price Predictor")
st.write("ประเมินราคาที่พักอาศัยในกรุงเทพฯ ด้วย AI")
st.divider()

# 2. รับค่าจากผู้ใช้ (Input Form)
col1, col2 = st.columns(2)

with col1:
    # ดึงรายชื่อทำเลมาจากข้อมูลจริง (หรือ Hardcode เอาไว้ก็ได้)
    locations = ['Sukhumvit', 'Ladprao', 'Siam', 'Sathorn', 'Ratchada', 
                 'Silom', 'Phrom Phong', 'Thonglor', 'Ari', 'Ekkamai']
    selected_location = st.selectbox("ทำเล (Location)", locations)
    
    prop_types = ['Condo', 'House', 'Apartment']
    selected_type = st.selectbox("ประเภท (Property Type)", prop_types)

with col2:
    area = st.number_input("พื้นที่ใช้สอย (sq. ft.)", min_value=100, max_value=5000, value=500)
    bedrooms = st.slider("จำนวนห้องนอน", 1, 10, 1)
    bathrooms = st.slider("จำนวนห้องน้ำ", 1, 5, 1)

# 3. เตรียมข้อมูลสำหรับทำนาย
# สำคัญมาก! ต้องสร้าง DataFrame ที่มีชื่อคอลัมน์ตรงกับตอนเทรนเป๊ะๆ
input_data = pd.DataFrame({
    'Property Type': [selected_type],
    'Location': [selected_location],
    'Area (sq. ft.)': [area],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms]
})

# 4. ปุ่มทำนาย
if st.button("💰 ประเมินราคา"):
    prediction = model.predict(input_data)
    price = prediction[0]
    
    st.success(f"ราคาประเมิน: {price:,.2f} บาท")
    
    # แสดงข้อมูลสรุป
    st.info(f"สเปค: {selected_type} ย่าน {selected_location}, ขนาด {area} ตร.ฟุต")