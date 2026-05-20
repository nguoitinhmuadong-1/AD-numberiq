# app.py
# Chạy local: streamlit run app.py
# Deploy Streamlit Cloud:
# - Main file path: app.py
# - Python: 3.11
# - requirements.txt nên có:
#   streamlit==1.40.2
#   streamlit-drawable-canvas==0.9.3
#   tensorflow-cpu==2.15.0
#   opencv-python-headless==4.9.0.80
#   numpy==1.26.4
#   pillow==10.4.0

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# =========================
# LOGO APP
# =========================
# Đặt logo của anh trong cùng thư mục với app.py và đặt tên là logo.png.
# Logo chỉ dùng làm icon tab / app icon, không hiển thị trong giao diện chính.
LOGO_PATH = "logo.png"

if os.path.exists(LOGO_PATH):
    logo = Image.open(LOGO_PATH)
    page_icon = logo
else:
    logo = None
    page_icon = "🤖"

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="AD Numeriq",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CSS GIAO DIỆN
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&display=swap');

* {
    font-family: 'Baloo 2', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(180, 90, 255, 0.45), transparent 28%),
        radial-gradient(circle at 80% 25%, rgba(60, 180, 255, 0.35), transparent 28%),
        radial-gradient(circle at 85% 75%, rgba(255, 120, 220, 0.35), transparent 30%),
        linear-gradient(135deg, #111344 0%, #17194d 48%, #25124e 100%);
    color: white;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.block-container {
    max-width: 1250px;
    padding-top: 20px;
}

h1, h2, h3, h4, p, label {
    color: white !important;
}

div[data-testid="stMarkdownContainer"] {
    color: white;
}

.big-title {
    font-size: 76px;
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: -3px;
}

.gradient-text {
    background: linear-gradient(90deg, #8fe3ff, #ffc8dd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.robot {
    font-size: 150px;
    text-align: center;
    filter: drop-shadow(0 25px 30px rgba(120, 180, 255, 0.45));
    animation: floatBot 3.6s ease-in-out infinite;
}

@keyframes floatBot {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-16px); }
}

.result-number {
    font-size: 120px;
    text-align: center;
    line-height: 1;
    font-weight: 900;
    background: linear-gradient(135deg, #8fe3ff, #ff9ad5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.confidence-text {
    font-size: 28px;
    text-align: center;
    font-weight: 900;
    color: #caffbf;
}

/* Nút bấm rõ chữ */
.stButton > button {
    border: 0 !important;
    color: #111827 !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    padding: 14px 30px !important;
    border-radius: 999px !important;
    background: linear-gradient(90deg, #ffffff, #dff3ff, #ffc8dd) !important;
    box-shadow: 0 18px 45px rgba(158, 214, 255, 0.38) !important;
    text-shadow: none !important;
}

.stButton > button p {
    color: #111827 !important;
    font-weight: 900 !important;
}

.stButton > button:hover {
    color: #000000 !important;
    background: linear-gradient(90deg, #bde0fe, #ffc8dd, #ffffff) !important;
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 22px 60px rgba(255, 200, 221, 0.45) !important;
}

.stButton > button:hover p {
    color: #000000 !important;
}

button[kind="secondary"] {
    color: #111827 !important;
    font-weight: 900 !important;
}

button[kind="secondary"] p {
    color: #111827 !important;
    font-weight: 900 !important;
}

canvas {
    border-radius: 26px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}

hr {
    border-color: rgba(255,255,255,0.13);
}

/* Làm sáng khu vực upload file */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 12px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #ffffff !important;
    font-weight: 800 !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #ffffff, #bde0fe, #ffc8dd) !important;
    color: #111827 !important;
    font-weight: 900 !important;
    border: none !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #111827 !important;
    font-weight: 900 !important;
}

[data-testid="stFileUploader"] section div {
    color: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL ANN
# =========================
@st.cache_resource
def load_model():
    """
    Ưu tiên load train.keras vì ổn định hơn khi deploy.
    Nếu không có train.keras thì mới thử train.h5.
    """
    if os.path.exists("train.keras"):
        return tf.keras.models.load_model("train.keras", compile=False)

    if os.path.exists("train.h5"):
        return tf.keras.models.load_model("train.h5", compile=False)

    st.error("Không tìm thấy model. Hãy upload file train.keras hoặc train.h5 lên GitHub.")
    st.stop()

model = load_model()

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "draw_mode" not in st.session_state:
    st.session_state.draw_mode = "draw"

# =========================
# XỬ LÝ ẢNH THEO LOGIC MNIST
# =========================
def center_and_resize(thresh):
    coords = cv2.findNonZero(thresh)

    if coords is None:
        img_28 = np.zeros((28, 28), dtype=np.uint8)
    else:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]

        target_size = 20

        if w > h:
            new_w = target_size
            new_h = max(1, int(h * target_size / w))
        else:
            new_h = target_size
            new_w = max(1, int(w * target_size / h))

        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_28 = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        img_28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

        # Căn tâm giống MNIST hơn
        M = cv2.moments(img_28)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            matrix = np.float32([
                [1, 0, 14 - cx],
                [0, 1, 14 - cy]
            ])

            img_28 = cv2.warpAffine(
                img_28,
                matrix,
                (28, 28),
                borderValue=0
            )

    img_ready = img_28.reshape((1, 28 * 28)).astype("float32") / 255.0
    return img_28, img_ready


def preprocess_canvas(image_rgba):
    img = image_rgba.astype(np.uint8)
    rgb = img[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Canvas nền đen, nét trắng
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    return center_and_resize(thresh)


def preprocess_uploaded_image(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Nếu ảnh nền trắng chữ đen thì đảo màu
    if np.mean(gray) > 127:
        gray = 255 - gray

    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    return center_and_resize(thresh)


def predict_digit(img_ready):
    preds = model.predict(img_ready, verbose=0)[0]
    digit = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    return digit, confidence, preds

# =========================
# COMPONENT: TÊN APP
# =========================
def show_app_title(title="AD Numeriq"):
    # Giữ bot cũ trong giao diện, logo chỉ làm icon tab / icon iPhone.
    st.markdown(f"## 🤖 {title}")

# =========================
# TRANG 1: HOME
# =========================
def home_page():
    nav1, nav2 = st.columns([0.55, 0.45])

    with nav1:
        show_app_title("AD Numeriq")

    with nav2:
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown("**Home**")
        m2.markdown("**Scanner**")
        m3.markdown("**Result**")
        m4.markdown("**MNIST**")

    st.markdown("---")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("#### ✨ AI Handwriting Recognition")

        st.markdown(
            """
            <div class="big-title">
                Make your<br>
                digit scan<br>
                <span class="gradient-text">awesome.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            Vẽ một chữ số từ **0 đến 9**. AD Numeriq sẽ xử lý ảnh theo logic MNIST:
            **grayscale → resize 28x28 → căn giữa → normalize → ANN predict**.
            """
        )

        st.write("")

        b1, b2 = st.columns([0.32, 0.68])

        with b1:
            if st.button("Get started →", use_container_width=True):
                st.session_state.page = "scanner"
                st.rerun()

        with b2:
            st.info("💡 Vẽ số to, nét dày, nằm giữa khung để AI đoán chính xác hơn.")

        st.write("")

        f1, f2, f3 = st.columns(3)

        with f1:
            st.success("🧠 ANN Model")

        with f2:
            st.info("🖼️ 28x28 Input")

        with f3:
            st.warning("⚡ Fast Predict")

    with right:
        # Giữ nguyên con bot cũ trong giao diện chính
        st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)

        st.markdown("## Hello!")
        st.markdown("### How may I help you today?")

        st.success("Input: Draw digit")
        st.info("Output: AI Prediction")
        st.warning("Mode: MNIST Scanner")

    st.markdown("---")

    st.markdown("## 🚀 How it works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### ✍️ 1. Draw")
        st.write("Người dùng vẽ số vào bảng vẽ.")

    with step2:
        st.markdown("### 🧹 2. Process")
        st.write("Ảnh được chuyển về grayscale và resize 28x28.")

    with step3:
        st.markdown("### 🤖 3. Predict")
        st.write("Model ANN dự đoán chữ số.")

    with step4:
        st.markdown("### 📊 4. Result")
        st.write("Hiển thị số dự đoán và độ tin cậy.")

# =========================
# TRANG 2: SCANNER
# =========================
def scanner_page():
    nav1, nav2, nav3 = st.columns([0.55, 0.25, 0.20])

    with nav1:
        show_app_title("AD Numeriq Scanner")

    with nav2:
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Draw**")
        c2.markdown("**Upload**")
        c3.markdown("**Predict**")

    with nav3:
        if st.button("← Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("---")

    st.markdown("# 🤖 AI Scanner Room")
    st.markdown("Vẽ số vào khung hoặc upload ảnh từ máy tính để AD Numeriq dự đoán.")

    col1, col2 = st.columns([1.05, 0.95], gap="large")

    with col1:
        st.markdown("### ✍️ Vẽ số tại đây")

        # Chọn màu nét theo chế độ hiện tại
        if st.session_state.draw_mode == "draw":
            stroke_color = "#FFFFFF"
            current_mode = "✏️ Đang vẽ"
        else:
            stroke_color = "#000000"
            current_mode = "🧽 Đang tẩy"

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 1)",
            stroke_width=25,
            stroke_color=stroke_color,
            background_color="#000000",
            width=430,
            height=430,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        # Một dòng phím chức năng dưới bảng vẽ
        btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1.4])

        with btn1:
            if st.button("✏️ Vẽ", use_container_width=True):
                st.session_state.draw_mode = "draw"
                st.rerun()

        with btn2:
            if st.button("🧽 Tẩy", use_container_width=True):
                st.session_state.draw_mode = "erase"
                st.rerun()

        with btn3:
            if st.button("🧹 Xóa", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.draw_mode = "draw"
                st.rerun()

        with btn4:
            st.success(current_mode)

        st.info("Nền đen + nét trắng sẽ giống dữ liệu MNIST hơn. Bạn vẽ số càng to và rõ thì AI càng dễ đoán.")

    with col2:
        st.markdown("### ✨ Kết quả AI")

        if canvas_result.image_data is not None:
            processed_28, img_ready = preprocess_canvas(canvas_result.image_data)

            if np.sum(processed_28) == 0:
                st.info("Bạn hãy vẽ một số vào khung trước nhé 😄")
                st.markdown('<div class="robot">🤖</div>', unsafe_allow_html=True)
            else:
                digit, confidence, preds = predict_digit(img_ready)

                st.markdown("#### AD Numeriq dự đoán là")

                st.markdown(
                    f'<div class="result-number">{digit}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence-text">Độ tin cậy: {confidence:.2f}%</div>',
                    unsafe_allow_html=True
                )

                st.write("")

                p1, p2 = st.columns([0.42, 0.58])

                with p1:
                    st.image(processed_28, caption="Ảnh 28x28", width=160)

                with p2:
                    with st.expander("📊 Xác suất từng số"):
                        for i, p in enumerate(preds):
                            st.progress(float(p), text=f"Số {i}: {p*100:.2f}%")

    st.markdown("---")

    st.markdown("# 🖼️ Upload Image")
    st.markdown("Hoặc bạn có thể upload ảnh số viết tay từ máy tính.")

    uploaded_file = st.file_uploader(
        "Chọn ảnh PNG/JPG có số viết tay",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        img_28, img_ready = preprocess_uploaded_image(pil_img)
        digit, confidence, preds = predict_digit(img_ready)

        u1, u2, u3 = st.columns(3)

        with u1:
            st.image(pil_img, caption="Ảnh gốc", use_container_width=True)

        with u2:
            st.image(img_28, caption="Sau xử lý 28x28", width=180)

        with u3:
            st.markdown("#### AD Numeriq dự đoán")
            st.markdown(
                f'<div class="result-number">{digit}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="confidence-text">{confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

# =========================
# RUN APP
# =========================
if st.session_state.page == "home":
    home_page()
else:
    scanner_page()
