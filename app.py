# app.py — Streamlit Toxicity Checker (lazy load + debug)
import os, json, numpy as np, pandas as pd, torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====== CONFIG ======
MODEL_DIR = "NLP_toxic"   # <-- SỬA ĐƯỜNG DẪN NÀY (thư mục chứa config.json, pytorch_model.bin)
MAX_LEN   = 192

st.set_page_config(page_title="Toxicity Checker", page_icon="🤖", layout="centered")
st.title("🤖 Toxicity Checker")
st.caption("Nhập câu, bấm **Kiểm tra**. App sẽ phân loại Toxic/Non-Toxic. (Model load theo yêu cầu)")

# Cho xem debug ở sidebar
with st.sidebar:
    st.header("⚙️ Debug info")
    st.write("**Model dir:**", os.path.abspath(MODEL_DIR))
    st.write("**Files in model dir:**")
    if os.path.isdir(MODEL_DIR):
        st.code("\n".join(sorted(os.listdir(MODEL_DIR))), language="text")
    else:
        st.error("❌ MODEL_DIR không tồn tại!")

# ====== SESSION STATE ======
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "tok" not in st.session_state:
    st.session_state.tok = None
if "model" not in st.session_state:
    st.session_state.model = None
if "device" not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "labels" not in st.session_state:
    st.session_state.labels = ["toxic"]     # sẽ cập nhật sau khi load
if "thresholds" not in st.session_state:
    st.session_state.thresholds = np.array([0.5], dtype=float)
if "num_labels" not in st.session_state:
    st.session_state.num_labels = 1

def load_model_safely():
    """Nạp model & tokenizer, có thông báo lỗi rõ ràng."""
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(st.session_state.device).eval()
    except Exception as e:
        st.error(f"❌ Không thể nạp model/tokenizer từ `{MODEL_DIR}`.\n\n{e}")
        return

    # labels & thresholds
    default_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    num_labels = int(model.config.num_labels)
    labels = default_labels[:num_labels] if num_labels > 1 else ["toxic"]

    thr_path = os.path.join(MODEL_DIR, "thresholds.json")
    if os.path.exists(thr_path):
        try:
            with open(thr_path, "r") as f:
                th_dict = json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Đọc thresholds.json lỗi, dùng 0.5. Chi tiết: {e}")
            th_dict = {lab: 0.5 for lab in labels}
    else:
        st.info("ℹ️ Không thấy thresholds.json, dùng threshold = 0.5 cho tất cả nhãn.")
        th_dict = {lab: 0.5 for lab in labels}
    thresholds = np.array([th_dict.get(l, 0.5) for l in labels], dtype=float)

    # set state
    st.session_state.tok = tok
    st.session_state.model = model
    st.session_state.labels = labels
    st.session_state.thresholds = thresholds
    st.session_state.num_labels = num_labels
    st.session_state.loaded = True
    st.success(f"✅ Model loaded. Labels: {labels} | Thresholds: {thresholds.tolist()}")

@torch.no_grad()
def predict_one(text: str):
    tok = st.session_state.tok
    model = st.session_state.model
    device = st.session_state.device

    enc = tok([text], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()[0]   # shape [L]
    return probs

# ====== FORM ======
with st.form("tox_form"):
    text = st.text_area("Nhập câu tiếng Anh", height=130,
                        placeholder="e.g., I hate you. You are disgusting!")
    col1, col2 = st.columns(2)
    with col1:
        btn_load = st.form_submit_button("🚀 Load model (nếu chưa)")
    with col2:
        btn_check = st.form_submit_button("🔎 Kiểm tra")

if btn_load:
    load_model_safely()

if btn_check:
    if not st.session_state.loaded:
        st.warning("Vui lòng bấm **Load model** trước.")
    elif not text.strip():
        st.warning("Vui lòng nhập câu.")
    else:
        # dự đoán
        probs = predict_one(text)
        labels = st.session_state.labels
        ths = st.session_state.thresholds
        preds = (probs >= ths).astype(int)

        if st.session_state.num_labels == 1:
            overall = bool(preds[0])
        else:
            overall = bool(preds.sum() > 0)

        # kết quả
        st.subheader("Kết quả")
        if overall:
            st.markdown('<div style="background:#FEF2F2;color:#991B1B;border:1px solid #FECACA;padding:10px 14px;border-radius:12px;font-weight:600;">Kết luận: TOXIC</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0;padding:10px 14px;border-radius:12px;font-weight:600;">Kết luận: NON-TOXIC</div>', unsafe_allow_html=True)

        # bảng xác suất
        df = pd.DataFrame({"label": labels, "probability": probs})
        st.dataframe(df.style.format({"probability": "{:.3f}"}), use_container_width=True)
        st.bar_chart(df.set_index("label"))

        st.caption("Bạn có thể thêm file thresholds.json trong thư mục model để tùy chỉnh ngưỡng theo nhãn.")

# ====== QUICK HELP ======
with st.expander("❓ Không thấy gì hiện lên? Thử kiểm tra nhanh"):
    st.markdown("""
1. Chạy đúng lệnh: `streamlit run app.py` (xem log ở terminal có báo lỗi không).
2. Đảm bảo `MODEL_DIR` là thư mục hợp lệ, có `config.json`, `pytorch_model.bin`, `tokenizer.json`/`vocab.txt`, v.v.
3. Nếu dùng Windows & trình duyệt Cốc Cốc, thử mở bằng Chrome/Edge khác hoặc tắt ad-block.
4. Không để lệnh tải model quá nặng ở đầu chương trình (bản này đã lazy load).
""")
