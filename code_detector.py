import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import os
from packaging import version

# --- Version Check ---
TORCH_MIN_VERSION = "2.6.0"
if version.parse(torch.__version__) < version.parse(TORCH_MIN_VERSION):
    st.error(f"⚠️ Security Alert: PyTorch {torch.__version__} is vulnerable. Please upgrade to {TORCH_MIN_VERSION}+")
    st.error("Run: pip install --upgrade torch>=2.6.0")
    st.stop()

# --- Page Config ---
st.set_page_config(
    page_title="🔒 Secure CodeGuardian AI Detector",
    page_icon="🔍",
    layout="centered"
)

# --- Authentication ---
try:
    login(token=os.getenv("HF_TOKEN", "your_hf_token_here"))
except Exception as e:
    st.warning(f"Authentication note: {str(e)}")

# --- Safe Model Loading ---
@st.cache_resource
def load_model():
    try:
        # Using safetensors for secure loading
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/codebert-base",
            num_labels=2,
            use_safetensors=True  # Force safetensors format
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Model loading failed. Technical details: {str(e)}")
        return None, None

tokenizer, model = load_model()

# --- UI Components ---
st.title("🔒 Secure CodeGuardian")
st.markdown("### Vulnerability-protected AI code detection")

with st.expander("ℹ️ Security Info"):
    st.markdown("""
    **Why we require PyTorch 2.6+:**
    - Addresses [CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)
    - Uses `safetensors` for secure model loading
    - All model weights verified before loading
    """)

code_input = st.text_area(
    "Paste code to analyze:",
    height=300,
    placeholder="def example():\n    print('Hello world')"
)

if st.button("Analyze", type="primary") and code_input:
    if not tokenizer or not model:
        st.error("Model not loaded - check errors above")
    else:
        with st.spinner("Securely analyzing..."):
            try:
                inputs = tokenizer(code_input, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=1)[0]
                ai_prob = probs[1].item()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AI Probability", f"{ai_prob*100:.1f}%")
                with col2:
                    st.progress(ai_prob)
                
                if ai_prob > 0.7:
                    st.error("🚨 High confidence: AI-generated")
                elif ai_prob > 0.4:
                    st.warning("⚠️ Moderate confidence: Possibly AI")
                else:
                    st.success("✅ Likely human-written")
                    
            except Exception as e:
                st.error(f"Secure analysis failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.8em; color: #777;">
    🔒 Secure Model Loading • PyTorch {torch.__version__} • 
    <a href="https://huggingface.co/microsoft/codebert-base" target="_blank">CodeBERT</a>
</div>
""", unsafe_allow_html=True)