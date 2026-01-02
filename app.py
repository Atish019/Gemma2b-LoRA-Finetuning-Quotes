import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Streamlit page setup
st.set_page_config(
    page_title="Gemma Quote Generator", 
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced 3D Background with Particles Animation
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap');
    
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Poppins', sans-serif;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    /* Floating particles background */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    
    .particle {
        position: absolute;
        background: rgba(138, 43, 226, 0.3);
        border-radius: 50%;
        animation: float 6s infinite ease-in-out;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container with glass morphism */
    .main-container {
        position: relative;
        z-index: 1;
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3rem;
        box-shadow: 
            0 8px 32px 0 rgba(138, 43, 226, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        animation: cardEntrance 1s ease-out;
    }
    
    @keyframes cardEntrance {
        from {
            opacity: 0;
            transform: translateY(50px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    /* Card glow effect */
    .glass-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        border-radius: 26px;
        z-index: -1;
        animation: rotate 4s linear infinite;
        opacity: 0.3;
        filter: blur(10px);
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Title styling with gradient text */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem !important;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #ffd89b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem !important;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 30px rgba(245, 87, 108, 0.7)); }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        font-size: 1.2rem !important;
        padding: 1.2rem 1.5rem !important;
        transition: all 0.4s ease !important;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.4) !important;
        background: rgba(255, 255, 255, 0.12) !important;
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
        font-style: italic;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.2rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        width: 100% !important;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Quote output styling */
    .quote-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        animation: quoteAppear 0.8s ease-out;
    }
    
    @keyframes quoteAppear {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .quote-text {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        line-height: 1.8;
        color: #ffffff;
        text-align: center;
        font-style: italic;
        margin: 0;
        position: relative;
    }
    
    .quote-text::before, .quote-text::after {
        content: '"';
        font-size: 4rem;
        color: rgba(102, 126, 234, 0.5);
        position: absolute;
        font-family: serif;
    }
    
    .quote-text::before {
        top: -1rem;
        left: -1rem;
    }
    
    .quote-text::after {
        bottom: -2rem;
        right: -1rem;
    }
    
    /* Loading animation */
    .loading-dots {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 2rem 0;
    }
    
    .dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        animation: bounce 1.4s infinite ease-in-out;
    }
    
    .dot:nth-child(1) { animation-delay: -0.32s; }
    .dot:nth-child(2) { animation-delay: -0.16s; }
    .dot:nth-child(3) { animation-delay: 0s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        flex: 1;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>

<!-- Floating Particles -->
<div class="particles" id="particles"></div>

<script>
// Create floating particles
function createParticles() {
    const particles = document.getElementById('particles');
    const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c'];
    
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random properties
        const size = Math.random() * 6 + 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const left = Math.random() * 100;
        const top = Math.random() * 100;
        const delay = Math.random() * 5;
        const duration = Math.random() * 10 + 5;
        
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.background = color;
        particle.style.left = `${left}%`;
        particle.style.top = `${top}%`;
        particle.style.animationDelay = `${delay}s`;
        particle.style.animationDuration = `${duration}s`;
        
        particles.appendChild(particle);
    }
}

createParticles();
</script>
""", unsafe_allow_html=True)

# JavaScript for particles (already included in CSS)

# Main content
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-title">Gemma Quote Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform your thoughts into inspiring AI-generated quotes</p>', unsafe_allow_html=True)

# Stats cards
st.markdown("""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-number">∞</div>
        <div class="stat-label">Creative Possibilities</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">AI</div>
        <div class="stat-label">Powered</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">💬</div>
        <div class="stat-label">Quotes Generated</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main glass card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

# Model loading section
MODEL_PATH = os.path.join("outputs")
HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=80, temperature=0.8, top_p=0.9)
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model with custom loading animation
if 'model_loaded' not in st.session_state:
    st.markdown('<div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)
    st.info("🔄 Loading AI model...")
    pipe = load_model()
    if pipe:
        st.session_state.model_loaded = True
        st.session_state.pipe = pipe
        st.success("✨ AI Model Ready! Start generating inspirational quotes.")
    else:
        st.error("❌ Failed to load model")
else:
    pipe = st.session_state.pipe

# Input section
if pipe:
    prompt = st.text_input(
        "✨ What's on your mind?",
        placeholder="Type the beginning of your quote here... (e.g., 'The secret of life is...', 'True happiness comes from...')",
        label_visibility="collapsed"
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button("🚀 Generate Magical Quote", use_container_width=True)
    
    # Generate output
    if generate_btn and prompt.strip():
        with st.spinner("🎨 Creating your inspirational quote..."):
            try:
                output = pipe(prompt)[0]["generated_text"]
                st.markdown(f'''
                <div class="quote-container">
                    <p class="quote-text">{output}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Copy to clipboard button
                st.code(output, language="text")
                
            except Exception as e:
                st.error(f"Error generating quote: {str(e)}")
    elif generate_btn:
        st.warning("💡 Please enter some text to generate a quote")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 4rem; color: rgba(255, 255, 255, 0.5); font-size: 0.9rem; padding: 2rem;">
    <div style="margin-bottom: 1rem;">
        ✨ Powered by Gemma-2B • Fine-tuned for Creative Quote Generation ✨
    </div>
    <div style="font-size: 0.8rem; opacity: 0.7;">
        Transform your thoughts into wisdom with AI magic
    </div>
</div>
""", unsafe_allow_html=True)
