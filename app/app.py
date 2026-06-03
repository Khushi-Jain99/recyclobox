import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import yaml

# Append the src directory to Python path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import WasteClassifier
from preprocessor import preprocess_frame

# Load configurations
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Disposal Guide Database
DISPOSAL_GUIDES = {
    'plastic': {
        'status': 'Recyclable',
        'color': '#3B82F6', # Blue
        'bg_color': 'rgba(59, 130, 246, 0.1)',
        'border_color': 'rgba(59, 130, 246, 0.3)',
        'bin_color': 'Blue Bin',
        'icon': '🥤',
        'guidelines': [
            "Rinse plastic bottles and containers to remove food residue.",
            "Remove caps and lids and place them separately in the bin.",
            "Compress bottles/jugs to save space in the bin.",
            "Only standard plastics (PET #1, HDPE #2) are widely accepted."
        ]
    },
    'paper': {
        'status': 'Recyclable',
        'color': '#10B981', # Green
        'bg_color': 'rgba(16, 185, 129, 0.1)',
        'border_color': 'rgba(16, 185, 129, 0.3)',
        'bin_color': 'Paper/Cardboard Bin',
        'icon': '📄',
        'guidelines': [
            "Keep paper clean and dry. Wet paper breaks down and clogs sorting machines.",
            "Flatten all cardboard boxes to maximize bin space.",
            "Remove plastic sleeves, excessive tape, and bubble wrap from packages.",
            "Do not recycle heavily soiled paper (e.g., greasy pizza boxes) - compost instead."
        ]
    },
    'glass': {
        'status': 'Recyclable',
        'color': '#06B6D4', # Cyan
        'bg_color': 'rgba(6, 182, 212, 0.1)',
        'border_color': 'rgba(6, 182, 212, 0.3)',
        'bin_color': 'Glass Bin',
        'icon': '🍾',
        'guidelines': [
            "Empty and rinse all glass jars and bottles completely.",
            "Remove metal caps and lids (recycle them with metals).",
            "Labels on bottles do not need to be removed.",
            "Do not mix mirrors, windows, or drinking glasses as they melt at different temperatures."
        ]
    },
    'metal': {
        'status': 'Recyclable',
        'color': '#84CC16', # Lime
        'bg_color': 'rgba(132, 204, 22, 0.1)',
        'border_color': 'rgba(132, 204, 22, 0.3)',
        'bin_color': 'Metal Bin',
        'icon': '🥫',
        'guidelines': [
            "Empty and rinse aluminum cans, tin cans, and steel food containers.",
            "Crush beverage cans to save bin volume.",
            "Clean aluminum foil can be rolled into a ball and recycled.",
            "Empty aerosol cans are recyclable in many areas (verify locally)."
        ]
    },
    'organic': {
        'status': 'Compostable',
        'color': '#8B5CF6', # Violet/Purple
        'bg_color': 'rgba(139, 92, 246, 0.1)',
        'border_color': 'rgba(139, 92, 246, 0.3)',
        'bin_color': 'Organic/Compost Bin',
        'icon': '🍎',
        'guidelines': [
            "Ideal for kitchen scraps (fruits, vegetables, coffee grounds, eggshells).",
            "Include yard waste like dry leaves, twigs, and grass clippings.",
            "Avoid placing meat, bones, dairy, fats, or pet waste in standard compost bins.",
            "Helps reduce methane emission from landfills by returning nutrients to the soil."
        ]
    },
    'hazardous': {
        'status': 'Special Hazard Disposal',
        'color': '#EF4444', # Red
        'bg_color': 'rgba(239, 68, 68, 0.1)',
        'border_color': 'rgba(239, 68, 68, 0.3)',
        'bin_color': 'Hazardous Waste Facility',
        'icon': '⚠️',
        'guidelines': [
            "DO NOT place batteries, electronics, bulbs, paints, or pesticides in regular bins.",
            "These items contain mercury, lead, or toxic chemicals that pollute groundwater.",
            "Store in a cool, dry place until you can drop them off.",
            "Take items to local e-waste collection points or specialized hazardous waste depots."
        ]
    },
    'non_recyclable': {
        'status': 'Landfill',
        'color': '#6B7280', # Gray
        'bg_color': 'rgba(107, 114, 128, 0.1)',
        'border_color': 'rgba(107, 114, 128, 0.3)',
        'bin_color': 'Landfill/Trash Bin',
        'icon': '🗑️',
        'guidelines': [
            "Includes diapers, sanitary products, ceramic dishes, broken mirrors, and styrofoam.",
            "Place in your standard gray/black landfill trash bin.",
            "Try to minimize the use of these materials by choosing reusable alternatives.",
            "Ensure trash bags are securely tied before throwing them in the collection cart."
        ]
    }
}

# Page configuration with custom title and styling
st.set_page_config(
    page_title="RecycloBox AI - Smart Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Theme using Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title text styling */
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        font-size: 1.2rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Style for Glassmorphism effect */
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    /* Sidebar styling refinement */
    .css-1633q7x {
        background-color: #0F172A;
    }
    
    .stProgress > div > div > div > div {
        background-color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="title-text">RecycloBox AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Intelligent Deep Learning Waste Classification & Disposal Guide</p>', unsafe_allow_html=True)

# Initialize Classifier
@st.cache_resource
def get_classifier():
    return WasteClassifier()

classifier = get_classifier()

# Sidebar Setup
st.sidebar.markdown("### ⚙️ Control Center")
demo_mode = st.sidebar.toggle("🌐 Demo / Simulation Mode", value=not os.path.exists(config['model']['weights_path']), 
                              help="Toggle mock inference if the model weights are not trained yet.")

if not os.path.exists(config['model']['weights_path']) and not demo_mode:
    st.sidebar.warning("⚠️ Trained model not found at `models/recyclobox_model.keras`. Enable Demo Mode to preview.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Details")
st.sidebar.markdown("""
- **Classes**: 7 Categories
- **Total Dataset Size**: 2,902 Images
- **Training Set**: 2,030 Images
- **Validation Set**: 433 Images
- **Test Set**: 439 Images
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Developer Training Command")
st.sidebar.code("python src/train.py", language="bash")
st.sidebar.info("Run this command in your terminal to train the deep learning model on your CPU/GPU.")

# Main Dashboard Layout with Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Input Media")
    
    # Input tabs: File Upload or Camera Capture
    input_tab1, input_tab2 = st.tabs(["📤 File Upload", "📷 Live Camera"])
    
    input_image = None
    
    with input_tab1:
        uploaded_file = st.file_uploader("Drag and drop an image of waste here...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)
            
    with input_tab2:
        camera_image = st.camera_input("Take a photo of the waste item:")
        if camera_image is not None:
            input_image = Image.open(camera_image)
            st.image(input_image, caption="Camera Capture", use_container_width=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# Run Inference or Simulation
if input_image is not None:
    # Convert PIL Image to OpenCV (BGR) format
    cv_image = np.array(input_image)
    if len(cv_image.shape) == 3: # Color image
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    prediction_result = None
    
    if demo_mode:
        # Simulate classification for demo purposes
        # Choose a class based on simple image metrics or randomly for the interface demonstration
        np.random.seed(sum(cv_image.flatten()) % 10000) # Semi-deterministic mock predictions based on image contents
        mock_idx = np.random.randint(0, len(config['dataset']['classes']))
        mock_class = config['dataset']['classes'][mock_idx]
        
        # Build mock probabilities
        mock_probs = np.random.dirichlet(np.ones(len(config['dataset']['classes'])) * 0.2)
        mock_probs[mock_idx] = max(mock_probs[mock_idx], 0.75) # Ensure selected class is highly confident
        mock_probs /= sum(mock_probs)
        
        prediction_result = {
            "class": mock_class,
            "confidence": float(mock_probs[mock_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(config['dataset']['classes'], mock_probs)}
        }
    else:
        # Run real model inference
        with st.spinner("Processing image through RecycloBox AI..."):
            prediction_result = classifier.predict_frame(cv_image)
            
    # Display Results in Column 2
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        if "error" in prediction_result:
            st.error(prediction_result["error"])
        else:
            predicted_class = prediction_result["class"]
            confidence = prediction_result["confidence"]
            probs = prediction_result["probabilities"]
            
            # Fetch disposal info
            guide = DISPOSAL_GUIDES[predicted_class]
            
            # Category Badge Header
            st.markdown(f"""
            <div style="background-color: {guide['bg_color']}; border: 1px solid {guide['border_color']}; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="status-badge" style="background-color: {guide['color']}; color: #FFFFFF;">
                            {guide['status']}
                        </span>
                        <h2 style="margin: 0; color: #FFFFFF; font-size: 2.2rem; font-weight: 700;">
                            {guide['icon']} {predicted_class.replace('_', ' ').title()}
                        </h2>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.9rem; color: #94A3B8;">Confidence</span>
                        <h3 style="margin: 0; color: {guide['color']}; font-size: 1.8rem; font-weight: 700;">
                            {confidence:.1%}
                        </h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Disposal Guide Card
            st.markdown(f"### 📍 Disposal Recommendation: **{guide['bin_color']}**")
            for line in guide['guidelines']:
                st.markdown(f"- {line}")
                
            st.markdown("---")
            
            # Probabilities bars
            st.markdown("### 📊 Classification Probability Distribution")
            for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                col_name, col_bar = st.columns([1, 3])
                with col_name:
                    st.write(cls.replace('_', ' ').title())
                with col_bar:
                    # Choose bar color matching classes
                    bar_color = DISPOSAL_GUIDES[cls]['color']
                    # Streamlit progress bar is simple, so we can write inline HTML for highly customized colors
                    st.markdown(f"""
                    <div style="background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; width: 100%; height: 20px; overflow: hidden; display: flex; align-items: center; margin-top: 4px;">
                        <div style="background-color: {bar_color}; width: {p * 100:.1f}%; height: 100%; border-radius: 10px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px;">
                            <span style="font-size: 0.75rem; font-weight: 700; color: #000000; display: {'block' if p > 0.1 else 'none'};">{p:.1%}</span>
                        </div>
                        <span style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; margin-left: 8px; display: {'block' if p <= 0.1 else 'none'};">{p:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with col2:
        st.markdown('<div class="premium-card" style="text-align: center; padding: 80px 40px;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 4rem; margin-bottom: 20px;">♻️</div>
        <h3 style="color: #FFFFFF; font-weight: 600;">Awaiting Waste Media</h3>
        <p style="color: #94A3B8; font-size: 1rem;">Upload a image from your file system or take a photo using your camera to identify the category and get disposal advice.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
