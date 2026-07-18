import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import get_resnet18, get_embedding_extractor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Land-Use Change Detector", layout="wide")

@st.cache_resource
def load_models():
    # Assume model is trained on 10 EuroSAT classes
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
               
    model = get_resnet18(num_classes=10)
    if os.path.exists('resnet18_finetuned.pt'):
        model.load_state_dict(torch.load('resnet18_finetuned.pt', map_location='cpu'))
    model.eval()
    
    extractor = get_embedding_extractor(model)
    extractor.eval()
    
    return model, extractor, classes

def process_image(img):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def main():
    st.title("🛰️ Satellite Image Land-Use Classifier & Change Detector")
    
    model, extractor, classes = load_models()
    
    # We use a default threshold derived from ROC curve, adjust as needed
    SIMILARITY_THRESHOLD = st.sidebar.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.85, 0.01)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("T1 (Before)")
        file1 = st.file_uploader("Upload T1 Image", type=["jpg", "png", "jpeg"], key="t1")
        if file1:
            img1 = Image.open(file1).convert('RGB')
            st.image(img1, caption="T1 Image", use_column_width=True)
            
    with col2:
        st.header("T2 (After)")
        file2 = st.file_uploader("Upload T2 Image", type=["jpg", "png", "jpeg"], key="t2")
        if file2:
            img2 = Image.open(file2).convert('RGB')
            st.image(img2, caption="T2 Image", use_column_width=True)
            
    if file1 and file2:
        st.markdown("---")
        st.header("Analysis Results")
        
        # Process images
        tensor1 = process_image(img1)
        tensor2 = process_image(img2)
        
        with torch.no_grad():
            # Classification
            out1 = model(tensor1)
            out2 = model(tensor2)
            
            prob1 = torch.nn.functional.softmax(out1, dim=1)[0]
            prob2 = torch.nn.functional.softmax(out2, dim=1)[0]
            
            class_idx1 = torch.argmax(prob1).item()
            class_idx2 = torch.argmax(prob2).item()
            
            # Embeddings & Similarity
            emb1 = extractor(tensor1)
            emb2 = extractor(tensor2)
            
            emb1 = nn.AdaptiveAvgPool2d((1, 1))(emb1).flatten(1)
            emb2 = nn.AdaptiveAvgPool2d((1, 1))(emb2).flatten(1)
            
            cos = nn.CosineSimilarity(dim=1)
            sim_score = cos(emb1, emb2).item()
            
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.subheader("T1 Prediction")
            st.write(f"**Class:** {classes[class_idx1]}")
            st.write(f"**Confidence:** {prob1[class_idx1]:.2%}")
            
        with col_res2:
            st.subheader("T2 Prediction")
            st.write(f"**Class:** {classes[class_idx2]}")
            st.write(f"**Confidence:** {prob2[class_idx2]:.2%}")
            
        with col_res3:
            st.subheader("Change Detection")
            st.write(f"**Similarity Score:** {sim_score:.4f}")
            is_changed = sim_score < SIMILARITY_THRESHOLD
            if is_changed:
                st.error("🚨 CHANGE DETECTED")
            else:
                st.success("✅ NO SIGNIFICANT CHANGE")
                
        # Heatmap visualization
        st.subheader("Change Heatmap")
        fig, ax = plt.subplots(figsize=(6, 2))
        
        # We simulate a 1x1 heatmap since we're only comparing single tiles
        heatmap_val = 1 if is_changed else 0
        heatmap_data = np.array([[heatmap_val]])
        
        sns.heatmap(heatmap_data, ax=ax, cmap='coolwarm', cbar=False, 
                   annot=np.array([["Changed" if is_changed else "Unchanged"]]), fmt='',
                   annot_kws={"size": 16}, xticklabels=False, yticklabels=False)
                   
        st.pyplot(fig)

if __name__ == '__main__':
    main()
