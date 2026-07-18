import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from models import get_resnet18, get_embedding_extractor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import random
from PIL import Image

DATA_DIR = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, '2750'), transform=transform)
    return full_dataset

def simulate_regions_and_extract_embeddings(extractor, dataset, grid_size=5, num_regions=100):
    """
    Simulates regions by creating grids of tiles.
    For each region, we create a T1 grid and a T2 grid.
    Some tiles in T2 will be changed to different classes to simulate land-cover change.
    """
    extractor.eval()
    
    # Group indices by class for easier sampling
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
        
    y_true = []
    similarities = []
    
    regions_t1 = []
    regions_t2 = []
    
    with torch.no_grad():
        for r in range(num_regions):
            t1_indices = []
            t2_indices = []
            labels = []
            
            for _ in range(grid_size * grid_size):
                base_class = random.randint(0, len(dataset.classes) - 1)
                idx1 = random.choice(class_indices[base_class])
                
                # 50% chance of change
                if random.random() < 0.5:
                    new_class = random.choice([c for c in range(len(dataset.classes)) if c != base_class])
                    idx2 = random.choice(class_indices[new_class])
                    labels.append(1) # Changed
                else:
                    idx2 = random.choice(class_indices[base_class])
                    labels.append(0) # Unchanged
                    
                t1_indices.append(idx1)
                t2_indices.append(idx2)
                
            # Extract embeddings
            t1_imgs = torch.stack([dataset[i][0] for i in t1_indices]).to(DEVICE)
            t2_imgs = torch.stack([dataset[i][0] for i in t2_indices]).to(DEVICE)
            
            emb1 = extractor(t1_imgs)
            emb2 = extractor(t2_imgs)
            
            emb1 = nn.AdaptiveAvgPool2d((1, 1))(emb1).flatten(1)
            emb2 = nn.AdaptiveAvgPool2d((1, 1))(emb2).flatten(1)
            
            # Cosine similarity
            cos = nn.CosineSimilarity(dim=1)
            sim = cos(emb1, emb2).cpu().numpy()
            
            y_true.extend(labels)
            similarities.extend(sim)
            
            regions_t1.append(t1_indices)
            regions_t2.append(t2_indices)
            
    return np.array(y_true), np.array(similarities), regions_t1, regions_t2

def plot_roc_curve(y_true, similarities):
    # For ROC, lower similarity means HIGHER probability of change
    # So we use 1 - similarity as the score
    scores = 1 - similarities
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Change Detection)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Calculate optimal threshold (Youden's J statistic)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold_score = thresholds[optimal_idx]
    optimal_similarity_threshold = 1 - optimal_threshold_score
    print(f"Optimal Cosine Similarity Threshold: {optimal_similarity_threshold:.4f}")
    return optimal_similarity_threshold

def generate_change_heatmaps(dataset, regions_t1, regions_t2, similarities, threshold, grid_size=5, num_samples=5):
    
    os.makedirs('heatmaps', exist_ok=True)
    
    # Denormalize for plotting
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # T1 Region
        t1_grid = np.zeros((grid_size*64, grid_size*64, 3))
        t2_grid = np.zeros((grid_size*64, grid_size*64, 3))
        heatmap_data = np.zeros((grid_size, grid_size))
        
        sim_start = i * (grid_size * grid_size)
        
        for r in range(grid_size):
            for c in range(grid_size):
                idx = r * grid_size + c
                img1 = inv_normalize(dataset[regions_t1[i][idx]][0]).permute(1, 2, 0).numpy()
                img2 = inv_normalize(dataset[regions_t2[i][idx]][0]).permute(1, 2, 0).numpy()
                
                img1 = np.clip(img1, 0, 1)
                img2 = np.clip(img2, 0, 1)
                
                t1_grid[r*64:(r+1)*64, c*64:(c+1)*64, :] = img1
                t2_grid[r*64:(r+1)*64, c*64:(c+1)*64, :] = img2
                
                # 1 if changed (similarity < threshold), else 0
                heatmap_data[r, c] = 1 if similarities[sim_start + idx] < threshold else 0
                
        axes[0].imshow(t1_grid)
        axes[0].set_title('T1 (Before)')
        axes[0].axis('off')
        
        axes[1].imshow(t2_grid)
        axes[1].set_title('T2 (After)')
        axes[1].axis('off')
        
        sns.heatmap(heatmap_data, ax=axes[2], cmap='coolwarm', cbar=False, linewidths=2, linecolor='white')
        axes[2].set_title('Change Heatmap (Red = Changed)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'heatmaps/change_region_{i+1}.png')
        plt.close()
        print(f"Saved heatmaps/change_region_{i+1}.png")

def main():
    print("Loading dataset...")
    dataset = load_dataset()
    
    print("Loading model and extractor...")
    model = get_resnet18(num_classes=10)
    if os.path.exists('resnet18_finetuned.pt'):
        model.load_state_dict(torch.load('resnet18_finetuned.pt', map_location=DEVICE))
    model = model.to(DEVICE)
    extractor = get_embedding_extractor(model).to(DEVICE)
    
    print("Simulating regions and extracting embeddings...")
    y_true, similarities, regions_t1, regions_t2 = simulate_regions_and_extract_embeddings(extractor, dataset)
    
    print("Plotting ROC curve and selecting threshold...")
    threshold = plot_roc_curve(y_true, similarities)
    
    print("Generating change heatmaps...")
    generate_change_heatmaps(dataset, regions_t1, regions_t2, similarities, threshold)
    print("Change detection pipeline complete.")

if __name__ == '__main__':
    main()
