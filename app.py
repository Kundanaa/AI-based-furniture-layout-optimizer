import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import LayoutPredictor

# Load trained model
model = LayoutPredictor()
model.load_state_dict(torch.load("layout_model.pth"))
model.eval()

def visualize_layout(room_dims, positions):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, room_dims[0])
    ax.set_ylim(0, room_dims[1])
    
    # Furniture types with their proportional dimensions
    furniture_names = ["Bed", "Desk", "Wardrobe", "Chair", "Sofa", "Table"]
    furniture_sizes = [
        (0.3 * room_dims[0], 0.4 * room_dims[1]),  # Bed
        (0.2 * room_dims[0], 0.2 * room_dims[1]),  # Desk
        (0.15 * room_dims[0], 0.3 * room_dims[1]), # Wardrobe
        (0.1 * room_dims[0], 0.1 * room_dims[1]),  # Chair
        (0.25 * room_dims[0], 0.2 * room_dims[1]), # Sofa
        (0.2 * room_dims[0], 0.15 * room_dims[1])  # Table
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5']
    
    # Only show requested number of items
    for i in range(min(len(positions), len(furniture_sizes))):
        x, y = positions[i]
        width, height = furniture_sizes[i]
        
        # Convert normalized coordinates to actual dimensions
        x_pos = x * room_dims[0]
        y_pos = y * room_dims[1]
        
        # Draw furniture
        rect = plt.Rectangle((x_pos, y_pos), width, height, 
                            color=colors[i], alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x_pos + width/2, y_pos + height/2, furniture_names[i], 
                ha='center', va='center', fontsize=10)
    
    # Add grid for better visualization
    ax.grid(linestyle='--', alpha=0.6)
    ax.set_title(f"Room Layout ({room_dims[0]}m × {room_dims[1]}m)")
    
    st.pyplot(fig)

# Streamlit UI
st.title("Furniture Layout Optimizer")
st.write("Generate optimized furniture arrangements based on room dimensions")

room_width = st.slider("Room Width (meters)", min_value=3.0, max_value=8.0, value=5.0, step=0.1)
room_height = st.slider("Room Height (meters)", min_value=3.0, max_value=6.0, value=4.0, step=0.1)
num_items = st.slider("Number of Items", min_value=3, max_value=6, value=6)
# room_type = st.selectbox("Room Type", ["Bedroom", "Office", "Living Room"])
# room_type_num = ["Bedroom", "Office", "Living Room"].index(room_type)
room_type_num=0
if st.button("Generate Layout"):
    # Create input with all 4 required features
    input_tensor = torch.FloatTensor([[room_width, room_height, num_items, room_type_num]])
    
    with torch.no_grad():
        output_tensor = model(input_tensor).detach().numpy()
    
    # Reshape to get furniture positions (already normalized 0-1)
    positions = output_tensor.reshape(-1, 2)[:num_items]
    
    # Visualize with proper furniture dimensions
    visualize_layout([room_width, room_height], positions)
    
    # Show furniture dimensions
    st.write("### Furniture Dimensions")
    furniture_data = [
        ("Bed", f"{0.3 * room_width:.1f}m × {0.4 * room_height:.1f}m"),
        ("Desk", f"{0.2 * room_width:.1f}m × {0.2 * room_height:.1f}m"),
        ("Wardrobe", f"{0.15 * room_width:.1f}m × {0.3 * room_height:.1f}m"),
        ("Chair", f"{0.1 * room_width:.1f}m × {0.1 * room_height:.1f}m"),
        ("Sofa", f"{0.25 * room_width:.1f}m × {0.2 * room_height:.1f}m"),
        ("Table", f"{0.2 * room_width:.1f}m × {0.15 * room_height:.1f}m")
    ]
    
    for i, (item, size) in enumerate(furniture_data[:num_items]):
        st.write(f"- {item}: {size}")
