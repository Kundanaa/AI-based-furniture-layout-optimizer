# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from model import LayoutPredictor
# from data_generator import generate_dataset

# def calculate_loss(predictions, labels, room_dims):
#     batch_size = predictions.size(0)
#     num_items = 6
    
#     # Reshape predictions to [batch_size, items, coordinates]
#     positions = predictions.view(batch_size, num_items, 2)
    
#     # Reshape labels to [batch_size, items, coordinates]
#     labels = labels.view(batch_size, num_items * 2).view(batch_size, num_items, 2)
    
#     loss = 0.0
    
#     # Furniture dimensions (normalized)
#     furniture_sizes = torch.tensor([
#         [0.3, 0.4], [0.2, 0.2], [0.15, 0.3],
#         [0.1, 0.1], [0.25, 0.2], [0.2, 0.15]
#     ], device=predictions.device)
    
#     overlap_weight = 50.0
#     clearance_weight = 30.0
#     boundary_weight = 20.0

#     # Overlap penalty
#     for i in range(num_items):
#         for j in range(i + 1, num_items):
#             box1 = torch.cat([positions[:, i], furniture_sizes[i].repeat(batch_size)], dim=1)
#             box2 = torch.cat([positions[:, j], furniture_sizes[j].repeat(batch_size)], dim=1)
#             loss += overlap_weight * overlap_penalty(box1, box2)
    
#     # Boundary penalty
#     for i in range(num_items):
#         x_pos_scaled = positions[:, i][:batch_size]
#         x, y = positions[:, i].chunk(2, dim=1)
#         w, h = furniture_sizes[i]
#         loss += boundary_weight * torch.mean(torch.relu(-x))   # Left boundary
#         loss += boundary_weight * torch.mean(torch.relu(-y))   # Bottom boundary
#         loss += boundary_weight * torch.mean(torch.relu(x + w - room_dims[0]))  # Right boundary
#         loss += boundary_weight * torch.mean(torch.relu(y + h - room_dims[1]))  # Top boundary

#     # Wall clearance penalty
#     for i in range(num_items):
#         bounds = 0.05
#         loss += clearance_weight * torch.mean(torch.relu(bounds - positions[:, i]))
#         loss += clearance_weight * torch.mean(torch.relu(positions[:, i] + furniture_sizes[i].repeat(batch_size) - (1 - bounds)))
    
#     return loss

# def overlap_penalty(box1, box2):
#     x1, y1, w1, h1 = box1.chunk(4, dim=1)
#     x2, y2, w2, h2 = box2.chunk(4, dim=1)
    
#     overlap_x = torch.relu(torch.min(x1 + w1, x2 + w2) - torch.max(x1, x2))
#     overlap_y = torch.relu(torch.min(y1 + h1, y2 + h2) - torch.max(y1, y2))
    
#     return torch.mean(overlap_x * overlap_y)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import LayoutPredictor
from data_generator import generate_dataset

# def calculate_loss(predictions, labels, room_dims):
#     batch_size = predictions.size(0)
#     loss = 0.0
#     num_items = 6
    
#     # Reshape predictions to [batch_size, items, coordinates]
#     positions = predictions.view(batch_size, num_items, 2)
#     labels = labels.view(batch_size, num_items, 2)
    
#     # Furniture dimensions (normalized)
#     furniture_sizes = torch.tensor([
#         [0.3, 0.4],  # Bed
#         [0.2, 0.2],  # Desk
#         [0.15, 0.3], # Wardrobe
#         [0.1, 0.1],  # Chair
#         [0.25, 0.2], # Sofa
#         [0.2, 0.15]  # Table
#     ], device=predictions.device)
    
#     overlap_weight = 50.0
#     #clearance_weight = 30.0
#     boundary_weight = 20.0

#     # Overlap penalty
#     for i in range(num_items):
#         for j in range(i + 1, num_items):
#             box1 = torch.cat([positions[:, i], furniture_sizes[i].repeat(batch_size, 1)], dim=1)
#             box2 = torch.cat([positions[:, j], furniture_sizes[j].repeat(batch_size, 1)], dim=1)
#             loss += overlap_weight * overlap_penalty(box1, box2)
    
#     # Boundary penalty
#     for i in range(num_items):
#         x_pos_scaled = positions[:, i][:batch_size]
#         x, y = positions[:, i].chunk(2, dim=1)
#         w, h = furniture_sizes[i]
#         loss += boundary_weight * torch.mean(torch.relu(-x))   # Left boundary
#         loss += boundary_weight * torch.mean(torch.relu(-y))   # Bottom boundary
#         loss += boundary_weight * torch.mean(torch.relu(x + w - room_dims[0]))  # Right boundary
#         loss += boundary_weight * torch.mean(torch.relu(y + h - room_dims[1]))  # Top boundary

    # Wall clearance penalty
    # for i in range(num_items):
    #     bounds = 0.05
    #     loss += clearance_weight * torch.mean(torch.relu(bounds - positions[:, i]))
    #     loss += clearance_weight * torch.mean(torch.relu(positions[:, i] + furniture_sizes[i].repeat(batch_size, 1) - (1 - bounds)))
    
    # Functional zone penalty (e.g., bed near walls)
#     loss += functional_zone_penalty(positions, room_dims)

#     return loss

# def overlap_penalty(box1, box2):
#     x1, y1, w1, h1 = box1.chunk(4, dim=1)
#     x2, y2, w2, h2 = box2.chunk(4, dim=1)
    
#     overlap_x = torch.relu(torch.min(x1 + w1, x2 + w2) - torch.max(x1, x2))
#     overlap_y = torch.relu(torch.min(y1 + h1, y2 + h2) - torch.max(y1, y2))
    
#     return torch.mean(overlap_x * overlap_y)

# def functional_zone_penalty(positions, room_dims):
#     penalty = 0.0
#     for i in range(positions.size(1)):
#         x, y = positions[:, i].chunk(2, dim=1)
        
#         if i == 0:  # Bed should be near walls
#             penalty += torch.mean(torch.relu(x - 0.8)) + torch.mean(torch.relu(y - 0.8))
        
#         elif i == 1:  # Desk should be near windows (top-left corner)
#             penalty += torch.mean(torch.relu(0.2 - x)) + torch.mean(torch.relu(0.2 - y))
        
#         elif i == 4:  # Sofa should be near center
#             penalty += torch.mean(torch.relu(abs(x - room_dims[0] / 2))) + torch.mean(torch.relu(abs(y - room_dims[1] / 2)))
    
#     return penalty * 10  # Increase weight of functional zone penalties


def calculate_loss(predictions, labels, room_dims):
    batch_size = predictions.size(0)
    loss = 0.0
    num_items =6
    
    # Reshape predictions to [batch_size, items, coordinates]
    positions = predictions.view(batch_size, 6, 2)
    labels = labels.view(batch_size, 6, 2)
    
    # Furniture dimensions (normalized)
    furniture_sizes = torch.tensor([
        [0.3, 0.4],  # Bed
        [0.2, 0.2],  # Desk
        [0.15, 0.3], # Wardrobe
        [0.1, 0.1],  # Chair
        [0.25, 0.2], # Sofa
        [0.2, 0.15]  # Table
    ], device=predictions.device)
    
    overlap_weight = 50.0
    clearance_weight = 20.0
    #boundary_weight = 10.0

    #Overlap penalty
    for i in range(6):
        for j in range(i + 1, 6):
            box1 = torch.cat([positions[:, i], furniture_sizes[i].repeat(batch_size, 1)], dim=1)
            box2 = torch.cat([positions[:, j], furniture_sizes[j].repeat(batch_size, 1)], dim=1)
            loss += overlap_weight * overlap_penalty(box1, box2)
    
    # Wall clearance penalty
    for i in range(6):
        # x, y = positions[:, i].chunk(2, dim=1)
        # w, h = furniture_sizes[i]
        # loss += boundary_weight * torch.mean(torch.relu(-x))   # Left boundary
        # loss += boundary_weight * torch.mean(torch.relu(-y))   # Bottom boundary
        # loss += boundary_weight * torch.mean(torch.relu(x + w - room_dims[0]))  # Right boundary
        # loss += boundary_weight * torch.mean(torch.relu(y + h - room_dims[1]))  # Top boundary

    # return loss
        bounds = 0.05
        loss += clearance_weight * torch.mean(torch.relu(bounds - positions[:, i]))
        loss += clearance_weight * torch.mean(torch.relu(positions[:, i] + furniture_sizes[i].repeat(batch_size, 1) - (1 - bounds)))
    
    # Functional zone penalty (e.g., bed near walls)
    loss += functional_zone_penalty(positions, room_dims)

    return loss

def overlap_penalty(box1, box2):
    x1, y1, w1, h1 = box1.chunk(4, dim=1)
    x2, y2, w2, h2 = box2.chunk(4, dim=1)
    
    overlap_x = torch.relu(torch.min(x1 + w1, x2 + w2) - torch.max(x1, x2))
    overlap_y = torch.relu(torch.min(y1 + h1, y2 + h2) - torch.max(y1, y2))
    
    return torch.mean(overlap_x * overlap_y)

def functional_zone_penalty(positions, room_dims):
    penalty = 0.0
    for i in range(positions.size(1)):
        # x, y = positions[:, i].chunk(2, dim=1)
        # if i == 0:  # Bed should be near walls
        #     penalty += torch.mean(torch.relu(x - 0.8)) + torch.mean(torch.relu(y - 0.8))
        # elif i == 1:  # Desk should be near windows (top-left corner)
        #     penalty += torch.mean(torch.relu(0.2 - x)) + torch.mean(torch.relu(0.2 - y))
        x, y = positions[:, i].chunk(2, dim=1)
        if i == 0:  # Bed should be near walls
            penalty += torch.mean(torch.relu(x - room_dims[0] * 0.8)) + torch.mean(torch.relu(y - room_dims[1] * 0.8))
        elif i == 1:  # Desk should be near windows (top-left corner)
            penalty += torch.mean(torch.relu(0.2 * room_dims[0] - x)) + torch.mean(torch.relu(0.2 * room_dims[1] - y))
    return penalty


# Generate synthetic dataset
data, labels = generate_dataset(100)

# Normalize labels by room dimensions
normalized_labels = []
for i in range(len(data)):
    room_w, room_h, num_items, room_type = data[i]
    # Flatten (x, y) tuples into a single list of normalized coordinates
    normalized_labels.append([coord / room_w if idx % 2 == 0 else coord / room_h 
                            for idx, coord in enumerate(labels[i])])

labels_tensor = torch.tensor(normalized_labels, dtype=torch.float32)


# Convert to tensors
data_tensor = torch.tensor(data, dtype=torch.float32)

#labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 12)

dataset = TensorDataset(data_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and optimizer
model = LayoutPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    total_loss = 0
    for batch_data, batch_labels in dataloader:
        # Forward pass
        predictions = model(batch_data)
        
        # Calculate custom loss
        loss = calculate_loss(predictions, batch_labels, batch_data[:, :2])
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print progress
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "layout_model.pth")
print("Training complete! Model saved as layout_model.pth")