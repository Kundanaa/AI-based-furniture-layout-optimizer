import numpy as np

# def generate_sample():
#     room_w = np.random.uniform(3, 8)
#     room_h = np.random.uniform(3, 6)
#     num_items = 6
#     room_type = np.random.randint(0, 3) 

#     positions = []
#     furniture_sizes = [
#         (0.3 * room_w, 0.4 * room_h),  # Bed
#         (0.2 * room_w, 0.2 * room_h),  # Desk
#         (0.15 * room_w, 0.3 * room_h), # Wardrobe
#         (0.1 * room_w, 0.1 * room_h),  # Chair
#         (0.25 * room_w, 0.2 * room_h), # Sofa
#         (0.2 * room_w, 0.15 * room_h)  # Table
#     ]
    
#     for size_idx in range(num_items):
#         attempts = 50  # Maximum attempts to find a valid position
#         for _ in range(attempts):
#             x = np.random.uniform(0 + furniture_sizes[size_idx][0], room_w - furniture_sizes[size_idx][0])
#             y = np.random.uniform(0 + furniture_sizes[size_idx][1], room_h - furniture_sizes[size_idx][1])
            
#             collision = check_collision(x / room_w,
#                                         y / room_h,
#                                         positions,
#                                         furniture_sizes[size_idx])
            
#             if not collision:
#                 positions.append((x / room_w,
#                                   y / room_h,
#                                   furniture_sizes[size_idx][0] / room_w,
#                                   furniture_sizes[size_idx][1] / room_h))
#                 break
    
#     return [room_w / 8.0, room_h / 6.0, num_items / 6.0, room_type / 2.0], positions

# def check_collision(x_scaled, y_scaled, positions_scaled, size_scaled):
#     min_spacing = 0.05  # Minimum spacing between furniture items
#     for px_scaled, py_scaled, pw_scaled, ph_scaled in positions_scaled:
#         if not (x_scaled + size_scaled[0] + min_spacing < px_scaled or 
#                 x_scaled > px_scaled + pw_scaled + min_spacing or 
#                 y_scaled + size_scaled[1] + min_spacing < py_scaled or 
#                 y_scaled > py_scaled + ph_scaled + min_spacing):
#             return True  # Collision detected
#     return False


# def generate_dataset(num_samples=100):
#     data = []
#     labels = []
    
#     for _ in range(num_samples):
#         sample_data, sample_labels = generate_sample()
#         data.append(sample_data)
        
#         # Flatten labels into a single list of coordinates [(x1, y1), (x2, y2), ...]
#         flattened_labels = [coord for position in sample_labels for coord in position[:2]]
#         labels.append(flattened_labels)
    
#     # Ensure all rows in labels have the same length
#     max_length = max(len(label) for label in labels)
#     for label in labels:
#         while len(label) < max_length:
#             label.extend([0.0])  # Pad with zeros
    
#     return np.array(data), np.array(labels)


# def generate_sample():
#     room_w = np.random.uniform(3, 8)
#     room_h = np.random.uniform(3, 6)
#     num_items = 6
#     room_type = np.random.randint(0, 3)  
    
#     positions = []
#     furniture_sizes = [
#         (0.3 * room_w, 0.4 * room_h),  # Bed
#         (0.2 * room_w, 0.2 * room_h),  # Desk
#         (0.15 * room_w, 0.3 * room_h), # Wardrobe
#         (0.1 * room_w, 0.1 * room_h),  # Chair
#         (0.25 * room_w, 0.2 * room_h), # Sofa
#         (0.2 * room_w, 0.15 * room_h)  # Table
#     ]
    
#     for size_idx in range(num_items):
#         while True:
#             if size_idx == 0:  # Bed near walls
#                 x = np.random.uniform(0, room_w * 0.2)
#                 y = np.random.uniform(0, room_h * 0.2)
#             elif size_idx == 1:  # Desk near windows (top-left corner)
#                 x = np.random.uniform(0, room_w * 0.3)
#                 y = np.random.uniform(room_h * 0.7, room_h)
#             elif size_idx == 4:  # Sofa near center
#                 x = np.random.uniform(room_w * 0.3, room_w * 0.7)
#                 y = np.random.uniform(room_h * 0.3, room_h * 0.7)
#             else:  # General placement for other items
#                 x = np.random.uniform(0, room_w - furniture_sizes[size_idx][0])
#                 y = np.random.uniform(0, room_h - furniture_sizes[size_idx][1])
            
#             collision = check_collision(x, y, positions,
#                                         furniture_sizes[size_idx], room_w, room_h)
            
#             if not collision:
#                 positions.append((x / room_w, y / room_h,
#                                 furniture_sizes[size_idx][0] / room_w,
#                                 furniture_sizes[size_idx][1] / room_h))
#                 break
    
#     return [room_w / 8.0, room_h / 6.0, num_items / 6.0, room_type / 2.0], positions

# def check_collision(x, y, positions, size, room_w, room_h):
#     for px, py, pw, ph in positions:
#         if not (x + size[0] < px or x > px + pw or y + size[1] < py or y > py + ph):
#             return True  # Collision detected
#     return False

# def generate_dataset(num_samples=100):
#     data = []
#     labels = []
    
#     for _ in range(num_samples):
#         sample_data, sample_labels = generate_sample()
#         data.append(sample_data)
        
#         # Flatten labels into a single list of coordinates
#         flattened_labels = [coord for position in sample_labels for coord in position]
#         labels.append(flattened_labels)
    
#     return np.array(data), np.array(labels)

def generate_sample():
    room_w = np.random.uniform(3, 8)
    room_h = np.random.uniform(3, 6)
    num_items = 6
    room_type = np.random.randint(0, 3)  
    
    positions = []
    furniture_sizes = [
        (0.3 * room_w, 0.4 * room_h),
        (0.2 * room_w, 0.2 * room_h),
        (0.15 * room_w, 0.3 * room_h),
        (0.1 * room_w, 0.1 * room_h),
        (0.25 * room_w, 0.2 * room_h),
        (0.2 * room_w, 0.15 * room_h)
    ]
    
    for size_idx in range(num_items):
        while True:
            x = np.random.uniform(furniture_sizes[size_idx][0], room_w - furniture_sizes[size_idx][0])
            y = np.random.uniform(furniture_sizes[size_idx][1], room_h - furniture_sizes[size_idx][1])
            
            collision = False
            for px, py in positions:
                if abs(px - x) < furniture_sizes[size_idx][0] and abs(py - y) < furniture_sizes[size_idx][1]:
                    collision = True
                    break
            
            if not collision:
                positions.append((x / room_w, y / room_h))  
                break
    
    return [room_w / 8.0, room_h / 6.0, num_items / 6.0, room_type / 2.0], positions


def generate_dataset(num_samples=100):
    data = []
    labels = []
    
    for _ in range(num_samples):
        sample_data, sample_labels = generate_sample()
        data.append(sample_data)
        labels.append(sample_labels)
    
    return np.array(data), np.array(labels).reshape(-1, len(sample_labels[0]) * len(sample_labels))
