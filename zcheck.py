import torch

# Load the YOLOv5 model
model = torch.load('best.pt', map_location=torch.device('cpu'))

# Extract class names
if 'names' in model:
    print(model['names'])
elif 'model' in model and hasattr(model['model'], 'names'):
    print(model['model'].names)
else:
    print("No class names defined in the model.")
