import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Xác định thiết bị (GPU nếu có, nếu không thì dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình ResNet50 với trọng số pre-trained
model = models.resnet50(pretrained=True).to(device)

# Đóng băng các tham số của mô hình (trừ lớp fully connected)
for param in model.parameters():
    param.requires_grad = False

# Thay thế lớp fully connected
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)  # Phân loại nhị phân: Healthy vs Tumor
).to(device)

# Tải trọng số từ file best_mri_model.pth
model.load_state_dict(torch.load('best_mri_model.pth', map_location=device))

# Định nghĩa biến đổi cho ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Hàm dự đoán ảnh
def predict_image(image_path, model, device, transform):
    try:
        # Load ảnh từ đường dẫn
        image = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh có 3 kênh (RGB)
        image_tensor = transform(image).unsqueeze(0).to(device)  # Thêm batch dimension

        # Chuyển mô hình sang chế độ đánh giá
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Danh sách các lớp
        classes = ['Healthy', 'Tumor']
        prediction = classes[predicted.item()]
        
        return image, prediction
    except Exception as e:
        print(f"Lỗi: {e}")
        return None, None

# Chương trình chính
if __name__ == "__main__":
    print("Đã load mô hình thành công!")
    while True:
        # Nhập đường dẫn ảnh từ người dùng
        image_path = input("Nhập đường dẫn ảnh (hoặc 'exit' để thoát): ")
        
        if image_path.lower() == 'exit':
            print("Đã thoát chương trình.")
            break
        
        # Dự đoán trên ảnh
        image, prediction = predict_image(image_path, model, device, transform)
        
        if image is not None and prediction is not None:
            print(f"Dự đoán: {prediction}")
            # Hiển thị ảnh và kết quả
            plt.imshow(image)
            plt.title(f"Dự đoán: {prediction}")
            plt.axis('off')
            plt.show()
        else:
            print("Không thể xử lý ảnh. Vui lòng kiểm tra đường dẫn và thử lại.")