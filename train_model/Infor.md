Dataset dùng đúng annotation hiện có: 0=Gach_Ten, 1=Ten.
Ảnh chữ rất ngang, nên input mặc định là 128x512 với resize-padding giữ tỉ lệ, không ép méo về 224x224.
CNN dùng ResNet18 pretrained ImageNet, output tạm thời là FC 2 lớp.
SVM train trên feature của train + val, test riêng trên test.
Kết quả trực quan gồm: training_curves.png, confusion matrix, ROC/PR curve, SVM grid heatmap, file dự đoán CSV, ảnh các mẫu phân loại sai.



chạy:
# Tao virtual environment
python -m venv .venv

# Cho phep activate trong PowerShell hien tai
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Kich hoat moi truong
.\.venv\Scripts\Activate.ps1

# Cap nhat pip va cai thu vien
python -m pip install --upgrade pip
pip install -r requirements.txt

# Kiem tra dataset visualization
python visualize_dataset.py --config config.yaml

# Chay pipeline train resnet18
python run_pipeline.py --config config.yaml

# Chay pipeline train resnet50
python run_pipeline.py --config config_resnet50.yaml

# Chay pipeline train convnext_tiny
python run_pipeline.py --config config_convnext_tiny.yaml

ResNet18 sinh feature 512 chieu. ResNet50 sinh feature 2048 chieu. ConvNeXt-Tiny sinh feature 768 chieu.
