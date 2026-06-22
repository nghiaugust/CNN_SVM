# Strike-Out Classifier

Dự án phân loại ảnh tên thành 2 lớp:

- `Gach_Ten`: tên bị gạch
- `Ten`: tên không bị gạch

Hỗ trợ các mô hình: ResNet18, ResNet50, ConvNeXt-Tiny, DeiT-Small, YOLOv8 classification. Mỗi mô hình CNN có thêm chế độ SVM trên đặc trưng trích từ backbone.

> Chạy các lệnh bên dưới từ thư mục `CNN_SVM/train_model`.

## 1. Cài môi trường

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng GPU NVIDIA CUDA 12.8:

```powershell
pip uninstall -y torch torchvision
pip install -r requirements-gpu-cu128.txt
pip install -r requirements.txt
```

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 2. Kiểm tra trước khi train

Chạy kiểm tra nhanh tất cả mô hình:

```powershell
python tools\preflight_check_models.py --device auto --batch-size 2 --no-backward
```

Chạy kiểm tra kỹ hơn, có backward một bước:

```powershell
python tools\preflight_check_models.py --device auto --batch-size 2
```

Kết quả lưu tại:

```text
runs/preflight/preflight_results.csv
```

## 3. Train tất cả mô hình

```powershell
python train\train_all_models.py --device auto
```

Train nhanh để thử pipeline:

```powershell
python train\train_all_models.py --device auto --cnn-epochs 1 --yolo-epochs 1
```

Train một số mô hình:

```powershell
python train\train_all_models.py --models resnet18,convnext_tiny,deit_small --device auto
```

Bỏ qua mô hình đã có trọng số:

```powershell
python train\train_all_models.py --skip-existing --device auto
```

## 4. Train riêng từng mô hình

```powershell
python train\run_pipeline.py --config configs\config.yaml --device auto
python train\run_pipeline.py --config configs\config_resnet50.yaml --device auto
python train\run_pipeline.py --config configs\config_convnext_tiny.yaml --device auto
python train\run_pipeline.py --config configs\config_deit_small.yaml --device auto
python train\train_yolov8.py --config configs\config_yolov8.yaml --device auto
```

## 5. Đánh giá tất cả mô hình

Đánh giá trên bộ `test` và `data/evaluation`:

```powershell
python evaluation\evaluate_all_models.py --device auto
```

Chỉ đánh giá một vài mô hình:

```powershell
python evaluation\evaluate_all_models.py --models resnet18,deit_small,yolov8 --device auto
```

Kết quả lưu tại:

```text
runs/evaluation_all/metrics.csv
runs/evaluation_all/predictions.csv
```

Các metric chính gồm: `precision`, `recall`, `f1`, `f1_macro`, `accuracy`, `inference_time_s`, `time_per_image_ms`. Mặc định `precision/recall/f1` tính cho lớp `Gach_Ten` (`positive-label=0`); có thể đổi bằng `--positive-label`.

## 6. Dữ liệu và output

Dữ liệu đang dùng:

```text
data/dataset/
  train_annotation.txt
  val_annotation.txt
  test_annotation.txt
  Gach_Ten/
  Ten/

data/evaluation/
  Gach_Ten/
  Ten/
```

Nơi lưu trọng số mặc định:

```text
runs/cnn_resnet18/best_cnn.pt
runs/svm_resnet18/svm_model.joblib
runs/cnn_resnet50/best_cnn.pt
runs/svm_resnet50/svm_model.joblib
runs/cnn_convnext_tiny/best_cnn.pt
runs/svm_convnext_tiny/svm_model.joblib
runs/cnn_deit_small/best_cnn.pt
runs/svm_deit_small/svm_model.joblib
runs/yolov8_cls/yolov8n_cls/weights/best.pt
```

Nếu config gốc ghi `dataset` nhưng dữ liệu nằm trong `data/dataset`, các script `train_all_models.py`, `preflight_check_models.py`, và `evaluate_all_models.py` sẽ tự fallback sang `data/dataset`.
