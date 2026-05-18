# CNN + SVM cho phân loại `Ten` và `Gach_Ten`

Pipeline gồm 2 giai đoạn:

1. Fine-tune `ResNet18` pretrained ImageNet với Cross-Entropy để CNN học đặc trưng chữ viết/nét gạch.
2. Cắt lớp FC cuối, trích vector 512 chiều, train `SVC(kernel="rbf")` bằng Grid Search trên `C` và `gamma`.

Ảnh trong dataset có tỉ lệ ngang/dọc khác nhau, nên loader dùng resize-padding giữ tỉ lệ về kích thước mặc định `128x512`. ResNet vẫn trả feature 512 chiều nhờ global average pooling.

## Cài đặt

```powershell
pip install -r requirements.txt
```

## Xem tổng quan dataset

```powershell
python visualize_dataset.py --config config.yaml
```

Kết quả nằm trong `runs/dataset_overview/`: phân bố nhãn, phân bố kích thước ảnh, lưới ảnh mẫu.

## Huấn luyện CNN

```powershell
python train_cnn.py --config config.yaml
```

Output chính trong `runs/cnn_resnet18/`:

- `best_cnn.pt`, `last_cnn.pt`
- `history.csv`
- `training_curves.png`
- `cnn_val_confusion.png`, `cnn_test_confusion.png`
- `cnn_*_classification_report.txt`
- `tensorboard/`

Xem TensorBoard:

```powershell
tensorboard --logdir runs/cnn_resnet18/tensorboard
```

## Trích feature và train SVM

```powershell
python train_svm.py --config config.yaml
```

Output chính trong `runs/svm_resnet18/`:

- `features.npz`
- `svm_model.joblib`
- `best_params.json`
- `grid_search_results.csv`
- `svm_grid_heatmap.png`
- `svm_test_confusion.png`
- `svm_test_roc_curve.png`
- `svm_test_pr_curve.png`
- `svm_test_classification_report.txt`
- `svm_test_predictions.csv`
- `svm_test_misclassified.png`

## Chạy toàn bộ pipeline

```powershell
python run_pipeline.py --config config.yaml
```

Nếu đã train CNN và chỉ muốn train lại SVM:

```powershell
python run_pipeline.py --config config.yaml --skip-cnn --force-extract
```

## Đánh giá lại model

CNN:

```powershell
python evaluate.py --config config.yaml --model-type cnn --split test --checkpoint runs/cnn_resnet18/best_cnn.pt
```

SVM:

```powershell
python evaluate.py --config config.yaml --model-type svm --split test --svm-model runs/svm_resnet18/svm_model.joblib --feature-cache runs/svm_resnet18/features.npz
```

## Compare ResNet18, ResNet50 and ConvNeXt-Tiny

Default config uses `model.name: resnet18` and writes to `runs/cnn_resnet18`, `runs/svm_resnet18`.

ResNet50 config is available in `config_resnet50.yaml`. It uses `model.name: resnet50`, writes to separate output folders, and extracts 2048-dim CNN features before training SVM.

ConvNeXt-Tiny config is available in `config_convnext_tiny.yaml`. It uses `model.name: convnext_tiny`, writes to separate output folders, and extracts 768-dim CNN features before training SVM.

```powershell
python run_pipeline.py --config config.yaml
python run_pipeline.py --config config_resnet50.yaml
python run_pipeline.py --config config_convnext_tiny.yaml
```

Important paths for comparison:

- ResNet18 CNN: `runs/cnn_resnet18/`
- ResNet18 SVM: `runs/svm_resnet18/`
- ResNet50 CNN: `runs/cnn_resnet50/`
- ResNet50 SVM: `runs/svm_resnet50/`
- ConvNeXt-Tiny CNN: `runs/cnn_convnext_tiny/`
- ConvNeXt-Tiny SVM: `runs/svm_convnext_tiny/`

## Deploy ResNet18 for Real Inference

The deploy package is in `deploy_resnet18/`. It can run either:

- CNN only: `best_cnn.pt`
- CNN + SVM: `best_cnn.pt` plus `svm_model.joblib`

Put trained weights into:

```text
deploy_resnet18/weights/best_cnn.pt
deploy_resnet18/weights/svm_model.joblib
```

Copy from training outputs:

```powershell
Copy-Item runs\cnn_resnet18\best_cnn.pt deploy_resnet18\weights\best_cnn.pt
Copy-Item runs\svm_resnet18\svm_model.joblib deploy_resnet18\weights\svm_model.joblib
```

Run CNN only:

```powershell
python deploy_resnet18\predict.py --mode cnn --input C:\path\to\image.jpg
```

Run ResNet18 + SVM:

```powershell
python deploy_resnet18\predict.py --mode svm --input C:\path\to\image.jpg
```

If you do not want to copy weights, pass paths directly:

```powershell
python deploy_resnet18\predict.py --mode svm --input C:\path\to\image.jpg --cnn-checkpoint runs\cnn_resnet18\best_cnn.pt --svm-model runs\svm_resnet18\svm_model.joblib
```
