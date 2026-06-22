# Deploy ResNet18 / ResNet18 + SVM

Package nay chay inference cho model ResNet18 da train bang `configs/config.yaml`.

## Dat weights

Neu chay tu thu muc `train_model/`:

```powershell
Copy-Item runs\cnn_resnet18\best_cnn.pt deploy\resnet18\weights\best_cnn.pt
Copy-Item runs\svm_resnet18\svm_model.joblib deploy\resnet18\weights\svm_model.joblib
```

Hoac truyen duong dan truc tiep bang `--cnn-checkpoint` va `--svm-model`.

## Predict

```powershell
python deploy\resnet18\predict.py --mode cnn --input C:\path\to\image.jpg
python deploy\resnet18\predict.py --mode svm --input C:\path\to\images --output svm_predictions.csv
```

Ket qua gom `path`, `pred_label`, `pred_name`, va cac cot `prob_*` neu model co xac suat.
