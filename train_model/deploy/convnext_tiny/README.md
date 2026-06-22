# Deploy ConvNeXt-Tiny / ConvNeXt-Tiny + SVM

Package nay chay inference cho model ConvNeXt-Tiny da train bang `configs/config_convnext_tiny.yaml`.

## Dat weights

Neu chay tu thu muc `train_model/`:

```powershell
Copy-Item runs\cnn_convnext_tiny\best_cnn.pt deploy\convnext_tiny\weights\best_cnn.pt
Copy-Item runs\svm_convnext_tiny\svm_model.joblib deploy\convnext_tiny\weights\svm_model.joblib
```

Hoac truyen duong dan truc tiep bang `--cnn-checkpoint` va `--svm-model`.

## Predict

```powershell
python deploy\convnext_tiny\predict.py --mode cnn --input C:\path\to\image.jpg
python deploy\convnext_tiny\predict.py --mode svm --input C:\path\to\images --output svm_predictions.csv
```

Ket qua gom `path`, `pred_label`, `pred_name`, va cac cot `prob_*` neu model co xac suat.
