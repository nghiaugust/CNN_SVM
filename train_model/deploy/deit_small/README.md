# Deploy DeiT-Small / DeiT-Small + SVM

Package nay chay inference cho model DeiT-Small da train bang `configs/config_deit_small.yaml`.

## Dat weights

Neu chay tu thu muc `train_model/`:

```powershell
Copy-Item runs\cnn_deit_small\best_cnn.pt deploy\deit_small\weights\best_cnn.pt
Copy-Item runs\svm_deit_small\svm_model.joblib deploy\deit_small\weights\svm_model.joblib
```

Hoac truyen duong dan truc tiep bang `--cnn-checkpoint` va `--svm-model`.

## Predict

```powershell
python deploy\deit_small\predict.py --mode cnn --input C:\path\to\image.jpg
python deploy\deit_small\predict.py --mode svm --input C:\path\to\images --output svm_predictions.csv
```

Ket qua gom `path`, `pred_label`, `pred_name`, va cac cot `prob_*` neu model co xac suat.
