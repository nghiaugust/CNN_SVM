# Deploy YOLOv8 Classification

Package nay chay inference cho YOLOv8 classification da train bang `configs/config_yolov8.yaml`.

## Dat weights

Neu chay tu thu muc `train_model/`:

```powershell
Copy-Item runs\yolov8_cls\yolov8n_cls\weights\best.pt deploy\yolov8\weights\best.pt
```

Hoac truyen duong dan truc tiep bang `--checkpoint`.

## Predict

```powershell
python deploy\yolov8\predict.py --input C:\path\to\image.jpg
python deploy\yolov8\predict.py --input C:\path\to\images --output yolov8_predictions.csv
```

Ket qua gom `path`, `pred_label`, `pred_name`, va cac cot `prob_*`.
