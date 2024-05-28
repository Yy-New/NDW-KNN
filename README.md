# NDW-KNN
![Uploading image.png…]()
### Step 1:
Run `python LogData.py` can logarithmically transform the data in the PublicData folder and obtain a LogPublicData folder. The data file name corresponds to the data file name of the PublicData folder.

### Step 2:
Run `python MissingGeneration.py` can generate a series of missing data from the data set in LogPublicData. The missing data is saved in the MMData folder.

### Step 3:
Run `python ./ImputeTest/IDWKNNImputeTest.py` to execute NDW-KNN imputation. NDW-KNN imputation data will be saved in the IDWKNNFillData folder.

### Step 4:
Run `python ./ImputeTest/KNNImputeTest.py` to execute KNN and NS-KNN imputation. KNN and NS-KNN imputation data will be saved in the KNNFillData folder.

### Step 5:
Run `python ./EvaluationNRMSE/IDWKNNEvaluation.py` to get the NRMSE of NDW-KNN interpolation data. The results are saved in the Results folder.

Run `python ./EvaluationNRMSE/KNNEvaluation.py` to get the NRMSE of KNN and NS-KNN interpolation data. The results are saved in the Results folder.

Run `python ./EvaluationNRMSE/AverageNRMSE.py` to get the average NRMSE of KNN, NS-KNN, NDW-KNN interpolation data. The results are saved in the AverageResult folder.

Run `python ./EvaluationMAPE/IDWKNNMAPE.py` to get the MAPE of NDW-KNN interpolation data. The results are saved in the Results folder.

Run `python ./EvaluationMAPE/KNNMAPE.py` to get the MAPE of KNN and NS-KNN interpolation data. The results are saved in the Results folder.

Run `python ./EvaluationMAPE/AverageMAPE.py` to get the average MAPE of KNN, NS-KNN, NDW-KNN interpolation data. The results are saved in the AverageResult folder.


