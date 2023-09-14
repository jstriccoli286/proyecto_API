from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from models import Prediction_Input
from models import Prediction_Out

import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier # modelo con árboles de decisión similar a XGBoost
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

import pickle

MODEL_PATH = 'boost.pkl'

#Load Scikit-learn model
loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
##print(loaded_model)

router = APIRouter()

preds = []

@router.get("/ml")
def get_prediction():
    return preds

@router.post("/ml", status_code = status.HTTP_201_CREATED, response_model= Prediction_Out)
def predict(pred_input: Prediction_Input):
    
    pred_input1 = {"id": [pred_input.id],"EK": [pred_input.EK],"EK_DMSNR_Curve": [pred_input.EK_DMSNR_Curve]}
    
    pred_input1 = pd.DataFrame(pred_input1)
    print(pred_input1)
    
    pred_input2 = pd.DataFrame(pred_input1.loc[:,"EK":"EK_DMSNR_Curve"])
    print(pred_input2)
    
    prediction_m = pd.DataFrame(loaded_model.predict(pred_input2),columns=['prediction']) #pred_input1.loc[0,"EK"],pred_input1.loc[0,"EK_DMSNR_Curve"]
    prediction_dict = {"id": str(pred_input1.loc[0,"id"]),"EK": float(pred_input1.loc[0,"EK"]), "EK_DMSNR_Curve": float(pred_input1.loc[0,"EK_DMSNR_Curve"]), "pred": float(prediction_m.iloc[0, 0])}
    preds.append(prediction_dict)
   
    print(prediction_m)
    print(prediction_m.iloc[0, 0])
    
    return JSONResponse(content=prediction_dict)


