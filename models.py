from pydantic import BaseModel

class Prediction_Input(BaseModel):
    id: int
    EK: float
    EK_DMSNR_Curve: float

class Prediction_Out(BaseModel):
    id: int
    EK: float
    EK_DMSNR_Curve: float
    Pred: float
