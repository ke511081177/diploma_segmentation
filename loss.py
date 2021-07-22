
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten

smooth = 1. # 用於防止分母爲0.
def dice_coef(y_true, y_pred):
    flatten_layer = Flatten()
    y_true_f = flatten_layer(y_true) # 將 y_true 拉伸爲一維.
    y_pred_f = flatten_layer(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
