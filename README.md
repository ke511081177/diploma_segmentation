# diploma_segmentation


## schoolClassification.py

- 把 image(**資料夾**)中的照片分類學校存在classify(**資料夾**)

## unet

- 訓練用於切割畢業證書以及印泥的U-net模型架構
  - data.py
    - 訓練資料處理
  - model.py
    - Model init

  - main.py
    - 訓練  

## main.py

- 各功能測試
  - 背景切割
  - 印泥切割
  - 學校分類 

  **備註: model檔案過大，未上傳**

## jiabaDictionary

- 用於補充訓練斷詞的字庫

## [Data連結](https://drive.google.com/drive/folders/1uUuRkW2yG3m6i-n_UlYMy8iv1C1VL_0x?usp=sharing)
- image 原圖
- 分類完資料
  - [ ] unsorted資料夾 做前處理 
