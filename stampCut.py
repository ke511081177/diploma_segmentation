import cv2
from tensorflow import keras
from loss import*
from func import*

def stampCut():
        
    stampModel = keras.models.load_model('unet_stamp_rgb_300_15_binary.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
    
    img  = cv2.imread('74.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tempImg = img
    rimg = cv2.resize(img, (256, 256))


    binary = np.reshape(rimg,(1,)+rimg.shape)
    
    mask = stampModel.predict(binary)
    
    mask = np.stack((mask,)*3, axis=-1)
    
    mask = np.reshape(mask,(256,256,3))
    
    mask = cv2.resize(mask,(img.shape[1],img.shape[0]))
    
    mask = mask[:,:,0]

    mask = np.uint8(mask)
    
    img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    squares = find_squares(img)
    
    if squares == 0:
        
        print('----ERROR------')
        return 0

    img = tempImg[min(squares[0][0][1],squares[0][2][1]):max(squares[0][0][1],squares[0][2][1]),
                min( squares[0][2][0],squares[0][0][0]):max(squares[0][2][0],squares[0][0][0])]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite('stamp.jpg',img)


