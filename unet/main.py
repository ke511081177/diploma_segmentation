from model import *
from data import *
import matplotlib.pyplot as plt
import random

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(2)


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip = True,
                    vertical_flip = True,
                    channel_shift_range = 0.1,
                    fill_mode='nearest'
                    )
myGene = trainGenerator(8,'data/stamp/train','image','label',data_gen_args,save_to_dir = None)

valGene = trainGenerator(8,"data/stamp/val",'image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_stamp_15.hdf5', monitor='loss',verbose=1, save_best_only=True)
history  = model.fit(myGene, validation_data = valGene,steps_per_epoch=100, epochs=6, validation_steps = 200, callbacks=[model_checkpoint])


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


testGene = testGenerator("data/stamp/test")
results = model.predict(testGene,10,verbose=1)
saveResult("data/stamp/test",results)

