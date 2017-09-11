from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers

'''ultraboost total = 2116   ==> 20%  423 on testing ==> 20%  338 on validation and rest on training  = 1355'''
'''stan smith total = 1776   ==> 20%  355 on testing ==> 20%  284 on validation and rest on training  = 1137'''
'''nmd total        = 2670   ==> 20%  534 on testing ==> 20%  427 on validation and rest on training  = 1709'''
'''converse         = 1476   ==> 20%  295 on testing ==> 20%  236 on validation and rest on training  = 945'''
'''am1              = 1171   ==> 20%  234 on testing ==> 20%  187 on validation and rest on training  = 750'''
'''jordan 3           1507   ==> 20 % 301 testing ===> 241 training '''
'''jordan 4           1735            347     277   '''
'''racer  1999                        400   320
yeezy 1691             338         270
air force 1  1408      281    225
''' '''jordan 1 2004     400    320 '''
''' converse high 2093   418   335 '''
''' 10 epochs final loss: 0.9971 - acc: 0.9177 - val_loss: 1.7646 - val_acc: 0.8619'''
'''    '''
imageWidth,imageHeight = 200,200
trainingPath='sneakers/TrainingSet'
validatingPath='sneakers/ValidationSet'
epochs_size=40
batch_size=32
num_class=11

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):

  x = base_model.output
  x = Flatten(input_shape=base_model.output_shape[1:])(x)
  x = Dense(1024, activation="relu")(x)
  x = Dropout(0.5)(x)

  predictions=Dense(nb_classes,activation='softmax')(x)
  model1 = Model(input = base_model.input, output = predictions)
  return model1


def setup_to_finetune(model):
   for layer in model.layers[:16]:
      layer.trainable = False
   for layer in model.layers[16:]:
      layer.trainable = True
   model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


datagen =  ImageDataGenerator(rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect')


'''image = np.dstack([1chan, 1chan, 1chan])'''

training_generator=datagen.flow_from_directory(directory=trainingPath,
                                              target_size=(200,200),
                                              batch_size=batch_size,
                                              seed=123)

validation_generator=datagen.flow_from_directory(directory=validatingPath,
                                              target_size=(200,200),
                                              batch_size=batch_size,
                                              seed=123)


base_model=applications.VGG16(weights='imagenet',include_top=False,input_shape=(200,200,3))

model = add_new_last_layer(base_model,num_class)

setup_to_transfer_learn(model,base_model)
'''
history_tl = model.fit_generator(
    training_generator,
    samples_per_epoch=6076,
    epochs=epochs_size,
    validation_data=validation_generator,
    nb_val_samples=1467)'''

setup_to_finetune(model)
'''10325,2577'''
'''12511,3122'''
model1= model.fit_generator(
    generator=training_generator,
    samples_per_epoch=37533,
    nb_epoch=epochs_size,
    validation_data=validation_generator,
    nb_val_samples=9366)

model_json=model1.model.to_json()
with open('16-3x-rgb-40e-11.json','w') as json_file:
    json_file.write(model_json)
model1.model.save_weights('16-3x-rgb-40e-11.h5')
print ('Model saved to disk')

'''to load after..
 load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
'''
