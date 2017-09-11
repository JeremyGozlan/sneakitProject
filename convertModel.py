from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
import coremltools

adidas=['NMD','StanSmith','Ultraboost','YeezyBoost350V2']
nike=['AirForce1','AirMax1','FlyknitRacer']
jordan=['AirJordan1','AirJordan3','AirJordan4']
converse = ['ConverseHigh','ConverseLow']

json_file = open('converse-3x-rgb-30e.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("converse-3x-rgb-30e.h5")
print "Loaded model from disk"

# evaluate loaded model on test data
loaded_model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
coreml_model = coremltools.converters.keras.convert(loaded_model, input_names ='image', image_input_names = 'image', class_labels = ['ConverseHigh','ConverseLow'])
coreml_model.save('ConverseModel.mlmodel')
