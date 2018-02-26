from keras.models import load_model

import numpy

model = load_model('diabetes.h5')

predict_me = numpy.array([[1, 126, 60, 0, 0, 30.1, 0.349, 47]])

predictions = model.predict(predict_me)
rounded = [round(output[0]) for output in predictions]

print(rounded)
