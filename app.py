from flask import Flask
from keras.models import load_model
import numpy
import tensorflow as tf

model = load_model('diabetes.h5')
# https://github.com/keras-team/keras/issues/2397#issuecomment-254919212
graph = tf.get_default_graph()

app = Flask(__name__)

@app.route("/")
def predict():
    with graph.as_default():
        predict_me = numpy.array([[1, 126, 60, 0, 0, 30.1, 0.349, 47]])
        predictions = model.predict(predict_me)
    
    return "chances of being diabetic are  %.2f" % predictions[0]
