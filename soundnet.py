from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from keras.models import Sequential
import numpy as np
import librosa
import json
from keras import backend as K

def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on mono audio files with a sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=22050, mono=True)
    audio = preprocess(audio)
    return audio

def build_model():
    """
    Builds up the SoundNet model and loads the weights from a given model file (8-layer model is kept at models/sound8.npy).
    :return:
    """
    model_weights = np.load('sound8.npy').item()
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, None, 1)))

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                          'kernel_size': 8, 'conv_strides': 2},
                         ]

    for x in filter_parameters:
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']


            model.add(BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var])
            model.add(Activation('relu'))
        if 'pool_size' in x:
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    return model

def predict_scene_from_audio_file(audio_file):
    model = build_model()
    audio = load_audio(audio_file)
    return model.predict(audio)


def predictions_to_scenes(prediction):
    scenes = []
    with open('categories_places2.txt', 'r') as f:
        categories = f.read().split('\n')
        for p in range(prediction.shape[1]):
            scenes.append(categories[np.argmax(prediction[0, p, :])])
    return scenes

def get_acoustic_instruments ( file_names ):
  acoustic_data = list()
  acoustic_family = list()
  for file in file_names:
    source = data.get(file).get("instrument_source")
    if source == 0:
      acoustic_data.append(file)
      acoustic_family.append(data.get(file).get("instrument_family"))
  return acoustic_data, acoustic_family

def remap_family ( families ):
  new_data = list()
  for i in families:
    if i == 1:
      new_data.append(0)
    if i == 2:
      new_data.append(1)
    if i == 3:
      new_data.append(2)
    if i == 4:
      new_data.append(3)
    if i == 5:
      new_data.append(4)
    if i == 7:
      new_data.append(5)
    if i == 8:
      new_data.append(6)
    if i == 10:
      new_data.append(7)
  return new_data

def get_sound_data(files):
  data_list = []
  for file_name in files:
    data_list.append(load_audio('../nsynth-test/audio/{}.wav'.format(file_name)))
  return data_list

#from keras import backend as K
def getActivations(data,number_layer,model):
    intermediate_tensor = []
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[number_layer].output])
    for audio in data:
        #getHiddenRepresentation       
        layer_output = get_layer_output([audio])[0]
        tensor = layer_output.reshape(1,-1)
        intermediate_tensor.append(tensor[0])
    return intermediate_tensor

#import json
with open("../nsynth-test/examples.json","r") as file:
  data = json.load(file)

file_names =  list(data.keys())

acoustic_data, acoustic_family = get_acoustic_instruments ( file_names )

acoustic_family = remap_family ( acoustic_family )

x_tr = np.array(get_sound_data(acoustic_data))

model = build_model()

activations = getActivations(x_tr,22,model)
x = np.asarray(activations)
y = np.asarray(acoustic_family)
np.save('activations.npy', x)
np.save('labels.npy', y)
