from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import librosa as lr
from tensorflow.keras import applications
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, Dense, Embedding,
                          Flatten, Input, MaxPooling2D, concatenate)
from tensorflow.keras.models import Model
from loguru import logger
from microquake.core.settings import settings

_version_ = 1.0


class SeismicModel:
    '''
    Class to classify mseed stream into one of the classes
        anthropogenic event, controlled explosion, earthquake, explosion,
        quarry blast
    '''
    REF_HEIGHT = 900.00
    REF_MAGNITUDE = -1.2

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __init__(self, model_name='multiclass-model.hdf5', ):
        '''
            :param model_name: Name of the model weight file name.
        '''
        self.base_directory = Path(settings.common_dir)/'../data/weights'
        # Model was trained at these dimensions
        self.D = (128, 128, 1)
        self.microquake_class_names = ['blast', 'crusher noise', 'drilling noise', 'electrical noise/lightning', 'mechanical noise', 'open pit blast', 
                'orepass noise', 'seismic event', 'surface event', 'test pulse']
        self.num_classes = len(self.microquake_class_names)
        self.model_file = self.base_directory/f"{model_name}"
        self.create_model()

    ###############################################
    # Librosa gives mel Spectrum which shows events clearly,
    # it is designed for audio frequencies which is suitable
    # to seismic events
    ################################################
    def librosa_spectrogram(self, tr, height=128, width=128):
        '''
            Using Librosa mel-spectrogram to obtain the spectrogram
            :param tr: stream trace
            :param height: image hieght
            :param width: image width
            :return: numpy array of spectrogram with height and width dimension
        '''
        data = self.get_norm_trace(tr).data
        signal = data * 255
        hl = int(signal.shape[0] // (width * 1.1))  # this will cut away 5%
        # from start and end
        spec = lr.feature.melspectrogram(signal, n_mels=height,
                                         hop_length=int(hl))
        img = lr.amplitude_to_db(spec)
        start = (img.shape[1] - width) // 2

        return img[:, start:start+width]

    #############################################
    # Data preparation
    #############################################
    @staticmethod
    def get_norm_trace(tr, taper=True):
        """
        :param tr: mseed stream
        :param taper: Boolean
        :return: normed composite trace
        """

        # c = tr[0]
        c = tr.composite()
        c[0].data = c[0].data / np.abs(c[0].data).max()  # we only
        # interested in c[0]
        c = c.detrend(type='demean')

        nan_in_context = np.any(np.isnan(c[0].data))

        if nan_in_context:
            logger.warning('NaN found in context trace. The NaN will be set '
                           'to 0.\nThis may cause instability')
            c[0].data = np.nan_to_num(c[0].data)

        if taper:
            c = c.taper(max_percentage=0.05)

        return c[0]

    @staticmethod
    def get_spectrogram(tr, nfft=512, noverlap=511):
        """
        :param tr: mseed stream
        :param nfft: The number of data points used in each block for the FFT
        :param noverlap: The number of points of overlap between blocks
        :param output_dir: directory to save spectrogram.png such as
        SPEC_BLAST_UG
        :return: RBG image array
        """
        rate = SeismicModel.get_norm_trace(tr).stats.sampling_rate
        data = SeismicModel.get_norm_trace(tr).data

        fig, ax = plt.subplots(1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        _, _, _, _ = ax.specgram(x=data, Fs=rate,
                                 noverlap=noverlap, NFFT=nfft)
        ax.axis('off')
        fig.set_size_inches(.64, .64)

        fig.canvas.draw()

        width, height = fig.get_size_inches() * fig.get_dpi()
        mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

        imarray = np.reshape(mplimage, (int(height), int(width), 3))
        plt.close(fig)

        return imarray

    @staticmethod
    def rgb2gray(rgb):
        """
        Convert RBG colored image to gray scale
        :param rgb: RGB image array
        :return: Gray scaled image array
        """

        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def normalize_gray(array):
        """
        :param array: Gray colored image array
        :return: Normalized gray colored image
        """

        return (array - array.min()) / (array.max() - array.min())

    @staticmethod
    def resnet50_layers(i):
        '''
            wrapper around the resnet 50 model, it starts by converting the
            one channel input to 3 channgels and then
            load resnet50 model
            :param i: input layer in this case the context trace
            :retun the flattend layer after the resent50 block
        '''
        x = Conv2D(filters=3, kernel_size=2, padding='same',
                   activation='relu')(i)

        x = applications.ResNet50(weights=None, include_top=False)(x)
        x = Flatten()(x)

        return x

    @staticmethod
    def conv_layers(i):
        '''
            create convolution layers for 2 second stream.
            :param i: input layers
            :return Flattened layer
        '''
        kern_size = (3, 3)
        dim = 64
        x = Conv2D(filters=16, kernel_size=2, padding='same',
                   activation='relu')(i)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(filters=32, kernel_size=2, padding='same',
                   activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Conv2D(filters=64, kernel_size=2, padding='same',
                   activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)

        X_shortcut = BatchNormalization()(x)

        for _ in range(4):
            y = Conv2D(filters=dim, kernel_size=kern_size,
                       activation='relu', padding='same')(X_shortcut)
            y = BatchNormalization()(y)
            # y = Conv2D(filters = dim, kernel_size = kern_size,
            # activation='relu', padding='same')(y)
            y = Conv2D(filters=dim, kernel_size=kern_size,
                       activation='relu', padding='same')(y)
            X_shortcut = Add()([y, X_shortcut])
        x = Flatten()(x)
        x = Dense(500, activation='relu')(x)
        x = BatchNormalization()(x)
        # x = Dense(128, activation='relu')(x)

        return x

    def create_model(self):
        """
        Create model and load weights
        """
        input_shape = (self.D[0], self.D[1], 1)
        i1 = Input(shape=input_shape, name="spectrogram")
        i2 = Input(shape=(1,), name='height', dtype='int32')
        i3 = Input(shape=input_shape, name='context_trace')
        i4 = Input(shape=(1,), name='magnitude', dtype='int32')
        emb = Embedding(3, 2)(i2)
        flat1 = Flatten()(emb)
        emb = Embedding(3, 2)(i4)
        flat2 = Flatten()(emb)

        x1 = self.conv_layers(i1)
        x2 = self.resnet50_layers(i3)

        x = concatenate([x1, x2], axis=-1)
        x = Dense(64, activation='relu')(x)
        x = concatenate([x,  flat1, flat2], axis=-1)
        x = Dense(32, activation='relu')(x)
        x = Dense(self.num_classes, activation='sigmoid')(x)
        self.model = Model([i1, i2, i3, i4], x)
        self.model.load_weights(f"{self.model_file.resolve()}")

    def predict(self, tr, context_trace, height, magnitude):
        """
        :param tr: Obspy stream object (2 s) that is good for discriminating
        between events
        :param context_trace: context trace object (20s) good for
        discriminating blast and earth quake
        :param height: the z-value of event.
        :return: dictionary of  event classes probability
        """
        spectrogram = self.librosa_spectrogram(context_trace, height=self.D[0], 
            width=self.D[1])
        contxt_img = self.normalize_gray(spectrogram)
        spectrogram = self.librosa_spectrogram(tr, height=self.D[0],
                                               width=self.D[1])
        normgram = self.normalize_gray(spectrogram)
        img = normgram[None, ..., None]  # Needed to in the form of batch
        # with one channel.
        contxt = contxt_img[None, ..., None]
        h = []
        m = []
        # We use the height as a category, greater than 900 or not

        if height >= self.REF_HEIGHT:
            h.append(1)
        else:
            h.append(0)

        if magnitude >= self.REF_MAGNITUDE:
            m.append(1)
        else:
            m.append(0)

        data = {'spectrogram': img, 'context_trace': contxt,
                'height': np.asarray(h), 'magnitude': np.asarray(m)}
        a = self.model.predict(data)

        classes = {}

        for p, n in zip(a.reshape(-1), self.microquake_class_names):
            classes[n] = p
        classes['noise/unknown'] = 1-np.max(a.reshape(-1))

        return classes
