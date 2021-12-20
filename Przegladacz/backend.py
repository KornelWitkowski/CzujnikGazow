import numpy as np

import pandas as pd

import keras.backend as keras
import tensorflow as tf


def compare_numpy_arrays_columns_with_nan(arr1, arr2, col_num):
    return ((arr1[:, col_num] == arr2[:, col_num])
            | (np.isnan(arr1[:, col_num])
            & np.isnan(arr2[:, col_num]))).all()


def sort_array_by_column(arr, column_num, desc=False):
    if desc:
        return arr[arr[:, column_num].argsort()[::-1]]
    return arr[arr[:, column_num].argsort()]


def read_file(path):
    try:
        spectrums_full_data = pd.read_csv(path)
    except:
        spectrums_full_data = pd.DataFrame([])

    if spectrums_full_data.shape[1] == 2056:
        spectrums_data_np = spectrums_full_data.to_numpy()
        concentrations = spectrums_data_np[:, 2:8]
        spectrums = spectrums_data_np[:, 8:]
        return None, concentrations, spectrums, spectrums_full_data

    return read_raw_file(path)


def read_raw_file(path):
    with open(path, encoding="utf-8") as file:
        lines = {index: line.split() for index, line in enumerate(file)}

    if "Start" not in lines[0]:
        return None

    spectrum_full_data = list(lines.values())

    indices_to_pop = []
    for index, line in lines.items():
        if len(line) != 2051:
            indices_to_pop.append(index)

    for index in indices_to_pop:
        lines.pop(index)

    for index, line in lines.items():
        lines[index] = list(map(float, line[3:]))

    indices = list(lines.keys())
    spectrums = np.array(list(lines.values()))

    return indices, None, spectrums, spectrum_full_data


def get_integer_from_string(string):
    if not string:
        return None
    integer_as_string = "".join([char for char in string if char.isdigit()])

    if integer_as_string == "":
        return None

    return int(integer_as_string)


def get_float_from_string(string):
    if not string:
        return None

    float_as_string = "".join(
        [char for char in string if char.isdigit() or char == "-" or char == "."]
    )

    if float_as_string == "":
        return None

    try:
        return float(float_as_string)
    except:
        return None


class SpectrumData:
    def __init__(
        self,
        indices=None,
        spectrum_full_data=None,
        spectrums=np.zeros(shape=(1, 2048)),
        concentrations=None,
        path=None,
    ):

        if path:
            content = read_file(path)
            if content is not None:
                indices, concentrations, spectrums, spectrum_full_data = content

        self.spectrum_full_data = spectrum_full_data
        self.wavelengths = np.loadtxt("wavelengths")
        self.spectrums = spectrums
        self.concentrations = concentrations

        if self.concentrations is not None:
            self.voltage_and_pressure = self.spectrum_full_data.to_numpy()[:, :2]
        else:
            self.voltage_and_pressure = None

        self.index = indices[0] if indices else 0
        self.indices = indices
        self.data_len = spectrums.shape[0]

    def get_spectrum(self):
        if self.indices:
            return self.spectrums[self.indices.index(self.index), :]
        return self.spectrums[self.index, :]

    def get_full_spectrum_data(self):
        if self.indices:
            return self.spectrums[self.indices.index(self.index), :]
        if self.voltage_and_pressure is not None:
            return np.concatenate([self.voltage_and_pressure[self.index, :], self.spectrums[self.index, :]])
        else:
            return self.spectrums[self.index, :]

    def get_concentrations(self):
        if self.concentrations is not None:
            return self.concentrations[self.index, :]
        return np.zeros((6,))

    def set_index(self, index):
        if index is None:
            return
        if self.indices:
            if index in self.indices:
                self.index = index
            return
        if index >= self.data_len or index < 0:
            return
        self.index = index
        return

    def set_next_index(self):
        if self.indices:
            next_index = self.indices.index(self.index) + 1
            if next_index >= len(self.indices):
                self.index = self.indices[0]
                return
            else:
                self.index = self.indices[next_index]
                return

        self.index = self.index + 1
        if self.index >= self.data_len:
            self.index = 0

    def set_previous_index(self):
        if self.indices:
            previous_index = self.indices.index(self.index) - 1
            if previous_index < 0:
                self.index = self.indices[-1]
                return
            else:
                self.index = self.indices[previous_index]
                return

        self.index = self.index - 1
        if self.index < 0:
            self.index = self.data_len - 1


class ConcentrationsPredictor:
    def __init__(self, path=None):
        if not path:
            self.model=None
            return

        model = tf.keras.models.load_model(path, custom_objects={"custom_loss": custom_loss}) if path else None
        input_shape = model.layers[0].get_input_at(0).get_shape().as_list()[1]
        output_shape = model.layers[-1].get_output_at(0).get_shape().as_list()[1]

        if output_shape != 6:
            self.model = None
            return
        if input_shape == 2050 or input_shape == 2048:
            self.model = model
            self.input_shape = input_shape
            return
        self.model = None
        return

    def predict_concentrations(self, spectrum):
        if self.model is not None:
            if self.input_shape == 2048:
                if spectrum.shape[0] == 2050:
                    return self.model.predict(spectrum[2:].reshape(1, 2048))[0]
                else:
                    return self.model.predict(spectrum.reshape(1, 2048))[0]
            if self.input_shape == 2050:
                if spectrum.shape[0] == 2050:
                    return self.model.predict(spectrum.reshape(1, 2050))[0]
        return np.zeros((6,))


def custom_loss(y_true, y_pred):
    mse = keras.mean(keras.square(y_true - y_pred), axis=-1)
    sum_constraint = keras.square(keras.sum(y_pred, axis=-1) - 1)

    return mse + sum_constraint
