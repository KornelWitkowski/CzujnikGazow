import numpy as np
import ntpath

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from PyQt5.QtGui import QFont, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from backend import SpectrumData, ConcentrationsPredictor
from backend import (
    get_float_from_string,
    get_integer_from_string,
    compare_numpy_arrays_columns_with_nan,
    sort_array_by_column,
)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setGeometry(200, 200, 1650, 1000)
        self.setWindowTitle("Przeglądacz")
        self.setWindowIcon(QIcon("icon.png"))

        self.frame = QFrame(self)
        self.layout = QGridLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        self.spectrum_data = SpectrumData()
        self.concentrations_predictor = ConcentrationsPredictor()

        self.x_min = self.spectrum_data.wavelengths[0]
        self.x_max = self.spectrum_data.wavelengths[-1]
        self.y_min = -200
        self.y_max = 10000

        self.file_select_panel = FileSelectPanel(self)
        self.file_select_panel.setFixedSize(450, 150)
        self.layout.addWidget(self.file_select_panel, *(0, 0, 1, 1))

        self.axes_setting_panel = AxesSettingPanel(self)
        self.axes_setting_panel.setFixedSize(450, 200)
        self.layout.addWidget(self.axes_setting_panel, *(1, 0, 1, 1))

        self.file_line_panel = FileLinePanel(self)
        self.file_line_panel.setFixedSize(450, 150)
        self.layout.addWidget(self.file_line_panel, *(2, 0, 1, 1))

        self.file_preview_table = FilePreviewTable(self)
        self.file_preview_table.setFixedSize(450, 400)
        self.layout.addWidget(self.file_preview_table, *(3, 0, 1, 1))

        self.spectrum_diagram = SpectrumDiagram(self)
        self.layout.addWidget(self.spectrum_diagram, *(0, 1, 10, 1))

    def refresh_plots(self):
        self.spectrum_diagram.refresh_plots(self)
        self.spectrum_diagram.draw()
        self.file_line_panel.line_number_title_label.setText(
            "Nr wiersza: " + str(self.spectrum_data.index + 1)
        )
        return

    def draw_new_plots(self):
        self.spectrum_diagram = SpectrumDiagram(self)
        self.layout.addWidget(self.spectrum_diagram, *(0, 1, 10, 1))


class SpectrumDiagram(FigureCanvas):
    def __init__(self, main_window):

        self.fig = Figure(figsize=(5, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_xlabel(r"$\lambda$ [nm]", fontsize=14)
        self.ax1.set_ylabel("Liczba zliczeń", fontsize=14)
        self.ax1.set_title("Widmo", fontsize=16)

        self.ax1.margins(0.1, 0.1)
        self.ax1.grid(which="both", axis="both")
        plot_spectrum_ref = self.ax1.plot(
            main_window.spectrum_data.wavelengths,
            main_window.spectrum_data.get_spectrum(),
        )
        self.plot_spectrum_ref = plot_spectrum_ref[0]
        self.ax1.set_xlim(main_window.x_min, main_window.x_max)
        self.ax1.set_ylim(main_window.y_min, main_window.y_max)

        self.labels = ["CO2", "N", "O", "Ar", "He", "Ne"]

        self.ax2 = self.fig.add_subplot(222)
        self.ax2.grid(True)
        self.ax2.set_ylim(0, 1.1)
        self.ax2.set_ylabel("Stężenie", fontsize=14)
        self.ax2.set_title("Rzeczywiste stężenie", fontsize=16)
        self.plot_concentrations_ref = self.ax2.bar(
            self.labels, main_window.spectrum_data.get_concentrations(), color="green"
        )

        self.ax3 = self.fig.add_subplot(224)
        self.ax3.grid(True)
        self.ax3.set_ylim(0, 1.1)
        self.ax3.set_ylabel("Stężenie", fontsize=14)
        self.ax3.set_title("Przewidywane stężenie", fontsize=16)
        self.plot_prediction_ref = self.ax3.bar(
            self.labels,
            main_window.concentrations_predictor.predict_concentrations(
                main_window.spectrum_data.get_full_spectrum_data()
            ),
            color="red",
        )

        FigureCanvas.__init__(self, self.fig)

        return

    def refresh_plots(self, main_window):
        self.plot_spectrum_ref.set_ydata(main_window.spectrum_data.get_spectrum())

        for i, concentrations in enumerate(
            main_window.spectrum_data.get_concentrations()
        ):
            self.plot_concentrations_ref[i].set_height(concentrations)

        for i, predicted_concentrations in enumerate(
            main_window.concentrations_predictor.predict_concentrations(
                main_window.spectrum_data.get_full_spectrum_data()
            )
        ):
            self.plot_prediction_ref[i].set_height(predicted_concentrations)

        return


class FilePreviewTable(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        self.layout = QGridLayout(self)

        self.spectrum_data = None
        self.row_number = 0
        self.columns_labels = []

        self.file_select_label = QLabel("Podgląd danych")
        self.file_select_label.setFont(QFont("Arial", 16))
        self.layout.addWidget(self.file_select_label, *(0, 0, 1, 1))

        self.blank_label = QLabel("")
        self.layout.addWidget(self.blank_label, *(1, 0, 1, 1))

        self.table_widget = QTableWidget()
        self.table_widget.setFont(QFont("Arial", 7))
        self.table_widget.setRowCount(self.row_number)
        self.table_widget.setColumnCount(9)
        self.vertical_header = self.table_widget.verticalHeader()
        self.vertical_header.setVisible(False)

        self.header = self.table_widget.horizontalHeader()
        self.header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.horizontalHeader().sectionClicked.connect(
            self.sort_table_by_column
        )

        self.table_widget.move(0, 0)
        self.layout.addWidget(self.table_widget, *(2, 0, 1, 1))

    def write_prepared_file_content(self):
        main_window = self.parent().parent()

        self.spectrum_data = main_window.spectrum_import numpy as np
import ntpath

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from PyQt5.QtGui import QFont, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from backend import SpectrumData, ConcentrationsPredictor
from backend import (
    get_float_from_string,
    get_integer_from_string,
    compare_numpy_arrays_columns_with_nan,
    sort_array_by_column,
)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setGeometry(200, 200, 1650, 1000)
        self.setWindowTitle("Przeglądacz")
        self.setWindowIcon(QIcon("icon.png"))

        self.frame = QFrame(self)
        self.layout = QGridLayout()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        self.spectrum_data = SpectrumData()
        self.concentrations_predictor = ConcentrationsPredictor()

        self.x_min = self.spectrum_data.wavelengths[0]
        self.x_max = self.spectrum_data.wavelengths[-1]
        self.y_min = -200
        self.y_max = 10000

        self.file_select_panel = FileSelectPanel(self)
        self.file_select_panel.setFixedSize(450, 150)
        self.layout.addWidget(self.file_select_panel, *(0, 0, 1, 1))

        self.axes_setting_panel = AxesSettingPanel(self)
        self.axes_setting_panel.setFixedSize(450, 200)
        self.layout.addWidget(self.axes_setting_panel, *(1, 0, 1, 1))

        self.file_line_panel = FileLinePanel(self)
        self.file_line_panel.setFixedSize(450, 150)
        self.layout.addWidget(self.file_line_panel, *(2, 0, 1, 1))

        self.file_preview_table = FilePreviewTable(self)
        self.file_preview_table.setFixedSize(450, 400)
        self.layout.addWidget(self.file_preview_table, *(3, 0, 1, 1))

        self.spectrum_diagram = SpectrumDiagram(self)
        self.layout.addWidget(self.spectrum_diagram, *(0, 1, 10, 1))

    def refresh_plots(self):
        self.spectrum_diagram.refresh_plots(self)
        self.spectrum_diagram.draw()
        self.file_line_panel.line_number_title_label.setText(
            "Nr wiersza: " + str(self.spectrum_data.index + 1)
        )
        return

    def draw_new_plots(self):
        self.spectrum_diagram = SpectrumDiagram(self)
        self.layout.addWidget(self.spectrum_diagram, *(0, 1, 10, 1))


class SpectrumDiagram(FigureCanvas):
    def __init__(self, main_window):

        self.fig = Figure(figsize=(5, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_xlabel(r"$\lambda$ [nm]", fontsize=14)
        self.ax1.set_ylabel("Liczba zliczeń", fontsize=14)
        self.ax1.set_title("Widmo", fontsize=16)

        self.ax1.margins(0.1, 0.1)
        self.ax1.grid(which="both", axis="both")
        plot_spectrum_ref = self.ax1.plot(
            main_window.spectrum_data.wavelengths,
            main_window.spectrum_data.get_spectrum(),
        )
        self.plot_spectrum_ref = plot_spectrum_ref[0]
        self.ax1.set_xlim(main_window.x_min, main_window.x_max)
        self.ax1.set_ylim(main_window.y_min, main_window.y_max)

        self.labels = ["CO2", "N", "O", "Ar", "He", "Ne"]

        self.ax2 = self.fig.add_subplot(222)
        self.ax2.grid(True)
        self.ax2.set_ylim(0, 1.1)
        self.ax2.set_ylabel("Stężenie", fontsize=14)
        self.ax2.set_title("Rzeczywiste stężenie", fontsize=16)
        self.plot_concentrations_ref = self.ax2.bar(
            self.labels, main_window.spectrum_data.get_concentrations(), color="green"
        )

        self.ax3 = self.fig.add_subplot(224)
        self.ax3.grid(True)
        self.ax3.set_ylim(0, 1.1)
        self.ax3.set_ylabel("Stężenie", fontsize=14)
        self.ax3.set_title("Przewidywane stężenie", fontsize=16)
        self.plot_prediction_ref = self.ax3.bar(
            self.labels,
            main_window.concentrations_predictor.predict_concentrations(
                main_window.spectrum_data.get_full_spectrum_data()
            ),
            color="red",
        )

        FigureCanvas.__init__(self, self.fig)

        return

    def refresh_plots(self, main_window):
        self.plot_spectrum_ref.set_ydata(main_window.spectrum_data.get_spectrum())

        for i, concentrations in enumerate(
            main_window.spectrum_data.get_concentrations()
        ):
            self.plot_concentrations_ref[i].set_height(concentrations)

        for i, predicted_concentrations in enumerate(
            main_window.concentrations_predictor.predict_concentrations(
                main_window.spectrum_data.get_full_spectrum_data()
            )
        ):
            self.plot_prediction_ref[i].set_height(predicted_concentrations)

        return


class FilePreviewTable(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        self.layout = QGridLayout(self)

        self.spectrum_data = None
        self.row_number = 0
        self.columns_labels = []

        self.file_select_label = QLabel("Podgląd danych")
        self.file_select_label.setFont(QFont("Arial", 16))
        self.layout.addWidget(self.file_select_label, *(0, 0, 1, 1))

        self.blank_label = QLabel("")
        self.layout.addWidget(self.blank_label, *(1, 0, 1, 1))

        self.table_widget = QTableWidget()
        self.table_widget.setFont(QFont("Arial", 7))
        self.table_widget.setRowCount(self.row_number)
        self.table_widget.setColumnCount(9)
        self.vertical_header = self.table_widget.verticalHeader()
        self.vertical_header.setVisible(False)

        self.header = self.table_widget.horizontalHeader()
        self.header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.horizontalHeader().sectionClicked.connect(
            self.sort_table_by_column
        )

        self.table_widget.move(0, 0)
        self.layout.addWidget(self.table_widget, *(2, 0, 1, 1))

    def write_prepared_file_content(self):
        main_window = self.parent().parent()

        self.spectrum_data = main_window.spectrum_data.spectrum_full_data
        self.spectrum_data_np = self.spectrum_data.to_numpy()
        indices = np.arange(1, self.spectrum_data_np.shape[0] + 1).reshape(
            self.spectrum_data_np.shape[0], 1
        )
        self.spectrum_data_np = np.concatenate(
            (indices, self.spectrum_data_np), axis=1
        )[:, :9]

        self.row_number = self.spectrum_data.shape[0]
        self.table_widget.setRowCount(self.row_number)

        self.columns_labels = ["index"] + self.spectrum_data.columns.to_list()[:8]
        self.table_widget.setHorizontalHeaderLabels(self.columns_labels)
        self.header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.header.setSectionResizeMode(2, QHeaderView.Stretch)

        for i in range(self.spectrum_data_np.shape[0]):
            for j in range(self.spectrum_data_np.shape[1]):
                cell_value = str(round(float(self.spectrum_data_np[i, j]), 3))
                if j == 0:
                    cell_value = str(int(float(cell_value)))
                self.table_widget.setItem(i, j, QTableWidgetItem(cell_value))

    def update_cells(self):
        for i in range(self.spectrum_data_np.shape[0]):
            for j in range(self.spectrum_data_np.shape[1]):
                cell_value = str(round(float(self.spectrum_data_np[i, j]), 3))
                if j == 0:
                    cell_value = str(int(float(cell_value)))

                self.table_widget.item(i, j).setText(cell_value)
        return

    def sort_table_by_column(self, header_number):
        if self.spectrum_data is None:
            return

        sorted_data = sort_array_by_column(
            arr=self.spectrum_data_np, column_num=header_number
        )

        if compare_numpy_arrays_columns_with_nan(
            sorted_data, self.spectrum_data_np, header_number
        ):
            sorted_data = sort_array_by_column(
                arr=self.spectrum_data_np, column_num=header_number, desc=True
            )

        self.spectrum_data_np = sorted_data
        self.update_cells()
        return

    def write_raw_file_content(self):
        main_window = self.parent().parent()

        spectrum_data = main_window.spectrum_data.spectrum_full_data
        self.row_number = len(spectrum_data) if spectrum_data is not None else 0

        self.table_widget.setRowCount(self.row_number)

        for i in range(self.row_number):
            for j in range(min(len(spectrum_data[i]), 9)):
                if j == 0:
                    self.table_widget.setItem(i, j, QTableWidgetItem(str(i + 1)))
                else:
                    self.table_widget.setItem(
                        i, j, QTableWidgetItem(str(spectrum_data[i][j - 1]))
                    )

        if spectrum_data is not None:
            self.header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)


class FileSelectPanel(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        self.layout = QGridLayout(self)

        self.file_select_label = QLabel("Wybór plików")
        self.file_select_label.setFont(QFont("Arial", 16))
        self.layout.addWidget(self.file_select_label, *(0, 0, 1, 1))

        self.blank_label = QLabel("")
        self.layout.addWidget(self.blank_label, *(1, 0, 1, 1))

        self.choose_file_button = QPushButton("Baza danych")
        self.choose_file_button.setFixedSize(150, 30)
        self.choose_file_button.clicked.connect(self.get_spectrums_path)
        self.layout.addWidget(self.choose_file_button, *(2, 0, 1, 2))

        self.current_file_label = QLabel("")
        self.current_file_label.setFixedSize(250, 30)
        self.current_file_label.setStyleSheet("border: 1px solid black;")
        self.current_file_label.setFont(QFont("Arial", 10))
        self.layout.addWidget(self.current_file_label, *(2, 2, 1, 2))

        self.choose_model_button = QPushButton("Sieć neuronowa")
        self.choose_model_button.setFixedSize(150, 30)
        self.choose_model_button.clicked.connect(self.get_model_path)
        self.layout.addWidget(self.choose_model_button, *(3, 0, 1, 2))

        self.current_model_label = QLabel("")
        self.current_model_label.setFixedSize(250, 30)
        self.current_model_label.setStyleSheet("border: 1px solid black;")
        self.current_model_label.setFont(QFont("Arial", 10))
        self.layout.addWidget(self.current_model_label, *(3, 2, 1, 2))

    def get_spectrums_path(self):
        main_window = self.parent().parent()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybór bazy danych",
            "",
            "Wszystkie pliki (*);;Pliki tekstowe (*.txt);; Pliki CSV (*.csv)",
            options=options,
        )
        self.current_file_label.setText(ntpath.basename(file_path))
        main_window.spectrum_data = SpectrumData(path=file_path)
        main_window.refresh_plots()

        self.reset_preview_table(main_window)

        if main_window.spectrum_data.concentrations is not None:
            main_window.file_preview_table.write_prepared_file_content()
            return

        main_window.file_preview_table.write_raw_file_content()
        return

    def get_model_path(self):
        main_window = self.parent().parent()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Wybór sieci neuronowej", "", "Pliki HDF (*.h5)", options=options
        )
        self.current_model_label.setText(ntpath.basename(file_path))
        main_window.concentrations_predictor = ConcentrationsPredictor(path=file_path)
        main_window.refresh_plots()

        return

    def reset_preview_table(self, main_window):
        main_window.file_preview_table.setParent(None)
        main_window.file_preview_table = FilePreviewTable(self)
        main_window.file_preview_table.setFixedSize(450, 400)
        main_window.layout.addWidget(main_window.file_preview_table, *(3, 0, 1, 1))


class AxesSettingPanel(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        self.layout = QGridLayout(self)

        self.axes_setting_label = QLabel("Zakres osi")
        self.axes_setting_label.setFont(QFont("Arial", 16))
        self.axes_setting_label.setFixedSize(150, 30)
        self.layout.addWidget(self.axes_setting_label, *(0, 0, 1, 2))

        self.blank_label1 = QLabel("")
        self.blank_label1.setFixedSize(150, 30)
        self.layout.addWidget(self.blank_label1, *(0, 2, 1, 2))

        self.blank_label2 = QLabel("")
        self.blank_label2.setFixedSize(150, 30)
        self.layout.addWidget(self.blank_label2, *(0, 4, 1, 2))

        self.blank_label3 = QLabel("")
        self.layout.addWidget(self.blank_label3, *(1, 0, 1, 1))

        self.x_min_label = QLabel("x <sub>min</sub>")
        self.x_min_label.setFixedSize(60, 40)
        self.x_min_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.x_min_label, *(2, 0, 1, 1))

        self.x_min_textbox = QLineEdit()
        self.x_min_textbox.setFixedSize(70, 30)
        self.layout.addWidget(self.x_min_textbox, *(2, 1, 1, 1))

        self.x_max_label = QLabel("x <sub>max</sub>")
        self.x_max_label.setFixedSize(60, 30)
        self.x_max_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.x_max_label, *(3, 0, 1, 1))

        self.x_max_textbox = QLineEdit()
        self.x_max_textbox.setFixedSize(70, 30)
        self.layout.addWidget(self.x_max_textbox, *(3, 1, 1, 1))

        self.y_min_label = QLabel("y <sub>min</sub>")
        self.y_min_label.setFixedSize(60, 30)
        self.y_min_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.y_min_label, *(2, 2, 1, 1))

        self.y_min_textbox = QLineEdit()
        self.y_min_textbox.setFixedSize(70, 30)
        self.layout.addWidget(self.y_min_textbox, *(2, 3, 1, 1))

        self.y_max_label = QLabel("y <sub>max</sub>")
        self.y_max_label.setFixedSize(60, 30)
        self.y_max_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.y_max_label, *(3, 2, 1, 1))

        self.y_max_textbox = QLineEdit()
        self.y_max_textbox.setFixedSize(70, 30)
        self.layout.addWidget(self.y_max_textbox, *(3, 3, 1, 1))

        self.blank_line = QLabel("")
        self.blank_line.setFixedSize(70, 30)
        self.layout.addWidget(self.blank_line, *(4, 1, 1, 1))

        self.set_axes_button = QPushButton("Ustaw")
        self.set_axes_button.setFixedSize(70, 30)
        self.set_axes_button.clicked.connect(self.set_diagram_axes_limits)
        self.layout.addWidget(self.set_axes_button, *(5, 1, 1, 1))

        self.reset_axes_button = QPushButton("Reset")
        self.reset_axes_button.setFixedSize(70, 30)
        self.reset_axes_button.clicked.connect(self.reset_diagram_axes_limits)
        self.layout.addWidget(self.reset_axes_button, *(5, 2, 1, 1))

    def reset_diagram_axes_limits(self):
        main_window = self.parent().parent()
        self.set_axes_limits(
            main_window.spectrum_data.wavelengths[0],
            main_window.spectrum_data.wavelengths[-1],
            -200,
            10000,
        )
        main_window.draw_new_plots()

        return

    def set_diagram_axes_limits(self):
        main_window = self.parent().parent()

        x_min = get_float_from_string(self.x_min_textbox.text())
        x_max = get_float_from_string(self.x_max_textbox.text())
        y_min = get_float_from_string(self.y_min_textbox.text())
        y_max = get_float_from_string(self.y_max_textbox.text())

        self.set_axes_limits(x_min, x_max, y_min, y_max)
        main_window.draw_new_plots()

        return

    def set_axes_limits(self, x_min=None, x_max=None, y_min=None, y_max=None):
        main_window = self.parent().parent()

        if (x_min is not None and x_max is not None) and (x_min < x_max):
            main_window.x_min = x_min
            main_window.x_max = x_max

        if (y_min is not None and y_max is not None) and (y_min < y_max):
            main_window.y_min = y_min
            main_window.y_max = y_max

        return


class FileLinePanel(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent=parent)
        self.layout = QGridLayout(self)

        self.line_number_title_label = QLabel("Nr wiersza")
        self.line_number_title_label.setFixedSize(200, 30)
        self.line_number_title_label.setFont(QFont("Arial", 16))
        self.layout.addWidget(self.line_number_title_label, *(0, 0, 1, 1))

        self.blank_label1 = QLabel("")
        self.blank_label1.setFixedSize(250, 30)
        self.layout.addWidget(self.blank_label1, *(0, 1, 1, 4))

        self.blank_label2 = QLabel("")
        self.blank_label2.setFixedSize(250, 30)
        self.layout.addWidget(self.blank_label2, *(0, 4, 1, 4))

        self.previous_spectrum_button = QPushButton("<")
        self.previous_spectrum_button.setFixedSize(50, 30)
        self.previous_spectrum_button.clicked.connect(self.draw_previous_spectrum)
        self.layout.addWidget(self.previous_spectrum_button, *(1, 1, 1, 1))

        self.next_spectrum_button = QPushButton(">")
        self.next_spectrum_button.setFixedSize(50, 30)
        self.next_spectrum_button.clicked.connect(self.draw_next_spectrum)
        self.layout.addWidget(self.next_spectrum_button, *(1, 2, 1, 1))

        self.set_line_textbox = QLineEdit()
        self.set_line_textbox.setFixedSize(50, 30)
        self.layout.addWidget(self.set_line_textbox, *(1, 3, 1, 1))

        self.set_line_button = QPushButton("Ustaw")
        self.set_line_button.setFixedSize(50, 30)
        self.set_line_button.clicked.connect(self.set_line)
        self.layout.addWidget(self.set_line_button, *(1, 4, 1, 1))

    def set_line(self):
        main_window = self.parent().parent()
        index = get_integer_from_string(self.set_line_textbox.text())
        main_window.spectrum_data.set_index(index - 1)
        main_window.refresh_plots()
        return

    def draw_next_spectrum(self):
        main_window = self.parent().parent()
        main_window.spectrum_data.set_next_index()
        main_window.refresh_plots()
        return

    def draw_previous_spectrum(self):
        main_window = self.parent().parent()
        main_window.spectrum_data.set_previous_index()
        main_window.refresh_plots()
        return
