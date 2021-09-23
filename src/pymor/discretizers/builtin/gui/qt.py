# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Visualization of grid data using Qt.

This module provides a few methods and classes for visualizing data
associated to grids. We use the `Qt <http://www.qt-project.org>`_ widget
toolkit for the GUI.
"""

import math as m
from pathlib import Path
from tempfile import NamedTemporaryFile
import subprocess
import sys

import numpy as np

from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.core.exceptions import QtMissing, IOLibsMissing
from pymor.core.pickle import dump
from pymor.discretizers.builtin.grids.vtkio import write_vtk
from pymor.discretizers.builtin.gui import vmin_vmax_vectorarray
from pymor.discretizers.builtin.gui.jupyter.vista import PyVistaPatchWidget
from pymor.discretizers.builtin.gui.matplotlib import Matplotlib1DWidget, MatplotlibPatchWidget
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


@defaults('method')
def background_visualization_method(method='ipython_if_possible'):
    assert method in ('ipython', 'ipython_if_possible', 'pymor-vis')

    if getattr(sys, '_called_from_test', False):
        return 'ipython'
    elif method == 'ipython_if_possible':
        try:
            from IPython import get_ipython
            ip = get_ipython()
        except ImportError:
            ip = None
        if ip is None:
            return 'pymor-vis'
        else:
            return 'ipython'
    else:
        return method


def _launch_qt_app(main_window_factory, block):
    """Wrapper to display plot in a separate process."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    if getattr(sys, '_called_from_test', False):
        QTimer.singleShot(1000, app.quit)
        block = True

    if not block:
        try:
            from IPython import get_ipython
            ip = get_ipython()
        except ImportError:
            ip = None
        if ip is None:
            logger = getLogger('pymor.discretizers.builtin.gui.qt')
            logger.warn('Not running within IPython. Falling back to blocking visualization.')
            block = True
        else:
            ip.run_line_magic('gui', 'qt')

    main_window = main_window_factory()
    main_window.show()

    if block:
        app.exec_()
    else:
        global _qt_app
        _qt_app = app                 # deleting the app ref somehow closes the window
        _qt_windows.add(main_window)  # need to keep ref to keep window alive


if config.HAVE_QT:
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QLCDNumber,
                                QAction, QStyle, QToolBar, QLabel, QFileDialog, QMessageBox)
    from qtpy.QtCore import Qt, QTimer

    class PlotMainWindow(QWidget):
        """Base class for plot main windows."""

        def __init__(self, vecarray_tuple, plot_panel, limits, length=1, title=None):
            assert all(isinstance(v, VectorArray) for v in vecarray_tuple)
            super().__init__()

            layout = QVBoxLayout()

            if title:
                title = QLabel('<b>' + title + '</b>')
                title.setAlignment(Qt.AlignHCenter)
                layout.addWidget(title)
            layout.addWidget(plot_panel)

            plot_panel.set(vecarray_tuple, limits)

            if length > 1:
                hlayout = QHBoxLayout()

                self.slider = QSlider(Qt.Horizontal)
                self.slider.setMinimum(0)
                self.slider.setMaximum(length - 1)
                self.slider.setTickPosition(QSlider.TicksBelow)
                hlayout.addWidget(self.slider)

                lcd = QLCDNumber(m.ceil(m.log10(length)))
                lcd.setDecMode()
                lcd.setSegmentStyle(QLCDNumber.Flat)
                hlayout.addWidget(lcd)

                layout.addLayout(hlayout)

                hlayout = QHBoxLayout()

                toolbar = QToolBar()
                self.a_play = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), 'Play', self)
                self.a_play.setCheckable(True)
                self.a_rewind = QAction(self.style().standardIcon(QStyle.SP_MediaSeekBackward), 'Rewind', self)
                self.a_toend = QAction(self.style().standardIcon(QStyle.SP_MediaSeekForward), 'End', self)
                self.a_step_backward = QAction(self.style().standardIcon(QStyle.SP_MediaSkipBackward),
                                               'Step Back', self)
                self.a_step_forward = QAction(self.style().standardIcon(QStyle.SP_MediaSkipForward), 'Step', self)
                self.a_loop = QAction(self.style().standardIcon(QStyle.SP_BrowserReload), 'Loop', self)
                self.a_loop.setCheckable(True)
                toolbar.addAction(self.a_play)
                toolbar.addAction(self.a_rewind)
                toolbar.addAction(self.a_toend)
                toolbar.addAction(self.a_step_backward)
                toolbar.addAction(self.a_step_forward)
                toolbar.addAction(self.a_loop)
                if hasattr(self, 'save'):
                    self.a_save = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save', self)
                    toolbar.addAction(self.a_save)
                    self.a_save.triggered.connect(self.save)
                hlayout.addWidget(toolbar)

                self.speed = QSlider(Qt.Horizontal)
                self.speed.setMinimum(0)
                self.speed.setMaximum(100)
                hlayout.addWidget(QLabel('Speed:'))
                hlayout.addWidget(self.speed)

                layout.addLayout(hlayout)

                self.timer = QTimer()
                self.timer.timeout.connect(self.update_solution)

                self.slider.valueChanged.connect(self.slider_changed)
                self.slider.valueChanged.connect(lcd.display)
                self.speed.valueChanged.connect(self.speed_changed)
                self.a_play.toggled.connect(self.toggle_play)
                self.a_rewind.triggered.connect(self.rewind)
                self.a_toend.triggered.connect(self.to_end)
                self.a_step_forward.triggered.connect(self.step_forward)
                self.a_step_backward.triggered.connect(self.step_backward)

                self.speed.setValue(50)

            elif hasattr(self, 'save'):
                hlayout = QHBoxLayout()
                toolbar = QToolBar()
                self.a_save = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save', self)
                toolbar.addAction(self.a_save)
                hlayout.addWidget(toolbar)
                layout.addLayout(hlayout)
                self.a_save.triggered.connect(self.save)

            self.setLayout(layout)
            self.plot_panel = plot_panel
            self.vecarray_tuple = vecarray_tuple
            self.length = length

        def slider_changed(self, ind):
            self.plot_panel.step(ind)

        def speed_changed(self, val):
            self.timer.setInterval(val * 20)

        def update_solution(self):
            ind = self.slider.value() + 1
            if ind >= self.length:
                if self.a_loop.isChecked():
                    ind = 0
                else:
                    self.a_play.setChecked(False)
                    return
            self.slider.setValue(ind)

        def toggle_play(self, checked):
            if checked:
                if self.slider.value() + 1 == self.length:
                    self.slider.setValue(0)
                self.timer.start()
            else:
                self.timer.stop()

        def rewind(self):
            self.slider.setValue(0)

        def to_end(self):
            self.a_play.setChecked(False)
            self.slider.setValue(self.length - 1)

        def step_forward(self):
            self.a_play.setChecked(False)
            ind = self.slider.value() + 1
            if ind == self.length and self.a_loop.isChecked():
                ind = 0
            if ind < self.length:
                self.slider.setValue(ind)

        def step_backward(self):
            self.a_play.setChecked(False)
            ind = self.slider.value() - 1
            if ind == -1 and self.a_loop.isChecked():
                ind = self.length - 1
            if ind >= 0:
                self.slider.setValue(ind)

        def closeEvent(self, event):
            try:
                self.deleteLater()
                _qt_windows.remove(self)
            except KeyError:
                pass  # we should be in blocking mode ...
            event.accept()


_qt_app = None
_qt_windows = set()


def visualize_patch(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, backend='pyvista', block=False, columns=2):
    """Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.

    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
        as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
        provided, in which case a subplot is created for each entry of the tuple. The
        lengths of all arrays have to agree.
    bounding_box
        A bounding box in which the grid is contained.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 2).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    separate_colorbars
        If `True`, use separate colorbars for each subplot.
    rescale_colorbars
        If `True`, rescale colorbars to data in each frame.
    backend
        Plot backend to use ('gl' or 'matplotlib').
    block
        If `True`, block execution until the plot window is closed.
    columns
        The number of columns in the visualizer GUI in case multiple plots are displayed
        at the same time.
    """
    if not config.HAVE_QT:
        raise QtMissing()

    assert backend in {'pyvista', 'matplotlib'}

    if not block:
        if background_visualization_method() == 'pymor-vis':
            data = dict(dim=2,
                        grid=grid, U=U, bounding_box=bounding_box, codim=codim, title=title, legend=legend,
                        separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                        backend=backend, columns=columns)
            with NamedTemporaryFile(mode='wb', delete=False) as f:
                dump(data, f)
                filename = f.name
            subprocess.Popen(['python', '-m', 'pymor.scripts.pymor_vis', '--delete', filename])
            return

    if backend == 'pyvista':
        if not config.HAVE_PYVISTA:
            logger = getLogger('pymor.discretizers.builtin.gui.qt.visualize_patch')
            logger.warning('import of PyVista failed, falling back to matplotlib; rendering will be slow')
            backend = 'matplotlib'
        if backend == 'matplotlib' and not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')
    else:
        if grid.dim == 1:
            return visualize_matplotlib_1d(grid=grid, U=U, codim=1, title=title, legend=legend,
                                           separate_plots=separate_colorbars, block=block)
        if not config.HAVE_MATPLOTLIB:
            raise ImportError('cannot visualize: import of matplotlib failed')

    # TODO extract class
    class MainWindow(PlotMainWindow):
        def __init__(self, grid, UT, bounding_box, codim, title, legend, separate_colorbars, rescale_colorbars, backend):
            if isinstance(UT, VectorArray):
                vecarray_tuple = (UT,)
            else:
                vecarray_tuple = UT

            assert (all(isinstance(u, VectorArray) for u in vecarray_tuple)
                    and all(len(u) == len(vecarray_tuple[0]) for u in vecarray_tuple))
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(U)
            if not separate_colorbars and len(vecarray_tuple) > 1:
                l = getLogger('pymor.discretizers.builtin.gui.qt.visualize_patch')
                l.warning('separate_colorbars=False not supported')
            if backend == 'pyvista':
                widget = PyVistaPatchWidget
            else:
                widget = MatplotlibPatchWidget
                separate_colorbars = True
            limits = vmin_vmax_vectorarray(vecarray_tuple, separate_colorbars=separate_colorbars,
                                                    rescale_colorbars=rescale_colorbars)

            class PlotPanel(QWidget):
                def __init__(self):
                    super().__init__()
                    layout = QHBoxLayout()
                    plot_layout = QGridLayout()
                    plots = [widget(parent=self, U=u, limits=limits, grid=grid, bounding_box=bounding_box,
                                    codim=codim)
                             for u in vecarray_tuple]
                    if legend:
                        for i, plot, l in zip(range(len(plots)), plots, legend):
                            subplot_layout = QVBoxLayout()
                            caption = QLabel(l)
                            caption.setAlignment(Qt.AlignHCenter)
                            subplot_layout.addWidget(caption)
                            if not separate_colorbars or backend == 'matplotlib':
                                subplot_layout.addWidget(plot)
                            else:
                                hlayout = QHBoxLayout()
                                hlayout.addWidget(plot)
                                subplot_layout.addLayout(hlayout)
                            plot_layout.addLayout(subplot_layout, int(i/columns), (i % columns), 1, 1)
                    else:
                        for i, plot in zip(range(len(plots)), plots):
                            if not separate_colorbars or backend == 'matplotlib':
                                plot_layout.addWidget(plot, int(i/columns), (i % columns), 1, 1)
                            else:
                                hlayout = QHBoxLayout()
                                hlayout.addWidget(plot)
                                plot_layout.addLayout(hlayout, int(i/columns), (i % columns), 1, 1)
                    layout.addLayout(plot_layout)
                    self.setLayout(layout)
                    self.plots = plots

                def set(self, U, limits):
                    self.U = U
                    self.limits = limits
                    for u, plot in zip(self.U, self.plots):
                        plot.set(u, limits=limits)

                def step(self, ind):
                    for plot in self.plots:
                        plot.step(ind)

            super().__init__(vecarray_tuple, PlotPanel(), limits=limits, title=title, length=len(vecarray_tuple[0]))
            self.grid = grid
            self.codim = codim

        def save(self):
            if not config.HAVE_VTKIO:
                msg = QMessageBox(QMessageBox.Critical, 'Error', str(IOLibsMissing()))
                msg.exec_()
                return
            filename = Path(QFileDialog.getSaveFileName(self, 'Save as vtk file')[0])
            base_name = filename.stem
            if base_name:
                if len(self.vecarray_tuple) == 1:
                    write_vtk(self.grid, self.vecarray_tuple[0], base_name, codim=self.codim)
                else:
                    for i, u in enumerate(self.vecarray_tuple):
                        write_vtk(self.grid, u, f'{base_name}-{i}',
                                  codim=self.codim)

    _launch_qt_app(lambda: MainWindow(grid, U, bounding_box, codim, title=title, legend=legend,
                                      separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                      backend=backend),
                   block)


def visualize_matplotlib_1d(grid, U, codim=1, title=None, legend=None, separate_plots=False, block=False):
    """Visualize scalar data associated to a one-dimensional |Grid| as a plot.

    The grid's |ReferenceElement| must be the line. The data can either
    be attached to the subintervals or vertices of the grid.

    Parameters
    ----------
    grid
        The underlying |Grid|.
    U
        |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
        as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
        provided, in which case several plots are made into the same axes. The
        lengths of all arrays have to agree.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 1).
    title
        Title of the plot.
    legend
        Description of the data that is plotted. Most useful if `U` is a tuple in which
        case `legend` has to be a tuple of strings of the same length.
    separate_plots
        If `True`, use subplots to visualize multiple |VectorArrays|.
    block
        If `True`, block execution until the plot window is closed.
    """
    if not config.HAVE_QT:
        raise QtMissing()
    if not config.HAVE_MATPLOTLIB:
        raise ImportError('cannot visualize: import of matplotlib failed')

    if not block:
        if background_visualization_method() == 'pymor-vis':
            data = dict(dim=1,
                        grid=grid, U=U, codim=codim, title=title, legend=legend,
                        separate_plots=separate_plots)
            with NamedTemporaryFile(mode='wb', delete=False) as f:
                dump(data, f)
                filename = f.name
            subprocess.Popen(['python', '-m', 'pymor.scripts.pymor_vis', '--delete', filename])
            return

    class MainWindow(PlotMainWindow):
        def __init__(self, grid, U, codim, title, legend, separate_plots):
            if isinstance(U, VectorArray):
                vecarray_tuple = (U,)
            else:
                vecarray_tuple = U
            assert (all(isinstance(u, VectorArray) for u in vecarray_tuple)
                    and all(len(u) == len(vecarray_tuple[0]) for u in vecarray_tuple))
            if isinstance(legend, str):
                legend = (legend,)
            assert legend is None or isinstance(legend, tuple) and len(legend) == len(vecarray_tuple)

            limits = vmin_vmax_vectorarray(vecarray_tuple, separate_colorbars=False, rescale_colorbars=False)

            plot_widget = Matplotlib1DWidget(vecarray_tuple, parent=None, grid=grid, count=len(vecarray_tuple),
                                             limits=limits, legend=legend, codim=codim, separate_plots=separate_plots)
            super().__init__(vecarray_tuple, plot_widget, limits=limits, title=title, length=len(vecarray_tuple[0]))
            self.grid = grid

    _launch_qt_app(lambda: MainWindow(grid, U, codim, title=title, legend=legend, separate_plots=separate_plots), block)
