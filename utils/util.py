import random
import torch
import numpy as np
import os
import pandas as pd
import json

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as plt_Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from pathlib import Path
from ast import literal_eval

def get_results_df(result_csv, aggregate=True, by_attitude=False):
    results_path= Path("./results/zero/")
    control_results = Path("./results/control/zero/")
    persona_cat_dict = json.load(open("./data/persona_cat.json", "r"))

    rename_model = {
    "gpt-4-0125-preview": "GPT-4",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral",
    "zephyr-7b-beta": "Zephyr",
    "Mistral-7B-Instruct-v0.2": "Mistral-inst",
    "gemma-7b-it": "Gemma-7b-inst",
    "gemma-2b-it": "Gemma-2b-inst"
}

    models = os.listdir(results_path)
    results = {model: pd.read_csv(results_path/model/result_csv, index_col=0) for model in models}

    all_results_dic = {}
    for model, df in results.items():
        if "attitude_scores"  in result_csv:
            results[model] = results[model].iloc[:,:5]
        elif "attitude" in result_csv:
            for column in df.columns:
                results[model][column] = results[model][column].str.replace(" ", ",")
                results[model][column] = results[model][column].apply(literal_eval)
        for persona in results[model].index.tolist():
            if by_attitude:
                all_results =  results[model].loc[persona]
                all_results_dic.setdefault(model, {})[persona] = all_results
            elif "attitude" in result_csv and "scores" not in result_csv:
                all_results =  np.concatenate(results[model].loc[persona].values.tolist(),axis=0)
                if aggregate: 
                    all_results = all_results.mean()
                all_results_dic.setdefault(model, {})[persona] = all_results
            else:
                all_results =  np.array(results[model].loc[persona].values.tolist())
                all_results_dic.setdefault(model, {})[persona] = all_results
    
    results_df = pd.DataFrame.from_dict(all_results_dic)
    
    results_df["persona_cat"] =  [persona_cat_dict[persona] for persona in results_df.index]
    
    results = {model: pd.read_csv(control_results/model/result_csv, index_col=0) for model in models}
    
    all_results_dic = {}
    for model, df in results.items():
        if "attitude_scores" in result_csv:
            results[model] = results[model].iloc[:,:5]
        elif "attitude" in result_csv:
            for column in df.columns:
                results[model][column] = results[model][column].str.replace(" ", ",")
                results[model][column] = results[model][column].apply(literal_eval)
        for persona in results[model].index.tolist():
            if by_attitude:
                all_results =  results[model].loc[persona]
                all_results_dic.setdefault(model, {})[persona] = all_results
            elif "attitude" in result_csv and "scores" not in result_csv:
                all_results =  np.concatenate(results[model].loc[persona].values.tolist(),axis=0)
                if aggregate: all_results = all_results.mean()
                all_results_dic.setdefault(model, {})[persona] = all_results
            else:
                all_results =  np.array(results[model].loc[persona].values.tolist())
                all_results_dic.setdefault(model, {})[persona] = all_results
    
    results_df_control = pd.DataFrame.from_dict(all_results_dic); results_df
    
    results_df_control["persona_cat"] = "control"

    return pd.concat([results_df, results_df_control], axis=0).rename(columns=rename_model)


def initialize_seeds(seed=42):
    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.detesrministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return plt_Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=plt_Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta