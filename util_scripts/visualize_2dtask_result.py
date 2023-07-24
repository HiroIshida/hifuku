import argparse
from typing import Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from hifuku.domain import DummyDomain
from hifuku.library.core import SolutionLibrary
from hifuku.script_utils import load_library

domain = DummyDomain  # global


def sample_colors(n_colors, cmap_name="hsv"):
    cmap = plt.cm.get_cmap(cmap_name)  # get the colormap
    colors = cmap(np.linspace(0, 1, n_colors))  # sample the colormap
    return colors


class SingleLibraryPlotter:
    singleton: SolutionLibrary
    iter_values: np.ndarray
    sdf_values: np.ndarray
    mesh: Tuple[np.ndarray, np.ndarray]

    def __init__(self, library: SolutionLibrary, n_grid: int):
        assert len(library.predictors) == 1
        pred = library.predictors[0]

        xlin = np.linspace(-2.2, 2.2, n_grid)
        ylin = np.linspace(-2.2, 2.2, n_grid)
        X, Y = np.meshgrid(xlin, ylin)

        pts = list(zip(X.flatten(), Y.flatten()))
        pts = np.array(pts)
        pts_torch = torch.from_numpy(np.array(pts)).float()
        mesh_torch = torch.empty((len(pts), 0))

        iter_preds, _ = pred.forward((mesh_torch, pts_torch))
        iter_values = iter_preds.cpu().detach().numpy().flatten() - config.n_max_call

        world = domain.task_type.get_world_type().sample()
        sdf = world.get_exact_sdf()
        sdf_values = -sdf(pts)

        self.singleton = library
        self.iter_values = iter_values
        self.sdf_values = sdf_values
        self.mesh = (X, Y)

    def visualize(
        self,
        fax,
        margin: float,
        contourf_kwargs_: Optional[Dict] = None,  # if None, don't plot, {} if use default
        contour_kwargs_: Optional[Dict] = None,  # if None, don't plot, {} if use default
        center_kwargs_: Optional[Dict] = None,
    ) -> None:
        fig, ax = fax
        X, Y = self.mesh
        iters_main = np.maximum(self.sdf_values, self.iter_values + margin)
        iters_main = iters_main.reshape(X.shape)

        if contourf_kwargs_ is not None:
            contourf_kwargs = {"colors": ["gray"], "alpha": 0.2}
            for key, val in contourf_kwargs_.items():
                contourf_kwargs[key] = val
            ax.contourf(X, Y, iters_main, levels=[-np.inf, 0], **contourf_kwargs)

        if contour_kwargs_ is not None:
            contour_kwargs = {"colors": ["black"], "linewidths": 2.5}
            for key, val in contour_kwargs_.items():
                contour_kwargs[key] = val
            ax.contour(X, Y, iters_main, levels=[0], **contour_kwargs)

        if center_kwargs_ is not None:
            center_kwargs = {"c": "black"}
            for key, val in center_kwargs_.items():
                center_kwargs[key] = val
            center = self.singleton.predictors[0].initial_solution.numpy()[-1]
            ax.scatter(center[0], center[1], label="selected", **center_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("-grid", type=int, default=600)
    parser.add_argument("-mode", type=str, default="step0")
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    n_step: Optional[int] = args.n
    assert n_step is not None
    mode: Literal["step1", "step2", "step3", "step0"] = args.mode

    # domain = EightRooms_SQP_Domain
    # domain = EightRooms_Lightning_Domain
    domain = DummyDomain
    config = domain.solver_config
    world = domain.task_type.get_world_type().sample()

    lib = load_library(domain, "cpu")

    singletons = lib.unbundle()
    plotters = [SingleLibraryPlotter(s, n_grid=args.grid) for s in singletons]

    fig, ax = plt.subplots()
    world.visualize((fig, ax))

    if mode == "step0":
        latest_margins = lib._margins_history[n_step - 1]
        assert n_step > 0
        for i in range(n_step - 1):
            margin = latest_margins[i]
            plotters[i].visualize((fig, ax), margin, {}, {}, {})
        plotters[n_step - 1].visualize(
            (fig, ax), latest_margins[n_step - 1], {"colors": ["red"]}, {}, {}
        )
    elif mode == "step1":
        candidates = lib._candidates_history[n_step]  # not n_step - 1
        latest_margins = lib._margins_history[n_step - 1]
        for i in range(n_step):
            margin = latest_margins[i]
            plotters[i].visualize((fig, ax), margin, {}, {}, {})
        # plot active sampling
        pts = np.array([traj.numpy()[-1] for traj in candidates])
        ax.scatter(pts[:, 0], pts[:, 1], c="b", s=1, label="candidates")
        selected = lib.predictors[n_step].initial_solution.numpy()[-1]
        ax.scatter(selected[0], selected[1], c="orange", marker="*", s=300, label="selected")

    elif mode == "step2":
        latest_margins = lib._margins_history[n_step - 1]
        if n_step > 0:
            for i in range(n_step):
                margin = latest_margins[i]
                plotters[i].visualize((fig, ax), margin, {}, {}, {})

        plotters[n_step].visualize(
            (fig, ax), 0.0, None, {"colors": "red", "linestyles": "dashed"}, {"c": "red"}
        )

    elif mode == "step3":
        latest_margins = lib._margins_history[n_step]
        prev_margins = lib._margins_history[n_step - 1]
        if n_step > 0:
            for i in range(n_step):
                plotters[i].visualize((fig, ax), latest_margins[i], None, {}, {})
                plotters[i].visualize((fig, ax), prev_margins[i], None, {"linestyles": "dashed"})

        # latest margin
        plotters[n_step].visualize(
            (fig, ax), 0.0, None, {"colors": "red", "linestyles": "dashed"}, {}
        )
        plotters[n_step].visualize(
            (fig, ax), latest_margins[n_step], None, {"colors": "red"}, {"c": "red"}
        )

    ax.set_xlim(-1.0, 2.2)
    ax.set_ylim(-1.8, 1.6)

    if args.save:
        if args.legend:
            legend = ax.legend()
            legend.get_frame().set_alpha(1.0)
            file_name = "./figs/legend-algo-expl-seq-{}.png".format(n_step)
        else:
            file_name = "./figs/algo-{}-{}.png".format(n_step, mode)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", which="both", length=0)
        plt.tight_layout(pad=0)
        plt.savefig(file_name, dpi=300)
    else:
        plt.show()
