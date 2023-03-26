from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.optimize import minimize
from sympy import symbols


class MarginOptimizer:
    obj: Callable[[np.ndarray], float]
    ineq_const1: Callable[[np.ndarray], float]
    ineq_const2: Callable[[np.ndarray], float]
    ineq_const3: Callable[[np.ndarray], float]
    m_real: int

    def __init__(self, alpha: float, x_real: float, y_real: float, m_real: int):
        x, y, mu_x, mu_y, eps, eps_dash, gamma = symbols(
            "x, y, mu_x, mu_y, eps, eps_dash, gamma", positive=True
        )
        expr = (x - eps_dash) ** 2 * eps / ((x - eps_dash) * (1 + eps) + (y + eps_dash)) - gamma
        tmp = sympy.solve(expr, eps)
        eps_depends_on_gamma = sympy.simplify(tmp[0])

        m, q = symbols("m, q", positive=True)

        gamma_value = sympy.sqrt(sympy.log(2 / q) / (2 * m))

        sym_margin = eps_depends_on_gamma.subs(gamma, gamma_value)
        sym_margin = sym_margin.subs(x, x_real)
        sym_margin = sym_margin.subs(y, y_real)
        sym_margin = sym_margin.subs(m, m_real)
        sym_margin = sympy.simplify(sym_margin)
        obj_tmp = sympy.lambdify((q, eps_dash), sym_margin)

        def obj(inp: np.ndarray) -> float:
            q, eps_dash = inp
            f = obj_tmp(q, eps_dash)
            return f

        def ineq_const1(inp):
            q, eps_dash = inp
            return alpha - (q + 2 * np.exp(-2 * m_real * eps_dash**2))

        def ineq_const2(inp):
            q, eps_dash = inp
            val = (x_real - eps_dash) - np.sqrt(np.log(2 / q) / (2 * m_real))
            return val

        def ineq_const3(inp):
            q, eps_dash = inp
            return 2 * x_real - eps_dash

        self.obj = obj
        self.ineq_const1 = ineq_const1
        self.ineq_const2 = ineq_const2
        self.ineq_const3 = ineq_const3
        self.m_real = m_real

    def optimize(self, grid_search: bool = False) -> float:

        bounds = ((-4, 0), (-4, 0))

        if grid_search:
            q_lin = np.linspace(bounds[0][0], bounds[0][1], 100)
            eps_lin = np.linspace(bounds[1][0], bounds[1][1], 100)
            Q, E = np.meshgrid(q_lin, eps_lin)
            pts = np.array(list(zip(Q.flatten(), E.flatten())))
            val1 = np.array([self.ineq_const1(10**p) for p in pts])
            val2 = np.array([self.ineq_const2(10**p) for p in pts])
            lindices_valid = np.logical_and(val1 > 0.0, val2 > 0.0)
            obj_values = np.array([self.obj(10**p) for p in pts])
            min_value = np.min(obj_values[lindices_valid])
            return min_value

        else:

            def objective(log_variable: np.ndarray) -> float:
                variable = 10**log_variable
                return self.obj(variable)

            def ineq_const(log_variable: np.ndarray) -> np.ndarray:
                variable = 10**log_variable
                val1 = self.ineq_const1(variable)
                val2 = self.ineq_const2(variable)
                return np.array([val1, val2])

            cons = {"type": "ineq", "fun": ineq_const}
            x0 = np.array([bounds[0][0], bounds[1][0]])
            result = minimize(objective, x0, method="trust-constr", bounds=bounds, constraints=cons)
            if not result.success:
                return np.inf

            q = (10**result.x)[0]
            (10**result.x)[1]
            np.log(2 / q) / (2 * self.m_real)

            confidence_half_interval = self.obj(10**result.x)
            return confidence_half_interval

    def plot(self):
        N = 200
        q_lin = np.linspace(-3, 0, N)
        e_lin = np.linspace(-3, 0, N)

        Q, E = np.meshgrid(q_lin, e_lin)
        pts = np.array(list(zip(Q.flatten(), E.flatten())))

        values1 = np.array([self.ineq_const1(10**p) for p in pts])
        mat1 = np.reshape(values1, (N, N))

        values2 = np.array([self.ineq_const2(10**p) for p in pts])
        mat2 = np.reshape(values2, (N, N))

        values4 = np.array([self.obj(10**p) for p in pts])
        mat4 = np.reshape(values4, (N, N))

        fig, ax = plt.subplots()

        ax.set_xlabel("q-value")
        ax.set_ylabel("eps-value")

        ax.contour(Q, E, mat1, levels=[0], colors="black")
        ax.contour(Q, E, mat2, levels=[0], colors="black")

        is_valid = np.logical_and(mat1 > 0, mat2 > 0)
        mat4[np.logical_not(is_valid)] = np.inf

        levels = np.linspace(-4, 2, 100)
        cf = ax.contourf(Q, E, np.log10(mat4), levels=levels, cmap="jet")

        levels = np.array([-2, -1])
        ax.contour(Q, E, np.log10(mat4), levels=levels, cmap="jet")

        fig.colorbar(cf)

        minima = pts[np.argmin(values4)]
        ax.scatter(minima[0], minima[1], c="k")
        print(10**minima)
        print("min value : {}".format(np.min(mat4)))

        plt.show()


@dataclass(frozen=True)
class CoverageResult:
    values_ground_truth: np.ndarray
    values_estimation: np.ndarray
    threshold: float

    def __post_init__(self):
        assert len(self.values_ground_truth) == len(self.values_estimation)
        self.values_ground_truth.flags.writeable = False
        self.values_estimation.flags.writeable = False

    def __len__(self) -> int:
        return len(self.values_ground_truth)

    def bootstrap_sampling(self) -> "CoverageResult":
        n = self.__len__()
        indices = np.random.randint(n, size=n)
        vgt = self.values_ground_truth[indices]
        vest = self.values_estimation[indices]
        return CoverageResult(vgt, vest, self.threshold)

    @cached_property
    def true_positive_rate(self) -> float:
        est_trues = np.sum(self.values_estimation <= self.threshold)
        true_positives = np.sum(self.true_positive_bools)
        return float(true_positives / est_trues)

    @cached_property
    def true_positive_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth <= self.threshold, self.values_estimation <= self.threshold
        )

    @cached_property
    def true_negative_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth > self.threshold, self.values_estimation > self.threshold
        )

    @cached_property
    def false_postive_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth > self.threshold, self.values_estimation <= self.threshold
        )

    @cached_property
    def false_negative_bools(self) -> np.ndarray:
        return np.logical_and(
            self.values_ground_truth <= self.threshold, self.values_estimation > self.threshold
        )

    def determine_margin(
        self, true_positive_lower_bound: float, confidence_coefficient: float = 0.05
    ) -> float:
        m = len(self.values_ground_truth)

        for margin in np.linspace(0.0, self.threshold, 100):
            # Z_1 estimated to be successful
            logicals1 = self.values_estimation + margin < self.threshold
            Z_1 = np.sum(logicals1) / m

            # Z_2 estimated to be successful and actually successful
            logicals2 = np.logical_and(logicals1, self.values_ground_truth < self.threshold)
            Z_2 = np.sum(logicals2) / m
            if Z_1 == 0:
                assert np.inf  # to indicate that the coresponding traj is useless

            # print("margin: {}, Z1: {}, Z2: {}".format(margin, Z_1, Z_2))
            opt = MarginOptimizer(confidence_coefficient, Z_1, Z_2, m)
            half_interval = opt.optimize(grid_search=False)
            assert half_interval > 0

            est_center = Z_2 / Z_1
            lower_bound = est_center - half_interval
            if lower_bound > true_positive_lower_bound:
                return margin
        return np.inf

    def __str__(self) -> str:
        string = "coverage result => "
        string += "n_sample: {}, ".format(len(self))
        string += "true positive: {}, ".format(sum(self.true_positive_bools))
        string += "true negative: {}, ".format(sum(self.true_negative_bools))
        string += "false positive: {}, ".format(sum(self.false_postive_bools))
        string += "false negative: {}".format(sum(self.false_negative_bools))
        return string
