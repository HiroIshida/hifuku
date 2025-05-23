import copy
import inspect
import logging
import pickle
import re
import signal
import time
import uuid
from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import tqdm
from mohou.trainer import TrainCache, TrainConfig, train
from mohou.utils import detect_device
from ompl import set_ompl_random_seed
from rpbench.interface import (
    AbstractTaskSolver,
    TaskBase,
    concat_param_seq,
    pack_param_seq,
    shuffle_param_seq,
)
from skmp.solver.interface import AbstractScratchSolver, ConfigT, ResultT
from skmp.trajectory import Trajectory

from hifuku.batch_network import BatchFCN
from hifuku.coverage import RealEstAggregate, optimize_latest_bias
from hifuku.datagen import (
    BatchBiasesOptimizerBase,
    BatchTaskSampler,
    BatchTaskSolver,
    DistributeBatchBiasesOptimizer,
    DistributeBatchTaskSampler,
    DistributedBatchTaskSolver,
    MultiProcesBatchBiasesOptimizer,
    MultiProcessBatchTaskSampler,
    MultiProcessBatchTaskSolver,
)
from hifuku.neuralnet import (
    AutoEncoderBase,
    ChannelSplitPixelAutoEncoder,
    CostPredictor,
    CostPredictorConfig,
    CostPredictorWithEncoder,
    CostPredictorWithEncoderConfig,
    NullAutoEncoder,
    PixelAutoEncoder,
    VoxelAutoEncoder,
    create_dataset_from_params_and_results,
)
from hifuku.pool import TaskPool, TaskT

logger = logging.getLogger(__name__)


@dataclass
class ProfileInfo:  # per each iteration
    t_total: Optional[float] = None
    t_determine_cand: Optional[float] = None
    t_dataset: Optional[float] = None
    t_train: Optional[float] = None
    t_bias: Optional[float] = None

    @property
    def has_all_info(self) -> bool:
        return all(
            [
                self.t_total is not None,
                self.t_determine_cand is not None,
                self.t_dataset is not None,
                self.t_train is not None,
                self.t_bias is not None,
            ]
        )

    @property
    def is_valid(self) -> bool:
        if not self.has_all_info:
            return False
        # sum of each time must be smaller than t_total
        return self.t_total > (self.t_dataset + self.t_train + self.t_determine_cand + self.t_bias)  # type: ignore[operator]

    @classmethod
    def from_total(cls, t_total: float) -> "ProfileInfo":
        # for backward compatibility
        return cls(t_total, None, None, None, None)

    @property
    def t_other(self) -> float:
        return self.t_total - (self.t_dataset + self.t_train + self.t_determine_cand + self.t_bias)  # type: ignore[operator]


@dataclass
class ActiveSamplerHistory:
    # states
    sampling_number_factor: float

    # the below are not states but history for postmortem analysis
    biases_history: List[List[float]]
    aggregate_list: List[RealEstAggregate]
    candidates_history: List[List[Trajectory]]
    elapsed_time_history: List[ProfileInfo]
    coverage_est_history: List[float]
    failure_count: int

    @classmethod
    def init(cls, sampling_number_factor: float) -> "ActiveSamplerHistory":
        return cls(sampling_number_factor, [], [], [], [], [], 0)

    def check_consistency(self) -> None:
        if len(self.elapsed_time_history) == 0:
            return
        assert len(self.aggregate_list) == len(self.biases_history)
        total_iter = len(self.aggregate_list) + self.failure_count
        assert len(self.candidates_history) == total_iter
        assert len(self.elapsed_time_history) == total_iter
        for elapsed_time in self.elapsed_time_history:
            assert elapsed_time.is_valid
        assert len(self.coverage_est_history) == total_iter
        assert all(
            [
                self.coverage_est_history[i] <= self.coverage_est_history[i + 1]
                for i in range(len(self.coverage_est_history) - 1)
            ]
        )

    def dump(self, base_path: Path) -> None:
        # with (base_path / "sampler_history.pkl").open(mode="wb") as f:
        #     pickle.dump(self, f)
        # dump using dict
        dic: Dict[str, Any] = {}
        dic["sampling_number_factor"] = self.sampling_number_factor
        dic["aggregate_list"] = [agg.to_dict() for agg in self.aggregate_list]
        dic["biases_history"] = self.biases_history
        dic["candidates_history"] = [pickle.dumps(cands) for cands in self.candidates_history]
        dic["elapsed_time_history"] = [asdict(e) for e in self.elapsed_time_history]
        dic["coverage_est_history"] = self.coverage_est_history
        dic["failure_count"] = self.failure_count
        with (base_path / "sampler_history.pkl").open(mode="wb") as f:
            pickle.dump(dic, f)

    @classmethod
    def load(cls, base_path: Path) -> "ActiveSamplerHistory":
        with (base_path / "sampler_history.pkl").open(mode="rb") as f:
            dic = pickle.load(f)
        sampling_number_factor = dic["sampling_number_factor"]
        aggregate_list = [RealEstAggregate.from_dict(d) for d in dic["aggregate_list"]]
        biases_history = dic["biases_history"]
        candidates_history = [pickle.loads(c) for c in dic["candidates_history"]]
        elapsed_time_history = [ProfileInfo(**d) for d in dic["elapsed_time_history"]]
        coverage_est_history = dic["coverage_est_history"]
        failure_count = dic["failure_count"]
        return cls(
            sampling_number_factor,
            biases_history,
            aggregate_list,
            candidates_history,
            elapsed_time_history,
            coverage_est_history,
            failure_count,
        )

    @property
    def total_iter(self) -> int:
        return len(self.aggregate_list) + self.failure_count

    @property
    def total_time(self) -> float:
        return sum([e.t_total for e in self.elapsed_time_history])  # type: ignore


@dataclass
class SolutionLibrary:
    max_admissible_cost: float
    ae_model_shared: Optional[AutoEncoderBase]
    predictors: List[Union[CostPredictor, CostPredictorWithEncoder]]
    init_solutions: List[Trajectory]
    biases: List[float]
    uuidval: str
    meta_data: Dict
    jit_compiled: bool = False
    batch_predictor: Optional[Callable] = None

    def __post_init__(self):
        if self.ae_model_shared is not None:
            assert self.ae_model_shared.trained
            for pred in self.predictors:
                assert isinstance(pred, CostPredictor)
        else:
            for pred in self.predictors:
                assert isinstance(pred, CostPredictorWithEncoder)

    @dataclass
    class InferenceResult:
        cost: float
        idx: int  # index of selected solution in the library
        init_solution: Trajectory

    @classmethod
    def initialize(
        cls,
        max_admissible_cost: float,
        ae_model: Optional[AutoEncoderBase],
        meta_data: Optional[Dict] = None,
    ) -> "SolutionLibrary":
        uuidval = str(uuid.uuid4())[-8:]
        if meta_data is None:
            meta_data = {}
        return cls(
            max_admissible_cost,
            ae_model,
            [],
            [],
            [],
            uuidval,
            meta_data,
        )

    def put_on_device(self, device: torch.device):
        if self.ae_model_shared is not None:
            self.ae_model_shared.put_on_device(device)
        for pred in self.predictors:
            pred.put_on_device(device)

    @property
    def device(self) -> torch.device:
        if self.ae_model_shared is not None:
            return self.ae_model_shared.get_device()
        else:
            pred: CostPredictorWithEncoder = self.predictors[0]  # type: ignore[assignment]
            return pred.device

    def jit_compile(self, batch_predictor: bool = False, n_batch_vec: int = 1) -> None:
        # batch input of image is not supported yet
        if n_batch_vec > 1:
            print("batch size larger than 1 is experimental")
        print("compling...")
        assert self.device.type == "cuda"
        if self.ae_model_shared is not None:

            if isinstance(self.ae_model_shared, nn.Module):
                n_grid = self.ae_model_shared.config.n_grid
                n_channel = self.ae_model_shared.config.n_channel
                if isinstance(
                    self.ae_model_shared, (PixelAutoEncoder, ChannelSplitPixelAutoEncoder)
                ):
                    dummy_input = torch.rand(1, n_channel, n_grid, n_grid).cuda()
                elif isinstance(self.ae_model_shared, VoxelAutoEncoder):
                    assert n_channel == 1, "TODO: support multi channel voxel auto encoder"
                    dummy_input = torch.rand(1, 1, n_grid, n_grid, n_grid).cuda()
                else:
                    assert False
                traced = torch.jit.trace(self.ae_model_shared.eval(), (dummy_input,))
                self.ae_model_shared = torch.jit.optimize_for_inference(traced)

            pred_config = self.predictors[0].config
            dummy_input = (
                torch.rand(1, pred_config.dim_conv_bottleneck).cuda(),
                torch.rand(n_batch_vec, pred_config.dim_task_descriptor).cuda(),
            )
        else:
            if n_batch_vec > 1:
                raise ValueError(
                    "TODO: batch size larger than 1 is not supported for non-shared ae"
                )
            pred_config = self.predictors[0].config
            assert isinstance(pred_config, CostPredictorWithEncoderConfig)
            dim_vector = pred_config.costpred_model.config.dim_task_descriptor
            n_grid = pred_config.ae_model.config.n_grid
            image_input = torch.rand(1, 1, n_grid, n_grid)
            vec_input = torch.rand(1, dim_vector)
            dummy_input = (image_input.cuda(), vec_input.cuda())

        if batch_predictor:
            linear_list = []
            expander_list = []
            for pred in self.predictors:
                linear_list.append(pred.linears)
                expander_list.append(pred.description_expand_linears)
            fcn_linears_batch = BatchFCN(linear_list).cuda()
            fcn_expanders_batch = BatchFCN(expander_list).cuda()

            class Tmp(nn.Module):
                def __init__(self, fcn_linears, fcn_expanders, n):
                    super().__init__()
                    self.fcn_linears = fcn_linears
                    self.fcn_expanders = fcn_expanders
                    self.n = n

                def forward(self, bottleneck, descriptor):
                    expanded = self.fcn_expanders(descriptor)
                    n_batch = expanded.shape[0]
                    encoded = bottleneck.repeat(n_batch, self.n, 1)
                    concat = torch.cat((encoded, expanded), dim=2)
                    tmp = self.fcn_linears(concat)
                    return tmp.squeeze(2)

                def cuda(self):
                    self.fcn_linears.cuda()
                    self.fcn_expanders.cuda()
                    return super().cuda()

            tmp = Tmp(fcn_linears_batch, fcn_expanders_batch, len(self.predictors)).cuda()
            traced = torch.jit.trace(tmp, dummy_input)
            self.batch_predictor = torch.jit.optimize_for_inference(traced)
        else:
            for i in range(len(self.predictors)):
                traced = torch.jit.trace(self.predictors[i], (dummy_input,))
                self.predictors[i] = torch.jit.optimize_for_inference(traced)
        print("done. note that you must warm up the gpu by running the inference once")
        self.jit_compiled = True

    def _infer_cost(self, task: TaskBase) -> np.ndarray:
        assert len(self.predictors) > 0
        has_shared_ae = self.ae_model_shared is not None
        if has_shared_ae:
            return self._infer_cost_with_shared_ae(task)
        else:
            return self._infer_cost_combined(task)

    def _infer_cost_combined(self, task: TaskBase) -> np.ndarray:
        expression = task.export_task_expression(use_matrix=True)
        matrix = expression.get_matrix()
        assert matrix is not None
        world_mat_np = np.expand_dims(matrix, axis=(0, 1))
        vec_np = np.array([expression.get_vector()])

        if self.jit_compiled:
            device = torch.device("cuda")
        else:
            device = self.device

        world_mat_torch = torch.from_numpy(world_mat_np).float().to(device)
        vec_torch = torch.from_numpy(vec_np).float().to(device)

        cost_list = []
        for pred, bias in zip(self.predictors, self.biases):
            # assert isinstance(pred, CostPredictorWithEncoder)
            costs = pred.forward((world_mat_torch, vec_torch))  # type: ignore
            cost_list.append(costs.item() + bias)
        return np.array(cost_list)

    def _infer_cost_with_shared_ae(self, task: TaskBase) -> np.ndarray:
        assert self.ae_model_shared is not None

        if self.jit_compiled:
            device = torch.device("cuda")
        else:
            device = self.device

        expression = task.export_task_expression(use_matrix=True)
        # NOTE(2024/11/3): avoid calling tensor.float() because it causes
        # performance issue and deteoriation as well.
        # this may be circumvented by limiting the torch thread number
        # but this requires nonnegligible overhead. thus
        vecs_np = np.array([expression.get_vector()]).astype(np.float32)
        vecs_torch = torch.from_numpy(vecs_np).to(device)

        matrix = expression.get_matrix()
        if matrix is None:
            matrix_torch = torch.empty((1, 0))
        else:
            # NOTE(2024/11/3): avoid calling tensor.float() (see above)
            is_channel_split = isinstance(self.ae_model_shared, ChannelSplitPixelAutoEncoder)
            if self.jit_compiled:
                is_channel_split = (
                    self.ae_model_shared.original_name == "ChannelSplitPixelAutoEncoder"
                )

            if is_channel_split:
                # No need to expand the dimension for channel split pixel auto encoder
                matrix_np = np.expand_dims(matrix, axis=(0,)).astype(np.float32)
            else:
                matrix_np = np.expand_dims(matrix, axis=(0, 1)).astype(np.float32)
            matrix_torch = torch.from_numpy(matrix_np).to(device)

        n_batch = 1
        encoded: torch.Tensor = self.ae_model_shared.forward(matrix_torch)

        if self.batch_predictor is not None:
            assert self.jit_compiled, "batch predictor must be available only when jit compiled"
            raw_costs = self.batch_predictor(encoded, vecs_torch)
            costs_numpy = raw_costs.detach().cpu().numpy().flatten()
            return costs_numpy + np.array(self.biases)
        else:
            encoded_repeated = encoded.repeat(n_batch, 1)
            cost_list = []
            for pred, bias in zip(self.predictors, self.biases):
                costs, _ = pred.forward((encoded_repeated, vecs_torch))
                costs.item() + bias
                cost_list.append(costs.item() + bias)
            return np.array(cost_list)

    def infer(self, task: TaskBase) -> InferenceResult:
        costs = self._infer_cost(task)
        cost_min = np.min(costs)
        idx_min = int(np.argmin(costs))
        return self.InferenceResult(cost_min, idx_min, self.init_solutions[idx_min])

    def dump(self, base_path: Path) -> None:
        # rather than directly saving the object, serialize it
        # to a dictionary and then save it for future compatibility
        device = torch.device("cpu")

        dic: Dict[str, Any] = {}
        dic["max_admissible_cost"] = self.max_admissible_cost
        if self.ae_model_shared is None:
            dic["ae_model_shared"] = None
        else:
            ae_model = copy.deepcopy(self.ae_model_shared)
            ae_model.put_on_device(device)
            dic["ae_model_shared"] = pickle.dumps(ae_model)
        dic["predictors"] = []
        for pred in self.predictors:
            pred_copied = copy.deepcopy(pred)
            pred_copied.put_on_device(device)
            dic["predictors"].append(pickle.dumps(pred_copied))
        dic["biases"] = self.biases
        dic["init_solutions"] = pickle.dumps(self.init_solutions)
        dic["uuidval"] = self.uuidval
        dic["meta_data"] = self.meta_data

        name = "Library-{}.pkl".format(self.uuidval)
        with (base_path / name).open(mode="wb") as f:
            pickle.dump(dic, f)

    @classmethod
    def load(
        cls,
        base_path: Path,
        device: Optional[torch.device] = None,
        n_experience_use: Optional[int] = None,
    ) -> "SolutionLibrary":
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        file_lst = []
        for file in base_path.iterdir():
            match = re.match(r"Library-(\w{8}).pkl", file.name)
            if match:
                file_lst.append((file, match.group(1)))
        if len(file_lst) == 0:
            raise FileNotFoundError("no library file found")
        if len(file_lst) > 1:
            raise ValueError("multiple library files found")

        file, uuidval = file_lst[0]
        with file.open(mode="rb") as f:
            dic = pickle.load(f)
        max_admissible_cost = dic["max_admissible_cost"]
        tmp = dic["ae_model_shared"]
        if tmp is None:
            ae_model_shared = None
        else:
            ae_model_shared = pickle.loads(tmp)
            ae_model_shared.put_on_device(device)
        pred_pickled_list = dic["predictors"]
        pred_list = []

        for pred_pickled in pred_pickled_list:
            pred = pickle.loads(pred_pickled)
            pred.put_on_device(device)
            pred_list.append(pred)
        biases = dic["biases"]
        init_solutions = pickle.loads(dic["init_solutions"])
        meta_data = dic["meta_data"]

        if n_experience_use is None:
            n_experience_use = len(pred_pickled_list)
        if n_experience_use > len(pred_pickled_list):
            raise ValueError(
                "n_experience_use must be smaller than the number of predictors in the library"
            )

        return cls(
            max_admissible_cost,
            ae_model_shared,
            pred_list[:n_experience_use],
            init_solutions[:n_experience_use],
            biases[:n_experience_use],
            uuidval,
            meta_data,
        )


@dataclass
class LibraryBasedSolverBase(AbstractTaskSolver[TaskT, ConfigT, ResultT]):
    library: SolutionLibrary
    solver: AbstractScratchSolver[ConfigT, ResultT]
    task: Optional[TaskT]
    timeout: Optional[float]
    previous_false_positive: Optional[bool]
    previous_est_positive: Optional[bool]
    _loginfo_fun: Callable
    _logwarn_fun: Callable

    @classmethod
    def init(
        cls,
        library: SolutionLibrary,
        solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        use_rospy_logger: bool = False,
    ) -> "LibraryBasedSolverBase[TaskT, ConfigT, ResultT]":
        # internal solver's timeout must be None
        # because inference time must be considered in timeout for fairness
        timeout_stashed = config.timeout  # stash this
        config.timeout = None
        solver = solver_type.init(config)

        if use_rospy_logger:
            import rospy

            # NOTE: don't know why but importing rospy at the top of the file
            # cause issue in logging in the training phase. That's why I import
            # it here.
            loginfo_fun = rospy.loginfo
            logwarn_fun = rospy.logwarn
        else:
            loginfo_fun = logger.info
            logwarn_fun = logger.warning

        return cls(library, solver, None, timeout_stashed, None, None, loginfo_fun, logwarn_fun)

    def setup(self, task: TaskT) -> None:
        self.solver.setup(task.export_problem())
        self.task = task

    def solve(self) -> ResultT:
        # NOTE: almost copied from skmp.solver.interface
        ts = time.time()

        if self.timeout is not None:
            assert self.timeout > 0

            def handler(sig, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, self.timeout)
            set_ompl_random_seed(0)  # to make result reproducible
        try:
            ret = self._solve()
        except TimeoutError:
            ret = self.solver.get_result_type().abnormal()

        if self.timeout is not None:
            signal.alarm(0)  # reset alarm

        ret.time_elapsed = time.time() - ts
        return ret

    @abstractmethod
    def _solve(self) -> ResultT:
        ...


@dataclass
class LibraryBasedGuaranteedSolver(LibraryBasedSolverBase[TaskT, ConfigT, ResultT]):
    def __post_init__(self):
        assert self.solver.config.n_max_call == self.library.max_admissible_cost

    def _solve(self) -> ResultT:
        self.previous_est_positive = None
        self.previous_false_positive = None

        ts = time.time()
        assert self.task is not None
        infer_res = self.library.infer(self.task)

        seems_infeasible = infer_res.cost > self.library.max_admissible_cost
        self._loginfo_fun(f"nit {infer_res.cost}: the {self.library.max_admissible_cost}")
        if seems_infeasible:
            self._logwarn_fun("seems infeasible")
            result_type = self.solver.get_result_type()
            res = result_type.abnormal()
            res.time_elapsed = None
            self.previous_est_positive = False
            self.previous_false_positive = False
            return res
        solver_result = self.solver.solve(infer_res.init_solution)
        solver_result.time_elapsed = time.time() - ts

        self.previous_est_positive = True
        self.previous_false_positive = solver_result.traj is None
        return solver_result


@dataclass
class LibraryBasedHeuristicSolver(LibraryBasedSolverBase[TaskT, ConfigT, ResultT]):
    def __post_init__(self):
        assert self.solver.config.n_max_call > 10000  # heuristic solver must run until timeout

    def _solve(self) -> ResultT:
        ts = time.time()
        assert self.task is not None
        infer_res = self.library.infer(self.task)
        solver_result = self.solver.solve(infer_res.init_solution)
        solver_result.time_elapsed = time.time() - ts
        return solver_result


@dataclass
class DifficultTaskPredicate:
    task_type: Type[TaskBase]
    library: SolutionLibrary
    th_min_cost: float
    th_max_cost: Optional[float] = None

    def __post_init__(self):
        # note: library must be put on cpu
        # to copy into forked processes
        self.library = copy.deepcopy(self.library)
        self.library.put_on_device(torch.device("cpu"))

    def __call__(self, task: TaskBase) -> bool:
        infer_res = self.library.infer(task)
        cost = infer_res.cost
        if cost < self.th_min_cost:
            return False
        if self.th_max_cost is None:
            return True
        else:
            return cost < self.th_max_cost


@dataclass
class LibrarySamplerConfig:
    # you have to tune
    sampling_number_factor: float = 5000
    acceptable_false_positive_rate: float = 0.1

    # if active sampling stop after memory error... and joint training ...
    compress_matrix: bool = False

    # maybe you have to tune maybe ...
    inc_coef_mult_snf: float = 1.1  # snf stands for sampling_number_factor
    dec_coef_mult_snf: float = 0.9
    threshold_inc_snf: float = 0.3  # if gain < expected * this, then increase snf
    threshold_dec_snf: float = 0.7  # if gain > expected * this, then decrease snf
    n_solution_candidate: int = 100
    n_difficult: int = 500
    early_stopping_patience: int = 10

    # experimental
    already_solvable_as_failure: bool = False

    # same for all settings (you dont have to tune)
    n_task_inner: int = 1  # this should be 1 always (2024/02/24)
    sample_from_difficult_region: bool = True
    train_config: TrainConfig = TrainConfig()
    ignore_useless_traj: bool = True
    costpred_model_config: Optional[Dict] = None
    n_validation: int = 10000
    n_validation_inner: int = 1
    n_optimize_biases_batch: int = 2000
    candidate_sample_scale: int = 4
    train_with_encoder: bool = False
    tmp_n_max_call_mult_factor: float = 1.5
    clamp_factor: float = 2.0

    def __post_init__(self):
        assert self.tmp_n_max_call_mult_factor <= self.clamp_factor


@dataclass
class SimpleSolutionLibrarySampler(Generic[TaskT, ConfigT, ResultT]):
    task_type: Type[TaskT]
    solver_config: ConfigT
    library: SolutionLibrary
    config: LibrarySamplerConfig
    task_pool: TaskPool[TaskT]
    tasks_validation: np.ndarray
    solver: BatchTaskSolver
    sampler: BatchTaskSampler
    biases_optimizer: BatchBiasesOptimizerBase
    test_false_positive_rate: bool
    project_path: Path
    sampler_history: ActiveSamplerHistory
    device: torch.device
    ae_model_pretrained: Optional[AutoEncoderBase]
    presampled_tasks_params: np.ndarray

    # below are class variables
    presampled_cache_file_name: ClassVar[str] = "presampled_tasks.cache"

    @property
    def train_pred_with_encoder(self) -> bool:
        return self.ae_model_pretrained is not None

    def at_first_iteration(self) -> bool:
        return len(self.library.predictors) == 0

    @classmethod
    def initialize(
        cls,
        task_type: Type[TaskT],
        solver_t: Type[AbstractScratchSolver[ConfigT, ResultT]],
        solver_config: ConfigT,
        ae_model: AutoEncoderBase,
        config: LibrarySamplerConfig,
        project_path: Path,
        task_pool: Optional[TaskPool[TaskT]] = None,
        solver: Optional[BatchTaskSolver[ConfigT, ResultT]] = None,
        sampler: Optional[BatchTaskSampler[TaskT]] = None,
        use_distributed: bool = False,
        test_false_positive_rate: bool = False,
        n_limit_batch_solver: Optional[int] = None,
        device: Optional[torch.device] = None,
        warm_start: bool = False,
    ) -> "SimpleSolutionLibrarySampler[TaskT, ConfigT, ResultT]":
        """
        use will be used only if either of solver and sampler is not set
        """

        frame = inspect.currentframe()
        assert frame is not None
        _, _, _, values = inspect.getargvalues(frame)
        logger.info("arg of initialize: {}".format(values))

        if device is None:
            device = detect_device()
        assert ae_model.get_device() == device
        meta_data = asdict(config)

        # setup library
        if warm_start:
            try:
                library = SolutionLibrary.load(project_path, torch.device(device))
                library.put_on_device(device)
            except Exception as e:
                message = f"failed to load library from warm start. {e}"
                logger.error(message)
                raise RuntimeError(message)
            loaded_metadata = library.meta_data

            for k, v in meta_data.items():
                if k in loaded_metadata:
                    if loaded_metadata[k] != v:
                        message = "metadata mismatch in warm start. {} != {}".format(
                            loaded_metadata[k], v
                        )
                        # warning because it's not fatal
                        logger.warning(message)
                else:
                    message = f"metadata mismatch in warm start. {k} is not in loaded metadata"
                    logger.warning(message)
        else:
            library = SolutionLibrary.initialize(
                solver_config.n_max_call,
                None if config.train_with_encoder else ae_model,
                meta_data,
            )

        # setup sampler history
        if warm_start:
            try:
                sampler_history = ActiveSamplerHistory.load(project_path)
            except Exception as e:
                message = f"failed to load sampler history from warm start. {e}"
                logger.error(message)
                raise RuntimeError(message)
        else:
            sampler_history = ActiveSamplerHistory.init(config.sampling_number_factor)

        # setup solver, sampler, optimizer
        if solver is None:
            solver = (
                DistributedBatchTaskSolver(
                    solver_t, solver_config, task_type, n_limit_batch=n_limit_batch_solver
                )
                if use_distributed
                else MultiProcessBatchTaskSolver(
                    solver_t, solver_config, task_type, n_limit_batch=n_limit_batch_solver
                )
            )
        assert solver.solver_t == solver_t
        assert solver.config == solver_config
        if sampler is None:
            sampler = (
                DistributeBatchTaskSampler() if use_distributed else MultiProcessBatchTaskSampler()
            )
        biases_optimizer = (
            DistributeBatchBiasesOptimizer()
            if use_distributed
            else MultiProcesBatchBiasesOptimizer()
        )

        # setup pools
        if task_pool is None:
            task_pool = TaskPool(task_type)

        # setup validation set
        project_path.mkdir(exist_ok=True)
        validation_cache_path = project_path / "{}-validation_set.cache".format(task_type.__name__)
        if warm_start:
            assert validation_cache_path.exists(), "validation cache must exist in warm start"
            with validation_cache_path.open(mode="rb") as f:
                tasks_validation: np.ndarray = pickle.load(f)
        else:
            if validation_cache_path.exists():
                raise RuntimeError(
                    f"validation cache {validation_cache_path} must not exist in cold start. remove it"
                )
            logger.info("start creating validation set")
            tasks_validation = sampler.sample_batch(
                config.n_validation, TaskPool(task_type).as_predicated()
            )
            with validation_cache_path.open(mode="wb") as f:
                pickle.dump(tasks_validation, f)
            logger.info("validation set with {} elements is created".format(len(tasks_validation)))
        assert len(tasks_validation) > 0

        # setup presampled tasks
        presample_cache_path = project_path / cls.presampled_cache_file_name
        presampled_task_params: Optional[np.ndarray] = None
        if warm_start:
            if not presample_cache_path.exists():
                logger.warn("it's bit strange that presample cache does not exist in warm start...")
                task_params = next(task_pool)  # sample once to get the shape
                presampled_task_params = np.array([task_params])
            else:
                with presample_cache_path.open(mode="rb") as f:
                    presampled_task_params = pickle.load(f)
        else:
            if presample_cache_path.exists():
                raise RuntimeError(
                    f"presample cache {presample_cache_path} must not exist in cold start. remove it"
                )
            task_params = next(task_pool)  # sample once to get the shape
            presampled_task_params = pack_param_seq([task_params])

        logger.info("library sampler config: {}".format(config))
        return cls(
            task_type,
            solver_config,
            library,
            config,
            task_pool,
            tasks_validation,
            solver,
            sampler,
            biases_optimizer,
            test_false_positive_rate,
            project_path,
            sampler_history,
            device,
            ae_model if config.train_with_encoder else None,
            presampled_task_params,
        )

    def step_active_sampling(self) -> bool:
        """
        return False if failed
        """
        prof_info = ProfileInfo()
        ts = time.time()
        self.sampler_history.check_consistency()
        assert len(self.sampler_history.aggregate_list) == len(self.library.predictors)

        logger.info("active sampling step")

        ts_determine_cand = time.time()
        init_solution, gain_expected = self._determine_init_solution(self.config.n_difficult)
        logger.info(f"sampling nuber factor: {self.sampler_history.sampling_number_factor}")
        n_task_now = int((1.0 / gain_expected) * self.sampler_history.sampling_number_factor)
        logger.info(f"n_task_now: {n_task_now}")
        prof_info.t_determine_cand = time.time() - ts_determine_cand

        predictor = self._train_predictor(init_solution, self.project_path, n_task_now, prof_info)

        ts_bias = time.time()
        ret = self._determine_biases(predictor, init_solution)
        prof_info.t_bias = time.time() - ts_bias

        if ret is None:
            logger.info("no bias set could increase coverage. dont add anything to library")
            if len(self.sampler_history.coverage_est_history) > 0:
                coverage_est = self.sampler_history.coverage_est_history[-1]
            else:
                coverage_est = 0.0
            self.sampler_history.failure_count += 1
        else:
            biases, aggregate, coverage_est = ret
            logger.info("bias for latest costpred is {}".format(biases[-1]))
            logger.debug("determined biases {}".format(biases))

            self.library.predictors.append(predictor)
            self.library.init_solutions.append(init_solution)
            self.library.biases = biases

            self.sampler_history.aggregate_list.append(aggregate)
            self.sampler_history.biases_history.append(copy.deepcopy(biases))

        prof_info.t_total = time.time() - ts

        self.sampler_history.elapsed_time_history.append(prof_info)
        self.sampler_history.coverage_est_history.append(coverage_est)
        coverage_this = self.sampler_history.coverage_est_history[-1]
        if len(self.sampler_history.coverage_est_history) > 1:
            coverage_previous = self.sampler_history.coverage_est_history[-2]
        else:
            coverage_previous = 0.0
        gain = coverage_this - coverage_previous
        logger.info(
            f"coverage this: {coverage_this}, coverage previous: {coverage_previous}, gain: {gain}"
        )
        achievement_rate = gain / gain_expected
        logger.info(
            f"expected gain: {gain_expected}, actual gain: {gain}, achievement rate: {achievement_rate}"
        )
        if achievement_rate < self.config.threshold_inc_snf:
            self.sampler_history.sampling_number_factor *= self.config.inc_coef_mult_snf
            logger.info(
                f"expected gain is {gain_expected} is too small. increase sampling number factor to {self.sampler_history.sampling_number_factor}"
            )
        elif gain > gain_expected * self.config.threshold_dec_snf:
            self.sampler_history.sampling_number_factor *= self.config.dec_coef_mult_snf
            logger.info(
                f"expected gain is high enough. decrease sampling number factor to {self.sampler_history.sampling_number_factor}"
            )

        logger.info("elapsed time in active sampling: {} min".format(prof_info.t_total / 60.0))
        logger.info("prof_info: {}".format(prof_info))
        t_total_list = [e.t_total for e in self.sampler_history.elapsed_time_history]
        logger.info("current elapsed time history: {}".format(t_total_list))
        logger.info(
            "current coverage est history: {}".format(self.sampler_history.coverage_est_history)
        )
        self.library.dump(self.project_path)
        self.sampler_history.dump(self.project_path)
        return True

    def _determine_biases(
        self,
        predictor: Union[CostPredictorWithEncoder, CostPredictor],
        init_solution: Trajectory,
    ) -> Optional[Tuple[List[float], RealEstAggregate, float]]:
        # TODO: move this whole "adjusting" operation to a different method
        logger.info("start measuring coverage")
        singleton_library = SolutionLibrary(
            self.library.max_admissible_cost,
            self.library.ae_model_shared,
            [predictor],
            [init_solution],
            [0.0],
            "dummy",
            {},
        )
        aggregate = self.measure_real_est(singleton_library, self.tasks_validation)
        logger.info(aggregate)
        agg_list = self.sampler_history.aggregate_list + [aggregate]

        # first solve the latest bias naively fixig all the past biases
        logger.info("determine bias using naive method")
        naive_result = optimize_latest_bias(
            agg_list,
            self.library.biases,
            self.solver_config.n_max_call,
            self.config.acceptable_false_positive_rate,
        )

        # solve the biases using cmaes if the library is not empty
        cmaes_result = None
        if len(self.library.predictors) > 0:
            logger.info("determine biases using cmaes")
            cma_std = self.solver_config.n_max_call * 0.5
            coverage_est_last = self.sampler_history.coverage_est_history[-1]
            results = self.biases_optimizer.optimize_batch(
                self.config.n_optimize_biases_batch,
                agg_list,
                self.solver_config.n_max_call,
                self.config.acceptable_false_positive_rate,
                cma_std,
                minimum_coverage=coverage_est_last,
            )
            # filter None of results
            result_list = [r for r in results if r is not None]
            if len(result_list) > 0:
                idx_max_cover = np.argmax([r.coverage for r in result_list])
                cmaes_result = result_list[idx_max_cover]

        # both method failed
        if naive_result is None and cmaes_result is None:
            logger.info("no biases set could increase coverage")
            return None

        # logging for later discussion in the paper
        result = None
        if naive_result is None:
            assert cmaes_result is not None  # to satisfy mypy
            logger.info(f"naive method failed. use cmaes result {cmaes_result.coverage}")
            result = cmaes_result
        elif cmaes_result is None:
            logger.info(
                f"cmaes method failed (or not executed). use naive result {naive_result.coverage}"
            )
            result = naive_result
        else:
            if cmaes_result.coverage > naive_result.coverage:
                logger.info(
                    f"cmaes result {cmaes_result.coverage} is better than naive result {naive_result.coverage}"
                )
                result = cmaes_result
            else:
                logger.info(
                    f"naive result {naive_result.coverage} is better than cmaes result {cmaes_result.coverage}"
                )
                result = naive_result
        assert result is not None

        return result.best_biases, aggregate, result.coverage

    def _consider_already_solvable_as_failure(
        self, params: np.ndarray, results: List[ResultT]
    ) -> List[ResultT]:
        if len(self.library.predictors) == 0:
            logger.info("no predictor in the library. do nothing")
            return results

        logger.info("consider already solvable as failure")
        results_new: List[ResultT] = []
        for param, result in tqdm.tqdm(zip(params, results)):
            task = self.task_type.from_task_param(param)
            inf_res = self.library.infer(task)
            already_solvable = inf_res.cost <= self.library.max_admissible_cost
            copied = copy.deepcopy(result)
            if already_solvable:
                copied.traj = None  # traj = None means failure
            results_new.append(copied)
        return results_new

    def _train_predictor(
        self,
        init_solution: Trajectory,
        project_path: Path,
        n_task: int,
        profile_info: ProfileInfo,
    ) -> Union[CostPredictorWithEncoder, CostPredictor]:

        ts_dataset = time.time()

        # NOTE: to my future self: you can't use presampled tasks if you'd like to
        # sample tasks using some predicate!!
        if len(self.presampled_tasks_params) < n_task:
            n_required = n_task - len(self.presampled_tasks_params)
            logger.info(
                f"presampled {len(self.presampled_tasks_params)} tasks are not enough. populate more: {n_required}"
            )
            predicated_pool = self.task_pool.as_predicated()
            tasks = self.sampler.sample_batch(n_required, predicated_pool)
            self.presampled_tasks_params = concat_param_seq(self.presampled_tasks_params, tasks)
            # save it to cache
            presample_cache_path = project_path / self.presampled_cache_file_name
            with presample_cache_path.open(mode="wb") as f:
                pickle.dump(self.presampled_tasks_params, f)
        assert len(self.presampled_tasks_params) >= n_task
        tasks = self.presampled_tasks_params[:n_task]

        logger.info("start generating dataset")
        init_solutions = [init_solution] * len(tasks)
        results = self.solver.solve_batch(
            tasks,
            init_solutions,
            tmp_n_max_call_mult_factor=self.config.tmp_n_max_call_mult_factor,
        )

        if self.config.already_solvable_as_failure:
            # NOTE: 2023/03/07: this method is found to be rather ineffective
            # this is probably because by considering as already-covered as failure
            # the rich information provided by theses sample are lost.
            results = self._consider_already_solvable_as_failure(tasks, results)
        logger.info("creating dataset")
        weights = None
        dataset = create_dataset_from_params_and_results(
            tasks,
            results,
            self.solver_config,
            self.task_type,
            weights,
            self.library.ae_model_shared,
            self.config.clamp_factor,
            compress_mesh=self.config.compress_matrix,
        )

        profile_info.t_dataset = time.time() - ts_dataset

        logger.info("start training model")
        ts_train = time.time()
        # determine 1dim tensor dimension by temp creation of a task
        # TODO: should I implement this as a method?
        task = self.task_type.sample()
        exp = task.export_task_expression(use_matrix=True)
        vector_desc = exp.get_vector()
        n_dim_vector_description = vector_desc.shape[0]

        # train
        if self.train_pred_with_encoder:
            assert self.ae_model_pretrained is not None
            ae_model = self.ae_model_pretrained
        else:
            assert self.library.ae_model_shared is not None
            ae_model = self.library.ae_model_shared
        n_bottleneck = ae_model.n_bottleneck

        if isinstance(ae_model, ChannelSplitPixelAutoEncoder):
            # NOTE: this case each channel is encoded separately
            # Thus we need to multiply the number of channels
            # it's quite adhoc though
            n_bottleneck *= ae_model.config.n_channel

        if self.config.costpred_model_config is not None:
            costpred_model_conf = CostPredictorConfig(
                n_dim_vector_description, n_bottleneck, **self.config.costpred_model_config
            )
        else:
            costpred_model_conf = CostPredictorConfig(n_dim_vector_description, n_bottleneck)

        if self.train_pred_with_encoder:
            assert self.ae_model_pretrained is not None
            costpred_model = CostPredictor(costpred_model_conf, self.device)
            ae_model_pretrained = copy.deepcopy(self.ae_model_pretrained)
            ae_model_pretrained.put_on_device(costpred_model.device)
            assert not isinstance(ae_model_pretrained, NullAutoEncoder)
            # the right above assertion ensure that ae_model_pretrained has a device...
            assert costpred_model.device == ae_model_pretrained.device  # type: ignore[attr-defined]
            conf = CostPredictorWithEncoderConfig(costpred_model, ae_model_pretrained)
            model: Union[CostPredictorWithEncoder, CostPredictor] = CostPredictorWithEncoder(conf)
        else:
            model = CostPredictor(costpred_model_conf, self.device)

        tcache = TrainCache.from_model(model)

        train(
            project_path,
            tcache,
            dataset,
            self.config.train_config,
            early_stopping_patience=self.config.early_stopping_patience,
            device=self.device,
            num_workers=6,
        )
        model.eval()
        profile_info.t_train = time.time() - ts_train
        return model

    def _sample_solution_canidates(
        self,
        n_sample: int,
        task_pool: TaskPool[TaskT],
    ) -> List[Trajectory]:

        if self.config.sample_from_difficult_region and not self.at_first_iteration():
            # because sampling from near-feasible-boundary is effective in most case....
            pred_bit_difficult = DifficultTaskPredicate(
                task_pool.task_type,
                self.library,
                self.solver_config.n_max_call,
                self.solver_config.n_max_call * 1.2,
            )
            predicated_pool_bit_difficult = task_pool.make_predicated(pred_bit_difficult, 40)

            # but, we also need to sample from far-boundary because some of the possible
            # feasible regions are disjoint from the ones obtained so far
            pred_difficult = DifficultTaskPredicate(
                task_pool.task_type, self.library, self.solver_config.n_max_call, None
            )
            predicated_pool_difficult = task_pool.make_predicated(pred_difficult, 40)
        else:
            # "sampling from difficult" get stuck when the classifier is not properly trained
            # which becomes task in test where classifier is trained with dummy small sample
            predicated_pool_bit_difficult = task_pool.as_predicated()
            predicated_pool_difficult = predicated_pool_bit_difficult

        prefix = "_sample_solution_canidates:"

        logger.info("{} start sampling solution solved difficult tasks".format(prefix))

        # TODO: dont hardcode
        n_batch = n_sample * self.config.candidate_sample_scale
        n_batch_little_difficult = int(n_batch * 0.5)
        n_batch_difficult = n_batch - n_batch_little_difficult

        feasible_solutions: List[Trajectory] = []
        while True:
            logger.info("{} sample batch".format(prefix))
            tasks1 = self.sampler.sample_batch(
                n_batch_little_difficult, predicated_pool_bit_difficult
            )
            tasks2 = self.sampler.sample_batch(n_batch_difficult, predicated_pool_difficult)
            tasks = concat_param_seq(tasks1, tasks2)

            # NOTE: shuffling is required asin the following sectino, for loop is existed
            # as soon as number of candidates exceeds n_sample
            # we need to "mixutre" bit-difficult and difficult tasks
            tasks = shuffle_param_seq(tasks)

            logger.info("{} solve batch".format(prefix))
            results = self.solver.solve_batch(tasks, None, use_default_solver=True)

            for result in results:
                if result.traj is not None:
                    feasible_solutions.append(result.traj)
                    if len(feasible_solutions) == n_sample:
                        return feasible_solutions
            logger.info("{} progress {} / {} ".format(prefix, len(feasible_solutions), n_sample))

    def _sample_difficult_tasks(
        self,
        n_sample: int,
        task_pool: TaskPool[TaskT],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        difficult_params_list: List[np.ndarray] = []
        easy_params_list: List[np.ndarray] = []
        with tqdm.tqdm(total=n_sample) as pbar:
            while len(difficult_params_list) < n_sample:
                logger.debug("try sampling difficutl task...")
                task_param = next(task_pool)
                task = self.task_type.from_task_param(task_param)
                infer_res = self.library.infer(task)
                cost = infer_res.cost
                is_difficult = cost > self.solver_config.n_max_call
                if is_difficult:
                    logger.debug("sampled! number: {}".format(len(difficult_params_list)))
                    difficult_params_list.append(task_param)
                    pbar.update(1)
                else:
                    easy_params_list.append(task_param)
        if len(easy_params_list) == 0:
            return pack_param_seq(difficult_params_list), None
        else:
            return pack_param_seq(difficult_params_list), pack_param_seq(easy_params_list)

    def _select_solution_candidates(
        self, candidates: List[Trajectory], task_params: np.ndarray
    ) -> Tuple[Trajectory, int]:
        logger.info("select single solution out of {} candidates".format(len(candidates)))

        candidates_repeated = []
        n_task = len(task_params)
        n_cand = len(candidates)
        for cand in candidates:
            candidates_repeated.extend([cand] * n_task)
        tasks_repeated = pack_param_seq(list(task_params) * n_cand)
        results = self.solver.solve_batch(tasks_repeated, candidates_repeated)
        assert len(results) == n_task * n_cand

        def split_list(lst, n):
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        results_list = split_list(
            results, n_task
        )  # split the result such that each list corresponds to candidate trajectory

        n_solved_max = 0
        best_cand: Optional[Trajectory] = None
        for idx_cand, results in enumerate(results_list):
            n_solved = sum([r.traj is not None for r in results])
            logger.info("cand_idx {}: {}".format(idx_cand, n_solved))
            if n_solved > n_solved_max:
                n_solved_max = n_solved
                best_cand = candidates[idx_cand]
        logger.info("n_solved_max of candidates: {}".format(n_solved_max))
        assert best_cand is not None
        return best_cand, n_solved_max

    def _determine_init_solution(self, n_difficult: int) -> Tuple[Trajectory, float]:
        n_repeat_budget = 2
        for i_repeat in range(n_repeat_budget):
            logger.info("sample solution candidates ({}-th repeat)".format(i_repeat))
            solution_candidates = self._sample_solution_canidates(
                self.config.n_solution_candidate, self.task_pool
            )
            self.sampler_history.candidates_history.append(solution_candidates)

            logger.info("sample difficult tasks")
            if self.at_first_iteration():
                difficult_params = pack_param_seq(
                    [next(self.task_pool) for _ in range(n_difficult)]
                )
                n_total = len(difficult_params)
            else:
                difficult_params, easy_params = self._sample_difficult_tasks(
                    n_difficult, self.task_pool
                )
                if easy_params is None:
                    n_total = len(difficult_params)
                else:
                    n_total = len(difficult_params) + len(easy_params)

            best_solution, n_solved_max = self._select_solution_candidates(
                solution_candidates, difficult_params
            )
            # the rate of solved difficult tasks / all
            rate_difficult_solved = n_solved_max / n_total
            if best_solution is not None:
                logger.info(f"rate of difficult tasks solved: {rate_difficult_solved}")
                logger.info("found best solution")
                return best_solution, rate_difficult_solved
        raise RuntimeError("consumed all repeat budget")

    def measure_real_est(self, lib: SolutionLibrary, task_params: np.ndarray) -> RealEstAggregate:
        logger.info("**compute est values")
        cost_est_list = []
        init_solutions_est_list = []
        for task_param in tqdm.tqdm(task_params):
            task = self.task_type.from_task_param(task_param)
            infer_result = lib.infer(task)
            cost_est_list.append(infer_result.cost)
            init_solutions_est_list.append(infer_result.init_solution)
        logger.info("**compute real values")

        results = self.solver.solve_batch(task_params, init_solutions_est_list)  # type: ignore

        cost_real_list = []
        for result in results:
            if result.traj is None:
                cost_real_list.append(np.inf)
            else:
                cost_real_list.append(result.n_call)

        aggregate = RealEstAggregate(
            np.array(cost_real_list), np.array(cost_est_list), self.solver_config.n_max_call
        )
        logger.info(aggregate)
        return aggregate

    def measure_coverage(self, task_params: np.ndarray) -> float:
        total_count = 0
        success_count = 0
        for task_param in task_params:
            task = self.task_type.from_task_param(task_param)
            infer_res = self.library.infer(task)
            total_count += 1
            if infer_res.cost < self.solver_config.n_max_call:
                success_count += 1
        return success_count / total_count
