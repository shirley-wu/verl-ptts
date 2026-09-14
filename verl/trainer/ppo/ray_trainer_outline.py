import logging
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict, DictConfig



from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.outline_utils import get_outline_formatter
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.reward import compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.import_utils import load_class_from_fqn
from verl.checkpoint_engine import CheckpointEngineManager
from verl.workers.config import FSDPEngineConfig
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool
from verl.single_controller.ray.base import split_resource_pool
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


logger = logging.getLogger(__name__)


class OutlineTrainer(RayPPOTrainer):
    def __init__(self, outline_formatter, config, *args, **kwargs):
        self.outline_formatter = outline_formatter
        super().__init__(config, *args, **kwargs)
        assert self.use_reward_loop
        assert config.outline.outline_prompt_key == config.data.prompt_key

    def _get_solution_gen_batch(self, test_batch, outlines_parsed):
        assert len(test_batch) == len(outlines_parsed)

        solution_prompt = test_batch.non_tensor_batch[self.config.outline.solution_prompt_key]
        data_source = test_batch.non_tensor_batch["data_source"]
        reward_model = test_batch.non_tensor_batch["reward_model"]

        # Build prompts for valid Q+O pairs
        outline_inds = []
        solution_inds = []
        gen_prompt = []
        gen_data_source = []
        gen_reward_model = []
        for i, outlines in enumerate(outlines_parsed):
            if outlines is not None:
                outline_inds.append(i)
                solution_inds.append(list(range(len(gen_prompt), len(gen_prompt) + len(outlines))))
                orig_prompt = solution_prompt[i]
                for j, outline in enumerate(outlines_parsed[i]):
                    gen_prompt.append([
                        {'role': turn['role'], 'content': turn['content'].replace("{outline}", outline)}
                        for turn in orig_prompt
                    ])
                    gen_data_source.append(data_source[i])
                    gen_reward_model.append(reward_model[i])

        outline_inds = np.array(outline_inds)
        solution_inds = np.array(solution_inds)
        if len(outline_inds) == 0:
            gen_batch = None
        else:
            gen_batch = DataProto(batch=None, non_tensor_batch={
                "raw_prompt": np.array(gen_prompt, dtype=object),
                "data_source": np.array(gen_data_source, dtype=object),
                "reward_model": np.array(gen_reward_model, dtype=object),
            })

        return gen_batch, outline_inds, solution_inds

    def _reward_outline_by_solution(self, outline_gen_batch, solution_gen_batch, outline_inds, solution_inds):
        solution_scores = solution_gen_batch.batch['rm_scores'].sum(dim=1)
        solution_scores = solution_scores[solution_inds].max(dim=1).values
        # allow partial format reward
        solution_scores[solution_scores == 0] = self.config.outline.outline_format_reward

        response_last_pos = outline_gen_batch.batch['response_mask'].sum(1) - 1
        outline_gen_batch.batch['rm_scores'][:] = 0
        outline_gen_batch.batch['rm_scores'][np.arange(len(response_last_pos)), response_last_pos] = \
            self.config.outline.outline_failed_reward

        response_last_pos = response_last_pos[outline_inds]
        outline_gen_batch.batch['rm_scores'][outline_inds, response_last_pos] = solution_scores
        return outline_gen_batch


    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            test_batch_copy = deepcopy(test_batch)

            # The invocation of the reward function is agnostic to whether a reward model is used.
            # Decisions about when (e.g., training vs. validation) and whether to invoke the reward model
            # are delegated to user-defined reward functions.
            # if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            #     return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
                "max_tokens": self.config.outline.outline_max_response_length,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad and process outputs first
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # sample_outputs.extend(output_texts)  # let's store the whole set of output outlines and solutions

            # 2nd rollout
            output_texts_parsed = self.outline_formatter.parse_outlines_batched(output_texts)
            solution_gen_batch, outline_inds_to_gen_solution, solution_inds_to_gen_solution = \
                self._get_solution_gen_batch(test_batch_copy, output_texts_parsed)
            solution_gen_batch.meta_info = {
                "eos_token_id": self.executor_tokenizer.eos_token_id,
                "pad_token_id": self.executor_tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            solution_gen_batch_padded, solution_pad_size = pad_dataproto_to_divisor(solution_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                raise NotImplementedError("Outline RL with frozen executor does not support non-async rollout mode.")
            else:
                if(self.config.outline.frozen_solution_rollout.enable):
                    self.checkpoint_manager.sleep_replicas()
                    self.async_executor_manager.wake_up()
                    solution_output_gen_batch_padded = self.async_executor_manager.generate_sequences(solution_gen_batch_padded)
                    self.async_executor_manager.sleep_replicas()
                    self.checkpoint_manager.update_weights()
                else:
                    solution_output_gen_batch_padded = \
                        self.async_rollout_manager.generate_sequences(solution_gen_batch_padded)
            solution_output_gen_batch = unpad_dataproto(solution_output_gen_batch_padded, pad_size=solution_pad_size)
            # assign pass@4 reward
            assert "rm_scores" in test_output_gen_batch.batch.keys(), "rm_scores not calculated"
            assert "rm_scores" in solution_output_gen_batch.batch.keys(), "rm_scores not calculated"
            test_output_gen_batch = self._reward_outline_by_solution(
                test_output_gen_batch, solution_output_gen_batch,
                outline_inds_to_gen_solution, solution_inds_to_gen_solution
            )

            # record output
            solution_ids = solution_output_gen_batch.batch["responses"]
            solution_texts = [self.executor_tokenizer.decode(ids, skip_special_tokens=True) for ids in solution_ids]
            outputs_to_store = [
                {'outline_text': output_texts[i], 'outline_parsed': output_texts_parsed[i], 'solutions': None}
                for i in range(len(output_texts))
            ]
            for i, j in zip(outline_inds_to_gen_solution, solution_inds_to_gen_solution):
                outputs_to_store[i]['solutions'] = [solution_texts[jj] for jj in j]
            sample_outputs.extend(outputs_to_store)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                raise NotImplementedError  # not supported by the outline trainer

                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights()

            print("validation generation end")

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            if not self.use_reward_loop:
                raise NotImplementedError  # not supported by the outline trainer
                reward_tensor, reward_extra_info = self._compute_reward_legacy(
                    test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
                )
            else:
                reward_tensor = test_batch.batch["rm_scores"]
                reward_extra_keys = test_batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: test_batch.non_tensor_batch[key] for key in reward_extra_keys}

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _get_frozen_executor_config(self):
        """
        Initialize config object for frozen solution model by setting the specified model path
        """
        frozen_executor_config = deepcopy(self.config)
        with open_dict(frozen_executor_config.actor_rollout_ref.model):
            frozen_executor_config.actor_rollout_ref.model.path = self.config.outline.frozen_solution_rollout.model_path
        
        with open_dict(frozen_executor_config.actor_rollout_ref.rollout):
            frozen_executor_config.actor_rollout_ref.rollout.enable_sleep_mode = True
            # so that sleep_replicas really frees the engine
            frozen_executor_config.actor_rollout_ref.rollout.free_cache_engine = True
            # load real weights instead of the default dummy weights (no weight sync for the frozen executor)
            frozen_executor_config.actor_rollout_ref.rollout.load_format = "auto"
        
        return frozen_executor_config 

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if self.use_rm and not self.use_reward_loop:
            raise RuntimeError("Reward model worker group is not supported, please set use_reward_loop=True")

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        if self.use_reward_loop:
            from verl.experimental.reward_loop import RewardLoopManager

            # initalize reward loop manager
            # reward model (colocate or standalone): get resource_pool
            # no reward model: resource_pool = None
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
            self.reward_loop_manager = RewardLoopManager(
                config=self.config,
                rm_resource_pool=resource_pool,
            )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = self.use_reward_loop and (
            not self.use_rm or self.config.reward_model.enable_resource_pool
        )
        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout

        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

        self.executor_tokenizer = self.tokenizer

        if(self.config.outline.frozen_solution_rollout.enable):
            executor_config = self._get_frozen_executor_config()
    
            executor_local_path = copy_to_local(executor_config.actor_rollout_ref.model.path)
            self.executor_tokenizer = hf_tokenizer(executor_local_path, trust_remote_code=True)

            if self.use_reward_loop:
                self.executor_reward_loop_manager = FrozenExecutorRewardLoopManager(
                    config=executor_config,
                    rm_resource_pool=resource_pool,
                )

            executor_reward_loop_handles = self.executor_reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

            self.async_executor_manager = FrozenExecutorAgentLoopManager(
                config=executor_config,
                worker_group=None,
                rollout_resource_pool=actor_rollout_resource_pool,
                reward_loop_worker_handles=executor_reward_loop_handles,
            )

            self.async_executor_manager.sleep_replicas()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["max_tokens"] = self.config.outline.outline_max_response_length

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                gen_batch_output_copy = deepcopy(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile(global_step=self.global_steps)
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            if(self.config.outline.frozen_solution_rollout.enable):
                                # Sleep rollout generator if we use a separate frozen executor model
                                self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    output_texts = self.tokenizer.batch_decode(gen_batch_output.batch["responses"],
                                                               skip_special_tokens=True)

                    # 2nd rollout
                    output_texts_parsed = self.outline_formatter.parse_outlines_batched(output_texts)
                    solution_gen_batch, outline_inds_to_gen_solution, solution_inds_to_gen_solution = \
                        self._get_solution_gen_batch(gen_batch_output_copy, output_texts_parsed)
                    solution_gen_batch.meta_info = {"global_steps": self.global_steps}

                    size_divisor = (
                        self.actor_rollout_wg.world_size
                        if not self.async_rollout_mode
                        else self.config.actor_rollout_ref.rollout.agent.num_workers
                    )
                    solution_gen_batch, solution_pad_size = pad_dataproto_to_divisor(solution_gen_batch, size_divisor)

                    # wrap within the timer
                    with marked_timer("step", timing_raw):
                        # generate a batch
                        with marked_timer("gen-solution", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                raise NotImplementedError("Outline RL with frozen executor does not support non-async rollout mode.")
                            else:
                                if(self.config.outline.frozen_solution_rollout.enable):
                                    if curr_step_profile:
                                        self.async_executor_manager.start_profile(global_step=self.global_steps)
                                    self.async_executor_manager.wake_up()
                                    solution_gen_batch = self.async_executor_manager.generate_sequences(solution_gen_batch)
                                    self.async_executor_manager.sleep_replicas()
                                    if curr_step_profile:
                                        self.async_executor_manager.stop_profile()
                                else:
                                    if curr_step_profile:
                                        self.async_rollout_manager.start_profile(global_step=self.global_steps)
                                    solution_gen_batch = self.async_rollout_manager.generate_sequences(solution_gen_batch)
                                    self.checkpoint_manager.sleep_replicas()
                                    if curr_step_profile:
                                        self.async_rollout_manager.stop_profile()

                            timing_raw.update(solution_gen_batch.meta_info["timing"])
                            solution_gen_batch.meta_info.pop("timing", None)

                    # assign pass@4 reward
                    #important - rollout manager need to be initialized with the reward model to compute RM score
                    assert "rm_scores" in gen_batch_output.batch.keys(), "rm_scores not calculated"
                    assert "rm_scores" in solution_gen_batch.batch.keys(), "rm_scores not calculated"
                    solution_gen_batch = unpad_dataproto(solution_gen_batch, pad_size=solution_pad_size)
                    gen_batch_output = self._reward_outline_by_solution(
                        gen_batch_output, solution_gen_batch,
                        outline_inds_to_gen_solution, solution_inds_to_gen_solution
                    )

                    # record output
                    solution_texts = self.executor_tokenizer.batch_decode(solution_gen_batch.batch["responses"],
                                                                 skip_special_tokens=True)
                    outputs_to_store = [
                        {'outline_text': output_texts[i], 'outline_parsed': output_texts_parsed[i], 'solutions': None}
                        for i in range(len(output_texts))
                    ]
                    for i, j in zip(outline_inds_to_gen_solution, solution_inds_to_gen_solution):
                        outputs_to_store[i]['solutions'] = [solution_texts[jj] for jj in j]

                    gen_batch_output.non_tensor_batch['outputs_to_log'] = np.array(outputs_to_store, dtype=object)

                    assert self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX, (
                        "REMAX is not supported by the outline trainer"
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            raise NotImplementedError  # not supported by the outline trainer

                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # Compute or extract reward_tensor and reward_extra_infos_dict for training
                        if not self.use_reward_loop:
                            raise NotImplementedError  # not supported by the outline trainer

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = self._compute_reward_legacy(
                                    batch, reward_fn=self.reward_fn, reward_for_val=False
                                )
                        else:
                            reward_tensor = batch.batch["rm_scores"]
                            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                            reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                                is_last_step
                                or self.global_steps % self.config.trainer.save_freq == 0
                                or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def _log_rollout_data(
            self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = batch.non_tensor_batch["outputs_to_log"].tolist()
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
 
def get_outline_trainer(config, *args, **kwargs):
    outline_formatter = get_outline_formatter(config.outline)
    return OutlineTrainer(config=config, outline_formatter=outline_formatter, *args, **kwargs)

class FrozenExecutorAgentLoopManager(AgentLoopManager):
    def _initialize_llm_servers(self, rollout_resource_pool: RayResourcePool):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = rollout_resource_pool.world_size
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        # Use a disjoint rank range so the frozen executor's named Ray actors do not
        # collide with the trainable rollout replicas that already use ranks 0..N-1.
        replica_rank_offset = world_size
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank_offset + replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        # Important: init with colocate mode so wake up doesn't sync weight into server.
        split_resource_pools = split_resource_pool(rollout_resource_pool, split_size=rollout_world_size)
        assert len(split_resource_pools) == len(self.rollout_replicas)
        # init_colocated: keep the executor workers in separate processes from the trainable rollout workers
        self._run_all(
            [
                server.init_colocated(resource_pool)
                for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
            ]
        )
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")
    
    def wake_up(self):
        self._run_all([r.wake_up() for r in self.rollout_replicas])
    
    def sleep_replicas(self):
        self._run_all([r.sleep() for r in self.rollout_replicas])

class FrozenExecutorRewardLoopManager(RewardLoopManager):
    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        super().__init__(config, rm_resource_pool)

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward_model.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.reward_loop_workers.append(
                self.reward_loop_workers_class.options(
                    name=f"frozen_executor_reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
            )
