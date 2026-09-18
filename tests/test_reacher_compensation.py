import unittest
from dataclasses import replace
import numpy as np
import torch
from sdob.examples.reacher.config import load_experiment_config
from sdob.examples.reacher.fixed_goal_runner import (ActionBiasCostModel, DisturbanceMPPIObserver, StabilizedMPPISolver, clip_action_to_bounds)

def _norm(value):
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))

class ReacherCompensationTests(unittest.TestCase):
    def test_action_bias_cost_model_clamps_rollout_actions_and_penalizes_bounds(self):
        class FakeCostModel:
            def __init__(self):
                self.last_candidates = None

            def parameters(self):
                return iter([torch.nn.Parameter(torch.ones(()))])

            def get_cost(self, info_dict, action_candidates):
                self.last_candidates = action_candidates.detach().clone()
                return torch.zeros(action_candidates.shape[:2])

        fake = FakeCostModel()
        model = ActionBiasCostModel(
            fake,
            action_dim=2,
            synthesis_gain=0.5,
            bound_penalty_weight=10.0,
        )
        model.set_bias(np.array([0.4, -0.2], dtype=np.float64))

        candidates = torch.tensor([[[[1.2, -1.3, 0.0, 0.0]]]])
        costs = model.get_cost({}, candidates)

        self.assertLessEqual(float(fake.last_candidates.max()), 1.0)
        self.assertGreaterEqual(float(fake.last_candidates.min()), -1.0)
        self.assertGreater(float(costs.item()), 0.0)


    def test_action_bias_cost_model_can_replace_terminal_cost_with_tail_cost(self):
        class FakeCostModel:
            def parameters(self):
                return iter([torch.nn.Parameter(torch.ones(()))])

            def get_cost(self, info_dict, action_candidates):
                info_dict["predicted_emb"] = torch.zeros(1, 1, 4, 2)
                info_dict["goal_emb"] = torch.ones(1, 1, 1, 2)
                return torch.full((1, 1), 10.0)

        model = ActionBiasCostModel(
            FakeCostModel(),
            action_dim=2,
            trajectory_cost_weight=0.5,
            trajectory_cost_steps=2,
            trajectory_cost_mode="replace",
        )
        costs = model.get_cost({"goal": torch.zeros(1, 1, 3, 4, 4)}, torch.zeros(1, 1, 1, 2))

        self.assertAlmostEqual(float(costs.item()), 1.0)


    def test_action_bias_regularization_penalizes_compensated_effective_action(self):
        class FakeCostModel:
            def parameters(self):
                return iter([torch.nn.Parameter(torch.ones(()))])

            def get_cost(self, info_dict, action_candidates):
                return torch.zeros(action_candidates.shape[:2])

        model = ActionBiasCostModel(
            FakeCostModel(),
            action_dim=2,
            synthesis_gain=1.0,
            bound_penalty_weight=0.0,
            action_reg_weight=10.0,
            smoothness_reg_weight=0.0,
        )
        model.set_bias(np.array([0.3, -0.2], dtype=np.float64))

        compensation_only = torch.tensor([[[[-0.3, 0.2]]]])
        cost = model.get_cost({}, compensation_only)

        self.assertAlmostEqual(float(cost.item()), 0.0, places=6)


    def test_action_bias_rollout_synthesis_makes_solver_choose_opposite_control(self):
        class FakeCostModel(torch.nn.Module):
            def __init__(self, desired_effective_action: float):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(()))
                self.desired_effective_action = float(desired_effective_action)

            def get_cost(self, info_dict, action_candidates):
                del info_dict
                return (
                    action_candidates[..., 0] - self.desired_effective_action
                ).square().mean(dim=-1)

        class FakeActionSpace:
            shape = (1, 1)
            low = np.array([[-1.0]], dtype=np.float32)
            high = np.array([[1.0]], dtype=np.float32)

        desired = 0.0
        disturbance_estimate = 0.12
        config = replace(
            load_experiment_config("sdob/examples/reacher/configs/reacher.yaml"),
            horizon=1,
            action_block=1,
            num_samples=1024,
            cem_steps=8,
            var_scale=0.5,
            mppi_temperature=0.05,
            adaptive_temperature=False,
            mppi_cost_normalize=False,
            mppi_min_variance=0.001,
            mppi_mean_update_rate=0.8,
            action_reg_weight=0.0,
            smoothness_reg_weight=0.0,
        )
        model = ActionBiasCostModel(
            FakeCostModel(desired),
            action_dim=1,
            synthesis_gain=1.0,
            action_reg_weight=0.0,
            smoothness_reg_weight=0.0,
        )
        model.set_bias(np.array([disturbance_estimate]))
        solver = StabilizedMPPISolver(model, config=config, batch_size=1, seed=7)
        solver.configure(action_space=FakeActionSpace(), n_envs=1, config=config)

        actions = solver.solve({"dummy": np.zeros((1, 1), dtype=np.float32)})["actions"]
        control_action = float(actions[0, 0, 0])
        effective_action = control_action + disturbance_estimate

        self.assertLess(abs(effective_action - desired), 0.04)
        self.assertLess(abs(control_action - (desired - disturbance_estimate)), 0.04)


    def test_disturbance_mppi_observer_scores_forward_rollouts(self):
        observer = DisturbanceMPPIObserver(
            action_dim=2,
            num_particles=256,
            particle_std=0.4,
            temperature=0.002,
            prior_weight=0.0,
            magnitude_prior_weight=0.0,
            filter_alpha=1.0,
            max_norm=1.0,
            seed=5,
        )
        observer.reset()
        nominal_block = np.array([[0.1, -0.2]], dtype=np.float64)
        true_d = np.array([0.25, -0.15], dtype=np.float64)
        feedback_latent = np.array(
            [2.0 * (nominal_block[0, 0] + true_d[0]), -3.0 * (nominal_block[0, 1] + true_d[1])],
            dtype=np.float64,
        )

        def rollout_fn(action_vectors):
            actions = torch.as_tensor(action_vectors[:, -2:], dtype=torch.float32)
            return np.asarray(
                torch.stack([2.0 * actions[:, 0], -3.0 * actions[:, 1]], dim=-1).tolist(),
                dtype=np.float64,
            )

        estimate = observer.update_from_rollout(
            rollout_fn=rollout_fn,
            action_block_plan=nominal_block,
            feedback_latent=feedback_latent,
            proposal_center=true_d,
        )

        self.assertLess(_norm(observer.last_innovation - true_d), 0.03)
        self.assertLess(_norm(estimate - true_d), 0.03)
        self.assertLess(observer.last_ess, 20.0)


    def test_clip_action_to_bounds_projects_infeasible_solver_output(self):
        action = np.array([2.5, -3.0], dtype=np.float64)

        clipped = clip_action_to_bounds(action)

        self.assertEqual(clipped.tolist(), [1.0, -1.0])
        self.assertEqual(action.tolist(), [2.5, -3.0])


