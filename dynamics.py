"""Domain (dynamics) randomization for MuJoCo environments.

Section VII of the survey: a policy trained on one fixed simulator often fails on a
real robot whose mass, friction or motors differ slightly. Peng et al. [11] showed
that randomizing these physical parameters during training makes the policy robust
to them. This wrapper does exactly that for any Gymnasium MuJoCo task:

  * At every reset, each body's mass is multiplied by a factor drawn from
    U[1 - mass_range, 1 + mass_range].
  * Every contact surface's sliding friction is multiplied by a factor drawn from
    U[1 - friction_range, 1 + friction_range].

FixedDynamics applies one fixed change instead (used to test a trained policy on
"a different robot", as a stand-in for the reality gap).
"""
import gymnasium as gym
import numpy as np


class _DynamicsBase(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        model = env.unwrapped.model
        self._mass0 = model.body_mass.copy()
        self._friction0 = model.geom_friction[:, 0].copy()

    def _set(self, mass_scale, friction_scale):
        model = self.env.unwrapped.model
        model.body_mass[:] = self._mass0 * mass_scale
        model.geom_friction[:, 0] = self._friction0 * friction_scale


class DomainRandomization(_DynamicsBase):
    """Draw new physical parameters at every episode."""

    def __init__(self, env, mass_range=0.3, friction_range=0.4):
        super().__init__(env)
        self.mass_range, self.friction_range = mass_range, friction_range

    def reset(self, **kwargs):
        rng = self.env.unwrapped.np_random
        n_body, n_geom = len(self._mass0), len(self._friction0)
        mass_scale = rng.uniform(1 - self.mass_range, 1 + self.mass_range, n_body)
        friction_scale = rng.uniform(1 - self.friction_range, 1 + self.friction_range, n_geom)
        self._set(mass_scale, friction_scale)
        return self.env.reset(**kwargs)


class FixedDynamics(_DynamicsBase):
    """Scale every mass and friction by one fixed factor (a 'different robot')."""

    def __init__(self, env, mass_scale=1.0, friction_scale=1.0):
        super().__init__(env)
        self._set(mass_scale, friction_scale)
