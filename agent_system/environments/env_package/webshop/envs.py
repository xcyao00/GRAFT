# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
        
        self._current_idx = None
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        # Carry the generic task identifier on every step so that the main
        # rollout loop can persist it into batch.non_tensor_batch['gamefile']
        # uniformly across environments.
        if self._current_idx is not None:
            info['extra.gamefile'] = str(int(self._current_idx))

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        
        self._current_idx = int(idx)
        info['extra.gamefile'] = str(self._current_idx)
        return obs, info

    def reset_to_state(self, idx, actions: list):
        """
        Reset the worker to the same goal as ``idx`` and replay ``actions``
        step by step in order to recover an intermediate state. Used by
        GraphGPO-Supp supplementary rollout for WebShop.

        Args:
            idx (int): Session / goal index used by the original rollout.
            actions (list[str]): List of *projected* action strings to replay
                (e.g. ``"search[red shoes]"`` / ``"click[buy now]"``). The
                strings should be exactly what was fed to ``self.env.step``
                during the original rollout so that the page sequence is
                identical.

        Returns:
            obs: Current observation after replaying the actions.
            info: Info dict (same shape as ``self.reset``) with
                ``available_actions``, ``won=False`` and
                ``extra.gamefile=str(idx)`` populated. ``task_score`` is
                only populated by ``self.step``; we set it to 0.0 here
                because the trajectory has not yet finished.
        """
        obs, _ = self.env.reset(session=idx)
        self._current_idx = int(idx)
        # Replay actions; if any step finishes the episode, the underlying
        # env may auto-reset (see ``WebAgentTextEnv.step``); we just stop
        # replaying in that case to avoid drifting onto a fresh session.
        for action in actions:
            try:
                obs, reward, done, _info = self.env.step(action)
            except Exception:
                # Best-effort replay: if a particular action no longer
                # corresponds to a clickable on the current page (e.g. due
                # to stochasticity in search results), stop here.
                break
            if done:
                break
        info = {
            'available_actions': self.env.get_available_actions(),
            'won': False,
            'task_score': 0.0,
            'extra.gamefile': str(int(idx)),
        }
        return obs, info

    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def reset_to_states(self, worker_indices: list, gamefiles: list, action_sequences: list):
        """
        Reset specific workers to intermediate states by re-seeding the same
        goal/session and replaying an action sequence. Mirrors the AlfWorld
        ``reset_to_states`` interface so that recipe-level supplementary
        rollout code can be written generically.

        Args:
            worker_indices (list[int]): Indices of workers to reset.
            gamefiles (list[str]): Per-worker session indices encoded as
                strings (the same value previously stored in
                ``info['extra.gamefile']``). The name ``gamefiles`` is kept
                for API compatibility with AlfWorld; for WebShop the value
                is interpreted as ``int(session_idx)``.
            action_sequences (list[list[str]]): Per-worker projected action
                strings to replay.

        Returns:
            obs_list (list[str]): Current observation per worker.
            info_list (list[dict]): Info dict per worker, populated with the
                same shape as ``WebshopWorker.reset``.
        """
        assert len(worker_indices) == len(gamefiles) == len(action_sequences), (
            "worker_indices, gamefiles and action_sequences must have the same length"
        )

        futures = []
        for idx, gamefile, actions in zip(worker_indices, gamefiles, action_sequences):
            try:
                session_idx = int(gamefile)
            except (TypeError, ValueError):
                session_idx = 0  # fallback; will likely fail to replay,
                                 # supplementary_rollout treats this as a skip.
            future = self._workers[idx].reset_to_state.remote(session_idx, list(actions))
            futures.append(future)

        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )