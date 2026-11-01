import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "lab1_pi_vi"))
from mountain_car_setup import N_ACTIONS, state_index


def height(position):
    """Height of the track at a given position (same formula as in the env's source code)."""
    return np.sin(3 * position) * 0.45 + 0.55


def energy(obs):
    """Mechanical energy of the car in the env's physics: potential (g * h) + kinetic (v^2 / 2).

    g = 0.0025 and h = sin(3p) / 3 come from the env's update: velocity += -0.0025 * cos(3p).
    Rolling freely keeps the energy constant; only the engine can add (or remove) energy.
    """
    position, velocity = obs
    return 0.0025 * np.sin(3 * position) / 3 + velocity ** 2 / 2


def height_reward(reward, obs, next_obs, k):
    """Reward function 1: reward + k * (height(s') - height(s)).

    Why height: to reach the flag the car must first roll back to gain speed.
    Height grows when climbing either side of the valley, so the swing itself is rewarded.
    """
    return reward + k * (height(next_obs[0]) - height(obs[0]))


def energy_reward(reward, obs, next_obs, k):
    """Reward function 2: reward + k * (energy(s') - energy(s)).

    Not punished for rolling down (height turns into speed, energy stays the same),
    rewarded only when the engine adds energy to the swing.
    """
    return reward + k * (energy(next_obs) - energy(obs))


def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))  # explore
    return int(Q[s].argmax())                # exploit


def mc_control(n_pos, n_vel, n_episodes=20_000, gamma=0.99, alpha=0.02, k=100, reward_fn=height_reward,
               eps_start=1.0, eps_decay=0.9995, eps_min=0.05, log_every=500, seed=0):
    """Learn Q only from played episodes

    After each episode, go backwards through it computing the return G_t, and update
    Q(s, a) <- Q(s, a) + alpha * (G_t - Q(s, a)) at the first visit of (s, a) in the episode.
    """
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    Q = np.zeros((n_pos * n_vel, N_ACTIONS))
    epsilon = eps_start
    log = {"episode": [], "mean_steps": [], "success_%": []}
    steps_hist, success_hist = [], []
    start = time.perf_counter()

    for ep in range(n_episodes):
        # 1. play one full episode with the epsilon-greedy policy
        obs, _ = env.reset(seed=seed + ep)
        episode = []  # (state, action, shaped reward)
        terminated = truncated = False
        while not (terminated or truncated):
            s = state_index(obs, n_pos, n_vel)
            a = epsilon_greedy(Q, s, epsilon, rng)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            episode.append((s, a, reward_fn(reward, obs, next_obs, k)))
            obs = next_obs

        # 2. find the first visit of every (s, a)
        first_visit = {}
        for t, (s, a, _) in enumerate(episode):
            first_visit.setdefault((s, a), t)

        # 3. go backwards, accumulate G_t, update Q at first visits
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G
            if first_visit[(s, a)] == t:
                Q[s, a] += alpha * (G - Q[s, a])

        # 4. decay exploration
        epsilon = max(eps_min, epsilon * eps_decay)

        # logging: learning curve
        steps_hist.append(len(episode))
        success_hist.append(terminated)
        if (ep + 1) % log_every == 0:
            log["episode"].append(ep + 1)
            log["mean_steps"].append(np.mean(steps_hist[-log_every:]))
            log["success_%"].append(np.mean(success_hist[-log_every:]) * 100)

    env.close()
    return {"Q": Q, "policy": Q.argmax(axis=1), "log": log, "time": time.perf_counter() - start,
            "env_steps": int(np.sum(steps_hist))}


def test_policy(policy, n_pos, n_vel, k, reward_fn=height_reward, n_episodes=100):
    """Greedy runs in the real env. Reports the original reward AND the shaped one."""
    env = gym.make("MountainCar-v0")
    steps, success, shaped = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_shaped, n, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            next_obs, reward, terminated, truncated, _ = env.step(int(policy[state_index(obs, n_pos, n_vel)]))
            total_shaped += reward_fn(reward, obs, next_obs, k)
            obs, n = next_obs, n + 1
        steps.append(n)
        success.append(terminated)
        shaped.append(total_shaped)
    env.close()
    steps, success = np.array(steps), np.array(success)
    return {
        "mean_steps": steps[success].mean() if success.any() else np.nan,
        "success_%": success.mean() * 100,
        "mean_reward": -steps.mean(),  # original reward: -1 per step
        "mean_shaped_reward": np.mean(shaped),
    }
