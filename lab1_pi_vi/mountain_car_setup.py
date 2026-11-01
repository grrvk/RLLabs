"""MountainCar-v0: discretization, model building, Policy Iteration, Value Iteration, testing."""

import time

import gymnasium as gym
import numpy as np

POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07
N_ACTIONS = 3  # 0 = left, 1 = do nothing, 2 = right


def discretize(obs, n_pos, n_vel):
    """Real (position, velocity) -> bin indices (i, j)."""
    i = int((obs[0] - POS_MIN) / (POS_MAX - POS_MIN) * n_pos)
    j = int((obs[1] - VEL_MIN) / (VEL_MAX - VEL_MIN) * n_vel)
    return min(max(i, 0), n_pos - 1), min(max(j, 0), n_vel - 1)


def undiscretize(i, j, n_pos, n_vel):
    """Bin indices (i, j) -> real coordinates of the bin centre."""
    pos = POS_MIN + (i + 0.5) * (POS_MAX - POS_MIN) / n_pos
    vel = VEL_MIN + (j + 0.5) * (VEL_MAX - VEL_MIN) / n_vel
    return pos, vel


def state_index(obs, n_pos, n_vel):
    """Real observation -> single state number s = i * n_vel + j (row in the P/R tables)."""
    i, j = discretize(obs, n_pos, n_vel)
    return i * n_vel + j


def build_model(n_pos, n_vel):
    """For every (state, action): put the car at the bin centre, do one step, record the result.

    P[s, a] - next state, R[s, a] - reward, done[s, a] - True if this step reached the flag.
    """
    env = gym.make("MountainCar-v0")
    env.reset()
    n_states = n_pos * n_vel
    P = np.zeros((n_states, N_ACTIONS), dtype=int)
    R = np.zeros((n_states, N_ACTIONS))
    done = np.zeros((n_states, N_ACTIONS), dtype=bool)

    for i in range(n_pos):
        for j in range(n_vel):
            s = i * n_vel + j
            for a in range(N_ACTIONS):
                env.unwrapped.state = np.array(undiscretize(i, j, n_pos, n_vel))
                next_obs, reward, terminated, _, _ = env.unwrapped.step(a)
                P[s, a] = state_index(next_obs, n_pos, n_vel)
                R[s, a] = reward
                done[s, a] = terminated
    env.close()
    return P, R, done


def q_values(V, P, R, done, gamma):
    """Q(s, a) = R(s, a) + gamma * V(s') for all states and actions; no future value after the flag."""
    return R + gamma * V[P] * (~done)


def policy_iteration(P, R, done, gamma=0.99, theta=1e-4, max_iters=10_000):
    n_states = len(P)
    states = np.arange(n_states)
    policy = np.ones(n_states, dtype=int)  # start: "do nothing" everywhere
    V = np.zeros(n_states)
    total_sweeps = 0
    changes = []  # how many states changed their action at each outer iteration
    start = time.perf_counter()

    for outer in range(1, max_iters + 1):
        # Policy Evaluation: V(s) = R(s, pi(s)) + gamma * V(s') until it stops changing
        r, next_s, not_done = R[states, policy], P[states, policy], ~done[states, policy]
        for _ in range(max_iters):
            V_new = r + gamma * V[next_s] * not_done
            delta = np.max(np.abs(V_new - V))
            V = V_new
            total_sweeps += 1
            if delta < theta:
                break

        # Policy Improvement: pi'(s) = argmax_a Q(s, a)
        new_policy = q_values(V, P, R, done, gamma).argmax(axis=1)
        changes.append(int(np.sum(new_policy != policy)))
        if changes[-1] == 0:
            break
        policy = new_policy

    return {"V": V, "policy": policy, "iterations": outer, "eval_sweeps": total_sweeps,
            "changes": changes, "time": time.perf_counter() - start}


def value_iteration(P, R, done, gamma=0.99, epsilon=1e-4, max_iters=10_000):
    V = np.zeros(len(P))
    deltas = []
    start = time.perf_counter()

    for it in range(1, max_iters + 1):
        V_new = q_values(V, P, R, done, gamma).max(axis=1)  # V(s) = max_a [R + gamma * V(s')]
        deltas.append(np.max(np.abs(V_new - V)))
        V = V_new
        if deltas[-1] < epsilon:
            break

    policy = q_values(V, P, R, done, gamma).argmax(axis=1)  # extract the policy once
    return {"V": V, "policy": policy, "iterations": it, "deltas": deltas,
            "time": time.perf_counter() - start}


def run_episode(policy, n_pos, n_vel, seed=None, render_mode=None):
    """Play one episode. Returns (total reward, reached flag?, list of observations, list of frames)."""
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    obs, _ = env.reset(seed=seed)
    total_reward, terminated, truncated = 0, False, False
    trajectory, frames = [obs], []
    while not (terminated or truncated):
        if render_mode == "rgb_array":
            frames.append(env.render())
        action = policy[state_index(obs, n_pos, n_vel)]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        trajectory.append(obs)
    env.close()
    return total_reward, terminated, np.array(trajectory), frames


def test_policy(policy, n_pos, n_vel, n_episodes=100):
    rewards, success = [], []
    for k in range(n_episodes):
        total_reward, reached, _, _ = run_episode(policy, n_pos, n_vel, seed=k)
        rewards.append(total_reward)
        success.append(reached)
    rewards, success = np.array(rewards), np.array(success)
    return {
        "mean_steps": -rewards[success].mean() if success.any() else np.nan,  # reward = -1 per step
        "success_%": success.mean() * 100,
        "mean_reward": rewards.mean(),
    }
