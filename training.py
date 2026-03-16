import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import ClipReward

# --- OFF TRACK PENALTY WRAPPER ---
class OffTrackPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.01, penaltyFactor = 5):
        super().__init__(env)
        self.penalty = penalty
        self.penaltyFactor = penaltyFactor

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        car = self.env.unwrapped.car
        wheels_on_grass = 0

        if car is not None:
            for wheel in car.wheels:
                if len(wheel.tiles) == 0:
                    wheels_on_grass += 1

            totalPenalty = 0.0
            if wheels_on_grass == 1:
                totalPenalty = self.penalty
            elif wheels_on_grass >= 2:
                totalPenalty = self.penalty * self.penaltyFactor * wheels_on_grass

            if wheels_on_grass > 0:
                reward -= totalPenalty

        return obs, reward, terminated, truncated, info

# --- RANDOM START WRAPPER ---
class RandomStartWrapper(gym.Wrapper):
    def __init__(self, env, min_speed=20, max_speed=40):
        super().__init__(env)
        self.min_speed = min_speed
        self.max_speed = max_speed

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        car = self.env.unwrapped.car

        if car is not None:
            speed = np.random.uniform(self.min_speed, self.max_speed)

            # 20% delle volte parte con un angolo casuale
            if np.random.rand() < 0.2:
                random_angle = np.random.uniform(-0.25, 0.25)
                car.hull.angle += random_angle

            angle = car.hull.angle
            vel_x = speed * np.cos(angle)
            vel_y = speed * np.sin(angle)
            car.hull.linearVelocity = (vel_x, vel_y)

        return obs, info

# --- COSTRUZIONE AMBIENTE ---
def make_env():
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    
    env = GrayscaleObservation(env, keep_dim=True)
    env = RandomStartWrapper(env)
    env = ClipReward(env, max_reward=2.0)
    env = OffTrackPenaltyWrapper(env, penalty=0.02)
    
    return env

# --- SETUP TRAINING ---
env = DummyVecEnv([make_env])

# Parametri PPO standard
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=0, 
    device="cuda"
)

TOTAL_STEPS = 300_000

model.learn(total_timesteps=TOTAL_STEPS)
    
# Salvataggio Finale

model.save("model.zip")
env.close()