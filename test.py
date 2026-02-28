import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

SEED = 1
MODEL = "models/model.zip"

def make_test_env():
    env = gym.make("CarRacing-v3", max_episode_steps=1000, render_mode="human")
    env = GrayscaleObservation(env, keep_dim=True)
    return env

 
env = DummyVecEnv([make_test_env])

env.seed(SEED)

model = PPO.load(MODEL, device="cpu")

obs = env.reset() 

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, info = env.step(action)
        
        env.render()
        
        if dones[0]:
            obs = env.reset()
            print("Episodio terminato")
            
except KeyboardInterrupt:
    pass
    
env.close()