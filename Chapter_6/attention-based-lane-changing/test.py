import gym
import highway_env
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

env_config = {
    "id": "highway-v0",
    "import_module": "highway_env",
    "lanes_count": 3,
    "vehicles_count": 30,
    "duration": 30,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    # "centering_position": [0.3, 0.5],
    "observation": {
                "type": "GrayscaleObservation",
                "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
                "stack_size": 1,
                "observation_shape": (150, 600)
                },
    # "observation": {
    #     "type": "Kinematics",
    #     "vehicles_count": 15,
    #     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    #     "features_range": {
    #         "x": [-100, 100],
    #         "y": [-100, 100],
    #         "vx": [-20, 20],
    #         "vy": [-20, 20]
    #     },
    #     # "absolute": True,
    #     # "order": "shuffled"
    # },
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    # "destination": "o1"
}

env = gym.make("highway-v0")
env.unwrapped.configure(env_config)
# Reset the environment to ensure configuration is applied
env.reset()

done = False
while not done:
    action = np.random.randint(5)
    obs, reward, done, _ = env.step(action)
    # obs = np.squeeze(obs)
    # img = Image.fromarray(obs.astype('uint8')).convert('L')
    # img.save('obs.png')
    print(obs.shape)
    # plt.pause(0.01)
    # plt.imshow(obs)
    env.render()


# python experiments.py evaluate configs/HighwayEnv/env_attention.json \
#                                configs/HighwayEnv/agents/DQNAgent/ego_attention.json \
#                                --train --episodes=4000 --name-from-config
