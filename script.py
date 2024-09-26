
from os.path import exists
from pathlib import Path
import uuid
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from tetris_env import TetrisEnv

romName = 'tetris.gb'


def create_env(i, config, seed=0):
    def _init():
        env = TetrisEnv(config)
        env.reset(seed=(seed + i))
        return env
    set_random_seed(seed)
    return _init


# required for Vectorized Environments as per https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vec-env
if __name__ == "__main__":
    # Normal gameboy speed = 60 frames per second
    # 1 action every 3 frames
    # 20 actions per second
    # 20 action/second * 60s = 1200 actions in 1 minute
    # episode length 1200 should equate to the AI playing tetris for 1 minute in every episode
    # every learning step has a number of episodes
    # with 5 learning steps, 20 episodes, 1200 steps in each episode, 3 frames between every action, and a normal game speed of 60 fps
    # the AI should learn Tetris for 100 minutes per CPU
    # If we use 10 threads to run 1 enviroment each, the AI will learn 1000 minutes of tetris in 100 minutes.
    # if we speed up the emulation to about 20x speed and do that on 10 threads/enviroments, the AI will learn 1000 minutes of Tetris in 5 minutes.

    frames_per_action = 3  # amount of frames between each action. The fastest the game can update seems to be every 3 frames when dropping a block
    episode_length = 1024 * 20  # number of steps in an episode
    episodes = 40  # number of episodes in each training iteration
    threads = 16  # number of enviroments used to train model in parallel
    learn_steps = 10  # (I don't think my interpretations of learn_steps is correct) number of gradient steps. how many times the model will try to learn / how many times a step towards the global minimum will be taken with gradient descent
    session_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
        'headless': True,
        'debug': False,  # pyboy debug mode
        'no_input': False,  # disable pyboy accepting user input
        'save_final_state': True,
        'action_frequency': frames_per_action,
        'max_actions': episode_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': False,
        'session_path': session_path,
        'rom_path': 'tetris.gb',
        'initial_state': 'tetris.gb.state'
    }

    env = SubprocVecEnv([create_env(i, env_config) for i in range(threads)])

    checkpoint_callback = CheckpointCallback(
        episode_length, session_path, 'tetris')

    # CNN => Convolutional Neural Network, this type of neural netowork is supposed to be good at working with image data
    # default learning rate for PPO is 0.0003
    # batch_size is the number of experiences that will be used in the stochastic gradient descent when trying to optimize
    # batch_size should be a factor of episode length * threads
    # n_epochs determines how may times the model will train with a batch (batch_size) of experiences when learn is called on the model
    # gamme is the discount factor, the discount factor will encourage the model to seek future rewards instead of immediate rewards.
    # higher gamma encourages the model to seek future rewards instead of short term rewards
    # https://openai.com/research/openai-five

    model = PPO(
        'CnnPolicy',
        env,
        learning_rate=0.0003,  # default 0.0003
        verbose=1,
        batch_size=512,
        n_steps=episode_length,
        n_epochs=10,
        gamma=0.999
    )

    for i in range(learn_steps):
        print(f'Learn Step {i}')
        # episode_length * threads to allow every enviroment to all the way to its max_actions
        # the internal step counter of the model is incremented by the number of enviroments after each iteration
        model.learn(
            total_timesteps=episode_length * threads * episodes, # *100,
            callback=checkpoint_callback
        )
