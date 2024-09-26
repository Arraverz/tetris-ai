from pyboy import PyBoy
from pyboy.utils import WindowEvent
import uuid
from pathlib import Path
from skimage.transform import resize_local_mean
import numpy as np
import random
import mediapy as media
import matplotlib.pyplot as plt
import json
import pandas as pd

from gymnasium import Env, spaces

# normal speed at level 0 a block goes down by 1 every 53 frames (I've seen it move down at 52 frames as well, though that only happened once)
# normal speed at level 9 a block goes down by 1 every 11 frames
# in hard mode at level 9 a block goes down by 1 every 4 frames
# normal down movement is independent from left or right movement
# holding down the block moves every 3 frames
# holding left or right stops the drop from holding down
# moving left or right updates every 9 frames


class TetrisEnv(Env):
    def __init__(self, config=None):
        self.rom_path = config['rom_path']
        self.initial_state = config['initial_state']
        self.action_frequency = config['action_frequency']
        self.max_actions = config['max_actions']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.headless = config['headless']
        self.session_path = config['session_path']
        self.instance_id = str(uuid.uuid4())[:8]
        self.all_runs = []
        self.agent_stats = []

        self.reset_count = 0
        self.action_count = 0
        self.reward_components = {
            'score': 0,
            'lines': 0,
            'invalid_moves': 0,
            'row_fullness': 0
        }
        self.invalid_move_penalties = 0
        self.total_reward = 0

        assert isinstance(self.session_path,
                          Path), "session_path must be instance of Path"
        self.session_path.mkdir(exist_ok=True)

        # actions sent through the API are toggled and need to be released.
        # releasing a button is a valid action
        self.valid_actions = [
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_DOWN,
            # WindowEvent.PASS, # maybe pass should not be an option either adjust left or right, spin the piece, or let it fall fast
        ]

        # height 144, width: 160, 3 color channels RGP
        self.game_shape = (144, 160, 3)
        self.input_box_size = self.game_shape[1] // 10
        self.padding_height = 4
        self.output_shape = (
            self.game_shape[0] + self.input_box_size + self.padding_height,
            self.game_shape[1],
            self.game_shape[2]
        )
        self.reduced_shape = (
            self.output_shape[0] // 4,
            self.output_shape[1] // 4,
            self.output_shape[2]
        )

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.reduced_shape, dtype=np.uint8)

        window_type = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
            gamerom_file=self.rom_path,
            window_type=window_type,
            debug=config['debug'],
            no_input=config['no_input'],
        )

        if not self.headless:
            self.pyboy.set_emulation_speed(20)

        self.screen = self.pyboy.botsupport_manager().screen()

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)

        with open(self.initial_state, "rb") as f:
            self.pyboy.load_state(f)

        # go from menu screen to the game screen
        # self.go_to_game_screen()

        self.agent_stats = []

        if self.save_video:
            base_dir = self.session_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)

            name = Path(
                f'reset_{self.reset_count}_id_{self.instance_id}').with_suffix('.mp4')

            self.frame_writer = media.VideoWriter(
                base_dir / name, self.output_shape[:2], fps=60)
            self.frame_writer.__enter__()

        self.action_count = 0
        self.reward_components = {
            'score': 0,
            'lines': 0,
            'invalid_moves': 0,
            'row_fullness': 0
        }
        self.invalid_move_penalties = 0
        self.total_reward = 0
        self.reset_count += 1

        return self.render(), {}

    def go_to_game_screen(self):
        # wait for a random amount of frames to make sure we get different RNG
        random_number = int(random.random() * 600)

        for i in range(random_number):
            self.pyboy.tick()
            pass

        # go to game mode select screen
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for i in range(5):
            self.pyboy.tick()
        # go to level select screen
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        for i in range(5):
            self.pyboy.tick()

        # start game
        self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
        self.pyboy.tick()
        self.pyboy.tick()

    def render(self, reduce_res=True, simple=True):
        width = self.game_shape[1]

        if simple:
            game_render = self.get_simple_render()
        else:
            game_render = self.screen.screen_ndarray()

        # boxes at the top of the screen representing the inputs A, B, Left, Right, Up
        input_render = self.get_input_render(self.input_box_size, width)
        # small black bar separating the input_render and the game_render
        padding_render = np.zeros(
            shape=(self.padding_height, width, 3), dtype=np.uint8)

        full_render = np.concatenate((
            input_render,
            padding_render,
            game_render
        ), axis=0)

        if reduce_res:
            full_render = (
                # maybe use downscale_local_mean instead?
                255*resize_local_mean(full_render, self.reduced_shape)
            ).astype(np.uint8)

        return full_render

    def get_simple_render(self):
        def make_box(value):
            return np.full(
                shape=(8, 8, 3),
                fill_value=value,
                dtype=np.uint8
            )

        game_render = self.screen.screen_ndarray()
        width = self.game_shape[1]
        height = self.game_shape[0]

        white = np.full((3,), 255, dtype=np.uint8)
        simple_render = np.zeros((0, width, 3), dtype=np.uint8)
        for i in range(0, height, 8):
            row = np.zeros((8, 0, 3), dtype=np.uint8)

            for j in range(0, width, 8):
                pixel = game_render[i][j]
                row = np.concatenate((
                    row,
                    make_box(value=255 if np.array_equal(pixel, white) else 0)
                ), axis=1)

            simple_render = np.concatenate((
                simple_render,
                row
            ), axis=0)

        return simple_render

    def get_input_render(self, boxSize, screenWidth):
        def make_input_box(active):
            return np.full(
                shape=(boxSize, boxSize, 3),
                fill_value=255 if active == True else 0,
                dtype=np.uint8
            )

        # Inputs are read from the I/O part of the memory, which is a single byte at 0xFF00
        arrow_input = self.get_arrow_input()
        button_input = self.get_button_input()
        border_padding = np.zeros(
            shape=(boxSize, boxSize//2, 3), dtype=np.uint8)
        between_box_render = np.zeros(
            shape=(boxSize, boxSize, 3), dtype=np.uint8)

        boxA = make_input_box(button_input['a'])
        boxB = make_input_box(button_input['b'])
        boxLeft = make_input_box(arrow_input['left'])
        boxRight = make_input_box(arrow_input['right'])
        boxDown = make_input_box(arrow_input['down'])

        input_render = np.concatenate((
            border_padding,
            boxA,
            between_box_render,
            boxB,
            between_box_render,
            boxLeft,
            between_box_render,
            boxRight,
            between_box_render,
            boxDown,
            border_padding,
        ), axis=1, dtype=np.uint8)

        if input_render.shape != (boxSize, screenWidth, 3):
            diff = screenWidth - input_render.shape[1]
            if diff == 0:
                raise Exception(
                    f'input_render width {input_render.shape[1]} equal to {screenWidth} but input_render.size is not equal to ({boxSize}, {screenWidth}, 3)')
            if diff < 0:
                raise Exception(
                    f'input_render width {input_render.shape[1]} is greater than {screenWidth}')

            fill = np.zeros(shape=(boxSize, diff, 3), dtype=np.uint8)
            input_render = np.concatenate(
                (input_render, fill), axis=1,  dtype=np.uint8
            )

        return input_render

    def get_arrow_input(self):
        # reset Bit 4 in 0xFF00 to select D-Pad inputs
        # 0 means input is set 1
        self.pyboy.set_memory_value(0xFF00, 0b11101111)
        joystick_byte = self.pyboy.get_memory_value(0xFF00)
        return {
            'right': not (joystick_byte & (1 << 0)),
            'left': not (joystick_byte & (1 << 1)),
            'up': not (joystick_byte & (1 << 2)),
            'down': not (joystick_byte & (1 << 3)),
        }

    def get_button_input(self):
        # reset Bit 5 in 0xFF00 to select D-Pad inputs
        # 0 means input is set 1
        self.pyboy.set_memory_value(0xFF00, 0b11011111)
        joystick_byte = self.pyboy.get_memory_value(0xFF00)
        return {
            'a': not (joystick_byte & (1 << 0)),
            'b': not (joystick_byte & (1 << 1)),
            'select': not (joystick_byte & (1 << 2)),
            'start': not (joystick_byte & (1 << 3)),
        }

    def step(self, action):

        self.execute_on_emulator(action)

        observation_memory = self.render()
        terminated = self.game_over()
        truncated = self.max_actions_reached()

        old_reward = self.total_reward
        self.update_reward(action, terminated, truncated)

        gained_reward = self.total_reward - old_reward

        self.save_and_print_info(terminated, truncated, observation_memory)

        self.action_count += 1

        return observation_memory, gained_reward, terminated, truncated, {}

    def execute_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)

        for i in range(self.action_frequency):
            tetrimino_collided = self.pyboy.get_memory_value(0xFF9B)
            # min x = 16, max x = 88
            # J => 129, 2 => 130, box => Tile 131, L => 132, T => Tile 133, S => 134, Line (horizontal) => 138,139,139,143, Line (vertical) => 128,136,136,137
            # Active sprite seems to always be in index 4 through 7, Next sprite is in 8 through 11
            if (tetrimino_collided and not self.pyboy.botsupport_manager().sprite(4).on_screen):
                # release all inputs when the tetrimo has landed and is no longer a sprite
                # I hope this solves the issue of the game ignoring Down input when the active tetrimo changes
                self.release_all_inputs()

            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.action_frequency-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()

        if self.save_video and self.fast_video:
            self.add_video_frame()

    def release_all_inputs(self):
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)

    def get_reward_components(self, action, terminated, truncated):
        reward = {
            'invalid_moves': 0,
            'lines': 0,
            'row_fullness': 0
        }

        reward['invalid_moves'] = self.get_invalid_move_penalty(action)
        reward['lines'] = self.get_lines()

    def update_reward(self, action, terminated, truncated):
        # 50% penalty for game over and don't calculate other rewards
        if terminated and self.total_reward > 0:
            self.total_reward -= abs(self.total_reward * 0.5)
            return

        # penalty for invalid moves like pressing a button that's alread held or releasing a button that's not held
        self.reward_components['invalid_moves'] += self.get_invalid_move_penalty(
            action)

        # new_score_reward = self.get_score()
        # if self.reward_components['score'] < new_score_reward:
        #     self.reward_components['score'] = new_score_reward

        new_line_reward = self.get_lines()
        if self.reward_components['lines'] < new_line_reward:
            self.reward_components['lines'] = new_line_reward

        rows = self.get_row_fullness()
        row_fullness = 0
        for i in range(len(rows)):
            # ignore full rows.
            # full rows get their own reward.
            # line count in memory updats before the row disappears
            # so awaring something for a full line here would be come a penalty when the row disappearts
            if rows[i] == 10:
                continue
            # top 2 rows should not be filled penalize it heavily
            if i in [0, 1]:
                row_fullness += -4*rows[i]
            elif i in [2, 3, 4, 5]:
                row_fullness += -2*rows[i]
            elif i in [6, 7, 8, 9]:
                row_fullness += 0*rows[i]
            elif i in [10, 11, 12, 13]:
                row_fullness += 2*rows[i]
            elif i in [14, 15, 16, 17]:
                row_fullness += 4*rows[i]

        self.reward_components['row_fullness'] = row_fullness

        # check for holes, holes are bad

        # predict position of piece if we just let it fall from where it is right now and use that to calculate the reward
        # use sprite map and background map, use np.roll on sprite map until any part of the piece would overlap with the background
        # active sprite seems to be in index 4-7 and I can get the 4 sprites that make up the active tetrimo with
        # self.pyboy.botsupport_manager().sprite(index)
        # each sprite has x and y coordinates. need to divide those by 8 to get the correct index for the background map
        # if I'm only looking at the game area I also need to subtract 16 first

        self.total_reward = self.reward_components['score'] + \
            self.reward_components['lines'] * 1024 + \
            self.reward_components['invalid_moves'] + \
            self.reward_components['row_fullness']

        # 50% bonus for surviving until max_actions
        if truncated and self.total_reward > 0:
            self.total_reward = self.total_reward * 1.5

    def get_invalid_move_penalty(self, action) -> float:
        button_input = self.get_button_input()
        arrow_input = self.get_arrow_input()
        input_action = self.valid_actions[action]

        # holding an arrow is valid
        # pressing down while left or right is already pressed will result in the down arrow being ignored
        # however holding left or right while already holding down is valid
        # cannot release a button that is not being held

        if input_action == WindowEvent.PRESS_BUTTON_A and button_input['a']:
            return -0.1
        elif input_action == WindowEvent.PRESS_BUTTON_B and button_input['b']:
            return -0.1
        elif input_action == WindowEvent.PRESS_ARROW_DOWN and (arrow_input['left'] or arrow_input['right']):
            return -0.1
        elif input_action == WindowEvent.RELEASE_BUTTON_A and not button_input['a']:
            return -0.1
        elif input_action == WindowEvent.RELEASE_BUTTON_B and not button_input['b']:
            return -0.1
        elif input_action == WindowEvent.RELEASE_ARROW_LEFT and not arrow_input['left']:
            return -0.1
        elif input_action == WindowEvent.RELEASE_ARROW_RIGHT and not arrow_input['right']:
            return -0.1
        elif input_action == WindowEvent.RELEASE_ARROW_DOWN and not arrow_input['down']:
            return -0.1

        return 0

    def get_row_fullness(self):
        map = self.pyboy.botsupport_manager().tilemap_background()
        map_array = map[2:12, 0:18]
        # 18 lines. First lines is highest on the screen
        lines = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for row in range(18):
            for col in range(10):
                tile = map_array[row][col]
                if tile != 47:
                    lines[row] += 1

        return lines

    def add_video_frame(self):
        self.frame_writer.add_image(
            self.render(reduce_res=False, simple=False))

    def append_agent_stats(self, action):
        score = self.reward_components['score']
        lines = self.reward_components['lines']
        invalid_moves = self.reward_components['invalid_moves']
        row_fullness = self.reward_components['row_fullness']
        level = self.get_level()

        # level doesn't affect the reward, because it's already factored into score, but it might be interesting for stats
        self.agent_stats.append({
            'action': self.action_count,
            'reward': self.total_reward,
            'score': score,
            'lines': lines,
            'invalid_moves': invalid_moves,
            'row_fullness': row_fullness,
            'level': level,
        })

    def get_score(self) -> int:
        # last 2 digits => one and ten digit
        score_2 = self.pyboy.get_memory_value(0xC0A0)
        # middle 2 digits => hundred and thousand digit
        score_1 = self.pyboy.get_memory_value(0xC0A1)
        # first 2 digits => ten thousand and hundred thousand digit
        score_0 = self.pyboy.get_memory_value(0xC0A2)

        return int(f'{score_0:02x}{score_1:02x}{score_2:02x}')

    def get_lines(self) -> int:
        lines_lo = self.pyboy.get_memory_value(0xFF9E)
        lines_hi = self.pyboy.get_memory_value(0xFF9F)
        return int(f'{lines_hi:02x}{lines_lo:02x}')

    def get_level(self) -> int:
        return self.pyboy.get_memory_value(0xFFA9)

    def game_over(self) -> bool:
        game_state = self.pyboy.get_memory_value(0xFFE1)
        # gamestate 4 is showing the game over screen
        # gamestate 13 is the death animation where the screen is filling with lines
        return game_state == 4 or game_state == 13

    def max_actions_reached(self) -> bool:
        return self.action_count >= self.max_actions

    def save_and_print_info(self, terminated, truncated, obs_memory):
        done = terminated or truncated
        if self.print_rewards:
            score = self.reward_components['score']
            lines = self.reward_components['lines']
            invalid_moves = self.reward_components['invalid_moves']
            row_fullness = self.reward_components['row_fullness']
            prog_string = f'step: {self.action_count:6d}  reward: {self.total_reward:10.2f}'
            prog_string += f'  score: {score:6d}'
            prog_string += f'  lines: {lines:4d}'
            prog_string += f'  row_fullness: {row_fullness:6d}'
            prog_string += f'  invalid_moves: {invalid_moves:8.1f}'
            if (terminated):
                prog_string += ' Terminated'
            if (truncated):
                prog_string += ' Truncated'

            print(f'\r{prog_string}', end='', flush=True)

        if self.action_count % 50 == 0:
            plt.imsave(
                self.session_path / Path(f'curframe_{self.instance_id}.png'),
                self.render()
            )

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.session_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path /
                    Path(
                        f'frame_{self.instance_id}_r{self.total_reward}_{self.reset_count}.png'),
                    obs_memory
                )

        if self.save_video and done:
            self.frame_writer.close()

        if done:
            self.all_runs.append(self.total_reward)
            with open(self.session_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)

            pd.DataFrame(self.agent_stats).to_csv(
                self.session_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')
