from pyboy import PyBoy
from pyboy import WindowEvent
import random
from time import sleep
import copy
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize, resize_local_mean

pyboy = PyBoy(
    'tetris.gb',
    debug=True,
    window_type='SDL2',  # headless or SDL2
    no_input=False,
    hide_window=True
)
validActions = [
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PASS,
]
screen = pyboy.botsupport_manager().screen()
counter = 0
state = {
    'score': 0,  # 0xC0A0, 0xC0A1, 0xC0A2 => 3-byte little-endian BCD. 123456 would be stored as $C0A0: $56 $C0A1: $34 $C0A2: $12
    # 'singles': 0,  # 0xC0AC   => Not a counter. this is set to 1 for 78 frames when 1 line is completed
    # 'doubles': 0,  # 0xC0B1   => Not a counter. this is set to 1 for 78 frames when 2 lines are completed
    # 'triples': 0,  # 0xC0B6   => Not a counter. this is set to 1 for 78 frames when 3 lines are completed
    # 'tetrises': 0,  # 0xC0BB  => Not a counter. this is set to 1 for 78 frames when 4 lines are completed
    # 'fastdrop_bonus_sum': 0,  # 0xC0C0
    # 'drops': 0,  # 0xC0C2 this seems unused. It's always 0 and the game does not have a hard drop button
    # 'slow_drop': 0,  # 0xC0C7
    # 'fastdrop_bonus_added': 0,  # 0xC0CE

    'level': 0,  # 0xFFA9
    'game_state': 0,  # 0xFFE1    gamestate = 4 => game over?
}
last_state = None


def reset():
    with open('tetris.gb.state', "rb") as f:
        pyboy.load_state(f)

    # wait for a random amount of frames to make sure we get different RNG
    random_number = int(random.random() * 600)
    print(f'wait {random_number} ticks')
    for i in range(random_number):
        pyboy.tick()
        pass

    # go to game mode select screen
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
    for i in range(5):
        pyboy.tick()
    # go to level select screen
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
    for i in range(5):
        pyboy.tick()

    # start game
    pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
    pyboy.tick()
    pyboy.tick()


game_shape = (144, 160, 3)
input_box_size = game_shape[1] // 10
padding_height = 4
full_output_shape = (
    int(game_shape[0] + input_box_size + padding_height),
    game_shape[1],
    game_shape[2]
)
reduced_shape = (
    full_output_shape[0] // 4,
    full_output_shape[1] // 4,
    full_output_shape[2]
)
print(full_output_shape)
print(reduced_shape)


def get_arrow_input():
    # reset Bit 4 in 0xFF00 to select D-Pad inputs
    # 0 means input is set 1
    pyboy.set_memory_value(0xFF00, 0b11101111)
    joystick_byte = pyboy.get_memory_value(0xFF00)
    return {
        'right': not (joystick_byte & (1 << 0)),
        'left': not (joystick_byte & (1 << 1)),
        'up': not (joystick_byte & (1 << 2)),
        'down': not (joystick_byte & (1 << 3)),
    }


def get_button_input():
    # reset Bit 5 in 0xFF00 to select D-Pad inputs
    # 0 means input is set 1
    pyboy.set_memory_value(0xFF00, 0b11011111)
    joystick_byte = pyboy.get_memory_value(0xFF00)
    return {
        'a': not (joystick_byte & (1 << 0)),
        'b': not (joystick_byte & (1 << 1)),
        'select': not (joystick_byte & (1 << 2)),
        'start': not (joystick_byte & (1 << 3)),
    }


def set_bit(value, bit):
    return value | (1 << bit)


def clear_bit(value, bit):
    return value & ~(1 << bit)


def render(reduce_res=True, simple=True):
    def make_box(value):
        return np.full(
            shape=(8, 8, 3),
            fill_value=value,
            dtype=np.uint8
        )

    game_render = screen.screen_ndarray()

    width = game_render.shape[1]
    height = game_render.shape[0]

    if simple:
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

    padding_render = np.zeros(shape=(padding_height, width, 3), dtype=np.uint8)

    input_render = get_input_render(input_box_size, width)

    full_render = np.concatenate((
        input_render,
        padding_render,
        simple_render if simple else game_render
    ), axis=0)
    if reduce_res:
        full_render = (
            # maybe use downscale_local_mean instead?
            255*resize_local_mean(full_render, reduced_shape)
        ).astype(np.uint8)

    return full_render


def get_row_fullness():
    map = pyboy.botsupport_manager().tilemap_background()
    map_array = map[2:12, 0:18]
    # 18 lines. First lines is highest on the screen
    lines = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for row in range(18):
        for col in range(10):
            tile = map_array[row][col]
            if tile != 47:
                lines[row] += 1

    return lines


def get_input_render(boxSize, screenWidth):
    def make_input_box(active):
        return np.full(
            shape=(boxSize, boxSize, 3),
            fill_value=255 if active == True else 0,
            dtype=np.uint8
        )

    # Inputs are read from the I/O part of the memory, which is a single byte at 0xFF00
    arrow_input = get_arrow_input()
    button_input = get_button_input()
    border_padding = np.zeros(shape=(boxSize, boxSize//2, 3), dtype=np.uint8)
    between_box_render = np.zeros(shape=(boxSize, boxSize, 3), dtype=np.uint8)

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


def get_game_area_sprite_map():
    # min x = 16, max x = 88
    # J => 129, 2 => 130, box => Tile 131, L => 132, T => Tile 133, S => 134, Line (horizontal) => 138,139,139,143, Line (vertical) => 128,136,136,137
    # Active sprite seems to always be in index 4 through 7, Next sprite is in 8 through 11
    if( not pyboy.botsupport_manager().sprite(4).on_screen):
        # tetrino collided and the sprite is not on screen => the tetrino has finished falling and is now part of the background 
        print('empty')

    # for s in range(40):
    #     sprite = pyboy.botsupport_manager().sprite(s)
    #     print(sprite)
    map = pyboy.botsupport_manager()

# pyboy.mb
# reset()

# pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
# pyboy.tick()


# frame_writer = media.VideoWriter('test.mp4', full_output_shape[:2], fps=60)
# small_frame_writer = media.VideoWriter(
#     'test_small.mp4', reduced_shape[:2], fps=60)
# frame_writer.__enter__()
# small_frame_writer.__enter__()
game_state = None
lines = None
level = None
row_fullness = None
while not pyboy.tick():

    # screen.screen_image().save(f'.\\test\\{counter}.jpg')
    # frame = render()
    # full_frame = render(reduce_res=False, simple=False)
    # simple_full_frame = render(reduce_res=False)

    # small_frame_writer.add_image(frame)
    # frame_writer.add_image(full_frame)
    # plt.imsave(
    #     Path(f'curframe.jpeg'),
    #     frame
    # )
    # plt.imsave(
    #     Path(f'full_curframe.jpeg'),
    #     simple_full_frame
    # )

    # lines_lo = pyboy.get_memory_value(0xFF9E)
    # lines_hi = pyboy.get_memory_value(0xFF9F)
    # new_lines = int(f'{lines_hi:02x}{lines_lo:02x}')

    # new_row_fullness = get_row_fullness()
    # for i in new_row_fullness:
    #     if i == 10:
    #         print(new_lines)
    #         print('test')
    # if row_fullness != new_row_fullness:
    #     print(new_row_fullness)
    #     row_fullness = new_row_fullness

    # if (lines != new_lines):
    #     print(new_lines)
    #     print(new_row_fullness)
    #     lines = new_lines

    # new_level = pyboy.get_memory_value(0xFFA9)
    # if (new_level != level):
    #     print(f'Level {new_level}')
    #     level = new_level
    tetrimino_collided = pyboy.get_memory_value(0xFF9B)
    if (tetrimino_collided != 0):
        get_game_area_sprite_map()
        # print("tetrimino_collided")

    new_game_state = pyboy.get_memory_value(0xFFE1)
    if (new_game_state != game_state):
        print(new_game_state)
        game_state = new_game_state
    #     if (new_game_state == 4):
    #         reset()

    counter += 1
    # if counter > 1000:
    #     break
    pass
# frame_writer.close()
# small_frame_writer.close()
# exit()

# while not pyboy.tick():
#     # screen.screen_image().save(f'.\\test\\{counter}.jpg')
#     # screen.screen_ndarray()
#     counter += 1

#     # last 2 digits => one and ten digit
#     score_2 = pyboy.get_memory_value(0xC0A0)
#     # middle 2 digits => hundred and thousand digit
#     score_1 = pyboy.get_memory_value(0xC0A1)
#     # first 2 digits => ten thousand and hundred thousand digit
#     score_0 = pyboy.get_memory_value(0xC0A2)
#     state['score'] = int(f'{score_0:02x}{score_1:02x}{score_2:02x}')
#     # state['singles'] = pyboy.get_memory_value(0xC0AC)
#     # state['doubles'] = pyboy.get_memory_value(0xC0B1)
#     # state['triples'] = pyboy.get_memory_value(0xC0B6)
#     # state['tetrises'] = pyboy.get_memory_value(0xC0BB)
#     # state['slow_drop'] = pyboy.get_memory_value(0xC0C2)

#     state['level'] = pyboy.get_memory_value(0xFFA9)
#     state['game_state'] = pyboy.get_memory_value(0xFFE1)

#     if (state != last_state):
#         print(f'Frames: {counter}')
#         print(state)
#         counter = 0

#     last_state = copy.copy(state)

#     pass
pyboy.stop()
