import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
WIDTH = 960
HEIGHT = 560
BG_COLOR = '#FFFFFF'
INK = '#17313C'
TEAL = '#008C9E'
GOLD = '#D17A1B'
LEFT_X_CENTER = 215
RIGHT_RECT = {'x_min': 588, 'x_max': 904, 'y_min': 80, 'y_max': 396}
ARROW_Y = 238
HEADER_Y = 36
COUNTER_Y = 435
LABEL_Y = 481
DUR_STATIC_PREVIEW = 1200
DUR_INITIAL_PARTIAL = 500
DUR_STEP = 80
FRAMES_PER_STEP = 6
DUR_FINAL_HOLD = 1600
LEFT_HEADS = []
for j in range(12):
    y = 139 + j * 18
    LEFT_HEADS.append({'label': f'H{j + 1}', 'y': y, 'x_points': [188, 242]})

def get_font(size, bold=False):
    p = Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf')
    if not p.exists():
        p = Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
    return ImageFont.truetype(str(p), size)

def cosine_ease(t):
    return (1 - np.cos(np.pi * t)) / 2

def generate_recursive_points(level):
    if level == 0:
        return np.array([[0.0, 0.0]])
    parent_pts = generate_recursive_points(level - 1)
    num_parents = len(parent_pts)
    child_pts = []
    mag = 1.0 / 2 ** ((level + 1) // 2)
    if level % 2 == 1:
        axis = 0
    else:
        axis = 1
    for pt in parent_pts:
        p1 = pt.copy()
        p2 = pt.copy()
        if axis == 0:
            p1[0] += mag
            p2[0] -= mag
        else:
            p1[1] += mag
            p2[1] -= mag
        child_pts.append(p1)
        child_pts.append(p2)
    return np.array(child_pts)

def scale_points_to_rect(points, rect):
    center = np.array([(rect['x_min'] + rect['x_max']) / 2, (rect['y_min'] + rect['y_max']) / 2])
    return np.asarray(points) * 150.0 + center

def draw_static_elements(d, complete=True):
    d.text((LEFT_X_CENTER, HEADER_Y), '12 separately normalized heads', font=get_font(23, bold=True), fill=INK, anchor='mm')
    d.text((745, HEADER_Y), ('One exact single-head readout' if complete else 'Building the Cartesian product'), font=get_font(23, bold=True), fill=INK, anchor='mm')
    for head in LEFT_HEADS:
        y = head['y']
        d.text((134, y), head['label'], font=get_font(15), fill=INK, anchor='rm')
        for x in head['x_points']:
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=TEAL)
    d.line((356, ARROW_Y, 494, ARROW_Y), fill=GOLD, width=4)
    d.polygon([(512, ARROW_Y), (491, ARROW_Y - 10), (491, ARROW_Y + 10)], fill=GOLD)
    d.text((432, 210), 'exact fusion', font=get_font(17), fill=INK, anchor='mm')

def draw_left_counter(d, active_head_idx=None):
    d.text((LEFT_X_CENTER, COUNTER_Y), '24', font=get_font(54, bold=True), fill=TEAL, anchor='mm')
    d.text((LEFT_X_CENTER, LABEL_Y), 'head–atom terms', font=get_font(21), fill=INK, anchor='mm')
    if active_head_idx is not None and 0 <= active_head_idx < 12:
        y = LEFT_HEADS[active_head_idx]['y']
        for x in LEFT_HEADS[active_head_idx]['x_points']:
            d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=GOLD, width=2)

def draw_right_grid(d, points, level):
    radius = max(1.2, 7 - 0.5 * level)
    for x, y in points:
        d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=GOLD)
    d.text((745, COUNTER_Y), str(2 ** level), font=get_font(54, bold=True), fill=GOLD, anchor='mm')
    unit = 'single-head atoms' if level == 12 else 'partial-product atoms'
    d.text((745, LABEL_Y), unit, font=get_font(21), fill=INK, anchor='mm')
    progress = '2^12 = 4096 · exact minimum' if level == 12 else f'{level} of 12 choices: 2^{level} combinations'
    d.text((745, 525), progress, font=get_font(17), fill=INK, anchor='mm')

def generate_animation():
    out = Path(__file__).resolve().parent
    levels = {j: scale_points_to_rect(generate_recursive_points(j), RIGHT_RECT) for j in range(13)}
    assert sum((len(h['x_points']) for h in LEFT_HEADS)) == 24
    assert [len(levels[j]) for j in range(13)] == [2 ** j for j in range(13)]
    frames = []
    durations = []

    def add(points, level, active, duration):
        im = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        d = ImageDraw.Draw(im)
        draw_static_elements(d, complete=(level == 12))
        draw_left_counter(d, active)
        draw_right_grid(d, points, level)
        frames.append(im)
        durations.append(duration)
    add(levels[12], 12, None, DUR_STATIC_PREVIEW)
    add(levels[0], 0, None, DUR_INITIAL_PARTIAL)
    for j in range(1, 13):
        parents = np.repeat(levels[j - 1], 2, axis=0)
        for f in range(FRAMES_PER_STEP):
            phase = cosine_ease((f + 1) / FRAMES_PER_STEP)
            points = parents + (levels[j] - parents) * phase
            assert np.all(points >= [RIGHT_RECT['x_min'], RIGHT_RECT['y_min']])
            assert np.all(points <= [RIGHT_RECT['x_max'], RIGHT_RECT['y_max']])
            active = None if j == 12 and f == FRAMES_PER_STEP - 1 else j - 1
            add(points, j, active, DUR_STEP)
    add(levels[12], 12, None, DUR_FINAL_HOLD)
    palette = frames[0].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    colors = palette.getpalette()
    white = next((i for i in range(256) if colors[3 * i:3 * i + 3] == [255, 255, 255]))
    path = out / 'atom_cost.gif'
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations, loop=0, background=white, disposal=2, optimize=True)
    qa = {'dimensions': [WIDTH, HEIGHT], 'stage_counts': [2 ** j for j in range(13)], 'durations_ms': durations, 'global_palette': True, 'gif_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / 'atom_cost_gif_qa.json').write_text(json.dumps(qa, indent=2) + '\n')
if __name__ == '__main__':
    generate_animation()
