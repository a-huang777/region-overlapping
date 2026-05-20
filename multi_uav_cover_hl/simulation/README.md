# 3D Simulation Visualizer

Run from the project root:

```bash
source /Users/gengmengen/miniconda3/etc/profile.d/conda.sh
conda activate test
python simulation/simulate_high_level_3d.py
```

Default models:

- Low-level model: `low_model/ppo_model_save.zip`
- High-level model: `check_point_high_level/version_3/model/ppo_high_level_final.zip`

Useful options:

```bash
python simulation/simulate_high_level_3d.py --episodes 5
python simulation/simulate_high_level_3d.py --frame-stride 2 --dpi 100
python simulation/simulate_high_level_3d.py --high-model check_point_high_level/version_2/model/ppo_high_level_final.zip
```

Generated GIFs are saved under `simulation/renders/`.

## Interactive HTML

The HTML exporter creates a browser-viewable simulation that can be rotated
with the mouse and zoomed with the wheel:

```bash
python simulation/simulate_high_level_3d_html.py
```

Useful options:

```bash
python simulation/simulate_high_level_3d_html.py --episodes 5
python simulation/simulate_high_level_3d_html.py --frame-stride 2
python simulation/simulate_high_level_3d_html.py --initial-zoom 14 --max-zoom 80 --render-scale 1.5
python simulation/simulate_high_level_3d_html.py --high-model check_point_high_level/version_2/model/ppo_high_level_final.zip
```

Generated HTML files are saved under `simulation/html/`.

Inside the HTML page:

- Drag the canvas to rotate the camera.
- Use the mouse wheel or the `Zoom` slider to change scene scale.
- Use the `Render` slider to trade sharpness for rendering speed.
