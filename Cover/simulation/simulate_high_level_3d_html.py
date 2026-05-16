"""
Run the trained high-level policy and export an interactive 3D HTML simulation.

This file intentionally avoids browser-side dependencies. The generated HTML
uses a small canvas renderer with mouse drag rotation, wheel zoom, and playback
controls, so it can be opened directly from disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(SIM_DIR / ".mplconfig"))
(SIM_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)

from stable_baselines3 import PPO

from simulation.simulate_high_level_3d import build_env, resolve_path, run_policy_rollout


def guidance_to_json(guidance_points: list[dict]) -> list[list[float]]:
    points = []
    for item in guidance_points:
        wx, wy = item["world"]
        points.append([float(wx), float(wy)])
    return points


def rollout_to_payload(
    env,
    result,
    episode_idx: int,
    frame_stride: int,
    altitude: float,
    initial_zoom: float,
    min_zoom: float,
    max_zoom: float,
    render_scale: float,
) -> dict:
    frames = result.frames[:: max(frame_stride, 1)]
    if frames and frames[-1] is not result.frames[-1]:
        frames.append(result.frames[-1])

    payload_frames = []
    trajectory = []
    for frame in frames:
        pos = frame["pos"][0]
        theta = float(frame["theta"][0])
        trajectory.append([float(pos[0]), float(pos[1])])
        payload_frames.append(
            {
                "pos": [float(pos[0]), float(pos[1]), float(altitude)],
                "theta": theta,
                "map": frame["map"].astype(int).tolist(),
                "guidance": guidance_to_json(frame.get("guidance_points", [])),
            }
        )

    low_env = env.low_env
    return {
        "episode": int(episode_idx),
        "envSize": float(low_env.cfg.env_size),
        "gridSize": int(low_env.cfg.grid_size),
        "gridRes": float(low_env.cfg.grid_res),
        "detRadius": float(low_env.cfg.det_radius),
        "uavRadius": float(low_env.cfg.uav_radius),
        "altitude": float(altitude),
        "start": [float(low_env.start_pos[0]), float(low_env.start_pos[1]), 0.15],
        "coverageRate": float(result.coverage_rate),
        "highSteps": int(result.high_steps),
        "totalReward": float(result.total_reward),
        "endReason": result.end_reason,
        "view": {
            "initialZoom": float(initial_zoom),
            "minZoom": float(min_zoom),
            "maxZoom": float(max_zoom),
            "renderScale": float(render_scale),
        },
        "trajectory": trajectory,
        "frames": payload_frames,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive 3D Coverage Simulation</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --line: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
    }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #app {
      display: grid;
      grid-template-rows: 1fr auto;
      height: 100%;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      background: linear-gradient(#1e293b, #0f172a 62%, #0b1120);
    }
    canvas:active { cursor: grabbing; }
    .hud {
      position: absolute;
      top: 14px;
      left: 14px;
      padding: 12px 14px;
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 8px;
      backdrop-filter: blur(8px);
      line-height: 1.5;
      font-size: 13px;
      pointer-events: none;
    }
    .hud strong {
      color: white;
      font-size: 14px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: var(--panel);
      border-top: 1px solid var(--line);
    }
    button {
      min-width: 76px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      font-size: 13px;
    }
    button:hover { border-color: var(--accent); }
    input[type="range"] { width: 100%; }
    .timeline {
      flex: 1 1 260px;
      min-width: 180px;
    }
    .controlGroup {
      display: grid;
      grid-template-columns: auto minmax(90px, 140px) auto;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .stat {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <div id="app">
    <div style="position: relative; min-height: 0;">
      <canvas id="scene"></canvas>
      <div class="hud" id="hud"></div>
    </div>
    <div class="controls">
      <button id="play">Pause</button>
      <input class="timeline" id="slider" type="range" min="0" max="0" value="0" />
      <label class="controlGroup">
        <span>Zoom</span>
        <input id="zoomSlider" type="range" min="2" max="40" step="0.1" value="11.5" />
        <span id="zoomStat">11.5x</span>
      </label>
      <label class="controlGroup">
        <span>Render</span>
        <input id="renderScaleSlider" type="range" min="0.5" max="3" step="0.25" value="1" />
        <span id="renderScaleStat">1.00x</span>
      </label>
      <button id="resetView">Reset View</button>
      <span class="stat" id="frameStat"></span>
      <span class="stat">Drag rotate · Wheel zoom</span>
    </div>
  </div>

  <script>
    const DATA = __DATA__;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const hud = document.getElementById("hud");
    const slider = document.getElementById("slider");
    const zoomSlider = document.getElementById("zoomSlider");
    const renderScaleSlider = document.getElementById("renderScaleSlider");
    const playButton = document.getElementById("play");
    const resetViewButton = document.getElementById("resetView");
    const frameStat = document.getElementById("frameStat");
    const zoomStat = document.getElementById("zoomStat");
    const renderScaleStat = document.getElementById("renderScaleStat");

    let frameIndex = 0;
    let playing = true;
    let yaw = -0.82;
    let pitch = -0.9;
    let zoom = DATA.view.initialZoom;
    let renderScale = DATA.view.renderScale;
    let dragging = false;
    let lastMouse = [0, 0];
    let lastTime = 0;

    slider.max = Math.max(DATA.frames.length - 1, 0);
    zoomSlider.min = String(DATA.view.minZoom);
    zoomSlider.max = String(DATA.view.maxZoom);
    zoomSlider.value = String(zoom);
    renderScaleSlider.value = String(renderScale);

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function syncViewControls() {
      zoomSlider.value = String(zoom);
      renderScaleSlider.value = String(renderScale);
      zoomStat.textContent = `${zoom.toFixed(1)}x`;
      renderScaleStat.textContent = `${renderScale.toFixed(2)}x`;
    }

    function setZoom(value) {
      zoom = clamp(value, DATA.view.minZoom, DATA.view.maxZoom);
      syncViewControls();
      render();
    }

    function setRenderScale(value) {
      renderScale = clamp(value, 0.5, 3.0);
      syncViewControls();
      resizeCanvas();
    }

    function resizeCanvas() {
      const ratio = (window.devicePixelRatio || 1) * renderScale;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      render();
    }

    function resetView() {
      yaw = -0.82;
      pitch = -0.9;
      zoom = DATA.view.initialZoom;
      renderScale = DATA.view.renderScale;
      syncViewControls();
      resizeCanvas();
      render();
    }

    function rotatePoint(p) {
      const cx = DATA.envSize / 2;
      const cy = DATA.envSize / 2;
      let x = p[0] - cx;
      let y = p[1] - cy;
      let z = p[2] || 0;

      const cyaw = Math.cos(yaw);
      const syaw = Math.sin(yaw);
      const x1 = x * cyaw - y * syaw;
      const y1 = x * syaw + y * cyaw;

      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);
      const y2 = y1 * cp - z * sp;
      const z2 = y1 * sp + z * cp;
      return [x1, y2, z2];
    }

    function project(p) {
      const rect = canvas.getBoundingClientRect();
      const r = rotatePoint(p);
      const scale = Math.min(rect.width, rect.height) * zoom / DATA.envSize;
      return [
        rect.width / 2 + r[0] * scale,
        rect.height / 2 + r[1] * scale,
        r[2]
      ];
    }

    function drawLine3(points, color, width = 1, alpha = 1) {
      if (points.length < 2) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      const first = project(points[0]);
      ctx.moveTo(first[0], first[1]);
      for (let i = 1; i < points.length; i++) {
        const p = project(points[i]);
        ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.restore();
    }

    function drawCircle3(center, radius, z, color, width = 1, alpha = 1, fill = false) {
      const pts = [];
      for (let i = 0; i <= 80; i++) {
        const a = i / 80 * Math.PI * 2;
        pts.push([center[0] + Math.cos(a) * radius, center[1] + Math.sin(a) * radius, z]);
      }
      drawLine3(pts, color, width, alpha);
      if (fill) {
        const start = project(pts[0]);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(start[0], start[1]);
        for (let i = 1; i < pts.length; i++) {
          const p = project(pts[i]);
          ctx.lineTo(p[0], p[1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
    }

    function polygonDepth(poly) {
      let depth = 0;
      for (const p of poly) depth += rotatePoint(p)[2];
      return depth / poly.length;
    }

    function drawGround(frame) {
      const cells = [];
      const gs = DATA.gridSize;
      const res = DATA.gridRes;
      for (let gx = 0; gx < gs; gx++) {
        for (let gy = 0; gy < gs; gy++) {
          const state = frame.map[gx][gy];
          let color = "#f8fafc";
          if (state === 1) color = "#7dd3fc";
          if (state === -1) color = "#475569";
          const x0 = gx * res;
          const y0 = gy * res;
          const x1 = x0 + res;
          const y1 = y0 + res;
          const poly = [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]];
          cells.push({ poly, color, depth: polygonDepth(poly) });
        }
      }
      cells.sort((a, b) => a.depth - b.depth);
      ctx.save();
      for (const cell of cells) {
        const p0 = project(cell.poly[0]);
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        for (let i = 1; i < cell.poly.length; i++) {
          const p = project(cell.poly[i]);
          ctx.lineTo(p[0], p[1]);
        }
        ctx.closePath();
        ctx.fillStyle = cell.color;
        ctx.globalAlpha = 0.88;
        ctx.fill();
        ctx.globalAlpha = 0.35;
        ctx.strokeStyle = "#94a3b8";
        ctx.lineWidth = 0.35;
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawPoint(p, radius, color, label = "") {
      const q = project(p);
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(q[0], q[1], radius, 0, Math.PI * 2);
      ctx.fill();
      if (label) {
        ctx.fillStyle = "#e5e7eb";
        ctx.font = "12px sans-serif";
        ctx.fillText(label, q[0] + 8, q[1] - 8);
      }
      ctx.restore();
    }

    function drawUav(frame) {
      const pos = frame.pos;
      const theta = frame.theta;
      const z = pos[2];
      const arm = 2.0;
      const c = Math.cos(theta);
      const s = Math.sin(theta);
      const local = [[arm, 0], [-arm, 0], [0, arm], [0, -arm]];
      const arms = local.map(([x, y]) => [pos[0] + x * c - y * s, pos[1] + x * s + y * c, z]);

      drawLine3([arms[0], arms[1]], "#111827", 3, 1);
      drawLine3([arms[2], arms[3]], "#111827", 3, 1);
      for (const r of arms) drawCircle3(r, 0.55, z + 0.1, "#020617", 1.8, 0.95);

      const nose = [pos[0] + Math.cos(theta) * 2.8, pos[1] + Math.sin(theta) * 2.8, z + 0.15];
      drawLine3([[pos[0], pos[1], z], nose], "#ef4444", 3, 1);
      drawPoint(pos, 7, "#2563eb", "UAV");
    }

    function drawSceneHelpers(frame) {
      const pos = frame.pos;
      drawCircle3([pos[0], pos[1]], DATA.detRadius, 0.08, "#2563eb", 1.5, 0.35);
      drawPoint(DATA.start, 5, "#1d4ed8", "Start");

      for (const g of frame.guidance) {
        drawPoint([g[0], g[1], 0.25], 6, "#ef4444", "Guidance");
        drawLine3([[g[0], g[1], 0.25], [g[0], g[1], DATA.altitude]], "#ef4444", 1.2, 0.45);
      }

      const trail = DATA.trajectory
        .slice(0, frameIndex + 1)
        .map(([x, y]) => [x, y, DATA.altitude * 0.72]);
      drawLine3(trail, "#f97316", 2.6, 0.92);
    }

    function drawAxes() {
      const e = DATA.envSize;
      drawLine3([[0, 0, 0], [e, 0, 0]], "#ef4444", 1.4, 0.85);
      drawLine3([[0, 0, 0], [0, e, 0]], "#22c55e", 1.4, 0.85);
      drawLine3([[0, 0, 0], [0, 0, DATA.altitude * 2]], "#38bdf8", 1.4, 0.85);
    }

    function render() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      const frame = DATA.frames[frameIndex];
      if (!frame) return;

      drawGround(frame);
      drawAxes();
      drawSceneHelpers(frame);
      drawUav(frame);

      hud.innerHTML = `
        <strong>Interactive 3D Coverage Simulation</strong><br>
        Episode: ${DATA.episode}<br>
        End: ${DATA.endReason}<br>
        Coverage: ${(DATA.coverageRate * 100).toFixed(2)}%<br>
        High steps: ${DATA.highSteps}<br>
        Reward: ${DATA.totalReward.toFixed(3)}
      `;
      frameStat.textContent = `Frame ${frameIndex + 1}/${DATA.frames.length}`;
      slider.value = String(frameIndex);
    }

    function tick(ts) {
      if (playing && DATA.frames.length > 0 && ts - lastTime > 125) {
        frameIndex = (frameIndex + 1) % DATA.frames.length;
        lastTime = ts;
        render();
      }
      requestAnimationFrame(tick);
    }

    canvas.addEventListener("mousedown", (event) => {
      dragging = true;
      lastMouse = [event.clientX, event.clientY];
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (event) => {
      if (!dragging) return;
      const dx = event.clientX - lastMouse[0];
      const dy = event.clientY - lastMouse[1];
      yaw += dx * 0.008;
      pitch = Math.max(-1.45, Math.min(-0.15, pitch + dy * 0.008));
      lastMouse = [event.clientX, event.clientY];
      render();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const step = event.shiftKey ? 1.03 : 1.1;
      setZoom(zoom * (event.deltaY > 0 ? 1 / step : step));
    }, { passive: false });
    slider.addEventListener("input", () => {
      frameIndex = Number(slider.value);
      render();
    });
    zoomSlider.addEventListener("input", () => {
      setZoom(Number(zoomSlider.value));
    });
    renderScaleSlider.addEventListener("input", () => {
      setRenderScale(Number(renderScaleSlider.value));
    });
    playButton.addEventListener("click", () => {
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    });
    resetViewButton.addEventListener("click", resetView);
    window.addEventListener("resize", resizeCanvas);

    syncViewControls();
    resizeCanvas();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


def write_html(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_text = json.dumps(payload, separators=(",", ":"))
    output_path.write_text(HTML_TEMPLATE.replace("__DATA__", data_text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an interactive 3D HTML simulation.")
    parser.add_argument("--low-model", default="low_model/ppo_model_save.zip", help="Path to the trained low-level PPO model.")
    parser.add_argument(
        "--high-model",
        default="check_point_high_level/version_3/model/ppo_high_level_final.zip",
        help="Path to the trained high-level PPO model.",
    )
    parser.add_argument("--output-dir", default="simulation/html", help="Directory for generated HTML files.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to export.")
    parser.add_argument("--seed", type=int, default=4000, help="Base random seed.")
    parser.add_argument("--grid-bins", type=int, default=5, help="High-level grid bins per axis.")
    parser.add_argument("--option-horizon", type=int, default=10, help="Low-level steps per high-level decision.")
    parser.add_argument("--max-high-steps", type=int, default=100, help="Maximum high-level decisions per episode.")
    parser.add_argument("--altitude", type=float, default=4.0, help="Rendered UAV altitude in meters.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Export every Nth recorded low-level frame.")
    parser.add_argument("--initial-zoom", type=float, default=11.5, help="Initial browser view zoom.")
    parser.add_argument("--min-zoom", type=float, default=2.0, help="Minimum browser view zoom.")
    parser.add_argument("--max-zoom", type=float, default=40.0, help="Maximum browser view zoom.")
    parser.add_argument("--render-scale", type=float, default=1.0, help="Initial canvas resolution multiplier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low_model = resolve_path(args.low_model)
    high_model = resolve_path(args.high_model)
    output_dir = resolve_path(args.output_dir)
    min_zoom = min(args.min_zoom, args.max_zoom)
    max_zoom = max(args.min_zoom, args.max_zoom)
    initial_zoom = min(max(args.initial_zoom, min_zoom), max_zoom)
    render_scale = min(max(args.render_scale, 0.5), 3.0)

    if not low_model.exists():
        raise FileNotFoundError(f"Low-level model not found: {low_model}")
    if not high_model.exists():
        raise FileNotFoundError(f"High-level model not found: {high_model}")

    env = build_env(
        low_model=low_model,
        grid_bins=args.grid_bins,
        option_horizon=args.option_horizon,
        max_high_steps=args.max_high_steps,
    )
    print(f"Loading high-level model: {high_model}")
    model = PPO.load(str(high_model), env=env, device="cpu")

    for episode_idx in range(1, args.episodes + 1):
        seed = args.seed + episode_idx
        print(f"Running episode {episode_idx} with seed {seed}...")
        result = run_policy_rollout(env, model, seed=seed)
        print(
            f"Episode {episode_idx}: end={result.end_reason}, "
            f"coverage={result.coverage_rate * 100:.2f}%, "
            f"high_steps={result.high_steps}, frames={len(result.frames)}, "
            f"reward={result.total_reward:.3f}"
        )
        payload = rollout_to_payload(
            env=env,
            result=result,
            episode_idx=episode_idx,
            frame_stride=args.frame_stride,
            altitude=args.altitude,
            initial_zoom=initial_zoom,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            render_scale=render_scale,
        )
        output_path = output_dir / f"high_level_3d_episode_{episode_idx}.html"
        write_html(payload, output_path)
        print(f"Interactive HTML saved: {output_path}")

    env.close()


if __name__ == "__main__":
    main()
