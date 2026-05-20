import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class GlobalUncoveredHeatmap:
    """
    维护全局“未覆盖热力图”，并根据局部信息熵提取全局引导点。

    约定输入 global_map 与 env.py 一致:
      -1: 障碍物
       0: 未覆盖
       1: 已覆盖
    """

    def __init__(self, env_size: float, grid_res: float, entropy_window: int = 9):
        if grid_res <= 0:
            raise ValueError("grid_res must be positive.")
        if entropy_window < 3 or entropy_window % 2 == 0:
            raise ValueError("entropy_window must be an odd integer >= 3.")

        self.env_size = float(env_size)
        self.grid_res = float(grid_res)
        self.entropy_window = int(entropy_window)

        self.grid_size = int(self.env_size / self.grid_res)
        self.uncovered_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.last_global_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

    def update_from_global_map(self, global_map: np.ndarray) -> np.ndarray:
        """
        根据实时 global_map 更新未覆盖热力图，并返回最新热力图。

        热度定义:
          - 先将未覆盖视作 1，其他(已覆盖/障碍物)视作 0
          - 在局部窗口内估计未覆盖概率 p
          - 计算 Shannon 熵 H(p) = -p log2 p - (1-p) log2(1-p)

        含义:
          - 熵高: 局部已覆盖/未覆盖混合明显，具有探索价值
          - 熵低: 局部全覆盖或全未覆盖，边际收益较低
        """
        if global_map.shape != (self.grid_size, self.grid_size):
            raise ValueError(
                f"global_map shape mismatch, expected {(self.grid_size, self.grid_size)}, got {global_map.shape}"
            )

        self.last_global_map = global_map.astype(np.int8, copy=True)

        unknown_mask = (self.last_global_map == 0).astype(np.float32)
        valid_mask = (self.last_global_map >= 0).astype(np.float32)

        half = self.entropy_window // 2
        padded_unknown = np.pad(unknown_mask, pad_width=half, mode="constant", constant_values=0.0)
        padded_valid = np.pad(valid_mask, pad_width=half, mode="constant", constant_values=0.0)

        heatmap = np.zeros_like(unknown_mask, dtype=np.float32)
        # 使用更稳健的 epsilon，避免 float32 下出现 log2(0) 数值告警
        eps = 1e-6

        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                if self.last_global_map[gx, gy] == -1:
                    continue

                ux0, ux1 = gx, gx + self.entropy_window
                uy0, uy1 = gy, gy + self.entropy_window

                local_unknown = padded_unknown[ux0:ux1, uy0:uy1]
                local_valid = padded_valid[ux0:ux1, uy0:uy1]

                valid_count = np.sum(local_valid)
                if valid_count < 1:
                    continue

                p = float(np.sum(local_unknown, dtype=np.float64) / (float(valid_count) + eps))
                p = float(np.clip(p, eps, 1.0 - eps))
                entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
                heatmap[gx, gy] = float(entropy)

        self.uncovered_heatmap = heatmap
        return self.uncovered_heatmap

    def get_top_guidance_points(
        self,
        k: int = 3,
        min_grid_dist: int = 5,
        only_uncovered: bool = True,
    ):
        """
        返回熵最大的 k 个全局引导点。

        参数:
          k: 返回点数
          min_grid_dist: 网格坐标最小间距，防止多个引导点聚集
          only_uncovered: True 时仅从未覆盖栅格中选点

        返回:
          List[dict], 每个元素包含:
            - grid: (gx, gy)
            - world: (x, y)  (米, 栅格中心点)
            - entropy: 熵值
        """
        if k <= 0:
            return []

        candidate_mask = np.ones_like(self.uncovered_heatmap, dtype=bool)
        candidate_mask &= (self.last_global_map >= 0)
        if only_uncovered:
            candidate_mask &= (self.last_global_map == 0)

        indices = np.argwhere(candidate_mask)
        if indices.size == 0:
            return []

        scores = self.uncovered_heatmap[candidate_mask]
        order = np.argsort(-scores)

        selected = []
        for idx in order:
            gx, gy = indices[idx]
            score = float(self.uncovered_heatmap[gx, gy])
            if score <= 0:
                continue

            too_close = False
            for item in selected:
                sx, sy = item["grid"]
                if (gx - sx) ** 2 + (gy - sy) ** 2 < (min_grid_dist ** 2):
                    too_close = True
                    break
            if too_close:
                continue

            wx = (gx + 0.5) * self.grid_res
            wy = (gy + 0.5) * self.grid_res

            selected.append(
                {
                    "grid": (int(gx), int(gy)),
                    "world": (float(wx), float(wy)),
                    "entropy": score,
                }
            )
            if len(selected) >= k:
                break

        return selected

    def get_frontier_guidance_points(
        self,
        k: int = 3,
        min_grid_dist: int = 8,
        min_cluster_size: int = 6,
        mode: str = "frontier",
        alpha: float = 0.4,
    ):
        """
        基于 frontier(已覆盖/未覆盖边界) 的引导点提取。
        相比仅按熵最大值挑点，这个方法通常更符合“下一步该扫哪里”的直觉。

        mode:
          - frontier: 纯边界推进
          - hybrid: 边界分数 + 深处探索分数
          - interior: 更偏向未覆盖深处
        alpha:
          深处探索分数权重（仅 hybrid/interior 生效）
        """
        if k <= 0:
            return []
        if mode not in {"frontier", "hybrid", "interior"}:
            raise ValueError("mode must be one of {'frontier', 'hybrid', 'interior'}")

        frontier_mask = self._build_frontier_mask()
        clusters = self._cluster_frontiers(frontier_mask, min_cluster_size=min_cluster_size)
        if not clusters:
            # 没有 frontier 时，回退到熵最大策略
            return self.get_top_guidance_points(k=k, min_grid_dist=min_grid_dist, only_uncovered=True)

        depth_map = self._compute_unknown_depth_map()
        depth_norm = float(np.max(depth_map)) if np.max(depth_map) > 0 else 1.0

        # 每个 cluster 内选一个代表点：优先选择“熵高 + cluster 大”的点
        cluster_candidates = []
        for cluster in clusters:
            cells = np.array(cluster, dtype=np.int32)
            entropies = self.uncovered_heatmap[cells[:, 0], cells[:, 1]]

            if mode == "frontier":
                scores = entropies + 0.01 * float(len(cluster))
            else:
                local_depth = depth_map[cells[:, 0], cells[:, 1]] / depth_norm
                if mode == "hybrid":
                    scores = entropies + alpha * local_depth + 0.01 * float(len(cluster))
                else:  # interior
                    scores = alpha * local_depth + 0.3 * entropies + 0.01 * float(len(cluster))

            local_best = int(np.argmax(scores))
            gx, gy = int(cells[local_best, 0]), int(cells[local_best, 1])

            cluster_score = float(scores[local_best])
            cluster_candidates.append((cluster_score, gx, gy, len(cluster)))

        cluster_candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for score, gx, gy, csize in cluster_candidates:
            too_close = False
            for item in selected:
                sx, sy = item["grid"]
                if (gx - sx) ** 2 + (gy - sy) ** 2 < (min_grid_dist ** 2):
                    too_close = True
                    break
            if too_close:
                continue

            wx = (gx + 0.5) * self.grid_res
            wy = (gy + 0.5) * self.grid_res
            selected.append(
                {
                    "grid": (gx, gy),
                    "world": (float(wx), float(wy)),
                    "entropy": float(self.uncovered_heatmap[gx, gy]),
                    "score": float(score),
                    "cluster_size": int(csize),
                }
            )
            if len(selected) >= k:
                break

        return selected

    def get_uncovered_region_guidance_points(
        self,
        k: int = 3,
        min_grid_dist: int = 10,
        depth_weight: float = 1.0,
        unknown_ratio_weight: float = 1.5,
        local_window: int = 11,
        edge_penalty_weight: float = 0.6,
    ):
        """
        纯未覆盖区域引导（不依赖与已覆盖边界连接）。

        目标：
          - 引导点只从未覆盖区域中选
          - 局部“覆盖越少(未覆盖占比越高)”的区域分数越高
          - 越处于未覆盖深处(离已覆盖区更远)的区域分数越高
          - 远离地图边界（减少贴边引导点）
        """
        if k <= 0:
            return []
        if local_window < 3 or local_window % 2 == 0:
            raise ValueError("local_window must be an odd integer >= 3.")

        unknown = (self.last_global_map == 0)
        if np.sum(unknown) == 0:
            return []

        depth_map = self._compute_unknown_depth_map()
        max_depth = float(np.max(depth_map)) if np.max(depth_map) > 0 else 1.0
        depth_norm = depth_map / max_depth

        # 计算每个未覆盖格局部窗口的未覆盖占比（越高越好）
        half = local_window // 2
        unknown_f = unknown.astype(np.float32)
        valid_f = (self.last_global_map >= 0).astype(np.float32)
        padded_unknown = np.pad(unknown_f, pad_width=half, mode="constant", constant_values=0.0)
        padded_valid = np.pad(valid_f, pad_width=half, mode="constant", constant_values=0.0)
        unknown_ratio_map = np.zeros_like(depth_norm, dtype=np.float32)

        eps = 1e-6
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                if not unknown[gx, gy]:
                    continue
                x0, x1 = gx, gx + local_window
                y0, y1 = gy, gy + local_window
                local_unknown = padded_unknown[x0:x1, y0:y1]
                local_valid = padded_valid[x0:x1, y0:y1]
                valid_count = float(np.sum(local_valid))
                if valid_count < 1.0:
                    continue
                unknown_ratio_map[gx, gy] = float(np.sum(local_unknown) / (valid_count + eps))

        edge_dist_map = self._compute_edge_distance_map()
        max_edge_dist = float(np.max(edge_dist_map)) if np.max(edge_dist_map) > 0 else 1.0
        edge_dist_norm = edge_dist_map / max_edge_dist

        edge_penalty = (1.0 - edge_dist_norm) * edge_penalty_weight
        score_map = depth_weight * depth_norm + unknown_ratio_weight * unknown_ratio_map - edge_penalty
        score_map[~unknown] = -1.0

        # 按分数从高到低选点，并做最小距离约束
        candidate_indices = np.argwhere(unknown)
        candidate_scores = score_map[unknown]
        order = np.argsort(-candidate_scores)

        selected = []
        for idx in order:
            gx, gy = candidate_indices[idx]
            score = float(score_map[gx, gy])
            if score <= 0:
                continue

            too_close = False
            for item in selected:
                sx, sy = item["grid"]
                if (gx - sx) ** 2 + (gy - sy) ** 2 < (min_grid_dist ** 2):
                    too_close = True
                    break
            if too_close:
                continue

            wx = (gx + 0.5) * self.grid_res
            wy = (gy + 0.5) * self.grid_res
            selected.append(
                {
                    "grid": (int(gx), int(gy)),
                    "world": (float(wx), float(wy)),
                    "score": score,
                    "unknown_ratio": float(unknown_ratio_map[gx, gy]),
                    "depth": float(depth_map[gx, gy]),
                    "edge_dist": float(edge_dist_map[gx, gy]),
                }
            )
            if len(selected) >= k:
                break

        return selected

    def _build_frontier_mask(self) -> np.ndarray:
        """
        frontier 定义：未覆盖格(0)且 8 邻域内存在已覆盖格(1)。
        """
        unknown = (self.last_global_map == 0)
        covered = (self.last_global_map == 1)
        frontier = np.zeros_like(unknown, dtype=bool)

        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for dx, dy in shifts:
            shifted = np.zeros_like(covered, dtype=bool)

            if dx >= 0:
                xs_src = slice(0, self.grid_size - dx)
                xs_dst = slice(dx, self.grid_size)
            else:
                xs_src = slice(-dx, self.grid_size)
                xs_dst = slice(0, self.grid_size + dx)

            if dy >= 0:
                ys_src = slice(0, self.grid_size - dy)
                ys_dst = slice(dy, self.grid_size)
            else:
                ys_src = slice(-dy, self.grid_size)
                ys_dst = slice(0, self.grid_size + dy)

            shifted[xs_dst, ys_dst] = covered[xs_src, ys_src]
            frontier |= (unknown & shifted)

        return frontier

    @staticmethod
    def _cluster_frontiers(frontier_mask: np.ndarray, min_cluster_size: int = 6):
        """
        8 邻域 BFS 聚类 frontier，返回 cluster 列表，每个 cluster 是 [(gx, gy), ...]
        """
        h, w = frontier_mask.shape
        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters = []

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        for x in range(h):
            for y in range(w):
                if (not frontier_mask[x, y]) or visited[x, y]:
                    continue

                queue = [(x, y)]
                visited[x, y] = True
                cluster = []

                while queue:
                    cx, cy = queue.pop()
                    cluster.append((cx, cy))

                    for dx, dy in neighbors:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if frontier_mask[nx, ny] and (not visited[nx, ny]):
                                visited[nx, ny] = True
                                queue.append((nx, ny))

                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

        return clusters

    def _compute_unknown_depth_map(self) -> np.ndarray:
        """
        近似“未覆盖深度图”:
        每个未覆盖格到最近已覆盖格的曼哈顿距离（两次动态规划近似）。
        距离越大，越偏未覆盖深处。
        """
        unknown = (self.last_global_map == 0)
        covered = (self.last_global_map == 1)

        h, w = self.last_global_map.shape
        inf = h + w + 10
        dist = np.full((h, w), inf, dtype=np.float32)
        dist[covered] = 0.0
        dist[self.last_global_map == -1] = inf

        for x in range(h):
            for y in range(w):
                if dist[x, y] == 0.0:
                    continue
                best = dist[x, y]
                if x > 0:
                    best = min(best, dist[x - 1, y] + 1.0)
                if y > 0:
                    best = min(best, dist[x, y - 1] + 1.0)
                dist[x, y] = best

        for x in range(h - 1, -1, -1):
            for y in range(w - 1, -1, -1):
                if dist[x, y] == 0.0:
                    continue
                best = dist[x, y]
                if x + 1 < h:
                    best = min(best, dist[x + 1, y] + 1.0)
                if y + 1 < w:
                    best = min(best, dist[x, y + 1] + 1.0)
                dist[x, y] = best

        dist[~unknown] = 0.0
        dist[np.isinf(dist)] = 0.0
        return dist

    def _compute_edge_distance_map(self) -> np.ndarray:
        """
        每个格子到地图边缘的最短网格距离。
        距离越小，越靠近边界。
        """
        h, w = self.last_global_map.shape
        edge_dist = np.zeros((h, w), dtype=np.float32)
        for x in range(h):
            for y in range(w):
                d = min(x, y, h - 1 - x, w - 1 - y)
                edge_dist[x, y] = float(d)
        return edge_dist


if __name__ == "__main__":
    # 测试调用：100x100 栅格区域，随机刷新覆盖分布并生成动态图
    env_size = 100.0
    grid_res = 1.0
    gsize = int(env_size / grid_res)
    rng = np.random.default_rng(42)

    # 动图参数
    num_frames = 30
    block_size = 10
    covered_ratio_low = 0.75
    covered_ratio_high = 0.93

    guide = GlobalUncoveredHeatmap(env_size=env_size, grid_res=grid_res, entropy_window=9)

    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(
        np.zeros((gsize, gsize), dtype=np.float32).T,
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        animated=True,
    )
    scat = ax.scatter([], [], c="red", marker="^", s=120, label="Guidance Points", animated=True)
    ax.legend(loc="upper right")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_xlim(0, gsize - 1)
    ax.set_ylim(0, gsize - 1)
    ax.grid(False)

    def build_random_global_map():
        gm = np.zeros((gsize, gsize), dtype=np.int8)
        covered_ratio = rng.uniform(covered_ratio_low, covered_ratio_high)

        for bx in range(0, gsize, block_size):
            for by in range(0, gsize, block_size):
                if rng.random() < covered_ratio:
                    x1 = min(bx + block_size, gsize)
                    y1 = min(by + block_size, gsize)
                    gm[bx:x1, by:y1] = 1

        # 固定少量障碍物，仅用于可视化对比
        gm[20:21, 20:21] = -1
        gm[65:66, 55:56] = -1
        return gm

    def frame_update(frame_idx):
        global_map = build_random_global_map()
        guide.update_from_global_map(global_map)
        points = guide.get_uncovered_region_guidance_points(
            k=5,
            min_grid_dist=10,
            depth_weight=1.2,
            unknown_ratio_weight=2.0,
            local_window=11,
            edge_penalty_weight=0.8,
        )

        # 未扫描(0)->灰色, 已扫描(1)->白色, 障碍(-1)->黑色
        show_map = np.full_like(global_map, 0.5, dtype=np.float32)
        show_map[global_map == 1] = 1.0
        show_map[global_map == -1] = 0.0
        image.set_data(show_map.T)

        if points:
            gx = [item["grid"][0] for item in points]
            gy = [item["grid"][1] for item in points]
            scat.set_offsets(np.column_stack([gx, gy]))
        else:
            scat.set_offsets(np.empty((0, 2)))

        ax.set_title(f"Global Guidance Dynamic Test  frame={frame_idx + 1}/{num_frames}")
        return image, scat

    # 刷新节奏调慢，便于肉眼观察每一帧引导点变化
    frame_interval_ms = 1200
    gif_fps = 1
    anim = FuncAnimation(fig, frame_update, frames=num_frames, interval=frame_interval_ms, blit=False, repeat=True)
    plt.tight_layout()

    gif_path = "global_guidance_test.gif"
    png_path = "global_guidance_test.png"
    anim.save(gif_path, writer=PillowWriter(fps=gif_fps))
    frame_update(0)
    plt.savefig(png_path, dpi=180)
    print(f"saved animation: {gif_path}")
    print(f"saved preview: {png_path}")
    plt.show()
