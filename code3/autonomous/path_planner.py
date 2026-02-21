"""
path_planner.py - A* 경로 계획
================================
2D BEV 장애물 격자 위에서 A* 알고리즘으로 목표 지점까지 경로 탐색
경로 → 조향각·속도 명령 변환 (STM32 CAN 전송용)
"""

import heapq
import math
import numpy as np
import cv2


class AStarPlanner:
    """
    BEV 격자 A* 경로 계획기

    Parameters
    ----------
    grid_size   : BEV 격자 크기 (기본 200×200)
    resolution  : 셀당 실제 거리 [m] (기본 0.5m)
    lookahead   : 조향 계산용 lookahead 거리 [cells]
    """

    def __init__(self,
                 grid_size: int = 200,
                 resolution: float = 0.5,
                 lookahead: int = 15):
        self.grid_size  = grid_size
        self.resolution = resolution
        self.lookahead  = lookahead
        self.ego_r      = grid_size // 2
        self.ego_c      = grid_size // 2

    # ── A* 핵심 ──────────────────────────────
    def _heuristic(self, a, b) -> float:
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _neighbors(self, r, c, grid):
        """8-방향 이동 (대각선 포함)"""
        H, W = grid.shape
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                    cost = 1.414 if (dr and dc) else 1.0
                    yield nr, nc, cost

    def plan(self,
             obstacle_grid: np.ndarray,
             goal_ahead_m: float = 8.0) -> list:
        """
        Parameters
        ----------
        obstacle_grid : (H, W) uint8  0=자유 / 255=장애물
        goal_ahead_m  : 전방 목표 거리 [m]

        Returns
        -------
        path : [(row, col), ...] 시작→목표, 경로 없으면 []
        """
        grid = (obstacle_grid > 0).astype(np.uint8)
        start = (self.ego_r, self.ego_c)

        # 전방 목표 (자유 공간 내에서 찾기)
        goal_cells = int(goal_ahead_m / self.resolution)
        goal_r = max(0, self.ego_r - goal_cells)
        goal_c = self.ego_c

        # 목표 주변 자유 셀 탐색
        goal = self._find_free_near(grid, goal_r, goal_c)
        if goal is None:
            return []

        # A* 탐색
        open_heap = [(0.0, start)]
        came_from = {start: None}
        g_score   = {start: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                return self._reconstruct(came_from, goal)

            for nr, nc, move_cost in self._neighbors(*cur, grid):
                nxt = (nr, nc)
                g_new = g_score[cur] + move_cost
                if g_new < g_score.get(nxt, float('inf')):
                    g_score[nxt]   = g_new
                    f_score        = g_new + self._heuristic(nxt, goal)
                    came_from[nxt] = cur
                    heapq.heappush(open_heap, (f_score, nxt))

        return []   # 경로 없음

    def _find_free_near(self, grid, r, c, radius=10):
        H, W = grid.shape
        for dist in range(radius + 1):
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                        return (nr, nc)
        return None

    def _reconstruct(self, came_from, goal):
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    # ── 경로 → 조향·속도 변환 ────────────────
    def path_to_command(self, path: list) -> tuple:
        """
        경로 → (steering, speed)
          steering : -1.0(좌) ~ +1.0(우)
          speed    :  0.0     ~  1.0

        Returns (0.0, 0.0) if path is empty
        """
        if len(path) < 2:
            return 0.0, 0.0

        # lookahead 지점 선택
        look = min(self.lookahead, len(path) - 1)
        tr, tc = path[look]
        sr, sc = path[0]

        # 열 차이 = 횡방향 오프셋 (오른쪽 양수)
        lateral = tc - sc
        steering = float(np.clip(lateral / (self.lookahead * 0.8), -1.0, 1.0))

        # 속도: 커브일수록 감속
        speed = float(np.clip(0.6 * (1.0 - 0.5 * abs(steering)), 0.15, 0.6))

        return steering, speed

    # ── 경로 시각화 ───────────────────────────
    def draw_path(self, bev_img: np.ndarray, path: list,
                  scale: float = 2.0) -> np.ndarray:
        """BEV 이미지에 경로 오버레이"""
        img = bev_img.copy()
        s   = int(scale)
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            cv2.line(img, (c1*s, r1*s), (c2*s, r2*s), (0, 255, 255), 1)
        # 목표점
        if path:
            gr, gc = path[-1]
            cv2.circle(img, (gc*s, gr*s), 5, (0, 0, 255), -1)
        return img
