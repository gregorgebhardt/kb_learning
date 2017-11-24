# Based on code from http://www.redblobgames.com/pathfinding/
# Copyright 2014 Red Blob Games <redblobgames@gmail.com>

import heapq
import numpy as np
from numpy.linalg import linalg


class SquareGrid:
    def __init__(self, width, height, resolution=1):
        self.width = int(width * resolution)
        self.height = int(height * resolution)
        self.walls = []
        self.resolution = resolution

    def in_bounds(self, coords):
        (x, y) = coords
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, coords):
        return coords not in self.walls

    # altered by KD, added diagonal neighbors
    def neighbors(self, coords):
        (x, y) = coords
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1),
                   (x - 1, y - 1)]
        if (x + y) % 2 == 0:
            results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

    def cost(self, from_node, to_node):
        dist = linalg.norm(np.asarray(to_node) - np.asarray(from_node)) / self.resolution
        cost = dist
        return cost


class GridWithObstacles(SquareGrid):
    def __init__(self, width, height, resolution=1):
        super().__init__(width, height, resolution)
        self.obstacles = []
        self.potential_radius = 0.4
        self.potential_factor = 10

    def cost(self, from_node, to_node):
        q = self.potential_radius
        dist = linalg.norm(np.asarray(to_node) - np.asarray(from_node)) / self.resolution
        cost = dist
        for obstacle in self.obstacles:
            d = linalg.norm(np.asarray(to_node) / self.resolution - np.asarray(obstacle))
            if d == 0.0:
                cost += np.inf
            elif d <= q:
                cost += self.potential_factor * dist * (1 / d - 1 / q) ** 2
        return cost


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class AStar:
    def __init__(self, grid):
        self.grid = grid

    def enforceBounds(self, pos):

        p0 = pos[0]
        if (pos[0] < 0):
            p0 = 0
        elif (pos[0] >= self.grid.width):
            p0 = self.grid.width - 1

        p1 = pos[1]
        if (pos[1] < 0):
            p1 = 0
        elif (pos[1] >= self.grid.width):
            p1 = self.grid.width - 1

        return tuple([p0, p1])

    def get_path(self, start, goal):
        start = np.asarray(start) * self.grid.resolution
        start = tuple(map(int, tuple(start)))
        goal = np.asarray(goal) * self.grid.resolution
        goal = tuple(map(int, tuple(goal)))
        start = self.enforceBounds(start)
        goal = self.enforceBounds(goal)

        came_from, cost_so_far = self._a_star_search(self.grid, start, goal)
        return np.asarray(self._reconstruct_path(came_from, start, goal)) / self.grid.resolution

    def _reconstruct_path(self, came_from, start, goal):
        current = goal
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.append(start)  # optional
        path.reverse()  # optional
        return path

    # altered by KD, using L2 distance
    def _heuristic(self, a, b):
        (x1, y1) = a
        (x2, y2) = b
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / self.grid.resolution

    def _a_star_search(self, graph, start, goal):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = dict()
        cost_so_far = dict()
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()
            if current == goal:
                break
            for next_node in graph.neighbors(current):
                new_cost = cost_so_far[current] + graph.cost(current, next_node)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self._heuristic(goal, next_node)
                    frontier.put(next_node, priority)
                    came_from[next_node] = current

        return came_from, cost_so_far

# TODO check this file
if __name__ == '__main__':
    grid = SquareGrid(10, 10, 2)
    astar = AStar(grid)
    #  print(astar.get_path((5, 2), (5, 8)))

    grid = GridWithObstacles(10, 10, 2)
    astar = AStar(grid)
    #  print(astar.get_path((5, 2), (5, 8)))
    grid.obstacles = [[5, 5], [7, 7]]
    grid.potential_radius = 5
    grid.potential_factor = 100
    astar = AStar(grid)
    print(astar.get_path((5, 2), (5, 8)))
