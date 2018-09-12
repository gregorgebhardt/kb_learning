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
        return tuple(coords) not in self.walls

    # altered by KD, added diagonal neighbors
    def neighbors(self, coords):
        neighbors_8 = np.dstack(np.meshgrid([-1, 0, 1], [-1, 0, 1])).reshape((-1, 2))
        results = neighbors_8 + coords

        # (x, y) = coords
        # results = [(x + 1, y),
        #            (x, y - 1),
        #            (x - 1, y),
        #            (x, y + 1),
        #            (x + 1, y + 1),
        #            (x + 1, y - 1),
        #            (x - 1, y + 1),
        #            (x - 1, y - 1)]
        # if (x + y) % 2 == 0:
        #     results.reverse()  # aesthetics

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
        self.potential_radius = 4.
        self.potential_factor = 20

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
    def __init__(self, grid, offset):
        self.grid = grid
        self.offset = offset

    def enforce_bounds(self, pos):
        pos = np.minimum(pos, [self.grid.width-1, self.grid.height-1])
        pos = np.maximum(pos, [0, 0])
        return pos

    def world_to_grid(self, world_point):
        return np.int32((world_point + self.offset) * self.grid.resolution)

    def grid_to_world(self, grid_points):
        return (np.float64(grid_points) / self.grid.resolution) - self.offset

    def get_path(self, start, goal):
        obj_rel_start = tuple(self.enforce_bounds(self.world_to_grid(start)))
        obj_rel_goal = tuple(self.enforce_bounds(self.world_to_grid(goal)))

        exploration_graph, _ = self._a_star_search(self.grid, obj_rel_start, obj_rel_goal)
        path = self._extract_path(exploration_graph, obj_rel_start, obj_rel_goal)

        world_path = np.concatenate([self.grid_to_world(path), [goal]])

        return world_path

    def _extract_path(self, exploration_graph, start, goal):
        path = [goal]
        while path[-1] != start:
            path.append(exploration_graph[path[-1]])
        path.reverse()  # optional
        return np.array(path)

    # altered by KD, using L2 distance
    def _heuristic(self, a, b):
        (x1, y1) = a
        (x2, y2) = b
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / self.grid.resolution

    def _a_star_search(self, world_graph, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        frontier = PriorityQueue()
        frontier.put(start, 0)
        exploration_graph = dict()
        exploration_costs = dict()
        exploration_graph[start] = None
        exploration_costs[start] = 0

        while not frontier.empty():
            current = frontier.get()
            if current == goal:
                break
            for next_node in world_graph.neighbors(current):
                next_node = tuple(next_node)
                new_cost = exploration_costs[current] + world_graph.cost(current, next_node)
                if next_node not in exploration_costs or new_cost < exploration_costs[next_node]:
                    exploration_costs[next_node] = new_cost
                    priority = new_cost + self._heuristic(goal, next_node)
                    frontier.put(next_node, priority)
                    exploration_graph[next_node] = current

        return exploration_graph, exploration_costs

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
