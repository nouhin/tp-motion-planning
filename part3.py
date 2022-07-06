import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from matplotlib import collections as mc

random.seed(21)

class Env:
    # A 2D environment with bounds [-1, 1] x [-1, 1]
    def __init__(self, walls=[]):
        self.walls = [([-1.0, -1.0], [1.0, -1.0]),
                      ([1.0, -1.0], [1.0, 1.0]),
                      ([1.0, 1.0], [-1.0, 1.0]),
                      ([-1.0, 1.0], [-1.0, -1.0])] + walls

    @staticmethod
    def robot_segments(pos, theta):
        return [([pos[0], pos[1]], [pos[0] + np.cos(theta) * 0.2, pos[1] + np.sin(theta) * 0.2]),
                ([pos[0], pos[1]], [pos[0] - np.sin(theta) * 0.35, pos[1] + np.cos(theta) * 0.35])]

    @staticmethod
    def intersect(a, b, c, d):
        # If line segments ab and cd have a true intersection, return the intersection point. Otherwise, return False
        # a, b, c and d are 2D points of the form [x, y]
        x1, x2, x3, x4 = a[0], b[0], c[0], d[0]
        y1, y2, y3, y4 = a[1], b[1], c[1], d[1]
        denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)
        if denom == 0:
            return False
        else:
            t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
            if t <= 0 or t >= 1:
                return False
            else:
                t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
                if t <= 0 or t >= 1:
                    return False
                else:
                    return [x3 + t * (x4 - x3), y3 + t * (y4 - y3)]

    @staticmethod
    def dist(a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    @staticmethod
    def random_action():
        return [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]

    @staticmethod
    def random_oriented_action(orientation):
        theta = random.uniform(orientation - 35.0 * np.pi/180.0, orientation + 35.0 * np.pi/180.0)
        return [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], theta

    def step_old(self, state, action, full=False):
        candidate = [state[0] + action[0], state[1] + action[1]]
        dist = np.infty
        for w in self.walls:  # Naive way to check for collisions
            pt = Env.intersect(state, candidate, w[0], w[1])
            if pt:
                candidate = pt
        if full:
            return candidate
        else:
            newstate = [state[0] + 0.99*(candidate[0] - state[0]), state[1] + 0.99*(candidate[1] - state[1])]
            if Env.dist(state, newstate) < 0.001:  # Reject steps that are too small
                return False
            else:
                return newstate

    def step(self, state, orientation, action, new_orientation, full=False):
        for i in range(21):
            candidate = [state[0] + i/20.0 * action[0], state[1] + i/20.0 * action[1]]
            cand_orientation = orientation + i/20.0 * (new_orientation - orientation)
            collision = False
            for z in Env.robot_segments(candidate, cand_orientation):  # Naive way to check for collisions
                for w in self.walls:
                    pt = Env.intersect(z[0], z[1], w[0], w[1])
                    if pt:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                assert(i>0)
                candidate = [state[0] + (i-1) / 20.0 * action[0], state[1] + (i-1) / 20.0 * action[1]]
                cand_orientation = orientation + (i-1) / 20.0 * (new_orientation - orientation)
                break
        if i == 0:
            return False, False
        return candidate, cand_orientation

    def plotwalls(self, fig, ax):
        lines = []
        rgbs = []
        for w in self.walls:
            lines.append(w)
            rgbs.append((0, 0, 0, 1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axis('equal')

    @staticmethod
    def plotrobot(fig, ax, pos, theta):
        lines = []
        rgbs = []
        for w in Env.robot_segments(pos, theta):
            lines.append(w)
            rgbs.append((0, 1, 1, 1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axis('equal')


class Tree:
    def __init__(self, init_state, orientation, parent=None, root=True):
        self.parent = parent
        self.state = init_state
        self.orientation = orientation
        self.local_inverse_density = 1
        self.successors = []
        self.root = root
        self.all_nodes = []

    def __all_edges(self):
        if not self.successors:
            return [], []
        else:
            lines = []
            rgbs = []
            for s in self.successors:
                # lines.append((self.state, s.state))
                # rgbs.append((1, 0, 0, 1))
                lnew = Env.robot_segments(s.state, s.orientation)
                lines += lnew
                rgbs += [(0,1,0,1)]*len(lnew)
                ladd, rgbadd = s.__all_edges()
                lines += ladd
                rgbs += rgbadd
            return lines, rgbs

    def plot(self, fig, ax):
        lines, rgbs = self.__all_edges()
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1))


def random_walls(env, n):
    for i in range(n):
        start = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        progress = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        end = env.step_old(start, progress, True)
        env.walls.append((start, end))


def random_expansion(t, env):
    s = random.choice(t.all_nodes + [t])
    orientation = s.orientation
    action, theta = Env.random_oriented_action(orientation)
    new_state, new_orientation = env.step(s.state, s.orientation, action, theta)
    if new_state:
        new_node = Tree(new_state, new_orientation, s, False)
        s.successors.append(new_node)
        t.all_nodes.append(new_node)


def rrt_expansion(t, env):
    sample = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    nearest_neighbor = t
    d = Env.dist(t.state, sample)
    for s in t.all_nodes:  # Naive way to get the nearest neighbor
        d_tmp = Env.dist(s.state, sample)
        if d_tmp < d:
            nearest_neighbor = s
            d = d_tmp
    orientation = nearest_neighbor.orientation
    action, theta = Env.random_oriented_action(orientation)
    new_state, new_orientation = env.step(nearest_neighbor.state, nearest_neighbor.orientation,
                                          action, theta)
    if new_state:
        new_node = Tree(new_state, new_orientation, nearest_neighbor, False)
        nearest_neighbor.successors.append(new_node)
        t.all_nodes.append(new_node)



env = Env()
random_walls(env, 10)
t = Tree([-0.5, -0.5], 0.0)
#
for i in range(200):
    # random_expansion(t, env)
    rrt_expansion(t, env)
#     # approx_rrt_expansion(t, env)


fig, ax = plt.subplots()
t.plot(fig, ax)
env.plotwalls(fig, ax)
plt.show()
