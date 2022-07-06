import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

random.seed(11)


class Env:
    # A 2D environment with bounds [-1, 1] x [-1, 1]
    def __init__(self, walls=[]):
        self.walls = [([-1.0, -1.0], [1.0, -1.0]), ([1.0, -1.0], [1.0, 1.0]), ([1.0, 1.0], [-1.0, 1.0]), ([-1.0, 1.0], [-1.0, -1.0])] + walls

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

    def step(self, state, action, full=False):
        candidate = [state[0] + action[0], state[1] + action[1]]
        dist = np.infty
        for w in self.walls:  # Naive way to check for collisions
            pt = Env.intersect(state, candidate, w[0], w[1])
            if pt:
                candidate = pt
        if full:
            return candidate
        else:
            newstate = [state[0] + 0.99 * (candidate[0] - state[0]), state[1] + 0.99 * (candidate[1] - state[1])]
            if Env.dist(state, newstate) < 0.001:  # Reject steps that are too small
                return False
            else:
                return newstate

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


class Tree:
    def __init__(self, init_state, parent=None, root=True):
        self.parent = parent
        if self.parent is not None:
            # Inherite color of previous iteration
            self.color = parent.color
        else:
            # generate a random color in RGB format
            color = [float(x) / 20 for x in np.random.choice(range(20), size=3)]
            color.append(1)
            # creates the color attribute for Tree class
            self.color = color
        self.state = init_state
        self.successors = []
        self.root = root
        self.all_nodes = []

    def __all_edges(self):
        if not self.successors:
            return [], []
        else:
            lines = []
            rgbs = []
            color = [float(x) / 256 for x in np.random.choice(range(256), size=3)]
            color.append(1)
            for s in self.successors:
                lines.append((self.state, s.state))
                rgbs.append(self.color)
                ladd, rgbadd = s.__all_edges()
                lines += ladd
                rgbs += rgbadd
            return lines, rgbs

    def plot(self, fig, ax, label):
        lines, rgbs = self.__all_edges()
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1, label=label))


def random_walls(env, n):
    for i in range(n):
        start = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        progress = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        end = env.step(start, progress, True)
        env.walls.append((start, end))


def random_expansion(t, env):
    s = random.choice(t.all_nodes + [t])
    new_state = env.step(s.state, Env.random_action())
    if new_state:
        new_node = Tree(new_state, s, False)
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
    action = [np.clip(sample[0] - nearest_neighbor.state[0], -0.1, 0.1), np.clip(sample[1] - nearest_neighbor.state[1], -0.1, 0.1)]
    new_state = env.step(nearest_neighbor.state, action)
    if new_state:
        new_node = Tree(new_state, nearest_neighbor, False)
        nearest_neighbor.successors.append(new_node)
        t.all_nodes.append(new_node)


def rrt_expansion_var1(t, env):
    sample = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    nearest_neighbor = t
    d = Env.dist(t.state, sample)
    for s in t.all_nodes:  # Naive way to get the nearest neighbor
        d_tmp = Env.dist(s.state, sample)
        if d_tmp < d:
            nearest_neighbor = s
            d = d_tmp
    new_state = env.step(nearest_neighbor.state, Env.random_action())
    if new_state:
        new_node = Tree(new_state, nearest_neighbor, False)
        nearest_neighbor.successors.append(new_node)
        t.all_nodes.append(new_node)


def rrt_expansion_var2(t, env):
    sample = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    nearest_neighbor = t
    d = Env.dist(t.state, sample)
    for s in random.sample(t.all_nodes + [t], len(t.all_nodes) // 2 + 1):
        d_tmp = Env.dist(s.state, sample)
        if d_tmp < d:
            nearest_neighbor = s
            d = d_tmp
    action = [np.clip(sample[0] - nearest_neighbor.state[0], -0.1, 0.1), np.clip(sample[1] - nearest_neighbor.state[1], -0.1, 0.1)]
    new_state = env.step(nearest_neighbor.state, action)
    if new_state:
        new_node = Tree(new_state, nearest_neighbor, False)
        nearest_neighbor.successors.append(new_node)
        t.all_nodes.append(new_node)


# this  function calculates the max distance from the origin for each tree
def get_max_distance(t, env):
    d = 0
    for s in t.all_nodes:
        d_tmp = Env.dist(t.state, s.state)
        if d_tmp > d:
            d = d_tmp
    return d


def main():
    # Initialisation
    env = Env()
    random_walls(env, 20)
    t_rand_expansion = Tree([-0.5, -0.5])
    t_rrt_expansion = Tree([-0.5, -0.5])
    fig, ax = plt.subplots()
    env.plotwalls(fig, ax)

    # Evaluation bases on constant number of itterations
    n = int(input("Enter num of iter :"))

    for i in range(n + 1):
        rrt_expansion(t_rrt_expansion, env)
        random_expansion(t_rand_expansion, env)
        t_rand_expansion.plot(fig, ax, "Random expansion")
        t_rrt_expansion.plot(fig, ax, "RRT Expansion")
        if i == 0:
            plt.legend()
        plt.pause(0.01)

    plt.show()

    d_random_expansion = get_max_distance(t_rand_expansion, env)
    d_rrt_expansion = get_max_distance(t_rrt_expansion, env)
    print("Performance of random expansion : max distance = ", d_random_expansion)
    print("Performance of of rrt expansion : max distance = ", d_rrt_expansion)

    del t_rand_expansion, t_rrt_expansion, fig, ax
    t_rand_expansion = Tree([-0.5, -0.5])
    t_rrt_expansion = Tree([-0.5, -0.5])
    fig, ax = plt.subplots()
    env.plotwalls(fig, ax)

    # Evaluation bases on constant time execution

    print("Evaluation bases on constant time execution")
    Time = float(input("Enter the time in seconds : "))
    print("Distance eval for", Time, "seconds")

    end_time = datetime.now() + timedelta(seconds=Time)
    while datetime.now() < end_time:
        rrt_expansion(t_rrt_expansion, env)

    end_time = datetime.now() + timedelta(seconds=Time)
    while datetime.now() < end_time:
        random_expansion(t_rand_expansion, env)

    t_rand_expansion.plot(fig, ax, "Random expansion")
    t_rrt_expansion.plot(fig, ax, "RRT Expansion")
    plt.legend()
    plt.show()

    d_random_expansion = get_max_distance(t_rand_expansion, env)
    d_rrt_expansion = get_max_distance(t_rrt_expansion, env)
    print("Performance of random expansion : max distance = ", d_random_expansion)
    print("Performance of of rrt expansion : max distance = ", d_rrt_expansion)

    # Evaluation based on number of walls for constant iterations

    print("Evaluation based on number of walls for constant iterations")
    numbers_of_walls = list(map(int, input("Enter number of walls for each test in one line :").split()))
    for n in numbers_of_walls:
        del t_rand_expansion, t_rrt_expansion, fig, ax, env
        env = Env()
        random_walls(env, n)
        t_rand_expansion = Tree([-0.5, -0.5])
        t_rrt_expansion = Tree([-0.5, -0.5])
        fig, ax = plt.subplots()
        env.plotwalls(fig, ax)
        for i in range(200):
            rrt_expansion(t_rrt_expansion, env)
            random_expansion(t_rand_expansion, env)
        t_rand_expansion.plot(fig, ax, "Random expansion")
        t_rrt_expansion.plot(fig, ax, "RRT Expansion")
        title = "Number of walls : " + str(n)
        ax.set_title(title)
        plt.legend()
        plt.show()
        d_random_expansion = get_max_distance(t_rand_expansion, env)
        d_rrt_expansion = get_max_distance(t_rrt_expansion, env)
        print("Performance of random expansion with", n, "walls : max distance = ", d_random_expansion)
        print("Performance of of rrt expansion with", n, "walls: max distance = ", d_rrt_expansion)

    # Evaluation of variantes of RRT implementaion :

    del t_rand_expansion, t_rrt_expansion, fig, ax, env
    env = Env()
    random_walls(env, 30)
    t_rrt_expansion_var1 = Tree([-0.5, -0.5])
    t_rrt_expansion_var2 = Tree([-0.5, -0.5])
    fig, ax = plt.subplots()
    env.plotwalls(fig, ax)

    print("Evaluation based on constant time execution of RRT variantes")
    Time = float(input("Enter the time in seconds : "))
    print("Distance eval for", Time, "seconds")

    end_time = datetime.now() + timedelta(seconds=Time)
    while datetime.now() < end_time:
        rrt_expansion_var1(t_rrt_expansion_var1, env)

    end_time = datetime.now() + timedelta(seconds=Time)
    while datetime.now() < end_time:
        rrt_expansion_var2(t_rrt_expansion_var2, env)

    t_rrt_expansion_var1.plot(fig, ax, "Variante 1")
    t_rrt_expansion_var2.plot(fig, ax, "Variante 2")
    plt.legend()
    plt.show()

    d_rrt_expansion_var1 = get_max_distance(t_rrt_expansion_var1, env)
    d_rrt_expansion_var2 = get_max_distance(t_rrt_expansion_var2, env)
    print("Performance of variante 1 : max distance = ", d_rrt_expansion_var1)
    print("Performance of variante 2 : max distance = ", d_rrt_expansion_var2)


if __name__ == "__main__":
    main()
