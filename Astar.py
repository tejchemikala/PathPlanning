import matplotlib.pyplot as plt
import numpy as np

def heuristic_distance(start_node, goal_node):
    return (goal_node[0] - start_node[0]) ** 2 + (goal_node[1] - start_node[1]) ** 2

directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

def plot_grid_with_path(grid, path):
    cmap = plt.get_cmap("tab20c", int(np.max(grid)) + 1)
    plt.figure(figsize=(len(grid[0]) / 2, len(grid) / 2))
    plt.imshow(grid, cmap=cmap, interpolation='none', origin='upper')

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                plt.plot(j, i, 's', color='k')
            elif grid[i][j] == 2:
                plt.plot(j, i, 'o', color='b')

    for node in path:
        plt.plot(node[1], node[0], 'o', color='r')

    plt.plot()
    plt.grid(True)
    plt.show()

def AStarImplementation(start_node, goal_node,obstacles):
    open_nodes = {}
    open_nodes[start_node] = (heuristic_distance(start_node, goal_node), 0)
    closed_nodes = {}
    children_parent_relation = {}
    
    while len(open_nodes):
        current_node_pair = min(open_nodes.items(), key=lambda x: x[1][0] + x[1][1])
        current_node, current_cost = current_node_pair[0], current_node_pair[1][1]
        
        if current_node == goal_node:
            path = reconstruct_path(children_parent_relation, current_node)
            return path
        
        del open_nodes[current_node]
        closed_nodes[current_node] = current_node_pair[1]
        
        for direction in directions:
            successor_node = (current_node[0] + direction[0], current_node[1] + direction[1])
            if(successor_node in obstacles):
                continue
            successor_current_cost = current_cost + heuristic_distance(current_node, successor_node)
            
            if successor_node in closed_nodes.keys() and closed_nodes[successor_node][1] <= successor_current_cost:
                continue
            
            if successor_node not in open_nodes.keys() or open_nodes[successor_node][1] > successor_current_cost:
                open_nodes[successor_node] = (successor_current_cost + heuristic_distance(successor_node, goal_node), successor_current_cost)
                children_parent_relation[successor_node] = current_node
        
    return None

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Test
start_node = (0, 0)
goal_node = (12,17)
obstacles = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)}
grid_size = (20,20)
grid = np.zeros(grid_size)
for obstacle in obstacles:
    grid[obstacle[0]][obstacle[1]] = 1
path = AStarImplementation(start_node, goal_node,obstacles)
for node in path:
    grid[node[0]][node[1]] = 2
plot_grid_with_path(grid, path)
print(path)
