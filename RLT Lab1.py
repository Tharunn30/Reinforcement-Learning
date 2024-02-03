#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Define the 3x3 grid world
grid_world_size = 3
start_position = (0, 0)
goal_position = (2, 2)
actions = ["left", "right", "up", "down"]

# Initialize Q-values for each state-action pair
Q_table = {(x, y): {action: 0.0 for action in actions} for x in range(grid_world_size) for y in range(grid_world_size)}

# Define hyperparameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.2
num_episodes = 200

# Function to choose an action using epsilon-greedy strategy
def choose_action(state):
    if np.random.rand() < exploration_rate:
        return np.random.choice(actions)
    else:
        return max(actions, key=lambda a: Q_table[state][a])

# Function to take an action and get the next state and reward
def take_action(state, action):
    x, y = state
    if action == "left" and x > 0:
        x -= 1
    elif action == "right" and x < grid_world_size - 1:
        x += 1
    elif action == "up" and y > 0:
        y -= 1
    elif action == "down" and y < grid_world_size - 1:
        y += 1

    next_state = (x, y)
    reward = 1 if next_state == goal_position else 0
    return next_state, reward

# Q-learning training
for episode in range(num_episodes):
    current_state = start_position
    while current_state != goal_position:
        action = choose_action(current_state)
        next_state, reward = take_action(current_state, action)
        current_q_value = Q_table[current_state][action]
        best_next_action = max(actions, key=lambda a: Q_table[next_state][a])
        new_q_value = current_q_value + learning_rate * (reward + discount_factor * Q_table[next_state][best_next_action] - current_q_value)
        Q_table[current_state][action] = new_q_value
        current_state = next_state

# Function to visualize the grid world with the agent's path
def visualize_grid_world(path):
    for y in range(grid_world_size - 1, -1, -1):
        for x in range(grid_world_size):
            position = (x, y)
            if position == start_position:
                print("S", end=" ")
            elif position == goal_position:
                print("G", end=" ")
            elif position in path:
                print("P", end=" ")
            else:
                print(".", end=" ")
        print()

# Function to use the learned Q-values to navigate the grid world
def navigate_grid_world():
    current_state = start_position
    path = [current_state]

    while current_state != goal_position:
        action = max(actions, key=lambda a: Q_table[current_state][a])
        next_state, _ = take_action(current_state, action)
        path.append(next_state)
        current_state = next_state

    visualize_grid_world(path)

# Call the function to navigate the grid world using learned Q-values
navigate_grid_world()


# In[ ]:




