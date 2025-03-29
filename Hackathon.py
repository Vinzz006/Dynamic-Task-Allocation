import numpy as np
import random

class DynamicTaskAllocator:
    def __init__(self, n_resources, n_tasks):
        self.n_resources = n_resources  # Number of resources (e.g., robots)
        self.n_tasks = n_tasks  # Number of tasks to be allocated
        
        self.resources_state = np.zeros(n_resources)  # All resources start as idle (0)
        self.q_table = np.zeros((n_resources, n_tasks))  # Q-table initialized with zeros
        
        self.alpha = 0.1  # Learning rate (how fast it learns)
        self.gamma = 0.9  # Discount factor (importance of future rewards)
        self.epsilon = 0.1  # Exploration rate (chance of random action)

    def choose_action(self, task):
        available_resources = [i for i in range(self.n_resources) if self.resources_state[i] == 0]
        
        if not available_resources:
            return -1  # No available resources

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_resources)  # Randomly choose an idle resource
        else:
            return max(available_resources, key=lambda x: self.q_table[x, task])  # Choose best resource

    def update_q_table(self, resource, task, reward, next_best_q):
        current_q = self.q_table[resource, task]
        self.q_table[resource, task] = current_q + self.alpha * (reward + self.gamma * next_best_q - current_q)

    def assign_task(self, task):
        resource = self.choose_action(task)
        if resource == -1:
            return -1  # No available resources
        
        self.resources_state[resource] = 1  # Mark resource as busy
        reward = 1  # Fixed reward for successful assignment
        next_best_q = np.max(self.q_table[resource])  # Best future Q-value
        self.update_q_table(resource, task, reward, next_best_q)
        
        return resource

    def complete_task(self, resource):
        self.resources_state[resource] = 0  # Make resource available again


# Main Execution
n_resources = 5
n_tasks = 10

allocator = DynamicTaskAllocator(n_resources, n_tasks)

for task in range(n_tasks):
    assigned_resource = allocator.assign_task(task)
    if assigned_resource != -1:
        print(f"Task {task} assigned to Resource {assigned_resource}")
        allocator.complete_task(assigned_resource)  # Free up resource
    else:
        print(f"Task {task} could not be assigned, no available resources.")
