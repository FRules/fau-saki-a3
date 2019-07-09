import csv
import numpy as np

NUMBER_OF_SLOTS = 4
COST_MAPPING = {
    0: 1,
    1: 2,
    2: 2,
    3: 3,
    4: 3,
    5: 4
}


def init_warehouse():
    return ['x' for i in range(NUMBER_OF_SLOTS)]


def get_storing_sequence(file):
    storing_sequence = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            action = (row[0][0] + row[1][0]).upper()
            storing_sequence.append(action)
    return storing_sequence


def find_first_empty_slot(warehouse):
    for i in range(len(warehouse)):
        if warehouse[i] == "x":
            return i


def find_first_slot_with_color(warehouse, color):
    for i in range(len(warehouse)):
        if warehouse[i] == color:
            return i


def get_distance_using_greedy_algorithm(warehouse, storing_sequence):
    distance = 0
    for action in storing_sequence:
        is_storing_action = action[0] == "S"
        target_color = action[1]
        if is_storing_action:
            slot = find_first_empty_slot(warehouse)
            warehouse[slot] = target_color
        else:
            slot = find_first_slot_with_color(warehouse, target_color)
            warehouse[slot] = "x"
        distance += COST_MAPPING.get(slot)
    return distance


def state_equals_warehouse(state, warehouse, action):
    for i in range(len(state)):
        if i == NUMBER_OF_SLOTS:
            if action == state[i]:
                return True
        if state[i] != warehouse[i]:
            return False


def get_warehouse_state_index(states, warehouse, action):
    for i in range(len(states)):
        if state_equals_warehouse(states[i], warehouse, action):
            return i


def get_distance_using_policy(warehouse, storing_sequence, policy):
    distance = 0
    for action in storing_sequence:
        is_storing_action = action[0] == "S"
        target_color = action[1]
        state_index = get_warehouse_state_index(states, warehouse, action)
        target_slot_according_to_policy = policies[state_index]
        if is_storing_action:
            warehouse[target_slot_according_to_policy] = target_color
        else:
            warehouse[target_slot_according_to_policy] = "x"
        distance += COST_MAPPING.get(target_slot_according_to_policy)
    return distance


warehouse = init_warehouse()
storing_sequence = get_storing_sequence("training_2x2.txt")
distance_greedy = get_distance_using_greedy_algorithm(warehouse, storing_sequence)
distance_policy = get_distance_using_policy(warehouse, storing_sequence, policy)

print(distance_greedy)
print(distance_policy)