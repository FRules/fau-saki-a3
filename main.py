import itertools
import numpy as np
import copy
import mdptoolbox
import csv

# Hyper-Parameter for the number of slots.
# Can be changed to 6, but this could result in a
# memory exception
NUMBER_OF_SLOTS = 4
TRAINING_FILE = "training_2x2.txt"
REWARD_MAPPING = {
    0: 6,
    1: 4,
    2: 4,
    3: 2,
    4: 2,
    5: 1
}


def get_percentage_of_actions_by_training_data(training_file):
    actions_dict = {"SR": 0, "SB": 0, "SW": 0, "RR": 0, "RB": 0, "RW": 0}
    sum = 0
    with open(training_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            action = (row[0][0] + row[1][0]).upper()
            actions_dict[action] = actions_dict[action] + 1
            sum += 1
    probabilities = {
        "SR": actions_dict["SR"] / sum,
        "SB": actions_dict["SB"] / sum,
        "SW": actions_dict["SW"] / sum,
        "RR": actions_dict["RR"] / sum,
        "RB": actions_dict["RB"] / sum,
        "RW": actions_dict["RW"] / sum,
    }
    return probabilities


def append_action_to_states(state_without_action, action):
    copied = list(copy.copy(state_without_action))
    copied.append(action)
    return np.array(copied)


def get_states():
    possibilities = ["x", "r", "w", "b"]
    states_without_actions = [p for p in itertools.product(possibilities, repeat=NUMBER_OF_SLOTS)]

    final_states = []
    for state_without_action in states_without_actions:
        final_states.append(append_action_to_states(state_without_action, "SB"))
        final_states.append(append_action_to_states(state_without_action, "SR"))
        final_states.append(append_action_to_states(state_without_action, "SW"))
        final_states.append(append_action_to_states(state_without_action, "RB"))
        final_states.append(append_action_to_states(state_without_action, "RR"))
        final_states.append(append_action_to_states(state_without_action, "RW"))

    return np.array(final_states)


def did_warehouse_change_except_in_slot(warehouse_slot, state_1, state_2):
    for i in range(len(state_1)):
        if i == warehouse_slot:
            continue
        if state_1[i] != state_2[i]:
            return True
    return False


def get_color(state):
    return state[NUMBER_OF_SLOTS][1].lower()


def is_store_action_possible(warehouse_slot, state_1, state_2):
    color_to_store = get_color(state_1)

    if did_warehouse_change_except_in_slot(warehouse_slot, state_1, state_2):
        return False

    if state_1[warehouse_slot] != "x" and state_1[warehouse_slot] != state_2[warehouse_slot]:
        # slot was overwritten, that is not possible
        return False

    if state_1[warehouse_slot] == "x" and state_2[warehouse_slot] != color_to_store:
        # empty slot was overwritten, but with something else we expected, not possible
        return False

    return True


def is_restore_action_possible(warehouse_slot, state_1, state_2):
    color_to_restore = get_color(state_1)

    if did_warehouse_change_except_in_slot(warehouse_slot, state_1, state_2):
        return False

    if state_1[warehouse_slot] != color_to_restore and state_1[warehouse_slot] != state_2[warehouse_slot]:
        # something changed which had nothing to do with the restore color command, not possible
        return False

    if state_1[warehouse_slot] == color_to_restore and state_2[warehouse_slot] != "x":
        # a color was replaced instead of just getting it, not possible
        return False

    return True


def is_transition_possible(warehouse_slot, state_1, state_2):
    action = state_1[NUMBER_OF_SLOTS][0]
    if action == "S":
        return is_store_action_possible(warehouse_slot, state_1, state_2)
    else:
        return is_restore_action_possible(warehouse_slot, state_1, state_2)


def distribute_transition_probability_matrix(tpm):
    for i_action, action in enumerate(tpm):
        for index, row_vector in enumerate(action):
            sum_of_probability = np.sum(row_vector)
            if sum_of_probability == 0:
                # no transition was possible for this state (e. g. restore from empty warehouse)
                # it should stay in the state then since the sum of each row has to equal one
                tpm[action, index, index] = 1
                continue
        # give every possible transition the same possibility by dividing every element by the sum
        # of the row. E. g. [0, 0, 1, 1, 0, 0] will be converted to [0, 0, 0.5, 0.5, 0, 0].
        # Also, for example [0.25, 0.15, 0.3] will be normalized to [0.35, 0.21, 0.44]
        # This will also result in the sum of the row equal to one.
        tpm[i_action] = action / action.sum(axis=1)[:, None]
    return tpm


def get_probability(probability_distribution, state):
    action = state[NUMBER_OF_SLOTS]
    return probability_distribution[action]


def get_transition_probability_matrix(probability_distribution, states):
    tpm = np.zeros((NUMBER_OF_SLOTS, len(states), len(states)))
    for warehouse_slot in range(NUMBER_OF_SLOTS):
        for i, state_1 in enumerate(states, start=0):
            for j, state_2 in enumerate(states, start=0):
                if is_transition_possible(warehouse_slot, state_1, state_2):
                    tpm[warehouse_slot, i, j] = get_probability(probability_distribution, state_2)
    return distribute_transition_probability_matrix(tpm)


def get_reward_for_store_action(state):
    reward = np.zeros(NUMBER_OF_SLOTS)

    for i in range(NUMBER_OF_SLOTS):
        if state[i] != "x":
            reward[i] = -1
        else:
            reward[i] = REWARD_MAPPING.get(i)
    return reward


def get_reward_for_restore_action(state):
    color_to_restore = get_color(state)
    reward = np.zeros(NUMBER_OF_SLOTS)

    for i in range(NUMBER_OF_SLOTS):
        if state[i] != color_to_restore:
            reward[i] = -1
        else:
            reward[i] = REWARD_MAPPING.get(i)
    return reward


def get_reward_matrix(states):
    rm = np.zeros((len(states), NUMBER_OF_SLOTS))
    for i, state_1 in enumerate(states, start=0):
        action = state_1[NUMBER_OF_SLOTS][0]
        if action == "S":
            reward = get_reward_for_store_action(state_1)
        else:
            reward = get_reward_for_restore_action(state_1)
        rm[i] = reward
    return rm


probability_distribution = get_percentage_of_actions_by_training_data(TRAINING_FILE)
states = get_states()
tpm = get_transition_probability_matrix(probability_distribution, states)
reward_matrix = get_reward_matrix(states)

print(tpm)
print(reward_matrix)

mdpResultPolicy = mdptoolbox.mdp.PolicyIteration(tpm, reward_matrix, 0.3, max_iter=len(states) * 10)
mdpResultValue = mdptoolbox.mdp.ValueIteration(tpm, reward_matrix, 0.3, max_iter=len(states) * 10)

# Run the MDP
mdpResultPolicy.run()
mdpResultValue.run()

print('PolicyIteration:')
print(mdpResultPolicy.policy)
print(mdpResultPolicy.V)
print(mdpResultPolicy.iter)

print('ValueIteration:')
print(mdpResultValue.policy)
print(mdpResultValue.V)
print(mdpResultValue.iter)


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
        target_color = action[1].lower()
        if is_storing_action:
            slot = find_first_empty_slot(warehouse)
            warehouse[slot] = target_color
        else:
            slot = find_first_slot_with_color(warehouse, target_color)
            warehouse[slot] = "x"
        distance += COST_MAPPING.get(slot)
    return distance


def state_equals_warehouse(state, warehouse, action):
    for i in range(len(warehouse)):
        if state[i] != warehouse[i]:
            return False
    return state[NUMBER_OF_SLOTS] == action


def get_warehouse_state_index(states, warehouse, action):
    for i in range(len(states)):
        if state_equals_warehouse(states[i], warehouse, action):
            return i


def get_distance_using_policy(warehouse, storing_sequence, policy):
    distance = 0
    for action in storing_sequence:
        is_storing_action = action[0] == "S"
        target_color = action[1].lower()
        state_index = get_warehouse_state_index(states, warehouse, action)
        target_slot_according_to_policy = policy[state_index]
        if is_storing_action:
            warehouse[target_slot_according_to_policy] = target_color
        else:
            warehouse[target_slot_according_to_policy] = "x"
        distance += COST_MAPPING.get(target_slot_according_to_policy)
    return distance


warehouse = init_warehouse()
storing_sequence = get_storing_sequence(TRAINING_FILE)
distance_greedy = get_distance_using_greedy_algorithm(warehouse, storing_sequence)

policyValueIteration = list(mdpResultValue.policy)
policyPolicyIteration = list(mdpResultPolicy.policy)
warehouse = init_warehouse()
distance_ValueIteration = get_distance_using_policy(warehouse, storing_sequence, policyValueIteration)
warehouse = init_warehouse()
distance_PolicyIteration = get_distance_using_policy(warehouse, storing_sequence, policyPolicyIteration)

print("Greedy: ", distance_greedy)
print("Value: ", distance_ValueIteration)
print("Policy: ", distance_PolicyIteration)