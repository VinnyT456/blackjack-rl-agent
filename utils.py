import numpy as np

card_prob = {
    '1': 4/52,
    '2': 4/52,
    '3': 4/52,
    '4': 4/52,
    '5': 4/52,
    '6': 4/52,
    '7': 4/52,
    '8': 4/52,
    '9': 4/52,
    '10': 16/52,
}

def encode_state(state):
    return np.array([
        state[0] / 32,
        state[1] / 11,
        int(state[2])
    ], dtype=np.float32)

def custom_action(state):
    #Check the current state's sum and ace
    usable_ace = bool(state[2])
    current_sum = int(state[0] * 32)

    #Find the total probability of busting
    bust_limit = 21 - current_sum
    bust_chance = sum([card_prob[str(i)] for i in range(bust_limit + 1, 11)])

    #Find the probability of busting when there's a usable ace.
    if usable_ace:
        bust_limit_11 = 21 - current_sum
        bust_11 = sum(card_prob[str(i)] for i in range(bust_limit_11 + 1, 11))

        bust_limit_1 = 21 - (current_sum - 10)
        bust_1 = sum(card_prob[str(i)] for i in range(bust_limit_1 + 1, 11))

        bust_chance = min(bust_11, bust_1)
    
    #Return the action based on a certain threshold
    return 1 if bust_chance < 0.5 else 0

def custom_reward(state, action):
    usable_ace = bool(state[2])
    current_sum = int(state[0] * 32)

    bust_limit = 21 - current_sum
    bust_chance = sum([card_prob[str(i)] for i in range(bust_limit + 1, 11)])

    if usable_ace:
        bust_limit_11 = 21 - current_sum
        bust_11 = sum(card_prob[str(i)] for i in range(bust_limit_11 + 1, 11))

        bust_limit_1 = 21 - (current_sum - 10)
        bust_1 = sum(card_prob[str(i)] for i in range(bust_limit_1 + 1, 11))

        bust_chance = min(bust_11, bust_1)
    
    if (action == 1):
        return 1 if bust_chance < 0.5 else -1
    else:
        return 1 if bust_chance > 0.5 else -1