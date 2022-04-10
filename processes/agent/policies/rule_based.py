import time


# Trackmania Tutorial Track 1
def rule_based_policy_test(start, **kwargs):
    # Base action : Accelerate
    action = {"rt": 1, "lt": 0, "ls": 0}

    dur = time.time() - start
    if 5.35 > dur > 3.85:
        action['ls'] = 1   # Turn right

    elif 6.9 > dur >= 5.45:
        action['ls'] = -1  # Turn left
        action['rt'] = max(1.8 * (dur - 5), 1)  # Reduce speed for a bit

    elif 8.30 > dur > 6.9:
        action['ls'] = 1   # Turn right

    else:
        action['ls'] = 0   # Straight ahead!

    return action, action
