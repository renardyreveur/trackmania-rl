import time


# Trackmania Tutorial Track 1
def rule_based_policy_test(start, **kwargs):
    # Base action : Accelerate
    action = {
        "rt_mu": 1, "lt_mu": 0, "ls_mu": 0,
        "rt_sigma": 0, "lt_sigma": 0, "ls_sigma": 0
    }

    dur = time.time() - start
    if 5.3 > dur > 4.0:
        action['ls_mu'] = 1   # Turn right

    elif 6.9 > dur >= 5.45:
        action['ls_mu'] = -1  # Turn left
        action['rt_mu'] = max(1.8 * dur - 5 * 1.8, 1)  # Reduce speed for a bit

    elif 8.30 > dur > 6.9:
        action['ls_mu'] = 1   # Turn right

    else:
        action['ls_mu'] = 0   # Straight ahead!

    return action
