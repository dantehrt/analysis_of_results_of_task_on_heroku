from utils import argmax
import sys
import pprint
import matplotlib.pyplot as plt
import pandas as pd


class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state. [page 646]"""

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)

        self.init = init

        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist

        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""

        return self.reward[state]

    def T(self, state, action):
        """Transition model. From a state and an action, return a list
        of (probability, result-state) pairs."""

        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""

        if state in self.terminals:
            return [None]
        else:
            return self.actlist

    def get_states_from_transitions(self, transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(tr[1] for actions in transitions.values()
                     for effects in actions.values()
                     for tr in effects)
            return s1.union(s2)
        else:
            print('Could not retrieve states from transitions')
            return None

    def check_consistency(self):

        # check that all states in transitions are valid
        assert set(self.states) == self.get_states_from_transitions(self.transitions)

        # check that init is a valid state
        assert self.init in self.states

        # check reward for each state
        assert set(self.reward.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001


class originalMDP(MDP):
    """MDPを継承"""

    def __init__(self, init, actlist, terminals, transitions, reward, states, max_repetition, max_switching):
        """Constructor for originalMDP"""
        self.max_repetition = max_repetition
        self.max_switching = max_switching
        MDP.__init__(self, init=init, actlist=actlist, terminals=terminals, transitions=transitions, reward=reward,
                     states=states, gamma=0.9)

    def T(self, state, action):
        if action is None:
            return [(1, state)]
        else:
            previous_task = state[0]
            repetition_times = state[1]
            switching_times = state[2]
            # 初期状態の時
            if previous_task == 0:
                return [(1 - self.transitions['repetition'][action][repetition_times],
                         (action, 1, 0)),
                        (self.transitions['repetition'][action][repetition_times],
                         (self.terminals[0]))]
            # 切り替え回数スイッチ回数共に最大になったとき，
            elif switching_times == self.max_switching and repetition_times == self.max_repetition:
                return [(1, (self.terminals[0]))]
            # 切り替え回数が最大のでタスクが同じ時
            elif switching_times == self.max_switching and previous_task != action:
                return [(1, (self.terminals[0]))]
            # 繰り返し回数が最大で，切り替わる時
            elif repetition_times == self.max_repetition and previous_task == action:
                return [(1, (self.terminals[0]))]
            # 前回とタスクが同じ時
            elif previous_task == action:
                is_repeat = True
            # 前回とタスクが異なる時
            elif previous_task != action:
                is_repeat = False
            else:
                raise ValueError('おかしい')

            if is_repeat:
                return [(1 - self.transitions['repetition'][action][repetition_times],
                         (action, repetition_times + 1, switching_times)),
                        (self.transitions['repetition'][action][repetition_times],
                         (self.terminals[0]))]
            else:
                return [(1 - self.transitions['switching'][switching_times],
                         (action, 1, switching_times + 1)),
                        (self.transitions['switching'][switching_times],
                         (self.terminals[0]))]


# 状態を作成
# state = (直前のタスクの種類，タスク繰り返し回数，切り替え回数）
def create_states(action_list, max_repetition, max_switching, init, terminal):
    states = []
    for action in action_list:
        for index_repetition in range(1, max_repetition + 1):
            for index_switching in range(max_switching + 1):
                states.append((action, index_repetition, index_switching))
    states.append(init)
    states.append(terminal)
    return states


def value_iteration(mdp, epsilon=0.001):
    U1 = {s: 0 for s in mdp.states}  # 内包表記
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def draw_route(pi, state, terminal):
    action = pi[state]
    if action is None:
        return (str(state), 'END')
    else:
        previous_task = state[0]
        repetition_times = state[1]
        switching_times = state[2]

        # 初期状態の時
        if previous_task == 0:
            next_state = (action, 1, switching_times)
        # 前回とタスクが同じ時
        elif previous_task == action:
            next_state = (previous_task, repetition_times + 1, switching_times)
        # 前回とタスクが異なる時
        elif previous_task != action:
            next_state = (action, 1, switching_times + 1)
        else:
            raise ValueError('おっかしいなぁ')

        if next_state not in pi.keys():
            return (str(state) + ' "' + str(next_state) + '"', 'END')
    answer = draw_route(pi, next_state, terminal)
    return (str(state) + ' ' + answer[0],
            str(pi[state]) + ' ' + answer[1])


def show_results(route):
    # route = '1 1 1 2 2 2 2 1 1 1 1 1 2 2 2 2 2 1 1 1 1 1 2 2 2 2 2 2 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1 1 1 1 2 2 2 2 2 2 2 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 END'
    states = route.split()

    num_of_1 = []
    num_of_2 = []
    message = ''
    previous_state = states[0]
    count = 0
    for state in states:
        if previous_state == state:
            count += 1
        else:
            message += 'task:{}, repetition_times:{} \n'.format(previous_state, count)
            if previous_state == '1':
                num_of_1.append(count)
                num_of_2.append(0)
            else:
                num_of_1.append(0)
                num_of_2.append(count)
            count = 1
        previous_state = state
    print(message)
    print(num_of_1)
    print(num_of_2)

    plt.bar([x for x in range(len(num_of_1))], num_of_1, label='task1')
    plt.bar([x for x in range(len(num_of_2))], num_of_2, label='task2')
    plt.xlabel('number_of_switching')
    plt.ylabel('number_of_repetition')
    plt.legend()
    plt.show()

    df = pd.concat([pd.Series(num_of_1, name='task1'), pd.Series(num_of_2, name='task2')], axis=1)
    df.to_csv('mdp_result.csv')

def solve_mdp():
    # 行動（t1を与えるかt2を与えるか）
    actlist = [1, 2]

    # タスクが同じだった時(max_repetition + 1個)，違った時(max_switching + 1個)の遷移する確率
    repetition_1 = [0, 0.00909090909090909, 0.01818181818181818, 0.02727272727272727, 0.03636363636363636,
                    0.045454545454545456, 0.05454545454545454, 0.06363636363636363, 0.07272727272727272,
                    0.08181818181818182, 0.09090909090909091, 0.13636363636363635, 0.18181818181818182, 0.2,
                    0.21818181818181817, 0.23636363636363636, 0.2545454545454545, 0.2727272727272727,
                    0.28787878787878785, 0.30303030303030304, 0.3181818181818182, 0.3333333333333333,
                    0.3484848484848485, 0.36363636363636365, 0.38181818181818183, 0.4, 0.4181818181818182,
                    0.43636363636363634, 0.45454545454545453, 0.4675324675324675, 0.4805194805194805,
                    0.49350649350649345, 0.5064935064935064, 0.5194805194805194, 0.5324675324675324, 0.5454545454545454,
                    0.5636363636363636, 0.5818181818181818, 0.6, 0.6181818181818182, 0.6363636363636364,
                    0.6493506493506493, 0.6623376623376623, 0.6753246753246753, 0.6883116883116883, 0.7012987012987013,
                    0.7142857142857143, 0.7272727272727273, 0.7575757575757576, 0.7878787878787878, 0.8181818181818181,
                    0.8484848484848485, 0.8787878787878788, 0.9090909090909091, 0.9119318181818181, 0.9147727272727273,
                    0.9176136363636364, 0.9204545454545454, 0.9232954545454545, 0.9261363636363636, 0.9289772727272727,
                    0.9318181818181818, 0.9346590909090909, 0.9375, 0.9403409090909091, 0.9431818181818181,
                    0.9460227272727273, 0.9488636363636364, 0.9517045454545454, 0.9545454545454546, 0.9573863636363636,
                    0.9602272727272727, 0.9630681818181818, 0.9659090909090909, 0.96875, 0.9715909090909091,
                    0.9744318181818181, 0.9772727272727273, 0.9801136363636364, 0.9829545454545454, 0.9857954545454546,
                    0.9886363636363636, 0.9914772727272727, 0.9943181818181819, 0.9971590909090909, 1.0]
    repetition_2 = [0, 0.015873015873015872, 0.031746031746031744, 0.047619047619047616, 0.06349206349206349,
                    0.07936507936507936, 0.09523809523809523, 0.1111111111111111, 0.12698412698412698,
                    0.14285714285714285, 0.15873015873015872, 0.1746031746031746, 0.19047619047619047,
                    0.20634920634920634, 0.2222222222222222, 0.25, 0.2777777777777778, 0.3055555555555555,
                    0.3333333333333333, 0.3472222222222222, 0.3611111111111111, 0.375, 0.38888888888888884,
                    0.4027777777777778, 0.41666666666666663, 0.4305555555555555, 0.4444444444444444, 0.4567901234567901,
                    0.4691358024691358, 0.48148148148148145, 0.49382716049382713, 0.5061728395061729,
                    0.5185185185185185, 0.5308641975308642, 0.5432098765432098, 0.5555555555555556, 0.6111111111111112,
                    0.6666666666666666, 0.6717171717171717, 0.6767676767676767, 0.6818181818181818, 0.6868686868686869,
                    0.6919191919191919, 0.6969696969696969, 0.702020202020202, 0.7070707070707071, 0.7121212121212122,
                    0.7171717171717171, 0.7222222222222222, 0.7272727272727273, 0.7323232323232323, 0.7373737373737373,
                    0.7424242424242424, 0.7474747474747475, 0.7525252525252525, 0.7575757575757576, 0.7626262626262627,
                    0.7676767676767677, 0.7727272727272727, 0.7777777777777778, 0.7916666666666666, 0.8055555555555556,
                    0.8194444444444444, 0.8333333333333333, 0.8472222222222222, 0.861111111111111, 0.875,
                    0.8888888888888888, 0.9027777777777777, 0.9166666666666666, 0.9305555555555556, 0.9444444444444444,
                    0.9583333333333333, 0.9722222222222222, 0.9861111111111112, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    switching = [0.006993006993006993, 0.013986013986013986, 0.02097902097902098, 0.027972027972027972,
                 0.03496503496503497, 0.04195804195804196, 0.04895104895104895, 0.055944055944055944,
                 0.06293706293706294, 0.06993006993006994, 0.07692307692307693, 0.15384615384615385, 0.3076923076923077,
                 0.34615384615384615, 0.38461538461538464, 0.4358974358974359, 0.48717948717948717, 0.5384615384615384,
                 0.5576923076923077, 0.5769230769230769, 0.5961538461538461, 0.6153846153846154, 0.6346153846153846,
                 0.6538461538461539, 0.6730769230769231, 0.6923076923076923, 0.6978021978021978, 0.7032967032967032,
                 0.7087912087912088, 0.7142857142857143, 0.7197802197802198, 0.7252747252747253, 0.7307692307692308,
                 0.7362637362637363, 0.7417582417582418, 0.7472527472527473, 0.7527472527472527, 0.7582417582417582,
                 0.7637362637362638, 0.7692307692307693, 0.8076923076923077, 0.8461538461538461, 0.8571428571428571,
                 0.8681318681318682, 0.8791208791208791, 0.8901098901098902, 0.9010989010989011, 0.9120879120879122,
                 0.9230769230769231, 0.9300699300699301, 0.9370629370629371, 0.9440559440559441, 0.951048951048951,
                 0.9580419580419581, 0.9650349650349651, 0.9720279720279721, 0.9790209790209791, 0.986013986013986,
                 0.993006993006993, 1.0]

    # repetition_1 = [1, 1, 1]
    # repetition_2 = [1, 1, 1]
    # switching = [0.8, 0.9, 0.9, 1]

    transitions = {'repetition': {1: repetition_1, 2: repetition_2}, 'switching': switching}

    # 初期状態，終了状態
    init = (0, 0, 0)
    terminal = (-100, -100, -100)

    # 最大繰り返し回数，最大スイッチ回数
    max_repetition = len(repetition_1) - 1
    max_switching = len(switching) - 1
    states = create_states(actlist, max_repetition, max_switching, init, terminal)

    reward = {}  # 報酬を作成(全部1)，終了状態だけ0に
    for state in states:
        reward[state] = 1
    reward[terminal] = 0

    mdp = originalMDP(init, actlist, [terminal], transitions, reward, states, max_repetition, max_switching)
    pi = best_policy(mdp, value_iteration(mdp))
    sys.setrecursionlimit(10000)
    states = draw_route(pi, init, terminal)
    print(states[0])
    print(states[1])
    show_results(states[1])

    # U = value_iteration(mdp)
    # # pprint.pprint(U)
    # s = (1, 20, 0)
    # print(argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp)))
    # print(expected_utility(1, s, U, mdp))
    # pprint.pprint([p * U[s1] for (p, s1) in mdp.T(s, 1)])
    # print(expected_utility(2, s, U, mdp))
    # pprint.pprint([p * U[s1] for (p, s1) in mdp.T(s, 2)])


def calculate_expected_value_of_completed_number(number_of_tasks, switch_frequency, t=0, n=0, m=0, count=0):
    # タスクが同じだった時(max_repetition + 1個)，違った時(max_switching + 1個)の遷移する確率
    repetition_1 = [0, 0.00909090909090909, 0.01818181818181818, 0.02727272727272727, 0.03636363636363636,
                    0.045454545454545456, 0.05454545454545454, 0.06363636363636363, 0.07272727272727272,
                    0.08181818181818182, 0.09090909090909091, 0.13636363636363635, 0.18181818181818182, 0.2,
                    0.21818181818181817, 0.23636363636363636, 0.2545454545454545, 0.2727272727272727,
                    0.28787878787878785, 0.30303030303030304, 0.3181818181818182, 0.3333333333333333,
                    0.3484848484848485, 0.36363636363636365, 0.38181818181818183, 0.4, 0.4181818181818182,
                    0.43636363636363634, 0.45454545454545453, 0.4675324675324675, 0.4805194805194805,
                    0.49350649350649345, 0.5064935064935064, 0.5194805194805194, 0.5324675324675324, 0.5454545454545454,
                    0.5636363636363636, 0.5818181818181818, 0.6, 0.6181818181818182, 0.6363636363636364,
                    0.6493506493506493, 0.6623376623376623, 0.6753246753246753, 0.6883116883116883, 0.7012987012987013,
                    0.7142857142857143, 0.7272727272727273, 0.7575757575757576, 0.7878787878787878, 0.8181818181818181,
                    0.8484848484848485, 0.8787878787878788, 0.9090909090909091, 0.9119318181818181, 0.9147727272727273,
                    0.9176136363636364, 0.9204545454545454, 0.9232954545454545, 0.9261363636363636, 0.9289772727272727,
                    0.9318181818181818, 0.9346590909090909, 0.9375, 0.9403409090909091, 0.9431818181818181,
                    0.9460227272727273, 0.9488636363636364, 0.9517045454545454, 0.9545454545454546, 0.9573863636363636,
                    0.9602272727272727, 0.9630681818181818, 0.9659090909090909, 0.96875, 0.9715909090909091,
                    0.9744318181818181, 0.9772727272727273, 0.9801136363636364, 0.9829545454545454, 0.9857954545454546,
                    0.9886363636363636, 0.9914772727272727, 0.9943181818181819, 0.9971590909090909, 1.0]
    repetition_2 = [0, 0.015873015873015872, 0.031746031746031744, 0.047619047619047616, 0.06349206349206349,
                    0.07936507936507936, 0.09523809523809523, 0.1111111111111111, 0.12698412698412698,
                    0.14285714285714285, 0.15873015873015872, 0.1746031746031746, 0.19047619047619047,
                    0.20634920634920634, 0.2222222222222222, 0.25, 0.2777777777777778, 0.3055555555555555,
                    0.3333333333333333, 0.3472222222222222, 0.3611111111111111, 0.375, 0.38888888888888884,
                    0.4027777777777778, 0.41666666666666663, 0.4305555555555555, 0.4444444444444444, 0.4567901234567901,
                    0.4691358024691358, 0.48148148148148145, 0.49382716049382713, 0.5061728395061729,
                    0.5185185185185185, 0.5308641975308642, 0.5432098765432098, 0.5555555555555556, 0.6111111111111112,
                    0.6666666666666666, 0.6717171717171717, 0.6767676767676767, 0.6818181818181818, 0.6868686868686869,
                    0.6919191919191919, 0.6969696969696969, 0.702020202020202, 0.7070707070707071, 0.7121212121212122,
                    0.7171717171717171, 0.7222222222222222, 0.7272727272727273, 0.7323232323232323, 0.7373737373737373,
                    0.7424242424242424, 0.7474747474747475, 0.7525252525252525, 0.7575757575757576, 0.7626262626262627,
                    0.7676767676767677, 0.7727272727272727, 0.7777777777777778, 0.7916666666666666, 0.8055555555555556,
                    0.8194444444444444, 0.8333333333333333, 0.8472222222222222, 0.861111111111111, 0.875,
                    0.8888888888888888, 0.9027777777777777, 0.9166666666666666, 0.9305555555555556, 0.9444444444444444,
                    0.9583333333333333, 0.9722222222222222, 0.9861111111111112, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    switching = [0.006993006993006993, 0.013986013986013986, 0.02097902097902098, 0.027972027972027972,
                 0.03496503496503497, 0.04195804195804196, 0.04895104895104895, 0.055944055944055944,
                 0.06293706293706294, 0.06993006993006994, 0.07692307692307693, 0.15384615384615385, 0.3076923076923077,
                 0.34615384615384615, 0.38461538461538464, 0.4358974358974359, 0.48717948717948717, 0.5384615384615384,
                 0.5576923076923077, 0.5769230769230769, 0.5961538461538461, 0.6153846153846154, 0.6346153846153846,
                 0.6538461538461539, 0.6730769230769231, 0.6923076923076923, 0.6978021978021978, 0.7032967032967032,
                 0.7087912087912088, 0.7142857142857143, 0.7197802197802198, 0.7252747252747253, 0.7307692307692308,
                 0.7362637362637363, 0.7417582417582418, 0.7472527472527473, 0.7527472527472527, 0.7582417582417582,
                 0.7637362637362638, 0.7692307692307693, 0.8076923076923077, 0.8461538461538461, 0.8571428571428571,
                 0.8681318681318682, 0.8791208791208791, 0.8901098901098902, 0.9010989010989011, 0.9120879120879122,
                 0.9230769230769231, 0.9300699300699301, 0.9370629370629371, 0.9440559440559441, 0.951048951048951,
                 0.9580419580419581, 0.9650349650349651, 0.9720279720279721, 0.9790209790209791, 0.986013986013986,
                 0.993006993006993, 1.0]
    # repetition_1 = [0, 0.1, 0.2, 1]
    # repetition_2 = [0, 0.4, 0.5, 1]
    # switching = [0.3, 1]
    repetition = {1: repetition_1, 2: repetition_2}

    # print('t:{}, n:{}, m:{}, count:{}'.format(t, n, m, count))
    if t == 0:
        # return str(1 - repetition[1][0]) + '*(' + calculate_expected_value_of_completed_number(number_of_tasks, switch_frequency, 1, 1, 0, count+1) + ')'
        return (1 - repetition[1][0]) * calculate_expected_value_of_completed_number(number_of_tasks, switch_frequency,
                                                                                     1, 1, 0, count + 1)
    elif count == number_of_tasks - 1:
        if n == switch_frequency:
            # return str(1 - switching[m]) + '*' + str(number_of_tasks) + '+' + str(switching[m]) + '*' + str(count)
            return (1 - switching[m]) * number_of_tasks + switching[m] * count
        else:
            # return str(1 - repetition[t][n]) + '*' + str(number_of_tasks) + '+' + str(repetition[t][n]) + '*' + str(count)
            return (1 - repetition[t][n]) * number_of_tasks + repetition[t][n] * count
    else:
        if n == switch_frequency:
            # return str(1-switching[m]) + '*(' + calculate_expected_value_of_completed_number(number_of_tasks, switch_frequency, 1 + t % 2, 1, m + 1, count+1) +')+' + str(switching[m]) + '*' + str(count)
            return (1 - switching[m]) * calculate_expected_value_of_completed_number(number_of_tasks, switch_frequency,
                                                                                     1 + t % 2, 1, m + 1, count + 1) + \
                   switching[m] * count
        else:
            # return str(1-repetition[t][n]) + '*(' + calculate_expected_value_of_completed_number(number_of_tasks, switch_frequency, t, n + 1, m, count+1) +')+' + str(repetition[t][n]) + '*' + str(count)
            return (1 - repetition[t][n]) * calculate_expected_value_of_completed_number(number_of_tasks,
                                                                                         switch_frequency, t, n + 1, m,
                                                                                         count + 1) + repetition[t][
                       n] * count


def show_graph_of_expected_number_of_completed_tasks():
    switch_frequency_list = [x for x in range(2, 25)]
    expected_number_of_completed_tasks_list = []
    for switch_frequency in switch_frequency_list:
        expected_number_of_completed_tasks_list.append(
            calculate_expected_value_of_completed_number(90, switch_frequency))
    print(expected_number_of_completed_tasks_list)
    plt.bar(switch_frequency_list, expected_number_of_completed_tasks_list)
    plt.xlabel('switch_frequency')
    plt.ylabel('expected_number_of_completed_tasks')
    plt.show()


if __name__ == '__main__':
    show_graph_of_expected_number_of_completed_tasks()
    # solve_mdp()
