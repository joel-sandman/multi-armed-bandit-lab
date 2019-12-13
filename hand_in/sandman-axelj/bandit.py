# epsilon-greedy example implementation of a multi-armed bandit
import random

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import simulator
import reference_bandit

# generic epsilon-greedy bandit
class Bandit:
    def __init__(self, arms, epsilon=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.frequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.expected_values = [0] * len(arms)
        self.x = 0.9
        self.discarded_arms = []
        self.runNumber = 0
        self.local_frequencies = [0] * len(arms)
        self.local_sums = [0] * len(arms)
        self.local_expected_values = [0] * len(arms)

    def run(self):
        if self.runNumber == 1000:
            self.epsilon = 0.1
            self.discarded_arms = []
            self.runNumber = 0
            self.local_frequencies = [0] * len(arms)
            self.local_sums = [0] * len(arms)
            self.local_expected_values = [0] * len(arms)
        self.runNumber += 1
        if min(self.local_frequencies) == 0:
            return self.arms[self.local_frequencies.index(min(self.local_frequencies))]
        if random.random() < self.epsilon:
            while True:
                arm_index = random.randint(0, len(arms) - 1)
                if arm_index not in self.discarded_arms:
                    break
            return self.arms[arm_index]
        return self.arms[self.local_expected_values.index(max(self.local_expected_values))]

    def give_feedback(self, arm, reward):
        arm_index = self.arms.index(arm)

        sum = self.sums[arm_index] + reward
        self.sums[arm_index] = sum
        local_sum = self.local_sums[arm_index] + reward
        self.local_sums[arm_index] = local_sum

        frequency = self.frequencies[arm_index] + 1
        self.frequencies[arm_index] = frequency
        local_frequency = self.local_frequencies[arm_index] + 1
        self.local_frequencies[arm_index] = local_frequency

        expected_value = sum / frequency
        self.expected_values[arm_index] = expected_value
        local_expected_value = local_sum / local_frequency
        self.local_expected_values[arm_index] = local_expected_value

        self.epsilon *= self.x
        if self.local_expected_values[arm_index] < -3:
            print("Discard arm: ", arm_index)
            self.discarded_arms.append(arm_index)

# configuration
arms = [
    'Configuration a',
    'Configuration b',
    'Configuration c',
    'Configuration d',
    'Configuration e',
    'Configuration f'
]

# instantiate bandits
bandit = Bandit(arms)
ref_bandit = reference_bandit.ReferenceBandit(arms)
