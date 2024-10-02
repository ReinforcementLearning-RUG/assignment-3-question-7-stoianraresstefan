import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_mdp.mdp.mdp import MDP
from rl_mdp.policy.policy import Policy
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator

def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """

    states = ['s0', 's1', 's2', 's3']
    actions = ['a0', 'a1']
    
    def transition_function(state, action):
        if state == 's0':
            if action == 'a0':
                return {'s1': 0.8, 's2': 0.2}
            elif action == 'a1':
                return {'s2': 1.0}
        elif state == 's1':
            if action == 'a0':
                return {'s1': 0.5, 's3': 0.5}
            elif action == 'a1':
                return {'s2': 1.0}
        elif state == 's2':
            if action == 'a0':
                return {'s3': 1.0}
            elif action == 'a1':
                return {'s2': 1.0}
        elif state == 's3':  # Terminal state
            return {}

    def reward_function(state, action):
        if state == 's0' and action == 'a0':
            return 0
        elif state == 's0' and action == 'a1':
            return 0
        elif state == 's1' and action == 'a0':
            return 1
        elif state == 's1' and action == 'a1':
            return -1
        elif state == 's2' and action == 'a0':
            return 2
        elif state == 's2' and action == 'a1':
            return -1
        else:
            return 0

    # Instantiate the MDP and Policy
    mdp = MDP(states, actions, transition_function, reward_function)
    policy = Policy()

    # Instantiate the Monte Carlo Evaluator
    mc_evaluator = MCEvaluator(mdp, policy)
    mc_evaluator.evaluate(episodes=1000)
    print("Monte Carlo Value Table:")
    print(mc_evaluator.get_value_table())

    # Instantiate the TD(0) Evaluator
    td_evaluator = TDEvaluator(mdp, alpha=0.1)
    td_value_table = td_evaluator.evaluate(policy, num_episodes=1000)
    print("TD(0) Value Table:")
    print(td_value_table)

    # Instantiate the TD(λ) Evaluator
    td_lambda_evaluator = TDLambdaEvaluator(mdp, alpha=0.1, lambd=0.9)
    td_lambda_value_table = td_lambda_evaluator.evaluate(policy, num_episodes=1000)
    print("TD(λ) Value Table:")
    print(td_lambda_value_table)

if __name__ == "__main__":
    main()
