import random
import numpy as np
from smartcab.environment import Environment
from smartcab.simulator import Simulator
from smartcab.agent import LearningAgent
from visuals import calculate_reliability, calculate_safety
import csv
import os
import pandas as pd
import ast

def run(alpha, trials):
    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=False)

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, epsilon=1.0, alpha=alpha, tolerance=0.05, trials=trials)

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, log_metrics=True, display=False, optimized=True)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10)

    return agent

def getQStats(agent):
    allStateAction = 0
    visitedStateAction = 0
    noStates = 0

    for state, action_reward in agent.Q.iteritems():
        noStates +=1
        for action, reward in action_reward.iteritems():
            allStateAction += 1
            visitedStateAction += 1 if reward != 0. else 0

    return noStates, allStateAction, visitedStateAction

def runExperiment(writer, alpha, trials):
    agent = run(alpha = alpha, trials = trials)
    noStates, allStateAction, visitedStateAction = getQStats(agent)
    data = pd.read_csv(os.path.join("logs", "sim_improved-learning.csv"))
    data['good_actions'] = data['actions'].apply(lambda x: ast.literal_eval(x)[0])
    testing_data = data[data['testing'] == True]
    avg_reward = testing_data['net_reward'].mean()
    safety_rating, _ = calculate_safety(testing_data)
    reliability_rating, _ = calculate_reliability(testing_data)
    writer.writerow({
        'alpha' : alpha,
        'trials' : trials,
        'states' : noStates,
        'all_s_a' : allStateAction,
        'visited_s_a' : visitedStateAction,
        'avg_reward': avg_reward,
        'safety': safety_rating,
        'reliability': reliability_rating
    })

def optimize():
    report_filename = os.path.join('logs', 'report_200.csv')
    fields = ['alpha', 'trials', 'states', 'all_s_a', 'visited_s_a', 'avg_reward', 'safety', 'reliability']
    np.random.seed(42)
    random.seed = 42

    with open(report_filename, "wb") as report:
        writer = csv.DictWriter(report, fieldnames=fields)
        writer.writeheader()
        for i in range(11):
            alpha, trials = 0.5, 200 + i * 10 #np.random.randint(20, 200)
            runExperiment(writer, alpha, trials)

if __name__ == '__main__':
    optimize()
