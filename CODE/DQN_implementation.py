import os
from environment import Environment
from agent import Agent

"""
Our Vector:
"""


def main():
    data_path = "".join(s+"/" for s in (os.getcwd().split('/')[:-1])) + "DATA/"
    data_train = data_path+"train_db/"
    data_db = data_path+"fer_db/train.json"

    # loading users
    list_users = sorted(list(map(int, os.listdir(data_train))))

    env = Environment()
    agent = Agent(env)

    env.set_path_train(data_train)
    env.set_path_files(data_db)

    # episodes
    for us in list_users:
        # reset episode with new user and get initial state

        # Todo initialize arbitrary
        # Q[num_states, num_actions]

        state = env.reset(us)
        done = False
        # Todo finish. check https://github.com/dennybritz/reinforcement-learning/tree/master/DQN
        # DQN with experience replace
        # check test_environment for usage
        while not done:

            # Todo select an action a

            # Observe reward and new state
            # example
            reward, nextx_state, done = env.step(agent.next_snippet, agent.change_db)

            # Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
            # state = next_state

if __name__ == "__main__":
    main()
