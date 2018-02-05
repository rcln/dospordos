import os
from environment import Environment
from agent import Agent
import numpy as np


def main():

    eps = 0.1

    data_path = "".join(s+"/" for s in (os.getcwd().split('/')[:-1])) + "DATA/"
    data_train = data_path+"train_db/"
    data_db = data_path+"fer_db/train.json"

    # loading users
    list_users = sorted(list(map(int, os.listdir(data_train))))

    env = Environment()
    agent = Agent(env)

    env.set_path_train(data_train)
    env.set_path_files(data_db)

    agent.print_model()

    # episodes
    for us in list_users:
        # reset episode with new user and get initial state
        # TODO fucntion to make this random
        replay_memory = []

        # initial state
        state = env.reset(us)
        done = False
        # DQN with experience replace
        while not done:
            """
            Select an action with an epsilon 
            """
            action_vector = None
            p = np.random.random()
            if p < eps:
                # TODO make a real random vector theo
                action_vector = [1,0,0,0,0,0]
            else:
                # a = argmaxQ(s,a)
                pass

            # Observe reward and new state
            # example
            reward, nextx_state, done = env.step(agent.next_snippet, agent.change_db)

            # Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
            # state = next_state
        break

if __name__ == "__main__":
    main()
