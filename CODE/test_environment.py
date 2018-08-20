# -*- coding:utf-8 -*-

from CODE.environment import Environment
from CODE.agent import Agent
import tensorflow as tf

"""
Problems:
* cuando detenerse
* familias semanticas 

** TODO: 
    -Implementar la red neuronal. [keras]
    -Implementar el TD-Gammon
    -Completar funciones del Agente
    
    - Revisar porque es tan lento y optimizarlo

"""

def main():

    env = Environment()
    env.set_path_files('../DATA/fer_db/train.json')
    env.set_path_train('../DATA/train_db/')
    # start new episode
    env.reset(1)

    # test of environment data
    print('data env', env.queues)
    print('data current queue env', env.current_queue)
    print('data size env current queue', env.queues[env.current_queue].qsize())

    agent = Agent(env)

    # action sending to environment by agent

    reward, state, done = env.step(agent.next_snippet, agent.add_current_db)
    # checking queue
    print('after size', env.queues[env.current_queue].qsize())
    print('current data', env.current_data)
    print('current text', env.current_text)
    print('reward', reward)
    print('state', state)
    print('done?', done)

    # changing to queue # 2
    env.step(agent.change_queue, agent.keep_db, 2)
    print('data size env current queue', env.queues[env.current_queue].qsize())
    print('env current queue', env.current_queue)

    # # changing to next queue
    # env.step(agent.change_queue, agent.keep_db)
    # print('data size env current queue', env.queues[env.current_queue].qsize())
    # print('env current queue', env.current_queue)

    pass

def test():
    env = Environment()
    agent = Agent(env)

    print('next_snippet', agent.next_snippet)
    print('add_current_db', agent.add_current_db)

    #reward, state, done = env.step(agent.next_snippet, agent.add_current_db)

    pass

if __name__ == "__main__":
    main()
    #test()

