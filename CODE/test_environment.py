# -*- coding:utf-8 -*-

from CODE.environment import Environment
from CODE.agent import Agent


def main():
    env = Environment()
    env.set_path_files('/home/urb/PycharmProjects/dospordos/DATA/train_db/')

    # start new episode
    env.reset(1)

    # data init
    # Test in environment of data
    print('data env', env.queries)
    print('env current queue size', env.queries[env.current_query].qsize())
    agent = Agent(env.queries, env.current_query)

    # current query
    print('agent current queue size', agent.current_query)

    # agent taking the action alone
    agent.next_snippet()
    # test of synchronization
    print('sync=?', agent.queries[agent.current_query].qsize(), env.queries[env.current_query].qsize())

    # getting the queries of episode
    queries = env.get_queries()
    print('queries', queries)

    # changing query in agent and checking size of queue
    # NOTE, is not synchronized with environment in this form
    agent.change_query(queries[0])
    print('agent change queue', agent.queries[agent.current_query].qsize())
    env.current_query = agent.current_query

    # using the enviroment in order to get reward, next_state and done
    # passing functions as argument
    # test current queue size
    print('env change queue', env.queries[env.current_query].qsize())

    reward, state, done = env.step(agent.next_snippet, agent.change_db)

    print('env queue after step', env.queries[env.current_query].qsize())
    print('agent same queue?', agent.queries[agent.current_query].qsize())

if __name__ == "__main__":
    main()
