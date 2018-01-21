# -*- coding:utf-8 -*-

from CODE.environment import Environment
from CODE.agent import Agent

# Todo, leer excel de Fernando para calcular accuracy..Reward.  JORGE
"""
ESTADO
Nombre: _extraido del query_
Bio: [(U,A), (U2,A2), ... ,(Un, An)]

6 positions  para  confidence 1cU, 1cA, 2cU, 2cA, 3cU, 3cA     curConf
newConf ==
common, Total  (U),   common, Total(A),  common, Total(U-A)
tf-idf  ..? del snippet

Reward=
Jaccard distance conjunto // overlap
* mejora. Expansión 

Entre más valores penalizar el  Parcel-BD

-NeuroNet y LSTM-Ner

[snippet,base] -> red -> acción

Problems:
* cuando detenerse
* familias semanticas 

** TODO: 
    -Theo confidence score function with spacy
    -Implementar la red neuronal. [keras]
    -Implementar el TD-Gammon

"""


def main():
    env = Environment()
    env.set_path_files('/home/urb/PycharmProjects/dospordos/DATA/train_db/')
    env.set_path_train('/home/urb/PycharmProjects/dospordos/DATA/db_fer/train.json')

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
