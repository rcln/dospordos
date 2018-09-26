# dospordos
Reinforcement learning using the technique of TD-Gammon, for highly qualified migrations localizations in Google-Bing-Duckduckgo-Citeseerx

# Running Examples 

`python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/train_db/ DQN 0 `

This means that you'll be using:
 - the train database (data source) with path ~/project/dospordos/DATA/db_v1_ns/train_db/ 
 - DQN model instead of a DDQN
  - 0 is for using Regex or NE and is_test is for using a train data source or test data source. 
 
 
 `python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/test_db/ -is_test=1`
 
 This means that you'll be using:
 - you'll be testing db_v1_ns data source with a test data source 
