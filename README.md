# dospordos
Reinforcement learning using the technique of TD-Gammon, for highly qualified migrations localizations in Google-Bing-Duckduckgo-Citeseerx

## Running Examples 

`python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/train_db/ DQN 0 `

This means that you'll be using:
 - the train database (data source) with path ~/project/dospordos/DATA/db_v1_ns/train_db/ 
 - DQN model instead of a DDQN
  - 0 is for using Regex or NE and is_test is for using a train data source or test data source. 
 
 
 `python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/test_db/ DQN 0 -is_test=1`
 
This means that you'll be using:
 - the test database db_v1_ns (data source)

Depending on the parameters given the data will be stored as DQN_0_db_v1_ns* in the
DATA directory

There are other optional parameters to run a specific range of users
which are -initial_range and -final_range

`python training_script.py /users/urbinagonzalez/project/dospordos/DATA/db_v1_ns/test_db/ DQN 0 -is_test=1 -final_range=45`

- This will run the users up to the user 45

