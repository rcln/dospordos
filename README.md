# dospordos
Reinforcement learning using the technique of TD-Gammon, for highly qualified migrations localizations in Google-Bing-Duckduckgo-Citeseerx

### How to run
The script is called `training_script.py` which has the following arguments:
 - "DB", "Path to training directory"
 - "ALG", "Algorithm to execute", default="DQN"
 - "is_RE", "Use of Regular Expression", default="0"
 - "-is_test", "The data is for testing", required=False, default=0
 - "-initial_range", "Initial range of users", required=False
 - "-final_range", help="Final range of users", required=False
 - "-is_db_v2", help="Is the second database", required=False
 
#### Running Examples 

`python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/train_db/ DQN 0 `

This means that you'll be using:
 - the train database (data source) with path ~/project/dospordos/DATA/db_v1_ns/train_db/ 
 - DQN model instead of a DDQN
  - 0 is for using Regex or NE and is_test is for using a train data source or test data source. 
 
 
 `python3 training_script.py ~/project/dospordos/DATA/db_v2_ns/test_db/ DQN 0 -is_test=1 -is_db_v2=1`
 
This means that you'll be using:
 - the test database db_v2_ns (data source)
 - is the second database

Depending on the parameters given the data will be stored as DQN_0_db_v1_ns* in the
DATA directory

There are other optional parameters to run a specific range of users
which are -initial_range and -final_range

`python training_script.py /users/urbinagonzalez/project/dospordos/DATA/db_v1_ns/test_db/ DQN 0 -is_test=1 -final_range=45`

- This will run the users up to the user 45. Is doing list_users[:45]

### Requirements

The data directory should have folders with numbers
 * ~/DATA/train
   * 3
   * 5
   * ...
   * 4904
   * 4905
   * ...

Besides running the build.sh
* python -m spacy download en
* Install keras, tensorflow

####Notes

In the class DQN of DQN_implementation.py you can set the callbacks used
for stopping the network.

`self.callbacks = [agent.EarlyStopByLossVal(value=0.1),
                          agent.EarlyStopping(patience=10)]`

##For testing
Use TESTS/evaluate_test_run script, you should already have all the pkl files you want to average and graph.
This is the format you should follow:

`python3 evaluate_test_run.py -r DQN_0_db_v1_ns_rm.pkl -acc DQN_0_db_v1_ns_acc.pkl -g 1`


For more details:

`python3 evaluate_test_run.py -h`


## Connection to cluster

`ssh -p 60022 urbinagonzalez@tal.lipn.univ-paris13.fr`

in GPU2
The directory for the project is

`~/project/dospordos`

The virtual environment is called `venv-dospordos`

There's a tmux session ready. To connect 

`tmux a -t base`
