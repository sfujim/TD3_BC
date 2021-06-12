# A Minimalist Approach to Offline Reinforcement Learning

TD3+BC is a simple approach to offline RL where only two changes are made to TD3: (1) a weighted behavior cloning loss is added to the policy update and (2) the states are normalized. Unlike competing methods there are no changes to architecture or underlying hyperparameters. 

### Usage
The paper results can be reproduced by running:
```
./run_experiments.sh
```
