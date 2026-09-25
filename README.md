# Morph RL
This repository contains the training and evaluation code for the modeling follow up of *Foot placement control underlies stable locomotion across species", De Comite and Seethapathi, PNAS, 2025.

This is still a work in progress...


## Installation
The file `requirements.txt` contains the list of packages required for running most of the code. Note that this code uses jax (and the default settings assumes you can run at least 8092 environments in parallel, hence have access to at least as many cuda cores).

## Running the code 

There are three different steps in the code. The first one consists in generating the different morphologies that will be used for training, the second one is the training itself, and the last one is the evaluation of the rollouts using the diverse foot placement metrics. 

To run the training code (which is what is being worked on right now) with the default parameters. It will automatically log the results in the *run/<run_name>* folder. 

```
python train.py <run_name>
```


# Todos 

Update the environment to allow for different type of velocity mode
- [x] We want a way to manually select the speed for the evaluation rollouts

Add the following features to the analysis pipeline 
- [ ] Proper contact detection (both location and timing)
- [ ] Speed step length and duration analysis
- [ ] Characterization of the gait patterns
- [x] Generate rollouts at different speeds 
- [ ] Update the foot placement control code