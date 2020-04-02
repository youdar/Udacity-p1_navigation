[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  
Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.


### Getting Started - Installation

The project was developed on Mac OSX, no other OS setup was tested

#### Requirements:
- need to have python 3 on your machine 
  (In the instruction below I will assume python 3.6, please change according to your version)

#### Installation:

For Mac OSX
(did not check on other OS)

1. Code location: [Udacity-p1_navigation](https://github.com/youdar/Udacity-p1_navigation)   
2. Open the link above and click on [Clone or Download] -> [Download zip]   
3. Unzip the downloaded file   
4. If you have pycharm   
   - open the folder with pycharm  
   - in the [perferences] - [Project Interpreter] create a virtual environment   
   - let pycharm install all packages in the `requirements.txt` file   
4. open a terminal and type   
   - `cd unzipped_path`   
     Example: `cd /Users/youval.dar/Downloads/Udacity-p1_navigation-master` 
   - `python3.6 -m venv /unzipped_path/venv`   
     Example: `python3.6 -m venv /Users/youval.dar/Downloads/Udacity-p1_navigation-master/venv`   
   - activate virtual environment:   
     Example: `source /Users/youval.dar/Downloads/Udacity-p1_navigation-master/venv/bin/activate`  
   - upgrade pip: `pip install pip --upgrade`   
   - install project dependencies  
     `pip install -r /Users/youval.dar/Downloads/Udacity-p1_navigation-master/requirements.txt`   
   - Continue to run the code and **when all is done**:   
     to deactivate virtual environment, in the terminal `deactivate`  

#### Running the code:

With your favorite code editor update   
`LOCAL_PATH` in the files `Navigation.ipynb` and `Navigation_tools.py` with the correct path of your local

**Using Pycharm IDE**
If you want to use a tool like pycharm, open `Navigation_tools.py`, which gives you great way to stop in mid process  
and see the values of different variable, go to the bottom of the code and comment/uncomment the section you want to run      
```
if __name__ == '__main__':
    o = BananaGame()
    # model_num=1 : DQN, model_num=2 : DQN with dropout (see model.py)
    # o.uniform_random(3)
    # scores = o.training_model(model_num=2)
    # o.plot_scores(scores)
    scores = o.trained_model(100, model_num=2)
    # o.plot_scores(scores)
```   
  
**Using Jupyter notebook**
If you want tot use the Jupyter notebook, in the command line (the environment we started earlier)   
type `jupyter notebook`  the notebook will open, click on the folders and go to your unzipped_path   
open `Navigation.ipynb` and run    
Note that unlike when running using pycharm, you might get sometimes `broken pipe` error when running 
different learning or playing tasks one after the other. If that happens just restart the notebook using   
the button on the top of the notebook.   

### Project files
- `Navigation_tools.py`: code containing functions used during to development
- `Navigation.ipynb`: Summary code
- `dqn_agent.py`: DQN agent
- `model.py`: Two model, one with dropout and one without
- `checkpoint.pth`: trained model settings for the model without dropout
- `checkpoint_dropout.pth`: trained model settings for the model with dropout layers
- `model.pt` is a copy of `checkpoint.pth`   
- `README.md`: this file
- `Report.md`: Info on the project
