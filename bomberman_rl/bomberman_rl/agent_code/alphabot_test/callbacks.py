import os
import pickle
import random
from types import BuiltinFunctionType

import numpy as np
from random import shuffle
import time

from .featureExtractor import *


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.


    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("alphabot_qtable.pkl"):
        self.logger.info("Setting up model from scratch.")
        self.model = {}
        
    else:
        self.logger.info("Loading model from saved state.")
        with open("alphabot_qtable.pkl", "rb") as file:
            self.model = pickle.load(file)
            self.logger.info(f"Qtable{self.model}")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #  Exploration vs exploitation
    epsilon = .1
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 20% wait. 0% bomb.
        return np.random.choice(ACTIONS, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5])

    self.logger.debug("Querying model for action.")
    # Log the start time
    start_time = time.time()
    
    features = state_to_features(self,game_state)
    features_string = ''.join(map(str, features))
    #print('hi',features_string)
    self.logger.debug(f"state{features_string}")
    
    q_values = {action: self.model.get((features_string, action), 0.0) for action in ACTIONS}
    action_taken= max(q_values, key=q_values.get)
    self.logger.debug(f"state{action_taken}")
    
    # Log the end time
    end_time = time.time()
    # Calculate and log the elapsed time
    elapsed_time = end_time - start_time
    self.logger.info(f"time taken to act is {elapsed_time}")
    
    return max(q_values, key=q_values.get)
   
    


def state_to_features(self,game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    #Logic for feature Engineering! Perhaps the most key aspect for a good performance!
    
    # Gather information about the game state
    
    if game_state is None:
        return None
    
    arena = game_state['field']
    coins = game_state['coins']
    _, score, isbomb, (x, y) = game_state['self']
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    #print('bombs',bombs)
    
    free_space = arena == 0
    others = [xy for (n, s, b, xy) in game_state['others']]
    for o in others:
        free_space[o] = False
    
    f_1= get_nearest_coin_direction_feature(self,arena,coins,(x,y),others)
    self.logger.info(f'Feature1 : {f_1}')
    
    f_2 =escapeUnderDanger(self,(x,y),others,arena,bombs,explosion_map)
    self.logger.info(f'Feature2 : {f_2}')
    
    f_3= neighbourUp(self,(x,y),arena,bombs, explosion_map)
    self.logger.info(f'Feature3 : {f_3}')
    
    f_4= neighbourRight(self,(x,y),arena,bombs, explosion_map)
    self.logger.info(f'Feature4 : {f_4}')
    
    f_5= neighbourDown(self,(x,y),arena,bombs, explosion_map)
    self.logger.info(f'Feature5 : {f_5}')
    
    f_6= neighbourLeft(self,(x,y),arena,bombs, explosion_map)
    self.logger.info(f'Feature6 : {f_6}')
    
    f_7=attackStrategy(self,f_1,f_2,f_3,f_4,f_5,f_6,(x,y),isbomb,others,arena,bombs,explosion_map)
    self.logger.info(f'Feature7 : {f_7}')
    
    return [f_1,f_2,f_3,f_4,f_5,f_6,f_7]
   
# what

'''
def get_nearest_coin_direction_feature(self,game_state: dict) -> int:    
    if game_state is None:
        self.logger.debug("Entering None Game States")
        return None
    # Gather information about the game state
    arena = game_state['field']
    coins = game_state['coins']
    _, score, bombs_left, (x, y) = game_state['self']
    
    free_space = arena == 0
    others = [xy for (n, s, b, xy) in game_state['others']]
    for o in others:
        free_space[o] = False
    coin = coinNearestToAgent(self,coins,(x, y),others)
    print(f'coin {coin}')
    d = look_for_targets(free_space,(x, y), coin, self.logger)
    
    if d == (x, y - 1): return 0 #('UP')
    if d == (x + 1, y): return 1 #'(RIGHT')
    if d == (x, y + 1): return 2 #('DOWN')
    if d == (x - 1, y): return 3 #'(LEFT') 
    if (d is None ): return 4 #'(No Coin)' 
    #What about WAIT? IF YOU ENTER DANGER TILE


    #COPIED FROM coin_collector_agent 
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of the closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards the closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
        
def coinNearestToAgent(self,coins,my_agent,others):
    #return (x,y) of coin else None
    for coin in coins:
        # Find distance of coin from enemies from current position to all targets, track closest
        print(f'coin {coin}')
        d1 = np.sum(np.abs(np.subtract(coin, others)), axis=1).min()
        d2 = np.sum(np.abs(np.subtract(coin, [my_agent])), axis=1).min()
        print('d1 and d2',d1,d2)
        self.logger.info(f'd1 : {d1}, d2 : {d2}')
        if(d2<=d1):
            self.logger.info(f'coin :{coin}')
            return [coin]
    return []

def isBombPlacedSafe(self):
    pass

def isunderThreat(self,my_agent): 
    # if under bomb radius and if explosion map==1

    pass



                 Assessment                     Neighbours                         Attack
Coin Collection  |  Under Threat   |  TOP     |   RIGHT | DOWN | LEFT | Safe to Place Bomb
    0            |(IN CURRENT POS)    WALL        |          |        |    0-Attack      
    1            |                 |  ENEMY       |                        
    2            | 0/1                crate
    3            |                 |  THREAT
    4 (NO COIN)  |                 |  Free
PLACE BOMB/OTHER MOVE
(WHAT ABOUT WAIT)|
YES WAIT, IF
NEXT STEP KILLS
YOU


    
                 |   
 1)place bomb only if killing enemy/destroying crate possible and coin not available in the nearest spot                

2) iF I AM UNDER ATTACK, SHOULD I JUST ESCAPE OR IF THERE IS A POSSIBILITY TO PLACE BOMB AND ESCAPE

NO :: WHAT ABOUT SEEING  NEAREST NEIGHBOURS (AND REPRESENT OBSTACLE/NOT)

HIGH LEVEL DESIGN OF AGENT
ORDER OF PRECEDENCE
1)IF AGENT IS UNDER THREAT, (NEVER WAIT (handle in reward)). : GOAL : MOVE TO SAFE TILE(THIS WILL REQUIRE INFORMATION ON NEIGHBORS IN ALL DIRECTION)
2) IF AGENT IN SAFE TILE AND SAFE TO PLACE BOMB AND THERE IS NO NEAREST COIN AND IT IS USEFUL TO PLACE BOMB, THEN ONLY DO IT
2.1) 'USEFUL' DEFINITION : 1) ATTACK ENEMY 2) DESTROY CRATE IF COINS ARE LEFT
3)COLLECT COINS IF IT IS NEAREST TO AGENT (THE DEFINITION OF NEAREST : FOR NOW JUST CONSIDER DISTANCE IN EMPTY GRID)

Remark: If NOT under threat AND NEIGHBOURS SAFE and NOT useful to place bomb, MOVE SOMEWHERE
GIVE APPROPRIATE REWARD TO ENSURE THIS
'''

'''
max_next_q_value = max(q_table.get_q_value(new_state, a) for a in possible_actions)

'''


'''
Safe to place bomb: if agent is not at a dead end and there is no nearest coin to to collect and crate can be destroyed from current position
'''

'''
HOW TO ESCAPE BOMB?

'''

'''
If my bomb active and i am in safe tile wait
'''

'''
 || coin collection(0,1,2,3,4)  || escape_ifunder_threat(0,1,2,3,4) || isattack(bomb) ||Neighbours(bool) || Navigate
'''

