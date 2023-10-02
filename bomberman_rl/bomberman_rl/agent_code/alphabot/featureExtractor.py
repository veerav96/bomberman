import os
import pickle
import random
from types import BuiltinFunctionType

import numpy as np
from random import shuffle
import time

def get_nearest_coin_direction_feature(self,arena,coins,my_agent,others) -> int:    
    
    (x,y)=my_agent
    free_space = arena == 0
    
    for o in others:
        free_space[o] = False
    coin = coinNearestToAgent(self,coins,(x, y),others)
    
    d = look_for_targets(free_space,(x, y), coin, self.logger)
    
    if d == (x, y - 1): return 0 #('UP')
    if d == (x + 1, y): return 1 #'(RIGHT')
    if d == (x, y + 1): return 2 #('DOWN')
    if d == (x - 1, y): return 3 #'(LEFT') 
    if (d is None ): return 4 #'(No Nearest Coin)' 


    #COPIED this function FROM coin_collector_agent (BFS)
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
    if d != 0:
        # No suitable target found, return None
        return None
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
        d1=100 # default high value(to handle no opponents)
        if(others):
            d1 = np.sum(np.abs(np.subtract(coin, others)), axis=1).min()
        
        d2 = np.sum(np.abs(np.subtract(coin, [my_agent])), axis=1).min()
        
        self.logger.info(f'd1 : {d1}, d2 : {d2}')
        if(d2<=d1):
            self.logger.info(f'coin :{coin}')
            return [coin]
    return []


#1 ==>ATTACK , 0 OTHERS
def attackStrategy(self,f_1,f_2,my_agent,isbomb,others,arena,bombs,explosion_map):
    if(f_1==4 and f_2==0 ):
        bombs.append((my_agent,3))
        if(isbomb):
            f_7= escapeUnderDanger(self,my_agent,others,arena,bombs,explosion_map)
            if(f_7!=5): #escape possible, so throw bomb
                return 1
    
    return 0

def isunderDanger(self,my_agent,bombs,arena): 
    #if agent can die if it stays in current tile #No need of explosion map
    width,height=arena.shape[0],arena.shape[1]
    # if under bomb radius and if explosion map==1
    #list of tiles under threat
    danger_tiles=[]
    for(xb,yb),t in bombs:
        for(i,j) in [(xb+h,yb) for h in range(0,4)] :
            if (0 <= i < width) and (0 <= j < height):
                if arena[i,j]!=-1:
                    danger_tiles.append((i,j))
                    if(my_agent in danger_tiles):
                        return True
                else:
                    break
    for(xb,yb),t in bombs:
        for(i,j) in [(xb-h,yb) for h in range(1,4)] :
            if (0 <= i < width) and (0 <= j < height):
                if arena[i,j]!=-1:
                    danger_tiles.append((i,j))
                    if(my_agent in danger_tiles):
                        return True
                else:
                    break
    for(xb,yb),t in bombs:
        for(i,j) in [(xb,yb+h) for h in range(1,4)] :
            if (0 <= i < width) and (0 <= j < height):
                if arena[i,j]!=-1:
                    danger_tiles.append((i,j))
                    if(my_agent in danger_tiles):
                        return True
                else:
                    break
    for(xb,yb),t in bombs:
        for(i,j) in [(xb,yb-h) for h in range(1,4)] :
            if (0 <= i < width) and (0 <= j < height):
                if arena[i,j]!=-1:
                    danger_tiles.append((i,j))
                    if(my_agent in danger_tiles):
                        return True
                else:
                    break
    return False

    #0 ==> NO DANGER #1 UP #2 RIGHT #3 DOWN #4 LEFT #5 NO ESCAPE POSSIBLE
    #use explosion map=1 not free and bombs with t=1 not free(not implemented), target nearest free tile
def escapeUnderDanger(self,my_agent,others,arena,bombs,explosion_map):
    if not isunderDanger(self,my_agent,bombs,arena):
        return 0
    (x,y)=my_agent
    free_space = arena == 0
    
    for o in others:
        free_space[o] = False
    #if Neighbour has just exploded in previous step, DO NOT STEP INTO IT
    if explosion_map[x,y-1]==1:
        free_space[x,y-1]=False
    if explosion_map[x+1,y]==1:
        free_space[x+1,y]=False
    if explosion_map[x-1,y]==1:
        free_space[x-1,y]=False
    if explosion_map[x-1,y]==1:
        free_space[x-1,y]=False
    #FIND LIST OF TARGET ESCAPE PLACES(FREE TILES AND SAFE TILES(NO BOMB EFFECT)) 
    targets=np.copy(free_space) 
    for (xb,yb),t in bombs:
        
        for(i,j) in [(xb+h,yb) for h in range(0,4)] :
            if (0 <= i < free_space.shape[0]) and (0 <= j < free_space.shape[1]):
                if arena[i,j]!=-1:
                    
                    targets[i,j]=False     
                else:
                    break
        for(i,j) in [(xb-h,yb) for h in range(1,4)] :
            if (0 <= i < free_space.shape[0]) and (0 <= j < free_space.shape[1]):
                if arena[i,j]!=-1:
                    targets[i,j]=False     
                else:
                    break
        for(i,j) in [(xb,yb+h) for h in range(1,4)] :
            if (0 <= i < free_space.shape[0]) and (0 <= j < free_space.shape[1]):
                if arena[i,j]!=-1:
                    targets[i,j]=False     
                else:
                    break
        for(i,j) in [(xb,yb-h) for h in range(1,4)] :
            if (0 <= i < free_space.shape[0]) and (0 <= j < free_space.shape[1]):
                if arena[i,j]!=-1:
                    targets[i,j]=False     
                else:
                    break      
                
    
    targets = [(i, j) for i in range(free_space.shape[0]) for j in range(free_space.shape[1]) if targets[i, j]]
    
    d= look_for_targets(free_space, (x,y), targets, logger=None)
    
    if d == (x, y - 1): return 1 #('UP')
    if d == (x + 1, y): return 2 #'(RIGHT')
    if d == (x, y + 1): return 3 #('DOWN')
    if d == (x - 1, y): return 4 #'(LEFT') 
    if d is None : return 5 #'(No Escape Possible')
    
#0 => FREE, 1==>OCCUPIED/DANGER FROM possible BOMB EXPLOSION
def neighbourUp(self,my_agent,arena,bombs, explosion_map):
    (x,y) =my_agent
    if 0<=x<arena.shape[0] and 0<=y-1<arena.shape[1]:
        bombs_t1 = [item for item in bombs if item[1] == 0] # filter bombs which will explode in next step
        dangerbool = isunderDanger(self,(x,y-1),bombs_t1,arena)
        #print('dangerbool',dangerbool)
        if arena[x][y-1]==0 and explosion_map[x][y-1]==0 and not dangerbool :
            return 0
    
    return 1

    
#0 => FREE, 1==>OCCUPIED/DANGER FROM possible BOMB EXPLOSION
def neighbourRight(self,my_agent,arena,bombs, explosion_map):
    (x,y) =my_agent
    if 0<=x+1<arena.shape[0] and 0<=y<arena.shape[1]:
        bombs_t1 = [item for item in bombs if item[1] == 0] # filter bombs which will explode in next step
        dangerbool = isunderDanger(self,(x+1,y),bombs_t1,arena)
        #print('dangerbool',dangerbool)
        if arena[x+1][y]==0 and explosion_map[x+1][y]==0 and not dangerbool:
            return 0
    
    return 1

#0 => FREE, 1==>OCCUPIED/DANGER FROM possible BOMB EXPLOSION
def neighbourDown(self,my_agent,arena,bombs, explosion_map):
    (x,y) =my_agent
    if 0<=x<arena.shape[0] and 0<=y+1<arena.shape[1]:
        bombs_t1 = [item for item in bombs if item[1] == 0] # filter bombs which will explode in next step
        dangerbool = isunderDanger(self,(x,y+1),bombs_t1,arena)
        #print('dangerbool',dangerbool)
        if arena[x][y+1]==0 and explosion_map[x][y+1]==0 and not dangerbool:
            return 0
    
    return 1

#0 => FREE, 1==>OCCUPIED/DANGER FROM possible BOMB EXPLOSION
def neighbourLeft(self,my_agent,arena,bombs, explosion_map):
    (x,y) =my_agent
    if 0<=x-1<arena.shape[0] and 0<=y<arena.shape[1]:
        bombs_t1 = [item for item in bombs if item[1] == 0] # filter bombs which will explode in next step
        dangerbool = isunderDanger(self,(x-1,y),bombs_t1,arena)
        #print('dangerbool',dangerbool)
        if arena[x-1][y]==0 and explosion_map[x-1][y]==0 and not dangerbool:
            return 0
    
    return 1



'''                  Assessment                     Neighbours                         Attack
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

'''      Coin collect carefully                 Defence            
 || coin collection(0,1,2,3,4)  || escape_ifunder_danger(0,1,2,3,4) || attackStrategey(bomb) ||Neighbours(bool) ||

 keep neighbors for training invalid actions and not to enter explosive area from safe tile

 Attack Strategy
 if safe to place bomb(i.e not under threat and cant kill myself in a dead end
):
    1)attack enemy if radius
    2)destroy crate in current move or move to

    #if neighbour position has more potential of killing enemy/crates

'''

