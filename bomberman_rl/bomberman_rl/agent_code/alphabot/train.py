from collections import namedtuple, deque
import numpy as np

import pickle
from typing import List

from numpy.lib.function_base import append

import events as e
from .callbacks import state_to_features,ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Hyperparameters For Q-learning
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9


# Events

#AUILLARY REWARDS
MOVED_GOOD = 'MOVED_GOOD'
MOVED_BAD = 'MOVED_BAD'
WAIT_GOOD = 'WAIT_GOOD'
WAIT_BAD = 'WAIT_BAD'
#CRATE_DESTROYED_GOOD = 'CRATE_DESTROYED_GOOD'
#CRATE_DESTROYED_BAD = 'CRATE_DESTROYED_BAD'
BOMB_PLACED_GOOD = 'BOMB_PLACED_GOOD'
BOMB_PLACED_BAD = 'BOMB_PLACED_BAD'
#UNDER_ATTACK_FROM_BOMB = 'UNDER_ATTACK_FROM_BOMB' #-0.1
#ESCAPED_FROM_BOMB ='ESCAPED_FROM_BOMB'          #+0.4
COIN_NEAR='COIN_NEAR'
COIN_FAR='COIN_FAR'
ESCAPE_SUCCESS='ESCAPE_SUCCESS'
ESCAPE_FAIL='ESCAPE_FAIL'




def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s') # should be (s, a, s', r)
     
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

   
    
    # Get state representations for Q-learning
    old_game_feature = state_to_features(self,old_game_state)
    new_game_feature = state_to_features(self,new_game_state)
    

     # Idea: Add your own events to hand out rewards
     # f_1(0 / 1 / 2 / 3 / 4) :: COLLECT COINS (a)Coin is present nearest to you b) NOT of (a)
     #rewards for f_1  #move towards nearest coin without getting killed :: direction up, right,down,left,None :
    f_1=old_game_feature[0]
    f_2=old_game_feature[1]
    f_3=old_game_feature[2]
    f_4=old_game_feature[3]
    f_5=old_game_feature[4]
    f_6=old_game_feature[5]
    f_7=old_game_feature[6]
    #events for Coin Collection Mode
    if(f_1 in (0,1,2,3)):
        if(f_2 in (1,2,3,4)):
            if((f_2==1 and self_action=='UP')or (f_2==2 and self_action=='RIGHT')or (f_2==3 and self_action=='DOWN') or (f_2==4 and self_action=='LEFT')):
                events.append(ESCAPE_SUCCESS)
            else:
                events.append(ESCAPE_FAIL)
        
        else :
            if((f_1==0 and self_action=='UP') or(f_1==1 and self_action=='RIGHT')or(f_1==2 and self_action=='DOWN') or (f_1==3 and self_action=='LEFT') ):
                
                events.append(COIN_NEAR)
            
            elif ((f_1==0 and self_action=='WAIT' and f_3==1)or(f_1==1 and self_action=='WAIT' and f_4==1) or (f_1==2 and self_action=='WAIT' and f_5==1)or (f_1==3 and self_action=='WAIT' and f_6==1) ):
                events.append(COIN_NEAR)
            
            elif(f_2 in (1,2,3,4)):
                if((f_2==1 and self_action=='UP')or (f_2==2 and self_action=='RIGHT')or (f_2==3 and self_action=='DOWN') or (f_2==4 and self_action=='LEFT')):
                    events.append(ESCAPE_SUCCESS)
            else:
                events.append(COIN_FAR)  
    #escape mode
    if(f_2 in (1,2,3,4)):
            if((f_2==1 and self_action=='UP')or (f_2==2 and self_action=='RIGHT')or (f_2==3 and self_action=='DOWN') or (f_2==4 and self_action=='LEFT')):
                
                events.append(ESCAPE_SUCCESS)
            else:
                events.append(ESCAPE_FAIL)
    #IF THERE IS NO DANGER, ATTACK
    if(f_7 in (0,1)):
        
        if(f_7==1 and self_action=='BOMB'):
            events.append(BOMB_PLACED_GOOD)
        elif(f_7==0 and self_action=='BOMB'):
            events.append(BOMB_PLACED_BAD)
        
        #NO COIN MODE AND NOT UNDER DANGER AND CANNOT PLACE BOMB
        if(f_1==4 and f_2==0 and f_7==0):
            
            if((f_3==0 and self_action=='UP') or (f_4==0 and self_action=='RIGHT') or (f_5==0 and self_action=='DOWN') or (f_3==6 and self_action=='LEFT')):
                
                events.append(MOVED_GOOD)
            else:
                events.append(MOVED_BAD)
        
    # Train the Q-learning agent
    reward = reward_from_events(self, events)
    
    train_q_learning(self, old_game_feature, self_action, new_game_feature, reward)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(self,old_game_state), self_action, state_to_features(self,new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(self,last_game_state), last_action, None, reward_from_events(self, events)))

    rounds= last_game_state['round']
    # Store the model
    if(rounds%1000==0):
        with open("alphabot_qtable.pkl", "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        #Max points I can Have in an episode( classic setting : 5*3 +9 =24)
        e.COIN_COLLECTED: 1,  #Game Reward
        e.KILLED_OPPONENT: 5, #Game Reward
        
        #Auxillary Rewards
        e.GOT_KILLED:-5,  #equal karma for getting killed and suicide
        e.KILLED_SELF:-5, #don't want to suicide even though opponent cannot catch me up in points (Lets maintain spirit of the game!)
        e.INVALID_ACTION:-5,
        #e.CRATE_DESTROYED:0.1
        MOVED_GOOD: 0.1,
        MOVED_BAD: -0.1,
        #CRATE_DESTROYED_GOOD: 0.1,
        #CRATE_DESTROYED_BAD: -0.1,
        BOMB_PLACED_GOOD: 0.1,
        BOMB_PLACED_BAD: -0.1,
        COIN_NEAR: 0.1,
        COIN_FAR: -0.1,
        ESCAPE_SUCCESS: 0.1,
        ESCAPE_FAIL: -0.1

        
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #print(f"Awarded {reward_sum} for events {', '.join(events)}")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def train_q_learning(self, old_state, action, new_state, reward):
    # Calculate the Q-value update using the Q-learning formula
    # Q(s, a) = Q(s, a) + alpha * [R(s, a) + gamma * max(Q(s', a')) - Q(s, a)]
    old_state_string = ''.join(map(str, old_state))
    new_state_string = ''.join(map(str, new_state))
    
    q_table=self.model
    #old_q_value = q_table[old_state, action]
    #max_next_q_value = np.max(q_table[new_state, :])
    #new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q_value - old_q_value)

    # Update the Q-table
    #q_table[old_state, action] = new_q_value    
    ####################
    old_q_value = q_table.get((old_state_string, action), 0.0)
    max_next_q_value = max(q_table.get((new_state_string, a), 0.0) for a in ACTIONS)
    new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q_value - old_q_value)

    # Update the Q-table
    q_table[(old_state_string, action)] = new_q_value   
