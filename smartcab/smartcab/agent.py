import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.fh = open('log.txt','w') #Record intermediate values for debugging
        self.count = 0 #the count of penalty
        self.rewards =0 #total reward
        self.msg = None #record basic information
        self.deadline = 0
        self.step = 0
        #Create a matrix represent the initial values for Q(s,a), there are 8 states and 4 actions
        self.prevState = None
        self.prevAction = None
        self.prevReward = None
        self.Q = np.zeros([2,4,4]) 


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required 
        #Record some parameters, added by richard 
        self.fh.write(str(self.msg) + '\n')        
        #self.fh.write(str(self.Q) + '\n')
        self.count = 0
        self.rewards = 0
        #self.step = 0
        self.deadline = 0
        self.msg = None #record basic information
        
      
    def findNeighboringStates(self):#Give current state, find next possible states
        states = []
        lights = ['red', 'green']
        actions = [None,'forward', 'left', 'right'] 
        for light in lights:
            for action in actions:
                state = {'light':light,'next_waypoint':action}
                states.append(state)
        return states    
         

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state        
        agent_state = self.env.agent_states[self] 
        light = inputs['light'] 
             
        state = {'light':light,'next_waypoint':self.next_waypoint}
        self.state = state
        
        #Previous State 
        #headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
        #distances =[(0, 0),(0, 1),(1, 0),(0, -1),(-1, 0),(1, 1),(1, -1),(-1, 1),(-1, -1)] #The signs of distance
        lights = ['red', 'green']
        actions = [None,'forward', 'left', 'right'] 
        # TODO: Select action according to your policy
        epsilon = 100.0/(self.step+100)#Decrease epsilon 
        if random.uniform(0,1)<epsilon: #Exploration/Exploitation Trade-off
            action = random.choice(actions)
        else:
            light_index = lights.index(light)
            nextway_index = actions.index(self.next_waypoint)
            Q = self.Q[light_index,nextway_index,:]
            best_action_index = Q.argmax() #Find the action index which has the largest value in Q
            action = actions[best_action_index]
            
        #action = self.next_waypoint
  
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        if self.prevState is not None:
            prev_light = self.prevState['light']
            prev_nextway = self.prevState['next_waypoint']
            light_index = lights.index(prev_light)
            nextway_index = actions.index(prev_nextway)
            action_index = actions.index(self.prevAction)
            # TODO: Learn policy based on state, action, reward
            #Predict next state S' according to current state S and action
            #next_state = {'light':light,'next_waypoint':action}
            #Find possible states of next state S'
            possible_states = self.findNeighboringStates()
            #Update Q values
            max_Q = 0
            for item in possible_states:#Find max values of Q(S', a')
                temp_light = item['light']
                temp_nextway = item['next_waypoint']
                light_temp_index = lights.index(temp_light)
                nextway_temp_index = actions.index(temp_nextway)  
                temp = self.Q[light_temp_index,nextway_temp_index,:].max()
                if temp > max_Q:
                    max_Q = temp
            alpha = 0.8
            gamma = 0.2
            self.Q[light_index, nextway_index, action_index] = (1 - alpha) * self.Q[light_index, nextway_index, action_index] + alpha * (self.prevReward + gamma * max_Q)
                

        #Record some values        
        self.rewards += reward #total rewards
        self.step += 1 #Global time
        self.deadline = deadline
        if reward == -1:
            self.count += 1 #count of penalties
        
        #Output debug information
        self.msg = {'Step': self.step,'Wrong Actions':self.count,'Total Rewards':self.rewards,'Deadline':self.deadline,'Done':self.env.done}
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print self.epsilon
        #print 'Current State', state
        #print 'Next State', next_state
        self.prevState = state #store this state
        self.prevAction = action
        self.prevReward = reward
 

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.2)  # reduce update_delay to speed up simulation
    sim.run(n_trials=101)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
