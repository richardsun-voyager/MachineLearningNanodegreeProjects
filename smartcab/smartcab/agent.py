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
        #Create a matrix represent the initial values for Q(s,a), there are 72 states and 4 actions
        self.Q = np.zeros([2,9,4,4])   


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
        
      
    def findNeighboringStates(self, current_state,t):#Give current state, find next possible states
        states = []
        actions = [None,'forward', 'left', 'right'] 
        for action in actions:
            states.append(self.findNextState(current_state,action,t))
        return states
    
    def findNextState(self,current_state,action,t):
        destination = self.env.agent_states[self]['destination'] 
        light = current_state['light']
        distance = current_state['distance']
        location = (distance[0]+destination[0],distance[1]+destination[1])
        heading = current_state['heading']
        move_okay = True
        if action == 'forward':
            if light != 'green':
                move_okay = False
        elif action == 'left':
            if light == 'green':
                heading = (heading[1], -heading[0])
            else:
                move_okay = False
        elif action == 'right':
            heading = (-heading[1], heading[0])

        if action is not None:
            if move_okay:
                location = ((location[0] + heading[0] - self.env.bounds[0]) % (self.env.bounds[2] - self.env.bounds[0] + 1) + self.env.bounds[0],
                            (location[1] + heading[1] - self.env.bounds[1]) % (self.env.bounds[3] - self.env.bounds[1] + 1) + self.env.bounds[1])  # wrap-around
                
        traffic_light = self.env.intersections[location]#Current traffic light
        traffic_light.update(t+1)#Next moment's traffic light
        light = 'green' if (traffic_light.state and heading[1] != 0) or ((not traffic_light.state) and heading[0] != 0) else 'red'
        distance = (location[0] - destination[0] , location[1] - destination[1] )
        return {'light': light, 'distance': distance, 'heading':heading}  # TODO: make this a namedtuple
    
    
         

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state        
        agent_state = self.env.agent_states[self] 
        light = inputs['light'] 
        location = agent_state['location']
        heading = agent_state['heading']  
        destination = agent_state['destination']
        distance = (location[0] - destination[0] , location[1] - destination[1] ) #difference between current location and destination       
        state = {'light':light,'distance':distance,'heading':heading}
        self.state = {'light':light,'distance':np.sign(distance),'heading':heading}
        
        #Current State 
        headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
        distances =[(0, 0),(0, 1),(1, 0),(0, -1),(-1, 0),(1, 1),(1, -1),(-1, 1),(-1, -1)] #The signs of distance
        light_states = ['red', 'green']
        actions = [None,'forward', 'left', 'right'] 

        distance_index = distances.index(tuple(np.sign(distance)))
        light_index = light_states.index(light)
        heading_index = headings.index(heading)
    

        # TODO: Select action according to your policy
        Q = self.Q[light_index,distance_index,heading_index, :]
        best_action_index = Q.argmax() #Find the action index which has the largest value in Q
        action = actions[best_action_index]
        epsilon = 100.0/(self.step+100)#Decrease epsilon  
        if random.uniform(0,1)<epsilon: #Exploration/Exploitation Trade-off
            action = random.choice(actions)                   
        action_index = actions.index(action)
        #action = self.next_waypoint
        #action = random.choice(actions)
         
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Predict next state S' according to current state S and action
        next_state = self.findNextState(state,action,t)
        
        #Find possible states of next state S'
        possible_states = self.findNeighboringStates(next_state,t+1)

        #Update Q values
        max_Q = 0
        for item in possible_states:#Find max values of Q(S', a')
            temp_light = item['light']
            temp_distance = item['distance']
            temp_heading = item['heading']
            distance_temp_index = distances.index(tuple(np.sign(temp_distance)))
            light_temp_index = light_states.index(temp_light)
            heading_temp_index = headings.index(temp_heading)
            temp = self.Q[light_temp_index,distance_temp_index, 
                          heading_temp_index, :].max()
            if temp > max_Q:
                max_Q = temp

        alpha = 2000.0/(self.step+2000)
        gamma = 0.2
        self.Q[light_index,distance_index,heading_index, action_index] = (1 - alpha) * 
        self.Q[light_index,distance_index,heading_index, action_index] + alpha * (reward + gamma * max_Q)
                

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
