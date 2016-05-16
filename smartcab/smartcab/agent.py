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
        self.gamma = 0.8
        self.learning_time = 1
        #Create a matrix represent the initial values for Q(s,a), there are 48 states and 4 actions
        self.Q = np.zeros([2,4,4,4,4])

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.Q = np.zeros([2,4,4,4,4])
        
    def findNeighbouringStates(self, location, heading):#Give current state, find next possible states
        states = []
        #The 1st state
        loc1 = location #stay still
        state = {'location': loc1, 'heading': heading}
        states.append(state)
        #The 2nd state
        loc2 = ((location[0] + heading[0] - self.env.bounds[0]) % (self.env.bounds[2] - self.env.bounds[0] + 1) + self.env.bounds[0],
                            (location[1] + heading[1] - self.env.bounds[1]) % (self.env.bounds[3] - self.env.bounds[1] + 1) + self.env.bounds[1])  #move forward directly
        state = {'location': loc2, 'heading': heading}
        states.append(state)
        #The 3rd state
        heading3 = (heading[1], -heading[0])
        loc3 = ((location[0] + heading3[0] - self.env.bounds[0]) % (self.env.bounds[2] - self.env.bounds[0] + 1) + self.env.bounds[0],
                            (location[1] + heading3[1] - self.env.bounds[1]) % (self.env.bounds[3] - self.env.bounds[1] + 1) + self.env.bounds[1])  #turn left
        state = {'location': loc3, 'heading': heading3}
        states.append(state)
        #The 4th state
        heading4 = (-heading[1], heading[0])
        loc4 = ((location[0] + heading4[0] - self.env.bounds[0]) % (self.env.bounds[2] - self.env.bounds[0] + 1) + self.env.bounds[0],
                            (location[1] + heading4[1] - self.env.bounds[1]) % (self.env.bounds[3] - self.env.bounds[1] + 1) + self.env.bounds[1])  #turn right
        state = {'location': loc4, 'heading': heading4}
        states.append(state)
        return states
    
    def findNextState(self,location,heading):
        light = 'green' if (self.env.intersections[location].state and heading[1] != 0) or ((not self.env.intersections[location].state) and heading[0] != 0) else 'red'

        # Populate oncoming, left, right
        oncoming = None
        left = None
        right = None
        agent = self.env.primary_agent
        for other_agent, other_state in self.env.agent_states.iteritems():
            if agent == other_agent or location != other_state['location'] or (heading[0] == other_state['heading'][0] and heading[1] == other_state['heading'][1]):
                continue
            other_heading = other_agent.get_next_waypoint()
            if (heading[0] * other_state['heading'][0] + heading[1] * other_state['heading'][1]) == -1:
                if oncoming != 'left':  # we don't want to override oncoming == 'left'
                    oncoming = other_heading
            elif (heading[1] == other_state['heading'][0] and -heading[0] == other_state['heading'][1]):
                if right != 'forward' and right != 'left':  # we don't want to override right == 'forward or 'left'
                    right = other_heading
            else:
                if left != 'forward':  # we don't want to override left == 'forward'
                    left = other_heading

        return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}  # TODO: make this a namedtuple
    
    def findNextLocation(self,location, heading, action=None):        
        light = 'green' if (self.env.intersections[location].state and heading[1] != 0) or ((not self.env.intersections[location].state) and heading[0] != 0) else 'red'

        # Move agent if within bounds and obeys traffic rules
        reward = 0  # reward/penalty
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
        return location, heading
    
    def findNeighboringStates(self,location,heading):
        states =[]
        actions = [None, 'forward', 'left', 'right']
        for action in actions:
            [next_location, next_heading] = self.findNextLocation(location,heading,action)
            states.append(self.findNextState(next_location,next_heading))
        return states
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # {'light': light, 'oncoming': oncoming, 'left': left, 'right': right} 
        state = inputs
        self.state = state
        agent_state = self.env.agent_states[self]   
        current_location = agent_state['location']
        current_heading = agent_state['heading']  
        #Current State 
        headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
        light_states = ['red', 'green']
        actions = [None, 'forward', 'left', 'right']
        current_light = inputs['light']
        current_oncoming = inputs['oncoming']
        current_left = inputs['left']
        current_right = inputs['right']
        light_index = light_states.index(current_light)
        oncoming_index = actions.index(current_oncoming)
        left_index = actions.index(current_left)
        right_index = actions.index(current_right)        

        # TODO: Select action according to your policy
        #action = random.choice(actions)  
        best_action_index = self.Q[light_index,oncoming_index,left_index,right_index, :].argmax()
        action = actions[best_action_index] if best_action_index != 0 else random.choice(actions)   
        action_index = actions.index(action)

         
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        #Predict next location and heading
        [next_location,next_heading] = self.findNextLocation(current_location,current_heading,action)
 
        # Predict next state S' according to current state S and action
        next_state = self.findNextState(next_location,next_heading)
               
        

        
        #Find possible states of next state S'
        possible_states = self.findNeighboringStates(next_location,next_heading)

        #Update Q values
        max_Q = 0
        for item in possible_states:#Find max values of Q(S', a')
            light = item['light']
            oncoming = item['oncoming']
            left = item['left']
            right = item['right']
            light_temp_index = light_states.index(light)
            oncoming_temp_index = actions.index(oncoming)
            left_temp_index = actions.index(left)
            right_temp_index = actions.index(right)
            temp = self.Q[light_temp_index,oncoming_temp_index,left_temp_index,right_temp_index, :].max()
            if temp > max_Q:
                max_Q = temp

        alpha = 0.8
        self.Q[light_index,oncoming_index,left_index,right_index, action_index] = (1 - alpha) * self.Q[light_index,oncoming_index,left_index,right_index, action_index] + alpha * (reward + self.gamma * max_Q)
                

               
        #print self.Q
        print t
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
