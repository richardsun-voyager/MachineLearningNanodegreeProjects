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
        self.learning_time = 300
        #Create a matrix represent the initial values for Q(s,a), there are 48 states and 4 actions
        self.Q = np.zeros([48,4,4])

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.Q = np.zeros([48,4,4])
        
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

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = self.env.agent_states[self]   
        current_location = state['location']
        current_heading = state['heading']  
        self.state = {'location': current_location, 'heading': current_heading} #States contain information of location and heading direction

        # TODO: Select action according to your policy
        actions = [None, 'forward', 'left', 'right']
        if t<self.learning_time:#If the learning process is not over
            action = random.choice(actions)
        else:
            current_state_index = (current_location[0] - 1) * 6 + current_location[1] - 1 #Give the state a label
            current_heading_index = headings.index(current_heading) #index of the heading
            best_action_index = self.Q[current_state_index, current_heading_index, :].argmax()
            action = actions[best_action_index]
            

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if t>self.learning_time:#
            return
        
       
        #Traffic light status
        light = 'green' if (self.env.intersections[current_location].state and current_heading[1] != 0) or ((not self.env.intersections[current_location].state) and current_heading[0] != 0) else 'red'

        # Predict next state S' according to current state S and action
        move_okay = True
        next_heading = current_heading
        next_location = current_location
        if action == 'forward':
            if light != 'green':
                move_okay = False
        elif action == 'left':
            if light == 'green':
                next_heading = (current_heading[1], -current_heading[0])
            else:
                move_okay = False
        elif action == 'right':
            next_heading = (-current_heading[1], current_heading[0])

        if action is not None:
            if move_okay:
                next_location = ((current_location[0] + next_heading[0] - self.env.bounds[0]) % (self.env.bounds[2] - self.env.bounds[0] + 1) + self.env.bounds[0],
                            (current_location[1] + next_heading[1] - self.env.bounds[1]) % (self.env.bounds[3] - self.env.bounds[1] + 1) + self.env.bounds[1])  # wrap-around
        
        #Find possible states of next state S'
        possible_states = self.findNeighbouringStates(next_location, next_heading)
        current_state_index = (current_location[0] - 1) * 6 + current_location[1] - 1 #Give the state a label
        headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
        alpha = 1.0/(t + 1) #We introduce alpha as 1/(t + 1)
        #Update Q values
        max_Q = 0
        for item in possible_states:#Find max values of Q(S', a')
            location = item['location']
            heading = item['heading']
            state_index = (location[0] - 1) * 6 + location[1] - 1
            temp = self.Q[state_index,headings.index(heading),:].max()
            if temp > max_Q:
                max_Q = temp
                
        self.Q[current_state_index, headings.index(current_heading),actions.index(action)] = (1 - alpha) * self.Q[current_state_index, headings.index(current_heading),actions.index(action)] + alpha * (reward + self.gamma * max_Q)
                

               
        print self.Q
        
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
