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
        #Create a matrix represent the initial values for Q(s,a), there are 48 states and 4 actions
        self.gamma = 0.8
        self.Q = np.zeros([48,4,4])

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.env.agent_states[self]
        
        # TODO: Select action according to your policy
        actions = [None, 'forward', 'left', 'right']
        action = random.choice(actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
        location = self.state['location']
        heading = self.state['heading']
        state_index = (location[0] - 1) * 6 + location[1] - 1 #Give the state a label
        #Find possible adjacent states
        
        if heading[0]>0:#East
            loc1 = state_index - 1
            if loc1 % 6 == 5:
                loc1 = loc1 + 6
            loc2 = state_index + 1
            if loc2 % 6 == 0:
                loc1 = loc1 - 6
            loc3 = (state_index + 6) % 48
            loc4 = state_index            
        elif heading[1]>0:#South
            loc1 = (state_index - 6) % 48
            loc2 = (state_index + 6) % 48
            loc3 = state_index + 1
            if loc3 % 6 ==0:
                loc3 = loc3 - 6
            loc4 = state_index
        elif heading[1]<0:#North
            loc1 = state_index - 1
            if loc1 % 6 == 5:
                loc1 = loc1 + 6
            loc2 = (state_index + 6) % 48
            loc3 = (state_index - 6) % 48
            loc4 = state_index
        else:#West
            loc1 = (state_index - 6) % 48
            loc2 = state_index - 1
            if loc2 % 6 == 5:
                loc2 = loc2 + 6
            loc3 = state_index + 1
            if loc3 % 6 ==0:
                loc3 = loc3 - 6
            loc4 = state_index
        
        possible_states = [loc1%48, loc2%48, loc3%48, loc4%48]
                
        self.Q[state_index,headings.index(heading), actions.index(action)] = reward + self.gamma * self.Q[possible_states,:,:].max()
        
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
