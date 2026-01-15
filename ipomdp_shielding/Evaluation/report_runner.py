" Evaluation Runner - generic method for running experiments "

from dataclasses import dataclass


from ..Models import IPOMDP
from .runtime_shield import RuntimeImpShield

@dataclass
class ReportRunner:
    ipomdp : IPOMDP
    perception : "State -> Obs"
    rt_shield : RuntimeImpShield
    action_selector : "{Actions} -> Action"
    length : int
    initial : "State x Action"


    def run(self):

        self.initialize()

        self.rt_shield.initialize(self.initial)

        state, action = self.initial

        obs = self.perception(state)

        for step in range(self.length):
            # Stop if we've reached FAIL state
            if state == "FAIL":
                break

            # get next action
            actions = self.rt_shield.next_actions((obs, action))
            action = self.action_selector(actions)

            # evaluator sees cur state, obs, and chosen action
            self.report(step, state, obs, action, self.rt_shield)

            # update state with action
            state = self.ipomdp.evolve(state, action)

            # update obs with new state
            obs = self.perception(state)

        return self.final(self.rt_shield)
    
    def initialize(self):
        pass
    
    def report(self, step, state, obs, action, rt_shield):
        pass


    def final(self, rt_shield):
        pass

        
