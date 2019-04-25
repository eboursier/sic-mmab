import numpy as np

class FullSensingMultiPlayerMAB():
    """
    Structure of stochastic MAB in the full sensing model
    """
    
    def __init__(self, means, nplayers, strategy, **kwargs):
        self.K = len(means)
        np.random.shuffle(means)
        self.means = np.array(means)
        self.M = nplayers
        self.players = [strategy(narms=self.K, **kwargs) for _ in range(nplayers)]
        
    def simulate_single_step_rewards(self):
        return np.random.binomial(1, self.means)   
      
    def simulate_single_step(self, plays):
        unique, counts = np.unique(plays, return_counts=True)
        # remove the collisions
        collisions = unique[counts>1]
        cols = np.array([p in collisions for p in plays]) # the value is 1 if there is collision
        rews = self.simulate_single_step_rewards()
        rewards = rews[plays]*(1-cols)
        return list(zip(rews[plays], cols)), rewards    
    
    def simulate(self, horizon=10000):
        """
        Return the vector of regret for each time step until horizon
        """
        
        rewards = []
        play_history = []
        
        for t in range(horizon):
            plays = np.zeros(self.M)
            plays = [(int)(player.play()) for player in self.players]
                
            obs, rews = self.simulate_single_step(plays)
            
            [self.players[i].update(plays[i], obs[i]) for i in range(self.M)]
            
            rewards.append(np.sum(rews))
            play_history.append(plays)
        
        top_means = -np.partition(-self.means, self.M)[:self.M]
        best_case_reward = np.sum(top_means)*np.arange(1, horizon+1)
        cumulated_reward = np.cumsum(rewards)
        
        regret = best_case_reward - cumulated_reward
        return regret, play_history



# if __name__ == "__main__":

# 	horizon = 5000
# 	K = 9
# 	means = np.linspace(0.9, 0.89, K)
# 	nplayers = 6
# 	n_simu = 10
# 	regret = []

# 	strat = SynchComm

#     for i in range(nalgo):
#         MAB = FullSensingMultiPlayerMAB(means, nplayers=nplayers, strategy=strat, T=horizon)
#         r, _ = MAB.simulate(horizon)
#         regret.append(r)
