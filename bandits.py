import numpy as np

class FullSensingMultiPlayerMAB():
    """
    Structure of stochastic MAB in the full sensing model (adapted to both Collision and Statistic Sensing settings)
    """
    
    def __init__(self, means, nplayers, strategy, **kwargs):
        self.K = len(means)
        np.random.shuffle(means)
        self.means = np.array(means)
        self.M = nplayers
        self.players = [strategy(narms=self.K, **kwargs) for _ in range(nplayers)] # list of all players and their strategy
        
    def simulate_single_step_rewards(self):
        return np.random.binomial(1, self.means)   
      
    def simulate_single_step(self, plays):
        """
        return to each player its stat and collision indicator where plays is the vector of plays by the players
        """
        unique, counts = np.unique(plays, return_counts=True) # compute the number of pulls per arm
        # remove the collisions
        collisions = unique[counts>1] # arms where collisions happen
        cols = np.array([p in collisions for p in plays]) # the value is 1 if there is collision
        rews = self.simulate_single_step_rewards() # generate the stats X_k(t)
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
            plays = [(int)(player.play()) for player in self.players] # plays of all players
                
            obs, rews = self.simulate_single_step(plays) # observations of all players
            
            [self.players[i].update(plays[i], obs[i]) for i in range(self.M)] # update strategies of all players
            
            rewards.append(np.sum(rews)) # list of rewards
            play_history.append(plays)
        
        top_means = -np.partition(-self.means, self.M)[:self.M]
        best_case_reward = np.sum(top_means)*np.arange(1, horizon+1)
        cumulated_reward = np.cumsum(rewards)
        
        regret = best_case_reward - cumulated_reward
        return regret, play_history