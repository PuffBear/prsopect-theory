import numpy as np

def calculate_bullwhip_effect(env):
    """
    Calculate the Bullwhip Effect for a given episode.
    Bullwhip Effect = Variance of Upstream Orders / Variance of Downstream Demand
    """
    unwrapped = env.unwrapped
    # Retail demand history across all markets
    demands = unwrapped.D.sum(axis=1)
    var_demand = np.var(demands)
    
    # Orders placed by all nodes upstream
    orders = unwrapped.R.sum(axis=1)
    var_orders = np.var(orders)
    
    if var_demand == 0:
        return 1.0 # Baseline if demand is perfectly constant
        
    return var_orders / var_demand

def calculate_systemic_lost_sales(env):
    """
    Calculate the total ratio of unfulfilled market demand over the episode.
    Lost sales occur when retailers cannot satisfy the market demand.
    """
    unwrapped = env.unwrapped
    total_unfulfilled = unwrapped.U.sum().sum()
    total_demand = unwrapped.D.sum().sum()
    
    if total_demand == 0:
        return 0.0
        
    return total_unfulfilled / total_demand
