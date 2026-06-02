import numpy as np

def compute_bullwhip(base_env):
    """
    Bullwhip Ratio = Var(upstream orders) / Var(customer demand)
    """
    var_demand = np.var(base_env.D.values)
    var_orders = np.var(base_env.R.values)
    if var_demand == 0:
        return 0
    return var_orders / var_demand

def compute_lost_sales_ratio(base_env):
    """
    Lost Sales Ratio = total_unfulfilled_demand / total_demand
    """
    total_unfulfilled = 0
    if hasattr(base_env, 'U'):
        total_unfulfilled = np.sum(base_env.U.values)
    elif hasattr(base_env, 'LS'):
        total_unfulfilled = np.sum(base_env.LS.values)
        
    total_demand = np.sum(base_env.D.values)
    if total_demand == 0:
        return 0
    return total_unfulfilled / total_demand

def compute_inventory_variance(base_env):
    """
    Inventory Variance = variance(network inventory) aggregated across the episode
    """
    return np.var(base_env.X.values)

def get_mean_economic_reward(base_env):
    """
    Mean Economic Reward = Average unshaped environment profit across the network
    """
    # Sum across all nodes for each timestep, then mean across the episode
    return np.mean(np.sum(base_env.P.values, axis=1))
