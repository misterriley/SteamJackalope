import numpy as np

def softmin_blend(signals, temperature=0.01):
    if not signals:
        return 0.0
    stack = np.stack(signals, axis=0) # (num_signals, num_games)
    scaled = -stack / temperature
    max_val = np.max(scaled, axis=0)
    exp_vals = np.exp(scaled - max_val)
    weights = exp_vals / np.sum(exp_vals, axis=0)
    return np.sum(stack * weights, axis=0)

# Antichamber -> Super reaKtor
t_sim = 0.1249
s_sim = 0.0160
top_sim = 0.0
# We also scale topic sim by 0.1 in the main logic? No, let's use the real values.
# Wait, I scale topic_sim by 0.1 in solve_user_taste.py.
top_sim_scaled = 0.0 * 0.1

print(f"Antichamber -> Super reaKtor Softmin (T=0.01): {softmin_blend([t_sim, s_sim, top_sim_scaled])}")

# Ori -> Deltarune
t_sim_o = 0.064
s_sim_o = 0.040
top_sim_o = 0.541
top_sim_o_scaled = 0.541 * 0.1 # 0.0541

print(f"Ori -> Deltarune Softmin (T=0.01): {softmin_blend([t_sim_o, s_sim_o, top_sim_o_scaled])}")
