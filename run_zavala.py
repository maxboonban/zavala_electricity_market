import time
import numpy as np
from jax import random

from zavala_funcs import (
    zavala,
    generate_instance,
    price_distortion,
    probability_feasible,
    expected_cumulative_regret,
)

# =========================
# Run only the Zavala baseline
# =========================
num_instances = 10
key = random.key(100)
keys = random.split(key, num_instances)
instances = []
for key in keys:
    instances.append(generate_instance(key, num_scenarios=10, num_g=10, num_d=10))

zavala_times = []
zavala_distortions = []
zavala_regrets = []
probs_feasible = []

for i in range(len(instances)):
    # print('Instance', i)
    probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar = instances[i]
    
    # Debug: Print instance fingerprint for reproducibility checking
    if i == 0:  # Only print for first instance to avoid spam
        print(f"Instance fingerprint - probs[:5]: {probs[:5].round(6)}")
        print(f"mc_g_i[:3]: {mc_g_i[:3].round(3)}, mv_d_j[:3]: {mv_d_j[:3].round(3)}")
        print(f"g_i_bar.mean(): {g_i_bar.mean():.3f}, d_j_bar.mean(): {d_j_bar.mean():.3f}")

    start_time = time.time()
    z_g_i, z_d_j, _, _, z_pi, z_Pi = zavala(probs, mc_g_i, mv_d_j, g_i_bar, d_j_bar)
    end_time = time.time()
    
    # Debug: Print dual values for first instance
    if i == 0:
        print(f"Day-ahead dual (pi): {z_pi:.6f}")
        print(f"Real-time duals (Pi): {z_Pi.round(6)}")
    
    probs_feasible.append(probability_feasible(probs, z_g_i, z_d_j, g_i_bar, d_j_bar))
    # zavala_times.append(end_time - start_time)
    # print('Zavala time:', end_time - start_time)
    zavala_distortions.append(price_distortion(probs, z_pi, z_Pi))
    zavala_regrets.append(expected_cumulative_regret(probs, z_g_i, z_d_j, z_pi, mc_g_i, mv_d_j, g_i_bar, d_j_bar))

print(f'Mean prob feasible: {np.mean(probs_feasible)}')
print(f'Zavala mean time: {np.mean(zavala_times)}')
print(f'Zavala mean distortion: {np.mean(zavala_distortions)}')
print(f'Zavala mean regret: {np.mean(zavala_regrets)}')

