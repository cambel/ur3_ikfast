# import ur3_kinematics as ur3_arm
import ur3_kinematics as ur3_arm

import numpy as np
import random

q = np.ones(6)*1.0
q = np.array([1.3131, -1.3424, 1.2247, -1.4563, -1.5629, -0.2531])

rot = [-0.000, -0.330, 0.944, 0.000]

# pose = ur3_arm.forward(q, 'q')
# print "Fk", np.round(pose, 4).tolist()

def stress_test():
    for _ in range(10000):
        act = [random.random()*0.2 for _ in range(3)]
        
        pose = np.array([act + rot])
        # pose = ur3_arm.forward(q, 'q')
        # print "FK result:", pose

        ur3_arm.inverse(pose)

def _best_ik_sol(sols, q_guess, weights=np.ones(6)):
    """ Get best IK solution """
    valid_sols = []
    for sol in sols:
        test_sol = np.ones(6) * 9999.
        for i in range(6):
            for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2. * np.pi
                        and abs(test_ang - q_guess[i]) <
                        abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if len(valid_sols) == 0:
        return None
    best_sol_ind = np.argmin(
        np.sum((weights * (valid_sols - np.array(q_guess)))**2, 1))
    return valid_sols[best_sol_ind]


pose = np.array([[-0.0813921290377793, 0.5172287777137272, 0.19673604340411155, -0.5006034812613676, 0.5007233861521643, 0.4994208785680137, 0.4992504693634886]])
sols = ur3_arm.inverse(pose)

print (np.round(_best_ik_sol(sols, q), 3).tolist())
