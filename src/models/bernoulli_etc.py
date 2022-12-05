def etc_b(b1, b2, n):
    # optimal m
    if b2 - b1 != 0:
        optimal = int(np.round(max(1,(4/(max(b1,b2)-min(b1,b2)) * np.log((n*(max(b1,b2)-min(b1,b2))/4))))))
    else:
        optimal = 1
    
    regret = []
    for j in range(1000):
        # optimal mean
        max_mean = max(b1, b2) 

        # reward accumulated during explore
        b1_reward = b1*(optimal)
        b2_reward = b2*(optimal)
        explore_reward = b1_reward + b2_reward

        # empirical mean for each arm
        b1_explore = sum(np.random.choice([0,1], optimal, p=[1-b1, b1]))/optimal
        b2_explore = sum(np.random.choice([0,1], optimal, p=[1-b2, b2]))/optimal

        # choosing theoretical mean to exploit with (based on higher empirical mean)
        chosen = b1 if b1_explore > b2_explore else b2

        # reward accumulated during exploit
        exploit_reward = chosen * (n - 2*optimal)

        # total reward (explore and exploit)
        total_reward = explore_reward + exploit_reward

        # optimal reward (explore and exploit using optimal mean)
        true_reward = max_mean * n

        regret.append(true_reward - total_reward)
    return np.mean(regret)