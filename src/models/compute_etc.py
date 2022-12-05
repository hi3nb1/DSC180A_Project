def etc(m1, m2, n, m='optimal'):
    explore = m
    
    # calculating optimal m if argument is given
    if m == 'optimal':
        if m2 - m1 != 0:
            explore = int(np.round(max(1,(4/(max(m1,m2)-min(m1,m2)) * np.log((n*(max(m1,m2)-min(m1,m2))/4))))))
        else:
            explore = 1

    regret = []
    
    # simulation of 1000 trials
    for j in range(1000):
        # optimal mean
        max_mean = max(m1, m2) 

        # reward accumulated during explore
        m1_reward = m1*(explore)
        m2_reward = m2*(explore)
        explore_reward = m1_reward + m2_reward

        # empirical mean for each arm
        m1_explore = sum([np.random.normal(m1, 1) for i in range(explore)])/(explore)
        m2_explore = sum([np.random.normal(m2, 1) for i in range(explore)])/(explore)

        # choosing theoretical mean to exploit with (based on higher empirical mean)
        chosen = m1 if m1_explore > m2_explore else m2

        # reward accumulated during exploit
        exploit_reward = chosen * (n - 2*explore)

        # total reward (explore and exploit)
        total_reward = explore_reward + exploit_reward

        # optimal reward (explore and exploit using optimal mean)
        true_reward = max_mean * n

        regret.append(true_reward - total_reward)
    return np.mean(regret)