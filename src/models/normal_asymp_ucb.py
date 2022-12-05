def ao_ucb_n(m1, m2, n):
    regret = []
    for i in range(100):
        true_reward = max(m1, m2) * n

        arm1_count = 1
        arm2_count = 1

        m1_empirical = [np.random.normal(m1, 1)]
        m2_empirical = [np.random.normal(m2, 1)]

        for j in range(n-2):
            ucb = [0,0]

            # arm1 calculation
            ucb[0] = np.mean(m1_empirical) + np.sqrt(2*np.log(1+(arm1_count+1)*np.log(np.log(arm1_count+1)))/arm1_count)
            
            # arm2 calculation
            ucb[1] = np.mean(m2_empirical) + np.sqrt(2*np.log(1+(arm2_count+1)*np.log(np.log(arm2_count+1)))/arm2_count)
            
            if ucb.index(max(ucb)) == 0:
                arm1_count += 1
                m1_empirical.append(np.random.normal(m1, 1))
            else:
                arm2_count += 1
                m2_empirical.append(np.random.normal(m2, 1))
                
        regret.append(true_reward - (len(m1_empirical)*m1 + len(m2_empirical)*m2))
    
    return [np.mean(regret), np.var(regret)]