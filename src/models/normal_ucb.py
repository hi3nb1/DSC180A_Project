def ucb_n(m1, m2, n, error):
    regret = []
    for i in range(100):
        true_reward = max(m1, m2) * n

        m1_empirical = []
        m2_empirical = []

        for j in range(n):
            ucb = [0,0]

            # arm1 calculation
            if len(m1_empirical) == 0:
                ucb[0] = float('inf')
            else:
                ucb[0] = np.mean(m1_empirical) + np.sqrt((2*np.log(1/error))/len(m1_empirical))

            # arm2 calculation
            if len(m2_empirical) == 0:
                ucb[1] = float('inf')
            else:
                ucb[1] = np.mean(m2_empirical) + np.sqrt((2*np.log(1/error))/len(m2_empirical))

            if ucb.index(max(ucb)) == 0:
                m1_empirical.append(np.random.normal(m1, 1))
            else:
                m2_empirical.append(np.random.normal(m2, 1))
                
        regret.append(true_reward - (len(m1_empirical)*m1 + len(m2_empirical)*m2))
    
    return [np.mean(regret), np.var(regret)]