def ucb_b(b1, b2, n, error):
    regret = []
    for i in range(100):
        true_reward = max(b1, b2) * n

        arm1_count = 0
        arm2_count = 0

        b1_empirical = []
        b2_empirical = []

        for j in range(n):
            ucb = [0,0]

            # arm1 calculation
            if arm1_count == 0:
                ucb[0] = float('inf')
            else:
                ucb[0] = np.mean(b1_empirical) + np.sqrt((2*np.log(1/error))/arm1_count)

            # arm2 calculation
            if arm2_count == 0:
                ucb[1] = float('inf')
            else:
                ucb[1] = np.mean(b2_empirical) + np.sqrt((2*np.log(1/error))/arm2_count)

            if ucb.index(max(ucb)) == 0:
                arm1_count += 1
                b1_empirical.append(np.random.choice([0,1], p=[1-b1, b1]))
            else:
                arm2_count += 1
                b2_empirical.append(np.random.choice([0,1], p=[1-b2, b2]))
        regret.append(true_reward - (len(b1_empirical)*b1 + len(b2_empirical)*b2))
        
    return [np.mean(regret), np.var(regret)]