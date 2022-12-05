def moss_b(b1, b2, n):
    regret = []
    for i in range(100):
        true_reward = max(b1, b2) * n

        arm1_count = 1
        arm2_count = 1

        b1_empirical = [np.random.choice([0,1], p=[1-b1, b1])]
        b2_empirical = [np.random.choice([0,1], p=[1-b2, b2])]

        for j in range(n-2):
            ucb = [0,0]

            # arm1 calculation
            ucb[0] = np.mean(b1_empirical) + np.sqrt((4/arm1_count)*np.log(max(1, n/(2*arm1_count))))

            # arm2 calculation
            ucb[1] = np.mean(b2_empirical) + np.sqrt((4/arm2_count)*np.log(max(1, n/(2*arm2_count))))

            if ucb.index(max(ucb)) == 0:
                arm1_count += 1
                b1_empirical.append(np.random.choice([0,1], p=[1-b1, b1]))
            else:
                arm2_count += 1
                b2_empirical.append(np.random.choice([0,1], p=[1-b2, b2]))
                
        regret.append(true_reward - (len(b1_empirical)*b1 + len(b2_empirical)*b2))
    
    return [np.mean(regret), np.var(regret)]