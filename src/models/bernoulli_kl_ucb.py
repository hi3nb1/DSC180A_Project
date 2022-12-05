def kl_ucb(b1, b2, n):
    def divergence(p, q):
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    
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
            p1 = np.mean(b1_empirical)
            bound1 = np.log(1 + arm1_count*np.log(np.log(arm1_count))) / arm1_count
            
            lower1 = p1
            upper1 = 1
            for k in range(10):
                q1 = (lower1 + upper1) / 2
                if (upper1 - lower1 < 0.0001):
                    break
                test = divergence(p1, q1)
                if test < bound1:
                    lower1 = q1
                else:
                    upper1 = q1
            

            # arm2 calculation
            p2 = np.mean(b2_empirical)
            bound2 = np.log(1 + arm2_count*np.log(np.log(arm2_count))) / arm2_count
            
            lower2 = p2
            upper2 = 1
            for k in range(10):
                q2 = (lower2 + upper2) / 2
                if (upper2 - lower2 < 0.0001):
                    break
                test = divergence(p2, q2)
                if test < bound2:
                    lower2 = q2
                else:
                    upper2 = q2

            if ucb.index(max(ucb)) == 0:
                arm1_count += 1
                b1_empirical.append(np.random.choice([0,1], p=[1-b1, b1]))
            else:
                arm2_count += 1
                b2_empirical.append(np.random.choice([0,1], p=[1-b2, b2]))
                
        regret.append(true_reward - (len(b1_empirical)*b1 + len(b2_empirical)*b2))
    
    return [np.mean(regret), np.var(regret)]