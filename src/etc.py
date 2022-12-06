import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normal(m1, m2, n, m):
    entries = []
    for each in m2:
        optimal = int(np.round(max(1,(4/(max(m1,each)-min(m1,each)) * np.log((n*(max(m1,each)-min(m1,each))/4))))))
        mlist = m + [optimal]
        for i in range(len(mlist)):
                
            row = [each-m1, i, mlist[i], n-2*mlist[i]]
            
            regret_list = []
            for j in range(10**3):
                # optimal mean
                max_mean = max(m1, each) 

                # reward accumulated during explore
                m1_reward = m1*(mlist[i])
                m2_reward = each*(mlist[i])
                explore_reward = m1_reward + m2_reward

                # empirical mean for each arm
                m1_explore = sum([np.random.normal(m1, 1) for i in range(mlist[i])])/(mlist[i])
                m2_explore = sum([np.random.normal(each, 1) for i in range(mlist[i])])/(mlist[i])

                # choosing theoretical mean to exploit with (based on higher empirical mean)
                chosen = m1 if m1_explore > m2_explore else each

                # reward accumulated during exploit
                exploit_reward = chosen * (n - 2*mlist[i])

                # total reward (explore and exploit)
                total_reward = explore_reward + exploit_reward

                # optimal reward (explore and exploit using optimal mean)
                true_reward = max_mean * n

                regret_list.append(true_reward - total_reward)
                    
            row.append(np.mean(regret_list))
            entries.append(row)
            
    return pd.DataFrame(entries, columns=['μ2 − μ1','m index', 'm', 'n - 2m', 'regret'])

##########

def bernoulli(b1, b2, n, m):
    entries = []
    for each in b2:
        optimal = int(np.round(max(1,(4/(max(b1,each)-min(b1,each)) * np.log((n*(max(b1,each)-min(b1,each))/4))))))
        mlist = m + [optimal]
        for i in range(len(mlist)):
                
            row = [each-b1, i, mlist[i], n-2*mlist[i]]
            
            regret_list = []
            for j in range(10**3):
                # optimal mean
                max_mean = max(b1, each) 

                # reward accumulated during explore
                b1_reward = b1*(mlist[i])
                b2_reward = each*(mlist[i])
                explore_reward = b1_reward + b2_reward

                # empirical mean for each arm
                b1_explore = sum(np.random.choice([0,1], mlist[i], p=[1-b1, b1]))/mlist[i]
                b2_explore = sum(np.random.choice([0,1], mlist[i], p=[1-each, each]))/mlist[i]
            
                # choosing theoretical mean to exploit with (based on higher empirical mean)
                chosen = b1 if b1_explore > b2_explore else each

                # reward accumulated during exploit
                exploit_reward = chosen * (n - 2*mlist[i])

                # total reward (explore and exploit)
                total_reward = explore_reward + exploit_reward

                # optimal reward (explore and exploit using optimal mean)
                true_reward = max_mean * n

                regret_list.append(true_reward - total_reward)
                    
            row.append(np.mean(regret_list))
            entries.append(row)
            
    return pd.DataFrame(entries, columns=['μ2 − μ1','m index', 'm', 'n - 2m', 'regret'])