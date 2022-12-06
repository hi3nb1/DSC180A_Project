import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normal(m1, m2, n, error, sim):
    regret = []
    for i in range(sim):
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

#####

def bernoulli(b1, b2, n, error, sim):
    regret = []
    for i in range(sim):
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

#####

def asymptotic(m1, m2, n, sim):
    regret = []
    for i in range(sim):
        true_reward = max(m1, m2) * n

        arm1_count = 1
        arm2_count = 1

        m1_empirical = [np.random.normal(m1, 1)]
        m2_empirical = [np.random.normal(m2, 1)]

        for j in range(n-2):
            ucb = [0,0]

            # arm1 calculation
            ucb[0] = np.mean(m1_empirical) + np.sqrt(2*np.log(1+j*np.log(j)**2)/arm1_count)
        
            # arm2 calculation
            ucb[1] = np.mean(m2_empirical) + np.sqrt(2*np.log(1+j*np.log(j)**2)/arm2_count)
            
            if ucb.index(max(ucb)) == 0:
                arm1_count += 1
                m1_empirical.append(np.random.normal(m1, 1))
            else:
                arm2_count += 1
                m2_empirical.append(np.random.normal(m2, 1))
                
        regret.append(true_reward - (len(m1_empirical)*m1 + len(m2_empirical)*m2))

    return [np.mean(regret), np.var(regret)]

#####

def moss_normal(m1, m2, n, sim):
    regret = []
    for i in range(sim):
        true_reward = max(m1, m2) * n

        arm1_count = 1
        arm2_count = 1

        m1_empirical = [np.random.normal(m1, 1)]
        m2_empirical = [np.random.normal(m2, 1)]

        for j in range(n-2):
            ucb = [0,0]

            # arm1 calculation
            ucb[0] = np.mean(m1_empirical) + np.sqrt((4/arm1_count)*np.log(max(1, n/(2*arm1_count))))
            
            # arm2 calculation
            ucb[1] = np.mean(m2_empirical) + np.sqrt((4/arm2_count)*np.log(max(1, n/(2*arm2_count))))

            if ucb.index(max(ucb)) == 0:
                arm1_count += 1
                m1_empirical.append(np.random.normal(m1, 1))
            else:
                arm2_count += 1 
                m2_empirical.append(np.random.normal(m2, 1))
                
        regret.append(true_reward - (len(m1_empirical)*m1 + len(m2_empirical)*m2))
    
    return [np.mean(regret), np.var(regret)]

#####

def moss_bernoulli(b1, b2, n, sim):
    regret = []
    for i in range(sim):
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

#####

def kl(b1, b2, n, sim):
    def divergence(p, q):
        if p == 0:
            if q == 0:
                return 0
            return np.log(1/(1-q))
        if p == 1:
            if q == 1:
                return 0
            return np.log(1/q)
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    
    regret = []
    for i in range(sim):
        true_reward = max(b1, b2) * n

        arm1_count = 1
        arm2_count = 1

        b1_empirical = [np.random.choice([0,1], p=[1-b1, b1])]
        b2_empirical = [np.random.choice([0,1], p=[1-b2, b2])]

        for j in range(n-2):
            ucb = [0,0]

            # arm1 calculation
            p1 = np.mean(b1_empirical)
            bound1 = np.log(1 + j*np.log(np.log(j))) / arm1_count
            
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
            
            ucb[0] = q1

            # arm2 calculation
            p2 = np.mean(b2_empirical)
            bound2 = np.log(1 + j*np.log(np.log(j))) / arm2_count
            
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
                    
            ucb[1] = q2
            
            if ucb.index(max(ucb)) == 0:
                arm1_count += 1
                b1_empirical.append(np.random.choice([0,1], p=[1-b1, b1]))
            else:
                arm2_count += 1
                b2_empirical.append(np.random.choice([0,1], p=[1-b2, b2]))
                
        regret.append(true_reward - (len(b1_empirical)*b1 + len(b2_empirical)*b2))
    
    return [np.mean(regret), np.var(regret)]

#####

def linear(d, lamb, n, a, v, sim):
    regret = []
    optimal = max([v*a[0], v*a[1]])*n
    
    for j in range(sim):
        arm1_reward = []
        arm2_reward = []
        
        delta = 1/n
        emp_mean = 0
        V = lamb
        theta = 0
        
        for t in range(1,n+1):
            beta = np.sqrt(lamb) + np.sqrt((2*np.log(1/delta)) + d * np.log((1+(t-1))/(delta*d)))
            arms = [0]*len(a)
            
            for i in range(len(a)):
                arms[i] = a[i]*theta + beta*np.sqrt((a[i]**2)*(1/V))
            
            if np.argmax(arms) == 0:
                reward = v*a[0] + np.random.normal(0,1)
                arm1_reward.append(reward)
                V = V + a[0]**2
                emp_mean = emp_mean + reward*a[0]
                theta = (1/V) * emp_mean
            else:
                reward = v*a[1] + np.random.normal(0,1)
                arm2_reward.append(reward)
                V = V + a[1]**2
                emp_mean = emp_mean + reward*a[1]
                theta = (1/V) * emp_mean
            
        regret.append(optimal - (v*a[0]*len(arm1_reward)+v*a[1]*len(arm2_reward)))
    return [np.mean(regret), np.var(regret)]