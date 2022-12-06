import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bay(arm1,arm2,priors):
    regret = []
    for j in range(100):
        arms = [0]*1000
        s=0
        f=0
        q=0
        total_reward = 0
        post = [priors[0], priors[1]]
        for i in range(999,0,-1):
            if i == 999:
                arms[-1] = [0,0]
            else:
                pull2 = arm2+max(arms[i+1])
                pull1 = (post[0]+s)/(post[0]+post[1]+q) + (post[0]+s)/(post[0]+post[1]+q) * max(arms[i+1]) + (1-(post[0]+s)/(post[0]+post[1]+q)) * max(arms[i+1])
                arms[i] = [pull1, pull2]
                if max(pull1,pull2) == pull1:
                    out = np.random.choice([0,1], p=[1-arm1, arm1])
                    if out == 1:
                        s += 1
                    else:
                        f += 1
                    q += 1
                    post = [post[0]+s, post[1]+f]

        for each in arms:
            pulled = np.argmax(each)
            if pulled == 0:
                total_reward += arm1
            else:
                total_reward += arm2
        optimal = 1000*(max(arm1,arm2))
        regret.append(optimal - total_reward)
    return [np.mean(regret), np.var(regret)]