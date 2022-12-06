import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normal(priors, means, n):
    regret = []
    for i in range(100):
        arm1_reward = []
        arm2_reward = []
        
        post1 = priors[0].copy()
        post2 = priors[1].copy()
        for j in range(n):
            arm1 = np.random.normal(post1[0], np.sqrt(post1[1]))
            arm2 = np.random.normal(post2[0], np.sqrt(post2[1]))
            
            if arm1 > arm2:
                reward = np.random.normal(means[0], 1)
                arm1_reward.append(reward)
                post1[0] = (post1[0]/post1[1] + np.mean(arm1_reward)/1**2) / (1/post1[1] + 1/1**2)       
                post1[1] = (1 + 1/(post1[1]))**(-1)
            else:
                reward = np.random.normal(means[1], 1)
                arm2_reward.append(reward)
                post2[0] = (post2[0]/post2[1] + np.mean(arm2_reward)/1**2) / (1/post2[1] + 1/1**2)
                post2[1] = (1 + 1/(post2[1]))**(-1)
            
            
        regret.append(max(means)*n - (len(arm1_reward)*means[0] + (len(arm2_reward)*means[1])))
    return [np.mean(regret), np.var(regret)]

#####

def bernoulli(priors, prob, n):
    regret = []
    for i in range(100):
        arm1_reward = []
        arm2_reward = []
        
        arm1_pull = [0,0]
        arm2_pull = [0,0]
        
        post1 = priors[0].copy()
        post2 = priors[1].copy()
        for j in range(n):
            arm1 = np.random.beta(post1[0], post1[1])
            arm2 = np.random.beta(post2[0], post2[1])
            
            if arm1 > arm2:
                reward = np.random.choice([0,1], p=[1-prob[0], prob[0]])
                if reward == 1:
                    arm1_pull[0] += 1
                else:
                    arm1_pull[1] += 1
                arm1_reward.append(reward)
                post1[0] = post1[0] + arm1_pull[0] 
                post1[1] = post1[1] + arm1_pull[1]
            else:
                reward = np.random.choice([0,1], p=[1-prob[1], prob[1]])
                if reward == 1:
                    arm2_pull[0] += 1
                else:
                    arm2_pull[1] += 1
                arm2_reward.append(reward)
                post2[0] = post2[0] + arm2_pull[0] 
                post2[1] = post2[1] + arm2_pull[1]
            
            
        regret.append(max(prob)*n - (len(arm1_reward)*prob[0] + (len(arm2_reward)*prob[1])))
    return [np.mean(regret), np.var(regret)]

#####

def linear(means, n, v, prior=[0,1]):
    regret = []
    optimal = max([v*means[0], v*means[1]]) * n
    
    for i in range(1000):
        arm1_reward = []
        arm2_reward = []
        post = prior.copy()
        
        for j in range(1,n+1):
            theta = np.random.normal(post[0], np.sqrt(post[1]))
            a_list = [theta * each for each in means]
            reward = v*means[np.argmax(a_list)] + np.random.normal(0,1)
            if np.argmax(a_list) == 0:
                arm1_reward.append(reward)
            else:
                arm2_reward.append(reward)
            
            post[0] = 1/((1/post[1]) + means[np.argmax(a_list)]**2) * ((1/post[1])*theta + reward*means[np.argmax(a_list)])
            post[1] = 1/((1/post[1]) + means[np.argmax(a_list)]**2)
            
        regret.append(optimal - (v*means[0]*len(arm1_reward)+v*means[1]*len(arm2_reward)))
    return [np.mean(regret), np.var(regret)]