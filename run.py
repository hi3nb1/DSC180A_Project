import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

cwd = os.getcwd()
out = os.path.join(cwd, 'output')
sys.path.insert(0, cwd+'/src')

import bayesian
import etc
import thompson
import ucb

test = 15 if sys.argv[1] == 'test' else 100
n = 10 if sys.argv[1] == 'test' else 1000

# ETC Normal Parameters
mean1 = 0
mean2 = [0.01*i for i in range(1,101)]
m = [25,50,75,100]
# ETC Normal Function
etc_normal = etc.normal(mean1, mean2, n, m)

# ETC Bernoulli Parameters
prob1 = 0
prob2 = [0.01*i for i in range(1,101)]
# ETC Bernoulli Function
etc_bernoulli = etc.bernoulli(prob1, prob2, n, m)

# --------------- #

# UCB Normal Parameters
error = 1/(1000*2)
# UCB Normal Function
ucb_normal = pd.DataFrame()
ucb_normal['mu2-mu1'] = [each - mean1 for each in mean2]
ucb_normal['regret'] = [ucb.normal(mean1, each, n, error, test)[0] for each in mean2]
ucb_normal['variance'] = [ucb.normal(mean1, each, n, error, test)[1] for each in mean2]

# UCB Bernoulli Parameters
ucb_prob1 = 1/2
# UCB Bernoulli Function
ucb_bernoulli = pd.DataFrame()
ucb_bernoulli['mu2-mu1'] = [each - ucb_prob1 for each in prob2]
ucb_bernoulli['regret'] = [ucb.bernoulli(ucb_prob1, each, n, error, test)[0] for each in prob2]
ucb_bernoulli['variance'] = [ucb.bernoulli(ucb_prob1, each, n, error, test)[1] for each in prob2]

# UCB Normal Asymptotic Function
ucb_asymp = pd.DataFrame()
ucb_asymp['mu2-mu1'] = [each - mean1 for each in mean2]
asymp_regret = []
asymp_var = []
for each in mean2:
    result = ucb.asymptotic(mean1, each, n, test)
    asymp_regret.append(result[0])
    asymp_var.append(result[1])
ucb_asymp['regret'] = asymp_regret
ucb_asymp['variance'] = asymp_var

# UCB Normal Moss Function
ucb_n_moss = pd.DataFrame()
ucb_n_moss['mu2-mu1'] = [each - mean1 for each in mean2]
ucb_n_moss['regret'] = [ucb.moss_normal(mean1, each, n, test)[0] for each in mean2]
ucb_n_moss['variance'] = [ucb.moss_normal(mean1, each, n, test)[1] for each in mean2]

# UCB Bernoulli Moss Function
ucb_b_moss = pd.DataFrame()
ucb_b_moss['mu2-mu1'] = [each - prob1 for each in prob2]
ucb_b_moss['regret'] = [ucb.moss_bernoulli(prob1, each, n, test)[0] for each in prob2]
ucb_b_moss['variance'] = [ucb.moss_bernoulli(prob1, each, n, test)[1] for each in prob2]

# UCB Bernoulli KL Function
kl = pd.DataFrame()
kl['mu2-mu1'] = [each - ucb_prob1 for each in prob2]
kl['regret'] = [ucb.kl(ucb_prob1, each, n, test)[0] for each in prob2]
kl['variance'] = [ucb.kl(ucb_prob1, each, n, test)[1] for each in prob2]

# UCB Linear Parameters
d = 1
lamb = 0.1
v_list = np.arange(-0.5,0.5, 0.01)
a_list = {
    '(0.1, -0.1)': [0.1,-0.1],
    '(0.1, -0.2)': [0.1,-0.2],
    '(0.1, 0.2)': [0.1,0.2]
}
# UCB Linear Function
lin_ucb_output = pd.DataFrame()
lin_ucb_v = []
lin_ucb_a = []
lin_ucb_regret = []
lin_ucb_var = []
for each in v_list:
    for key,val in a_list.items():
        lin_ucb_v.append(each)
        lin_ucb_a.append(key)
        result = ucb.linear(d, lamb, n, val, each, test)
        lin_ucb_regret.append(result[0])
        lin_ucb_var.append(result[1])
        
lin_ucb_output['(a1, a2)'] = lin_ucb_a
lin_ucb_output['v'] = lin_ucb_v
lin_ucb_output['regret'] = lin_ucb_regret
lin_ucb_output['variance'] = lin_ucb_var

# ------------- #

# Thompson Normal Parameters
thomp_mean2 = np.arange(0,1,0.05)
mean_dict = {
    'm1':[[0,1], [0,1]],
    'm2':[[0,1], [1/2,1]],
    'm3':[[1/2,1], [0,1]],
    'm4':[[0,0.1], [1/2,0.1]],
    'm5':[[1/2,0.1], [0,0.1]]
}
# Thompson Normal Function
n_results = {}
for key,value in mean_dict.items():
    ndiff = []
    nregret = []
    nvariance = []
    for each in thomp_mean2:
        ndiff.append(each - mean1)
        mean, var = thompson.normal(value.copy(), [mean1, each], n, test)
        nregret.append(mean)
        nvariance.append(var)
        
    nout = pd.DataFrame()
    nout['mu2-mu1'] = ndiff
    nout['regret'] = nregret
    nout['variance'] = nvariance
    n_results[key+'- '+str(value)] = nout

# Thompson Bernoulli Parameters
thomp_prob1 = 0.5
thomp_prob2 = np.arange(0,1,0.05)
beta_dict = {
    'b1':[[1,1], [1,1]],
    'b2':[[1,1], [1,3]],
    'b3':[[10,10], [10,10]],
    'b4':[[10,10], [10,30]]
}
# Thompson Bernoulli Function
b_results = {}
for key,value in beta_dict.items():
    bdiff = []
    bregret = []
    bvariance = []
    for each in thomp_prob2:
        bdiff.append(each - prob1)
        mean, var = thompson.bernoulli(value.copy(), [thomp_prob1, each], n, test)
        bregret.append(mean)
        bvariance.append(var)
        
    bout = pd.DataFrame()
    bout['mu2-mu1'] = bdiff
    bout['regret'] = bregret
    bout['variance'] = bvariance
    b_results[key+'- '+str(value)] = bout

# Thompson Linear Function
thomp_output = pd.DataFrame()
thomp_v = []
thomp_a = []
thomp_regret = []
thomp_var = []
for each in v_list:
    for key,val in a_list.items():
        thomp_v.append(each)
        thomp_a.append(key)
        result = thompson.linear(val, n, each, test)
        thomp_regret.append(result[0])
        thomp_var.append(result[1])
        
thomp_output['(a1, a2)'] = thomp_a
thomp_output['v'] = thomp_v
thomp_output['regret'] = thomp_regret
thomp_output['variance'] = thomp_var

# ------------- #    

# Bayesian Parameters
arm1 = np.arange(0,1.05,0.05)
arm2 = 0.5
priors = [
    [1,1],
    [1,3],
    [10,10],
    [10,30]
]
# Bayesian Function
bay_mu_diff = []
bay_beta = []
bay_regret = []
bay_variance = []
for each in arm1:
    for every in priors:
        bay_mu_diff.append(arm2 - each)
        bay_beta.append(every)
        result = bayesian.bay(each,arm2,every, test)
        bay_regret.append(result[0])
        bay_variance.append(result[1])
    
bay_df = pd.DataFrame()
bay_df['mu_diff'] = bay_mu_diff
bay_df['beta'] = bay_beta
bay_df['regret'] = bay_regret
bay_df['variance'] = bay_variance

etc_normal.to_csv(os.path.join(out, 'etc_normal'), index=False)
etc_bernoulli.to_csv(os.path.join(out, 'etc_bernoulli'), index=False)
ucb_normal.to_csv(os.path.join(out, 'ucb_normal'), index=False)
ucb_bernoulli.to_csv(os.path.join(out, 'ucb_bernoulli'), index=False)
ucb_asymp.to_csv(os.path.join(out, 'ucb_asymp'), index=False)
ucb_n_moss.to_csv(os.path.join(out, 'ucb_n_moss'), index=False)
ucb_b_moss.to_csv(os.path.join(out, 'ucb_b_moss'), index=False)
kl.to_csv(os.path.join(out, 'kl'), index=False)
lin_ucb_output.to_csv(os.path.join(out, 'lin_ucb_output'), index=False)
nout.to_csv(os.path.join(out, 'thompson_n'), index=False)
bout.to_csv(os.path.join(out, 'thompson_b'), index=False)
thomp_output.to_csv(os.path.join(out, 'lin_thomp_output'), index=False)
bay_df.to_csv(os.path.join(out, 'bayesian'), index=False)
