import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#### time parameters ####
#########################

T = 100
R = 1

#### model parameters ####
##########################

Np = 200
weight_limit = 1 / (Np ** 3.0)
ess = 10.0 / Np
nu = 1.0 / (2 ** Np)
phi = 0.23
sigma = 0.1
vol = [-2.5, -0.5]
p00 = 0.95
p11 = 0.01


#### Utils ####
###############

def p_ess(particles):
    return 1 / sum(np.square(particles["weights"]))


def g_ess(particles):
    return 1 / (max(particles["weights"]))


def markov_switch(i, p00, p11):
    u = np.random.uniform()
    if (i == 0):
        if u <= p00:
            return 0
        else:
            return 1
    else:
        if u <= p11:
            return 1
        else:
            return 0


def generate_states(ht, j):
    h_tp1 = np.random.normal(loc=vol[j] + phi * ht, scale=sigma)
    y_tp1 = np.random.normal(loc=0.0, scale=np.exp(h_tp1 / 2))
    return [h_tp1, y_tp1]


def generate_path(T):
    path = {"state": [], "obs": []}
    H = [np.random.normal(loc=vol[0],scale=sigma)]
    Y = []
    j = 0
    for t in range(T):
        j = markov_switch(j, p00, p11)
        state = generate_states(H[t], j)
        H.append(state[0])
        Y.append(state[1])
    path["state"] = H
    path["obs"] = Y
    return path


#### SIR Particle Filter ####
#############################

def process(ht, j):
    h_tp1 = vol[j] + phi * ht
    return h_tp1


def prediction(particles, Np):
    for i in range(Np):
        particles["switch"][i] = markov_switch(particles["switch"][i], p00, p11)
        particles["p_ant"][i] = particles["p"][i]
        particles["p"][i] = np.random.normal(loc=process(particles["p_ant"][i], particles["switch"][i]), scale=sigma)
    return particles


def update_weights(particles, observation, Np):
    for i in range(Np):
        p = particles["p"][i]
        particles["weights"][i] = norm.pdf(observation, 0, np.exp(p))
        if (particles["weights"][i] < weight_limit):
            particles["weights"][i] = weight_limit
    particles["weights"] /= sum(particles["weights"])
    return particles


def resampling(particles, Np):
    indexes = np.random.choice(Np, Np, p=particles["weights"])
    Zt = particles["p"][:]
    for i in range(Np):
        particles["p"][i] = Zt[indexes[i]]
        particles["weights"][i] = 1.0 / Np
    particles = prediction(particles, Np)
    return particles


def init_forward(m0, s0, Np):
    particles = {"p": [], "p_ant": [], "weights": [], "switch": []}
    h0 = np.random.normal(m0, s0, Np)
    w = []
    for i in range(Np):
        w.append(1.0 / Np)
        particles["p"].append(h0[i])
        particles["p_ant"].append(0)
        particles["switch"].append(0)
    particles["weights"] = np.array(w)
    return particles


def forward_sir_filter(m0, s0, path, T):
    particles = init_forward(m0, s0, Np)
    P = [particles]
    r = 0
    for t in range(0, T, 1):
        obs = path["obs"][t]
        p = {}
        particles = prediction(particles, Np)
        particles = update_weights(particles, obs, Np)
        p["weights"] = particles["weights"][:]
        if 1.0 / g_ess(particles) >= ess:
            r = r + 1
            particles = resampling(particles, Np)
        p["p"] = particles["p"][:]
        p["p_ant"] = particles["p_ant"][:]
        P.append(p)
    print("forward resampling rate: " + str(r * 100 / T) + "%")
    return P


#### Auxiliary Particle Filter ####
###################################


def mean_apf(particles):
    for i in range(Np):
        particles["switch"][i] = markov_switch(particles["switch"][i], p00, p11)
        particles["mean"][i] = vol[particles["switch"][i]] + phi * particles["p"][i]
    return particles


def index_apf(particles, observation, Np):
    indexes = []
    for i in range(Np):
        w_obs = norm.pdf(observation, 0, np.exp(particles["mean"][i]))
        if w_obs < weight_limit:
            w_obs = weight_limit
        prob = particles["weights"] * w_obs
        prob /= sum(prob)
        indexes.append(np.random.choice(Np, 1, p=prob)[0])
    return indexes


def process_apf(ht, j):
    h_tp1 = vol[j] + phi * ht
    return h_tp1


def prediction_apf(particles, Np, indexes):
    for i in range(Np):
        particles["p_ant"][i] = particles["p"][i]
        particles["p"][i] = np.random.normal(loc=process_apf(particles["p_ant"][indexes[i]], particles["switch"][i]), scale=sigma)
    return particles


def update_weights_apf(particles, observation, Np, indexes):
    for i in range(Np):
        p = particles["p"][i]
        m = particles["mean"][indexes[i]]
        particles["weights"][i] = norm.pdf(observation, 0, np.exp(p)) / norm.pdf(observation, 0, np.exp(m))
        if (particles["weights"][i] < weight_limit):
            particles["weights"][i] = weight_limit
    particles["weights"] /= sum(particles["weights"])
    return particles

def init_forward_apf(m0, s0, Np):
    particles = {"p": [], "p_ant": [], "weights": [], "switch": [], "mean":[]}
    h0 = np.random.normal(m0, s0, Np)
    w = []
    for i in range(Np):
        w.append(1.0 / Np)
        particles["p"].append(h0[i])
        particles["p_ant"].append(0)
        particles["switch"].append(0)
        particles["mean"].append(h0[i])
    particles["weights"] = np.array(w)
    return particles

def forward_aux_filter(m0, s0, path, T):
    particles = init_forward_apf(m0, s0, Np)
    P = [particles]
    for t in range(0, T, 1):
        obs = path["obs"][t]
        p = {}
        particles = mean_apf(particles)
        indexes = index_apf(particles, obs, Np)
        particles = prediction_apf(particles, Np, indexes)
        particles = update_weights_apf(particles, obs, Np, indexes)
        p["weights"] = particles["weights"][:]
        p["p"] = particles["p"][:]
        p["p_ant"] = particles["p_ant"][:]
        P.append(p)
    return P


#### Particle Smoother ####
###########################


def particle_smoother(m0, s0, path, T):
    p_forward = forward_aux_filter(m0, s0, path, T)
    # p_backward = backward_filter(path, T)
    smoothing_weights = []
    Xest = []
    for t in range(2, T, 1):
        p_f = p_forward[t - 1]
        w_f = p_forward[t - 1]["weights"]
        xsum = np.multiply(np.transpose(p_f["p"])[0], w_f)
        Xest.append(sum(xsum))
    return Xest


#### Simulations ####
#####################


path = generate_path(T)
X = np.transpose(path["state"])
Y = np.transpose(path["obs"])

Xest = np.zeros((R, T - 2))
Mest = np.zeros((R, T - 3))

for i in range(R):
    print("Run " + str(i + 1))
    Xest[i] = particle_smoother(0, 0.01, path, T)

linex, = plt.plot(range(T - 2), X[2:T], label="X")
liney, = plt.plot(range(T - 2), Y[2:T], label="Y")
linexest, = plt.plot(range(T - 2), np.mean(Xest, axis=0), label="Xest")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(handles=[linexest, linex, liney])
plt.show()
