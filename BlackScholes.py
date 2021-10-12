import numpy as np
from scipy.stats import norm

#We define some functions that we will be using repeatedly
#We use the norm package from scipy to import the density and distribution functions of a Gaussian

def GeomBrown(S0, mu, vol, T, n=1):
    '''Gives a lognormal random variable at expiry time'''
    mean = (mu - vol**2 / 2)*T
    sigma = vol * np.sqrt(T)
    W = np.random.lognormal(mean, sigma, size=(n,1))
    return S0 * W


def GeomBrownFull(S0, mu, vol, T, n=1, time_delta = .1):
    ''' Gives a Geometric Browinan motion with prescribed time deltas'''
    n_columns = int(T/time_delta)
    W = np.random.normal(0, vol*np.sqrt(time_delta), size=(n,n_columns))
    x = (mu - vol*vol/2)*time_delta 
    y = vol * W
    X = np.exp(x + y)
    X = np.cumprod(X, axis = 1)
    X = np.hstack((np.ones((n,1)), X))
    return S0*X


def d1(S, K, vol, r, T, t):
    return (np.log(S/K) + (r+vol**2/2)*(T-t))/(vol*np.sqrt(T-t))


def d2(S, K, vol, r, T, t):
    return (np.log(S/K) + (r-vol**2/2)*(T-t))/(vol*np.sqrt(T-t))


def call(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    n2 = d2(S, K, vol, r, T, t)
    return S*norm.cdf(n1) - np.exp(-r*(T-t))*K*norm.cdf(n2)


def put(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    n2 = d2(S, K, vol, r, T, t)
    return -S*norm.cdf(-n1) + np.exp(-r*(T-t))*K*norm.cdf(-n2)


def call_delta(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    return norm.cdf(n1)


def put_delta(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    return -norm.cdf(-n1)


def call_gamma(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    return norm.pdf(n1)/(S * vol * np.sqrt(T-t))


#There is no put gamma as it is the same as for the call


def call_vega(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    return S * np.sqrt(T-t) * norm.pdf(n1)


#There is no put vega as it is the same as for the call


def call_theta(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    n2 = d2(S, K, vol, r, T, t)
    x = S * norm.pdf(n1) * vol / (2 * np.sqrt(T-t))
    y = r * np.exp(-r * (T-t)) * norm.cdf(n2)
    return -x - y


def put_theta(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    n2 = d2(S, K, vol, r, T, t)
    x = S * norm.pdf(n1) * vol / (2 * np.sqrt(T-t))
    y = r * np.exp(-r * (T-t)) * norm.cdf(-n2)
    return -x + y


def call_rho(S, K, vol, r, T, t):
    n2 = d2(S, K, vol, r, T, t)
    return K * (T-t) * np.exp(-r * (T-t)) * norm.cdf(n2)


def put_rho(S, K, vol, r, T, t):
    n2 = d2(S, K, vol, r, T, t)
    return -K * (T-t) * np.exp(-r * (T-t)) * norm.cdf(-n2)


#We define objects that represent Black Scholes objects


class BlackScholes:
    def __init__(self, S, vol, r, T, t):
        self.spot = S
        self.vol = vol
        self.rate = r
        self.expiry = T 
        self.time = t

    def __add__(self, other):
        helper = BlackScholes(self.spot, self.vol, self.rate, self.expiry, self.time)
        helper.payoff = lambda x : self.payoff(x) + other.payoff(x)
        return helper

    def __sub__(self, other):
        helper = BlackScholes(self.spot, self.vol, self.rate, self.expiry, self.time)
        helper.payoff = lambda x : self.payoff(x) - other.payoff(x)
        return helper

    def payoff(x):
        pass


class Call(BlackScholes):
    def __init__(self, S, K, vol, r, T, t):
        super().__init__(S, vol, r, T, t)
        self.strike = K

    def payoff(self, x):
        return np.maximum(x - self.strike, 0)

    def price(self):
        return call(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def delta(self):
        return call_delta(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def gamma(self):
        return call_gamma(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def vega(self):
        return call_vega(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def rho(self):
        return call_rho(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def theta(self):
        return call_theta(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)


class Put(BlackScholes):
    def __init__(self, S, K, vol, r, T, t):
        super().__init__(S, vol, r, T, t)
        self.strike = K

    def payoff(self, x):
        return np.maximum(self.strike - x, 0)

    def price(self):
        return put(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def delta(self):
        return put_delta(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def gamma(self):
        return call_gamma(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def vega(self):
        return call_vega(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def rho(self):
        return put_rho(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)

    def theta(self):
        return put_theta(self.spot, self.strike, self.vol, self.rate, self.expiry, self.time)


class DigitalCall(BlackScholes):
    def __init__(self, S, K, vol, r, T, t):
        super().__init__(S, vol, r, T, t)
        self.strike = K

    def payoff(self, x):
        return 1.0 * (x > self.strike)

    def price(self):
        S = self.spot
        K = self.strike
        vol = self.vol
        T = self.expiry
        t = self.time
        r = self.rate
        n2 = d2(S, K, vol, r, T, t)
        return np.exp(-r *(T-t)) * norm.cdf(n2)


class DigitalPut(BlackScholes):
    def __init__(self, S, K, vol, r, T, t):
        super().__init__(S, vol, r, T, t)
        self.strike = K

    def payoff(self, x):
        return 1.0 * (x < self.strike)

    def price(self):
        S = self.spot
        K = self.strike
        vol = self.vol
        T = self.expiry
        t = self.time
        r = self.rate
        n2 = d2(S, K, vol, r, T, t)
        return np.exp(-r *(T-t)) * norm.cdf(-n2)


class MonteCarlo:
    def __init__(self, option, num_steps):
        self.option = option
        self.stop = num_steps

    def price(self):
        option = self.option
        spot = option.spot
        rate = option.rate
        vol = option.vol
        expiry = option.expiry - option.time
        num_steps = self.stop
        spot_at_expiry = GeomBrown(spot, rate, vol, expiry, num_steps).reshape(-1)
        payoffs = option.payoff(spot_at_expiry)
        payoffs = np.exp(-rate * expiry) * payoffs
        return np.mean(payoffs), np.std(payoffs)/np.sqrt(num_steps)

    def delta(self, epsilon = .001):
        option = self.option
        spot = option.spot
        rate = option.rate
        vol = option.vol
        expiry = option.expiry - option.time
        num_steps = self.stop
        spot_at_expiry = GeomBrown(spot, rate, vol, expiry, num_steps).reshape(-1)
        spot_at_expiry_epsilon = (1 + (epsilon/spot)) * spot_at_expiry
        values = option.payoff(spot_at_expiry)
        values_epsilon = option.payoff(spot_at_expiry_epsilon)
        deltas = (values_epsilon - values)/epsilon
        return np.mean(deltas), np.std(deltas)/np.sqrt(num_steps)

    def gamma(self, epsilon = .001):
        option = self.option
        spot = option.spot
        rate = option.rate
        vol = option.vol
        expiry = option.expiry - option.time
        num_steps = self.stop
        spot_at_expiry = GeomBrown(spot, rate, vol, expiry, num_steps).reshape(-1)
        spot_at_expiry_plusepsilon = (1 + (epsilon/spot)) * spot_at_expiry
        spot_at_expiry_minusepsilon = (1 - (epsilon/spot)) * spot_at_expiry
        values = option.payoff(spot_at_expiry)
        values_plusepsilon = option.payoff(spot_at_expiry_plusepsilon)
        values_minusepsilon = option.payoff(spot_at_expiry_minusepsilon)
        deltas = (values_plusepsilon - 2*values + values_minusepsilon)/epsilon**2
        return np.mean(deltas), np.std(deltas)/np.sqrt(num_steps)


option = Call(77, 95, .5, .01, 5/12, 0)
suma = option + option
mc = MonteCarlo(option, 100000)
a = mc.price()
print("{}".format(a))
b = suma.payoff(100)
print("{}".format(b))