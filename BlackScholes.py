import numpy as np
from scipy.stats import norm

#We define some functions that we will be using repeatedly
#We use the norm package from scipy to import the density and distribution functions of a Gaussian

def GeomBrown(S0, mu, vol, T, n=1):
    '''Gives a lognormal random print("{}".format(a))variable at expiry time'''
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
    return -N(-n1)


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

    def gamma(self):
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

    def gamma(self):
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
        return np.exp(-rate * expiry) * np.mean(payoffs)




option = DigitalPut(55, 66, .2, .1, 1, 0)
mc = MonteCarlo(option, 10000)
a = mc.price()
print("{}".format(a))
b = option.price()
print("{}".format(b))