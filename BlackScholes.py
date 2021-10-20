import numpy as np
import matplotlib.pyplot as plt
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
    y = r * K * np.exp(-r * (T-t)) * norm.cdf(n2)
    return -x - y


def put_theta(S, K, vol, r, T, t):
    n1 = d1(S, K, vol, r, T, t)
    n2 = d2(S, K, vol, r, T, t)
    x = S * norm.pdf(n1) * vol / (2 * np.sqrt(T-t))
    y = r * K * np.exp(-r * (T-t)) * norm.cdf(-n2)
    return -x + y


def call_rho(S, K, vol, r, T, t):
    n2 = d2(S, K, vol, r, T, t)
    return K * (T-t) * np.exp(-r * (T-t)) * norm.cdf(n2)


def put_rho(S, K, vol, r, T, t):
    n2 = d2(S, K, vol, r, T, t)
    return -K * (T-t) * np.exp(-r * (T-t)) * norm.cdf(-n2)


def digital_call(S, K, vol, r, T, t):
    n2 = d2(S, K, vol, r, T, t)
    return np.exp(-r *(T-t)) * norm.cdf(n2)


def digital_put(S, K, vol, r, T, t):
    n2 = d2(S, K, vol, r, T, t)
    return np.exp(-r *(T-t)) * norm.cdf(-n2)


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
        if (hasattr(self, "price") * hasattr(other, "price")):
            helper.price = lambda  : self.price() + other.price()

        if (hasattr(self, "delta") * hasattr(other, "delta")):
            helper.delta = lambda  : self.delta() + other.delta()

        if (hasattr(self, "gamma") * hasattr(other, "gamma")):
            helper.gamma = lambda  : self.gamma() + other.gamma()

        if (hasattr(self, "vega") * hasattr(other, "vega")):
            helper.vega = lambda  : self.vega() + other.vega()

        if (hasattr(self, "theta") * hasattr(other, "theta")):
            helper.theta = lambda  : self.theta() + other.theta()

        if (hasattr(self, "rho") * hasattr(other, "rho")):
            helper.rho = lambda  : self.rho() + other.rho()

        return helper

    def __sub__(self, other):
        helper = BlackScholes(self.spot, self.vol, self.rate, self.expiry, self.time)
        helper.payoff = lambda x : self.payoff(x) - other.payoff(x)

        if (hasattr(self, "price") * hasattr(other, "price")):
            helper.price = lambda  : self.price() - other.price()

        if (hasattr(self, "delta") * hasattr(other, "delta")):
            helper.delta = lambda  : self.delta() - other.delta()

        if (hasattr(self, "gamma") * hasattr(other, "gamma")):
            helper.gamma = lambda  : self.gamma() - other.gamma()

        if (hasattr(self, "vega") * hasattr(other, "vega")):
            helper.vega = lambda  : self.vega() - other.vega()

        if (hasattr(self, "theta") * hasattr(other, "theta")):
            helper.theta = lambda  : self.theta() - other.theta()

        if (hasattr(self, "rho") * hasattr(other, "rho")):
            helper.rho = lambda  : self.rho() - other.rho()

        return helper

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            helper = BlackScholes(self.spot, self.vol, self.rate, self.expiry, self.time)
            helper.payoff = lambda x : other * self.payoff(x)

            if hasattr(self, "price"):
                helper.price = lambda : other * self.price()

            if hasattr(self, "delta"):
                helper.delta = lambda : other * self.delta()

            if hasattr(self, "gamma"):
                helper.gamma = lambda : other * self.gamma()

            if hasattr(self, "vega"):
                helper.vega = lambda : other * self.vega()

            if hasattr(self, "theta"):
                helper.theta = lambda : other * self.theta()

            if hasattr(self, "rho"):
                helper.rho = lambda : other * self.rho()

            return helper

    def __rmul__(self, other):
        return self.__mul__(other)

    def payoff(self, x):
        pass

    def plot_time(self, time_delta = .01):
        if hasattr(self, "bs_formula"):
            S = self.spot
            K = self.strike
            vol = self.vol
            r = self.rate
            T = self.expiry
            times = np.arange(self.time, T, time_delta)
            func = self.bs_formula()
            plt.grid()
            plt.xlabel("Time elapsed in years")
            plt.ylabel("Price of the option")
            plt.plot(times, func(S, K, vol, r, T, times))
        else:
            print("Unable to plot because there is no explicit formula for price.")

    def plot_spot(self, spot_delta = .01):
        if hasattr(self, "bs_formula"):
            S = np.arange(.5 * self.spot, 1.5 * self.spot, spot_delta)
            K = self.strike
            vol = self.vol
            r = self.rate
            T = self.expiry
            t = self.time
            func = self.bs_formula()
            plt.grid()
            plt.xlabel("Value of the spot")
            plt.ylabel("Price of the option")
            plt.plot(S, func(S, K, vol, r, T, t))
        else:
            print("Unable to plot because there is no explicit formula for price.")


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

    def bs_formula(self):
        func = call
        return func


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

    def bs_formula(self):
        func = put
        return func


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
        r = self.rate
        T = self.expiry
        t = self.time
        return digital_call(S, K, vol, r, T, t)

    def bs_formula(self):
        func = digital_call
        return func


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
        r = self.rate
        T = self.expiry
        t = self.time
        return digital_put(S, K, vol, r, T, t)

    def bs_formula(self):
        func = digital_put
        return func


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

    def vega(self, epsilon = .001):
        option = self.option
        spot = option.spot
        rate = option.rate
        vol = option.vol
        expiry = option.expiry - option.time
        num_steps = self.stop
        normals = np.random.normal(0,1, num_steps)
        vol_epsilon = vol + epsilon
        exponent = (rate-vol**2/2)*expiry - vol*np.sqrt(expiry)*normals
        exponent_epsilon = (rate-vol_epsilon**2/2)*expiry - vol_epsilon*np.sqrt(expiry)*normals
        spot_at_expiry = spot * np.exp(exponent)
        spot_at_expiry_epsilon = spot * np.exp(exponent_epsilon)
        values = option.payoff(spot_at_expiry)
        values_epsilon = option.payoff(spot_at_expiry_epsilon)
        vegas = (values_epsilon - values)/epsilon
        return np.mean(vegas), np.std(vegas)/np.sqrt(num_steps)

    def rho(self, epsilon = .001):
        option = self.option
        spot = option.spot
        rate = option.rate
        vol = option.vol
        expiry = option.expiry - option.time
        num_steps = self.stop
        normals = np.random.normal(0,1, num_steps)
        rate_epsilon = rate + epsilon
        exponent = (rate-vol**2/2)*expiry - vol*np.sqrt(expiry)*normals
        exponent_epsilon = (rate_epsilon-vol**2/2)*expiry - vol*np.sqrt(expiry)*normals
        spot_at_expiry = spot * np.exp(exponent)
        spot_at_expiry_epsilon = spot * np.exp(exponent_epsilon)
        values = np.exp(-rate * expiry) * option.payoff(spot_at_expiry)
        values_epsilon = np.exp(-rate_epsilon * expiry) * option.payoff(spot_at_expiry_epsilon)
        rhos = (values_epsilon - values)/epsilon
        return np.mean(rhos), np.std(rhos)/np.sqrt(num_steps)

    def theta(self, epsilon = .001):
        option = self.option
        spot = option.spot
        rate = option.rate
        vol = option.vol
        expiry = option.expiry - option.time
        num_steps = self.stop
        normals = np.random.normal(0,1, num_steps)
        expiry_epsilon = expiry - epsilon
        exponent = (rate-vol**2/2)*expiry - vol*np.sqrt(expiry)*normals
        exponent_epsilon = (rate-vol**2/2)*expiry_epsilon - vol*np.sqrt(expiry_epsilon)*normals
        spot_at_expiry = spot * np.exp(exponent)
        spot_at_expiry_epsilon = spot * np.exp(exponent_epsilon)
        values = np.exp(-rate * expiry) * option.payoff(spot_at_expiry)
        values_epsilon =  np.exp(-rate * expiry_epsilon) * option.payoff(spot_at_expiry_epsilon)
        thetas = ( values_epsilon - values)/epsilon
        return np.mean(thetas), np.std(thetas)/np.sqrt(num_steps)





option = Call(49, 50, 1.04, .01, 3/12, 0)
option2 = Call(49, 110, 1.04, .01, 3/12, 0)

suma = 2 * option - option2 + DigitalPut(49, 85, 1.04, .01, 3/12, 0)

mc = MonteCarlo(suma, 10000)
a = mc.price()
b = suma.price()
c = hasattr(mc, "price")

print("{}".format(a))
print("{}".format(b))
print("{}".format(c))