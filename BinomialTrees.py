import numpy as np
from BlackScholes import *

NUMER_OF_STEPS = 200

class BinomialTree:
	def __init__(self, S, vol, r, T, t, num_steps = NUMER_OF_STEPS):
		self.spot = S 
		self.vol = vol 
		self.rate = r 
		self.expiry = T 
		self.time = t
		self.num_steps = num_steps
		self.time_delta = (T-t)/num_steps

	def up(self):
		vol = self.vol
		T = self.expiry
		t = self.time
		num_steps = self.num_steps
		time_delta = self.time_delta
		return  np.exp(vol * np.sqrt(time_delta))

	def down(self):
		return 1 / self.up()

	def exponent(self):
		time_delta = self.time_delta
		r = self.rate
		return np.exp(r * time_delta)

	def prob(self):
		a = self.exponent()
		return (a - self.down())/(self.up() - self.down())

	def make_tree(self):
		num_steps = self.num_steps + 1
		time_delta = self.time_delta
		up = self.up()
		down = self.down()
		tree = np.zeros([num_steps, num_steps]) #np.empty([num_steps, num_steps]) is faster
		for j in range(num_steps):
			for i in range(j+1):
				tree[i,j] = self. spot * up**(j-i) * down**(i)
		return tree


class AmericanCall(BlackScholes):
	def __init__(self, S, K, vol, r, T, t):
		super().__init__(S, vol, r, T, t)
		self.strike = K

	def make_tree(self, num_steps = NUMER_OF_STEPS):
		S = self.spot
		vol = self.vol
		r = self.rate
		T = self.expiry
		t = self.time
		TreeObj = BinomialTree(S, vol, r, T, t, num_steps)
		time_delta = TreeObj.time_delta
		prob = TreeObj.prob()
		discount = 1/TreeObj.exponent()
		tree = TreeObj.make_tree()
		func = lambda x : np.maximum(x - self.strike, 0)
		for i in range(num_steps + 1):
			tree[i, num_steps] = func(tree[i, num_steps])
		for j in range(num_steps - 1, -1 , -1):
			for i in range(j + 1):
				x = discount * (prob * tree[i, j+1]  +(1-prob) * tree[i+1, j+1])
				tree[i, j] = np.maximum(x, func(tree[i,j]))
		return tree

	def price(self, num_steps = NUMER_OF_STEPS):
		return self.make_tree(num_steps)[0,0]


class AmericanPut(BlackScholes):
	def __init__(self, S, K, vol, r, T, t):
		super().__init__(S, vol, r, T, t)
		self.strike = K

	def make_tree(self, num_steps = NUMER_OF_STEPS):
		S = self.spot
		vol = self.vol
		r = self.rate
		T = self.expiry
		t = self.time
		TreeObj = BinomialTree(S, vol, r, T, t, num_steps)
		time_delta = TreeObj.time_delta
		prob = TreeObj.prob()
		discount = 1/TreeObj.exponent()
		tree = TreeObj.make_tree()
		func = lambda x : np.maximum(self.strike - x, 0)
		for i in range(num_steps + 1):
			tree[i, num_steps] = func(tree[i, num_steps])
		for j in range(num_steps - 1, -1 , -1):
			for i in range(j + 1):
				x = discount * (prob * tree[i, j+1]  +(1-prob) * tree[i+1, j+1])
				tree[i, j] = np.maximum(x, func(tree[i,j]))
		return tree

	def price(self, num_steps = NUMER_OF_STEPS):
		return self.make_tree(num_steps)[0,0]
