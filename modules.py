import torch
from torch import nn
from functools import reduce
from operator import __mul__


class Parallel(nn.ModuleList):
	''' Passes inputs through multiple `nn.Module`s in parallel. Returns a tuple of outputs. '''

	def __init__(self, *args):
		super().__init__(args)

	def forward(self, xs):
		if isinstance(xs, torch.Tensor):
			return tuple(m(xs) for m in self)
		assert len(xs) == len(self)
		return tuple(m(X) for m, X in zip(self, xs))


class SplitTensor(nn.Module):
	''' Wrapper around `torch.split` '''

	def __init__(self, size_or_sizes, dim):
		super().__init__()
		self.size_or_sizes = size_or_sizes
		self.dim = dim

	def forward(self, X):
		return X.split(self.size_or_sizes, dim=self.dim)


class SliceTensor(nn.Module):
	''' Applies the specified slices to the tensor. '''

	def __init__(self, slices):
		super().__init__()
		self.slices = tuple(slice(*s) for s in slices)

	def forward(self, x):
		return x[self.slices]

class Clone(Parallel):
	def __init__(self, n):
		super().__init__(*[nn.Identity() for _ in range(n)])

class ConcatTensors(nn.Module):
	''' Wrapper around `torch.cat` '''
	def __init__(self, dim=1):
		super().__init__()
		self.dim = dim

	def forward(self, xs):
		return torch.cat(xs, dim=self.dim)


class AddTensors(nn.Module):
	''' Adds all its inputs together. '''

	def forward(self, xs):
		return sum(xs)


class ElementwiseMultiplyTensors(nn.Module):
	''' Elementwise multiplies all its inputs together. '''
	def forward(self, xs):
		return reduce(__mul__, xs)


class MergeDictsLambda(nn.Module):
	def __init__(self, fn, keys=['out']):
		super().__init__()
		self.fn = fn
		self.keys = keys

	def forward(self, dicts):
		out = {}
		for key in self.keys:
			out[key] = self.fn(tuple(X[key] for X in dicts))
		return out
