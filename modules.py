class Split(nn.Module):
	''' Wrapper around `torch.split` '''
	def __init__(self, size, dim):
		super(Split, self).__init__()
		self.size = size
		self.dim = dim

	def forward(self, X):
		return X.split(self.size, self.dim)

class Parallel(nn.ModuleList):
	''' Passes inputs through multiple `nn.Module`s in parallel. Returns a tuple of outputs. '''

	def forward(self, Xs):
		if isinstance(Xs, torch.Tensor):
			return tuple(m(Xs) for m in self)
		assert len(Xs) == len(self)
		return tuple(m(X) for m, X in zip(self, Xs))

def Clone(n=2):
	return Parallel(nn.Identity() for _ in range(n))

class Concat(nn.Module):
	''' Concatenates an iterable input of tensors along `dim` '''
	def __init__(self, dim=1):
		super(Concat, self).__init__()
		self.dim = dim

	def forward(self, Xs):
		return torch.cat(Xs, dim=self.dim)

class Add(nn.Module):
	''' Sums an iterable input of tensors '''
	def forward(self, Xs):
		return sum(Xs)

class AddDict(nn.Module):
	def __init__(self, keys=['out']):
		super(AddDict, self).__init__()
		self.keys = keys

	def forward(self, Xs):
		out = OrderedDict()
		for key in self.keys:
			out[key] = sum(X[key] for X in Xs)
		return out
