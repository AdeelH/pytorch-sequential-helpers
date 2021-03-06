# pytorch-sequential-helpers
Some helper modules that allow complex networks (particulary those with parallel data flows) to be expressed as a single Sequential.

# Example

**Pass same input to 2 different NN branches and merge the results by adding**
```{python3}
nn.Sequential(
  Parallel(branch1, branch2),
  Add()
)
```

**Split the RGB channels of a batch of images, pass each to a different NN branch, and concat the results**
```{python3}
nn.Sequential(
  Split((1, 1, 1), dim=1),
  Parallel(branch1, branch2, branch3),
  Concat(dim=1)
)
```
