"""A small automatic-differentiation engine on NumPy arrays (Week 1).

A Tensor holds an array of numbers, its data. A tensor computed from other
tensors also remembers

    parents        the tensors it was computed from,
    op             the name of the operation (only for printing),
    backward_rule  a function that takes the gradient of the loss with respect
                   to this tensor and returns the gradients with respect to its
                   parents: the chain rule for this one operation (in the notes,
                   a vector-Jacobian product).

loss.backward() goes through the tensors from the loss back to the inputs and
applies these rules. That is all of reverse-mode differentiation. From Week 2
on the course uses PyTorch, whose interface this engine copies.

Example:

    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    print(x.grad)          # [2. 4. 6.], since dy/dx = 2x
"""
import numpy as np


class Tensor:
    """An array that remembers how it was computed.

    data           the numbers (a NumPy array of floats)
    grad           after backward(): the gradient of the loss with respect to
                   this tensor, with the same shape as data (else None)
    requires_grad  whether gradients should be computed for this tensor
    """

    def __init__(self, data, requires_grad=False, parents=(), op=""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=float)
        self.shape = self.data.shape
        self.grad = None
        self.parents = parents
        self.op = op
        self.backward_rule = None
        # a tensor needs a gradient if we asked for one, or if any of its parents needs one
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    def __repr__(self):
        if self.requires_grad:
            return f"Tensor({self.data}, requires_grad=True)"
        return f"Tensor({self.data})"

    def item(self):
        """The value of a tensor that holds a single number, as a Python float."""
        return self.data.item()

    def detach(self):
        """A copy of the data, cut off from the graph."""
        return Tensor(self.data)

    # Every operation below follows the same pattern:
    #   1. compute the output from the inputs' data, recording the inputs as its parents;
    #   2. define backward_rule(grad): given dL/d(output), return [dL/d(parent) for each parent];
    #   3. attach the rule to the output and return it.

    # ------------------------------------------------------------------ arithmetic

    def __add__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data + other.data, parents=(self, other), op="+")
        def backward_rule(grad):
            # d(a + b)/da = 1 and d(a + b)/db = 1
            return [unbroadcast(grad, self.shape), unbroadcast(grad, other.shape)]
        out.backward_rule = backward_rule
        return out

    def __mul__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data * other.data, parents=(self, other), op="*")
        def backward_rule(grad):
            # d(a * b)/da = b and d(a * b)/db = a
            return [unbroadcast(grad * other.data, self.shape), unbroadcast(grad * self.data, other.shape)]
        out.backward_rule = backward_rule
        return out

    def __truediv__(self, other):
        other = to_tensor(other)
        out = Tensor(self.data / other.data, parents=(self, other), op="/")
        def backward_rule(grad):
            # d(a / b)/da = 1 / b and d(a / b)/db = -a / b^2
            grad_self = grad / other.data
            grad_other = -grad * self.data / other.data ** 2
            return [unbroadcast(grad_self, self.shape), unbroadcast(grad_other, other.shape)]
        out.backward_rule = backward_rule
        return out

    def __pow__(self, exponent):
        """self ** exponent, where the exponent is a plain number."""
        out = Tensor(self.data ** exponent, parents=(self,), op=f"**{exponent}")
        def backward_rule(grad):
            # d(a^n)/da = n a^(n-1)
            return [grad * exponent * self.data ** (exponent - 1)]
        out.backward_rule = backward_rule
        return out

    def __matmul__(self, other):
        """Matrix product of two 2-D tensors. Write a vector as a column, of shape (n, 1)."""
        other = to_tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("@ needs two 2-D tensors; reshape a vector to (n, 1) or (1, n)")
        out = Tensor(self.data @ other.data, parents=(self, other), op="@")
        def backward_rule(grad):
            # Z = A B with G = dL/dZ:  dL/dA = G B^T  and  dL/dB = A^T G
            return [grad @ other.data.T, self.data.T @ grad]
        out.backward_rule = backward_rule
        return out

    # Operations built from the ones above, so they need no backward rule of their own.
    # Python calls __radd__ for 2 + x, __rsub__ for 2 - x, and so on, when x is a Tensor.

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-to_tensor(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return to_tensor(other) - self

    def __rtruediv__(self, other):
        return to_tensor(other) / self

    # ---------------------------------------------------------- functions of each entry

    def exp(self):
        out = Tensor(np.exp(self.data), parents=(self,), op="exp")
        def backward_rule(grad):
            return [grad * out.data]                        # d(e^a)/da = e^a
        out.backward_rule = backward_rule
        return out

    def log(self):
        out = Tensor(np.log(self.data), parents=(self,), op="log")
        def backward_rule(grad):
            return [grad / self.data]                       # d(log a)/da = 1 / a
        out.backward_rule = backward_rule
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), parents=(self,), op="tanh")
        def backward_rule(grad):
            return [grad * (1 - out.data ** 2)]             # d(tanh a)/da = 1 - tanh(a)^2
        out.backward_rule = backward_rule
        return out

    def sigmoid(self):
        # 1 / (1 + e^-a), written with tanh so that e^-a cannot overflow for very negative a
        out = Tensor(0.5 * (1 + np.tanh(0.5 * self.data)), parents=(self,), op="sigmoid")
        def backward_rule(grad):
            return [grad * out.data * (1 - out.data)]       # d(sigmoid a)/da = sigmoid(a) (1 - sigmoid(a))
        out.backward_rule = backward_rule
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), parents=(self,), op="relu")
        def backward_rule(grad):
            return [grad * (self.data > 0)]                 # slope 1 where a > 0, else 0
        out.backward_rule = backward_rule
        return out

    # ------------------------------------------------------------------- reductions

    def sum(self, axis=None, keepdims=False):
        """Sum of all entries, or along one axis."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), parents=(self,), op="sum")
        def backward_rule(grad):
            # every entry that went into a sum gets the gradient of that sum
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)           # put back the axis the sum removed
            return [grad * np.ones(self.shape)]             # copy it to every entry
        out.backward_rule = backward_rule
        return out

    def mean(self, axis=None, keepdims=False):
        total = self.sum(axis=axis, keepdims=keepdims)
        count = self.data.size / total.data.size            # how many entries went into each sum
        return total / count

    # ------------------------------------------------------------------------ shapes

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), parents=(self,), op="reshape")
        def backward_rule(grad):
            return [grad.reshape(self.shape)]
        out.backward_rule = backward_rule
        return out

    def transpose(self):
        """Transpose of a 2-D tensor (PyTorch writes x.T)."""
        out = Tensor(self.data.T, parents=(self,), op="transpose")
        def backward_rule(grad):
            return [grad.T]
        out.backward_rule = backward_rule
        return out

    def __getitem__(self, index):
        """Entries selected by NumPy indexing, e.g. x[2], x[:, 0], or x[rows, cols]."""
        out = Tensor(self.data[index], parents=(self,), op="index")
        def backward_rule(grad):
            # each selected entry gets its gradient; the others get 0. np.add.at (instead of
            # full[index] += grad) makes an entry selected twice receive both gradients.
            full = np.zeros(self.shape)
            np.add.at(full, index, grad)
            return [full]
        out.backward_rule = backward_rule
        return out

    # ------------------------------------------------------------- softmax and friends

    def logsumexp(self, axis=-1, keepdims=False):
        """log(sum(exp(x))) along an axis, computed stably by first subtracting the maximum."""
        m = self.data.max(axis=axis, keepdims=True)
        result = m + np.log(np.exp(self.data - m).sum(axis=axis, keepdims=True))
        softmax = np.exp(self.data - result)                # the derivative of logsumexp is softmax
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        out = Tensor(result, parents=(self,), op="logsumexp")
        def backward_rule(grad):
            if not keepdims:
                grad = np.expand_dims(grad, axis)
            return [grad * softmax]
        out.backward_rule = backward_rule
        return out

    def log_softmax(self, axis=-1):
        return self - self.logsumexp(axis=axis, keepdims=True)

    def softmax(self, axis=-1):
        return self.log_softmax(axis=axis).exp()

    # ---------------------------------------------------------------------- backward

    def backward(self, grad=None):
        """Compute the gradient of this tensor with respect to every tensor it depends on.

        The results go into .grad of the leaves: the tensors created with
        requires_grad=True, such as parameters. As in PyTorch, gradients are added
        to .grad, not written over it, so clear them before each new backward pass
        (optimizer.zero_grad()).
        """
        if not self.requires_grad:
            raise RuntimeError("this tensor does not depend on any tensor with requires_grad=True")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() without an argument needs a single number; pass the gradient")
            grad = np.ones(self.shape)
        add_to_grad(self, np.array(grad, dtype=float))

        # Visit each tensor after every tensor that uses it, so that its gradient is complete.
        for t in reversed(topological_order(self)):
            if not t.parents:
                continue                                    # a leaf: its .grad is the answer
            for parent, parent_grad in zip(t.parents, t.backward_rule(t.grad)):
                if parent.requires_grad:
                    add_to_grad(parent, parent_grad)
            t.grad = None                                   # as in PyTorch, keep gradients only at leaves


# ------------------------------------------------------------------------ helpers

def to_tensor(x):
    """Wrap a number or an array as a Tensor (a constant: no gradient)."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def add_to_grad(t, grad):
    """Add a gradient contribution to t.grad. A tensor used several times receives one per use."""
    if t.grad is None:
        t.grad = grad
    else:
        t.grad = t.grad + grad


def unbroadcast(grad, shape):
    """Sum grad down to the given shape, undoing NumPy broadcasting.

    If b has shape (2,) and is added to each row of a (4, 2) array, every entry of
    b is used 4 times, so its gradient is the sum of the gradient's 4 rows.
    Broadcasting is reuse, and reused values add up their gradients.
    """
    while grad.ndim > len(shape):                           # axes that broadcasting added in front
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):                     # axes of length 1 that it stretched
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def topological_order(root):
    """The tensors that root depends on, each listed after all of its parents.

    Only tensors that need gradients are listed. This is a depth-first search.
    Python allows about 1000 nested calls, which is plenty for the graphs of
    Week 1; PyTorch walks much deeper graphs without recursion.
    """
    order = []
    visited = set()
    def visit(t):
        if t in visited or not t.requires_grad:
            return
        visited.add(t)
        for parent in t.parents:
            visit(parent)
        order.append(t)                                     # only after all of its parents

    visit(root)
    return order


# ------------------------------------------------------------------------- losses

def cross_entropy(logits, targets):
    """Average of -log softmax(logits)[target] over the rows of logits.

    logits has shape (examples, classes); targets holds one class index per example.
    """
    targets = np.array(targets, dtype=int)
    log_probs = logits.log_softmax(axis=1)
    rows = np.arange(len(targets))
    return -log_probs[rows, targets].mean()


def binary_cross_entropy_with_logits(logits, targets):
    """Average of -[y log s + (1 - y) log(1 - s)], where s = sigmoid(z) and z are the logits.

    logits and targets must have the same shape.
    """
    z = logits.data
    y = np.array(targets, dtype=float)
    if z.shape != y.shape:
        raise ValueError(f"logits have shape {z.shape} but targets have shape {y.shape}")
    # The same quantity, rearranged so that exp never overflows:
    # -[y log s + (1 - y) log(1 - s)] = max(z, 0) - y z + log(1 + e^-|z|)
    losses = np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))
    out = Tensor(losses.mean(), parents=(logits,), op="bce")
    def backward_rule(grad):
        s = 0.5 * (1 + np.tanh(0.5 * z))                    # sigmoid(z)
        return [grad * (s - y) / z.size]                    # the derivative of each loss is s - y
    out.backward_rule = backward_rule
    return out


# ------------------------------------------------------------------ checking gradients

def numerical_grad(f, x, eps=1e-6):
    """Estimate d f() / d x by central differences, nudging one entry of x at a time.

    f is a function with no arguments that returns a single-number Tensor.
    """
    grad = np.zeros(x.shape)
    for i in np.ndindex(x.shape):                           # every index of x, e.g. (0, 0), (0, 1), ...
        old = x.data[i]
        x.data[i] = old + eps
        up = f().item()
        x.data[i] = old - eps
        down = f().item()
        x.data[i] = old                                     # restore
        grad[i] = (up - down) / (2 * eps)
    return grad


def gradcheck(f, inputs, eps=1e-6):
    """Largest relative difference between autograd and central differences, over all inputs."""
    for x in inputs:
        x.grad = None
    f().backward()
    worst = 0.0
    for x in inputs:
        if x.grad is None:                                  # f does not depend on x
            auto = np.zeros(x.shape)
        else:
            auto = x.grad
        numeric = numerical_grad(f, x, eps)
        error = np.abs(auto - numeric) / np.maximum(1.0, np.abs(numeric))
        worst = max(worst, error.max())
    return float(worst)
