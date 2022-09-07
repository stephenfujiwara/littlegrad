import math

class Value:
    def __init__(self, data, children=(), op="", label='') -> None:
        self.data = data
        self.grad = 0
        self._backward = lambda:None
        self._prev = set(children)
        self._op = op
        self.label = label
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        # if other is not Value object, wrap it to be a Value object.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op='+')

        def _backward():
            # the resulting node of every operation will store a function to compute the gradient with respect to that operation
            # and chain that "local" gradient with the gradient computed from operations after it.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward       

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward =_backward

        return out
    
    def __rmul__(self, other): # will be called when something tries to multipy with a Value object
        return self * other
    
    def __pow__(self, other): # pow function that only allows raising to a constant. x^k
        assert(isinstance(other, (int, float)))
        out = Value(self.data ** other, children=(self, ), op=f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
        
    def __truediv__(self, other):
        return self * other ** -1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, children=(self, ), op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, children=(self,), op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), children=(self, ), op='exp')

        def _backward():
            # the local gradient of e^x is just e^x
            local_grad = out.data
            # remember, self.grad is the global gradient w.r. to self,
            # local_grad is the gradient w.r.t to the operation performed
            # out.grad is the global gradient w.r. to the result node of the operation. 
            self.grad = local_grad * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        # topological sort to order vertices such that for every edge uv between u and v, u comes before v.
        # then it needs to be reversed, so backpropogation can begin from the very last vertex.
        topo = []
        visited = set()
        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_sort(child)
                topo.append(v)
        topological_sort(self)

        # set the gradient of self w.r.t self as 1
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rtruediv__(self, other): # other / self
        return other * self**-1    