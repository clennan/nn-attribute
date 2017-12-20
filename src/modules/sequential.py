
from src.modules.module import Module


class Sequential(Module):

    def __init__(self, modules):

        self.modules = modules

    def forward(self, x):
        print('-------------------------------------------------')
        print('Forward pass...')
        print('-------------------------------------------------')
        for m in self.modules:
            print(m.name+':', x.get_shape().as_list())
            x = m.forward(x)
        print('-------------------------------------------------')
        return x

    def lrp(self, R):
        print('Backpropagating relevances...')
        print('-------------------------------------------------')
        for m in self.modules[::-1]:
            print(m.name+':', R.get_shape().as_list())
            R = m.lrp(R)
        print('-------------------------------------------------')
        return R
