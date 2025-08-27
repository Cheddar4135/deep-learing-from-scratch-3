import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError("이 메서드는 반드시 오버라이드해야 합니다.")


class Square(Function):
    def forward(self, x):
        return x**2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_dff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


"""test
y=x^2 함수에서 x = 2.0일때 수치 미분한 결과 구하기
"""
f = Square()
x = Variable(np.array(2.0))
dy = numerical_dff(f, x)
print(dy)

"""test
합성함수의 미분
참고: 함수도 객체이기 때문에 다른 함수에 인자로 전달할 수 있다.
"""


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_dff(f, x)
print(dy)

"""수치미분의 문제점
1. 자릿수 누락으로 인한 오차 포함
2. 어마무시한 계산량
신경망에서는 매개변수를 수백만 개 사용하게 되는데 이 모두를 수치 미분으로 구하는 건 쉽지 않다.
그래서 역전파가 등장했다. 역전파는 복잡한 알고리즘이라 정확한 구현 성공했는지 확인하기 위해 수치 미분 결과와 비교하기도. (=gradient checking) 이 기울기 확인은 10단계에서 구현한다.
"""
