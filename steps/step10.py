"""10.1 파이썬 단위 테스트
uniitest
    파이썬 표준 라이브러리에서 제공하는 단위 테스트 프레임워크
    단위 테스트란? 코드의 작은 단위(함수, 클래스 등)가 예상대로 동작하는지 자동으로 검사하는 테스트
    사용방법
        1. 테스트 클래스 작성
            unittest.TestCase 를 상속받아 테스트 클래스를 만든다.
            테스트할 메서드의 이름을 test_로 시작하게 만든다.
        2. assert 메서드로 결과 검증
            self.assertEqual, self.assertTrue 등 다양한 assert 메서드로 결과를 확인한다.
        3. 테스트 실행
            파일 맨 아래 아래 코드를 추가한다.
            if __name__ == '__main__' :
                unittest.main()
            터미널에서 해당 파일을 실행하면 테스트가 자동으로 수행된다.
"""

import unittest
import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


# square함수의 forward, backward 테스트
# 이렇게 테스트 케이스가 많아질수록 square 함수의 신뢰도도 높아진다.
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)


""" 10.3 gradient 확인을 위한 자동 테스트
앞 절에서는 역전파 테스트를 작성하며 미분의 기댓값을 손으로 계산해 입력했다. 이부분을 수치 미분을 이용해 자동화해보자.
    1. 기울기 확인을 할 무작위 입력값을 하나 생성한다.
    2. 역전파로 미분값을 구하고, numerical_diff 함수를 사용해 수치 미분으로도 계산해본다.
    3. 두 메서드로 각각 구한 값들이 거의 일치하는지 확인한다.

함수설명: np.random.rand(1)
    0 이상 1 미만의 난수(무작위 실수) 1개를 원소로 갖는 1차원 넘파이 배열 (numpy.ndarray)을 생성합니다.
        arr = np.random.rand(1)
        print(arr)  # 예: [0.37454012]
함수설명: np.allclose(a,b)
    ndarray 인스턴스 a,b의 값이 가까운지 판정한다.
    얼마나 가까워야하는지는 np.allclose(a, b, rtol=1e-5, atol=le-08) 과 같이 인수 rtol과 atol로 지정할 수 있다.
    가까우면 True, 다르다 판정되면 False를 반환한다.
"""


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    # 추가 부분
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


if __name__ == "__main__":
    unittest.main()
