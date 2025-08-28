import numpy as np
import unittest

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError()
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        """재귀버전
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)
    
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x = f.input
            y = f.output
            """y = self (x)
            이러면 늘 역전파 테스트 결과가 1.0이 나올거임
            반복문을 돌면서 여러 Variable에 대해 backward 수행해야하는데 self.grad만 계속 참조하니까 항상 맨 처음 시작한 변수의 grad인 1.0만 사용하게 되는 것

            """
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        """output.creator = self (x)
        이렇게 다른 클래스의 인스턴스 변수에 접근해 값 설정하는 것은 객체지향적이지 않음 
        객체의 내부 상태를 외부에서 직접 변경하면 캡슐화가 깨짐
        Variable 내부 구현이 바뀌면 Function 코드도 바꿔야할 수 있음
        객체의 내부 상태는 객체 자신이 책임지고 관리해야하고, 외부에서는 메서드로만 접근해야함
        코드의 유지보수성을 위해 set_creator 같은 메서드를 통해 설정하는 것이 바람직함
        """
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self,x):
        return x**2
    
    def backward(self, gy):
        x = self.input.data
        return 2*x*gy
        
class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self,gy):
        x = self.input.data
        return np.exp(x) * gy

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


def num_diff(f,x,eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    """x0 = x - esp, x1 = x + esp (x)
    현재 x는 일반 스칼라나 np.ndarray가 아닌 Variable 클래스이다. 
    따라서 x-esp 처럼 숫자와 바로 더할 수 없다. x.data로 값을 꺼내야한다.
    또한 f는 function 클래스 인스턴스다.
    즉 들어가는 인자는 무조건 Variable 변수여야하기때문에 다시 Variable 상태로 만들어준다.

    """
    y0 = f(x0) 
    y1 = f(x1)
    return (y1.data - y0.data) / (eps * 2)

"""class TestCase01(unittest.TestCase):
    def test_func1(self, x):
        y = square(exp(square(x)))
        y.backward()
        expected = num_diff(y,x)
        self.assertEqual(expected, x.grad)
    
이 코드가 잘못된점
    1. 테스트 메서드의 인자는 self만 있어야한다.
    2. 테스트 내부에서 입력값을 직접 생성해야한다.
    3. 실수/배열 비교는 assertTrue(np.alloclose(...))을 사용해야한다.
참고 : 
    - 관례적으로 클래스명은 TestSquare 처럼 테스트대상에 맞게 작성해야함
    - 메서드는 반드시 test_ 로 시작해야 uniitest가 테스트 메서드로 인식함
"""
class TestSquare(unittest.TestCase):
    def test_backward(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = num_diff(square,x)
        flg = np.allclose(num_grad, x.grad)
        #flg는 flag의 약어로, 보통 True, False 값을 저장할 때 많이 사용하는 변수명
        self.assertTrue(flg)

if __name__ == "__main__":
    unittest.main()