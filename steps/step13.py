import numpy as np
""" 가변 길이 인수(역전파 편)
수정할 곳
    1. Variable 클래스의 backward 메서드
    2. Function 클래스 상속한 각 함수 클래스 - Square, Exp의 backward 메서드
테스트
    z=x^2 + y^2 라는 계산을 z.backward()를 호출해서 미분 계산을 자동으로 처리할 것
"""

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
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]

        while funcs:
            f = funcs.pop()
            gys = [ output.grad for output in f.outputs]  # 출력변수인 outputs에 담겨있는 미분값들을 리스트에 담는다.
            gxs = f.backward(*gys)  # 함수f의 역전파를 호출한다. 이때 리스트 언팩으로 풀어서 인수 전달

            if not isinstance(gxs, tuple):  # 기본 반환형은 튜플, 단일값 반환시에도 튜플이 되도록 변환시켜준다.
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):  # f.input[i]번째 변수의 미분값은 gxs[i]에 대응된다.
                x.grad = gx  # 모든 Variable 인스턴스 각각에 알맞은 미분값을 설정한다.

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy  # 각 편미분값이 1이니까 gy 그대로 전달되는거야!


def add(x0, x1):
    return Add()(x0, x1)


##유배
class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data  # 개선된 Function 클래스에 맞춰
        return 2 * x * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data  # 개선된 Function 클래스에 맞춰
        return np.exp(x) * gy


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


# Test
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(z.grad)
print(x.grad)
print(y.grad)
