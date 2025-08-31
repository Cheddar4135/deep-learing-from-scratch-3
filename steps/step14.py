import numpy as np

"""step13의 문제점
같은 변수를 반복해서 사용할 경우 의도대로 동작하지 않을 수 있다.
같은 변수를 사용할 시에는 전파되는 미분값의 합을 구하도록 수정해보자.
    1. 한 연산에 동일한 변수를 반복해 사용할 시 생기는 문제 해결하자.
    2. 다른 계산에 똑같은 변수 재사용시 생기는 문제 해결하자.
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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:  # 같은 변수를 반복해 사용할 경우
                    x.grad = (
                        x.grad + gx
                    )  # x.grad+=gx처럼 쓰면 문제가 되는 경우가 있다고 한다.

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


# Test1 : 동일 변수를 사용해 역전파 계산 (문제 해결)
x = Variable(np.array(3.0))
y = add(x, x)  # y = x + x
y.backward()
print(x.grad)  # dy/dx = 2

# Test2 : 같은 변수를 사용해 '다른'계산을 할 경우 계산이 꼬이는 문제 발생
y = add(add(x, x), x)  # y = (x + x) + x
y.backward()
print(x.grad)  # dy/dx = 3 but 5가 출력된다.

"""왜 x.grad가 5가 되는가? (보충 설명)
Test2만 실행했다면 3.0이 정상적으로 출력된다.
그러나 Test1을 실행한 후, Test2를 실행하면 5가 출력된다.
이미 x.grad = 2.0이 저장되어있기 때문이다.

계산 그래프 구조
    x, x -[add1] -> t
    t, x -[add2] -> y

순전파
    x = 3
    t = 3 + 3 = 6
    y = 6 + 3 = 9

역전파과정
    1. y.backward() 호출 -> y.grad = 1.0
    2. add2의 backward
        입력: t,x
        두 입력 모두에 대해 grad가 1.0씩 전달됨
        t.grad = 1.0
        x.grad = x.grad + 1.0 = 3.0이 되어버림.
    3. add1의 backward
        입력: x,x
        두 입력 모두에 대해 grad가 t.grad=1.0씩 전달됨
        x.grad = x.grad + 1.0 (여기서 x.grad = 3.0이 이미 저장되어있음) = 4.0 (첫 번째 x)
        x.grad = x.grad + 1.0 = 4.0 + 1.0 = 5.0
    4. 최종적으로
        x.grad는 5.0d이 되어버린다.

왜 이런 문제가 생기는가?
    x가 여러 번 그래프에 등장할 때, backward를 여러 번 호출하면서 grad가 누적
    중간에 xgrad가 이미 더해진 상태에서 또 더해지기 때문

해결방법
    역전파 전에 grad를 0으로 초기화해거나, 각 연산 (test1,test2)마다 새로운 Variable을 사용해줘야한다.
    
전략
    Variable 클래스에 미분값을 초기화하는 cleargrad 메서드를 추가하고,
    새로운 연산 시에는 미분값을 꼭 초기화하도록 하자.

"""


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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None


# Test1
x = Variable(np.array(3.0))
y = add(x, x)  # y = x + x
y.backward()
print(x.grad)  # dy/dx = 2

# Test2 : 변수에 누적된 미적값을 초기화해준다.
x.cleargrad()  # 미분값 초기화
y = add(add(x, x), x)  # y = (x + x) + x
y.backward()
print(x.grad)  # dy/dx = 3 (정상 출력)

# PR Test 
# PR Test
