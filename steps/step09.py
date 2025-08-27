import numpy as np

""" 9.1 파이썬 함수로 이용하기
지금까지의 DeZero는 함수를 파이썬 클래스로 정의해 사용했다.
가령 함수 Square 클래스를 사용하려면, 인스턴스 f를 생성한 다음 그 인스턴스를 호출하는 두 단계로 구분해 적어야 했다.
    x = Variable(np.array(0.5))
    f = Square()
    y = f(x)
물론 y = Square()(x) 형태로 한줄에 적을 수도 있지만, 이러면 모양새가 좋지 않다.
더 바람직한 해법은 '파이썬 함수'를 지원하는 것이다.
"""


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(
            self
        )  # 출력변수(output)에 "내가 너의 창조자임"을 기억시킨다.
        self.input = input
        self.output = output  # output 역시 인스턴스 변수에 저장해둔다.
        return output

    def forward(self, x):
        raise NotImplementedError("이 메서드는 반드시 오버라이드해야 합니다.")

    def backward(self, gy):
        raise NotImplementedError("이 메서드는 반드시 오버라이드해야 합니다.")


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# 추가한 부분
def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


# Test
x = Variable(np.array(0.5))
y = square(exp(square(x)))  # 합성함수 연속하여 적용도 가능해짐
y.grad = np.array(1.0)
y.backward()
print(x.grad)


""" 9.2 backward 메서드 간소화
목표: 테스트시 y.grad = np.array(1.0) 부분도 생략할 수 있도록 해보자.
함수설명: np.ones_like(self.data)
    self.data와 형상과 데이터타입이 같은 ndarray 인스턴스를 생성하여 모든 요소를 1로 채워 돌려준다.
    self.data가 스칼라이고 32비트 부동소수점 숫자면 self.grad도 스칼라+32비트 부동소수점 숫자 타입으로 만들어진다.
    참고로 np.array(1.0)은 64비트 부동소수점 숫자 타입으로 만들어준다.

"""


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # 개선 부분
        if self.grad is None:
            self.grad = np.ones_like(
                self.data
            )  # 만약 변수의 grad가 None이면 자동으로 미분값을 1로 생성한다.

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


""" 9.3 ndarray만 취급하기
DeZero의 Variable은 데이터로 ndarray 인스턴스만 취급하게끔 의도했다.
하지만 사용자가 float이나 int 같은 의도치 않은 데이터타입을 사용하는 경우도 있을 수 있다. ex) Variable(1.0), Variable(3)
이런 경우를 대비해 Variable에 ndarray 인스턴스 외의 데이터를 넣을 경우 즉시 오류를 일으키도록 한다.

함수 설명: isinstance(객체, 클래스)
    객체가 해당 클래스의 인스턴스면 True, 아니면 False를 반환한다.
    예시:  
        a = 3
        print(isinstance(a, int)) #True

        b = np.array([1,2,3])
        print(isinstance(b, float)) #False
참고 : 한국어로 에러 메시지 작성했더니 터미널에서 깨져보여서 영어로 바꿈. 
"""


class Variable:
    def __init__(self, data):
        # 개선부분
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}is not supported.".format(type(data)))
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


# TypeError Test
x = Variable(1.0)

"""ndarray만 취급하기(보완)
문제: 0차원 ndarray 인스턴스를 사용하여 계산하면 결과의 데이터 타입이 numpy.float64나 numpy.float32 등이 된다.
    x = np.array(1.0)
    y = x ** 2
    print(type(x, x.ndim))  # <class 'numpy.ndarray'> 0
    print(type(y))          # <class 'numpy.float64'>

    따라서 이런 상황에 대해서도 대처를 해줘야한다.
    Dezero 함수의 계산 결과(출력)가 numpy.float64나 numpy.float32가 되지 않도록 해야한다.
"""


def as_array(x):
    if np.isscalar(x):  # x가 스칼라 타입(int, float 타입 등)인지 확인
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(
            as_array(y)
        )  # 출력 결과인 output이 항상 ndarray 인스턴스가 되도록 보장한다.
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError("This method must be overridden.")

    def backward(self, gy):
        raise NotImplementedError("This method must be overridden.")
