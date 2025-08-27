import numpy as np

"""7.1 역전파 자동화의 시작

역전파 자동화
    목표: 순전파를 한 번만 해주면 어떤 계산이라도 상관없이 역전파가 한번에 자동으로 이루어지는 구조를 만들자.

Variable과 Function 클래스 연결시키기
    함수 입장에서 본 변수는 : 입력변수(input)과 출력변수(output)
    변수 입장에서 본 함수는 : 변수 자신을 만들어준 부모이자 창조자 (물론 창조자인 함수가 존재하지 않는 변수 - 예컨대 사용자에 의해 만들어진 변수 등 -도 있지만)

"""

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None  # 인스턴스 변수 추가

    def set_creator(self, func):  # 매서드 추가
        self.creator = func


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # 출력변수(output)에 "내가 너의 창조자임"을 기억시킨다.
        self.input = input
        self.output = output      # output 역시 인스턴스 변수에 저장해둔다.
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

"""Test

assert문
    조건이 참인지 검사하는 파이썬의 내장 명령문
    조건이 참이면 아무 일도 일어나지 않고, 거짓이면 AssertionError 예외가 발생한다.
    주로 디버깅이나 테스트 단계에서 "이 조건이 반드시 성립해야 한다"는 것을 확인할 때 사용한다.
    즉, 코드의 동작이 예상대로 되는지 자동으로 체크하는 용도로 사용한다.

"""
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 거꾸로 거슬러 올라가며 변수와 함수 사이 연결이 잘 되어있는지 확인한다.
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

"""7.2 역전파 도전!
변수와 함수의 관계를 이용해 역전파를 시도해보자.
    1. 함수를 가져온다.
    2. 함수의 입력을 가져온다.
    3. 함수의 backward 메서드를 호출한다.
"""
# b -> [C] -> y
# b.grad <- [C.backward] <- y.grad (=1)
y.grad = np.array(1.0)
C = y.creator                   # 1. 함수를 가져온다.
b = C.input                     # 2. 함수의 입력을 가져온다.
b.grad = C.backward(y.grad)     # 3. 함수의 backward 메서드를 호출한다.

# a -> [B] -> b
# a.grad <- [B.backward] <- b.grad
B = b.creator
a = B.input
a.grad = B.backward(b.grad)

# x -> [A] -> a
# x.grad <- [A.backward] <- a.grad
A = a.creator
x = A.input
x.grad = A.backward(a.grad)

print(x.grad)

"""7.3 backward 메서드 추가
이제 7.2 반복작업을 자동화해보자.
Variable 클래스에 backward라는 새로운 메서드를 추가한다.
    1. Variable의 creator에서 함수를 얻어오고,
    2. 그 함수의 입력을 가져온다.
    3. 함수의 backward 메서드를 호출한다.
    4. 자신보다 하나 앞에 놓인 변수의 backward 메서드를 호출한다.
    만약 Variable 인스턴스의 creator가 None이면 역전파가 중단된다.
    창조주가 없으므로 이 Variable 인스턴스는 함수 바깥에서 생성됐음을 뜻한다.(높은 확률로 사용자가 만들어 건넨 변수)
"""
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None  

    def set_creator(self, func):  
        self.creator = func
    
    def backward(self):
        f = self.creator        # 1. 함수를 가져온다.
        if f is not None:
            x = f.input         # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad)  # 3. 함수의 backward 메서드를 호출한다.
            x.backward()        # 하나 앞 변수의 backward 메서드를 호출한다.(재귀)

"""Test
변수 y의 backward 메서드를 호출하면 역전파가 한번에 자동으로 진행된다. - 자동 미분  완성!
"""
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#역전파
y.grad = np.array(1.0)
y.backward()
print(x.grad)