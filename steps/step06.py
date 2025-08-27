import numpy as np

""" Variable 클래스 추가 구현
인스턴스 변수인 data와 grad는 모두 넘파이 다차원 배열 (np.ndarray)라고 가정한다.
grad는 None으로 초기화해둔 다음, 나중에 실제로 역전파를 하면 미분값을 계산하여 대입한다.

참고: 벡터나 행렬 등 다변수에 대한 미분은 기울기(gradient)라고 한다. 
"""


class Valuable:
    def __init__(self, data):
        self.data = data
        self.grad = None


""" Function 클래스 추가 구현
이전 단계까지의 Function 클래스는 일반적인 계산을 하는 순전파 (forward 메서드) 기능만 지원한 상태였다.
이번 단계에서는 학습을 위한 '역전파(backward)' 기능을 추가한다.

추가된 핵심 기능:
1. 입력된 input을 인스턴스 변수 self.input에 저장
   - forward 연산이 끝난 후에 저장한다.
   - forward가 예외로 실패했을 때는 input을 굳이 기억할 필요가 없으므로,
     "output 생성 이후"에 저장하는 것이 설계상 더 적절하다.
2. 역전파(backward) 메서드 정의
   - 각 Function은 자신이 담당하는 연산의 미분(gradient)을 계산할 수 있어야 한다.

정리:
- forward: 입력 -> 출력 (순전파)
- backward: 출력 gradient -> 입력 gradient (역전파)
"""


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Valuable(y)
        self.input = input  # 입력 변수를 기억한다.
        return output

    def forward(self, x):
        raise NotImplementedError("이 메서드는 반드시 오버라이드해야 합니다.")

    def backward(self, gy):
        raise NotImplementedError("이 메서드는 반드시 오버라이드해야 합니다.")


""" Square와 Exp 클래스 추가 구현

backward 메서드의 역할:
- gy: 출력쪽에서 흘러온 gradient. 즉, 역전파 과정에서 전해지는 '∂L/∂output' (출력에 대한 미분값)
- x : forward 시 입력값 (self.input.data로 접근).
- gx: 입력변수(input)의 gradient. 즉, '∂L/∂input' 

체인 룰(Chain Rule):
- 전체 미분은 국소 미분(local gradient) * 전파된 미분(gy) 로 계산된다.
- gx = gy * f'(x)

즉, 각 단계에서 local gradient와 gy를 곱해 나가는 것이 backward 메서드.
x.grad <- [f.backward] <- y.grad 

Square 클래스
- forward: y = x^2
- backward: dy/dx = 2x
  따라서, gx = gy * (2x)

Exp 클래스
- forward: y = exp(x)
- backward: dy/dx = exp(x)
  따라서, gx = gy * exp(x)
"""


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


""" Test
x -> [A] -> a -> [B] -> b -> [C] -> y
x.grad <- [A.backward] <- a.grad <- [B.backward] <- b.grad <- [C.backward] <- y.grad (=1)
"""
# 순전파 진행
A = Square()
B = Exp()
C = Square()

x = Valuable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 역전파 진행
y.grad = np.array(1.0)  # 역전파는 dy/dy = 1에서 시작한다.
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)


"""Problem
역전파 순서에 맞춰 backward를 호출하는 코드를 우리가 일일이 작성해 넣는 건 불편하다.
다음단계에서 이를 자동화해보자.
"""
