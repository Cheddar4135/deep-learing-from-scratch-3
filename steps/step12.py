""" 12.1 첫번째 개선 : 함수를 사용하기 쉽게

파이썬 가변 인자 문법 (*)
    리스트를 사용하는 대신 임의 개수의 인자(가변 길이 인수)를 건네 함수를 호출할 수 있다.
    사용예시:
        def f(*x):
            print(x)
        
        f(1,2,3,4,5) # --> 1,2,3,4,5
        f(1,2)       # --> 1,2
"""

import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
""" 12.2 두번째 개선 : 함수를 구현하기 쉽게

기존 forward 메서드 코드는 인수는 리스트로 전달되고, 결과는 튜플을 반환했다.
(before)
    class Add(Function):
        def forward(self,xs):
            x0,x1 = xs
            y = x0 + x1
            return (y,)
(after)
    class Add(Function):
        def forward(self,x0,x1):
            y = x0 + x1
            return y

리스트 언팩(unpack)이란?
    리스트의 원소를 낱개로 풀어서 전달하는 기법
    예를 들어 xs = [x0, x1] 일 때 self.forward(*xs)를 하면 self.forward(x0,x1)으로 호출하는 것과 동일하게 동작한다.

왜 굳이 ys를 튜플로 바꿔주는걸까?
    리스트 컴프리헨션을 돌리기 위해, 즉 반복 가능한 객체로 만들어주기 위해서이다.
    리스트로 바꿔도 되지만, 파이썬에서 함수가 여러 값을 반환할 때 자동으로 튜플이 되기 때문에 
    예외상황인 함수가 하나만 반환할때 (ys,)처럼 튜플로 만들어주는 처리가 필요하다.
"""

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)                          #별표을 붙여 언팩(list unpack)
        if not isinstance(ys, tuple):                   #ys가 튜플이 아닌 경우(즉 하나의 값만 반환 시) 튜플로 변경한다.
            ys = (ys,)
        
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

#Add 클래스를 '파이썬 함수'로 사용할 수 있도록 코드 추가
def add(x0,x1):
    return Add()(x0,x1)


#최종 사용법
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(3.0))
y = add(x0,x1)
print(y.data)