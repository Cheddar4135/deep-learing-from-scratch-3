""" 11.1 Function 클래스 수정
지금까지 Function 클래스는 하나의 인수만 받고 하나의 값을 반환했다.
가변 길이 입출력을 처리할 수 있다록 확장해보자.
    이번 step의 방법 : 입출력을 list로 감싸기
    이번 step 결과의 한계:
        - 사용자의 입장에서 입출력을 리스트로 담아 건네주고 받아야하는 것은 다소 귀찮다.
        - 사용자가 준 입력이 몇개인지 정확히 알아야한다.

    보완 방향(이후 step에서):
        - 함수의 입력/출력을 가변 인자(*inputs, *outputs)로 받아, 
          사용자가 리스트로 감싸지 않아도 자연스럽게 여러 인수를 처리할 수 있도록 개선할 수 있다.
        - 단일 입력/출력일 때도 리스트로 감싸지 않고 바로 값을 넘기고 받을 수 있도록 개선하면 사용성이 좋아진다.
"""
# 필요한 부분만 구현했습니다. (기존 전체 구현 코드는 review01.py 참고)
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
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs

""" 11.2 Add 클래스 구현
Add 클래스의 forward 메서드를 구현해보자.

"""

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs   # 인자 xs는 변수가 두 개 담긴 리스트라고 가정 (현 step에서는)
        y = x0 + x1
        return (y, )  
        # (y,)는 요소가 하나인 튜플을 의미한다. return (y)라고 하면 그냥 y 값 자체를 반환해버림
        # 그러면 __call__ 이 튜플의 각 요소를 접근해 Variable로 감싸서 리스트로 반환해준다.
    
xs = [Variable(np.array(2)), Variable(np.array(3))] #리스트로 준비
f = Add()
ys = f(xs)      #여기서 반환되는 것은 outputs, 즉 리스트이다. 
y = ys[0]       #따라서 여기서 첫 번째 Variable 객체를 접근한다. 
print(y.data)
