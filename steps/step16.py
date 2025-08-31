"""목표: 역전파 함수 호출 순서 개선 :  generation(세대) 기록 방식

문법 설명 : lambda
    이름 없는 함수를 만드는 파이썬 문법이다.
    간단한 함수를 한 줄로 빠르게 만들 때 유용하다.

    예를 들어 def add(a,b,): return a+b --> add = lambda a,b: a+b 로 표현할 수 있다.

    1. 리스트 정렬 기준으로 사용
        lst = [(1, 'a'), (3, 'c'), (2, 'b')]
        lst.sort(key=lambda x: x[0])        # 원소 x를 입력받아 x[0]을 반환한 걸 기준으로 정렬
        print(lst)                          # [(1, 'a'), (2, 'b'), (3, 'c')]

    2. map 함수와 함께 사용
        map(함수, 리스트) : 리스트의 각 원소에 함수를 적용해서 이터러블 객체를 반환한다.
            nums = [1,2,3]
            squard = list(map(lambda x: x**2, nums))
            print(squard)                       # [1,4,9]

    3. filter 함수와 함께 사용
        filter(함수, 리스트) : 리스트의 각 원소에 함수를 적용해서, True 인 값만 남긴다.
            nums = [1,2,3,4]
            even = list(filter(lambda x: x % 2 == 0, nums))
            print(even)                         # [2,4]

"""

import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported.".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0  # 인스턴스 변수 generation 추가, 0으로 초기화

    def set_creator(self, func):
        self.creator = func
        self.generation = (
            func.generation + 1
        )  # 부모 함수의 세대보다 1만큼 큰 값으로 설정한다.

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)  # 아하 list에 원소 추가는 append, set에 원소 추가는 add
                funcs.sort(
                    key=lambda x: x.generation
                )  # 각 원소 x (Function 객체)의 generation 값을 기준으로 정렬

        add_func(self.creator)

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
                    add_func(x.creator)  # 수정 전 : funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max(
            [x.generation for x in inputs]
        )  # 입력 변수 세대 중 가장 큰 값을 선택한다.
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
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


# Test : y = (x^2)^2 + (x^2)^2 = 2x^4
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)


""" 발전 : heapq 사용한 버전

1. heapq.heappush(heap, (-f.generation, ..., f))
    파이썬 heapq는 최소 힙
    따라서 “generation이 클수록 먼저”를 만족하기 위해 우선순위를 -generation으로 넣는다.
    그러면 heappop()이 가장 큰 generation을 가진 함수를 먼저 꺼내게 된다.
2. heapq.heappush(heap, (-f.generation, id(f), f)) --> id(f)???
    힙에 넣는 항목이 튜플이면, 첫 요소가 같을 때 두 번째 요소로 비교되는데, 이 두번째 요소가 비교 가능 객체가 아니면 에러가 날 수 있다.
    따라서 같은 generation끼리의 비교 시 에러를 방지하기 위해 항상 비교 가능한 정수인 id(f)를 두 번째 요소로 둔다.

3. _,_,f = heapq.heappop(heap)
    힙에 들어가는 원소는 3개짜리 튜플이다.
        1) -f.generation → 우선순위(큰 generation 먼저 나오도록 음수화)
        2) id(f) → 같은 generation끼리 충돌 방지용 tie-breaker
        3) f → 실제 Function 객체
    앞의 두 값(-generation, id(f))은 버리고, 마지막 f만 쓰기 때문에
    언더스코어 _로 관례적으로 안쓰는 변수라고 표현해준다.

"""
import heapq


class Variable:
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        heap = []
        seen_set = set()

        def add_func(f):
            if f is None:
                return
            if f not in seen_set:
                # -f.generation : 큰 generation 먼저 나오도록
                # id(f) : tie-breaker
                heapq.heappush(heap, (-f.generation, id(f), f))
                seen_set.add(f)

        add_func(self.creator)

        while heap:
            _, _, f = heapq.heappop(heap)

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
                    add_func(x.creator)
