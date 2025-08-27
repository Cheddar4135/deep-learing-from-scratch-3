# Before (재귀)
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

# After (반복문)
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
                funcs.append(x.creator)   #하나 앞의 함수를 리스트에 추가한다.


"""왜 굳이 반복문으로 구현?
1. 구현 확장성
    15단계에서 알 수 있다.
    복잡한 계산 그래프를 다룰 때, 방금 전환한 구현 덕분에 부드럽게 확장할 수 있다.
2. 처리 효율
    재귀는 함수를 재귀적으로 호출할 때마다 중간 결과를 스택에 쌓으면서 처리를 이어간다.
    반복문은 이러한 오버헤드가 없고 메모리 사용량이 더 적다.
    또한 재귀 호출이 너무 깊어지만 (호출 횟수가 너무 많아지면) 파이썬에서는 RecursionError가 발생할 수 있다.
    반복문은 이런 제한이 없고, 함수 호출과 반환에 드는 시간이 드는 재귀보다 실행 속도가 더 빠를 수 있다.

"""