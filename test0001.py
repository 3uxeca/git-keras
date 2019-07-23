
# import tensorflow as tf
# import keras


# print("Hello VSCode")
# a,b = map(int, (input().split()))
# print(a+b)

# def hello():
#     global hi
#     hi = "hello"

# def hello2():
#     print(hi)

# hi = "freeristea"
# hello()
# hello2()

# def sum_many(*args):
#     sum = 1
#     for i in args:
#         sum *= i
    
#     return sum
# print(sum_many(1,2,3))


# 난수를 사용한 예제. 난수를 생성한 후 어떤 숫자인지 맞추는 프로그램!
from random import *

n = randint(1, 100)

while True:
    ans = input("Guess my number (Q to exit): ")
    if ans.upper() == "Q":
        break
    ians = int(ans)
    if (n == ians):
        print("Correct!")
        break
    elif (n > ians):
        print("Choose higher number")
    else:
        print("Choose lower number")
