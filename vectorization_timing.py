import numpy as np
import time

a = np.random.rand(999999)
b = np.random.rand(999999)

print(a)
print(b)

solution = 0
start = time.time()
for i in range(999999):
    solution += a[i]*b[i]
finish = time.time()
for_loop_time = 1000*(finish-start)

print(f'Solution is {solution}')
print(f'For loop: {for_loop_time}ms\n')

start = time.time()
solution = np.dot(a, b)  # Vectorization
finish = time.time()
vectorization_time = 1000*(finish-start)

print(f'Solution is {solution}')
print(f'Vectorization: {vectorization_time}ms\n')

print(f'Vectoriztion is {for_loop_time/vectorization_time} times faster than for loop.')
