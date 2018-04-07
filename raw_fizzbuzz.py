length = 99
arr = [0] * (length-1)
i = 1
while i < length:
    if n % 3 == 0 and n % 5 == 0:
        arr[n-1] = 'FizzBuzz'
    elif n % 3 == 0:
        arr[n-1] = 'Fizz'
    elif n % 5 == 0:
        arr[n-1] = 'Buzz'
    else:
        arr