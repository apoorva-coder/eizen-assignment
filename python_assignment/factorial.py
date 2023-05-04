import threading

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def threaded_factorial(n):
    thread = threading.Thread(target=factorial, args=(n,))
    thread.start()
    thread.join()

if __name__ == '__main__':
    n = int(input("Enter a number: "))
    threaded_factorial(n)
