from math import sqrt

w = int(input())

c = 0

if w % 2 == 0:

    for n in range(2, int(sqrt(w)) + 1, 2):

        if w % n == 0:

            if (w / n) % 2 == 0:
                c += 1

else:

    for n in range(1, int(sqrt(w)) + 1, 2):

        if w % n == 0:

            if (w / n) % 2 == 1:
                c += 1

print(c)


