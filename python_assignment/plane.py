n, m = list(map(int, input().split()))
g = list(map(int, input().split()))
p = list(map(int, input().split()))
g.sort()
p.sort()
g = g[::-1]
p = p[::-1]

if (p[0] < g[0]):
    print(-1)
else:
    i = 1
    mxt = 1
    j = 1
    time = 1
    while (i < n):
        while (p[j] < g[i]):
            time = time + 2
            i = i + 1
        mxt = max(mxt, time)
        time = 1
        i = i + 1
        j = j + 1
        if (j == m and i != n):
            time = 2 * (((n - i) // m) + ((n - i) % m))
            mxt += time
            break
    print(mxt)