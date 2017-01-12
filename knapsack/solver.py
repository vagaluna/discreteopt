#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])
Node = namedtuple("Node", ['level', 'flags', 'value', 'room', 'est'])


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # output_data = trivial(items, capacity)

    if len(items) < 400:
        obj, opt, solution = dp(items, capacity)
    else:
        obj, opt, solution = bs(items, capacity)

    output_data = str(obj) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def trivial(items, capacity):
    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def dp(items, capacity):
    K = capacity
    n = len(items)
    prev = [0 for i in range(0, K + 1)]
    curr = [0 for i in range(0, K + 1)]
    flag = [0 for i in range(0, n)]

    for i in xrange(0, n):
        item = items[i]
        for k in xrange(0, K + 1):
            w = item.weight
            if w <= k:
                v1 = prev[k]
                v2 = item.value + prev[k - w]
                if v2 > v1:
                    curr[k] = v2
                    flag[i] |= 1 << k
                else:
                    curr[k] = v1
            else:
                curr[k] = prev[k]
        prev, curr = curr, prev

    k = K
    solution = [0] * n
    for i in xrange(n-1, -1, -1):
        mask = 1 << k
        if flag[i] & mask > 0:
            solution[i] = 1
            k = k - items[i].weight
        else:
            solution[i] = 0

    return prev[K], 1, solution


def bs(items, capacity):
    sorted_items = sorted(items, key=lambda item: item.value * 1.0 / item.weight, reverse=True)
    N = len(items)
    MAX_QUEUE_SIZE = 5000 if N < 10000 else 500

    def estimate_optimal(taken_flags, room):
        # room should be positive
        est = 0
        n = len(taken_flags)
        remain = room
        for item in sorted_items:
            i = item.index
            if i >= n:
                if item.weight <= remain:
                    est += item.value
                    remain -= item.weight
                else:
                    frac = remain * 1.0 / item.weight
                    est += item.value * frac
                    break
        return est

    def expand(parent, taken):
        level = parent.level + 1
        if not taken:
            room = parent.room
            value = parent.value
            flags = parent.flags + [0]
        else:
            room = parent.room - items[level - 1].weight
            value = parent.value + items[level - 1].value
            flags = parent.flags + [1]

        if room < 0:
            value = None
            est = None
        elif room == 0:
            est = value
        else:
            est = value + estimate_optimal(flags, room)
        child = Node(level, flags, value, room, est)
        return child

    def insert_node(node, queue, max_queue_size):
        n = len(queue)
        if n == 0:
            queue.append(node)
            return

        e = node.est
        low = 0
        high = n
        pos = None
        while low <= high:
            if low == high:
                pos = low
                break

            mid = (low + high) / 2
            mid_est = queue[mid].est
            pre = mid - 1
            if pre >= 0:
                pre_est = queue[pre].est
                if pre_est < e:
                    high = mid - 1
                elif mid_est > e:
                    low = mid + 1
                else:
                    pos = mid
                    break
            else:
                if mid_est <= e:
                    pos = 0
                    break
                else:
                    low = mid + 1
        queue.insert(pos, node)

        if len(queue) > max_queue_size:
            del queue[n]

    best_queue = []
    root_est = estimate_optimal([], capacity)
    root = Node(0, [], 0, capacity, root_est)
    insert_node(root, best_queue, MAX_QUEUE_SIZE)

    best_value = 0
    best_solution = None

    while len(best_queue) != 0:
        best = best_queue.pop(0)
        if best.level == N or best.est <= best_value:
            continue

        # if best.level < N and best.est > best_value:
        for c in [expand(best, True), expand(best, False)]:
            if c.value is None:
                continue

            if c.level == N and c.value > best_value:
                best_value = c.value
                best_solution = c.flags
            elif c.level < N and c.est > best_value:
                insert_node(c, best_queue, MAX_QUEUE_SIZE)

    return best_value, 0, best_solution


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

