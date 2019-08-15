list1 = ['Google', 'Runoob', 1997, 2000]
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]

print('list1:', list1)
print('list2:', list2)
print('list3:', list3)

print('list3[1]:', list3[1])
print('len(list3):', len(list3))
print('max(list3):', max(list3))
print('min(list3):', min(list3))

list3.append('e')
print('After list3.append(\'e\'):', list3)

list3.pop()
print('After list3.pop():', list3)

list3.remove('b')
print('After list3.remove(\'b\'):', list3)
