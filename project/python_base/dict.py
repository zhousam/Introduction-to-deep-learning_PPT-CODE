dict = {'Name': 'Runoob', 'Age': 7}

dict['Age'] = 8  # 更新 Age
dict['School'] = "MIT"  # 添加信息
print("dict['Age']: ", dict['Age'])
print("dict['School']: ", dict['School'])

del dict['Name']  # 删除键 'Name'
print('Afterdel dict[\'Name\']:', dict)
dict.clear()  # 清空字典
print('Afterdel dict.clear():', dict)
del dict  # 删除字典
# print("dict['School']: ", dict['School'])

