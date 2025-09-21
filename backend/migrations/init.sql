-- 初始化数据脚本

-- 插入默认的Python代币
INSERT INTO tokens (id, name, symbol, price, total_supply)
VALUES ('default-python-token', 'Python Token', 'PYTHON', 1.0, 1000000.0);

-- 创建管理员用户（密码: admin123）
INSERT INTO users (id, username, email, hashed_password, is_admin, e_coin_balance, python_token_balance)
VALUES (
    'admin-user-id',
    'admin',
    'admin@edupath.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY/MrLJFQa1tuFW',
    1,
    1000.0,
    1000.0
);

-- 插入示例知识胶囊
INSERT INTO capsules (id, title, description, content, code, category, author_id, status)
VALUES
('capsule-1', 'Python变量和数据类型', '了解Python中的基本数据类型',
'Python中有多种数据类型，包括整数(int)、浮点数(float)、字符串(str)、布尔值(bool)等。变量是存储数据的容器，在Python中不需要声明变量类型。',
'# 整数
age = 25
# 浮点数
price = 19.99
# 字符串
name = "Alice"
# 布尔值
is_student = True',
'基础语法', 'admin-user-id', 'approved'),

('capsule-2', 'Python列表', '学习Python中的列表数据结构',
'列表是Python中最常用的数据结构之一，可以存储多个元素，元素可以是不同类型。列表是可变的，可以添加、删除、修改元素。',
'# 创建列表
fruits = ["apple", "banana", "orange"]
# 访问元素
print(fruits[0])  # apple
# 添加元素
fruits.append("grape")
# 删除元素
fruits.remove("banana")',
'数据结构', 'admin-user-id', 'approved');