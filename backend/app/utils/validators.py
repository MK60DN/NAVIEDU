import re
from typing import Optional


def validate_email(email: str) -> bool:
    """验证邮箱格式"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None


def validate_username(username: str) -> bool:
    """验证用户名格式"""
    # 3-20个字符，只允许字母数字和下划线
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None


def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """验证密码强度"""
    if len(password) < 6:
        return False, "密码至少需要6个字符"

    if len(password) > 100:
        return False, "密码不能超过100个字符"

    # MVP版本简化密码要求，正式版可以增加复杂度要求
    return True, None


def validate_amount(amount: float) -> bool:
    """验证金额"""
    return amount > 0 and amount <= 10000


def sanitize_html(text: str) -> str:
    """清理HTML内容，防止XSS攻击"""
    # 简单的HTML标签清理
    import html
    return html.escape(text)


def validate_capsule_content(content: str) -> tuple[bool, Optional[str]]:
    """验证知识胶囊内容"""
    if len(content) < 10:
        return False, "内容至少需要10个字符"

    if len(content) > 5000:
        return False, "内容不能超过5000个字符"

    return True, None


def validate_code(code: str) -> tuple[bool, Optional[str]]:
    """验证代码内容"""
    if code and len(code) > 2000:
        return False, "代码不能超过2000个字符"

    # 检查是否包含危险代码
    dangerous_patterns = [
        r'__import__',
        r'eval\(',
        r'exec\(',
        r'compile\(',
        r'globals\(',
        r'locals\(',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return False, "代码包含不安全的内容"

    return True, None