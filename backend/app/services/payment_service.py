from sqlalchemy.orm import Session
from app.models import User, Transaction
from typing import Dict, Any
import hashlib
import time


class PaymentService:
    """支付服务 - MVP版本仅模拟支付流程"""

    @staticmethod
    def create_payment_order(
            db: Session,
            user_id: str,
            amount: float,
            payment_method: str
    ) -> Dict[str, Any]:
        """创建支付订单（模拟）"""
        order_id = hashlib.md5(f"{user_id}{amount}{time.time()}".encode()).hexdigest()

        # MVP版本直接返回成功，实际应接入支付SDK
        return {
            "order_id": order_id,
            "amount": amount,
            "payment_method": payment_method,
            "status": "pending",
            "payment_url": f"https://payment.example.com/pay/{order_id}"  # 模拟支付链接
        }

    @staticmethod
    def verify_payment(
            db: Session,
            order_id: str,
            user_id: str
    ) -> bool:
        """验证支付（模拟）"""
        # MVP版本直接返回成功
        # 实际应该调用支付平台API验证
        return True

    @staticmethod
    def process_recharge(
            db: Session,
            user_id: str,
            amount: float,
            payment_method: str,
            order_id: str = None
    ) -> Dict[str, Any]:
        """处理充值"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        # 增加E币余额
        user.e_coin_balance += amount

        # 记录交易
        transaction = Transaction(
            user_id=user_id,
            type="recharge",
            amount=amount,
            currency="E_COIN",
            description=f"充值 {amount} E币 - {payment_method}",
            status="completed"
        )
        db.add(transaction)
        db.commit()

        return {
            "success": True,
            "new_balance": user.e_coin_balance,
            "transaction_id": transaction.id
        }


payment_service = PaymentService()