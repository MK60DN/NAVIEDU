import sys
print("Python version:", sys.version)
print("-" * 50)

# 测试导入
try:
    from app import create_app
    print("✓ App module imported successfully")
    app = create_app()
    print("✓ App created successfully")
except ImportError as e:
    print(f"✗ Import Error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()