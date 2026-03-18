import os
try:
    os.makedirs("test_models", exist_ok=True)
    with open("test_models/success.txt", "w") as f:
        f.write("Directory creation worked")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
