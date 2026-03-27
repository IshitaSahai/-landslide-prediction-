import shutil
src = r'C:\Users\WIN 10\.gemini\antigravity\brain\17cb129e-9f07-4018-87f8-46152fec0401\landslide_mitigation_1774625559740.png'
dst = r'c:\Users\WIN 10\Downloads\landslide\static\mitigation.png'
try:
    shutil.copy2(src, dst)
    print("Successfully copied image")
except Exception as e:
    print(f"Error: {e}")
