
import subprocess
import os
import time
print("Python setup")


if not os.path.exists("..\\python\\.venv"):
    print("Check virtual install") 
    subprocess.run(['python','-m', 'venv', '..\\python\\.venv'], shell = True)
    subprocess.run(['..\\python\\.venv\\Scripts\\Activate.bat','&&', 'pip','install', '-r', '..\\python\\requirements.txt', '&&', 'python', '..\\python\\eval.py'], shell = True)
else: 
    subprocess.run(['..\\python\\.venv\\Scripts\\Activate.bat', '&&', 'python', '..\\python\\eval.py'], shell = True)
# '&&', '..\\python\\.venv\\Scripts\\Activate.bat', '&&', 'pip install -r ...\\python\\requirements.txt', '&&', 'python', '..\\python\\eval.py'])                    




