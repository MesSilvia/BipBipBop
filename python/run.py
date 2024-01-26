
import subprocess
import os
import time



if not os.path.exists("..\\python\\.venv"):
    #Run if .venv still doesn't exists
    print("Installing required packages (Wait time: 140 s)......") 
    subprocess.run(['python','-m', 'venv', '..\\python\\.venv'], shell = True)
    subprocess.run(['..\\python\\.venv\\Scripts\\Activate.bat','&&', 'pip','install', '-r', '..\\python\\requirements.txt', '&&', 'python', '..\\python\\eval.py'], shell = True)
else: 
    #Run if .venv exists 
    print("Communication setup (Wait time: 20 s)")
    subprocess.run(['..\\python\\.venv\\Scripts\\Activate.bat', '&&', 'python', '..\\python\\eval.py'], shell = True)
# '&&', '..\\python\\.venv\\Scripts\\Activate.bat', '&&', 'pip install -r ...\\python\\requirements.txt', '&&', 'python', '..\\python\\eval.py'])                    




