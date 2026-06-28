import platform
import os
from datetime import datetime

def createmcdir():
    os_name = platform.system()

    if os_name == "Windows":
        #print("You are using Windows", end=' - ')
        fp = os.path.join(os.path.expanduser("~"), 'Desktop', 'pypscresults', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))
        os.makedirs(fp)
    elif os_name == "Linux":
        #print("You are using Linux.", end=' - ')
        # Get current working directory
        fp = os.path.join(os.getcwd(), 'pypscresults', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))
        #fp = os.path.join(os.path.expanduser("~"), 'pypscresults', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))
        os.makedirs(fp)
    elif os_name == "Darwin":
        #print("You are using macOS.", end=' - ')
        fp = os.path.join(os.getcwd(), 'pypscresults', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))
        #fp = os.path.join(os.path.expanduser("~"), 'pypscresults', datetime.now().strftime('MCresult-'+'%Y-%m-%d-%H%M%S'))
        os.makedirs(fp)
    else:
        raise ValueError(f"I stop here - Unknown OS: {os_name}")
        
    #print(f'I created {fp} folder to save pypsc results.')
    
    return fp