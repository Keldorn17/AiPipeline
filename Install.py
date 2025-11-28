import subprocess
import sys
import os

def install_requirements():
    if not os.path.exists("requirements.txt"):
        print("requirements.txt nem található!")
        sys.exit(1)

    print("Csomagok telepítése...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Telepítés kész.")

def run_main():
    if not os.path.exists("Main.py"):
        print("Main.py nem található!")
        sys.exit(1)

    print("Program futtatása...")
    subprocess.check_call([sys.executable, "Main.py"])

if __name__ == "__main__":
    install_requirements()
    run_main()
