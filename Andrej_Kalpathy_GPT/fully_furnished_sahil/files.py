import os

path = os.getcwd()

print(path)


os.chdir("./")
print(f"os.getcwd() is {os.getcwd()}")
