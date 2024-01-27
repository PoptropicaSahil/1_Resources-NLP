import time
from functools import cache


def add_2(a: int, b: float) -> float:
    """hi

    Args:
        a (int): hihi
        b (float): hihihih

    Returns:
        float: yoyo
    """
    return str(a + b)


@cache
def add(x, y):
    """adds two numbers together

    Args:
        x (int): int1 yyyyyyyyyyyo
        y (int): int2
    """
    print("running...")
    time.sleep(3)
    return x + y


if __name__ == "__main__":
    add(2, 3)
    print(add.__name__)
    print(add.__doc__)

    print(f"NOT CACHED: {add(2,5)}")
    print(f"CACHED: {add(2,5)}")
    print(f"NOT CACHED: {add(5,5)}")
    print(f"NOT CACHED: {add(3,5)}")

    print(add_2("fine", "hi"))
