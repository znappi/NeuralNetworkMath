import numpy as np

def main():
    print("🚀 Hello from Docker!")

    # Создаем простую матрицу 2x2
    matrix = np.array([[1, 2], [3, 4]])

    # Выполняем матричное умножение
    result = np.dot(matrix, matrix)

    print("Result:")
    print(result)

if __name__ == "__main__":
    main()
