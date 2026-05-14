import numpy as np


def partial_pivot_gauss(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    swaps = []
    multipliers = []

    for k in range(n - 1):
        pivot = k + np.argmax(np.abs(A[k:, k]))
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
            swaps.append((k + 1, pivot + 1))
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            multipliers.append((i + 1, k + 1, m))
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]
    return x, A, b, swaps, multipliers


def gauss_seidel(A, b, tol=1e-8, max_iter=200000):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    x = np.zeros(n)

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            left = np.dot(A[i, :i], x[:i])
            right = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (b[i] - left - right) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, k
    return x, max_iter


def main():
    A = np.array(
        [
            [10.0, 7.0, 8.0, 7.0],
            [7.0, 5.0, 6.0, 5.0],
            [8.0, 6.0, 10.0, 9.0],
            [7.0, 5.0, 9.0, 10.0],
        ]
    )
    b = np.array([32.0, 23.0, 33.0, 31.0])
    b_hat = np.array([32.01, 22.99, 33.01, 30.99])

    x, U, y, swaps, multipliers = partial_pivot_gauss(A, b)
    x_hat, U_hat, y_hat, _, _ = partial_pivot_gauss(A, b_hat)

    delta_x = x_hat - x
    delta_b = b_hat - b
    A_inv = np.linalg.inv(A)

    print("A =")
    print(A)
    print("\nb =", b)
    print("b_hat =", b_hat)

    print("\nSwaps:", swaps)
    print("Multipliers:")
    for i, k, m in multipliers:
        print(f"m{i}{k} = {m}")

    print("\nUpper triangular matrix for b:")
    print(U)
    print("Transformed right-hand side:", y)
    print("x =", x)

    print("\nUpper triangular matrix for b_hat:")
    print(U_hat)
    print("Transformed right-hand side:", y_hat)
    print("x_hat =", x_hat)

    print("\ndelta_x =", delta_x)
    print("delta_b =", delta_b)

    cond1 = np.linalg.norm(A, 1) * np.linalg.norm(A_inv, 1)
    cond_inf = np.linalg.norm(A, np.inf) * np.linalg.norm(A_inv, np.inf)

    rel_x_1 = np.linalg.norm(delta_x, 1) / np.linalg.norm(x, 1)
    rel_b_1 = np.linalg.norm(delta_b, 1) / np.linalg.norm(b, 1)
    bound_1 = cond1 * rel_b_1

    rel_x_inf = np.linalg.norm(delta_x, np.inf) / np.linalg.norm(x, np.inf)
    rel_b_inf = np.linalg.norm(delta_b, np.inf) / np.linalg.norm(b, np.inf)
    bound_inf = cond_inf * rel_b_inf

    print("\nA^{-1} =")
    print(A_inv)

    print("\n1-norm report:")
    print("cond_1(A) =", cond1)
    print("||delta_b||_1 / ||b||_1 =", rel_b_1)
    print("bound_1 =", bound_1)
    print("||delta_x||_1 / ||x||_1 =", rel_x_1)

    print("\ninf-norm report:")
    print("cond_inf(A) =", cond_inf)
    print("||delta_b||_inf / ||b||_inf =", rel_b_inf)
    print("bound_inf =", bound_inf)
    print("||delta_x||_inf / ||x||_inf =", rel_x_inf)

    eigvals = np.linalg.eigvals(-np.linalg.inv(np.tril(A)) @ np.triu(A, 1))
    rho = np.max(np.abs(eigvals))
    print("\nGauss-Seidel spectral radius =", rho)

    x_gs, it_gs = gauss_seidel(A, b, tol=1e-8)
    xh_gs, ith_gs = gauss_seidel(A, b_hat, tol=1e-8)
    print("Gauss-Seidel for b:", it_gs, x_gs)
    print("Gauss-Seidel for b_hat:", ith_gs, xh_gs)


if __name__ == "__main__":
    main()
