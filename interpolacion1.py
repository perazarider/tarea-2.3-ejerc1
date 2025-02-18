import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 6*x**2 + 11*x - 6

def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2  

x0, x1, x2 = 1.0, 2.0, 3.0  
x_points = np.array([x0, x1, x2])
y_points = f(x_points)


x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]


root = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)


real_root = 2.0  
error_absoluto = abs(real_root - root)
error_relativo = error_absoluto / abs(real_root)
error_cuadratico = error_absoluto**2


plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 6x^2 + 11x - 6", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")  
plt.show()

# Resultados
print(f"Raíz aproximada usando interpolación: {root:.6f}")
print(f"Error absoluto: {error_absoluto:.6f}")
print(f"Error relativo: {error_relativo:.6f}")
print(f"Error cuadrático: {error_cuadratico:.6f}")
