"""
Моделирование диффузии цифрового продукта на сети агентов
с применением модели Басса и центральности по собственному значению.

Структура сети генерируется по модели предпочтительного присоединения
Барабаши–Альберт (Barabási, Albert, 1999), стандартной для моделирования
экономических и социальных сетей.
"""

import numpy as np
import csv

# Фиксируем зерно случайности для воспроизводимости расчётов
np.random.seed(42)


def generate_network_ba(n_total=1000, m_links=4, weight_range=(0.3, 0.9)):
    """
    Генерация сети по модели Барабаши–Альберт (Barabási, Albert, 1999).

    Алгоритм:
    1. Начать с малой связной сети из m_links + 1 узлов
    2. Каждый следующий узел добавляется и устанавливает m_links связей
    3. Вероятность соединения с существующим узлом пропорциональна
       его числу связей (preferential attachment, "богатые богатеют")

    Это даёт распределение степеней по степенному закону P(k) ~ k^(-3),
    что соответствует реальным экономическим и социальным сетям.
    """
    A = np.zeros((n_total, n_total))

    # Начальная клика из m_links + 1 узлов
    n_init = m_links + 1
    for i in range(n_init):
        for j in range(n_init):
            if i != j:
                w = np.random.uniform(weight_range[0], weight_range[1])
                A[i, j] = w

    degrees = np.zeros(n_total)
    for i in range(n_init):
        degrees[i] = (A[i, :] > 0).sum()

    # Добавление новых узлов с предпочтительным присоединением
    for new_node in range(n_init, n_total):
        existing = np.arange(new_node)
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = np.random.choice(existing, size=m_links,
                                    replace=False, p=probs)

        for t in targets:
            w = np.random.uniform(weight_range[0], weight_range[1])
            A[new_node, t] = w
            A[t, new_node] = w
            degrees[new_node] += 1
            degrees[t] += 1

    return A, degrees


def compute_centrality(A, tol=1e-6, max_iter=1000):
    """
    Расчёт центральности по собственному значению степенным методом.
    """
    n = A.shape[0]
    x = np.ones(n) / np.sqrt(n)

    lambda1 = 0
    n_iter = max_iter
    for k in range(max_iter):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new = x_new / norm
        lambda1_new = x_new @ A @ x_new / (x_new @ x_new)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            lambda1 = lambda1_new
            n_iter = k + 1
            break
        x = x_new
        lambda1 = lambda1_new

    # Второе собственное значение через дефляцию
    A_deflated = A - lambda1 * np.outer(x, x)
    eigenvalues = np.linalg.eigvals(A_deflated)
    lambda2 = np.max(np.abs(eigenvalues.real))

    # Нормировка: максимальная компонента = 1
    x_normalized = x / x.max()

    return x_normalized, lambda1, lambda2, n_iter


def bass_diffusion(p, q, m, N0, T):
    """
    Дискретная модель Басса.
        n(t) = p*(m - N(t-1)) + q*(N(t-1)/m)*(m - N(t-1))
        N(t) = N(t-1) + n(t)
    """
    N = np.zeros(T + 1)
    n = np.zeros(T + 1)
    N[0] = N0

    for t in range(1, T + 1):
        increment = p * (m - N[t - 1]) + q * (N[t - 1] / m) * (m - N[t - 1])
        n[t] = increment
        N[t] = N[t - 1] + increment

    return N, n


def run_all():
    print("МОДЕЛИРОВАНИЕ ДИФФУЗИИ ЦИФРОВОГО ПРОДУКТА НА СЕТИ")

    # === Параметры модели ===
    N_TOTAL = 1000
    M_LINKS = 4
    P = 0.015
    M = 1000
    N0 = 50
    T = 10

    Q_MASS = 0.45
    Q_TARGET = 0.70
    Q_COMB = 0.57

    # === Этап 1: чистая модель Басса (раздел 2.2) ===
    print("\n[Этап 1] Расчёт чистой модели Басса (раздел 2.2)")
    N_pure, n_pure = bass_diffusion(P, Q_MASS, M, 0, T)
    for t in range(1, T + 1):
        print(f"  Год {t}: N(t)={N_pure[t]:.1f}, n(t)={n_pure[t]:.1f}")

    peak_pure = np.argmax(n_pure[1:]) + 1
    t_star = np.log(Q_MASS / P) / (P + Q_MASS)
    print(f"  Теоретическая точка перегиба t* = {t_star:.2f} года")
    print(f"  Эмпирический пик прироста: год {peak_pure}")

    # === Этап 2: генерация сети Барабаши-Альберт (раздел 3.1) ===
    print("\n[Этап 2] Генерация сети Барабаши-Альберт (раздел 3.1)")
    A, degrees = generate_network_ba(n_total=N_TOTAL, m_links=M_LINKS)
    print(f"  Узлов: {N_TOTAL}, связей: {int((A > 0).sum() / 2)}")
    print(f"  Средняя степень: {degrees.mean():.1f}, "
          f"максимальная: {int(degrees.max())}")

    # === Этап 3: расчёт центральности (раздел 3.3) ===
    print("\n[Этап 3] Расчёт центральности по собственному значению")
    x, lambda1, lambda2, n_iter = compute_centrality(A)
    print(f"  λ1 = {lambda1:.3f}, λ2 = {lambda2:.3f}")
    print(f"  λ2/λ1 = {lambda2/lambda1:.3f}, итераций: {n_iter}")

    sorted_x = np.sort(x)[::-1]
    total = sorted_x.sum()
    print(f"  Топ-5%:  {sorted_x[:50].sum() / total * 100:.1f}%")
    print(f"  Топ-10%: {sorted_x[:100].sum() / total * 100:.1f}%")
    print(f"  Топ-20%: {sorted_x[:200].sum() / total * 100:.1f}%")

    # === Этап 4: три сценария проникновения (раздел 3.3) ===
    print("\n[Этап 4] Расчёт трёх сценариев проникновения")
    N_mass, n_mass = bass_diffusion(P, Q_MASS, M, N0, T)
    N_target, n_target = bass_diffusion(P, Q_TARGET, M, N0, T)
    N_comb, n_comb = bass_diffusion(P, Q_COMB, M, N0, T)

    print("  Год | Массовый | Точечный | Комбинированный")
    for t in range(T + 1):
        print(f"  {t:3d} | {N_mass[t]:8.1f} | {N_target[t]:8.1f} | "
              f"{N_comb[t]:8.1f}")


if __name__ == "__main__":
    run_all()
