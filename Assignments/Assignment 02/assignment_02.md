# CSE 6363 Machine Learning

- Student ID: 1001778270
- Name: Bo Lin

## Assignment 02

### HW 6

> Solve SVM for a data set with 3 data instances in 2 dimensions: (1,1,+),
> (0,-1,+), (-1,1,-). Here, the first 2 number are the 2-dimension coordinates,
> ‘+’ in 3rd place is positive class, and ‘-’ in 3rd place is negative class.
> Your task is to: (1) write down dual-problem using those 3 data instances.
> (2) compute alpha’s. (3) compute w and b. (4) compute margin.

For a detailed process on how to get the Lagrange dual problem, please check the `SVM.pdf` file

(1).

According to the Lagrange multiplier method and slackness:

$$ \min \limits_{\alpha} \quad \frac{1}{2} \sum_{n=1}^{N}{\sum_{m=1}^{N}{\alpha_n \alpha_m y_n y_m x_n^T x_m}} - \sum_{n=1}^{N}{\alpha_n} $$
$$ \text{s.t.} \quad \sum_{n=1}^{N}{\alpha_n y_n} = 0; $$
$$ \alpha_n \geq 0, \  \text{for n = 1, 2, ..., N}$$

bring $x, y, \alpha$ into the equation:

$$ \min \limits_{\alpha} \quad \frac{1}{2} (2 \alpha_1^2 - 2 \alpha_1 \alpha_2 + \alpha_2^2  - 2 \alpha_2 \alpha_3 + 2 \alpha_3^2) - \alpha_1 - \alpha_2 - \alpha_3 $$
$$ \text{s.t.} \quad \sum_{n=1}^{N}{\alpha_n y_n} = 0; $$
$$ \alpha_n \geq 0, \  \text{for n = 1, 2, ..., N}$$

simplify:

$$ \min \limits_{\alpha} \quad \alpha_1^2 - \alpha_1 \alpha_2 + \frac{1}{2} \alpha_2^2  - \alpha_2 \alpha_3 + \alpha_3^2 - \alpha_1 - \alpha_2 - \alpha_3 $$
$$ \text{s.t.} \quad \sum_{n=1}^{N}{\alpha_n y_n} = 0; $$
$$ \alpha_n \geq 0, \  \text{for n = 1, 2, ..., N}$$

(2).

use QP (Quadratic programming) to solve the alpha:

$$ \alpha_1 = 0.375, \alpha_2 = 0.25, \alpha_3 = 0.625$$

$$ \alpha^T = (0.375, 0.25, 0.625)$$

(3).

follow the KKT conditions:

$$ w = \sum{\alpha_n y_n x_n} $$

so the $w^T = (1, -0.5)$

because $\alpha > 0$:

$$ \alpha(1 - y_n(w^T x_n + b)) = 0 $$

$$b = y_n - w^T x_n$$

so the $b = 0.5$

the hyperplane is $x_1 - 0.5x_2 + 0.5 = 0$

(4).

margin is $\frac{2}{\parallel w \parallel}$

so the margin is $\frac{4}{\sqrt{5}}$

---

### HW 7

> Solve SVM when data are non-separable, when minimizing the violations of the
> misclassification, i.e., on those slack variables.
>
> $$ \min \limits_{w} \frac{1}{2}{\parallel w \parallel}_2^2 + C(\sum_{i=1}^{N}{\xi_i})^k $$
> $$ \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \  i = 1, ..., n. $$
> $$ \xi_i \geq 0, \  i = 1, ..., n. $$
>
> Your task is to: derive the dual-problem of the above primal-problem when $k=2$ .

Lagrange function with lagrange multipliers $\alpha_i$ and $\beta_i$:

$$ \mathcal{L}(w, b, \xi, \alpha, \beta) = \frac{1}{2} \parallel w \parallel_2^2 + C (\sum_{i=1}^{n}\xi_i)^2 + \sum_{i=1}^{n}{\alpha_i(1-\xi_i-y_i(w^T x_i + b))} + \sum_{i=1}^{n}{\beta_i(-\xi_i)} $$

Lagrange dual:

$$ \max \limits_{\alpha_i \geq 0, \beta_i \geq 0} (\min \limits_{w, b, \xi} \mathcal{L}(w, b, \xi, \alpha, \beta)) $$

So the problem becomes:

$$ \max \limits_{\alpha_i \geq 0, \beta_i \geq 0} (\min \limits_{w, b, \xi} \frac{1}{2} \parallel w \parallel_2^2 + C (\sum_{i=1}^{n}\xi_i)^2 + \sum_{i=1}^{n}{\alpha_i(1-\xi_i-y_i(w^T x_i + b))} + \sum_{i=1}^{n}{\beta_i(-\xi_i)}) $$

Use KKT Conditions:

$$ \frac{\partial{\mathcal{L}}}{\partial{\xi_i}} = 0 = 2C\sum{\xi} - \alpha_i - \beta_i $$

$$ \beta_i = 2C\sum{\xi} - \alpha_i $$

Replace $\beta_i$ with $2C\sum{\xi} - \alpha_i$ to simplify the problem:

$$ \max \limits_{0 \leq \alpha_i \leq 2C\sum\xi, \beta_i = 2C\sum\xi - \alpha} (\min \limits_{w, b} (\frac{1}{2} \parallel w \parallel_2^2 + \sum_{i=1}^{n}{\alpha_i(1-y_i(w^T x_i + b))} + C \sum_{i=1}^{n}{\xi_i})) $$

Follow the process of hard margin SVM:

$$ \max \limits_{0 \leq \alpha_i \leq 2C\sum\xi, \beta = 2C\sum\xi - \alpha, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}} (-\frac{1}{2} \parallel w \parallel_2^2 + \sum_{i=1}^{n}{\alpha_i} - C \sum_{i=1}^{n}{\xi_i}) $$

Convet $\max$ into $\min$:

$$ \min \limits_{0 \leq \alpha_i \leq 2C\sum\xi, \beta = 2C\sum\xi - \alpha, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}} (\frac{1}{2} \parallel w \parallel_2^2 - \sum_{i=1}^{n}{\alpha_i} + C \sum_{i=1}^{n}{\xi_i}) $$

So the dual problem is:

$$ \min \limits_{\alpha} \quad \frac{1}{2} \sum_{i=1}^{n}{\sum_{j=1}^{n}{\alpha_i \alpha_j y_i y_j x_i^T x_j}} - \sum_{i=1}^{n}{\alpha_i} + C \sum_{i=1}^{n}{\xi_i} $$
$$ \text{s.t.} \quad \sum_{i=1}^{n}{\alpha_i y_i} = 0; $$
$$ 0 \leq \alpha_i \leq 2C\sum\xi, \  \text{for i = 1, 2, ..., n}$$
