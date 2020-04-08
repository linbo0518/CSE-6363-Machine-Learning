# SVM推导

- Author: Bo Lin (@linbo0518)
- Created Date: 03/18/2020
- Modified Date: 03/25/2020

## 线性支持向量机

存在数据集 $(x_1, y_1), (x_2, y_2) ... (x_n, y_n)$ ，其中包含两类，两类分别用 $+1$ 和 $-1$ 表示

存在超平面：$w^T x + b = 0$ 可以正确划分空间中的两类数据，所以：

$$ w^T x + b < 0 \quad \text{when} \  y = -1 $$
$$ w^T x + b > 0 \quad \text{when} \  y = +1 $$

超平面上存在两点 $x_1, x_2$ , 则：

$$ w^T x_1 + b = 0, \  w^T x_2 + b = 0 $$

$$ w^T x_1 = -b, \  w^T x_2 = -b $$

$$ w^T (x_1 - x_2) = 0 $$

因为 $x_1 - x_2$ 为超平面上的向量，所以， $w^T$ 为超平面的法向量：

$$ w^T \perp \text{Hyperplane} $$

平面外一点 $x$ 到平面的距离为：

$$ \text{distance} = | \frac{w^T}{\parallel w \parallel} (x - x_1) | $$

$$ \text{distance} = | \frac{1}{\parallel w \parallel} (w^T x - w^T x_1) | $$

$$ \text{distance} = \frac{1}{\parallel w \parallel} | w^T x + b | $$

因为 $y$ 与 $w^T x + b$ 同号：

$$ y(w^T x + b) > 0 $$

所以可以脱掉绝对值符号：

$$ \text{distance} = \frac{1}{\parallel w \parallel} y (w^T x + b) $$

所以对于SVM问题可以表示成：

$$ \max \limits_{w, b} \quad \text{margin}(w, b) $$
$$ \text{subject to(s.t.)} \quad \text{every} \ y(w^T x + b) > 0 $$

其中 $\text{margin} = \min \limits_{1,2...n} \  \frac{1}{\parallel w \parallel} y_n(w^T x + b)$ ，
表示为数据点中到超平面的最小距离，也即距离超平面最近的点到超平面的距离

将 $w, b$ 进行等比例的放缩， $w^T x + b = 0$ 与 $kw^T x + kb = 0$ 表示为同一个超平面，所以将其放缩为特殊值以简化计算：

$$ \min \limits_{1,2...n} \  \frac{1}{\parallel w \parallel} y_n(w^T x_n + b) = 1 \quad \rightarrow \quad \text{margin}(w, b) = \frac{1}{\parallel w \parallel} $$

于是，问题可以被简化为：

$$ \max \limits_{w, b} \quad \frac{1}{\parallel w \parallel} $$
$$ \text{s.t.} \quad \min \limits_{1,2...n} \  \frac{1}{\parallel w \parallel} y_n(w^T x_n + b) = 1 $$

因为服从于所有数据点到决策超平面的距离最小值为1，所以可以将整个问题放缩来去掉去最小值符号：

$$ y(w^T x + b) \geq 1, \quad \text{for all} \  n $$

> **讨论**：
>
> 因为对于所有的数据点，若最小值为1，那么必然对于所有的点来说都大于等于1，
> 但这其中存在一个问题，如果所有的数据点都大于1，并不存在等于1的数据点，那么这个缩放就会存在问题。
> 通过反证法，我们假设缩放存在问题，所有的数据点都大于1，例如 $y(w^T x + b) > 1.414$
> 前文中提到对于 $w, b$ 可以缩放而不改变决策超平面，所有为了达到大于等于1，
> 所有要对 $w，b$ 进行缩放， 得到 $\frac{w}{1.414}, \frac{b}{1.414}$ 来满足条件。
> 我们已知最大间隔 $\text{margin} = \frac{1}{\parallel w \parallel}$，
> 那么缩放后最大间隔 $\text{margin}' = \frac{1.414}{\parallel w \parallel}$,
> 所以就会出现 $\text{margin}' > \text{margin}$ ，导致存在比最大间隔更优的解，
> 故矛盾，必然存在等于1的解，该缩放不存在问题。

于是，问题可以转化为：

$$ \max \limits_{w, b} \quad \frac{1}{\parallel w \parallel} $$
$$ \text{s.t.} \quad y_n(w^T x_n + b) \geq 1, \quad \text{for all} \  n$$

进一步简化问题，将最大值问题转化为最小值问题（取倒数），同时去掉 $w$ 外面的二范数，问题就可以表示为：

$$ \min \limits_{w, b} \quad \frac{1}{2} w^t w $$
$$ \text{s.t.} \quad y_n(w^T x_n + b) \geq 1, \quad \text{for all} \  n$$

该问题为一个有条件的优化问题，符合二次规划形式问题的标准形式，可直接通过二次规划相关工具求解，此处略过。

所以，当找到该问题的最优解的 $x$ 时，那么此时的 $x$ 就是支持向量候选。

## 对偶问题

针对有条件的优化问题，可以使用拉格朗日乘数法转化为看似没有条件的问题进行求解，
根据拉格朗日乘数法将约束条件乘上一个系数 $\alpha$ 加入到优化函数中，进行求解。
对于拉格朗日乘数法，有多少个约束条件就需要多少个乘数，使用针对SVM问题，需要服从 $N$ 个条件，
那么对于该问题的拉格朗日乘数法就需要 $N$ 个系数 $\alpha$。

所以，根据拉格朗日乘数法，可以将原问题表示成：

$$ \mathcal{L}(w, b, \alpha) = \frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n(1 - y_n(w^T x + b))} $$

其中 $\frac{1}{2} w^T w$ 为原问题的优化目标， $\sum_{n=1}^{N}{\alpha_n(1 - y_n(w^T x + b))}$ 为原问题的约束条件

所以，原问题就变成：

$$ \min \limits_{w, b}(\max \limits_{\alpha_n \geq 0} \mathcal{L}(w, b, \alpha)) $$

> **证明**：
>
> - 对于任何不符合条件的 $w, b$ ： $\max \limits_{\alpha_n \geq 0}(\text{Obj} + \sum_{n=1}^{N}\alpha_n(\text{postive})) \rightarrow \infty$
> - 对于任何符合条件的 $w, b$ ： $\max \limits_{\alpha_n \geq 0}(\text{Obj} + \sum_{n=1}^{N}\alpha_n(\text{non-postive})) = \text{Obj}$
>
> 然后通过最小化操作 $\min \limits_{w, b}(\text{Obj}, \text{if violate}; \  \infty, \text{if feasible})$
>
> 所以 $\text{SVM} \equiv \min \limits_{w, b}(\max \limits_{\alpha_n \geq 0} \mathcal{L}(w, b, \alpha))$

此时，如果我们选取一个固定的 $\alpha$ ，称为 $\alpha'$ 满足 $\alpha' \geq 0$ ， 便可得到如下关系：

$$ \min \limits_{w, b}(\max \limits_{\alpha_n \geq 0} \mathcal{L}(w, b, \alpha)) \geq \min \limits_{w, b}\mathcal{L}(w, b, \alpha') $$

所以，固定 $\alpha$ 可以使得右式一定小于等于左式，此时再通过取最大值操作可以取得左式的下界，得到：

$$ \min \limits_{w, b}(\max \limits_{\alpha_n \geq 0} \mathcal{L}(w, b, \alpha)) \geq \max \limits_{\alpha_n \geq 0}(\min \limits_{w, b}\mathcal{L}(w, b, \alpha)) $$

> **讨论**：
>
> 这样做的目的是什么呢？首先我们通过拉格朗日函数转化得到的问题称为拉格朗日对偶问题，
> 然后对于左式大于等于右式这种形式，在对偶问题里我们称取得原问题的下界叫做弱对偶（weak duality），
> 这样转化的好处是，原先我们需要在约束 $\alpha_n \geq 0$ 的条件下求最大值，
> 现在我们可以不考虑任何约束条件，直接求解拉格朗日函数在 $w, b$ 为何值时的最小值，相当于没有约束的优化问题。
>
> 那么怎么样才能实现强对偶（strong duality）呢？也就是右式问题与左式问题相同，可以解得同样的最优解？
> 前面我们提到可以使用二次规划标准形式来进行求解，对于弱对偶问题，
> 如果问题是二次规划形式并且满足以下条件便可以认为是强对偶关系：
>
> - 问题为凸问题；SVM问题刚好是一个凸问题
> - 问题有解；对于SVM问题来说就是可分
> - 有线性条件；对于SVM问题可以使用二次规划求解，二次规划是线性条件
>
> 上述三个条件都满足，那么我们就可以称右式是左式的强对偶关系，也就是说解右边的式子和解左边的式子是一样的，
> 存在一组解对于左边来说是最好的，对于右边来说也是最好的。

那么现在我们的问题就可以表示成：

$$ \max \limits_{\alpha_n \geq 0}(\min \limits_{w, b}(\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n(1 - y_n(w^T x_n + b))})) $$

现在我们的里层最小化问题已经没有任何条件约束的最优化问题，那么当我们要求最小值问题的时候，
便可以通过其偏微分为0时，我们先对 $b$ 求解：

$$ \frac{\partial{\mathcal{L}(w, b, \alpha)}}{\partial{b}} = 0 = - \sum_{n=1}^{N}{\alpha_n y_n} $$

$$ \sum_{n=1}^{N}{\alpha_n y_n} = 0 $$

此时，当解为最佳解时 $b$ 前面的系数 $- \sum_{n=1}^{N}{\alpha_n y_n}$ 为0，那么我们可以去掉 $b$ ，
同时加上约束条件：

$$ \max \limits_{\alpha_n \geq 0}(\min \limits_{w, b}(\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n(1 - y_n(w^T x_n))} - \sum_{n=1}^{N}{\alpha_n y_n} b)) $$

$$ \max \limits_{\alpha_n \geq 0, \sum{\alpha_n y_n} = 0}(\min \limits_{w}(\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n(1 - y_n(w^T x_n))})) $$

此时，问题变成了与 $b$ 无关，只与 $w$ 相关的最小值问题，然后我们再对 $w$ 求解：

$$ \frac{\partial{\mathcal{L}(w, b, \alpha)}}{\partial{w_i}} = 0 = w_i - \sum_{n=1}^{N}{\alpha_n y_n x_{n,i}} $$

$$ w_i = \sum_{n=1}^{N}{\alpha_n y_n x_{n,i}} $$

此时，我们的最佳解需要满足 $w_i = \sum_{n=1}^{N}{\alpha_n y_n x_{n,i}}$ ，所以此时：

$$ \max \limits_{\alpha_n \geq 0, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}}(\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n} - \sum_{n=1}^{N}{\alpha_n y_n x_{n}} w) $$

$$ \max \limits_{\alpha_n \geq 0, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}}(\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n} - w^T w) $$

$$ \max \limits_{\alpha_n \geq 0, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}}(-\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n}) $$

$$ \max \limits_{\alpha_n \geq 0, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}}(-\frac{1}{2} \parallel \sum_{n=1}^{N}{\alpha_n y_n x_{n}}  \parallel^2 + \sum_{n=1}^{N}{\alpha_n}) $$

此时我们已经不需要考虑 $w, b$，将原问题转化为来一个只需要考虑 $\alpha$ 的问题，求解算出 $\alpha$ ，便可以求出相应的 $w, b$

## KKT条件

当我们把上文的问题求解得到相应的 $w, b, \alpha$ ，这些解会满足一些关系，我们称这个关系叫做KKT条件。
KKT的内容是：

如果求解得到的 $w, b, \alpha$ 满足原始问题与对偶问题都是最佳解的话，需要满足以下条件：

- 满足原始问题条件； $y_n(w^T x_n + b) \geq 1$
- 满足对偶问题条件； $\alpha_n \geq 0$
- 满足对偶问题内部问题的最优化结果； $\sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}$
- 满足原始问题内部问题的最优化结果； $\alpha(1 - y_n(w^T x_n + b)) = 0$

所以如果找到满足上述条件的最优化解 $w, b, \alpha$ ，那么一定是原始问题和对偶问题的最优化解，
得到来最优化的 $\alpha$ ，我们就可以通过 $\alpha$ 求的相应的 $w, b$。

此时问题是一个最大化问题，不利于求解，我们可以通过取负号的方式转化为最小化问题，于是问题变成：

$$ \min \limits_{\alpha_n \geq 0, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}}(\frac{1}{2} \parallel \sum_{n=1}^{N}{\alpha_n y_n x_{n}}  \parallel^2 - \sum_{n=1}^{N}{\alpha_n}) $$

进一步化简，展开平方项，可以表示成：

$$ \min \limits_{\alpha_n \geq 0, \sum{\alpha_n y_n} = 0, w = \sum{\alpha_n y_n x_n}}(\frac{1}{2} \sum_{n=1}^{N}{\sum_{m=1}^{N}{\alpha_n \alpha_m y_n y_m x_n^T x_m}} - \sum_{n=1}^{N}{\alpha_n}) $$

所以最终我们将SVM转化成的拉格朗日对偶形式为：

$$ \min \limits_{\alpha} \quad \frac{1}{2} \sum_{n=1}^{N}{\sum_{m=1}^{N}{\alpha_n \alpha_m y_n y_m x_n^T x_m}} - \sum_{n=1}^{N}{\alpha_n} $$
$$ \text{s.t.} \quad \sum_{n=1}^{N}{\alpha_n y_n} = 0; $$
$$ \alpha_n \geq 0, \quad \text{for} \  n = 1, 2, ..., N $$

对于如何求解 $\alpha$ 我们依旧需要通过二次规划方法进行求解，此处略过。

现在我们可以通过解得最佳化的 $\alpha$ 来反推出 $w, b$：

首先针对 $w$ ，只存在一个条件： $w = \sum{\alpha_n y_n x_n}$

然后对于 $b$ ，存在两个条件： $y_n(w^T x_n + b) \geq 1$ 和 $\alpha_n(1 - y_n(w^T x_n + b)) = 0$

对于求解 $b$ 的第一个条件，只能告诉我们 $b$ 的范围，并不能求得确定的值，所以我们使用第二个条件：

当求解 $\alpha > 0$ 时： $1 - y_n(w^T x_n + b) = 0$ ，求得 $b = y_n - w^T x_n$

此时 $y_n(w^T x_n + b) = 1$ ， 上文我们提到的缩放问题，点到超平面的距离最小为1，那么当前的数据点就是距离超平面距离为1的点，
也就是我们所说的支持向量。

> **讨论**：
>
> 当我们求解出 $\alpha$ 之后，可以根据 $\alpha$ 再反推出 $w, b$ ，之前我们讨论过当 $\alpha > 0$ 时，
> 该点就是我们所说的支持向量，若 $\alpha = 0$ 时，根据 $w = \sum{\alpha_n y_n x_n}$ 我们可以不用计算 $w$ ,
> 因为此时 $w$ 已经为零了。 所以只需要考虑 $\alpha > 0$ 时 $w, b$ 的值，也就是说，SVM算法通过计算找出了哪些是支持向量，
> 若不是支持向量便不再需要计算。

## 软间隔支持向量机

之前的线性支持向量机也叫硬间隔支持向量机，是基于数据完全可分的假设，若想做到数据完全可分，
就需要进行特征转换来增加数据的纬度空间，从而在高维空间内找到一个可以完全分开数据的超平面，
通过增加模型的复杂度来实现完全可分，也带来来过拟合的风险。根据VC Bound理论，
模型在预测阶段的误差是由训练阶段的误差和模型复杂度共同决定的，所以这类模型的泛化性较差，
对于数据噪声无法判别。

那么如果我们允许SVM忽略少量的错误是否可以解决这个问题呢？答案是肯定的，
那么我们就可以将硬间隔SVM问题转化为软间隔SVM问题，这里我们需要引入松弛变量 $\xi$ ，
来度量忽略某些错误的程度：

$$ \xi=
\begin{cases}
0& \text{when} \  y(w^T x + b) \geq 1 \\
1 - y(w^T x + b)& \text{otherwise}
\end{cases}$$

那么我们的问题就变成了：

$$ \min \limits_{w} \quad \frac{1}{2} w^T w + C \sum_{n=1}^{N}{\xi_n} $$
$$ \text{s.t.} \quad y_n(w^T x_n + b) \geq 1 - \xi_n, \quad \text{for} \  n = 1, 2, ..., N; $$
$$ \xi_n \geq 0, \quad \text{for} \  n = 1, 2, ..., N $$

> **讨论**：
>
> $C$ 是调节间隔和犯错程度的参数
>
> - 如果 $C$ 越大，意味着违反完全可分的数据越少，间隔可能会越小
> - 如果 $C$ 越小，意味着违反完全可分的数据越多，间隔可能会越大

然后便可以根据拉格朗日乘数法，可以将原问题表示成：

$$ \mathcal{L}(w, b, \xi, \alpha, \beta) = \frac{1}{2} w^T w + C \sum_{n=1}^{N}\xi_n + \sum_{n=1}^{N}{\alpha_n(1 - \xi_n - y_n(w^T x_n + b))} + \sum_{n=1}^{N}{\beta_n(- \xi_n)} $$

所以，对偶问题就变成：

$$ \max \limits_{\alpha_n \geq 0, \beta_n \geq 0} (\min \limits_{w, b, \xi} \mathcal{L}(w, b, \xi, \alpha, \beta)) $$

代入拉格朗日函数：

$$ \max \limits_{\alpha_n \geq 0, \beta_n \geq 0} (\min \limits_{w, b, \xi} (\frac{1}{2} w^T w + C \sum_{n=1}^{N}\xi_n + \sum_{n=1}^{N}{\alpha_n(1 - \xi_n - y_n(w^T x_n + b))} + \sum_{n=1}^{N}{\beta_n(- \xi_n)})) $$

通过KKT条件来简化该问题：

$$ \frac{\partial{\mathcal{L}}}{\partial{\xi_n}} = 0 = C - \alpha_n - \beta_n $$

$$ \beta_n = C - \alpha_n \quad 0 \leq \alpha_n \leq C $$

将 $\beta_n = C - \alpha_n$ 代入到问题中：

$$ \max \limits_{0 \leq \alpha_n \leq C, \beta_n = C - \alpha} (\min \limits_{w, b} (\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n(1 - y_n(w^T x_n + b))} + \sum_{n=1}^{N}(C - \alpha_n - \beta_n) \xi_n )) $$

消掉所有和 $xi$ 相关的项：

$$ \max \limits_{0 \leq \alpha_n \leq C, \beta_n = C - \alpha} (\min \limits_{w, b} (\frac{1}{2} w^T w + \sum_{n=1}^{N}{\alpha_n(1 - y_n(w^T x_n + b))})) $$

然后进行和推导硬间隔支持向量机一样的步骤得到：

$$ \frac{\partial{\mathcal{L}}}{\partial{b}} = 0 \longrightarrow \sum_{n=1}^{N}{\alpha_n y_n} = 0 $$

$$ \frac{\partial{\mathcal{L}}}{\partial{w_i}} = 0 \longrightarrow w_i = \sum_{n=1}^{N}{\alpha_n y_n x_{n,i}} $$

最终我们的对偶问题变成：

$$ \min \limits_{\alpha} \quad \frac{1}{2} \sum_{n=1}^{N}{\sum_{m=1}^{N}{\alpha_n \alpha_m y_n y_m x_n^T x_m}} - \sum_{n=1}^{N}{\alpha_n} $$
$$ \text{s.t.} \quad \sum_{n=1}^{N}{\alpha_n y_n} = 0; $$
$$ 0 \leq \alpha_n \leq C, \quad \text{for} \  n = 1, 2, ..., N $$

此时便可以和之前一样通过二次规划方法进行求解。

当我们求解出 $\alpha$ 后，可以通过 $\alpha$ 计算出 $w, b$。

对于 $w$ ，计算方法和硬间隔支持向量机相同： $w = \sum{\alpha_n y_n x_n}$

对于 $b$ ，可以使用条件： $\alpha_n(1 - \xi_n - y_n(w^T x_n + b)) = 0$ 和 $(C - \alpha_n) \xi_n = 0$

当求解 $\alpha > 0$ 时： $1 - \xi_n - y_n(w^T x_n + b) = 0$ ，求得 $b = y_n - y_n \xi_n - w^T x_n$
这便陷入了一个循环问题：求解 $b$ 需要求解 $\xi$, 但是求解 $\xi$ 需要求解 $b$。所以此时我们需要使用第二个条件，
当 $\alpha < C$ 也就是 $\xi = 0$，此时 $b = y_n - w^T x_n$。

> **讨论**：
>
> 在求解软间隔支持向量机对偶问题的过程中，我们需要处理两个关系：
> $$ \alpha_n(1 - \xi_n - y_n(w^T x + b)) = 0 $$
> $$ (C - \alpha_n) \xi_n = 0 $$
>
> - 当 $0 < \alpha < C$ 时：$\xi = 0$，此时数据点落在边界上，是自由支持向量
> - 当 $\alpha = 0$ 时：$\xi = 0$，此时数据点落在边界之外，是非支持向量
> - 当 $\alpha = C$ 时：$\xi = 1 - y_n(w^T x + b)$，此时数据点落在边界之内，是有界支持向量
