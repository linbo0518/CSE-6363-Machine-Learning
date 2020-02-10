# CSE 6363 Machine Learning

- Student ID: 1001778270
- Name: Bo Lin

## Assignment 01

- **HW1: Textbook  Exercise 1.1(p.6, p.58)**

$$y(x, w) = \sum^{M}_{j=0}{w_jx^j}$$

$$E(w) = \frac{1}{2}\sum^{N}_{n=1}\{y(x_n, w) - t_n\}^2$$

$$\frac{\partial{E}}{\partial{w_i}} = \sum^{N}_{n=1}(\sum^{M}_{j=0}w_jx_n^i-t_n)x^i_n$$

make $\frac{\partial{E}}{\partial{w_i}} = 0$, so:

$$\frac{\partial{E}}{\partial{w_i}} = \sum^{N}_{n=1}(\sum^{M}_{j=0}w_jx_n^j-t_n)x^i_n = 0$$

$$\sum^{N}_{n=1}\sum^{M}_{j=0}w_jx^jx^i_n = \sum^{N}_{n=1}{x^i_nt_n}$$

$$\sum^{N}_{n=1}\sum^{M}_{j=0}w_jx_n^{i+j} = \sum^{N}_{n=1}{x^i_nt_n}$$

so:

$$\sum^{M}_{j=0}\sum^{N}_{i=1}x_n^{i+j}w_j = \sum^{N}_{n=1}{x^i_nt_n}$$

---

- **HW2: Show that when M=1, the results of HW1 is identical the results of linear regression.**

When $M = 1$:

$$y(x, w) = w_1x + w_0$$

The formula of linear regression is:

$$y = wx + b$$

two formulas are the identical.

---

- **HW3: Textbook  Exercise 1.2.**

$$E(w) = \frac{1}{2}\sum^{N}_{n=1}\{y(x_n, w) - t_n\}^2 + \frac{\lambda}{2} {\lVert w \rVert}^2$$

$$\frac{\partial{E}}{\partial{w_i}} = \sum^{N}_{n=1}(\sum^{M}_{j=0}w_jx_n^i-t_n)x^i_n + \lambda w_i$$

make $\frac{\partial{E}}{\partial{w_i}} = 0$, so:

$$\sum^{M}_{j=0}\sum^{N}_{i=1}x_n^{i+j}w_j + \lambda w_i = \sum^{N}_{n=1}{x^i_nt_n}$$

---

- **HW4: A problem on a multiple-choice quiz is answered correctly with probability 0.9 if a student is prepared. An unprepared student guesses between 4 possible answers, so the probability of choosing the right answer is 1/4. Seventy-five percent of students prepare for the quiz. If Mr. X gives a correct answer to this problem, what is the chance that he did not prepare for the quiz?**

We know that:

$$(P_c|P_p) = 0.9 \quad and \quad (P_c|P_u) = 0.25$$

$$P_p = 0.25 \quad and \quad P_u = 0.25$$

So for the $P_c$:

$$P_c = (P_c|P_p) \times P_p + (P_c|P_u) \times P_u$$

So $(P_u|P_c)$ is:

$$(P_u|P_c)  = \frac{(P_c|P_u) \times P_u}{P_c}$$

$$= \frac{(P_c|P_u) \times P_u}{(P_c|P_p) \times P_p + (P_c|P_u) \times P_u}$$

So $(P_u|P_c) = 0.08474$

---

- **HW5: At a plant, 20% of all the produced parts are subject to a special electronic inspection. It is known that any produced part which was inspected electronically has no defects with probability 0.95. For a part that was not inspected electronically this probability is only 0.7. A customer receives a part and finds defects in it. What is the probability that this part went through an electronic inspection?**

$$P_i = 0.2 \quad so \quad P_{ni} = 0.8$$

$$(P_{nd}|P_i) = 0.95 \quad so \quad (P_d|P_i) = 0.05$$

$$(P_{nd}|P_{ni}) = 0.7 \quad so \quad (P_d|P_{ni}) = 0.3$$

$$P_d = (P_d|P_i) \times P_i + (P_d|P_{ni}) \times P_{ni}$$

So $P_d = 0.25$

$$(P_i|P_d) = \frac{(P_d|P_i) \times P_i}{p_d}$$

So $(P_i|P_d) = 0.04$
