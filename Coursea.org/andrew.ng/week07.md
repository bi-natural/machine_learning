## Weekly 7

Goals : SVM(Support Vector Machine)
   - type of supervised machine learning
   - cleaner an more powerful.

Recall *logistic regression*

  if $y=1$, then $h_{\theta} \approx 1$ and $\Theta^T x \gg 0$
  if $y=0$, then $h_{\theta} \approx 0$ and $\Theta^T x \ll 0$

  - cost function : $J(\theta)$

  $ J(\theta) = {1 \over m} \sum_{i=1}^{m} y^{(i)} log(h_{\theta}x^{(i)}) + (1 - y^{(i)})  log(1 - h_{\theta}(x^{(i)}))$

  $ J(\theta) = {1 \over m} \sum_{i=1}^{m} y^{(i)} log(1/{1 + e^{-{\theta}^T x^{(i)}}}) - (1 - y^{(i)})  log(1/{1 + e^{-{\theta}^T x^{(i)}}}) + (1 - y^{(i)}) $

### new notations

Set $cost_0(z)$, $cost_1(z)$
  - cost for classifying when y=0 / y=1, k는 임의의 상수값 (선의 기울기값)
  - $cost_0(z) = max(0, k(1+z))$
  - $cost_1(z) = max(0, k(1-z))$

Replace with $cost_0$, $cost_1$ to (cost function of regularized logistic regression)
  $ J(\theta) = C \sum_{i=1}^{m} y^{(i)} cost_1(\theta^T x^{(i)}) + (1 - y^{(i)})  cost_0(\theta^T x^{(i)}) + 1/2 \sum_{j=1}^{n} \Theta_{j}^{2}$
, when $ C = 1 / \lambda$

- 좀더 regularized시키려면, C값을 줄이면 $\lambda$가 커지는 효과로 오버피팅에 개선 효과가 있다  
- 반대로, 덜 .regularized시키려면 C값을 키워야 한다. (underfiting을 개선)

### Large Margin Intuition

SVM는 large margin classifier 로 생각할 수 있다. classifier를 하더라도 얼마나 넉넉한 마진을 확보하여 안정적으로 경계를 세울 수 있는지이다.


### LMC(Large Margin Classification)에 대한 수학적 백그라운드

벡터에 의한 euclid distance.

### Kernals

SVM를 사용하기 위해 Kernel를 사용함. (좀더 복잡하고 non-linear한 Classification에 사용)

유사성함수(similarity function): 어떤 임의의 landmark $l^{(i)}$에 대한 유사성을 정의한다.

$ f_i = similarity(x, l^{(i)}) =  \exp (-{{\lVert x - l^{(i)}\rVert^2} \over {2 \sigma^2}})$

Kernel (similarity) fuctions:

```
function f = kernel (x1, x2)
  % f = $f_i$
  % x1 = x^{(i)}
  % x2 = landmark (l^{(i)})
```
  $ f = exp (-{{\lVert x1 - x2 \rVert ^2} \over { 2 \sigma^2}}) $
```
return
```
  - Note: Gaussian Kernel를 사용하기 전에 feature scaling를 해야 함.

  $\lVert x - l \rVert ^ 2$ = $\sum_{i=1}^{n} (x_i - l_i)$

Choose of Kernel

  - Gaussian Kernel
  - Polynomail Kernel : k(x, l) = $(x^T l + constant)^n$,
  - String Kernl, chi-square kenel, histogram intersection kernel, ..


### Multi-class Classification
