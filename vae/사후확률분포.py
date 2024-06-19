import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 사전 분포 설정 (Beta 분포)
prior_a, prior_b = 1, 1

# 관찰된 데이터: 성공 횟수, 실패 횟수
obs_success, obs_failure = 10, 5

# 사후 분포 파라미터 업데이트
post_a = prior_a + obs_success
post_b = prior_b + obs_failure

# x축 값 생성
x = np.linspace(0, 1, 100)

# 사전 분포와 사후 분포 그래프 생성
plt.figure(figsize=(10, 5))
plt.plot(x, beta.pdf(x, prior_a, prior_b), label='Prior Distribution', color='blue')
plt.plot(x, beta.pdf(x, post_a, post_b), label='Posterior Distribution', color='red')
plt.title('Bayesian Update: Prior and Posterior Distributions')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()