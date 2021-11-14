# New Adversarial Attack Papers
**update daily**

## **1. Poisoning Knowledge Graph Embeddings via Relation Inference Patterns**

通过关系推理模式毒化知识图嵌入 cs.LG

Joint Conference of the 59th Annual Meeting of the Association for  Computational Linguistics and the 11th International Joint Conference on  Natural Language Processing (ACL-IJCNLP 2021)

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2111.06345v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: We study the problem of generating data poisoning attacks against Knowledge Graph Embedding (KGE) models for the task of link prediction in knowledge graphs. To poison KGE models, we propose to exploit their inductive abilities which are captured through the relationship patterns like symmetry, inversion and composition in the knowledge graph. Specifically, to degrade the model's prediction confidence on target facts, we propose to improve the model's prediction confidence on a set of decoy facts. Thus, we craft adversarial additions that can improve the model's prediction confidence on decoy facts through different inference patterns. Our experiments demonstrate that the proposed poisoning attacks outperform state-of-art baselines on four KGE models for two publicly available datasets. We also find that the symmetry pattern based attacks generalize across all model-dataset combinations which indicates the sensitivity of KGE models to this pattern.

摘要: 研究了针对知识图中链接预测任务的知识图嵌入(KGE)模型产生数据中毒攻击的问题。为了毒化KGE模型，我们提出开发KGE模型的归纳能力，这些能力是通过知识图中的对称性、反转和合成等关系模式捕捉到的。具体地说，为了降低模型对目标事实的预测置信度，我们提出了提高模型对一组诱饵事实的预测置信度的方法。因此，我们设计了对抗性的加法，通过不同的推理模式来提高模型对诱饵事实的预测置信度。我们的实验表明，提出的中毒攻击在两个公开可用的数据集的四个KGE模型上的性能优于最新的基线。我们还发现，基于对称模式的攻击在所有模型-数据集组合中都是通用的，这表明了KGE模型对该模式的敏感性。



## **2. Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes**

Qu反化：利用量化伪像实现对抗性结果 cs.LG

Accepted to NeurIPS 2021 [Poster]

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2110.13541v2)

**Authors**: Sanghyun Hong, Michael-Andrei Panaitescu-Liess, Yiğitcan Kaya, Tudor Dumitraş

**Abstracts**: Quantization is a popular technique that $transforms$ the parameter representation of a neural network from floating-point numbers into lower-precision ones ($e.g.$, 8-bit integers). It reduces the memory footprint and the computational cost at inference, facilitating the deployment of resource-hungry models. However, the parameter perturbations caused by this transformation result in $behavioral$ $disparities$ between the model before and after quantization. For example, a quantized model can misclassify some test-time samples that are otherwise classified correctly. It is not known whether such differences lead to a new security vulnerability. We hypothesize that an adversary may control this disparity to introduce specific behaviors that activate upon quantization. To study this hypothesis, we weaponize quantization-aware training and propose a new training framework to implement adversarial quantization outcomes. Following this framework, we present three attacks we carry out with quantization: (i) an indiscriminate attack for significant accuracy loss; (ii) a targeted attack against specific samples; and (iii) a backdoor attack for controlling the model with an input trigger. We further show that a single compromised model defeats multiple quantization schemes, including robust quantization techniques. Moreover, in a federated learning scenario, we demonstrate that a set of malicious participants who conspire can inject our quantization-activated backdoor. Lastly, we discuss potential counter-measures and show that only re-training consistently removes the attack artifacts. Our code is available at https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation

摘要: 量化是一种流行的技术，它将神经网络的参数表示从浮点数转换为低精度数字(例如$，8位整数)。它减少了推理时的内存占用和计算成本，便于部署资源匮乏的模型。然而，这种变换引起的参数扰动导致了量化前后模型之间的$行为$$差异$。例如，量化模型可能会错误分类一些本来可以正确分类的测试时间样本。目前尚不清楚这种差异是否会导致新的安全漏洞。我们假设对手可以控制这种差异，以引入量化后激活的特定行为。为了研究这一假设，我们将量化意识训练武器化，并提出了一个新的训练框架来实现对抗性量化结果。在此框架下，我们提出了三种量化的攻击：(I)不加区别的攻击，造成显著的精度损失；(Ii)针对特定样本的有针对性的攻击；以及(Iii)用输入触发器控制模型的后门攻击。我们进一步表明，单一的折衷模型击败了包括鲁棒量化技术在内的多个量化方案。此外，在联合学习场景中，我们演示了一组合谋的恶意参与者可以注入我们的量化激活后门。最后，我们讨论了潜在的对策，并表明只有重新训练才能始终如一地移除攻击伪影。我们的代码可在https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation获得



## **3. Robust Deep Reinforcement Learning through Adversarial Loss**

对抗性损失下的鲁棒深度强化学习 cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2008.01976v2)

**Authors**: Tuomas Oikarinen, Wang Zhang, Alexandre Megretski, Luca Daniel, Tsui-Wei Weng

**Abstracts**: Recent studies have shown that deep reinforcement learning agents are vulnerable to small adversarial perturbations on the agent's inputs, which raises concerns about deploying such agents in the real world. To address this issue, we propose RADIAL-RL, a principled framework to train reinforcement learning agents with improved robustness against $l_p$-norm bounded adversarial attacks. Our framework is compatible with popular deep reinforcement learning algorithms and we demonstrate its performance with deep Q-learning, A3C and PPO. We experiment on three deep RL benchmarks (Atari, MuJoCo and ProcGen) to show the effectiveness of our robust training algorithm. Our RADIAL-RL agents consistently outperform prior methods when tested against attacks of varying strength and are more computationally efficient to train. In addition, we propose a new evaluation method called Greedy Worst-Case Reward (GWC) to measure attack agnostic robustness of deep RL agents. We show that GWC can be evaluated efficiently and is a good estimate of the reward under the worst possible sequence of adversarial attacks. All code used for our experiments is available at https://github.com/tuomaso/radial_rl_v2.

摘要: 最近的研究表明，深度强化学习Agent容易受到Agent输入的小的对抗性扰动，这引起了人们对在现实世界中部署此类Agent的担忧。为了解决这个问题，我们提出了RADIUS-RL，一个原则性的框架来训练强化学习Agent，使其对$l_p$-范数有界的攻击具有更好的鲁棒性。我们的框架与流行的深度强化学习算法兼容，并通过深度Q-学习、A3C和PPO验证了其性能。我们在三个深度RL基准(Atari、MuJoCo和ProcGen)上进行了实验，以验证我们的鲁棒训练算法的有效性。我们的Radius-RL代理在针对不同强度的攻击进行测试时，性能一直优于以前的方法，并且在训练时的计算效率更高。此外，我们还提出了一种新的评估方法--贪婪最坏情况奖励(GWC)来度量深度RL代理的攻击不可知性。我们表明，GWC可以被有效地评估，并且是在可能的最坏的对抗性攻击序列下的一个很好的奖励估计。我们实验使用的所有代码都可以在https://github.com/tuomaso/radial_rl_v2.上找到



## **4. Trustworthy Medical Segmentation with Uncertainty Estimation**

基于不确定性估计的可信医学分割 eess.IV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05978v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Rasool Ghulam, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.

摘要: 深度学习(DL)由于其精确性、效率和客观性，在重塑医疗系统方面有着巨大的希望。然而，DL模型对噪声和非分布输入的脆性阻碍了它们在临床上的部署。大多数系统在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。本文介绍了一种新的贝叶斯深度学习框架，用于分段神经网络中的不确定性量化，特别是编解码器的体系结构。该框架使用一阶泰勒级数近似，通过最大化证据下界来传播和学习给定训练数据的模型参数分布的前两个矩(均值和协方差)。输出由两幅图组成：分割后的图像和分割的不确定性图。通过预测分布的协方差矩阵来捕捉分割决策中的不确定性。我们在磁共振成像和计算机断层扫描的医学图像分割数据上对所提出的框架进行了评估。我们在多个基准数据集上的实验表明，与现有的分割模型相比，该框架对噪声和敌意攻击具有更强的鲁棒性。此外，该框架的不确定性图将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌意攻击破坏的补丁相关联。因此，当模型做出错误的预测或通过在不确定性图中呈现更高的值来错过部分分割结构(例如，肿瘤)时，该模型可以自我评估其分割决策。



## **5. Robust Learning via Ensemble Density Propagation in Deep Neural Networks**

基于集成密度传播的深度神经网络鲁棒学习 cs.LG

submitted to 2020 IEEE International Workshop on Machine Learning for  Signal Processing

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05953v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Lyudmila Mihaylova

**Abstracts**: Learning in uncertain, noisy, or adversarial environments is a challenging task for deep neural networks (DNNs). We propose a new theoretically grounded and efficient approach for robust learning that builds upon Bayesian estimation and Variational Inference. We formulate the problem of density propagation through layers of a DNN and solve it using an Ensemble Density Propagation (EnDP) scheme. The EnDP approach allows us to propagate moments of the variational probability distribution across the layers of a Bayesian DNN, enabling the estimation of the mean and covariance of the predictive distribution at the output of the model. Our experiments using MNIST and CIFAR-10 datasets show a significant improvement in the robustness of the trained models to random noise and adversarial attacks.

摘要: 对于深度神经网络(DNNs)来说，在不确定、噪声或敌对环境中学习是一项具有挑战性的任务。在贝叶斯估计和变分推理的基础上，提出了一种新的具有理论基础的、高效的鲁棒学习方法。我们用集合密度传播(ENDP)方案描述了DNN各层间的密度传播问题，并对其进行了求解。ENDP方法允许我们在贝叶斯DNN的各层之间传播变分概率分布的矩，从而能够在模型的输出处估计预测分布的均值和协方差。我们使用MNIST和CIFAR-10数据集进行的实验表明，训练后的模型对随机噪声和敌意攻击的鲁棒性有了显着的提高。



## **6. Audio Attacks and Defenses against AED Systems -- A Practical Study**

AED系统的音频攻击与防御--一项实用研究 cs.SD

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2106.07428v4)

**Authors**: Rodrigo dos Santos, Shirin Nilizadeh

**Abstracts**: In this paper, we evaluate deep learning-enabled AED systems against evasion attacks based on adversarial examples. We test the robustness of multiple security critical AED tasks, implemented as CNNs classifiers, as well as existing third-party Nest devices, manufactured by Google, which run their own black-box deep learning models. Our adversarial examples use audio perturbations made of white and background noises. Such disturbances are easy to create, to perform and to reproduce, and can be accessible to a large number of potential attackers, even non-technically savvy ones.   We show that an adversary can focus on audio adversarial inputs to cause AED systems to misclassify, achieving high success rates, even when we use small levels of a given type of noisy disturbance. For instance, on the case of the gunshot sound class, we achieve nearly 100% success rate when employing as little as 0.05 white noise level. Similarly to what has been previously done by works focusing on adversarial examples from the image domain as well as on the speech recognition domain. We then, seek to improve classifiers' robustness through countermeasures. We employ adversarial training and audio denoising. We show that these countermeasures, when applied to audio input, can be successful, either in isolation or in combination, generating relevant increases of nearly fifty percent in the performance of the classifiers when these are under attack.

摘要: 在这篇文章中，我们评估了深度学习使能的AED系统对逃避攻击的抵抗能力，这是基于敌意的例子。我们测试了多个安全关键AED任务(实现为CNNS分类器)以及由Google制造的现有第三方Nest设备的健壮性，这些设备运行自己的黑盒深度学习模型。我们的对抗性例子使用由白噪声和背景噪声构成的音频扰动。这样的干扰很容易制造、执行和复制，而且可以被大量潜在的攻击者接触到，即使是不懂技术的人也可以接触到。我们表明，即使当我们使用少量的给定类型的噪声干扰时，对手也可以专注于音频对手输入，导致AED系统错误分类，从而实现高成功率。例如，在枪声类的情况下，当使用低至0.05的白噪声水平时，我们获得了近100%的成功率。类似于先前通过关注来自图像域以及语音识别领域的对抗性示例的作品所做的工作。然后，通过对策寻求提高分类器的鲁棒性。我们采用对抗性训练和音频去噪。我们表明，当这些对策应用于音频输入时，无论是单独应用还是组合应用，都可以取得成功，当分类器受到攻击时，分类器的性能会相应提高近50%。



## **7. A black-box adversarial attack for poisoning clustering**

一种针对中毒聚类的黑盒对抗性攻击 cs.LG

18 pages, Pattern Recognition 2022

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2009.05474v4)

**Authors**: Antonio Emanuele Cinà, Alessandro Torcinovich, Marcello Pelillo

**Abstracts**: Clustering algorithms play a fundamental role as tools in decision-making and sensible automation processes. Due to the widespread use of these applications, a robustness analysis of this family of algorithms against adversarial noise has become imperative. To the best of our knowledge, however, only a few works have currently addressed this problem. In an attempt to fill this gap, in this work, we propose a black-box adversarial attack for crafting adversarial samples to test the robustness of clustering algorithms. We formulate the problem as a constrained minimization program, general in its structure and customizable by the attacker according to her capability constraints. We do not assume any information about the internal structure of the victim clustering algorithm, and we allow the attacker to query it as a service only. In the absence of any derivative information, we perform the optimization with a custom approach inspired by the Abstract Genetic Algorithm (AGA). In the experimental part, we demonstrate the sensibility of different single and ensemble clustering algorithms against our crafted adversarial samples on different scenarios. Furthermore, we perform a comparison of our algorithm with a state-of-the-art approach showing that we are able to reach or even outperform its performance. Finally, to highlight the general nature of the generated noise, we show that our attacks are transferable even against supervised algorithms such as SVMs, random forests, and neural networks.

摘要: 聚类算法在决策和明智的自动化过程中起着基础性的作用。由于这些应用的广泛使用，对这类算法进行抗对抗噪声的鲁棒性分析已成为当务之急。然而，就我们所知，目前只有几部著作解决了这个问题。为了填补这一空白，在这项工作中，我们提出了一种用于制作敌意样本的黑盒对抗性攻击，以测试聚类算法的健壮性。我们将问题描述为一个有约束的最小化规划，其结构一般，攻击者可以根据她的能力约束进行定制。我们不假定有关受害者群集算法的内部结构的任何信息，并且我们仅允许攻击者将其作为服务进行查询。在没有任何导数信息的情况下，受抽象遗传算法(AGA)的启发，采用定制的方法进行优化。在实验部分，我们展示了不同的单一聚类算法和集成聚类算法在不同场景下对我们制作的敌意样本的敏感度。此外，我们将我们的算法与最先进的方法进行了比较，结果表明我们能够达到甚至超过它的性能。最后，为了突出生成噪声的一般性质，我们证明了我们的攻击即使是针对支持向量机、随机森林和神经网络等有监督算法也是可以转移的。



## **8. Universal Multi-Party Poisoning Attacks**

普遍存在的多方中毒攻击 cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/1809.03474v3)

**Authors**: Saeed Mahloujifar, Mohammad Mahmoody, Ameer Mohammed

**Abstracts**: In this work, we demonstrate universal multi-party poisoning attacks that adapt and apply to any multi-party learning process with arbitrary interaction pattern between the parties. More generally, we introduce and study $(k,p)$-poisoning attacks in which an adversary controls $k\in[m]$ of the parties, and for each corrupted party $P_i$, the adversary submits some poisoned data $\mathcal{T}'_i$ on behalf of $P_i$ that is still ``$(1-p)$-close'' to the correct data $\mathcal{T}_i$ (e.g., $1-p$ fraction of $\mathcal{T}'_i$ is still honestly generated). We prove that for any ``bad'' property $B$ of the final trained hypothesis $h$ (e.g., $h$ failing on a particular test example or having ``large'' risk) that has an arbitrarily small constant probability of happening without the attack, there always is a $(k,p)$-poisoning attack that increases the probability of $B$ from $\mu$ to by $\mu^{1-p \cdot k/m} = \mu + \Omega(p \cdot k/m)$. Our attack only uses clean labels, and it is online.   More generally, we prove that for any bounded function $f(x_1,\dots,x_n) \in [0,1]$ defined over an $n$-step random process $\mathbf{X} = (x_1,\dots,x_n)$, an adversary who can override each of the $n$ blocks with even dependent probability $p$ can increase the expected output by at least $\Omega(p \cdot \mathrm{Var}[f(\mathbf{x})])$.

摘要: 在这项工作中，我们展示了通用的多方中毒攻击，适用于任何具有任意交互模式的多方学习过程。更一般地，我们引入和研究了$(k，p)$中毒攻击，在这种攻击中，敌手控制着当事人的$k\in[m]$，并且对于每个被破坏的一方$P_i$，对手代表$P_i$提交一些有毒数据$\数学{T}‘_i$，该数据仍然是’‘$(1-p)$-接近’‘正确数据$\数学{T}_i$(例如，$1-p$分数$\我们证明了对于最终训练假设$h$的任何“坏”性质$B$(例如，$h$在特定的测试用例上失败或具有“大的”风险)，在没有攻击的情况下发生的概率任意小的恒定概率，总是存在$(k，p)$中毒攻击，它将$B$的概率从$\µ$增加到$\µ^{1-p\CDOT k/m}=\Mu+\Omega(p\CDOT k/m)$。我们的攻击只使用干净的标签，而且是在线的。更一般地，我们证明了对于定义在$n$步随机过程$\mathbf{X}=(x_1，\点，x_n)$上的[0，1]$中的任何有界函数$f(x_1，\dots，x_n)\，能够以偶数依赖概率$p$覆盖$n$块中的每个块的对手可以使期望输出至少增加$\Omega(p\cdot\mathm{var}[f(\mam



## **9. Distributionally Robust Trajectory Optimization Under Uncertain Dynamics via Relative Entropy Trust-Regions**

基于相对熵信赖域的不确定动态分布鲁棒轨迹优化 eess.SY

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2103.15388v2)

**Authors**: Hany Abdulsamad, Tim Dorau, Boris Belousov, Jia-Jie Zhu, Jan Peters

**Abstracts**: Trajectory optimization and model predictive control are essential techniques underpinning advanced robotic applications, ranging from autonomous driving to full-body humanoid control. State-of-the-art algorithms have focused on data-driven approaches that infer the system dynamics online and incorporate posterior uncertainty during planning and control. Despite their success, such approaches are still susceptible to catastrophic errors that may arise due to statistical learning biases, unmodeled disturbances or even directed adversarial attacks. In this paper, we tackle the problem of dynamics mismatch and propose a distributionally robust optimal control formulation that alternates between two relative entropy trust-region optimization problems. Our method finds the worst-case maximum entropy Gaussian posterior over the dynamics parameters and the corresponding robust policy. We show that our approach admits a closed-form backward-pass for a certain class of systems and demonstrate the resulting robustness on linear and nonlinear numerical examples.

摘要: 轨迹优化和模型预测控制是支撑先进机器人应用的关键技术，从自动驾驶到全身仿人控制。最先进的算法专注于数据驱动的方法，这些方法在线推断系统动态，并在计划和控制过程中纳入后验不确定性。尽管这些方法取得了成功，但它们仍然容易受到灾难性错误的影响，这些错误可能是由于统计学习偏差、未建模的干扰甚至是定向的对抗性攻击而产生的。本文对动态失配问题进行了撞击研究，提出了一种在两个相对熵信赖域优化问题之间交替的分布鲁棒最优控制公式。我们的方法求出了动力学参数的最坏情况下的最大熵高斯后验分布，并给出了相应的鲁棒策略。我们证明了我们的方法对于一类系统允许闭合形式的后向传递，并在线性和非线性数值例子上证明了所得到的结果的鲁棒性。



## **10. Sparse Adversarial Video Attacks with Spatial Transformations**

基于空间变换的稀疏对抗性视频攻击 cs.CV

The short version of this work will appear in the BMVC 2021  conference

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05468v1)

**Authors**: Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Qiang Ni

**Abstracts**: In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA. Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index (SSIM) measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.

摘要: 近年来，大量的研究工作集中在图像的对抗性攻击上，而对抗性视频攻击的研究很少。我们提出了一种针对视频的对抗性攻击策略，称为DeepSAVA。我们的模型通过一个统一的优化框架同时包括加性扰动和空间变换，其中采用结构相似指数(SSIM)度量对抗距离。我们设计了一种有效和新颖的优化方案，它交替使用贝叶斯优化来识别视频中最有影响力的帧，以及基于随机梯度下降(SGD)的优化来产生加性和空间变换的扰动。这样做使DeepSAVA能够对视频执行非常稀疏的攻击，以保持人的不可感知性，同时在攻击成功率和对手可转移性方面仍获得最先进的性能。我们在不同类型的深度神经网络和视频数据集上的密集实验证实了DeepSAVA的优越性。



## **11. Are Transformers More Robust Than CNNs?**

变形金刚比CNN更健壮吗？ cs.CV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05464v1)

**Authors**: Yutong Bai, Jieru Mei, Alan Yuille, Cihang Xie

**Abstracts**: Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks, recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations.   With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. The code and models are publicly available at https://github.com/ytongbai/ViTs-vs-CNNs.

摘要: 变压器作为一种强有力的视觉识别工具应运而生。除了展示好胜在广泛的可视基准上的性能外，最近的研究还认为，变形金刚比卷积神经网络(CNN)更健壮。然而，令人惊讶的是，我们发现这些结论是从不公平的实验环境中得出的，在这些实验环境中，变形金刚和CNN在不同的尺度上进行了比较，并应用了不同的训练框架。在本文中，我们的目标是提供变压器和CNN之间的第一次公平和深入的比较，重点是鲁棒性评估。有了我们的统一训练设置，我们首先挑战了以前的信念，即在衡量对手的健壮性时，变形金刚优于CNN。更令人惊讶的是，我们发现，如果CNN恰当地采用了变形金刚的训练食谱，它们在防御对手攻击方面可以很容易地像变形金刚一样健壮。虽然关于分布外样本的泛化，我们表明(外部)大规模数据集的预训练并不是使Transformers获得比CNN更好的性能的基本要求。此外，我们的消融表明，这种更强的概括性在很大程度上得益于变形金刚的自我关注式架构本身，而不是其他培训设置。我们希望这项工作可以帮助社区更好地理解Transformers和CNN的健壮性，并对其进行基准测试。代码和模型可在https://github.com/ytongbai/ViTs-vs-CNNs.上公开获得



## **12. Statistical Perspectives on Reliability of Artificial Intelligence Systems**

人工智能系统可靠性的统计透视 cs.SE

40 pages

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05391v1)

**Authors**: Yili Hong, Jiayi Lian, Li Xu, Jie Min, Yueyao Wang, Laura J. Freeman, Xinwei Deng

**Abstracts**: Artificial intelligence (AI) systems have become increasingly popular in many areas. Nevertheless, AI technologies are still in their developing stages, and many issues need to be addressed. Among those, the reliability of AI systems needs to be demonstrated so that the AI systems can be used with confidence by the general public. In this paper, we provide statistical perspectives on the reliability of AI systems. Different from other considerations, the reliability of AI systems focuses on the time dimension. That is, the system can perform its designed functionality for the intended period. We introduce a so-called SMART statistical framework for AI reliability research, which includes five components: Structure of the system, Metrics of reliability, Analysis of failure causes, Reliability assessment, and Test planning. We review traditional methods in reliability data analysis and software reliability, and discuss how those existing methods can be transformed for reliability modeling and assessment of AI systems. We also describe recent developments in modeling and analysis of AI reliability and outline statistical research challenges in this area, including out-of-distribution detection, the effect of the training set, adversarial attacks, model accuracy, and uncertainty quantification, and discuss how those topics can be related to AI reliability, with illustrative examples. Finally, we discuss data collection and test planning for AI reliability assessment and how to improve system designs for higher AI reliability. The paper closes with some concluding remarks.

摘要: 人工智能(AI)系统在许多领域变得越来越受欢迎。然而，人工智能技术仍处于发展阶段，许多问题需要解决。其中，需要证明人工智能系统的可靠性，以便普通公众可以放心地使用人工智能系统。在这篇文章中，我们提供了关于人工智能系统可靠性的统计观点。与其他考虑因素不同，人工智能系统的可靠性侧重于时间维度。也就是说，系统可以在预期的时间段内执行其设计的功能。介绍了一种用于人工智能可靠性研究的智能统计框架，该框架包括五个组成部分：系统结构、可靠性度量、故障原因分析、可靠性评估和测试规划。我们回顾了可靠性数据分析和软件可靠性的传统方法，并讨论了如何将这些现有方法转化为人工智能系统的可靠性建模和评估。我们还描述了人工智能可靠性建模和分析的最新进展，并概述了该领域的统计研究挑战，包括分布失调检测、训练集的影响、对抗性攻击、模型精度和不确定性量化，并用说明性例子讨论了这些主题如何与人工智能可靠性相关。最后，我们讨论了人工智能可靠性评估的数据收集和测试规划，以及如何改进系统设计以提高人工智能可靠性。论文最后以一些结束语结束。



## **13. TDGIA:Effective Injection Attacks on Graph Neural Networks**

TDGIA：对图神经网络的有效注入攻击 cs.LG

KDD 2021 research track paper

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2106.06663v2)

**Authors**: Xu Zou, Qinkai Zheng, Yuxiao Dong, Xinyu Guan, Evgeny Kharlamov, Jialiang Lu, Jie Tang

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. However, recent studies have shown that GNNs are vulnerable to adversarial attacks. In this paper, we study a recently-introduced realistic attack scenario on graphs -- graph injection attack (GIA). In the GIA scenario, the adversary is not able to modify the existing link structure and node attributes of the input graph, instead the attack is performed by injecting adversarial nodes into it. We present an analysis on the topological vulnerability of GNNs under GIA setting, based on which we propose the Topological Defective Graph Injection Attack (TDGIA) for effective injection attacks. TDGIA first introduces the topological defective edge selection strategy to choose the original nodes for connecting with the injected ones. It then designs the smooth feature optimization objective to generate the features for the injected nodes. Extensive experiments on large-scale datasets show that TDGIA can consistently and significantly outperform various attack baselines in attacking dozens of defense GNN models. Notably, the performance drop on target GNNs resultant from TDGIA is more than double the damage brought by the best attack solution among hundreds of submissions on KDD-CUP 2020.

摘要: 图神经网络(GNNs)在各种实际应用中取得了良好的性能。然而，最近的研究表明，GNN很容易受到敌意攻击。本文研究了最近引入的一种图的现实攻击场景--图注入攻击(GIA)。在GIA场景中，敌手不能修改输入图的现有链接结构和节点属性，而是通过向其中注入敌方节点来执行攻击。分析了GIA环境下GNNs的拓扑脆弱性，在此基础上提出了针对有效注入攻击的拓扑缺陷图注入攻击(TDGIA)。TDGIA首先引入拓扑缺陷边选择策略，选择原始节点与注入节点连接。然后设计平滑特征优化目标，为注入节点生成特征。在大规模数据集上的广泛实验表明，TDGIA在攻击数十个防御GNN模型时，可以一致且显着地优于各种攻击基线。值得注意的是，TDGIA对目标GNN造成的性能下降是KDD-Cup 2020上百份提交的最佳攻击解决方案带来的损害的两倍多。



## **14. A Unified Game-Theoretic Interpretation of Adversarial Robustness**

对抗性稳健性的统一博弈论解释 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2103.07364v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, i.e. the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.

摘要: 本文提供了一个统一的视角来解释不同的对抗性攻击和防御方法，即DNNs的输入变量之间的多阶交互的观点。基于多阶交互，我们发现对抗性攻击主要影响高阶交互来欺骗DNN。此外，我们发现对抗性训练的DNN的鲁棒性来自于特定类别的低阶交互。我们的发现为统一对抗性扰动和鲁棒性提供了一种潜在的方法，可以对现有的防御方法进行原则性的解释。此外，我们的发现还修正了以往对对抗性习得特征的形状偏向的不准确理解。



## **15. Membership Inference Attacks Against Self-supervised Speech Models**

针对自监督语音模型的隶属度推理攻击 cs.CR

Submitted to ICASSP 2022. Source code available at  https://github.com/RayTzeng/s3m-membership-inference

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05113v1)

**Authors**: Wei-Cheng Tseng, Wei-Tsung Kao, Hung-yi Lee

**Abstracts**: Recently, adapting the idea of self-supervised learning (SSL) on continuous speech has started gaining attention. SSL models pre-trained on a huge amount of unlabeled audio can generate general-purpose representations that benefit a wide variety of speech processing tasks. Despite their ubiquitous deployment, however, the potential privacy risks of these models have not been well investigated. In this paper, we present the first privacy analysis on several SSL speech models using Membership Inference Attacks (MIA) under black-box access. The experiment results show that these pre-trained models are vulnerable to MIA and prone to membership information leakage with high adversarial advantage scores in both utterance-level and speaker-level. Furthermore, we also conduct several ablation studies to understand the factors that contribute to the success of MIA.

摘要: 最近，将自我监督学习(SSL)的思想应用于连续语音的研究开始受到关注。在大量未标记音频上预先训练的SSL模型可以生成有利于各种语音处理任务的通用表示。然而，尽管它们的部署无处不在，但这些模型的潜在隐私风险还没有得到很好的调查。本文首次对几种SSL语音模型在黑盒访问下使用成员推理攻击(MIA)进行了隐私分析。实验结果表明，这些预训练模型在话语级和说话人级都有较高的对手优势得分，容易受到MIA的影响，容易泄露成员信息。此外，我们还进行了几项消融研究，以了解导致MIA成功的因素。



## **16. A Statistical Difference Reduction Method for Escaping Backdoor Detection**

一种逃避后门检测的统计减差方法 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05077v1)

**Authors**: Pengfei Xia, Hongjing Niu, Ziqiang Li, Bin Li

**Abstracts**: Recent studies show that Deep Neural Networks (DNNs) are vulnerable to backdoor attacks. An infected model behaves normally on benign inputs, whereas its prediction will be forced to an attack-specific target on adversarial data. Several detection methods have been developed to distinguish inputs to defend against such attacks. The common hypothesis that these defenses rely on is that there are large statistical differences between the latent representations of clean and adversarial inputs extracted by the infected model. However, although it is important, comprehensive research on whether the hypothesis must be true is lacking. In this paper, we focus on it and study the following relevant questions: 1) What are the properties of the statistical differences? 2) How to effectively reduce them without harming the attack intensity? 3) What impact does this reduction have on difference-based defenses? Our work is carried out on the three questions. First, by introducing the Maximum Mean Discrepancy (MMD) as the metric, we identify that the statistical differences of multi-level representations are all large, not just the highest level. Then, we propose a Statistical Difference Reduction Method (SDRM) by adding a multi-level MMD constraint to the loss function during training a backdoor model to effectively reduce the differences. Last, three typical difference-based detection methods are examined. The F1 scores of these defenses drop from 90%-100% on the regularly trained backdoor models to 60%-70% on the models trained with SDRM on all two datasets, four model architectures, and four attack methods. The results indicate that the proposed method can be used to enhance existing attacks to escape backdoor detection algorithms.

摘要: 最近的研究表明，深度神经网络(DNNs)很容易受到后门攻击。被感染的模型在良性输入上表现正常，而它的预测将被迫在对抗性数据上针对攻击特定的目标。已经开发了几种检测方法来区分输入以防御此类攻击。这些防御所依赖的共同假设是，由感染模型提取的干净和敌对输入的潜在表示之间存在很大的统计差异。然而，尽管这很重要，但关于这一假设是否一定是真的缺乏全面的研究。本文针对这一问题进行了研究：1)统计差异的性质是什么？2)如何在不影响攻击强度的情况下有效地降低统计差异？3)这种减少对基于差异的防御有什么影响？(2)如何在不影响攻击强度的情况下有效地减少统计差异？3)这种减少对基于差异的防御有什么影响？我们的工作就是围绕这三个问题展开的。首先，通过引入最大平均差异(MMD)作为度量，我们发现多级表示的统计差异都很大，而不仅仅是最高级别。然后，在后门模型训练过程中，通过在损失函数中加入多级MMD约束，提出了一种统计差值缩减方法(SDRM)，有效地减小了差值。最后，分析了三种典型的基于差分的检测方法。在所有两个数据集、四个模型体系结构和四种攻击方法上，这些防御的F1得分从定期训练的后门模型的90%-100%下降到使用SDRM训练的模型的60%-70%。实验结果表明，该方法可用于增强现有的逃避后门检测算法的攻击。



## **17. Tightening the Approximation Error of Adversarial Risk with Auto Loss Function Search**

用自动损失函数搜索法缩小对抗性风险的逼近误差 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05063v1)

**Authors**: Pengfei Xia, Ziqiang Li, Bin Li

**Abstracts**: Numerous studies have demonstrated that deep neural networks are easily misled by adversarial examples. Effectively evaluating the adversarial robustness of a model is important for its deployment in practical applications. Currently, a common type of evaluation is to approximate the adversarial risk of a model as a robustness indicator by constructing malicious instances and executing attacks. Unfortunately, there is an error (gap) between the approximate value and the true value. Previous studies manually design attack methods to achieve a smaller error, which is inefficient and may miss a better solution. In this paper, we establish the tightening of the approximation error as an optimization problem and try to solve it with an algorithm. More specifically, we first analyze that replacing the non-convex and discontinuous 0-1 loss with a surrogate loss, a necessary compromise in calculating the approximation, is one of the main reasons for the error. Then we propose AutoLoss-AR, the first method for searching loss functions for tightening the approximation error of adversarial risk. Extensive experiments are conducted in multiple settings. The results demonstrate the effectiveness of the proposed method: the best-discovered loss functions outperform the handcrafted baseline by 0.9%-2.9% and 0.7%-2.0% on MNIST and CIFAR-10, respectively. Besides, we also verify that the searched losses can be transferred to other settings and explore why they are better than the baseline by visualizing the local loss landscape.

摘要: 大量研究表明，深度神经网络很容易被对抗性例子所误导。有效地评估模型的对抗健壮性对于其在实际应用中的部署具有重要意义。目前，一种常见的评估方法是通过构建恶意实例和执行攻击来近似模型的敌意风险作为健壮性指标。不幸的是，近似值和真实值之间存在误差(差距)。以往的研究都是通过手工设计攻击方法来实现较小的错误，效率较低，可能会错过更好的解决方案。本文将逼近误差的收紧问题建立为优化问题，并尝试用算法求解。更具体地说，我们首先分析了用替代损失代替非凸的、不连续的0-1损失是造成误差的主要原因之一，这是计算近似时的一种必要的折衷。在此基础上，提出了第一种搜索损失函数的方法AutoLoss-AR，以减小对手风险的逼近误差。在多个环境中进行了广泛的实验。结果证明了该方法的有效性：在MNIST和CIFAR-10上，最好发现的损失函数的性能分别比手工制作的基线高0.9%-2.9%和0.7%-2.0%。此外，我们还验证了搜索到的损失可以转移到其他设置，并通过可视化本地损失情况来探索为什么它们比基线更好。



## **18. GraphAttacker: A General Multi-Task GraphAttack Framework**

GraphAttacker：一个通用的多任务GraphAttack框架 cs.LG

17 pages,9 figeures

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2101.06855v2)

**Authors**: Jinyin Chen, Dunjie Zhang, Zhaoyan Ming, Kejie Huang, Wenrong Jiang, Chen Cui

**Abstracts**: Graph neural networks (GNNs) have been successfully exploited in graph analysis tasks in many real-world applications. The competition between attack and defense methods also enhances the robustness of GNNs. In this competition, the development of adversarial training methods put forward higher requirement for the diversity of attack examples. By contrast, most attack methods with specific attack strategies are difficult to satisfy such a requirement. To address this problem, we propose GraphAttacker, a novel generic graph attack framework that can flexibly adjust the structures and the attack strategies according to the graph analysis tasks. GraphAttacker generates adversarial examples through alternate training on three key components: the multi-strategy attack generator (MAG), the similarity discriminator (SD), and the attack discriminator (AD), based on the generative adversarial network (GAN). Furthermore, we introduce a novel similarity modification rate SMR to conduct a stealthier attack considering the change of node similarity distribution. Experiments on various benchmark datasets demonstrate that GraphAttacker can achieve state-of-the-art attack performance on graph analysis tasks of node classification, graph classification, and link prediction, no matter the adversarial training is conducted or not. Moreover, we also analyze the unique characteristics of each task and their specific response in the unified attack framework. The project code is available at https://github.com/honoluluuuu/GraphAttacker.

摘要: 图神经网络(GNNs)已被成功地应用于许多实际应用中的图分析任务中。攻防手段的竞争也增强了GNNs的健壮性。在本次比赛中，对抗性训练方法的发展对进攻实例的多样性提出了更高的要求。相比之下，大多数具有特定攻击策略的攻击方法很难满足这一要求。针对这一问题，我们提出了一种新的通用图攻击框架GraphAttacker，该框架可以根据图分析任务灵活调整攻击结构和攻击策略。GraphAttacker在生成对抗网络(GAN)的基础上，通过交替训练多策略攻击生成器(MAG)、相似判别器(SD)和攻击鉴别器(AD)这三个关键部件来生成对抗性实例。此外，考虑到节点相似度分布的变化，引入了一种新的相似度修改率SMR来进行隐身攻击。在不同的基准数据集上的实验表明，无论是否进行对抗性训练，GraphAttacker都可以在节点分类、图分类和链接预测等图分析任务上获得最先进的攻击性能。此外，我们还分析了在统一攻击框架下每个任务的独特性和它们的具体响应。项目代码可在https://github.com/honoluluuuu/GraphAttacker.上找到



## **19. Reversible Attack based on Local Visual Adversarial Perturbation**

基于局部视觉对抗扰动的可逆攻击 cs.CV

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2110.02700v2)

**Authors**: Li Chen, Shaowei Zhu, Zhaoxia Yin

**Abstracts**: Deep learning is getting more and more outstanding performance in many tasks such as autonomous driving and face recognition and also has been challenged by different kinds of attacks. Adding perturbations that are imperceptible to human vision in an image can mislead the neural network model to get wrong results with high confidence. Adversarial Examples are images that have been added with specific noise to mislead a deep neural network model However, adding noise to images destroys the original data, making the examples useless in digital forensics and other fields. To prevent illegal or unauthorized access of image data such as human faces and ensure no affection to legal use reversible adversarial attack technique is rise. The original image can be recovered from its reversible adversarial example. However, the existing reversible adversarial examples generation strategies are all designed for the traditional imperceptible adversarial perturbation. How to get reversibility for locally visible adversarial perturbation? In this paper, we propose a new method for generating reversible adversarial examples based on local visual adversarial perturbation. The information needed for image recovery is embedded into the area beyond the adversarial patch by reversible data hiding technique. To reduce image distortion and improve visual quality, lossless compression and B-R-G embedding principle are adopted. Experiments on ImageNet dataset show that our method can restore the original images error-free while ensuring the attack performance.

摘要: 深度学习在自动驾驶、人脸识别等任务中表现越来越突出，也受到了各种攻击的挑战。在图像中添加人眼无法察觉的扰动可能会误导神经网络模型，使其在高置信度下得到错误的结果。对抗性的例子是添加了特定噪声的图像，以误导深度神经网络模型。然而，向图像添加噪声会破坏原始数据，使这些示例在数字取证和其他领域中无用。为了防止对人脸等图像数据的非法或未经授权的访问，确保不影响合法使用，可逆对抗攻击技术应运而生。原始图像可以从其可逆的对抗性示例中恢复。然而，现有的可逆对抗性实例生成策略都是针对传统的潜意识对抗性扰动而设计的。如何获得局部可见的对抗性扰动的可逆性？本文提出了一种基于局部可视对抗性扰动的可逆对抗性实例生成方法。利用可逆数据隐藏技术将图像恢复所需的信息嵌入到敌方补丁之外的区域。为了减少图像失真，提高视觉质量，采用无损压缩和B-R-G嵌入原理。在ImageNet数据集上的实验表明，该方法可以在保证攻击性能的前提下无差错地恢复原始图像。



## **20. On Robustness of Neural Ordinary Differential Equations**

关于神经常微分方程的稳健性 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/1910.05513v3)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks. \url{https://github.com/HanshuYAN/TisODE}

摘要: 近年来，神经常微分方程(ODE)在各个研究领域受到越来越多的关注。已有一些研究神经常微分方程的优化问题和逼近能力的工作，但其鲁棒性尚不清楚。在这项工作中，我们通过从经验和理论上探索神经ODE的稳健性来填补这一重要空白。我们首先对基于ODENET的神经网络(ODENet)的鲁棒性进行了实证研究，方法是将ODENet暴露在具有各种类型扰动的输入中，然后研究相应输出的变化。与传统的卷积神经网络(CNNs)相比，我们发现ODENet对随机高斯扰动和敌意攻击示例都具有更强的鲁棒性。然后，我们通过利用连续时间颂歌的流的某些理想性质，即积分曲线是不相交的，来提供对这一现象的深刻理解。我们的工作表明，由于其固有的鲁棒性，使用神经ODE作为构建鲁棒深层网络模型的基础挡路是很有前途的。为了进一步增强香草神经微分方程组的鲁棒性，我们提出了时不变稳态神经微分方程组(TisODE)，它通过时不变性和施加稳态约束来规则化扰动数据上的流动。我们表明，TisODE方法的性能优于香草神经ODE方法，并且还可以与其他最先进的体系结构方法相结合来构建更健壮的深层网络。\url{https://github.com/HanshuYAN/TisODE}



## **21. Bayesian Framework for Gradient Leakage**

梯度泄漏的贝叶斯框架 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04706v1)

**Authors**: Mislav Balunović, Dimitar I. Dimitrov, Robin Staab, Martin Vechev

**Abstracts**: Federated learning is an established method for training machine learning models without sharing training data. However, recent work has shown that it cannot guarantee data privacy as shared gradients can still leak sensitive information. To formalize the problem of gradient leakage, we propose a theoretical framework that enables, for the first time, analysis of the Bayes optimal adversary phrased as an optimization problem. We demonstrate that existing leakage attacks can be seen as approximations of this optimal adversary with different assumptions on the probability distributions of the input data and gradients. Our experiments confirm the effectiveness of the Bayes optimal adversary when it has knowledge of the underlying distribution. Further, our experimental evaluation shows that several existing heuristic defenses are not effective against stronger attacks, especially early in the training process. Thus, our findings indicate that the construction of more effective defenses and their evaluation remains an open problem.

摘要: 联合学习是一种在不共享训练数据的情况下训练机器学习模型的既定方法。然而，最近的研究表明，它不能保证数据隐私，因为共享梯度仍然可能泄露敏感信息。为了形式化梯度泄漏问题，我们提出了一个理论框架，该框架首次能够将贝叶斯最优对手表述为优化问题进行分析。我们证明了现有的泄漏攻击可以看作是对输入数据的概率分布和梯度的不同假设的最优对手的近似。我们的实验证实了贝叶斯最优对手在知道潜在分布的情况下的有效性。此外，我们的实验评估表明，现有的几种启发式防御方法对更强的攻击并不有效，特别是在训练过程的早期。因此，我们的研究结果表明，构建更有效的防御体系及其评估仍然是一个悬而未决的问题。



## **22. HAPSSA: Holistic Approach to PDF Malware Detection Using Signal and Statistical Analysis**

HAPSSA：基于信号和统计分析的PDF恶意软件整体检测方法 cs.CR

Submitted version - MILCOM 2021 IEEE Military Communications  Conference

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04703v1)

**Authors**: Tajuddin Manhar Mohammed, Lakshmanan Nataraj, Satish Chikkagoudar, Shivkumar Chandrasekaran, B. S. Manjunath

**Abstracts**: Malicious PDF documents present a serious threat to various security organizations that require modern threat intelligence platforms to effectively analyze and characterize the identity and behavior of PDF malware. State-of-the-art approaches use machine learning (ML) to learn features that characterize PDF malware. However, ML models are often susceptible to evasion attacks, in which an adversary obfuscates the malware code to avoid being detected by an Antivirus. In this paper, we derive a simple yet effective holistic approach to PDF malware detection that leverages signal and statistical analysis of malware binaries. This includes combining orthogonal feature space models from various static and dynamic malware detection methods to enable generalized robustness when faced with code obfuscations. Using a dataset of nearly 30,000 PDF files containing both malware and benign samples, we show that our holistic approach maintains a high detection rate (99.92%) of PDF malware and even detects new malicious files created by simple methods that remove the obfuscation conducted by malware authors to hide their malware, which are undetected by most antiviruses.

摘要: 恶意PDF文档对各种安全组织构成严重威胁，这些组织需要现代威胁情报平台来有效地分析和表征PDF恶意软件的身份和行为。最先进的方法使用机器学习(ML)来学习PDF恶意软件的特征。然而，ML模型经常容易受到规避攻击，在这种攻击中，敌手混淆恶意软件代码以避免被防病毒程序检测到。在本文中，我们推导了一种简单而有效的整体PDF恶意软件检测方法，该方法利用恶意软件二进制文件的信号和统计分析。这包括组合来自各种静电的正交特征空间模型和动态恶意软件检测方法，以在面临代码混淆时实现普遍的鲁棒性。使用包含近30,000个包含恶意软件和良性样本的PDF文件的数据集，我们显示，我们的整体方法保持了较高的PDF恶意软件检测率(99.92%)，甚至可以检测到通过简单方法创建的新恶意文件，这些新的恶意文件消除了恶意软件作者为隐藏恶意软件而进行的混淆，而这些文件是大多数防病毒软件无法检测到的。



## **23. A Separation Result Between Data-oblivious and Data-aware Poisoning Attacks**

数据迟钝和数据感知中毒攻击的分离结果 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2003.12020v2)

**Authors**: Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Abhradeep Thakurta

**Abstracts**: Poisoning attacks have emerged as a significant security threat to machine learning algorithms. It has been demonstrated that adversaries who make small changes to the training set, such as adding specially crafted data points, can hurt the performance of the output model. Some of the stronger poisoning attacks require the full knowledge of the training data. This leaves open the possibility of achieving the same attack results using poisoning attacks that do not have the full knowledge of the clean training set.   In this work, we initiate a theoretical study of the problem above. Specifically, for the case of feature selection with LASSO, we show that full-information adversaries (that craft poisoning examples based on the rest of the training data) are provably stronger than the optimal attacker that is oblivious to the training set yet has access to the distribution of the data. Our separation result shows that the two setting of data-aware and data-oblivious are fundamentally different and we cannot hope to always achieve the same attack or defense results in these scenarios.

摘要: 中毒攻击已经成为机器学习算法的重大安全威胁。已经证明，对训练集进行微小更改的对手，例如添加巧尽心思构建的数据点，可能会损害输出模型的性能。一些更强的中毒攻击需要完全了解训练数据。这使得使用不完全了解干净训练集的中毒攻击获得相同的攻击结果的可能性仍然存在。在这项工作中，我们开始了对上述问题的理论研究。具体地说，对于使用套索进行特征选择的情况，我们证明了全信息对手(基于训练数据的睡觉来制作中毒实例)比最优攻击者(即对训练集是迟钝但可以访问数据分布的攻击者)更强大。我们的分离结果表明，数据感知和数据迟钝这两个设置是根本不同的，我们不能指望在这些场景下总是能达到相同的攻防效果。



## **24. DeepSteal: Advanced Model Extractions Leveraging Efficient Weight Stealing in Memories**

DeepSteal：高级模型提取，利用记忆中有效的重量窃取 cs.CR

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04625v1)

**Authors**: Adnan Siraj Rakin, Md Hafizul Islam Chowdhuryy, Fan Yao, Deliang Fan

**Abstracts**: Recent advancements of Deep Neural Networks (DNNs) have seen widespread deployment in multiple security-sensitive domains. The need of resource-intensive training and use of valuable domain-specific training data have made these models a top intellectual property (IP) for model owners. One of the major threats to the DNN privacy is model extraction attacks where adversaries attempt to steal sensitive information in DNN models. Recent studies show hardware-based side channel attacks can reveal internal knowledge about DNN models (e.g., model architectures) However, to date, existing attacks cannot extract detailed model parameters (e.g., weights/biases). In this work, for the first time, we propose an advanced model extraction attack framework DeepSteal that effectively steals DNN weights with the aid of memory side-channel attack. Our proposed DeepSteal comprises two key stages. Firstly, we develop a new weight bit information extraction method, called HammerLeak, through adopting the rowhammer based hardware fault technique as the information leakage vector. HammerLeak leverages several novel system-level techniques tailed for DNN applications to enable fast and efficient weight stealing. Secondly, we propose a novel substitute model training algorithm with Mean Clustering weight penalty, which leverages the partial leaked bit information effectively and generates a substitute prototype of the target victim model. We evaluate this substitute model extraction method on three popular image datasets (e.g., CIFAR-10/100/GTSRB) and four DNN architectures (e.g., ResNet-18/34/Wide-ResNet/VGG-11). The extracted substitute model has successfully achieved more than 90 % test accuracy on deep residual networks for the CIFAR-10 dataset. Moreover, our extracted substitute model could also generate effective adversarial input samples to fool the victim model.

摘要: 近年来，深度神经网络(DNNs)在多个安全敏感领域得到了广泛的应用。对资源密集型培训的需求和对有价值的特定领域培训数据的使用已使这些模型成为模型所有者的最高知识产权(IP)。DNN隐私面临的主要威胁之一是模型提取攻击，即攻击者试图窃取DNN模型中的敏感信息。最近的研究表明，基于硬件的侧信道攻击可以揭示DNN模型(例如，模型体系结构)的内部知识，然而，到目前为止，现有的攻击不能提取详细的模型参数(例如，权重/偏差)。在这项工作中，我们首次提出了一个高级模型提取攻击框架DeepSteal，该框架可以借助记忆边信道攻击有效地窃取DNN权重。我们建议的DeepSteal包括两个关键阶段。首先，通过采用基于Rowhammer的硬件故障技术作为信息泄漏向量，提出了一种新的加权比特信息提取方法HammerLeak。HammerLeak利用针对DNN应用的几种新颖的系统级技术来实现快速高效的重量盗窃。其次，提出了一种基于均值聚类权重惩罚的替身模型训练算法，该算法有效地利用了部分泄露的比特信息，生成了目标受害者模型的替身原型。我们在三个流行的图像数据集(如CIFAR10/10 0/GTSRB)和四个数字近邻结构(如Resnet-18/34/Wide-Resnet/VGG-11)上对该替身模型提取方法进行了评估。所提取的替身模型在CIFAR-10数据集上的深层残差网络上的测试准确率已成功达到90%以上。此外，我们提取的替身模型还可以生成有效的敌意输入样本来愚弄受害者模型。



## **25. Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training**

安全胜过遗憾：通过对抗性训练防止妄想对手 cs.LG

NeurIPS 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2102.04716v3)

**Authors**: Lue Tao, Lei Feng, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Delusive attacks aim to substantially deteriorate the test accuracy of the learning model by slightly perturbing the features of correctly labeled training examples. By formalizing this malicious attack as finding the worst-case training data within a specific $\infty$-Wasserstein ball, we show that minimizing adversarial risk on the perturbed data is equivalent to optimizing an upper bound of natural risk on the original data. This implies that adversarial training can serve as a principled defense against delusive attacks. Thus, the test accuracy decreased by delusive attacks can be largely recovered by adversarial training. To further understand the internal mechanism of the defense, we disclose that adversarial training can resist the delusive perturbations by preventing the learner from overly relying on non-robust features in a natural setting. Finally, we complement our theoretical findings with a set of experiments on popular benchmark datasets, which show that the defense withstands six different practical attacks. Both theoretical and empirical results vote for adversarial training when confronted with delusive adversaries.

摘要: 妄想攻击的目的是通过对正确标记的训练样本的特征进行轻微扰动来显著降低学习模型的测试精度。通过将这种恶意攻击形式化为在特定的$\infty$-Wasserstein球中寻找最坏情况的训练数据，我们表明最小化扰动数据上的敌意风险等价于优化原始数据上的自然风险上界。这意味着对抗性训练可以作为对抗妄想攻击的原则性防御。因此，通过对抗性训练可以在很大程度上恢复由于妄想攻击而降低的测试精度。为了进一步了解防御的内在机制，我们揭示了对抗性训练可以通过防止学习者在自然环境中过度依赖非鲁棒特征来抵抗妄想干扰。最后，我们通过在流行的基准数据集上的一组实验来补充我们的理论发现，这些实验表明该防御系统可以抵御六种不同的实际攻击。在面对妄想性对手时，无论是理论结果还是经验结果都支持对抗性训练。



## **26. Robust and Information-theoretically Safe Bias Classifier against Adversarial Attacks**

抗敌意攻击的稳健且信息理论安全的偏向分类器 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04404v1)

**Authors**: Lijia Yu, Xiao-Shan Gao

**Abstracts**: In this paper, the bias classifier is introduced, that is, the bias part of a DNN with Relu as the activation function is used as a classifier. The work is motivated by the fact that the bias part is a piecewise constant function with zero gradient and hence cannot be directly attacked by gradient-based methods to generate adversaries such as FGSM. The existence of the bias classifier is proved an effective training method for the bias classifier is proposed. It is proved that by adding a proper random first-degree part to the bias classifier, an information-theoretically safe classifier against the original-model gradient-based attack is obtained in the sense that the attack generates a totally random direction for generating adversaries. This seems to be the first time that the concept of information-theoretically safe classifier is proposed. Several attack methods for the bias classifier are proposed and numerical experiments are used to show that the bias classifier is more robust than DNNs against these attacks in most cases.

摘要: 本文介绍了偏向分类器，即以RELU为激活函数的DNN的偏向部分作为分类器。该工作的动机在于偏差部分是零梯度的分段常数函数，因此不能被基于梯度的方法直接攻击来生成诸如FGSM之类的对手。证明了偏向分类器的存在性，提出了一种有效的偏向分类器训练方法。证明了通过在偏向分类器中增加适当的随机一阶部分，在攻击产生一个完全随机的攻击方向的意义下，得到了一个针对原始模型梯度攻击的信息论安全的分类器。这似乎是首次提出信息理论安全分类器的概念。提出了几种针对偏向分类器的攻击方法，并通过数值实验表明，在大多数情况下，偏向分类器比DNNs对这些攻击具有更强的鲁棒性。



## **27. Get a Model! Model Hijacking Attack Against Machine Learning Models**

找个模特来！针对机器学习模型的模型劫持攻击 cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04394v1)

**Authors**: Ahmed Salem, Michael Backes, Yang Zhang

**Abstracts**: Machine learning (ML) has established itself as a cornerstone for various critical applications ranging from autonomous driving to authentication systems. However, with this increasing adoption rate of machine learning models, multiple attacks have emerged. One class of such attacks is training time attack, whereby an adversary executes their attack before or during the machine learning model training. In this work, we propose a new training time attack against computer vision based machine learning models, namely model hijacking attack. The adversary aims to hijack a target model to execute a different task than its original one without the model owner noticing. Model hijacking can cause accountability and security risks since a hijacked model owner can be framed for having their model offering illegal or unethical services. Model hijacking attacks are launched in the same way as existing data poisoning attacks. However, one requirement of the model hijacking attack is to be stealthy, i.e., the data samples used to hijack the target model should look similar to the model's original training dataset. To this end, we propose two different model hijacking attacks, namely Chameleon and Adverse Chameleon, based on a novel encoder-decoder style ML model, namely the Camouflager. Our evaluation shows that both of our model hijacking attacks achieve a high attack success rate, with a negligible drop in model utility.

摘要: 机器学习(ML)已经成为从自动驾驶到身份验证系统等各种关键应用的基石。然而，随着机器学习模型采用率的不断提高，出现了多种攻击。这种攻击的一类是训练时间攻击，由此对手在机器学习模型训练之前或期间执行他们的攻击。在这项工作中，我们提出了一种新的针对基于计算机视觉的机器学习模型的训练时间攻击，即模型劫持攻击。敌手的目标是劫持目标模型，以便在模型所有者不察觉的情况下执行与其原始任务不同的任务。劫持模特可能会导致责任和安全风险，因为被劫持的模特所有者可能会因为让他们的模特提供非法或不道德的服务而被陷害。模型劫持攻击的发起方式与现有的数据中毒攻击方式相同。然而，模型劫持攻击的一个要求是隐蔽性，即用于劫持目标模型的数据样本应该与模型的原始训练数据集相似。为此，我们基于一种新的编解码器风格的ML模型，即伪装器，提出了两种不同模型的劫持攻击，即变色龙攻击和逆变色龙攻击。我们的评估表明，我们的两种模型劫持攻击都达到了很高的攻击成功率，而模型效用的下降可以忽略不计。



## **28. Geometrically Adaptive Dictionary Attack on Face Recognition**

人脸识别中的几何自适应字典攻击 cs.CV

Accepted at WACV 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04371v1)

**Authors**: Junyoung Byun, Hyojun Go, Changick Kim

**Abstracts**: CNN-based face recognition models have brought remarkable performance improvement, but they are vulnerable to adversarial perturbations. Recent studies have shown that adversaries can fool the models even if they can only access the models' hard-label output. However, since many queries are needed to find imperceptible adversarial noise, reducing the number of queries is crucial for these attacks. In this paper, we point out two limitations of existing decision-based black-box attacks. We observe that they waste queries for background noise optimization, and they do not take advantage of adversarial perturbations generated for other images. We exploit 3D face alignment to overcome these limitations and propose a general strategy for query-efficient black-box attacks on face recognition named Geometrically Adaptive Dictionary Attack (GADA). Our core idea is to create an adversarial perturbation in the UV texture map and project it onto the face in the image. It greatly improves query efficiency by limiting the perturbation search space to the facial area and effectively recycling previous perturbations. We apply the GADA strategy to two existing attack methods and show overwhelming performance improvement in the experiments on the LFW and CPLFW datasets. Furthermore, we also present a novel attack strategy that can circumvent query similarity-based stateful detection that identifies the process of query-based black-box attacks.

摘要: 基于CNN的人脸识别模型带来了显著的性能提升，但它们容易受到对手的干扰。最近的研究表明，即使对手只能访问模型的硬标签输出，他们也可以愚弄模型。然而，由于需要大量的查询来发现不可察觉的对抗性噪声，因此减少查询的数量对这些攻击至关重要。在本文中，我们指出了现有基于决策的黑盒攻击的两个局限性。我们观察到，它们将查询浪费在背景噪声优化上，并且它们没有利用为其他图像生成的对抗性扰动。我们利用三维人脸对齐来克服这些限制，并提出了一种通用的人脸识别黑盒攻击策略，称为几何自适应字典攻击(GADA)。我们的核心想法是在UV纹理贴图中创建对抗性扰动，并将其投影到图像中的脸部。通过将扰动搜索空间限制在人脸区域，并有效地循环使用先前的扰动，极大地提高了查询效率。我们将GADA策略应用于现有的两种攻击方法，在LFW和CPLFW数据集上的实验表明，GADA策略的性能有了显著的提高。此外，我们还提出了一种新的攻击策略，可以规避基于查询相似度的状态检测，识别基于查询的黑盒攻击过程。



## **29. Robustness of Graph Neural Networks at Scale**

图神经网络在尺度上的鲁棒性 cs.LG

39 pages, 22 figures, 17 tables NeurIPS 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2110.14038v3)

**Authors**: Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann

**Abstracts**: Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.

摘要: 图神经网络(GNNs)因其普及性和应用的多样性而变得越来越重要。然而，现有的关于它们易受敌意攻击的研究依赖于相对较小的图表。我们解决了这一差距，并研究了如何大规模攻击和防御GNN。我们提出了两种稀疏性感知的一阶优化攻击，这两种攻击在对节点数目为二次的多个参数进行优化的情况下仍能保持有效的表示。我们证明了常见的代理损失并不能很好地适用于针对GNNs的全局攻击。我们的替代方案可以使攻击强度加倍。此外，为了提高GNNs的可靠性，我们设计了一个健壮的聚集函数--软中值，从而在所有尺度上都能进行有效的防御。与以前的工作相比，我们在大于100倍的图上使用标准GNN来评估我们的攻击和防御。我们甚至通过将我们的技术扩展到可伸缩的GNN来进一步扩展一个数量级。



## **30. Characterizing the adversarial vulnerability of speech self-supervised learning**

语音自监督学习的对抗性脆弱性表征 cs.SD

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04330v1)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.

摘要: 一个名为语音处理通用性能基准(SUBB)的排行榜推动了语音表示学习的研究，该基准测试旨在以最小的体系结构和少量的数据对共享的自监督学习(SSL)语音模型在各种下游语音任务中的性能进行基准测试。出色的演示了语音SSL上行模型通过最小程度的适配提高了各种下行任务的性能。随着上游自我监督学习模型和下游任务的范式越来越受到语言学界的关注，表征这种范式的对抗性鲁棒性是当务之急。在本文中，我们首次尝试研究了该范式在零知识和有限知识两种攻击下的攻击脆弱性。实验结果表明，Superb提出的范式对有限知识的攻击具有很强的脆弱性，零知识攻击产生的攻击具有可移植性。Xab测试验证精心设计的敌意攻击的隐蔽性。



## **31. Graph Robustness Benchmark: Benchmarking the Adversarial Robustness of Graph Machine Learning**

图健壮性基准：对图机器学习的对抗性健壮性进行基准测试 cs.LG

21 pages, 12 figures, NeurIPS 2021 Datasets and Benchmarks Track

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04314v1)

**Authors**: Qinkai Zheng, Xu Zou, Yuxiao Dong, Yukuo Cen, Da Yin, Jiarong Xu, Yang Yang, Jie Tang

**Abstracts**: Adversarial attacks on graphs have posed a major threat to the robustness of graph machine learning (GML) models. Naturally, there is an ever-escalating arms race between attackers and defenders. However, the strategies behind both sides are often not fairly compared under the same and realistic conditions. To bridge this gap, we present the Graph Robustness Benchmark (GRB) with the goal of providing a scalable, unified, modular, and reproducible evaluation for the adversarial robustness of GML models. GRB standardizes the process of attacks and defenses by 1) developing scalable and diverse datasets, 2) modularizing the attack and defense implementations, and 3) unifying the evaluation protocol in refined scenarios. By leveraging the GRB pipeline, the end-users can focus on the development of robust GML models with automated data processing and experimental evaluations. To support open and reproducible research on graph adversarial learning, GRB also hosts public leaderboards across different scenarios. As a starting point, we conduct extensive experiments to benchmark baseline techniques. GRB is open-source and welcomes contributions from the community. Datasets, codes, leaderboards are available at https://cogdl.ai/grb/home.

摘要: 图的敌意攻击已经成为图机器学习(GML)模型健壮性的主要威胁。当然，攻击者和防御者之间的军备竞赛不断升级。然而，在相同的现实条件下，双方背后的战略往往是不公平的比较。为了弥补这一差距，我们提出了图健壮性基准(GRB)，目的是为GML模型的对抗健壮性提供一个可扩展的、统一的、模块化的和可重现的评估。GRB通过1)开发可扩展和多样化的数据集，2)将攻击和防御实现模块化，3)在细化的场景中统一评估协议，从而标准化了攻击和防御的过程。通过利用GRB管道，最终用户可以专注于开发具有自动数据处理和实验评估功能的健壮GML模型。为了支持关于图形对抗性学习的开放和可重复的研究，GRB还在不同的场景中主持公共排行榜。作为起点，我们进行了广泛的实验来对基线技术进行基准测试。GRB是开源的，欢迎来自社区的贡献。有关数据集、代码和排行榜的信息，请访问https://cogdl.ai/grb/home.



## **32. DeepMoM: Robust Deep Learning With Median-of-Means**

DeepMoM：基于均值中值的稳健深度学习 stat.ML

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2105.14035v2)

**Authors**: Shih-Ting Huang, Johannes Lederer

**Abstracts**: Data used in deep learning is notoriously problematic. For example, data are usually combined from diverse sources, rarely cleaned and vetted thoroughly, and sometimes corrupted on purpose. Intentional corruption that targets the weak spots of algorithms has been studied extensively under the label of "adversarial attacks." In contrast, the arguably much more common case of corruption that reflects the limited quality of data has been studied much less. Such "random" corruptions are due to measurement errors, unreliable sources, convenience sampling, and so forth. These kinds of corruption are common in deep learning, because data are rarely collected according to strict protocols -- in strong contrast to the formalized data collection in some parts of classical statistics. This paper concerns such corruption. We introduce an approach motivated by very recent insights into median-of-means and Le Cam's principle, we show that the approach can be readily implemented, and we demonstrate that it performs very well in practice. In conclusion, we believe that our approach is a very promising alternative to standard parameter training based on least-squares and cross-entropy loss.

摘要: 深度学习中使用的数据是出了名的问题。例如，数据通常来自不同的来源，很少被彻底清理和审查，有时还会被故意破坏。针对算法弱点的故意腐败已经在“对抗性攻击”的标签下进行了广泛的研究。相比之下，可以说更常见的反映数据质量有限的腐败案件的研究要少得多。这种“随机”损坏是由于测量误差、来源不可靠、采样方便等原因造成的。这种类型的损坏在深度学习中很常见，因为数据很少根据严格的协议收集--这与经典统计中某些部分的形式化数据收集形成了强烈对比。本文关注的是这样的腐败现象。我们介绍了一种基于对均值中位数和Le Cam原理的最新见解的方法，我们证明了该方法可以很容易地实现，并且我们证明了它在实践中表现得非常好。总之，我们认为我们的方法是一种非常有前途的替代基于最小二乘和交叉熵损失的标准参数训练的方法。



## **33. On the Effectiveness of Small Input Noise for Defending Against Query-based Black-Box Attacks**

小输入噪声对抵抗基于查询的黑盒攻击的有效性研究 cs.CR

Accepted at WACV 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2101.04829v2)

**Authors**: Junyoung Byun, Hyojun Go, Changick Kim

**Abstracts**: While deep neural networks show unprecedented performance in various tasks, the vulnerability to adversarial examples hinders their deployment in safety-critical systems. Many studies have shown that attacks are also possible even in a black-box setting where an adversary cannot access the target model's internal information. Most black-box attacks are based on queries, each of which obtains the target model's output for an input, and many recent studies focus on reducing the number of required queries. In this paper, we pay attention to an implicit assumption of query-based black-box adversarial attacks that the target model's output exactly corresponds to the query input. If some randomness is introduced into the model, it can break the assumption, and thus, query-based attacks may have tremendous difficulty in both gradient estimation and local search, which are the core of their attack process. From this motivation, we observe even a small additive input noise can neutralize most query-based attacks and name this simple yet effective approach Small Noise Defense (SND). We analyze how SND can defend against query-based black-box attacks and demonstrate its effectiveness against eight state-of-the-art attacks with CIFAR-10 and ImageNet datasets. Even with strong defense ability, SND almost maintains the original classification accuracy and computational speed. SND is readily applicable to pre-trained models by adding only one line of code at the inference.

摘要: 虽然深度神经网络在各种任务中表现出前所未有的性能，但对敌意示例的脆弱性阻碍了它们在安全关键系统中的部署。许多研究表明，即使在对手无法访问目标模型内部信息的黑盒设置中，攻击也是可能的。大多数黑盒攻击都是基于查询的，每个查询都会获取目标模型的输出作为输入，最近的许多研究都集中在减少所需查询的数量上。在本文中，我们注意到基于查询的黑盒对抗攻击的一个隐含假设，即目标模型的输出与查询输入精确对应。如果在模型中引入一定的随机性，可能会打破假设，因此，基于查询的攻击在梯度估计和局部搜索这两个攻击过程的核心上都可能会有很大的困难。从这一动机出发，我们观察到即使是很小的加性输入噪声也可以中和大多数基于查询的攻击，并将这种简单而有效的方法命名为小噪声防御(SND)。我们分析了SND如何防御基于查询的黑盒攻击，并使用CIFAR-10和ImageNet数据集验证了它对八种最先进的攻击的有效性。即使具有很强的防御能力，SND也几乎保持了原有的分类精度和计算速度。通过在推理处只添加一行代码，SND很容易适用于预先训练的模型。



## **34. Defense Against Explanation Manipulation**

对解释操纵的防御 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04303v1)

**Authors**: Ruixiang Tang, Ninghao Liu, Fan Yang, Na Zou, Xia Hu

**Abstracts**: Explainable machine learning attracts increasing attention as it improves transparency of models, which is helpful for machine learning to be trusted in real applications. However, explanation methods have recently been demonstrated to be vulnerable to manipulation, where we can easily change a model's explanation while keeping its prediction constant. To tackle this problem, some efforts have been paid to use more stable explanation methods or to change model configurations. In this work, we tackle the problem from the training perspective, and propose a new training scheme called Adversarial Training on EXplanations (ATEX) to improve the internal explanation stability of a model regardless of the specific explanation method being applied. Instead of directly specifying explanation values over data instances, ATEX only puts requirement on model predictions which avoids involving second-order derivatives in optimization. As a further discussion, we also find that explanation stability is closely related to another property of the model, i.e., the risk of being exposed to adversarial attack. Through experiments, besides showing that ATEX improves model robustness against manipulation targeting explanation, it also brings additional benefits including smoothing explanations and improving the efficacy of adversarial training if applied to the model.

摘要: 可解释机器学习由于提高了模型的透明性而受到越来越多的关注，这有助于机器学习在实际应用中得到信任。然而，最近已经证明解释方法容易受到操纵，在这些方法中，我们可以很容易地改变模型的解释，同时保持其预测不变。为了解决撞击的这一问题，已经做出了一些努力，使用更稳定的解释方法或改变模型配置。在这项工作中，我们从训练的角度对这一问题进行了撞击研究，并提出了一种新的训练方案，称为对抗性解释训练(ATEX)，以提高模型的内部解释稳定性，而不考虑具体的解释方法。ATEX没有直接指定数据实例上的解释值，而是只对模型预测提出了要求，避免了优化中涉及二阶导数的问题。作为进一步的讨论，我们还发现解释稳定性与模型的另一个性质，即暴露于敌意攻击的风险密切相关。通过实验表明，ATEX除了提高了模型对操作目标解释的鲁棒性外，如果将其应用到模型中，还可以带来平滑解释和提高对抗性训练效果等额外的好处。



## **35. A Unified Game-Theoretic Interpretation of Adversarial Robustness**

对抗性稳健性的统一博弈论解释 cs.LG

the previous version is arXiv:2103.07364, but I mistakenly apply a  new ID for the paper

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.03536v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, \emph{i.e.} the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.

摘要: 本文提供了一个统一的视角来解释不同的对抗性攻击和防御方法，即DNNs的输入变量之间的多阶交互的观点。基于多阶交互，我们发现对抗性攻击主要影响高阶交互来欺骗DNN。此外，我们发现对抗性训练的DNN的鲁棒性来自于特定类别的低阶交互。我们的发现为统一对抗性扰动和鲁棒性提供了一种潜在的方法，可以对现有的防御方法进行原则性的解释。此外，我们的发现还修正了以往对对抗性习得特征的形状偏向的不准确理解。



## **36. Generative Dynamic Patch Attack**

生成式动态补丁攻击 cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04266v1)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.

摘要: 对抗性补丁攻击是一系列攻击算法，通过扰动图像的一部分来欺骗深层神经网络模型。现有的补丁攻击大多考虑在与输入无关的位置(预定义位置或随机位置)注入敌意补丁。这种攻击设置对于攻击来说可能是足够的，但在用于对抗性训练时有相当大的限制。因此，用现有补丁攻击训练的鲁棒模型不能有效防御其他对抗性攻击。本文首先提出了一种端到端的补丁攻击算法--生成性动态补丁攻击(GDPA)，该算法对每幅输入图像分别生成补丁模式和补丁位置。我们证明了GDPA是一个通用的攻击框架，只需少量的配置更改，就可以生成动态/静电和可见/不可见的补丁。其次，GDPA可以很容易地集成到对抗性训练中，以提高模型对各种对抗性攻击的鲁棒性。在VGGFace、交通标志和ImageNet上的大量实验表明，GDPA比最新的补丁攻击具有更高的攻击成功率，而带有GDPA的对抗性训练模型对敌意补丁攻击表现出比竞争方法更好的鲁棒性。我们的源代码可以在https://github.com/lxuniverse/gdpa.上找到



## **37. Natural Adversarial Objects**

自然对抗性客体 cs.CV

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2111.04204v1)

**Authors**: Felix Lau, Nishant Subramani, Sasha Harrison, Aerin Kim, Elliot Branson, Rosanne Liu

**Abstracts**: Although state-of-the-art object detection methods have shown compelling performance, models often are not robust to adversarial attacks and out-of-distribution data. We introduce a new dataset, Natural Adversarial Objects (NAO), to evaluate the robustness of object detection models. NAO contains 7,934 images and 9,943 objects that are unmodified and representative of real-world scenarios, but cause state-of-the-art detection models to misclassify with high confidence. The mean average precision (mAP) of EfficientDet-D7 drops 74.5% when evaluated on NAO compared to the standard MSCOCO validation set.   Moreover, by comparing a variety of object detection architectures, we find that better performance on MSCOCO validation set does not necessarily translate to better performance on NAO, suggesting that robustness cannot be simply achieved by training a more accurate model.   We further investigate why examples in NAO are difficult to detect and classify. Experiments of shuffling image patches reveal that models are overly sensitive to local texture. Additionally, using integrated gradients and background replacement, we find that the detection model is reliant on pixel information within the bounding box, and insensitive to the background context when predicting class labels. NAO can be downloaded at https://drive.google.com/drive/folders/15P8sOWoJku6SSEiHLEts86ORfytGezi8.

摘要: 虽然最先进的目标检测方法已经显示出令人信服的性能，但模型通常对敌意攻击和分布外的数据并不健壮。我们引入了一个新的数据集--自然对抗性对象(NAO)来评估目标检测模型的健壮性。NAO包含7934张图像和9943个对象，这些图像和对象未经修改，可以代表真实世界的场景，但会导致最先进的检测模型高度可信地错误分类。与标准MSCOCO验证集相比，在NAO上评估EfficientDet-D7的平均平均精度(MAP)下降了74.5%。此外，通过比较各种目标检测体系结构，我们发现在MSCOCO验证集上更好的性能并不一定转化为在NAO上更好的性能，这表明鲁棒性不能简单地通过训练更精确的模型来实现。我们进一步调查了为什么NAO中的例子很难检测和分类。混洗图像块的实验表明，模型对局部纹理过于敏感。此外，通过使用集成梯度和背景替换，我们发现该检测模型依赖于边界框内的像素信息，并且在预测类别标签时对背景上下文不敏感。NAO可从https://drive.google.com/drive/folders/15P8sOWoJku6SSEiHLEts86ORfytGezi8.下载



## **38. Adversarial Attacks on Multi-task Visual Perception for Autonomous Driving**

自主驾驶多任务视觉感知的对抗性攻击 cs.CV

Accepted for publication at Journal of Imaging Science and Technology

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2107.07449v2)

**Authors**: Ibrahim Sobh, Ahmed Hamed, Varun Ravi Kumar, Senthil Yogamani

**Abstracts**: Deep neural networks (DNNs) have accomplished impressive success in various applications, including autonomous driving perception tasks, in recent years. On the other hand, current deep neural networks are easily fooled by adversarial attacks. This vulnerability raises significant concerns, particularly in safety-critical applications. As a result, research into attacking and defending DNNs has gained much coverage. In this work, detailed adversarial attacks are applied on a diverse multi-task visual perception deep network across distance estimation, semantic segmentation, motion detection, and object detection. The experiments consider both white and black box attacks for targeted and un-targeted cases, while attacking a task and inspecting the effect on all the others, in addition to inspecting the effect of applying a simple defense method. We conclude this paper by comparing and discussing the experimental results, proposing insights and future work. The visualizations of the attacks are available at https://youtu.be/6AixN90budY.

摘要: 近年来，深度神经网络(DNNs)在包括自主驾驶感知任务在内的各种应用中取得了令人印象深刻的成功。另一方面，当前的深度神经网络很容易被敌意攻击所欺骗。此漏洞引起了严重关注，尤其是在安全关键型应用程序中。因此，攻击和防御DNN的研究得到了广泛的报道。在这项工作中，详细的对抗性攻击应用于一个跨越距离估计、语义分割、运动检测和目标检测的多样化的多任务视觉感知深度网络。实验同时考虑了针对目标和非目标情况的白盒攻击和黑盒攻击，同时攻击一个任务并检查对所有其他任务的影响，此外还检查了应用简单防御方法的效果。最后，通过对实验结果的比较和讨论，总结了本文的研究成果，并提出了自己的见解和未来的工作方向。这些攻击的可视化可在https://youtu.be/6AixN90budY.上查看。



## **39. On the Convergence of Prior-Guided Zeroth-Order Optimization Algorithms**

关于先验引导零阶优化算法的收敛性 stat.ML

NeurIPS 2021; code available at https://github.com/csy530216/pg-zoo

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2107.10110v2)

**Authors**: Shuyu Cheng, Guoqiang Wu, Jun Zhu

**Abstracts**: Zeroth-order (ZO) optimization is widely used to handle challenging tasks, such as query-based black-box adversarial attacks and reinforcement learning. Various attempts have been made to integrate prior information into the gradient estimation procedure based on finite differences, with promising empirical results. However, their convergence properties are not well understood. This paper makes an attempt to fill up this gap by analyzing the convergence of prior-guided ZO algorithms under a greedy descent framework with various gradient estimators. We provide a convergence guarantee for the prior-guided random gradient-free (PRGF) algorithms. Moreover, to further accelerate over greedy descent methods, we present a new accelerated random search (ARS) algorithm that incorporates prior information, together with a convergence analysis. Finally, our theoretical results are confirmed by experiments on several numerical benchmarks as well as adversarial attacks.

摘要: 零阶(ZO)优化被广泛用于处理具有挑战性的任务，如基于查询的黑盒对抗攻击和强化学习。为了将先验信息集成到基于有限差分的梯度估计过程中，已经进行了各种尝试，并取得了令人满意的经验结果。然而，人们对它们的收敛性还没有很好的理解。本文试图通过分析先验引导ZO算法在具有不同梯度估计器的贪婪下降框架下的收敛性来填补这一空白。我们给出了先验引导的随机无梯度(PRGF)算法的收敛性保证。此外，为了进一步加速贪婪下降算法，我们提出了一种结合先验信息的加速随机搜索(ARS)算法，并进行了收敛性分析。最后，通过在几个数值基准和对抗性攻击上的实验验证了我们的理论结果。



## **40. On the Robustness of Domain Constraints**

论域约束的健壮性 cs.CR

Accepted to the 28th ACM Conference on Computer and Communications  Security. Seoul, South Korea

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2105.08619v2)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Yohan Beugin, Michael J. Weisman, Patrick McDaniel

**Abstracts**: Machine learning is vulnerable to adversarial examples-inputs designed to cause models to perform poorly. However, it is unclear if adversarial examples represent realistic inputs in the modeled domains. Diverse domains such as networks and phishing have domain constraints-complex relationships between features that an adversary must satisfy for an attack to be realized (in addition to any adversary-specific goals). In this paper, we explore how domain constraints limit adversarial capabilities and how adversaries can adapt their strategies to create realistic (constraint-compliant) examples. In this, we develop techniques to learn domain constraints from data, and show how the learned constraints can be integrated into the adversarial crafting process. We evaluate the efficacy of our approach in network intrusion and phishing datasets and find: (1) up to 82% of adversarial examples produced by state-of-the-art crafting algorithms violate domain constraints, (2) domain constraints are robust to adversarial examples; enforcing constraints yields an increase in model accuracy by up to 34%. We observe not only that adversaries must alter inputs to satisfy domain constraints, but that these constraints make the generation of valid adversarial examples far more challenging.

摘要: 机器学习很容易受到敌意例子的影响-输入的目的是导致模型表现不佳。然而，目前还不清楚对抗性的例子是否代表了模型域中的现实输入。不同的域(如网络和网络钓鱼)都有域约束，即攻击者必须满足才能实现攻击的功能之间的复杂关系(以及任何特定于对手的目标)。在这篇文章中，我们探索领域约束如何限制对手的能力，以及对手如何调整他们的策略来创建现实的(符合约束的)示例。在这方面，我们开发了从数据中学习领域约束的技术，并展示了如何将学习到的约束集成到对抗性的制作过程中。我们评估了我们的方法在网络入侵和网络钓鱼数据集上的有效性，发现：(1)由最先进的手工算法产生的敌意示例高达82%违反了域约束，(2)域约束对敌意示例是健壮的；强制约束可以使模型准确率提高高达34%。我们观察到，不仅对手必须改变输入才能满足域约束，而且这些约束使得生成有效的对抗性示例更具挑战性。



## **41. Adversarial Robustness of Deep Code Comment Generation**

深层代码注释生成的对抗健壮性 cs.SE

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2108.00213v2)

**Authors**: Yu Zhou, Xiaoqing Zhang, Juanjuan Shen, Tingting Han, Taolue Chen, Harald Gall

**Abstracts**: Deep neural networks (DNNs) have shown remarkable performance in a variety of domains such as computer vision, speech recognition, or natural language processing. Recently they also have been applied to various software engineering tasks, typically involving processing source code. DNNs are well-known to be vulnerable to adversarial examples, i.e., fabricated inputs that could lead to various misbehaviors of the DNN model while being perceived as benign by humans. In this paper, we focus on the code comment generation task in software engineering and study the robustness issue of the DNNs when they are applied to this task. We propose ACCENT, an identifier substitution approach to craft adversarial code snippets, which are syntactically correct and semantically close to the original code snippet, but may mislead the DNNs to produce completely irrelevant code comments. In order to improve the robustness, ACCENT also incorporates a novel training method, which can be applied to existing code comment generation models. We conduct comprehensive experiments to evaluate our approach by attacking the mainstream encoder-decoder architectures on two large-scale publicly available datasets. The results show that ACCENT efficiently produces stable attacks with functionality-preserving adversarial examples, and the generated examples have better transferability compared with baselines. We also confirm, via experiments, the effectiveness in improving model robustness with our training method.

摘要: 深度神经网络(DNNs)在计算机视觉、语音识别、自然语言处理等领域表现出显著的性能。最近，它们还被应用于各种软件工程任务，通常涉及处理源代码。众所周知，DNN很容易受到敌意示例的攻击，即在人类认为DNN模型是良性的情况下，可能会导致DNN模型的各种错误行为的捏造输入。本文针对软件工程中的代码注释生成任务，研究了DNN应用于该任务时的健壮性问题。我们提出了一种标识符替换方法Accent来制作敌意代码片段，这些代码片段在语法上是正确的，在语义上也接近于原始代码片段，但可能会误导DNN生成完全不相关的代码注释。为了提高鲁棒性，Accent还引入了一种新的训练方法，该方法可以应用于现有的代码注释生成模型。我们在两个大规模公开可用的数据集上进行了全面的实验，通过攻击主流的编解码器架构来评估我们的方法。实验结果表明，重音算法能有效地产生稳定的攻击，且生成的实例与基线相比具有更好的可移植性。通过实验，我们也证实了我们的训练方法在提高模型鲁棒性方面的有效性。



## **42. Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks**

绘制健壮的暂存券：在随机初始化的网络中发现具有天生健壮性的子网 cs.LG

Accepted at NeurIPS 2021

**SubmitDate**: 2021-11-06    [paper-pdf](http://arxiv.org/pdf/2110.14068v2)

**Authors**: Yonggan Fu, Qixuan Yu, Yang Zhang, Shang Wu, Xu Ouyang, David Cox, Yingyan Lin

**Abstracts**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial attacks, i.e., an imperceptible perturbation to the input can mislead DNNs trained on clean images into making erroneous predictions. To tackle this, adversarial training is currently the most effective defense method, by augmenting the training set with adversarial samples generated on the fly. Interestingly, we discover for the first time that there exist subnetworks with inborn robustness, matching or surpassing the robust accuracy of the adversarially trained networks with comparable model sizes, within randomly initialized networks without any model training, indicating that adversarial training on model weights is not indispensable towards adversarial robustness. We name such subnetworks Robust Scratch Tickets (RSTs), which are also by nature efficient. Distinct from the popular lottery ticket hypothesis, neither the original dense networks nor the identified RSTs need to be trained. To validate and understand this fascinating finding, we further conduct extensive experiments to study the existence and properties of RSTs under different models, datasets, sparsity patterns, and attacks, drawing insights regarding the relationship between DNNs' robustness and their initialization/overparameterization. Furthermore, we identify the poor adversarial transferability between RSTs of different sparsity ratios drawn from the same randomly initialized dense network, and propose a Random RST Switch (R2S) technique, which randomly switches between different RSTs, as a novel defense method built on top of RSTs. We believe our findings about RSTs have opened up a new perspective to study model robustness and extend the lottery ticket hypothesis.

摘要: 深度神经网络(DNNs)很容易受到敌意攻击，即输入的不知不觉的扰动会误导训练在干净图像上的DNN做出错误的预测。对撞击来说，对抗性训练是目前最有效的防御方法，通过使用飞翔上生成的对抗性样本来扩大训练集。有趣的是，我们首次发现，在没有任何模型训练的随机初始化网络中，存在具有天生鲁棒性的子网络，其鲁棒性精度与具有相似模型大小的对抗性训练网络相当或超过，这表明对抗性模型权重的训练对于对抗性鲁棒性来说并不是必不可少的。我们将这样的子网命名为健壮的暂存票(RST)，它本质上也是有效的。与流行的彩票假设不同，原始的密集网络和识别出的RST都不需要训练。为了验证和理解这一有趣的发现，我们进一步进行了大量的实验，研究了不同模型、数据集、稀疏模式和攻击下RST的存在和性质，得出了DNNs的健壮性与其初始化/过参数化之间的关系。此外，我们还发现了来自同一随机初始化密集网络的不同稀疏比的RST之间的对抗性较差，并提出了一种在RST之上随机切换的随机RST切换(R2S)技术，作为一种新的防御方法。我们相信，我们关于RST的发现为研究模型的稳健性和扩展彩票假说开辟了一个新的视角。



## **43. TRS: Transferability Reduced Ensemble via Encouraging Gradient Diversity and Model Smoothness**

TRS：通过鼓励梯度多样性和模型光滑性来减少集成的可转移性 cs.LG

Proceedings of the 35th Conference on Neural Information Processing  Systems (NeurIPS 2021)

**SubmitDate**: 2021-11-06    [paper-pdf](http://arxiv.org/pdf/2104.00671v2)

**Authors**: Zhuolin Yang, Linyi Li, Xiaojun Xu, Shiliang Zuo, Qian Chen, Benjamin Rubinstein, Pan Zhou, Ce Zhang, Bo Li

**Abstracts**: Adversarial Transferability is an intriguing property - adversarial perturbation crafted against one model is also effective against another model, while these models are from different model families or training processes. To better protect ML systems against adversarial attacks, several questions are raised: what are the sufficient conditions for adversarial transferability and how to bound it? Is there a way to reduce the adversarial transferability in order to improve the robustness of an ensemble ML model? To answer these questions, in this work we first theoretically analyze and outline sufficient conditions for adversarial transferability between models; then propose a practical algorithm to reduce the transferability between base models within an ensemble to improve its robustness. Our theoretical analysis shows that only promoting the orthogonality between gradients of base models is not enough to ensure low transferability; in the meantime, the model smoothness is an important factor to control the transferability. We also provide the lower and upper bounds of adversarial transferability under certain conditions. Inspired by our theoretical analysis, we propose an effective Transferability Reduced Smooth(TRS) ensemble training strategy to train a robust ensemble with low transferability by enforcing both gradient orthogonality and model smoothness between base models. We conduct extensive experiments on TRS and compare with 6 state-of-the-art ensemble baselines against 8 whitebox attacks on different datasets, demonstrating that the proposed TRS outperforms all baselines significantly.

摘要: 对抗性可转移性是一个有趣的性质-针对一个模型精心设计的对抗性扰动对另一个模型也有效，而这些模型来自不同的模型家族或训练过程。为了更好地保护ML系统免受对抗性攻击，提出了几个问题：对抗性可转移性的充分条件是什么？如何对其进行约束？有没有办法降低对抗性转移，以提高集成ML模型的稳健性？为了回答这些问题，本文首先从理论上分析和概括了模型间对抗性转移的充分条件，然后提出了一种实用的算法来降低集成内基础模型之间的转移，以提高其鲁棒性。理论分析表明，仅提高基础模型梯度间的正交性并不足以保证较低的可转移性，同时，模型的光滑性是控制可转移性的重要因素。在一定条件下，我们还给出了对抗性转移的上下界。在理论分析的启发下，我们提出了一种有效的可转移性简化平滑(TRS)集成训练策略，通过加强基模型之间的梯度正交性和模型平滑性来训练具有低可转移性的鲁棒集成。我们在TRS上进行了大量的实验，并与6个最新的集成基线在不同数据集上的8个白盒攻击进行了比较，结果表明提出的TRS的性能明显优于所有的基线。



## **44. Reconstructing Training Data from Diverse ML Models by Ensemble Inversion**

基于集成反演的不同ML模型训练数据重构 cs.LG

9 pages, 8 figures, WACV 2022

**SubmitDate**: 2021-11-05    [paper-pdf](http://arxiv.org/pdf/2111.03702v1)

**Authors**: Qian Wang, Daniel Kurz

**Abstracts**: Model Inversion (MI), in which an adversary abuses access to a trained Machine Learning (ML) model attempting to infer sensitive information about its original training data, has attracted increasing research attention. During MI, the trained model under attack (MUA) is usually frozen and used to guide the training of a generator, such as a Generative Adversarial Network (GAN), to reconstruct the distribution of the original training data of that model. This might cause leakage of original training samples, and if successful, the privacy of dataset subjects will be at risk if the training data contains Personally Identifiable Information (PII). Therefore, an in-depth investigation of the potentials of MI techniques is crucial for the development of corresponding defense techniques. High-quality reconstruction of training data based on a single model is challenging. However, existing MI literature does not explore targeting multiple models jointly, which may provide additional information and diverse perspectives to the adversary.   We propose the ensemble inversion technique that estimates the distribution of original training data by training a generator constrained by an ensemble (or set) of trained models with shared subjects or entities. This technique leads to noticeable improvements of the quality of the generated samples with distinguishable features of the dataset entities compared to MI of a single ML model. We achieve high quality results without any dataset and show how utilizing an auxiliary dataset that's similar to the presumed training data improves the results. The impact of model diversity in the ensemble is thoroughly investigated and additional constraints are utilized to encourage sharp predictions and high activations for the reconstructed samples, leading to more accurate reconstruction of training images.

摘要: 模型反转(MI)是指敌手滥用对经过训练的机器学习(ML)模型的访问，试图推断关于其原始训练数据的敏感信息，已引起越来越多的研究关注。在MI过程中，训练的攻击下模型(MUA)通常被冻结，并用于指导生成器(如生成性对抗网络)的训练，以重构该模型的原始训练数据的分布。这可能会导致原始训练样本的泄漏，如果成功，如果训练数据包含个人身份信息(PII)，则数据集对象的隐私将面临风险。因此，深入研究MI技术的潜力对于发展相应的防御技术至关重要。基于单一模型的高质量训练数据重建具有挑战性。然而，现有的MI文献没有探索联合瞄准多个模型，这可能会为对手提供额外的信息和不同的视角。我们提出了集成反演技术，该技术通过训练一个生成器来估计原始训练数据的分布，该生成器受具有共享主题或实体的训练模型的集成(或集合)约束。与单个ML模型的MI相比，该技术导致具有数据集实体的可区分特征的生成样本的质量显著改善。我们在没有任何数据集的情况下实现了高质量的结果，并展示了如何利用与假定的训练数据相似的辅助数据集来改善结果。深入研究了集成中模型多样性的影响，并利用附加约束来鼓励对重建样本的精确预测和高激活，从而导致更准确的训练图像重建。



## **45. Visualizing the Emergence of Intermediate Visual Patterns in DNNs**

在DNNs中可视化中间视觉模式的出现 cs.CV

**SubmitDate**: 2021-11-05    [paper-pdf](http://arxiv.org/pdf/2111.03505v1)

**Authors**: Mingjie Li, Shaobo Wang, Quanshi Zhang

**Abstracts**: This paper proposes a method to visualize the discrimination power of intermediate-layer visual patterns encoded by a DNN. Specifically, we visualize (1) how the DNN gradually learns regional visual patterns in each intermediate layer during the training process, and (2) the effects of the DNN using non-discriminative patterns in low layers to construct disciminative patterns in middle/high layers through the forward propagation. Based on our visualization method, we can quantify knowledge points (i.e., the number of discriminative visual patterns) learned by the DNN to evaluate the representation capacity of the DNN. Furthermore, this method also provides new insights into signal-processing behaviors of existing deep-learning techniques, such as adversarial attacks and knowledge distillation.

摘要: 提出了一种将DNN编码的中间层视觉模式的识别力可视化的方法。具体地说，我们可视化了(1)DNN如何在训练过程中逐渐学习各中间层的区域视觉模式，以及(2)DNN在低层使用非区分模式通过前向传播构建中高层区分模式的效果。基于我们的可视化方法，我们可以量化DNN学习的知识点(即区分视觉模式的数量)来评估DNN的表示能力。此外，该方法还为现有深度学习技术(如对抗性攻击和知识提取)的信号处理行为提供了新的见解。



## **46. Adversarial Attacks on Knowledge Graph Embeddings via Instance Attribution Methods**

基于实例属性方法的知识图嵌入对抗性攻击 cs.LG

2021 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2021)

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.03120v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: Despite the widespread use of Knowledge Graph Embeddings (KGE), little is known about the security vulnerabilities that might disrupt their intended behaviour. We study data poisoning attacks against KGE models for link prediction. These attacks craft adversarial additions or deletions at training time to cause model failure at test time. To select adversarial deletions, we propose to use the model-agnostic instance attribution methods from Interpretable Machine Learning, which identify the training instances that are most influential to a neural model's predictions on test instances. We use these influential triples as adversarial deletions. We further propose a heuristic method to replace one of the two entities in each influential triple to generate adversarial additions. Our experiments show that the proposed strategies outperform the state-of-art data poisoning attacks on KGE models and improve the MRR degradation due to the attacks by up to 62% over the baselines.

摘要: 尽管KGE(Knowledge Graph Embedding，知识图嵌入)被广泛使用，但人们对可能破坏其预期行为的安全漏洞知之甚少。我们研究了针对链接预测的KGE模型的数据中毒攻击。这些攻击在训练时精心设计敌意的添加或删除，从而在测试时导致模型失败。为了选择对抗性删除，我们建议使用可解释机器学习中的与模型无关的实例属性方法，该方法识别对神经模型对测试实例的预测影响最大的训练实例。我们使用这些有影响力的三元组作为对抗性删除。我们进一步提出了一种启发式方法来替换每个有影响力的三元组中的两个实体中的一个，以生成对抗性加法。我们的实验表明，所提出的策略比现有的针对KGE模型的数据中毒攻击具有更好的性能，并且使由于攻击而导致的MRR降级在基线上提高了高达62%。



## **47. Scanflow: A multi-graph framework for Machine Learning workflow management, supervision, and debugging**

Scanflow：一个用于机器学习工作流管理、监督和调试的多图框架 cs.LG

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.03003v1)

**Authors**: Gusseppe Bravo-Rocca, Peini Liu, Jordi Guitart, Ajay Dholakia, David Ellison, Jeffrey Falkanger, Miroslav Hodak

**Abstracts**: Machine Learning (ML) is more than just training models, the whole workflow must be considered. Once deployed, a ML model needs to be watched and constantly supervised and debugged to guarantee its validity and robustness in unexpected situations. Debugging in ML aims to identify (and address) the model weaknesses in not trivial contexts. Several techniques have been proposed to identify different types of model weaknesses, such as bias in classification, model decay, adversarial attacks, etc., yet there is not a generic framework that allows them to work in a collaborative, modular, portable, iterative way and, more importantly, flexible enough to allow both human- and machine-driven techniques. In this paper, we propose a novel containerized directed graph framework to support and accelerate end-to-end ML workflow management, supervision, and debugging. The framework allows defining and deploying ML workflows in containers, tracking their metadata, checking their behavior in production, and improving the models by using both learned and human-provided knowledge. We demonstrate these capabilities by integrating in the framework two hybrid systems to detect data drift distribution which identify the samples that are far from the latent space of the original distribution, ask for human intervention, and whether retrain the model or wrap it with a filter to remove the noise of corrupted data at inference time. We test these systems on MNIST-C, CIFAR-10-C, and FashionMNIST-C datasets, obtaining promising accuracy results with the help of human involvement.

摘要: 机器学习(ML)不仅仅是训练模型，还必须考虑整个工作流程。一旦部署，就需要监视ML模型，并不断地对其进行监督和调试，以确保其在意外情况下的有效性和健壮性。ML中的调试旨在识别(并解决)在不平凡的上下文中的模型弱点。已经提出了几种技术来识别不同类型的模型弱点，例如分类偏差、模型衰减、对抗性攻击等，但是还没有一个通用的框架允许它们以协作、模块化、可移植、迭代的方式工作，更重要的是，足够灵活地允许人和机器驱动的技术。本文提出了一种新的容器化有向图框架来支持和加速端到端ML工作流管理、监督和调试。该框架允许在容器中定义和部署ML工作流，跟踪它们的元数据，检查它们在生产中的行为，并通过使用学习到的知识和人工提供的知识来改进模型。我们通过在框架中集成两个混合系统来检测数据漂移分布来展示这些能力，这两个系统识别远离原始分布潜在空间的样本，要求人工干预，以及是重新训练模型还是用过滤包裹模型，以在推理时消除损坏数据的噪声。我们在MNIST-C、CIFAR-10-C和FashionMNIST-C数据集上测试了这些系统，在人工参与的帮助下获得了令人满意的准确性结果。



## **48. Attacking Deep Reinforcement Learning-Based Traffic Signal Control Systems with Colluding Vehicles**

用合谋车辆攻击基于深度强化学习的交通信号控制系统 cs.LG

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.02845v1)

**Authors**: Ao Qu, Yihong Tang, Wei Ma

**Abstracts**: The rapid advancements of Internet of Things (IoT) and artificial intelligence (AI) have catalyzed the development of adaptive traffic signal control systems (ATCS) for smart cities. In particular, deep reinforcement learning (DRL) methods produce the state-of-the-art performance and have great potentials for practical applications. In the existing DRL-based ATCS, the controlled signals collect traffic state information from nearby vehicles, and then optimal actions (e.g., switching phases) can be determined based on the collected information. The DRL models fully "trust" that vehicles are sending the true information to the signals, making the ATCS vulnerable to adversarial attacks with falsified information. In view of this, this paper first time formulates a novel task in which a group of vehicles can cooperatively send falsified information to "cheat" DRL-based ATCS in order to save their total travel time. To solve the proposed task, we develop CollusionVeh, a generic and effective vehicle-colluding framework composed of a road situation encoder, a vehicle interpreter, and a communication mechanism. We employ our method to attack established DRL-based ATCS and demonstrate that the total travel time for the colluding vehicles can be significantly reduced with a reasonable number of learning episodes, and the colluding effect will decrease if the number of colluding vehicles increases. Additionally, insights and suggestions for the real-world deployment of DRL-based ATCS are provided. The research outcomes could help improve the reliability and robustness of the ATCS and better protect the smart mobility systems.

摘要: 物联网(IoT)和人工智能(AI)的快速发展促进了智能城市自适应交通信号控制系统(ATCS)的发展。尤其是深度强化学习(DRL)方法具有最先进的性能和巨大的实际应用潜力。在现有的基于DRL的ATCS中，受控信号收集附近车辆的交通状态信息，然后可以基于收集的信息来确定最优动作(例如，切换相位)。DRL模型完全“信任”车辆正在向信号发送真实的信息，使得ATCS容易受到带有伪造信息的敌意攻击。有鉴于此，本文首次提出了一种新颖的任务，即一组车辆可以协同发送伪造信息来“欺骗”基于DRL的ATC，以节省它们的总行程时间。为了解决这一问题，我们开发了CollusionVeh，这是一个通用的、有效的车辆共谋框架，由路况编码器、车辆解释器和通信机制组成。我们利用我们的方法对已建立的基于DRL的ATCS进行攻击，并证明了在合理的学习场景数下，合谋车辆的总行驶时间可以显著减少，并且合谋效应随着合谋车辆数量的增加而降低。此外，还为基于DRL的ATCS的实际部署提供了见解和建议。研究成果有助于提高ATCS的可靠性和鲁棒性，更好地保护智能移动系统。



## **49. Adversarial Attacks on Graph Classification via Bayesian Optimisation**

基于贝叶斯优化的图分类对抗性攻击 stat.ML

NeurIPS 2021. 11 pages, 8 figures, 2 tables (24 pages, 17 figures, 8  tables including references and appendices)

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.02842v1)

**Authors**: Xingchen Wan, Henry Kenlay, Binxin Ru, Arno Blaas, Michael A. Osborne, Xiaowen Dong

**Abstracts**: Graph neural networks, a popular class of models effective in a wide range of graph-based learning tasks, have been shown to be vulnerable to adversarial attacks. While the majority of the literature focuses on such vulnerability in node-level classification tasks, little effort has been dedicated to analysing adversarial attacks on graph-level classification, an important problem with numerous real-life applications such as biochemistry and social network analysis. The few existing methods often require unrealistic setups, such as access to internal information of the victim models, or an impractically-large number of queries. We present a novel Bayesian optimisation-based attack method for graph classification models. Our method is black-box, query-efficient and parsimonious with respect to the perturbation applied. We empirically validate the effectiveness and flexibility of the proposed method on a wide range of graph classification tasks involving varying graph properties, constraints and modes of attack. Finally, we analyse common interpretable patterns behind the adversarial samples produced, which may shed further light on the adversarial robustness of graph classification models.

摘要: 图神经网络是一类在广泛的基于图的学习任务中有效的流行模型，已被证明容易受到敌意攻击。虽然大多数文献集中在节点级分类任务中的此类漏洞，但很少有人致力于分析对图级分类的敌意攻击，这是许多现实应用(如生物化学和社会网络分析)中的一个重要问题。现有的少数方法通常需要不切实际的设置，例如访问受害者模型的内部信息，或者不切实际地进行大量查询。提出了一种新的基于贝叶斯优化的图分类模型攻击方法。我们的方法是黑箱的，查询效率高，并且相对于所应用的扰动是简约的。我们通过实验验证了该方法在涉及不同的图属性、约束和攻击模式的广泛的图分类任务上的有效性和灵活性。最后，我们分析了产生的对抗性样本背后常见的可解释模式，这可能进一步揭示图分类模型的对抗性鲁棒性。



## **50. Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models**

对抗性胶水：语言模型健壮性评估的多任务基准 cs.CL

Oral Presentation in NeurIPS 2021 (Datasets and Benchmarks Track). 24  pages, 4 figures, 12 tables

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.02840v1)

**Authors**: Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, Bo Li

**Abstracts**: Large-scale pre-trained language models have achieved tremendous success across a wide range of natural language understanding (NLU) tasks, even surpassing human performance. However, recent studies reveal that the robustness of these models can be challenged by carefully crafted textual adversarial examples. While several individual datasets have been proposed to evaluate model robustness, a principled and comprehensive benchmark is still missing. In this paper, we present Adversarial GLUE (AdvGLUE), a new multi-task benchmark to quantitatively and thoroughly explore and evaluate the vulnerabilities of modern large-scale language models under various types of adversarial attacks. In particular, we systematically apply 14 textual adversarial attack methods to GLUE tasks to construct AdvGLUE, which is further validated by humans for reliable annotations. Our findings are summarized as follows. (i) Most existing adversarial attack algorithms are prone to generating invalid or ambiguous adversarial examples, with around 90% of them either changing the original semantic meanings or misleading human annotators as well. Therefore, we perform a careful filtering process to curate a high-quality benchmark. (ii) All the language models and robust training methods we tested perform poorly on AdvGLUE, with scores lagging far behind the benign accuracy. We hope our work will motivate the development of new adversarial attacks that are more stealthy and semantic-preserving, as well as new robust language models against sophisticated adversarial attacks. AdvGLUE is available at https://adversarialglue.github.io.

摘要: 大规模的预训练语言模型在广泛的自然语言理解(NLU)任务中取得了巨大的成功，甚至超过了人类的表现。然而，最近的研究表明，这些模型的稳健性可能会受到精心设计的文本对抗性例子的挑战。虽然已经提出了几个单独的数据集来评估模型的稳健性，但仍然缺乏一个原则性和综合性的基准。本文提出了一种新的多任务基准--对抗性粘合剂(AdvGLUE)，用以定量、深入地研究和评估现代大规模语言模型在各种类型的对抗性攻击下的脆弱性。特别是，我们系统地应用了14种文本对抗性攻击方法来粘合任务来构建AdvGLUE，并进一步验证了该方法的可靠性。我们的发现总结如下。(I)现有的对抗性攻击算法大多容易产生无效或歧义的对抗性示例，其中90%左右的算法要么改变了原有的语义，要么误导了人类的注释者。因此，我们执行仔细的筛选过程来策划一个高质量的基准。(Ii)我们测试的所有语言模型和稳健训练方法在AdvGLUE上的表现都很差，分数远远落后于良性准确率。我们希望我们的工作将促进更隐蔽性和语义保持的新的对抗性攻击的发展，以及针对复杂的对抗性攻击的新的健壮语言模型的开发。有关AdvGLUE的信息，请访问https://adversarialglue.github.io.。



