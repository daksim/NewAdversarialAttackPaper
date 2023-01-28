# Latest Adversarial Attack Papers
**update at 2023-01-28 10:40:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Certified Interpretability Robustness for Class Activation Mapping**

类激活映射的证明可解释性鲁棒性 cs.LG

13 pages, 5 figures. Accepted to Machine Learning for Autonomous  Driving Workshop at NeurIPS 2020

**SubmitDate**: 2023-01-26    [abs](http://arxiv.org/abs/2301.11324v1) [paper-pdf](http://arxiv.org/pdf/2301.11324v1)

**Authors**: Alex Gu, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, Luca Daniel

**Abstract**: Interpreting machine learning models is challenging but crucial for ensuring the safety of deep networks in autonomous driving systems. Due to the prevalence of deep learning based perception models in autonomous vehicles, accurately interpreting their predictions is crucial. While a variety of such methods have been proposed, most are shown to lack robustness. Yet, little has been done to provide certificates for interpretability robustness. Taking a step in this direction, we present CORGI, short for Certifiably prOvable Robustness Guarantees for Interpretability mapping. CORGI is an algorithm that takes in an input image and gives a certifiable lower bound for the robustness of the top k pixels of its CAM interpretability map. We show the effectiveness of CORGI via a case study on traffic sign data, certifying lower bounds on the minimum adversarial perturbation not far from (4-5x) state-of-the-art attack methods.

摘要: 解释机器学习模型是具有挑战性的，但对于确保自动驾驶系统中深层网络的安全至关重要。由于基于深度学习的感知模型在自动驾驶汽车中的普遍应用，准确解释它们的预测是至关重要的。虽然已经提出了各种这样的方法，但大多数都被证明缺乏稳健性。然而，在提供可解释性健壮性证书方面所做的工作很少。朝着这个方向迈出了一步，我们提出了CORGI，即可证明可证明的可解释映射的健壮性保证。CORGI是一种算法，它接收输入图像，并为其CAM可解释图的前k个像素的稳健性给出可证明的下界。通过对交通标志数据的案例研究，我们展示了COGI的有效性，证明了最小对抗性扰动的下界与最先进的攻击方法(4-5x)不远。



## **2. Hybrid Protection of Digital FIR Filters**

数字FIR滤波器的混合保护 cs.CR

**SubmitDate**: 2023-01-26    [abs](http://arxiv.org/abs/2301.11115v1) [paper-pdf](http://arxiv.org/pdf/2301.11115v1)

**Authors**: Levent Aksoy, Quang-Linh Nguyen, Felipe Almeida, Jaan Raik, Marie-Lise Flottes, Sophie Dupuis, Samuel Pagliarini

**Abstract**: A digital Finite Impulse Response (FIR) filter is a ubiquitous block in digital signal processing applications and its behavior is determined by its coefficients. To protect filter coefficients from an adversary, efficient obfuscation techniques have been proposed, either by hiding them behind decoys or replacing them by key bits. In this article, we initially introduce a query attack that can discover the secret key of such obfuscated FIR filters, which could not be broken by existing prominent attacks. Then, we propose a first of its kind hybrid technique, including both hardware obfuscation and logic locking using a point function for the protection of parallel direct and transposed forms of digital FIR filters. Experimental results show that the hybrid protection technique can lead to FIR filters with higher security while maintaining the hardware complexity competitive or superior to those locked by prominent logic locking methods. It is also shown that the protected multiplier blocks and FIR filters are resilient to existing attacks. The results on different forms and realizations of FIR filters show that the parallel direct form FIR filter has a promising potential for a secure design.

摘要: 数字有限脉冲响应(FIR)滤波器是数字信号处理应用中普遍存在的一个模块，其性能由其系数决定。为了保护滤波系数不被攻击者攻击，已经提出了有效的混淆技术，要么将它们隐藏在诱饵后面，要么用密钥位代替它们。在本文中，我们首先介绍了一种查询攻击，它可以发现这种模糊FIR滤波器的密钥，而这些密钥是现有的显著攻击无法破解的。然后，我们提出了一种第一种混合技术，包括硬件混淆和使用点函数的逻辑锁定，用于保护并行的直接和转置形式的数字FIR滤波器。实验结果表明，该混合保护技术可以在保持硬件复杂度与传统逻辑锁定方法相当或更好的情况下，使FIR滤波器具有更高的安全性。结果还表明，受保护的乘法器块和FIR滤波器对现有的攻击具有很强的抵抗力。对不同形式和不同实现方式的FIR滤波器的结果表明，并行直接形式FIR滤波器在安全设计方面具有很好的潜力。



## **3. Improving the Transferability of Adversarial Attacks on Face Recognition with Beneficial Perturbation Feature Augmentation**

利用有益扰动特征增强提高人脸识别中敌意攻击的可转移性 cs.CV

**SubmitDate**: 2023-01-26    [abs](http://arxiv.org/abs/2210.16117v2) [paper-pdf](http://arxiv.org/pdf/2210.16117v2)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Ping Li

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. To improve the transferability of adversarial face examples, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of adversarial examples to surrogate FR models by constantly generating new models that have the similar effect of hard samples to craft the adversarial examples. Specifically, in the backpropagation, BPFA records the gradients on pre-selected features and uses the gradient on the input image to craft the adversarial example. In the next forward propagation, BPFA leverages the recorded gradients to add perturbations (i.e., beneficial perturbations) that can be pitted against the adversarial example on their corresponding features. The optimization process of the adversarial example and the optimization process of the beneficial perturbations added on the features correspond to a minimax two-player game. Extensive experiments demonstrate that BPFA can significantly boost the transferability of adversarial attacks on FR.

摘要: 人脸识别(FR)模型很容易被敌意的例子所愚弄，这些例子是通过在良性的人脸图像上添加难以察觉的扰动来构建的。为了提高对抗性人脸样本的可转移性，我们提出了一种新的攻击方法，称为有益扰动特征增强攻击(BPFA)，该方法通过不断生成具有硬样本相似效果的新模型来构造对抗性样本，从而减少了对抗性样本对替代FR模型的过度拟合。具体地说，在反向传播中，BPFA记录预先选择的特征上的梯度，并使用输入图像上的梯度来制作对抗性例子。在下一次前向传播中，BPFA利用记录的梯度来添加扰动(即，有益的扰动)，这些扰动可以在其相应的特征上与对抗性示例相比较。对抗性例子的优化过程和添加在特征上的有益扰动的优化过程对应于极小极大两人博弈。大量实验表明，BPFA能够显著提高对抗性攻击对FR的可转移性。



## **4. Revisiting the Adversarial Robustness-Accuracy Tradeoff in Robot Learning**

机器人学习中对抗性稳健性与准确性权衡的再认识 cs.RO

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2204.07373v2) [paper-pdf](http://arxiv.org/pdf/2204.07373v2)

**Authors**: Mathias Lechner, Alexander Amini, Daniela Rus, Thomas A. Henzinger

**Abstract**: Adversarial training (i.e., training on adversarially perturbed input data) is a well-studied method for making neural networks robust to potential adversarial attacks during inference. However, the improved robustness does not come for free but rather is accompanied by a decrease in overall model accuracy and performance. Recent work has shown that, in practical robot learning applications, the effects of adversarial training do not pose a fair trade-off but inflict a net loss when measured in holistic robot performance. This work revisits the robustness-accuracy trade-off in robot learning by systematically analyzing if recent advances in robust training methods and theory in conjunction with adversarial robot learning, are capable of making adversarial training suitable for real-world robot applications. We evaluate three different robot learning tasks ranging from autonomous driving in a high-fidelity environment amenable to sim-to-real deployment to mobile robot navigation and gesture recognition. Our results demonstrate that, while these techniques make incremental improvements on the trade-off on a relative scale, the negative impact on the nominal accuracy caused by adversarial training still outweighs the improved robustness by an order of magnitude. We conclude that although progress is happening, further advances in robust learning methods are necessary before they can benefit robot learning tasks in practice.

摘要: 对抗性训练(即对对抗性扰动的输入数据进行训练)是一种研究得很好的方法，可以使神经网络在推理过程中对潜在的对抗性攻击具有健壮性。然而，稳健性的提高并不是免费的，而是伴随着总体模型精度和性能的下降。最近的工作表明，在实际的机器人学习应用中，对抗性训练的效果并不构成公平的权衡，而是在衡量整体机器人性能时造成净损失。这项工作回顾了机器人学习中的稳健性和精确度之间的权衡，系统地分析了健壮训练方法和理论的最新进展以及对抗性机器人学习是否能够使对抗性训练适用于真实世界的机器人应用。我们评估了三种不同的机器人学习任务，从高保真环境中的自动驾驶到模拟现实的部署，再到移动机器人导航和手势识别。我们的结果表明，虽然这些技术在相对规模上对权衡做出了渐进的改进，但对抗性训练对标称精度的负面影响仍然超过了健壮性改善的一个数量级。我们的结论是，尽管正在取得进展，但在稳健学习方法能够在实践中有利于机器人学习任务之前，还需要进一步改进。



## **5. RobustPdM: Designing Robust Predictive Maintenance against Adversarial Attacks**

RobustPdM：针对敌方攻击设计稳健的预测性维护 cs.CR

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10822v1) [paper-pdf](http://arxiv.org/pdf/2301.10822v1)

**Authors**: Ayesha Siddique, Ripan Kumar Kundu, Gautam Raj Mode, Khaza Anuarul Hoque

**Abstract**: The state-of-the-art predictive maintenance (PdM) techniques have shown great success in reducing maintenance costs and downtime of complicated machines while increasing overall productivity through extensive utilization of Internet-of-Things (IoT) and Deep Learning (DL). Unfortunately, IoT sensors and DL algorithms are both prone to cyber-attacks. For instance, DL algorithms are known for their susceptibility to adversarial examples. Such adversarial attacks are vastly under-explored in the PdM domain. This is because the adversarial attacks in the computer vision domain for classification tasks cannot be directly applied to the PdM domain for multivariate time series (MTS) regression tasks. In this work, we propose an end-to-end methodology to design adversarially robust PdM systems by extensively analyzing the effect of different types of adversarial attacks and proposing a novel adversarial defense technique for DL-enabled PdM models. First, we propose novel MTS Projected Gradient Descent (PGD) and MTS PGD with random restarts (PGD_r) attacks. Then, we evaluate the impact of MTS PGD and PGD_r along with MTS Fast Gradient Sign Method (FGSM) and MTS Basic Iterative Method (BIM) on Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Convolutional Neural Network (CNN), and Bi-directional LSTM based PdM system. Our results using NASA's turbofan engine dataset show that adversarial attacks can cause a severe defect (up to 11X) in the RUL prediction, outperforming the effectiveness of the state-of-the-art PdM attacks by 3X. Furthermore, we present a novel approximate adversarial training method to defend against adversarial attacks. We observe that approximate adversarial training can significantly improve the robustness of PdM models (up to 54X) and outperforms the state-of-the-art PdM defense methods by offering 3X more robustness.

摘要: 最先进的预测性维护(PDM)技术通过广泛利用物联网(IoT)和深度学习(DL)，在降低复杂机器的维护成本和停机时间方面取得了巨大成功，同时提高了整体生产率。不幸的是，物联网传感器和数字图书馆算法都容易受到网络攻击。例如，DL算法以其对对抗性例子的敏感性而闻名。这种对抗性攻击在产品数据管理领域被极大地忽视了。这是因为用于分类任务的计算机视觉领域中的对抗性攻击不能直接应用于用于多变量时间序列(MTS)回归任务的PDM域。在这项工作中，我们通过广泛分析不同类型的对抗性攻击的影响，提出了一种端到端的方法来设计对抗性健壮的产品数据管理系统，并提出了一种新的针对DL使能的产品数据管理模型的对抗性防御技术。首先，我们提出了新的MTS投影梯度下降(PGD)攻击和带随机重启的MTS PGD(PGD_R)攻击。然后，结合MTS快速梯度符号法(FGSM)和MTS基本迭代法(BIM)，评价了MTS、PGD和PGD_r对基于长短期记忆(LSTM)、门控递归单元(GRU)、卷积神经网络(CNN)和双向LSTM的产品数据管理系统的影响。我们使用NASA的涡扇发动机数据集的结果表明，敌意攻击可以导致RUL预测中的严重缺陷(高达11倍)，比最先进的PDM攻击的有效性高出3倍。此外，我们还提出了一种新的近似对抗性训练方法来防御对抗性攻击。我们观察到，近似对抗性训练可以显著提高PDM模型的健壮性(高达54倍)，并通过提供3倍以上的健壮性来超越最新的PDM防御方法。



## **6. Characterizing the Influence of Graph Elements**

刻画图元素的影响 cs.LG

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2210.07441v2) [paper-pdf](http://arxiv.org/pdf/2210.07441v2)

**Authors**: Zizhang Chen, Peizhao Li, Hongfu Liu, Pengyu Hong

**Abstract**: Influence function, a method from robust statistics, measures the changes of model parameters or some functions about model parameters concerning the removal or modification of training instances. It is an efficient and useful post-hoc method for studying the interpretability of machine learning models without the need for expensive model re-training. Recently, graph convolution networks (GCNs), which operate on graph data, have attracted a great deal of attention. However, there is no preceding research on the influence functions of GCNs to shed light on the effects of removing training nodes/edges from an input graph. Since the nodes/edges in a graph are interdependent in GCNs, it is challenging to derive influence functions for GCNs. To fill this gap, we started with the simple graph convolution (SGC) model that operates on an attributed graph and formulated an influence function to approximate the changes in model parameters when a node or an edge is removed from an attributed graph. Moreover, we theoretically analyzed the error bound of the estimated influence of removing an edge. We experimentally validated the accuracy and effectiveness of our influence estimation function. In addition, we showed that the influence function of an SGC model could be used to estimate the impact of removing training nodes/edges on the test performance of the SGC without re-training the model. Finally, we demonstrated how to use influence functions to guide the adversarial attacks on GCNs effectively.

摘要: 影响函数是稳健统计中的一种方法，它度量模型参数的变化或与训练实例的删除或修改有关的模型参数的某些函数。对于研究机器学习模型的可解释性，它是一种有效和有用的后处理方法，而不需要昂贵的模型重新训练。近年来，处理图形数据的图形卷积网络(GCNS)引起了人们的极大关注。然而，以前还没有关于GCNS影响函数的研究来阐明从输入图中移除训练节点/边的效果。由于图中的节点/边在GCNS中是相互依赖的，因此推导GCNS的影响函数是一项具有挑战性的工作。为了填补这一空白，我们从对属性图进行操作的简单图卷积(SGC)模型开始，并建立了一个影响函数来逼近当从属性图中移除节点或边时模型参数的变化。此外，我们还从理论上分析了去除边缘对估计影响的误差界。我们通过实验验证了我们的影响估计函数的准确性和有效性。此外，我们还证明了SGC模型的影响函数可以用来估计去除训练节点/边对SGC测试性能的影响，而不需要重新训练模型。最后，我们演示了如何使用影响函数来有效地指导对GCNS的对抗性攻击。



## **7. Extending Adversarial Attacks to Produce Adversarial Class Probability Distributions**

扩展对抗性攻击以产生对抗性类别概率分布 cs.LG

Final version as accepted in JMLR. Attribution requirements are  provided at http://jmlr.org/papers/v24/21-0326.html

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2004.06383v3) [paper-pdf](http://arxiv.org/pdf/2004.06383v3)

**Authors**: Jon Vadillo, Roberto Santana, Jose A. Lozano

**Abstract**: Despite the remarkable performance and generalization levels of deep learning models in a wide range of artificial intelligence tasks, it has been demonstrated that these models can be easily fooled by the addition of imperceptible yet malicious perturbations to natural inputs. These altered inputs are known in the literature as adversarial examples. In this paper, we propose a novel probabilistic framework to generalize and extend adversarial attacks in order to produce a desired probability distribution for the classes when we apply the attack method to a large number of inputs. This novel attack paradigm provides the adversary with greater control over the target model, thereby exposing, in a wide range of scenarios, threats against deep learning models that cannot be conducted by the conventional paradigms. We introduce four different strategies to efficiently generate such attacks, and illustrate our approach by extending multiple adversarial attack algorithms. We also experimentally validate our approach for the spoken command classification task and the Tweet emotion classification task, two exemplary machine learning problems in the audio and text domain, respectively. Our results demonstrate that we can closely approximate any probability distribution for the classes while maintaining a high fooling rate and even prevent the attacks from being detected by label-shift detection methods.

摘要: 尽管深度学习模型在广泛的人工智能任务中具有显著的性能和泛化水平，但事实证明，这些模型很容易被自然输入中添加难以察觉但恶意的扰动所愚弄。这些改变的输入在文献中被称为对抗性的例子。在本文中，我们提出了一种新的概率框架来推广和扩展对抗性攻击，以便在对大量输入应用攻击方法时产生期望的类概率分布。这种新的攻击模式为对手提供了对目标模型的更大控制，从而在广泛的场景中暴露了传统模式无法进行的对深度学习模型的威胁。我们介绍了四种不同的策略来有效地生成此类攻击，并通过扩展多个对抗性攻击算法来说明我们的方法。我们还对语音和文本领域中的两个典型机器学习问题--口语命令分类任务和推文情感分类任务--进行了实验验证。我们的结果表明，我们可以在保持较高的愚弄率的同时，接近类的任何概率分布，甚至可以防止标签移位检测方法检测到攻击。



## **8. On the Adversarial Robustness of Camera-based 3D Object Detection**

基于摄像机的三维目标检测的对抗性研究 cs.CV

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10766v1) [paper-pdf](http://arxiv.org/pdf/2301.10766v1)

**Authors**: Shaoyuan Xie, Zichao Li, Zeyu Wang, Cihang Xie

**Abstract**: In recent years, camera-based 3D object detection has gained widespread attention for its ability to achieve high performance with low computational cost. However, the robustness of these methods to adversarial attacks has not been thoroughly examined. In this study, we conduct the first comprehensive investigation of the robustness of leading camera-based 3D object detection methods under various adversarial conditions. Our experiments reveal five interesting findings: (a) the use of accurate depth estimation effectively improves robustness; (b) depth-estimation-free approaches do not show superior robustness; (c) bird's-eye-view-based representations exhibit greater robustness against localization attacks; (d) incorporating multi-frame benign inputs can effectively mitigate adversarial attacks; and (e) addressing long-tail problems can enhance robustness. We hope our work can provide guidance for the design of future camera-based object detection modules with improved adversarial robustness.

摘要: 近年来，基于摄像机的三维目标检测由于能够以较低的计算代价获得较高的检测性能而受到广泛关注。然而，这些方法对对抗性攻击的稳健性还没有得到彻底的检验。在这项研究中，我们首次全面考察了主流的基于摄像机的3D目标检测方法在各种对抗条件下的稳健性。我们的实验揭示了五个有趣的发现：(A)使用精确的深度估计有效地提高了稳健性；(B)无深度估计的方法没有表现出更好的稳健性；(C)基于鸟瞰视图的表示对局部化攻击表现出更强的稳健性；(D)结合多帧良性输入可以有效地缓解对抗性攻击；以及(E)解决长尾问题可以增强稳健性。我们希望我们的工作能够为未来基于摄像机的目标检测模块的设计提供指导，提高对抗的健壮性。



## **9. A Study on FGSM Adversarial Training for Neural Retrieval**

面向神经检索的FGSM对抗性训练研究 cs.IR

Accepted at ECIR 2023

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10576v1) [paper-pdf](http://arxiv.org/pdf/2301.10576v1)

**Authors**: Simon Lupart, Stéphane Clinchant

**Abstract**: Neural retrieval models have acquired significant effectiveness gains over the last few years compared to term-based methods. Nevertheless, those models may be brittle when faced to typos, distribution shifts or vulnerable to malicious attacks. For instance, several recent papers demonstrated that such variations severely impacted models performances, and then tried to train more resilient models. Usual approaches include synonyms replacements or typos injections -- as data-augmentation -- and the use of more robust tokenizers (characterBERT, BPE-dropout). To further complement the literature, we investigate in this paper adversarial training as another possible solution to this robustness issue. Our comparison includes the two main families of BERT-based neural retrievers, i.e. dense and sparse, with and without distillation techniques. We then demonstrate that one of the most simple adversarial training techniques -- the Fast Gradient Sign Method (FGSM) -- can improve first stage rankers robustness and effectiveness. In particular, FGSM increases models performances on both in-domain and out-of-domain distributions, and also on queries with typos, for multiple neural retrievers.

摘要: 与基于术语的方法相比，神经检索模型在过去几年中获得了显著的有效性收益。然而，当面临打字错误、分布变化或易受恶意攻击时，这些模型可能会变得脆弱。例如，最近的几篇论文证明了这种变化严重影响了模型的性能，然后试图训练更具弹性的模型。通常的方法包括同义词替换或打字错误注入--作为数据增强--以及使用更健壮的标记器(CharacterBERT，BPE-Dropout)。为了进一步补充文献，我们在本文中调查了对抗性训练作为这个稳健性问题的另一种可能的解决方案。我们的比较包括两个主要的基于BERT的神经检索器家族，即密集和稀疏，使用和不使用蒸馏技术。然后，我们展示了一种最简单的对抗性训练技术--快速梯度符号方法(FGSM)--可以提高第一阶段排名者的稳健性和有效性。特别是，FGSM提高了模型在域内和域外分布上的性能，以及对于多个神经检索器的具有拼写错误的查询的性能。



## **10. A Data-Centric Approach for Improving Adversarial Training Through the Lens of Out-of-Distribution Detection**

一种以数据为中心的方法，通过非分布检测镜头改善对手训练 cs.LG

Accepted to CSICC 2023

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10454v1) [paper-pdf](http://arxiv.org/pdf/2301.10454v1)

**Authors**: Mohammad Azizmalayeri, Arman Zarei, Alireza Isavand, Mohammad Taghi Manzuri, Mohammad Hossein Rohban

**Abstract**: Current machine learning models achieve super-human performance in many real-world applications. Still, they are susceptible against imperceptible adversarial perturbations. The most effective solution for this problem is adversarial training that trains the model with adversarially perturbed samples instead of original ones. Various methods have been developed over recent years to improve adversarial training such as data augmentation or modifying training attacks. In this work, we examine the same problem from a new data-centric perspective. For this purpose, we first demonstrate that the existing model-based methods can be equivalent to applying smaller perturbation or optimization weights to the hard training examples. By using this finding, we propose detecting and removing these hard samples directly from the training procedure rather than applying complicated algorithms to mitigate their effects. For detection, we use maximum softmax probability as an effective method in out-of-distribution detection since we can consider the hard samples as the out-of-distribution samples for the whole data distribution. Our results on SVHN and CIFAR-10 datasets show the effectiveness of this method in improving the adversarial training without adding too much computational cost.

摘要: 目前的机器学习模型在许多真实世界的应用中都取得了超乎人类的性能。尽管如此，他们还是很容易受到难以察觉的对抗性干扰。这个问题最有效的解决方案是对抗性训练，它用对抗性扰动的样本而不是原始样本来训练模型。近年来，已经开发了各种方法来改进对抗性训练，例如数据增强或修改训练攻击。在这项工作中，我们从一个新的以数据为中心的角度来研究同样的问题。为此，我们首先证明了现有的基于模型的方法可以等价于对硬训练样本施加较小的扰动或优化权重。通过使用这一发现，我们建议直接从训练过程中检测和移除这些硬样本，而不是应用复杂的算法来减轻它们的影响。对于检测，我们使用最大Softmax概率作为非分布检测的有效方法，因为我们可以将硬样本视为整个数据分布的非分布样本。在SVHN和CIFAR-10数据集上的实验结果表明，该方法在不增加太多计算代价的情况下，有效地改善了对抗性训练。



## **11. BDMMT: Backdoor Sample Detection for Language Models through Model Mutation Testing**

BDMMT：基于模型突变测试的语言模型后门样本检测 cs.CL

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.10412v1) [paper-pdf](http://arxiv.org/pdf/2301.10412v1)

**Authors**: Jiali Wei, Ming Fan, Wenjing Jiao, Wuxia Jin, Ting Liu

**Abstract**: Deep neural networks (DNNs) and natural language processing (NLP) systems have developed rapidly and have been widely used in various real-world fields. However, they have been shown to be vulnerable to backdoor attacks. Specifically, the adversary injects a backdoor into the model during the training phase, so that input samples with backdoor triggers are classified as the target class. Some attacks have achieved high attack success rates on the pre-trained language models (LMs), but there have yet to be effective defense methods. In this work, we propose a defense method based on deep model mutation testing. Our main justification is that backdoor samples are much more robust than clean samples if we impose random mutations on the LMs and that backdoors are generalizable. We first confirm the effectiveness of model mutation testing in detecting backdoor samples and select the most appropriate mutation operators. We then systematically defend against three extensively studied backdoor attack levels (i.e., char-level, word-level, and sentence-level) by detecting backdoor samples. We also make the first attempt to defend against the latest style-level backdoor attacks. We evaluate our approach on three benchmark datasets (i.e., IMDB, Yelp, and AG news) and three style transfer datasets (i.e., SST-2, Hate-speech, and AG news). The extensive experimental results demonstrate that our approach can detect backdoor samples more efficiently and accurately than the three state-of-the-art defense approaches.

摘要: 深度神经网络(DNN)和自然语言处理(NLP)系统发展迅速，并广泛应用于现实世界的各个领域。然而，它们已被证明容易受到后门攻击。具体地说，对手在训练阶段向模型注入后门，以便具有后门触发器的输入样本被归类为目标类。一些攻击已经在预先训练的语言模型(LMS)上取得了很高的攻击成功率，但还没有有效的防御方法。在这项工作中，我们提出了一种基于深度模型突变测试的防御方法。我们的主要理由是，如果我们对LMS施加随机突变，那么后门样本比干净样本要健壮得多，而且后门样本是可以推广的。我们首先确认了模型突变测试在检测后门样本中的有效性，并选择了最合适的突变算子。然后，我们通过检测后门样本来系统地防御三个被广泛研究的后门攻击级别(即字符级别、单词级别和句子级别)。我们还首次尝试防御最新的风格级后门攻击。我们在三个基准数据集(即IMDB、Yelp和AG新闻)和三个风格传输数据集(即SST-2、仇恨演讲和AG新闻)上对我们的方法进行了评估。大量的实验结果表明，我们的方法可以比三种最先进的防御方法更有效和更准确地检测后门样本。



## **12. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

自适应神经网络的动态感知敌意攻击 cs.CV

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2210.08159v2) [paper-pdf](http://arxiv.org/pdf/2210.08159v2)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods.

摘要: 本文研究了自适应神经网络的动态感知对抗攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络体系结构在整个攻击过程中是固定的。然而，这一假设对最近提出的许多自适应神经网络并不成立，这些自适应神经网络基于输入自适应地停用不必要的执行单元来提高计算效率。它导致了严重的梯度滞后问题，使得当前步骤的学习攻击由于之后的体系结构变化而无效。为了解决这个问题，我们提出了一种引导梯度法(LGM)，并展示了滞后梯度的显著影响。更具体地说，我们重新制定了梯度，以了解网络体系结构的潜在动态变化，以便在网络体系结构动态变化时，学习到的攻击比不知道动态变化的方法更好地“引导”下一步。在典型的自适应神经网络上对2D图像和3D点云进行的大量实验表明，与动态未知攻击方法相比，我们的LGM具有令人印象深刻的对抗性攻击性能。



## **13. Blockchain-aided Secure Semantic Communication for AI-Generated Content in Metaverse**

区块链辅助Metverse中AI生成内容的安全语义通信 cs.CR

10 pages, 8 figures, journal

**SubmitDate**: 2023-01-25    [abs](http://arxiv.org/abs/2301.11289v1) [paper-pdf](http://arxiv.org/pdf/2301.11289v1)

**Authors**: Yijing Lin, Hongyang Du, Dusit Niyato, Jiangtian Nie, Jiayi Zhang, Yanyu Cheng, Zhaohui Yang

**Abstract**: The construction of virtual transportation networks requires massive data to be transmitted from edge devices to Virtual Service Providers (VSP) to facilitate circulations between the physical and virtual domains in Metaverse. Leveraging semantic communication for reducing information redundancy, VSPs can receive semantic data from edge devices to provide varied services through advanced techniques, e.g., AI-Generated Content (AIGC), for users to explore digital worlds. But the use of semantic communication raises a security issue because attackers could send malicious semantic data with similar semantic information but different desired content to break Metaverse services and cause wrong output of AIGC. Therefore, in this paper, we first propose a blockchain-aided semantic communication framework for AIGC services in virtual transportation networks to facilitate interactions of the physical and virtual domains among VSPs and edge devices. We illustrate a training-based targeted semantic attack scheme to generate adversarial semantic data by various loss functions. We also design a semantic defense scheme that uses the blockchain and zero-knowledge proofs to tell the difference between the semantic similarities of adversarial and authentic semantic data and to check the authenticity of semantic data transformations. Simulation results show that the proposed defense method can reduce the semantic similarity of the adversarial semantic data and the authentic ones by up to 30% compared with the attack scheme.

摘要: 虚拟交通网络的构建需要将大量数据从边缘设备传输到虚拟服务提供商(VSP)，以促进Metverse中物理域和虚拟域之间的流通。利用语义通信减少信息冗余，VSP可以从边缘设备接收语义数据，通过先进的技术提供多样化的服务，例如人工智能生成内容(AIGC)，供用户探索数字世界。但是，语义通信的使用带来了一个安全问题，因为攻击者可以发送具有相似语义信息但不同期望内容的恶意语义数据来破坏Metverse服务，并导致AIGC的错误输出。为此，本文首先提出了一种基于区块链的虚拟交通网络中AIGC服务的语义通信框架，以促进VSP和边缘设备之间物理域和虚拟域的交互。给出了一种基于训练的目标语义攻击方案，通过各种损失函数生成对抗性语义数据。我们还设计了一个语义防御方案，使用区块链和零知识证明来区分敌意语义数据和真实语义数据的语义相似性，并检查语义数据转换的真实性。仿真结果表明，与攻击方案相比，该防御方法可以将敌意语义数据与真实语义数据的语义相似度降低30%。



## **14. To Trust or Not To Trust Prediction Scores for Membership Inference Attacks**

信任还是不信任成员关系推断攻击的预测分数 cs.LG

15 pages, 8 figures, 10 tables

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2111.09076v3) [paper-pdf](http://arxiv.org/pdf/2111.09076v3)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstract**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures. Consequently, MIAs will miserably fail since overconfidence leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated, and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of models and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions, the more they reveal the training data.

摘要: 成员关系推理攻击(MIA)的目的是确定特定样本是否被用来训练预测模型。知道这一点确实可能会导致隐私被侵犯。然而，大多数MIA都利用模型的预测分数--每个输出给定一些输入的概率--遵循这样一种直觉，即训练后的模型在其训练数据上往往表现不同。我们认为，对于许多现代深度网络体系结构来说，这是一种谬误。因此，MIA将悲惨地失败，因为过度自信不仅会导致已知域上的高假阳性率，而且还会导致分布外数据的高假阳性率，并隐含地充当对MIA的防御。具体地说，使用生成性对抗性网络，我们能够产生潜在无限数量的样本，这些样本被错误地归类为训练数据的一部分。换句话说，MIA的威胁被高估了，泄露的信息比之前假设的要少。此外，在模型的过度自信和他们对MIA的敏感性之间实际上存在着权衡：分类器知道的越多，他们不知道的时候，做出低置信度预测的人就越多，他们透露的训练数据就越多。



## **15. Robustness through Data Augmentation Loss Consistency**

通过数据增强丢失一致性实现的健壮性 cs.LG

40 pages

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2110.11205v3) [paper-pdf](http://arxiv.org/pdf/2110.11205v3)

**Authors**: Tianjian Huang, Shaunak Halbe, Chinnadhurai Sankar, Pooyan Amini, Satwik Kottur, Alborz Geramifard, Meisam Razaviyayn, Ahmad Beirami

**Abstract**: While deep learning through empirical risk minimization (ERM) has succeeded at achieving human-level performance at a variety of complex tasks, ERM is not robust to distribution shifts or adversarial attacks. Synthetic data augmentation followed by empirical risk minimization (DA-ERM) is a simple and widely used solution to improve robustness in ERM. In addition, consistency regularization can be applied to further improve the robustness of the model by forcing the representation of the original sample and the augmented one to be similar. However, existing consistency regularization methods are not applicable to covariant data augmentation, where the label in the augmented sample is dependent on the augmentation function. For example, dialog state covaries with named entity when we augment data with a new named entity. In this paper, we propose data augmented loss invariant regularization (DAIR), a simple form of consistency regularization that is applied directly at the loss level rather than intermediate features, making it widely applicable to both invariant and covariant data augmentation regardless of network architecture, problem setup, and task. We apply DAIR to real-world learning problems involving covariant data augmentation: robust neural task-oriented dialog state tracking and robust visual question answering. We also apply DAIR to tasks involving invariant data augmentation: robust regression, robust classification against adversarial attacks, and robust ImageNet classification under distribution shift. Our experiments show that DAIR consistently outperforms ERM and DA-ERM with little marginal computational cost and sets new state-of-the-art results in several benchmarks involving covariant data augmentation. Our code of all experiments is available at: https://github.com/optimization-for-data-driven-science/DAIR.git

摘要: 虽然通过经验风险最小化(ERM)的深度学习在各种复杂任务中成功地实现了人类水平的绩效，但ERM对分布变化或对手攻击并不健壮。合成数据增强和经验风险最小化(DA-ERM)是提高ERM稳健性的一种简单而广泛使用的解决方案。此外，一致性正则化可以通过强制原始样本和扩展样本的表示相似来进一步提高模型的稳健性。然而，现有的一致性正则化方法不适用于协变数据扩充，即扩充样本中的标签依赖于扩充函数。例如，当我们使用新的命名实体扩充数据时，对话框状态与命名实体相关。在本文中，我们提出了数据增强损失不变正则化(DAIR)，这是一种简单的一致性正则化形式，直接应用于丢失级而不是中间特征，使其广泛适用于不变和协变数据增强，而不受网络结构、问题设置和任务的影响。我们将DAIR应用于涉及协变数据扩充的真实世界学习问题：稳健的神经面向任务的对话状态跟踪和稳健的视觉问答。我们还将DAIR应用于涉及不变数据扩充的任务：稳健回归、对抗对手攻击的稳健分类以及分布漂移下的稳健ImageNet分类。我们的实验表明，DAIR的性能始终优于ERM和DA-ERM，而边际计算开销很小，并且在几个涉及协变数据增强的基准测试中创造了新的最先进的结果。我们所有实验的代码都在：https://github.com/optimization-for-data-driven-science/DAIR.git上。



## **16. RAIN: RegulArization on Input and Network for Black-Box Domain Adaptation**

RAIN：黑箱域自适应的输入和网络规则化 cs.CV

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2208.10531v2) [paper-pdf](http://arxiv.org/pdf/2208.10531v2)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstract**: Source-Free domain adaptation transits the source-trained model towards target domain without exposing the source data, trying to dispel these concerns about data privacy and security. However, this paradigm is still at risk of data leakage due to adversarial attacks on the source model. Hence, the Black-Box setting only allows to use the outputs of source model, but still suffers from overfitting on the source domain more severely due to source model's unseen weights. In this paper, we propose a novel approach named RAIN (RegulArization on Input and Network) for Black-Box domain adaptation from both input-level and network-level regularization. For the input-level, we design a new data augmentation technique as Phase MixUp, which highlights task-relevant objects in the interpolations, thus enhancing input-level regularization and class consistency for target models. For network-level, we develop a Subnetwork Distillation mechanism to transfer knowledge from the target subnetwork to the full target network via knowledge distillation, which thus alleviates overfitting on the source domain by learning diverse target representations. Extensive experiments show that our method achieves state-of-the-art performance on several cross-domain benchmarks under both single- and multi-source black-box domain adaptation.

摘要: 无源域自适应在不暴露源数据的情况下将源训练模型过渡到目标域，试图消除这些对数据隐私和安全的担忧。然而，由于对源模型的对抗性攻击，该范例仍然面临数据泄露的风险。因此，黑盒设置只允许使用源模型的输出，但由于源模型的不可见权重，仍然受到源域上的过度拟合的更严重影响。本文从输入级正则化和网络级正则化两个方面提出了一种新的黑箱域自适应方法RAIN。对于输入层，我们设计了一种新的数据增强技术--阶段混合，它在内插中突出与任务相关的对象，从而增强了输入层的正则性和目标模型的类一致性。对于网络级，我们提出了一种子网络精馏机制，通过知识精馏将知识从目标子网络传递到整个目标网络，从而通过学习不同的目标表示来缓解源域的过度匹配。大量的实验表明，在单源和多源黑盒领域自适应的情况下，我们的方法在多个跨域基准测试上都达到了最好的性能。



## **17. Robust Fair Clustering: A Novel Fairness Attack and Defense Framework**

稳健公平聚类：一种新的公平攻防框架 cs.LG

Accepted to the 11th International Conference on Learning  Representations (ICLR 2023)

**SubmitDate**: 2023-01-24    [abs](http://arxiv.org/abs/2210.01953v2) [paper-pdf](http://arxiv.org/pdf/2210.01953v2)

**Authors**: Anshuman Chhabra, Peizhao Li, Prasant Mohapatra, Hongfu Liu

**Abstract**: Clustering algorithms are widely used in many societal resource allocation applications, such as loan approvals and candidate recruitment, among others, and hence, biased or unfair model outputs can adversely impact individuals that rely on these applications. To this end, many fair clustering approaches have been recently proposed to counteract this issue. Due to the potential for significant harm, it is essential to ensure that fair clustering algorithms provide consistently fair outputs even under adversarial influence. However, fair clustering algorithms have not been studied from an adversarial attack perspective. In contrast to previous research, we seek to bridge this gap and conduct a robustness analysis against fair clustering by proposing a novel black-box fairness attack. Through comprehensive experiments, we find that state-of-the-art models are highly susceptible to our attack as it can reduce their fairness performance significantly. Finally, we propose Consensus Fair Clustering (CFC), the first robust fair clustering approach that transforms consensus clustering into a fair graph partitioning problem, and iteratively learns to generate fair cluster outputs. Experimentally, we observe that CFC is highly robust to the proposed attack and is thus a truly robust fair clustering alternative.

摘要: 聚类算法广泛应用于许多社会资源分配应用中，如贷款审批和候选人招聘等，因此，有偏见或不公平的模型输出可能会对依赖这些应用的个人产生不利影响。为此，最近提出了许多公平的聚类方法来解决这个问题。由于潜在的重大危害，确保公平的聚类算法即使在敌意影响下也能提供一致的公平输出是至关重要的。然而，公平分簇算法还没有从对抗攻击的角度进行研究。与以往的研究不同，我们试图弥补这一差距，并通过提出一种新的黑盒公平攻击来进行针对公平聚类的健壮性分析。通过全面的实验，我们发现最新的模型非常容易受到我们的攻击，因为它会显著降低它们的公平性能。最后，我们提出了共识公平聚类(CFC)，这是第一种将共识聚类转化为公平图划分问题的稳健公平聚类方法，并迭代地学习生成公平聚类输出。在实验上，我们观察到cfc对所提出的攻击具有高度的健壮性，因此是一种真正健壮的公平集群替代方案。



## **18. DODEM: DOuble DEfense Mechanism Against Adversarial Attacks Towards Secure Industrial Internet of Things Analytics**

DODEM：针对安全工业物联网分析的双重防御机制 cs.CR

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09740v1) [paper-pdf](http://arxiv.org/pdf/2301.09740v1)

**Authors**: Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstract**: Industrial Internet of Things (I-IoT) is a collaboration of devices, sensors, and networking equipment to monitor and collect data from industrial operations. Machine learning (ML) methods use this data to make high-level decisions with minimal human intervention. Data-driven predictive maintenance (PDM) is a crucial ML-based I-IoT application to find an optimal maintenance schedule for industrial assets. The performance of these ML methods can seriously be threatened by adversarial attacks where an adversary crafts perturbed data and sends it to the ML model to deteriorate its prediction performance. The models should be able to stay robust against these attacks where robustness is measured by how much perturbation in input data affects model performance. Hence, there is a need for effective defense mechanisms that can protect these models against adversarial attacks. In this work, we propose a double defense mechanism to detect and mitigate adversarial attacks in I-IoT environments. We first detect if there is an adversarial attack on a given sample using novelty detection algorithms. Then, based on the outcome of our algorithm, marking an instance as attack or normal, we select adversarial retraining or standard training to provide a secondary defense layer. If there is an attack, adversarial retraining provides a more robust model, while we apply standard training for regular samples. Since we may not know if an attack will take place, our adaptive mechanism allows us to consider irregular changes in data. The results show that our double defense strategy is highly efficient where we can improve model robustness by up to 64.6% and 52% compared to standard and adversarial retraining, respectively.

摘要: 工业物联网(I-IoT)是设备、传感器和网络设备的协作，用于监控和收集工业运营中的数据。机器学习(ML)方法使用这些数据做出高层决策，只需最少的人工干预。数据驱动的预测性维护是一种基于ML的物联网应用，用于为工业资产寻找最优的维护计划。这些最大似然方法的性能会受到敌意攻击的严重威胁，在这种攻击中，敌手伪造扰动数据并将其发送到最大似然模型以降低其预测性能。模型应该能够保持对这些攻击的健壮性，其中健壮性是通过输入数据中的扰动对模型性能的影响来衡量的。因此，需要有效的防御机制来保护这些模型免受对手攻击。在这项工作中，我们提出了一种双重防御机制来检测和缓解I-IoT环境中的敌意攻击。我们首先使用新颖性检测算法检测给定样本上是否存在敌意攻击。然后，根据算法的结果，将实例标记为攻击或正常，选择对抗性再训练或标准训练来提供二次防御层。如果发生攻击，对抗性再训练提供了一个更稳健的模型，而我们对常规样本应用标准训练。由于我们可能不知道是否会发生攻击，我们的自适应机制允许我们考虑数据的不规则变化。结果表明，我们的双重防御策略是高效的，与标准和对抗性再训练相比，我们可以分别提高64.6%和52%的模型稳健性。



## **19. ESWORD: Implementation of Wireless Jamming Attacks in a Real-World Emulated Network**

ESWORD：无线干扰攻击在真实网络中的实现 cs.NI

6 pages, 7 figures, 1 table. IEEE Wireless Communications and  Networking Conference (WCNC), Glasgow, Scotland, March 2023

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09615v1) [paper-pdf](http://arxiv.org/pdf/2301.09615v1)

**Authors**: Clifton Paul Robinson, Leonardo Bonati, Tara Van Nieuwstadt, Teddy Reiss, Pedram Johari, Michele Polese, Hieu Nguyen, Curtis Watson, Tommaso Melodia

**Abstract**: Wireless jamming attacks have plagued wireless communication systems and will continue to do so going forward with technological advances. These attacks fall under the category of Electronic Warfare (EW), a continuously growing area in both attack and defense of the electromagnetic spectrum, with one subcategory being electronic attacks. Jamming attacks fall under this specific subcategory of EW as they comprise adversarial signals that attempt to disrupt, deny, degrade, destroy, or deceive legitimate signals in the electromagnetic spectrum. While jamming is not going away, recent research advances have started to get the upper hand against these attacks by leveraging new methods and techniques, such as machine learning. However, testing such jamming solutions on a wide and realistic scale is a daunting task due to strict regulations on spectrum emissions. In this paper, we introduce eSWORD, the first large-scale framework that allows users to safely conduct real-time and controlled jamming experiments with hardware-in-the-loop. This is done by integrating eSWORD into the Colosseum wireless network emulator that enables large-scale experiments with up to 50 software-defined radio nodes. We compare the performance of eSWORD with that of real-world jamming systems by using an over-the-air wireless testbed (ensuring safe measures were taken when conducting experiments). Our experimental results demonstrate that eSWORD follows similar patterns in throughput, signal-to-noise ratio, and link status to real-world jamming experiments, testifying to the high accuracy of the emulated eSWORD setup.

摘要: 无线干扰攻击已经困扰着无线通信系统，并将随着技术的进步继续这样做。这些攻击属于电子战(EW)类别，这是一个在电磁频谱攻击和防御方面不断增长的领域，其中一个子类别是电子攻击。干扰攻击属于电子战这一特定的子类别，因为它们包含试图干扰、拒绝、降低、破坏或欺骗电磁频谱中的合法信号的敌对信号。虽然干扰不会消失，但最近的研究进展已经开始利用新的方法和技术，如机器学习，在对抗这些攻击方面占据上风。然而，由于对频谱排放的严格规定，在广泛和现实的范围内测试这种干扰解决方案是一项艰巨的任务。在本文中，我们介绍了eSWORD，这是第一个大规模框架，允许用户安全地进行实时和受控的半实物干扰实验。这是通过将eSWORD集成到Colosseum无线网络仿真器中来实现的，该仿真器可以使用多达50个软件定义的无线电节点进行大规模实验。我们使用空中无线试验台将eSWORD与真实干扰系统的性能进行了比较(确保在进行实验时采取了安全措施)。我们的实验结果表明，eSWORD在吞吐量、信噪比和链路状态方面与真实世界的干扰实验具有相似的规律，证明了仿真的eSWORD设置的高精度。



## **20. BayBFed: Bayesian Backdoor Defense for Federated Learning**

贝叶斯联邦储备银行：联邦学习的贝叶斯后门防御 cs.LG

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09508v1) [paper-pdf](http://arxiv.org/pdf/2301.09508v1)

**Authors**: Kavita Kumari, Phillip Rieger, Hossein Fereidooni, Murtuza Jadliwala, Ahmad-Reza Sadeghi

**Abstract**: Federated learning (FL) allows participants to jointly train a machine learning model without sharing their private data with others. However, FL is vulnerable to poisoning attacks such as backdoor attacks. Consequently, a variety of defenses have recently been proposed, which have primarily utilized intermediary states of the global model (i.e., logits) or distance of the local models (i.e., L2-norm) from the global model to detect malicious backdoors. However, as these approaches directly operate on client updates, their effectiveness depends on factors such as clients' data distribution or the adversary's attack strategies. In this paper, we introduce a novel and more generic backdoor defense framework, called BayBFed, which proposes to utilize probability distributions over client updates to detect malicious updates in FL: it computes a probabilistic measure over the clients' updates to keep track of any adjustments made in the updates, and uses a novel detection algorithm that can leverage this probabilistic measure to efficiently detect and filter out malicious updates. Thus, it overcomes the shortcomings of previous approaches that arise due to the direct usage of client updates; as our probabilistic measure will include all aspects of the local client training strategies. BayBFed utilizes two Bayesian Non-Parametric extensions: (i) a Hierarchical Beta-Bernoulli process to draw a probabilistic measure given the clients' updates, and (ii) an adaptation of the Chinese Restaurant Process (CRP), referred by us as CRP-Jensen, which leverages this probabilistic measure to detect and filter out malicious updates. We extensively evaluate our defense approach on five benchmark datasets: CIFAR10, Reddit, IoT intrusion detection, MNIST, and FMNIST, and show that it can effectively detect and eliminate malicious updates in FL without deteriorating the benign performance of the global model.

摘要: 联合学习(FL)允许参与者共同训练机器学习模型，而不与其他人共享他们的私人数据。然而，FL很容易受到后门攻击等中毒攻击。因此，最近提出了各种防御措施，它们主要利用全局模型的中间状态(即Logits)或局部模型与全局模型的距离(即L2范数)来检测恶意后门。然而，由于这些方法直接操作于客户端更新，其有效性取决于客户端的数据分布或对手的攻击策略等因素。在本文中，我们介绍了一种新颖的、更通用的后门防御框架BayBFed，它提出利用客户端更新的概率分布来检测FL中的恶意更新：它计算客户端更新的概率度量来跟踪更新中的任何调整，并使用一种新的检测算法来利用这种概率度量来有效地检测和过滤恶意更新。因此，它克服了以前因直接使用客户更新而出现的缺点；因为我们的概率衡量标准将包括当地客户培训战略的所有方面。BayBFed利用两个贝叶斯非参数扩展：(I)层次化的Beta-Bernoulli过程来得出给定客户更新的概率度量，以及(Ii)对中式餐厅过程(CRP)的适应，我们称之为CRP-Jensen，它利用这种概率度量来检测和过滤恶意更新。我们在CIFAR10、Reddit、IoT入侵检测、MNIST和FMNIST五个基准数据集上对我们的防御方法进行了广泛的评估，结果表明，它可以在不影响全局模型的良性性能的情况下，有效地检测和消除FL中的恶意更新。



## **21. Practical Adversarial Attacks Against AI-Driven Power Allocation in a Distributed MIMO Network**

分布式MIMO网络中人工智能驱动功率分配的实用对抗性攻击 eess.SP

6 pages, 10 figures, accepted for presentation in International  Conference on Communications (ICC) 2023 in Communication and Information  System Security Symposium

**SubmitDate**: 2023-01-23    [abs](http://arxiv.org/abs/2301.09305v1) [paper-pdf](http://arxiv.org/pdf/2301.09305v1)

**Authors**: Ömer Faruk Tuna, Fehmi Emre Kadan, Leyli Karaçay

**Abstract**: In distributed multiple-input multiple-output (D-MIMO) networks, power control is crucial to optimize the spectral efficiencies of users and max-min fairness (MMF) power control is a commonly used strategy as it satisfies uniform quality-of-service to all users. The optimal solution of MMF power control requires high complexity operations and hence deep neural network based artificial intelligence (AI) solutions are proposed to decrease the complexity. Although quite accurate models can be achieved by using AI, these models have some intrinsic vulnerabilities against adversarial attacks where carefully crafted perturbations are applied to the input of the AI model. In this work, we show that threats against the target AI model which might be originated from malicious users or radio units can substantially decrease the network performance by applying a successful adversarial sample, even in the most constrained circumstances. We also demonstrate that the risk associated with these kinds of adversarial attacks is higher than the conventional attack threats. Detailed simulations reveal the effectiveness of adversarial attacks and the necessity of smart defense techniques.

摘要: 在分布式多输入多输出(D-MIMO)网络中，功率控制是优化用户频谱效率的关键，而最大-最小公平(MMF)功率控制是一种常用的策略，因为它满足所有用户的统一服务质量。MMF功率控制的最优解需要很高的运算复杂度，因此提出了基于深度神经网络的人工智能解决方案来降低复杂度。虽然使用人工智能可以实现相当准确的模型，但这些模型在对抗对手攻击时存在一些固有的漏洞，其中精心设计的扰动应用于AI模型的输入。在这项工作中，我们表明，针对目标人工智能模型的威胁可能来自恶意用户或无线电单元，通过应用成功的对抗性样本可以显著降低网络性能，即使在最受限制的环境中也是如此。我们还证明了与这些类型的对抗性攻击相关的风险比常规攻击威胁更高。详细的仿真揭示了对抗性攻击的有效性和智能防御技术的必要性。



## **22. ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning**

ContraBERT：通过对比学习增强代码预训练模型 cs.SE

**SubmitDate**: 2023-01-22    [abs](http://arxiv.org/abs/2301.09072v1) [paper-pdf](http://arxiv.org/pdf/2301.09072v1)

**Authors**: Shangqing Liu, Bozhi Wu, Xiaofei Xie, Guozhu Meng, Yang Liu

**Abstract**: Large-scale pre-trained models such as CodeBERT, GraphCodeBERT have earned widespread attention from both academia and industry. Attributed to the superior ability in code representation, they have been further applied in multiple downstream tasks such as clone detection, code search and code translation. However, it is also observed that these state-of-the-art pre-trained models are susceptible to adversarial attacks. The performance of these pre-trained models drops significantly with simple perturbations such as renaming variable names. This weakness may be inherited by their downstream models and thereby amplified at an unprecedented scale. To this end, we propose an approach namely ContraBERT that aims to improve the robustness of pre-trained models via contrastive learning. Specifically, we design nine kinds of simple and complex data augmentation operators on the programming language (PL) and natural language (NL) data to construct different variants. Furthermore, we continue to train the existing pre-trained models by masked language modeling (MLM) and contrastive pre-training task on the original samples with their augmented variants to enhance the robustness of the model. The extensive experiments demonstrate that ContraBERT can effectively improve the robustness of the existing pre-trained models. Further study also confirms that these robustness-enhanced models provide improvements as compared to original models over four popular downstream tasks.

摘要: CodeBERT、GraphCodeBERT等大规模的预训练模型得到了学术界和产业界的广泛关注。由于在代码表示方面的优越能力，它们已被进一步应用于克隆检测、代码搜索和代码翻译等多个下游任务。然而，也有人观察到，这些最先进的预先训练的模型容易受到对手的攻击。这些预先训练的模型的性能随着简单的扰动而显著下降，例如重命名变量名称。这一弱点可能会被它们的下游模式继承，从而以前所未有的规模放大。为此，我们提出了一种名为ContraBERT的方法，旨在通过对比学习来提高预训练模型的稳健性。具体来说，我们设计了九种简单和复杂的数据扩充算子，分别对程序设计语言(PL)和自然语言(NL)数据构造不同的变体。此外，我们继续通过掩蔽语言建模(MLM)和对比预训练任务对现有的预训练模型进行训练，以增强模型的稳健性。大量实验表明，ContraBERT能够有效地提高现有预训练模型的稳健性。进一步的研究还证实，与原始模型相比，这些增强了稳健性的模型在四个流行的下游任务中提供了改进。



## **23. Provable Unrestricted Adversarial Training without Compromise with Generalizability**

可证明的不受限制的对抗性训练，不妥协于泛化 cs.LG

**SubmitDate**: 2023-01-22    [abs](http://arxiv.org/abs/2301.09069v1) [paper-pdf](http://arxiv.org/pdf/2301.09069v1)

**Authors**: Lilin Zhang, Ning Yang, Yanchao Sun, Philip S. Yu

**Abstract**: Adversarial training (AT) is widely considered as the most promising strategy to defend against adversarial attacks and has drawn increasing interest from researchers. However, the existing AT methods still suffer from two challenges. First, they are unable to handle unrestricted adversarial examples (UAEs), which are built from scratch, as opposed to restricted adversarial examples (RAEs), which are created by adding perturbations bound by an $l_p$ norm to observed examples. Second, the existing AT methods often achieve adversarial robustness at the expense of standard generalizability (i.e., the accuracy on natural examples) because they make a tradeoff between them. To overcome these challenges, we propose a unique viewpoint that understands UAEs as imperceptibly perturbed unobserved examples. Also, we find that the tradeoff results from the separation of the distributions of adversarial examples and natural examples. Based on these ideas, we propose a novel AT approach called Provable Unrestricted Adversarial Training (PUAT), which can provide a target classifier with comprehensive adversarial robustness against both UAE and RAE, and simultaneously improve its standard generalizability. Particularly, PUAT utilizes partially labeled data to achieve effective UAE generation by accurately capturing the natural data distribution through a novel augmented triple-GAN. At the same time, PUAT extends the traditional AT by introducing the supervised loss of the target classifier into the adversarial loss and achieves the alignment between the UAE distribution, the natural data distribution, and the distribution learned by the classifier, with the collaboration of the augmented triple-GAN. Finally, the solid theoretical analysis and extensive experiments conducted on widely-used benchmarks demonstrate the superiority of PUAT.

摘要: 对抗训练(AT)被广泛认为是防御对抗攻击的最有前途的策略，引起了越来越多的研究人员的兴趣。然而，现有的AT方法仍然面临着两个挑战。首先，它们不能处理从头开始构建的无限制对抗性示例(UAE)，而受限对抗性示例(RAE)是通过将受$l_p$范数约束的扰动添加到观察到的示例来创建的。第二，现有的AT方法往往以牺牲标准泛化能力(即对自然样本的准确性)为代价来实现对抗的健壮性，因为它们在两者之间进行了权衡。为了克服这些挑战，我们提出了一种独特的观点，将UAE理解为潜移默化的未被观察到的例子。此外，我们还发现，这种权衡是由于对抗性例子和自然例子的分布分离造成的。基于这些思想，我们提出了一种新的AT方法，称为可证明的无限制对抗训练(PUAT)，它可以为目标分类器提供对UAE和RAE都具有全面的对抗健壮性，同时提高其标准泛化能力。特别是，PUAT利用部分标记的数据，通过一种新的增强型三层GaN准确地捕获自然数据分布，从而实现有效的UAE生成。同时，PUAT通过在对手损失中引入目标分类器的监督损失来扩展传统的AT，并通过扩展的三层GAN协作实现了UAE分布、自然数据分布和分类器学习的分布之间的对齐。最后，在广泛使用的基准上进行了扎实的理论分析和广泛的实验，证明了PUAT的优越性。



## **24. SUPER-Net: Trustworthy Medical Image Segmentation with Uncertainty Propagation in Encoder-Decoder Networks**

超网：编解码网中具有不确定性传播的可信医学图像分割 eess.IV

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2111.05978v3) [paper-pdf](http://arxiv.org/pdf/2111.05978v3)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Hassan M. Fathallah-Shaykh, Ghulam Rasool

**Abstract**: Deep Learning (DL) holds great promise in reshaping the healthcare industry owing to its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most models produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian DL framework for uncertainty quantification in segmentation neural networks: SUPER-Net: trustworthy medical image Segmentation with Uncertainty Propagation in Encoder-decodeR Networks. SUPER-Net analytically propagates, using Taylor series approximations, the first two moments (mean and covariance) of the posterior distribution of the model parameters across the nonlinear layers. In particular, SUPER-Net simultaneously learns the mean and covariance without expensive post-hoc Monte Carlo sampling or model ensembling. The output consists of two simultaneous maps: the segmented image and its pixelwise uncertainty map, which corresponds to the covariance matrix of the predictive distribution. We conduct an extensive evaluation of SUPER-Net on medical image segmentation of Magnetic Resonances Imaging and Computed Tomography scans under various noisy and adversarial conditions. Our experiments on multiple benchmark datasets demonstrate that SUPER-Net is more robust to noise and adversarial attacks than state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed SUPER-Net associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts, or adversarial attacks. Perhaps more importantly, the model exhibits the ability of self-assessment of its segmentation decisions, notably when making erroneous predictions due to noise or adversarial examples.

摘要: 深度学习(DL)由于其精确度、效率和客观性，在重塑医疗行业方面前景光明。然而，DL模型对噪声和分布外输入的脆性阻碍了它们在临床上的部署。大多数模型在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。介绍了一种新的用于分割神经网络不确定性量化的贝叶斯DL框架：超网：编解码器网络中的具有不确定性传播的可信医学图像分割。利用泰勒级数近似，超网络解析地传播模型参数在非线性层上的后验分布的前两个矩(均值和协方差)。特别是，超级网络同时学习均值和协方差，而不需要昂贵的事后蒙特卡罗采样或模型集成。输出由两个同时映射组成：分割图像及其像素化不确定性映射，其对应于预测分布的协方差矩阵。在各种噪声和对抗性条件下，我们对超网在磁共振成像和计算机断层扫描医学图像分割中的应用进行了广泛的评估。我们在多个基准数据集上的实验表明，与最新的分割模型相比，超级网络对噪声和敌意攻击具有更强的鲁棒性。此外，所提出的超网的不确定性映射将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌对攻击破坏的补丁相关联。也许更重要的是，该模型展示了对其分割决策进行自我评估的能力，特别是当由于噪声或对抗性例子而做出错误预测时。



## **25. Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network**

三维稀疏卷积网络的动态感知对抗攻击 cs.CV

We have improved the quality of this work and updated a new version  to address the limitations of the proposed method

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2112.09428v2) [paper-pdf](http://arxiv.org/pdf/2112.09428v2)

**Authors**: An Tao, Yueqi Duan, He Wang, Ziyi Wu, Pengliang Ji, Haowen Sun, Jie Zhou, Jiwen Lu

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem in deep neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed networks, e.g. 3D sparse convolution network, which contains input-dependent execution to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture changes afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we re-formulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on various datasets show that our LGM achieves impressive performance on semantic segmentation and classification. Compared with the dynamic-unaware methods, LGM achieves about 20% lower mIoU averagely on the ScanNet and S3DIS datasets. LGM also outperforms the recent point cloud attacks.

摘要: 本文研究了深层神经网络中动态感知的敌意攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络体系结构在整个攻击过程中是固定的。然而，这一假设并不适用于最近提出的许多网络，例如3D稀疏卷积网络，它包含依赖输入的执行以提高计算效率。它导致了严重的梯度滞后问题，使得当前步骤的学习攻击由于之后的体系结构变化而无效。为了解决这个问题，我们提出了一种引导梯度法(LGM)，并展示了滞后梯度的显著影响。更具体地说，我们重新制定了梯度以感知网络体系结构的潜在动态变化，以便在网络体系结构动态变化时，学习到的攻击比不感知动态变化的方法更好地“引导”下一步。在各种数据集上的大量实验表明，我们的LGM在语义分割和分类方面取得了令人印象深刻的性能。与动态未知方法相比，LGM在ScanNet和S3DIS数据集上的MIU值平均降低了20%左右。LGM的性能也优于最近的点云攻击。



## **26. Passive Defense Against 3D Adversarial Point Clouds Through the Lens of 3D Steganalysis**

基于3D隐写分析镜头的3D对抗点云被动防御 cs.MM

This paper is out-of-date

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2205.08738v2) [paper-pdf](http://arxiv.org/pdf/2205.08738v2)

**Authors**: Jiahao Zhu

**Abstract**: Nowadays, 3D data plays an indelible role in the computer vision field. However, extensive studies have proved that deep neural networks (DNNs) fed with 3D data, such as point clouds, are susceptible to adversarial examples, which aim to misguide DNNs and might bring immeasurable losses. Currently, 3D adversarial point clouds are chiefly generated in three fashions, i.e., point shifting, point adding, and point dropping. These point manipulations would modify geometrical properties and local correlations of benign point clouds more or less. Motivated by this basic fact, we propose to defend such adversarial examples with the aid of 3D steganalysis techniques. Specifically, we first introduce an adversarial attack and defense model adapted from the celebrated Prisoners' Problem in steganography to help us comprehend 3D adversarial attack and defense more generally. Then we rethink two significant but vague concepts in the field of adversarial example, namely, active defense and passive defense, from the perspective of steganalysis. Most importantly, we design a 3D adversarial point cloud detector through the lens of 3D steganalysis. Our detector is double-blind, that is to say, it does not rely on the exact knowledge of the adversarial attack means and victim models. To enable the detector to effectively detect malicious point clouds, we craft a 64-D discriminant feature set, including features related to first-order and second-order local descriptions of point clouds. To our knowledge, this work is the first to apply 3D steganalysis to 3D adversarial example defense. Extensive experimental results demonstrate that the proposed 3D adversarial point cloud detector can achieve good detection performance on multiple types of 3D adversarial point clouds.

摘要: 如今，3D数据在计算机视觉领域发挥着不可磨灭的作用。然而，大量的研究已经证明，以点云等三维数据为基础的深度神经网络(DNN)很容易受到敌意例子的影响，这些例子旨在误导DNN，并可能带来不可估量的损失。目前，三维对抗性点云的生成主要有三种方式，即点移位、点加点和点删除。这些点操作会或多或少地改变良性点云的几何性质和局部相关性。在这一基本事实的推动下，我们建议借助3D隐写分析技术来为这些对抗性例子进行辩护。具体地说，我们首先介绍了一种改编自隐写术中著名囚犯问题的对抗性攻防模型，以帮助我们更全面地理解3D对抗性攻防。然后，我们从隐写分析的角度重新思考了对抗性例证领域中两个重要而模糊的概念，即主动防御和被动防御。最重要的是，我们通过3D隐写分析的镜头设计了一个3D对抗点云检测器。我们的检测器是双盲的，也就是说，它不依赖于对抗性攻击手段和受害者模型的准确知识。为了使检测器能够有效地检测恶意点云，我们构造了一个64维判别特征集，包括与点云的一阶和二阶局部描述相关的特征。据我们所知，这项工作是首次将3D隐写分析应用于3D对抗实例防御。大量的实验结果表明，本文提出的三维对抗性点云检测器对多种类型的三维对抗性点云具有较好的检测性能。



## **27. How Potent are Evasion Attacks for Poisoning Federated Learning-Based Signal Classifiers?**

对于毒化基于联合学习的信号分类器，规避攻击的效力有多大？ eess.SP

6 pages, Accepted to IEEE ICC 2023

**SubmitDate**: 2023-01-21    [abs](http://arxiv.org/abs/2301.08866v1) [paper-pdf](http://arxiv.org/pdf/2301.08866v1)

**Authors**: Su Wang, Rajeev Sahay, Christopher G. Brinton

**Abstract**: There has been recent interest in leveraging federated learning (FL) for radio signal classification tasks. In FL, model parameters are periodically communicated from participating devices, training on their own local datasets, to a central server which aggregates them into a global model. While FL has privacy/security advantages due to raw data not leaving the devices, it is still susceptible to several adversarial attacks. In this work, we reveal the susceptibility of FL-based signal classifiers to model poisoning attacks, which compromise the training process despite not observing data transmissions. In this capacity, we develop an attack framework in which compromised FL devices perturb their local datasets using adversarial evasion attacks. As a result, the training process of the global model significantly degrades on in-distribution signals (i.e., signals received over channels with identical distributions at each edge device). We compare our work to previously proposed FL attacks and reveal that as few as one adversarial device operating with a low-powered perturbation under our attack framework can induce the potent model poisoning attack to the global classifier. Moreover, we find that more devices partaking in adversarial poisoning will proportionally degrade the classification performance.

摘要: 最近，人们对利用联合学习(FL)进行无线电信号分类任务感兴趣。在FL中，模型参数从参与设备周期性地传送到中央服务器，在它们自己的本地数据集上进行训练，中央服务器将它们聚集成全局模型。虽然FL由于原始数据不会离开设备而具有隐私/安全优势，但它仍然容易受到几种对手攻击。在这项工作中，我们揭示了基于FL的信号分类器对中毒攻击建模的敏感性，尽管没有观察到数据传输，但这会影响训练过程。在这种能力下，我们开发了一个攻击框架，在该框架中，受攻击的FL设备使用对抗性逃避攻击来扰乱其本地数据集。结果，全局模型的训练过程在分布内信号(即，在每个边缘设备处具有相同分布的信道上接收的信号)上显著降级。我们将我们的工作与以前提出的FL攻击进行了比较，发现在我们的攻击框架下，只要一个敌方设备在低功率扰动下操作就可以诱导出对全局分类器的强有力的模型中毒攻击。此外，我们还发现，参与对抗性中毒的设备越多，分类性能就越差。



## **28. Robot Skill Learning Via Classical Robotics-Based Generated Datasets: Advantages, Disadvantages, and Future Improvement**

基于经典机器人生成数据集的机器人技能学习：优势、劣势和未来改进 cs.RO

**SubmitDate**: 2023-01-20    [abs](http://arxiv.org/abs/2301.08794v1) [paper-pdf](http://arxiv.org/pdf/2301.08794v1)

**Authors**: Batu Kaan Oezen

**Abstract**: Why do we not profit from our long-existing classical robotics knowledge and look for some alternative way for data collection? The situation ignoring all existing methods might be such a waste. This article argues that a dataset created using a classical robotics algorithm is a crucial part of future development. This developed classic algorithm has a perfect domain adaptation and generalization property, and most importantly, collecting datasets based on them is quite easy. It is well known that current robot skill-learning approaches perform exceptionally badly in the unseen domain, and their performance against adversarial attacks is quite limited as long as they do not have a very exclusive big dataset. Our experiment is the initial steps of using a dataset created by classical robotics codes. Our experiment investigated possible trajectory collection based on classical robotics. It addressed some advantages and disadvantages and pointed out other future development ideas.

摘要: 为什么我们不从我们长期存在的经典机器人知识中获益，寻找一些替代的数据收集方式呢？忽视所有现有方法的情况可能是这样一种浪费。本文认为，使用经典机器人算法创建的数据集是未来开发的关键部分。这种改进的经典算法具有良好的领域适应性和泛化能力，最重要的是，基于它们的数据集的收集非常容易。众所周知，目前的机器人技能学习方法在看不见的领域表现得非常糟糕，只要它们没有非常独特的大数据集，它们对抗对手攻击的性能就相当有限。我们的实验是使用由经典机器人代码创建的数据集的初始步骤。我们的实验研究了基于经典机器人的可能的轨迹收集。它指出了一些优点和缺点，并指出了未来的其他发展思路。



## **29. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

StratDef：基于ML的恶意软件检测中对抗攻击的战略防御 cs.LG

**SubmitDate**: 2023-01-20    [abs](http://arxiv.org/abs/2202.07568v4) [paper-pdf](http://arxiv.org/pdf/2202.07568v4)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了一种基于移动目标防御方法的战略防御系统StratDef。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，在现有的防御系统中，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **30. On the Relationship Between Information-Theoretic Privacy Metrics And Probabilistic Information Privacy**

论信息论隐私度量与概率信息隐私的关系 cs.IT

**SubmitDate**: 2023-01-20    [abs](http://arxiv.org/abs/2301.08401v1) [paper-pdf](http://arxiv.org/pdf/2301.08401v1)

**Authors**: Chong Xiao Wang, Wee Peng Tay

**Abstract**: Information-theoretic (IT) measures based on $f$-divergences have recently gained interest as a measure of privacy leakage as they allow for trading off privacy against utility using only a single-value characterization. However, their operational interpretations in the privacy context are unclear. In this paper, we relate the notion of probabilistic information privacy (IP) to several IT privacy metrics based on $f$-divergences. We interpret probabilistic IP under both the detection and estimation frameworks and link it to differential privacy, thus allowing a precise operational interpretation of these IT privacy metrics. We show that the $\chi^2$-divergence privacy metric is stronger than those based on total variation distance and Kullback-Leibler divergence. Therefore, we further develop a data-driven empirical risk framework based on the $\chi^2$-divergence privacy metric and realized using deep neural networks. This framework is agnostic to the adversarial attack model. Empirical experiments demonstrate the efficacy of our approach.

摘要: 基于$f$-分歧的信息理论(IT)措施最近受到了人们的关注，作为隐私泄露的衡量标准，因为它们允许仅使用单一价值描述来权衡隐私和效用。然而，他们在隐私背景下的操作解释并不清楚。在本文中，我们将概率信息隐私(IP)的概念与几种基于$f$-离散度的IT隐私度量联系起来。我们在检测和评估框架下解释概率知识产权，并将其链接到不同的隐私，从而允许对这些IT隐私指标进行精确的操作解释。我们证明了$X^2-散度隐私度量比基于全变差距离和Kullback-Leibler散度的隐私度量更强。因此，我们进一步发展了一个数据驱动的经验风险框架，该框架基于散度隐私度量，并使用深度神经网络实现。该框架与对抗性攻击模型无关。实证实验证明了该方法的有效性。



## **31. BO-DBA: Query-Efficient Decision-Based Adversarial Attacks via Bayesian Optimization**

BO-DBA：基于贝叶斯优化的高效查询决策对抗攻击 cs.LG

**SubmitDate**: 2023-01-19    [abs](http://arxiv.org/abs/2106.02732v2) [paper-pdf](http://arxiv.org/pdf/2106.02732v2)

**Authors**: Zhuosheng Zhang, Shucheng Yu

**Abstract**: Decision-based attacks (DBA), wherein attackers perturb inputs to spoof learning algorithms by observing solely the output labels, are a type of severe adversarial attacks against Deep Neural Networks (DNNs) requiring minimal knowledge of attackers. State-of-the-art DBA attacks relying on zeroth-order gradient estimation require an excessive number of queries. Recently, Bayesian optimization (BO) has shown promising in reducing the number of queries in score-based attacks (SBA), in which attackers need to observe real-valued probability scores as outputs. However, extending BO to the setting of DBA is nontrivial because in DBA only output labels instead of real-valued scores, as needed by BO, are available to attackers. In this paper, we close this gap by proposing an efficient DBA attack, namely BO-DBA. Different from existing approaches, BO-DBA generates adversarial examples by searching so-called \emph{directions of perturbations}. It then formulates the problem as a BO problem that minimizes the real-valued distortion of perturbations. With the optimized perturbation generation process, BO-DBA converges much faster than the state-of-the-art DBA techniques. Experimental results on pre-trained ImageNet classifiers show that BO-DBA converges within 200 queries while the state-of-the-art DBA techniques need over 15,000 queries to achieve the same level of perturbation distortion. BO-DBA also shows similar attack success rates even as compared to BO-based SBA attacks but with less distortion.

摘要: 基于决策的攻击(DBA)是一种针对深度神经网络(DNN)的严重对抗性攻击，攻击者只需了解最少的攻击者知识，即可通过只观察输出标签来干扰输入以欺骗学习算法。依赖零阶梯度估计的最先进的DBA攻击需要过多的查询。最近，贝叶斯优化(BO)在减少基于分数的攻击(SBA)中的查询数量方面显示出了良好的前景，在SBA中，攻击者需要观察实值概率分数作为输出。然而，将BO扩展到DBA的设置并不容易，因为在DBA中，攻击者只能使用输出标签，而不是BO所需的实值分数。在本文中，我们通过提出一种有效的DBA攻击，即BO-DBA来弥补这一差距。与现有方法不同，BO-DBA通过搜索所谓的\emph(扰动方向)来生成对抗性实例。然后将该问题表示为使扰动的实值失真最小化的BO问题。通过优化的扰动生成过程，BO-DBA的收敛速度比最先进的DBA技术快得多。在预先训练的ImageNet分类器上的实验结果表明，BO-DBA在200个查询内收敛，而最先进的DBA技术需要超过15,000个查询才能达到相同程度的扰动失真。与基于BO的SBA攻击相比，BO-DBA也显示出相似的攻击成功率，但失真更小。



## **32. RNAS-CL: Robust Neural Architecture Search by Cross-Layer Knowledge Distillation**

RNAS-CL：基于跨层知识提取的稳健神经结构搜索 cs.CV

17 pages, 12 figures

**SubmitDate**: 2023-01-19    [abs](http://arxiv.org/abs/2301.08092v1) [paper-pdf](http://arxiv.org/pdf/2301.08092v1)

**Authors**: Utkarsh Nath, Yancheng Wang, Yingzhen Yang

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Neural Architecture Search (NAS), one of the driving tools of deep neural networks, demonstrates superior performance in prediction accuracy in various machine learning applications. However, it is unclear how it performs against adversarial attacks. Given the presence of a robust teacher, it would be interesting to investigate if NAS would produce robust neural architecture by inheriting robustness from the teacher. In this paper, we propose Robust Neural Architecture Search by Cross-Layer Knowledge Distillation (RNAS-CL), a novel NAS algorithm that improves the robustness of NAS by learning from a robust teacher through cross-layer knowledge distillation. Unlike previous knowledge distillation methods that encourage close student/teacher output only in the last layer, RNAS-CL automatically searches for the best teacher layer to supervise each student layer. Experimental result evidences the effectiveness of RNAS-CL and shows that RNAS-CL produces small and robust neural architecture.

摘要: 深度神经网络很容易受到敌意攻击。神经结构搜索(NAS)是深度神经网络的驱动工具之一，在各种机器学习应用中表现出了优异的预测精度。然而，目前尚不清楚它在对抗对手攻击时的表现如何。鉴于有一位健壮的老师在场，研究NAS是否会通过继承老师的健壮性来产生健壮的神经架构将是一件有趣的事情。本文提出了一种基于跨层知识蒸馏的稳健神经结构搜索算法(RNAS-CL)，该算法通过跨层知识蒸馏向健壮的教师学习，从而提高了NAS的稳健性。不同于以往的知识提炼方法，RNAS-CL只在最后一层鼓励接近的学生/教师输出，RNAS-CL自动搜索最好的教师层来监督每一层学生。实验结果证明了RNAS-CL的有效性，并表明RNAS-CL产生了小而健壮的神经结构。



## **33. Evaluating the Robustness of Trigger Set-Based Watermarks Embedded in Deep Neural Networks**

深度神经网络中基于触发集的水印稳健性评估 cs.CR

15 pages, accepted at IEEE TDSC

**SubmitDate**: 2023-01-19    [abs](http://arxiv.org/abs/2106.10147v2) [paper-pdf](http://arxiv.org/pdf/2106.10147v2)

**Authors**: Suyoung Lee, Wonho Song, Suman Jana, Meeyoung Cha, Sooel Son

**Abstract**: Trigger set-based watermarking schemes have gained emerging attention as they provide a means to prove ownership for deep neural network model owners. In this paper, we argue that state-of-the-art trigger set-based watermarking algorithms do not achieve their designed goal of proving ownership. We posit that this impaired capability stems from two common experimental flaws that the existing research practice has committed when evaluating the robustness of watermarking algorithms: (1) incomplete adversarial evaluation and (2) overlooked adaptive attacks. We conduct a comprehensive adversarial evaluation of 11 representative watermarking schemes against six of the existing attacks and demonstrate that each of these watermarking schemes lacks robustness against at least two non-adaptive attacks. We also propose novel adaptive attacks that harness the adversary's knowledge of the underlying watermarking algorithm of a target model. We demonstrate that the proposed attacks effectively break all of the 11 watermarking schemes, consequently allowing adversaries to obscure the ownership of any watermarked model. We encourage follow-up studies to consider our guidelines when evaluating the robustness of their watermarking schemes via conducting comprehensive adversarial evaluation that includes our adaptive attacks to demonstrate a meaningful upper bound of watermark robustness.

摘要: 基于触发器集的水印方案因其为深度神经网络模型拥有者提供了一种证明所有权的手段而受到越来越多的关注。在这篇文章中，我们认为最新的基于触发式集合的水印算法并没有达到他们设计的证明所有权的目标。我们假设这种能力受损源于现有研究实践在评估水印算法的稳健性时所犯的两个常见的实验缺陷：(1)不完全的对抗性评估和(2)忽略自适应攻击。针对现有的6种攻击，我们对11种代表性的水印方案进行了全面的对抗性评估，证明了每种方案对至少两种非自适应攻击都缺乏稳健性。我们还提出了新的自适应攻击，利用对手对目标模型的底层水印算法的了解。我们证明了所提出的攻击有效地破坏了全部11个水印方案，从而允许攻击者隐藏任何水印模型的所有权。我们鼓励后续研究在评估他们的水印方案的稳健性时考虑我们的指导方针，通过进行全面的对抗性评估，包括我们的自适应攻击来证明水印稳健性的有意义的上界。



## **34. Exposing Fine-Grained Adversarial Vulnerability of Face Anti-Spoofing Models**

暴露Face反欺骗模型的细粒度攻击漏洞 cs.CV

**SubmitDate**: 2023-01-18    [abs](http://arxiv.org/abs/2205.14851v2) [paper-pdf](http://arxiv.org/pdf/2205.14851v2)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Face anti-spoofing aims to discriminate the spoofing face images (e.g., printed photos) from live ones. However, adversarial examples greatly challenge its credibility, where adding some perturbation noise can easily change the predictions. Previous works conducted adversarial attack methods to evaluate the face anti-spoofing performance without any fine-grained analysis that which model architecture or auxiliary feature is vulnerable to the adversary. To handle this problem, we propose a novel framework to expose the fine-grained adversarial vulnerability of the face anti-spoofing models, which consists of a multitask module and a semantic feature augmentation (SFA) module. The multitask module can obtain different semantic features for further evaluation, but only attacking these semantic features fails to reflect the discrimination-related vulnerability. We then design the SFA module to introduce the data distribution prior for more discrimination-related gradient directions for generating adversarial examples. Comprehensive experiments show that SFA module increases the attack success rate by nearly 40$\%$ on average. We conduct this fine-grained adversarial analysis on different annotations, geometric maps, and backbone networks (e.g., Resnet network). These fine-grained adversarial examples can be used for selecting robust backbone networks and auxiliary features. They also can be used for adversarial training, which makes it practical to further improve the accuracy and robustness of the face anti-spoofing models.

摘要: 人脸反欺骗的目的是区分伪造的人脸图像(如打印的照片)和活的人脸图像。然而，对抗性的例子极大地挑战了它的可信度，在那里添加一些扰动噪声很容易改变预测。以往的工作采用对抗性攻击的方法来评估人脸的反欺骗性能，没有任何细粒度的分析来确定哪个模型、架构或辅助特征容易受到对手的攻击。为了解决这个问题，我们提出了一种新的框架来暴露人脸反欺骗模型的细粒度攻击漏洞，该框架由多任务模块和语义特征增强(SFA)模块组成。多任务模块可以获得不同的语义特征用于进一步的评估，但仅攻击这些语义特征并不能反映与歧视相关的脆弱性。然后，我们设计了SFA模块来引入数据分布，以获得更多与区分相关的梯度方向，以生成对抗性示例。综合实验表明，SFA模块的攻击成功率平均提高了近40美元。我们在不同的注释、几何地图和骨干网络(例如RESNET网络)上进行了这种细粒度的对抗性分析。这些细粒度的对抗性实例可用于选择健壮的主干网络和辅助特征。它们还可以用于对抗性训练，从而进一步提高人脸反欺骗模型的准确性和稳健性。



## **35. Generative Adversarial Networks to infer velocity components in rotating turbulent flows**

生成对抗性网络用于推断旋转湍流中的速度分量 physics.flu-dyn

**SubmitDate**: 2023-01-18    [abs](http://arxiv.org/abs/2301.07541v1) [paper-pdf](http://arxiv.org/pdf/2301.07541v1)

**Authors**: Tianyi Li, Michele Buzzicotti, Luca Biferale, Fabio Bonaccorso

**Abstract**: Inference problems for two-dimensional snapshots of rotating turbulent flows are studied. We perform a systematic quantitative benchmark of point-wise and statistical reconstruction capabilities of the linear Extended Proper Orthogonal Decomposition (EPOD) method, a non-linear Convolutional Neural Network (CNN) and a Generative Adversarial Network (GAN). We attack the important task of inferring one velocity component out of the measurement of a second one, and two cases are studied: (I) both components lay in the plane orthogonal to the rotation axis and (II) one of the two is parallel to the rotation axis. We show that EPOD method works well only for the former case where both components are strongly correlated, while CNN and GAN always outperform EPOD both concerning point-wise and statistical reconstructions. For case (II), when the input and output data are weakly correlated, all methods fail to reconstruct faithfully the point-wise information. In this case, only GAN is able to reconstruct the field in a statistical sense. The analysis is performed using both standard validation tools based on L2 spatial distance between the prediction and the ground truth and more sophisticated multi-scale analysis using wavelet decomposition. Statistical validation is based on standard Jensen-Shannon divergence between the probability density functions, spectral properties and multi-scale flatness.

摘要: 研究了旋转湍流二维快照的推断问题。我们对线性扩展本征正交分解(EPOD)方法、非线性卷积神经网络(CNN)和生成性对抗网络(GAN)的逐点重建和统计重建能力进行了系统的定量基准测试。我们提出了从第二个速度分量的测量中推断出一个速度分量的重要任务，并研究了两种情况：(I)两个分量都位于与旋转轴垂直的平面上；(Ii)两个分量中的一个平行于旋转轴。我们发现，EPOD方法只适用于强相关的前一种情况，而CNN和GAN在逐点重建和统计重建方面总是优于EPOD。对于情况(II)，当输入和输出数据弱相关时，所有方法都不能忠实地重建逐点信息。在这种情况下，只有GaN能够在统计意义上重建场。分析既使用了基于预测与地面真实之间的L2空间距离的标准验证工具，也使用了使用小波分解的更复杂的多尺度分析。统计验证基于概率密度函数、光谱特性和多尺度平坦度之间的标准Jensen-Shannon散度。



## **36. Accurate Detection of Paroxysmal Atrial Fibrillation with Certified-GAN and Neural Architecture Search**

GAN认证和神经结构搜索对阵发性房颤的准确检测 cs.LG

19 pages

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.10173v1) [paper-pdf](http://arxiv.org/pdf/2301.10173v1)

**Authors**: Mehdi Asadi, Fatemeh Poursalim, Mohammad Loni, Masoud Daneshtalab, Mikael Sjödin, Arash Gharehbaghi

**Abstract**: This paper presents a novel machine learning framework for detecting Paroxysmal Atrial Fibrillation (PxAF), a pathological characteristic of Electrocardiogram (ECG) that can lead to fatal conditions such as heart attack. To enhance the learning process, the framework involves a Generative Adversarial Network (GAN) along with a Neural Architecture Search (NAS) in the data preparation and classifier optimization phases. The GAN is innovatively invoked to overcome the class imbalance of the training data by producing the synthetic ECG for PxAF class in a certified manner. The effect of the certified GAN is statistically validated. Instead of using a general-purpose classifier, the NAS automatically designs a highly accurate convolutional neural network architecture customized for the PxAF classification task. Experimental results show that the accuracy of the proposed framework exhibits a high value of 99% which not only enhances state-of-the-art by up to 5.1%, but also improves the classification performance of the two widely-accepted baseline methods, ResNet-18, and Auto-Sklearn, by 2.2% and 6.1%.

摘要: 提出了一种新的用于检测阵发性房颤(PxAF)的机器学习框架。阵发性房颤是心电图的一种病理特征，可导致心脏病发作等致命疾病。为了增强学习过程，该框架在数据准备和分类器优化阶段涉及生成性对抗网络(GAN)以及神经结构搜索(NAS)。通过以认证的方式产生PxAF类的合成心电，创新性地调用GAN来克服训练数据的类不平衡。对认证的GaN的效果进行了统计验证。NAS不使用通用分类器，而是自动设计为PxAF分类任务定制的高精度卷积神经网络结构。实验结果表明，该框架的正确率高达99%，不仅使最新的分类正确率提高了5.1%，而且将两种被广泛接受的基线分类方法ResNet-18和Auto-SkLearning的分类性能分别提高了2.2%和6.1%。



## **37. Denoising Diffusion Probabilistic Models as a Defense against Adversarial Attacks**

抗敌意攻击的扩散概率模型去噪 cs.LG

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.06871v1) [paper-pdf](http://arxiv.org/pdf/2301.06871v1)

**Authors**: Lars Lien Ankile, Anna Midgley, Sebastian Weisshaar

**Abstract**: Neural Networks are infamously sensitive to small perturbations in their inputs, making them vulnerable to adversarial attacks. This project evaluates the performance of Denoising Diffusion Probabilistic Models (DDPM) as a purification technique to defend against adversarial attacks. This works by adding noise to an adversarial example before removing it through the reverse process of the diffusion model. We evaluate the approach on the PatchCamelyon data set for histopathologic scans of lymph node sections and find an improvement of the robust accuracy by up to 88\% of the original model's accuracy, constituting a considerable improvement over the vanilla model and our baselines. The project code is located at https://github.com/ankile/Adversarial-Diffusion.

摘要: 神经网络对其输入中的微小扰动非常敏感，这使得它们很容易受到对手的攻击。该项目评估了去噪扩散概率模型(DDPM)作为一种防御对手攻击的净化技术的性能。这种方法的工作原理是，在通过扩散模型的反向过程将其移除之前，将噪声添加到敌对示例中。我们在PatchCamelyon数据集上对该方法进行了评估，发现该方法的稳健准确率比原始模型的精度提高了88%，这比Vanilla模型和我们的基线有了很大的改善。项目代码位于https://github.com/ankile/Adversarial-Diffusion.



## **38. Database Matching Under Noisy Synchronization Errors**

噪声同步误差下的数据库匹配 cs.IT

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.06796v1) [paper-pdf](http://arxiv.org/pdf/2301.06796v1)

**Authors**: Serhat Bakirtas, Elza Erkip

**Abstract**: The re-identification or de-anonymization of users from anonymized data through matching with publicly-available correlated user data has raised privacy concerns, leading to the complementary measure of obfuscation in addition to anonymization. Recent research provides a fundamental understanding of the conditions under which privacy attacks, in the form of database matching, are successful in the presence of obfuscation. Motivated by synchronization errors stemming from the sampling of time-indexed databases, this paper presents a unified framework considering both obfuscation and synchronization errors and investigates the matching of databases under noisy entry repetitions. By investigating different structures for the repetition pattern, replica detection and seeded deletion detection algorithms are devised and sufficient and necessary conditions for successful matching are derived. Finally, the impacts of some variations of the underlying assumptions, such as adversarial deletion model, seedless database matching and zero-rate regime, on the results are discussed. Overall, our results provide insights into the privacy-preserving publication of anonymized and obfuscated time-indexed data as well as the closely-related problem of the capacity of synchronization channels.

摘要: 通过与公开可用的相关用户数据进行匹配，从匿名数据中重新识别用户或取消匿名化，引起了对隐私的担忧，导致了除了匿名化之外的混淆的补充措施。最近的研究提供了一个基本的理解，即在存在混淆的情况下，以数据库匹配的形式进行的隐私攻击是成功的。从时间索引数据库采样产生的同步误差出发，提出了一种同时考虑混淆和同步误差的统一框架，并研究了噪声条目重复情况下的数据库匹配问题。通过研究重复模式的不同结构，设计了副本检测和种子删除检测算法，得到了匹配成功的充要条件。最后，讨论了对抗性删除模型、无种子数据库匹配和零速率机制等基本假设的变化对结果的影响。总体而言，我们的结果为匿名和混淆时间索引数据的隐私保护发布以及与同步通道容量密切相关的问题提供了见解。



## **39. Adversarial AI in Insurance: Pervasiveness and Resilience**

保险业中的对抗性人工智能：普及性和韧性 cs.LG

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2301.07520v1) [paper-pdf](http://arxiv.org/pdf/2301.07520v1)

**Authors**: Elisa Luciano, Matteo Cattaneo, Ron Kenett

**Abstract**: The rapid and dynamic pace of Artificial Intelligence (AI) and Machine Learning (ML) is revolutionizing the insurance sector. AI offers significant, very much welcome advantages to insurance companies, and is fundamental to their customer-centricity strategy. It also poses challenges, in the project and implementation phase. Among those, we study Adversarial Attacks, which consist of the creation of modified input data to deceive an AI system and produce false outputs. We provide examples of attacks on insurance AI applications, categorize them, and argue on defence methods and precautionary systems, considering that they can involve few-shot and zero-shot multilabelling. A related topic, with growing interest, is the validation and verification of systems incorporating AI and ML components. These topics are discussed in various sections of this paper.

摘要: 人工智能(AI)和机器学习(ML)的快速而动态的步伐正在给保险行业带来革命性的变化。人工智能为保险公司提供了重要的、非常受欢迎的优势，是其以客户为中心的战略的基础。在项目和实施阶段，这也带来了挑战。其中，我们研究了对抗性攻击，它包括创建修改的输入数据来欺骗AI系统并产生虚假输出。我们提供了针对保险人工智能应用程序的攻击示例，对它们进行了分类，并就防御方法和预防系统进行了辩论，考虑到它们可能涉及少发和零发的多标签。一个相关的话题日益引起人们的兴趣，那就是对包含AI和ML组件的系统进行确认和验证。这些主题将在本白皮书的各个部分中进行讨论。



## **40. Imperceptible Adversarial Attack via Invertible Neural Networks**

基于逆神经网络的潜伏性敌意攻击 cs.CV

**SubmitDate**: 2023-01-17    [abs](http://arxiv.org/abs/2211.15030v3) [paper-pdf](http://arxiv.org/pdf/2211.15030v3)

**Authors**: Zihan Chen, Ziyue Wang, Junjie Huang, Wentao Zhao, Xiao Liu, Dejian Guan

**Abstract**: Adding perturbations via utilizing auxiliary gradient information or discarding existing details of the benign images are two common approaches for generating adversarial examples. Though visual imperceptibility is the desired property of adversarial examples, conventional adversarial attacks still generate traceable adversarial perturbations. In this paper, we introduce a novel Adversarial Attack via Invertible Neural Networks (AdvINN) method to produce robust and imperceptible adversarial examples. Specifically, AdvINN fully takes advantage of the information preservation property of Invertible Neural Networks and thereby generates adversarial examples by simultaneously adding class-specific semantic information of the target class and dropping discriminant information of the original class. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that the proposed AdvINN method can produce less imperceptible adversarial images than the state-of-the-art methods and AdvINN yields more robust adversarial examples with high confidence compared to other adversarial attacks.

摘要: 通过利用辅助梯度信息添加扰动或丢弃良性图像的现有细节是生成对抗性示例的两种常见方法。虽然视觉不可感知性是对抗性例子的理想属性，但传统的对抗性攻击仍然产生可追踪的对抗性扰动。在本文中，我们介绍了一种新的基于可逆神经网络(AdvINN)的对抗性攻击方法，以产生健壮且不可察觉的对抗性示例。具体而言，AdvINN充分利用了可逆神经网络的信息保持性，通过同时添加目标类的类特定语义信息和丢弃原类的判别信息来生成对抗性实例。在CIFAR-10、CIFAR-100和ImageNet-1K上的大量实验表明，所提出的AdvINN方法可以产生比现有方法更少的不可察觉的对抗性图像，并且与其他对抗性攻击相比，AdvINN产生更健壮的对抗性例子和更高的置信度。



## **41. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

私人眼睛：视频会议中通过眼镜反射窥视文本屏幕的限度 cs.CR

**SubmitDate**: 2023-01-16    [abs](http://arxiv.org/abs/2205.03971v3) [paper-pdf](http://arxiv.org/pdf/2205.03971v3)

**Authors**: Yan Long, Chen Yan, Shilin Xiao, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstract**: Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual and graphical information gleaming from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our models and experimental results in a controlled lab setting show it is possible to reconstruct and recognize with over 75% accuracy on-screen texts that have heights as small as 10 mm with a 720p webcam. We further apply this threat model to web textual contents with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution towards 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Besides textual targets, a case study on recognizing a closed-world dataset of Alexa top 100 websites with 720p webcams shows a maximum recognition accuracy of 94% with 10 participants even without using machine-learning models. Our research proposes near-term mitigations including a software prototype that users can use to blur the eyeglass areas of their video streams. For possible long-term defenses, we advocate an individual reflection testing procedure to assess threats under various settings, and justify the importance of following the principle of least privilege for privacy-sensitive scenarios.

摘要: 通过数学建模和人体实验，这项研究探索了新兴的网络摄像头可能在多大程度上泄露从网络摄像头捕获的眼镜反射中闪烁的可识别的文本和图形信息。我们工作的主要目标是测量、计算和预测随着未来网络摄像头技术的发展而产生的可识别性的因素、限制和阈值。我们的工作利用视频帧序列上的多帧超分辨率技术，探索和表征了基于光学攻击的可行威胁模型。我们的模型和在受控实验室环境下的实验结果表明，使用720p网络摄像头可以重建和识别高度低至10 mm的屏幕文本，准确率超过75%。我们进一步将该威胁模型应用于具有不同攻击者能力的Web文本内容，以找出文本变得可识别的阈值。我们对20名参与者的用户研究表明，目前的720p网络摄像头足以让对手在大字体网站上重建文本内容。我们的模型进一步表明，向4K摄像头的演变将使文本泄漏的门槛倾斜到重建流行网站上的大多数标题文本。除了文本目标，在使用720P网络摄像头识别Alexa前100名网站的封闭世界数据集上的案例研究显示，即使不使用机器学习模型，在10个参与者的情况下，最高识别准确率也达到94%。我们的研究提出了近期的缓解措施，包括一个软件原型，用户可以使用它来模糊他们视频流的眼镜区域。对于可能的长期防御，我们主张使用个人反射测试程序来评估各种环境下的威胁，并证明在隐私敏感场景中遵循最小特权原则的重要性。



## **42. Defending Backdoor Attacks on Vision Transformer via Patch Processing**

利用补丁处理防御视觉转换器的后门攻击 cs.CV

**SubmitDate**: 2023-01-16    [abs](http://arxiv.org/abs/2206.12381v2) [paper-pdf](http://arxiv.org/pdf/2206.12381v2)

**Authors**: Khoa D. Doan, Yingjie Lao, Peng Yang, Ping Li

**Abstract**: Vision Transformers (ViTs) have a radically different architecture with significantly less inductive bias than Convolutional Neural Networks. Along with the improvement in performance, security and robustness of ViTs are also of great importance to study. In contrast to many recent works that exploit the robustness of ViTs against adversarial examples, this paper investigates a representative causative attack, i.e., backdoor. We first examine the vulnerability of ViTs against various backdoor attacks and find that ViTs are also quite vulnerable to existing attacks. However, we observe that the clean-data accuracy and backdoor attack success rate of ViTs respond distinctively to patch transformations before the positional encoding. Then, based on this finding, we propose an effective method for ViTs to defend both patch-based and blending-based trigger backdoor attacks via patch processing. The performances are evaluated on several benchmark datasets, including CIFAR10, GTSRB, and TinyImageNet, which show the proposed novel defense is very successful in mitigating backdoor attacks for ViTs. To the best of our knowledge, this paper presents the first defensive strategy that utilizes a unique characteristic of ViTs against backdoor attacks.   The paper will appear in the Proceedings of the AAAI'23 Conference. This work was initially submitted in November 2021 to CVPR'22, then it was re-submitted to ECCV'22. The paper was made public in June 2022. The authors sincerely thank all the referees from the Program Committees of CVPR'22, ECCV'22, and AAAI'23.

摘要: 与卷积神经网络相比，视觉转换器(VITS)具有完全不同的体系结构，具有明显更少的感应偏差。随着性能的提高，VITS的安全性和健壮性也具有重要的研究意义。与最近许多利用VITS对敌意例子的健壮性的工作不同，本文研究了一种典型的致因攻击，即后门攻击。我们首先检查VITS对各种后门攻击的脆弱性，发现VITS也很容易受到现有攻击的攻击。然而，我们观察到VITS的干净数据准确性和后门攻击成功率对位置编码之前的补丁变换有明显的响应。然后，基于这一发现，我们提出了一种VITS通过补丁处理来防御基于补丁和基于混合的触发后门攻击的有效方法。在包括CIFAR10、GTSRB和TinyImageNet在内的几个基准数据集上进行了性能评估，表明所提出的新型防御在缓解VITS后门攻击方面是非常成功的。据我们所知，本文提出了第一种利用VITS的独特特性来抵御后门攻击的防御策略。这篇论文将发表在AAAI‘23会议论文集上。这项工作最初于2021年11月提交给CVPR‘22，然后再次提交给ECCV’22。这篇论文于2022年6月发表。作者衷心感谢来自CVPR‘22、ECCV’22和AAAI‘23项目委员会的所有裁判。



## **43. Meta Generative Attack on Person Reidentification**

基于元生成攻击的人的再识别 cs.CV

**SubmitDate**: 2023-01-16    [abs](http://arxiv.org/abs/2301.06286v1) [paper-pdf](http://arxiv.org/pdf/2301.06286v1)

**Authors**: A V Subramanyam

**Abstract**: Adversarial attacks have been recently investigated in person re-identification. These attacks perform well under cross dataset or cross model setting. However, the challenges present in cross-dataset cross-model scenario does not allow these models to achieve similar accuracy. To this end, we propose our method with the goal of achieving better transferability against different models and across datasets. We generate a mask to obtain better performance across models and use meta learning to boost the generalizability in the challenging cross-dataset cross-model setting. Experiments on Market-1501, DukeMTMC-reID and MSMT-17 demonstrate favorable results compared to other attacks.

摘要: 最近，对抗性攻击已被调查在个人重新识别。这些攻击在跨数据集或跨模型设置下表现良好。然而，跨数据集跨模型场景中存在的挑战不允许这些模型达到类似的精度。为此，我们提出了我们的方法，目标是实现更好的对不同模型和跨数据集的可移植性。我们生成一个掩码来获得更好的跨模型性能，并使用元学习来提高在具有挑战性的跨数据集跨模型设置中的泛化能力。在Market-1501、DukeMTMC-Reid和mSMT-17上的实验表明，与其他攻击相比，攻击效果更好。



## **44. A Search-Based Testing Approach for Deep Reinforcement Learning Agents**

一种基于搜索的深度强化学习代理测试方法 cs.SE

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2206.07813v2) [paper-pdf](http://arxiv.org/pdf/2206.07813v2)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstract**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on Deep-Q-Learning agents which are widely used as benchmarks and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.

摘要: 在过去的十年中，深度强化学习(DRL)算法被越来越多地用于解决各种决策问题，如自动驾驶和机器人技术。然而，当这些算法部署在安全关键环境中时，它们面临着巨大的挑战，因为它们经常表现出可能导致潜在关键错误的错误行为。评估DRL代理安全性的一种方法是对其进行测试，以检测在其执行期间可能导致严重故障的故障。这提出了一个问题，即我们如何有效地测试DRL政策，以确保它们的正确性和对安全要求的遵守。大多数现有的测试DRL代理的工作使用对抗性攻击，扰乱代理的状态或动作。然而，这样的攻击往往会导致不切实际的环境状况。他们的主要目标是测试DRL代理的健壮性，而不是测试代理策略与需求的符合性。由于DRL环境的状态空间巨大、测试执行成本高以及DRL算法的黑箱性质，对DRL代理进行穷举测试是不可能的。在本文中，我们提出了一种基于搜索的强化学习代理测试方法(STARLA)，通过在有限的测试预算内有效地搜索代理的失败执行来测试DRL代理的策略。我们使用机器学习模型和专门的遗传算法将搜索范围缩小到故障剧集。我们将Starla应用于被广泛用作基准测试的Deep-Q-Learning代理上，结果表明它在检测到更多与代理策略相关的错误方面明显优于随机测试。我们还研究了如何使用搜索结果提取表征DRL代理故障情节的规则。此类规则可用于了解代理发生故障的条件，从而评估其部署风险。



## **45. SoK: Data Privacy in Virtual Reality**

SOK：虚拟现实中的数据隐私 cs.HC

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2301.05940v1) [paper-pdf](http://arxiv.org/pdf/2301.05940v1)

**Authors**: Gonzalo Munilla Garrido, Vivek Nair, Dawn Song

**Abstract**: The adoption of virtual reality (VR) technologies has rapidly gained momentum in recent years as companies around the world begin to position the so-called "metaverse" as the next major medium for accessing and interacting with the internet. While consumers have become accustomed to a degree of data harvesting on the web, the real-time nature of data sharing in the metaverse indicates that privacy concerns are likely to be even more prevalent in the new "Web 3.0." Research into VR privacy has demonstrated that a plethora of sensitive personal information is observable by various would-be adversaries from just a few minutes of telemetry data. On the other hand, we have yet to see VR parallels for many privacy-preserving tools aimed at mitigating threats on conventional platforms. This paper aims to systematize knowledge on the landscape of VR privacy threats and countermeasures by proposing a comprehensive taxonomy of data attributes, protections, and adversaries based on the study of 68 collected publications. We complement our qualitative discussion with a statistical analysis of the risk associated with various data sources inherent to VR in consideration of the known attacks and defenses. By focusing on highlighting the clear outstanding opportunities, we hope to motivate and guide further research into this increasingly important field.

摘要: 近年来，虚拟现实(VR)技术的采用势头迅速增强，世界各地的公司开始将所谓的“虚拟现实”定位为访问互联网和与互联网互动的下一个主要媒介。虽然消费者已经习惯了在一定程度上从网络上获取数据，但虚拟世界中数据共享的实时性质表明，对隐私的担忧可能会在新的“Web 3.0”中更加普遍。对VR隐私的研究表明，各种潜在的对手只需几分钟的遥测数据就可以观察到过多的敏感个人信息。另一方面，我们还没有看到许多旨在缓解传统平台上威胁的隐私保护工具的VR相似之处。本文旨在通过对收集到的68种出版物的研究，提出数据属性、保护和对手的全面分类，以系统化关于虚拟现实隐私威胁和对策的知识。考虑到已知的攻击和防御，我们用与VR固有的各种数据源相关的风险的统计分析来补充我们的定性讨论。通过重点突出明确的突出机遇，我们希望激励和指导对这一日益重要的领域的进一步研究。



## **46. Deepfake Detection using Biological Features: A Survey**

利用生物特征进行深伪检测的研究进展 cs.CV

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2301.05819v1) [paper-pdf](http://arxiv.org/pdf/2301.05819v1)

**Authors**: Kundan Patil, Shrushti Kale, Jaivanti Dhokey, Abhishek Gulhane

**Abstract**: Deepfake is a deep learning-based technique that makes it easy to change or modify images and videos. In investigations and court, visual evidence is commonly employed, but these pieces of evidence may now be suspect due to technological advancements in deepfake. Deepfakes have been used to blackmail individuals, plan terrorist attacks, disseminate false information, defame individuals, and foment political turmoil. This study describes the history of deepfake, its development and detection, and the challenges based on physiological measurements such as eyebrow recognition, eye blinking detection, eye movement detection, ear and mouth detection, and heartbeat detection. The study also proposes a scope in this field and compares the different biological features and their classifiers. Deepfakes are created using the generative adversarial network (GANs) model, and were once easy to detect by humans due to visible artifacts. However, as technology has advanced, deepfakes have become highly indistinguishable from natural images, making it important to review detection methods.

摘要: Deepfac是一种基于深度学习的技术，可以轻松地更改或修改图像和视频。在调查和法庭上，通常使用视觉证据，但由于深度假冒技术的进步，这些证据现在可能会受到怀疑。Deepfake被用来勒索个人、策划恐怖袭击、传播虚假信息、诽谤个人，并煽动政治动荡。这项研究描述了深度假象的历史、发展和检测，以及基于生理测量的挑战，如眉毛识别、眨眼检测、眼动检测、耳朵和嘴巴检测以及心跳检测。本研究还提出了这一领域的范围，并比较了不同的生物学特征及其分类器。Deepfake是使用生成性对抗网络(GANS)模型创建的，由于可见的人工制品，一度很容易被人类发现。然而，随着技术的进步，深伪与自然图像已经变得非常难以区分，这使得审查检测方法变得重要。



## **47. $A^{3}D$: A Platform of Searching for Robust Neural Architectures and Efficient Adversarial Attacks**

$A^{3}D$：搜索健壮的神经体系结构和高效的对抗性攻击的平台 cs.LG

**SubmitDate**: 2023-01-14    [abs](http://arxiv.org/abs/2203.03128v2) [paper-pdf](http://arxiv.org/pdf/2203.03128v2)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen

**Abstract**: The robustness of deep neural networks (DNN) models has attracted increasing attention due to the urgent need for security in many applications. Numerous existing open-sourced tools or platforms are developed to evaluate the robustness of DNN models by ensembling the majority of adversarial attack or defense algorithms. Unfortunately, current platforms do not possess the ability to optimize the architectures of DNN models or the configuration of adversarial attacks to further enhance the robustness of models or the performance of adversarial attacks. To alleviate these problems, in this paper, we first propose a novel platform called auto adversarial attack and defense ($A^{3}D$), which can help search for robust neural network architectures and efficient adversarial attacks. In $A^{3}D$, we employ multiple neural architecture search methods, which consider different robustness evaluation metrics, including four types of noises: adversarial noise, natural noise, system noise, and quantified metrics, resulting in finding robust architectures. Besides, we propose a mathematical model for auto adversarial attack, and provide multiple optimization algorithms to search for efficient adversarial attacks. In addition, we combine auto adversarial attack and defense together to form a unified framework. Among auto adversarial defense, the searched efficient attack can be used as the new robustness evaluation to further enhance the robustness. In auto adversarial attack, the searched robust architectures can be utilized as the threat model to help find stronger adversarial attacks. Experiments on CIFAR10, CIFAR100, and ImageNet datasets demonstrate the feasibility and effectiveness of the proposed platform, which can also provide a benchmark and toolkit for researchers in the application of automated machine learning in evaluating and improving the DNN model robustnesses.

摘要: 由于许多应用对安全性的迫切需求，深度神经网络(DNN)模型的稳健性受到越来越多的关注。许多现有的开源工具或平台被开发来通过集成大多数对抗性攻击或防御算法来评估DNN模型的健壮性。遗憾的是，目前的平台不具备优化DNN模型的体系结构或对抗性攻击的配置以进一步增强模型的健壮性或对抗性攻击的性能的能力。为了缓解这些问题，在本文中，我们首先提出了一个新的平台，称为自动对抗攻击和防御($A^{3}D$)，它可以帮助寻找健壮的神经网络结构和高效的对抗攻击。在$A^{3}D$中，我们使用了多种神经体系结构搜索方法，这些方法考虑了不同的健壮性评估指标，包括四种类型的噪声：对抗性噪声、自然噪声、系统噪声和量化指标，从而找到健壮的体系结构。此外，我们还提出了自动对抗性攻击的数学模型，并提供了多种优化算法来寻找有效的对抗性攻击。此外，我们将自动对抗性攻击和防御结合在一起，形成了一个统一的框架。在自动对抗性防御中，搜索到的高效攻击可以作为新的健壮性评估，进一步增强健壮性。在汽车对抗攻击中，搜索到的稳健结构可以作为威胁模型，帮助发现更强的对抗攻击。在CIFAR10、CIFAR100和ImageNet数据集上的实验证明了该平台的可行性和有效性，该平台还可以为研究人员在自动机器学习应用中评估和改进DNN模型的健壮性提供一个基准和工具包。



## **48. Threat Models over Space and Time: A Case Study of E2EE Messaging Applications**

空间和时间上的威胁模型：E2EE消息传递应用程序的案例研究 cs.CR

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05653v1) [paper-pdf](http://arxiv.org/pdf/2301.05653v1)

**Authors**: Partha Das Chowdhury, Maria Sameen, Jenny Blessing, Nicholas Boucher, Joseph Gardiner, Tom Burrows, Ross Anderson, Awais Rashid

**Abstract**: Threat modelling is foundational to secure systems engineering and should be done in consideration of the context within which systems operate. On the other hand, the continuous evolution of both the technical sophistication of threats and the system attack surface is an inescapable reality. In this work, we explore the extent to which real-world systems engineering reflects the changing threat context. To this end we examine the desktop clients of six widely used end-to-end-encrypted mobile messaging applications to understand the extent to which they adjusted their threat model over space (when enabling clients on new platforms, such as desktop clients) and time (as new threats emerged). We experimented with short-lived adversarial access against these desktop clients and analyzed the results with respect to two popular threat elicitation frameworks, STRIDE and LINDDUN. The results demonstrate that system designers need to both recognise the threats in the evolving context within which systems operate and, more importantly, to mitigate them by rescoping trust boundaries in a manner that those within the administrative boundary cannot violate security and privacy properties. Such a nuanced understanding of trust boundary scopes and their relationship with administrative boundaries allows for better administration of shared components, including securing them with safe defaults.

摘要: 威胁建模是安全系统工程的基础，应该考虑到系统运行的环境。另一方面，威胁的技术复杂性和系统攻击面的不断演变是一个不可避免的现实。在这项工作中，我们探索现实世界系统工程反映不断变化的威胁背景的程度。为此，我们研究了六个广泛使用的端到端加密移动消息传递应用程序的桌面客户端，以了解它们在空间(当在新平台上启用客户端时)和时间(当出现新威胁时)对其威胁模型进行调整的程度。我们对这些桌面客户端进行了短暂的恶意访问试验，并分析了两个流行的威胁诱导框架STRIDE和LINDDUN的结果。结果表明，系统设计者需要在系统运行的不断变化的环境中认识到威胁，更重要的是，通过以行政边界内的信任边界不能违反安全和隐私属性的方式重新应对信任边界来缓解这些威胁。对信任边界作用域及其与管理边界的关系的这种细致入微的理解允许更好地管理共享组件，包括使用安全缺省值保护它们。



## **49. Resilient Model Predictive Control of Distributed Systems Under Attack Using Local Attack Identification**

基于局部攻击识别的分布式系统攻击弹性模型预测控制 cs.SY

Submitted for review to Springer Natural Computer Science on November  18th 2022

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05547v1) [paper-pdf](http://arxiv.org/pdf/2301.05547v1)

**Authors**: Sarah Braun, Sebastian Albrecht, Sergio Lucia

**Abstract**: With the growing share of renewable energy sources, the uncertainty in power supply is increasing. In addition to the inherent fluctuations in the renewables, this is due to the threat of deliberate malicious attacks, which may become more revalent with a growing number of distributed generation units. Also in other safety-critical technology sectors, control systems are becoming more and more decentralized, causing the targets for attackers and thus the risk of attacks to increase. It is thus essential that distributed controllers are robust toward these uncertainties and able to react quickly to disturbances of any kind. To this end, we present novel methods for model-based identification of attacks and combine them with distributed model predictive control to obtain a resilient framework for adaptively robust control. The methodology is specially designed for distributed setups with limited local information due to privacy and security reasons. To demonstrate the efficiency of the method, we introduce a mathematical model for physically coupled microgrids under the uncertain influence of renewable generation and adversarial attacks, and perform numerical experiments, applying the proposed method for microgrid control.

摘要: 随着可再生能源的份额越来越大，电力供应的不确定性也在增加。除了可再生能源的内在波动之外，这是由于故意恶意攻击的威胁，随着分布式发电机组数量的增加，恶意攻击可能会变得更加普遍。同样在其他安全关键技术部门，控制系统正变得越来越分散，导致攻击者的目标，从而增加了攻击的风险。因此，分布式控制器对这些不确定性具有健壮性并能够对任何类型的干扰做出快速反应是至关重要的。为此，我们提出了新的基于模型的攻击识别方法，并将其与分布式模型预测控制相结合，得到了一种具有弹性的自适应鲁棒控制框架。由于隐私和安全原因，该方法是专门为本地信息有限的分布式设置而设计的。为了验证该方法的有效性，我们引入了在可再生发电和恶意攻击的不确定影响下的物理耦合微电网的数学模型，并将该方法应用于微电网控制，进行了数值实验。



## **50. PMFault: Faulting and Bricking Server CPUs through Management Interfaces**

PM故障：通过管理界面对服务器CPU进行故障修复和加固 cs.CR

For demo and source code, visit https://zt-chen.github.io/PMFault/

**SubmitDate**: 2023-01-13    [abs](http://arxiv.org/abs/2301.05538v1) [paper-pdf](http://arxiv.org/pdf/2301.05538v1)

**Authors**: Zitai Chen, David Oswald

**Abstract**: Apart from the actual CPU, modern server motherboards contain other auxiliary components, for example voltage regulators for power management. Those are connected to the CPU and the separate Baseboard Management Controller (BMC) via the I2C-based PMBus.   In this paper, using the case study of the widely used Supermicro X11SSL motherboard, we show how remotely exploitable software weaknesses in the BMC (or other processors with PMBus access) can be used to access the PMBus and then perform hardware-based fault injection attacks on the main CPU. The underlying weaknesses include insecure firmware encryption and signing mechanisms, a lack of authentication for the firmware upgrade process and the IPMI KCS control interface, as well as the motherboard design (with the PMBus connected to the BMC and SMBus by default).   First, we show that undervolting through the PMBus allows breaking the integrity guarantees of SGX enclaves, bypassing Intel's countermeasures against previous undervolting attacks like Plundervolt/V0ltPwn. Second, we experimentally show that overvolting outside the specified range has the potential of permanently damaging Intel Xeon CPUs, rendering the server inoperable. We assess the impact of our findings on other server motherboards made by Supermicro and ASRock.   Our attacks, dubbed PMFault, can be carried out by a privileged software adversary and do not require physical access to the server motherboard or knowledge of the BMC login credentials.   We responsibly disclosed the issues reported in this paper to Supermicro and discuss possible countermeasures at different levels. To the best of our knowledge, the 12th generation of Supermicro motherboards, which was designed before we reported PMFault to Supermicro, is not vulnerable.

摘要: 除了实际的CPU，现代服务器主板还包含其他辅助组件，例如用于电源管理的电压调节器。它们通过基于I2C的PMBus连接到CPU和单独的底板管理控制器(BMC)。本文以广泛使用的SuperMicro X11SSL主板为例，展示了如何利用BMC(或其他具有PMBus访问权限的处理器)中可远程利用的软件漏洞来访问PMBus，然后对主CPU执行基于硬件的故障注入攻击。潜在的弱点包括不安全的固件加密和签名机制，缺乏对固件升级过程和IPMI KCS控制接口的身份验证，以及主板设计(PMBus默认连接到BMC和SMBus)。首先，我们展示了通过PMBus的欠压允许打破SGX飞地的完整性保证，绕过英特尔针对之前的欠压攻击的对策，如Plundervolt/V0ltPwn。其次，我们的实验表明，超出指定范围的过电压有可能永久损坏Intel Xeon CPU，导致服务器无法运行。我们评估我们的发现对SuperMicro和ASRock制造的其他服务器主板的影响。我们的攻击，称为PMbug，可以由特权软件对手执行，不需要物理访问服务器主板或了解BMC登录凭据。我们负责任地向SuperMicro披露了本文报告的问题，并在不同层面讨论了可能的对策。据我们所知，第12代超微主板是在我们向超微报告PM故障之前设计的，并不容易受到攻击。



