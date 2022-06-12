# Latest Adversarial Attack Papers
**update at 2022-06-13 06:31:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Network insensitivity to parameter noise via adversarial regularization**

基于对抗性正则化的网络对参数噪声不敏感性 cs.LG

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2106.05009v3)

**Authors**: Julian Büchel, Fynn Faber, Dylan R. Muir

**Abstracts**: Neuromorphic neural network processors, in the form of compute-in-memory crossbar arrays of memristors, or in the form of subthreshold analog and mixed-signal ASICs, promise enormous advantages in compute density and energy efficiency for NN-based ML tasks. However, these technologies are prone to computational non-idealities, due to process variation and intrinsic device physics. This degrades the task performance of networks deployed to the processor, by introducing parameter noise into the deployed model. While it is possible to calibrate each device, or train networks individually for each processor, these approaches are expensive and impractical for commercial deployment. Alternative methods are therefore needed to train networks that are inherently robust against parameter variation, as a consequence of network architecture and parameters. We present a new adversarial network optimisation algorithm that attacks network parameters during training, and promotes robust performance during inference in the face of parameter variation. Our approach introduces a regularization term penalising the susceptibility of a network to weight perturbation. We compare against previous approaches for producing parameter insensitivity such as dropout, weight smoothing and introducing parameter noise during training. We show that our approach produces models that are more robust to targeted parameter variation, and equally robust to random parameter variation. Our approach finds minima in flatter locations in the weight-loss landscape compared with other approaches, highlighting that the networks found by our technique are less sensitive to parameter perturbation. Our work provides an approach to deploy neural network architectures to inference devices that suffer from computational non-idealities, with minimal loss of performance. ...

摘要: 神经形态神经网络处理器，以记忆电阻的计算-内存交叉开关阵列的形式，或以亚阈值模拟和混合信号ASIC的形式，有望在基于神经网络的ML任务的计算密度和能量效率方面具有巨大的优势。然而，由于工艺变化和本征器件物理，这些技术容易出现计算上的非理想化。这会在部署的模型中引入参数噪声，从而降低部署到处理器的网络的任务性能。虽然有可能校准每个设备，或者为每个处理器单独训练网络，但这些方法成本高昂，对于商业部署来说不切实际。因此，需要替代方法来训练作为网络体系结构和参数的结果而对参数变化具有内在健壮性的网络。我们提出了一种新的对抗性网络优化算法，该算法在训练过程中攻击网络参数，在面对参数变化的情况下提高推理过程中的鲁棒性。我们的方法引入了一个正则化项，惩罚了网络对权重扰动的敏感性。我们比较了以往产生参数不敏感度的方法，如丢弃、权重平滑和在训练过程中引入参数噪声。我们表明，我们的方法产生的模型对目标参数变化更稳健，对随机参数变化同样稳健。与其他方法相比，我们的方法在减肥场景中更平坦的位置找到了最小值，突出表明我们的技术找到的网络对参数扰动不那么敏感。我们的工作提供了一种方法，以最小的性能损失将神经网络结构部署到遭受计算非理想影响的推理设备。..。



## **2. Unlearning Protected User Attributes in Recommendations with Adversarial Training**

在带有对抗性训练的推荐中忘记受保护的用户属性 cs.IR

Accepted at SIGIR 2022

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04500v1)

**Authors**: Christian Ganhör, David Penz, Navid Rekabsaz, Oleg Lesota, Markus Schedl

**Abstracts**: Collaborative filtering algorithms capture underlying consumption patterns, including the ones specific to particular demographics or protected information of users, e.g. gender, race, and location. These encoded biases can influence the decision of a recommendation system (RS) towards further separation of the contents provided to various demographic subgroups, and raise privacy concerns regarding the disclosure of users' protected attributes. In this work, we investigate the possibility and challenges of removing specific protected information of users from the learned interaction representations of a RS algorithm, while maintaining its effectiveness. Specifically, we incorporate adversarial training into the state-of-the-art MultVAE architecture, resulting in a novel model, Adversarial Variational Auto-Encoder with Multinomial Likelihood (Adv-MultVAE), which aims at removing the implicit information of protected attributes while preserving recommendation performance. We conduct experiments on the MovieLens-1M and LFM-2b-DemoBias datasets, and evaluate the effectiveness of the bias mitigation method based on the inability of external attackers in revealing the users' gender information from the model. Comparing with baseline MultVAE, the results show that Adv-MultVAE, with marginal deterioration in performance (w.r.t. NDCG and recall), largely mitigates inherent biases in the model on both datasets.

摘要: 协作过滤算法捕获潜在的消费模式，包括特定的人口统计数据或受保护的用户信息，例如性别、种族和位置。这些编码的偏见会影响推荐系统(RS)对提供给各种人口统计子组的内容的进一步分离的决策，并引起关于披露用户的受保护属性的隐私问题。在这项工作中，我们研究了从RS算法的学习交互表示中移除用户特定的受保护信息的可能性和挑战，同时保持其有效性。具体地说，我们将对抗性训练融入到最新的MultVAE体系结构中，形成了一种新的模型--对抗性多项似然变分自动编码器(ADV-MultVAE)，其目的是在保持推荐性能的同时消除受保护属性的隐含信息。我们在MovieLens-1M和LFM-2b-DemoBias数据集上进行了实验，并评估了基于外部攻击者无法从模型中透露用户性别信息的偏差缓解方法的有效性。与基线的MultVAE相比，结果显示ADV-MultVAE，性能略有下降(W.r.t.NDCG和Recall)，在很大程度上缓解了模型在这两个数据集上的固有偏差。



## **3. Subfield Algorithms for Ideal- and Module-SVP Based on the Decomposition Group**

基于分解群的理想和模SVP的子场算法 cs.CR

29 pages plus appendix, to appear in Banach Center Publications

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2105.03219v3)

**Authors**: Christian Porter, Andrew Mendelsohn, Cong Ling

**Abstracts**: Whilst lattice-based cryptosystems are believed to be resistant to quantum attack, they are often forced to pay for that security with inefficiencies in implementation. This problem is overcome by ring- and module-based schemes such as Ring-LWE or Module-LWE, whose keysize can be reduced by exploiting its algebraic structure, allowing for faster computations. Many rings may be chosen to define such cryptoschemes, but cyclotomic rings, due to their cyclic nature allowing for easy multiplication, are the community standard. However, there is still much uncertainty as to whether this structure may be exploited to an adversary's benefit. In this paper, we show that the decomposition group of a cyclotomic ring of arbitrary conductor can be utilised to significantly decrease the dimension of the ideal (or module) lattice required to solve a given instance of SVP. Moreover, we show that there exist a large number of rational primes for which, if the prime ideal factors of an ideal lie over primes of this form, give rise to an "easy" instance of SVP. It is important to note that the work on ideal SVP does not break Ring-LWE, since its security reduction is from worst case ideal SVP to average case Ring-LWE, and is one way.

摘要: 虽然基于格子的密码系统被认为能够抵抗量子攻击，但它们经常被迫为这种安全性买单，因为实现效率低下。这个问题可以通过基于环和模块的方案来解决，例如环-LWE或模块-LWE，其密钥大小可以通过利用其代数结构来减小，从而允许更快的计算。可以选择许多环来定义这样的密码方案，但割圆环由于其循环性质允许容易相乘，是社区标准。然而，对于这种结构是否会被利用来为对手谋取利益，仍然存在很大的不确定性。在这篇文章中，我们证明了任意导体的分圆环的分解群可以用来显著降低求解给定SVP实例所需的理想(或模)格的维度。此外，我们还证明了存在大量的有理素数，对于这些有理素数，如果理想的素数理想因子位于这种形式的素数之上，则会产生SVP的“简单”实例。值得注意的是，关于理想SVP的工作不会破坏Ring-LWE，因为它的安全性降低是从最坏情况的理想SVP到平均情况的Ring-LWE，并且是单向的。



## **4. CARLA-GeAR: a Dataset Generator for a Systematic Evaluation of Adversarial Robustness of Vision Models**

Carla-Gear：用于系统评估视觉模型对抗稳健性的数据集生成器 cs.CV

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04365v1)

**Authors**: Federico Nesti, Giulio Rossolini, Gianluca D'Amico, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: Adversarial examples represent a serious threat for deep neural networks in several application domains and a huge amount of work has been produced to investigate them and mitigate their effects. Nevertheless, no much work has been devoted to the generation of datasets specifically designed to evaluate the adversarial robustness of neural models. This paper presents CARLA-GeAR, a tool for the automatic generation of photo-realistic synthetic datasets that can be used for a systematic evaluation of the adversarial robustness of neural models against physical adversarial patches, as well as for comparing the performance of different adversarial defense/detection methods. The tool is built on the CARLA simulator, using its Python API, and allows the generation of datasets for several vision tasks in the context of autonomous driving. The adversarial patches included in the generated datasets are attached to billboards or the back of a truck and are crafted by using state-of-the-art white-box attack strategies to maximize the prediction error of the model under test. Finally, the paper presents an experimental study to evaluate the performance of some defense methods against such attacks, showing how the datasets generated with CARLA-GeAR might be used in future work as a benchmark for adversarial defense in the real world. All the code and datasets used in this paper are available at http://carlagear.retis.santannapisa.it.

摘要: 对抗性的例子在几个应用领域对深度神经网络构成了严重的威胁，并且已经产生了大量的工作来调查它们并减轻它们的影响。然而，没有太多的工作致力于生成专门设计来评估神经模型的对抗性稳健性的数据集。本文介绍了一个自动生成照片真实感合成数据集的工具Carla-Gear，它可以用来系统地评估神经模型对物理对抗性补丁的对抗性健壮性，以及比较不同对抗性防御/检测方法的性能。该工具建立在Carla模拟器上，使用其PythonAPI，并允许在自动驾驶的背景下为几个视觉任务生成数据集。生成的数据集中包含的对抗性补丁被附加到广告牌或卡车后部，并通过使用最先进的白盒攻击策略来制作，以最大限度地提高测试模型的预测误差。最后，本文给出了一个实验研究，评估了一些防御方法对这类攻击的性能，展示了使用Carla-Gear生成的数据集如何在未来的工作中用作现实世界中对抗性防御的基准。本文中使用的所有代码和数据集都可以在http://carlagear.retis.santannapisa.it.上找到



## **5. Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks**

即插即用攻击：朝向健壮灵活的模型反转攻击 cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2201.12179v4)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, Kristian Kersting

**Abstracts**: Model inversion attacks (MIAs) aim to create synthetic images that reflect the class-wise characteristics from a target classifier's private training data by exploiting the model's learned knowledge. Previous research has developed generative MIAs that use generative adversarial networks (GANs) as image priors tailored to a specific target model. This makes the attacks time- and resource-consuming, inflexible, and susceptible to distributional shifts between datasets. To overcome these drawbacks, we present Plug & Play Attacks, which relax the dependency between the target model and image prior, and enable the use of a single GAN to attack a wide range of targets, requiring only minor adjustments to the attack. Moreover, we show that powerful MIAs are possible even with publicly available pre-trained GANs and under strong distributional shifts, for which previous approaches fail to produce meaningful results. Our extensive evaluation confirms the improved robustness and flexibility of Plug & Play Attacks and their ability to create high-quality images revealing sensitive class characteristics.

摘要: 模型反转攻击(MIA)的目的是利用目标分类器的学习知识，从目标分类器的私有训练数据中创建反映类别特征的合成图像。以前的研究已经开发出生成性MIA，它使用生成性对抗网络(GANS)作为针对特定目标模型量身定做的图像先验。这使得攻击耗费时间和资源，不灵活，并且容易受到数据集之间的分布变化的影响。为了克服这些缺点，我们提出了即插即用攻击，它放松了目标模型和图像先验之间的依赖，使单个GAN能够攻击范围广泛的目标，只需要对攻击进行微小的调整。此外，我们表明，即使在公开可用的预先训练的GAN和强烈的分布变化下，强大的MIA也是可能的，以前的方法无法产生有意义的结果。我们广泛的评估证实了即插即用攻击的健壮性和灵活性的提高，以及它们创建揭示敏感类别特征的高质量图像的能力。



## **6. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

学习打破深度感知散列：用例NeuralHash cs.LG

Accepted by ACM FAccT 2022 as Oral

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2111.06628v4)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.

摘要: 苹果最近公布了其深度感知哈希系统NeuralHash，用于在文件上传到其iCloud服务之前检测用户设备上的儿童性虐待材料(CSAM)。公众很快就对保护用户隐私和系统的可靠性提出了批评。本文首次提出了基于NeuralHash的深度感知哈希算法的综合实证分析。具体地说，我们证明了当前的深度感知散列可能并不健壮。攻击者可以通过在图像中应用微小的更改来操纵散列值，这可以是由基于梯度的方法引起的，也可以只是通过执行标准图像转换来强制或防止散列冲突。这种攻击让恶意行为者很容易利用检测系统：从隐藏滥用材料到陷害无辜用户，一切皆有可能。此外，使用散列值，仍然可以对存储在用户设备上的数据进行推断。在我们看来，根据我们的结果，当前形式的深度感知散列通常还不能用于健壮的客户端扫描，不应该从隐私的角度使用。



## **7. Bounding Training Data Reconstruction in Private (Deep) Learning**

私密(深度)学习中的边界训练数据重构 cs.LG

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2201.12383v3)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.

摘要: 在ML中，差异隐私被广泛接受为防止数据泄露的事实上的方法，传统观点认为，它提供了针对隐私攻击的强大保护。然而，现有的DP语义保证侧重于成员关系推理，这可能会高估对手的能力，并且不适用于成员身份本身不敏感的情况。本文首先在形式化威胁模型下给出了DP机制抵抗训练数据重构攻击的语义保证。我们发现，两种不同的隐私记账方法--Renyi Differential Privacy和Fisher信息泄漏--都提供了对数据重构攻击的强大语义保护。



## **8. Blacklight: Scalable Defense for Neural Networks against Query-Based Black-Box Attacks**

Blacklight：针对基于查询的黑盒攻击的神经网络可扩展防御 cs.CR

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2006.14042v3)

**Authors**: Huiying Li, Shawn Shan, Emily Wenger, Jiayun Zhang, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Deep learning systems are known to be vulnerable to adversarial examples. In particular, query-based black-box attacks do not require knowledge of the deep learning model, but can compute adversarial examples over the network by submitting queries and inspecting returns. Recent work largely improves the efficiency of those attacks, demonstrating their practicality on today's ML-as-a-service platforms.   We propose Blacklight, a new defense against query-based black-box adversarial attacks. The fundamental insight driving our design is that, to compute adversarial examples, these attacks perform iterative optimization over the network, producing image queries highly similar in the input space. Blacklight detects query-based black-box attacks by detecting highly similar queries, using an efficient similarity engine operating on probabilistic content fingerprints. We evaluate Blacklight against eight state-of-the-art attacks, across a variety of models and image classification tasks. Blacklight identifies them all, often after only a handful of queries. By rejecting all detected queries, Blacklight prevents any attack to complete, even when attackers persist to submit queries after account ban or query rejection. Blacklight is also robust against several powerful countermeasures, including an optimal black-box attack that approximates white-box attacks in efficiency. Finally, we illustrate how Blacklight generalizes to other domains like text classification.

摘要: 众所周知，深度学习系统很容易受到敌意例子的攻击。特别是，基于查询的黑盒攻击不需要深度学习模型的知识，但可以通过提交查询和检查返回来计算网络上的对抗性示例。最近的工作在很大程度上提高了这些攻击的效率，证明了它们在今天的ML即服务平台上的实用性。我们提出了Blacklight，一种新的针对基于查询的黑盒对抗攻击的防御方案。驱动我们设计的基本见解是，为了计算对抗性的例子，这些攻击在网络上执行迭代优化，产生在输入空间中高度相似的图像查询。Blacklight使用对概率内容指纹进行操作的高效相似性引擎，通过检测高度相似的查询来检测基于查询的黑盒攻击。我们针对各种型号和图像分类任务中的八种最先进的攻击对Blacklight进行评估。Blacklight通常只在几个问题之后就能识别出所有这些问题。通过拒绝所有检测到的查询，Blacklight可以阻止任何攻击完成，即使攻击者在帐户禁用或查询拒绝后仍坚持提交查询。Blacklight对几种强大的对策也很健壮，包括在效率上接近白盒攻击的最佳黑盒攻击。最后，我们说明了Blacklight如何推广到文本分类等其他领域。



## **9. Adversarial Text Normalization**

对抗性文本规范化 cs.CL

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.04137v1)

**Authors**: Joanna Bitton, Maya Pavlova, Ivan Evtimov

**Abstracts**: Text-based adversarial attacks are becoming more commonplace and accessible to general internet users. As these attacks proliferate, the need to address the gap in model robustness becomes imminent. While retraining on adversarial data may increase performance, there remains an additional class of character-level attacks on which these models falter. Additionally, the process to retrain a model is time and resource intensive, creating a need for a lightweight, reusable defense. In this work, we propose the Adversarial Text Normalizer, a novel method that restores baseline performance on attacked content with low computational overhead. We evaluate the efficacy of the normalizer on two problem areas prone to adversarial attacks, i.e. Hate Speech and Natural Language Inference. We find that text normalization provides a task-agnostic defense against character-level attacks that can be implemented supplementary to adversarial retraining solutions, which are more suited for semantic alterations.

摘要: 基于文本的敌意攻击正变得越来越常见，普通互联网用户也可以访问。随着这些攻击的激增，解决模型健壮性差距的需求变得迫在眉睫。虽然对对抗性数据的再训练可能会提高性能，但仍然存在一种额外的字符级攻击，这些模型在这种攻击上步履蹒跚。此外，重新训练模型的过程是时间和资源密集型的，这就产生了对轻型、可重复使用的防御的需求。在这项工作中，我们提出了对抗性文本规格化器，这是一种新的方法，以较低的计算开销恢复受攻击内容的基线性能。我们评估了归一化在两个容易受到敌意攻击的问题领域的有效性，即仇恨言论和自然语言推理。我们发现，文本归一化提供了一种针对字符级攻击的与任务无关的防御，可以实现对对抗性再训练解决方案的补充，后者更适合于语义变化。



## **10. PrivHAR: Recognizing Human Actions From Privacy-preserving Lens**

PrivHAR：从隐私保护镜头识别人类行为 cs.CV

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03891v1)

**Authors**: Carlos Hinojosa, Miguel Marquez, Henry Arguello, Ehsan Adeli, Li Fei-Fei, Juan Carlos Niebles

**Abstracts**: The accelerated use of digital cameras prompts an increasing concern about privacy and security, particularly in applications such as action recognition. In this paper, we propose an optimizing framework to provide robust visual privacy protection along the human action recognition pipeline. Our framework parameterizes the camera lens to successfully degrade the quality of the videos to inhibit privacy attributes and protect against adversarial attacks while maintaining relevant features for activity recognition. We validate our approach with extensive simulations and hardware experiments.

摘要: 数码相机的加速使用促使人们越来越关注隐私和安全，特别是在动作识别等应用中。在这篇文章中，我们提出了一个优化的框架，以提供稳健的视觉隐私保护沿人类行为识别管道。我们的框架对摄像机镜头进行了参数化处理，成功地降低了视频的质量，从而抑制了隐私属性并防止了敌意攻击，同时保持了活动识别的相关特征。我们通过大量的仿真和硬件实验来验证我们的方法。



## **11. Standalone Neural ODEs with Sensitivity Analysis**

带敏感度分析的独立神经网络模型 cs.LG

25 pages, 15 figures; typos corrected

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2205.13933v2)

**Authors**: Rym Jaroudi, Lukáš Malý, Gabriel Eilertsen, B. Tomas Johansson, Jonas Unger, George Baravdish

**Abstracts**: This paper presents the Standalone Neural ODE (sNODE), a continuous-depth neural ODE model capable of describing a full deep neural network. This uses a novel nonlinear conjugate gradient (NCG) descent optimization scheme for training, where the Sobolev gradient can be incorporated to improve smoothness of model weights. We also present a general formulation of the neural sensitivity problem and show how it is used in the NCG training. The sensitivity analysis provides a reliable measure of uncertainty propagation throughout a network, and can be used to study model robustness and to generate adversarial attacks. Our evaluations demonstrate that our novel formulations lead to increased robustness and performance as compared to ResNet models, and that it opens up for new opportunities for designing and developing machine learning with improved explainability.

摘要: 提出了一种能够描述完整深度神经网络的连续深度神经网络模型--独立神经网络模型(SNODE)。该算法采用一种新的非线性共轭梯度(NCG)下降优化方案进行训练，其中可以引入Soblev梯度来改善模型权重的平滑程度。我们还给出了神经敏感度问题的一般公式，并展示了如何将其用于NCG训练。敏感度分析为不确定性在整个网络中的传播提供了可靠的度量，并可用于研究模型的健壮性和生成对抗性攻击。我们的评估表明，与ResNet模型相比，我们的新公式导致了更高的稳健性和性能，并为设计和开发具有更好解释性的机器学习开辟了新的机会。



## **12. Wavelet Regularization Benefits Adversarial Training**

小波正则化有利于对抗训练 cs.CV

Preprint version

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03727v1)

**Authors**: Jun Yan, Huilin Yin, Xiaoyang Deng, Ziming Zhao, Wancheng Ge, Hao Zhang, Gerhard Rigoll

**Abstracts**: Adversarial training methods are state-of-the-art (SOTA) empirical defense methods against adversarial examples. Many regularization methods have been proven to be effective with the combination of adversarial training. Nevertheless, such regularization methods are implemented in the time domain. Since adversarial vulnerability can be regarded as a high-frequency phenomenon, it is essential to regulate the adversarially-trained neural network models in the frequency domain. Faced with these challenges, we make a theoretical analysis on the regularization property of wavelets which can enhance adversarial training. We propose a wavelet regularization method based on the Haar wavelet decomposition which is named Wavelet Average Pooling. This wavelet regularization module is integrated into the wide residual neural network so that a new WideWaveletResNet model is formed. On the datasets of CIFAR-10 and CIFAR-100, our proposed Adversarial Wavelet Training method realizes considerable robustness under different types of attacks. It verifies the assumption that our wavelet regularization method can enhance adversarial robustness especially in the deep wide neural networks. The visualization experiments of the Frequency Principle (F-Principle) and interpretability are implemented to show the effectiveness of our method. A detailed comparison based on different wavelet base functions is presented. The code is available at the repository: \url{https://github.com/momo1986/AdversarialWaveletTraining}.

摘要: 对抗性训练方法是针对对抗性例子的最先进的(SOTA)经验防御方法。许多正则化方法与对抗性训练相结合已被证明是有效的。然而，这种正则化方法是在时间域中实现的。由于对抗性脆弱性可以看作是一种高频现象，因此对对抗性训练的神经网络模型进行频域调整是非常必要的。面对这些挑战，我们对小波的正则化性质进行了理论分析，以增强对抗性训练。提出了一种基于Haar小波分解的小波正则化方法--小波平均池化。将该小波正则化模块集成到宽残差神经网络中，形成了一种新的宽小波响应网络模型。在CIFAR-10和CIFAR-100的数据集上，本文提出的对抗性小波训练方法在不同类型的攻击下具有较好的鲁棒性。验证了小波正则化方法可以增强对抗攻击的稳健性，特别是在深度广泛的神经网络中。通过对频率原理(F原理)和可解释性的可视化实验，验证了该方法的有效性。对不同的小波基函数进行了详细的比较。代码可以在存储库中找到：\url{https://github.com/momo1986/AdversarialWaveletTraining}.



## **13. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

Prada：针对神经排序模型的实用黑箱对抗性攻击 cs.IR

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2204.01321v3)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed. In this paper, we introduce the Word Substitution Ranking Attack (WSRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers cannot directly get access to the model information, but can only query the target model to obtain the rank positions of the partial retrieved list. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations. Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.

摘要: 近年来，神经网络排序模型(NRM)取得了显著的成功，尤其是利用预先训练好的语言模型。然而，深层神经模型因其易受敌意例子的攻击而臭名昭著。鉴于我们对神经信息检索模型的日益依赖，对抗性攻击可能成为一种新型的Web垃圾邮件技术。因此，在部署NRM之前，研究潜在的敌意攻击以识别NRM的漏洞是很重要的。在本文中，我们引入了针对NRMS的单词替换排名攻击(WSRA)任务，该任务旨在通过在目标文档的文本中添加对抗性扰动来提升其排名。重点研究了基于决策的黑盒攻击环境，攻击者不能直接获取模型信息，只能通过查询目标模型获得部分检索列表的排名位置。这种攻击设置在现实世界的搜索引擎中是现实的。提出了一种新的基于伪相关性的对抗性排序攻击方法(PRADA)，该方法通过学习基于伪相关反馈(PRF)的代理模型来生成用于发现对抗性扰动的梯度。在两个网络搜索基准数据集上的实验表明，Prada可以超越现有的攻击策略，并成功地利用文本的微小不可分辨扰动来欺骗NRM。



## **14. Latent Boundary-guided Adversarial Training**

潜在边界制导的对抗性训练 cs.LG

To appear in Machine Learning

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03717v1)

**Authors**: Xiaowei Zhou, Ivor W. Tsang, Jie Yin

**Abstracts**: Deep Neural Networks (DNNs) have recently achieved great success in many classification tasks. Unfortunately, they are vulnerable to adversarial attacks that generate adversarial examples with a small perturbation to fool DNN models, especially in model sharing scenarios. Adversarial training is proved to be the most effective strategy that injects adversarial examples into model training to improve the robustness of DNN models to adversarial attacks. However, adversarial training based on the existing adversarial examples fails to generalize well to standard, unperturbed test data. To achieve a better trade-off between standard accuracy and adversarial robustness, we propose a novel adversarial training framework called LAtent bounDary-guided aDvErsarial tRaining (LADDER) that adversarially trains DNN models on latent boundary-guided adversarial examples. As opposed to most of the existing methods that generate adversarial examples in the input space, LADDER generates a myriad of high-quality adversarial examples through adding perturbations to latent features. The perturbations are made along the normal of the decision boundary constructed by an SVM with an attention mechanism. We analyze the merits of our generated boundary-guided adversarial examples from a boundary field perspective and visualization view. Extensive experiments and detailed analysis on MNIST, SVHN, CelebA, and CIFAR-10 validate the effectiveness of LADDER in achieving a better trade-off between standard accuracy and adversarial robustness as compared with vanilla DNNs and competitive baselines.

摘要: 近年来，深度神经网络(DNN)在许多分类任务中取得了巨大的成功。不幸的是，它们很容易受到对抗性攻击，生成带有微小扰动的对抗性示例来愚弄DNN模型，特别是在模型共享场景中。对抗性训练被证明是将对抗性实例注入模型训练以提高DNN模型对对抗性攻击的稳健性的最有效策略。然而，基于现有对抗性实例的对抗性训练不能很好地推广到标准的、不受干扰的测试数据。为了在标准准确率和对手健壮性之间实现更好的折衷，我们提出了一种新的对手训练框架，称为潜在边界制导的对抗性训练(LIDA)，该框架针对潜在的边界制导的对抗性实例对DNN模型进行对抗性训练。与大多数现有的在输入空间生成对抗性实例的方法不同，梯形图通过对潜在特征添加扰动来生成大量高质量的对抗性实例。扰动沿具有注意力机制的支持向量机构造的决策边界的法线进行。我们从边界场的角度和可视化的角度分析了我们生成的边界制导的对抗性例子的优点。在MNIST、SVHN、CelebA和CIFAR-10上的大量实验和详细分析验证了梯形算法在标准准确率和对手健壮性之间取得了比普通DNN和竞争基线更好的折衷。



## **15. Autoregressive Perturbations for Data Poisoning**

数据中毒的自回归摄动 cs.LG

21 pages, 13 figures. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03693v1)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstracts**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.

摘要: 从社交媒体上窃取数据作为获取数据集的一种手段的盛行，导致对未经授权使用数据的担忧日益加剧。数据中毒攻击被认为是防止抓取的堡垒，因为它们通过添加微小的、不可察觉的干扰而使数据“无法学习”。不幸的是，现有方法需要目标体系结构和完整数据集的知识，以便可以训练代理网络，其参数用于生成攻击。在这项工作中，我们引入了自回归(AR)中毒，这是一种在不访问更广泛的数据集的情况下生成有毒数据的方法。所提出的AR扰动是通用的，可以应用于不同的数据集，并且可能毒害不同的体系结构。与现有的无法学习的方法相比，我们的AR毒药对常见的防御措施更具抵抗力，例如对抗性训练和强大的数据增强。我们的分析进一步提供了对有效数据毒害的原因的洞察。



## **16. SHORTSTACK: Distributed, Fault-tolerant, Oblivious Data Access**

ShortStack：分布式、容错、不经意的数据访问 cs.CR

Full version of USENIX OSDI'22 paper

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2205.14281v2)

**Authors**: Midhul Vuppalapati, Kushal Babel, Anurag Khandelwal, Rachit Agarwal

**Abstracts**: Many applications that benefit from data offload to cloud services operate on private data. A now-long line of work has shown that, even when data is offloaded in an encrypted form, an adversary can learn sensitive information by analyzing data access patterns. Existing techniques for oblivious data access-that protect against access pattern attacks-require a centralized and stateful trusted proxy to orchestrate data accesses from applications to cloud services. We show that, in failure-prone deployments, such a centralized and stateful proxy results in violation of oblivious data access security guarantees and/or system unavailability. We thus initiate the study of distributed, fault-tolerant, oblivious data access.   We present SHORTSTACK, a distributed proxy architecture for oblivious data access in failure-prone deployments. SHORTSTACK achieves the classical obliviousness guarantee--access patterns observed by the adversary being independent of the input--even under a powerful passive persistent adversary that can force failure of arbitrary (bounded-sized) subset of proxy servers at arbitrary times. We also introduce a security model that enables studying oblivious data access with distributed, failure-prone, servers. We provide a formal proof that SHORTSTACK enables oblivious data access under this model, and show empirically that SHORTSTACK performance scales near-linearly with number of distributed proxy servers.

摘要: 许多受益于数据分流到云服务的应用程序都在私有数据上运行。目前的一系列工作表明，即使以加密的形式卸载数据，对手也可以通过分析数据访问模式来获取敏感信息。现有的不经意数据访问技术--防止访问模式攻击--需要一个集中的、有状态的可信代理来协调从应用程序到云服务的数据访问。我们表明，在容易出现故障的部署中，这种集中式和有状态的代理会导致违反不经意的数据访问安全保证和/或系统不可用。因此，我们开始了对分布式、容错、不经意数据访问的研究。我们提出了ShortStack，这是一种分布式代理体系结构，用于在容易出现故障的部署中进行不经意的数据访问。ShortStack实现了经典的遗忘保证--攻击者观察到的访问模式独立于输入--即使在强大的被动持久对手下也是如此，该对手可以在任意时间强制任意(有限大小的)代理服务器子集发生故障。我们还介绍了一个安全模型，该模型能够研究使用分布式的、容易发生故障的服务器的不经意的数据访问。我们给出了一个形式化的证明，证明了在该模型下，ShortStack能够实现不经意的数据访问，并通过实验证明了ShortStack的性能随着分布式代理服务器的数量近似线性地扩展。



## **17. Dap-FL: Federated Learning flourishes by adaptive tuning and secure aggregation**

DAP-FL：联合学习通过自适应调整和安全聚合蓬勃发展 cs.CR

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03623v1)

**Authors**: Qian Chen, Zilong Wang, Jiawei Chen, Haonan Yan, Xiaodong Lin

**Abstracts**: Federated learning (FL), an attractive and promising distributed machine learning paradigm, has sparked extensive interest in exploiting tremendous data stored on ubiquitous mobile devices. However, conventional FL suffers severely from resource heterogeneity, as clients with weak computational and communication capability may be unable to complete local training using the same local training hyper-parameters. In this paper, we propose Dap-FL, a deep deterministic policy gradient (DDPG)-assisted adaptive FL system, in which local learning rates and local training epochs are adaptively adjusted by all resource-heterogeneous clients through locally deployed DDPG-assisted adaptive hyper-parameter selection schemes. Particularly, the rationality of the proposed hyper-parameter selection scheme is confirmed through rigorous mathematical proof. Besides, due to the thoughtlessness of security consideration of adaptive FL systems in previous studies, we introduce the Paillier cryptosystem to aggregate local models in a secure and privacy-preserving manner. Rigorous analyses show that the proposed Dap-FL system could guarantee the security of clients' private local models against chosen-plaintext attacks and chosen-message attacks in a widely used honest-but-curious participants and active adversaries security model. In addition, through ingenious and extensive experiments, the proposed Dap-FL achieves higher global model prediction accuracy and faster convergence rates than conventional FL, and the comprehensiveness of the adjusted local training hyper-parameters is validated. More importantly, experimental results also show that the proposed Dap-FL achieves higher model prediction accuracy than two state-of-the-art RL-assisted FL methods, i.e., 6.03% higher than DDPG-based FL and 7.85% higher than DQN-based FL.

摘要: 联合学习(FL)作为一种极具吸引力和前景的分布式机器学习范式，引起了人们对利用存储在无处不在的移动设备上的海量数据的广泛兴趣。然而，传统的FL存在严重的资源异构性问题，计算和通信能力较弱的客户端可能无法使用相同的局部训练超参数来完成局部训练。本文提出了一种深度确定性策略梯度(DDPG)辅助的自适应FL系统DAP-FL，该系统通过本地部署的深度确定性策略梯度(DDPG)辅助的自适应超参数选择机制，由所有资源不同的客户端自适应地调整局部学习率和局部训练周期。通过严格的数学证明，验证了所提出的超参数选择方案的合理性。此外，由于以往研究中对自适应FL系统的安全性考虑不够，我们引入了Paillier密码体制，以安全和保护隐私的方式聚合局部模型。严格的分析表明，在广泛使用的诚实但好奇的参与者和主动对手的安全模型中，所提出的DAP-FL系统能够保证客户的私有本地模型免受选择明文攻击和选择消息攻击。此外，通过巧妙和广泛的实验，提出的Dap-FL比传统FL具有更高的全局模型预测精度和更快的收敛速度，并验证了调整后的局部训练超参数的全面性。更重要的是，实验结果还表明，Dap-FL的模型预测精度高于两种最先进的RL辅助FL方法，即比基于DDPG的FL高6.03%，比基于DQN的FL高7.85%。



## **18. Random and Adversarial Bit Error Robustness: Energy-Efficient and Secure DNN Accelerators**

随机和对抗性误码稳健性：节能和安全的DNN加速器 cs.LG

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2104.08323v2)

**Authors**: David Stutz, Nandhini Chandramoorthy, Matthias Hein, Bernt Schiele

**Abstracts**: Deep neural network (DNN) accelerators received considerable attention in recent years due to the potential to save energy compared to mainstream hardware. Low-voltage operation of DNN accelerators allows to further reduce energy consumption, however, causes bit-level failures in the memory storing the quantized weights. Furthermore, DNN accelerators are vulnerable to adversarial attacks on voltage controllers or individual bits. In this paper, we show that a combination of robust fixed-point quantization, weight clipping, as well as random bit error training (RandBET) or adversarial bit error training (AdvBET) improves robustness against random or adversarial bit errors in quantized DNN weights significantly. This leads not only to high energy savings for low-voltage operation as well as low-precision quantization, but also improves security of DNN accelerators. In contrast to related work, our approach generalizes across operating voltages and accelerators and does not require hardware changes. Moreover, we present a novel adversarial bit error attack and are able to obtain robustness against both targeted and untargeted bit-level attacks. Without losing more than 0.8%/2% in test accuracy, we can reduce energy consumption on CIFAR10 by 20%/30% for 8/4-bit quantization. Allowing up to 320 adversarial bit errors, we reduce test error from above 90% (chance level) to 26.22%.

摘要: 由于与主流硬件相比，深度神经网络(DNN)加速器具有节能的潜力，近年来受到了极大的关注。DNN加速器的低电压操作允许进一步降低能耗，然而，这会导致存储量化权重的存储器中的位级故障。此外，DNN加速器容易受到针对电压控制器或单个比特的敌意攻击。在这篇文章中，我们证明了稳健的定点量化、权重削减以及随机误码训练(RandBET)或对抗误码训练(AdvBET)的组合显著地提高了对量化DNN权重中的随机或敌意误码的鲁棒性。这不仅为低电压操作和低精度量化带来了高节能，而且还提高了DNN加速器的安全性。与相关工作相比，我们的方法适用于工作电压和加速器，不需要更改硬件。此外，我们提出了一种新的对抗性比特错误攻击，并且能够对目标和非目标比特级攻击获得健壮性。在不损失超过0.8%/2%的测试精度的情况下，我们可以将CIFAR10上8/4位量化的能耗降低20%/30%。允许多达320个对抗性比特错误，我们将测试错误从90%以上(概率水平)降低到26.22%。



## **19. Optimal Clock Synchronization with Signatures**

利用签名实现最优时钟同步 cs.DC

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2203.02553v2)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.

摘要: 通过增加可容忍的错误方的数量，可以使用加密签名来提高分布式系统对对手攻击的恢复能力。虽然这一点已经得到了广泛的研究，但在容错时钟同步的背景下，甚至在完全连接的系统中，这一点也没有得到充分的研究。这里，尽管本地时钟速率在$1$和$\vartheta>1$之间变化，端到端通信延迟在$d-u$和$d$之间变化，以及来自恶意方的干扰，但$n$节点系统的诚实方被要求计算小偏差(即最大相位偏移)的输出时钟。到目前为止，只有已知的歪斜$d$时钟脉冲能够以$\lceil n/2\rceil$(PODC`19)的(最优的)弹性产生，改进了没有签名的$\lceil n/3\rceil$保持的紧凑界限(STEC`84，PODC`85)。由于通常是$d\gg u$和$\vartheta-1\ll 1$，这远远不是即使在无故障的情况下也适用的$u+(\vartheta-1)d$的下限(IPL‘01)。我们证明了$theta(u+(vartheta-1)d)$在从$lceil n/3\rceil$到$lceil n/2\rceil-1$的斜斜度上的上下界是匹配的。在假设对手不能伪造签名的情况下，给出上界的算法是确定性的。即使时钟最初是完全同步的，诚实节点之间的消息延迟是已知的，$\vartheta$任意接近于1，并且同步算法是随机的，这个下界仍然成立。这对寻求利用签名来提供更可靠时间的网络设计人员具有至关重要的影响。与没有签名的设置相比，它们必须确保攻击者不能轻松绕过具有故障端点的链路上的延迟下限。



## **20. Towards Understanding and Mitigating Audio Adversarial Examples for Speaker Recognition**

说话人识别中音频对抗性实例的理解与缓解 cs.SD

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03393v1)

**Authors**: Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Feng Wang, Jiashui Wang

**Abstracts**: Speaker recognition systems (SRSs) have recently been shown to be vulnerable to adversarial attacks, raising significant security concerns. In this work, we systematically investigate transformation and adversarial training based defenses for securing SRSs. According to the characteristic of SRSs, we present 22 diverse transformations and thoroughly evaluate them using 7 recent promising adversarial attacks (4 white-box and 3 black-box) on speaker recognition. With careful regard for best practices in defense evaluations, we analyze the strength of transformations to withstand adaptive attacks. We also evaluate and understand their effectiveness against adaptive attacks when combined with adversarial training. Our study provides lots of useful insights and findings, many of them are new or inconsistent with the conclusions in the image and speech recognition domains, e.g., variable and constant bit rate speech compressions have different performance, and some non-differentiable transformations remain effective against current promising evasion techniques which often work well in the image domain. We demonstrate that the proposed novel feature-level transformation combined with adversarial training is rather effective compared to the sole adversarial training in a complete white-box setting, e.g., increasing the accuracy by 13.62% and attack cost by two orders of magnitude, while other transformations do not necessarily improve the overall defense capability. This work sheds further light on the research directions in this field. We also release our evaluation platform SPEAKERGUARD to foster further research.

摘要: 说话人识别系统(SRSS)最近被证明容易受到敌意攻击，这引发了严重的安全问题。在这项工作中，我们系统地研究了基于变换和对抗性训练的安全SRSS防御。根据SRSS的特点，我们提出了22种不同的变换，并用最近在说话人识别方面有希望的7种对抗性攻击(4个白盒和3个黑盒)对它们进行了全面的评估。在仔细考虑防御评估中的最佳实践的情况下，我们分析了转换抵御自适应攻击的强度。我们还评估和理解了它们与对抗性训练相结合时对抗适应性攻击的有效性。我们的研究提供了许多有价值的见解和发现，其中许多是新的或与图像和语音识别领域的结论不一致的，例如，可变比特率和恒定比特率语音压缩具有不同的性能，一些不可微变换仍然有效地对抗当前有希望的规避技术，这些技术在图像领域通常效果很好。与完全白盒环境下的单一对抗性训练相比，本文提出的新的特征级变换结合对抗性训练是相当有效的，例如提高了13.62%的准确率和两个数量级的攻击代价，而其他变换并不一定提高整体防御能力。这项工作进一步揭示了这一领域的研究方向。我们还发布了我们的评估平台SPEAKERGUARD，以促进进一步的研究。



## **21. Building Robust Ensembles via Margin Boosting**

通过提高利润率来构建稳健的整体 cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03362v1)

**Authors**: Dinghuai Zhang, Hongyang Zhang, Aaron Courville, Yoshua Bengio, Pradeep Ravikumar, Arun Sai Suggala

**Abstracts**: In the context of adversarial robustness, a single model does not usually have enough power to defend against all possible adversarial attacks, and as a result, has sub-optimal robustness. Consequently, an emerging line of work has focused on learning an ensemble of neural networks to defend against adversarial attacks. In this work, we take a principled approach towards building robust ensembles. We view this problem from the perspective of margin-boosting and develop an algorithm for learning an ensemble with maximum margin. Through extensive empirical evaluation on benchmark datasets, we show that our algorithm not only outperforms existing ensembling techniques, but also large models trained in an end-to-end fashion. An important byproduct of our work is a margin-maximizing cross-entropy (MCE) loss, which is a better alternative to the standard cross-entropy (CE) loss. Empirically, we show that replacing the CE loss in state-of-the-art adversarial training techniques with our MCE loss leads to significant performance improvement.

摘要: 在对抗性稳健性的背景下，单个模型通常不具有足够的能力来防御所有可能的对抗性攻击，因此具有次优的稳健性。因此，一项新兴的工作重点是学习一组神经网络，以抵御对手的攻击。在这项工作中，我们采取了一种原则性的方法来建立稳健的合奏。我们从边际提升的角度来考虑这一问题，并提出了一个学习具有最大边际的集成的算法。通过在基准数据集上的广泛实验评估，我们的算法不仅优于现有的集成技术，而且优于以端到端方式训练的大型模型。我们工作的一个重要副产品是边际最大化交叉熵(MCE)损失，这是标准交叉熵(CE)损失的更好替代。经验表明，用我们的MCE损失取代最先进的对抗性训练技术中的CE损失会导致显著的性能改进。



## **22. Adaptive Regularization for Adversarial Training**

自适应正则化在对抗性训练中的应用 stat.ML

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03353v1)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstracts**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to use a data-adaptive regularization for robustifying a prediction model. We apply more regularization to data which are more vulnerable to adversarial attacks and vice versa. Even though the idea of data-adaptive regularization is not new, our data-adaptive regularization has a firm theoretical base of reducing an upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on clean samples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.

摘要: 对抗性训练是为了提高对抗攻击的稳健性，因为它很容易产生人类无法察觉的数据扰动来欺骗给定的深度神经网络。在本文中，我们提出了一种新的对抗性训练算法，该算法在理论上动机良好，在经验上优于其他现有的算法。该算法的一个新特点是使用数据自适应正则化来增强预测模型的健壮性。我们对更容易受到对手攻击的数据应用更多的正则化，反之亦然。尽管数据自适应正则化的思想并不新鲜，但我们的数据自适应正则化在降低稳健风险上界方面有着坚实的理论基础。数值实验表明，我们提出的算法同时提高了泛化(对干净样本的准确率)和稳健性(对敌意攻击的准确率)，达到了最好的性能。



## **23. AS2T: Arbitrary Source-To-Target Adversarial Attack on Speaker Recognition Systems**

AS2T：说话人识别系统的任意源-目标对抗攻击 cs.SD

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03351v1)

**Authors**: Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Yang Liu

**Abstracts**: Recent work has illuminated the vulnerability of speaker recognition systems (SRSs) against adversarial attacks, raising significant security concerns in deploying SRSs. However, they considered only a few settings (e.g., some combinations of source and target speakers), leaving many interesting and important settings in real-world attack scenarios alone. In this work, we present AS2T, the first attack in this domain which covers all the settings, thus allows the adversary to craft adversarial voices using arbitrary source and target speakers for any of three main recognition tasks. Since none of the existing loss functions can be applied to all the settings, we explore many candidate loss functions for each setting including the existing and newly designed ones. We thoroughly evaluate their efficacy and find that some existing loss functions are suboptimal. Then, to improve the robustness of AS2T towards practical over-the-air attack, we study the possible distortions occurred in over-the-air transmission, utilize different transformation functions with different parameters to model those distortions, and incorporate them into the generation of adversarial voices. Our simulated over-the-air evaluation validates the effectiveness of our solution in producing robust adversarial voices which remain effective under various hardware devices and various acoustic environments with different reverberation, ambient noises, and noise levels. Finally, we leverage AS2T to perform thus far the largest-scale evaluation to understand transferability among 14 diverse SRSs. The transferability analysis provides many interesting and useful insights which challenge several findings and conclusion drawn in previous works in the image domain. Our study also sheds light on future directions of adversarial attacks in the speaker recognition domain.

摘要: 最近的工作揭示了说话人识别系统(SRSS)对对手攻击的脆弱性，这引发了人们在部署SRSS时的重大安全担忧。然而，他们只考虑了几个设置(例如，源说话人和目标说话人的一些组合)，将许多有趣和重要的设置留在了现实世界的攻击场景中。在这项工作中，我们提出了AS2T，这是该领域的第一次攻击，覆盖了所有设置，从而允许攻击者使用任意来源和目标说话人来创建敌意语音，用于三个主要识别任务中的任何一个。由于现有的损失函数都不能适用于所有的设置，因此我们探索了每个设置的许多候选损失函数，包括现有的和新设计的损失函数。我们对它们的有效性进行了深入的评估，发现现有的一些损失函数是次优的。然后，为了提高AS2T对实际空中攻击的稳健性，我们研究了空中传输中可能出现的失真，利用不同参数的不同变换函数对这些失真进行建模，并将其融入到对抗声音的生成中。我们的模拟空中评估验证了我们的解决方案在产生健壮的对抗性声音方面的有效性，这些声音在各种硬件设备和具有不同混响、环境噪声和噪声水平的各种声学环境中仍然有效。最后，我们利用AS2T执行到目前为止最大规模的评估，以了解14个不同SRS之间的可转移性。可转移性分析提供了许多有趣和有用的见解，挑战了图像领域以前工作中得出的一些发现和结论。我们的研究也为说话人识别领域未来的对抗性攻击提供了方向。



## **24. Subject Membership Inference Attacks in Federated Learning**

联合学习中的主体成员推理攻击 cs.LG

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03317v1)

**Authors**: Anshuman Suri, Pallika Kanani, Virendra J. Marathe, Daniel W. Peterson

**Abstracts**: Privacy in Federated Learning (FL) is studied at two different granularities: item-level, which protects individual data points, and user-level, which protects each user (participant) in the federation. Nearly all of the private FL literature is dedicated to studying privacy attacks and defenses at these two granularities. Recently, subject-level privacy has emerged as an alternative privacy granularity to protect the privacy of individuals (data subjects) whose data is spread across multiple (organizational) users in cross-silo FL settings. An adversary might be interested in recovering private information about these individuals (a.k.a. \emph{data subjects}) by attacking the trained model. A systematic study of these patterns requires complete control over the federation, which is impossible with real-world datasets. We design a simulator for generating various synthetic federation configurations, enabling us to study how properties of the data, model design and training, and the federation itself impact subject privacy risk. We propose three attacks for \emph{subject membership inference} and examine the interplay between all factors within a federation that affect the attacks' efficacy. We also investigate the effectiveness of Differential Privacy in mitigating this threat. Our takeaways generalize to real-world datasets like FEMNIST, giving credence to our findings.

摘要: 联合学习(FL)中的隐私在两个不同的粒度上进行研究：项级和用户级，前者保护单个数据点，后者保护联合中的每个用户(参与者)。几乎所有的私人FL文献都致力于在这两个粒度上研究隐私攻击和防御。最近，主题级别隐私已经作为一种替代隐私粒度出现，以保护其数据在跨竖井FL设置中跨多个(组织)用户分布的个人(数据主体)的隐私。对手可能对恢复这些个人的私人信息感兴趣(也称为。\emph{数据主题})攻击训练的模型。对这些模式的系统研究需要完全控制联邦，这在现实世界的数据集中是不可能的。我们设计了一个模拟器来生成各种合成联邦配置，使我们能够研究数据的属性、模型设计和训练以及联邦本身如何影响主体隐私风险。我们提出了三种针对主体成员关系推理的攻击，并考察了影响攻击效果的联邦内所有因素之间的相互作用。我们还研究了差异隐私在缓解这一威胁方面的有效性。我们的结论是推广到像FEMNIST这样的真实世界数据集，这让我们的发现更可信。



## **25. Quickest Change Detection in the Presence of Transient Adversarial Attacks**

存在瞬时敌意攻击时的最快变化检测 eess.SP

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03245v1)

**Authors**: Thirupathaiah Vasantam, Don Towsley, Venugopal V. Veeravalli

**Abstracts**: We study a monitoring system in which the distributions of sensors' observations change from a nominal distribution to an abnormal distribution in response to an adversary's presence. The system uses the quickest change detection procedure, the Shewhart rule, to detect the adversary that uses its resources to affect the abnormal distribution, so as to hide its presence. The metric of interest is the probability of missed detection within a predefined number of time-slots after the changepoint. Assuming that the adversary's resource constraints are known to the detector, we find the number of required sensors to make the worst-case probability of missed detection less than an acceptable level. The distributions of observations are assumed to be Gaussian, and the presence of the adversary affects their mean. We also provide simulation results to support our analysis.

摘要: 我们研究了一个监测系统，其中传感器的观测值的分布随着对手的出现而从名义分布变为非正常分布。该系统使用最快的变化检测过程--休哈特规则来检测利用其资源影响异常分布的对手，从而隐藏其存在。感兴趣的度量是在变化点之后的预定数量的时隙内遗漏检测的概率。假设检测器知道对手的资源限制，我们找到使最坏情况下的漏检概率小于可接受水平所需的传感器数量。观测值的分布被假定为高斯分布，而对手的存在会影响其平均值。我们还提供了仿真结果来支持我们的分析。



## **26. Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning**

基于可解释深度强化学习的无人机制导规划鲁棒对抗攻击检测 cs.LG

13 pages, 20 figures

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.02670v2)

**Authors**: Thomas Hickling, Nabil Aouf, Phillippa Spencer

**Abstracts**: The danger of adversarial attacks to unprotected Uncrewed Aerial Vehicle (UAV) agents operating in public is growing. Adopting AI-based techniques and more specifically Deep Learning (DL) approaches to control and guide these UAVs can be beneficial in terms of performance but add more concerns regarding the safety of those techniques and their vulnerability against adversarial attacks causing the chances of collisions going up as the agent becomes confused. This paper proposes an innovative approach based on the explainability of DL methods to build an efficient detector that will protect these DL schemes and thus the UAVs adopting them from potential attacks. The agent is adopting a Deep Reinforcement Learning (DRL) scheme for guidance and planning. It is formed and trained with a Deep Deterministic Policy Gradient (DDPG) with Prioritised Experience Replay (PER) DRL scheme that utilises Artificial Potential Field (APF) to improve training times and obstacle avoidance performance. The adversarial attacks are generated by Fast Gradient Sign Method (FGSM) and Basic Iterative Method (BIM) algorithms and reduced obstacle course completion rates from 80\% to 35\%. A Realistic Synthetic environment for UAV explainable DRL based planning and guidance including obstacles and adversarial attacks is built. Two adversarial attack detectors are proposed. The first one adopts a Convolutional Neural Network (CNN) architecture and achieves an accuracy in detection of 80\%. The second detector is developed based on a Long Short Term Memory (LSTM) network and achieves an accuracy of 91\% with much faster computing times when compared to the CNN based detector.

摘要: 对在公共场合工作的无保护无人驾驶飞行器(UAV)特工进行敌意攻击的危险正在增加。采用基于人工智能的技术，更具体地说，深度学习(DL)方法来控制和引导这些无人机，在性能方面可能是有益的，但也增加了人们对这些技术的安全性及其对抗对手攻击的脆弱性的更多担忧，随着代理变得困惑，碰撞的可能性会增加。本文提出了一种基于DL方法的可解释性的创新方法，以构建一个有效的检测器来保护这些DL方案，从而保护采用这些方案的无人机免受潜在的攻击。该代理正在采用深度强化学习(DRL)方案来指导和规划。它是利用深度确定性策略梯度(DDPG)和优先经验重播(PER)DRL方案形成和训练的，该方案利用人工势场(APF)来改进训练时间和避障性能。采用快速梯度符号法(FGSM)和基本迭代法(BIM)算法生成对抗性攻击，将障碍路径完成率从80%降低到35%。建立了包括障碍物和对抗性攻击在内的基于DRL的无人机可解释规划和制导的现实综合环境。提出了两种对抗性攻击检测器。第一种方法采用卷积神经网络(CNN)结构，检测精度达到80%。第二种检测器是基于长短期记忆(LSTM)网络开发的，与基于CNN的检测器相比，具有91%的精度和更快的计算时间。



## **27. VLC Physical Layer Security through RIS-aided Jamming Receiver for 6G Wireless Networks**

基于RIS辅助干扰接收机的6G无线网络VLC物理层安全 cs.CR

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2205.09026v2)

**Authors**: Simone Soderi, Alessandro Brighente, Federico Turrin, Mauro Conti

**Abstracts**: Visible Light Communication (VLC) is one the most promising enabling technology for future 6G networks to overcome Radio-Frequency (RF)-based communication limitations thanks to a broader bandwidth, higher data rate, and greater efficiency. However, from the security perspective, VLCs suffer from all known wireless communication security threats (e.g., eavesdropping and integrity attacks). For this reason, security researchers are proposing innovative Physical Layer Security (PLS) solutions to protect such communication. Among the different solutions, the novel Reflective Intelligent Surface (RIS) technology coupled with VLCs has been successfully demonstrated in recent work to improve the VLC communication capacity. However, to date, the literature still lacks analysis and solutions to show the PLS capability of RIS-based VLC communication. In this paper, we combine watermarking and jamming primitives through the Watermark Blind Physical Layer Security (WBPLSec) algorithm to secure VLC communication at the physical layer. Our solution leverages RIS technology to improve the security properties of the communication. By using an optimization framework, we can calculate RIS phases to maximize the WBPLSec jamming interference schema over a predefined area in the room. In particular, compared to a scenario without RIS, our solution improves the performance in terms of secrecy capacity without any assumption about the adversary's location. We validate through numerical evaluations the positive impact of RIS-aided solution to increase the secrecy capacity of the legitimate jamming receiver in a VLC indoor scenario. Our results show that the introduction of RIS technology extends the area where secure communication occurs and that by increasing the number of RIS elements the outage probability decreases.

摘要: 可见光通信(VLC)是未来6G网络最有前途的使能技术之一，可以克服基于射频(RF)的通信限制，因为它具有更宽的带宽、更高的数据速率和更高的效率。然而，从安全的角度来看，VLC受到所有已知的无线通信安全威胁(例如，窃听和完整性攻击)。为此，安全研究人员提出了创新的物理层安全(PLS)解决方案来保护此类通信。在不同的解决方案中，新型的反射智能表面(RIS)技术与VLC相结合已经在最近的工作中被成功地展示出来，以提高VLC的通信容量。然而，到目前为止，文献仍然缺乏分析和解决方案来展示基于RIS的VLC通信的偏最小二乘能力。在本文中，我们通过水印盲物理层安全(WBPLSec)算法将水印和干扰基元相结合来保护物理层的VLC通信。我们的解决方案利用RIS技术来提高通信的安全属性。通过使用优化框架，我们可以计算RIS相位，以最大化房间中预定义区域内的WBPLSec干扰方案。特别是，与没有RIS的场景相比，我们的方案在保密能力方面提高了性能，而不需要假设对手的位置。我们通过数值评估验证了RIS辅助解决方案对提高VLC室内场景中合法干扰接收机的保密容量的积极影响。我们的结果表明，RIS技术的引入扩展了安全通信发生的区域，并且随着RIS单元数量的增加，中断概率降低。



## **28. Sampling without Replacement Leads to Faster Rates in Finite-Sum Minimax Optimization**

有限和极小极大优化中无替换抽样的快速算法 math.OC

48 pages, 3 figures

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.02953v1)

**Authors**: Aniket Das, Bernhard Schölkopf, Michael Muehlebach

**Abstracts**: We analyze the convergence rates of stochastic gradient algorithms for smooth finite-sum minimax optimization and show that, for many such algorithms, sampling the data points without replacement leads to faster convergence compared to sampling with replacement. For the smooth and strongly convex-strongly concave setting, we consider gradient descent ascent and the proximal point method, and present a unified analysis of two popular without-replacement sampling strategies, namely Random Reshuffling (RR), which shuffles the data every epoch, and Single Shuffling or Shuffle Once (SO), which shuffles only at the beginning. We obtain tight convergence rates for RR and SO and demonstrate that these strategies lead to faster convergence than uniform sampling. Moving beyond convexity, we obtain similar results for smooth nonconvex-nonconcave objectives satisfying a two-sided Polyak-{\L}ojasiewicz inequality. Finally, we demonstrate that our techniques are general enough to analyze the effect of data-ordering attacks, where an adversary manipulates the order in which data points are supplied to the optimizer. Our analysis also recovers tight rates for the incremental gradient method, where the data points are not shuffled at all.

摘要: 我们分析了光滑有限和极大极小优化问题的随机梯度算法的收敛速度，并证明了对于许多这类算法，对数据点进行不替换采样比用替换采样可以更快地收敛。对于光滑和强凸-强凹的情况，我们考虑了梯度下降上升和近似点方法，并对两种流行的无替换抽样策略进行了统一的分析，即随机重洗(RR)和单次洗牌(SO)。随机重洗(RR)是每一个时期都要洗牌的抽样策略，而单次洗牌(SO)是只在开始洗牌的抽样策略。我们得到了RR和SO的紧收敛速度，并证明了这两种策略比均匀抽样的收敛速度更快。超越凸性，我们得到了满足双边Polyak-L ojasiewicz不等式的光滑非凸-非凹目标的类似结果。最后，我们演示了我们的技术足够通用，可以分析数据排序攻击的影响，在这种攻击中，对手操纵向优化器提供数据点的顺序。我们的分析还恢复了增量梯度法的紧缩率，在这种方法中，数据点根本没有被洗牌。



## **29. A Robust Deep Learning Enabled Semantic Communication System for Text**

一种支持深度学习的健壮文本语义交流系统 eess.SP

6 pages

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02596v1)

**Authors**: Xiang Peng, Zhijin Qin, Danlan Huang, Xiaoming Tao, Jianhua Lu, Guangyi Liu, Chengkang Pan

**Abstracts**: With the advent of the 6G era, the concept of semantic communication has attracted increasing attention. Compared with conventional communication systems, semantic communication systems are not only affected by physical noise existing in the wireless communication environment, e.g., additional white Gaussian noise, but also by semantic noise due to the source and the nature of deep learning-based systems. In this paper, we elaborate on the mechanism of semantic noise. In particular, we categorize semantic noise into two categories: literal semantic noise and adversarial semantic noise. The former is caused by written errors or expression ambiguity, while the latter is caused by perturbations or attacks added to the embedding layer via the semantic channel. To prevent semantic noise from influencing semantic communication systems, we present a robust deep learning enabled semantic communication system (R-DeepSC) that leverages a calibrated self-attention mechanism and adversarial training to tackle semantic noise. Compared with baseline models that only consider physical noise for text transmission, the proposed R-DeepSC achieves remarkable performance in dealing with semantic noise under different signal-to-noise ratios.

摘要: 随着6G时代的到来，语义沟通的概念越来越受到关注。与传统的通信系统相比，语义通信系统不仅受到无线通信环境中存在的物理噪声(例如附加的高斯白噪声)的影响，而且由于基于深度学习的系统的来源和性质而受到语义噪声的影响。本文对语义噪声的产生机制进行了详细的阐述。特别地，我们将语义噪声分为两类：字面语义噪声和对抗性语义噪声。前者是由书面错误或表达歧义引起的，后者是通过语义通道对嵌入层进行扰动或攻击造成的。为了防止语义噪声对语义通信系统的影响，我们提出了一种健壮的深度学习语义通信系统(R-DeepSC)，该系统利用校准的自我注意机制和对抗性训练来应对语义噪声。与仅考虑物理噪声的文本传输基线模型相比，R-DeepSC在处理不同信噪比下的语义噪声方面取得了显著的性能。



## **30. Certified Robustness in Federated Learning**

联合学习中的认证稳健性 cs.LG

17 pages, 10 figures. Code available at  https://github.com/MotasemAlfarra/federated-learning-with-pytorch

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02535v1)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstracts**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(\ie personalized) models, and find that the robustness of local models degrades as they diverge from the global model

摘要: 由于联邦学习在训练分布式数据上的机器学习模型方面的有效性，它最近获得了极大的关注和普及。然而，与单节点监督学习设置中一样，在联合学习中训练的模型容易受到称为对抗性攻击的不可察觉的输入转换的影响，从而质疑其在安全相关应用中的部署。在这项工作中，我们研究了联合训练、个性化和经过认证的健壮性之间的相互作用。特别是，我们采用了随机化平滑，这是一种广泛使用和可扩展的认证方法，用于认证在联合设置上训练的深层网络不受输入扰动和转换的影响。我们发现，与仅基于本地数据进行训练相比，简单的联合平均技术不仅在建立更准确的模型方面是有效的，而且在可证明的健壮性方面也更有效。我们进一步分析了个性化，这是联合训练中的一种流行技术，它增加了模型对本地数据的偏差，并对稳健性进行了分析。我们展示了个性化比这两者(即只在本地数据上训练和联合训练)在建立更健壮的模型和更快的训练方面的几个优势。最后，我们研究了全局模型和局部模型的混合模型的稳健性，发现局部模型的稳健性随着偏离全局模型而降低



## **31. Fast Adversarial Training with Adaptive Step Size**

步长自适应的快速对抗性训练 cs.LG

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2206.02417v1)

**Authors**: Zhichao Huang, Yanbo Fan, Chen Liu, Weizhong Zhang, Yong Zhang, Mathieu Salzmann, Sabine Süsstrunk, Jue Wang

**Abstracts**: While adversarial training and its variants have shown to be the most effective algorithms to defend against adversarial attacks, their extremely slow training process makes it hard to scale to large datasets like ImageNet. The key idea of recent works to accelerate adversarial training is to substitute multi-step attacks (e.g., PGD) with single-step attacks (e.g., FGSM). However, these single-step methods suffer from catastrophic overfitting, where the accuracy against PGD attack suddenly drops to nearly 0% during training, destroying the robustness of the networks. In this work, we study the phenomenon from the perspective of training instances. We show that catastrophic overfitting is instance-dependent and fitting instances with larger gradient norm is more likely to cause catastrophic overfitting. Based on our findings, we propose a simple but effective method, Adversarial Training with Adaptive Step size (ATAS). ATAS learns an instancewise adaptive step size that is inversely proportional to its gradient norm. The theoretical analysis shows that ATAS converges faster than the commonly adopted non-adaptive counterparts. Empirically, ATAS consistently mitigates catastrophic overfitting and achieves higher robust accuracy on CIFAR10, CIFAR100 and ImageNet when evaluated on various adversarial budgets.

摘要: 尽管对抗性训练及其变体已被证明是防御对抗性攻击的最有效算法，但它们的训练过程极其缓慢，很难扩展到像ImageNet这样的大型数据集。最近加速对抗性训练的工作的关键思想是用单步攻击(例如FGSM)代替多步攻击(例如PGD)。然而，这些单步方法存在灾难性的过拟合问题，在训练过程中对PGD攻击的准确率突然下降到近0%，破坏了网络的健壮性。在这项工作中，我们从训练实例的角度来研究这一现象。我们发现，灾难性过拟合是依赖于实例的，并且具有较大梯度范数的拟合实例更有可能导致灾难性过拟合。基于我们的研究结果，我们提出了一种简单而有效的方法--自适应步长对抗性训练(ATAS)。ATAS学习一种与其梯度范数成反比的实例化自适应步长。理论分析表明，与常用的非自适应算法相比，ATAS的收敛速度更快。从经验上看，ATAS一致地缓解了灾难性的过拟合，并在CIFAR10、CIFAR100和ImageNet上实现了更高的稳健精度，当在各种对抗性预算上进行评估时。



## **32. The art of defense: letting networks fool the attacker**

防御艺术：让网络愚弄攻击者 cs.CV

**SubmitDate**: 2022-06-06    [paper-pdf](http://arxiv.org/pdf/2104.02963v3)

**Authors**: Jinlai Zhang, Yinpeng Dong, Binbin Liu, Bo Ouyang, Jihong Zhu, Minchi Kuang, Houqing Wang, Yanmei Meng

**Abstracts**: Robust environment perception is critical for autonomous cars, and adversarial defenses are the most effective and widely studied ways to improve the robustness of environment perception. However, all of previous defense methods decrease the natural accuracy, and the nature of the DNNs itself has been overlooked. To this end, in this paper, we propose a novel adversarial defense for 3D point cloud classifier that makes full use of the nature of the DNNs. Due to the disorder of point cloud, all point cloud classifiers have the property of permutation invariant to the input point cloud. Based on this nature, we design invariant transformations defense (IT-Defense). We show that, even after accounting for obfuscated gradients, our IT-Defense is a resilient defense against state-of-the-art (SOTA) 3D attacks. Moreover, IT-Defense do not hurt clean accuracy compared to previous SOTA 3D defenses. Our code is available at: {\footnotesize{\url{https://github.com/cuge1995/IT-Defense}}}.

摘要: 稳健的环境感知是自动驾驶汽车的关键，而对抗防御是提高环境感知健壮性的最有效和被广泛研究的方法。然而，以往的防御方法都降低了DNN的自然准确率，并且忽略了DNN本身的性质。为此，在本文中，我们提出了一种新的针对三维点云分类器的对抗性防御方案，该方案充分利用了DNN的性质。由于点云的无序性，所有的点云分类器都具有对输入点云的置换不变性。基于这一性质，我们设计了不变变换防御(IT-Defense)。我们表明，即使在考虑模糊梯度之后，我们的IT防御也是针对最先进的(SOTA)3D攻击的弹性防御。此外，与以前的Sota 3D防御相比，IT防御不会损害干净的准确性。我们的代码请访问：{\footnotesize{\url{https://github.com/cuge1995/IT-Defense}}}.



## **33. Quantized and Distributed Subgradient Optimization Method with Malicious Attack**

具有恶意攻击的量化分布式次梯度优化方法 math.OC

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02272v1)

**Authors**: Iyanuoluwa Emiola, Chinwendu Enyioha

**Abstracts**: This paper considers a distributed optimization problem in a multi-agent system where a fraction of the agents act in an adversarial manner. Specifically, the malicious agents steer the network of agents away from the optimal solution by sending false information to their neighbors and consume significant bandwidth in the communication process. We propose a distributed gradient-based optimization algorithm in which the non-malicious agents exchange quantized information with one another. We prove convergence of the solution to a neighborhood of the optimal solution, and characterize the solutions obtained under the communication-constrained environment and presence of malicious agents. Numerical simulations to illustrate the results are also presented.

摘要: 本文考虑多智能体系统中的分布式优化问题，其中部分智能体以对抗性的方式行动。具体地说，恶意代理通过向其邻居发送虚假信息来引导代理网络远离最佳解决方案，并在通信过程中消耗大量带宽。我们提出了一种基于梯度的分布式优化算法，在该算法中非恶意代理之间交换量化信息。我们证明了解收敛到最优解的一个邻域，并刻画了在通信受限环境和恶意代理存在的情况下所得到的解。文中还给出了数值模拟结果。



## **34. Vanilla Feature Distillation for Improving the Accuracy-Robustness Trade-Off in Adversarial Training**

在对抗性训练中提高精确度和稳健性权衡的普通特征提取 cs.CV

12 pages

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02158v1)

**Authors**: Guodong Cao, Zhibo Wang, Xiaowei Dong, Zhifei Zhang, Hengchang Guo, Zhan Qin, Kui Ren

**Abstracts**: Adversarial training has been widely explored for mitigating attacks against deep models. However, most existing works are still trapped in the dilemma between higher accuracy and stronger robustness since they tend to fit a model towards robust features (not easily tampered with by adversaries) while ignoring those non-robust but highly predictive features. To achieve a better robustness-accuracy trade-off, we propose the Vanilla Feature Distillation Adversarial Training (VFD-Adv), which conducts knowledge distillation from a pre-trained model (optimized towards high accuracy) to guide adversarial training towards higher accuracy, i.e., preserving those non-robust but predictive features. More specifically, both adversarial examples and their clean counterparts are forced to be aligned in the feature space by distilling predictive representations from the pre-trained/clean model, while previous works barely utilize predictive features from clean models. Therefore, the adversarial training model is updated towards maximally preserving the accuracy as gaining robustness. A key advantage of our method is that it can be universally adapted to and boost existing works. Exhaustive experiments on various datasets, classification models, and adversarial training algorithms demonstrate the effectiveness of our proposed method.

摘要: 对抗性训练已被广泛探索用于减轻对深度模型的攻击。然而，大多数现有的工作仍然陷于更高的准确率和更强的稳健性之间的两难境地，因为它们倾向于向健壮的特征(不易被对手篡改)拟合模型，而忽略了那些非健壮但高度预测的特征。为了达到更好的稳健性和精确度之间的权衡，我们提出了Vanilla特征提取对抗训练(VFD-ADV)，它从预先训练的模型中进行知识提取(向高准确度优化)，以引导对抗训练朝着更高的准确率方向发展，即保留那些非稳健但可预测的特征。更具体地说，通过从预先训练/干净的模型中提取预测表示，迫使对抗性例子和它们的干净例子在特征空间中对齐，而以前的工作几乎不利用来自干净模型的预测特征。因此，对抗性训练模型在获得稳健性的同时，朝着最大限度地保持准确性的方向更新。我们方法的一个关键优势是它可以普遍适用于并促进现有的工作。在各种数据集、分类模型和对抗性训练算法上的详尽实验证明了该方法的有效性。



## **35. Federated Adversarial Training with Transformers**

与变形金刚进行联合对抗性训练 cs.LG

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2206.02131v1)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Olivier Déforges

**Abstracts**: Federated learning (FL) has emerged to enable global model training over distributed clients' data while preserving its privacy. However, the global trained model is vulnerable to the evasion attacks especially, the adversarial examples (AEs), carefully crafted samples to yield false classification. Adversarial training (AT) is found to be the most promising approach against evasion attacks and it is widely studied for convolutional neural network (CNN). Recently, vision transformers have been found to be effective in many computer vision tasks. To the best of the authors' knowledge, there is no work that studied the feasibility of AT in a FL process for vision transformers. This paper investigates such feasibility with different federated model aggregation methods and different vision transformer models with different tokenization and classification head techniques. In order to improve the robust accuracy of the models with the not independent and identically distributed (Non-IID), we propose an extension to FedAvg aggregation method, called FedWAvg. By measuring the similarities between the last layer of the global model and the last layer of the client updates, FedWAvg calculates the weights to aggregate the local models updates. The experiments show that FedWAvg improves the robust accuracy when compared with other state-of-the-art aggregation methods.

摘要: 联合学习(FL)已经出现，以实现对分布式客户数据的全局模型训练，同时保护其隐私。然而，全局训练的模型很容易受到逃避攻击，尤其是对抗性例子(AEs)，精心制作的样本会产生错误的分类。对抗训练(AT)被认为是对抗逃避攻击的最有前途的方法，卷积神经网络(CNN)对其进行了广泛的研究。最近，视觉转换器被发现在许多计算机视觉任务中是有效的。就作者所知，还没有研究在视觉转换器的FL过程中AT的可行性的工作。本文采用不同的联邦模型聚合方法和不同标记化和分类头技术的视觉转换器模型，研究了这种方法的可行性。为了提高非独立同分布(Non-IID)模型的稳健精度，提出了一种扩展的FedAvg集结方法，称为FedWAvg。通过测量全局模型的最后一层和客户端更新的最后一层之间的相似性，FedWAvg计算权重以聚合本地模型更新。实验表明，FedWAvg与其他最先进的聚合方法相比，提高了健壮性。



## **36. Data-Efficient Backdoor Attacks**

数据高效的后门攻击 cs.CV

Accepted to IJCAI 2022 Long Oral

**SubmitDate**: 2022-06-05    [paper-pdf](http://arxiv.org/pdf/2204.12281v2)

**Authors**: Pengfei Xia, Ziqiang Li, Wei Zhang, Bin Li

**Abstracts**: Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-and-Updating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability. The prototype code of our method is now available at https://github.com/xpf/Data-Efficient-Backdoor-Attacks.

摘要: 最近的研究证明，深度神经网络很容易受到后门攻击。具体地说，通过将少量有毒样本混合到训练集中，可以恶意控制训练模型的行为。现有的攻击方法通过从良性集合中随机选择一些干净的数据，然后在其中嵌入触发器来构建这样的攻击者。然而，这种选择策略忽略了这样一个事实，即每个有毒样本对后门注入的贡献是不相等的，这降低了中毒的效率。在本文中，我们将通过选择来提高有毒数据效率的问题描述为一个优化问题，并提出了一种过滤和更新策略(FUS)来解决该问题。在CIFAR-10和ImageNet-10上的实验结果表明，该方法是有效的：与随机选择策略相比，只需47%~75%的中毒样本量即可获得相同的攻击成功率。更重要的是，根据一种设置选择的对手可以很好地推广到其他设置，表现出很强的可转移性。我们方法的原型代码现已在https://github.com/xpf/Data-Efficient-Backdoor-Attacks.上提供



## **37. Connecting adversarial attacks and optimal transport for domain adaptation**

连接对抗性攻击和最优传输以实现域自适应 cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2205.15424v2)

**Authors**: Arip Asadulaev, Vitaly Shutov, Alexander Korotin, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: We present a novel algorithm for domain adaptation using optimal transport. In domain adaptation, the goal is to adapt a classifier trained on the source domain samples to the target domain. In our method, we use optimal transport to map target samples to the domain named source fiction. This domain differs from the source but is accurately classified by the source domain classifier. Our main idea is to generate a source fiction by c-cyclically monotone transformation over the target domain. If samples with the same labels in two domains are c-cyclically monotone, the optimal transport map between these domains preserves the class-wise structure, which is the main goal of domain adaptation. To generate a source fiction domain, we propose an algorithm that is based on our finding that adversarial attacks are a c-cyclically monotone transformation of the dataset. We conduct experiments on Digits and Modern Office-31 datasets and achieve improvement in performance for simple discrete optimal transport solvers for all adaptation tasks.

摘要: 我们提出了一种新的基于最优传输的域自适应算法。在领域自适应中，目标是使在源域样本上训练的分类器适应于目标域。在我们的方法中，我们使用最优传输将目标样本映射到名为源虚构的域。此域与源不同，但源域分类器会对其进行准确分类。我们的主要思想是通过目标域上的c-循环单调变换来生成源小说。如果两个结构域中具有相同标记的样本是c-循环单调的，那么这些结构域之间的最优传输映射保持了类结构，这是结构域适应的主要目标。为了生成源虚构领域，我们提出了一种算法，该算法基于我们的发现，即对抗性攻击是数据集的c循环单调变换。我们在Digits和现代Office-31数据集上进行了实验，并在所有适应任务的简单离散最优传输求解器的性能上取得了改进。



## **38. A General Framework for Evaluating Robustness of Combinatorial Optimization Solvers on Graphs**

评价图上组合优化求解器稳健性的通用框架 math.OC

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2201.00402v2)

**Authors**: Han Lu, Zenan Li, Runzhong Wang, Qibing Ren, Junchi Yan, Xiaokang Yang

**Abstracts**: Solving combinatorial optimization (CO) on graphs is among the fundamental tasks for upper-stream applications in data mining, machine learning and operations research. Despite the inherent NP-hard challenge for CO, heuristics, branch-and-bound, learning-based solvers are developed to tackle CO problems as accurately as possible given limited time budgets. However, a practical metric for the sensitivity of CO solvers remains largely unexplored. Existing theoretical metrics require the optimal solution which is infeasible, and the gradient-based adversarial attack metric from deep learning is not compatible with non-learning solvers that are usually non-differentiable. In this paper, we develop the first practically feasible robustness metric for general combinatorial optimization solvers. We develop a no worse optimal cost guarantee thus do not require optimal solutions, and we tackle the non-differentiable challenge by resorting to black-box adversarial attack methods. Extensive experiments are conducted on 14 unique combinations of solvers and CO problems, and we demonstrate that the performance of state-of-the-art solvers like Gurobi can degenerate by over 20% under the given time limit bound on the hard instances discovered by our robustness metric, raising concerns about the robustness of combinatorial optimization solvers.

摘要: 求解图上的组合优化问题是数据挖掘、机器学习和运筹学中上游应用的基本任务之一。尽管CO存在固有的NP-Hard挑战，但启发式、分支定界、基于学习的求解器被开发出来，以在有限的时间预算内尽可能准确地处理CO问题。然而，CO解算器灵敏度的实用指标在很大程度上仍未被探索。现有的理论度量要求最优解是不可行的，基于深度学习的基于梯度的敌意攻击度量与通常不可微的非学习求解器不兼容。在这篇文章中，我们为一般的组合优化求解器发展了第一个实用可行的稳健性度量。我们开发了一个不会更差的最优成本保证，因此不需要最优解决方案，我们通过求助于黑箱对抗性攻击方法来应对不可区分的挑战。在14个独特的求解器和CO问题组合上进行了广泛的实验，我们证明了最先进的求解器，如Gurobi，在我们的健壮性度量发现的困难实例上，在给定的时间限制下，性能可以退化超过20%，这引起了人们对组合优化求解器的健壮性的担忧。



## **39. Guided Diffusion Model for Adversarial Purification**

对抗性净化中的引导扩散模型 cs.CV

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2205.14969v2)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.

摘要: 随着深度神经网络(DNN)在各种算法和框架中的广泛应用，安全威胁已成为人们关注的问题之一。对抗性攻击干扰了基于DNN的图像分类器，攻击者可以故意在输入图像上添加不可察觉的对抗性扰动来愚弄分类器。在本文中，我们提出了一种新的净化方法，称为引导扩散净化模型(GDMP)，以帮助保护分类器免受对手攻击。该方法的核心是将净化嵌入到去噪扩散概率模型(DDPM)的扩散去噪过程中，使其扩散过程能够淹没带有逐渐增加的高斯噪声的对抗性扰动，并在引导去噪过程后同时去除这两种噪声。在不同数据集上的综合实验表明，所提出的GDMP将对抗性攻击引起的扰动减少到较小的范围，从而显著提高了分类的正确性。GDMP在CIFAR10数据集上的稳健准确率提高了5%，在PGD攻击下达到了90.1%。此外，GDMP在具有挑战性的ImageNet数据集上获得了70.94%的健壮性。



## **40. Soft Adversarial Training Can Retain Natural Accuracy**

软对抗训练可以保持自然的准确性 cs.LG

7 pages, 6 figures

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01904v1)

**Authors**: Abhijith Sharma, Apurva Narayan

**Abstracts**: Adversarial training for neural networks has been in the limelight in recent years. The advancement in neural network architectures over the last decade has led to significant improvement in their performance. It sparked an interest in their deployment for real-time applications. This process initiated the need to understand the vulnerability of these models to adversarial attacks. It is instrumental in designing models that are robust against adversaries. Recent works have proposed novel techniques to counter the adversaries, most often sacrificing natural accuracy. Most suggest training with an adversarial version of the inputs, constantly moving away from the original distribution. The focus of our work is to use abstract certification to extract a subset of inputs for (hence we call it 'soft') adversarial training. We propose a training framework that can retain natural accuracy without sacrificing robustness in a constrained setting. Our framework specifically targets moderately critical applications which require a reasonable balance between robustness and accuracy. The results testify to the idea of soft adversarial training for the defense against adversarial attacks. At last, we propose the scope of future work for further improvement of this framework.

摘要: 近年来，神经网络的对抗性训练一直是人们关注的焦点。在过去的十年中，神经网络结构的进步导致了它们的性能的显著提高。这引发了人们对它们在实时应用程序中的部署的兴趣。这一过程引发了了解这些模型在对抗攻击中的脆弱性的需要。它在设计对对手具有健壮性的模型方面很有帮助。最近的作品提出了新的技术来对抗对手，最常见的是牺牲了自然的准确性。大多数人建议使用对抗性版本的投入进行培训，不断远离原始分布。我们的工作重点是使用抽象认证来提取对抗性训练的输入子集(因此我们称之为“软”)。我们提出了一种训练框架，它可以在约束环境下保持自然的准确性，而不会牺牲鲁棒性。我们的框架专门针对需要在健壮性和准确性之间取得合理平衡的中等关键应用程序。结果证明了软对抗性训练对对抗攻击的防御思想。最后，对该框架的进一步完善提出了下一步的工作范围。



## **41. Saliency Attack: Towards Imperceptible Black-box Adversarial Attack**

突显攻击：向潜伏的黑盒对抗性攻击 cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01898v1)

**Authors**: Zeyu Dai, Shengcai Liu, Ke Tang, Qing Li

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, even in the black-box setting where the attacker is only accessible to the model output. Recent studies have devised effective black-box attacks with high query efficiency. However, such performance is often accompanied by compromises in attack imperceptibility, hindering the practical use of these approaches. In this paper, we propose to restrict the perturbations to a small salient region to generate adversarial examples that can hardly be perceived. This approach is readily compatible with many existing black-box attacks and can significantly improve their imperceptibility with little degradation in attack success rate. Further, we propose the Saliency Attack, a new black-box attack aiming to refine the perturbations in the salient region to achieve even better imperceptibility. Extensive experiments show that compared to the state-of-the-art black-box attacks, our approach achieves much better imperceptibility scores, including most apparent distortion (MAD), $L_0$ and $L_2$ distances, and also obtains significantly higher success rates judged by a human-like threshold on MAD. Importantly, the perturbations generated by our approach are interpretable to some extent. Finally, it is also demonstrated to be robust to different detection-based defenses.

摘要: 深度神经网络很容易受到敌意例子的攻击，即使在攻击者只能通过模型输出访问的黑盒环境中也是如此。最近的研究已经设计出有效的黑盒攻击，具有很高的查询效率。然而，这样的表现往往伴随着攻击隐蔽性的妥协，阻碍了这些方法的实际使用。在本文中，我们建议将扰动限制在一个很小的显著区域内，以产生难以察觉的对抗性例子。这种方法很容易与许多现有的黑盒攻击兼容，并且可以在几乎不降低攻击成功率的情况下显著提高它们的隐蔽性。此外，我们提出了显著攻击，这是一种新的黑盒攻击，旨在细化显著区域的扰动，以获得更好的不可见性。大量的实验表明，与最新的黑盒攻击相比，我们的方法获得了更好的不可见性分数，包括最明显失真(MAD)、$L0$和$L2$距离，并且以MAD上类似人类的阈值来判断成功率。重要的是，我们的方法产生的扰动在某种程度上是可以解释的。最后，还证明了该算法对不同的基于检测的防御具有较强的鲁棒性。



## **42. Reward Poisoning Attacks on Offline Multi-Agent Reinforcement Learning**

基于离线多智能体强化学习的奖励毒化攻击 cs.LG

**SubmitDate**: 2022-06-04    [paper-pdf](http://arxiv.org/pdf/2206.01888v1)

**Authors**: Young Wu, Jermey McMahan, Xiaojin Zhu, Qiaomin Xie

**Abstracts**: We expose the danger of reward poisoning in offline multi-agent reinforcement learning (MARL), whereby an attacker can modify the reward vectors to different learners in an offline data set while incurring a poisoning cost. Based on the poisoned data set, all rational learners using some confidence-bound-based MARL algorithm will infer that a target policy - chosen by the attacker and not necessarily a solution concept originally - is the Markov perfect dominant strategy equilibrium for the underlying Markov Game, hence they will adopt this potentially damaging target policy in the future. We characterize the exact conditions under which the attacker can install a target policy. We further show how the attacker can formulate a linear program to minimize its poisoning cost. Our work shows the need for robust MARL against adversarial attacks.

摘要: 我们揭示了离线多智能体强化学习(MAIL)中奖励中毒的危险，即攻击者可以在离线数据集中修改不同学习者的奖励向量，同时招致中毒成本。基于中毒数据集，所有使用基于置信度的Marl算法的理性学习者都会推断，由攻击者选择的目标策略-最初不一定是解的概念-是潜在马尔可夫博弈的马尔可夫完美支配策略均衡，因此他们将在未来采用这种潜在的破坏性目标策略。我们描述了攻击者可以安装目标策略的确切条件。我们进一步展示了攻击者如何制定一个线性规划来最小化其中毒成本。我们的工作表明，需要健壮的Marl来抵御对手攻击。



## **43. Kallima: A Clean-label Framework for Textual Backdoor Attacks**

Kallima：一种针对文本后门攻击的干净标签框架 cs.CR

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01832v1)

**Authors**: Xiaoyi Chen, Yinpeng Dong, Zeyu Sun, Shengfang Zhai, Qingni Shen, Zhonghai Wu

**Abstracts**: Although Deep Neural Network (DNN) has led to unprecedented progress in various natural language processing (NLP) tasks, research shows that deep models are extremely vulnerable to backdoor attacks. The existing backdoor attacks mainly inject a small number of poisoned samples into the training dataset with the labels changed to the target one. Such mislabeled samples would raise suspicion upon human inspection, potentially revealing the attack. To improve the stealthiness of textual backdoor attacks, we propose the first clean-label framework Kallima for synthesizing mimesis-style backdoor samples to develop insidious textual backdoor attacks. We modify inputs belonging to the target class with adversarial perturbations, making the model rely more on the backdoor trigger. Our framework is compatible with most existing backdoor triggers. The experimental results on three benchmark datasets demonstrate the effectiveness of the proposed method.

摘要: 尽管深度神经网络(DNN)在各种自然语言处理(NLP)任务中取得了前所未有的进步，但研究表明，深度模型极易受到后门攻击。现有的后门攻击主要是将少量有毒样本注入训练数据集，并将标签更改为目标样本。这种贴错标签的样本在人工检查时会引起怀疑，可能会揭示攻击。为了提高文本后门攻击的隐蔽性，我们提出了第一个干净标签框架Kallima，用于合成模仿风格的后门样本来开发隐蔽的文本后门攻击。我们使用对抗性扰动来修改属于目标类的输入，使模型更依赖于后门触发器。我们的框架与大多数现有的后门触发器兼容。在三个基准数据集上的实验结果证明了该方法的有效性。



## **44. Almost Tight L0-norm Certified Robustness of Top-k Predictions against Adversarial Perturbations**

Top-k预测对敌方扰动的几乎紧L0范数认证稳健性 cs.CR

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2011.07633v2)

**Authors**: Jinyuan Jia, Binghui Wang, Xiaoyu Cao, Hongbin Liu, Neil Zhenqiang Gong

**Abstracts**: Top-k predictions are used in many real-world applications such as machine learning as a service, recommender systems, and web searches. $\ell_0$-norm adversarial perturbation characterizes an attack that arbitrarily modifies some features of an input such that a classifier makes an incorrect prediction for the perturbed input. $\ell_0$-norm adversarial perturbation is easy to interpret and can be implemented in the physical world. Therefore, certifying robustness of top-$k$ predictions against $\ell_0$-norm adversarial perturbation is important. However, existing studies either focused on certifying $\ell_0$-norm robustness of top-$1$ predictions or $\ell_2$-norm robustness of top-$k$ predictions. In this work, we aim to bridge the gap. Our approach is based on randomized smoothing, which builds a provably robust classifier from an arbitrary classifier via randomizing an input. Our major theoretical contribution is an almost tight $\ell_0$-norm certified robustness guarantee for top-$k$ predictions. We empirically evaluate our method on CIFAR10 and ImageNet. For instance, our method can build a classifier that achieves a certified top-3 accuracy of 69.2\% on ImageNet when an attacker can arbitrarily perturb 5 pixels of a testing image.

摘要: Top-k预测被用于许多真实世界的应用中，例如机器学习即服务、推荐系统和网络搜索。$\ell_0$-范数对抗性扰动刻画了这样一种攻击：任意修改输入的某些特征，使得分类器对扰动输入做出错误的预测。$\ell_0$-范数对抗性摄动很容易解释，并且可以在物理世界中实现。因此，证明top-$k$预测对$\ell_0$-范数对抗扰动的稳健性是很重要的。然而，现有的研究要么集中于证明TOP-$1$预测的$\ELL_0$-范数稳健性，要么集中于证明TOP-$K$预测的$\ELL_2$-范数稳健性。在这项工作中，我们的目标是弥合这一差距。我们的方法基于随机化平滑，通过随机化输入，从任意分类器构建可证明稳健的分类器。我们的主要理论贡献是对top-$k$预测提供了几乎紧的$\ell_0$-范数证明的稳健性保证。我们在CIFAR10和ImageNet上对我们的方法进行了实证评估。例如，我们的方法可以构建一个分类器，当攻击者可以任意扰乱测试图像的5个像素时，该分类器在ImageNet上的认证准确率为69.2\%。



## **45. Gradient Obfuscation Checklist Test Gives a False Sense of Security**

梯度混淆核对表测试给人一种错误的安全感 cs.CV

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01705v1)

**Authors**: Nikola Popovic, Danda Pani Paudel, Thomas Probst, Luc Van Gool

**Abstracts**: One popular group of defense techniques against adversarial attacks is based on injecting stochastic noise into the network. The main source of robustness of such stochastic defenses however is often due to the obfuscation of the gradients, offering a false sense of security. Since most of the popular adversarial attacks are optimization-based, obfuscated gradients reduce their attacking ability, while the model is still susceptible to stronger or specifically tailored adversarial attacks. Recently, five characteristics have been identified, which are commonly observed when the improvement in robustness is mainly caused by gradient obfuscation. It has since become a trend to use these five characteristics as a sufficient test, to determine whether or not gradient obfuscation is the main source of robustness. However, these characteristics do not perfectly characterize all existing cases of gradient obfuscation, and therefore can not serve as a basis for a conclusive test. In this work, we present a counterexample, showing this test is not sufficient for concluding that gradient obfuscation is not the main cause of improvements in robustness.

摘要: 针对敌意攻击的一组流行的防御技术是基于向网络中注入随机噪声。然而，这种随机防御的主要健壮性来源往往是由于对梯度的混淆，提供了一种错误的安全感。由于大多数流行的对抗性攻击都是基于优化的，模糊梯度降低了它们的攻击能力，而该模型仍然容易受到更强或特定定制的对抗性攻击。最近，已经确定了五个特征，当稳健性的改善主要由梯度混淆引起时，通常观察到这些特征。自那以后，使用这五个特征作为充分的测试来确定梯度混淆是否是健壮性的主要来源已经成为一种趋势。然而，这些特征并不能完美地描述所有现有的梯度模糊情况，因此不能作为决定性测试的基础。在这项工作中，我们提供了一个反例，表明这个测试不足以得出梯度混淆不是健壮性提高的主要原因的结论。



## **46. Evaluating Transfer-based Targeted Adversarial Perturbations against Real-World Computer Vision Systems based on Human Judgments**

基于人的判断评估基于迁移的目标对抗性扰动对真实世界计算机视觉系统的影响 cs.CV

technical report

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01467v1)

**Authors**: Zhengyu Zhao, Nga Dang, Martha Larson

**Abstracts**: Computer vision systems are remarkably vulnerable to adversarial perturbations. Transfer-based adversarial images are generated on one (source) system and used to attack another (target) system. In this paper, we take the first step to investigate transfer-based targeted adversarial images in a realistic scenario where the target system is trained on some private data with its inventory of semantic labels not publicly available. Our main contributions include an extensive human-judgment-based evaluation of attack success on the Google Cloud Vision API and additional analysis of the different behaviors of Google Cloud Vision in face of original images vs. adversarial images. Resources are publicly available at \url{https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/google_results.zip}.

摘要: 计算机视觉系统非常容易受到对抗性干扰的影响。基于传输的敌意图像在一个(源)系统上生成，并用于攻击另一个(目标)系统。在本文中，我们第一步研究了基于转移的目标敌意图像，在现实场景中，目标系统是在一些私有数据上进行训练的，其语义标签库不是公开的。我们的主要贡献包括对Google Cloud Vision API的攻击成功进行了广泛的基于人的判断的评估，以及对Google Cloud Vision在面对原始图像和对手图像时的不同行为进行了额外的分析。资源可在\url{https://github.com/ZhengyuZhao/Targeted-Tansfer/blob/main/google_results.zip}.上公开获得



## **47. Adversarial Attacks on Human Vision**

对人类视觉的对抗性攻击 cs.CV

21 pages, 8 figures, 1 table

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01365v1)

**Authors**: Victor A. Mateescu, Ivan V. Bajić

**Abstracts**: This article presents an introduction to visual attention retargeting, its connection to visual saliency, the challenges associated with it, and ideas for how it can be approached. The difficulty of attention retargeting as a saliency inversion problem lies in the lack of one-to-one mapping between saliency and the image domain, in addition to the possible negative impact of saliency alterations on image aesthetics. A few approaches from recent literature to solve this challenging problem are reviewed, and several suggestions for future development are presented.

摘要: 这篇文章介绍了视觉注意重定目标，它与视觉显著的联系，与之相关的挑战，以及如何处理它的想法。注意力重定向作为显著反转问题的困难在于缺乏显著与图像域之间的一对一映射，以及显著变化可能对图像美学造成的负面影响。从最近的文献中回顾了一些解决这一挑战性问题的方法，并对未来的发展提出了几点建议。



## **48. On the Privacy Properties of GAN-generated Samples**

GaN样品的保密特性研究 cs.LG

AISTATS 2021

**SubmitDate**: 2022-06-03    [paper-pdf](http://arxiv.org/pdf/2206.01349v1)

**Authors**: Zinan Lin, Vyas Sekar, Giulia Fanti

**Abstracts**: The privacy implications of generative adversarial networks (GANs) are a topic of great interest, leading to several recent algorithms for training GANs with privacy guarantees. By drawing connections to the generalization properties of GANs, we prove that under some assumptions, GAN-generated samples inherently satisfy some (weak) privacy guarantees. First, we show that if a GAN is trained on m samples and used to generate n samples, the generated samples are (epsilon, delta)-differentially-private for (epsilon, delta) pairs where delta scales as O(n/m). We show that under some special conditions, this upper bound is tight. Next, we study the robustness of GAN-generated samples to membership inference attacks. We model membership inference as a hypothesis test in which the adversary must determine whether a given sample was drawn from the training dataset or from the underlying data distribution. We show that this adversary can achieve an area under the ROC curve that scales no better than O(m^{-1/4}).

摘要: 生成性对抗网络(GAN)的隐私影响是一个非常感兴趣的话题，导致了最近几个用于训练具有隐私保证的GAN的算法。通过与GANS的泛化性质的联系，我们证明了在某些假设下，GAN生成的样本内在地满足某些(弱)隐私保证。首先，我们证明了如果一个GaN被训练在m个样本上并用来产生n个样本，所产生的样本对于(epsilon，Delta)对是(epsilon，Delta)-差分-私有的，其中Delta尺度为O(n/m)。我们证明了在某些特殊条件下，这个上界是紧的。接下来，我们研究了GAN生成的样本对成员推理攻击的稳健性。我们将成员推理建模为假设检验，其中对手必须确定给定的样本是从训练数据集还是从底层数据分布中提取的。我们证明了这个对手可以在ROC曲线下获得一个尺度不超过O(m^{-1/4})的区域。



## **49. Adaptive Adversarial Training to Improve Adversarial Robustness of DNNs for Medical Image Segmentation and Detection**

自适应对抗训练提高DNN在医学图像分割和检测中的对抗鲁棒性 eess.IV

8 pages

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2206.01736v1)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Recent methods based on Deep Neural Networks (DNNs) have reached high accuracy for medical image analysis, including the three basic tasks: segmentation, landmark detection, and object detection. It is known that DNNs are vulnerable to adversarial attacks, and the adversarial robustness of DNNs could be improved by adding adversarial noises to training data (i.e., adversarial training). In this study, we show that the standard adversarial training (SAT) method has a severe issue that limits its practical use: it generates a fixed level of noise for DNN training, and it is difficult for the user to choose an appropriate noise level, because a high noise level may lead to a large reduction in model performance, and a low noise level may have little effect. To resolve this issue, we have designed a novel adaptive-margin adversarial training (AMAT) method that generates adaptive adversarial noises for DNN training, which are dynamically tailored for each individual training sample. We have applied our AMAT method to state-of-the-art DNNs for the three basic tasks, using five publicly available datasets. The experimental results demonstrate that our AMAT method outperforms the SAT method in adversarial robustness on noisy data and prediction accuracy on clean data. Please contact the author for the source code.

摘要: 近年来，基于深度神经网络(DNNS)的医学图像分析方法已经达到了很高的精度，包括分割、地标检测和目标检测三个基本任务。众所周知，DNN容易受到对抗性攻击，通过在训练数据中添加对抗性噪声(即对抗性训练)可以提高DNN的对抗性健壮性。在这项研究中，我们发现标准的对抗训练(SAT)方法有一个严重的问题限制了它的实际应用：它为DNN训练产生固定的噪声水平，用户很难选择合适的噪声水平，因为高水平的噪声可能会导致模型性能的大幅下降，而低水平的噪声可能影响很小。为了解决这一问题，我们设计了一种新的自适应差值对抗训练方法(AMAT)，该方法为DNN训练生成自适应对抗噪声，这些噪声是为每个训练样本动态定制的。我们已经将我们的AMAT方法应用于三个基本任务的最先进的DNN，使用了五个公开可用的数据集。实验结果表明，我们的AMAT方法在对噪声数据的对抗稳健性和对干净数据的预测精度方面优于SAT方法。请联系作者以获取源代码。



## **50. A Barrier Certificate-based Simplex Architecture with Application to Microgrids**

一种基于屏障证书的单纯形体系结构及其在微网格中的应用 eess.SY

**SubmitDate**: 2022-06-02    [paper-pdf](http://arxiv.org/pdf/2202.09710v2)

**Authors**: Amol Damare, Shouvik Roy, Scott A. Smolka, Scott D. Stoller

**Abstracts**: We present Barrier Certificate-based Simplex (BC-Simplex), a new, provably correct design for runtime assurance of continuous dynamical systems. BC-Simplex is centered around the Simplex Control Architecture, which consists of a high-performance advanced controller which is not guaranteed to maintain safety of the plant, a verified-safe baseline controller, and a decision module that switches control of the plant between the two controllers to ensure safety without sacrificing performance. In BC-Simplex, Barrier certificates are used to prove that the baseline controller ensures safety. Furthermore, BC-Simplex features a new automated method for deriving, from the barrier certificate, the conditions for switching between the controllers. Our method is based on the Taylor expansion of the barrier certificate and yields computationally inexpensive switching conditions. We consider a significant application of BC-Simplex to a microgrid featuring an advanced controller in the form of a neural network trained using reinforcement learning. The microgrid is modeled in RTDS, an industry-standard high-fidelity, real-time power systems simulator. Our results demonstrate that BC-Simplex can automatically derive switching conditions for complex systems, the switching conditions are not overly conservative, and BC-Simplex ensures safety even in the presence of adversarial attacks on the neural controller.

摘要: 提出了基于屏障证书的单纯形(BC-Simplex)，这是一种新的、可证明是正确的连续动态系统运行时保证设计。BC-Simplex围绕Simplex控制架构展开，该架构由不能保证维护工厂安全的高性能高级控制器、经过验证的安全基准控制器以及在两个控制器之间切换工厂控制以确保安全而不牺牲性能的决策模块组成。在BC-Simplex中，屏障证书被用来证明基线控制器确保了安全性。此外，BC-Simplex具有一种新的自动方法，用于从屏障证书推导控制器之间切换的条件。我们的方法是基于障碍证书的泰勒展开式，并且产生了计算上不昂贵的切换条件。我们考虑了BC-单纯形在微电网中的一个重要应用，该微网具有一个采用强化学习训练的神经网络形式的高级控制器。微电网是在RTDS中建模的，RTDS是一种行业标准的高保真、实时电力系统仿真器。结果表明，BC-单纯形能够自动推导出复杂系统的切换条件，切换条件不会过于保守，即使在神经控制器受到敌意攻击的情况下，BC-单纯形也能保证安全性。



