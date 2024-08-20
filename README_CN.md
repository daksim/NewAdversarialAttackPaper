# Latest Adversarial Attack Papers
**update at 2024-08-20 11:03:06**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Fight Perturbations with Perturbations: Defending Adversarial Attacks via Neuron Influence**

用扰动对抗扰动：通过神经元影响防御对抗攻击 cs.CV

Final version. Accepted to IEEE Transactions on Dependable and Secure  Computing

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2112.13060v3) [paper-pdf](http://arxiv.org/pdf/2112.13060v3)

**Authors**: Ruoxi Chen, Haibo Jin, Haibin Zheng, Jinyin Chen, Zhenguang Liu

**Abstract**: The vulnerabilities of deep learning models towards adversarial attacks have attracted increasing attention, especially when models are deployed in security-critical domains. Numerous defense methods, including reactive and proactive ones, have been proposed for model robustness improvement. Reactive defenses, such as conducting transformations to remove perturbations, usually fail to handle large perturbations. The proactive defenses that involve retraining, suffer from the attack dependency and high computation cost. In this paper, we consider defense methods from the general effect of adversarial attacks that take on neurons inside the model. We introduce the concept of neuron influence, which can quantitatively measure neurons' contribution to correct classification. Then, we observe that almost all attacks fool the model by suppressing neurons with larger influence and enhancing those with smaller influence. Based on this, we propose \emph{Neuron-level Inverse Perturbation} (NIP), a novel defense against general adversarial attacks. It calculates neuron influence from benign examples and then modifies input examples by generating inverse perturbations that can in turn strengthen neurons with larger influence and weaken those with smaller influence.

摘要: 深度学习模型对敌意攻击的脆弱性引起了越来越多的关注，特别是当模型部署在安全关键领域时。为了提高模型的稳健性，人们提出了多种防御方法，包括被动防御和主动防御。被动防御，例如进行变换以消除扰动，通常无法处理大扰动。主动防御涉及再训练，存在攻击依赖性和计算代价高等问题。在本文中，我们从对抗性攻击对模型内部神经元的一般影响出发，考虑防御方法。我们引入了神经元影响的概念，它可以定量地衡量神经元对正确分类的贡献。然后，我们观察到几乎所有的攻击都通过抑制影响较大的神经元和增强影响较小的神经元来愚弄模型。在此基础上，我们提出了一种新的防御一般敌意攻击的方法--神经元水平逆摄动(NIP)。它从良性样本中计算神经元的影响，然后通过产生逆扰动来修改输入样本，反过来可以增强影响较大的神经元，削弱影响较小的神经元。



## **2. Detecting Adversarial Attacks in Semantic Segmentation via Uncertainty Estimation: A Deep Analysis**

通过不确定性估计检测语义分割中的对抗性攻击：深入分析 cs.CV

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10021v1) [paper-pdf](http://arxiv.org/pdf/2408.10021v1)

**Authors**: Kira Maag, Roman Resner, Asja Fischer

**Abstract**: Deep neural networks have demonstrated remarkable effectiveness across a wide range of tasks such as semantic segmentation. Nevertheless, these networks are vulnerable to adversarial attacks that add imperceptible perturbations to the input image, leading to false predictions. This vulnerability is particularly dangerous in safety-critical applications like automated driving. While adversarial examples and defense strategies are well-researched in the context of image classification, there is comparatively less research focused on semantic segmentation. Recently, we have proposed an uncertainty-based method for detecting adversarial attacks on neural networks for semantic segmentation. We observed that uncertainty, as measured by the entropy of the output distribution, behaves differently on clean versus adversely perturbed images, and we utilize this property to differentiate between the two. In this extended version of our work, we conduct a detailed analysis of uncertainty-based detection of adversarial attacks including a diverse set of adversarial attacks and various state-of-the-art neural networks. Our numerical experiments show the effectiveness of the proposed uncertainty-based detection method, which is lightweight and operates as a post-processing step, i.e., no model modifications or knowledge of the adversarial example generation process are required.

摘要: 深度神经网络在语义分割等一系列任务中表现出了显著的有效性。然而，这些网络很容易受到敌意攻击，这些攻击会给输入图像添加不可察觉的扰动，导致错误预测。该漏洞在自动驾驶等安全关键型应用程序中尤其危险。虽然在图像分类的背景下，对抗性例子和防御策略已经得到了很好的研究，但针对语义分割的研究相对较少。最近，我们提出了一种基于不确定性的神经网络敌意攻击检测方法，用于语义分割。我们观察到，通过输出分布的熵来衡量不确定性，在干净的图像和反向扰动的图像上表现出不同的行为，我们利用这一特性来区分两者。在我们工作的这个扩展版本中，我们对基于不确定性的对抗性攻击检测进行了详细的分析，包括一组不同的对抗性攻击和各种最先进的神经网络。我们的数值实验表明了基于不确定性的检测方法的有效性，该方法是轻量级的，并且作为后处理步骤进行操作，即不需要修改模型或了解对抗性示例的生成过程。



## **3. Adversarial Prompt Tuning for Vision-Language Models**

视觉语言模型的对抗性即时调优 cs.CV

ECCV 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2311.11261v3) [paper-pdf](http://arxiv.org/pdf/2311.11261v3)

**Authors**: Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang

**Abstract**: With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code is available at https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.

摘要: 随着多通道学习的快速发展，诸如CLIP等预先训练的视觉语言模型在弥合视觉和语言通道之间的差距方面显示出了显著的能力。然而，这些模型仍然容易受到敌意攻击，特别是在图像模式方面，这带来了相当大的安全风险。本文介绍了对抗性提示调优(AdvPT)技术，这是一种在VLMS中增强图像编码器对抗性稳健性的新技术。AdvPT创新性地利用可学习的文本提示，并将其与对抗性图像嵌入相结合，以解决VLM中固有的漏洞，而无需进行广泛的参数培训或修改模型体系结构。我们证明，AdvPT提高了对白盒和黑盒攻击的抵抗力，并与现有的基于图像处理的防御技术相结合，显示出协同效应，进一步增强了防御能力。全面的实验分析提供了对对抗性即时调整的见解，这是一种致力于通过修改文本输入来提高对对抗性图像的抵抗力的新范式，为未来稳健的多通道学习研究铺平了道路。这些发现为增强VLM的安全性开辟了新的可能性。我们的代码可以在https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning.上找到



## **4. Segment-Anything Models Achieve Zero-shot Robustness in Autonomous Driving**

分段任意模型在自动驾驶中实现零攻击鲁棒性 cs.CV

Accepted to IAVVC 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09839v1) [paper-pdf](http://arxiv.org/pdf/2408.09839v1)

**Authors**: Jun Yan, Pengyu Wang, Danni Wang, Weiquan Huang, Daniel Watzenig, Huilin Yin

**Abstract**: Semantic segmentation is a significant perception task in autonomous driving. It suffers from the risks of adversarial examples. In the past few years, deep learning has gradually transitioned from convolutional neural network (CNN) models with a relatively small number of parameters to foundation models with a huge number of parameters. The segment-anything model (SAM) is a generalized image segmentation framework that is capable of handling various types of images and is able to recognize and segment arbitrary objects in an image without the need to train on a specific object. It is a unified model that can handle diverse downstream tasks, including semantic segmentation, object detection, and tracking. In the task of semantic segmentation for autonomous driving, it is significant to study the zero-shot adversarial robustness of SAM. Therefore, we deliver a systematic empirical study on the robustness of SAM without additional training. Based on the experimental results, the zero-shot adversarial robustness of the SAM under the black-box corruptions and white-box adversarial attacks is acceptable, even without the need for additional training. The finding of this study is insightful in that the gigantic model parameters and huge amounts of training data lead to the phenomenon of emergence, which builds a guarantee of adversarial robustness. SAM is a vision foundation model that can be regarded as an early prototype of an artificial general intelligence (AGI) pipeline. In such a pipeline, a unified model can handle diverse tasks. Therefore, this research not only inspects the impact of vision foundation models on safe autonomous driving but also provides a perspective on developing trustworthy AGI. The code is available at: https://github.com/momo1986/robust_sam_iv.

摘要: 语义分割是自动驾驶中一项重要的感知任务。它面临着对抗性例子的风险。在过去的几年里，深度学习逐渐从参数相对较少的卷积神经网络(CNN)模型过渡到参数数量巨大的基础模型。任意分割模型(SAM)是一个通用的图像分割框架，它能够处理各种类型的图像，并且能够识别和分割图像中的任意对象，而不需要对特定对象进行训练。它是一个统一的模型，可以处理不同的下游任务，包括语义分割、目标检测和跟踪。在自主驾驶的语义分割任务中，研究SAM的零射击对抗健壮性具有重要意义。因此，我们在没有额外训练的情况下对SAM的稳健性进行了系统的实证研究。基于实验结果，即使不需要额外的训练，SAM在黑盒腐败和白盒对抗攻击下的零射击对抗健壮性也是可以接受的。这项研究的发现是有洞察力的，因为庞大的模型参数和大量的训练数据导致了涌现现象，这为对手的稳健性提供了保证。SAM是一个视觉基础模型，可以被视为人工通用智能(AGI)管道的早期原型。在这样的管道中，统一的模型可以处理不同的任务。因此，本研究不仅考察了视觉基础模型对安全自动驾驶的影响，而且为开发可信赖的自动驾驶系统提供了一个视角。代码可从以下网址获得：https://github.com/momo1986/robust_sam_iv.



## **5. Patch of Invisibility: Naturalistic Physical Black-Box Adversarial Attacks on Object Detectors**

隐形补丁：对物体检测器的自然主义物理黑匣子对抗攻击 cs.CV

Accepted at MLCS @ ECML-PKDD 2024

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2303.04238v5) [paper-pdf](http://arxiv.org/pdf/2303.04238v5)

**Authors**: Raz Lapid, Eylon Mizrahi, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called "white-box" attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a direct, black-box, gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. To our knowledge this is the first and only method that performs black-box physical attacks directly on object-detection models, which results with a model-agnostic attack. We show that our proposed method works both digitally and physically. We compared our approach against four different black-box attacks with different configurations. Our approach outperformed all other approaches that were tested in our experiments by a large margin.

摘要: 近年来，针对深度学习模型的对抗性攻击受到越来越多的关注。这一领域的工作主要集中在基于梯度的技术上，即所谓的“白盒”攻击，即攻击者可以访问目标模型的内部参数；这种假设在现实世界中通常是不现实的。一些攻击还使用整个像素空间来愚弄给定的模型，这既不实用也不物理(即，现实世界)。相反，我们在这里提出了一种直接的、黑盒的、无梯度的方法，该方法使用预先训练的生成性对抗网络(GAN)的学习图像流形来为目标检测器生成自然的物理对抗斑块。据我们所知，这是第一种也是唯一一种直接对目标检测模型执行黑盒物理攻击的方法，这导致了与模型无关的攻击。我们证明了我们提出的方法在数字和物理上都是有效的。我们将我们的方法与四种不同配置的不同黑盒攻击进行了比较。我们的方法远远超过了在我们的实验中测试的所有其他方法。



## **6. Regularization for Adversarial Robust Learning**

对抗鲁棒学习的正规化 cs.LG

51 pages, 5 figures

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09672v1) [paper-pdf](http://arxiv.org/pdf/2408.09672v1)

**Authors**: Jie Wang, Rui Gao, Yao Xie

**Abstract**: Despite the growing prevalence of artificial neural networks in real-world applications, their vulnerability to adversarial attacks remains to be a significant concern, which motivates us to investigate the robustness of machine learning models. While various heuristics aim to optimize the distributionally robust risk using the $\infty$-Wasserstein metric, such a notion of robustness frequently encounters computation intractability. To tackle the computational challenge, we develop a novel approach to adversarial training that integrates $\phi$-divergence regularization into the distributionally robust risk function. This regularization brings a notable improvement in computation compared with the original formulation. We develop stochastic gradient methods with biased oracles to solve this problem efficiently, achieving the near-optimal sample complexity. Moreover, we establish its regularization effects and demonstrate it is asymptotic equivalence to a regularized empirical risk minimization (ERM) framework, by considering various scaling regimes of the regularization parameter $\eta$ and robustness level $\rho$. These regimes yield gradient norm regularization, variance regularization, or a smoothed gradient norm regularization that interpolates between these extremes. We numerically validate our proposed method in supervised learning, reinforcement learning, and contextual learning and showcase its state-of-the-art performance against various adversarial attacks.

摘要: 尽管人工神经网络在现实世界中的应用越来越普遍，但它们对对手攻击的脆弱性仍然是一个重要的问题，这促使我们研究机器学习模型的健壮性。虽然各种启发式方法的目标是使用$\infty$-Wasserstein度量来优化分布健壮性风险，但这样的健壮性概念经常遇到计算困难。为了解决计算上的挑战，我们开发了一种新的对抗性训练方法，将$Phi$-发散正则化整合到分布稳健的风险函数中。与原公式相比，这种正则化方法在计算上有了显著的改进。我们发展了带有有偏预言的随机梯度方法来有效地解决这一问题，获得了接近最优的样本复杂度。此外，我们建立了它的正则化效应，并证明了它与正则化经验风险最小化(ERM)框架的渐近等价，通过考虑正则化参数和稳健性水平的不同标度机制。这些区域产生在这些极值之间内插的梯度范数正则化、方差正则化或平滑的梯度范数正则化。我们在监督学习、强化学习和上下文学习中对我们提出的方法进行了数值验证，并展示了它在抵抗各种对手攻击方面的最新表现。



## **7. Symbiotic Game and Foundation Models for Cyber Deception Operations in Strategic Cyber Warfare**

战略网络战中网络欺骗行动的共生博弈和基础模型 cs.CR

40 pages, 7 figures, 2 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2403.10570v2) [paper-pdf](http://arxiv.org/pdf/2403.10570v2)

**Authors**: Tao Li, Quanyan Zhu

**Abstract**: We are currently facing unprecedented cyber warfare with the rapid evolution of tactics, increasing asymmetry of intelligence, and the growing accessibility of hacking tools. In this landscape, cyber deception emerges as a critical component of our defense strategy against increasingly sophisticated attacks. This chapter aims to highlight the pivotal role of game-theoretic models and foundation models (FMs) in analyzing, designing, and implementing cyber deception tactics. Game models (GMs) serve as a foundational framework for modeling diverse adversarial interactions, allowing us to encapsulate both adversarial knowledge and domain-specific insights. Meanwhile, FMs serve as the building blocks for creating tailored machine learning models suited to given applications. By leveraging the synergy between GMs and FMs, we can advance proactive and automated cyber defense mechanisms by not only securing our networks against attacks but also enhancing their resilience against well-planned operations. This chapter discusses the games at the tactical, operational, and strategic levels of warfare, delves into the symbiotic relationship between these methodologies, and explores relevant applications where such a framework can make a substantial impact in cybersecurity. The chapter discusses the promising direction of the multi-agent neurosymbolic conjectural learning (MANSCOL), which allows the defender to predict adversarial behaviors, design adaptive defensive deception tactics, and synthesize knowledge for the operational level synthesis and adaptation. FMs serve as pivotal tools across various functions for MANSCOL, including reinforcement learning, knowledge assimilation, formation of conjectures, and contextual representation. This chapter concludes with a discussion of the challenges associated with FMs and their application in the domain of cybersecurity.

摘要: 我们目前正面临着前所未有的网络战，战术的快速演变，情报的日益不对称，以及黑客工具的日益普及。在这种情况下，网络欺骗成为我们抵御日益复杂的攻击的防御战略的关键组成部分。本章旨在强调博弈论模型和基础模型(FM)在分析、设计和实施网络欺骗战术中的关键作用。游戏模型(GM)作为建模各种对抗性交互的基本框架，允许我们封装对抗性知识和特定领域的见解。同时，FM作为创建适合特定应用的定制机器学习模型的构建块。通过利用GM和FM之间的协同作用，我们可以推进主动和自动化的网络防御机制，不仅确保我们的网络免受攻击，而且增强其对精心规划的行动的弹性。本章讨论了战争的战术、作战和战略层面的游戏，深入探讨了这些方法之间的共生关系，并探索了此类框架可以在网络安全中产生重大影响的相关应用。本章讨论了多智能体神经符号猜想学习(MANSCOL)的发展方向，它允许防御者预测对手的行为，设计自适应的防御性欺骗策略，并为作战级别的综合和适应综合知识。FMS是MANSCOL的各种功能的关键工具，包括强化学习、知识同化、猜想的形成和上下文表示。本章最后讨论了与FMS相关的挑战及其在网络安全领域的应用。



## **8. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于意识的一致框架打击敌对攻击 cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.

摘要: 自从最近引入了功能越来越强大的大型语言模型(LLM)以来，对人工智能对齐的研究有了很大的增长。不幸的是，现代的校准方法仍然不能完全防止模型受到故意攻击时的有害反应。这些漏洞可能导致LLMS被操纵来生成危险内容：从创建危险材料的说明到煽动暴力或支持不道德行为。为了帮助缓解这个问题，我们引入了Bergeron：一个旨在提高LLM抵御攻击的健壮性的框架，而不需要任何额外的参数微调。Bergeron被组织成两级；辅助LLM充当主要LLM的监护人。此框架可以更好地保护主要模型免受来袭攻击，同时监控其输出中是否有任何有害内容。经验分析认为，通过使用Bergeron来补充模型与现有的比对训练，我们可以显著提高多个常用的商业和开源LLM的稳健性和安全性。具体地说，我们发现，与没有这种支持的型号相比，集成了Bergeron的型号平均抵抗攻击的能力要高出近7倍。



## **9. Interpreting Global Perturbation Robustness of Image Models using Axiomatic Spectral Importance Decomposition**

使用公理谱重要性分解解释图像模型的全局扰动鲁棒性 cs.AI

Accepted by Transactions on Machine Learning Research (TMLR 2024)

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.01139v2) [paper-pdf](http://arxiv.org/pdf/2408.01139v2)

**Authors**: Róisín Luo, James McDermott, Colm O'Riordan

**Abstract**: Perturbation robustness evaluates the vulnerabilities of models, arising from a variety of perturbations, such as data corruptions and adversarial attacks. Understanding the mechanisms of perturbation robustness is critical for global interpretability. We present a model-agnostic, global mechanistic interpretability method to interpret the perturbation robustness of image models. This research is motivated by two key aspects. First, previous global interpretability works, in tandem with robustness benchmarks, e.g. mean corruption error (mCE), are not designed to directly interpret the mechanisms of perturbation robustness within image models. Second, we notice that the spectral signal-to-noise ratios (SNR) of perturbed natural images exponentially decay over the frequency. This power-law-like decay implies that: Low-frequency signals are generally more robust than high-frequency signals -- yet high classification accuracy can not be achieved by low-frequency signals alone. By applying Shapley value theory, our method axiomatically quantifies the predictive powers of robust features and non-robust features within an information theory framework. Our method, dubbed as \textbf{I-ASIDE} (\textbf{I}mage \textbf{A}xiomatic \textbf{S}pectral \textbf{I}mportance \textbf{D}ecomposition \textbf{E}xplanation), provides a unique insight into model robustness mechanisms. We conduct extensive experiments over a variety of vision models pre-trained on ImageNet to show that \textbf{I-ASIDE} can not only \textbf{measure} the perturbation robustness but also \textbf{provide interpretations} of its mechanisms.

摘要: 扰动稳健性评估由各种扰动引起的模型的脆弱性，例如数据损坏和对抗性攻击。理解扰动稳健性的机制对于全局可解释性至关重要。我们提出了一种模型不可知的全局机械可解释性方法来解释图像模型的扰动稳健性。这项研究的动机有两个关键方面。首先，以前的全局可解释性与稳健性基准一起工作，例如平均破坏误差(MCE)，不是被设计成直接解释图像模型中的扰动稳健性的机制。其次，我们注意到受扰动的自然图像的光谱信噪比(SNR)随频率呈指数衰减。这种类似幂规律的衰减意味着：低频信号通常比高频信号更健壮--然而，仅靠低频信号不能达到高分类精度。通过应用Shapley值理论，我们的方法在信息论框架内公理地量化了稳健特征和非稳健特征的预测能力。我们的方法称为Textbf{i-side}(Textbf{I}MAGE\Textbf{A}X-Ait\Textbf{S}频谱\Textbf{I}M重要\Textbf{D}分解\Textbf{E}解释)，提供了对模型健壮性机制的独特见解。我们在ImageNet上预先训练的各种视觉模型上进行了大量的实验，结果表明，文本bf{i-side}不仅可以测量扰动的稳健性，而且可以对其机制进行解释。



## **10. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

通过对抗权重调整增强对抗可移植性 cs.CR

13 pages

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09469v1) [paper-pdf](http://arxiv.org/pdf/2408.09469v1)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji, Shouling_Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.

摘要: 深度神经网络(DNN)很容易受到敌意例子(AE)的攻击，这些例子误导了模型，同时对人类观察者来说是良性的。一个关键的问题是AEs的可转移性，这使得黑盒攻击能够在不直接访问目标模型的情况下进行。然而，以往的许多攻击都未能解释对抗性转移的内在机制。在本文中，我们重新思考了可转让实体的性质，并对可转让的提法进行了改造。在此机制的基础上，我们分析了不同体系结构模型之间的AEs泛化，并证明了我们可以找到局部扰动来缓解代理模型和目标模型之间的差距。我们进一步建立了模型光滑性与平坦局部极大值之间的内在联系，这两者都有助于AEs的可转移性。在此基础上，提出了一种新的对抗性攻击算法AWT是一种无数据调整方法，它结合了基于梯度和基于模型的攻击方法来增强AE的可转移性。在ImageNet上对不同体系结构的各种模型进行的大量实验表明，AWT的攻击性能优于其他攻击，基于CNN和基于Transformer的模型的攻击成功率比最先进的攻击分别提高了近5%和10%。



## **11. WPN: An Unlearning Method Based on N-pair Contrastive Learning in Language Models**

WPN：一种基于语言模型N对对比学习的去学习方法 cs.CL

ECAI 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09459v1) [paper-pdf](http://arxiv.org/pdf/2408.09459v1)

**Authors**: Guitao Chen, Yunshen Wang, Hongye Sun, Guang Chen

**Abstract**: Generative language models (LMs) offer numerous advantages but may produce inappropriate or harmful outputs due to the harmful knowledge acquired during pre-training. This knowledge often manifests as undesirable correspondences, such as "harmful prompts" leading to "harmful outputs," which our research aims to mitigate through unlearning techniques.However, existing unlearning methods based on gradient ascent can significantly impair the performance of LMs. To address this issue, we propose a novel approach called Weighted Positional N-pair (WPN) Learning, which leverages position-weighted mean pooling within an n-pair contrastive learning framework. WPN is designed to modify the output distribution of LMs by eliminating specific harmful outputs (e.g., replacing toxic responses with neutral ones), thereby transforming the model's behavior from "harmful prompt-harmful output" to "harmful prompt-harmless response".Experiments on OPT and GPT-NEO LMs show that WPN effectively reduces the proportion of harmful responses, achieving a harmless rate of up to 95.8\% while maintaining stable performance on nine common benchmarks (with less than 2\% degradation on average). Moreover, we provide empirical evidence to demonstrate WPN's ability to weaken the harmful correspondences in terms of generalizability and robustness, as evaluated on out-of-distribution test sets and under adversarial attacks.

摘要: 生成语言模型(LMS)有许多优点，但可能会产生不适当或有害的输出，因为在预培训期间获得了有害的知识。这些知识经常表现为不希望看到的对应关系，例如“有害提示”导致“有害输出”，我们的研究旨在通过遗忘技术来缓解这种情况。然而，现有的基于梯度上升的遗忘方法会显著影响LMS的性能。为了解决这个问题，我们提出了一种新的方法，称为加权位置N对(WPN)学习，它利用n对对比学习框架中的位置加权平均池。在OPT和GPT-neo LMS上的实验表明，WPN有效地减少了有害响应的比例，在9个常用基准上保持了稳定的性能(平均降级小于2)，从而改变了LMS的输出分布。此外，我们提供了经验证据来证明WPN在泛化能力和稳健性方面能够削弱有害的对应关系，如在分布外测试集上和在敌意攻击下的评估。



## **12. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

基于XAI检测Deepfake检测器上的对抗攻击 cs.CR

Accepted at TMLR 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2403.02955v2) [paper-pdf](http://arxiv.org/pdf/2403.02955v2)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.

摘要: 我们介绍了一种利用可解释人工智能(XAI)来识别针对深度假冒检测器的对抗性攻击的新方法。在一个以数字进步为特征的时代，深度假冒已经成为一种强有力的工具，创造了对高效检测系统的需求。然而，这些系统经常成为抑制其性能的对抗性攻击的目标。我们解决了这个问题，通过利用XAI的能力开发了一个可防御的深度伪检测器。所提出的方法使用XAI为给定的方法生成可解释性地图，提供人工智能模型中决策因素的显式可视化。随后，我们采用了一个预先训练的特征抽取器来处理输入图像及其对应的XAI图像。然后使用从该过程中提取的特征嵌入来训练简单而有效的分类器。我们的方法不仅有助于深度假冒的检测，还有助于增强对可能的敌意攻击的理解，准确地定位潜在的漏洞。此外，该方法不会改变深度伪检测器的性能。这篇论文展示了令人振奋的结果，为未来的深度伪检测机制提供了一条潜在的途径。我们相信，这项研究将对社区做出有价值的贡献，引发关于保护深度假冒探测器的迫切需要的讨论。



## **13. Adversarial Attacked Teacher for Unsupervised Domain Adaptive Object Detection**

无监督领域自适应对象检测的对抗攻击教师 cs.CV

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09431v1) [paper-pdf](http://arxiv.org/pdf/2408.09431v1)

**Authors**: Kaiwen Wang, Yinzhe Shen, Martin Lauer

**Abstract**: Object detectors encounter challenges in handling domain shifts. Cutting-edge domain adaptive object detection methods use the teacher-student framework and domain adversarial learning to generate domain-invariant pseudo-labels for self-training. However, the pseudo-labels generated by the teacher model tend to be biased towards the majority class and often mistakenly include overconfident false positives and underconfident false negatives. We reveal that pseudo-labels vulnerable to adversarial attacks are more likely to be low-quality. To address this, we propose a simple yet effective framework named Adversarial Attacked Teacher (AAT) to improve the quality of pseudo-labels. Specifically, we apply adversarial attacks to the teacher model, prompting it to generate adversarial pseudo-labels to correct bias, suppress overconfidence, and encourage underconfident proposals. An adaptive pseudo-label regularization is introduced to emphasize the influence of pseudo-labels with high certainty and reduce the negative impacts of uncertain predictions. Moreover, robust minority objects verified by pseudo-label regularization are oversampled to minimize dataset imbalance without introducing false positives. Extensive experiments conducted on various datasets demonstrate that AAT achieves superior performance, reaching 52.6 mAP on Clipart1k, surpassing the previous state-of-the-art by 6.7%.

摘要: 物体探测器在处理区域移位时遇到了挑战。前沿领域自适应目标检测方法使用师生框架和领域对抗学习来生成领域不变的伪标签，用于自我训练。然而，教师模式生成的伪标签往往偏向于大多数班级，并且经常错误地包括过度自信的假阳性和不足自信的假阴性。我们发现，易受对手攻击的伪标签更有可能是低质量的。为了解决这个问题，我们提出了一个简单而有效的框架，称为对抗性攻击教师(AAT)，以提高伪标签的质量。具体地说，我们对教师模型应用对抗性攻击，促使它生成对抗性伪标签来纠正偏见，抑制过度自信，并鼓励信心不足的建议。引入了一种自适应伪标签正则化方法，以突出高确定性伪标签的影响，减少不确定预测带来的负面影响。此外，通过伪标签正则化验证的健壮少数对象被过采样，以最小化数据集的不平衡而不会引入误报。在不同的数据集上进行的广泛实验表明，AAT取得了优越的性能，在Clipart1k上达到了52.6MAP，超过了以前的最先进水平6.7%。



## **14. Rethinking Impersonation and Dodging Attacks on Face Recognition Systems**

重新思考模仿并躲避对人脸识别系统的攻击 cs.CV

Accepted to ACM MM 2024

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2401.08903v4) [paper-pdf](http://arxiv.org/pdf/2401.08903v4)

**Authors**: Fengfan Zhou, Qianyu Zhou, Bangjie Yin, Hui Zheng, Xuequan Lu, Lizhuang Ma, Hefei Ling

**Abstract**: Face Recognition (FR) systems can be easily deceived by adversarial examples that manipulate benign face images through imperceptible perturbations. Adversarial attacks on FR encompass two types: impersonation (targeted) attacks and dodging (untargeted) attacks. Previous methods often achieve a successful impersonation attack on FR, however, it does not necessarily guarantee a successful dodging attack on FR in the black-box setting. In this paper, our key insight is that the generation of adversarial examples should perform both impersonation and dodging attacks simultaneously. To this end, we propose a novel attack method termed as Adversarial Pruning (Adv-Pruning), to fine-tune existing adversarial examples to enhance their dodging capabilities while preserving their impersonation capabilities. Adv-Pruning consists of Priming, Pruning, and Restoration stages. Concretely, we propose Adversarial Priority Quantification to measure the region-wise priority of original adversarial perturbations, identifying and releasing those with minimal impact on absolute model output variances. Then, Biased Gradient Adaptation is presented to adapt the adversarial examples to traverse the decision boundaries of both the attacker and victim by adding perturbations favoring dodging attacks on the vacated regions, preserving the prioritized features of the original perturbations while boosting dodging performance. As a result, we can maintain the impersonation capabilities of original adversarial examples while effectively enhancing dodging capabilities. Comprehensive experiments demonstrate the superiority of our method compared with state-of-the-art adversarial attack methods.

摘要: 人脸识别(FR)系统很容易被敌意的例子欺骗，这些例子通过潜移默化的扰动来操纵良性的人脸图像。对FR的敌意攻击包括两种类型：模仿(目标)攻击和躲避(非目标)攻击。以往的方法往往能成功地实现对FR的模仿攻击，但在黑盒环境下，这并不一定能保证对FR的成功躲避攻击。在本文中，我们的主要观点是，生成敌意示例应该同时执行模仿攻击和躲避攻击。为此，我们提出了一种新的攻击方法，称为对抗性剪枝(ADV-Puning)，对现有的对抗性实例进行微调，以增强它们的躲避能力，同时保持它们的模拟能力。高级修剪包括启动、修剪和恢复三个阶段。具体地说，我们提出了对抗性优先级量化来度量原始对抗性扰动的区域优先级，识别并释放那些对绝对模型输出方差影响最小的扰动。然后，通过在空闲区域上添加有利于躲避攻击的扰动，保留了原始扰动的优先特征，同时提高了躲避性能，提出了有偏梯度自适应算法，使敌意例子能够穿越攻击者和受害者的决策边界。因此，我们可以在保持原始对抗性例子的模拟能力的同时，有效地增强躲避能力。综合实验表明，与现有的对抗性攻击方法相比，该方法具有一定的优越性。



## **15. Malacopula: adversarial automatic speaker verification attacks using a neural-based generalised Hammerstein model**

Malacopula：使用基于神经的广义Hammerstein模型的对抗性自动说话人验证攻击 eess.AS

Accepted at ASVspoof Workshop 2024

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09300v1) [paper-pdf](http://arxiv.org/pdf/2408.09300v1)

**Authors**: Massimiliano Todisco, Michele Panariello, Xin Wang, Héctor Delgado, Kong Aik Lee, Nicholas Evans

**Abstract**: We present Malacopula, a neural-based generalised Hammerstein model designed to introduce adversarial perturbations to spoofed speech utterances so that they better deceive automatic speaker verification (ASV) systems. Using non-linear processes to modify speech utterances, Malacopula enhances the effectiveness of spoofing attacks. The model comprises parallel branches of polynomial functions followed by linear time-invariant filters. The adversarial optimisation procedure acts to minimise the cosine distance between speaker embeddings extracted from spoofed and bona fide utterances. Experiments, performed using three recent ASV systems and the ASVspoof 2019 dataset, show that Malacopula increases vulnerabilities by a substantial margin. However, speech quality is reduced and attacks can be detected effectively under controlled conditions. The findings emphasise the need to identify new vulnerabilities and design defences to protect ASV systems from adversarial attacks in the wild.

摘要: 我们提出了Malacopula，这是一种基于神经的广义Hammerstein模型，旨在向欺骗的语音话语引入对抗性扰动，以便它们更好地欺骗自动说话人验证（ASV）系统。Malacopula使用非线性过程来修改语音话语，增强了欺骗攻击的有效性。该模型由多项函数的并行分支组成，后面是线性时不变过滤器。对抗优化过程的作用是最小化从欺骗和真实话语中提取的说话者嵌入之间的cos距离。使用三个最近的ASV系统和ASVspoof 2019数据集进行的实验表明，Malacopula大幅增加了漏洞。然而，语音质量会降低，并且可以在受控条件下有效检测攻击。研究结果强调，需要识别新的漏洞并设计防御措施，以保护ASV系统免受野外对抗攻击。



## **16. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

PADetBench：针对对象检测的物理攻击基准 cs.CV

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09181v1) [paper-pdf](http://arxiv.org/pdf/2408.09181v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications.   However, conducting physical experiments is extremely time-consuming and labor-intensive.   Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models.   To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation.   This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world.   Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis.   In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics.   Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack

摘要: 针对目标检测的物理攻击由于其重要的实际意义而受到越来越多的关注。然而，进行物理实验是极其耗时和劳动密集型的。此外，物理动力学和跨域转换在现实世界中面临严格规范的挑战，导致不一致的评估和比较，严重阻碍了物理稳健模型的发展。为了应对这些挑战，我们探索利用真实的模拟在受控的物理动力学和跨域转换下，彻底和严格地基准具有公平性的物理攻击。这解决了在现实世界中无法实现的捕获相同的对抗性图像的问题。我们的基准包括20种物理攻击方法、48个对象探测器、全面的物理动力学和评估指标。我们还提供用于数据集生成、检测、评估和进一步分析的端到端管道。此外，我们根据我们的基准进行了8064组评估，其中包括对受控物理动力学的整体评估和进一步的详细消融研究。通过这些实验，我们对物理攻击性能和物理对抗健壮性进行了深入的分析，得出了有价值的观察结果，并讨论了未来研究的潜在方向。码基：https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **17. Training Verifiably Robust Agents Using Set-Based Reinforcement Learning**

使用基于集的强化学习训练可验证稳健的代理 cs.LG

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09112v1) [paper-pdf](http://arxiv.org/pdf/2408.09112v1)

**Authors**: Manuel Wendl, Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Reinforcement learning often uses neural networks to solve complex control tasks. However, neural networks are sensitive to input perturbations, which makes their deployment in safety-critical environments challenging. This work lifts recent results from formally verifying neural networks against such disturbances to reinforcement learning in continuous state and action spaces using reachability analysis. While previous work mainly focuses on adversarial attacks for robust reinforcement learning, we train neural networks utilizing entire sets of perturbed inputs and maximize the worst-case reward. The obtained agents are verifiably more robust than agents obtained by related work, making them more applicable in safety-critical environments. This is demonstrated with an extensive empirical evaluation of four different benchmarks.

摘要: 强化学习通常使用神经网络来解决复杂的控制任务。然而，神经网络对输入扰动很敏感，这使得它们在安全关键环境中的部署具有挑战性。这项工作将正式验证神经网络对抗此类干扰的最新结果提升到使用可达性分析在连续状态和动作空间中进行强化学习。虽然之前的工作主要集中在对抗攻击以实现鲁棒强化学习，但我们利用整组受干扰的输入来训练神经网络，并最大化最坏情况下的回报。可以验证，获得的代理比通过相关工作获得的代理更稳健，使它们更适用于安全关键的环境。对四个不同基准的广泛实证评估证明了这一点。



## **18. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多通道大型语言模型(MLLM)在各种多通道任务中表现出令人印象深刻的性能。另一方面，附加图像模式的集成可能允许恶意用户在图像中注入有害内容以越狱。与基于文本的LLMS不同，在LLMS中，攻击者需要使用特定的算法选择离散的令牌来隐藏其恶意意图，而图像信号的连续性为攻击者提供了直接注入有害意图的机会。在这项工作中，我们提出了一种简单而有效的越狱防御机制--$\extbf{bathe}$($\extbf{ba}$ck door$\extbf{T}$rigger S$\extbf{h}$i$\extbf{e}$ld)。我们的工作是基于生成式语言模型对越狱后门攻击和虚拟提示后门攻击的最新研究。越狱后门攻击使用有害指令和手动创建的字符串作为触发器，使后门模型生成被禁止的响应。我们假设有害指令可以作为触发器，如果我们将拒绝响应设置为触发响应，那么反向模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一点，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为‘’楔形‘’。我们的综合实验表明，BAIT有效地缓解了各种类型的越狱攻击，并且能够自适应地防御看不见的攻击，对MLLMS的性能影响最小。



## **19. HookChain: A new perspective for Bypassing EDR Solutions**

HookChain：询问EDR解决方案的新视角 cs.CR

50 pages, 23 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2404.16856v3) [paper-pdf](http://arxiv.org/pdf/2404.16856v3)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.

摘要: 在当前的数字安全生态系统中，威胁发展迅速且复杂，开发终端检测和响应(EDR)解决方案的公司正在不断寻找创新，不仅要跟上形势，还要预测新出现的攻击媒介。在此背景下，本文介绍了HookChain，从另一个角度介绍了广为人知的技术，这些技术结合在一起时，提供了针对传统EDR系统的另一层复杂规避。通过IAT挂钩技术、动态SSN解析和间接系统调用的精确组合，HookChain以一种仅作用于Ntdll.dll的EDR保持警惕的眼睛看不到的方式重定向Windows子系统的执行流，而不需要更改所涉及的应用程序和恶意软件的源代码。这项工作不仅挑战了目前的网络安全惯例，而且还揭示了未来保护战略的一条有希望的道路，充分利用了对持续演变是数字安全有效性的关键的理解。通过开发和探索HookChain技术，这项研究对终端安全方面的知识体系做出了重大贡献，刺激了能够有效应对不断变化的数字威胁动态的更健壮和适应性更强的解决方案的开发。这项工作旨在激发人们对安全技术研究和开发的深刻反思和进步，这些技术总是领先于对手几步。



## **20. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

询问、参与、攻击：针对图像到文本模型的有效基于决策的黑匣子定向攻击 cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.

摘要: 虽然图像到文本模型在各种视觉语言任务中显示出了显著的进步，但它们仍然容易受到对手的攻击。现有的针对图像到文本模型的白盒攻击需要访问目标模型的体系结构、渐变和参数，导致实用性较低。最近提出的灰盒攻击虽然提高了实用性，但它们在训练过程中存在语义丢失问题，限制了它们的针对性攻击性能。为了推进图像到文本模型的对抗性攻击，本文重点研究了一个具有挑战性的场景：基于决策的黑箱定向攻击，攻击者只能访问最终的输出文本，并且目标是执行定向攻击。具体地说，我们将基于决策的黑盒定向攻击问题描述为一个大规模优化问题。为了有效地解决优化问题，提出了一个三阶段过程\textit{Ask}引导攻击者创建满足特定语义的目标文本。\textit{attend}识别图像中要攻击的关键区域，从而减少了后续\textit{攻击}的搜索空间。利用进化算法攻击与目标文本语义相关的关键区域，从而在不丢失语义的情况下实现目标攻击。在基于变压器和基于CNN+RNN的图文转换模型上的实验结果证实了该方法的有效性。



## **21. Stochastic Bandits Robust to Adversarial Attacks**

对对抗攻击具有鲁棒性的随机盗贼 cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08859v1) [paper-pdf](http://arxiv.org/pdf/2408.08859v1)

**Authors**: Xuchuang Wang, Jinhang Zuo, Xutong Liu, John C. S. Lui, Mohammad Hajiesmaili

**Abstract**: This paper investigates stochastic multi-armed bandit algorithms that are robust to adversarial attacks, where an attacker can first observe the learner's action and {then} alter their reward observation. We study two cases of this model, with or without the knowledge of an attack budget $C$, defined as an upper bound of the summation of the difference between the actual and altered rewards. For both cases, we devise two types of algorithms with regret bounds having additive or multiplicative $C$ dependence terms. For the known attack budget case, we prove our algorithms achieve the regret bound of ${O}((K/\Delta)\log T + KC)$ and $\tilde{O}(\sqrt{KTC})$ for the additive and multiplicative $C$ terms, respectively, where $K$ is the number of arms, $T$ is the time horizon, $\Delta$ is the gap between the expected rewards of the optimal arm and the second-best arm, and $\tilde{O}$ hides the logarithmic factors. For the unknown case, we prove our algorithms achieve the regret bound of $\tilde{O}(\sqrt{KT} + KC^2)$ and $\tilde{O}(KC\sqrt{T})$ for the additive and multiplicative $C$ terms, respectively. In addition to these upper bound results, we provide several lower bounds showing the tightness of our bounds and the optimality of our algorithms. These results delineate an intrinsic separation between the bandits with attacks and corruption models [Lykouris et al., 2018].

摘要: 研究了对敌方攻击具有鲁棒性的随机多臂盗贼算法，其中攻击者可以首先观察到学习者的行为，然后改变他们的奖励观察。我们研究了该模型的两种情况，在有或不知道攻击预算$C$的情况下，该预算被定义为实际和改变的奖励之间的差值之和的上界。对于这两种情况，我们设计了两种类型的算法，它们具有加性或乘性$C$依赖项的后悔界。对于已知的攻击预算情形，我们证明了我们的算法对于加性和乘性$C$项分别达到了遗憾界$O}((K/\Delta)\log T+KC)$和$\tide{O}(\Sqrt{KTC})$，其中$K$是武器数，$T$是时间范围，$\Delta$是最优ARM和次优ARM的期望回报之间的差距，$\tide{O}$隐藏了对数因子。对于未知情况，我们证明了对于加性项和乘性项，我们的算法分别达到了[0}(KT}+KC^2)$和[O](KCSQRT{T})$的遗憾界。除了这些上界结果外，我们还提供了几个下界，表明了我们的界的紧密性和我们的算法的最优性。这些结果描绘了具有攻击和腐败模式的土匪之间的内在分离[Lykouris等人，2018年]。



## **22. Potion: Towards Poison Unlearning**

药剂：走向毒药的学习 cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2406.09173v2) [paper-pdf](http://arxiv.org/pdf/2406.09173v2)

**Authors**: Stefan Schoepf, Jack Foster, Alexandra Brintrup

**Abstract**: Adversarial attacks by malicious actors on machine learning systems, such as introducing poison triggers into training datasets, pose significant risks. The challenge in resolving such an attack arises in practice when only a subset of the poisoned data can be identified. This necessitates the development of methods to remove, i.e. unlearn, poison triggers from already trained models with only a subset of the poison data available. The requirements for this task significantly deviate from privacy-focused unlearning where all of the data to be forgotten by the model is known. Previous work has shown that the undiscovered poisoned samples lead to a failure of established unlearning methods, with only one method, Selective Synaptic Dampening (SSD), showing limited success. Even full retraining, after the removal of the identified poison, cannot address this challenge as the undiscovered poison samples lead to a reintroduction of the poison trigger in the model. Our work addresses two key challenges to advance the state of the art in poison unlearning. First, we introduce a novel outlier-resistant method, based on SSD, that significantly improves model protection and unlearning performance. Second, we introduce Poison Trigger Neutralisation (PTN) search, a fast, parallelisable, hyperparameter search that utilises the characteristic "unlearning versus model protection" trade-off to find suitable hyperparameters in settings where the forget set size is unknown and the retain set is contaminated. We benchmark our contributions using ResNet-9 on CIFAR10 and WideResNet-28x10 on CIFAR100. Experimental results show that our method heals 93.72% of poison compared to SSD with 83.41% and full retraining with 40.68%. We achieve this while also lowering the average model accuracy drop caused by unlearning from 5.68% (SSD) to 1.41% (ours).

摘要: 恶意行为者对机器学习系统的对抗性攻击，如将有毒触发器引入训练数据集，构成了巨大的风险。解决此类攻击的挑战出现在实践中，当只能识别有毒数据的子集时。这就需要开发方法来从仅有有毒数据的子集的已训练模型中移除(即取消学习)有毒触发器。这项任务的要求与关注隐私的遗忘有很大不同，在隐私遗忘中，模型要忘记的所有数据都是已知的。以前的工作表明，未发现的中毒样本会导致已有的遗忘方法的失败，只有一种方法-选择性突触抑制(SSD)-显示出有限的成功。即使在去除已识别的毒物之后进行全面的再培训，也不能解决这一挑战，因为未发现的毒物样本会导致在模型中重新引入毒物触发器。我们的工作解决了两个关键挑战，以推进毒物忘却学习的艺术水平。首先，我们提出了一种新的基于SSD的抗孤立点方法，该方法显著改善了模型保护和遗忘性能。其次，我们引入了毒药触发中和(PTN)搜索，这是一种快速、可并行的超参数搜索，它利用“遗忘与模型保护”的权衡特性，在忘记集大小未知且保留集受到污染的情况下找到合适的超参数。我们使用CIFAR10上的ResNet-9和CIFAR100上的WideResNet-28x10对我们的贡献进行基准测试。实验结果表明，与SSD的83.41%和完全再训练的40.68%相比，我们的方法可以治愈93.72%的毒物。我们实现了这一点，同时也将因遗忘而导致的平均模型精度下降从5.68%(SSD)降至1.41%(我们的)。



## **23. ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale**

ASVspoof 5：众包语音数据、Deepfakes和大规模对抗性攻击 eess.AS

8 pages, ASVspoof 5 Workshop (Interspeech2024 Satellite)

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08739v1) [paper-pdf](http://arxiv.org/pdf/2408.08739v1)

**Authors**: Xin Wang, Hector Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges that promote the study of speech spoofing and deepfake attacks, and the design of detection solutions. Compared to previous challenges, the ASVspoof 5 database is built from crowdsourced data collected from a vastly greater number of speakers in diverse acoustic conditions. Attacks, also crowdsourced, are generated and tested using surrogate detection models, while adversarial attacks are incorporated for the first time. New metrics support the evaluation of spoofing-robust automatic speaker verification (SASV) as well as stand-alone detection solutions, i.e., countermeasures without ASV. We describe the two challenge tracks, the new database, the evaluation metrics, baselines, and the evaluation platform, and present a summary of the results. Attacks significantly compromise the baseline systems, while submissions bring substantial improvements.

摘要: ASVspoof 5是一系列挑战的第五版，旨在促进语音欺骗和深度伪造攻击的研究以及检测解决方案的设计。与之前的挑战相比，ASVspoof 5数据库是根据从不同声学条件下大量扬声器收集的众包数据构建的。攻击（也是众包的）是使用代理检测模型生成和测试的，而对抗性攻击则是首次纳入的。新指标支持对欺骗稳健的自动说话者验证（SASV）以及独立检测解决方案的评估，即没有ASV的对策。我们描述了两个挑战轨道、新数据库、评估指标、基线和评估平台，并提供结果摘要。攻击严重损害了基线系统，而提交则带来了实质性改进。



## **24. A Novel Buffered Federated Learning Framework for Privacy-Driven Anomaly Detection in IIoT**

用于IIoT中隐私驱动异常检测的新型缓冲联邦学习框架 cs.CR

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08722v1) [paper-pdf](http://arxiv.org/pdf/2408.08722v1)

**Authors**: Samira Kamali Poorazad, Chafika Benzaid, Tarik Taleb

**Abstract**: Industrial Internet of Things (IIoT) is highly sensitive to data privacy and cybersecurity threats. Federated Learning (FL) has emerged as a solution for preserving privacy, enabling private data to remain on local IIoT clients while cooperatively training models to detect network anomalies. However, both synchronous and asynchronous FL architectures exhibit limitations, particularly when dealing with clients with varying speeds due to data heterogeneity and resource constraints. Synchronous architecture suffers from straggler effects, while asynchronous methods encounter communication bottlenecks. Additionally, FL models are prone to adversarial inference attacks aimed at disclosing private training data. To address these challenges, we propose a Buffered FL (BFL) framework empowered by homomorphic encryption for anomaly detection in heterogeneous IIoT environments. BFL utilizes a novel weighted average time approach to mitigate both straggler effects and communication bottlenecks, ensuring fairness between clients with varying processing speeds through collaboration with a buffer-based server. The performance results, derived from two datasets, show the superiority of BFL compared to state-of-the-art FL methods, demonstrating improved accuracy and convergence speed while enhancing privacy preservation.

摘要: 工业物联网(IIoT)对数据隐私和网络安全威胁高度敏感。联合学习(FL)已经成为一种保护隐私的解决方案，使私有数据能够保留在本地IIoT客户端上，同时协作训练模型来检测网络异常。然而，同步和异步FL体系结构都表现出局限性，特别是在处理由于数据异构性和资源限制而具有不同速度的客户端时。同步体系结构会受到落后者的影响，而异步方法则会遇到通信瓶颈。此外，FL模型容易受到旨在泄露私人训练数据的对抗性推理攻击。为了应对这些挑战，我们提出了一种基于同态加密的缓冲FL(BFL)框架，用于异类IIoT环境中的异常检测。BFL使用一种新颖的加权平均时间方法来缓解掉队效应和通信瓶颈，通过与基于缓冲区的服务器协作来确保具有不同处理速度的客户端之间的公平性。在两个数据集上的实验结果表明，BFL方法与最先进的FL方法相比具有更好的性能，在增强隐私保护的同时，提高了准确率和收敛速度。



## **25. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIIR：基于互信息的对抗鲁棒性的掩蔽图像建模 cs.CV

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2312.04960v3) [paper-pdf](http://arxiv.org/pdf/2312.04960v3)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve excellent performance in various tasks, but they are also vulnerable to adversarial attacks. Building robust ViTs is highly dependent on dedicated Adversarial Training (AT) strategies. However, current ViTs' adversarial training only employs well-established training approaches from convolutional neural network (CNN) training, where pre-training provides the basis for AT fine-tuning with the additional help of tailored data augmentations. In this paper, we take a closer look at the adversarial robustness of ViTs by providing a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained by utilizing the MI bounds. Based on this finding, we propose a masked autoencoder-based pre-training method, MIMIR, that employs an MI penalty to facilitate the adversarial training of ViTs. Extensive experiments show that MIMIR outperforms state-of-the-art adversarially trained ViTs on benchmark datasets with higher natural and robust accuracy, indicating that ViTs can substantially benefit from exploiting MI. In addition, we consider two adaptive attacks by assuming that the adversary is aware of the MIMIR design, which further verifies the provided robustness.

摘要: 视觉变形器(VITS)在各种任务中取得了优异的性能，但它们也容易受到对手的攻击。建立强大的VITS高度依赖于专门的对手训练(AT)策略。然而，目前VITS的对抗性训练只采用卷积神经网络(CNN)训练中成熟的训练方法，其中预训练是AT微调的基础，并辅之以定制的数据扩充。在本文中，我们通过在基于自动编码器的自监督预训练中提供一种新的理论互信息(MI)分析来进一步研究VITS的对抗健壮性。具体地说，我们证明了在基于VIT的自动编码器中，敌意示例与其潜在表示之间的MI应该通过利用MI界来限制。基于这一发现，我们提出了一种基于掩蔽自动编码器的预训练方法MIMIR，该方法使用MI惩罚来促进VITS的对抗性训练。大量的实验表明，MIMIR在基准数据集上的性能优于经过对手训练的最先进的VITS，具有更高的自然和稳健的准确率，表明VITS可以从利用MI中获得实质性的好处。此外，通过假设攻击者知道Mimir设计，我们考虑了两种自适应攻击，这进一步验证了所提供的稳健性。



## **26. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08685v1) [paper-pdf](http://arxiv.org/pdf/2408.08685v1)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, especially for topology attacks, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attack. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph.

摘要: 图神经网络(GNN)很容易受到敌意干扰，尤其是对拓扑攻击，许多提高GNN稳健性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。



## **27. Towards Physical World Backdoor Attacks against Skeleton Action Recognition**

针对骨架动作识别的物理世界后门攻击 cs.CR

Accepted by ECCV 2024

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08671v1) [paper-pdf](http://arxiv.org/pdf/2408.08671v1)

**Authors**: Qichen Zheng, Yi Yu, Siyuan Yang, Jun Liu, Kwok-Yan Lam, Alex Kot

**Abstract**: Skeleton Action Recognition (SAR) has attracted significant interest for its efficient representation of the human skeletal structure. Despite its advancements, recent studies have raised security concerns in SAR models, particularly their vulnerability to adversarial attacks. However, such strategies are limited to digital scenarios and ineffective in physical attacks, limiting their real-world applicability. To investigate the vulnerabilities of SAR in the physical world, we introduce the Physical Skeleton Backdoor Attacks (PSBA), the first exploration of physical backdoor attacks against SAR. Considering the practicalities of physical execution, we introduce a novel trigger implantation method that integrates infrequent and imperceivable actions as triggers into the original skeleton data. By incorporating a minimal amount of this manipulated data into the training set, PSBA enables the system misclassify any skeleton sequences into the target class when the trigger action is present. We examine the resilience of PSBA in both poisoned and clean-label scenarios, demonstrating its efficacy across a range of datasets, poisoning ratios, and model architectures. Additionally, we introduce a trigger-enhancing strategy to strengthen attack performance in the clean label setting. The robustness of PSBA is tested against three distinct backdoor defenses, and the stealthiness of PSBA is evaluated using two quantitative metrics. Furthermore, by employing a Kinect V2 camera, we compile a dataset of human actions from the real world to mimic physical attack situations, with our findings confirming the effectiveness of our proposed attacks. Our project website can be found at https://qichenzheng.github.io/psba-website.

摘要: 骨骼动作识别(SAR)以其对人体骨骼结构的有效表示而引起了人们的极大兴趣。尽管它取得了进展，但最近的研究提出了对SAR模型的安全担忧，特别是它们对对手攻击的脆弱性。然而，这些策略仅限于数字场景，在物理攻击中无效，限制了它们在现实世界中的适用性。为了研究物理世界中SAR的脆弱性，我们引入了物理骨架后门攻击(PSBA)，这是对SAR物理后门攻击的首次探索。考虑到物理执行的实用性，我们引入了一种新的触发器植入方法，将不频繁和不可察觉的动作作为触发器集成到原始骨架数据中。通过将最少量的这种操作数据合并到训练集中，PSBA使系统能够在存在触发动作时将任何骨架序列错误地分类到目标类中。我们检查了PSBA在中毒和干净标签情况下的弹性，展示了它在一系列数据集、中毒比率和模型体系结构中的有效性。此外，我们还引入了触发增强策略来增强干净标签设置下的攻击性能。PSBA的健壮性针对三种不同的后门防御进行了测试，并使用两个定量指标来评估PSBA的隐蔽性。此外，通过使用Kinect V2摄像头，我们汇编了真实世界中人类行为的数据集，以模拟物理攻击情况，我们的发现证实了我们提出的攻击的有效性。我们的项目网站可以在https://qichenzheng.github.io/psba-website.找到



## **28. A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning**

一个偷偷摸摸的犯错者：针对分裂学习的以冲突为导向的重建攻击 cs.CR

Accepted to CVPR 2024

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2405.04115v2) [paper-pdf](http://arxiv.org/pdf/2405.04115v2)

**Authors**: Xiaoyang Xu, Mengda Yang, Wenzhe Yi, Ziang Li, Juan Wang, Hongxin Hu, Yong Zhuang, Yaxin Liu

**Abstract**: Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.

摘要: Split Learning(SL)是一种分布式学习框架，以其隐私保护功能和最小的计算要求而闻名。以前的研究一直强调，通过服务器对手重建训练数据，SL系统中潜在的隐私泄露。然而，这些研究往往依赖强假设或折衷系统效用来提高攻击性能。介绍了一种新的基于SL的半诚实数据重构攻击--面向特征的重构攻击(FORA)。与以前的工作不同，FORA依赖于有限的先验知识，特别是服务器使用来自公共的辅助样本，而不知道任何客户的私人信息。这使得Fora能够悄悄地进行攻击，并实现稳健的性能。Fora利用的关键漏洞是受害者客户端输出的粉碎数据中暴露的模型表示首选项。FORA通过特征级迁移学习构造了一个替代客户，旨在更好地模拟受害客户的表征偏好。利用这个替代客户端，服务器训练攻击模型以有效地重建私有数据。广泛的实验表明，与最先进的方法相比，FORA的性能更优越。此外，本文还系统地评估了该方法在不同环境和先进防御策略下的适用性。



## **29. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

稳健的神经信息检索：对抗性和非分布性的角度 cs.IR

Survey paper

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.06992v2) [paper-pdf](http://arxiv.org/pdf/2407.06992v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.

摘要: 神经信息检索(IR)模型的最新进展显著提高了它们在各种IR任务中的有效性。这些模型的稳健性对于确保它们在实践中的可靠性至关重要，也引起了人们的极大关注。随着对稳健IR的广泛研究的提出，我们认为现在是巩固当前状况、从现有方法中收集见解并为未来发展奠定基础的好时机。我们认为信息检索的稳健性是一个多方面的概念，强调了它对对抗攻击、分布外(OOD)场景和性能差异的必要性。以对抗性和面向对象的稳健性为重点，我们分别剖析了密集检索模型(DRM)和神经排名模型(NRM)的稳健性解决方案，将它们识别为神经IR管道的关键组件。我们提供了对现有方法、数据集和评估度量的深入讨论，揭示了大型语言模型时代的挑战和未来方向。据我们所知，这是关于神经IR模型稳健性的第一次全面调查，我们还将在SIGIR2024\url{https://sigir2024-robust-information-retrieval.github.io}.上进行我们的第一次教程演示在组织现有工作的同时，我们还介绍了稳健IR基准(BSTIR)，这是一个用于稳健神经信息检索的异质评估基准，可在\url{https://github.com/Davion-Liu/BestIR}.希望本研究为今后研究信息检索模型的健壮性提供有用的线索，并为开发可信搜索引擎\url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.提供帮助



## **30. Efficient Image-to-Image Diffusion Classifier for Adversarial Robustness**

具有对抗鲁棒性的高效图像到图像扩散分类器 cs.CV

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08502v1) [paper-pdf](http://arxiv.org/pdf/2408.08502v1)

**Authors**: Hefei Mei, Minjing Dong, Chang Xu

**Abstract**: Diffusion models (DMs) have demonstrated great potential in the field of adversarial robustness, where DM-based defense methods can achieve superior defense capability without adversarial training. However, they all require huge computational costs due to the usage of large-scale pre-trained DMs, making it difficult to conduct full evaluation under strong attacks and compare with traditional CNN-based methods. Simply reducing the network size and timesteps in DMs could significantly harm the image generation quality, which invalidates previous frameworks. To alleviate this issue, we redesign the diffusion framework from generating high-quality images to predicting distinguishable image labels. Specifically, we employ an image translation framework to learn many-to-one mapping from input samples to designed orthogonal image labels. Based on this framework, we introduce an efficient Image-to-Image diffusion classifier with a pruned U-Net structure and reduced diffusion timesteps. Besides the framework, we redesign the optimization objective of DMs to fit the target of image classification, where a new classification loss is incorporated in the DM-based image translation framework to distinguish the generated label from those of other classes. We conduct sufficient evaluations of the proposed classifier under various attacks on popular benchmarks. Extensive experiments show that our method achieves better adversarial robustness with fewer computational costs than DM-based and CNN-based methods. The code is available at https://github.com/hfmei/IDC.

摘要: 扩散模型在对抗鲁棒性领域显示出了巨大的潜力，基于扩散模型的防御方法可以在不需要对手训练的情况下获得优越的防御能力。然而，由于它们都需要使用大规模的预先训练的DM，因此它们都需要巨大的计算代价，这使得在强攻击下进行充分评估并与传统的基于CNN的方法进行比较是困难的。简单地减少DM中的网络大小和时间步长可能会严重损害映像生成质量，从而使之前的框架失效。为了缓解这个问题，我们重新设计了扩散框架，从生成高质量的图像到预测可区分的图像标签。具体地说，我们使用一个图像转换框架来学习从输入样本到设计的正交图像标签的多对一映射。基于该框架，我们提出了一种高效的图像到图像扩散分类器，该分类器具有修剪的U网结构和减少的扩散时间。除了该框架外，我们还重新设计了DMS的优化目标，以适应图像分类的目标，其中在基于DM的图像翻译框架中引入了新的分类损失，以区分生成的标签和其他类别的标签。在各种针对流行基准的攻击下，我们对提出的分类器进行了充分的评估。大量实验表明，与基于DM的方法和基于CNN的方法相比，我们的方法具有更好的对抗健壮性和更少的计算代价。代码可在https://github.com/hfmei/IDC.上获得



## **31. DFT-Based Adversarial Attack Detection in MRI Brain Imaging: Enhancing Diagnostic Accuracy in Alzheimer's Case Studies**

MRI脑部成像中基于DART的对抗性攻击检测：提高阿尔茨海默病病例研究的诊断准确性 eess.IV

10 pages, 4 figures, conference

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08489v1) [paper-pdf](http://arxiv.org/pdf/2408.08489v1)

**Authors**: Mohammad Hossein Najafi, Mohammad Morsali, Mohammadmahdi Vahediahmar, Saeed Bagheri Shouraki

**Abstract**: Recent advancements in deep learning, particularly in medical imaging, have significantly propelled the progress of healthcare systems. However, examining the robustness of medical images against adversarial attacks is crucial due to their real-world applications and profound impact on individuals' health. These attacks can result in misclassifications in disease diagnosis, potentially leading to severe consequences. Numerous studies have explored both the implementation of adversarial attacks on medical images and the development of defense mechanisms against these threats, highlighting the vulnerabilities of deep neural networks to such adversarial activities. In this study, we investigate adversarial attacks on images associated with Alzheimer's disease and propose a defensive method to counteract these attacks. Specifically, we examine adversarial attacks that employ frequency domain transformations on Alzheimer's disease images, along with other well-known adversarial attacks. Our approach utilizes a convolutional neural network (CNN)-based autoencoder architecture in conjunction with the two-dimensional Fourier transform of images for detection purposes. The simulation results demonstrate that our detection and defense mechanism effectively mitigates several adversarial attacks, thereby enhancing the robustness of deep neural networks against such vulnerabilities.

摘要: 最近在深度学习方面的进展，特别是在医学成像方面，极大地推动了医疗保健系统的进步。然而，由于医学图像在现实世界中的应用以及对个人健康的深刻影响，检测其对敌意攻击的稳健性至关重要。这些攻击可能导致疾病诊断中的错误分类，可能导致严重后果。许多研究都探索了对医学图像的对抗性攻击的实现和针对这些威胁的防御机制的发展，突出了深层神经网络对这种对抗性活动的脆弱性。在这项研究中，我们调查了与阿尔茨海默病相关的图像的对抗性攻击，并提出了一种防御方法来对抗这些攻击。具体地说，我们研究了对阿尔茨海默病图像进行频域变换的对抗性攻击，以及其他众所周知的对抗性攻击。我们的方法利用基于卷积神经网络(CNN)的自动编码器体系结构，并结合图像的二维傅立叶变换进行检测。仿真结果表明，我们的检测和防御机制有效地缓解了多个对抗性攻击，从而增强了深度神经网络对此类漏洞的鲁棒性。



## **32. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

不确定性和校准对可能性比隶属推理攻击的影响 cs.IT

13 pages, 20 figures

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2402.10686v2) [paper-pdf](http://arxiv.org/pdf/2402.10686v2)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.

摘要: 在成员关系推理攻击(MIA)中，攻击者利用典型机器学习模型表现出的过度自信来确定是否使用特定数据点来训练目标模型。在本文中，我们在信息论框架内分析了最新的似然比攻击(LIRA)的性能，该框架允许研究真实数据生成过程中的任意不确定性的影响、有限训练数据集引起的认知不确定性的影响以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收信息递减的反馈：置信度向量(CV)披露，其中输出概率向量被释放；真实标签置信度(TLC)披露，其中模型仅提供分配给真实标签的概率；以及决策集(DS)披露，其中产生与保形预测相同的自适应预测集。我们得出了MIA对手的优势界限，目的是对不确定性和校准对MIA有效性的影响提供见解。仿真结果表明，推导出的解析界很好地预测了MIA的有效性。



## **33. A Multi-task Adversarial Attack Against Face Authentication**

针对人脸认证的多任务对抗攻击 cs.CV

Accepted by ACM Transactions on Multimedia Computing, Communications,  and Applications

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08205v1) [paper-pdf](http://arxiv.org/pdf/2408.08205v1)

**Authors**: Hanrui Wang, Shuo Wang, Cunjian Chen, Massimo Tistarelli, Zhe Jin

**Abstract**: Deep-learning-based identity management systems, such as face authentication systems, are vulnerable to adversarial attacks. However, existing attacks are typically designed for single-task purposes, which means they are tailored to exploit vulnerabilities unique to the individual target rather than being adaptable for multiple users or systems. This limitation makes them unsuitable for certain attack scenarios, such as morphing, universal, transferable, and counter attacks. In this paper, we propose a multi-task adversarial attack algorithm called MTADV that are adaptable for multiple users or systems. By interpreting these scenarios as multi-task attacks, MTADV is applicable to both single- and multi-task attacks, and feasible in the white- and gray-box settings. Furthermore, MTADV is effective against various face datasets, including LFW, CelebA, and CelebA-HQ, and can work with different deep learning models, such as FaceNet, InsightFace, and CurricularFace. Importantly, MTADV retains its feasibility as a single-task attack targeting a single user/system. To the best of our knowledge, MTADV is the first adversarial attack method that can target all of the aforementioned scenarios in one algorithm.

摘要: 基于深度学习的身份管理系统，如人脸认证系统，容易受到对手攻击。然而，现有攻击通常是为单任务目的而设计的，这意味着它们是为利用单个目标特有的漏洞而定制的，而不是适应多个用户或系统。这种限制使得它们不适合某些攻击场景，例如变形攻击、通用攻击、可传输攻击和反击攻击。本文提出了一种适用于多用户或多系统的多任务对抗性攻击算法MTADV。通过将这些场景解释为多任务攻击，MTADV既适用于单任务攻击，也适用于多任务攻击，并且在白盒和灰盒环境下都是可行的。此外，MTADV对包括LFW、CelebA和CelebA-HQ在内的各种人脸数据集都是有效的，并且可以与不同的深度学习模型一起工作，例如FaceNet、InsightFace和CurousarFace。重要的是，MTADV保持了其作为针对单个用户/系统的单任务攻击的可行性。据我们所知，MTADV是第一种可以在一个算法中针对上述所有场景的对抗性攻击方法。



## **34. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

前置指导：大型语言模型防御越狱攻击的方向盘 cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08924v1) [paper-pdf](http://arxiv.org/pdf/2408.08924v1)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance.

摘要: 近年来，大型语言模型的快速发展在各种任务中取得了显著的性能。然而，研究表明，LLMS容易受到越狱攻击，在越狱攻击中，攻击者可以通过精心制作的提示来诱导生成有害内容。此漏洞对安全使用和推广LLMS构成重大挑战。现有的防御方法从不同的角度提供保护，但往往存在有效性不足或对模型能力产生重大影响的问题。本文提出了一种即插即用、易于部署的越狱防御框架--前缀引导(PG)，它通过直接设置模型输出的前几个令牌来引导模型识别有害提示。这种方法将模型固有的安全功能与外部分类器相结合，以防御越狱攻击。我们在三个模型和五种攻击方法上演示了PG的有效性。与基线相比，我们的方法总体上更有效。此外，在Just-Eval基准上的结果进一步证实了PG在保持模型性能方面的优越性。



## **35. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

大型语言模型Sentinel：对抗性纯化的LLM代理 cs.CL

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2405.20770v2) [paper-pdf](http://arxiv.org/pdf/2405.20770v2)

**Authors**: Guang Lin, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.

摘要: 在过去的两年里，大型语言模型(LLM)的使用取得了快速发展。虽然这些LLM提供了相当大的便利，但它们也引发了安全问题，因为LLM容易受到一些精心设计的文本扰动的敌意攻击。本文介绍了一种新的防御技术--大语言模型哨兵(LLAMOS)，该技术旨在通过在将对抗性文本实例输入目标LLM之前对其进行提纯来增强LLMS的对抗性健壮性。我们的方法包括两个主要部分：a)代理指令，它可以模拟一个新的代理进行对抗性防御，改变最少的字符，在防御攻击的同时保持句子的原始含义；b)防御指导，它提供修改干净或对抗性示例的策略，以确保目标LLMS的有效防御和准确输出。值得注意的是，防御代理展示了强大的防御能力，即使没有从对手的例子中学习。此外，我们还进行了一个有趣的对抗性实验，在这个实验中，我们开发了两个代理，一个用于防御，一个用于攻击，并让他们相互对抗。在敌对的互动中，两个代理都没有完全击败另一个。在开源和闭源LLMS上的大量实验表明，我们的方法有效地防御了对手攻击，从而增强了对手攻击的健壮性。



## **36. Security Challenges of Complex Space Applications: An Empirical Study**

复杂太空应用的安全挑战：实证研究 cs.CR

Presented at the ESA Security for Space Systems (3S) conference on  the 28th of May 2024

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08061v1) [paper-pdf](http://arxiv.org/pdf/2408.08061v1)

**Authors**: Tomas Paulik

**Abstract**: Software applications in the space and defense industries have their unique characteristics: They are complex in structure, mission-critical, and often targets of state-of-the-art cyber attacks sponsored by adversary nation states. These applications have typically a high number of stakeholders in their software component supply chain, data supply chain, and user base. The aforementioned factors make such software applications potentially vulnerable to bad actors, as the widely adopted DevOps tools and practices were not designed for high-complexity and high-risk environments.   In this study, I investigate the security challenges of the development and management of complex space applications, which differentiate the process from the commonly used practices. My findings are based on interviews with five domain experts from the industry and are further supported by a comprehensive review of relevant publications.   To illustrate the dynamics of the problem, I present and discuss an actual software supply chain structure used by Thales Alenia Space, which is one of the largest suppliers of the European Space Agency. Subsequently, I discuss the four most critical security challenges identified by the interviewed experts: Verification of software artifacts, verification of the deployed application, single point of security failure, and data tampering by trusted stakeholders. Furthermore, I present best practices which could be used to overcome each of the given challenges, and whether the interviewed experts think their organization has access to the right tools to address them. Finally, I propose future research of new DevSecOps strategies, practices, and tools which would enable better methods of software integrity verification in the space and defense industries.

摘要: 航天和国防工业中的软件应用程序有其独特的特点：它们结构复杂，任务关键，经常成为敌对民族国家支持的最先进网络攻击的目标。这些应用程序通常在其软件组件供应链、数据供应链和用户群中拥有大量的利益相关者。上述因素使这类软件应用程序可能容易受到不良行为者的攻击，因为广泛采用的DevOps工具和做法不是为高复杂性和高风险环境设计的。在这项研究中，我调查了复杂空间应用程序开发和管理的安全挑战，这一过程有别于通常使用的做法。我的发现是基于对该行业五位领域专家的采访，并得到了对相关出版物的全面审查的进一步支持。为了说明问题的动态，我提出并讨论了泰利斯·阿莱尼亚空间公司使用的实际软件供应链结构，该公司是欧洲航天局的最大供应商之一。随后，我讨论了受访专家确定的四个最关键的安全挑战：验证软件构件、验证已部署的应用程序、单点安全故障以及受信任的利益相关者篡改数据。此外，我提出了可用于克服每一项特定挑战的最佳做法，以及受访专家是否认为他们的组织有权使用适当的工具来应对这些挑战。最后，我提出了对新的DevSecOps策略、实践和工具的未来研究，这些策略、实践和工具将使空间和国防行业中的软件完整性验证方法变得更好。



## **37. LiD-FL: Towards List-Decodable Federated Learning**

LiD-FL：迈向列表可解码联邦学习 cs.LG

26 pages, 5 figures

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.04963v2) [paper-pdf](http://arxiv.org/pdf/2408.04963v2)

**Authors**: Hong Liu, Liren Shan, Han Bao, Ronghui You, Yuhao Yi, Jiancheng Lv

**Abstract**: Federated learning is often used in environments with many unverified participants. Therefore, federated learning under adversarial attacks receives significant attention. This paper proposes an algorithmic framework for list-decodable federated learning, where a central server maintains a list of models, with at least one guaranteed to perform well. The framework has no strict restriction on the fraction of honest workers, extending the applicability of Byzantine federated learning to the scenario with more than half adversaries. Under proper assumptions on the loss function, we prove a convergence theorem for our method. Experimental results, including image classification tasks with both convex and non-convex losses, demonstrate that the proposed algorithm can withstand the malicious majority under various attacks.

摘要: 联邦学习通常用于有许多未经验证的参与者的环境中。因此，对抗攻击下的联邦学习受到了高度关注。本文提出了一种用于列表可解码联邦学习的算法框架，其中中央服务器维护一个模型列表，至少有一个模型保证表现良好。该框架对诚实工人的比例没有严格限制，将拜占庭联邦学习的适用性扩展到有一半以上对手的场景。在对损失函数的适当假设下，我们证明了我们方法的收敛性定理。实验结果（包括具有凸损失和非凸损失的图像分类任务）表明，所提出的算法可以抵御各种攻击下的恶意多数。



## **38. A Survey of Trojan Attacks and Defenses to Deep Neural Networks**

特洛伊木马对深度神经网络的攻击和防御综述 cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08920v1) [paper-pdf](http://arxiv.org/pdf/2408.08920v1)

**Authors**: Lingxin Jin, Xianyu Wen, Wei Jiang, Jinyu Zhan

**Abstract**: Deep Neural Networks (DNNs) have found extensive applications in safety-critical artificial intelligence systems, such as autonomous driving and facial recognition systems. However, recent research has revealed their susceptibility to Neural Network Trojans (NN Trojans) maliciously injected by adversaries. This vulnerability arises due to the intricate architecture and opacity of DNNs, resulting in numerous redundant neurons embedded within the models. Adversaries exploit these vulnerabilities to conceal malicious Trojans within DNNs, thereby causing erroneous outputs and posing substantial threats to the efficacy of DNN-based applications. This article presents a comprehensive survey of Trojan attacks against DNNs and the countermeasure methods employed to mitigate them. Initially, we trace the evolution of the concept from traditional Trojans to NN Trojans, highlighting the feasibility and practicality of generating NN Trojans. Subsequently, we provide an overview of notable works encompassing various attack and defense strategies, facilitating a comparative analysis of their approaches. Through these discussions, we offer constructive insights aimed at refining these techniques. In recognition of the gravity and immediacy of this subject matter, we also assess the feasibility of deploying such attacks in real-world scenarios as opposed to controlled ideal datasets. The potential real-world implications underscore the urgency of addressing this issue effectively.

摘要: 深度神经网络(DNN)在安全关键的人工智能系统中得到了广泛的应用，如自动驾驶和面部识别系统。然而，最近的研究表明，他们对对手恶意注入的神经网络特洛伊木马(NN特洛伊木马)很敏感。这一漏洞是由于DNN的复杂体系结构和不透明造成的，导致模型中嵌入了大量冗余神经元。攻击者利用这些漏洞在DNN中隐藏恶意特洛伊木马程序，从而导致错误输出，并对基于DNN的应用程序的效率构成重大威胁。本文对针对DNN的特洛伊木马攻击进行了全面的调查，并提出了缓解木马攻击的对策。首先，我们追溯了从传统特洛伊木马到NN特洛伊木马的概念演变，强调了生成NN特洛伊木马的可行性和实用性。随后，我们提供了包含各种攻击和防御策略的著名作品的概述，便于对它们的方法进行比较分析。通过这些讨论，我们提供了旨在完善这些技术的建设性见解。认识到这一主题的严重性和紧迫性，我们还评估了在现实世界场景中部署此类攻击的可行性，而不是受控的理想数据集。潜在的现实影响突显了有效解决这一问题的紧迫性。



## **39. Robust Active Learning (RoAL): Countering Dynamic Adversaries in Active Learning with Elastic Weight Consolidation**

稳健主动学习（RoAL）：通过弹性权重巩固对抗主动学习中的动态对手 cs.LG

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.07364v2) [paper-pdf](http://arxiv.org/pdf/2408.07364v2)

**Authors**: Ricky Maulana Fajri, Yulong Pei, Lu Yin, Mykola Pechenizkiy

**Abstract**: Despite significant advancements in active learning and adversarial attacks, the intersection of these two fields remains underexplored, particularly in developing robust active learning frameworks against dynamic adversarial threats. The challenge of developing robust active learning frameworks under dynamic adversarial attacks is critical, as these attacks can lead to catastrophic forgetting within the active learning cycle. This paper introduces Robust Active Learning (RoAL), a novel approach designed to address this issue by integrating Elastic Weight Consolidation (EWC) into the active learning process. Our contributions are threefold: First, we propose a new dynamic adversarial attack that poses significant threats to active learning frameworks. Second, we introduce a novel method that combines EWC with active learning to mitigate catastrophic forgetting caused by dynamic adversarial attacks. Finally, we conduct extensive experimental evaluations to demonstrate the efficacy of our approach. The results show that RoAL not only effectively counters dynamic adversarial threats but also significantly reduces the impact of catastrophic forgetting, thereby enhancing the robustness and performance of active learning systems in adversarial environments.

摘要: 尽管在主动学习和对抗性攻击方面取得了重大进展，但这两个领域的交集仍未得到充分探索，特别是在针对动态对抗性威胁制定强有力的主动学习框架方面。在动态对抗性攻击下开发健壮的主动学习框架的挑战是至关重要的，因为这些攻击可能会导致主动学习周期内的灾难性遗忘。本文介绍了一种新的方法--稳健主动学习(ROAL)，它将弹性权重合并(EWC)融入到主动学习过程中，从而解决了这个问题。我们的贡献有三个方面：首先，我们提出了一种新的动态对抗性攻击，它对主动学习框架构成了重大威胁。其次，我们提出了一种新的方法，将EWC和主动学习相结合来缓解动态对手攻击导致的灾难性遗忘。最后，我们进行了广泛的实验评估，以证明我们的方法的有效性。结果表明，ROAL不仅能有效地对抗动态对手威胁，而且能显著降低灾难性遗忘的影响，从而增强主动学习系统在对抗环境中的健壮性和性能。



## **40. Enhancing Adversarial Attacks via Parameter Adaptive Adversarial Attack**

通过参数自适应对抗攻击增强对抗攻击 cs.LG

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07733v1) [paper-pdf](http://arxiv.org/pdf/2408.07733v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Chenyu Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: In recent times, the swift evolution of adversarial attacks has captured widespread attention, particularly concerning their transferability and other performance attributes. These techniques are primarily executed at the sample level, frequently overlooking the intrinsic parameters of models. Such neglect suggests that the perturbations introduced in adversarial samples might have the potential for further reduction. Given the essence of adversarial attacks is to impair model integrity with minimal noise on original samples, exploring avenues to maximize the utility of such perturbations is imperative. Against this backdrop, we have delved into the complexities of adversarial attack algorithms, dissecting the adversarial process into two critical phases: the Directional Supervision Process (DSP) and the Directional Optimization Process (DOP). While DSP determines the direction of updates based on the current samples and model parameters, it has been observed that existing model parameters may not always be conducive to adversarial attacks. The impact of models on adversarial efficacy is often overlooked in current research, leading to the neglect of DSP. We propose that under certain conditions, fine-tuning model parameters can significantly enhance the quality of DSP. For the first time, we propose that under certain conditions, fine-tuning model parameters can significantly improve the quality of the DSP. We provide, for the first time, rigorous mathematical definitions and proofs for these conditions, and introduce multiple methods for fine-tuning model parameters within DSP. Our extensive experiments substantiate the effectiveness of the proposed P3A method. Our code is accessible at: https://anonymous.4open.science/r/P3A-A12C/

摘要: 近年来，对抗性攻击的迅速演变引起了人们的广泛关注，特别是关于它们的可转移性和其他性能属性。这些技术主要是在样本级别执行的，经常忽略模型的内在参数。这种忽视表明，对抗性样本中引入的扰动可能有进一步减少的潜力。鉴于对抗性攻击的本质是在原始样本上以最小的噪声破坏模型的完整性，探索最大化此类扰动的效用的方法势在必行。在此背景下，我们深入研究了对抗攻击算法的复杂性，将对抗过程分解为两个关键阶段：方向监督过程(DSP)和方向优化过程(DOP)。虽然DSP根据当前样本和模型参数来确定更新方向，但已经观察到，现有的模型参数可能并不总是有利于对抗性攻击。目前的研究往往忽略了模型对对抗效能的影响，从而导致了对数字信号处理器的忽视。我们提出，在一定条件下，微调模型参数可以显著提高数字信号处理器的质量。首次提出在一定条件下，微调模型参数可以显著提高数字信号处理的质量。我们首次为这些条件提供了严格的数学定义和证明，并介绍了在DSP中微调模型参数的多种方法。我们的大量实验证明了提出的P3A方法的有效性。我们的代码可通过以下网址访问：https://anonymous.4open.science/r/P3A-A12C/



## **41. TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases**

TabularBench：在现实世界的实例中对Tabular深度学习的对抗鲁棒性进行基准测试 cs.LG

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07579v1) [paper-pdf](http://arxiv.org/pdf/2408.07579v1)

**Authors**: Thibault Simonetto, Salah Ghamizi, Maxime Cordy

**Abstract**: While adversarial robustness in computer vision is a mature research field, fewer researchers have tackled the evasion attacks against tabular deep learning, and even fewer investigated robustification mechanisms and reliable defenses. We hypothesize that this lag in the research on tabular adversarial attacks is in part due to the lack of standardized benchmarks. To fill this gap, we propose TabularBench, the first comprehensive benchmark of robustness of tabular deep learning classification models. We evaluated adversarial robustness with CAA, an ensemble of gradient and search attacks which was recently demonstrated as the most effective attack against a tabular model. In addition to our open benchmark (https://github.com/serval-uni-lu/tabularbench) where we welcome submissions of new models and defenses, we implement 7 robustification mechanisms inspired by state-of-the-art defenses in computer vision and propose the largest benchmark of robust tabular deep learning over 200 models across five critical scenarios in finance, healthcare and security. We curated real datasets for each use case, augmented with hundreds of thousands of realistic synthetic inputs, and trained and assessed our models with and without data augmentations. We open-source our library that provides API access to all our pre-trained robust tabular models, and the largest datasets of real and synthetic tabular inputs. Finally, we analyze the impact of various defenses on the robustness and provide actionable insights to design new defenses and robustification mechanisms.

摘要: 虽然计算机视觉中的对抗稳健性是一个成熟的研究领域，但针对表格式深度学习的逃避攻击的研究人员还很少，更少的人研究了鲁棒机制和可靠的防御措施。我们假设，表格对抗性攻击研究的这种滞后部分是由于缺乏标准化的基准。为了填补这一空白，我们提出了第一个全面的表格式深度学习分类模型稳健性基准TumularBch。我们使用CAA来评估对手的健壮性，CAA是梯度攻击和搜索攻击的集合，最近被证明是针对表格模型的最有效的攻击。除了我们欢迎提交新模型和防御措施的开放基准(https://github.com/serval-uni-lu/tabularbench)之外，我们还实施了7个受计算机视觉最先进防御措施启发的鲁棒性机制，并提出了针对金融、医疗保健和安全领域的五个关键场景的200多个模型的最大稳健表格深度学习基准。我们为每个用例管理真实的数据集，使用数十万真实的合成输入进行扩充，并在有和没有数据扩充的情况下对我们的模型进行训练和评估。我们开源了我们的库，它提供了对我们所有预先训练的健壮表格模型的API访问，以及真实和合成表格输入的最大数据集。最后，我们分析了各种防御对健壮性的影响，并为设计新的防御和鲁棒性机制提供了可操作的见解。



## **42. Achieving Data Efficient Neural Networks with Hybrid Concept-based Models**

利用混合基于概念的模型实现数据高效的神经网络 cs.LG

11 pages, 8 figures, appendix

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07438v1) [paper-pdf](http://arxiv.org/pdf/2408.07438v1)

**Authors**: Tobias A. Opsahl, Vegard Antun

**Abstract**: Most datasets used for supervised machine learning consist of a single label per data point. However, in cases where more information than just the class label is available, would it be possible to train models more efficiently? We introduce two novel model architectures, which we call hybrid concept-based models, that train using both class labels and additional information in the dataset referred to as concepts. In order to thoroughly assess their performance, we introduce ConceptShapes, an open and flexible class of datasets with concept labels. We show that the hybrid concept-based models outperform standard computer vision models and previously proposed concept-based models with respect to accuracy, especially in sparse data settings. We also introduce an algorithm for performing adversarial concept attacks, where an image is perturbed in a way that does not change a concept-based model's concept predictions, but changes the class prediction. The existence of such adversarial examples raises questions about the interpretable qualities promised by concept-based models.

摘要: 大多数用于有监督机器学习的数据集都由每个数据点的单个标签组成。然而，在除了类标签之外还有更多信息的情况下，是否有可能更有效地训练模型？我们引入了两种新的模型体系结构，我们称之为混合基于概念的模型，它们使用类别标签和数据集中的额外信息进行训练，称为概念。为了彻底评估它们的性能，我们引入了概念形状，这是一类开放的、灵活的带有概念标签的数据集。结果表明，基于概念的混合模型在精度方面优于标准计算机视觉模型和以前提出的基于概念的模型，特别是在稀疏数据环境下。我们还介绍了一种执行对抗性概念攻击的算法，其中图像被扰动的方式不会改变基于概念的模型的概念预测，但会改变类别预测。这种对抗性例子的存在引发了人们对基于概念的模型所承诺的可解释性的质疑。



## **43. A Survey of Fragile Model Watermarking**

脆弱模型水印综述 cs.CR

There will be more revisions to the wording and image additions, and  we hope to withdraw the previous version

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2406.04809v5) [paper-pdf](http://arxiv.org/pdf/2406.04809v5)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.

摘要: 模型脆弱水印受到神经网络对抗攻击领域和传统多媒体脆弱水印的启发，逐渐成为检测篡改的有力工具，并在近年来得到了快速发展。与广泛用于识别模型版权的稳健水印不同，模型的脆弱水印旨在识别模型是否遭受了意外更改，例如后门、中毒、压缩等。这些更改可能会给模型用户带来未知的风险，例如在经典自动驾驶场景中将停车标志误识别为限速标志。本文概述了模型脆弱水印领域自诞生以来的相关工作，对其进行了分类，揭示了该领域的发展轨迹，从而为模型脆弱水印的未来工作提供了全面的综述。



## **44. BadMerging: Backdoor Attacks Against Model Merging**

BadMerging：针对模型合并的后门攻击 cs.CR

To appear in ACM Conference on Computer and Communications Security  (CCS), 2024

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07362v1) [paper-pdf](http://arxiv.org/pdf/2408.07362v1)

**Authors**: Jinghuai Zhang, Jianfeng Chi, Zheng Li, Kunlin Cai, Yang Zhang, Yuan Tian

**Abstract**: Fine-tuning pre-trained models for downstream tasks has led to a proliferation of open-sourced task-specific models. Recently, Model Merging (MM) has emerged as an effective approach to facilitate knowledge transfer among these independently fine-tuned models. MM directly combines multiple fine-tuned task-specific models into a merged model without additional training, and the resulting model shows enhanced capabilities in multiple tasks. Although MM provides great utility, it may come with security risks because an adversary can exploit MM to affect multiple downstream tasks. However, the security risks of MM have barely been studied. In this paper, we first find that MM, as a new learning paradigm, introduces unique challenges for existing backdoor attacks due to the merging process. To address these challenges, we introduce BadMerging, the first backdoor attack specifically designed for MM. Notably, BadMerging allows an adversary to compromise the entire merged model by contributing as few as one backdoored task-specific model. BadMerging comprises a two-stage attack mechanism and a novel feature-interpolation-based loss to enhance the robustness of embedded backdoors against the changes of different merging parameters. Considering that a merged model may incorporate tasks from different domains, BadMerging can jointly compromise the tasks provided by the adversary (on-task attack) and other contributors (off-task attack) and solve the corresponding unique challenges with novel attack designs. Extensive experiments show that BadMerging achieves remarkable attacks against various MM algorithms. Our ablation study demonstrates that the proposed attack designs can progressively contribute to the attack performance. Finally, we show that prior defense mechanisms fail to defend against our attacks, highlighting the need for more advanced defense.

摘要: 为下游任务微调预先训练的模型导致了开源特定于任务的模型的激增。最近，模型合并(MM)已成为促进这些独立微调模型之间知识转移的一种有效方法。MM直接将多个微调的特定于任务的模型组合成一个合并模型，而无需额外的培训，所得到的模型在多个任务中显示出增强的能力。虽然MM提供了强大的实用程序，但它可能伴随着安全风险，因为攻击者可以利用MM来影响多个下游任务。然而，MM的安全风险几乎没有被研究过。在本文中，我们首先发现MM作为一种新的学习范式，由于合并过程给现有的后门攻击带来了独特的挑战。为了应对这些挑战，我们引入了BadMerging，这是第一个专门为MM设计的后门攻击。值得注意的是，BadMerging允许对手通过贡献一个后备的特定任务模型来危害整个合并的模型。BadMerging包括一种两阶段攻击机制和一种新的基于特征内插的损失机制，以增强嵌入式后门对不同合并参数变化的鲁棒性。考虑到合并模型可能包含来自不同领域的任务，BadMerging可以联合妥协对手(任务上攻击)和其他贡献者(任务外攻击)提供的任务，并以新颖的攻击设计解决相应的独特挑战。大量实验表明，BadMerging对各种MM算法具有显著的攻击效果。我们的烧蚀研究表明，所提出的攻击设计可以逐渐对攻击性能做出贡献。最后，我们展示了先前的防御机制无法防御我们的攻击，强调了需要更高级的防御。



## **45. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

图代理网络：赋予节点推理能力以对抗复原力 cs.LG

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2306.06909v3) [paper-pdf](http://arxiv.org/pdf/2306.06909v3)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **46. Elephants Do Not Forget: Differential Privacy with State Continuity for Privacy Budget**

大象不要忘记：隐私预算的状态连续性差异隐私 cs.CR

In Proceedings of the 2024 ACM SIGSAC Conference on Computer and  Communications Security (CCS 2024)

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2401.17628v2) [paper-pdf](http://arxiv.org/pdf/2401.17628v2)

**Authors**: Jiankai Jin, Chitchanok Chuengsatiansup, Toby Murray, Benjamin I. P. Rubinstein, Yuval Yarom, Olga Ohrimenko

**Abstract**: Current implementations of differentially-private (DP) systems either lack support to track the global privacy budget consumed on a dataset, or fail to faithfully maintain the state continuity of this budget. We show that failure to maintain a privacy budget enables an adversary to mount replay, rollback and fork attacks - obtaining answers to many more queries than what a secure system would allow. As a result the attacker can reconstruct secret data that DP aims to protect - even if DP code runs in a Trusted Execution Environment (TEE). We propose ElephantDP, a system that aims to provide the same guarantees as a trusted curator in the global DP model would, albeit set in an untrusted environment. Our system relies on a state continuity module to provide protection for the privacy budget and a TEE to faithfully execute DP code and update the budget. To provide security, our protocol makes several design choices including the content of the persistent state and the order between budget updates and query answers. We prove that ElephantDP provides liveness (i.e., the protocol can restart from a correct state and respond to queries as long as the budget is not exceeded) and DP confidentiality (i.e., an attacker learns about a dataset as much as it would from interacting with a trusted curator). Our implementation and evaluation of the protocol use Intel SGX as a TEE to run the DP code and a network of TEEs to maintain state continuity. Compared to an insecure baseline, we observe 1.1-3.2$\times$ overheads and lower relative overheads for complex DP queries.

摘要: 当前的差异私有(DP)系统的实现要么缺乏对跟踪数据集上消耗的全球隐私预算的支持，要么不能忠实地维持该预算的状态连续性。我们表明，未能维持隐私预算使对手能够发动重播、回滚和分叉攻击--获得比安全系统所允许的更多的查询答案。因此，攻击者可以重建DP旨在保护的秘密数据-即使DP代码在可信执行环境(TEE)中运行。我们提出了ElephantDP，这是一个旨在提供与全局DP模型中的可信管理员相同的保证的系统，尽管它设置在不可信的环境中。我们的系统依赖于状态连续性模块来为隐私预算提供保护，并依靠TEE来忠实地执行DP代码和更新预算。为了提供安全性，我们的协议做出了几个设计选择，包括持久状态的内容以及预算更新和查询答案之间的顺序。我们证明了ElephantDP提供了活跃性(即，只要不超过预算，协议就可以从正确的状态重新启动并响应查询)和DP机密性(即，攻击者通过与可信管理员交互来了解关于数据集的信息)。我们的协议实现和评估使用Intel SGX作为TEE来运行DP代码，并使用TEE网络来保持状态连续性。与不安全的基准相比，对于复杂的DP查询，我们观察到1.1-3.2$\倍$的开销和较低的相对开销。



## **47. Exploiting Leakage in Password Managers via Injection Attacks**

通过注入攻击利用密码管理器中的泄露 cs.CR

Full version of the paper published in USENIX Security 2024

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2408.07054v1) [paper-pdf](http://arxiv.org/pdf/2408.07054v1)

**Authors**: Andrés Fábrega, Armin Namavari, Rachit Agarwal, Ben Nassi, Thomas Ristenpart

**Abstract**: This work explores injection attacks against password managers. In this setting, the adversary (only) controls their own application client, which they use to "inject" chosen payloads to a victim's client via, for example, sharing credentials with them. The injections are interleaved with adversarial observations of some form of protected state (such as encrypted vault exports or the network traffic received by the application servers), from which the adversary backs out confidential information. We uncover a series of general design patterns in popular password managers that lead to vulnerabilities allowing an adversary to efficiently recover passwords, URLs, usernames, and attachments. We develop general attack templates to exploit these design patterns and experimentally showcase their practical efficacy via analysis of ten distinct password manager applications. We disclosed our findings to these vendors, many of which deployed mitigations.

摘要: 这项工作探讨了针对密码管理器的注入攻击。在这种设置中，对手（仅）控制他们自己的应用程序客户端，他们使用该客户端通过例如与受害者共享凭证等方式将选择的有效负载“注入”到受害者的客户端。注入与某种形式的受保护状态（例如加密的保险库出口或应用服务器接收的网络流量）的敌对观察交织在一起，对手从中撤回机密信息。我们发现了流行的密码管理器中的一系列通用设计模式，这些模式会导致漏洞，使对手能够有效地恢复密码、URL、用户名和附件。我们开发通用攻击模板来利用这些设计模式，并通过分析十个不同的密码管理器应用程序来实验性地展示它们的实际功效。我们向这些供应商披露了我们的调查结果，其中许多供应商都部署了缓解措施。



## **48. Maintaining Adversarial Robustness in Continuous Learning**

在持续学习中保持对抗稳健性 cs.LG

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2402.11196v2) [paper-pdf](http://arxiv.org/pdf/2402.11196v2)

**Authors**: Xiaolei Ru, Xiaowei Cao, Zijia Liu, Jack Murdoch Moore, Xin-Ya Zhang, Xia Zhu, Wenjia Wei, Gang Yan

**Abstract**: Adversarial robustness is essential for security and reliability of machine learning systems. However, adversarial robustness enhanced by defense algorithms is easily erased as the neural network's weights update to learn new tasks. To address this vulnerability, it is essential to improve the capability of neural networks in terms of robust continual learning. Specially, we propose a novel gradient projection technique that effectively stabilizes sample gradients from previous data by orthogonally projecting back-propagation gradients onto a crucial subspace before using them for weight updates. This technique can maintaining robustness by collaborating with a class of defense algorithms through sample gradient smoothing. The experimental results on four benchmarks including Split-CIFAR100 and Split-miniImageNet, demonstrate that the superiority of the proposed approach in mitigating rapidly degradation of robustness during continual learning even when facing strong adversarial attacks.

摘要: 对抗健壮性对于机器学习系统的安全性和可靠性至关重要。然而，随着神经网络的权重更新以学习新的任务，防御算法增强的对抗性健壮性很容易被抹去。为了解决这一弱点，必须提高神经网络在稳健的持续学习方面的能力。特别是，我们提出了一种新的梯度投影技术，该技术通过将反向传播梯度正交投影到关键子空间上，然后使用它们进行权重更新，有效地稳定了来自先前数据的样本梯度。该技术通过与一类防御算法协作，通过样本梯度平滑来保持鲁棒性。在Split-CIFAR100和Split-mini ImageNet四个基准测试上的实验结果表明，该方法在连续学习过程中即使面对强大的对手攻击也能有效缓解健壮性的快速退化。



## **49. DePatch: Towards Robust Adversarial Patch for Evading Person Detectors in the Real World**

Depatch：迈向在现实世界中躲避人员探测器的强大对抗补丁 cs.CV

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2408.06625v1) [paper-pdf](http://arxiv.org/pdf/2408.06625v1)

**Authors**: Jikang Cheng, Ying Zhang, Zhongyuan Wang, Zou Qin, Chen Li

**Abstract**: Recent years have seen an increasing interest in physical adversarial attacks, which aim to craft deployable patterns for deceiving deep neural networks, especially for person detectors. However, the adversarial patterns of existing patch-based attacks heavily suffer from the self-coupling issue, where a degradation, caused by physical transformations, in any small patch segment can result in a complete adversarial dysfunction, leading to poor robustness in the complex real world. Upon this observation, we introduce the Decoupled adversarial Patch (DePatch) attack to address the self-coupling issue of adversarial patches. Specifically, we divide the adversarial patch into block-wise segments, and reduce the inter-dependency among these segments through randomly erasing out some segments during the optimization. We further introduce a border shifting operation and a progressive decoupling strategy to improve the overall attack capabilities. Extensive experiments demonstrate the superior performance of our method over other physical adversarial attacks, especially in the real world.

摘要: 近年来，人们对物理对抗性攻击越来越感兴趣，这种攻击旨在设计可部署的模式来欺骗深层神经网络，特别是用于个人探测器。然而，现有的基于补丁的攻击的对抗性模式严重受到自耦合问题的影响，在任何一个小的补丁片段中，由物理变换引起的退化可能导致完全的对抗性功能障碍，导致在复杂的现实世界中健壮性较差。基于这一观察，我们引入了解耦对抗性补丁(Depatch)攻击来解决对抗性补丁的自耦合问题。具体地说，我们将敌意补丁分成分块的分段，并在优化过程中通过随机删除一些分段来减少这些分段之间的相互依赖。我们进一步引入了边界移动操作和渐进式解耦策略，以提高整体攻击能力。大量实验表明，该方法的性能优于其他物理对抗性攻击，尤其是在真实世界中。



## **50. Fooling SHAP with Output Shuffling Attacks**

利用输出洗牌攻击愚弄SHAP cs.LG

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06509v1) [paper-pdf](http://arxiv.org/pdf/2408.06509v1)

**Authors**: Jun Yuan, Aritra Dasgupta

**Abstract**: Explainable AI~(XAI) methods such as SHAP can help discover feature attributions in black-box models. If the method reveals a significant attribution from a ``protected feature'' (e.g., gender, race) on the model output, the model is considered unfair. However, adversarial attacks can subvert the detection of XAI methods. Previous approaches to constructing such an adversarial model require access to underlying data distribution, which may not be possible in many practical scenarios. We relax this constraint and propose a novel family of attacks, called shuffling attacks, that are data-agnostic. The proposed attack strategies can adapt any trained machine learning model to fool Shapley value-based explanations. We prove that Shapley values cannot detect shuffling attacks. However, algorithms that estimate Shapley values, such as linear SHAP and SHAP, can detect these attacks with varying degrees of effectiveness. We demonstrate the efficacy of the attack strategies by comparing the performance of linear SHAP and SHAP using real-world datasets.

摘要: Shap等可解释AI~(XAI)方法可以帮助发现黑盒模型中的特征属性。如果该方法揭示了模型输出上的“受保护特征”(例如，性别、种族)的显著属性，则该模型被认为是不公平的。然而，对抗性攻击可以颠覆XAI方法的检测。以前构建这种对抗性模型的方法需要访问底层数据分布，这在许多实际情况下可能是不可能的。我们放松了这一限制，提出了一类新的攻击，称为改组攻击，是数据不可知的。所提出的攻击策略可以采用任何经过训练的机器学习模型来愚弄基于Shapley值的解释。我们证明了Shapley值不能检测到洗牌攻击。然而，估计Shapley值的算法，如线性Shap和Shap，可以以不同程度的有效性检测到这些攻击。我们通过使用真实数据集比较线性Shap和Shap的性能来验证攻击策略的有效性。



