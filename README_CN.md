# Latest Adversarial Attack Papers
**update at 2023-10-06 10:46:09**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. OMG-ATTACK: Self-Supervised On-Manifold Generation of Transferable Evasion Attacks**

OMG-Attack：自我监督的流形上的可转移规避攻击 cs.LG

ICCV 2023, AROW Workshop

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03707v1) [paper-pdf](http://arxiv.org/pdf/2310.03707v1)

**Authors**: Ofir Bar Tal, Adi Haviv, Amit H. Bermano

**Abstract**: Evasion Attacks (EA) are used to test the robustness of trained neural networks by distorting input data to misguide the model into incorrect classifications. Creating these attacks is a challenging task, especially with the ever-increasing complexity of models and datasets. In this work, we introduce a self-supervised, computationally economical method for generating adversarial examples, designed for the unseen black-box setting. Adapting techniques from representation learning, our method generates on-manifold EAs that are encouraged to resemble the data distribution. These attacks are comparable in effectiveness compared to the state-of-the-art when attacking the model trained on, but are significantly more effective when attacking unseen models, as the attacks are more related to the data rather than the model itself. Our experiments consistently demonstrate the method is effective across various models, unseen data categories, and even defended models, suggesting a significant role for on-manifold EAs when targeting unseen models.

摘要: 回避攻击(EA)被用来测试训练神经网络的稳健性，方法是扭曲输入数据以将模型误导到错误的分类。创建这些攻击是一项具有挑战性的任务，特别是在模型和数据集的日益复杂的情况下。在这项工作中，我们介绍了一种自监督的、计算上经济的方法来生成对抗性实例，该方法是针对不可见的黑箱设置而设计的。我们的方法采用了表示学习的技术，生成了流形上的EA，鼓励它们类似于数据分布。在攻击受过训练的模型时，这些攻击在有效性上与最先进的攻击相当，但在攻击看不见的模型时，这些攻击的效率要高得多，因为攻击更多地与数据相关，而不是模型本身。我们的实验一直表明，该方法对各种模型、不可见数据类别，甚至是防御模型都是有效的，这表明在针对不可见模型时，流形上的EA具有重要的作用。



## **2. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03684v1) [paper-pdf](http://arxiv.org/pdf/2310.03684v1)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。



## **3. Certification of Deep Learning Models for Medical Image Segmentation**

医学图像分割中深度学习模型的验证 eess.IV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03664v1) [paper-pdf](http://arxiv.org/pdf/2310.03664v1)

**Authors**: Othmane Laousy, Alexandre Araujo, Guillaume Chassagnon, Nikos Paragios, Marie-Pierre Revel, Maria Vakalopoulou

**Abstract**: In medical imaging, segmentation models have known a significant improvement in the past decade and are now used daily in clinical practice. However, similar to classification models, segmentation models are affected by adversarial attacks. In a safety-critical field like healthcare, certifying model predictions is of the utmost importance. Randomized smoothing has been introduced lately and provides a framework to certify models and obtain theoretical guarantees. In this paper, we present for the first time a certified segmentation baseline for medical imaging based on randomized smoothing and diffusion models. Our results show that leveraging the power of denoising diffusion probabilistic models helps us overcome the limits of randomized smoothing. We conduct extensive experiments on five public datasets of chest X-rays, skin lesions, and colonoscopies, and empirically show that we are able to maintain high certified Dice scores even for highly perturbed images. Our work represents the first attempt to certify medical image segmentation models, and we aspire for it to set a foundation for future benchmarks in this crucial and largely uncharted area.

摘要: 在医学成像中，分割模型在过去十年中有了显著的改进，现在每天都在临床实践中使用。然而，与分类模型类似，分割模型也会受到对抗性攻击的影响。在像医疗保健这样的安全关键领域，验证模型预测是至关重要的。随机化平滑是最近引入的，它为验证模型和获得理论保证提供了一个框架。在这篇文章中，我们首次提出了一种基于随机平滑和扩散模型的医学图像分割基线。我们的结果表明，利用扩散概率模型的去噪能力可以帮助我们克服随机平滑的局限性。我们在胸部X光、皮肤损伤和结肠镜检查的五个公共数据集上进行了广泛的实验，经验表明，即使是高度扰动的图像，我们也能够保持高认证Dice分数。我们的工作代表着认证医学图像分割模型的第一次尝试，我们渴望它为这一关键且基本上未知的领域的未来基准奠定基础。



## **4. Targeted Adversarial Attacks on Generalizable Neural Radiance Fields**

可推广神经辐射场上的定向对抗性攻击 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03578v1) [paper-pdf](http://arxiv.org/pdf/2310.03578v1)

**Authors**: Andras Horvath, Csaba M. Jozsa

**Abstract**: Neural Radiance Fields (NeRFs) have recently emerged as a powerful tool for 3D scene representation and rendering. These data-driven models can learn to synthesize high-quality images from sparse 2D observations, enabling realistic and interactive scene reconstructions. However, the growing usage of NeRFs in critical applications such as augmented reality, robotics, and virtual environments could be threatened by adversarial attacks.   In this paper we present how generalizable NeRFs can be attacked by both low-intensity adversarial attacks and adversarial patches, where the later could be robust enough to be used in real world applications. We also demonstrate targeted attacks, where a specific, predefined output scene is generated by these attack with success.

摘要: 神经辐射场(NERF)是最近出现的一种用于3D场景表示和渲染的强大工具。这些数据驱动的模型可以学习从稀疏的2D观测合成高质量的图像，从而实现逼真和交互的场景重建。然而，在增强现实、机器人和虚拟环境等关键应用中越来越多地使用nerf可能会受到对手攻击的威胁。在这篇文章中，我们介绍了可推广的NERF如何同时被低强度的对抗性攻击和对抗性补丁攻击，其中后者可以足够健壮，以用于现实世界的应用。我们还演示了目标攻击，这些攻击成功地生成了特定的、预定义的输出场景。



## **5. Enhancing Adversarial Robustness via Score-Based Optimization**

通过基于分数的优化增强对手的健壮性 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2307.04333v2) [paper-pdf](http://arxiv.org/pdf/2307.04333v2)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.

摘要: 对抗性攻击有可能通过引入轻微的扰动来误导深度神经网络分类器。开发能够缓解这些攻击影响的算法，对于确保人工智能的安全使用至关重要。最近的研究表明，基于分数的扩散模型在对抗防御中是有效的。然而，现有的基于扩散的防御依赖于对扩散模型的逆随机微分方程的顺序模拟，这在计算上效率低下，并且产生次优结果。在本文中，我们提出了一种新的对抗防御方案ScoreOpt，该方案在测试时优化对手样本，在基于分数的先验的指导下，朝着原始干净数据的方向进行优化。我们在包括CIFAR10、CIFAR100和ImageNet在内的多个数据集上进行了全面的实验。实验结果表明，该方法在稳健性和推理速度上均优于现有的对抗性防御方法。



## **6. AdvRain: Adversarial Raindrops to Attack Camera-based Smart Vision Systems**

AdvRain：敌意雨滴攻击基于摄像头的智能视觉系统 cs.CV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2303.01338v2) [paper-pdf](http://arxiv.org/pdf/2303.01338v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Muhammad Shafique

**Abstract**: Vision-based perception modules are increasingly deployed in many applications, especially autonomous vehicles and intelligent robots. These modules are being used to acquire information about the surroundings and identify obstacles. Hence, accurate detection and classification are essential to reach appropriate decisions and take appropriate and safe actions at all times. Current studies have demonstrated that "printed adversarial attacks", known as physical adversarial attacks, can successfully mislead perception models such as object detectors and image classifiers. However, most of these physical attacks are based on noticeable and eye-catching patterns for generated perturbations making them identifiable/detectable by human eye or in test drives. In this paper, we propose a camera-based inconspicuous adversarial attack (\textbf{AdvRain}) capable of fooling camera-based perception systems over all objects of the same class. Unlike mask based fake-weather attacks that require access to the underlying computing hardware or image memory, our attack is based on emulating the effects of a natural weather condition (i.e., Raindrops) that can be printed on a translucent sticker, which is externally placed over the lens of a camera. To accomplish this, we provide an iterative process based on performing a random search aiming to identify critical positions to make sure that the performed transformation is adversarial for a target classifier. Our transformation is based on blurring predefined parts of the captured image corresponding to the areas covered by the raindrop. We achieve a drop in average model accuracy of more than $45\%$ and $40\%$ on VGG19 for ImageNet and Resnet34 for Caltech-101, respectively, using only $20$ raindrops.

摘要: 基于视觉的感知模块越来越多地部署在许多应用中，特别是自动驾驶汽车和智能机器人。这些模块被用来获取关于周围环境的信息和识别障碍物。因此，准确的检测和分类对于作出适当的决定并始终采取适当和安全的行动至关重要。目前的研究表明，被称为物理对抗性攻击的“印刷对抗性攻击”可以成功地误导对象检测器和图像分类器等感知模型。然而，大多数这些物理攻击都是基于所产生的扰动的明显和醒目的模式，使得它们可以被人眼或试驾识别/检测。在这篇文章中，我们提出了一种基于摄像机的隐蔽敌意攻击(Textbf{AdvRain})，能够在同一类对象上欺骗基于摄像机的感知系统。与需要访问底层计算硬件或图像内存的基于面具的假天气攻击不同，我们的攻击基于模拟自然天气条件(即雨滴)的影响，可以将其打印在半透明贴纸上，该贴纸外部放置在相机的镜头上。为了实现这一点，我们提供了一种迭代过程，该过程基于执行旨在识别关键位置的随机搜索，以确保所执行的变换对目标分类器是对抗性的。我们的变换是基于模糊与雨滴覆盖的区域相对应的捕获图像的预定义部分。在VGG19(ImageNet)和Resnet34(Caltech-101)上，仅使用$20$雨滴，我们的平均模型精度分别下降了$45$和$40$。



## **7. Efficient Biologically Plausible Adversarial Training**

有效的生物学上看似合理的对抗性训练 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2309.17348v3) [paper-pdf](http://arxiv.org/pdf/2309.17348v3)

**Authors**: Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi

**Abstract**: Artificial Neural Networks (ANNs) trained with Backpropagation (BP) show astounding performance and are increasingly often used in performing our daily life tasks. However, ANNs are highly vulnerable to adversarial attacks, which alter inputs with small targeted perturbations that drastically disrupt the models' performance. The most effective method to make ANNs robust against these attacks is adversarial training, in which the training dataset is augmented with exemplary adversarial samples. Unfortunately, this approach has the drawback of increased training complexity since generating adversarial samples is very computationally demanding. In contrast to ANNs, humans are not susceptible to adversarial attacks. Therefore, in this work, we investigate whether biologically-plausible learning algorithms are more robust against adversarial attacks than BP. In particular, we present an extensive comparative analysis of the adversarial robustness of BP and Present the Error to Perturb the Input To modulate Activity (PEPITA), a recently proposed biologically-plausible learning algorithm, on various computer vision tasks. We observe that PEPITA has higher intrinsic adversarial robustness and, with adversarial training, has a more favourable natural-vs-adversarial performance trade-off as, for the same natural accuracies, PEPITA's adversarial accuracies decrease in average by 0.26% and BP's by 8.05%.

摘要: 用反向传播(BP)训练的人工神经网络(ANN)表现出惊人的性能，越来越多地被用于执行我们的日常生活任务。然而，人工神经网络非常容易受到对抗性攻击，这些攻击通过小的有针对性的扰动改变输入，从而极大地破坏模型的性能。使神经网络对这些攻击具有健壮性的最有效的方法是对抗性训练，其中训练数据集被示例性对抗性样本扩充。不幸的是，这种方法的缺点是增加了训练的复杂性，因为生成对抗性样本的计算要求非常高。与人工神经网络不同，人类不容易受到敌意攻击。因此，在这项工作中，我们调查了生物上可信的学习算法是否比BP算法对对手攻击更健壮。特别是，我们对BP的对抗健壮性进行了广泛的比较分析，并提出了在各种计算机视觉任务中扰动输入以调节活动的误差(PEITA)，这是一种最近提出的生物学上看似合理的学习算法。我们观察到，Pepita具有更高的内在对抗健壮性，并且经过对抗训练后，具有更有利的自然与对抗性能权衡，对于相同的自然精度，Pepita的对抗精度平均下降0.26%，BP的平均下降8.05%。



## **8. An Integrated Algorithm for Robust and Imperceptible Audio Adversarial Examples**

一种稳健不可察觉的音频对抗性实例综合算法 cs.SD

Proc. 3rd Symposium on Security and Privacy in Speech Communication

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03349v1) [paper-pdf](http://arxiv.org/pdf/2310.03349v1)

**Authors**: Armin Ettenhofer, Jan-Philipp Schulze, Karla Pizzi

**Abstract**: Audio adversarial examples are audio files that have been manipulated to fool an automatic speech recognition (ASR) system, while still sounding benign to a human listener. Most methods to generate such samples are based on a two-step algorithm: first, a viable adversarial audio file is produced, then, this is fine-tuned with respect to perceptibility and robustness. In this work, we present an integrated algorithm that uses psychoacoustic models and room impulse responses (RIR) in the generation step. The RIRs are dynamically created by a neural network during the generation process to simulate a physical environment to harden our examples against transformations experienced in over-the-air attacks. We compare the different approaches in three experiments: in a simulated environment and in a realistic over-the-air scenario to evaluate the robustness, and in a human study to evaluate the perceptibility. Our algorithms considering psychoacoustics only or in addition to the robustness show an improvement in the signal-to-noise ratio (SNR) as well as in the human perception study, at the cost of an increased word error rate (WER).

摘要: 音频敌意的例子是被操纵以愚弄自动语音识别(ASR)系统的音频文件，同时对人类收听者来说听起来仍然是良性的。大多数生成此类样本的方法都是基于两步算法：首先生成一个可行的敌意音频文件，然后在可感知性和稳健性方面进行微调。在这项工作中，我们提出了一种在生成步骤中使用心理声学模型和房间脉冲响应(RIR)的综合算法。RIR是由神经网络在生成过程中动态创建的，以模拟物理环境，以强化我们的示例，防止在空中攻击中经历的转换。我们在三个实验中比较了不同的方法：在模拟环境中和在现实的空中场景中评估稳健性，以及在人体研究中评估感知能力。我们的算法只考虑了心理声学，或者除了稳健性之外，在信噪比(SNR)和人类感知研究中都得到了改善，但代价是字错误率(WER)增加。



## **9. Untargeted White-box Adversarial Attack with Heuristic Defence Methods in Real-time Deep Learning based Network Intrusion Detection System**

基于实时深度学习的网络入侵检测系统中基于启发式防御方法的非目标白盒攻击 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03334v1) [paper-pdf](http://arxiv.org/pdf/2310.03334v1)

**Authors**: Khushnaseeb Roshan, Aasim Zafar, Sheikh Burhan Ul Haque

**Abstract**: Network Intrusion Detection System (NIDS) is a key component in securing the computer network from various cyber security threats and network attacks. However, consider an unfortunate situation where the NIDS is itself attacked and vulnerable more specifically, we can say, How to defend the defender?. In Adversarial Machine Learning (AML), the malicious actors aim to fool the Machine Learning (ML) and Deep Learning (DL) models to produce incorrect predictions with intentionally crafted adversarial examples. These adversarial perturbed examples have become the biggest vulnerability of ML and DL based systems and are major obstacles to their adoption in real-time and mission-critical applications such as NIDS. AML is an emerging research domain, and it has become a necessity for the in-depth study of adversarial attacks and their defence strategies to safeguard the computer network from various cyber security threads. In this research work, we aim to cover important aspects related to NIDS, adversarial attacks and its defence mechanism to increase the robustness of the ML and DL based NIDS. We implemented four powerful adversarial attack techniques, namely, Fast Gradient Sign Method (FGSM), Jacobian Saliency Map Attack (JSMA), Projected Gradient Descent (PGD) and Carlini & Wagner (C&W) in NIDS. We analyzed its performance in terms of various performance metrics in detail. Furthermore, the three heuristics defence strategies, i.e., Adversarial Training (AT), Gaussian Data Augmentation (GDA) and High Confidence (HC), are implemented to improve the NIDS robustness under adversarial attack situations. The complete workflow is demonstrated in real-time network with data packet flow. This research work provides the overall background for the researchers interested in AML and its implementation from a computer network security point of view.

摘要: 网络入侵检测系统是保障计算机网络免受各种网络安全威胁和网络攻击的重要组成部分。然而，考虑到一个不幸的情况，NIDS本身也受到攻击和攻击，更具体地说，我们可以说，如何保护防御者？在对抗性机器学习(AML)中，恶意行为者旨在欺骗机器学习(ML)和深度学习(DL)模型，通过故意制作的对抗性示例来产生错误的预测。这些对抗性扰动的例子已经成为基于ML和DL的系统的最大漏洞，并成为它们在实时和任务关键型应用(如网络入侵检测系统)中采用的主要障碍。反洗钱是一个新兴的研究领域，深入研究敌方攻击及其防御策略已成为保障计算机网络免受各种网络安全威胁的必要手段。在这项研究工作中，我们的目标是涵盖与网络入侵检测系统相关的重要方面，对手攻击及其防御机制，以增加基于ML和DL的网络入侵检测系统的健壮性。在网络入侵检测系统中实现了四种强大的对抗性攻击技术，即快速梯度符号方法(FGSM)、雅可比显著图攻击(JSMA)、投影梯度下降(PGD)和Carlini&Wagner(C&W)。我们从各种性能度量的角度详细分析了它的性能。在此基础上，提出了三种启发式防御策略，即对抗性训练(AT)、高斯数据增强(GDA)和高置信度(HC)，以提高网络入侵检测系统在对抗性攻击情况下的鲁棒性。整个工作流程在实时网络中以数据包流的形式进行演示。这项研究工作为从计算机网络安全的角度对AML及其实现感兴趣的研究人员提供了总体背景。



## **10. Certifiably Robust Graph Contrastive Learning**

可证明稳健的图对比学习 cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03312v1) [paper-pdf](http://arxiv.org/pdf/2310.03312v1)

**Authors**: Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang

**Abstract**: Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.

摘要: 图对比学习(GCL)是一种流行的无监督图表示学习方法。然而，已有研究表明，GCL在图结构和节点属性上都容易受到敌意攻击。虽然已经提出了一些经验方法来增强GCL的稳健性，但GCL的可证明稳健性仍未得到探索。在本文中，我们在GCL中开发了第一个可证明的健壮性框架。具体地说，我们首先提出了一个统一的标准来评价和证明GCL的健壮性。然后，我们引入了一种新的技术，随机Edgedrop平滑技术，以确保对任何GCL模型都具有可证明的稳健性，并且这种经证明的稳健性可以在下游任务中被证明保持。在此基础上，提出了一种有效的鲁棒GCL训练方法。在真实数据集上的大量实验表明，我们提出的方法在提供有效的可证明的稳健性和增强任何GCL模型的稳健性方面是有效的。RES的源代码可在https://github.com/ventr1c/RES-GCL.上找到



## **11. BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection**

BaDExpert：提取后门功能以进行准确的后门输入检测 cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2308.12439v2) [paper-pdf](http://arxiv.org/pdf/2308.12439v2)

**Authors**: Tinghao Xie, Xiangyu Qi, Ping He, Yiming Li, Jiachen T. Wang, Prateek Mittal

**Abstract**: We present a novel defense, against backdoor attacks on Deep Neural Networks (DNNs), wherein adversaries covertly implant malicious behaviors (backdoors) into DNNs. Our defense falls within the category of post-development defenses that operate independently of how the model was generated. The proposed defense is built upon a novel reverse engineering approach that can directly extract backdoor functionality of a given backdoored model to a backdoor expert model. The approach is straightforward -- finetuning the backdoored model over a small set of intentionally mislabeled clean samples, such that it unlearns the normal functionality while still preserving the backdoor functionality, and thus resulting in a model (dubbed a backdoor expert model) that can only recognize backdoor inputs. Based on the extracted backdoor expert model, we show the feasibility of devising highly accurate backdoor input detectors that filter out the backdoor inputs during model inference. Further augmented by an ensemble strategy with a finetuned auxiliary model, our defense, BaDExpert (Backdoor Input Detection with Backdoor Expert), effectively mitigates 17 SOTA backdoor attacks while minimally impacting clean utility. The effectiveness of BaDExpert has been verified on multiple datasets (CIFAR10, GTSRB and ImageNet) across various model architectures (ResNet, VGG, MobileNetV2 and Vision Transformer).

摘要: 针对深度神经网络(DNNS)的后门攻击，提出了一种新的防御方法，即攻击者秘密地在DNN中植入恶意行为(后门)。我们的防御属于开发后防御的范畴，其运作独立于模型是如何生成的。建议的防御建立在一种新的逆向工程方法之上，该方法可以直接将给定后门模型的后门功能提取到后门专家模型中。这种方法很简单--在一小部分故意错误标记的干净样本上优化后门模型，以便它在保留后门功能的同时取消学习正常功能，从而产生只能识别后门输入的模型(称为后门专家模型)。基于提取的后门专家模型，我们证明了设计高精度的后门输入检测器的可行性，该检测器在模型推理过程中过滤掉后门输入。我们的防御系统BaDExpert(带有后门专家的后门输入检测)通过精细的辅助模型进一步增强了整体策略，有效地减少了17次SOTA后门攻击，同时将对清洁实用程序的影响降至最低。BaDExpert的有效性已经在各种模型架构(ResNet、VGG、MobileNetV2和Vision Transformer)的多个数据集(CIFAR10、GTSRB和ImageNet)上得到了验证。



## **12. Burning the Adversarial Bridges: Robust Windows Malware Detection Against Binary-level Mutations**

烧毁敌意桥梁：针对二进制级突变的强大Windows恶意软件检测 cs.LG

12 pages

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03285v1) [paper-pdf](http://arxiv.org/pdf/2310.03285v1)

**Authors**: Ahmed Abusnaina, Yizhen Wang, Sunpreet Arora, Ke Wang, Mihai Christodorescu, David Mohaisen

**Abstract**: Toward robust malware detection, we explore the attack surface of existing malware detection systems. We conduct root-cause analyses of the practical binary-level black-box adversarial malware examples. Additionally, we uncover the sensitivity of volatile features within the detection engines and exhibit their exploitability. Highlighting volatile information channels within the software, we introduce three software pre-processing steps to eliminate the attack surface, namely, padding removal, software stripping, and inter-section information resetting. Further, to counter the emerging section injection attacks, we propose a graph-based section-dependent information extraction scheme for software representation. The proposed scheme leverages aggregated information within various sections in the software to enable robust malware detection and mitigate adversarial settings. Our experimental results show that traditional malware detection models are ineffective against adversarial threats. However, the attack surface can be largely reduced by eliminating the volatile information. Therefore, we propose simple-yet-effective methods to mitigate the impacts of binary manipulation attacks. Overall, our graph-based malware detection scheme can accurately detect malware with an area under the curve score of 88.32\% and a score of 88.19% under a combination of binary manipulation attacks, exhibiting the efficiency of our proposed scheme.

摘要: 对于健壮的恶意软件检测，我们探索了现有恶意软件检测系统的攻击面。我们对实际的二进制级黑盒恶意软件实例进行了根本原因分析。此外，我们还揭示了检测引擎中易失性特征的敏感性，并展示了它们的可利用性。突出软件内部易变的信息通道，介绍了消除攻击面的三个软件预处理步骤，即填充去除、软件剥离和区间信息重置。此外，针对目前出现的段注入攻击，我们提出了一种基于图的段依赖信息提取方法。建议的方案利用软件中各个部分的聚合信息来实现健壮的恶意软件检测和缓解敌意设置。我们的实验结果表明，传统的恶意软件检测模型不能有效地对抗恶意威胁。然而，通过消除不稳定的信息，可以大大减少攻击面。因此，我们提出了简单而有效的方法来缓解二进制操纵攻击的影响。总体而言，基于图的恶意软件检测方案能够准确地检测出恶意软件，在二进制操纵攻击的组合下，曲线下面积得分为88.32%，得分为88.19%，表明了该方案的有效性。



## **13. Network Cascade Vulnerability using Constrained Bayesian Optimization**

基于约束贝叶斯优化的网络级联漏洞 cs.SI

13 pages, 5 figures

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2304.14420v2) [paper-pdf](http://arxiv.org/pdf/2304.14420v2)

**Authors**: Albert Lam, Mihai Anitescu, Anirudh Subramanyam

**Abstract**: Measures of power grid vulnerability are often assessed by the amount of damage an adversary can exact on the network. However, the cascading impact of such attacks is often overlooked, even though cascades are one of the primary causes of large-scale blackouts. This paper explores modifications of transmission line protection settings as candidates for adversarial attacks, which can remain undetectable as long as the network equilibrium state remains unaltered. This forms the basis of a black-box function in a Bayesian optimization procedure, where the objective is to find protection settings that maximize network degradation due to cascading. Notably, our proposed method is agnostic to the choice of the cascade simulator and its underlying assumptions. Numerical experiments reveal that, against conventional wisdom, maximally misconfiguring the protection settings of all network lines does not cause the most cascading. More surprisingly, even when the degree of misconfiguration is limited due to resource constraints, it is still possible to find settings that produce cascades comparable in severity to instances where there are no resource constraints.

摘要: 电网脆弱性的衡量标准通常是根据对手对网络造成的破坏程度来评估的。然而，这类攻击的连锁影响往往被忽视，尽管连锁是大规模停电的主要原因之一。本文探讨了输电线路保护设置的修改作为对抗性攻击的候选对象，只要网络平衡状态保持不变，这种攻击就可以保持不可检测。这形成了贝叶斯优化过程中的黑盒函数的基础，其中的目标是找到使由于级联而导致的网络降级最大化的保护设置。值得注意的是，我们提出的方法与叶栅模拟器的选择及其基本假设无关。数值实验表明，与传统观点相反，最大限度地错误配置所有网络线路的保护设置并不会导致最大程度的级联。更令人惊讶的是，即使错误配置的程度因资源限制而受到限制，仍有可能找到在严重程度上与没有资源限制的情况相媲美的设置。



## **14. LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference**

LinGCN：同态加密推理的结构线性化图卷积网络 cs.LG

NeurIPS 2023 accepted publication

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.14331v3) [paper-pdf](http://arxiv.org/pdf/2309.14331v3)

**Authors**: Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding

**Abstract**: The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a model with minimized multiplication depth. (2) A compact node-wise polynomial replacement policy with a second-order trainable activation function, steered towards superior convergence by a two-level distillation approach from an all-ReLU based teacher model. (3) an enhanced HE solution that enables finer-grained operator fusion for node-wise activation functions, further reducing multiplication level consumption in HE-based inference. Our experiments on the NTU-XVIEW skeleton joint dataset reveal that LinGCN excels in latency, accuracy, and scalability for homomorphically encrypted inference, outperforming solutions such as CryptoGCN. Remarkably, LinGCN achieves a 14.2x latency speedup relative to CryptoGCN, while preserving an inference accuracy of 75% and notably reducing multiplication depth.

摘要: 图形卷积网络(GCN)模型大小的增长彻底改变了许多应用程序，在个人医疗保健和金融系统等领域超过了人类的表现。由于对客户数据的潜在敌意攻击，GCNS在云中的部署引发了隐私问题。为了解决安全问题，使用同态加密(HE)的隐私保护机器学习(PPML)保护敏感的客户端数据。然而，它在实际应用中引入了大量的计算开销。为了应对这些挑战，我们提出了LinGCN框架，该框架旨在减少乘法深度并优化基于HE的GCN推理的性能。LinGCN围绕三个关键元素构建：(1)可微结构线性化算法，辅以参数化的离散指标函数，与模型权重共同训练以满足优化目标。该策略促进了细粒度的节点级非线性位置选择，使得模型的乘法深度最小化。(2)一种具有二阶可训练激活函数的紧凑节点多项式替换策略，该策略通过基于全REU的教师模型的两级蒸馏方法引导到优越的收敛。(3)增强的HE解决方案，支持针对节点激活函数的更细粒度的算子融合，进一步减少基于HE的推理中的乘法级别消耗。我们在NTU-XVIEW骨骼关节数据集上的实验表明，LinGCN在同态加密推理的延迟、准确性和可扩展性方面都优于CryptoGCN等解决方案。值得注意的是，与CryptoGCN相比，LinGCN实现了14.2倍的延迟加速，同时保持了75%的推理准确率，并显著减少了乘法深度。



## **15. Misusing Tools in Large Language Models With Visual Adversarial Examples**

大型语言模型中的误用工具与视觉对抗性例子 cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03185v1) [paper-pdf](http://arxiv.org/pdf/2310.03185v1)

**Authors**: Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.

摘要: 大型语言模型(LLM)正在得到增强，具有使用工具和处理多种模式的能力。这些新功能带来了新的好处，但也带来了新的安全风险。在这项工作中，我们展示了攻击者可以使用可视化的对抗性示例来导致攻击者所需的工具使用。例如，攻击者可能会导致受害者LLM删除日历事件、泄露私人对话并预订酒店。与以前的工作不同，我们的攻击可以影响连接到LLM的用户资源的机密性和完整性，同时具有隐蔽性和对多个输入提示的通用性。我们使用基于梯度的对抗性训练来构建这些攻击，并在多个维度上表征性能。我们发现，我们的敌意图像可以操纵LLM调用遵循真实语法的工具(~98%)，同时保持与干净图像的高度相似(~0.9SSIM)。此外，使用人工评分和自动度量，我们发现攻击没有显著影响用户和LLM之间的对话(及其语义)。



## **16. Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors**

被夷为平地：机器学习钓鱼网页检测器上的高效查询攻击 cs.CR

Proceedings of the 16th ACM Workshop on Artificial Intelligence and  Security (AISec '23), November 30, 2023, Copenhagen, Denmark

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03166v1) [paper-pdf](http://arxiv.org/pdf/2310.03166v1)

**Authors**: Biagio Montaruli, Luca Demetrio, Maura Pintor, Luca Compagna, Davide Balzarotti, Battista Biggio

**Abstract**: Machine-learning phishing webpage detectors (ML-PWD) have been shown to suffer from adversarial manipulations of the HTML code of the input webpage. Nevertheless, the attacks recently proposed have demonstrated limited effectiveness due to their lack of optimizing the usage of the adopted manipulations, and they focus solely on specific elements of the HTML code. In this work, we overcome these limitations by first designing a novel set of fine-grained manipulations which allow to modify the HTML code of the input phishing webpage without compromising its maliciousness and visual appearance, i.e., the manipulations are functionality- and rendering-preserving by design. We then select which manipulations should be applied to bypass the target detector by a query-efficient black-box optimization algorithm. Our experiments show that our attacks are able to raze to the ground the performance of current state-of-the-art ML-PWD using just 30 queries, thus overcoming the weaker attacks developed in previous work, and enabling a much fairer robustness evaluation of ML-PWD.

摘要: 机器学习钓鱼网页检测器(ML-PWD)已被证明受到输入网页的HTML码的恶意操纵。然而，最近提出的攻击由于缺乏对采用的操作的优化使用而表现出有限的有效性，并且它们仅集中在HTML代码的特定元素上。在这项工作中，我们通过首先设计一组新颖的细粒度操作来克服这些限制，这些操作允许在不影响输入钓鱼网页的恶意和视觉外观的情况下修改其HTML代码，即这些操作是通过设计来保留功能和渲染的。然后，我们通过查询效率高的黑盒优化算法选择应该应用哪些操作来绕过目标检测器。我们的实验表明，我们的攻击能够在仅使用30个查询的情况下将当前最先进的ML-PWD的性能夷为平地，从而克服了以前工作中开发的较弱的攻击，并使得对ML-PWD的健壮性评估更加公平。



## **17. LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**

LLM撒谎：幻觉不是臭虫，而是作为对抗性例子的特征 cs.CL

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.01469v2) [paper-pdf](http://arxiv.org/pdf/2310.01469v2)

**Authors**: Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Li Yuan

**Abstract**: Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still can not completely trust their answer, since LLMs suffer from hallucination--fabricating non-existent facts to cheat users without perception. And the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that non-sense prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. This phenomenon forces us to revisit that hallucination may be another view of adversarial examples, and it shares similar features with conventional adversarial examples as the basic feature of LLMs. Therefore, we formalize an automatic hallucination triggering method as the hallucination attack in an adversarial way. Finally, we explore basic feature of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub.

摘要: 大型语言模型(LLM)，包括GPT-3.5、骆驼和Palm，似乎知识渊博，能够适应许多任务。然而，我们仍然不能完全相信他们的答案，因为LLMS患有幻觉--捏造不存在的事实来欺骗用户而不加察觉。它们存在和普遍存在的原因尚不清楚。在这篇文章中，我们证明了由随机令牌组成的无意义提示也可以诱导LLMS做出幻觉反应。这一现象迫使我们重新审视幻觉可能是对抗性例子的另一种观点，它与传统的对抗性例子有着相似的特征，是LLMS的基本特征。因此，我们将一种自动幻觉触发方法形式化为对抗性的幻觉攻击。最后，探讨了被攻击对抗性提示的基本特征，并提出了一种简单有效的防御策略。我们的代码在GitHub上发布。



## **18. Optimizing Key-Selection for Face-based One-Time Biometrics via Morphing**

基于变形的人脸一次性生物识别优化密钥选择 cs.CV

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.02997v1) [paper-pdf](http://arxiv.org/pdf/2310.02997v1)

**Authors**: Daile Osorio-Roig, Mahdi Ghafourian, Christian Rathgeb, Ruben Vera-Rodriguez, Christoph Busch, Julian Fierrez

**Abstract**: Nowadays, facial recognition systems are still vulnerable to adversarial attacks. These attacks vary from simple perturbations of the input image to modifying the parameters of the recognition model to impersonate an authorised subject. So-called privacy-enhancing facial recognition systems have been mostly developed to provide protection of stored biometric reference data, i.e. templates. In the literature, privacy-enhancing facial recognition approaches have focused solely on conventional security threats at the template level, ignoring the growing concern related to adversarial attacks. Up to now, few works have provided mechanisms to protect face recognition against adversarial attacks while maintaining high security at the template level. In this paper, we propose different key selection strategies to improve the security of a competitive cancelable scheme operating at the signal level. Experimental results show that certain strategies based on signal-level key selection can lead to complete blocking of the adversarial attack based on an iterative optimization for the most secure threshold, while for the most practical threshold, the attack success chance can be decreased to approximately 5.0%.

摘要: 如今，面部识别系统仍然容易受到对手的攻击。这些攻击从对输入图像的简单扰动到修改识别模型的参数以假冒授权对象不等。所谓的隐私增强面部识别系统主要是为了保护存储的生物测定参考数据，即模板。在文献中，增强隐私的面部识别方法只关注模板级别的常规安全威胁，而忽略了与对抗性攻击相关的日益增长的担忧。到目前为止，很少有工作提供机制来保护人脸识别免受对手攻击，同时保持模板级别的高安全性。在本文中，我们提出了不同的密钥选择策略，以提高在信号级运行的竞争可取消方案的安全性。实验结果表明，基于信号级密钥选择的某些策略可以在迭代优化最安全门限的基础上实现对敌方攻击的完全阻断，而对于最实用的门限，攻击成功的几率可以降低到约5.0%。



## **19. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

获得一轮保证：对认证健壮性的浮点攻击 cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2205.10159v4) [paper-pdf](http://arxiv.org/pdf/2205.10159v4)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: Adversarial examples pose a security risk as they can alter decisions of a machine learning classifier through slight input perturbations. Certified robustness has been proposed as a mitigation where given an input $\mathbf{x}$, a classifier returns a prediction and a certified radius $R$ with a provable guarantee that any perturbation to $\mathbf{x}$ with $R$-bounded norm will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples against state-of-the-art certifications in two threat models, that differ in how the norm of the perturbation is computed. We show that the attack can be carried out against linear classifiers that have exact certifiable guarantees and against neural networks that have conservative certifications. In the weak threat model, our experiments demonstrate attack success rates over 50% on random linear classifiers, up to 23% on the MNIST dataset for linear SVM, and up to 15% for a neural network. In the strong threat model, the success rates are lower but positive. The floating-point errors exploited by our attacks can range from small to large (e.g., $10^{-13}$ to $10^{3}$) - showing that even negligible errors can be systematically exploited to invalidate guarantees provided by certified robustness. Finally, we propose a formal mitigation approach based on bounded interval arithmetic, encouraging future implementations of robustness certificates to account for limitations of modern computing architecture to provide sound certifiable guarantees.

摘要: 对抗性的例子构成了安全风险，因为它们可以通过轻微的输入扰动来改变机器学习分类器的决定。证明的稳健性已经被提出作为一种缓解方法，其中给定一个输入$\mathbf{x}$，分类器返回一个预测和一个证明的半径$R$，并且可证明地保证，对$\mathbf{x}$的任何扰动都不会改变分类器的预测。在这项工作中，我们证明了这些保证可能会由于浮点表示的限制而失效，从而导致舍入误差。我们设计了一个四舍五入的搜索方法，可以有效地利用这个漏洞在两个威胁模型中找到针对最新认证的敌意例子，这两个威胁模型的扰动范数的计算方式不同。我们证明了该攻击可以针对具有精确可证明保证的线性分类器和具有保守认证的神经网络来执行。在弱威胁模型中，我们的实验表明，在随机线性分类器上的攻击成功率超过50%，线性支持向量机在MNIST数据集上的攻击成功率高达23%，而神经网络的攻击成功率高达15%。在强威胁模型中，成功率较低，但却是积极的。我们的攻击利用的浮点错误可以从小到大(例如，$10^{-13}$到$10^{3}$)-表明即使是可以忽略的错误也可以被系统地利用来使经证明的健壮性提供的保证无效。最后，我们提出了一种基于有界区间算术的形式化缓解方法，鼓励未来实现健壮性证书，以解决现代计算体系结构的局限性，提供可靠的可证明保证。



## **20. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

GPTFUZZER：自动生成越狱提示的Red Teaming大型语言模型 cs.AI

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.10253v2) [paper-pdf](http://arxiv.org/pdf/2309.10253v2)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.

摘要: 大型语言模型(LLM)最近经历了巨大的流行，并被广泛使用，从随意的对话到人工智能驱动的编程。然而，尽管LLM取得了相当大的成功，但它们并不完全可靠，可以就如何进行有害或非法活动提供详细指导。虽然安全措施可以降低此类输出的风险，但对抗性越狱攻击仍然可以利用LLMS产生有害内容。这些越狱模板通常是手动制作的，这使得大规模测试具有挑战性。在本文中，我们介绍了一种新的黑盒越狱模糊框架GPTFuzz，该框架受到AFL模糊框架的启发。GPTFuzz不是手动设计，而是自动生成用于红队LLM的越狱模板。在其核心，GPTFuzz以人类编写的模板作为初始种子，然后对它们进行突变以产生新的模板。我们详细介绍了GPTFuzz的三个关键组成部分：用于平衡效率和可变性的种子选择策略，用于创建语义等价或相似句子的变异算子，以及用于评估越狱攻击成功的判断模型。我们在不同的攻击场景下，针对各种商业和开源LLM，包括ChatGPT、骆驼2和维库纳，对GPTFuzz进行了评估。我们的结果表明，GPTFuzz一致地生成了成功率较高的越狱模板，超过了人工制作的模板。值得注意的是，GPTFuzz对ChatGPT和Llama-2模型的攻击成功率超过90%，即使在初始种子模板不是最优的情况下也是如此。我们预计，GPTFuzz将有助于研究人员和从业者检查LLM的稳健性，并将鼓励进一步探索增强LLM的安全性。



## **21. SlowFormer: Universal Adversarial Patch for Attack on Compute and Energy Efficiency of Inference Efficient Vision Transformers**

SlowFormer：计算攻击的通用对抗性补丁和推理高效视觉转换器的能量效率 cs.CV

Code is available at https://github.com/UCDvision/SlowFormer

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.02544v1) [paper-pdf](http://arxiv.org/pdf/2310.02544v1)

**Authors**: KL Navaneet, Soroush Abbasi Koohpayegani, Essam Sleiman, Hamed Pirsiavash

**Abstract**: Recently, there has been a lot of progress in reducing the computation of deep models at inference time. These methods can reduce both the computational needs and power usage of deep models. Some of these approaches adaptively scale the compute based on the input instance. We show that such models can be vulnerable to a universal adversarial patch attack, where the attacker optimizes for a patch that when pasted on any image, can increase the compute and power consumption of the model. We run experiments with three different efficient vision transformer methods showing that in some cases, the attacker can increase the computation to the maximum possible level by simply pasting a patch that occupies only 8\% of the image area. We also show that a standard adversarial training defense method can reduce some of the attack's success. We believe adaptive efficient methods will be necessary for the future to lower the power usage of deep models, so we hope our paper encourages the community to study the robustness of these methods and develop better defense methods for the proposed attack.

摘要: 近年来，在减少深层模型推理时的计算量方面取得了很大进展。这些方法可以减少深层模型的计算需求和功耗。其中一些方法根据输入实例自适应地扩展计算。我们证明了这样的模型容易受到普遍的对抗性补丁攻击，在这种攻击中，攻击者优化了一个补丁，当该补丁粘贴到任何图像上时，可以增加模型的计算和功耗。我们用三种不同的高效视觉变换方法进行了实验，结果表明，在某些情况下，攻击者只需粘贴一个只占图像面积8%的补丁，就可以将计算量增加到最大可能的水平。我们还表明，标准的对抗性训练防守方法可以减少一些攻击的成功。我们相信，未来将需要自适应的高效方法来降低深层模型的功耗，因此我们希望本文鼓励社会各界研究这些方法的健壮性，并针对所提出的攻击开发更好的防御方法。



## **22. Jailbreaker in Jail: Moving Target Defense for Large Language Models**

监狱里的越狱者：大型语言模型的移动目标防御 cs.CR

MTD Workshop in CCS'23

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02417v1) [paper-pdf](http://arxiv.org/pdf/2310.02417v1)

**Authors**: Bocheng Chen, Advait Paliwal, Qiben Yan

**Abstract**: Large language models (LLMs), known for their capability in understanding and following instructions, are vulnerable to adversarial attacks. Researchers have found that current commercial LLMs either fail to be "harmless" by presenting unethical answers, or fail to be "helpful" by refusing to offer meaningful answers when faced with adversarial queries. To strike a balance between being helpful and harmless, we design a moving target defense (MTD) enhanced LLM system. The system aims to deliver non-toxic answers that align with outputs from multiple model candidates, making them more robust against adversarial attacks. We design a query and output analysis model to filter out unsafe or non-responsive answers. %to achieve the two objectives of randomly selecting outputs from different LLMs. We evaluate over 8 most recent chatbot models with state-of-the-art adversarial queries. Our MTD-enhanced LLM system reduces the attack success rate from 37.5\% to 0\%. Meanwhile, it decreases the response refusal rate from 50\% to 0\%.

摘要: 大型语言模型(LLM)以其理解和遵循指令的能力而闻名，容易受到对手攻击。研究人员发现，当前的商业LLM要么无法提供不道德的答案，要么无法通过在面对敌对问题时提供有意义的答案而无法提供“帮助”。为了在有益和无害之间取得平衡，我们设计了一种增强的移动目标防御LLM系统。该系统旨在提供无毒的答案，与来自多个模型候选人的输出保持一致，使它们更强大地抵御对手攻击。我们设计了一个查询和输出分析模型来过滤掉不安全或无响应的答案。%，以实现从不同LLM中随机选择输出的两个目标。我们使用最先进的对抗性查询评估了超过8个最新的聊天机器人模型。我们的MTD增强型LLM系统将攻击成功率从37.5%降低到0%。同时，它将响应拒绝率从50%降低到0。



## **23. Exploring Model Learning Heterogeneity for Boosting Ensemble Robustness**

探索提高集成稳健性的模型学习异构性 cs.CV

Accepted by IEEE ICDM 2023

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02237v1) [paper-pdf](http://arxiv.org/pdf/2310.02237v1)

**Authors**: Yanzhao Wu, Ka-Ho Chow, Wenqi Wei, Ling Liu

**Abstract**: Deep neural network ensembles hold the potential of improving generalization performance for complex learning tasks. This paper presents formal analysis and empirical evaluation to show that heterogeneous deep ensembles with high ensemble diversity can effectively leverage model learning heterogeneity to boost ensemble robustness. We first show that heterogeneous DNN models trained for solving the same learning problem, e.g., object detection, can significantly strengthen the mean average precision (mAP) through our weighted bounding box ensemble consensus method. Second, we further compose ensembles of heterogeneous models for solving different learning problems, e.g., object detection and semantic segmentation, by introducing the connected component labeling (CCL) based alignment. We show that this two-tier heterogeneity driven ensemble construction method can compose an ensemble team that promotes high ensemble diversity and low negative correlation among member models of the ensemble, strengthening ensemble robustness against both negative examples and adversarial attacks. Third, we provide a formal analysis of the ensemble robustness in terms of negative correlation. Extensive experiments validate the enhanced robustness of heterogeneous ensembles in both benign and adversarial settings. The source codes are available on GitHub at https://github.com/git-disl/HeteRobust.

摘要: 深度神经网络集成具有提高复杂学习任务泛化性能的潜力。本文给出了形式化分析和实证评估，表明具有高集成多样性的异质深层集成可以有效地利用模型学习的异构性来提高集成的稳健性。我们首先证明了通过我们的加权包围盒集成共识方法，为解决相同的学习问题而训练的异类DNN模型，例如目标检测，可以显著地提高平均平均精度(MAP)。其次，通过引入基于连通成分标记(CCL)的比对，进一步构建了用于解决不同学习问题的异质模型集成，如对象检测和语义分割。结果表明，这种两层异构性驱动的集成构建方法可以组成一个集成团队，促进集成成员模型之间的高集成多样性和低负相关性，增强集成对反例和对手攻击的稳健性。第三，我们从负相关的角度对集成的稳健性进行了形式化分析。广泛的实验验证了在良性和对抗性环境下异质集成的增强的稳健性。源代码可在GitHub上获得，网址为https://github.com/git-disl/HeteRobust.



## **24. Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs**

多通道LLMS中滥用图像和声音进行间接指令注入 cs.CR

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2307.10490v4) [paper-pdf](http://arxiv.org/pdf/2307.10490v4)

**Authors**: Eugene Bagdasaryan, Tsung-Yin Hsieh, Ben Nassi, Vitaly Shmatikov

**Abstract**: We demonstrate how images and sounds can be used for indirect prompt and instruction injection in multi-modal LLMs. An attacker generates an adversarial perturbation corresponding to the prompt and blends it into an image or audio recording. When the user asks the (unmodified, benign) model about the perturbed image or audio, the perturbation steers the model to output the attacker-chosen text and/or make the subsequent dialog follow the attacker's instruction. We illustrate this attack with several proof-of-concept examples targeting LLaVa and PandaGPT.

摘要: 我们演示了图像和声音如何在多模式LLMS中用于间接提示和指令注入。攻击者生成与提示相对应的对抗性扰动，并将其混合到图像或音频记录中。当用户询问(未修改的、良性的)模型有关受干扰的图像或音频时，扰动引导模型输出攻击者选择的文本和/或使后续对话遵循攻击者的指令。我们用几个针对LLaVa和PandaGPT的概念验证示例来说明这种攻击。



## **25. FLEDGE: Ledger-based Federated Learning Resilient to Inference and Backdoor Attacks**

Fledge：基于分类帐的联邦学习对推理和后门攻击的弹性 cs.CR

To appear in Annual Computer Security Applications Conference (ACSAC)  2023

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02113v1) [paper-pdf](http://arxiv.org/pdf/2310.02113v1)

**Authors**: Jorge Castillo, Phillip Rieger, Hossein Fereidooni, Qian Chen, Ahmad Sadeghi

**Abstract**: Federated learning (FL) is a distributed learning process that uses a trusted aggregation server to allow multiple parties (or clients) to collaboratively train a machine learning model without having them share their private data. Recent research, however, has demonstrated the effectiveness of inference and poisoning attacks on FL. Mitigating both attacks simultaneously is very challenging. State-of-the-art solutions have proposed the use of poisoning defenses with Secure Multi-Party Computation (SMPC) and/or Differential Privacy (DP). However, these techniques are not efficient and fail to address the malicious intent behind the attacks, i.e., adversaries (curious servers and/or compromised clients) seek to exploit a system for monetization purposes. To overcome these limitations, we present a ledger-based FL framework known as FLEDGE that allows making parties accountable for their behavior and achieve reasonable efficiency for mitigating inference and poisoning attacks. Our solution leverages crypto-currency to increase party accountability by penalizing malicious behavior and rewarding benign conduct. We conduct an extensive evaluation on four public datasets: Reddit, MNIST, Fashion-MNIST, and CIFAR-10. Our experimental results demonstrate that (1) FLEDGE provides strong privacy guarantees for model updates without sacrificing model utility; (2) FLEDGE can successfully mitigate different poisoning attacks without degrading the performance of the global model; and (3) FLEDGE offers unique reward mechanisms to promote benign behavior during model training and/or model aggregation.

摘要: 联合学习(FL)是一种分布式学习过程，它使用可信的聚合服务器来允许多方(或客户端)协作训练机器学习模型，而不让他们共享他们的私人数据。然而，最近的研究已经证明了推理和中毒攻击对外语的有效性。同时减轻这两种攻击是非常具有挑战性的。最先进的解决方案已经提出使用具有安全多方计算(SMPC)和/或差分隐私(DP)的中毒防御。然而，这些技术效率不高，无法解决攻击背后的恶意意图，即对手(好奇的服务器和/或受攻击的客户端)试图利用系统来赚钱。为了克服这些局限性，我们提出了一个基于分类账的FL框架，称为FLEGGE，它允许各方对他们的行为负责，并在减轻推理和中毒攻击方面获得合理的效率。我们的解决方案利用加密货币通过惩罚恶意行为和奖励良性行为来增加当事人的责任。我们在四个公共数据集上进行了广泛的评估：Reddit、MNIST、Fashion-MNIST和CIFAR-10。我们的实验结果表明：(1)Fledge在不牺牲模型效用的情况下为模型更新提供了强大的隐私保障；(2)Fledge在不降低全局模型性能的情况下成功地缓解了不同的中毒攻击；(3)Fledge在模型训练和/或模型聚合期间提供了独特的奖励机制来促进良性行为。



## **26. Certifiers Make Neural Networks Vulnerable to Availability Attacks**

认证器使神经网络容易受到可用性攻击 cs.LG

Published at 16th ACM Workshop on Artificial Intelligence and  Security (AISec '23)

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2108.11299v5) [paper-pdf](http://arxiv.org/pdf/2108.11299v5)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: To achieve reliable, robust, and safe AI systems, it is vital to implement fallback strategies when AI predictions cannot be trusted. Certifiers for neural networks are a reliable way to check the robustness of these predictions. They guarantee for some predictions that a certain class of manipulations or attacks could not have changed the outcome. For the remaining predictions without guarantees, the method abstains from making a prediction, and a fallback strategy needs to be invoked, which typically incurs additional costs, can require a human operator, or even fail to provide any prediction. While this is a key concept towards safe and secure AI, we show for the first time that this approach comes with its own security risks, as such fallback strategies can be deliberately triggered by an adversary. In addition to naturally occurring abstains for some inputs and perturbations, the adversary can use training-time attacks to deliberately trigger the fallback with high probability. This transfers the main system load onto the fallback, reducing the overall system's integrity and/or availability. We design two novel availability attacks, which show the practical relevance of these threats. For example, adding 1% poisoned data during training is sufficient to trigger the fallback and hence make the model unavailable for up to 100% of all inputs by inserting the trigger. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the broad applicability of these attacks. An initial investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, specific solutions.

摘要: 为了实现可靠、健壮和安全的人工智能系统，在人工智能预测无法信任的情况下实施后备策略至关重要。神经网络的认证器是检验这些预测的稳健性的可靠方法。他们为某些预测提供了保证，即某种操纵或攻击不可能改变结果。对于没有保证的其余预测，该方法放弃进行预测，并且需要调用后备策略，这通常会产生额外的成本，可能需要人工操作员，甚至不能提供任何预测。虽然这是一个安全可靠的人工智能的关键概念，但我们第一次表明，这种方法也有自己的安全风险，因为这样的后备战略可能会被对手故意触发。除了对一些输入和扰动自然发生的弃权之外，对手还可以利用训练时间攻击来高概率地故意触发回退。这会将主系统负载转移到备用系统上，从而降低整个系统的完整性和/或可用性。我们设计了两个新的可用性攻击，表明了这些威胁的实际相关性。例如，在训练期间添加1%的有毒数据就足以触发回退，从而通过插入触发器使模型对高达100%的所有输入不可用。我们在多个数据集、模型体系结构和认证器上的广泛实验证明了这些攻击的广泛适用性。对潜在防御措施的初步调查表明，目前的方法不足以缓解这一问题，这突显了需要新的、具体的解决方案。



## **27. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

DeepZero：深度模型训练的零阶放大优化 cs.LG

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02025v1) [paper-pdf](http://arxiv.org/pdf/2310.02025v1)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinate-wise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsity-induced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box.

摘要: 当一阶信息难以或不可能获得时，零阶(ZO)优化已成为解决机器学习(ML)问题的一种流行技术。然而，ZO优化的可伸缩性仍然是一个悬而未决的问题：它的使用主要限于相对较小的ML问题，例如样本智慧的敌意攻击生成。据我们所知，没有任何先前的工作证明ZO优化在不显著降低性能的情况下训练深度神经网络(DNN)的有效性。为了克服这一障碍，我们开发了DeepZero，这是一个原则性的ZO深度学习(DL)框架，可以通过三项主要创新从头开始将ZO优化扩展到DNN培训。首先，我们证明了坐标梯度估计(CGE)在训练精度和计算效率上优于随机向量梯度估计。其次，我们提出了一种稀疏诱导的ZO训练协议，该协议扩展了仅使用有限差分的模型剪枝方法，以探索和利用CGE中的稀疏DL先验。第三，开发了特征重用和前向并行化的方法，推进了ZO训练的实际实现。我们的广泛实验表明，DeepZero在CIFAR-10上训练的ResNet-20上达到了最先进的精度(SOTA)，首次接近FO训练性能。此外，我们还展示了DeepZero在认证对抗防御和基于DL的偏微分方程纠错中的实际应用，比SOTA提高了10%-20%。我们相信，我们的研究结果将对未来可伸缩ZO优化的研究起到启发作用，并为进一步推进黑盒动态链接库做出贡献。



## **28. Beyond Labeling Oracles: What does it mean to steal ML models?**

除了给甲骨文贴标签：窃取ML模型意味着什么？ cs.LG

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.01959v1) [paper-pdf](http://arxiv.org/pdf/2310.01959v1)

**Authors**: Avital Shafran, Ilia Shumailov, Murat A. Erdogdu, Nicolas Papernot

**Abstract**: Model extraction attacks are designed to steal trained models with only query access, as is often provided through APIs that ML-as-a-Service providers offer. ML models are expensive to train, in part because data is hard to obtain, and a primary incentive for model extraction is to acquire a model while incurring less cost than training from scratch. Literature on model extraction commonly claims or presumes that the attacker is able to save on both data acquisition and labeling costs. We show that the attacker often does not. This is because current attacks implicitly rely on the adversary being able to sample from the victim model's data distribution. We thoroughly evaluate factors influencing the success of model extraction. We discover that prior knowledge of the attacker, i.e. access to in-distribution data, dominates other factors like the attack policy the adversary follows to choose which queries to make to the victim model API. Thus, an adversary looking to develop an equally capable model with a fixed budget has little practical incentive to perform model extraction, since for the attack to work they need to collect in-distribution data, saving only on the cost of labeling. With low labeling costs in the current market, the usefulness of such attacks is questionable. Ultimately, we demonstrate that the effect of prior knowledge needs to be explicitly decoupled from the attack policy. To this end, we propose a benchmark to evaluate attack policy directly.

摘要: 模型提取攻击旨在窃取仅具有查询访问权限的训练模型，这通常是通过ML-as-a-Service提供商提供的API提供的。ML模型的训练成本很高，部分原因是数据很难获得，而模型提取的主要动机是在获得模型的同时产生比从头开始培训更少的成本。有关模型提取的文献通常声称或假设攻击者能够节省数据获取和标记成本。我们发现攻击者通常不会这样做。这是因为当前的攻击隐含地依赖于对手能够从受害者模型的数据分布中进行采样。我们对影响模型提取成功的因素进行了深入的评估。我们发现，攻击者的先验知识，即对分发内数据的访问，主导了其他因素，如攻击者选择对受害者模型API进行哪些查询所遵循的攻击策略。因此，希望开发具有固定预算的同等能力的模型的对手几乎没有执行模型提取的实际动机，因为要使攻击发挥作用，他们需要收集分发内数据，这只节省了标记成本。由于当前市场的标签成本较低，此类攻击的用处值得怀疑。最终，我们证明了先验知识的影响需要与攻击策略明确地分离。为此，我们提出了一个直接评估攻击策略的基准。



## **29. Multi-Static ISAC in Cell-Free Massive MIMO: Precoder Design and Privacy Assessment**

无蜂窝海量MIMO中的多静态ISAC：预编码设计和隐私评估 eess.SP

Submitted to the 2023 IEEE Globecom Workshop on Enabling Security,  Trust, and Privacy in 6G Wireless Systems

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2309.13368v3) [paper-pdf](http://arxiv.org/pdf/2309.13368v3)

**Authors**: Isabella W. G. da Silva, Diana P. M. Osorio, Markku Juntti

**Abstract**: A multi-static sensing-centric integrated sensing and communication (ISAC) network can take advantage of the cell-free massive multiple-input multiple-output infrastructure to achieve remarkable diversity gains and reduced power consumption. While the conciliation of sensing and communication requirements is still a challenge, the privacy of the sensing information is a growing concern that should be seriously taken on the design of these systems to prevent other attacks. This paper tackles this issue by assessing the probability of an internal adversary to infer the target location information from the received signal by considering the design of transmit precoders that jointly optimizes the sensing and communication requirements in a multi-static-based cell-free ISAC network. Our results show that the multi-static setting facilitates a more precise estimation of the location of the target than the mono-static implementation.

摘要: 以多静态传感为中心的集成传感与通信(ISAC)网络可以利用无蜂窝的大规模多输入多输出基础设施来实现显着的分集增益和降低功耗。虽然传感和通信要求的协调仍然是一个挑战，但传感信息的隐私是一个日益令人担忧的问题，应该在这些系统的设计中认真对待，以防止其他攻击。本文通过评估内部敌手从接收信号中推断目标位置信息的概率来解决这个问题，考虑了在基于多静态的无小区ISAC网络中联合优化感知和通信需求的发射预编码器的设计。我们的结果表明，与单基地实现相比，多基地设置有助于更精确地估计目标的位置。



## **30. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

无法学习的例子给人一种错误的安全感：用可学习的例子穿透不可利用的数据 cs.LG

Accepted in MM 2023

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2305.09241v5) [paper-pdf](http://arxiv.org/pdf/2305.09241v5)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning. Our code is available at \url{https://github.com/jiangw-0/LE_JCDP}.

摘要: 保护数据不被未经授权的利用对隐私和安全至关重要，特别是在最近对安全漏洞的猖獗研究中，例如对抗性/成员攻击。为此，最近提出了不可学习的例子(UE)作为一种强制保护，通过向数据添加不可察觉的扰动，使得训练在这些数据上的模型不能根据原始的干净分布对它们进行准确的分类。不幸的是，我们发现UE提供了一种错误的安全感，因为它们无法阻止未经授权的用户利用其他不受保护的数据来取消保护，方法是将无法学习的数据再次变为可学习的数据。受此观察的启发，我们正式定义了一种新的威胁，引入了去除了保护的可学习未经授权示例(LES)。这种方法的核心是一种新颖的净化过程，将UE投射到LES的流形上。这是通过一种新的联合条件扩散模型来实现的，该模型根据UE和LES之间的像素和感知相似性来对UE进行去噪。大量的实验表明，在不同的场景下，LE对监督UE和非监督UE都提供了最先进的对抗性能，这是针对监督学习和非监督学习的UE的第一个可推广的对策。我们的代码可在\url{https://github.com/jiangw-0/LE_JCDP}.



## **31. DRSM: De-Randomized Smoothing on Malware Classifier Providing Certified Robustness**

DRSM：恶意软件分类器上的去随机化平滑提供认证的健壮性 cs.CR

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2303.13372v3) [paper-pdf](http://arxiv.org/pdf/2303.13372v3)

**Authors**: Shoumik Saha, Wenxiao Wang, Yigitcan Kaya, Soheil Feizi, Tudor Dumitras

**Abstract**: Machine Learning (ML) models have been utilized for malware detection for over two decades. Consequently, this ignited an ongoing arms race between malware authors and antivirus systems, compelling researchers to propose defenses for malware-detection models against evasion attacks. However, most if not all existing defenses against evasion attacks suffer from sizable performance degradation and/or can defend against only specific attacks, which makes them less practical in real-world settings. In this work, we develop a certified defense, DRSM (De-Randomized Smoothed MalConv), by redesigning the de-randomized smoothing technique for the domain of malware detection. Specifically, we propose a window ablation scheme to provably limit the impact of adversarial bytes while maximally preserving local structures of the executables. After showing how DRSM is theoretically robust against attacks with contiguous adversarial bytes, we verify its performance and certified robustness experimentally, where we observe only marginal accuracy drops as the cost of robustness. To our knowledge, we are the first to offer certified robustness in the realm of static detection of malware executables. More surprisingly, through evaluating DRSM against 9 empirical attacks of different types, we observe that the proposed defense is empirically robust to some extent against a diverse set of attacks, some of which even fall out of the scope of its original threat model. In addition, we collected 15.5K recent benign raw executables from diverse sources, which will be made public as a dataset called PACE (Publicly Accessible Collection(s) of Executables) to alleviate the scarcity of publicly available benign datasets for studying malware detection and provide future research with more representative data of the time.

摘要: 机器学习(ML)模型用于恶意软件检测已有二十多年的历史。因此，这引发了恶意软件作者和反病毒系统之间持续的军备竞赛，迫使研究人员提出针对逃避攻击的恶意软件检测模型的防御方案。然而，大多数(如果不是全部)现有的规避攻击防御系统会出现相当大的性能降级，并且/或者只能防御特定的攻击，这使得它们在现实世界的设置中不太实用。在这项工作中，我们通过重新设计恶意软件检测领域的去随机化平滑技术，开发了一种经过认证的防御机制--去随机化平滑MalConv(De-Randomized Smooth ed MalConv)。具体地说，我们提出了一种窗口消融方案，在最大限度地保留可执行文件的局部结构的同时，可证明地限制了敌意字节的影响。在展示了DRSM在理论上如何对具有连续敌意字节的攻击具有健壮性之后，我们通过实验验证了它的性能和被证明的健壮性，其中我们只观察到随着健壮性的代价，精确度仅有边际下降。据我们所知，我们是第一个在恶意软件可执行文件静态检测领域提供经过认证的健壮性的公司。更令人惊讶的是，通过对DRSM针对9种不同类型的经验攻击的评估，我们观察到所提出的防御在一定程度上对各种攻击具有经验上的健壮性，其中一些攻击甚至超出了其原始威胁模型的范围。此外，我们从不同的来源收集了15.5k个最近的良性原始可执行文件，这些数据将作为一个名为PACE(公开可访问的可执行文件集合(S))的数据集公开，以缓解用于研究恶意软件检测的可公开可用的良性数据集的稀缺性，并为未来的研究提供更具代表性的当时数据。



## **32. Robust Offline Reinforcement Learning -- Certify the Confidence Interval**

稳健的离线强化学习--验证可信区间 cs.LG

the theoretical and experimental were only partial and incomplete

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2309.16631v2) [paper-pdf](http://arxiv.org/pdf/2309.16631v2)

**Authors**: Jiarui Yao, Simon Shaolei Du

**Abstract**: Currently, reinforcement learning (RL), especially deep RL, has received more and more attention in the research area. However, the security of RL has been an obvious problem due to the attack manners becoming mature. In order to defend against such adversarial attacks, several practical approaches are developed, such as adversarial training, data filtering, etc. However, these methods are mostly based on empirical algorithms and experiments, without rigorous theoretical analysis of the robustness of the algorithms. In this paper, we develop an algorithm to certify the robustness of a given policy offline with random smoothing, which could be proven and conducted as efficiently as ones without random smoothing. Experiments on different environments confirm the correctness of our algorithm.

摘要: 目前，强化学习，特别是深度强化学习在该领域受到越来越多的关注。然而，随着攻击方式的日趋成熟，RL的安全性已经成为一个明显的问题。为了防御这种对抗性攻击，人们提出了几种实用的方法，如对抗性训练、数据过滤等。然而，这些方法大多基于经验算法和实验，没有对算法的稳健性进行严格的理论分析。在这篇文章中，我们开发了一种算法来证明给定策略的稳健性离线在随机平滑，这可以证明和执行的效率与没有随机平滑的策略。在不同环境下的实验验证了算法的正确性。



## **33. Decision-Dominant Strategic Defense Against Lateral Movement for 5G Zero-Trust Multi-Domain Networks**

5G零信任多域网络决策主导型侧向移动防御 cs.CR

55 pages, 1 table, and 1 figures

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01675v1) [paper-pdf](http://arxiv.org/pdf/2310.01675v1)

**Authors**: Tao Li, Yunian Pan, Quanyan Zhu

**Abstract**: Multi-domain warfare is a military doctrine that leverages capabilities from different domains, including air, land, sea, space, and cyberspace, to create a highly interconnected battle network that is difficult for adversaries to disrupt or defeat. However, the adoption of 5G technologies on battlefields presents new vulnerabilities due to the complexity of interconnections and the diversity of software, hardware, and devices from different supply chains. Therefore, establishing a zero-trust architecture for 5G-enabled networks is crucial for continuous monitoring and fast data analytics to protect against targeted attacks. To address these challenges, we propose a proactive end-to-end security scheme that utilizes a 5G satellite-guided air-ground network. Our approach incorporates a decision-dominant learning-based method that can thwart the lateral movement of adversaries targeting critical assets on the battlefield before they can conduct reconnaissance or gain necessary access or credentials. We demonstrate the effectiveness of our game-theoretic design, which uses a meta-learning framework to enable zero-trust monitoring and decision-dominant defense against attackers in emerging multi-domain battlefield networks.

摘要: 多领域战争是一种军事学说，它利用空中、陆地、海洋、太空和网络空间等不同领域的能力，创建一个高度互联的战斗网络，对手难以破坏或击败。然而，由于互联的复杂性以及来自不同供应链的软件、硬件和设备的多样性，5G技术在战场上的采用带来了新的脆弱性。因此，为支持5G的网络建立零信任架构对于持续监控和快速数据分析以防范定向攻击至关重要。为了应对这些挑战，我们提出了一种主动的端到端安全方案，该方案利用5G卫星制导的空地网络。我们的方法结合了一种基于决策的学习方法，可以在敌人进行侦察或获得必要的访问或证书之前，阻止他们在战场上瞄准关键资产的横向移动。我们展示了我们的博弈论设计的有效性，它使用元学习框架来实现对新兴多域战场网络中的攻击者的零信任监控和决策主导防御。



## **34. Adversarial Client Detection via Non-parametric Subspace Monitoring in the Internet of Federated Things**

联邦物联网中基于非参数空间监控的敌意客户端检测 cs.LG

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01537v1) [paper-pdf](http://arxiv.org/pdf/2310.01537v1)

**Authors**: Xianjian Xie, Xiaochen Xian, Dan Li, Andi Wang

**Abstract**: The Internet of Federated Things (IoFT) represents a network of interconnected systems with federated learning as the backbone, facilitating collaborative knowledge acquisition while ensuring data privacy for individual systems. The wide adoption of IoFT, however, is hindered by security concerns, particularly the susceptibility of federated learning networks to adversarial attacks. In this paper, we propose an effective non-parametric approach FedRR, which leverages the low-rank features of the transmitted parameter updates generated by federated learning to address the adversarial attack problem. Besides, our proposed method is capable of accurately detecting adversarial clients and controlling the false alarm rate under the scenario with no attack occurring. Experiments based on digit recognition using the MNIST datasets validated the advantages of our approach.

摘要: 联合物联网(IoFT)代表了一个以联合学习为骨干的互联系统网络，促进了协作知识获取，同时确保了单个系统的数据隐私。然而，IoFT的广泛采用受到安全担忧的阻碍，特别是联合学习网络容易受到对手攻击。在本文中，我们提出了一种有效的非参数方法FedRR，该方法利用联邦学习产生的传输参数更新的低阶特征来解决对抗性攻击问题。此外，在没有攻击的情况下，我们提出的方法能够准确地检测敌意客户端并控制误警率。基于MNIST数据集的数字识别实验验证了该方法的优越性。



## **35. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

掩码和恢复：使用掩码自动编码器在测试时进行盲后门保护 cs.LG

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2303.15564v2) [paper-pdf](http://arxiv.org/pdf/2303.15564v2)

**Authors**: Tao Sun, Lu Pang, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from a suspicious model regardless of image benignity. We focus on test-time image purification methods that incapacitate possible triggers while keeping semantic contents intact. Due to diverse trigger patterns and sizes, the heuristic trigger search in image space can be unscalable. We circumvent such barrier by leveraging the strong reconstruction power of generative models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It detects possible triggers in the token space using image structural similarity and label consistency between the test image and MAE restorations. The detection results are then refined by considering trigger topology. Finally, we fuse MAE restorations adaptively into a purified image for making prediction. Our approach is blind to the model architectures, trigger patterns and image benignity. Extensive experiments under different backdoor settings validate its effectiveness and generalizability. Code is available at https://github.com/tsun/BDMAE.

摘要: 深度神经网络很容易受到后门攻击，在后门攻击中，对手通过使用特殊触发器覆盖图像来恶意操纵模型行为。现有的后门防御方法通常需要访问一些验证数据和模型参数，这在许多真实世界的应用中是不切实际的，例如当模型作为云服务提供时。在本文中，我们讨论了测试时的盲后门防御的实际任务，特别是对于黑盒模型。每个测试图像的真实标签都需要从可疑模型中动态恢复，而不考虑图像的亲和性。我们专注于测试时间图像净化方法，这些方法在保持语义内容完整的同时使可能的触发器失效。由于触发器的模式和大小不同，启发式触发器搜索在图像空间中可能无法扩展。我们利用生成式模型强大的重构能力绕过了这一障碍，提出了一种基于掩蔽自动编码器的盲防框架(BDMAE)。它使用测试图像和MAE恢复之间的图像结构相似性和标签一致性来检测令牌空间中可能的触发因素。然后考虑触发拓扑结构对检测结果进行细化。最后，我们自适应地将MAE恢复融合到一个净化的图像中进行预测。我们的方法是对模型架构、触发模式和图像亲和性视而不见。在不同后门设置下的大量实验验证了该方法的有效性和通用性。代码可在https://github.com/tsun/BDMAE.上找到



## **36. Distributed Energy Resources Cybersecurity Outlook: Vulnerabilities, Attacks, Impacts, and Mitigations**

分布式能源网络安全展望：漏洞、攻击、影响和缓解 cs.CR

IEEE Systems Journal (2023)

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2205.11171v4) [paper-pdf](http://arxiv.org/pdf/2205.11171v4)

**Authors**: Ioannis Zografopoulos, Nikos D. Hatziargyriou, Charalambos Konstantinou

**Abstract**: The digitization and decentralization of the electric power grid are key thrusts for an economically and environmentally sustainable future. Towards this goal, distributed energy resources (DER), including rooftop solar panels, battery storage, electric vehicles, etc., are becoming ubiquitous in power systems. Power utilities benefit from DERs as they minimize operational costs; at the same time, DERs grant users and aggregators control over the power they produce and consume. DERs are interconnected, interoperable, and support remotely controllable features, thus, their cybersecurity is of cardinal importance. DER communication dependencies and the diversity of DER architectures widen the threat surface and aggravate the cybersecurity posture of power systems. In this work, we focus on security oversights that reside in the cyber and physical layers of DERs and can jeopardize grid operations. Existing works have underlined the impact of cyberattacks targeting DER assets, however, they either focus on specific system components (e.g., communication protocols), do not consider the mission-critical objectives of DERs, or neglect the adversarial perspective (e.g., adversary/attack models) altogether. To address these omissions, we comprehensively analyze adversarial capabilities and objectives when manipulating DER assets, and then present how protocol and device-level vulnerabilities can materialize into cyberattacks impacting power system operations. Finally, we provide mitigation strategies to thwart adversaries and directions for future DER cybersecurity research.

摘要: 电网的数字化和分散化是实现经济和环境可持续未来的关键推动力。为了实现这一目标，分布式能源(DER)在电力系统中变得无处不在，包括屋顶太阳能电池板、电池储存、电动汽车等。电力公用事业受益于DER，因为它们最大限度地降低了运营成本；同时，DER使用户和聚合器能够控制他们生产和消耗的电力。DER是互联的、可互操作的，并支持远程控制的功能，因此，其网络安全至关重要。DER通信的依赖性和DER体系结构的多样性扩大了威胁面，加剧了电力系统的网络安全态势。在这项工作中，我们重点关注驻留在DER的网络层和物理层并可能危及电网运营的安全疏忽。现有的工作强调了针对DER资产的网络攻击的影响，然而，它们要么专注于特定的系统组件(例如，通信协议)，没有考虑DER的关键任务目标，要么完全忽视了对抗性的观点(例如，对手/攻击模型)。为了解决这些疏漏，我们全面分析了操纵DER资产时的对抗能力和目标，然后介绍了协议和设备级漏洞如何转化为影响电力系统运行的网络攻击。最后，我们提供了挫败对手的缓解策略和未来网络安全研究的方向。



## **37. Fooling the Textual Fooler via Randomizing Latent Representations**

通过随机化潜在表征来愚弄文本愚民 cs.CL

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01452v1) [paper-pdf](http://arxiv.org/pdf/2310.01452v1)

**Authors**: Duy C. Hoang, Quang H. Nguyen, Saurav Manchanda, MinLong Peng, Kok-Seng Wong, Khoa D. Doan

**Abstract**: Despite outstanding performance in a variety of NLP tasks, recent studies have revealed that NLP models are vulnerable to adversarial attacks that slightly perturb the input to cause the models to misbehave. Among these attacks, adversarial word-level perturbations are well-studied and effective attack strategies. Since these attacks work in black-box settings, they do not require access to the model architecture or model parameters and thus can be detrimental to existing NLP applications. To perform an attack, the adversary queries the victim model many times to determine the most important words in an input text and to replace these words with their corresponding synonyms. In this work, we propose a lightweight and attack-agnostic defense whose main goal is to perplex the process of generating an adversarial example in these query-based black-box attacks; that is to fool the textual fooler. This defense, named AdvFooler, works by randomizing the latent representation of the input at inference time. Different from existing defenses, AdvFooler does not necessitate additional computational overhead during training nor relies on assumptions about the potential adversarial perturbation set while having a negligible impact on the model's accuracy. Our theoretical and empirical analyses highlight the significance of robustness resulting from confusing the adversary via randomizing the latent space, as well as the impact of randomization on clean accuracy. Finally, we empirically demonstrate near state-of-the-art robustness of AdvFooler against representative adversarial word-level attacks on two benchmark datasets.

摘要: 尽管在各种自然语言处理任务中表现出色，但最近的研究表明，自然语言处理模型容易受到对抗性攻击，这些攻击会轻微扰乱输入，导致模型行为不当。在这些攻击中，对抗性词级扰动是研究较多、效果较好的攻击策略。由于这些攻击在黑盒设置中工作，因此它们不需要访问模型体系结构或模型参数，因此可能对现有的NLP应用程序有害。为了执行攻击，对手多次查询受害者模型，以确定输入文本中最重要的单词，并用它们对应的同义词替换这些单词。在这项工作中，我们提出了一个轻量级和攻击不可知的防御，其主要目标是在这些基于查询的黑盒攻击中迷惑生成敌对示例的过程，即愚弄文本傻瓜。这种防御被称为AdvFooler，其工作原理是在推理时随机化输入的潜在表示。与现有的防御方法不同，AdvFooler在训练过程中不需要额外的计算开销，也不依赖于对潜在对手扰动集的假设，同时对模型的精度影响可以忽略不计。我们的理论和经验分析强调了通过随机化潜在空间来迷惑对手而产生的稳健性的重要性，以及随机化对干净精度的影响。最后，我们在两个基准数据集上通过实验验证了AdvFooler对典型的对抗性词级攻击的近乎最先进的健壮性。



## **38. Counterfactual Image Generation for adversarially robust and interpretable Classifiers**

用于对抗稳健和可解释分类器的反事实图像生成 cs.CV

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00761v1) [paper-pdf](http://arxiv.org/pdf/2310.00761v1)

**Authors**: Rafael Bischof, Florian Scheidegger, Michael A. Kraus, A. Cristiano I. Malossi

**Abstract**: Neural Image Classifiers are effective but inherently hard to interpret and susceptible to adversarial attacks. Solutions to both problems exist, among others, in the form of counterfactual examples generation to enhance explainability or adversarially augment training datasets for improved robustness. However, existing methods exclusively address only one of the issues. We propose a unified framework leveraging image-to-image translation Generative Adversarial Networks (GANs) to produce counterfactual samples that highlight salient regions for interpretability and act as adversarial samples to augment the dataset for more robustness. This is achieved by combining the classifier and discriminator into a single model that attributes real images to their respective classes and flags generated images as "fake". We assess the method's effectiveness by evaluating (i) the produced explainability masks on a semantic segmentation task for concrete cracks and (ii) the model's resilience against the Projected Gradient Descent (PGD) attack on a fruit defects detection problem. Our produced saliency maps are highly descriptive, achieving competitive IoU values compared to classical segmentation models despite being trained exclusively on classification labels. Furthermore, the model exhibits improved robustness to adversarial attacks, and we show how the discriminator's "fakeness" value serves as an uncertainty measure of the predictions.

摘要: 神经图像分类器是有效的，但本质上很难解释，并且容易受到对手的攻击。除其他外，这两个问题的解决方案都以反事实实例生成的形式存在，以增强可解释性或相反地增强训练数据集以提高稳健性。然而，现有的方法只解决其中一个问题。我们提出了一个统一的框架，利用图像到图像翻译生成性对抗网络(GANS)来产生反事实样本，突出突出可解释性的显著区域，并作为对抗样本来扩大数据集以获得更好的稳健性。这是通过将分类器和鉴别器组合到单个模型中来实现的，该模型将真实图像归因于它们各自的类别，并将生成的图像标记为“假”。我们通过评估(I)产生的混凝土裂缝语义分割任务的可解释性掩模和(Ii)模型对水果缺陷检测问题的投影梯度下降(PGD)攻击的弹性来评估该方法的有效性。我们制作的显著图具有很强的描述性，尽管专门针对分类标签进行了培训，但与经典的细分模型相比，获得了具有竞争力的IOU值。此外，该模型对敌意攻击表现出了更好的稳健性，并且我们展示了鉴别器的“虚假”值如何作为预测的不确定性度量。



## **39. Silent Killer: A Stealthy, Clean-Label, Black-Box Backdoor Attack**

沉默的杀手：一次隐形的、干净的、黑匣子的后门攻击 cs.CR

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2301.02615v2) [paper-pdf](http://arxiv.org/pdf/2301.02615v2)

**Authors**: Tzvi Lederer, Gallil Maimon, Lior Rokach

**Abstract**: Backdoor poisoning attacks pose a well-known risk to neural networks. However, most studies have focused on lenient threat models. We introduce Silent Killer, a novel attack that operates in clean-label, black-box settings, uses a stealthy poison and trigger and outperforms existing methods. We investigate the use of universal adversarial perturbations as triggers in clean-label attacks, following the success of such approaches under poison-label settings. We analyze the success of a naive adaptation and find that gradient alignment for crafting the poison is required to ensure high success rates. We conduct thorough experiments on MNIST, CIFAR10, and a reduced version of ImageNet and achieve state-of-the-art results.

摘要: 后门中毒攻击对神经网络构成了众所周知的风险。然而，大多数研究都集中在宽松的威胁模型上。我们介绍了Silent Killer，这是一种在干净标签、黑匣子设置下运行的新型攻击，使用了一种隐形的毒药和触发器，性能优于现有方法。我们调查了在有毒标签设置下这种方法的成功之后，普遍的对抗性扰动作为干净标签攻击的触发器的使用。我们分析了天真适应的成功，发现需要用梯度对齐来制作毒药，以确保高成功率。我们在MNIST、CIFAR10和ImageNet的精简版本上进行了深入的实验，并获得了最先进的结果。



## **40. DISCO Might Not Be Funky: Random Intelligent Reflective Surface Configurations That Attack**

迪斯科可能并不时髦：攻击的随机智能反射面配置 eess.SP

This work has been submitted for possible publication. We will share  the main codes of this work via GitHub soon. arXiv admin note: text overlap  with arXiv:2308.15716

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00687v1) [paper-pdf](http://arxiv.org/pdf/2310.00687v1)

**Authors**: Huan Huang, Lipeng Dai, Hongliang Zhang, Chongfu Zhang, Zhongxing Tian, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Emerging intelligent reflective surfaces (IRSs) significantly improve system performance, but also pose a signifcant risk for physical layer security (PLS). Unlike the extensive research on legitimate IRS-enhanced communications, in this article we present an adversarial IRS-based fully-passive jammer (FPJ). We describe typical application scenarios for Disco IRS (DIRS)-based FPJ, where an illegitimate IRS with random, time-varying reflection properties acts like a "disco ball" to randomly change the propagation environment. We introduce the principles of DIRS-based FPJ and overview existing investigations of the technology, including a design example employing one-bit phase shifters. The DIRS-based FPJ can be implemented without either jamming power or channel state information (CSI) for the legitimate users (LUs). It does not suffer from the energy constraints of traditional active jammers, nor does it require any knowledge of the LU channels. In addition to the proposed jamming attack, we also propose an anti-jamming strategy that requires only statistical rather than instantaneous CSI. Furthermore, we present a data frame structure that enables the legitimate access point (AP) to estimate the statistical CSI in the presence of the DIRS jamming. Typical cases are discussed to show the impact of the DIRS-based FPJ and the feasibility of the anti-jamming precoder. Moreover, we outline future research directions and challenges for the DIRS-based FPJ and its anti-jamming precoding to stimulate this line of research and pave the way for practical applications.

摘要: 新兴的智能反射表面(IRS)显著提高了系统性能，但也对物理层安全(PLS)构成了重大风险。不同于对合法IRS增强通信的广泛研究，本文提出了一种基于IRS的对抗性全被动干扰机(FPJ)。我们描述了基于Disco IRS(DIRS)的FPJ的典型应用场景，其中具有随机、时变反射属性的非法IRS充当随机改变传播环境的迪斯科球。我们介绍了基于DIRS的FPJ的原理，并综述了该技术的现有研究成果，包括一个使用一位移相器的设计实例。对于合法用户(LU)，可以在没有干扰功率或信道状态信息(CSI)的情况下实现基于DIRS的FPJ。它不受传统有源干扰器的能量限制，也不需要任何LU信道的知识。除了提出的干扰攻击外，我们还提出了一种只需要统计而不需要瞬时CSI的干扰策略。此外，我们提出了一种数据帧结构，使合法接入点(AP)能够在存在DIRS干扰的情况下估计统计CSI。通过典型算例说明了基于DIRS的预编码法的影响和抗扰预编码法的可行性。此外，我们还概述了基于DIRS的FPJ及其抗干扰性预编码的未来研究方向和挑战，以激励这一研究方向，为实际应用铺平道路。



## **41. A Survey of Robustness and Safety of 2D and 3D Deep Learning Models Against Adversarial Attacks**

2D和3D深度学习模型对敌方攻击的稳健性和安全性综述 cs.LG

Submitted to CSUR

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00633v1) [paper-pdf](http://arxiv.org/pdf/2310.00633v1)

**Authors**: Yanjie Li, Bin Xie, Songtao Guo, Yuanyuan Yang, Bin Xiao

**Abstract**: Benefiting from the rapid development of deep learning, 2D and 3D computer vision applications are deployed in many safe-critical systems, such as autopilot and identity authentication. However, deep learning models are not trustworthy enough because of their limited robustness against adversarial attacks. The physically realizable adversarial attacks further pose fatal threats to the application and human safety. Lots of papers have emerged to investigate the robustness and safety of deep learning models against adversarial attacks. To lead to trustworthy AI, we first construct a general threat model from different perspectives and then comprehensively review the latest progress of both 2D and 3D adversarial attacks. We extend the concept of adversarial examples beyond imperceptive perturbations and collate over 170 papers to give an overview of deep learning model robustness against various adversarial attacks. To the best of our knowledge, we are the first to systematically investigate adversarial attacks for 3D models, a flourishing field applied to many real-world applications. In addition, we examine physical adversarial attacks that lead to safety violations. Last but not least, we summarize present popular topics, give insights on challenges, and shed light on future research on trustworthy AI.

摘要: 得益于深度学习的快速发展，2D和3D计算机视觉应用被部署在许多安全关键系统中，如自动驾驶和身份认证。然而，深度学习模型不够可信，因为它们对对手攻击的健壮性有限。物理上可实现的对抗性攻击进一步对应用程序和人类安全构成致命威胁。已有大量文献研究深度学习模型对敌意攻击的稳健性和安全性。为了得到可信的人工智能，我们首先从不同的角度构建了一个通用的威胁模型，然后全面回顾了2D和3D对抗性攻击的最新进展。我们将对抗性例子的概念扩展到不知不觉的扰动之外，并整理了170多篇论文，以给出深度学习模型对各种对抗性攻击的健壮性的概述。据我们所知，我们是第一个系统地研究针对3D模型的对抗性攻击的公司，这是一个应用于许多现实世界应用的蓬勃发展的领域。此外，我们还研究了导致违反安全规定的物理对抗性攻击。最后但同样重要的是，我们总结了当前的热门话题，给出了对挑战的见解，并对未来值得信赖的人工智能的研究进行了展望。



## **42. Generating Transferable and Stealthy Adversarial Patch via Attention-guided Adversarial Inpainting**

通过注意力引导的对抗性修复生成可转移和隐蔽的对抗性补丁 cs.CV

Submitted to ICLR2024

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2308.05320v2) [paper-pdf](http://arxiv.org/pdf/2308.05320v2)

**Authors**: Yanjie Li, Mingxing Duan, Xuelong Dai, Bin Xiao

**Abstract**: Adversarial patch attacks can fool the face recognition (FR) models via small patches. However, previous adversarial patch attacks often result in unnatural patterns that are easily noticeable. Generating transferable and stealthy adversarial patches that can efficiently deceive the black-box FR models while having good camouflage is challenging because of the huge stylistic difference between the source and target images. To generate transferable, natural-looking, and stealthy adversarial patches, we propose an innovative two-stage attack called Adv-Inpainting, which extracts style features and identity features from the attacker and target faces, respectively and then fills the patches with misleading and inconspicuous content guided by attention maps. In the first stage, we extract multi-scale style embeddings by a pyramid-like network and identity embeddings by a pretrained FR model and propose a novel Attention-guided Adaptive Instance Normalization layer (AAIN) to merge them via background-patch cross-attention maps. The proposed layer can adaptively fuse identity and style embeddings by fully exploiting priority contextual information. In the second stage, we design an Adversarial Patch Refinement Network (APR-Net) with a novel boundary variance loss, a spatial discounted reconstruction loss, and a perceptual loss to boost the stealthiness further. Experiments demonstrate that our attack can generate adversarial patches with improved visual quality, better stealthiness, and stronger transferability than state-of-the-art adversarial patch attacks and semantic attacks.

摘要: 对抗性补丁攻击可以通过小补丁欺骗人脸识别(FR)模型。然而，以前的对抗性补丁攻击通常会导致很容易注意到的不自然模式。由于源图像和目标图像之间的巨大风格差异，生成可转移的隐身对抗性补丁在具有良好伪装性的同时可以有效地欺骗黑盒FR模型是具有挑战性的。为了生成可转移的、看起来自然的、隐蔽的敌意补丁，我们提出了一种新的两阶段攻击，称为ADV-Inaint，该攻击分别从攻击者和目标人脸提取风格特征和身份特征，然后在注意图的指导下向补丁填充误导性和不明显的内容。在第一阶段，我们通过金字塔状网络提取多尺度样式嵌入，通过预训练FR模型提取身份嵌入，并提出了一种新的注意力引导的自适应实例归一化层(AAIN)，通过背景-补丁交叉注意映射将它们合并。通过充分利用优先级上下文信息，该层可以自适应地融合身份和样式嵌入。在第二阶段，我们设计了一种新的边界方差损失、空间折扣重建损失和感知损失的对抗性补丁精化网络(APR-Net)，以进一步提高隐蔽性。实验表明，与现有的对抗性补丁攻击和语义攻击相比，该攻击生成的对抗性补丁具有更好的视觉质量、更好的隐蔽性和更强的可转移性。



## **43. Understanding Adversarial Transferability in Federated Learning**

理解联合学习中的对抗性迁移 cs.LG

10 pages of the main paper. 21 pages in total

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00616v1) [paper-pdf](http://arxiv.org/pdf/2310.00616v1)

**Authors**: Yijiang Li, Ying Gao, Haohan Wang

**Abstract**: We investigate the robustness and security issues from a novel and practical setting: a group of malicious clients has impacted the model during training by disguising their identities and acting as benign clients, and only revealing their adversary position after the training to conduct transferable adversarial attacks with their data, which is usually a subset of the data that FL system is trained with. Our aim is to offer a full understanding of the challenges the FL system faces in this practical setting across a spectrum of configurations. We notice that such an attack is possible, but the federated model is more robust compared with its centralized counterpart when the accuracy on clean images is comparable. Through our study, we hypothesized the robustness is from two factors: the decentralized training on distributed data and the averaging operation. We provide evidence from both the perspective of empirical experiments and theoretical analysis. Our work has implications for understanding the robustness of federated learning systems and poses a practical question for federated learning applications.

摘要: 我们从一个新颖和实用的环境中研究了该模型的健壮性和安全性问题：一群恶意客户端在训练过程中通过伪装自己的身份和扮演良性客户端来影响模型，并且在训练结束后只暴露他们的对手位置，用他们的数据进行可转移的对抗性攻击，这些数据通常是FL系统训练时使用的数据的子集。我们的目标是全面了解FL系统在各种配置的实际环境中所面临的挑战。我们注意到，这样的攻击是可能的，但当对干净图像的准确性与之相当时，联合模型比集中式模型更健壮。通过我们的研究，我们假设稳健性来自两个因素：分布式数据的分散训练和平均操作。我们从实证实验和理论分析两个角度提供了证据。我们的工作对于理解联邦学习系统的健壮性具有重要意义，并为联邦学习应用提出了一个实际问题。



## **44. On the Onset of Robust Overfitting in Adversarial Training**

论对抗性训练中健壮性过适应的发生 cs.LG

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00607v1) [paper-pdf](http://arxiv.org/pdf/2310.00607v1)

**Authors**: Chaojian Yu, Xiaolong Shi, Jun Yu, Bo Han, Tongliang Liu

**Abstract**: Adversarial Training (AT) is a widely-used algorithm for building robust neural networks, but it suffers from the issue of robust overfitting, the fundamental mechanism of which remains unclear. In this work, we consider normal data and adversarial perturbation as separate factors, and identify that the underlying causes of robust overfitting stem from the normal data through factor ablation in AT. Furthermore, we explain the onset of robust overfitting as a result of the model learning features that lack robust generalization, which we refer to as non-effective features. Specifically, we provide a detailed analysis of the generation of non-effective features and how they lead to robust overfitting. Additionally, we explain various empirical behaviors observed in robust overfitting and revisit different techniques to mitigate robust overfitting from the perspective of non-effective features, providing a comprehensive understanding of the robust overfitting phenomenon. This understanding inspires us to propose two measures, attack strength and data augmentation, to hinder the learning of non-effective features by the neural network, thereby alleviating robust overfitting. Extensive experiments conducted on benchmark datasets demonstrate the effectiveness of the proposed methods in mitigating robust overfitting and enhancing adversarial robustness.

摘要: 对抗性训练(AT)是一种广泛用于构建稳健神经网络的算法，但它存在稳健过拟合问题，其基本机制尚不清楚。在这项工作中，我们将正常数据和对抗性扰动作为独立的因素，通过AT中的因子消融，确定了稳健过拟合的根本原因源于正常数据。此外，我们解释了由于缺乏稳健泛化的模型学习特征而导致的稳健过拟合的开始，我们称之为非有效特征。具体地说，我们提供了无效特征的产生以及它们如何导致稳健过拟合的详细分析。此外，我们解释了在稳健过拟合中观察到的各种经验行为，并从非有效特征的角度回顾了缓解稳健过拟合的不同技术，提供了对稳健过拟合现象的全面理解。这种理解启发我们提出了两种措施，攻击强度和数据增强，以阻止神经网络学习无效特征，从而缓解鲁棒过拟合。在基准数据集上进行的大量实验表明，所提出的方法在缓解鲁棒过拟合和增强对手稳健性方面是有效的。



## **45. Physical Adversarial Attack meets Computer Vision: A Decade Survey**

身体对抗攻击与计算机视觉相遇：十年综述 cs.CV

19 pages. Under Review

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2209.15179v3) [paper-pdf](http://arxiv.org/pdf/2209.15179v3)

**Authors**: Hui Wei, Hao Tang, Xuemei Jia, Zhixiang Wang, Hanxun Yu, Zhubo Li, Shin'ichi Satoh, Luc Van Gool, Zheng Wang

**Abstract**: Despite the impressive achievements of Deep Neural Networks (DNNs) in computer vision, their vulnerability to adversarial attacks remains a critical concern. Extensive research has demonstrated that incorporating sophisticated perturbations into input images can lead to a catastrophic degradation in DNNs' performance. This perplexing phenomenon not only exists in the digital space but also in the physical world. Consequently, it becomes imperative to evaluate the security of DNNs-based systems to ensure their safe deployment in real-world scenarios, particularly in security-sensitive applications. To facilitate a profound understanding of this topic, this paper presents a comprehensive overview of physical adversarial attacks. Firstly, we distill four general steps for launching physical adversarial attacks. Building upon this foundation, we uncover the pervasive role of artifacts carrying adversarial perturbations in the physical world. These artifacts influence each step. To denote them, we introduce a new term: adversarial medium. Then, we take the first step to systematically evaluate the performance of physical adversarial attacks, taking the adversarial medium as a first attempt. Our proposed evaluation metric, hiPAA, comprises six perspectives: Effectiveness, Stealthiness, Robustness, Practicability, Aesthetics, and Economics. We also provide comparative results across task categories, together with insightful observations and suggestions for future research directions.

摘要: 尽管深度神经网络(DNN)在计算机视觉方面取得了令人印象深刻的成就，但它们对对手攻击的脆弱性仍然是一个令人担忧的问题。大量研究表明，在输入图像中加入复杂的扰动会导致DNN性能的灾难性下降。这种令人困惑的现象不仅存在于数字空间，也存在于物理世界。因此，迫切需要评估基于DNNS的系统的安全性，以确保它们在现实世界场景中的安全部署，特别是在安全敏感的应用中。为了促进对这一主题的深入理解，本文对物理对抗性攻击进行了全面的概述。首先，我们提炼出发动身体对抗攻击的四个一般步骤。在此基础上，我们揭示了在物理世界中携带对抗性扰动的人工制品的普遍作用。这些人工制品会影响每一步。为了表示它们，我们引入了一个新的术语：对抗性媒介。然后，以对抗性媒介为第一次尝试，对物理对抗性攻击的性能进行了系统的评估。我们提出的评估指标HIPAA包括六个角度：有效性、隐蔽性、健壮性、实用性、美观性和经济性。我们还提供了跨任务类别的比较结果，以及有洞察力的观察结果和对未来研究方向的建议。



## **46. Understanding the Robustness of Randomized Feature Defense Against Query-Based Adversarial Attacks**

了解随机化特征防御对基于查询的对手攻击的稳健性 cs.LG

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00567v1) [paper-pdf](http://arxiv.org/pdf/2310.00567v1)

**Authors**: Quang H. Nguyen, Yingjie Lao, Tung Pham, Kok-Seng Wong, Khoa D. Doan

**Abstract**: Recent works have shown that deep neural networks are vulnerable to adversarial examples that find samples close to the original image but can make the model misclassify. Even with access only to the model's output, an attacker can employ black-box attacks to generate such adversarial examples. In this work, we propose a simple and lightweight defense against black-box attacks by adding random noise to hidden features at intermediate layers of the model at inference time. Our theoretical analysis confirms that this method effectively enhances the model's resilience against both score-based and decision-based black-box attacks. Importantly, our defense does not necessitate adversarial training and has minimal impact on accuracy, rendering it applicable to any pre-trained model. Our analysis also reveals the significance of selectively adding noise to different parts of the model based on the gradient of the adversarial objective function, which can be varied during the attack. We demonstrate the robustness of our defense against multiple black-box attacks through extensive empirical experiments involving diverse models with various architectures.

摘要: 最近的研究表明，深度神经网络很容易受到敌意样本的攻击，这些样本找到的样本接近原始图像，但会导致模型错误分类。即使只访问模型的输出，攻击者也可以使用黑盒攻击来生成这样的对抗性示例。在这项工作中，我们提出了一种简单而轻量级的防御黑盒攻击的方法，即在推理时向模型中间层的隐藏特征添加随机噪声。理论分析表明，该方法有效地提高了模型抵抗基于分数和基于决策的黑盒攻击的能力。重要的是，我们的防御不需要对抗性训练，对准确性的影响很小，使其适用于任何预先训练的模型。我们的分析还揭示了根据攻击过程中可能变化的对抗性目标函数的梯度来选择性地向模型的不同部分添加噪声的意义。我们通过广泛的经验实验证明了我们对多个黑盒攻击的防御能力，这些实验涉及不同架构的不同模型。



## **47. Black-box Attacks on Image Activity Prediction and its Natural Language Explanations**

图像活跃度预测的黑盒攻击及其自然语言解释 cs.CV

Accepted at ICCV2023 AROW Workshop

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2310.00503v1) [paper-pdf](http://arxiv.org/pdf/2310.00503v1)

**Authors**: Alina Elena Baia, Valentina Poggioni, Andrea Cavallaro

**Abstract**: Explainable AI (XAI) methods aim to describe the decision process of deep neural networks. Early XAI methods produced visual explanations, whereas more recent techniques generate multimodal explanations that include textual information and visual representations. Visual XAI methods have been shown to be vulnerable to white-box and gray-box adversarial attacks, with an attacker having full or partial knowledge of and access to the target system. As the vulnerabilities of multimodal XAI models have not been examined, in this paper we assess for the first time the robustness to black-box attacks of the natural language explanations generated by a self-rationalizing image-based activity recognition model. We generate unrestricted, spatially variant perturbations that disrupt the association between the predictions and the corresponding explanations to mislead the model into generating unfaithful explanations. We show that we can create adversarial images that manipulate the explanations of an activity recognition model by having access only to its final output.

摘要: 可解释人工智能(XAI)方法旨在描述深度神经网络的决策过程。早期的XAI方法产生了视觉解释，而更新的技术产生了包括文本信息和视觉表示的多模式解释。可视化XAI方法已被证明容易受到白盒和灰盒对抗性攻击，攻击者具有目标系统的全部或部分知识和访问权限。由于多模式XAI模型的脆弱性尚未得到检验，本文首次评估了一种自合理化的基于图像的活动识别模型生成的自然语言解释对黑盒攻击的稳健性。我们产生了不受限制的、空间上不同的扰动，破坏了预测和相应解释之间的关联，从而误导模型产生不可信的解释。我们展示了我们可以创建敌意图像，通过只访问活动识别模型的最终输出来操纵活动识别模型的解释。



## **48. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

评估大语言模型的指令跟随健壮性以实现快速注入 cs.CL

The data and code can be found at  https://github.com/Leezekun/Adv-Instruct-Eval

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2308.10819v2) [paper-pdf](http://arxiv.org/pdf/2308.10819v2)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of instruction-following LLMs against adversarial instructions injected in the prompt. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these injected adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction injection attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being ``overfitted'' to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text. The data and code can be found at \url{https://github.com/Leezekun/Adv-Instruct-Eval}.

摘要: 大型语言模型(LLM)在遵循说明方面表现出非凡的熟练程度，这使它们在面向客户的应用程序中具有价值。然而，它们令人印象深刻的能力也引发了人们对对抗性指令带来的风险放大的担忧，这些指令可以被注入第三方攻击者输入的模型中，以操纵LLMS的原始指令并提示意外的操作和内容。因此，了解LLMS准确识别应遵循哪些指令以确保在现实世界场景中安全部署的能力至关重要。在本文中，我们提出了一个开创性的基准，用于自动评估指令跟随LLMS对提示中注入的敌意指令的健壮性。这一基准的目的是量化LLM受注入的敌意指令的影响程度，并评估它们区分这些注入的对抗性指令和原始用户指令的能力。通过使用最先进的指令跟随LLM进行的实验，我们发现它们对敌意指令注入攻击的健壮性存在显著的局限性。此外，我们的研究结果表明，流行的指导性调整模型倾向于在没有真正理解哪些指令应该被遵循的情况下，“过度适应”地遵循提示中的任何指导语。这突出表明，需要解决培训模型理解提示的挑战，而不是仅仅遵循指导短语和完成正文。数据和代码可在\url{https://github.com/Leezekun/Adv-Instruct-Eval}.上找到



## **49. Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems**

(深度)强化学习中的连通超水平集及其在极大极小定理中的应用 cs.LG

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2303.12981v3) [paper-pdf](http://arxiv.org/pdf/2303.12981v3)

**Authors**: Sihan Zeng, Thinh T. Doan, Justin Romberg

**Abstract**: The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger "equiconnectedness" property. To our best knowledge, these are novel and previously unknown discoveries.   We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting robust reinforcement learning problem under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.

摘要: 本文的目的是加深对强化学习中策略优化问题的优化环境的理解。具体地说，我们证明了目标函数关于策略参数的超水平集无论在表格设置下还是在由一类神经网络表示的策略下都是连通集。此外，我们还证明了作为政策参数和报酬的函数的优化目标满足较强的等连通性。据我们所知，这些都是以前未知的新奇发现。我们给出了这些超水平集的连通性在鲁棒强化学习的极大极小定理推导中的一个应用。我们证明了任何一边是凸的，另一边是等连通的极小极大优化程序都遵守极小极大等式(即存在纳什均衡)。我们发现，在对抗性奖励攻击下，这种结构被表现为一个有趣的鲁棒强化学习问题，并且它的极小极大等式的有效性随之而来。这是第一次在文献中确立这样的结果。



## **50. Human-Producible Adversarial Examples**

人类可产生的对抗性例子 cs.CV

Submitted to ICLR 2024

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2310.00438v1) [paper-pdf](http://arxiv.org/pdf/2310.00438v1)

**Authors**: David Khachaturov, Yue Gao, Ilia Shumailov, Robert Mullins, Ross Anderson, Kassem Fawaz

**Abstract**: Visual adversarial examples have so far been restricted to pixel-level image manipulations in the digital world, or have required sophisticated equipment such as 2D or 3D printers to be produced in the physical real world. We present the first ever method of generating human-producible adversarial examples for the real world that requires nothing more complicated than a marker pen. We call them $\textbf{adversarial tags}$. First, building on top of differential rendering, we demonstrate that it is possible to build potent adversarial examples with just lines. We find that by drawing just $4$ lines we can disrupt a YOLO-based model in $54.8\%$ of cases; increasing this to $9$ lines disrupts $81.8\%$ of the cases tested. Next, we devise an improved method for line placement to be invariant to human drawing error. We evaluate our system thoroughly in both digital and analogue worlds and demonstrate that our tags can be applied by untrained humans. We demonstrate the effectiveness of our method for producing real-world adversarial examples by conducting a user study where participants were asked to draw over printed images using digital equivalents as guides. We further evaluate the effectiveness of both targeted and untargeted attacks, and discuss various trade-offs and method limitations, as well as the practical and ethical implications of our work. The source code will be released publicly.

摘要: 到目前为止，视觉对抗性的例子仅限于数字世界中像素级的图像处理，或者需要复杂的设备，如2D或3D打印机，才能在现实世界中生产。我们提出了有史以来第一种为现实世界生成人类可产生的对抗性例子的方法，它只需要一支记号笔就可以了。我们称它们为$\extbf{对抗性标签}$。首先，在差分渲染的基础上，我们演示了仅用线条就可以构建强大的对抗性示例。我们发现，通过只画$4$线，我们可以在$54.8\$案例中中断基于YOLO的模型；将$9$线增加到$9$线会中断$81.8\$测试案例。接下来，我们设计了一种改进的直线放置方法，使其不受人的绘制误差的影响。我们在数字和模拟世界中对我们的系统进行了彻底的评估，并证明我们的标签可以被未经培训的人应用。我们通过进行一项用户研究来展示我们产生真实世界对抗性例子的方法的有效性，在该研究中，参与者被要求使用数字等价物作为指南来绘制印刷图像。我们进一步评估定向和非定向攻击的有效性，并讨论各种取舍和方法限制，以及我们工作的实际和伦理影响。源代码将公开发布。



