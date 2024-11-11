# Latest Adversarial Attack Papers
**update at 2024-11-11 09:21:10**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)**

骨折-抱歉-长凳：揭露对话回合中攻击的框架，这些攻击削弱了SORRY长凳（自动多枪越狱）的拒绝功效和防御 cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2408.16163v2) [paper-pdf](http://arxiv.org/pdf/2408.16163v2)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.

摘要: 本文介绍了FRACTURED-SORRY-Bench，这是一个用于评估大型语言模型（LLM）针对多轮对话攻击的安全性的框架。基于SORRY-Bench数据集，我们提出了一种简单而有效的方法，通过将有害的查询分解为看似无害的子问题来生成对抗性提示。与基线方法相比，我们的方法在GPT-4、GPT-4 o、GPT-4 o-mini和GPT-3.5-Turbo模型中实现了+46.22%的攻击成功率（SVR）最大增加。我们证明这种技术对当前的LLM安全措施构成了挑战，并强调了对微妙的多回合攻击进行更强大的防御的必要性。



## **2. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

梯度Cuff：通过探索拒绝损失景观来检测对大型语言模型的越狱攻击 cs.CR

Accepted by NeurIPS 2024. Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2403.00867v3) [paper-pdf](http://arxiv.org/pdf/2403.00867v3)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.

摘要: 大型语言模型(LLM)正在成为一种重要的生成性人工智能工具，用户输入一个查询，LLM生成一个答案。为了减少危害和误用，已经做出努力，使用先进的培训技术，如从人类反馈中强化学习(RLHF)，使这些LLM与人类价值保持一致。然而，最近的研究强调了LLMS在旨在颠覆嵌入的安全护栏的对抗性越狱企图中的脆弱性。为了应对这一挑战，本文定义并研究了LLMS的拒绝丢失，提出了一种检测越狱企图的梯度袖口方法。梯度袖口利用拒绝丢失场景中观察到的独特属性，包括函数值及其光滑性，设计了一种有效的两步检测策略。在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B-V1.5)和六种越狱攻击(GCG、AutoDAN、Pair、TAP、Base64和LRL)上的实验结果表明，梯度袖口可以显著提高LLM对恶意越狱查询的拒绝能力，同时通过调整检测阈值保持该模型对良性用户查询的性能。



## **3. Attention Masks Help Adversarial Attacks to Bypass Safety Detectors**

注意力口罩帮助对抗攻击绕过安全检测器 cs.CR

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04772v1) [paper-pdf](http://arxiv.org/pdf/2411.04772v1)

**Authors**: Yunfan Shi

**Abstract**: Despite recent research advancements in adversarial attack methods, current approaches against XAI monitors are still discoverable and slower. In this paper, we present an adaptive framework for attention mask generation to enable stealthy, explainable and efficient PGD image classification adversarial attack under XAI monitors. Specifically, we utilize mutation XAI mixture and multitask self-supervised X-UNet for attention mask generation to guide PGD attack. Experiments on MNIST (MLP), CIFAR-10 (AlexNet) have shown that our system can outperform benchmark PGD, Sparsefool and SOTA SINIFGSM in balancing among stealth, efficiency and explainability which is crucial for effectively fooling SOTA defense protected classifiers.

摘要: 尽管最近研究在对抗攻击方法方面取得了进展，但当前针对XAI监视器的方法仍然是可行的，而且速度较慢。在本文中，我们提出了一个用于注意力屏蔽生成的自适应框架，以在XAI监视器下实现隐蔽、可解释和高效的PVD图像分类对抗攻击。具体来说，我们利用突变XAI混合物和多任务自我监督X-UNet来生成注意力屏蔽来引导PVD攻击。在MNIST（MLP）、CIFAR-10（AlexNet）上的实验表明，我们的系统在隐身性、效率和可解释性之间的平衡方面优于基准PVD、Sparsefool和SOTA SINIFGSM，这对于有效欺骗SOTA防御保护分类器至关重要。



## **4. MISGUIDE: Security-Aware Attack Analytics for Smart Grid Load Frequency Control**

MISGUIDE：用于智能电网负载频率控制的安全感知攻击分析 cs.CE

12 page journal

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04731v1) [paper-pdf](http://arxiv.org/pdf/2411.04731v1)

**Authors**: Nur Imtiazul Haque, Prabin Mali, Mohammad Zakaria Haider, Mohammad Ashiqur Rahman, Sumit Paudyal

**Abstract**: Incorporating advanced information and communication technologies into smart grids (SGs) offers substantial operational benefits while increasing vulnerability to cyber threats like false data injection (FDI) attacks. Current SG attack analysis tools predominantly employ formal methods or adversarial machine learning (ML) techniques with rule-based bad data detectors to analyze the attack space. However, these attack analytics either generate simplistic attack vectors detectable by the ML-based anomaly detection models (ADMs) or fail to identify critical attack vectors from complex controller dynamics in a feasible time. This paper introduces MISGUIDE, a novel defense-aware attack analytics designed to extract verifiable multi-time slot-based FDI attack vectors from complex SG load frequency control dynamics and ADMs, utilizing the Gurobi optimizer. MISGUIDE can identify optimal (maliciously triggering under/over frequency relays in minimal time) and stealthy attack vectors. Using real-world load data, we validate the MISGUIDE-identified attack vectors through real-time hardware-in-the-loop (OPALRT) simulations of the IEEE 39-bus system.

摘要: 将先进的信息和通信技术融入智能电网(SGS)可以带来巨大的运营效益，同时增加了对虚假数据注入(FDI)攻击等网络威胁的脆弱性。目前的SG攻击分析工具主要使用形式化方法或对抗性机器学习(ML)技术和基于规则的坏数据检测器来分析攻击空间。然而，这些攻击分析要么生成基于ML的异常检测模型(ADMS)可以检测到的简单攻击向量，要么无法在可行的时间内从复杂的控制器动态中识别关键攻击向量。本文介绍了一种新的防御感知攻击分析工具MisGuide，它利用Gurobi优化器从复杂的SG负载频率控制动态和ADMS中提取可验证的基于多时隙的FDI攻击向量。误导可以识别最优(在最短时间内恶意触发频率下/频率上的继电器)和隐蔽攻击载体。使用真实负荷数据，通过IEEE 39节点系统的实时硬件在环仿真(OPALRT)验证了误导识别的攻击向量。



## **5. Neural Fingerprints for Adversarial Attack Detection**

用于对抗性攻击检测的神经指纹 cs.CV

14 pages

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04533v1) [paper-pdf](http://arxiv.org/pdf/2411.04533v1)

**Authors**: Haim Fisher, Moni Shahar, Yehezkel S. Resheff

**Abstract**: Deep learning models for image classification have become standard tools in recent years. A well known vulnerability of these models is their susceptibility to adversarial examples. These are generated by slightly altering an image of a certain class in a way that is imperceptible to humans but causes the model to classify it wrongly as another class. Many algorithms have been proposed to address this problem, falling generally into one of two categories: (i) building robust classifiers (ii) directly detecting attacked images. Despite the good performance of these detectors, we argue that in a white-box setting, where the attacker knows the configuration and weights of the network and the detector, they can overcome the detector by running many examples on a local copy, and sending only those that were not detected to the actual model. This problem is common in security applications where even a very good model is not sufficient to ensure safety. In this paper we propose to overcome this inherent limitation of any static defence with randomization. To do so, one must generate a very large family of detectors with consistent performance, and select one or more of them randomly for each input. For the individual detectors, we suggest the method of neural fingerprints. In the training phase, for each class we repeatedly sample a tiny random subset of neurons from certain layers of the network, and if their average is sufficiently different between clean and attacked images of the focal class they are considered a fingerprint and added to the detector bank. During test time, we sample fingerprints from the bank associated with the label predicted by the model, and detect attacks using a likelihood ratio test. We evaluate our detectors on ImageNet with different attack methods and model architectures, and show near-perfect detection with low rates of false detection.

摘要: 近年来，用于图像分类的深度学习模型已成为标准工具。这些模型的一个众所周知的弱点是它们容易受到对抗性例子的影响。它们是通过以人类无法察觉的方式稍微改变某个类的图像来生成的，但会导致模型将其错误地归类为另一个类。已经提出了许多算法来解决这个问题，通常分为两类：(I)构建稳健的分类器(Ii)直接检测受攻击的图像。尽管这些检测器的性能很好，但我们认为，在白盒设置中，攻击者知道网络和检测器的配置和权重，他们可以通过在本地副本上运行许多示例，并仅将未检测到的示例发送到实际模型来克服检测器。这个问题在安全应用程序中很常见，即使是非常好的模型也不足以确保安全。在这篇文章中，我们建议用随机化来克服任何静态防御的固有局限性。要做到这一点，必须生成具有一致性能的非常大的检测器家族，并为每个输入随机选择一个或多个检测器。对于单个检测器，我们建议采用神经指纹的方法。在训练阶段，对于每一类，我们重复从网络的某些层对神经元的微小随机子集进行采样，如果它们的平均值在焦点类的干净图像和被攻击的图像之间有足够的差异，则它们被认为是指纹并添加到检测器库中。在测试期间，我们从与模型预测的标签相关联的银行中采样指纹，并使用似然比检验来检测攻击。我们在ImageNet上用不同的攻击方法和模型架构对我们的检测器进行了评估，结果表明我们的检测器在低误检率的情况下接近完美的检测。



## **6. Undermining Image and Text Classification Algorithms Using Adversarial Attacks**

使用对抗攻击削弱图像和文本分类算法 cs.CR

Accepted for presentation at Electronic Imaging Conference 2025

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.03348v2) [paper-pdf](http://arxiv.org/pdf/2411.03348v2)

**Authors**: Langalibalele Lunga, Suhas Sreehari

**Abstract**: Machine learning models are prone to adversarial attacks, where inputs can be manipulated in order to cause misclassifications. While previous research has focused on techniques like Generative Adversarial Networks (GANs), there's limited exploration of GANs and Synthetic Minority Oversampling Technique (SMOTE) in text and image classification models to perform adversarial attacks. Our study addresses this gap by training various machine learning models and using GANs and SMOTE to generate additional data points aimed at attacking text classification models. Furthermore, we extend our investigation to face recognition models, training a Convolutional Neural Network(CNN) and subjecting it to adversarial attacks with fast gradient sign perturbations on key features identified by GradCAM, a technique used to highlight key image characteristics CNNs use in classification. Our experiments reveal a significant vulnerability in classification models. Specifically, we observe a 20 % decrease in accuracy for the top-performing text classification models post-attack, along with a 30 % decrease in facial recognition accuracy. This highlights the susceptibility of these models to manipulation of input data. Adversarial attacks not only compromise the security but also undermine the reliability of machine learning systems. By showcasing the impact of adversarial attacks on both text classification and face recognition models, our study underscores the urgent need for develop robust defenses against such vulnerabilities.

摘要: 机器学习模型容易受到对抗性攻击，在这种攻击中，输入可能被操纵以导致错误分类。虽然以前的研究主要集中在生成性对抗网络(GANS)等技术上，但在文本和图像分类模型中使用生成性对抗网络(GANS)和合成少数过采样技术(SMOTE)来执行敌意攻击的探索有限。我们的研究通过训练各种机器学习模型并使用Gans和Smote生成旨在攻击文本分类模型的额外数据点来解决这一差距。此外，我们将研究扩展到人脸识别模型，训练卷积神经网络(CNN)，并对GradCAM识别的关键特征进行快速梯度符号扰动的对抗性攻击，这是一种用于突出CNN用于分类的关键图像特征的技术。我们的实验揭示了分类模型中的一个重大漏洞。具体地说，我们观察到攻击后表现最好的文本分类模型的准确率下降了20%，面部识别准确率下降了30%。这突显了这些模型对输入数据操纵的敏感性。对抗性攻击不仅破坏了机器学习系统的安全性，而且破坏了系统的可靠性。通过展示对抗性攻击对文本分类和人脸识别模型的影响，我们的研究强调了开发针对此类漏洞的强大防御的迫切需要。



## **7. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

针对医学成像中对抗攻击的鲁棒共形预测的游戏理论防御 cs.LG

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04376v1) [paper-pdf](http://arxiv.org/pdf/2411.04376v1)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.

摘要: 对抗性攻击对深度学习模型的可靠性和安全性构成了严重威胁，特别是在医学成像等关键领域。本文介绍了一种新的框架，它将保形预测与博弈论防御策略相结合，以增强模型对已知和未知对手扰动的稳健性。我们主要研究了三个问题：在已知攻击(RQ1)下构造有效且高效的共形预测集，通过保守阈值(RQ2)确保未知攻击下的覆盖，以及在零和博弈框架(RQ3)下确定最优防御策略。我们的方法包括针对特定的攻击类型训练专门的防御模型，并使用最大和最小分类器来有效地聚合防御。在包括PathMNIST、OrganAMNIST和TIseMNIST在内的MedMNIST数据集上进行的大量实验表明，我们的方法在保持高覆盖率的同时最小化了预测集的大小。博弈论分析表明，最优防御策略往往收敛到一个奇异的稳健模型，在所有评估的数据集上表现优于统一和简单的策略。这项工作推进了不确定性量化和对抗稳健性方面的最新进展，为在对抗环境中部署深度学习模型提供了可靠的机制。



## **8. Towards Secured Smart Grid 2.0: Exploring Security Threats, Protection Models, and Challenges**

迈向安全智能电网2.0：探索安全威胁、保护模型和挑战 cs.NI

30 pages, 21 figures, 5 tables, accepted to appear in IEEE COMST

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.04365v2) [paper-pdf](http://arxiv.org/pdf/2411.04365v2)

**Authors**: Lan-Huong Nguyen, Van-Linh Nguyen, Ren-Hung Hwang, Jian-Jhih Kuo, Yu-Wen Chen, Chien-Chung Huang, Ping-I Pan

**Abstract**: Many nations are promoting the green transition in the energy sector to attain neutral carbon emissions by 2050. Smart Grid 2.0 (SG2) is expected to explore data-driven analytics and enhance communication technologies to improve the efficiency and sustainability of distributed renewable energy systems. These features are beyond smart metering and electric surplus distribution in conventional smart grids. Given the high dependence on communication networks to connect distributed microgrids in SG2, potential cascading failures of connectivity can cause disruption to data synchronization to the remote control systems. This paper reviews security threats and defense tactics for three stakeholders: power grid operators, communication network providers, and consumers. Through the survey, we found that SG2's stakeholders are particularly vulnerable to substation attacks/vandalism, malware/ransomware threats, blockchain vulnerabilities and supply chain breakdowns. Furthermore, incorporating artificial intelligence (AI) into autonomous energy management in distributed energy resources of SG2 creates new challenges. Accordingly, adversarial samples and false data injection on electricity reading and measurement sensors at power plants can fool AI-powered control functions and cause messy error-checking operations in energy storage, wrong energy estimation in electric vehicle charging, and even fraudulent transactions in peer-to-peer energy trading models. Scalable blockchain-based models, physical unclonable function, interoperable security protocols, and trustworthy AI models designed for managing distributed microgrids in SG2 are typical promising protection models for future research.

摘要: 许多国家正在推动能源领域的绿色转型，以期在2050年前实现碳排放中性。智能电网2.0(SG2)预计将探索数据驱动的分析和增强通信技术，以提高分布式可再生能源系统的效率和可持续性。这些功能超越了传统智能电网中的智能计量和剩余电量分配。考虑到在SG2中高度依赖通信网络来连接分布式微电网，潜在的级联连接故障可能会导致与远程控制系统的数据同步中断。本文回顾了电网运营商、通信网络提供商和消费者这三个利益相关者面临的安全威胁和防御策略。通过调查，我们发现SG2的S利益相关者特别容易受到变电站攻击/破坏、恶意软件/勒索软件威胁、区块链漏洞和供应链故障的影响。此外，将人工智能(AI)融入SG2分布式能源的自主能源管理中也带来了新的挑战。因此，发电厂电量读数和测量传感器上的敌意样本和虚假数据注入可能会愚弄人工智能支持的控制功能，并导致储能中混乱的错误检查操作，电动汽车充电中的错误能量估计，甚至P2P能源交易模式中的欺诈性交易。基于区块链的可扩展模型、物理不可克隆功能、可互操作的安全协议以及SG2中为管理分布式微网格而设计的可信AI模型是未来研究的典型保护模型。



## **9. $B^4$: A Black-Box Scrubbing Attack on LLM Watermarks**

$B ' 4 $：对LLM水印的黑匣子清除攻击 cs.CL

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.01222v3) [paper-pdf](http://arxiv.org/pdf/2411.01222v3)

**Authors**: Baizhou Huang, Xiao Pu, Xiaojun Wan

**Abstract**: Watermarking has emerged as a prominent technique for LLM-generated content detection by embedding imperceptible patterns. Despite supreme performance, its robustness against adversarial attacks remains underexplored. Previous work typically considers a grey-box attack setting, where the specific type of watermark is already known. Some even necessitates knowledge about hyperparameters of the watermarking method. Such prerequisites are unattainable in real-world scenarios. Targeting at a more realistic black-box threat model with fewer assumptions, we here propose $B^4$, a black-box scrubbing attack on watermarks. Specifically, we formulate the watermark scrubbing attack as a constrained optimization problem by capturing its objectives with two distributions, a Watermark Distribution and a Fidelity Distribution. This optimization problem can be approximately solved using two proxy distributions. Experimental results across 12 different settings demonstrate the superior performance of $B^4$ compared with other baselines.

摘要: 通过嵌入不可察觉的模式，水印已经成为LLM生成的内容检测的一种重要技术。尽管具有最高的性能，但它对对手攻击的健壮性仍未得到充分开发。以前的工作通常考虑灰盒攻击设置，其中特定类型的水印已知。有些甚至需要关于水印方法的超参数的知识。这样的先决条件在现实世界的场景中是无法实现的。针对一种假设更少、更逼真的黑盒威胁模型，本文提出了一种针对水印的黑盒擦除攻击--$B^4$。具体地说，我们将水印洗涤攻击描述为一个约束优化问题，通过两个分布来捕获其目标，即水印分布和保真度分布。这个优化问题可以使用两个代理分布近似地解决。在12个不同设置下的实验结果表明，与其他基线相比，$B^4$具有更好的性能。



## **10. Transferable Learned Image Compression-Resistant Adversarial Perturbations**

可转移习得图像抗压缩对抗扰动 cs.CV

Accepted by BMVC 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2401.03115v2) [paper-pdf](http://arxiv.org/pdf/2401.03115v2)

**Authors**: Yang Sui, Zhuohang Li, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Zhenzhong Chen

**Abstract**: Adversarial attacks can readily disrupt the image classification system, revealing the vulnerability of DNN-based recognition tasks. While existing adversarial perturbations are primarily applied to uncompressed images or compressed images by the traditional image compression method, i.e., JPEG, limited studies have investigated the robustness of models for image classification in the context of DNN-based image compression. With the rapid evolution of advanced image compression, DNN-based learned image compression has emerged as the promising approach for transmitting images in many security-critical applications, such as cloud-based face recognition and autonomous driving, due to its superior performance over traditional compression. Therefore, there is a pressing need to fully investigate the robustness of a classification system post-processed by learned image compression. To bridge this research gap, we explore the adversarial attack on a new pipeline that targets image classification models that utilize learned image compressors as pre-processing modules. Furthermore, to enhance the transferability of perturbations across various quality levels and architectures of learned image compression models, we introduce a saliency score-based sampling method to enable the fast generation of transferable perturbation. Extensive experiments with popular attack methods demonstrate the enhanced transferability of our proposed method when attacking images that have been post-processed with different learned image compression models.

摘要: 敌意攻击很容易破坏图像分类系统，暴露了基于DNN的识别任务的脆弱性。虽然现有的对抗性扰动主要应用于未压缩图像或使用传统图像压缩方法(即JPEG)压缩的图像，但在基于DNN的图像压缩环境下，已有有限的研究调查了图像分类模型的稳健性。随着先进图像压缩技术的迅速发展，基于DNN的学习图像压缩技术以其优于传统压缩的性能，在基于云的人脸识别和自动驾驶等安全关键应用中成为一种很有前途的图像传输方法。因此，迫切需要充分研究学习图像压缩后处理的分类系统的稳健性。为了弥补这一研究空白，我们探索了一种新的管道上的敌意攻击，该管道的目标是使用学习的图像压缩器作为预处理模块的图像分类模型。此外，为了增强扰动在学习图像压缩模型的不同质量水平和体系结构上的可转移性，我们引入了基于显著分数的采样方法来快速生成可转移的扰动。对常用攻击方法的大量实验表明，当攻击经过不同学习图像压缩模型后处理的图像时，所提出的方法具有更强的可转移性。



## **11. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

Available at: https://proceedings.mlr.press/v244/pizarro24a.html

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2305.17000v8) [paper-pdf](http://arxiv.org/pdf/2305.17000v8)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **12. Reassessing Noise Augmentation Methods in the Context of Adversarial Speech**

对抗性言语背景下重新评估噪音增强方法 eess.AS

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2409.01813v3) [paper-pdf](http://arxiv.org/pdf/2409.01813v3)

**Authors**: Karla Pizzi, Matías Pizarro, Asja Fischer

**Abstract**: In this study, we investigate if noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different state-of-the-art ASR architectures, where each of the ASR architectures is trained under three different augmentation conditions: one subject to background noise, speed variations, and reverberations, another subject to speed variations only, and a third without any form of data augmentation. The results demonstrate that noise augmentation not only improves model performance on noisy speech but also the model's robustness to adversarial attacks.

摘要: 在这项研究中，我们研究了噪音增强训练是否可以同时提高自动语音识别（ASB）系统中的对抗鲁棒性。我们对四种不同最先进的ASB架构的对抗鲁棒性进行了比较分析，其中每个ASB架构都在三种不同的增强条件下训练：一种受到背景噪音、速度变化和回响的影响，另一种仅受到速度变化的影响，第三种没有任何形式的数据增强。结果表明，噪音增强不仅提高了模型在含噪语音上的性能，还提高了模型对对抗攻击的鲁棒性。



## **13. Robustifying automatic speech recognition by extracting slowly varying features**

通过提取缓慢变化的特征来增强自动语音识别 eess.AS

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2112.07400v3) [paper-pdf](http://arxiv.org/pdf/2112.07400v3)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.

摘要: 在过去的几年里，深度学习系统被证明在敌意攻击下是非常脆弱的。基于神经网络的自动语音识别(ASR)系统也不例外。定向和非定向攻击可以修改音频输入信号，使人类仍能识别相同的单词，而ASR系统则被引导预测不同的转录。在本文中，我们提出了一种针对目标攻击的防御机制，即在将输入输入到ASR系统之前，通过慢速特征分析、低通滤波或两者结合的方法，从音频信号中去除快速变化的特征。我们对以这种方式处理的数据训练的混合ASR模型进行了实证分析。虽然得到的模型在良性数据上表现得相当好，但它们对目标对手攻击的健壮性要强得多：我们最终提出的模型在干净数据上的性能与基准模型相似，但健壮性要高四倍以上。



## **14. Fundamental Limits of Routing Attack on Network Overload**

网络过载路由攻击的基本限制 cs.NI

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.03749v1) [paper-pdf](http://arxiv.org/pdf/2411.03749v1)

**Authors**: Xinyu Wu, Eytan Modiano

**Abstract**: We quantify the threat of network adversaries to inducing \emph{network overload} through \emph{routing attacks}, where a subset of network nodes are hijacked by an adversary. We develop routing attacks on the hijacked nodes for two objectives related to overload: \emph{no-loss throughput minimization} and \emph{loss maximization}. The first objective attempts to identify a routing attack that minimizes the network's throughput that is guaranteed to survive. We develop a polynomial-time algorithm that can output the optimal routing attack in multi-hop networks with global information on the network's topology, and an algorithm with an approximation ratio of $2$ under partial information. The second objective attempts to maximize the throughput loss. We demonstrate that this problem is NP-hard, and develop two approximation algorithms with multiplicative and additive guarantees respectively in single-hop networks. We further investigate the adversary's optimal selection of nodes to hijack that can maximize network overload. We propose a heuristic polynomial-time algorithm to solve this NP-hard problem, and prove its optimality in special cases. We validate the near-optimal performance of the proposed algorithms over a wide range of network settings. Our results demonstrate that the proposed algorithms can accurately quantify the risk of overload given an arbitrary set of hijacked nodes and identify the critical nodes that should be protected against routing attacks.

摘要: 我们量化了网络攻击者通过网络攻击来诱导网络过载的威胁，其中网络节点的子集被攻击者劫持。针对与负载相关的两个目标：无损失吞吐量最小化和损失最大化，我们对被劫持节点进行了路由攻击。第一个目标是尝试识别能够最大限度地减少网络吞吐量的路由攻击，保证网络能够存活下来。我们提出了一个多项式时间算法，可以在多跳网络中输出关于网络拓扑的全局信息的最优路由攻击，以及在部分信息下的近似比为$2$的算法。第二个目标试图最大化吞吐量损失。我们证明了该问题是NP难的，并在单跳网络中提出了两种分别具有乘性和加性保证的近似算法。我们进一步研究了攻击者如何选择最优的节点进行劫持，从而最大化网络过载。我们提出了一个启发式多项式时间算法来解决这个NP-Hard问题，并在特殊情况下证明了它的最优性。我们验证了所提出的算法在广泛的网络环境下的近乎最优的性能。实验结果表明，在给定任意一组被劫持节点的情况下，所提出的算法可以准确地量化过载的风险，并识别出应该防止路由攻击的关键节点。



## **15. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2409.20002v2) [paper-pdf](http://arxiv.org/pdf/2409.20002v2)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型的广泛应用对其推理性能的优化提出了强烈的需求。目前用于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，而在很大程度上忽略了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道来自共享缓存和GPU内存分配，可以用来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了在传统计算系统中观察到的安全挑战，突显出迫切需要解决LLM服务基础设施中潜在的信息泄漏问题。在本文中，我们报告了一种新的攻击策略，旨在利用LLM部署中固有的计时侧通道，特别是针对广泛用于提高LLM推理性能的键值(KV)缓存和语义缓存。我们的方法利用计时测量和分类模型来检测缓存命中，允许对手以高精度推断私人提示。我们还提出了一种逐令牌搜索算法来高效地恢复缓存中的共享提示前缀，证明了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑盒测试的实验研究表明，这种隐私风险是完全现实的，具有显著的后果。我们的发现强调了强有力的缓解措施的必要性，以保护LLM系统免受此类新出现的威胁。



## **16. Formal Logic-guided Robust Federated Learning against Poisoning Attacks**

针对中毒攻击的形式逻辑引导鲁棒联邦学习 cs.CR

12 pages, 4 figures, 6 tables

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.03231v2) [paper-pdf](http://arxiv.org/pdf/2411.03231v2)

**Authors**: Dung Thuy Nguyen, Ziyan An, Taylor T. Johnson, Meiyi Ma, Kevin Leach

**Abstract**: Federated Learning (FL) offers a promising solution to the privacy concerns associated with centralized Machine Learning (ML) by enabling decentralized, collaborative learning. However, FL is vulnerable to various security threats, including poisoning attacks, where adversarial clients manipulate the training data or model updates to degrade overall model performance. Recognizing this threat, researchers have focused on developing defense mechanisms to counteract poisoning attacks in FL systems. However, existing robust FL methods predominantly focus on computer vision tasks, leaving a gap in addressing the unique challenges of FL with time series data. In this paper, we present FLORAL, a defense mechanism designed to mitigate poisoning attacks in federated learning for time-series tasks, even in scenarios with heterogeneous client data and a large number of adversarial participants. Unlike traditional model-centric defenses, FLORAL leverages logical reasoning to evaluate client trustworthiness by aligning their predictions with global time-series patterns, rather than relying solely on the similarity of client updates. Our approach extracts logical reasoning properties from clients, then hierarchically infers global properties, and uses these to verify client updates. Through formal logic verification, we assess the robustness of each client contribution, identifying deviations indicative of adversarial behavior. Experimental results on two datasets demonstrate the superior performance of our approach compared to existing baseline methods, highlighting its potential to enhance the robustness of FL to time series applications. Notably, FLORAL reduced the prediction error by 93.27% in the best-case scenario compared to the second-best baseline. Our code is available at https://anonymous.4open.science/r/FLORAL-Robust-FTS.

摘要: 联合学习(FL)通过支持分散的、协作的学习，为集中式机器学习(ML)相关的隐私问题提供了一个有前途的解决方案。然而，FL容易受到各种安全威胁，包括中毒攻击，即敌对客户端操纵训练数据或模型更新以降低整体模型性能。认识到这种威胁，研究人员专注于开发防御机制来对抗FL系统中的中毒攻击。然而，现有的稳健的外语学习方法主要集中在计算机视觉任务上，在用时间序列数据解决外语的独特挑战方面留下了空白。在本文中，我们提出了一种防御机制FLOLAR，该机制被设计用于在时间序列任务的联合学习中缓解中毒攻击，即使在具有异质客户端数据和大量对抗性参与者的场景中也是如此。与传统的以模型为中心的防御不同，FLORAL利用逻辑推理来评估客户的可信度，方法是将他们的预测与全球时间序列模式保持一致，而不是仅仅依赖客户更新的相似性。我们的方法从客户端提取逻辑推理属性，然后分层推断全局属性，并使用这些属性来验证客户端更新。通过形式逻辑验证，我们评估每个客户贡献的健壮性，识别指示对抗性行为的偏差。在两个数据集上的实验结果表明，与现有的基线方法相比，我们的方法具有更好的性能，突出了它在增强FL对时间序列应用的稳健性方面的潜力。值得注意的是，与次好的基线相比，FLORAL在最好的情况下将预测误差降低了93.27%。我们的代码可以在https://anonymous.4open.science/r/FLORAL-Robust-FTS.上找到



## **17. Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset**

通过虚构面部身份数据集对视觉语言模型取消学习进行基准测试 cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03554v1) [paper-pdf](http://arxiv.org/pdf/2411.03554v1)

**Authors**: Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, Yejin Choi, Muhao Chen, Chaowei Xiao

**Abstract**: Machine unlearning has emerged as an effective strategy for forgetting specific information in the training data. However, with the increasing integration of visual data, privacy concerns in Vision Language Models (VLMs) remain underexplored. To address this, we introduce Facial Identity Unlearning Benchmark (FIUBench), a novel VLM unlearning benchmark designed to robustly evaluate the effectiveness of unlearning algorithms under the Right to be Forgotten setting. Specifically, we formulate the VLM unlearning task via constructing the Fictitious Facial Identity VQA dataset and apply a two-stage evaluation pipeline that is designed to precisely control the sources of information and their exposure levels. In terms of evaluation, since VLM supports various forms of ways to ask questions with the same semantic meaning, we also provide robust evaluation metrics including membership inference attacks and carefully designed adversarial privacy attacks to evaluate the performance of algorithms. Through the evaluation of four baseline VLM unlearning algorithms within FIUBench, we find that all methods remain limited in their unlearning performance, with significant trade-offs between model utility and forget quality. Furthermore, our findings also highlight the importance of privacy attacks for robust evaluations. We hope FIUBench will drive progress in developing more effective VLM unlearning algorithms.

摘要: 机器遗忘已经成为一种遗忘训练数据中特定信息的有效策略。然而，随着视觉数据的日益集成，视觉语言模型(VLM)中的隐私问题仍然没有得到充分的研究。为了解决这个问题，我们引入了面部身份遗忘基准(FIUB边)，这是一个新的VLM遗忘基准，设计用于在被遗忘的权利设置下稳健地评估遗忘算法的有效性。具体地说，我们通过构建虚拟面部身份VQA数据集来制定VLM遗忘任务，并应用旨在精确控制信息源及其暴露水平的两阶段评估管道。在评估方面，由于VLM支持多种形式的具有相同语义的问题，我们还提供了健壮的评估指标，包括成员关系推理攻击和精心设计的对抗性隐私攻击来评估算法的性能。通过对FIUBuch四种基线VLM遗忘算法的评估，我们发现所有方法的遗忘性能都是有限的，模型效用和遗忘质量之间存在着显著的权衡。此外，我们的发现还强调了隐私攻击对于稳健评估的重要性。我们希望FIUB边将推动在开发更有效的VLM遗忘算法方面取得进展。



## **18. Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS**

引入扰动能力评分（PS）增强ML-NIDS对抗规避攻击的鲁棒性 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2409.07448v2) [paper-pdf](http://arxiv.org/pdf/2409.07448v2)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: As network security threats continue to evolve, safeguarding Machine Learning (ML)-based Network Intrusion Detection Systems (NIDS) from adversarial attacks is crucial. This paper introduces the notion of feature perturb-ability and presents a novel Perturb-ability Score (PS) metric that identifies NIDS features susceptible to manipulation in the problem-space by an attacker. By quantifying a feature's susceptibility to perturbations within the problem-space, the PS facilitates the selection of features that are inherently more robust against evasion adversarial attacks on ML-NIDS during the feature selection phase. These features exhibit natural resilience to perturbations, as they are heavily constrained by the problem-space limitations and correlations of the NIDS domain. Furthermore, manipulating these features may either disrupt the malicious function of evasion adversarial attacks on NIDS or render the network traffic invalid for processing (or both). This proposed novel approach employs a fresh angle by leveraging network domain constraints as a defense mechanism against problem-space evasion adversarial attacks targeting ML-NIDS. We demonstrate the effectiveness of our PS-guided feature selection defense in enhancing NIDS robustness. Experimental results across various ML-based NIDS models and public datasets show that selecting only robust features (low-PS features) can maintain solid detection performance while significantly reducing vulnerability to evasion adversarial attacks. Additionally, our findings verify that the PS effectively identifies NIDS features highly vulnerable to problem-space perturbations.

摘要: 随着网络安全威胁的不断演变，保护基于机器学习(ML)的网络入侵检测系统(NID)免受对手攻击至关重要。引入了特征扰动能力的概念，提出了一种新的扰动能力得分(PS)度量方法，用于识别网络入侵检测系统在问题空间中易受攻击者操纵的特征。通过量化特征对问题空间内扰动的敏感性，PS有助于选择在特征选择阶段对ML-NID的逃避攻击具有更强稳健性的特征。这些功能表现出对扰动的自然弹性，因为它们受到NIDS域的问题空间限制和相关性的严重限制。此外，操纵这些功能可能会破坏躲避对NID的恶意攻击的恶意功能，或者使网络流量无法处理(或两者兼而有之)。该方法从一个全新的角度，利用网络域约束作为针对ML-NID的问题空间规避攻击的防御机制。我们展示了我们的PS引导的特征选择防御在增强网络入侵检测系统健壮性方面的有效性。在各种基于ML的网络入侵检测模型和公共数据集上的实验结果表明，只选择健壮的特征(低PS特征)可以保持可靠的检测性能，同时显著降低对躲避对手攻击的脆弱性。此外，我们的发现验证了PS有效地识别了高度易受问题空间扰动影响的NIDS特征。



## **19. Oblivious Defense in ML Models: Backdoor Removal without Detection**

ML模型中的无意防御：在不检测的情况下删除后门 cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03279v1) [paper-pdf](http://arxiv.org/pdf/2411.03279v1)

**Authors**: Shafi Goldwasser, Jonathan Shafer, Neekon Vafa, Vinod Vaikuntanathan

**Abstract**: As society grows more reliant on machine learning, ensuring the security of machine learning systems against sophisticated attacks becomes a pressing concern. A recent result of Goldwasser, Kim, Vaikuntanathan, and Zamir (2022) shows that an adversary can plant undetectable backdoors in machine learning models, allowing the adversary to covertly control the model's behavior. Backdoors can be planted in such a way that the backdoored machine learning model is computationally indistinguishable from an honest model without backdoors.   In this paper, we present strategies for defending against backdoors in ML models, even if they are undetectable. The key observation is that it is sometimes possible to provably mitigate or even remove backdoors without needing to detect them, using techniques inspired by the notion of random self-reducibility. This depends on properties of the ground-truth labels (chosen by nature), and not of the proposed ML model (which may be chosen by an attacker).   We give formal definitions for secure backdoor mitigation, and proceed to show two types of results. First, we show a "global mitigation" technique, which removes all backdoors from a machine learning model under the assumption that the ground-truth labels are close to a Fourier-heavy function. Second, we consider distributions where the ground-truth labels are close to a linear or polynomial function in $\mathbb{R}^n$. Here, we show "local mitigation" techniques, which remove backdoors with high probability for every inputs of interest, and are computationally cheaper than global mitigation. All of our constructions are black-box, so our techniques work without needing access to the model's representation (i.e., its code or parameters). Along the way we prove a simple result for robust mean estimation.

摘要: 随着社会变得越来越依赖机器学习，确保机器学习系统免受复杂攻击的安全成为一个紧迫的问题。Goldwasser，Kim，Vaikuntanathan和Zamir(2022)最近的一个结果表明，对手可以在机器学习模型中植入不可检测的后门，允许对手秘密控制模型的行为。后门可以被植入这样一种方式，即被后门的机器学习模型在计算上与没有后门的诚实模型没有区别。在本文中，我们提出了在ML模型中防御后门的策略，即使它们是不可检测的。关键的观察结果是，使用受随机自还原概念启发的技术，有时可以在不需要检测到后门的情况下，以可证明的方式减轻甚至删除后门。这取决于基本事实标签的属性(自然选择)，而不是建议的ML模型(可能由攻击者选择)的属性。我们给出了安全后门缓解的正式定义，并继续展示了两种类型的结果。首先，我们展示了一种“全局缓解”技术，该技术在假设基本事实标签接近傅立叶重函数的情况下，从机器学习模型中移除所有后门。其次，我们考虑基本事实标号在$\mathbb{R}^n$中接近于线性或多项式函数的分布。在这里，我们展示了“局部缓解”技术，它为每个感兴趣的输入高概率地移除后门，并且在计算上比全局缓解更便宜。我们的所有构造都是黑盒结构，因此我们的技术无需访问模型的表示形式(即其代码或参数)即可工作。在此过程中，我们证明了稳健均值估计的一个简单结果。



## **20. Gradient-Guided Conditional Diffusion Models for Private Image Reconstruction: Analyzing Adversarial Impacts of Differential Privacy and Denoising**

用于私人图像重建的对象引导条件扩散模型：分析差异隐私和去噪的对抗影响 cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03053v1) [paper-pdf](http://arxiv.org/pdf/2411.03053v1)

**Authors**: Tao Huang, Jiayang Meng, Hong Chen, Guolong Zheng, Xu Yang, Xun Yi, Hua Wang

**Abstract**: We investigate the construction of gradient-guided conditional diffusion models for reconstructing private images, focusing on the adversarial interplay between differential privacy noise and the denoising capabilities of diffusion models. While current gradient-based reconstruction methods struggle with high-resolution images due to computational complexity and prior knowledge requirements, we propose two novel methods that require minimal modifications to the diffusion model's generation process and eliminate the need for prior knowledge. Our approach leverages the strong image generation capabilities of diffusion models to reconstruct private images starting from randomly generated noise, even when a small amount of differentially private noise has been added to the gradients. We also conduct a comprehensive theoretical analysis of the impact of differential privacy noise on the quality of reconstructed images, revealing the relationship among noise magnitude, the architecture of attacked models, and the attacker's reconstruction capability. Additionally, extensive experiments validate the effectiveness of our proposed methods and the accuracy of our theoretical findings, suggesting new directions for privacy risk auditing using conditional diffusion models.

摘要: 我们研究了用于重建私人图像的梯度引导的条件扩散模型的构造，重点研究了差分隐私噪声与扩散模型的去噪能力之间的对抗性相互作用。由于计算复杂性和先验知识的要求，现有的基于梯度的重建方法难以处理高分辨率的图像，我们提出了两种新的方法，它们只需要对扩散模型的生成过程进行最小程度的修改，并且不需要先验知识。我们的方法利用扩散模型强大的图像生成能力，从随机生成的噪声开始重建私人图像，即使在梯度中添加了少量的差分私人噪声。我们还对差分隐私噪声对重建图像质量的影响进行了全面的理论分析，揭示了噪声大小、攻击模型的体系结构和攻击者的重建能力之间的关系。此外，大量的实验验证了我们提出的方法的有效性和我们的理论发现的准确性，为使用条件扩散模型进行隐私风险审计提供了新的方向。



## **21. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

对抗性马尔科夫游戏：基于决策的自适应攻击和防御 cs.AI

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2312.13435v2) [paper-pdf](http://arxiv.org/pdf/2312.13435v2)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.

摘要: 尽管做出了相当大的努力来使它们健壮，但现实世界中基于ML的系统仍然容易受到基于决策的攻击，因为到目前为止，对其操作健壮性的确凿证据被证明是难以处理的。健壮性评估的规范方法要求自适应攻击，即完全了解防御并量身定做以绕过它。在这项研究中，我们引入了一个更广泛的适应性概念，并展示了攻击和防御如何从它和通过互动相互学习中受益。我们提出并评估了一个框架，用于通过形成的竞争博弈自适应地优化黑盒攻击和防御。要可靠地衡量健壮性，重要的是要针对现实和最坏情况下的攻击进行评估。因此，我们通过自适应控制来增强攻击和可供其使用的躲避武器，并观察到同样可以对防御做同样的事情，然后我们首先分开评估它们，然后在多智能体的角度下进行联合评估。我们演示了控制系统如何响应的主动防御是面对基于决策的攻击时模型强化的必要补充；然后说明如何通过自适应攻击来规避这些防御，最终只会引发主动和自适应防御。我们通过广泛的理论和经验调查验证了我们的观察结果，以确认启用AI的对手对基于黑盒ML的系统构成了相当大的威胁，重新点燃了众所周知的军备竞赛，其中防御也必须启用AI。简而言之，我们解决了适应性对手带来的挑战，并开发了适应性防御，从而制定了有效的策略，以确保部署在现实世界中的基于ML的系统的健壮性。



## **22. Flashy Backdoor: Real-world Environment Backdoor Attack on SNNs with DVS Cameras**

浮华的后门：现实环境对使用DVS摄像机的SNN进行后门攻击 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03022v1) [paper-pdf](http://arxiv.org/pdf/2411.03022v1)

**Authors**: Roberto Riaño, Gorka Abad, Stjepan Picek, Aitor Urbieta

**Abstract**: While security vulnerabilities in traditional Deep Neural Networks (DNNs) have been extensively studied, the susceptibility of Spiking Neural Networks (SNNs) to adversarial attacks remains mostly underexplored. Until now, the mechanisms to inject backdoors into SNN models have been limited to digital scenarios; thus, we present the first evaluation of backdoor attacks in real-world environments.   We begin by assessing the applicability of existing digital backdoor attacks and identifying their limitations for deployment in physical environments. To address each of the found limitations, we present three novel backdoor attack methods on SNNs, i.e., Framed, Strobing, and Flashy Backdoor. We also assess the effectiveness of traditional backdoor procedures and defenses adapted for SNNs, such as pruning, fine-tuning, and fine-pruning. The results show that while these procedures and defenses can mitigate some attacks, they often fail against stronger methods like Flashy Backdoor or sacrifice too much clean accuracy, rendering the models unusable.   Overall, all our methods can achieve up to a 100% Attack Success Rate while maintaining high clean accuracy in every tested dataset. Additionally, we evaluate the stealthiness of the triggers with commonly used metrics, finding them highly stealthy. Thus, we propose new alternatives more suited for identifying poisoned samples in these scenarios. Our results show that further research is needed to ensure the security of SNN-based systems against backdoor attacks and their safe application in real-world scenarios. The code, experiments, and results are available in our repository.

摘要: 虽然传统深度神经网络(DNN)的安全漏洞已经得到了广泛的研究，但尖峰神经网络(SNN)对敌意攻击的敏感性仍未得到充分的研究。到目前为止，将后门注入SNN模型的机制仅限于数字场景；因此，我们提出了对现实环境中的后门攻击的第一次评估。我们首先评估现有数字后门攻击的适用性，并确定它们在物理环境中部署的局限性。针对这些缺陷，我们提出了三种新的针对SNN的后门攻击方法，即框架式、选通式和闪光式后门。我们还评估了适用于SNN的传统后门程序和防御措施的有效性，例如修剪、微调和精细修剪。结果表明，虽然这些程序和防御措施可以缓解一些攻击，但它们往往无法通过华而不实的后门等更强大的方法，或者牺牲太多干净的准确性，导致模型无法使用。总体而言，我们的所有方法都可以实现高达100%的攻击成功率，同时在每个测试数据集中保持高清洁准确率。此外，我们使用常用的指标评估触发器的隐蔽性，发现它们具有高度的隐蔽性。因此，我们提出了更适合于在这些场景中识别有毒样本的新的替代方案。我们的结果表明，需要进一步的研究来确保基于SNN的系统免受后门攻击的安全性，以及它们在现实场景中的安全应用。代码、实验和结果都可以在我们的存储库中找到。



## **23. Region-Guided Attack on the Segment Anything Model (SAM)**

对分段任意模型（Sam）的区域引导攻击 cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02974v1) [paper-pdf](http://arxiv.org/pdf/2411.02974v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao

**Abstract**: The Segment Anything Model (SAM) is a cornerstone of image segmentation, demonstrating exceptional performance across various applications, particularly in autonomous driving and medical imaging, where precise segmentation is crucial. However, SAM is vulnerable to adversarial attacks that can significantly impair its functionality through minor input perturbations. Traditional techniques, such as FGSM and PGD, are often ineffective in segmentation tasks due to their reliance on global perturbations that overlook spatial nuances. Recent methods like Attack-SAM-K and UAD have begun to address these challenges, but they frequently depend on external cues and do not fully leverage the structural interdependencies within segmentation processes. This limitation underscores the need for a novel adversarial strategy that exploits the unique characteristics of segmentation tasks. In response, we introduce the Region-Guided Attack (RGA), designed specifically for SAM. RGA utilizes a Region-Guided Map (RGM) to manipulate segmented regions, enabling targeted perturbations that fragment large segments and expand smaller ones, resulting in erroneous outputs from SAM. Our experiments demonstrate that RGA achieves high success rates in both white-box and black-box scenarios, emphasizing the need for robust defenses against such sophisticated attacks. RGA not only reveals SAM's vulnerabilities but also lays the groundwork for developing more resilient defenses against adversarial threats in image segmentation.

摘要: Segment Anything Model(SAM)是图像分割的基石，在各种应用中表现出卓越的性能，特别是在自动驾驶和医学成像中，准确的分割至关重要。然而，SAM很容易受到对抗性攻击，这些攻击可能会通过微小的输入扰动显著损害其功能。传统的分割技术，如FGSM和PGD，在分割任务中往往是无效的，因为它们依赖于忽略空间细微差别的全局扰动。最近的方法，如攻击-SAM-K和UAD已经开始解决这些挑战，但它们经常依赖外部线索，并且没有充分利用分割过程中的结构相互依赖。这一限制强调了需要一种利用分段任务的独特特征的新的对抗性策略。作为回应，我们引入了专门为SAM设计的区域制导攻击(RGA)。RGA利用区域引导地图(RGM)来处理分割的区域，从而实现了将大片段分割并扩展小片段的有针对性的扰动，从而导致SAM的错误输出。我们的实验表明，RGA在白盒和黑盒场景中都取得了很高的成功率，强调了对这种复杂攻击的稳健防御的必要性。RGA不仅揭示了SAM的漏洞，而且为开发更具弹性的防御图像分割中的对抗性威胁奠定了基础。



## **24. Bias in the Mirror: Are LLMs opinions robust to their own adversarial attacks ?**

镜子中的偏见：LLM的观点是否能抵御自己的对抗攻击？ cs.CL

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2410.13517v2) [paper-pdf](http://arxiv.org/pdf/2410.13517v2)

**Authors**: Virgile Rennard, Christos Xypolopoulos, Michalis Vazirgiannis

**Abstract**: Large language models (LLMs) inherit biases from their training data and alignment processes, influencing their responses in subtle ways. While many studies have examined these biases, little work has explored their robustness during interactions. In this paper, we introduce a novel approach where two instances of an LLM engage in self-debate, arguing opposing viewpoints to persuade a neutral version of the model. Through this, we evaluate how firmly biases hold and whether models are susceptible to reinforcing misinformation or shifting to harmful viewpoints. Our experiments span multiple LLMs of varying sizes, origins, and languages, providing deeper insights into bias persistence and flexibility across linguistic and cultural contexts.

摘要: 大型语言模型（LLM）从其训练数据和对齐过程中继承了偏差，以微妙的方式影响其响应。虽然许多研究已经检查了这些偏差，但很少有工作探索它们在互动过程中的稳健性。在本文中，我们引入了一种新颖的方法，其中两个LLM实例进行自我辩论，争论相反的观点以说服模型的中立版本。通过此，我们评估偏见的存在程度，以及模型是否容易强化错误信息或转向有害观点。我们的实验跨越了不同规模、起源和语言的多个LLM，为跨语言和文化背景的偏见持续性和灵活性提供了更深入的见解。



## **25. Towards Efficient Transferable Preemptive Adversarial Defense**

迈向高效的可转让先发制人的对抗防御 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2407.15524v3) [paper-pdf](http://arxiv.org/pdf/2407.15524v3)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, we have devised a proactive strategy for "attacking" the medias before it is attacked by the third party, so that when the protected medias are further attacked, the adversarial perturbations are automatically neutralized. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks. The proposed methodology will be made available on GitHub.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对不起眼的扰动(即对抗性攻击)的敏感性而变得不可信任。攻击者可能会利用这种敏感性来操纵预测。为了防御此类攻击，我们制定了一种主动策略，在媒体受到第三方攻击之前对其进行“攻击”，以便当受保护的媒体进一步受到攻击时，对手的干扰会自动被中和。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。我们还设计了我们所知的第一个有效的白盒自适应恢复攻击，并证明了我们的防御策略添加的保护是不可逆转的，除非主干模型、算法和设置完全受损。这项工作为主动防御对抗性攻击提供了新的方向。拟议的方法将在GitHub上提供。



## **26. Query-Efficient Adversarial Attack Against Vertical Federated Graph Learning**

针对垂直联邦图学习的查询高效对抗攻击 cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02809v1) [paper-pdf](http://arxiv.org/pdf/2411.02809v1)

**Authors**: Jinyin Chen, Wenbo Mu, Luxin Zhang, Guohan Huang, Haibin Zheng, Yao Cheng

**Abstract**: Graph neural network (GNN) has captured wide attention due to its capability of graph representation learning for graph-structured data. However, the distributed data silos limit the performance of GNN. Vertical federated learning (VFL), an emerging technique to process distributed data, successfully makes GNN possible to handle the distributed graph-structured data. Despite the prosperous development of vertical federated graph learning (VFGL), the robustness of VFGL against the adversarial attack has not been explored yet. Although numerous adversarial attacks against centralized GNNs are proposed, their attack performance is challenged in the VFGL scenario. To the best of our knowledge, this is the first work to explore the adversarial attack against VFGL. A query-efficient hybrid adversarial attack framework is proposed to significantly improve the centralized adversarial attacks against VFGL, denoted as NA2, short for Neuron-based Adversarial Attack. Specifically, a malicious client manipulates its local training data to improve its contribution in a stealthy fashion. Then a shadow model is established based on the manipulated data to simulate the behavior of the server model in VFGL. As a result, the shadow model can improve the attack success rate of various centralized attacks with a few queries. Extensive experiments on five real-world benchmarks demonstrate that NA2 improves the performance of the centralized adversarial attacks against VFGL, achieving state-of-the-art performance even under potential adaptive defense where the defender knows the attack method. Additionally, we provide interpretable experiments of the effectiveness of NA2 via sensitive neurons identification and visualization of t-SNE.

摘要: 图神经网络(GNN)因其对图结构数据的图表示学习能力而受到广泛关注。然而，分布式数据孤岛限制了GNN的性能。垂直联合学习(VFL)是一种新兴的分布式数据处理技术，它成功地使GNN处理分布式图结构数据成为可能。尽管垂直联邦图学习(VFGL)得到了蓬勃的发展，但VFGL对敌意攻击的健壮性还没有得到研究。虽然许多针对集中式GNN的对抗性攻击被提出，但它们的攻击性能在VFGL场景中受到挑战。据我们所知，这是第一个探索VFGL对抗性攻击的工作。针对VFGL的集中式对抗攻击，提出了一种查询高效的混合对抗攻击框架，简称NA2，即基于神经元的对抗攻击。具体地说，恶意客户端操纵其本地训练数据，以秘密方式提高其贡献。然后根据处理后的数据建立阴影模型，在VFGL中模拟服务器模型的行为。因此，影子模型可以提高查询次数较少的各种集中式攻击的攻击成功率。在五个真实基准上的广泛实验表明，NA2提高了对VFGL的集中式对抗性攻击的性能，即使在防御者知道攻击方法的潜在自适应防御下也获得了最先进的性能。此外，我们还通过敏感神经元的识别和t-SNE的可视化提供了Na2有效性的可解释性实验。



## **27. TRANSPOSE: Transitional Approaches for Spatially-Aware LFI Resilient FSM Encoding**

TRANSPOSE：空间感知LFI弹性RSM编码的过渡方法 cs.CR

14 pages, 11 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02798v1) [paper-pdf](http://arxiv.org/pdf/2411.02798v1)

**Authors**: Muhtadi Choudhury, Minyan Gao, Avinash Varna, Elad Peer, Domenic Forte

**Abstract**: Finite state machines (FSMs) regulate sequential circuits, including access to sensitive information and privileged CPU states. Courtesy of contemporary research on laser attacks, laser-based fault injection (LFI) is becoming even more precise where an adversary can thwart chip security by altering individual flip-flop (FF) values. Different laser models, e.g., bit flip, bit set, and bit reset, have been developed to appreciate LFI on practical targets. As traditional approaches may incorporate substantial overhead, state-based SPARSE and transition-based TAMED countermeasures were proposed in our prior work to improve FSM resiliency efficiently. TAMED overcame SPARSE's limitation of being too conservative, and generating multiple LFI resilient encodings for contemporary LFI models on demand. SPARSE, however, incorporated design layout information into its vulnerability estimation which makes its vulnerability estimation metric more accurate. In this paper, we extend TAMED by proposing a transition-based encoding CAD framework (TRANSPOSE), that incorporates spatial transitional vulnerability metrics to quantify design susceptibility of FSMs based on both the bit flip model and the set-reset models. TRANSPOSE also incorporates floorplan optimization into its framework to accommodate secure spatial inter-distance of FF-sensitive regions. All TRANSPOSE approaches are demonstrated on 5 multifarious benchmarks and outperform existing FSM encoding schemes/frameworks in terms of security and overhead.

摘要: 有限状态机(FSM)管理时序电路，包括访问敏感信息和特权CPU状态。多亏了当代对激光攻击的研究，基于激光的故障注入(LFI)正变得更加精确，其中对手可以通过改变单个触发器(FF)值来破坏芯片安全。已经开发了不同的激光模型，例如比特翻转、比特设置和比特重置，以在实际目标上评价LFI。由于传统方法可能会带来较大的开销，我们在以前的工作中提出了基于状态的稀疏和基于转移的驯化对策来有效地提高有限状态机的弹性。驯服克服了Sparse过于保守的局限性，并根据需要为当代LFI模型生成多种LFI弹性编码。然而，Sparse将设计布局信息融入到其漏洞估计中，从而使其漏洞估计度量更加准确。在本文中，我们通过提出一个基于转移的编码CAD框架(TRANSPOSE)来扩展TAMD，该框架结合了空间转移脆弱性度量来量化基于位翻转模型和设置-重置模型的FSM的设计敏感度。Transspose还将布局优化整合到其框架中，以适应FF敏感区域的安全空间间距。所有转置方法都在5个不同的基准上进行了演示，并在安全性和开销方面优于现有的FSM编码方案/框架。



## **28. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

随机猴子在起作用：随机增强轻松打破LLM安全一致 cs.LG

Under peer review

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02785v1) [paper-pdf](http://arxiv.org/pdf/2411.02785v1)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.

摘要: 大型语言模型(LLM)的安全一致性最近已成为模型开发人员的一个重要目标。作为回应，越来越多的工作一直在研究如何通过各种越狱方法绕过安全对准，例如对抗性攻击。然而，这些越狱方法可能相当昂贵，或者涉及大量的创造力和努力，从而引入了恶意用户是高资源或老练的假设。在本文中，我们研究了输入提示的简单随机增强如何影响最新的LLMS的安全对齐有效性，例如Llama 3和Qwen 2。我们对17个不同的模型进行了深入的评估，并研究了在多个维度的随机增强下的安全性交集：增强类型、模型大小、量化、基于微调的防御和解码策略(例如采样温度)。我们表明，低资源和简单的攻击者，即$\textit{随机猴子}$，可以显著提高他们绕过对齐的机会，每个提示只需25个随机扩展。



## **29. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

指导合作：对复杂网络中囚犯困境的对抗攻击 physics.soc-ph

19 pages, 4 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2406.19692v4) [paper-pdf](http://arxiv.org/pdf/2406.19692v4)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.

摘要: 本研究探讨了对抗攻击概念在复杂网络中囚犯困境游戏中控制合作演变的应用。具体来说，它提出了一种简单的对抗攻击方法，通过向社交网络添加小扰动来推动玩家的策略走向目标状态。在模型和现实世界网络上对所提出的方法进行了评估。数值模拟表明，与其他技术相比，所提出的方法可以有效地促进协作，且扰动要小得多。此外，这项研究表明，对抗性攻击也可能有助于抑制合作（促进叛逃）。研究结果表明，对社交网络的对抗性攻击可以成为促进和抑制合作的有力工具，为控制社会系统中的合作行为开辟了新的可能性，同时也凸显了潜在的风险。



## **30. Semantic-Aligned Adversarial Evolution Triangle for High-Transferability Vision-Language Attack**

高可移植性视觉语言攻击的语义对齐对抗进化三角 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02669v1) [paper-pdf](http://arxiv.org/pdf/2411.02669v1)

**Authors**: Xiaojun Jia, Sensen Gao, Qing Guo, Ke Ma, Yihao Huang, Simeng Qin, Yang Liu, Ivor Tsang Fellow, Xiaochun Cao

**Abstract**: Vision-language pre-training (VLP) models excel at interpreting both images and text but remain vulnerable to multimodal adversarial examples (AEs). Advancing the generation of transferable AEs, which succeed across unseen models, is key to developing more robust and practical VLP models. Previous approaches augment image-text pairs to enhance diversity within the adversarial example generation process, aiming to improve transferability by expanding the contrast space of image-text features. However, these methods focus solely on diversity around the current AEs, yielding limited gains in transferability. To address this issue, we propose to increase the diversity of AEs by leveraging the intersection regions along the adversarial trajectory during optimization. Specifically, we propose sampling from adversarial evolution triangles composed of clean, historical, and current adversarial examples to enhance adversarial diversity. We provide a theoretical analysis to demonstrate the effectiveness of the proposed adversarial evolution triangle. Moreover, we find that redundant inactive dimensions can dominate similarity calculations, distorting feature matching and making AEs model-dependent with reduced transferability. Hence, we propose to generate AEs in the semantic image-text feature contrast space, which can project the original feature space into a semantic corpus subspace. The proposed semantic-aligned subspace can reduce the image feature redundancy, thereby improving adversarial transferability. Extensive experiments across different datasets and models demonstrate that the proposed method can effectively improve adversarial transferability and outperform state-of-the-art adversarial attack methods. The code is released at https://github.com/jiaxiaojunQAQ/SA-AET.

摘要: 视觉语言预训练(VLP)模型在解释图像和文本方面表现出色，但仍然容易受到多模式对抗性例子(AEs)的影响。推动可转移实体的产生是开发更稳健和实用的VLP模型的关键，这种可转移实体在看不见的模型中取得了成功。以往的方法通过增加图文对来增强对抗性实例生成过程中的多样性，旨在通过扩展图文特征的对比度空间来提高可转移性。然而，这些方法只关注当前企业的多样性，在可转让性方面的收益有限。为了解决这一问题，我们建议在优化过程中利用对抗性轨迹上的交集区域来增加AEs的多样性。具体地说，我们建议从由干净的、历史的和当前的对抗性例子组成的对抗性进化三角形中进行采样，以增强对抗性多样性。我们提供了一个理论分析，以证明所提出的对抗性进化三角的有效性。此外，我们发现冗余的非活动维度会主导相似性计算，扭曲特征匹配，并使AEs依赖于模型，降低了可转移性。因此，我们提出在语义图文特征对比空间中生成AEs，它可以将原始特征空间投影到语义语料库的子空间。提出的语义对齐子空间可以减少图像特征的冗余度，从而提高对抗性转移能力。在不同的数据集和模型上进行的大量实验表明，该方法可以有效地提高对抗攻击的可转移性，并优于最新的对抗攻击方法。该代码在https://github.com/jiaxiaojunQAQ/SA-AET.上发布



## **31. Class-Conditioned Transformation for Enhanced Robust Image Classification**

用于增强鲁棒图像分类的类别条件变换 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.15409v2) [paper-pdf](http://arxiv.org/pdf/2303.15409v2)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex M. Bronstein

**Abstract**: Robust classification methods predominantly concentrate on algorithms that address a specific threat model, resulting in ineffective defenses against other threat models. Real-world applications are exposed to this vulnerability, as malicious attackers might exploit alternative threat models. In this work, we propose a novel test-time threat model agnostic algorithm that enhances Adversarial-Trained (AT) models. Our method operates through COnditional image transformation and DIstance-based Prediction (CODIP) and includes two main steps: First, we transform the input image into each dataset class, where the input image might be either clean or attacked. Next, we make a prediction based on the shortest transformed distance. The conditional transformation utilizes the perceptually aligned gradients property possessed by AT models and, as a result, eliminates the need for additional models or additional training. Moreover, it allows users to choose the desired balance between clean and robust accuracy without training. The proposed method achieves state-of-the-art results demonstrated through extensive experiments on various models, AT methods, datasets, and attack types. Notably, applying CODIP leads to substantial robust accuracy improvement of up to $+23\%$, $+20\%$, $+26\%$, and $+22\%$ on CIFAR10, CIFAR100, ImageNet and Flowers datasets, respectively.

摘要: 稳健的分类方法主要集中在处理特定威胁模型的算法上，导致对其他威胁模型的防御无效。现实世界中的应用程序会暴露于此漏洞，因为恶意攻击者可能会利用其他威胁模型。在这项工作中，我们提出了一种新的测试时间威胁模型不可知算法，该算法增强了对手训练(AT)模型。该方法通过条件图像变换和基于距离的预测(CODIP)进行操作，主要包括两个步骤：首先，将输入图像转换为每个数据集类，其中输入图像可能是干净的，也可能是被攻击的。接下来，我们基于最短变换距离进行预测。条件变换利用了AT模型所具有的感知对齐的梯度特性，因此不需要额外的模型或额外的训练。此外，它允许用户在干净和稳健的精确度之间选择所需的平衡，而无需培训。通过在各种模型、AT方法、数据集和攻击类型上的大量实验，该方法获得了最先进的结果。值得注意的是，在CIFAR10、CIFAR100、ImageNet和Flowers数据集上，应用CODIP可以显著提高精度，分别达到+2 3$、$+2 0$、$+2 6$和$+2 2$。



## **32. Attacking Vision-Language Computer Agents via Pop-ups**

通过弹出窗口攻击视觉语言计算机代理 cs.CL

10 pages, preprint

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02391v1) [paper-pdf](http://arxiv.org/pdf/2411.02391v1)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.

摘要: 由大视觉和语言模型(VLM)驱动的自主代理在完成日常计算机任务方面表现出了巨大的潜力，例如浏览网页预订旅行和操作桌面软件，这需要代理理解这些界面。尽管这样的视觉输入越来越多地集成到代理应用程序中，但围绕它们存在哪些类型的风险和攻击仍不清楚。在这项工作中，我们证明了VLM代理可以很容易地受到一组精心设计的敌意弹出窗口的攻击，人类用户通常会识别并忽略这些弹出窗口。这种干扰会导致工程师单击这些弹出窗口，而不是像往常一样执行任务。将这些弹出窗口集成到OSWorld和VisualWebArena等现有代理测试环境中，攻击成功率(代理单击弹出窗口的频率)平均为86%，任务成功率降低47%。基本的防御技术，如要求代理忽略弹出窗口或包括广告通知，对攻击无效。



## **33. Swiper: a new paradigm for efficient weighted distributed protocols**

Swiper：高效加权分布式协议的新范式 cs.DC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2307.15561v2) [paper-pdf](http://arxiv.org/pdf/2307.15561v2)

**Authors**: Andrei Tonkikh, Luciano Freitas

**Abstract**: The majority of fault-tolerant distributed algorithms are designed assuming a nominal corruption model, in which at most a fraction $f_n$ of parties can be corrupted by the adversary. However, due to the infamous Sybil attack, nominal models are not sufficient to express the trust assumptions in open (i.e., permissionless) settings. Instead, permissionless systems typically operate in a weighted model, where each participant is associated with a weight and the adversary can corrupt a set of parties holding at most a fraction $f_w$ of the total weight.   In this paper, we suggest a simple way to transform a large class of protocols designed for the nominal model into the weighted model. To this end, we formalize and solve three novel optimization problems, which we collectively call the weight reduction problems, that allow us to map large real weights into small integer weights while preserving the properties necessary for the correctness of the protocols. In all cases, we manage to keep the sum of the integer weights to be at most linear in the number of parties, resulting in extremely efficient protocols for the weighted model. Moreover, we demonstrate that, on weight distributions that emerge in practice, the sum of the integer weights tends to be far from the theoretical worst case and, sometimes, even smaller than the number of participants.   While, for some protocols, our transformation requires an arbitrarily small reduction in resilience (i.e., $f_w = f_n - \epsilon$), surprisingly, for many important problems, we manage to obtain weighted solutions with the same resilience ($f_w = f_n$) as nominal ones. Notable examples include erasure-coded distributed storage and broadcast protocols, verifiable secret sharing, and asynchronous consensus.

摘要: 大多数容错分布式算法的设计都假设了一个名义上的腐败模型，在该模型中，至多只有$f_n$的参与者可以被对手破坏。然而，由于臭名昭著的Sybil攻击，名义模型不足以表达开放(即无许可)环境下的信任假设。取而代之的是，未经许可的系统通常在加权模型中运行，其中每个参与者与一个权重相关联，并且对手可以破坏至多持有总权重的一小部分$f_w$的一组当事人。在本文中，我们提出了一种简单的方法，将为标称模型设计的一大类协议转换为加权模型。为此，我们形式化并解决了三个新的优化问题，我们统称为重量减轻问题，它们允许我们将大的实数权重映射为小的整数权重，同时保持协议正确性所必需的性质。在所有情况下，我们设法保持整数权重之和在参与方数量中最多是线性的，从而产生用于加权模型的极其高效的协议。此外，我们证明，在实际出现的权重分布上，整数权重的总和往往远离理论上最坏的情况，有时甚至比参与者的数量更少。虽然对于某些协议，我们的变换需要任意小的弹性降低(即$f_w=f_n-\epsilon$)，但令人惊讶的是，对于许多重要问题，我们设法获得具有与名义问题相同的弹性($f_w=f_n$)的加权解。值得注意的例子包括擦除编码的分布式存储和广播协议、可验证的秘密共享和异步共识。



## **34. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2409.13174v2) [paper-pdf](http://arxiv.org/pdf/2409.13174v2)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompts, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable Analyses of how VLAMs respond to different physical security threats. Our project page is in this link: https://chaducheng.github.io/Manipulat-Facing-Threats/.

摘要: 最近，在多模式大语言模型(MLLM)的推动下，视觉语言动作模型(VLAM)被提出以在机器人操作任务的开放词汇场景中实现更好的性能。由于操作任务涉及与物理世界的直接交互，因此确保该任务执行过程中的健壮性和安全性始终是一个非常关键的问题。本文通过综合当前MLLMS的安全研究现状和物理世界中操纵任务的具体应用场景，对VLAMS在面临潜在物理威胁的情况下进行综合评估。具体地说，我们提出了物理脆弱性评估管道(PVEP)，它可以结合尽可能多的视觉通道物理威胁来评估VLAMS的物理健壮性。PVEP中的物理威胁具体包括分发外、基于排版的视觉提示和对抗性补丁攻击。通过比较VLAM在受到攻击前后的性能波动，我们对VLAM如何应对不同的物理安全威胁提供了一般性的分析。我们的项目页面位于以下链接：https://chaducheng.github.io/Manipulat-Facing-Threats/.



## **35. Alignment-Based Adversarial Training (ABAT) for Improving the Robustness and Accuracy of EEG-Based BCIs**

基于对齐的对抗训练（ABAT）用于提高基于脑电的BCI的稳健性和准确性 cs.HC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02094v1) [paper-pdf](http://arxiv.org/pdf/2411.02094v1)

**Authors**: Xiaoqing Chen, Ziwei Wang, Dongrui Wu

**Abstract**: Machine learning has achieved great success in electroencephalogram (EEG) based brain-computer interfaces (BCIs). Most existing BCI studies focused on improving the decoding accuracy, with only a few considering the adversarial security. Although many adversarial defense approaches have been proposed in other application domains such as computer vision, previous research showed that their direct extensions to BCIs degrade the classification accuracy on benign samples. This phenomenon greatly affects the applicability of adversarial defense approaches to EEG-based BCIs. To mitigate this problem, we propose alignment-based adversarial training (ABAT), which performs EEG data alignment before adversarial training. Data alignment aligns EEG trials from different domains to reduce their distribution discrepancies, and adversarial training further robustifies the classification boundary. The integration of data alignment and adversarial training can make the trained EEG classifiers simultaneously more accurate and more robust. Experiments on five EEG datasets from two different BCI paradigms (motor imagery classification, and event related potential recognition), three convolutional neural network classifiers (EEGNet, ShallowCNN and DeepCNN) and three different experimental settings (offline within-subject cross-block/-session classification, online cross-session classification, and pre-trained classifiers) demonstrated its effectiveness. It is very intriguing that adversarial attacks, which are usually used to damage BCI systems, can be used in ABAT to simultaneously improve the model accuracy and robustness.

摘要: 机器学习在基于脑电(EEG)的脑机接口(BCI)领域取得了巨大的成功。现有的脑机接口研究大多集中在提高译码精度上，很少考虑对抗性安全性。虽然在计算机视觉等其他应用领域已经提出了许多对抗防御方法，但以往的研究表明，这些方法直接扩展到BCI会降低对良性样本的分类精度。这一现象极大地影响了对抗性防御方法对基于脑电信号的脑机接口的适用性。为了缓解这一问题，我们提出了基于对齐的对抗性训练(ABAT)，它在对抗性训练之前进行脑电数据对齐。数据对齐将来自不同领域的脑电试验对齐以减少它们的分布差异，而对抗性训练进一步强化了分类边界。将数据对齐和对抗性训练相结合，可以同时使训练后的脑电分类器更准确、更健壮。在两种不同的BCI范例(运动图像分类和事件相关电位识别)、三种卷积神经网络分类器(EEGNet、ShallowCNN和DeepCNN)以及三种不同的实验设置(离线内跨块/会话分类、在线跨会话分类和预先训练的分类器)上的实验证明了该方法的有效性。非常耐人寻味的是，通常用于破坏BCI系统的对抗性攻击可以同时用于提高模型的精度和鲁棒性。



## **36. Exploiting LLM Quantization**

利用LLM量化 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2405.18137v2) [paper-pdf](http://arxiv.org/pdf/2405.18137v2)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.

摘要: 量化利用较低精度的权重来减少大型语言模型(LLM)的内存使用，这是在商用硬件上部署LLM的关键技术。虽然LLM量化对效用的影响已经被广泛研究，但这项工作首次从安全的角度研究了它的不利影响。我们发现，广泛使用的量化方法可以被利用来产生有害的量化LLM，即使全精度对应的看起来是良性的，潜在地诱骗用户部署恶意量化模型。我们使用一个三阶段攻击框架演示了这一威胁：(I)首先，我们通过对敌方任务的微调来获得恶意LLM；(Ii)接下来，我们量化恶意模型，并计算映射到相同量化模型的所有全精度模型的约束；(Iii)最后，使用投影梯度下降，我们在确保其权重满足步骤(Ii)中计算的约束的同时，从全精度模型中排除有毒行为。这一过程导致LLM完全精确地表现出良性行为，但当量化时，它遵循在步骤(I)中注入的对抗性行为。我们通过实验演示了这种攻击在三种不同场景中的可行性和严重性：易受攻击的代码生成、内容注入和过度拒绝攻击。在实践中，对手可能会在LLM社区中心(如拥抱脸)上托管产生的全精度模型，使数百万用户面临在他们的设备上部署其恶意量化版本的威胁。



## **37. DeSparsify: Adversarial Attack Against Token Sparsification Mechanisms in Vision Transformers**

DeSparsify：针对Vision Transformers中代币稀疏机制的对抗攻击 cs.CV

18 pages, 6 figures

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2402.02554v2) [paper-pdf](http://arxiv.org/pdf/2402.02554v2)

**Authors**: Oryan Yehezkel, Alon Zolfi, Amit Baras, Yuval Elovici, Asaf Shabtai

**Abstract**: Vision transformers have contributed greatly to advancements in the computer vision domain, demonstrating state-of-the-art performance in diverse tasks (e.g., image classification, object detection). However, their high computational requirements grow quadratically with the number of tokens used. Token sparsification mechanisms have been proposed to address this issue. These mechanisms employ an input-dependent strategy, in which uninformative tokens are discarded from the computation pipeline, improving the model's efficiency. However, their dynamism and average-case assumption makes them vulnerable to a new threat vector - carefully crafted adversarial examples capable of fooling the sparsification mechanism, resulting in worst-case performance. In this paper, we present DeSparsify, an attack targeting the availability of vision transformers that use token sparsification mechanisms. The attack aims to exhaust the operating system's resources, while maintaining its stealthiness. Our evaluation demonstrates the attack's effectiveness on three token sparsification mechanisms and examines the attack's transferability between them and its effect on the GPU resources. To mitigate the impact of the attack, we propose various countermeasures.

摘要: 视觉转换器为计算机视觉领域的进步做出了巨大贡献，在各种任务(例如，图像分类、目标检测)中展示了最先进的性能。然而，它们的高计算需求随着使用的令牌数量的增加而呈二次曲线增长。为了解决这个问题，已经提出了令牌稀疏机制。这些机制采用了一种依赖输入的策略，在该策略中，没有提供信息的令牌被从计算流水线中丢弃，从而提高了模型的效率。然而，它们的动态性和平均情况假设使它们容易受到新的威胁向量的攻击--精心设计的敌意示例能够愚弄稀疏机制，导致最差情况下的性能。在本文中，我们提出了DeSparsify，一种针对使用令牌稀疏机制的视觉转换器的可用性的攻击。这次攻击旨在耗尽操作系统的资源，同时保持其隐蔽性。我们的评估证明了攻击在三种令牌稀疏机制上的有效性，并检查了攻击在它们之间的可传递性及其对GPU资源的影响。为了减轻攻击的影响，我们提出了各种对策。



## **38. LiDAttack: Robust Black-box Attack on LiDAR-based Object Detection**

LiDAttack：对基于LiDART的对象检测的鲁棒黑匣子攻击 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01889v1) [paper-pdf](http://arxiv.org/pdf/2411.01889v1)

**Authors**: Jinyin Chen, Danxin Liao, Sheng Xiang, Haibin Zheng

**Abstract**: Since DNN is vulnerable to carefully crafted adversarial examples, adversarial attack on LiDAR sensors have been extensively studied. We introduce a robust black-box attack dubbed LiDAttack. It utilizes a genetic algorithm with a simulated annealing strategy to strictly limit the location and number of perturbation points, achieving a stealthy and effective attack. And it simulates scanning deviations, allowing it to adapt to dynamic changes in real world scenario variations. Extensive experiments are conducted on 3 datasets (i.e., KITTI, nuScenes, and self-constructed data) with 3 dominant object detection models (i.e., PointRCNN, PointPillar, and PV-RCNN++). The results reveal the efficiency of the LiDAttack when targeting a wide range of object detection models, with an attack success rate (ASR) up to 90%.

摘要: 由于DNN容易受到精心设计的对抗示例的影响，因此对LiDART传感器的对抗攻击已经得到了广泛的研究。我们引入了一种名为LiDAttack的强大黑匣子攻击。它利用遗传算法和模拟退变策略来严格限制扰动点的位置和数量，实现隐蔽有效的攻击。它模拟扫描偏差，使其能够适应现实世界场景变化的动态变化。对3个数据集进行了广泛的实验（即，KITTI、nuScenes和自构建数据），具有3个主要对象检测模型（即PointRCNN、PointPillar和PV-RCNN++）。结果揭示了LiDAttack在针对广泛的对象检测模型时的效率，攻击成功率（ASB）高达90%。



## **39. Learning predictable and robust neural representations by straightening image sequences**

通过拉直图像序列学习可预测且鲁棒的神经表示 cs.CV

Accepted at NeurIPS 2024

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01777v1) [paper-pdf](http://arxiv.org/pdf/2411.01777v1)

**Authors**: Xueyan Niu, Cristina Savin, Eero P. Simoncelli

**Abstract**: Prediction is a fundamental capability of all living organisms, and has been proposed as an objective for learning sensory representations. Recent work demonstrates that in primate visual systems, prediction is facilitated by neural representations that follow straighter temporal trajectories than their initial photoreceptor encoding, which allows for prediction by linear extrapolation. Inspired by these experimental findings, we develop a self-supervised learning (SSL) objective that explicitly quantifies and promotes straightening. We demonstrate the power of this objective in training deep feedforward neural networks on smoothly-rendered synthetic image sequences that mimic commonly-occurring properties of natural videos. The learned model contains neural embeddings that are predictive, but also factorize the geometric, photometric, and semantic attributes of objects. The representations also prove more robust to noise and adversarial attacks compared to previous SSL methods that optimize for invariance to random augmentations. Moreover, these beneficial properties can be transferred to other training procedures by using the straightening objective as a regularizer, suggesting a broader utility for straightening as a principle for robust unsupervised learning.

摘要: 预测是所有生物体的一种基本能力，并被认为是学习感觉表征的目标。最近的工作表明，在灵长类视觉系统中，预测是由神经表征促进的，这些神经表征遵循比其初始光感受器编码更直接的时间轨迹，这允许通过线性外推进行预测。受这些实验结果的启发，我们制定了一个明确量化和促进矫正的自我监督学习(SSL)目标。我们展示了这一目标在平滑渲染的合成图像序列上训练深度前馈神经网络的能力，这些合成图像序列模仿自然视频的常见属性。学习的模型包含神经嵌入，这些神经嵌入是预测性的，但也分解了对象的几何、光度和语义属性。与之前的针对随机增加的不变性进行优化的SSL方法相比，该表示还被证明对噪声和敌意攻击更健壮。此外，通过使用校直目标作为正则化规则，这些有益的属性可以转移到其他训练过程中，这表明校直作为稳健的无监督学习的原则具有更广泛的实用性。



## **40. Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges**

医学图像分析的对抗性攻击与防御调查：方法与挑战 eess.IV

Accepted by ACM Computing Surveys (CSUR) (DOI:  https://doi.org/10.1145/3702638)

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.14133v2) [paper-pdf](http://arxiv.org/pdf/2303.14133v2)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attacks and defenses for medical image analysis with a systematic taxonomy in terms of the application scenario. We also provide a unified framework for different types of adversarial attack and defense methods in the context of medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions. Code is available on \href{https://github.com/tomvii/Adv_MIA}{\color{red}{GitHub}}.

摘要: 深度学习技术在计算机辅助医学图像分析中取得了优异的性能，但仍然容易受到潜移默化的对抗性攻击，导致临床实践中潜在的误诊。相反，近年来在防御深度医疗诊断系统中这些量身定做的对抗性例子方面也取得了显著进展。在这篇论述中，我们对医学图像分析中对抗性攻击和防御的最新进展进行了全面的综述，并根据应用场景对其进行了系统的分类。我们还为医学图像分析中不同类型的对抗性攻击和防御方法提供了一个统一的框架。为了进行公平的比较，我们建立了一个新的基准，用于在不同场景下通过对抗性训练获得对抗性健壮的医疗诊断模型。据我们所知，这是第一份对反面稳健的医疗诊断模型进行彻底评估的调查报告。通过对定性和定量结果的分析，我们对当前医学图像分析系统中对抗性攻击和防御的挑战进行了详细的讨论，以阐明未来的研究方向。代码可在\href{https://github.com/tomvii/Adv_MIA}{\color{red}{GitHub}}.上找到



## **41. A General Recipe for Contractive Graph Neural Networks -- Technical Report**

压缩图神经网络的通用配方--技术报告 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01717v1) [paper-pdf](http://arxiv.org/pdf/2411.01717v1)

**Authors**: Maya Bechler-Speicher, Moshe Eliasof

**Abstract**: Graph Neural Networks (GNNs) have gained significant popularity for learning representations of graph-structured data due to their expressive power and scalability. However, despite their success in domains such as social network analysis, recommendation systems, and bioinformatics, GNNs often face challenges related to stability, generalization, and robustness to noise and adversarial attacks. Regularization techniques have shown promise in addressing these challenges by controlling model complexity and improving robustness. Building on recent advancements in contractive GNN architectures, this paper presents a novel method for inducing contractive behavior in any GNN through SVD regularization. By deriving a sufficient condition for contractiveness in the update step and applying constraints on network parameters, we demonstrate the impact of SVD regularization on the Lipschitz constant of GNNs. Our findings highlight the role of SVD regularization in enhancing the stability and generalization of GNNs, contributing to the development of more robust graph-based learning algorithms dynamics.

摘要: 图神经网络(GNN)由于其表达能力和可扩展性，在学习图结构数据的表示方面受到了极大的欢迎。然而，尽管GNN在社会网络分析、推荐系统和生物信息学等领域取得了成功，但它们经常面临与稳定性、泛化以及对噪声和对手攻击的健壮性相关的挑战。正则化技术通过控制模型复杂性和提高稳健性，在解决这些挑战方面表现出了希望。基于压缩GNN结构的最新进展，本文提出了一种通过奇异值分解正则化来诱导任意GNN收缩行为的新方法。通过推导更新过程中收敛的充分条件和对网络参数的约束，我们证明了奇异值分解正则化对GNN的Lipschitz常数的影响。我们的发现突出了奇异值分解正则化在增强GNN的稳定性和泛化方面的作用，有助于开发更健壮的基于图的学习算法动态。



## **42. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

UniGuard：为多模式大型语言模型越狱攻击建立通用安全护栏 cs.CL

14 pages

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01703v1) [paper-pdf](http://arxiv.org/pdf/2411.01703v1)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but are vulnerable to multimodal jailbreak attacks, where adversaries meticulously craft inputs to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard is trained such that the likelihood of generating harmful responses in a toxic corpus is minimized, and can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities and attack strategies. It demonstrates impressive generalizability across multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4, MiniGPT-4, and InstructBLIP, thereby broadening the scope of our solution.

摘要: 多模式大型语言模型（MLLM）彻底改变了视觉语言理解，但很容易受到多模式越狱攻击，对手精心设计输入以引发有害或不当的反应。我们提出了UniGuard，这是一种新型的多模式安全护栏，它联合考虑了单模式和跨模式有害信号。UniGuard经过训练，可以最大限度地降低在有毒的数据库中生成有害响应的可能性，并且可以以最小的计算成本无缝地应用于推理期间的任何输入提示。大量实验证明了UniGuard在多种模式和攻击策略中的通用性。它在多个最先进的MLLM（包括LLaVA、Gemini Pro、GPT-4、MiniGPT-4和DirecectBLIP）上展示了令人印象深刻的通用性，从而扩大了我们解决方案的范围。



## **43. Building the Self-Improvement Loop: Error Detection and Correction in Goal-Oriented Semantic Communications**

建立自我改进循环：面向目标的语义通信中的错误检测和纠正 cs.NI

7 pages, 8 figures, this paper has been accepted for publication in  IEEE CSCN 2024

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01544v1) [paper-pdf](http://arxiv.org/pdf/2411.01544v1)

**Authors**: Peizheng Li, Xinyi Lin, Adnan Aijaz

**Abstract**: Error detection and correction are essential for ensuring robust and reliable operation in modern communication systems, particularly in complex transmission environments. However, discussions on these topics have largely been overlooked in semantic communication (SemCom), which focuses on transmitting meaning rather than symbols, leading to significant improvements in communication efficiency. Despite these advantages, semantic errors -- stemming from discrepancies between transmitted and received meanings -- present a major challenge to system reliability. This paper addresses this gap by proposing a comprehensive framework for detecting and correcting semantic errors in SemCom systems. We formally define semantic error, detection, and correction mechanisms, and identify key sources of semantic errors. To address these challenges, we develop a Gaussian process (GP)-based method for latent space monitoring to detect errors, alongside a human-in-the-loop reinforcement learning (HITL-RL) approach to optimize semantic model configurations using user feedback. Experimental results validate the effectiveness of the proposed methods in mitigating semantic errors under various conditions, including adversarial attacks, input feature changes, physical channel variations, and user preference shifts. This work lays the foundation for more reliable and adaptive SemCom systems with robust semantic error management techniques.

摘要: 在现代通信系统中，尤其是在复杂的传输环境中，错误检测和纠错对于确保健壮和可靠的操作是必不可少的。然而，在注重传递意义而不是符号的语义交际(SemCom)中，对这些话题的讨论在很大程度上被忽视了，导致交际效率的显著提高。尽管有这些优点，语义错误--源于发送和接收的含义之间的差异--对系统可靠性构成了重大挑战。本文提出了一个检测和纠正SemCom系统中的语义错误的综合框架，以解决这一差距。我们正式定义了语义错误、检测和纠正机制，并确定了语义错误的关键来源。为了应对这些挑战，我们开发了一种基于高斯过程(GP)的潜在空间监测方法来检测错误，并结合人在环强化学习(HITL-RL)方法来使用用户反馈来优化语义模型配置。实验结果验证了该方法在对抗性攻击、输入特征变化、物理通道变化和用户偏好变化等多种情况下的有效性。这项工作为具有健壮的语义错误管理技术的SemCom系统的可靠性和自适应性奠定了基础。



## **44. Privacy-Preserving Customer Churn Prediction Model in the Context of Telecommunication Industry**

电信行业背景下保护隐私的客户流失预测模型 cs.LG

26 pages, 14 tables, 13 figures

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01447v1) [paper-pdf](http://arxiv.org/pdf/2411.01447v1)

**Authors**: Joydeb Kumar Sana, M Sohel Rahman, M Saifur Rahman

**Abstract**: Data is the main fuel of a successful machine learning model. A dataset may contain sensitive individual records e.g. personal health records, financial data, industrial information, etc. Training a model using this sensitive data has become a new privacy concern when someone uses third-party cloud computing. Trained models also suffer privacy attacks which leads to the leaking of sensitive information of the training data. This study is conducted to preserve the privacy of training data in the context of customer churn prediction modeling for the telecommunications industry (TCI). In this work, we propose a framework for privacy-preserving customer churn prediction (PPCCP) model in the cloud environment. We have proposed a novel approach which is a combination of Generative Adversarial Networks (GANs) and adaptive Weight-of-Evidence (aWOE). Synthetic data is generated from GANs, and aWOE is applied on the synthetic training dataset before feeding the data to the classification algorithms. Our experiments were carried out using eight different machine learning (ML) classifiers on three openly accessible datasets from the telecommunication sector. We then evaluated the performance using six commonly employed evaluation metrics. In addition to presenting a data privacy analysis, we also performed a statistical significance test. The training and prediction processes achieve data privacy and the prediction classifiers achieve high prediction performance (87.1\% in terms of F-Measure for GANs-aWOE based Na\"{\i}ve Bayes model). In contrast to earlier studies, our suggested approach demonstrates a prediction enhancement of up to 28.9\% and 27.9\% in terms of accuracy and F-measure, respectively.

摘要: 数据是成功的机器学习模型的主要燃料。数据集可能包含敏感的个人记录，如个人健康记录、金融数据、行业信息等。当有人使用第三方云计算时，使用这些敏感数据训练模型已成为一个新的隐私问题。训练后的模型还会遭受隐私攻击，导致训练数据的敏感信息泄露。这项研究是为了在电信行业(TCI)的客户流失预测建模的背景下保护训练数据的隐私。在这项工作中，我们提出了一种云环境下隐私保护的客户流失预测(PPCCP)模型框架。我们提出了一种新的方法，它是生成性对抗网络(GANS)和自适应证据权重(AWOE)的结合。从GANS生成合成数据，并在将数据提供给分类算法之前对合成训练数据集应用aWOE。我们的实验是在来自电信部门的三个开放可访问的数据集上使用八个不同的机器学习(ML)分类器进行的。然后，我们使用六个常用的评估指标来评估性能。除了提供数据隐私分析外，我们还执行了统计显著性测试。训练和预测过程实现了数据保密性，预测分类器获得了较高的预测性能(对于基于Gans-aWOE的Na2VeBayes模型的F度量为87.1%)，与以前的研究相比，我们的方法在预测准确率和F度量方面分别提高了28.9%和27.9%。



## **45. Enhancing Cyber-Resilience in Integrated Energy System Scheduling with Demand Response Using Deep Reinforcement Learning**

使用深度强化学习增强具有需求响应的综合能源系统调度中的网络弹性 eess.SY

Accepted by Applied Energy, Manuscript ID: APEN-D-24-03080

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2311.17941v2) [paper-pdf](http://arxiv.org/pdf/2311.17941v2)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen, Mohammad Shahidehpor

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, the state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack, incorporating cyber-attacks as adversaries directly into the scheduling process. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy, integrating adversarial training into the learning process to against cyber-attacks. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10% improvement in economic performance under cyber-attack scenarios.

摘要: 多能流优化调度是利用可再生能源、提高综合能源系统稳定性和经济性的有效方法。然而，工业企业稳定的需求供应面临着挑战，这些挑战来自资源和负载带来的不确定性，以及采用先进信息和通信技术的网络攻击的影响越来越大。针对这些挑战，提出了一种基于状态对抗性深度强化学习(DRL)的集成需求响应(IDR)支持的IES的无模型弹性调度方法。该方法设计了一个IDR程序来研究电-气-热柔性负荷的相互作用能力。此外，状态-对抗性马尔可夫决策过程(SA-MDP)模型刻画了网络攻击下IES的能量调度问题，将网络攻击作为对手直接纳入调度过程。针对网络攻击对调度策略的影响，提出了一种状态对抗性软行为者-批评者(SA-SAC)算法，将对抗性训练融入到学习过程中来对抗网络攻击。仿真结果表明，该方法能够很好地处理资源和负荷带来的不确定性，减轻网络攻击对调度策略的影响，保证各种能源的稳定需求。此外，该方法还表现出了对网络攻击的恢复能力。与原来的软演员-批评者(SAC)算法相比，该算法在网络攻击场景下的经济性能提高了10%。



## **46. High-Frequency Anti-DreamBooth: Robust Defense against Personalized Image Synthesis**

高频反DreamBooth：针对个性化图像合成的强大防御 cs.CV

ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond. Our  code is available at https://github.com/mti-lab/HF-ADB

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2409.08167v3) [paper-pdf](http://arxiv.org/pdf/2409.08167v3)

**Authors**: Takuto Onikubo, Yusuke Matsui

**Abstract**: Recently, text-to-image generative models have been misused to create unauthorized malicious images of individuals, posing a growing social problem. Previous solutions, such as Anti-DreamBooth, add adversarial noise to images to protect them from being used as training data for malicious generation. However, we found that the adversarial noise can be removed by adversarial purification methods such as DiffPure. Therefore, we propose a new adversarial attack method that adds strong perturbation on the high-frequency areas of images to make it more robust to adversarial purification. Our experiment showed that the adversarial images retained noise even after adversarial purification, hindering malicious image generation.

摘要: 最近，文本到图像的生成模型被滥用来创建未经授权的恶意个人图像，造成了日益严重的社会问题。以前的解决方案（例如Anti-DreamBooth）会向图像添加对抗性噪音，以保护它们不被用作恶意生成的训练数据。然而，我们发现对抗性噪音可以通过迪夫Pure等对抗性净化方法去除。因此，我们提出了一种新的对抗性攻击方法，该方法在图像的高频区域添加强扰动，使其对对抗性净化更稳健。我们的实验表明，即使在对抗净化之后，对抗图像也会保留噪音，从而阻碍了恶意图像的生成。



## **47. AED: An black-box NLP classifier model attacker**

AED：黑匣子NLP分类器模型攻击者 cs.LG

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2112.11660v4) [paper-pdf](http://arxiv.org/pdf/2112.11660v4)

**Authors**: Yueyang Liu, Yan Huang, Zhipeng Cai

**Abstract**: Deep Neural Networks (DNNs) have been successful in solving real-world tasks in domains such as connected and automated vehicles, disease, and job hiring. However, their implications are far-reaching in critical application areas. Hence, there is a growing concern regarding the potential bias and robustness of these DNN models. A transparency and robust model is always demanded in high-stakes domains where reliability and safety are enforced, such as healthcare and finance. While most studies have focused on adversarial image attack scenarios, fewer studies have investigated the robustness of DNN models in natural language processing (NLP) due to their adversarial samples are difficult to generate. To address this gap, we propose a word-level NLP classifier attack model called "AED," which stands for Attention mechanism enabled post-model Explanation with Density peaks clustering algorithm for synonyms search and substitution. AED aims to test the robustness of NLP DNN models by interpretability their weaknesses and exploring alternative ways to optimize them. By identifying vulnerabilities and providing explanations, AED can help improve the reliability and safety of DNN models in critical application areas such as healthcare and automated transportation. Our experiment results demonstrate that compared with other existing models, AED can effectively generate adversarial examples that can fool the victim model while maintaining the original meaning of the input.

摘要: 深度神经网络(DNN)已经成功地解决了联网和自动化车辆、疾病和就业等领域的现实世界任务。然而，它们对关键应用领域的影响是深远的。因此，人们越来越关注这些DNN模型的潜在偏差和稳健性。在要求可靠性和安全性的高风险领域，如医疗保健和金融，总是需要透明和健壮的模型。虽然大多数研究都集中在对抗性图像攻击场景上，但很少有人研究DNN模型在自然语言处理(NLP)中的稳健性，因为它们的对抗性样本很难生成。为了解决这一问题，我们提出了一种词级NLP分类器攻击模型AED，它代表了注意力机制启用的模型后解释，并使用密度峰值聚类算法进行同义词搜索和替换。AED的目的是测试NLP DNN模型的健壮性，通过解释它们的弱点并探索其他方法来优化它们。通过识别漏洞和提供解释，AED可以帮助提高DNN模型在医疗保健和自动化交通等关键应用领域的可靠性和安全性。实验结果表明，与其他已有的模型相比，AED能够有效地生成能够愚弄受害者模型的对抗性实例，同时保持输入的原始含义。



## **48. What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks**

Prettts越狱LLMS有哪些功能？调查攻击背后的机制 cs.CR

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2411.03343v1) [paper-pdf](http://arxiv.org/pdf/2411.03343v1)

**Authors**: Nathalie Maria Kirch, Severin Field, Stephen Casper

**Abstract**: While `jailbreaks' have been central to research on the safety and reliability of LLMs (large language models), the underlying mechanisms behind these attacks are not well understood. Some prior works have used linear methods to analyze jailbreak prompts or model refusal. Here, however, we compare linear and nonlinear methods to study the features in prompts that contribute to successful jailbreaks. We do this by probing for jailbreak success based only on the portions of the latent representations corresponding to prompt tokens. First, we introduce a dataset of 10,800 jailbreak attempts from 35 attack methods. We then show that different jailbreaking methods work via different nonlinear features in prompts. Specifically, we find that while probes can distinguish between successful and unsuccessful jailbreaking prompts with a high degree of accuracy, they often transfer poorly to held-out attack methods. We also show that nonlinear probes can be used to mechanistically jailbreak the LLM by guiding the design of adversarial latent perturbations. These mechanistic jailbreaks are able to jailbreak Gemma-7B-IT more reliably than 34 of the 35 techniques that it was trained on. Ultimately, our results suggest that jailbreaks cannot be thoroughly understood in terms of universal or linear prompt features alone.

摘要: 虽然“越狱”一直是研究大型语言模型(LLMS)安全性和可靠性的核心，但这些攻击背后的潜在机制并未得到很好的理解。以前的一些工作已经使用线性方法来分析越狱提示或模型拒绝。然而，在这里，我们比较线性和非线性方法来研究有助于成功越狱的提示中的特征。我们通过仅基于与提示令牌相对应的潜在表示的部分来探测越狱成功来实现这一点。首先，我们介绍了一个来自35种攻击方法的10,800次越狱尝试的数据集。然后，我们展示了不同的越狱方法通过不同的提示中的非线性特征来工作。具体地说，我们发现，虽然探测器可以高度准确地区分成功和不成功的越狱提示，但它们往往很难转变为坚持攻击的方法。我们还表明，通过指导对抗性潜在扰动的设计，可以使用非线性探测器来机械地越狱LLM。这些机械式越狱技术能够更可靠地越狱Gema-7B-IT，而不是它所训练的35种技术中的34种。最终，我们的结果表明，仅从普遍的或线性的提示特征来看，不能彻底理解越狱。



## **49. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

了解和改进对抗性协作过滤以实现稳健推荐 cs.IR

To appear in NeurIPS 2024

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2410.22844v2) [paper-pdf](http://arxiv.org/pdf/2410.22844v2)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.

摘要: 对抗性协同过滤(ACF)通过对抗性训练将对抗性扰动应用于用户和项目嵌入，被广泛认为是提高协同过滤推荐系统对中毒攻击的稳健性的有效策略。此外，大量研究表明，与传统的推荐算法相比，自适应过滤算法也能提高推荐性能。尽管取得了这些经验上的成功，但关于ACF在性能和稳健性方面的有效性的理论理解仍然不清楚。为了弥补这一差距，在本文中，我们首先从理论上证明，在干净和有毒的数据环境下，与相同训练周期的传统CF相比，ACF可以获得更低的推荐误差。此外，通过建立ACF优化过程中推荐误差减少的界限，我们发现根据不同用户的嵌入尺度对不同用户应用个性化的扰动幅度可以进一步提高ACF的有效性。在这些理论理解的基础上，我们提出了个性化幅度对抗协同过滤(PamaCF)。大量实验表明，PamaCF在有效防御各种类型的中毒攻击的同时，显著提高了推荐性能。



## **50. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01084v1) [paper-pdf](http://arxiv.org/pdf/2411.01084v1)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or \textit{red-teamers}, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary \textit{string compositions}, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者（\textit{red-teamers}）采用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的\textit{字符串合成}，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合大量字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



