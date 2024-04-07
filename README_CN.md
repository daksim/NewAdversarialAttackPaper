# Latest Adversarial Attack Papers
**update at 2024-04-07 11:06:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Knowledge Distillation-Based Model Extraction Attack using Private Counterfactual Explanations**

基于知识蒸馏的私有反事实搜索模型抽取攻击 cs.LG

15 pages

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03348v1) [paper-pdf](http://arxiv.org/pdf/2404.03348v1)

**Authors**: Fatima Ezzeddine, Omran Ayoub, Silvia Giordano

**Abstract**: In recent years, there has been a notable increase in the deployment of machine learning (ML) models as services (MLaaS) across diverse production software applications. In parallel, explainable AI (XAI) continues to evolve, addressing the necessity for transparency and trustworthiness in ML models. XAI techniques aim to enhance the transparency of ML models by providing insights, in terms of the model's explanations, into their decision-making process. Simultaneously, some MLaaS platforms now offer explanations alongside the ML prediction outputs. This setup has elevated concerns regarding vulnerabilities in MLaaS, particularly in relation to privacy leakage attacks such as model extraction attacks (MEA). This is due to the fact that explanations can unveil insights about the inner workings of the model which could be exploited by malicious users. In this work, we focus on investigating how model explanations, particularly Generative adversarial networks (GANs)-based counterfactual explanations (CFs), can be exploited for performing MEA within the MLaaS platform. We also delve into assessing the effectiveness of incorporating differential privacy (DP) as a mitigation strategy. To this end, we first propose a novel MEA methodology based on Knowledge Distillation (KD) to enhance the efficiency of extracting a substitute model of a target model exploiting CFs. Then, we advise an approach for training CF generators incorporating DP to generate private CFs. We conduct thorough experimental evaluations on real-world datasets and demonstrate that our proposed KD-based MEA can yield a high-fidelity substitute model with reduced queries with respect to baseline approaches. Furthermore, our findings reveal that the inclusion of a privacy layer impacts the performance of the explainer, the quality of CFs, and results in a reduction in the MEA performance.

摘要: 近年来，机器学习(ML)模型即服务(MLaaS)在各种生产软件应用程序中的部署显著增加。同时，可解释人工智能(XAI)继续发展，解决了ML模型中透明度和可信性的必要性。XAI技术旨在通过提供对ML模型的解释方面的见解来提高ML模型的透明度。与此同时，一些MLaaS平台现在除了ML预测输出外，还提供解释。这一设置加剧了人们对MLaaS漏洞的担忧，特别是与隐私泄露攻击有关的漏洞，如模型提取攻击(MEA)。这是因为解释可以揭示该模型的内部工作原理，这可能会被恶意用户利用。在这项工作中，我们重点研究如何利用模型解释，特别是基于生成性对抗网络(GANS)的反事实解释(CFS)来在MLaaS平台上执行MEA。我们还深入评估了将差异隐私(DP)作为缓解策略的有效性。为此，我们首先提出了一种新的基于知识蒸馏(KD)的MEA方法，以提高利用CFS提取目标模型替代模型的效率。然后，我们提出了一种训练包含DP的CF生成器以生成私有CF的方法。我们在真实世界的数据集上进行了深入的实验评估，并证明了我们提出的基于KD的MEA可以产生高保真的替代模型，相对于基线方法减少了查询。此外，我们的研究结果还表明，隐私层的加入影响了解释者的表现，也影响了解释的质量，并导致了MEA表现的下降。



## **2. Meta Invariance Defense Towards Generalizable Robustness to Unknown Adversarial Attacks**

面向未知对抗攻击的广义鲁棒性的Meta不变性防御 cs.CV

Accepted by IEEE TPAMI in 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03340v1) [paper-pdf](http://arxiv.org/pdf/2404.03340v1)

**Authors**: Lei Zhang, Yuhang Zhou, Yi Yang, Xinbo Gao

**Abstract**: Despite providing high-performance solutions for computer vision tasks, the deep neural network (DNN) model has been proved to be extremely vulnerable to adversarial attacks. Current defense mainly focuses on the known attacks, but the adversarial robustness to the unknown attacks is seriously overlooked. Besides, commonly used adaptive learning and fine-tuning technique is unsuitable for adversarial defense since it is essentially a zero-shot problem when deployed. Thus, to tackle this challenge, we propose an attack-agnostic defense method named Meta Invariance Defense (MID). Specifically, various combinations of adversarial attacks are randomly sampled from a manually constructed Attacker Pool to constitute different defense tasks against unknown attacks, in which a student encoder is supervised by multi-consistency distillation to learn the attack-invariant features via a meta principle. The proposed MID has two merits: 1) Full distillation from pixel-, feature- and prediction-level between benign and adversarial samples facilitates the discovery of attack-invariance. 2) The model simultaneously achieves robustness to the imperceptible adversarial perturbations in high-level image classification and attack-suppression in low-level robust image regeneration. Theoretical and empirical studies on numerous benchmarks such as ImageNet verify the generalizable robustness and superiority of MID under various attacks.

摘要: 尽管深度神经网络(DNN)模型为计算机视觉任务提供了高性能的解决方案，但已被证明极易受到对手攻击。目前的防御主要针对已知攻击，而对未知攻击的对抗健壮性被严重忽视。此外，常用的自适应学习和微调技术不适合对抗性防御，因为它在部署时本质上是一个零命中问题。因此，为了应对这一挑战，我们提出了一种与攻击无关的防御方法，称为元不变性防御(MID)。具体地，从人工构建的攻击者池中随机抽取各种对抗性攻击组合，组成针对未知攻击的不同防御任务，其中学生编码者通过多一致性蒸馏来监督，通过元原则学习攻击不变特征。提出的MID算法有两个优点：1)良性样本和敌方样本之间像素级、特征级和预测级的充分提取有利于发现攻击不变性。2)该模型同时实现了在高层图像分类中对不可察觉的对抗性扰动的鲁棒性和在低级稳健图像再生中的攻击抑制。在ImageNet等众多基准测试上的理论和实证研究验证了MID在各种攻击下的泛化健壮性和优越性。



## **3. Learn What You Want to Unlearn: Unlearning Inversion Attacks against Machine Unlearning**

学习你想要忘记的东西：忘记学习反转攻击针对机器忘记学习 cs.CR

To Appear in the 45th IEEE Symposium on Security and Privacy, May  20-23, 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03233v1) [paper-pdf](http://arxiv.org/pdf/2404.03233v1)

**Authors**: Hongsheng Hu, Shuo Wang, Tian Dong, Minhui Xue

**Abstract**: Machine unlearning has become a promising solution for fulfilling the "right to be forgotten", under which individuals can request the deletion of their data from machine learning models. However, existing studies of machine unlearning mainly focus on the efficacy and efficiency of unlearning methods, while neglecting the investigation of the privacy vulnerability during the unlearning process. With two versions of a model available to an adversary, that is, the original model and the unlearned model, machine unlearning opens up a new attack surface. In this paper, we conduct the first investigation to understand the extent to which machine unlearning can leak the confidential content of the unlearned data. Specifically, under the Machine Learning as a Service setting, we propose unlearning inversion attacks that can reveal the feature and label information of an unlearned sample by only accessing the original and unlearned model. The effectiveness of the proposed unlearning inversion attacks is evaluated through extensive experiments on benchmark datasets across various model architectures and on both exact and approximate representative unlearning approaches. The experimental results indicate that the proposed attack can reveal the sensitive information of the unlearned data. As such, we identify three possible defenses that help to mitigate the proposed attacks, while at the cost of reducing the utility of the unlearned model. The study in this paper uncovers an underexplored gap between machine unlearning and the privacy of unlearned data, highlighting the need for the careful design of mechanisms for implementing unlearning without leaking the information of the unlearned data.

摘要: 机器遗忘已经成为一种很有前途的解决方案，可以实现“被遗忘的权利”，根据这种权利，个人可以请求从机器学习模型中删除他们的数据。然而，现有的机器遗忘研究主要集中在遗忘方法的有效性和效率上，而忽略了对遗忘过程中隐私漏洞的研究。由于一个模型有两个版本可供对手使用，即原始模型和未学习模型，机器遗忘打开了一个新的攻击面。在本文中，我们进行了第一次调查，以了解机器遗忘可以在多大程度上泄露未学习数据的机密内容。具体地说，在机器学习即服务的背景下，我们提出了遗忘反转攻击，只需访问原始的和未学习的模型，就可以揭示未学习样本的特征和标签信息。通过在不同模型体系结构的基准数据集上以及在精确和近似代表遗忘方法上的大量实验，评估了所提出的遗忘反转攻击的有效性。实验结果表明，该攻击能够泄露未学习数据的敏感信息。因此，我们确定了三种可能的防御措施，它们有助于减轻拟议的攻击，同时代价是减少未学习模型的效用。本文的研究揭示了机器遗忘和未学习数据隐私之间的未被探索的差距，强调了需要仔细设计机制来实现遗忘而不泄露未学习数据的信息。



## **4. FACTUAL: A Novel Framework for Contrastive Learning Based Robust SAR Image Classification**

FACTUAL：一种基于对比学习的稳健SAR图像分类新框架 cs.CV

2024 IEEE Radar Conference

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03225v1) [paper-pdf](http://arxiv.org/pdf/2404.03225v1)

**Authors**: Xu Wang, Tian Ye, Rajgopal Kannan, Viktor Prasanna

**Abstract**: Deep Learning (DL) Models for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR), while delivering improved performance, have been shown to be quite vulnerable to adversarial attacks. Existing works improve robustness by training models on adversarial samples. However, by focusing mostly on attacks that manipulate images randomly, they neglect the real-world feasibility of such attacks. In this paper, we propose FACTUAL, a novel Contrastive Learning framework for Adversarial Training and robust SAR classification. FACTUAL consists of two components: (1) Differing from existing works, a novel perturbation scheme that incorporates realistic physical adversarial attacks (such as OTSA) to build a supervised adversarial pre-training network. This network utilizes class labels for clustering clean and perturbed images together into a more informative feature space. (2) A linear classifier cascaded after the encoder to use the computed representations to predict the target labels. By pre-training and fine-tuning our model on both clean and adversarial samples, we show that our model achieves high prediction accuracy on both cases. Our model achieves 99.7% accuracy on clean samples, and 89.6% on perturbed samples, both outperforming previous state-of-the-art methods.

摘要: 用于合成孔径雷达(SAR)自动目标识别(ATR)的深度学习(DL)模型虽然提供了更好的性能，但已被证明非常容易受到对手攻击。已有的工作通过训练对抗性样本上的模型来提高鲁棒性。然而，由于主要关注随机操纵图像的攻击，他们忽视了此类攻击在现实世界中的可行性。本文提出了一种用于对抗性训练和稳健SAR分类的新型对比学习框架FACTAL。FACTUAL由两部分组成：(1)与已有工作不同，提出了一种新的扰动方案，该方案结合了真实的物理对抗攻击(如OTSA)来构建一个有监督的对抗预训练网络。该网络利用类别标签将干净的和受干扰的图像聚在一起，形成一个更具信息量的特征空间。(2)在编码器之后级联一个线性分类器，使用计算的表示来预测目标标签。通过对干净样本和对抗性样本的预训练和微调，我们证明了我们的模型在这两种情况下都达到了很高的预测精度。我们的模型在清洁样本上达到了99.7%的准确率，在扰动样本上达到了89.6%的准确率，都超过了以前最先进的方法。



## **5. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

无线网络的鲁棒联邦学习：信道估计演示 cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03088v1) [paper-pdf](http://arxiv.org/pdf/2404.03088v1)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.

摘要: 联合学习（FL）为无线网络中的训练模型提供了一种保护隐私的协作方法，信道估计成为一个很有前途的应用。尽管对FL赋能信道估计进行了广泛的研究，但与FL相关的安全问题仍需要精心关注。在小型基站（SBS）用作在缓存数据上训练的本地模型，宏基站（MBS）用作全局模型设置的场景中，攻击者可以利用FL的漏洞，利用各种对抗性攻击或部署策略发起攻击。本文对这些漏洞进行了分析，提出了相应的解决方案，并通过仿真进行了验证。



## **6. Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning**

对抗性规避攻击在网络中的实际性：测试动态学习的影响 cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2306.05494v2) [paper-pdf](http://arxiv.org/pdf/2306.05494v2)

**Authors**: Mohamed el Shehaby, Ashraf Matrawy

**Abstract**: Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy compared to traditional models in processing and classifying large volumes of data. However, ML has been found to have several flaws, most importantly, adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the suitability of these attacks against ML-based network security entities, especially NIDS, due to the wide difference between different domains regarding the generation of adversarial attacks.   To further explore the practicality of adversarial attacks against ML-based NIDS in-depth, this paper presents three distinct contributions: identifying numerous practicality issues for evasion adversarial attacks on ML-NIDS using an attack tree threat model, introducing a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS, and investigating how the dynamicity of some real-world ML models affects adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effectiveness of adversarial attacks. While adversarial attacks can compromise ML-based NIDSs, our aim is to highlight the significant gap between research and real-world practicality in this domain, warranting attention.

摘要: 机器学习(ML)已经变得无处不在，与传统模型相比，它在处理和分类海量数据方面具有自动化的性质和较高的准确率，因此在网络入侵检测系统(NIDS)中的应用是不可避免的。然而，ML被发现有几个缺陷，最重要的是对抗性攻击，其目的是欺骗ML模型产生错误的预测。虽然大多数对抗性攻击研究集中在计算机视觉数据集，但最近的研究探索了这些攻击对基于ML的网络安全实体，特别是网络入侵检测系统的适用性，这是因为不同领域之间关于对抗性攻击生成的巨大差异。为了进一步深入探讨针对基于ML的网络入侵检测系统的对抗性攻击的实用性，本文提出了三个不同的贡献：利用攻击树威胁模型识别针对ML-NID的逃避对抗性攻击的大量实用性问题，引入与针对基于ML的网络入侵检测系统的对抗性攻击相关的实用性问题的分类，以及研究一些真实世界的ML模型的动态性如何影响对网络入侵检测系统的对抗性攻击。我们的实验表明，即使在没有对抗性训练的情况下，持续的再训练也会降低对抗性攻击的有效性。虽然敌意攻击可能会危及基于ML的NIDS，但我们的目标是突出这一领域的研究和现实世界实用之间的显著差距，值得关注。



## **7. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV—28K：评估多模大语言模型抗越狱攻击鲁棒性的基准 cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03027v1) [paper-pdf](http://arxiv.org/pdf/2404.03027v1)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **8. Adversarial Attacks and Dimensionality in Text Classifiers**

文本分类器中的对抗性攻击和模糊性 cs.LG

This paper is accepted for publication at EURASIP Journal on  Information Security in 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02660v1) [paper-pdf](http://arxiv.org/pdf/2404.02660v1)

**Authors**: Nandish Chattopadhyay, Atreya Goswami, Anupam Chattopadhyay

**Abstract**: Adversarial attacks on machine learning algorithms have been a key deterrent to the adoption of AI in many real-world use cases. They significantly undermine the ability of high-performance neural networks by forcing misclassifications. These attacks introduce minute and structured perturbations or alterations in the test samples, imperceptible to human annotators in general, but trained neural networks and other models are sensitive to it. Historically, adversarial attacks have been first identified and studied in the domain of image processing. In this paper, we study adversarial examples in the field of natural language processing, specifically text classification tasks. We investigate the reasons for adversarial vulnerability, particularly in relation to the inherent dimensionality of the model. Our key finding is that there is a very strong correlation between the embedding dimensionality of the adversarial samples and their effectiveness on models tuned with input samples with same embedding dimension. We utilize this sensitivity to design an adversarial defense mechanism. We use ensemble models of varying inherent dimensionality to thwart the attacks. This is tested on multiple datasets for its efficacy in providing robustness. We also study the problem of measuring adversarial perturbation using different distance metrics. For all of the aforementioned studies, we have run tests on multiple models with varying dimensionality and used a word-vector level adversarial attack to substantiate the findings.

摘要: 对机器学习算法的对抗性攻击一直是许多现实世界用例中采用人工智能的关键威慑因素。它们通过强制错误分类，大大削弱了高性能神经网络的能力。这些攻击在测试样本中引入微小的和结构化的扰动或改变，通常人类注释员察觉不到，但经过训练的神经网络和其他模型对此很敏感。历史上，对抗性攻击最早是在图像处理领域被识别和研究的。在本文中，我们研究了自然语言处理领域中的对抗性实例，特别是文本分类任务。我们研究了对抗脆弱性的原因，特别是与模型的固有维度有关的原因。我们的关键发现是，对抗性样本的嵌入维度与其在具有相同嵌入维度的输入样本调整的模型上的有效性之间存在很强的相关性。我们利用这种敏感性来设计一种对抗性防御机制。我们使用不同固有维度的集合模型来阻止攻击。这在多个数据集上进行了测试，以确定其在提供稳健性方面的有效性。我们还研究了使用不同的距离度量来度量对手扰动的问题。对于上述所有研究，我们在不同维度的多个模型上进行了测试，并使用了单词向量级别的对抗性攻击来证实这些发现。



## **9. Adversary-Augmented Simulation to evaluate fairness on HyperLedger Fabric**

在Hyperledger Fabric上评估公平性的对抗增强仿真 cs.CR

10 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2403.14342v2) [paper-pdf](http://arxiv.org/pdf/2403.14342v2)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Sara Tucci-Piergiovanni, Pierre-Yves Piriou

**Abstract**: This paper presents a novel adversary model specifically tailored to distributed systems, aiming to assess the security of blockchain networks. Building upon concepts such as adversarial assumptions, goals, and capabilities, our proposed adversary model classifies and constrains the use of adversarial actions based on classical distributed system models, defined by both failure and communication models. The objective is to study the effects of these allowed actions on the properties of distributed protocols under various system models. A significant aspect of our research involves integrating this adversary model into the Multi-Agent eXperimenter (MAX) framework. This integration enables fine-grained simulations of adversarial attacks on blockchain networks. In this paper, we particularly study four distinct fairness properties on Hyperledger Fabric with the Byzantine Fault Tolerant Tendermint consensus algorithm being selected for its ordering service. We define novel attacks that combine adversarial actions on both protocols, with the aim of violating a specific client-fairness property. Simulations confirm our ability to violate this property and allow us to evaluate the impact of these attacks on several order-fairness properties that relate orders of transaction reception and delivery.

摘要: 提出了一种新的针对分布式系统的敌手模型，旨在评估区块链网络的安全性。基于对抗性假设、目标和能力等概念，我们提出的对抗性模型对基于经典分布式系统模型的对抗性动作的使用进行分类和限制，该模型由故障模型和通信模型定义。目的是研究在不同的系统模型下，这些允许的操作对分布式协议性能的影响。我们研究的一个重要方面涉及将这种对手模型集成到多代理实验者(MAX)框架中。这种整合可以对区块链网络上的对抗性攻击进行细粒度的模拟。本文以拜占庭容错Tendermint一致性算法为排序服务，详细研究了Hyperledger织物的四个不同的公平性。我们定义了新的攻击，它结合了两种协议上的对抗性操作，目的是违反特定的客户端公平属性。模拟证实了我们违反这一属性的能力，并允许我们评估这些攻击对几个与交易接收和交付顺序相关的顺序公平属性的影响。



## **10. Unsegment Anything by Simulating Deformation**

模拟变形解分割任何东西 cs.CV

CVPR 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02585v1) [paper-pdf](http://arxiv.org/pdf/2404.02585v1)

**Authors**: Jiahao Lu, Xingyi Yang, Xinchao Wang

**Abstract**: Foundation segmentation models, while powerful, pose a significant risk: they enable users to effortlessly extract any objects from any digital content with a single click, potentially leading to copyright infringement or malicious misuse. To mitigate this risk, we introduce a new task "Anything Unsegmentable" to grant any image "the right to be unsegmented". The ambitious pursuit of the task is to achieve highly transferable adversarial attacks against all prompt-based segmentation models, regardless of model parameterizations and prompts. We highlight the non-transferable and heterogeneous nature of prompt-specific adversarial noises. Our approach focuses on disrupting image encoder features to achieve prompt-agnostic attacks. Intriguingly, targeted feature attacks exhibit better transferability compared to untargeted ones, suggesting the optimal update direction aligns with the image manifold. Based on the observations, we design a novel attack named Unsegment Anything by Simulating Deformation (UAD). Our attack optimizes a differentiable deformation function to create a target deformed image, which alters structural information while preserving achievable feature distance by adversarial example. Extensive experiments verify the effectiveness of our approach, compromising a variety of promptable segmentation models with different architectures and prompt interfaces. We release the code at https://github.com/jiahaolu97/anything-unsegmentable.

摘要: 基础分割模型虽然功能强大，但也带来了重大风险：它们使用户能够轻松地通过一次点击从任何数字内容中提取任何对象，这可能会导致侵犯版权或恶意滥用。为了减轻这种风险，我们引入了一个新的任务“任何不可分割的”，以授予任何图像“被取消分割的权利”。这项任务的雄心勃勃的追求是实现对所有基于提示的分割模型的高度可转移的对抗性攻击，而不考虑模型的参数化和提示。我们强调了即时特定对抗性噪音的不可转移性和异质性。我们的方法专注于破坏图像编码器功能，以实现与提示无关的攻击。有趣的是，与非目标攻击相比，目标特征攻击表现出更好的可转移性，这表明最佳更新方向与图像流形一致。在此基础上，我们设计了一种新的攻击方法，称为通过模拟变形来不分割任何东西(UAD)。我们的攻击优化了一个可微变形函数来生成目标变形图像，该变形图像改变了目标的结构信息，同时通过对抗性例子保持了可达到的特征距离。大量的实验验证了我们的方法的有效性，折衷了各种具有不同体系结构和提示界面的可提示分割模型。我们在https://github.com/jiahaolu97/anything-unsegmentable.上发布代码



## **11. A Unified Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

基于部件感知能力的视觉自监督编码器统一隶属度推理方法 cs.CV

Membership Inference, Self-supervised learning

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02462v1) [paper-pdf](http://arxiv.org/pdf/2404.02462v1)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we aim to perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses with the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Our code is available at https://github.com/JiePKU/PartCrop

摘要: 自我监督学习在利用大量未标记数据方面表现出了希望，但它也面临着重大的隐私问题，特别是在视觉方面。在本文中，我们的目标是在一种更现实的环境下对视觉自我监督模型进行隶属度推理：当对手攻击时，自我监督训练方法和细节是未知的，因为他在实践中通常面临一个黑箱系统。在这种情况下，考虑到自监督模型可以用完全不同的自监督范型来训练，例如蒙版图像建模和对比学习，训练细节复杂，我们提出了一种统一的隶属度推理方法PartCrop。它的动机是模型之间共享的部件感知能力和对训练数据的更强的部件响应。具体地说，PartCrop裁剪图像中对象的一部分，以在表示空间中使用图像查询响应。我们使用三个广泛使用的图像数据集对具有不同训练协议和结构的自监督模型进行了广泛的攻击。实验结果验证了PartCrop的有效性和泛化能力。此外，为了防御PartCrop，我们评估了两种常见的方法，即提前停止和区分隐私，并提出了一种称为缩小作物尺度范围的定制方法。防御实验表明，这些方法都是有效的。我们的代码可以在https://github.com/JiePKU/PartCrop上找到



## **12. Designing a Photonic Physically Unclonable Function Having Resilience to Machine Learning Attacks**

设计一个对机器学习攻击具有弹性的光子物理不可克隆函数 cs.CR

14 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02440v1) [paper-pdf](http://arxiv.org/pdf/2404.02440v1)

**Authors**: Elena R. Henderson, Jessie M. Henderson, Hiva Shahoei, William V. Oxford, Eric C. Larson, Duncan L. MacFarlane, Mitchell A. Thornton

**Abstract**: Physically unclonable functions (PUFs) are designed to act as device 'fingerprints.' Given an input challenge, the PUF circuit should produce an unpredictable response for use in situations such as root-of-trust applications and other hardware-level cybersecurity applications. PUFs are typically subcircuits present within integrated circuits (ICs), and while conventional IC PUFs are well-understood, several implementations have proven vulnerable to malicious exploits, including those perpetrated by machine learning (ML)-based attacks. Such attacks can be difficult to prevent because they are often designed to work even when relatively few challenge-response pairs are known in advance. Hence the need for both more resilient PUF designs and analysis of ML-attack susceptibility. Previous work has developed a PUF for photonic integrated circuits (PICs). A PIC PUF not only produces unpredictable responses given manufacturing-introduced tolerances, but is also less prone to electromagnetic radiation eavesdropping attacks than a purely electronic IC PUF. In this work, we analyze the resilience of the proposed photonic PUF when subjected to ML-based attacks. Specifically, we describe a computational PUF model for producing the large datasets required for training ML attacks; we analyze the quality of the model; and we discuss the modeled PUF's susceptibility to ML-based attacks. We find that the modeled PUF generates distributions that resemble uniform white noise, explaining the exhibited resilience to neural-network-based attacks designed to exploit latent relationships between challenges and responses. Preliminary analysis suggests that the PUF exhibits similar resilience to generative adversarial networks, and continued development will show whether more-sophisticated ML approaches better compromise the PUF and -- if so -- how design modifications might improve resilience.

摘要: 物理上不可克隆的功能(PUF)被设计成充当设备的“指纹”。在给定输入挑战的情况下，PUF电路应产生不可预测的响应，以便在信任根应用程序和其他硬件级别的网络安全应用程序等情况下使用。PUF通常是集成电路(IC)中存在的子电路，虽然传统的IC PUF是众所周知的，但事实证明，一些实现容易受到恶意利用，包括那些由基于机器学习(ML)的攻击所造成的利用。此类攻击可能很难预防，因为它们通常被设计为即使在事先知道的挑战-响应对相对较少的情况下也能发挥作用。因此，需要更具弹性的PUF设计和ML攻击敏感性分析。以前的工作已经开发了一种用于光子集成电路(PIC)的PUF。PIC PUF不仅在制造引入容差的情况下产生不可预测的响应，而且比纯电子IC PUF更不容易受到电磁辐射窃听攻击。在这项工作中，我们分析了所提出的光子PUF在遭受基于ML的攻击时的弹性。具体地说，我们描述了一个计算PUF模型，用于产生训练ML攻击所需的大数据集；我们分析了该模型的质量；我们讨论了所建模型的PUF对基于ML的攻击的敏感性。我们发现，建模的PUF生成类似于均匀白噪声的分布，解释了对旨在利用挑战和响应之间的潜在关系的基于神经网络的攻击表现出的弹性。初步分析表明，PUF对生成性对抗网络表现出类似的弹性，继续发展将表明更复杂的ML方法是否会更好地折衷PUF，以及--如果是--设计修改如何提高弹性。



## **13. One Noise to Rule Them All: Multi-View Adversarial Attacks with Universal Perturbation**

一个噪声统治他们所有：具有普遍扰动的多视图对抗攻击 cs.CV

6 pages, 4 figures, presented at ICAIA, Springer to publish under  Algorithms for Intelligent Systems

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02287v1) [paper-pdf](http://arxiv.org/pdf/2404.02287v1)

**Authors**: Mehmet Ergezer, Phat Duong, Christian Green, Tommy Nguyen, Abdurrahman Zeybey

**Abstract**: This paper presents a novel universal perturbation method for generating robust multi-view adversarial examples in 3D object recognition. Unlike conventional attacks limited to single views, our approach operates on multiple 2D images, offering a practical and scalable solution for enhancing model scalability and robustness. This generalizable method bridges the gap between 2D perturbations and 3D-like attack capabilities, making it suitable for real-world applications.   Existing adversarial attacks may become ineffective when images undergo transformations like changes in lighting, camera position, or natural deformations. We address this challenge by crafting a single universal noise perturbation applicable to various object views. Experiments on diverse rendered 3D objects demonstrate the effectiveness of our approach. The universal perturbation successfully identified a single adversarial noise for each given set of 3D object renders from multiple poses and viewpoints. Compared to single-view attacks, our universal attacks lower classification confidence across multiple viewing angles, especially at low noise levels. A sample implementation is made available at https://github.com/memoatwit/UniversalPerturbation.

摘要: 提出了一种新的通用摄动方法，用于在三维物体识别中生成健壮的多视点对抗性样本。与传统的仅限于单视图的攻击不同，我们的方法在多个2D图像上操作，为增强模型的可扩展性和健壮性提供了实用且可扩展的解决方案。这种可推广的方法弥合了2D扰动和类似3D的攻击能力之间的差距，使其适用于现实世界的应用。现有的对抗性攻击可能会在图像经历光照、相机位置或自然变形等变化时变得无效。我们通过制作适用于各种对象视图的单一通用噪声扰动来解决这一挑战。在不同渲染的3D对象上的实验证明了该方法的有效性。通用扰动成功地为来自多个姿势和视点的每一组给定的3D对象渲染识别了单个对抗性噪声。与单视图攻击相比，我们的通用攻击降低了多个视角的分类置信度，特别是在低噪声水平下。在https://github.com/memoatwit/UniversalPerturbation.上提供了一个示例实现



## **14. Towards Robust 3D Pose Transfer with Adversarial Learning**

基于对抗学习的鲁棒3D姿势传递 cs.CV

CVPR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02242v1) [paper-pdf](http://arxiv.org/pdf/2404.02242v1)

**Authors**: Haoyu Chen, Hao Tang, Ehsan Adeli, Guoying Zhao

**Abstract**: 3D pose transfer that aims to transfer the desired pose to a target mesh is one of the most challenging 3D generation tasks. Previous attempts rely on well-defined parametric human models or skeletal joints as driving pose sources. However, to obtain those clean pose sources, cumbersome but necessary pre-processing pipelines are inevitable, hindering implementations of the real-time applications. This work is driven by the intuition that the robustness of the model can be enhanced by introducing adversarial samples into the training, leading to a more invulnerable model to the noisy inputs, which even can be further extended to directly handling the real-world data like raw point clouds/scans without intermediate processing. Furthermore, we propose a novel 3D pose Masked Autoencoder (3D-PoseMAE), a customized MAE that effectively learns 3D extrinsic presentations (i.e., pose). 3D-PoseMAE facilitates learning from the aspect of extrinsic attributes by simultaneously generating adversarial samples that perturb the model and learning the arbitrary raw noisy poses via a multi-scale masking strategy. Both qualitative and quantitative studies show that the transferred meshes given by our network result in much better quality. Besides, we demonstrate the strong generalizability of our method on various poses, different domains, and even raw scans. Experimental results also show meaningful insights that the intermediate adversarial samples generated in the training can successfully attack the existing pose transfer models.

摘要: 三维姿态变换是三维生成中最具挑战性的任务之一，其目的是将期望的姿态传递到目标网格上。以前的尝试依赖于定义明确的参数人体模型或骨骼关节作为驱动姿势源。然而，为了获得这些干净的姿态源，繁琐但必要的预处理流水线是不可避免的，阻碍了实时应用的实现。这项工作是由这样一种直觉驱动的，即通过在训练中引入对抗性样本可以增强模型的稳健性，从而产生对噪声输入更不敏感的模型，这甚至可以进一步扩展到直接处理真实世界的数据，如原始点云/扫描而不需要中间处理。此外，我们提出了一种新的3D姿势掩蔽自动编码器(3D-PoseMAE)，这是一种定制的MAE，可以有效地学习3D外部表示(即姿势)。3D-PoseMAE通过同时生成扰动模型的敌意样本和通过多尺度掩蔽策略学习任意原始噪声姿势，从而便于从外部属性方面进行学习。定性和定量的研究都表明，我们的网络给出的传输网格的质量要好得多。此外，我们还证明了我们的方法在各种姿态、不同领域甚至原始扫描上都具有很强的通用性。实验结果还表明，训练中生成的中间对抗性样本能够成功攻击现有的姿势转移模型。



## **15. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

使用简单的自适应攻击破解领先的安全一致LLM cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02151v1) [paper-pdf](http://arxiv.org/pdf/2404.02151v1)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). We provide the code, prompts, and logs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全对齐的LLM也不能抵抗简单的自适应越狱攻击。首先，我们演示了如何成功地利用对logpros的访问来越狱：我们最初设计了一个对抗性提示模板(有时适用于目标LLM)，然后在后缀上应用随机搜索来最大化目标logprob(例如，令牌“Sure”)，可能需要多次重新启动。通过这种方式，我们获得了近100%的攻击成功率-根据GPT-4作为判断-在GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gema-7B和R2D2上，它们都经过了对抗GCG攻击的恶意训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有不暴露日志问题的Claude模型。此外，我们还展示了如何在受限的令牌集合上使用随机搜索来查找有毒模型中的特洛伊木马字符串--这项任务与越狱有许多相似之处--正是这种算法为我们带来了SATML‘24特洛伊木马检测大赛的第一名。这些攻击背后的共同主题是自适应至关重要：不同的模型容易受到不同提示模板的攻击(例如，R2D2对上下文中的学习提示非常敏感)，一些模型基于其API具有独特的漏洞(例如，预填充Claude)，并且在某些设置中，基于先验知识限制令牌搜索空间至关重要(例如，对于木马检测)。我们在https://github.com/tml-epfl/llm-adaptive-attacks.上提供攻击的代码、提示和日志



## **16. READ: Improving Relation Extraction from an ADversarial Perspective**

阅读：从对抗角度改进关系提取 cs.CL

Accepted by findings of NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02931v1) [paper-pdf](http://arxiv.org/pdf/2404.02931v1)

**Authors**: Dawei Li, William Hogan, Jingbo Shang

**Abstract**: Recent works in relation extraction (RE) have achieved promising benchmark accuracy; however, our adversarial attack experiments show that these works excessively rely on entities, making their generalization capability questionable. To address this issue, we propose an adversarial training method specifically designed for RE. Our approach introduces both sequence- and token-level perturbations to the sample and uses a separate perturbation vocabulary to improve the search for entity and context perturbations. Furthermore, we introduce a probabilistic strategy for leaving clean tokens in the context during adversarial training. This strategy enables a larger attack budget for entities and coaxes the model to leverage relational patterns embedded in the context. Extensive experiments show that compared to various adversarial training methods, our method significantly improves both the accuracy and robustness of the model. Additionally, experiments on different data availability settings highlight the effectiveness of our method in low-resource scenarios. We also perform in-depth analyses of our proposed method and provide further hints. We will release our code at https://github.com/David-Li0406/READ.

摘要: 最近在关系抽取(RE)方面的工作已经取得了令人满意的基准精度；然而，我们的对抗攻击实验表明，这些工作过度依赖实体，使得它们的泛化能力受到质疑。为了解决这个问题，我们提出了一种专门为RE设计的对抗性训练方法。我们的方法将序列级和令牌级的扰动引入样本，并使用单独的扰动词汇表来改进实体和上下文扰动的搜索。此外，我们引入了一种概率策略，在对抗性训练期间将干净的令牌留在上下文中。该策略为实体提供了更大的攻击预算，并诱使模型利用嵌入在上下文中的关系模式。大量实验表明，与各种对抗性训练方法相比，该方法显著提高了模型的准确性和稳健性。此外，在不同数据可用性设置上的实验突出了我们的方法在低资源场景下的有效性。我们还对我们提出的方法进行了深入的分析，并提供了进一步的提示。我们将在https://github.com/David-Li0406/READ.上发布我们的代码



## **17. Red-Teaming Segment Anything Model**

Red—Team Segment Anything Model cs.CV

CVPR 2024 - The 4th Workshop of Adversarial Machine Learning on  Computer Vision: Robustness of Foundation Models

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02067v1) [paper-pdf](http://arxiv.org/pdf/2404.02067v1)

**Authors**: Krzysztof Jankowski, Bartlomiej Sobieski, Mateusz Kwiatkowski, Jakub Szulc, Michal Janik, Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Foundation models have emerged as pivotal tools, tackling many complex tasks through pre-training on vast datasets and subsequent fine-tuning for specific applications. The Segment Anything Model is one of the first and most well-known foundation models for computer vision segmentation tasks. This work presents a multi-faceted red-teaming analysis that tests the Segment Anything Model against challenging tasks: (1) We analyze the impact of style transfer on segmentation masks, demonstrating that applying adverse weather conditions and raindrops to dashboard images of city roads significantly distorts generated masks. (2) We focus on assessing whether the model can be used for attacks on privacy, such as recognizing celebrities' faces, and show that the model possesses some undesired knowledge in this task. (3) Finally, we check how robust the model is to adversarial attacks on segmentation masks under text prompts. We not only show the effectiveness of popular white-box attacks and resistance to black-box attacks but also introduce a novel approach - Focused Iterative Gradient Attack (FIGA) that combines white-box approaches to construct an efficient attack resulting in a smaller number of modified pixels. All of our testing methods and analyses indicate a need for enhanced safety measures in foundation models for image segmentation.

摘要: 基础模型已经成为关键工具，通过对大量数据集进行预培训并随后针对特定应用进行微调来处理许多复杂任务。任意分割模型是最早也是最著名的计算机视觉分割任务的基础模型之一。这项工作提出了一个多方面的红团队分析，针对具有挑战性的任务测试了Segment Anything Model：(1)我们分析了样式转移对分段掩模的影响，表明将不利的天气条件和雨滴应用于城市道路的仪表板图像会显著扭曲生成的掩模。(2)我们重点评估了该模型是否可以用于隐私攻击，如识别名人的脸，并表明该模型在该任务中具有一些不需要的知识。(3)最后，我们检验了该模型对文本提示下的分割模板攻击的健壮性。我们不仅展示了流行的白盒攻击的有效性和对黑盒攻击的抵抗力，而且还引入了一种新的专注于方法的迭代梯度攻击(FIGA)，它结合了白盒方法来构造有效的攻击，从而减少了修改的像素数。我们所有的测试方法和分析都表明，需要在图像分割的基础模型中加强安全措施。



## **18. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

开源LLM有多值得信赖？恶意示威下的评估显示其脆弱性 cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.

摘要: 开源大型语言模型(LLM)的快速发展显著地推动了人工智能的发展。然而，人们对他们的可信性仍知之甚少。在缺乏足够可信度的情况下大规模部署这些模型可能会带来重大风险，这突显了迅速发现这些问题的必要性。在这项工作中，我们对开源LLM的可信性进行了对抗性评估，从八个不同的方面对它们进行了仔细的审查，包括毒性、刻板印象、伦理、幻觉、公平性、奉承、隐私和对对抗性演示的健壮性。我们提出了AdvCoU，一种基于话语的扩展链(CUU)提示策略，它结合了精心制作的恶意演示来进行可信度攻击。我们广泛的实验涵盖了最近一系列具有代表性的开源LLM，包括Vicuna、MPT、Falcon、Mistral和Llama 2。经验结果强调了我们攻击策略在不同方面的有效性。更有趣的是，我们的结果分析显示，在一般NLP任务中性能优越的模型并不总是具有更大的可信度；事实上，较大的模型可能更容易受到攻击。此外，经过指令调整、专注于指令遵循的模型往往更容易受到影响，尽管针对安全对齐的微调LLM被证明在减轻对手信任攻击方面是有效的。



## **19. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

解密局部差分隐私、平均贝叶斯隐私和最大贝叶斯隐私之间的相互作用 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16591v3) [paper-pdf](http://arxiv.org/pdf/2403.16591v3)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.

摘要: 机器学习的快速发展导致了各种隐私定义的出现，因为它对隐私构成了威胁，包括局部差异隐私(LDP)的概念。尽管这种衡量隐私的传统方法在许多领域得到了广泛的接受和应用，但它仍然显示出一定的局限性，从未能阻止推论披露到缺乏对对手背景知识的考虑。在这项全面的研究中，我们介绍了贝叶斯隐私，并深入研究了自民党与其贝叶斯同行之间的错综复杂的关系，揭示了对效用-隐私权衡的新见解。我们引入了一个框架，该框架封装了攻击和防御战略，突出了它们的相互作用和有效性。首先揭示了LDP与最大贝叶斯隐私度之间的关系，证明了在均匀先验分布下，满足$xi-LDP的机制将满足$\xi-MBP，反之，$\xi-MBP也赋予2$\xi-LDP。我们的下一个理论贡献是建立在平均贝叶斯隐私度(ABP)和最大贝叶斯隐私度(MBP)之间的严格定义和关系上，用方程$\epsilon_{p，a}\leq\frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p，m}+\epsilon)\cdot(e^{\epsilon_{p，m}+\epsilon}-1)}$来封装。这些关系加强了我们对各种机制提供的隐私保障的理解。我们的工作不仅为未来的经验探索奠定了基础，也承诺促进隐私保护算法的设计，从而促进可信机器学习解决方案的开发。



## **20. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

PatchCURE：提高对抗补丁防御的可证明鲁棒性、模型效用和计算效率 cs.CV

USENIX Security 2024. (extended) technical report

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2310.13076v2) [paper-pdf](http://arxiv.org/pdf/2310.13076v2)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.

摘要: 针对对抗性补丁攻击的最先进防御现在可以实现强大的可证明的健壮性，同时模型效用略有下降。然而，这种令人印象深刻的性能通常是以比无防御模型多10-100倍的推理时间计算为代价的--研究界见证了可证明的健壮性、模型实用性和计算效率之间的激烈三方权衡。在本文中，我们提出了一个名为PatchCURE的防御框架来解决这个权衡问题。PatchCURE为调整防御性能提供了足够的“旋钮”，并允许我们构建一系列防御：最健壮的PatchCURE实例可以与任何现有最先进的防御实例的性能相媲美(无需考虑效率)；最高效的PatchCURE实例具有与无防御模型相似的推理效率。值得注意的是，PatchCURE在所有不同的效率水平上实现了最先进的稳健性和实用性能，例如，当需要计算效率接近无防御模型时，绝对清洁准确率为16%-23%，并且经过认证的稳健精确度优于以前的防御系统。PatchCURE防御体系使我们能够灵活地选择适当的防御，以满足实践中给定的计算和/或效用约束。



## **21. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

人性化机器生成内容：通过对抗攻击规避AI文本检测 cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.

摘要: 随着大型语言模型(LLM)的发展，面对虚假信息传播、知识产权保护和防止学术剽窃等恶意使用案例，检测文本是否由机器生成变得越来越具有挑战性。虽然训练有素的文本检测器在看不见的测试数据上表现出了良好的性能，但最近的研究表明，这些检测器在处理诸如释义等敌意攻击时存在漏洞。在本文中，我们提出了一个更广泛类别的对抗性攻击的框架，旨在对机器生成的内容执行微小的扰动以逃避检测。我们考虑了两种攻击环境：白盒和黑盒，并在动态场景中使用对抗性学习来评估当前检测模型对此类攻击的稳健性的潜在增强。实验结果表明，当前的检测模型可以在短短10秒内被攻破，导致机器生成的文本被错误分类为人类书写的内容。此外，我们还探讨了改进模型在迭代对抗学习中的稳健性的前景。虽然在模型稳健性方面观察到了一些改进，但实际应用仍然面临着巨大的挑战。这些发现为人工智能文本检测器的未来发展指明了方向，强调了需要更准确和更稳健的检测方法。



## **22. Defense without Forgetting: Continual Adversarial Defense with Anisotropic & Isotropic Pseudo Replay**

不忘防御：具有各向异性和各向同性伪重放的连续对抗防御 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01828v1) [paper-pdf](http://arxiv.org/pdf/2404.01828v1)

**Authors**: Yuhang Zhou, Zhongyun Hua

**Abstract**: Deep neural networks have demonstrated susceptibility to adversarial attacks. Adversarial defense techniques often focus on one-shot setting to maintain robustness against attack. However, new attacks can emerge in sequences in real-world deployment scenarios. As a result, it is crucial for a defense model to constantly adapt to new attacks, but the adaptation process can lead to catastrophic forgetting of previously defended against attacks. In this paper, we discuss for the first time the concept of continual adversarial defense under a sequence of attacks, and propose a lifelong defense baseline called Anisotropic \& Isotropic Replay (AIR), which offers three advantages: (1) Isotropic replay ensures model consistency in the neighborhood distribution of new data, indirectly aligning the output preference between old and new tasks. (2) Anisotropic replay enables the model to learn a compromise data manifold with fresh mixed semantics for further replay constraints and potential future attacks. (3) A straightforward regularizer mitigates the 'plasticity-stability' trade-off by aligning model output between new and old tasks. Experiment results demonstrate that AIR can approximate or even exceed the empirical performance upper bounds achieved by Joint Training.

摘要: 深度神经网络已显示出对敌意攻击的敏感性。对抗性防守技术通常集中在一次射击的设置上，以保持对攻击的健壮性。然而，在现实世界的部署场景中，新的攻击可能会按顺序出现。因此，对于防御模型来说，不断适应新的攻击是至关重要的，但适应过程可能会导致灾难性地忘记以前防御攻击的方式。本文首次讨论了一系列攻击下的连续对抗防御的概念，并提出了一种称为各向异性和各向同性重放(AIR)的终身防御基线，它具有三个优点：(1)各向同性重放保证了新数据在邻域分布上的模型一致性，间接地对齐了新旧任务之间的输出偏好。(2)各向异性重放使模型能够学习具有新鲜混合语义的折衷数据流形，用于进一步的重放约束和潜在的未来攻击。(3)通过调整新任务和旧任务之间的模型输出，直接的正则化可以缓解“塑性-稳定性”之间的权衡。实验结果表明，AIR可以接近甚至超过联合训练所获得的经验性能上限。



## **23. Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models**

越狱提示攻击：一种针对扩散模型的可控对抗攻击 cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02928v1) [paper-pdf](http://arxiv.org/pdf/2404.02928v1)

**Authors**: Jiachen Ma, Anda Cao, Zhiqing Xiao, Jie Zhang, Chao Ye, Junbo Zhao

**Abstract**: The fast advance of the image generation community has attracted attention worldwide. The safety issue needs to be further scrutinized and studied. There have been a few works around this area mostly achieving a post-processing design, model-specific, or yielding suboptimal image quality generation. Despite that, in this article, we discover a black-box attack method that enjoys three merits. It enables (i)-attacks both directed and semantic-driven that theoretically and practically pose a hazard to this vast user community, (ii)-surprisingly surpasses the white-box attack in a black-box manner and (iii)-without requiring any post-processing effort. Core to our approach is inspired by the concept guidance intriguing property of Classifier-Free guidance (CFG) in T2I models, and we discover that conducting frustratingly simple guidance in the CLIP embedding space, coupled with the semantic loss and an additionally sensitive word list works very well. Moreover, our results expose and highlight the vulnerabilities in existing defense mechanisms.

摘要: 图像生成社区的快速发展已经引起了全世界的关注。安全问题需要进一步审查和研究。围绕这一领域已经有一些工作，主要是实现后期处理设计，特定于模型，或产生次优的图像质量生成。尽管如此，在本文中，我们发现了一种具有三个优点的黑盒攻击方法。它实现了(I)定向和语义驱动的攻击，这些攻击在理论上和实践上对这个庞大的用户社区构成了威胁，(Ii)在黑盒方式上出人意料地超过了白盒攻击，(Iii)-不需要任何后处理工作。我们方法的核心是受到T2I模型中无分类器指导(CFG)的概念指导的启发，我们发现在片段嵌入空间进行令人沮丧的简单指导，再加上语义损失和额外的敏感词表，效果非常好。此外，我们的结果暴露和突出了现有防御机制中的漏洞。



## **24. Security Allocation in Networked Control Systems under Stealthy Attacks**

隐身攻击下网络控制系统的安全分配 eess.SY

12 pages, 3 figures, and 1 table, journal submission

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2308.16639v2) [paper-pdf](http://arxiv.org/pdf/2308.16639v2)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper considers the problem of security allocation in a networked control system under stealthy attacks. The system is comprised of interconnected subsystems represented by vertices. A malicious adversary selects a single vertex on which to conduct a stealthy data injection attack with the purpose of maximally disrupting a distant target vertex while remaining undetected. Defense resources against the adversary are allocated by a defender on several selected vertices. First, the objectives of the adversary and the defender with uncertain targets are formulated in a probabilistic manner, resulting in an expected worst-case impact of stealthy attacks. Next, we provide a graph-theoretic necessary and sufficient condition under which the cost for the defender and the expected worst-case impact of stealthy attacks are bounded. This condition enables the defender to restrict the admissible actions to dominating sets of the graph representing the network. Then, the security allocation problem is solved through a Stackelberg game-theoretic framework. Finally, the obtained results are validated through a numerical example of a 50-vertex networked control system.

摘要: 研究了网络控制系统在隐身攻击下的安全分配问题。该系统由由顶点表示的相互连接的子系统组成。恶意攻击者选择单个顶点在其上进行隐形数据注入攻击，目的是在保持未被检测的情况下最大限度地破坏远处的目标顶点。针对对手的防御资源由防御者在几个选定的顶点上分配。首先，目标不确定的对手和防御者的目标是以概率的方式制定的，导致了预期的最坏情况下的隐形攻击影响。接下来，我们给出了一个图论的充要条件，在这个充要条件下，防御者的代价和隐身攻击的预期最坏影响是有界的。这一条件使防御者能够将允许的动作限制在表示网络的图的支配集上。然后，通过Stackelberg博弈论框架解决了安全分配问题。最后，通过一个50点网络控制系统的数值算例对所得结果进行了验证。



## **25. ADVREPAIR:Provable Repair of Adversarial Attack**

ADVREPAIR：对抗攻击的可证明修复 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01642v1) [paper-pdf](http://arxiv.org/pdf/2404.01642v1)

**Authors**: Zhiming Chi, Jianan Ma, Pengfei Yang, Cheng-Chao Huang, Renjue Li, Xiaowei Huang, Lijun Zhang

**Abstract**: Deep neural networks (DNNs) are increasingly deployed in safety-critical domains, but their vulnerability to adversarial attacks poses serious safety risks. Existing neuron-level methods using limited data lack efficacy in fixing adversaries due to the inherent complexity of adversarial attack mechanisms, while adversarial training, leveraging a large number of adversarial samples to enhance robustness, lacks provability. In this paper, we propose ADVREPAIR, a novel approach for provable repair of adversarial attacks using limited data. By utilizing formal verification, ADVREPAIR constructs patch modules that, when integrated with the original network, deliver provable and specialized repairs within the robustness neighborhood. Additionally, our approach incorporates a heuristic mechanism for assigning patch modules, allowing this defense against adversarial attacks to generalize to other inputs. ADVREPAIR demonstrates superior efficiency, scalability and repair success rate. Different from existing DNN repair methods, our repair can generalize to general inputs, thereby improving the robustness of the neural network globally, which indicates a significant breakthrough in the generalization capability of ADVREPAIR.

摘要: 深度神经网络(DNN)越来越多地被部署在安全关键领域，但它们对对手攻击的脆弱性构成了严重的安全风险。由于对抗性攻击机制的内在复杂性，现有的利用有限数据的神经元级别的方法在固定对手方面缺乏有效性，而对抗性训练利用大量的对抗性样本来增强稳健性，缺乏可证性。本文提出了一种利用有限数据可证明修复对抗性攻击的新方法--ADVREPAIR。通过使用正式验证，ADVREPAIR构建了补丁模块，当与原始网络集成时，可在健壮性邻域内提供可证明和专门的修复。此外，我们的方法结合了分配补丁模块的启发式机制，允许这种针对对手攻击的防御推广到其他输入。ADVREPAIR表现出卓越的效率、可扩展性和修复成功率。与现有的DNN修复方法不同，我们的修复方法可以推广到一般输入，从而提高了神经网络的全局鲁棒性，这表明ADVREPAIR在泛化能力方面取得了重大突破。



## **26. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

黑盒神经排序模型的多粒度对抗攻击 cs.IR

Accepted by SIGIR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01574v1) [paper-pdf](http://arxiv.org/pdf/2404.01574v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word-level or sentence-level, to a target document. However, limiting perturbations to a single level of granularity may reduce the flexibility of creating adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step are influenced by the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.

摘要: 对抗性排序攻击因其在探测漏洞方面的成功，从而增强了神经排序模型的稳健性而受到越来越多的关注。传统的攻击方法对目标文档采用单一粒度的扰动，例如单词级或句子级。然而，将扰动限制在单一的粒度级别可能会降低创建对抗性示例的灵活性，从而降低攻击的潜在威胁。因此，我们专注于通过结合多粒度扰动来生成高质量的对抗性实例。实现这一目标需要处理组合爆炸问题，这需要确定跨所有可能级别的粒度、位置和文本片段的扰动的最佳组合。为了应对这一挑战，我们将多粒度的对抗性攻击转化为一个连续的决策过程，其中下一攻击步骤中的扰动受到当前攻击步骤中扰动文档的影响。由于攻击过程只能访问最终状态，没有直接的中间信号，因此我们使用强化学习来执行多粒度攻击。在强化学习过程中，两个代理协作识别多粒度漏洞作为攻击目标，并将扰动候选组织成最终的扰动序列。实验结果表明，我们的攻击方法在攻击有效性和不可感知性方面都超过了主流基线。



## **27. MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**

MMCert：针对多模态模型的对抗攻击的可证明防御 cs.CV

To appear in CVPR'24

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.19080v3) [paper-pdf](http://arxiv.org/pdf/2403.19080v3)

**Authors**: Yanting Wang, Hongye Fu, Wei Zou, Jinyuan Jia

**Abstract**: Different from a unimodal model whose input is from a single modality, the input (called multi-modal input) of a multi-modal model is from multiple modalities such as image, 3D points, audio, text, etc. Similar to unimodal models, many existing studies show that a multi-modal model is also vulnerable to adversarial perturbation, where an attacker could add small perturbation to all modalities of a multi-modal input such that the multi-modal model makes incorrect predictions for it. Existing certified defenses are mostly designed for unimodal models, which achieve sub-optimal certified robustness guarantees when extended to multi-modal models as shown in our experimental results. In our work, we propose MMCert, the first certified defense against adversarial attacks to a multi-modal model. We derive a lower bound on the performance of our MMCert under arbitrary adversarial attacks with bounded perturbations to both modalities (e.g., in the context of auto-driving, we bound the number of changed pixels in both RGB image and depth image). We evaluate our MMCert using two benchmark datasets: one for the multi-modal road segmentation task and the other for the multi-modal emotion recognition task. Moreover, we compare our MMCert with a state-of-the-art certified defense extended from unimodal models. Our experimental results show that our MMCert outperforms the baseline.

摘要: 与单通道模型的输入来自单一通道不同，多通道模型的输入(称为多通道输入)来自图像、3D点、音频、文本等多个通道。与单通道模型类似，许多现有的研究表明，多通道模型也容易受到对抗性扰动的影响，攻击者可以在多通道输入的所有通道中添加小的扰动，从而使得多通道模型对其做出错误的预测。现有的认证防御大多是针对单模模型设计的，如我们的实验结果所示，当扩展到多模模型时，它们获得了次优的认证稳健性保证。在我们的工作中，我们提出了MMCert，这是第一个认证的多模式对抗攻击防御模型。我们得到了MMCert在两种模式都有界扰动的任意攻击下的性能下界(例如，在自动驾驶的背景下，我们限制了RGB图像和深度图像中变化的像素数量)。我们使用两个基准数据集来评估我们的MMCert：一个用于多模式道路分割任务，另一个用于多模式情感识别任务。此外，我们将我们的MMCert与从单模模型扩展而来的最先进的认证防御进行了比较。我们的实验结果表明，我们的MMCert的性能优于基线。



## **28. Rumor Detection with a novel graph neural network approach**

基于图神经网络的谣言检测方法 cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16206v3) [paper-pdf](http://arxiv.org/pdf/2403.16206v3)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.

摘要: 谣言在社交媒体上的广泛传播对人们的日常生活造成了负面影响，给公众带来了潜在的恐慌、恐惧和心理健康问题。如何尽早揭穿谣言仍是一个具有挑战性的问题。现有的研究主要是利用信息传播结构来发现谣言，而很少有人关注用户之间的相关性，他们可能会协同传播谣言以获得更大的人气。在本文中，我们提出了一种新的检测模型，该模型同时学习用户相关性和信息传播的表示，以检测社交媒体上的谣言。具体地说，我们利用图神经网络从描述用户和源推文之间的相关性的二部图中学习用户相关性的表示，以及用树结构表示信息传播。然后，我们结合这两个模块的学习表示来对谣言进行分类。由于恶意用户在部署后有意颠覆我们的模型，我们进一步开发了一种贪婪攻击方案，分析了图攻击、评论攻击和联合攻击三种对抗性攻击的代价。在两个公开数据集上的评估结果表明，该模型的性能优于最新的谣言检测模型。我们还证明了我们的方法在早期谣言检测中表现良好。此外，与现有的最佳检测方法相比，本文提出的检测方法对敌意攻击具有更强的鲁棒性。重要的是，我们证明了攻击者要颠覆用户相关性模式需要付出很高的代价，这说明了考虑用户相关性对谣言检测的重要性。



## **29. Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats**

对抗威胁下的基础模型集成联邦学习的脆弱性 cs.CR

Chen Wu and Xi Li are equal contribution. The corresponding author is  Jiaqi Wang

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2401.10375v2) [paper-pdf](http://arxiv.org/pdf/2401.10375v2)

**Authors**: Chen Wu, Xi Li, Jiaqi Wang

**Abstract**: Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we find that existing FL defense strategies offer limited protection against this novel attack approach. This research highlights the critical need for enhanced security measures in FL in the era of FMs.

摘要: 联合学习(FL)解决了机器学习中与数据隐私和安全相关的关键问题，但在某些情况下存在数据不足和不平衡的问题。基础模型(FM)的出现为现有FL框架的局限性提供了潜在的解决方案，例如通过生成用于模型初始化的合成数据。然而，由于FMS固有的安全问题，将FMS整合到FL中可能会带来新的风险，这在很大程度上仍未被探索。为了弥补这一差距，我们首次对FM集成FL(FM-FL)在对手威胁下的脆弱性进行了研究。基于FM-FL的统一框架，我们提出了一种新的攻击策略，利用FM的安全问题来危害FL客户端模型。通过在图像域和文本域使用著名的模型和基准数据集进行广泛的实验，我们揭示了FM-FL在不同FL配置下对这种新威胁的高度敏感性。此外，我们发现，现有的FL防御策略对这种新的攻击方法提供的保护有限。这项研究强调了在FMS时代加强FL安全措施的迫切需要。



## **30. Can Biases in ImageNet Models Explain Generalization?**

ImagNet模型中的偏差能解释泛化吗？ cs.CV

Accepted at CVPR2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01509v1) [paper-pdf](http://arxiv.org/pdf/2404.01509v1)

**Authors**: Paul Gavrikov, Janis Keuper

**Abstract**: The robust generalization of models to rare, in-distribution (ID) samples drawn from the long tail of the training distribution and to out-of-training-distribution (OOD) samples is one of the major challenges of current deep learning methods. For image classification, this manifests in the existence of adversarial attacks, the performance drops on distorted images, and a lack of generalization to concepts such as sketches. The current understanding of generalization in neural networks is very limited, but some biases that differentiate models from human vision have been identified and might be causing these limitations. Consequently, several attempts with varying success have been made to reduce these biases during training to improve generalization. We take a step back and sanity-check these attempts. Fixing the architecture to the well-established ResNet-50, we perform a large-scale study on 48 ImageNet models obtained via different training methods to understand how and if these biases - including shape bias, spectral biases, and critical bands - interact with generalization. Our extensive study results reveal that contrary to previous findings, these biases are insufficient to accurately predict the generalization of a model holistically. We provide access to all checkpoints and evaluation code at https://github.com/paulgavrikov/biases_vs_generalization

摘要: 将模型推广到从训练分布的长尾中提取的稀有分布内(ID)样本和训练分布外(OOD)样本是当前深度学习方法的主要挑战之一。对于图像分类，这表现在存在对抗性攻击，对失真图像的性能下降，以及对草图等概念缺乏泛化。目前对神经网络泛化的理解非常有限，但已经发现了一些将模型与人类视觉区分开来的偏差，并可能导致这些限制。因此，已经进行了几次尝试，但取得了不同的成功，以减少培训期间的这些偏见，以改进泛化。我们退后一步，理智地检查这些尝试。将架构固定到成熟的ResNet-50，我们对通过不同训练方法获得的48个ImageNet模型进行了大规模研究，以了解这些偏差-包括形状偏差、光谱偏差和关键频带-如何以及是否与泛化相互作用。我们广泛的研究结果表明，与以前的发现相反，这些偏差不足以准确地整体预测模型的概括性。我们允许访问https://github.com/paulgavrikov/biases_vs_generalization上的所有检查点和评估代码



## **31. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

利用潜在对抗训练防御不可预见的故障模式 cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.05030v3) [paper-pdf](http://arxiv.org/pdf/2403.05030v3)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use it to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 尽管开发人员进行了广泛的诊断和调试，但人工智能系统有时会表现出有害的意外行为。找到并修复这些攻击是具有挑战性的，因为攻击面太大了--要详尽地搜索可能引发有害行为的输入并不容易。红队和对抗性训练(AT)通常用于提高健壮性，然而，根据经验，它们难以修复与训练期间使用的攻击不同的失败模式。在这项工作中，我们利用潜在的对手训练(LAT)来防御漏洞，而不会生成引发漏洞的输入。随后，利用网络实际用于预测的概念的压缩、抽象和结构化的潜在表示。我们使用它来删除特洛伊木马程序，并防御抵抗类的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，相对于AT，LAT通常可以提高对新攻击的健壮性和对干净数据的性能。这表明，LAT可以成为一种很有前途的工具，用于防御开发人员未明确识别的故障模式。



## **32. Robust One-Class Classification with Signed Distance Function using 1-Lipschitz Neural Networks**

基于1—Lipschitz神经网络的带符号距离函数单类分类 cs.LG

27 pages, 11 figures, International Conference on Machine Learning  2023, (ICML 2023)

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2303.01978v2) [paper-pdf](http://arxiv.org/pdf/2303.01978v2)

**Authors**: Louis Bethune, Paul Novello, Thibaut Boissin, Guillaume Coiffier, Mathieu Serrurier, Quentin Vincenot, Andres Troya-Galvis

**Abstract**: We propose a new method, dubbed One Class Signed Distance Function (OCSDF), to perform One Class Classification (OCC) by provably learning the Signed Distance Function (SDF) to the boundary of the support of any distribution. The distance to the support can be interpreted as a normality score, and its approximation using 1-Lipschitz neural networks provides robustness bounds against $l2$ adversarial attacks, an under-explored weakness of deep learning-based OCC algorithms. As a result, OCSDF comes with a new metric, certified AUROC, that can be computed at the same cost as any classical AUROC. We show that OCSDF is competitive against concurrent methods on tabular and image data while being way more robust to adversarial attacks, illustrating its theoretical properties. Finally, as exploratory research perspectives, we theoretically and empirically show how OCSDF connects OCC with image generation and implicit neural surface parametrization. Our code is available at https://github.com/Algue-Rythme/OneClassMetricLearning

摘要: 我们提出了一种新的方法，称为一类符号距离函数(OCSDF)，通过可证明地学习符号距离函数(SDF)到任意分布的支持度边界来执行一类分类(OCC)。到支持点的距离可以解释为正态得分，其使用1-Lipschitz神经网络的逼近提供了对$L2$对手攻击的稳健界，这是基于深度学习的OCC算法的一个未被充分挖掘的弱点。因此，OCSDF附带了一种新的衡量标准-认证AUROC，其计算成本可以与任何经典AUROC相同。我们证明了OCSDF在表格和图像数据上与并发方法相比具有竞争力，同时对对手攻击具有更强的健壮性，说明了它的理论性质。最后，作为探索性的研究视角，我们从理论和经验上展示了OCSDF如何将OCC与图像生成和隐式神经表面参数化联系起来。我们的代码可以在https://github.com/Algue-Rythme/OneClassMetricLearning上找到



## **33. The twin peaks of learning neural networks**

学习神经网络的双峰 cs.LG

37 pages, 31 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2401.12610v2) [paper-pdf](http://arxiv.org/pdf/2401.12610v2)

**Authors**: Elizaveta Demyanenko, Christoph Feinauer, Enrico M. Malatesta, Luca Saglietti

**Abstract**: Recent works demonstrated the existence of a double-descent phenomenon for the generalization error of neural networks, where highly overparameterized models escape overfitting and achieve good test performance, at odds with the standard bias-variance trade-off described by statistical learning theory. In the present work, we explore a link between this phenomenon and the increase of complexity and sensitivity of the function represented by neural networks. In particular, we study the Boolean mean dimension (BMD), a metric developed in the context of Boolean function analysis. Focusing on a simple teacher-student setting for the random feature model, we derive a theoretical analysis based on the replica method that yields an interpretable expression for the BMD, in the high dimensional regime where the number of data points, the number of features, and the input size grow to infinity. We find that, as the degree of overparameterization of the network is increased, the BMD reaches an evident peak at the interpolation threshold, in correspondence with the generalization error peak, and then slowly approaches a low asymptotic value. The same phenomenology is then traced in numerical experiments with different model classes and training setups. Moreover, we find empirically that adversarially initialized models tend to show higher BMD values, and that models that are more robust to adversarial attacks exhibit a lower BMD.

摘要: 最近的工作证明了神经网络泛化误差存在双下降现象，即高度过参数的模型避免了过拟合并获得了良好的测试性能，这与统计学习理论所描述的标准偏差-方差权衡不一致。在目前的工作中，我们探索了这种现象与神经网络表示的函数的复杂性和敏感度的增加之间的联系。特别是，我们研究了布尔平均维度(BMD)，这是在布尔函数分析的背景下发展起来的一种度量。针对一个简单的教师-学生随机特征模型，我们基于复制品方法进行了理论分析，在数据点数目、特征数目和输入大小都增长到无穷大的高维区域中，给出了一个可解释的BMD表达式。我们发现，随着网络的超参数化程度的增加，BMD在与泛化误差峰值相对应的内插阈值处达到一个明显的峰值，然后缓慢地接近一个较低的渐近值。然后在不同模型类别和训练设置的数值实验中追踪相同的现象学。此外，我们从经验上发现，对抗性初始化的模型往往显示出较高的BMD值，而对对抗性攻击越健壮的模型显示出较低的BMD。



## **34. Foundations of Cyber Resilience: The Confluence of Game, Control, and Learning Theories**

网络弹性的基础：游戏、控制和学习理论的融合 eess.SY

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01205v1) [paper-pdf](http://arxiv.org/pdf/2404.01205v1)

**Authors**: Quanyan Zhu

**Abstract**: Cyber resilience is a complementary concept to cybersecurity, focusing on the preparation, response, and recovery from cyber threats that are challenging to prevent. Organizations increasingly face such threats in an evolving cyber threat landscape. Understanding and establishing foundations for cyber resilience provide a quantitative and systematic approach to cyber risk assessment, mitigation policy evaluation, and risk-informed defense design. A systems-scientific view toward cyber risks provides holistic and system-level solutions. This chapter starts with a systemic view toward cyber risks and presents the confluence of game theory, control theory, and learning theories, which are three major pillars for the design of cyber resilience mechanisms to counteract increasingly sophisticated and evolving threats in our networks and organizations. Game and control theoretic methods provide a set of modeling frameworks to capture the strategic and dynamic interactions between defenders and attackers. Control and learning frameworks together provide a feedback-driven mechanism that enables autonomous and adaptive responses to threats. Game and learning frameworks offer a data-driven approach to proactively reason about adversarial behaviors and resilient strategies. The confluence of the three lays the theoretical foundations for the analysis and design of cyber resilience. This chapter presents various theoretical paradigms, including dynamic asymmetric games, moving horizon control, conjectural learning, and meta-learning, as recent advances at the intersection. This chapter concludes with future directions and discussions of the role of neurosymbolic learning and the synergy between foundation models and game models in cyber resilience.

摘要: 网络复原力是网络安全的补充概念，侧重于预防具有挑战性的网络威胁的准备、响应和恢复。在不断变化的网络威胁环境中，组织面临的此类威胁越来越多。理解和建立网络复原力的基础为网络风险评估、缓解政策评估和风险知情防御设计提供了一种量化和系统的方法。系统科学的网络风险观提供了整体和系统级的解决方案。本章从系统地看待网络风险开始，介绍了博弈论、控制论和学习理论的融合，这三个理论是设计网络弹性机制的三大支柱，以对抗我们网络和组织中日益复杂和不断变化的威胁。博弈论和控制论方法提供了一套模型框架来捕捉防御者和攻击者之间的战略和动态交互。控制和学习框架共同提供了一种反馈驱动的机制，使其能够对威胁做出自主和适应性的反应。游戏和学习框架提供了一种数据驱动的方法来主动推理对手行为和弹性策略。三者的融合为网络韧性的分析和设计奠定了理论基础。本章介绍了各种理论范式，包括动态不对称博弈、移动视野控制、猜想学习和元学习，作为交叉路口的最新进展。本章最后对神经符号学习的作用以及基础模型和游戏模型在网络韧性中的协同作用进行了未来的方向和讨论。



## **35. The Best Defense is Attack: Repairing Semantics in Textual Adversarial Examples**

最好的防御是攻击：文本对抗示例中的语义修复 cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2305.04067v2) [paper-pdf](http://arxiv.org/pdf/2305.04067v2)

**Authors**: Heng Yang, Ke Li

**Abstract**: Recent studies have revealed the vulnerability of pre-trained language models to adversarial attacks. Existing adversarial defense techniques attempt to reconstruct adversarial examples within feature or text spaces. However, these methods struggle to effectively repair the semantics in adversarial examples, resulting in unsatisfactory performance and limiting their practical utility. To repair the semantics in adversarial examples, we introduce a novel approach named Reactive Perturbation Defocusing (Rapid). Rapid employs an adversarial detector to identify fake labels of adversarial examples and leverage adversarial attackers to repair the semantics in adversarial examples. Our extensive experimental results conducted on four public datasets, convincingly demonstrate the effectiveness of Rapid in various adversarial attack scenarios. To address the problem of defense performance validation in previous works, we provide a demonstration of adversarial detection and repair based on our work, which can be easily evaluated at https://tinyurl.com/22ercuf8.

摘要: 最近的研究揭示了预先训练的语言模型在对抗性攻击中的脆弱性。现有的对抗性防御技术试图在特征或文本空间内重建对抗性示例。然而，这些方法难以有效地修复对抗性实例中的语义，导致性能不佳，限制了它们的实用价值。为了修复对抗性例子中的语义，我们引入了一种新的方法--反应性扰动散焦(Rapid)。RAPID使用对抗性检测器来识别对抗性实例的虚假标签，并利用对抗性攻击者来修复对抗性实例中的语义。我们在四个公开数据集上进行的大量实验结果令人信服地证明了Rapid在各种对抗性攻击场景中的有效性。为了解决前人工作中的防御性能验证问题，我们在工作的基础上提供了一个对手检测和修复的演示，该演示可以在https://tinyurl.com/22ercuf8.上轻松地进行评估



## **36. Poisoning Decentralized Collaborative Recommender System and Its Countermeasures**

分布式协同推荐系统中毒及其对策 cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01177v1) [paper-pdf](http://arxiv.org/pdf/2404.01177v1)

**Authors**: Ruiqi Zheng, Liang Qu, Tong Chen, Kai Zheng, Yuhui Shi, Hongzhi Yin

**Abstract**: To make room for privacy and efficiency, the deployment of many recommender systems is experiencing a shift from central servers to personal devices, where the federated recommender systems (FedRecs) and decentralized collaborative recommender systems (DecRecs) are arguably the two most representative paradigms. While both leverage knowledge (e.g., gradients) sharing to facilitate learning local models, FedRecs rely on a central server to coordinate the optimization process, yet in DecRecs, the knowledge sharing directly happens between clients. Knowledge sharing also opens a backdoor for model poisoning attacks, where adversaries disguise themselves as benign clients and disseminate polluted knowledge to achieve malicious goals like promoting an item's exposure rate. Although research on such poisoning attacks provides valuable insights into finding security loopholes and corresponding countermeasures, existing attacks mostly focus on FedRecs, and are either inapplicable or ineffective for DecRecs. Compared with FedRecs where the tampered information can be universally distributed to all clients once uploaded to the cloud, each adversary in DecRecs can only communicate with neighbor clients of a small size, confining its impact to a limited range. To fill the gap, we present a novel attack method named Poisoning with Adaptive Malicious Neighbors (PAMN). With item promotion in top-K recommendation as the attack objective, PAMN effectively boosts target items' ranks with several adversaries that emulate benign clients and transfers adaptively crafted gradients conditioned on each adversary's neighbors. Moreover, with the vulnerabilities of DecRecs uncovered, a dedicated defensive mechanism based on user-level gradient clipping with sparsified updating is proposed. Extensive experiments demonstrate the effectiveness of the poisoning attack and the robustness of our defensive mechanism.

摘要: 为了给隐私和效率腾出空间，许多推荐系统的部署正在经历从中央服务器到个人设备的转变，其中联合推荐系统(FedRecs)和分散协作推荐系统(DecRecs)可以说是两个最具代表性的范例。虽然两者都利用知识(例如，梯度)共享来促进学习本地模型，但FedRecs依赖中央服务器来协调优化过程，而在DecRecs中，知识共享直接发生在客户之间。知识共享也为模型中毒攻击打开了后门，对手将自己伪装成良性客户，传播受污染的知识，以实现恶意目标，如提高物品的曝光率。虽然对这类中毒攻击的研究为发现安全漏洞和相应的对策提供了宝贵的见解，但现有的攻击大多集中在FedRecs上，不适用于DECRecs，或者对DECRecs无效。与FedRecs中被篡改的信息一旦上传到云中就可以统一分发到所有客户端相比，DecRecs中的每个对手只能与小规模的邻居客户端通信，将其影响限制在有限的范围内。为了填补这一空白，我们提出了一种新的攻击方法--自适应恶意邻居投毒攻击(PAMN)。以TOP-K推荐中的条目推广为攻击目标，PAMN有效地提升了目标条目的排名，多个对手模仿良性客户端，并根据每个对手的邻居自适应地传输定制的梯度。此外，针对DecRecs存在的漏洞，提出了一种基于稀疏更新的用户级梯度裁剪的专用防御机制。大量的实验证明了中毒攻击的有效性和我们防御机制的健壮性。



## **37. The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness**

输入扰动的双刃剑实现鲁棒精确公平 cs.LG

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01356v1) [paper-pdf](http://arxiv.org/pdf/2404.01356v1)

**Authors**: Xuran Li, Peng Wu, Yanting Chen, Xingjun Ma, Zhen Zhang, Kaixiang Dong

**Abstract**: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input perturbations to robust accurate fairness in DNN and the potential of using benign perturbations to correct adversarial instances.

摘要: 深度神经网络(DNN)对敌意输入扰动非常敏感，导致预测精度或个体公平性降低。为了联合刻画预测精度和个体公平性对对抗扰动的敏感性，我们引入了一种新的健壮性定义，称为鲁棒准确公平性。非正式地讲，稳健准确的公平性要求在受到输入扰动时，对实例及其类似实例的预测与基本事实一致。我们提出了一种称为RAFair的对抗性攻击方法，以揭露DNN中虚假或有偏见的对抗性缺陷，这些缺陷要么欺骗准确性，要么损害个体公平性。然后，我们证明了这样的对抗性实例可以通过精心设计的良性扰动来有效地解决，修正他们的预测是准确和公平的。我们的工作探索了输入扰动对DNN中稳健的准确公平性的双刃剑，以及使用良性扰动来纠正敌对实例的可能性。



## **38. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队游戏：红色团队语言模型的游戏理论框架 cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2310.00322v3) [paper-pdf](http://arxiv.org/pdf/2310.00322v3)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **39. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

LogoStyleFool：通过标志风格转移来削弱视频识别系统 cs.CV

14 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2312.09935v2) [paper-pdf](http://arxiv.org/pdf/2312.09935v2)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.

摘要: 视频识别系统很容易受到敌意例子的攻击。最近的研究表明，基于风格迁移和基于补丁的无限制扰动可以有效地提高攻击效率。然而，这些攻击面临两个主要挑战：1)向所有像素添加大的风格化扰动会降低视频的自然度，并且这种扰动很容易被检测到。2)基于补丁的视频攻击不能扩展到有针对性的攻击，因为强化学习的搜索空间有限，这是近年来在视频攻击中广泛使用的。本文针对视频黑盒的设置，通过在干净的视频中添加一个风格化的标识，提出了一种新的攻击框架--LogoStyleFool。我们将攻击分为三个阶段：样式参考选择、基于强化学习的标识样式迁移和扰动优化。我们通过将扰动范围缩小到区域标志来解决第一个挑战，而第二个挑战是通过在强化学习后补充优化阶段来解决的。实验结果表明，在攻击性能和语义保持方面，LogoStyleFool在攻击性能和语义保持方面都优于三种最先进的基于补丁的攻击。同时，与现有的两种基于补丁的防御方法相比，LogoStyleFool仍然保持其性能。我们认为，我们的研究有助于提高安全界对这种次区域风格的转移袭击的关注。



## **40. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过风格转移欺骗视频分类系统 cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2203.16000v4) [paper-pdf](http://arxiv.org/pdf/2203.16000v4)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在有针对性的攻击中，还考虑了目标类置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后使用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。此外，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以愚弄视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **41. BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks**

BadPart：针对像素回归任务的统一黑盒对抗补丁攻击 cs.CV

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00924v1) [paper-pdf](http://arxiv.org/pdf/2404.00924v1)

**Authors**: Zhiyuan Cheng, Zhaoyi Liu, Tengda Guo, Shiwei Feng, Dongfang Liu, Mingjie Tang, Xiangyu Zhang

**Abstract**: Pixel-wise regression tasks (e.g., monocular depth estimation (MDE) and optical flow estimation (OFE)) have been widely involved in our daily life in applications like autonomous driving, augmented reality and video composition. Although certain applications are security-critical or bear societal significance, the adversarial robustness of such models are not sufficiently studied, especially in the black-box scenario. In this work, we introduce the first unified black-box adversarial patch attack framework against pixel-wise regression tasks, aiming to identify the vulnerabilities of these models under query-based black-box attacks. We propose a novel square-based adversarial patch optimization framework and employ probabilistic square sampling and score-based gradient estimation techniques to generate the patch effectively and efficiently, overcoming the scalability problem of previous black-box patch attacks. Our attack prototype, named BadPart, is evaluated on both MDE and OFE tasks, utilizing a total of 7 models. BadPart surpasses 3 baseline methods in terms of both attack performance and efficiency. We also apply BadPart on the Google online service for portrait depth estimation, causing 43.5% relative distance error with 50K queries. State-of-the-art (SOTA) countermeasures cannot defend our attack effectively.

摘要: 像素级回归任务(如单目深度估计(MDE)和光流估计(OFE))在自动驾驶、增强现实和视频合成等应用中广泛应用于我们的日常生活中。虽然某些应用是安全关键的或具有社会意义的，但这些模型的对抗健壮性没有得到充分的研究，特别是在黑盒场景中。在这项工作中，我们引入了第一个针对像素回归任务的统一黑盒对抗性补丁攻击框架，旨在识别这些模型在基于查询的黑盒攻击下的脆弱性。提出了一种新的基于平方的对抗性补丁优化框架，并利用概率平方采样和基于分数的梯度估计技术有效地生成了补丁，克服了以往黑盒补丁攻击的可扩展性问题。我们的攻击原型名为BadPart，在MDE和OFE任务上进行了评估，总共使用了7个模型。BadPart在攻击性能和效率方面都超过了3种基线方法。我们还将BadPart应用于Google在线服务上进行人像深度估计，在50K查询中导致了43.5%的相对距离误差。最先进的(SOTA)对策不能有效地防御我们的攻击。



## **42. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

一个令人尴尬的简单的后门攻击防御SSL cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2403.15918v2) [paper-pdf](http://arxiv.org/pdf/2403.15918v2)

**Authors**: Aryan Satpathy, Nilaksh Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et. al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.

摘要: 自我监督学习(SSL)已经成为一种强大的范式，可以在缺乏人类监督的情况下处理数据环境。无需使用标签数据即可学习有意义的任务的能力使SSL成为在没有标签的情况下管理大量数据的流行方法。然而，最近的研究表明，SSL容易受到后门攻击，在后门攻击中，可以控制模型，可能是恶意的，以适应对手的动机。Li et.Al(2022)提出了一种新的基于频率的后门攻击：Ctrl。他们表明，CTRL可以用来有效地、秘密地控制使用SSL训练的受害者模型。在这项工作中，我们针对基于频率的攻击设计了两种防御策略：一种适用于模型训练之前，另一种应用于模型推理中。我们的第一个贡献是利用下游任务的不变性以一种可推广的方式防御后门攻击。我们观察到，在整个实验中，ASR(攻击成功率)降低了60%以上。我们的推理时间防御依赖于攻击的规避，并使用亮度通道来防御攻击。使用对象分类作为SSL的下游任务，我们演示了成功的防御策略，不需要对模型进行重新训练。代码可在https://github.com/Aryan-Satpathy/Backdoor.上找到



## **43. Machine Learning Robustness: A Primer**

机器学习鲁棒性：入门 cs.LG

arXiv admin note: text overlap with arXiv:2305.10862 by other authors

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.00897v1) [paper-pdf](http://arxiv.org/pdf/2404.00897v1)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.

摘要: 本章探讨了机器学习(ML)中稳健性的基本概念及其在人工智能(AI)系统中建立可信度的不可或缺的作用。讨论开始于对稳健性的详细定义，将其描述为ML模型在不同和意外的环境条件下保持稳定性能的能力。ML稳健性通过几个方面进行剖析：它与通用性的互补性；它作为值得信赖的人工智能的要求的地位；它的对抗性与非对抗性方面；它的量化指标；以及它的可再现性和可解释性等指标。本章深入探讨了阻碍健壮性的因素，如数据偏差、模型复杂性和未指定的ML管道的陷阱。它从广泛的角度考察了健壮性评估的关键技术，包括涵盖数字和物理领域的对抗性攻击。它涵盖了深度学习(DL)软件测试方法的非对抗性数据转移和细微差别。讨论继续探索增强健壮性的改进策略，从去偏向和增强等以数据为中心的方法开始。进一步的考试包括各种以模型为中心的方法，如转移学习、对抗性训练和随机平滑。最后，讨论了训练后的方法，包括集合技术、修剪和模型修复，这些方法成为使模型对不可预测的情况更具弹性的成本效益策略。本章强调了在通过现有方法估计和实现ML健壮性方面的持续挑战和限制。它为未来对这一关键概念的研究提供了见解和方向，这是值得信赖的人工智能系统的先决条件。



## **44. Privacy Re-identification Attacks on Tabular GANs**

针对Tabular GAN的隐私重识别攻击 cs.CR

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00696v1) [paper-pdf](http://arxiv.org/pdf/2404.00696v1)

**Authors**: Abdallah Alshantti, Adil Rasheed, Frank Westad

**Abstract**: Generative models are subject to overfitting and thus may potentially leak sensitive information from the training data. In this work. we investigate the privacy risks that can potentially arise from the use of generative adversarial networks (GANs) for creating tabular synthetic datasets. For the purpose, we analyse the effects of re-identification attacks on synthetic data, i.e., attacks which aim at selecting samples that are predicted to correspond to memorised training samples based on their proximity to the nearest synthetic records. We thus consider multiple settings where different attackers might have different access levels or knowledge of the generative model and predictive, and assess which information is potentially most useful for launching more successful re-identification attacks. In doing so we also consider the situation for which re-identification attacks are formulated as reconstruction attacks, i.e., the situation where an attacker uses evolutionary multi-objective optimisation for perturbing synthetic samples closer to the training space. The results indicate that attackers can indeed pose major privacy risks by selecting synthetic samples that are likely representative of memorised training samples. In addition, we notice that privacy threats considerably increase when the attacker either has knowledge or has black-box access to the generative models. We also find that reconstruction attacks through multi-objective optimisation even increase the risk of identifying confidential samples.

摘要: 生成性模型容易过度拟合，因此可能会泄漏训练数据中的敏感信息。在这项工作中。我们调查了使用生成性对抗网络(GANS)来创建表格合成数据集可能产生的隐私风险。为此，我们分析了重新识别攻击对合成数据的影响，即，旨在根据样本与最近的合成记录的接近程度来选择与记忆的训练样本相对应的样本的攻击。因此，我们考虑了不同攻击者可能具有不同访问级别或生成性模型和预测性知识的多个设置，并评估哪些信息可能对发起更成功的重新识别攻击最有用。在这样做的同时，我们还考虑了将重新识别攻击描述为重构攻击的情况，即攻击者使用进化多目标优化来扰动更接近训练空间的合成样本的情况。结果表明，攻击者通过选择可能代表记忆的训练样本的合成样本，确实可以构成重大的隐私风险。此外，我们注意到，当攻击者知道或拥有对生成模型的黑盒访问权限时，隐私威胁会显著增加。我们还发现，通过多目标优化进行的重建攻击甚至增加了识别机密样本的风险。



## **45. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

无模型是最好的模型：生成纯代码实现来替换设备上的DL模型 cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2403.16479v2) [paper-pdf](http://arxiv.org/pdf/2403.16479v2)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.8% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.

摘要: 最近的研究表明，部署的深度学习(DL)模型，如张量流精简(TFLite)模型，可以很容易地被攻击者从现实世界的应用和设备中提取出来，从而产生多种攻击，如对抗性攻击。尽管保护部署在设备上的DL模型越来越受到关注，但没有一种现有方法可以完全防止上述威胁。传统的软件保护技术已经得到了广泛的探索，如果设备上的模型可以用纯代码实现，如C++，这将打开重用现有软件保护技术的可能性。然而，由于DL模型的复杂性，目前还没有一种自动的方法可以将DL模型转换为纯代码。为了填补这一空白，我们提出了一种新的方法CustomDLCoder，它可以自动提取设备上的模型信息，并为广泛的DL模型合成定制的可执行程序。CustomDLCoder首先解析DL模型，提取其后端计算单元，将计算单元配置为图形，然后生成定制代码来实现和部署ML解决方案，而不需要显式的模型表示。合成的程序隐藏了DL部署环境的模型信息，因为它不需要保留显式的模型表示，从而防止了对DL模型的许多攻击。此外，它还提高了ML的性能，因为定制的代码删除了模型解析和预处理步骤，只保留了数据计算过程。我们的实验结果表明，CustomDLCoder通过禁止设备上的模型嗅探提高了模型的安全性。在x86-64和ARM64平台上，与原有的设备上平台(即TFLite)相比，该方法的模型推理速度分别提高了21.8%和24.3%。最重要的是，它可以在x86-64和ARM64平台上分别显著降低68.8%和36.0%的内存消耗。



## **46. Embodied Active Defense: Leveraging Recurrent Feedback to Counter Adversarial Patches**

主动防御：利用循环反馈对抗对抗补丁 cs.CV

27pages

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.00540v1) [paper-pdf](http://arxiv.org/pdf/2404.00540v1)

**Authors**: Lingxuan Wu, Xiao Yang, Yinpeng Dong, Liuwei Xie, Hang Su, Jun Zhu

**Abstract**: The vulnerability of deep neural networks to adversarial patches has motivated numerous defense strategies for boosting model robustness. However, the prevailing defenses depend on single observation or pre-established adversary information to counter adversarial patches, often failing to be confronted with unseen or adaptive adversarial attacks and easily exhibiting unsatisfying performance in dynamic 3D environments. Inspired by active human perception and recurrent feedback mechanisms, we develop Embodied Active Defense (EAD), a proactive defensive strategy that actively contextualizes environmental information to address misaligned adversarial patches in 3D real-world settings. To achieve this, EAD develops two central recurrent sub-modules, i.e., a perception module and a policy module, to implement two critical functions of active vision. These models recurrently process a series of beliefs and observations, facilitating progressive refinement of their comprehension of the target object and enabling the development of strategic actions to counter adversarial patches in 3D environments. To optimize learning efficiency, we incorporate a differentiable approximation of environmental dynamics and deploy patches that are agnostic to the adversary strategies. Extensive experiments demonstrate that EAD substantially enhances robustness against a variety of patches within just a few steps through its action policy in safety-critical tasks (e.g., face recognition and object detection), without compromising standard accuracy. Furthermore, due to the attack-agnostic characteristic, EAD facilitates excellent generalization to unseen attacks, diminishing the averaged attack success rate by 95 percent across a range of unseen adversarial attacks.

摘要: 深度神经网络对敌意补丁的脆弱性激发了许多增强模型稳健性的防御策略。然而，主流的防御依赖于单一的观察或预先建立的对手信息来对抗对抗性补丁，往往无法对抗看不见的或自适应的对抗性攻击，并且在动态3D环境中很容易表现出不令人满意的性能。受人类主动感知和循环反馈机制的启发，我们开发了体现主动防御(EAD)，这是一种主动防御策略，它主动地将环境信息与上下文关联起来，以应对3D现实世界中未对齐的敌方补丁。为了实现这一目标，EAD开发了两个中央循环子模块，即感知模块和政策模块，以实现主动视觉的两个关键功能。这些模型反复处理一系列信念和观察，有助于逐步完善其对目标对象的理解，并使开发战略行动来对抗3D环境中的敌意补丁。为了优化学习效率，我们结合了环境动态的可微近似，并部署了与对手策略无关的补丁。大量实验表明，EAD通过其在安全关键任务(如人脸识别和目标检测)中的操作策略，在不影响标准准确性的情况下，仅需几个步骤即可显著增强针对各种补丁的稳健性。此外，由于攻击不可知的特性，EAD有助于对看不见的攻击进行出色的泛化，使一系列看不见的对抗性攻击的平均攻击成功率降低95%。



## **47. An Unsupervised Adversarial Autoencoder for Cyber Attack Detection in Power Distribution Grids**

一种用于配电网网络攻击检测的无监督对抗自动编码器 cs.CR

**SubmitDate**: 2024-03-31    [abs](http://arxiv.org/abs/2404.02923v1) [paper-pdf](http://arxiv.org/pdf/2404.02923v1)

**Authors**: Mehdi Jabbari Zideh, Mohammad Reza Khalghani, Sarika Khushalani Solanki

**Abstract**: Detection of cyber attacks in smart power distribution grids with unbalanced configurations poses challenges due to the inherent nonlinear nature of these uncertain and stochastic systems. It originates from the intermittent characteristics of the distributed energy resources (DERs) generation and load variations. Moreover, the unknown behavior of cyber attacks, especially false data injection attacks (FDIAs) in the distribution grids with complex temporal correlations and the limited amount of labeled data increases the vulnerability of the grids and imposes a high risk in the secure and reliable operation of the grids. To address these challenges, this paper proposes an unsupervised adversarial autoencoder (AAE) model to detect FDIAs in unbalanced power distribution grids integrated with DERs, i.e., PV systems and wind generation. The proposed method utilizes long short-term memory (LSTM) in the structure of the autoencoder to capture the temporal dependencies in the time-series measurements and leverages the power of generative adversarial networks (GANs) for better reconstruction of the input data. The advantage of the proposed data-driven model is that it can detect anomalous points for the system operation without reliance on abstract models or mathematical representations. To evaluate the efficacy of the approach, it is tested on IEEE 13-bus and 123-bus systems with historical meteorological data (wind speed, ambient temperature, and solar irradiance) as well as historical real-world load data under three types of data falsification functions. The comparison of the detection results of the proposed model with other unsupervised learning methods verifies its superior performance in detecting cyber attacks in unbalanced power distribution grids.

摘要: 由于这些不确定和随机系统固有的非线性特性，在具有不平衡配置的智能配电网中检测网络攻击是一项挑战。它源于分布式能源发电和负荷变化的间歇性特征。此外，网络攻击的未知行为，特别是配电网中的虚假数据注入攻击(FDIA)，具有复杂的时间相关性和有限的标签数据量，增加了网格的脆弱性，给网格的安全可靠运行带来了很高的风险。为了应对这些挑战，提出了一种无监督对抗性自动编码器(AAE)模型来检测包含DER的不平衡配电网中的FDIA，即光伏系统和风力发电。该方法利用自动编码器结构中的长短期记忆(LSTM)来捕获时间序列测量中的时间相关性，并利用生成性对抗网络(GANS)的能力来更好地重构输入数据。提出的数据驱动模型的优点是，它可以检测系统运行的异常点，而不依赖于抽象的模型或数学表示。为了评估该方法的有效性，在IEEE 13节点和123节点系统上测试了三种数据伪造函数下的历史气象数据(风速、环境温度和太阳辐照度)以及历史真实负荷数据。将该模型的检测结果与其他非监督学习方法的检测结果进行了比较，验证了该模型在检测不平衡配电网网络攻击方面的优越性。



## **48. AttackNet: Enhancing Biometric Security via Tailored Convolutional Neural Network Architectures for Liveness Detection**

AttackNet：通过定制的卷积神经网络架构增强生物识别安全性，用于活性检测 cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.03769v2) [paper-pdf](http://arxiv.org/pdf/2402.03769v2)

**Authors**: Oleksandr Kuznetsov, Dmytro Zakharov, Emanuele Frontoni, Andrea Maranesi

**Abstract**: Biometric security is the cornerstone of modern identity verification and authentication systems, where the integrity and reliability of biometric samples is of paramount importance. This paper introduces AttackNet, a bespoke Convolutional Neural Network architecture, meticulously designed to combat spoofing threats in biometric systems. Rooted in deep learning methodologies, this model offers a layered defense mechanism, seamlessly transitioning from low-level feature extraction to high-level pattern discernment. Three distinctive architectural phases form the crux of the model, each underpinned by judiciously chosen activation functions, normalization techniques, and dropout layers to ensure robustness and resilience against adversarial attacks. Benchmarking our model across diverse datasets affirms its prowess, showcasing superior performance metrics in comparison to contemporary models. Furthermore, a detailed comparative analysis accentuates the model's efficacy, drawing parallels with prevailing state-of-the-art methodologies. Through iterative refinement and an informed architectural strategy, AttackNet underscores the potential of deep learning in safeguarding the future of biometric security.

摘要: 生物特征安全是现代身份验证和认证系统的基石，其中生物特征样本的完整性和可靠性至关重要。本文介绍了AttackNet，一种定制的卷积神经网络结构，精心设计用于对抗生物识别系统中的欺骗威胁。该模型植根于深度学习方法，提供了一种分层防御机制，从低级特征提取无缝过渡到高级模式识别。三个不同的体系结构阶段构成了该模型的核心，每个阶段都以明智地选择的激活函数、归一化技术和丢弃层为基础，以确保对对手攻击的健壮性和弹性。在不同的数据集上对我们的模型进行基准测试，肯定了它的威力，展示了与当代模型相比的卓越性能指标。此外，一项详细的比较分析强调了该模型的有效性，将其与流行的最先进的方法进行了比较。通过迭代改进和明智的架构策略，AttackNet强调了深度学习在保障生物识别安全的未来方面的潜力。



## **49. Bidirectional Consistency Models**

双向一致性模型 cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2403.18035v2) [paper-pdf](http://arxiv.org/pdf/2403.18035v2)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, largely reducing the number of iterations. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.

摘要: 扩散模型(DM)能够通过迭代地对随机向量去噪来生成非常高质量的样本，该过程对应于沿着概率流常微分方程式(PF ODE)移动。有趣的是，DM还可以通过沿PF ODE向后移动来将输入图像反转为噪声，这是下游任务(如插补和图像编辑)的关键操作。然而，这一过程的迭代性质限制了其速度，阻碍了其更广泛的应用。最近，一致性模型(CM)已经出现，通过近似PF ODE的积分来解决这一挑战，大大减少了迭代次数。然而，由于没有显式的常微分方程组解算器，使得反演过程变得更加复杂。为了解决这个问题，我们引入了双向一致性模型(BCM)，它学习一个单一的神经网络，允许沿着PF ODE进行前向和后向遍历，有效地将生成和反转任务统一在一个框架内。值得注意的是，我们提出的方法支持一步生成和反转，同时还允许使用额外的步骤来提高生成质量或减少重建误差。此外，通过利用模型的双向一致性，我们引入了一种采样策略，该策略可以在保留生成的图像内容的同时增强FID。我们进一步展示了我们的模型在几个下游任务中的能力，如插补和修复，并展示了潜在的应用程序，包括压缩图像的盲恢复和防御黑盒攻击。



## **50. STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario**

STBA：查询受限黑盒情形下DNN鲁棒性评估 cs.CV

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2404.00362v1) [paper-pdf](http://arxiv.org/pdf/2404.00362v1)

**Authors**: Renyang Liu, Kwok-Yan Lam, Wei Zhou, Sixing Wu, Jun Zhao, Dongting Hu, Mingming Gong

**Abstract**: Many attack techniques have been proposed to explore the vulnerability of DNNs and further help to improve their robustness. Despite the significant progress made recently, existing black-box attack methods still suffer from unsatisfactory performance due to the vast number of queries needed to optimize desired perturbations. Besides, the other critical challenge is that adversarial examples built in a noise-adding manner are abnormal and struggle to successfully attack robust models, whose robustness is enhanced by adversarial training against small perturbations. There is no doubt that these two issues mentioned above will significantly increase the risk of exposure and result in a failure to dig deeply into the vulnerability of DNNs. Hence, it is necessary to evaluate DNNs' fragility sufficiently under query-limited settings in a non-additional way. In this paper, we propose the Spatial Transform Black-box Attack (STBA), a novel framework to craft formidable adversarial examples in the query-limited scenario. Specifically, STBA introduces a flow field to the high-frequency part of clean images to generate adversarial examples and adopts the following two processes to enhance their naturalness and significantly improve the query efficiency: a) we apply an estimated flow field to the high-frequency part of clean images to generate adversarial examples instead of introducing external noise to the benign image, and b) we leverage an efficient gradient estimation method based on a batch of samples to optimize such an ideal flow field under query-limited settings. Compared to existing score-based black-box baselines, extensive experiments indicated that STBA could effectively improve the imperceptibility of the adversarial examples and remarkably boost the attack success rate under query-limited settings.

摘要: 已经提出了许多攻击技术来探索DNN的脆弱性，并进一步帮助提高它们的健壮性。尽管最近取得了显著的进展，但现有的黑盒攻击方法仍然存在性能不佳的问题，这是因为需要大量的查询来优化期望的扰动。此外，另一个关键的挑战是，以添加噪声的方式构建的对抗性样本是不正常的，并且难以成功地攻击健壮模型，而健壮模型的健壮性通过对抗小扰动的对抗性训练来增强。毫无疑问，上述两个问题将大大增加暴露的风险，并导致无法深入挖掘DNN的脆弱性。因此，有必要以一种非额外的方式充分评估DNN在查询受限设置下的脆弱性。在本文中，我们提出了空间变换黑盒攻击(STBA)，这是一个新的框架，可以在查询受限的情况下创建强大的对手示例。具体地说，STBA在清洁图像的高频部分引入了流场来生成对抗性实例，并采用了以下两个过程来增强其自然性，显著提高了查询效率：a)将估计的流场应用于干净图像的高频部分来生成对抗性实例，而不是在良性图像中引入外部噪声；b)在查询受限的情况下，利用一种基于批量样本的高效梯度估计方法来优化这样的理想流场。大量实验表明，与已有的基于分数的黑盒基线相比，STBA能够有效地提高对抗性实例的隐蔽性，显著提高查询受限环境下的攻击成功率。



