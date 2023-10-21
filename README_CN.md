# Latest Adversarial Attack Papers
**update at 2023-10-21 11:22:52**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. OODRobustBench: benchmarking and analyzing adversarial robustness under distribution shift**

OODRobustBch：分布转移下的对手健壮性基准测试与分析 cs.LG

in submission

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12793v1) [paper-pdf](http://arxiv.org/pdf/2310.12793v1)

**Authors**: Lin Li, Yifei Wang, Chawin Sitawarin, Michael Spratling

**Abstract**: Existing works have made great progress in improving adversarial robustness, but typically test their method only on data from the same distribution as the training data, i.e. in-distribution (ID) testing. As a result, it is unclear how such robustness generalizes under input distribution shifts, i.e. out-of-distribution (OOD) testing. This is a concerning omission as such distribution shifts are unavoidable when methods are deployed in the wild. To address this issue we propose a benchmark named OODRobustBench to comprehensively assess OOD adversarial robustness using 23 dataset-wise shifts (i.e. naturalistic shifts in input distribution) and 6 threat-wise shifts (i.e., unforeseen adversarial threat models). OODRobustBench is used to assess 706 robust models using 60.7K adversarial evaluations. This large-scale analysis shows that: 1) adversarial robustness suffers from a severe OOD generalization issue; 2) ID robustness correlates strongly with OOD robustness, in a positive linear way, under many distribution shifts. The latter enables the prediction of OOD robustness from ID robustness. Based on this, we are able to predict the upper limit of OOD robustness for existing robust training schemes. The results suggest that achieving OOD robustness requires designing novel methods beyond the conventional ones. Last, we discover that extra data, data augmentation, advanced model architectures and particular regularization approaches can improve OOD robustness. Noticeably, the discovered training schemes, compared to the baseline, exhibit dramatically higher robustness under threat shift while keeping high ID robustness, demonstrating new promising solutions for robustness against both multi-attack and unforeseen attacks.

摘要: 现有的工作在提高对手的稳健性方面已经取得了很大的进展，但通常只在来自与训练数据相同的分布的数据上测试他们的方法，即内分布(ID)测试。因此，目前还不清楚这种稳健性如何在输入分布漂移(即分布外(OOD)测试)下得到推广。这是一个令人担忧的遗漏，因为当方法部署在野外时，这种分布变化是不可避免的。为了解决这一问题，我们提出了一个名为OODRobustBch的基准，该基准使用23个数据集方向的变化(即输入分布的自然变化)和6个威胁方向的变化(即不可预见的对手威胁模型)来综合评估OOD对手威胁的健壮性。使用60.7K的对抗性评估，OODRobustBch被用来评估706个稳健模型。这一大规模的分析表明：1)对手的健壮性受到严重的OOD泛化问题的影响；2)ID健壮性与OOD健壮性在许多分布变化下以正线性的方式强烈相关。后者使得能够从ID健壮性预测OOD健壮性。在此基础上，我们能够预测现有健壮训练方案的OOD稳健性上限。结果表明，要实现面向对象设计的健壮性，需要设计出超越传统方法的新方法。最后，我们发现额外的数据、数据扩充、先进的模型结构和特定的正则化方法可以提高面向对象设计的健壮性。值得注意的是，与基线相比，发现的训练方案在保持高ID稳健性的同时，在威胁转移下表现出显著更高的稳健性，展示了针对多攻击和不可预见攻击的稳健性的新的有前途的解决方案。



## **2. TorKameleon: Improving Tor's Censorship Resistance with K-anonymization and Media-based Covert Channels**

TorKameleon：通过K-匿名化和基于媒体的隐蔽渠道提高Tor的审查抵抗力 cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2303.17544v3) [paper-pdf](http://arxiv.org/pdf/2303.17544v3)

**Authors**: Afonso Vilalonga, João S. Resende, Henrique Domingos

**Abstract**: Anonymity networks like Tor significantly enhance online privacy but are vulnerable to correlation attacks by state-level adversaries. While covert channels encapsulated in media protocols, particularly WebRTC-based encapsulation, have demonstrated effectiveness against passive traffic correlation attacks, their resilience against active correlation attacks remains unexplored, and their compatibility with Tor has been limited. This paper introduces TorKameleon, a censorship evasion solution designed to protect Tor users from both passive and active correlation attacks. TorKameleon employs K-anonymization techniques to fragment and reroute traffic through multiple TorKameleon proxies, while also utilizing covert WebRTC-based channels or TLS tunnels to encapsulate user traffic.

摘要: 像Tor这样的匿名网络极大地增强了在线隐私，但容易受到国家级对手的关联攻击。虽然封装在媒体协议中的隐蔽信道，特别是基于WebRTC的封装，已经证明了对被动流量关联攻击的有效性，但它们对主动关联攻击的弹性还没有被探索，并且它们与ToR的兼容性已经受到限制。本文介绍了TorKameleon，一个旨在保护Tor用户免受被动和主动相关攻击的审查逃避解决方案。TorKameleon使用K-匿名技术通过多个TorKameleon代理对流量进行分段和重新路由，同时还利用基于WebRTC的隐蔽通道或TLS隧道来封装用户流量。



## **3. WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks**

WaveAttack：基于非对称频率混淆的深层神经网络后门攻击 cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.11595v2) [paper-pdf](http://arxiv.org/pdf/2310.11595v2)

**Authors**: Jun Xia, Zhihao Yue, Yingbo Zhou, Zhiwei Ling, Xian Wei, Mingsong Chen

**Abstract**: Due to the popularity of Artificial Intelligence (AI) technology, numerous backdoor attacks are designed by adversaries to mislead deep neural network predictions by manipulating training samples and training processes. Although backdoor attacks are effective in various real scenarios, they still suffer from the problems of both low fidelity of poisoned samples and non-negligible transfer in latent space, which make them easily detectable by existing backdoor detection algorithms. To overcome the weakness, this paper proposes a novel frequency-based backdoor attack method named WaveAttack, which obtains image high-frequency features through Discrete Wavelet Transform (DWT) to generate backdoor triggers. Furthermore, we introduce an asymmetric frequency obfuscation method, which can add an adaptive residual in the training and inference stage to improve the impact of triggers and further enhance the effectiveness of WaveAttack. Comprehensive experimental results show that WaveAttack not only achieves higher stealthiness and effectiveness, but also outperforms state-of-the-art (SOTA) backdoor attack methods in the fidelity of images by up to 28.27\% improvement in PSNR, 1.61\% improvement in SSIM, and 70.59\% reduction in IS.

摘要: 由于人工智能(AI)技术的普及，许多后门攻击都是由对手设计的，通过操纵训练样本和训练过程来误导深度神经网络预测。虽然后门攻击在各种真实场景中都是有效的，但它们仍然存在有毒样本保真度低和潜在空间传输不可忽略的问题，这使得它们很容易被现有的后门检测算法检测到。针对这一缺陷，提出了一种新的基于频率的后门攻击方法WaveAttack，该方法通过离散小波变换(DWT)提取图像高频特征来生成后门触发器。此外，我们还引入了一种非对称频率混淆方法，在训练和推理阶段加入自适应残差，以改善触发的影响，进一步增强WaveAttack的有效性。综合实验结果表明，WaveAttack不仅获得了更高的隐蔽性和有效性，而且在图像保真度方面优于最新的SOTA后门攻击方法，峰值信噪比提高了28.27，SSIM提高了1.61，IS降低了70.59。



## **4. Learn from the Past: A Proxy based Adversarial Defense Framework to Boost Robustness**

借鉴过去：一种基于代理的增强健壮性的对抗防御框架 cs.LG

16 Pages

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12713v1) [paper-pdf](http://arxiv.org/pdf/2310.12713v1)

**Authors**: Yaohua Liu, Jiaxin Gao, Zhu Liu, Xianghao Jiao, Xin Fan, Risheng Liu

**Abstract**: In light of the vulnerability of deep learning models to adversarial samples and the ensuing security issues, a range of methods, including Adversarial Training (AT) as a prominent representative, aimed at enhancing model robustness against various adversarial attacks, have seen rapid development. However, existing methods essentially assist the current state of target model to defend against parameter-oriented adversarial attacks with explicit or implicit computation burdens, which also suffers from unstable convergence behavior due to inconsistency of optimization trajectories. Diverging from previous work, this paper reconsiders the update rule of target model and corresponding deficiency to defend based on its current state. By introducing the historical state of the target model as a proxy, which is endowed with much prior information for defense, we formulate a two-stage update rule, resulting in a general adversarial defense framework, which we refer to as `LAST' ({\bf L}earn from the P{\bf ast}). Besides, we devise a Self Distillation (SD) based defense objective to constrain the update process of the proxy model without the introduction of larger teacher models. Experimentally, we demonstrate consistent and significant performance enhancements by refining a series of single-step and multi-step AT methods (e.g., up to $\bf 9.2\%$ and $\bf 20.5\%$ improvement of Robust Accuracy (RA) on CIFAR10 and CIFAR100 datasets, respectively) across various datasets, backbones and attack modalities, and validate its ability to enhance training stability and ameliorate catastrophic overfitting issues meanwhile.

摘要: 鉴于深度学习模型对对抗性样本的脆弱性以及随之而来的安全问题，旨在增强模型对各种对抗性攻击的稳健性的一系列方法，包括作为突出代表的对抗性训练(AT)，得到了迅速的发展。然而，现有的方法本质上是帮助目标模型的当前状态来防御具有显式或隐式计算负担的面向参数的对抗性攻击，而这种对抗性攻击也存在由于优化轨迹不一致而导致的不稳定收敛行为。与前人的工作不同，本文针对目标模型的现状，重新考虑了目标模型的更新规则以及相应的不足进行防御。通过引入目标模型的历史状态作为代理，赋予目标模型大量的先验信息用于防御，我们制定了一个两阶段更新规则，从而得到了一个通用的对抗性防御框架，我们称之为`last‘(L从P{\bf ast}中赚取)。此外，我们设计了一个基于自蒸馏(SD)的防御目标来约束代理模型的更新过程，而不需要引入更大的教师模型。在实验上，我们通过提炼一系列单步和多步AT方法(例如，分别在CIFAR10和CIFAR100数据集上提高稳健精度(RA)，分别高达9.2美元和20.5美元)，展示了持续和显著的性能提升，并验证了其增强训练稳定性和改善灾难性过拟合问题的能力。



## **5. Generating Robust Adversarial Examples against Online Social Networks (OSNs)**

生成针对在线社交网络(OSN)的强大敌意示例 cs.MM

26 pages, 9 figures

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12708v1) [paper-pdf](http://arxiv.org/pdf/2310.12708v1)

**Authors**: Jun Liu, Jiantao Zhou, Haiwei Wu, Weiwei Sun, Jinyu Tian

**Abstract**: Online Social Networks (OSNs) have blossomed into prevailing transmission channels for images in the modern era. Adversarial examples (AEs) deliberately designed to mislead deep neural networks (DNNs) are found to be fragile against the inevitable lossy operations conducted by OSNs. As a result, the AEs would lose their attack capabilities after being transmitted over OSNs. In this work, we aim to design a new framework for generating robust AEs that can survive the OSN transmission; namely, the AEs before and after the OSN transmission both possess strong attack capabilities. To this end, we first propose a differentiable network termed SImulated OSN (SIO) to simulate the various operations conducted by an OSN. Specifically, the SIO network consists of two modules: 1) a differentiable JPEG layer for approximating the ubiquitous JPEG compression and 2) an encoder-decoder subnetwork for mimicking the remaining operations. Based upon the SIO network, we then formulate an optimization framework to generate robust AEs by enforcing model outputs with and without passing through the SIO to be both misled. Extensive experiments conducted over Facebook, WeChat and QQ demonstrate that our attack methods produce more robust AEs than existing approaches, especially under small distortion constraints; the performance gain in terms of Attack Success Rate (ASR) could be more than 60%. Furthermore, we build a public dataset containing more than 10,000 pairs of AEs processed by Facebook, WeChat or QQ, facilitating future research in the robust AEs generation. The dataset and code are available at https://github.com/csjunjun/RobustOSNAttack.git.

摘要: 在现代，在线社交网络(OSN)已经发展成为流行的图像传输渠道。对抗性例子(AE)被发现被故意设计来误导深度神经网络(DNN)，对OSN进行的不可避免的有损操作是脆弱的。因此，在通过OSN传输后，AE将失去攻击能力。在这项工作中，我们的目标是设计一种新的框架来生成能够在OSN传输中幸存下来的健壮的AE，即OSN传输前后的AE都具有很强的攻击能力。为此，我们首先提出了一种称为模拟OSN(SIO)的可区分网络来模拟OSN进行的各种操作。具体地说，SIO网络由两个模块组成：1)用于近似普遍存在的JPEG压缩的可区分JPEG层和2)用于模拟其余操作的编解码子网络。然后，基于SIO网络，我们制定了一个优化框架，通过强制模型输出在通过和不通过SIO的情况下都被误导来生成稳健的AE。在脸书、微信和QQ上进行的大量实验表明，我们的攻击方法比现有的方法产生了更强的攻击效果，特别是在小失真约束下；攻击成功率方面的性能增益可以超过60%。此外，我们还建立了一个公共数据集，其中包含了超过10,000对由脸书、微信或QQ处理的实体实体，为未来稳健的实体实体的研究提供了便利。数据集和代码可在https://github.com/csjunjun/RobustOSNAttack.git.上获得



## **6. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

刀片：联邦学习中拜占庭攻击和防御的统一基准套件 cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2206.05359v3) [paper-pdf](http://arxiv.org/pdf/2206.05359v3)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across clients, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial (Byzantine) clients aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques.   This paper investigates existing techniques in Byzantine-resilient FL and introduces an open-source benchmark suite for convenient and fair performance comparisons. Our investigation begins with a systematic study of Byzantine attack and defense strategies. Subsequently, we present \ours, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. The design of \ours incorporates key characteristics derived from our systematic study, encompassing the attacker's capabilities and knowledge, defense strategy categories, and factors influencing robustness. Blades contains built-in implementations of representative attack and defense strategies and offers user-friendly interfaces for seamlessly integrating new ideas.

摘要: 联合学习(FL)促进了跨客户的分布式培训，保护了他们的数据隐私。FL固有的分布式结构引入了漏洞，特别是来自敌意(拜占庭)客户端的漏洞，旨在歪曲本地更新以利于其优势。尽管有太多关于拜占庭式外语的研究，但学术界还没有建立一个全面的基准套件，这是公正评估和比较不同技术的关键。本文研究了拜占庭弹性FL中的现有技术，并介绍了一个开源的基准测试套件，用于方便和公平地进行性能比较。我们的调查始于对拜占庭攻防战略的系统研究。随后，我们提出了一个可伸缩、可扩展且易于配置的基准测试套件，它支持研究人员和开发人员在拜占庭弹性FL中有效地实现和验证针对基线算法的新策略。我们的设计包含了来自我们系统研究的关键特征，包括攻击者的能力和知识、防御策略类别和影响健壮性的因素。Blade包含典型攻击和防御策略的内置实现，并提供用户友好的界面，以无缝集成新想法。



## **7. Automatic Hallucination Assessment for Aligned Large Language Models via Transferable Adversarial Attacks**

基于可转移对抗性攻击的对齐大语言模型的自动幻觉评估 cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12516v1) [paper-pdf](http://arxiv.org/pdf/2310.12516v1)

**Authors**: Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao

**Abstract**: Although remarkable progress has been achieved in preventing large language model (LLM) hallucinations using instruction tuning and retrieval augmentation, it remains challenging to measure the reliability of LLMs using human-crafted evaluation data which is not available for many tasks and domains and could suffer from data leakage. Inspired by adversarial machine learning, this paper aims to develop a method of automatically generating evaluation data by appropriately modifying existing data on which LLMs behave faithfully. Specifically, this paper presents AutoDebug, an LLM-based framework to use prompting chaining to generate transferable adversarial attacks in the form of question-answering examples. We seek to understand the extent to which these examples trigger the hallucination behaviors of LLMs.   We implement AutoDebug using ChatGPT and evaluate the resulting two variants of a popular open-domain question-answering dataset, Natural Questions (NQ), on a collection of open-source and proprietary LLMs under various prompting settings. Our generated evaluation data is human-readable and, as we show, humans can answer these modified questions well. Nevertheless, we observe pronounced accuracy drops across multiple LLMs including GPT-4. Our experimental results show that LLMs are likely to hallucinate in two categories of question-answering scenarios where (1) there are conflicts between knowledge given in the prompt and their parametric knowledge, or (2) the knowledge expressed in the prompt is complex. Finally, we find that the adversarial examples generated by our method are transferable across all considered LLMs. The examples generated by a small model can be used to debug a much larger model, making our approach cost-effective.

摘要: 虽然在使用指令调整和提取增强来防止大语言模型(LLM)幻觉方面取得了显著的进展，但使用人工制作的评估数据来衡量LLMS的可靠性仍然是具有挑战性的，这些数据对于许多任务和领域都是不可用的，并且可能受到数据泄漏的影响。受对抗性机器学习的启发，本文旨在开发一种自动生成评价数据的方法，该方法通过适当修改现有的数据来忠实地执行LLMS。具体地说，本文提出了AutoDebug，这是一个基于LLM的框架，使用提示链以问答示例的形式生成可转移的对抗性攻击。我们试图了解这些例子在多大程度上触发了LLM的幻觉行为。我们使用ChatGPT实现了AutoDebug，并在各种提示设置下，在一组开源和专有LLM上评估了一个流行的开放领域问答数据集的两个变体-自然问题(Natural Questions，NQ)。我们生成的评估数据是人类可读的，如我们所示，人类可以很好地回答这些修改后的问题。然而，我们观察到包括GPT-4在内的多个LLM的准确率显著下降。我们的实验结果表明，在两种类型的问答场景中，LLM可能会产生幻觉：(1)提示中给出的知识与其参数知识之间存在冲突；(2)提示中表达的知识复杂。最后，我们发现，我们的方法生成的对抗性例子可以在所有考虑的LLM之间转移。由小模型生成的示例可用于调试大得多的模型，从而使我们的方法具有成本效益。



## **8. Red Teaming Language Model Detectors with Language Models**

具有语言模型的Red Teaming语言模型检测器 cs.CL

Preprint. Accepted by TACL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2305.19713v2) [paper-pdf](http://arxiv.org/pdf/2305.19713v2)

**Authors**: Zhouxing Shi, Yihan Wang, Fan Yin, Xiangning Chen, Kai-Wei Chang, Cho-Jui Hsieh

**Abstract**: The prevalence and strong capability of large language models (LLMs) present significant safety and ethical risks if exploited by malicious users. To prevent the potentially deceptive usage of LLMs, recent works have proposed algorithms to detect LLM-generated text and protect LLMs. In this paper, we investigate the robustness and reliability of these LLM detectors under adversarial attacks. We study two types of attack strategies: 1) replacing certain words in an LLM's output with their synonyms given the context; 2) automatically searching for an instructional prompt to alter the writing style of the generation. In both strategies, we leverage an auxiliary LLM to generate the word replacements or the instructional prompt. Different from previous works, we consider a challenging setting where the auxiliary LLM can also be protected by a detector. Experiments reveal that our attacks effectively compromise the performance of all detectors in the study with plausible generations, underscoring the urgent need to improve the robustness of LLM-generated text detection systems.

摘要: 如果被恶意用户利用，大语言模型(LLM)的流行和强大的能力会带来巨大的安全和道德风险。为了防止潜在的欺骗性使用LLMS，最近的工作提出了检测LLM生成的文本并保护LLMS的算法。在本文中，我们研究了这些LLM检测器在对抗攻击下的健壮性和可靠性。我们研究了两种类型的攻击策略：1)用给定上下文的同义词替换LLM输出中的某些单词；2)自动搜索指令提示以改变生成的写作风格。在这两种策略中，我们利用辅助LLM来生成单词替换或指令提示。与以前的工作不同，我们考虑了一个具有挑战性的设置，其中辅助LLM也可以受到探测器的保护。实验表明，我们的攻击有效地折衷了研究中所有检测器的性能，生成了看似合理的代码，这突显了提高LLM生成的文本检测系统的健壮性的迫切需要。



## **9. Unraveling the Connections between Privacy and Certified Robustness in Federated Learning Against Poisoning Attacks**

解开联合学习抗中毒攻击中隐私与认证稳健性之间的联系 cs.CR

ACM CCS 2023

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2209.04030v2) [paper-pdf](http://arxiv.org/pdf/2209.04030v2)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Qinbin Li, Sanmi Koyejo, Bo Li

**Abstract**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As local training data comes from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is usually trained in a differentially private way (DPFL). Thus, in this paper, we ask: What are the underlying connections between differential privacy and certified robustness in FL against poisoning attacks? Can we leverage the innate privacy property of DPFL to provide certified robustness for FL? Can we further improve the privacy of FL to improve such robustness certification? We first investigate both user-level and instance-level privacy of FL and provide formal privacy analysis to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack inefficacy for DPFL on both user and instance levels. Theoretically, we provide the certified robustness of DPFL based on both criteria given a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of poisoning attacks on different datasets. We find that increasing the level of privacy protection in DPFL results in stronger certified attack inefficacy; however, it does not necessarily lead to a stronger certified prediction. Thus, achieving the optimal certified prediction requires a proper balance between privacy and utility loss.

摘要: 联合学习(FL)提供了一种有效的范例来联合训练利用来自分布式用户的数据的全局模型。由于本地训练数据来自可能不值得信任的不同用户，多项研究表明，FL容易受到中毒攻击。同时，为了保护本地用户的隐私，FL通常会以一种不同的私人方式进行培训(DPFL)。因此，在这篇文章中，我们问：区别隐私和FL对中毒攻击的认证健壮性之间有什么潜在的联系？我们能否利用DPFL与生俱来的隐私属性为FL提供经过认证的健壮性？我们能否进一步改善FL的隐私，以提高这种健壮性认证？我们首先对FL的用户级和实例级隐私进行了研究，并提供了形式化的隐私分析，以实现改进的实例级隐私。然后，我们提供了两个健壮性认证标准：DPFL在用户和实例级别上的认证预测和认证攻击无效。理论上，在给定有限数量的敌意用户或实例的情况下，我们基于这两个标准提供了DPFL的证明的健壮性。在经验上，我们在不同数据集的一系列中毒攻击下进行了广泛的实验来验证我们的理论。我们发现，增加DPFL中的隐私保护级别会导致更强的认证攻击无效；然而，这并不一定会导致更强的认证预测。因此，要实现最佳验证预测，需要在隐私和效用损失之间取得适当的平衡。



## **10. CAT: Closed-loop Adversarial Training for Safe End-to-End Driving**

CAT：端到端安全驾驶的闭环对抗性训练 cs.LG

7th Conference on Robot Learning (CoRL 2023)

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12432v1) [paper-pdf](http://arxiv.org/pdf/2310.12432v1)

**Authors**: Linrui Zhang, Zhenghao Peng, Quanyi Li, Bolei Zhou

**Abstract**: Driving safety is a top priority for autonomous vehicles. Orthogonal to prior work handling accident-prone traffic events by algorithm designs at the policy level, we investigate a Closed-loop Adversarial Training (CAT) framework for safe end-to-end driving in this paper through the lens of environment augmentation. CAT aims to continuously improve the safety of driving agents by training the agent on safety-critical scenarios that are dynamically generated over time. A novel resampling technique is developed to turn log-replay real-world driving scenarios into safety-critical ones via probabilistic factorization, where the adversarial traffic generation is modeled as the multiplication of standard motion prediction sub-problems. Consequently, CAT can launch more efficient physical attacks compared to existing safety-critical scenario generation methods and yields a significantly less computational cost in the iterative learning pipeline. We incorporate CAT into the MetaDrive simulator and validate our approach on hundreds of driving scenarios imported from real-world driving datasets. Experimental results demonstrate that CAT can effectively generate adversarial scenarios countering the agent being trained. After training, the agent can achieve superior driving safety in both log-replay and safety-critical traffic scenarios on the held-out test set. Code and data are available at https://metadriverse.github.io/cat.

摘要: 驾驶安全是自动驾驶汽车的首要任务。与以往处理事故多发交通事件的算法设计在策略层面上的工作正交，本文从环境增强的角度研究了一种端到端安全驾驶的闭环对抗性训练(CAT)框架。CAT旨在通过对驾驶代理进行安全关键场景培训，不断提高驾驶代理的安全性，这些场景是随着时间的推移动态生成的。提出了一种新的重采样技术，通过概率因式分解将日志重放的真实驾驶场景转化为安全关键场景，其中对抗性流量的生成被建模为标准运动预测子问题的乘法。因此，与现有的安全关键场景生成方法相比，CAT可以发起更有效的物理攻击，并在迭代学习管道中产生显著更低的计算成本。我们将CAT集成到MetaDrive模拟器中，并在从真实驾驶数据集导入的数百个驾驶场景上验证了我们的方法。实验结果表明，CAT能够有效地生成对抗被训练智能体的对抗性场景。经过培训后，该代理可以在日志重放和安全关键交通场景中在坚持测试集上实现卓越的驾驶安全。有关代码和数据，请访问https://metadriverse.github.io/cat.



## **11. Segment Anything Meets Universal Adversarial Perturbation**

任何遇到普遍对手的扰动的细分 cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12431v1) [paper-pdf](http://arxiv.org/pdf/2310.12431v1)

**Authors**: Dongshen Han, Sheng Zheng, Chaoning Zhang

**Abstract**: As Segment Anything Model (SAM) becomes a popular foundation model in computer vision, its adversarial robustness has become a concern that cannot be ignored. This works investigates whether it is possible to attack SAM with image-agnostic Universal Adversarial Perturbation (UAP). In other words, we seek a single perturbation that can fool the SAM to predict invalid masks for most (if not all) images. We demonstrate convetional image-centric attack framework is effective for image-independent attacks but fails for universal adversarial attack. To this end, we propose a novel perturbation-centric framework that results in a UAP generation method based on self-supervised contrastive learning (CL), where the UAP is set to the anchor sample and the positive sample is augmented from the UAP. The representations of negative samples are obtained from the image encoder in advance and saved in a memory bank. The effectiveness of our proposed CL-based UAP generation method is validated by both quantitative and qualitative results. On top of the ablation study to understand various components in our proposed method, we shed light on the roles of positive and negative samples in making the generated UAP effective for attacking SAM.

摘要: 随着Segment Anything Model(SAM)成为计算机视觉中一种流行的基础模型，其对抗健壮性已成为一个不容忽视的问题。这项工作调查是否有可能用图像不可知的通用对抗扰动(UAP)来攻击SAM。换句话说，我们寻找一个单一的扰动，它可以愚弄SAM来预测大多数(如果不是全部)图像的无效掩码。我们证明了传递式图像中心攻击框架对于图像无关攻击是有效的，但对于通用对抗性攻击是无效的。为此，我们提出了一种新的以扰动为中心的框架，该框架导致了一种基于自监督对比学习(CL)的UAP生成方法，其中UAP被设置为锚定样本，正样本从UAP增加。事先从图像编码器获取负样本的表示，并将其存储在存储体中。定量和定性结果验证了本文提出的基于CL的UAP生成方法的有效性。在消融研究以了解我们提出的方法中的各个组成部分的基础上，我们阐明了正样本和负样本在使生成的UAP有效攻击SAM方面所起的作用。



## **12. One-Bit Byzantine-Tolerant Distributed Learning via Over-the-Air Computation**

基于空中计算的一位拜占庭容忍分布式学习 eess.SP

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.11998v2) [paper-pdf](http://arxiv.org/pdf/2310.11998v2)

**Authors**: Yuhan Yang, Youlong Wu, Yuning Jiang, Yuanming Shi

**Abstract**: Distributed learning has become a promising computational parallelism paradigm that enables a wide scope of intelligent applications from the Internet of Things (IoT) to autonomous driving and the healthcare industry. This paper studies distributed learning in wireless data center networks, which contain a central edge server and multiple edge workers to collaboratively train a shared global model and benefit from parallel computing. However, the distributed nature causes the vulnerability of the learning process to faults and adversarial attacks from Byzantine edge workers, as well as the severe communication and computation overhead induced by the periodical information exchange process. To achieve fast and reliable model aggregation in the presence of Byzantine attacks, we develop a signed stochastic gradient descent (SignSGD)-based Hierarchical Vote framework via over-the-air computation (AirComp), where one voting process is performed locally at the wireless edge by taking advantage of Bernoulli coding while the other is operated over-the-air at the central edge server by utilizing the waveform superposition property of the multiple-access channels. We comprehensively analyze the proposed framework on the impacts including Byzantine attacks and the wireless environment (channel fading and receiver noise), followed by characterizing the convergence behavior under non-convex settings. Simulation results validate our theoretical achievements and demonstrate the robustness of our proposed framework in the presence of Byzantine attacks and receiver noise.

摘要: 分布式学习已经成为一种很有前途的计算并行范例，可以实现从物联网(IoT)到自动驾驶和医疗保健行业的广泛智能应用。本文研究了无线数据中心网络中的分布式学习，该网络包含一个中心边缘服务器和多个边缘工作者，以协作地训练共享的全局模型并受益于并行计算。然而，分布式的性质使得学习过程容易受到拜占庭边缘工作者的错误和敌意攻击，以及周期性的信息交换过程带来的严重的通信和计算开销。为了在拜占庭攻击下实现快速、可靠的模型聚合，提出了一种基于带符号随机梯度下降(SignSGD)的空中分级投票框架(AirComp)，其中一个投票过程在无线边缘利用Bernoulli编码在本地执行，另一个投票过程在中心边缘服务器利用多址信道的波形叠加特性进行空中操作。我们综合分析了该框架对拜占庭攻击和无线环境(信道衰落和接收器噪声)的影响，并刻画了非凸环境下的收敛行为。仿真结果验证了我们的理论成果，并证明了我们提出的框架在存在拜占庭攻击和接收器噪声的情况下具有良好的鲁棒性。



## **13. REVAMP: Automated Simulations of Adversarial Attacks on Arbitrary Objects in Realistic Scenes**

REVAMP：对现实场景中任意对象的对抗性攻击的自动模拟 cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12243v1) [paper-pdf](http://arxiv.org/pdf/2310.12243v1)

**Authors**: Matthew Hull, Zijie J. Wang, Duen Horng Chau

**Abstract**: Deep Learning models, such as those used in an autonomous vehicle are vulnerable to adversarial attacks where an attacker could place an adversarial object in the environment, leading to mis-classification. Generating these adversarial objects in the digital space has been extensively studied, however successfully transferring these attacks from the digital realm to the physical realm has proven challenging when controlling for real-world environmental factors. In response to these limitations, we introduce REVAMP, an easy-to-use Python library that is the first-of-its-kind tool for creating attack scenarios with arbitrary objects and simulating realistic environmental factors, lighting, reflection, and refraction. REVAMP enables researchers and practitioners to swiftly explore various scenarios within the digital realm by offering a wide range of configurable options for designing experiments and using differentiable rendering to reproduce physically plausible adversarial objects. We will demonstrate and invite the audience to try REVAMP to produce an adversarial texture on a chosen object while having control over various scene parameters. The audience will choose a scene, an object to attack, the desired attack class, and the number of camera positions to use. Then, in real time, we show how this altered texture causes the chosen object to be mis-classified, showcasing the potential of REVAMP in real-world scenarios. REVAMP is open-source and available at https://github.com/poloclub/revamp.

摘要: 深度学习模型，如用于自动驾驶汽车的模型，容易受到对抗性攻击，攻击者可以在环境中放置对抗性对象，导致错误分类。在数字空间中生成这些敌对对象已经被广泛研究，然而，事实证明，在控制现实世界的环境因素时，成功地将这些攻击从数字领域转移到物理领域是具有挑战性的。为了应对这些限制，我们引入了REVAMP，这是一个易于使用的Python库，它是第一个用于创建具有任意对象的攻击场景并模拟真实环境因素、照明、反射和折射的工具。Revamp使研究人员和从业者能够通过提供广泛的可配置选项来设计实验并使用可区分渲染来再现物理上看似合理的对抗性对象，从而快速探索数字领域中的各种场景。我们将演示并邀请观众尝试改装，在选择的对象上产生一种对抗性的纹理，同时控制各种场景参数。观众将选择一个场景，一个要攻击的对象，想要的攻击等级，以及要使用的摄像机位置数量。然后，我们实时地展示了这种改变的纹理是如何导致所选对象被错误分类的，展示了在真实世界场景中的改造潜力。Revamp是开源的，可在https://github.com/poloclub/revamp.上获得



## **14. A Black-Box Attack on Code Models via Representation Nearest Neighbor Search**

基于表示最近邻搜索的代码模型黑盒攻击 cs.CR

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2305.05896v3) [paper-pdf](http://arxiv.org/pdf/2305.05896v3)

**Authors**: Jie Zhang, Wei Ma, Qiang Hu, Shangqing Liu, Xiaofei Xie, Yves Le Traon, Yang Liu

**Abstract**: Existing methods for generating adversarial code examples face several challenges: limted availability of substitute variables, high verification costs for these substitutes, and the creation of adversarial samples with noticeable perturbations. To address these concerns, our proposed approach, RNNS, uses a search seed based on historical attacks to find potential adversarial substitutes. Rather than directly using the discrete substitutes, they are mapped to a continuous vector space using a pre-trained variable name encoder. Based on the vector representation, RNNS predicts and selects better substitutes for attacks. We evaluated the performance of RNNS across six coding tasks encompassing three programming languages: Java, Python, and C. We employed three pre-trained code models (CodeBERT, GraphCodeBERT, and CodeT5) that resulted in a cumulative of 18 victim models. The results demonstrate that RNNS outperforms baselines in terms of ASR and QT. Furthermore, the perturbation of adversarial examples introduced by RNNS is smaller compared to the baselines in terms of the number of replaced variables and the change in variable length. Lastly, our experiments indicate that RNNS is efficient in attacking defended models and can be employed for adversarial training.

摘要: 现有的生成对抗性代码示例的方法面临着几个挑战：替换变量的可用性有限，这些替换的验证成本很高，以及创建具有明显扰动的对抗性样本。为了解决这些问题，我们提出的方法RNNS使用基于历史攻击的搜索种子来寻找潜在的对手替代品。不是直接使用离散代换，而是使用预先训练的变量名称编码器将它们映射到连续的向量空间。基于向量表示，RNNS预测并选择更好的攻击替代方案。我们评估了RNNS在六个编码任务中的性能，包括三种编程语言：Java、Python和C。我们使用了三个预先训练的代码模型(CodeBERT、GraphCodeBERT和CodeT5)，从而累积了18个受害者模型。结果表明，RNNS在ASR和QT方面优于基线。此外，与基线相比，RNNS引入的对抗性例子在替换变量的数量和变量长度的变化方面的扰动较小。最后，我们的实验表明，RNNS在攻击防御模型方面是有效的，并且可以用于对抗性训练。



## **15. Black-Box Training Data Identification in GANs via Detector Networks**

基于检测器网络的GANS黑盒训练数据识别 cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12063v1) [paper-pdf](http://arxiv.org/pdf/2310.12063v1)

**Authors**: Lukman Olagoke, Salil Vadhan, Seth Neel

**Abstract**: Since their inception Generative Adversarial Networks (GANs) have been popular generative models across images, audio, video, and tabular data. In this paper we study whether given access to a trained GAN, as well as fresh samples from the underlying distribution, if it is possible for an attacker to efficiently identify if a given point is a member of the GAN's training data. This is of interest for both reasons related to copyright, where a user may want to determine if their copyrighted data has been used to train a GAN, and in the study of data privacy, where the ability to detect training set membership is known as a membership inference attack. Unlike the majority of prior work this paper investigates the privacy implications of using GANs in black-box settings, where the attack only has access to samples from the generator, rather than access to the discriminator as well. We introduce a suite of membership inference attacks against GANs in the black-box setting and evaluate our attacks on image GANs trained on the CIFAR10 dataset and tabular GANs trained on genomic data. Our most successful attack, called The Detector, involve training a second network to score samples based on their likelihood of being generated by the GAN, as opposed to a fresh sample from the distribution. We prove under a simple model of the generator that the detector is an approximately optimal membership inference attack. Across a wide range of tabular and image datasets, attacks, and GAN architectures, we find that adversaries can orchestrate non-trivial privacy attacks when provided with access to samples from the generator. At the same time, the attack success achievable against GANs still appears to be lower compared to other generative and discriminative models; this leaves the intriguing open question of whether GANs are in fact more private, or if it is a matter of developing stronger attacks.

摘要: 自诞生以来，生成性对抗网络(GANS)一直是图像、音频、视频和表格数据中流行的生成性模型。在本文中，我们研究是否允许攻击者访问训练的GaN以及来自底层分布的新鲜样本，如果攻击者有可能有效地识别给定的点是否是GaN的训练数据的成员。这是因为与版权有关的原因，其中用户可能想要确定他们的受版权保护的数据是否已被用于训练GAN，以及在数据隐私研究中，其中检测训练集成员资格的能力被称为成员关系推断攻击。与大多数以前的工作不同，本文研究了在黑盒环境中使用Gans的隐私影响，在黑盒环境中，攻击者只能访问来自生成器的样本，而不能访问鉴别器。我们介绍了一套针对黑盒环境中的GAN的成员关系推理攻击，并评估了我们对在CIFAR10数据集上训练的图像GAN和在基因组数据上训练的表格GAN的攻击。我们最成功的攻击被称为探测器，它涉及训练第二个网络，根据样本由GAN生成的可能性对样本进行评分，而不是根据分布中的新样本进行评分。在一个简单的生成器模型下，我们证明了该检测器是一种近似最优的成员推理攻击。在广泛的表格和图像数据集、攻击和GAN架构中，我们发现，当提供对生成器样本的访问权限时，攻击者可以策划不平凡的隐私攻击。与此同时，与其他生成性和歧视性模型相比，对甘斯的攻击成功率似乎仍然较低；这留下了一个耐人寻味的悬而未决的问题，即甘斯实际上是更私密，还是开发了更强大的攻击。



## **16. Exploring Decision-based Black-box Attacks on Face Forgery Detection**

基于决策的黑盒攻击在人脸伪造检测中的应用研究 cs.CV

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12017v1) [paper-pdf](http://arxiv.org/pdf/2310.12017v1)

**Authors**: Zhaoyu Chen, Bo Li, Kaixun Jiang, Shuang Wu, Shouhong Ding, Wenqiang Zhang

**Abstract**: Face forgery generation technologies generate vivid faces, which have raised public concerns about security and privacy. Many intelligent systems, such as electronic payment and identity verification, rely on face forgery detection. Although face forgery detection has successfully distinguished fake faces, recent studies have demonstrated that face forgery detectors are very vulnerable to adversarial examples. Meanwhile, existing attacks rely on network architectures or training datasets instead of the predicted labels, which leads to a gap in attacking deployed applications. To narrow this gap, we first explore the decision-based attacks on face forgery detection. However, applying existing decision-based attacks directly suffers from perturbation initialization failure and low image quality. First, we propose cross-task perturbation to handle initialization failures by utilizing the high correlation of face features on different tasks. Then, inspired by using frequency cues by face forgery detection, we propose the frequency decision-based attack. We add perturbations in the frequency domain and then constrain the visual quality in the spatial domain. Finally, extensive experiments demonstrate that our method achieves state-of-the-art attack performance on FaceForensics++, CelebDF, and industrial APIs, with high query efficiency and guaranteed image quality. Further, the fake faces by our method can pass face forgery detection and face recognition, which exposes the security problems of face forgery detectors.

摘要: 人脸伪造生成技术生成了生动的人脸，这引发了公众对安全和隐私的担忧。许多智能系统，如电子支付和身份验证，都依赖于人脸伪造检测。虽然人脸伪造检测已经成功地区分了虚假人脸，但最近的研究表明，人脸伪造检测器非常容易受到敌意例子的攻击。同时，现有的攻击依赖于网络结构或训练数据集，而不是预测的标签，这导致了对已部署应用的攻击存在缺口。为了缩小这一差距，我们首先探讨了基于决策的人脸伪造检测攻击。然而，直接应用现有的基于决策的攻击方法存在扰动初始化失败和图像质量不高的问题。首先，我们利用人脸特征在不同任务上的高度相关性，提出了跨任务扰动来处理初始化失败。然后，受人脸伪造检测中利用频率线索的启发，提出了基于频率判决的攻击方法。我们在频域中加入扰动，然后在空间域中约束视觉质量。实验结果表明，该方法对FaceForensics++、CelebDF和工业API具有较高的攻击性能，具有较高的查询效率和图像质量保证。此外，该方法生成的假人脸能够通过人脸伪造检测和人脸识别，暴露了人脸伪造检测器的安全问题。



## **17. Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks**

视频质量评估模型中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2309.13609v2) [paper-pdf](http://arxiv.org/pdf/2309.13609v2)

**Authors**: Ao-Xiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang

**Abstract**: No-Reference Video Quality Assessment (NR-VQA) plays an essential role in improving the viewing experience of end-users. Driven by deep learning, recent NR-VQA models based on Convolutional Neural Networks (CNNs) and Transformers have achieved outstanding performance. To build a reliable and practical assessment system, it is of great necessity to evaluate their robustness. However, such issue has received little attention in the academic community. In this paper, we make the first attempt to evaluate the robustness of NR-VQA models against adversarial attacks, and propose a patch-based random search method for black-box attack. Specifically, considering both the attack effect on quality score and the visual quality of adversarial video, the attack problem is formulated as misleading the estimated quality score under the constraint of just-noticeable difference (JND). Built upon such formulation, a novel loss function called Score-Reversed Boundary Loss is designed to push the adversarial video's estimated quality score far away from its ground-truth score towards a specific boundary, and the JND constraint is modeled as a strict $L_2$ and $L_\infty$ norm restriction. By this means, both white-box and black-box attacks can be launched in an effective and imperceptible manner. The source code is available at https://github.com/GZHU-DVL/AttackVQA.

摘要: 无参考视频质量评估(NR-VQA)对于改善终端用户的观看体验起着至关重要的作用。在深度学习的推动下，最近基于卷积神经网络(CNN)和变压器的NR-VQA模型取得了优异的性能。为了建立一个可靠、实用的评估体系，对它们的稳健性进行评估是非常必要的。然而，这一问题在学术界却鲜有人关注。本文首次尝试评估了NR-VQA模型对对手攻击的稳健性，并提出了一种基于补丁的黑盒攻击随机搜索方法。具体地说，综合考虑攻击对视频质量分数的影响和对抗性视频的视觉质量，在JND约束下，将攻击问题描述为误导估计质量分数。在此基础上，设计了一种新的损失函数--分数反向边界损失函数，将对抗性视频的估计质量分数远离其真实分数推向特定的边界，并将JND约束建模为严格的$L_2$和$L_\inty$范数限制。通过这种手段，白盒攻击和黑盒攻击都可以有效地、隐蔽地发动。源代码可在https://github.com/GZHU-DVL/AttackVQA.上找到



## **18. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

PromptBitch：评估大型语言模型在对抗性提示下的稳健性 cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2306.04528v4) [paper-pdf](http://arxiv.org/pdf/2306.04528v4)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. Code is available at: https://github.com/microsoft/promptbench.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptBtch，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些对抗性提示旨在模仿打字或同义词等看似合理的用户错误，旨在评估微小的偏差如何在保持语义完整性的同时影响LLM结果。这些提示随后被用于不同的任务，如情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4788个对抗性提示，仔细评估了8个任务和13个数据集。我们的研究结果表明，当代的LLM对敌意提示并不健壮。此外，我们还提供了全面的分析，以了解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。代码可从以下网址获得：https://github.com/microsoft/promptbench.



## **19. Quantifying Privacy Risks of Prompts in Visual Prompt Learning**

视觉提示学习中提示的隐私风险量化 cs.CR

To appear in the 33rd USENIX Security Symposium, August 14-16, 2024

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11970v1) [paper-pdf](http://arxiv.org/pdf/2310.11970v1)

**Authors**: Yixin Wu, Rui Wen, Michael Backes, Pascal Berrang, Mathias Humbert, Yun Shen, Yang Zhang

**Abstract**: Large-scale pre-trained models are increasingly adapted to downstream tasks through a new paradigm called prompt learning. In contrast to fine-tuning, prompt learning does not update the pre-trained model's parameters. Instead, it only learns an input perturbation, namely prompt, to be added to the downstream task data for predictions. Given the fast development of prompt learning, a well-generalized prompt inevitably becomes a valuable asset as significant effort and proprietary data are used to create it. This naturally raises the question of whether a prompt may leak the proprietary information of its training data. In this paper, we perform the first comprehensive privacy assessment of prompts learned by visual prompt learning through the lens of property inference and membership inference attacks. Our empirical evaluation shows that the prompts are vulnerable to both attacks. We also demonstrate that the adversary can mount a successful property inference attack with limited cost. Moreover, we show that membership inference attacks against prompts can be successful with relaxed adversarial assumptions. We further make some initial investigations on the defenses and observe that our method can mitigate the membership inference attacks with a decent utility-defense trade-off but fails to defend against property inference attacks. We hope our results can shed light on the privacy risks of the popular prompt learning paradigm. To facilitate the research in this direction, we will share our code and models with the community.

摘要: 通过一种名为快速学习的新范式，大规模预先训练的模型越来越多地适应下游任务。与微调相反，快速学习不会更新预先训练的模型的参数。相反，它只学习要添加到下游任务数据以进行预测的输入扰动，即提示。鉴于快速学习的快速发展，一个通用的快速学习不可避免地成为一项宝贵的资产，因为需要大量的努力和专有数据来创建它。这自然提出了一个问题，即提示是否会泄露其训练数据的专有信息。本文首次通过属性推理和成员关系推理攻击的视角，对视觉提示学习学习到的提示进行了全面的隐私评估。我们的经验评估表明，提示容易受到这两种攻击。我们还证明了攻击者能够以有限的代价发起成功的属性推理攻击。此外，我们还证明了在放松对抗性假设的情况下，针对提示的成员推理攻击可以成功。我们进一步对防御进行了一些初步的研究，观察到我们的方法可以在效用和防御之间进行良好的权衡来缓解成员关系推理攻击，但不能防御属性推理攻击。我们希望我们的结果能够揭示流行的快速学习范式的隐私风险。为了促进这方面的研究，我们将与社会各界分享我们的代码和模型。



## **20. IMAP: Intrinsically Motivated Adversarial Policy**

IMAP：内在动机的对抗政策 cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2305.02605v2) [paper-pdf](http://arxiv.org/pdf/2305.02605v2)

**Authors**: Xiang Zheng, Xingjun Ma, Shengjie Wang, Xinyu Wang, Chao Shen, Cong Wang

**Abstract**: Reinforcement learning agents are susceptible to evasion attacks during deployment. In single-agent environments, these attacks can occur through imperceptible perturbations injected into the inputs of the victim policy network. In multi-agent environments, an attacker can manipulate an adversarial opponent to influence the victim policy's observations indirectly. While adversarial policies offer a promising technique to craft such attacks, current methods are either sample-inefficient due to poor exploration strategies or require extra surrogate model training under the black-box assumption. To address these challenges, in this paper, we propose Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box adversarial policy learning in both single- and multi-agent environments. We formulate four types of adversarial intrinsic regularizers -- maximizing the adversarial state coverage, policy coverage, risk, or divergence -- to discover potential vulnerabilities of the victim policy in a principled way. We also present a novel Bias-Reduction (BR) method to boost IMAP further. Our experiments validate the effectiveness of the four types of adversarial intrinsic regularizers and BR in enhancing black-box adversarial policy learning across a variety of environments. Our IMAP successfully evades two types of defense methods, adversarial training and robust regularizer, decreasing the performance of the state-of-the-art robust WocaR-PPO agents by 34%-54% across four single-agent tasks. IMAP also achieves a state-of-the-art attacking success rate of 83.91% in the multi-agent game YouShallNotPass.

摘要: 强化学习代理在部署过程中容易受到逃避攻击。在单代理环境中，这些攻击可以通过注入受害者策略网络的输入的不可察觉的扰动来发生。在多智能体环境中，攻击者可以操纵对手来间接影响受害者策略的观察。虽然对抗性策略提供了一种很有希望的技术来策划这样的攻击，但目前的方法要么由于糟糕的探索策略而样本效率低下，要么需要在黑箱假设下进行额外的代理模型训练。为了应对这些挑战，在本文中，我们提出了内在激励的对抗策略(IMAP)，用于在单代理和多代理环境中有效的黑盒对抗策略学习。我们制定了四种类型的对抗性内在规则化--最大化对抗性状态覆盖、保单覆盖、风险或分歧--以原则性的方式发现受害者政策的潜在脆弱性。我们还提出了一种新的偏置减少(BR)方法来进一步提高IMAP。我们的实验验证了四种类型的对抗性内在正则化和BR在增强各种环境下的黑箱对抗性策略学习方面的有效性。我们的IMAP成功地避开了两种防御方法，对抗性训练和稳健正则化，使最先进的稳健WocaR-PPO代理在四个单代理任务中的性能降低了34%-54%。IMAP在多智能体游戏YouShallNotPass中的进攻成功率也达到了83.91%。



## **21. Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**

面向稳健多智能体协作感知的恶意智能体检测 cs.CR

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11901v1) [paper-pdf](http://arxiv.org/pdf/2310.11901v1)

**Authors**: Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang

**Abstract**: Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.

摘要: 近年来，多智能体协作(MAC)感知被提出，并在许多应用中优于传统的单智能体感知，如自主驾驶。然而，由于信息的交换，MAC感知比单代理感知更容易受到敌意攻击。攻击者可以很容易地通过从附近的恶意代理发送有害信息来降低受害者代理的性能。在本文中，我们将对抗性攻击扩展到一项重要的感知任务--MAC对象检测，在这种情况下，对抗性训练等一般防御手段不再有效地对抗这些攻击。更重要的是，我们提出了恶意代理检测(Made)，这是一种针对MAC感知的反应性防御，可以由每个代理部署以准确检测并随后删除其本地协作网络中的任何潜在恶意代理。特别地，Made使用基于双假设检验的半监督异常检测器独立地检查网络中的每个代理，并结合Benjamini-Hochberg过程来控制推理的误检率。对于这两种假设检验，我们分别提出了一个匹配损失统计量和一个协作重建损失统计量，这两个统计量都是基于待检查代理和部署检测器的自我代理之间的一致性。我们在基准3D数据集V2X-SIM和真实道路数据集DAIR-V2X上进行了综合评估，结果表明，在Made的保护下，与最佳情况下的Oracle防御者相比，对抗我们的攻击的平均精度分别下降了1.28%和0.34%，远低于对抗性训练的8.92%和10.00%。



## **22. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

IRAD：抵抗敌意攻击的隐式表示驱动图像重采样 cs.CV

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11890v1) [paper-pdf](http://arxiv.org/pdf/2310.11890v1)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.

摘要: 我们引入了一种新的方法来对抗敌意攻击，即图像重采样。图像重采样将离散图像转换为新图像，模拟由几何变换指定的场景重新捕获或重新渲染的过程。我们的想法背后的基本原理是，图像重采样可以在保留基本语义信息的同时减轻对抗性扰动的影响，从而在防御对抗性攻击方面具有固有的优势。为了验证这一概念，我们提出了一种利用图像重采样来防御敌意攻击的综合研究。我们已经开发了使用内插策略和协调移动量的基本重采样方法。我们的分析表明，这些基本方法可以部分缓解对抗性攻击。然而，它们也有明显的局限性：清晰图像的准确性显著下降，而对抗性例子的准确性提高并不显著。我们提出了隐式表示驱动的图像重采样(IRAD)来克服这些限制。首先，我们构造了一个隐式连续表示，它使我们能够表示连续坐标空间内的任何输入图像。其次，我们介绍了SampleNet，它可以根据不同的输入自动生成像素方向的移位以进行重采样。此外，我们可以将我们的方法扩展到最先进的基于扩散的方法，以更少的时间步骤加速它，同时保持其防御能力。大量实验表明，该方法在保持对清晰图像的较高准确率的同时，显著增强了不同深度模型对各种攻击的抵抗能力。



## **23. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

生成还是不生成？安全驱动的未学习扩散模型仍然很容易生成不安全的图像。目前而言 cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11868v1) [paper-pdf](http://arxiv.org/pdf/2310.11868v1)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of complex and diverse images. However, these models also introduce potential safety hazards, such as the production of harmful content and infringement of data copyrights. Although there have been efforts to create safety-driven unlearning methods to counteract these challenges, doubts remain about their capabilities. To bridge this uncertainty, we propose an evaluation framework built upon adversarial attacks (also referred to as adversarial prompts), in order to discern the trustworthiness of these safety-driven unlearned DMs. Specifically, our research explores the (worst-case) robustness of unlearned DMs in eradicating unwanted concepts, styles, and objects, assessed by the generation of adversarial prompts. We develop a novel adversarial learning approach called UnlearnDiff that leverages the inherent classification capabilities of DMs to streamline the generation of adversarial prompts, making it as simple for DMs as it is for image classification attacks. This technique streamlines the creation of adversarial prompts, making the process as intuitive for generative modeling as it is for image classification assaults. Through comprehensive benchmarking, we assess the unlearning robustness of five prevalent unlearned DMs across multiple tasks. Our results underscore the effectiveness and efficiency of UnlearnDiff when compared to state-of-the-art adversarial prompting methods. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.

摘要: 扩散模型(DM)的最新进展使复杂多样图像的生成发生了革命性的变化。然而，这些模式也带来了潜在的安全隐患，如制作有害内容和侵犯数据版权。尽管已经努力创造安全驱动的遗忘方法来应对这些挑战，但人们仍然对它们的能力持怀疑态度。为了弥补这种不确定性，我们提出了一个建立在对抗性攻击(也称为对抗性提示)基础上的评估框架，以辨别这些安全驱动的未学习DM的可信度。具体地说，我们的研究探索了未学习的DM在消除不需要的概念、风格和对象方面的(最坏情况)健壮性，通过生成对抗性提示来评估。我们开发了一种新的对抗性学习方法UnlearnDiff，该方法利用DM固有的分类能力来简化对抗性提示的生成，使其对于DM来说就像对图像分类攻击一样简单。这项技术简化了对抗性提示的创建，使该过程对于生成性建模和图像分类攻击一样直观。通过全面的基准测试，我们评估了五个流行的未学习DM在多个任务中的遗忘健壮性。我们的结果强调了UnlearnDiff与最先进的对抗性提示方法相比的有效性和效率。有关代码，请访问https://github.com/OPTML-Group/Diffusion-MU-Attack.警告：本文包含可能具有攻击性的模型输出。



## **24. Revisiting Transferable Adversarial Image Examples: Attack Categorization, Evaluation Guidelines, and New Insights**

重访可转移的敌意图像实例：攻击分类、评估指南和新见解 cs.CR

Code is available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11850v1) [paper-pdf](http://arxiv.org/pdf/2310.11850v1)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes, Qi Li, Chao Shen

**Abstract**: Transferable adversarial examples raise critical security concerns in real-world, black-box attack scenarios. However, in this work, we identify two main problems in common evaluation practices: (1) For attack transferability, lack of systematic, one-to-one attack comparison and fair hyperparameter settings. (2) For attack stealthiness, simply no comparisons. To address these problems, we establish new evaluation guidelines by (1) proposing a novel attack categorization strategy and conducting systematic and fair intra-category analyses on transferability, and (2) considering diverse imperceptibility metrics and finer-grained stealthiness characteristics from the perspective of attack traceback. To this end, we provide the first large-scale evaluation of transferable adversarial examples on ImageNet, involving 23 representative attacks against 9 representative defenses. Our evaluation leads to a number of new insights, including consensus-challenging ones: (1) Under a fair attack hyperparameter setting, one early attack method, DI, actually outperforms all the follow-up methods. (2) A state-of-the-art defense, DiffPure, actually gives a false sense of (white-box) security since it is indeed largely bypassed by our (black-box) transferable attacks. (3) Even when all attacks are bounded by the same $L_p$ norm, they lead to dramatically different stealthiness performance, which negatively correlates with their transferability performance. Overall, our work demonstrates that existing problematic evaluations have indeed caused misleading conclusions and missing points, and as a result, hindered the assessment of the actual progress in this field.

摘要: 在现实世界的黑盒攻击场景中，可转移的敌意示例会引起严重的安全问题。然而，在这项工作中，我们发现了常见评估实践中的两个主要问题：(1)对于攻击的可转移性，缺乏系统的、一对一的攻击比较和公平的超参数设置。(2)对于攻击隐蔽性，根本没有可比性。针对这些问题，我们建立了新的评估准则：(1)提出了一种新的攻击分类策略，并对可转移性进行了系统和公平的类内分析；(2)从攻击追溯的角度考虑了不同的隐蔽性度量和更细粒度的隐蔽性特征。为此，我们首次对ImageNet上可转移的对抗性例子进行了大规模评估，涉及对9个代表性防御的23个代表性攻击。我们的评估导致了一些新的见解，包括挑战共识的：(1)在公平的攻击超参数设置下，一种早期攻击方法DI实际上优于所有后续方法。(2)最先进的防御系统DiffPure实际上给人一种错误的(白盒)安全感觉，因为它确实在很大程度上被我们的(黑盒)可转移攻击绕过了。(3)即使所有攻击都受到相同的$L_p$范数的约束，它们的隐蔽性性能也会有很大的不同，这与它们的可转移性性能呈负相关。总体而言，我们的工作表明，现有的有问题的评价确实造成了误导性的结论和遗漏，因此阻碍了对这一领域实际进展的评估。



## **25. Adversarial Training for Physics-Informed Neural Networks**

物理信息神经网络的对抗性训练 cs.LG

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.11789v1) [paper-pdf](http://arxiv.org/pdf/2310.11789v1)

**Authors**: Yao Li, Shengzhu Shi, Zhichang Guo, Boying Wu

**Abstract**: Physics-informed neural networks have shown great promise in solving partial differential equations. However, due to insufficient robustness, vanilla PINNs often face challenges when solving complex PDEs, especially those involving multi-scale behaviors or solutions with sharp or oscillatory characteristics. To address these issues, based on the projected gradient descent adversarial attack, we proposed an adversarial training strategy for PINNs termed by AT-PINNs. AT-PINNs enhance the robustness of PINNs by fine-tuning the model with adversarial samples, which can accurately identify model failure locations and drive the model to focus on those regions during training. AT-PINNs can also perform inference with temporal causality by selecting the initial collocation points around temporal initial values. We implement AT-PINNs to the elliptic equation with multi-scale coefficients, Poisson equation with multi-peak solutions, Burgers equation with sharp solutions and the Allen-Cahn equation. The results demonstrate that AT-PINNs can effectively locate and reduce failure regions. Moreover, AT-PINNs are suitable for solving complex PDEs, since locating failure regions through adversarial attacks is independent of the size of failure regions or the complexity of the distribution.

摘要: 物理信息神经网络在求解偏微分方程组方面显示出巨大的前景。然而，由于健壮性不足，香草PINN在求解复杂的偏微分方程组时经常面临挑战，特别是那些涉及多尺度行为或具有尖锐或振荡特征的解的情况。为了解决这些问题，基于投影的梯度下降对抗攻击，我们提出了一种称为AT-PINN的PINN对抗训练策略。AT-PINN通过使用对抗性样本对模型进行微调来增强PINN的稳健性，可以准确地识别模型的故障位置，并在训练过程中驱动模型关注这些区域。AT-PINN还可以通过在时间初始值周围选择初始配置点来执行与时间因果关系的推理。我们实现了AT-PINN到具有多尺度系数的椭圆型方程、具有多峰解的泊松方程、具有锐解的Burgers方程和Allen-Cahn方程。结果表明，AT-PINN能够有效地定位和减少失效区域。此外，AT-PINN适合于求解复杂的偏微分方程组，因为通过对抗性攻击来定位故障区域与故障区域的大小或分布的复杂性无关。



## **26. Evading Watermark based Detection of AI-Generated Content**

基于规避水印的人工智能生成内容检测 cs.LG

To appear in ACM Conference on Computer and Communications Security  (CCS), 2023

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2305.03807v4) [paper-pdf](http://arxiv.org/pdf/2305.03807v4)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process a watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed image evades detection while maintaining its visual quality. We show the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to AI-generated images and thus better maintain their visual quality than existing popular post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work shows the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new methods. Our code is publicly available: https://github.com/zhengyuan-jiang/WEvade.

摘要: 生成性人工智能模型可以生成极其逼真的内容，对信息的真实性提出了越来越大的挑战。为了应对这些挑战，水印被用来检测人工智能生成的内容。具体地说，水印在发布之前被嵌入到人工智能生成的内容中。如果可以从内容中解码类似的水印，则该内容被检测为人工智能生成的内容。在这项工作中，我们对这种基于水印的人工智能内容检测的稳健性进行了系统的研究。我们专注于人工智能生成的图像。我们的工作表明，攻击者可以通过在水印图像上添加一个人类无法察觉的小扰动来对其进行后处理，从而在保持其视觉质量的同时逃避检测。我们从理论和经验上证明了我们的攻击的有效性。此外，为了逃避检测，我们的对抗性后处理方法向人工智能生成的图像添加了更小的扰动，从而比现有的流行的后处理方法，如JPEG压缩、高斯模糊和亮度/对比度，更好地保持了它们的视觉质量。我们的工作显示了现有基于水印的人工智能生成内容检测的不足，突出了对新方法的迫切需求。我们的代码是公开提供的：https://github.com/zhengyuan-jiang/WEvade.



## **27. Attacks Meet Interpretability (AmI) Evaluation and Findings**

攻击符合可解释性(AMI)评估和调查结果 cs.CR

5 pages, 4 figures

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.08808v2) [paper-pdf](http://arxiv.org/pdf/2310.08808v2)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.

摘要: 为了考察模型解释在检测敌意实例方面的有效性，我们复制了两篇论文的结果：攻击满足解释性：对抗性样本的属性导向检测和AMI(攻击满足解释性)对对抗性实例的稳健性。然后进行实验和案例研究，找出两部作品的局限性。我们发现攻击满足可解释性(AMI)高度依赖于超参数的选择。因此，通过不同的超参数选择，阿米仍然能够检测到尼古拉斯·卡里尼的攻击。最后，我们对AMI等防御技术的未来评估工作提出了建议。



## **28. The Efficacy of Transformer-based Adversarial Attacks in Security Domains**

基于变形金刚的对抗性攻击在安全域的有效性 cs.CR

Accepted to IEEE Military Communications Conference (MILCOM), AI for  Cyber Workshop, 2023

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11597v1) [paper-pdf](http://arxiv.org/pdf/2310.11597v1)

**Authors**: Kunyang Li, Kyle Domico, Jean-Charles Noirot Ferrand, Patrick McDaniel

**Abstract**: Today, the security of many domains rely on the use of Machine Learning to detect threats, identify vulnerabilities, and safeguard systems from attacks. Recently, transformer architectures have improved the state-of-the-art performance on a wide range of tasks such as malware detection and network intrusion detection. But, before abandoning current approaches to transformers, it is crucial to understand their properties and implications on cybersecurity applications. In this paper, we evaluate the robustness of transformers to adversarial samples for system defenders (i.e., resiliency to adversarial perturbations generated on different types of architectures) and their adversarial strength for system attackers (i.e., transferability of adversarial samples generated by transformers to other target models). To that effect, we first fine-tune a set of pre-trained transformer, Convolutional Neural Network (CNN), and hybrid (an ensemble of transformer and CNN) models to solve different downstream image-based tasks. Then, we use an attack algorithm to craft 19,367 adversarial examples on each model for each task. The transferability of these adversarial examples is measured by evaluating each set on other models to determine which models offer more adversarial strength, and consequently, more robustness against these attacks. We find that the adversarial examples crafted on transformers offer the highest transferability rate (i.e., 25.7% higher than the average) onto other models. Similarly, adversarial examples crafted on other models have the lowest rate of transferability (i.e., 56.7% lower than the average) onto transformers. Our work emphasizes the importance of studying transformer architectures for attacking and defending models in security domains, and suggests using them as the primary architecture in transfer attack settings.

摘要: 今天，许多域的安全依赖于使用机器学习来检测威胁、识别漏洞和保护系统免受攻击。最近，变压器架构在恶意软件检测和网络入侵检测等广泛任务上提高了最先进的性能。但是，在放弃目前的变压器方法之前，了解它们的特性及其对网络安全应用的影响是至关重要的。在本文中，我们评估了变压器对系统防御者对抗样本的稳健性(即对不同类型体系结构上产生的对抗扰动的恢复能力)和它们对系统攻击者的对抗强度(即变压器产生的对抗样本对其他目标模型的可转移性)。为此，我们首先微调了一套预先训练的变压器、卷积神经网络(CNN)和混合(变压器和CNN的集成)模型，以解决不同的下游基于图像的任务。然后，我们使用攻击算法在每个模型上为每个任务制作19,367个对抗性示例。这些对抗性示例的可转移性是通过评估其他模型上的每个集合来衡量的，以确定哪些模型提供了更强的对抗性，从而对这些攻击具有更强的稳健性。我们发现，在变压器上制作的对抗性例子提供了最高的可转移率(即，比平均水平高25.7%)到其他模型上。同样，在其他模型上制作的对抗性例子在变压器上的可转移率最低(即，比平均水平低56.7%)。我们的工作强调了研究安全领域中攻击和防御模型的变压器体系结构的重要性，并建议将它们作为传输攻击环境中的主要体系结构。



## **29. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

联合学习中通过后门攻击的对抗性健壮性去坚固性 cs.LG

8 pages, 6 main pages of text, 4 figures, 2 tables. Made for a  Neurips workshop on backdoor attacks

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11594v1) [paper-pdf](http://arxiv.org/pdf/2310.11594v1)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: In today's data-driven landscape, the delicate equilibrium between safeguarding user privacy and unleashing data potential stands as a paramount concern. Federated learning, which enables collaborative model training without necessitating data sharing, has emerged as a privacy-centric solution. This decentralized approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data. Our research, initially spurred by test-time evasion attacks, investigates the intersection of adversarial training and backdoor attacks within federated learning, introducing Adversarial Robustness Unhardening (ARU). ARU is employed by a subset of adversaries to intentionally undermine model robustness during decentralized training, rendering models susceptible to a broader range of evasion attacks. We present extensive empirical experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our findings inform strategies for enhancing ARU to counter current defensive measures and highlight the limitations of existing defenses, offering insights into bolstering defenses against ARU.

摘要: 在当今数据驱动的格局中，保护用户隐私和释放数据潜力之间的微妙平衡是一个最重要的问题。联合学习是一种以隐私为中心的解决方案，它能够在不需要共享数据的情况下进行协作模型培训。这种分散的方法带来了安全挑战，尤其是毒化和后门攻击，即恶意实体注入被破坏的数据。我们的研究最初受到测试时间逃避攻击的启发，研究了联邦学习中对抗性训练和后门攻击的交集，引入了对抗性健壮性不硬化(ARU)。ARU被一部分攻击者利用来在分散训练期间故意破坏模型的健壮性，使模型容易受到更大范围的逃避攻击。我们提供了广泛的经验实验，评估ARU对对手训练的影响，以及现有针对中毒和后门攻击的健壮聚合防御。我们的发现为增强ARU以对抗当前防御措施的战略提供了参考，并突出了现有防御的局限性，为加强对ARU的防御提供了见解。



## **30. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

一种利用安全性和活跃性增强性能的双层区块链分片协议 cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11373v1) [paper-pdf](http://arxiv.org/pdf/2310.11373v1)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.

摘要: 分片对于提高区块链可伸缩性至关重要。现有的协议忽略了不同的对抗性攻击，限制了交易吞吐量。本文提出了一种突破性的分片协议Reetum，解决了这个问题，提高了区块链的可扩展性。RENETUM采用两阶段方法，根据运行时敌意攻击调整事务吞吐量。它包括两层的“控制”和“流程”分片。进程碎片包含至少一个可信节点，而控制碎片包含大多数可信节点。在第一阶段，事务被写入块，并由流程碎片中的节点投票表决。一致接受的障碍得到确认。在第二阶段，未获得一致接受的块由控制碎片投票表决。如果多数人投赞成票，就会接受阻止，从而消除第一阶段的反对者和沉默的选民。第一阶段使用一致投票，涉及的节点更少，支持更多的并行进程碎片。控制碎片最终确定决策并解决纠纷。实验证实了ReNetum的创新设计，提供了高交易吞吐量和对各种网络攻击的稳健性，性能优于现有的区块链网络分片协议。



## **31. CARSO: Blending Adversarial Training and Purification Improves Adversarial Robustness**

CARSO：混合对战训练和净化提高对战健壮性 cs.CV

19 pages, 1 figure, 9 tables

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2306.06081v3) [paper-pdf](http://arxiv.org/pdf/2306.06081v3)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a mutually-beneficial, robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the adversarially-trained model itself, and an aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of varied, strong adaptive attacks, across different image datasets and classifier architectures, shows that CARSO is able to defend itself against foreseen and unforeseen threats, including adaptive end-to-end attacks devised for stochastic defences. Paying a tolerable clean accuracy toll, our method improves by a significant margin the state of the art for CIFAR-10 and CIFAR-100 $\ell_\infty$ robust classification accuracy against AutoAttack. Code and pre-trained models are available at https://github.com/emaballarin/CARSO .

摘要: 在这项工作中，我们提出了一种新的用于图像分类的对抗性防御机制-CASO-以一种互惠的、增强稳健性的方式融合了对抗性训练和对抗性净化的范例。该方法建立在对抗性训练的分类器之上，并学习将其与潜在扰动输入相关联的内部表示映射到试探性干净重构的分布上。来自这种分布的多个样本由对抗性训练的模型本身进行分类，其输出的聚集最终构成感兴趣的稳健预测。在不同的图像数据集和分类器体系结构上，通过对各种不同的、强自适应攻击的基准测试进行的实验评估表明，CARSO能够防御可预见和不可预见的威胁，包括为随机防御而设计的自适应端到端攻击。支付了可容忍的干净精度代价，我们的方法显著地改善了CIFAR-10和CIFAR-100$\ell_\inty$相对于AutoAttack的稳健分类精度。代码和预先培训的模型可在https://github.com/emaballarin/CARSO上找到。



## **32. A Comprehensive Study of the Robustness for LiDAR-based 3D Object Detectors against Adversarial Attacks**

基于LiDAR的3D目标检测器抗敌意攻击能力的综合研究 cs.CV

30 pages, 14 figures. Accepted by IJCV

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2212.10230v3) [paper-pdf](http://arxiv.org/pdf/2212.10230v3)

**Authors**: Yifan Zhang, Junhui Hou, Yixuan Yuan

**Abstract**: Recent years have witnessed significant advancements in deep learning-based 3D object detection, leading to its widespread adoption in numerous applications. As 3D object detectors become increasingly crucial for security-critical tasks, it is imperative to understand their robustness against adversarial attacks. This paper presents the first comprehensive evaluation and analysis of the robustness of LiDAR-based 3D detectors under adversarial attacks. Specifically, we extend three distinct adversarial attacks to the 3D object detection task, benchmarking the robustness of state-of-the-art LiDAR-based 3D object detectors against attacks on the KITTI and Waymo datasets. We further analyze the relationship between robustness and detector properties. Additionally, we explore the transferability of cross-model, cross-task, and cross-data attacks. Thorough experiments on defensive strategies for 3D detectors are conducted, demonstrating that simple transformations like flipping provide little help in improving robustness when the applied transformation strategy is exposed to attackers. \revise{Finally, we propose balanced adversarial focal training, based on conventional adversarial training, to strike a balance between accuracy and robustness.} Our findings will facilitate investigations into understanding and defending against adversarial attacks on LiDAR-based 3D object detectors, thus advancing the field. The source code is publicly available at \url{https://github.com/Eaphan/Robust3DOD}.

摘要: 近年来，基于深度学习的3D目标检测技术取得了长足的进步，在众多应用中得到了广泛的应用。随着3D对象检测器对安全关键任务变得越来越重要，了解它们对对手攻击的稳健性是当务之急。本文首次对基于LiDAR的3D探测器在对抗攻击下的健壮性进行了全面的评估和分析。具体地说，我们将三种不同的对抗性攻击扩展到3D对象检测任务，以基准测试最先进的基于LiDAR的3D对象检测器针对Kitti和Waymo数据集的攻击的健壮性。我们进一步分析了稳健性与检测器特性之间的关系。此外，我们还探讨了跨模型、跨任务和跨数据攻击的可转移性。对3D探测器的防御策略进行了深入的实验，表明当应用的变换策略暴露给攻击者时，像翻转这样的简单变换对提高稳健性几乎没有帮助。修订{最后，我们建议在传统对抗训练的基础上进行平衡的对抗焦点训练，以在准确性和稳健性之间取得平衡。}我们的研究结果将有助于研究如何理解和防御基于LiDAR的3D对象探测器上的敌对攻击，从而推动该领域的发展。源代码可在\url{https://github.com/Eaphan/Robust3DOD}.}上公开获取



## **33. An adversarially robust data-market for spatial, crowd-sourced data**

空间、众包数据的相对强大的数据市场 cs.DS

13 pages, 7 figures

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2206.06299v3) [paper-pdf](http://arxiv.org/pdf/2206.06299v3)

**Authors**: Aida Manzano Kharman, Christian Jursitzky, Quan Zhou, Pietro Ferraro, Jakub Marecek, Pierre Pinson, Robert Shorten

**Abstract**: We describe an architecture for a decentralised data market for applications in which agents are incentivised to collaborate to crowd-source their data. The architecture is designed to reward data that furthers the market's collective goal, and distributes reward fairly to all those that contribute with their data. We show that the architecture is resilient to Sybil, wormhole, and data poisoning attacks. In order to evaluate the resilience of the architecture, we characterise its breakdown points for various adversarial threat models in an automotive use case.

摘要: 我们描述了一种用于应用程序的去中心化数据市场的架构，在该架构中，代理被激励合作以众包他们的数据。该架构旨在奖励推动市场集体目标的数据，并将奖励公平地分配给所有贡献其数据的人。我们表明，该体系结构对Sybil、虫洞和数据中毒攻击具有弹性。为了评估该体系结构的弹性，我们在一个汽车用例中描述了它的各种对抗性威胁模型的故障点。



## **34. Patch of Invisibility: Naturalistic Physical Black-Box Adversarial Attacks on Object Detectors**

隐形补丁：对物体探测器的自然主义物理黑匣子对抗性攻击 cs.CV

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2303.04238v4) [paper-pdf](http://arxiv.org/pdf/2303.04238v4)

**Authors**: Raz Lapid, Eylon Mizrahi, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called ``white-box'' attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a direct, black-box, gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. To our knowledge this is the first and only method that performs black-box physical attacks directly on object-detection models, which results with a model-agnostic attack. We show that our proposed method works both digitally and physically. We compared our approach against four different black-box attacks with different configurations. Our approach outperformed all other approaches that were tested in our experiments by a large margin.

摘要: 近年来，针对深度学习模型的对抗性攻击受到越来越多的关注。这方面的工作主要集中在基于梯度的技术，即所谓的“白盒”攻击，其中攻击者可以访问目标模型的内部参数；这种假设在现实世界中通常是不现实的。一些攻击还使用整个像素空间来愚弄给定的模型，这既不实用也不物理(即，现实世界)。相反，我们在这里提出了一种直接的、黑盒的、无梯度的方法，该方法使用预先训练的生成性对抗网络(GAN)的学习图像流形来为目标检测器生成自然的物理对抗斑块。据我们所知，这是第一种也是唯一一种直接对目标检测模型执行黑盒物理攻击的方法，这导致了与模型无关的攻击。我们证明了我们提出的方法在数字和物理上都是有效的。我们将我们的方法与四种不同配置的不同黑盒攻击进行了比较。我们的方法远远超过了在我们的实验中测试的所有其他方法。



## **35. It Is All About Data: A Survey on the Effects of Data on Adversarial Robustness**

这一切都与数据有关：数据对对手健壮性影响的调查 cs.LG

Accepted to ACM Computing Surveys, 40 pages, 24 figures

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2303.09767v3) [paper-pdf](http://arxiv.org/pdf/2303.09767v3)

**Authors**: Peiyu Xiong, Michael Tegegn, Jaskeerat Singh Sarin, Shubhraneel Pal, Julia Rubin

**Abstract**: Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to confuse the model into making a mistake. Such examples pose a serious threat to the applicability of machine-learning-based systems, especially in life- and safety-critical domains. To address this problem, the area of adversarial robustness investigates mechanisms behind adversarial attacks and defenses against these attacks. This survey reviews a particular subset of this literature that focuses on investigating properties of training data in the context of model robustness under evasion attacks. It first summarizes the main properties of data leading to adversarial vulnerability. It then discusses guidelines and techniques for improving adversarial robustness by enhancing the data representation and learning procedures, as well as techniques for estimating robustness guarantees given particular data. Finally, it discusses gaps of knowledge and promising future research directions in this area.

摘要: 对抗性的例子是机器学习模型的输入，攻击者故意设计这些模型来混淆模型，使其出错。这些例子对基于机器学习的系统的适用性构成了严重威胁，特别是在生命和安全关键领域。为了解决这个问题，对抗性稳健性领域调查了对抗性攻击背后的机制和对这些攻击的防御。这项调查回顾了这篇文献的一个特定子集，重点是在逃避攻击下的模型稳健性背景下调查训练数据的属性。它首先总结了导致对抗性漏洞的数据的主要属性。然后讨论了通过加强数据表示和学习过程来提高对手稳健性的指导方针和技术，以及在给定特定数据的情况下估计稳健性保证的技术。最后，讨论了该领域的知识差距和未来的研究方向。



## **36. Defending Black-box Classifiers by Bayesian Boundary Correction**

基于贝叶斯边界校正的黑盒分类器防御 cs.CV

arXiv admin note: text overlap with arXiv:2203.04713

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2306.16979v2) [paper-pdf](http://arxiv.org/pdf/2306.16979v2)

**Authors**: He Wang, Yunfeng Diao

**Abstract**: Classifiers based on deep neural networks have been recently challenged by Adversarial Attack, where the widely existing vulnerability has invoked the research in defending them from potential threats. Given a vulnerable classifier, existing defense methods are mostly white-box and often require re-training the victim under modified loss functions/training regimes. While the model/data/training specifics of the victim are usually unavailable to the user, re-training is unappealing, if not impossible for reasons such as limited computational resources. To this end, we propose a new black-box defense framework. It can turn any pre-trained classifier into a resilient one with little knowledge of the model specifics. This is achieved by new joint Bayesian treatments on the clean data, the adversarial examples and the classifier, for maximizing their joint probability. It is further equipped with a new post-train strategy which keeps the victim intact. We name our framework Bayesian Boundary Correction (BBC). BBC is a general and flexible framework that can easily adapt to different data types. We instantiate BBC for image classification and skeleton-based human activity recognition, for both static and dynamic data. Exhaustive evaluation shows that BBC has superior robustness and can enhance robustness without severely hurting the clean accuracy, compared with existing defense methods.

摘要: 基于深度神经网络的分类器最近受到了敌意攻击的挑战，其中广泛存在的漏洞引发了保护它们免受潜在威胁的研究。在给定一个易受攻击的分类器的情况下，现有的防御方法大多是白盒的，并且经常需要根据修改的损失函数/训练制度重新训练受害者。虽然受害者的模型/数据/培训细节通常对用户不可用，但重新培训是没有吸引力的，如果不是因为有限的计算资源等原因不可能的话。为此，我们提出了一种新的黑盒防御框架。它可以将任何预先训练的分类器变成一个有弹性的分类器，而对模型细节知之甚少。这是通过对干净数据、对抗性样本和分类器进行新的联合贝叶斯处理来实现的，以最大化它们的联合概率。它进一步配备了新的列车后战略，使受害者完好无损。我们将我们的框架命名为贝叶斯边界校正(BBC)。BBC是一个通用和灵活的框架，可以很容易地适应不同的数据类型。对于静态和动态数据，我们实例化了用于图像分类和基于骨骼的人体活动识别的BBC。详尽的评估表明，与现有的防御方法相比，BBC具有更好的稳健性，可以在不严重损害干净精度的情况下增强稳健性。



## **37. Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks**

对抗性攻击揭示的大型语言模型中的漏洞调查 cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10844v1) [paper-pdf](http://arxiv.org/pdf/2310.10844v1)

**Authors**: Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are swiftly advancing in architecture and capability, and as they integrate more deeply into complex systems, the urgency to scrutinize their security properties grows. This paper surveys research in the emerging interdisciplinary field of adversarial attacks on LLMs, a subfield of trustworthy ML, combining the perspectives of Natural Language Processing and Security. Prior work has shown that even safety-aligned LLMs (via instruction tuning and reinforcement learning through human feedback) can be susceptible to adversarial attacks, which exploit weaknesses and mislead AI systems, as evidenced by the prevalence of `jailbreak' attacks on models like ChatGPT and Bard. In this survey, we first provide an overview of large language models, describe their safety alignment, and categorize existing research based on various learning structures: textual-only attacks, multi-modal attacks, and additional attack methods specifically targeting complex systems, such as federated learning or multi-agent systems. We also offer comprehensive remarks on works that focus on the fundamental sources of vulnerabilities and potential defenses. To make this field more accessible to newcomers, we present a systematic review of existing works, a structured typology of adversarial attack concepts, and additional resources, including slides for presentations on related topics at the 62nd Annual Meeting of the Association for Computational Linguistics (ACL'24).

摘要: 大型语言模型(LLM)在体系结构和功能方面正在迅速发展，随着它们更深入地集成到复杂系统中，审查其安全属性的紧迫性也在增长。本文结合自然语言处理和安全的角度，对可信ML的一个子领域--LLMS的对抗性攻击这一新兴交叉学科领域的研究进行了综述。先前的工作表明，即使是与安全一致的LLM(通过指令调整和通过人类反馈的强化学习)也可能容易受到对手攻击，这些攻击利用弱点并误导人工智能系统，对ChatGPT和Bard等模型的“越狱”攻击盛行就是明证。在这次调查中，我们首先提供了大型语言模型的概述，描述了它们的安全对齐，并基于各种学习结构对现有研究进行了分类：纯文本攻击、多模式攻击以及专门针对复杂系统的额外攻击方法，如联合学习或多代理系统。我们还对侧重于漏洞的根本来源和潜在防御的工作进行了全面的评论。为了使这个领域更容易为新手所接受，我们提供了对现有工作的系统回顾，对抗性攻击概念的结构化类型学，以及额外的资源，包括在第62届计算语言学协会年会(ACL‘24)上相关主题的演示幻灯片。



## **38. Regularization properties of adversarially-trained linear regression**

逆训练线性回归的正则化性质 stat.ML

Accepted (spotlight) NeurIPS 2023; A preliminary version of this work  titled: "Surprises in adversarially-trained linear regression" was made  available under a different identifier: arXiv:2205.12695

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10807v1) [paper-pdf](http://arxiv.org/pdf/2310.10807v1)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Francis Bach, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against it. Formulated as a min-max problem, it searches for the best solution when the training data were corrupted by the worst-case attacks. Linear models are among the simple models where vulnerabilities can be observed and are the focus of our study. In this case, adversarial training leads to a convex optimization problem which can be formulated as the minimization of a finite sum. We provide a comparative analysis between the solution of adversarial training in linear regression and other regularization methods. Our main findings are that: (A) Adversarial training yields the minimum-norm interpolating solution in the overparameterized regime (more parameters than data), as long as the maximum disturbance radius is smaller than a threshold. And, conversely, the minimum-norm interpolator is the solution to adversarial training with a given radius. (B) Adversarial training can be equivalent to parameter shrinking methods (ridge regression and Lasso). This happens in the underparametrized region, for an appropriate choice of adversarial radius and zero-mean symmetrically distributed covariates. (C) For $\ell_\infty$-adversarial training -- as in square-root Lasso -- the choice of adversarial radius for optimal bounds does not depend on the additive noise variance. We confirm our theoretical findings with numerical examples.

摘要: 最先进的机器学习模型很容易受到相反构造的非常小的输入扰动的影响。对抗性训练是一种有效的防御方法。它被描述为一个最小-最大问题，当训练数据被最坏情况下的攻击破坏时，它搜索最优解。线性模型是可以观察到漏洞的简单模型之一，也是我们研究的重点。在这种情况下，对抗性训练导致了一个凸优化问题，该问题可以表示为有限和的最小化。我们将线性回归的对抗性训练方法与其他正则化方法进行了比较分析。我们的主要发现是：(A)只要最大扰动半径小于某一阈值，对抗性训练就能在过参数(参数多于数据)的情况下产生最小范数插值解。反之，最小范数插值器是具有给定半径的对抗性训练的解。(B)对抗性训练可等同于参数缩减法(岭回归和套索)。这发生在欠参数区域，以适当选择对抗性半径和零均值对称分布协变量。(C)对于对抗性训练--如在平方根套索中--最优界的对抗性半径的选择不取决于加性噪声方差。我们用数值例子证实了我们的理论发现。



## **39. Fast Adversarial Label-Flipping Attack on Tabular Data**

针对表格数据的快速敌意翻转标签攻击 cs.LG

10 pages

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10744v1) [paper-pdf](http://arxiv.org/pdf/2310.10744v1)

**Authors**: Xinglong Chang, Gillian Dobbie, Jörg Wicker

**Abstract**: Machine learning models are increasingly used in fields that require high reliability such as cybersecurity. However, these models remain vulnerable to various attacks, among which the adversarial label-flipping attack poses significant threats. In label-flipping attacks, the adversary maliciously flips a portion of training labels to compromise the machine learning model. This paper raises significant concerns as these attacks can camouflage a highly skewed dataset as an easily solvable classification problem, often misleading machine learning practitioners into lower defenses and miscalculations of potential risks. This concern amplifies in tabular data settings, where identifying true labels requires expertise, allowing malicious label-flipping attacks to easily slip under the radar. To demonstrate this risk is inherited in the adversary's objective, we propose FALFA (Fast Adversarial Label-Flipping Attack), a novel efficient attack for crafting adversarial labels. FALFA is based on transforming the adversary's objective and employs linear programming to reduce computational complexity. Using ten real-world tabular datasets, we demonstrate FALFA's superior attack potential, highlighting the need for robust defenses against such threats.

摘要: 机器学习模型越来越多地应用于网络安全等可靠性要求高的领域。然而，这些模型仍然容易受到各种攻击，其中对抗性标签翻转攻击构成了重大威胁。在翻转标签攻击中，对手恶意翻转训练标签的一部分，以破坏机器学习模型。本文提出了重要的关注，因为这些攻击可以将高度倾斜的数据集伪装成一个容易解决的分类问题，经常误导机器学习从业者降低防御能力和错误计算潜在风险。这种担忧在表格数据设置中放大，在表格数据设置中，识别真实标签需要专业知识，从而允许恶意标签翻转攻击很容易被忽视。为了证明这种风险是在对手的目标中继承的，我们提出了一种新的高效的攻击方法--快速对手标签翻转攻击(Falfa)。FALFA基于变换对手的目标，并使用线性规划来降低计算复杂性。使用10个真实世界的表格数据集，我们展示了法尔法优越的攻击潜力，强调了对此类威胁进行强大防御的必要性。



## **40. Passive Inference Attacks on Split Learning via Adversarial Regularization**

基于对抗性正则化的分裂学习被动推理攻击 cs.CR

16 pages, 16 figures

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10483v1) [paper-pdf](http://arxiv.org/pdf/2310.10483v1)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.

摘要: 分裂学习(Split Learning，SL)已成为传统联合学习的一种实用有效的替代方案。虽然以前攻击SL的尝试通常依赖于过于强烈的假设或目标明确、易于利用的模型，但我们寻求开发更实际的攻击。我们介绍了SDAR，这是一种针对SL的新型攻击框架，具有诚实但好奇的服务器。SDAR利用辅助数据和对抗性正则化学习客户私有模型的可解码模拟器，该模拟器可以有效地推断客户在香草SL下的私有特征，以及U形SL下的特征和标签。我们在两种配置下都进行了大量的实验，以验证我们提出的攻击的有效性。值得注意的是，在具有挑战性但实用的场景中，现有的被动攻击难以有效地重建客户端的私有数据，SDAR始终实现与主动攻击相当的攻击性能。在CIFAR-10上，在7的深度分裂水平上，SDAR实现了私有特征重建，在普通SL和U形SL上的均方误差都小于0.025，在U形背景下获得了98%以上的标签推理准确率，而现有的攻击无法产生非平凡的结果。



## **41. DANAA: Towards transferable attacks with double adversarial neuron attribution**

DANAA：具有双重对抗神经元属性的可转移攻击 cs.CV

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10427v1) [paper-pdf](http://arxiv.org/pdf/2310.10427v1)

**Authors**: Zhibo Jin, Zhiyu Zhu, Xinyi Wang, Jiayu Zhang, Jun Shen, Huaming Chen

**Abstract**: While deep neural networks have excellent results in many fields, they are susceptible to interference from attacking samples resulting in erroneous judgments. Feature-level attacks are one of the effective attack types, which targets the learnt features in the hidden layers to improve its transferability across different models. Yet it is observed that the transferability has been largely impacted by the neuron importance estimation results. In this paper, a double adversarial neuron attribution attack method, termed `DANAA', is proposed to obtain more accurate feature importance estimation. In our method, the model outputs are attributed to the middle layer based on an adversarial non-linear path. The goal is to measure the weight of individual neurons and retain the features that are more important towards transferability. We have conducted extensive experiments on the benchmark datasets to demonstrate the state-of-the-art performance of our method. Our code is available at: https://github.com/Davidjinzb/DANAA

摘要: 虽然深度神经网络在许多领域都有很好的效果，但它们容易受到攻击样本的干扰，从而导致错误的判断。特征级攻击是一种有效的攻击类型，它针对隐含层中的学习特征，以提高其在不同模型上的可移植性。然而，观察到神经元重要性估计结果在很大程度上影响了神经网络的可转移性。为了获得更准确的特征重要性估计，本文提出了一种双重对抗神经元属性攻击方法--DANAA。在我们的方法中，模型输出被归因于基于对抗性非线性路径的中间层。其目标是测量单个神经元的重量，并保留对可转移性更重要的特征。我们在基准数据集上进行了广泛的实验，以证明我们的方法具有最先进的性能。我们的代码请访问：https://github.com/Davidjinzb/DANAA



## **42. Privacy in Large Language Models: Attacks, Defenses and Future Directions**

大型语言模型中的隐私：攻击、防御和未来方向 cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10383v1) [paper-pdf](http://arxiv.org/pdf/2310.10383v1)

**Authors**: Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Yangqiu Song

**Abstract**: The advancement of large language models (LLMs) has significantly enhanced the ability to effectively tackle various downstream NLP tasks and unify these tasks into generative pipelines. On the one hand, powerful language models, trained on massive textual data, have brought unparalleled accessibility and usability for both models and users. On the other hand, unrestricted access to these models can also introduce potential malicious and unintentional privacy risks. Despite ongoing efforts to address the safety and privacy concerns associated with LLMs, the problem remains unresolved. In this paper, we provide a comprehensive analysis of the current privacy attacks targeting LLMs and categorize them according to the adversary's assumed capabilities to shed light on the potential vulnerabilities present in LLMs. Then, we present a detailed overview of prominent defense strategies that have been developed to counter these privacy attacks. Beyond existing works, we identify upcoming privacy concerns as LLMs evolve. Lastly, we point out several potential avenues for future exploration.

摘要: 大型语言模型(LLM)的发展极大地增强了有效地处理各种下游NLP任务并将这些任务统一到生成管道中的能力。一方面，强大的语言模型，基于海量文本数据的训练，为模型和用户带来了无与伦比的可及性和可用性。另一方面，不受限制地访问这些模型也可能带来潜在的恶意和无意的隐私风险。尽管正在努力解决与低密度脂蛋白相关的安全和隐私问题，但这个问题仍然没有得到解决。在本文中，我们对当前针对LLMS的隐私攻击进行了全面的分析，并根据对手假设的能力对它们进行了分类，以揭示LLMS中存在的潜在漏洞。然后，我们详细概述了为应对这些隐私攻击而开发的主要防御策略。除了现有的工作，我们发现随着LLM的发展，即将到来的隐私问题。最后，我们指出了未来可能的几个探索方向。



## **43. A White-Box False Positive Adversarial Attack Method on Contrastive Loss-Based Offline Handwritten Signature Verification Models**

基于对比损失的离线手写签名验证模型的白盒假阳性对抗攻击方法 cs.CV

8 pages, 2 figures

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2308.08925v2) [paper-pdf](http://arxiv.org/pdf/2308.08925v2)

**Authors**: Zhongliang Guo, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang

**Abstract**: In this paper, we tackle the challenge of white-box false positive adversarial attacks on contrastive loss-based offline handwritten signature verification models. We propose a novel attack method that treats the attack as a style transfer between closely related but distinct writing styles. To guide the generation of deceptive images, we introduce two new loss functions that enhance the attack success rate by perturbing the Euclidean distance between the embedding vectors of the original and synthesized samples, while ensuring minimal perturbations by reducing the difference between the generated image and the original image. Our method demonstrates state-of-the-art performance in white-box attacks on contrastive loss-based offline handwritten signature verification models, as evidenced by our experiments. The key contributions of this paper include a novel false positive attack method, two new loss functions, effective style transfer in handwriting styles, and superior performance in white-box false positive attacks compared to other white-box attack methods.

摘要: 针对基于对比损失的离线手写签名验证模型，提出了白盒假阳性对抗性攻击的挑战。我们提出了一种新的攻击方法，将攻击视为密切相关但截然不同的写作风格之间的一种风格转换。为了指导欺骗图像的生成，我们引入了两个新的损失函数，通过扰动原始样本和合成样本嵌入向量之间的欧几里德距离来提高攻击成功率，同时通过减小生成图像和原始图像之间的差异来确保最小的扰动。实验证明，我们的方法在基于对比损失的离线手写签名验证模型上具有最好的白盒攻击性能。本文的主要贡献包括一种新的误报攻击方法，两个新的损失函数，有效的笔迹风格转换，以及与其他白盒攻击方法相比具有更好的白盒误报攻击性能。



## **44. A Non-monotonic Smooth Activation Function**

一种非单调光滑激活函数 cs.LG

12 Pages

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10126v1) [paper-pdf](http://arxiv.org/pdf/2310.10126v1)

**Authors**: Koushik Biswas, Meghana Karri, Ulaş Bağcı

**Abstract**: Activation functions are crucial in deep learning models since they introduce non-linearity into the networks, allowing them to learn from errors and make adjustments, which is essential for learning complex patterns. The essential purpose of activation functions is to transform unprocessed input signals into significant output activations, promoting information transmission throughout the neural network. In this study, we propose a new activation function called Sqish, which is a non-monotonic and smooth function and an alternative to existing ones. We showed its superiority in classification, object detection, segmentation tasks, and adversarial robustness experiments. We got an 8.21% improvement over ReLU on the CIFAR100 dataset with the ShuffleNet V2 model in the FGSM adversarial attack. We also got a 5.87% improvement over ReLU on image classification on the CIFAR100 dataset with the ShuffleNet V2 model.

摘要: 激活函数在深度学习模型中至关重要，因为它们将非线性引入网络，使网络能够从错误中学习并进行调整，这对学习复杂模式至关重要。激活函数的基本目的是将未经处理的输入信号转换为重要的输出激活，从而促进整个神经网络中的信息传输。在这项研究中，我们提出了一种新的激活函数，称为SQISH，它是一个非单调的光滑函数，可以替代现有的激活函数。我们在分类、目标检测、分割任务和对手健壮性实验中展示了它的优越性。在FGSM对抗性攻击中，我们在CIFAR100数据集上使用ShuffleNet V2模型比RELU提高了8.21%。在使用ShuffleNet V2模型的CIFAR100数据集上，我们的分类性能也比RELU提高了5.87%。



## **45. Evading Detection Actively: Toward Anti-Forensics against Forgery Localization**

主动逃避侦查：走向反取证反伪证本土化 cs.CV

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10036v1) [paper-pdf](http://arxiv.org/pdf/2310.10036v1)

**Authors**: Long Zhuo, Shenghai Luo, Shunquan Tan, Han Chen, Bin Li, Jiwu Huang

**Abstract**: Anti-forensics seeks to eliminate or conceal traces of tampering artifacts. Typically, anti-forensic methods are designed to deceive binary detectors and persuade them to misjudge the authenticity of an image. However, to the best of our knowledge, no attempts have been made to deceive forgery detectors at the pixel level and mis-locate forged regions. Traditional adversarial attack methods cannot be directly used against forgery localization due to the following defects: 1) they tend to just naively induce the target forensic models to flip their pixel-level pristine or forged decisions; 2) their anti-forensics performance tends to be severely degraded when faced with the unseen forensic models; 3) they lose validity once the target forensic models are retrained with the anti-forensics images generated by them. To tackle the three defects, we propose SEAR (Self-supErvised Anti-foRensics), a novel self-supervised and adversarial training algorithm that effectively trains deep-learning anti-forensic models against forgery localization. SEAR sets a pretext task to reconstruct perturbation for self-supervised learning. In adversarial training, SEAR employs a forgery localization model as a supervisor to explore tampering features and constructs a deep-learning concealer to erase corresponding traces. We have conducted largescale experiments across diverse datasets. The experimental results demonstrate that, through the combination of self-supervised learning and adversarial learning, SEAR successfully deceives the state-of-the-art forgery localization methods, as well as tackle the three defects regarding traditional adversarial attack methods mentioned above.

摘要: 反取证旨在消除或隐藏篡改文物的痕迹。通常，反取证方法旨在欺骗双星探测器，并说服他们误判图像的真实性。然而，据我们所知，没有人试图在像素级别欺骗伪造探测器，也没有错误定位伪造区域。传统的对抗性攻击方法由于以下缺陷不能直接用于伪证定位：1)它们往往只是幼稚地诱导目标取证模型反转其像素级原始或伪造的决策；2)当面对看不见的取证模型时，它们的反取证性能往往严重下降；3)一旦目标取证模型被用它们生成的反取证图像重新训练，它们就失去了有效性。针对这三个缺陷，我们提出了一种新的自监督对抗性训练算法SEAR(Self-Supervised Anti-Forensics)，该算法能够有效地训练深度学习反取证模型，防止伪证定位。SEAR设置了一个借口任务来重构用于自我监督学习的扰动。在对抗性训练中，SEAR使用伪本地化模型作为监督者来探索篡改特征，并构造一个深度学习隐蔽器来消除相应的痕迹。我们已经在不同的数据集上进行了大规模的实验。实验结果表明，通过自监督学习和对抗性学习的结合，SEAR成功地欺骗了现有的伪造定位方法，解决了传统对抗性攻击方法的三大缺陷。



## **46. Black-box Targeted Adversarial Attack on Segment Anything (SAM)**

针对Segment Anything(SAM)的黑箱定向对抗性攻击 cs.CV

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10010v1) [paper-pdf](http://arxiv.org/pdf/2310.10010v1)

**Authors**: Sheng Zheng, Chaoning Zhang

**Abstract**: Deep recognition models are widely vulnerable to adversarial examples, which change the model output by adding quasi-imperceptible perturbation to the image input. Recently, Segment Anything Model (SAM) has emerged to become a popular foundation model in computer vision due to its impressive generalization to unseen data and tasks. Realizing flexible attacks on SAM is beneficial for understanding the robustness of SAM in the adversarial context. To this end, this work aims to achieve a targeted adversarial attack (TAA) on SAM. Specifically, under a certain prompt, the goal is to make the predicted mask of an adversarial example resemble that of a given target image. The task of TAA on SAM has been realized in a recent arXiv work in the white-box setup by assuming access to prompt and model, which is thus less practical. To address the issue of prompt dependence, we propose a simple yet effective approach by only attacking the image encoder. Moreover, we propose a novel regularization loss to enhance the cross-model transferability by increasing the feature dominance of adversarial images over random natural images. Extensive experiments verify the effectiveness of our proposed simple techniques to conduct a successful black-box TAA on SAM.

摘要: 深度识别模型很容易受到敌意例子的影响，这些例子通过在图像输入中添加准不可察觉的扰动来改变模型输出。近年来，分段任意模型(Segment Anything Model，SAM)以其对未知数据和任务的良好泛化能力，成为计算机视觉中一种流行的基础模型。实现对SAM的灵活攻击有助于理解SAM在对抗环境下的健壮性。为此，本工作旨在实现对SAM的有针对性的对抗性攻击。具体地说，在一定的提示下，目标是使对抗性例子的预测掩模与给定目标图像的掩模相似。SAM上的TAA任务已经在最近的白盒设置中通过假设对提示和模型的访问来实现，因此这是不太实际的。为了解决提示依赖的问题，我们提出了一种简单而有效的方法，只攻击图像编码器。此外，我们提出了一种新的正则化损失，通过增加对抗性图像对随机自然图像的特征优势来增强跨模型的可转移性。大量的实验验证了我们提出的简单技术在SAM上成功进行黑盒TAA的有效性。



## **47. Towards Deep Learning Models Resistant to Transfer-based Adversarial Attacks via Data-centric Robust Learning**

基于数据中心稳健学习的抗迁移攻击深度学习模型研究 cs.CR

9 pages

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2310.09891v1) [paper-pdf](http://arxiv.org/pdf/2310.09891v1)

**Authors**: Yulong Yang, Chenhao Lin, Xiang Ji, Qiwei Tian, Qian Li, Hongshan Yang, Zhibo Wang, Chao Shen

**Abstract**: Transfer-based adversarial attacks raise a severe threat to real-world deep learning systems since they do not require access to target models. Adversarial training (AT), which is recognized as the strongest defense against white-box attacks, has also guaranteed high robustness to (black-box) transfer-based attacks. However, AT suffers from heavy computational overhead since it optimizes the adversarial examples during the whole training process. In this paper, we demonstrate that such heavy optimization is not necessary for AT against transfer-based attacks. Instead, a one-shot adversarial augmentation prior to training is sufficient, and we name this new defense paradigm Data-centric Robust Learning (DRL). Our experimental results show that DRL outperforms widely-used AT techniques (e.g., PGD-AT, TRADES, EAT, and FAT) in terms of black-box robustness and even surpasses the top-1 defense on RobustBench when combined with diverse data augmentations and loss regularizations. We also identify other benefits of DRL, for instance, the model generalization capability and robust fairness.

摘要: 基于迁移的对抗性攻击对现实世界的深度学习系统构成了严重威胁，因为它们不需要访问目标模型。对抗性训练(AT)被公认为对白盒攻击的最强防御，也保证了对(黑盒)基于传输的攻击的高健壮性。然而，由于AT在整个训练过程中对对手实例进行优化，因此其计算开销很大。在这篇文章中，我们证明了这种繁重的优化对于AT抵抗基于传输的攻击是不必要的。相反，在训练之前进行一次对抗性增强就足够了，我们将这种新的防御范式命名为数据中心稳健学习(DRL)。我们的实验结果表明，DRL在黑盒稳健性方面优于广泛使用的AT技术(如PGD-AT、TRADS、EAT和FAT)，当结合不同的数据增强和损失规则时，甚至超过RobustBitch的TOP-1防御。我们还确定了DRL的其他优点，例如，模型泛化能力和稳健的公平性。



## **48. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

DIFFender：基于扩散的补丁攻击对抗性防御 cs.CV

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2306.09124v2) [paper-pdf](http://arxiv.org/pdf/2306.09124v2)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is not satisfactory. In this paper, we propose DIFFender, a novel defense method that leverages a text-guided diffusion model to defend against adversarial patches. DIFFender includes two main stages: patch localization and patch restoration. In the localization stage, we find and exploit an intriguing property of the diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ the diffusion model to reconstruct the adversarial regions in the images while preserving the integrity of the visual content. Importantly, these two stages are carefully guided by a unified diffusion model, thus we can utilize the close interaction between them to improve the whole defense performance. Moreover, we propose a few-shot prompt-tuning algorithm to fine-tune the diffusion model, enabling the pre-trained diffusion model to easily adapt to the defense task. We conduct extensive experiments on the image classification and face recognition tasks, demonstrating that our proposed method exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple patch attack methods.

摘要: 对抗性攻击，尤其是补丁攻击，对深度学习模型的健壮性和可靠性构成了严重的威胁。开发可靠的防御补丁攻击对于现实世界的应用是至关重要的，但目前在这一领域的研究并不令人满意。在本文中，我们提出了一种新的防御方法DIFFender，它利用文本引导的扩散模型来防御恶意补丁。DIFFender包括两个主要阶段：补丁定位和补丁恢复。在本地化阶段，我们发现并利用扩散模型的一个有趣的性质来有效地识别敌方补丁的位置。在恢复阶段，我们使用扩散模型来重建图像中的对抗性区域，同时保持视觉内容的完整性。重要的是，这两个阶段都被一个统一的扩散模型谨慎地指导着，因此我们可以利用它们之间的密切互动来提高整体防御性能。此外，我们还提出了几次快速调整算法来微调扩散模型，使预先训练好的扩散模型能够很容易地适应防御任务。我们在图像分类和人脸识别任务上进行了大量的实验，证明了我们的方法在强自适应攻击下表现出了良好的鲁棒性，并且能够很好地适用于各种场景、不同的分类器和多种补丁攻击方法。



## **49. Are Your Explanations Reliable? Investigating the Stability of LIME in Explaining Text Classifiers by Marrying XAI and Adversarial Attack**

你的解释可靠吗？结合Xai和对抗性攻击考察LIME在解释文本分类器中的稳定性 cs.LG

14 pages, 6 figures. Replacement by the updated version to be  published in EMNLP 2023

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2305.12351v2) [paper-pdf](http://arxiv.org/pdf/2305.12351v2)

**Authors**: Christopher Burger, Lingwei Chen, Thai Le

**Abstract**: LIME has emerged as one of the most commonly referenced tools in explainable AI (XAI) frameworks that is integrated into critical machine learning applications--e.g., healthcare and finance. However, its stability remains little explored, especially in the context of text data, due to the unique text-space constraints. To address these challenges, in this paper, we first evaluate the inherent instability of LIME on text data to establish a baseline, and then propose a novel algorithm XAIFooler to perturb text inputs and manipulate explanations that casts investigation on the stability of LIME as a text perturbation optimization problem. XAIFooler conforms to the constraints to preserve text semantics and original prediction with small perturbations, and introduces Rank-biased Overlap (RBO) as a key part to guide the optimization of XAIFooler that satisfies all the requirements for explanation similarity measure. Extensive experiments on real-world text datasets demonstrate that XAIFooler significantly outperforms all baselines by large margins in its ability to manipulate LIME's explanations with high semantic preservability.

摘要: LIME已成为可解释人工智能(XAI)框架中最常引用的工具之一，该框架集成到关键的机器学习应用程序中--例如医疗保健和金融。然而，由于独特的文本空间限制，它的稳定性仍然很少被研究，特别是在文本数据的上下文中。为了应对这些挑战，本文首先评估了LIME对文本数据的内在不稳定性以建立基线，然后提出了一种新的算法XAIFooler来扰动文本输入并操纵解释，将LIME的稳定性研究归结为一个文本扰动优化问题。XAIFooler遵循在小扰动下保持文本语义和原始预测的约束，并引入偏序重叠(RBO)作为关键部分来指导XAIFooler的优化，使其满足解释相似性度量的所有要求。在真实文本数据集上的大量实验表明，XAIFooler在处理LIME的解释方面远远超过所有基线，具有很高的语义保存性。



## **50. Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

拜占庭稳健合作多智能体强化学习的贝叶斯博弈 cs.GT

**SubmitDate**: 2023-10-15    [abs](http://arxiv.org/abs/2305.12872v2) [paper-pdf](http://arxiv.org/pdf/2305.12872v2)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Ruixiao Xu, Xin Yu, Jiakai Wang, Aishan Liu, Yaodong Yang, Xianglong Liu

**Abstract**: In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex post robust Bayesian Markov perfect equilibrium, which we proof to exist and weakly dominates the equilibrium of previous robust MARL approaches. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experimentation on matrix games, level-based foraging and StarCraft II indicate that, even under worst-case perturbations, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies, demonstrating resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.

摘要: 在这项研究中，我们探讨了协作多智能体强化学习(c-Marl)对拜占庭故障的稳健性，在拜占庭故障中，任何智能体都可以由于故障或对手攻击而执行任意的、最坏的操作。为了解决任何智能体都可能是对抗性的不确定性，我们提出了一种贝叶斯对抗性鲁棒DEC-POMDP(BARDEC-POMDP)框架，该框架将拜占庭对手视为自然决定的类型，由单独的转换表示。这使代理能够基于他们对其他代理类型的后验信念来学习策略，促进与确定的盟友的合作，并将受到对手操纵的脆弱性降至最低。我们将BARDEC-POMDP的最优解定义为一个事后稳健的贝叶斯马尔可夫完全均衡，并证明了它的存在，并且弱控制了以前的稳健Marl方法的均衡。为了实现这一均衡，我们提出了一个在特定条件下几乎必然收敛的双时间尺度的行动者-批评者算法。在矩阵游戏、基于关卡的觅食和星际争霸II上的实验表明，即使在最坏的情况下，我们的方法也成功地获得了复杂的微观管理技能，并自适应地与盟友结盟，展示了对非遗忘对手、随机盟友、基于观察的攻击和基于转移的攻击的弹性。



