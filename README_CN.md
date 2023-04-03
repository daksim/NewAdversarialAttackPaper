# Latest Adversarial Attack Papers
**update at 2023-04-03 09:09:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Packet-Level Adversarial Network Traffic Crafting using Sequence Generative Adversarial Networks**

基于序列生成式对抗网络的分组级对抗网络流量制作 cs.CR

The authors agreed to withdraw the manuscript due to privacy reason

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2103.04794v2) [paper-pdf](http://arxiv.org/pdf/2103.04794v2)

**Authors**: Qiumei Cheng, Shiying Zhou, Yi Shen, Dezhang Kong, Chunming Wu

**Abstract**: The surge in the internet of things (IoT) devices seriously threatens the current IoT security landscape, which requires a robust network intrusion detection system (NIDS). Despite superior detection accuracy, existing machine learning or deep learning based NIDS are vulnerable to adversarial examples. Recently, generative adversarial networks (GANs) have become a prevailing method in adversarial examples crafting. However, the nature of discrete network traffic at the packet level makes it hard for GAN to craft adversarial traffic as GAN is efficient in generating continuous data like image synthesis. Unlike previous methods that convert discrete network traffic into a grayscale image, this paper gains inspiration from SeqGAN in sequence generation with policy gradient. Based on the structure of SeqGAN, we propose Attack-GAN to generate adversarial network traffic at packet level that complies with domain constraints. Specifically, the adversarial packet generation is formulated into a sequential decision making process. In this case, each byte in a packet is regarded as a token in a sequence. The objective of the generator is to select a token to maximize its expected end reward. To bypass the detection of NIDS, the generated network traffic and benign traffic are classified by a black-box NIDS. The prediction results returned by the NIDS are fed into the discriminator to guide the update of the generator. We generate malicious adversarial traffic based on a real public available dataset with attack functionality unchanged. The experimental results validate that the generated adversarial samples are able to deceive many existing black-box NIDS.

摘要: 物联网(IoT)设备的激增严重威胁到当前的物联网安全格局，这需要一个强大的网络入侵检测系统(NIDS)。尽管检测精度很高，但现有的基于机器学习或深度学习的网络入侵检测系统很容易受到敌意例子的攻击。近年来，生成性对抗性网络(GANS)已成为对抗性实例制作中的一种流行方法。然而，分组级别的离散网络流量的性质使得GAN很难创建敌意流量，因为GAN在生成图像合成等连续数据方面是有效的。不同于以往将离散网络流量转换为灰度图像的方法，本文从策略梯度序列生成中得到了SeqGAN的启发。基于SeqGAN的结构，我们提出了攻击GAN，在包级生成符合域约束的敌意网络流量。具体地说，敌意分组生成被表述为连续的决策过程。在这种情况下，分组中的每个字节被视为序列中的令牌。生成器的目标是选择一个令牌来最大化其预期的最终回报。为了绕过网络入侵检测系统的检测，生成的网络流量和良性流量被黑盒网络入侵检测系统分类。网络入侵检测系统返回的预测结果被送入鉴别器以指导生成器的更新。我们基于真实的公共可用数据集生成恶意敌意流量，攻击功能保持不变。实验结果表明，生成的对抗性样本能够欺骗许多已有的黑盒网络入侵检测系统。



## **2. Fooling Polarization-based Vision using Locally Controllable Polarizing Projection**

利用局部可控偏振投影愚弄偏振视觉 cs.CV

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17890v1) [paper-pdf](http://arxiv.org/pdf/2303.17890v1)

**Authors**: Zhuoxiao Li, Zhihang Zhong, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng

**Abstract**: Polarization is a fundamental property of light that encodes abundant information regarding surface shape, material, illumination and viewing geometry. The computer vision community has witnessed a blossom of polarization-based vision applications, such as reflection removal, shape-from-polarization, transparent object segmentation and color constancy, partially due to the emergence of single-chip mono/color polarization sensors that make polarization data acquisition easier than ever. However, is polarization-based vision vulnerable to adversarial attacks? If so, is that possible to realize these adversarial attacks in the physical world, without being perceived by human eyes? In this paper, we warn the community of the vulnerability of polarization-based vision, which can be more serious than RGB-based vision. By adapting a commercial LCD projector, we achieve locally controllable polarizing projection, which is successfully utilized to fool state-of-the-art polarization-based vision algorithms for glass segmentation and color constancy. Compared with existing physical attacks on RGB-based vision, which always suffer from the trade-off between attack efficacy and eye conceivability, the adversarial attackers based on polarizing projection are contact-free and visually imperceptible, since naked human eyes can rarely perceive the difference of viciously manipulated polarizing light and ordinary illumination. This poses unprecedented risks on polarization-based vision, both in the monochromatic and trichromatic domain, for which due attentions should be paid and counter measures be considered.

摘要: 偏振是光的一个基本属性，它编码了关于表面形状、材料、照明和观察几何的丰富信息。计算机视觉领域已经见证了基于偏振的视觉应用的蓬勃发展，例如反射去除、从偏振形状、透明对象分割和颜色恒定，部分原因是单芯片单色/颜色偏振传感器的出现使得偏振数据的获取比以往任何时候都更加容易。然而，基于极化的愿景容易受到对手的攻击吗？如果是这样的话，有可能在物理世界中实现这些对抗性攻击，而不被人眼察觉吗？在本文中，我们警告社区基于偏振的视觉的脆弱性，这可能比基于RGB的视觉更严重。通过采用商用LCD投影仪，实现了局部可控的偏振投影，并成功地将其用于欺骗最先进的基于偏振的视觉算法，以实现玻璃分割和颜色恒定。与现有的基于RGB视觉的物理攻击相比，基于偏振投影的对抗性攻击者是非接触式的，视觉上不可感知，因为肉眼很少察觉到恶意操纵的偏振光和普通照明的差异。这给基于偏振的视觉带来了前所未有的风险，无论是在单色领域还是在三色领域，都应该给予应有的关注，并考虑采取对策。



## **3. Pentimento: Data Remanence in Cloud FPGAs**

Pentimento：云现场可编程门阵列中的数据存储 cs.CR

17 Pages, 8 Figures

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17881v1) [paper-pdf](http://arxiv.org/pdf/2303.17881v1)

**Authors**: Colin Drewes, Olivia Weng, Andres Meza, Alric Althoff, David Kohlbrenner, Ryan Kastner, Dustin Richmond

**Abstract**: Cloud FPGAs strike an alluring balance between computational efficiency, energy efficiency, and cost. It is the flexibility of the FPGA architecture that enables these benefits, but that very same flexibility that exposes new security vulnerabilities. We show that a remote attacker can recover "FPGA pentimenti" - long-removed secret data belonging to a prior user of a cloud FPGA. The sensitive data constituting an FPGA pentimento is an analog imprint from bias temperature instability (BTI) effects on the underlying transistors. We demonstrate how this slight degradation can be measured using a time-to-digital (TDC) converter when an adversary programs one into the target cloud FPGA.   This technique allows an attacker to ascertain previously safe information on cloud FPGAs, even after it is no longer explicitly present. Notably, it can allow an attacker who knows a non-secret "skeleton" (the physical structure, but not the contents) of the victim's design to (1) extract proprietary details from an encrypted FPGA design image available on the AWS marketplace and (2) recover data loaded at runtime by a previous user of a cloud FPGA using a known design. Our experiments show that BTI degradation (burn-in) and recovery are measurable and constitute a security threat to commercial cloud FPGAs.

摘要: 云现场可编程门阵列在计算效率、能源效率和成本之间取得了诱人的平衡。正是FPGA架构的灵活性实现了这些优势，但同样的灵活性也暴露了新的安全漏洞。我们展示了远程攻击者可以恢复“fpga pentimenti”--属于云fpga先前用户的长时间删除的秘密数据。构成现场可编程门阵列的敏感数据是偏置温度不稳定性(BTI)对底层晶体管影响的模拟印记。我们演示了当对手将时间-数字(TDC)转换器编程到目标云FPGA中时，如何测量这种轻微的降级。这项技术允许攻击者确定云现场可编程门阵列上以前的安全信息，即使它不再显式存在。值得注意的是，它可以让知道受害者设计的非机密“骨架”(物理结构，但不是内容)的攻击者(1)从AWS Marketplace上提供的加密的FPGA设计图像中提取专有细节，以及(2)恢复云FPGA的前用户在运行时使用已知设计加载的数据。我们的实验表明，BTI的退化(老化)和恢复是可测量的，并对商用云现场可编程门阵列构成安全威胁。



## **4. The Blockchain Imitation Game**

区块链模仿游戏 cs.CR

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17877v1) [paper-pdf](http://arxiv.org/pdf/2303.17877v1)

**Authors**: Kaihua Qin, Stefanos Chaliasos, Liyi Zhou, Benjamin Livshits, Dawn Song, Arthur Gervais

**Abstract**: The use of blockchains for automated and adversarial trading has become commonplace. However, due to the transparent nature of blockchains, an adversary is able to observe any pending, not-yet-mined transactions, along with their execution logic. This transparency further enables a new type of adversary, which copies and front-runs profitable pending transactions in real-time, yielding significant financial gains.   Shedding light on such "copy-paste" malpractice, this paper introduces the Blockchain Imitation Game and proposes a generalized imitation attack methodology called Ape. Leveraging dynamic program analysis techniques, Ape supports the automatic synthesis of adversarial smart contracts. Over a timeframe of one year (1st of August, 2021 to 31st of July, 2022), Ape could have yielded 148.96M USD in profit on Ethereum, and 42.70M USD on BNB Smart Chain (BSC).   Not only as a malicious attack, we further show the potential of transaction and contract imitation as a defensive strategy. Within one year, we find that Ape could have successfully imitated 13 and 22 known Decentralized Finance (DeFi) attacks on Ethereum and BSC, respectively. Our findings suggest that blockchain validators can imitate attacks in real-time to prevent intrusions in DeFi.

摘要: 使用区块链进行自动化和对抗性交易已变得司空见惯。然而，由于区块链的透明性质，对手能够观察到任何未决的、尚未挖掘的事务及其执行逻辑。这种透明度进一步催生了一种新型的对手，它实时复制并领先于盈利的待决交易，产生了巨大的财务收益。针对这种“复制-粘贴”的弊端，本文介绍了区块链模仿游戏，并提出了一种通用的模仿攻击方法论APE。利用动态程序分析技术，APE支持对抗性智能合同的自动合成。在一年的时间范围内(2021年8月1日至2022年7月31日)，Ape在以太上获得了1.4896亿美元的利润，在BNB Smart Chain(BSC)上获得了427万美元的利润。不仅作为一种恶意攻击，我们进一步展示了交易和合同模仿作为一种防御策略的潜力。我们发现，在一年内，APE可以成功地模仿已知的对以太和BSC的13次和22次去中心化金融(Defi)攻击。我们的发现表明，区块链验证器可以实时模拟攻击，以防止入侵Defi。



## **5. Towards Adversarially Robust Continual Learning**

走向对抗性稳健的持续学习 cs.LG

ICASSP 2023

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17764v1) [paper-pdf](http://arxiv.org/pdf/2303.17764v1)

**Authors**: Tao Bai, Chen Chen, Lingjuan Lyu, Jun Zhao, Bihan Wen

**Abstract**: Recent studies show that models trained by continual learning can achieve the comparable performances as the standard supervised learning and the learning flexibility of continual learning models enables their wide applications in the real world. Deep learning models, however, are shown to be vulnerable to adversarial attacks. Though there are many studies on the model robustness in the context of standard supervised learning, protecting continual learning from adversarial attacks has not yet been investigated. To fill in this research gap, we are the first to study adversarial robustness in continual learning and propose a novel method called \textbf{T}ask-\textbf{A}ware \textbf{B}oundary \textbf{A}ugmentation (TABA) to boost the robustness of continual learning models. With extensive experiments on CIFAR-10 and CIFAR-100, we show the efficacy of adversarial training and TABA in defending adversarial attacks.

摘要: 最近的研究表明，通过持续学习训练的模型可以达到与标准监督学习相当的性能，而持续学习模型的学习灵活性使其在现实世界中得到了广泛的应用。然而，深度学习模型被证明容易受到对手的攻击。虽然已经有很多关于标准监督学习背景下的模型稳健性的研究，但保护连续学习免受对手攻击的研究尚未见报道。为了填补这一研究空白，我们首次对持续学习中的对手稳健性进行了研究，并提出了一种新的方法--Taba(Taba)来提高持续学习模型的稳健性。通过在CIFAR-10和CIFAR-100上的大量实验，我们展示了对抗性训练和Taba在防御对抗性攻击方面的有效性。



## **6. CitySpec with Shield: A Secure Intelligent Assistant for Requirement Formalization**

CitySpec with Shield：需求形式化的安全智能助手 cs.AI

arXiv admin note: substantial text overlap with arXiv:2206.03132

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2302.09665v2) [paper-pdf](http://arxiv.org/pdf/2302.09665v2)

**Authors**: Zirong Chen, Issa Li, Haoxiang Zhang, Sarah Preum, John A. Stankovic, Meiyi Ma

**Abstract**: An increasing number of monitoring systems have been developed in smart cities to ensure that the real-time operations of a city satisfy safety and performance requirements. However, many existing city requirements are written in English with missing, inaccurate, or ambiguous information. There is a high demand for assisting city policymakers in converting human-specified requirements to machine-understandable formal specifications for monitoring systems. To tackle this limitation, we build CitySpec, the first intelligent assistant system for requirement specification in smart cities. To create CitySpec, we first collect over 1,500 real-world city requirements across different domains (e.g., transportation and energy) from over 100 cities and extract city-specific knowledge to generate a dataset of city vocabulary with 3,061 words. We also build a translation model and enhance it through requirement synthesis and develop a novel online learning framework with shielded validation. The evaluation results on real-world city requirements show that CitySpec increases the sentence-level accuracy of requirement specification from 59.02% to 86.64%, and has strong adaptability to a new city and a new domain (e.g., the F1 score for requirements in Seattle increases from 77.6% to 93.75% with online learning). After the enhancement from the shield function, CitySpec is now immune to most known textual adversarial inputs (e.g., the attack success rate of DeepWordBug after the shield function is reduced to 0% from 82.73%). We test the CitySpec with 18 participants from different domains. CitySpec shows its strong usability and adaptability to different domains, and also its robustness to malicious inputs.

摘要: 智能城市中开发了越来越多的监控系统，以确保城市的实时运行满足安全和性能要求。然而，许多现有的城市要求都是用英语编写的，信息缺失、不准确或含糊不清。对协助城市政策制定者将人类指定的要求转换为机器可理解的监测系统正式规范的需求很高。为了解决这一局限性，我们建立了CitySpec，这是智能城市中第一个用于需求规范的智能辅助系统。为了创建CitySpec，我们首先从100多个城市收集超过1500个不同领域(例如交通和能源)的真实城市需求，并提取特定于城市的知识来生成包含3061个单词的城市词汇数据库。我们还构建了翻译模型，并通过需求合成对其进行了增强，并开发了一种新型的屏蔽验证的在线学习框架。对实际城市需求的评估结果表明，CitySpec将需求描述的句子级准确率从59.02%提高到86.64%，并且对新城市和新领域具有很强的适应性(例如，通过在线学习，西雅图需求的F1得分从77.6%提高到93.75%)。在盾牌功能增强后，CitySpec现在对大多数已知的文本对手输入免疫(例如，盾牌功能将DeepWordBug的攻击成功率从82.73%降低到0%)。我们测试了来自不同领域的18名参与者的CitySpec。CitySpec显示了其强大的可用性和对不同领域的适应性，以及对恶意输入的健壮性。



## **7. Generating Adversarial Samples in Mini-Batches May Be Detrimental To Adversarial Robustness**

小批量生成敌方样本可能会损害敌方的健壮性 cs.LG

6 pages, 3 figures

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.17720v1) [paper-pdf](http://arxiv.org/pdf/2303.17720v1)

**Authors**: Timothy Redgrave, Colton Crum

**Abstract**: Neural networks have been proven to be both highly effective within computer vision, and highly vulnerable to adversarial attacks. Consequently, as the use of neural networks increases due to their unrivaled performance, so too does the threat posed by adversarial attacks. In this work, we build towards addressing the challenge of adversarial robustness by exploring the relationship between the mini-batch size used during adversarial sample generation and the strength of the adversarial samples produced. We demonstrate that an increase in mini-batch size results in a decrease in the efficacy of the samples produced, and we draw connections between these observations and the phenomenon of vanishing gradients. Next, we formulate loss functions such that adversarial sample strength is not degraded by mini-batch size. Our findings highlight a potential risk for underestimating the true (practical) strength of adversarial attacks, and a risk of overestimating a model's robustness. We share our codes to let others replicate our experiments and to facilitate further exploration of the connections between batch size and adversarial sample strength.

摘要: 神经网络已经被证明在计算机视觉中是高度有效的，并且非常容易受到对手的攻击。因此，随着神经网络因其无与伦比的性能而越来越多地使用，对抗性攻击构成的威胁也随之增加。在这项工作中，我们通过探索在对抗样本生成过程中使用的小批量大小与产生的对抗样本强度之间的关系，来解决对抗稳健性的挑战。我们证明，小批量大小的增加会导致所生产样本的有效性降低，我们将这些观察结果与梯度消失现象联系起来。接下来，我们制定损失函数，使得对抗性样本强度不会因小批量大小而退化。我们的发现突出了低估对抗性攻击的真实(实际)强度的潜在风险，以及高估模型的稳健性的风险。我们分享我们的代码，让其他人复制我们的实验，并促进进一步探索批次大小和敌对样本强度之间的联系。



## **8. TorKameleon: Improving Tor's Censorship Resistance With K-anonimization and Media-based Covert Channels**

TorKameleon：通过K-无名化和基于媒体的隐蔽渠道提高Tor的审查抵抗力 cs.CR

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.17544v1) [paper-pdf](http://arxiv.org/pdf/2303.17544v1)

**Authors**: João Afonso Vilalonga, João S. Resende, Henrique Domingos

**Abstract**: The use of anonymity networks such as Tor and similar tools can greatly enhance the privacy and anonymity of online communications. Tor, in particular, is currently the most widely used system for ensuring anonymity on the Internet. However, recent research has shown that Tor is vulnerable to correlation attacks carried out by state-level adversaries or colluding Internet censors. Therefore, new and more effective solutions emerged to protect online anonymity. Promising results have been achieved by implementing covert channels based on media traffic in modern anonymization systems, which have proven to be a reliable and practical approach to defend against powerful traffic correlation attacks. In this paper, we present TorKameleon, a censorship evasion solution that better protects Tor users from powerful traffic correlation attacks carried out by state-level adversaries. TorKameleon can be used either as a fully integrated Tor pluggable transport or as a standalone anonymization system that uses K-anonymization and encapsulation of user traffic in covert media channels. Our main goal is to protect users from machine and deep learning correlation attacks on anonymization networks like Tor. We have developed the TorKameleon prototype and performed extensive validations to verify the accuracy and experimental performance of the proposed solution in the Tor environment, including state-of-the-art active correlation attacks. As far as we know, we are the first to develop and study a system that uses both anonymization mechanisms described above against active correlation attacks.

摘要: 使用Tor等匿名网络和类似工具可以极大地增强在线通信的隐私和匿名性。特别是，Tor是目前在互联网上使用最广泛的匿名系统。然而，最近的研究表明，Tor很容易受到国家级对手或串通的互联网审查机构进行的关联攻击。因此，出现了新的、更有效的解决方案来保护在线匿名性。通过在现代匿名化系统中实现基于媒体流量的隐蔽通道，已经取得了可喜的结果，这被证明是一种可靠和实用的方法来防御强大的流量相关攻击。在本文中，我们提出了TorKameleon，一个审查逃避解决方案，可以更好地保护Tor用户免受来自国家级对手的强大流量关联攻击。TorKameleon既可以用作完全集成的ToR可插拔传输，也可以用作独立的匿名系统，该系统使用K-匿名化和对隐蔽媒体通道中的用户流量进行封装。我们的主要目标是保护用户免受Tor等匿名化网络上的机器和深度学习关联攻击。我们已经开发了TorKameleon原型，并进行了广泛的验证，以验证所提出的解决方案在ToR环境下的准确性和实验性能，包括最新的主动相关攻击。据我们所知，我们是第一个开发和研究使用上述两种匿名化机制来对抗主动相关攻击的系统。



## **9. Boosting Physical Layer Black-Box Attacks with Semantic Adversaries in Semantic Communications**

语义通信中利用语义对手增强物理层黑盒攻击 eess.SP

accepted by ICC2023

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.16523v2) [paper-pdf](http://arxiv.org/pdf/2303.16523v2)

**Authors**: Zeju Li, Xinghan Liu, Guoshun Nan, Jinfei Zhou, Xinchen Lyu, Qimei Cui, Xiaofeng Tao

**Abstract**: End-to-end semantic communication (ESC) system is able to improve communication efficiency by only transmitting the semantics of the input rather than raw bits. Although promising, ESC has also been shown susceptible to the crafted physical layer adversarial perturbations due to the openness of wireless channels and the sensitivity of neural models. Previous works focus more on the physical layer white-box attacks, while the challenging black-box ones, as more practical adversaries in real-world cases, are still largely under-explored. To this end, we present SemBLK, a novel method that can learn to generate destructive physical layer semantic attacks for an ESC system under the black-box setting, where the adversaries are imperceptible to humans. Specifically, 1) we first introduce a surrogate semantic encoder and train its parameters by exploring a limited number of queries to an existing ESC system. 2) Equipped with such a surrogate encoder, we then propose a novel semantic perturbation generation method to learn to boost the physical layer attacks with semantic adversaries. Experiments on two public datasets show the effectiveness of our proposed SemBLK in attacking the ESC system under the black-box setting. Finally, we provide case studies to visually justify the superiority of our physical layer semantic perturbations.

摘要: 端到端语义通信(ESC)系统能够通过只传输输入的语义而不是原始比特来提高通信效率。尽管ESC前景看好，但由于无线信道的开放性和神经模型的敏感性，ESC也被证明对精心设计的物理层对抗性扰动很敏感。以前的工作更多地关注物理层的白盒攻击，而具有挑战性的黑盒攻击，作为现实世界中更实际的对手，在很大程度上仍然没有得到充分的探索。为此，我们提出了一种新的方法SemBLK，它可以学习在黑盒环境下对ESC系统产生破坏性的物理层语义攻击，其中对手对人类是不可察觉的。具体来说，1)我们首先引入了一个代理语义编码器，并通过探索对现有ESC系统的有限数量的查询来训练它的参数。2)在代理编码器的基础上，提出了一种新的语义扰动生成方法，学习如何利用语义攻击增强物理层攻击。在两个公开数据集上的实验表明，本文提出的SemBLK在黑盒环境下对ESC系统的攻击是有效的。最后，我们提供了案例研究，以直观地证明我们的物理层语义扰动的优越性。



## **10. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

训练数据重构的非渐近下界 cs.LG

Corrected minor typos and restructured appendix

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.16372v2) [paper-pdf](http://arxiv.org/pdf/2303.16372v2)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We investigate semantic guarantees of private learning algorithms for their resilience to training Data Reconstruction Attacks (DRAs) by informed adversaries. To this end, we derive non-asymptotic minimax lower bounds on the adversary's reconstruction error against learners that satisfy differential privacy (DP) and metric differential privacy (mDP). Furthermore, we demonstrate that our lower bound analysis for the latter also covers the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget. Motivated by the theoretical improvements conferred by metric DP, we extend the privacy analysis of popular deep learning algorithms such as DP-SGD and Projected Noisy SGD to cover the broader notion of metric differential privacy.

摘要: 我们研究了私有学习算法的语义保证，因为它们对来自知情对手的训练数据重建攻击(DRA)具有韧性。为此，我们得到了满足差分隐私(DP)和度量差分隐私(MDP)的敌手重构误差的非渐近极大下界。此外，我们证明了对后者的下界分析也涵盖了高维机制，其中，输入数据的维度可能大于对手的查询预算。在度量DP的理论改进的推动下，我们扩展了DP-SGD和Projected Noise SGD等流行深度学习算法的隐私分析，以涵盖更广泛的度量差异隐私的概念。



## **11. Understanding the Robustness of 3D Object Detection with Bird's-Eye-View Representations in Autonomous Driving**

理解自动驾驶中鸟瞰表示的3D目标检测的稳健性 cs.CV

8 pages, CVPR2023

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.17297v1) [paper-pdf](http://arxiv.org/pdf/2303.17297v1)

**Authors**: Zijian Zhu, Yichi Zhang, Hai Chen, Yinpeng Dong, Shu Zhao, Wenbo Ding, Jiachen Zhong, Shibao Zheng

**Abstract**: 3D object detection is an essential perception task in autonomous driving to understand the environments. The Bird's-Eye-View (BEV) representations have significantly improved the performance of 3D detectors with camera inputs on popular benchmarks. However, there still lacks a systematic understanding of the robustness of these vision-dependent BEV models, which is closely related to the safety of autonomous driving systems. In this paper, we evaluate the natural and adversarial robustness of various representative models under extensive settings, to fully understand their behaviors influenced by explicit BEV features compared with those without BEV. In addition to the classic settings, we propose a 3D consistent patch attack by applying adversarial patches in the 3D space to guarantee the spatiotemporal consistency, which is more realistic for the scenario of autonomous driving. With substantial experiments, we draw several findings: 1) BEV models tend to be more stable than previous methods under different natural conditions and common corruptions due to the expressive spatial representations; 2) BEV models are more vulnerable to adversarial noises, mainly caused by the redundant BEV features; 3) Camera-LiDAR fusion models have superior performance under different settings with multi-modal inputs, but BEV fusion model is still vulnerable to adversarial noises of both point cloud and image. These findings alert the safety issue in the applications of BEV detectors and could facilitate the development of more robust models.

摘要: 三维物体检测是自动驾驶中一项重要的感知任务，目的是了解周围环境。鸟眼视图(BEV)表示显著提高了具有相机输入的3D探测器的性能，符合流行的基准。然而，这些视觉相关的Bev模型的鲁棒性与自动驾驶系统的安全性密切相关，目前还缺乏系统的了解。在本文中，我们评估了各种典型模型在广泛环境下的自然健壮性和对抗健壮性，以充分了解受显式BEV特征影响的行为，并与没有BEV特征的模型进行比较。在经典场景的基础上，通过在3D空间中应用对抗性补丁来保证时空一致性，提出了一种3D一致性补丁攻击方法，更适合于自主驾驶场景。通过大量的实验，我们发现：1)BEV模型在不同的自然条件和常见的污染情况下，由于具有丰富的空间表达能力，往往比以往的方法更稳定；2)BEV模型更容易受到对抗性噪声的影响，这主要是由于冗余的BEV特征造成的；3)Camera-LiDAR融合模型在不同的多模式输入情况下具有更好的性能，但BEV融合模型仍然容易受到点云和图像的对抗性噪声的影响。这些发现警告了BEV探测器应用中的安全问题，并可能有助于开发更稳健的模型。



## **12. Adversarial Attack and Defense for Dehazing Networks**

解扰网络的对抗性攻击与防御 cs.CV

**SubmitDate**: 2023-03-30    [abs](http://arxiv.org/abs/2303.17255v1) [paper-pdf](http://arxiv.org/pdf/2303.17255v1)

**Authors**: Jie Gui, Xiaofeng Cong, Chengwei Peng, Yuan Yan Tang, James Tin-Yau Kwok

**Abstract**: The research on single image dehazing task has been widely explored. However, as far as we know, no comprehensive study has been conducted on the robustness of the well-trained dehazing models. Therefore, there is no evidence that the dehazing networks can resist malicious attacks. In this paper, we focus on designing a group of attack methods based on first order gradient to verify the robustness of the existing dehazing algorithms. By analyzing the general goal of image dehazing task, five attack methods are proposed, which are prediction, noise, mask, ground-truth and input attack. The corresponding experiments are conducted on six datasets with different scales. Further, the defense strategy based on adversarial training is adopted for reducing the negative effects caused by malicious attacks. In summary, this paper defines a new challenging problem for image dehazing area, which can be called as adversarial attack on dehazing networks (AADN). Code is available at https://github.com/guijiejie/AADN.

摘要: 单幅图像去霾任务的研究已经得到了广泛的探索。然而，据我们所知，目前还没有对训练有素的去霾模型的稳健性进行全面的研究。因此，没有证据表明除霾网络能够抵御恶意攻击。在本文中，我们重点设计了一组基于一阶梯度的攻击方法，以验证现有去霾算法的健壮性。通过分析图像去噪任务的总体目标，提出了预测攻击、噪声攻击、掩模攻击、地面真实攻击和输入攻击等五种攻击方法。在6个不同尺度的数据集上进行了相应的实验。此外，还采用了基于对抗性训练的防御策略，以减少恶意攻击带来的负面影响。综上所述，本文定义了图像去污领域的一个新的挑战性问题--对抗性网络攻击。代码可在https://github.com/guijiejie/AADN.上找到



## **13. A Tensor-based Convolutional Neural Network for Small Dataset Classification**

一种基于张量的卷积神经网络用于小数据集分类 cs.CV

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.17061v1) [paper-pdf](http://arxiv.org/pdf/2303.17061v1)

**Authors**: Zhenhua Chen, David Crandall

**Abstract**: Inspired by the ConvNets with structured hidden representations, we propose a Tensor-based Neural Network, TCNN. Different from ConvNets, TCNNs are composed of structured neurons rather than scalar neurons, and the basic operation is neuron tensor transformation. Unlike other structured ConvNets, where the part-whole relationships are modeled explicitly, the relationships are learned implicitly in TCNNs. Also, the structured neurons in TCNNs are high-rank tensors rather than vectors or matrices. We compare TCNNs with current popular ConvNets, including ResNets, MobileNets, EfficientNets, RegNets, etc., on CIFAR10, CIFAR100, and Tiny ImageNet. The experiment shows that TCNNs have higher efficiency in terms of parameters. TCNNs also show higher robustness against white-box adversarial attacks on MNIST compared to ConvNets.

摘要: 受结构化隐含表示的ConvNets的启发，我们提出了一种基于张量的神经网络TCNN。与ConvNets不同，TCNN是由结构化神经元而不是标量神经元组成的，其基本运算是神经元张量变换。与其他结构化ConvNet不同的是，部分-整体关系是显式建模的，而TCNN中的关系是隐式学习的。此外，TCNN中的结构化神经元是高阶张量，而不是向量或矩阵。在CIFAR10、CIFAR100和Tiny ImageNet上，我们将TCNN与目前流行的ConvNet进行了比较，包括ResNet、MobileNets、EfficientNets、RegNet等。实验表明，TCNN具有较高的参数效率。与ConvNet相比，TCNN对MNIST上的白盒攻击表现出更高的健壮性。



## **14. Beyond Empirical Risk Minimization: Local Structure Preserving Regularization for Improving Adversarial Robustness**

超越经验风险最小化：提高对抗性稳健性的局部结构保持正则化 cs.LG

13 pages, 4 figures

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.16861v1) [paper-pdf](http://arxiv.org/pdf/2303.16861v1)

**Authors**: Wei Wei, Jiahuan Zhou, Ying Wu

**Abstract**: It is broadly known that deep neural networks are susceptible to being fooled by adversarial examples with perturbations imperceptible by humans. Various defenses have been proposed to improve adversarial robustness, among which adversarial training methods are most effective. However, most of these methods treat the training samples independently and demand a tremendous amount of samples to train a robust network, while ignoring the latent structural information among these samples. In this work, we propose a novel Local Structure Preserving (LSP) regularization, which aims to preserve the local structure of the input space in the learned embedding space. In this manner, the attacking effect of adversarial samples lying in the vicinity of clean samples can be alleviated. We show strong empirical evidence that with or without adversarial training, our method consistently improves the performance of adversarial robustness on several image classification datasets compared to the baselines and some state-of-the-art approaches, thus providing promising direction for future research.

摘要: 众所周知，深度神经网络很容易被人类察觉不到的扰动的对抗性例子所愚弄。为了提高对手的稳健性，人们提出了各种防御措施，其中最有效的是对抗训练方法。然而，这些方法大多独立处理训练样本，需要大量的样本来训练一个稳健的网络，而忽略了这些样本之间潜在的结构信息。在这项工作中，我们提出了一种新的局部结构保持(LSP)正则化，目的是在学习的嵌入空间中保持输入空间的局部结构。以此方式，可以减轻位于清洁样本附近的对抗性样本的攻击效果。我们的实验结果表明，无论有无对抗性训练，我们的方法与基线和一些最新的方法相比，在几个图像分类数据集上的对抗性稳健性都得到了持续的提高，从而为未来的研究提供了有希望的方向。



## **15. Improving the Transferability of Adversarial Attacks on Face Recognition with Beneficial Perturbation Feature Augmentation**

利用有益扰动特征增强提高人脸识别中敌意攻击的可转移性 cs.CV

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2210.16117v3) [paper-pdf](http://arxiv.org/pdf/2210.16117v3)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Ping Li

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. The existence of adversarial face examples poses a great threat to the security of society. In order to build a more sustainable digital nation, in this paper, we improve the transferability of adversarial face examples to expose more blind spots of existing FR models. Though generating hard samples has shown its effectiveness in improving the generalization of models in training tasks, the effectiveness of utilizing this idea to improve the transferability of adversarial face examples remains unexplored. To this end, based on the property of hard samples and the symmetry between training tasks and adversarial attack tasks, we propose the concept of hard models, which have similar effects as hard samples for adversarial attack tasks. Utilizing the concept of hard models, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of adversarial examples to surrogate FR models by constantly generating new hard models to craft the adversarial examples. Specifically, in the backpropagation, BPFA records the gradients on pre-selected feature maps and uses the gradient on the input image to craft the adversarial example. In the next forward propagation, BPFA leverages the recorded gradients to add beneficial perturbations on their corresponding feature maps to increase the loss. Extensive experiments demonstrate that BPFA can significantly boost the transferability of adversarial attacks on FR.

摘要: 人脸识别(FR)模型很容易被敌意的例子所愚弄，这些例子是通过在良性的人脸图像上添加难以察觉的扰动来构建的。敌对面孔的存在对社会安全构成了极大的威胁。为了建设一个更可持续的数字国家，本文通过提高对抗性人脸样本的可转移性来暴露现有FR模型的更多盲点。尽管硬样本的生成在训练任务中提高了模型的泛化能力，但利用硬样本来提高对抗性人脸样本的可转移性的有效性仍未被探讨。为此，基于硬样本的性质和训练任务与对抗性攻击任务之间的对称性，我们提出了硬模型的概念，对于对抗性攻击任务，硬模型具有类似硬样本的效果。利用硬模型的概念，我们提出了一种新的攻击方法，称为有益扰动特征增强攻击(BPFA)，该方法通过不断生成新的硬模型来构造对抗性实例，从而减少了对抗性实例对替代FR模型的过度拟合。具体地说，在反向传播中，BPFA记录预先选择的特征地图上的梯度，并使用输入图像上的梯度来制作对抗性例子。在下一次前向传播中，BPFA利用记录的梯度在其相应的特征图上添加有益的扰动以增加损失。大量实验表明，BPFA能够显著提高对抗性攻击对FR的可转移性。



## **16. Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness**

不平衡的梯度：高估对手稳健性的微妙原因 cs.CV

To appear in Machine Learning

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2006.13726v4) [paper-pdf](http://arxiv.org/pdf/2006.13726v4)

**Authors**: Xingjun Ma, Linxi Jiang, Hanxun Huang, Zejia Weng, James Bailey, Yu-Gang Jiang

**Abstract**: Evaluating the robustness of a defense model is a challenging task in adversarial robustness research. Obfuscated gradients have previously been found to exist in many defense methods and cause a false signal of robustness. In this paper, we identify a more subtle situation called Imbalanced Gradients that can also cause overestimated adversarial robustness. The phenomenon of imbalanced gradients occurs when the gradient of one term of the margin loss dominates and pushes the attack towards to a suboptimal direction. To exploit imbalanced gradients, we formulate a Margin Decomposition (MD) attack that decomposes a margin loss into individual terms and then explores the attackability of these terms separately via a two-stage process. We also propose a multi-targeted and ensemble version of our MD attack. By investigating 24 defense models proposed since 2018, we find that 11 models are susceptible to a certain degree of imbalanced gradients and our MD attack can decrease their robustness evaluated by the best standalone baseline attack by more than 1%. We also provide an in-depth investigation on the likely causes of imbalanced gradients and effective countermeasures. Our code is available at https://github.com/HanxunH/MDAttack.

摘要: 评估防御模型的健壮性是对抗健壮性研究中一项具有挑战性的任务。模糊梯度先前已被发现存在于许多防御方法中，并导致错误的稳健性信号。在这篇文章中，我们识别了一种更微妙的情况，称为不平衡梯度，它也可能导致高估的对手稳健性。当保证金损失的一项的梯度占主导地位，并将攻击推向次优方向时，就会出现不平衡的梯度现象。为了利用不平衡梯度，我们提出了一个边际分解(MD)攻击，该攻击将边际损失分解为单个项，然后通过两个阶段的过程分别研究这些项的可攻击性。我们还提出了我们的MD攻击的多目标和整体版本。通过调查2018年以来提出的24种防御模型，我们发现11种模型对一定程度的不平衡梯度敏感，我们的MD攻击可以使其以最佳独立基线攻击评估的健壮性降低1%以上。我们还对梯度失衡的可能原因进行了深入的调查，并提出了有效的对策。我们的代码可以在https://github.com/HanxunH/MDAttack.上找到



## **17. Targeted Adversarial Attacks on Wind Power Forecasts**

对风电预测的有针对性的对抗性攻击 cs.LG

20 pages, including appendix, 12 figures

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.16633v1) [paper-pdf](http://arxiv.org/pdf/2303.16633v1)

**Authors**: René Heinrich, Christoph Scholz, Stephan Vogt, Malte Lehna

**Abstract**: In recent years, researchers proposed a variety of deep learning models for wind power forecasting. These models predict the wind power generation of wind farms or entire regions more accurately than traditional machine learning algorithms or physical models. However, latest research has shown that deep learning models can often be manipulated by adversarial attacks. Since wind power forecasts are essential for the stability of modern power systems, it is important to protect them from this threat. In this work, we investigate the vulnerability of two different forecasting models to targeted, semitargeted, and untargeted adversarial attacks. We consider a Long Short-Term Memory (LSTM) network for predicting the power generation of a wind farm and a Convolutional Neural Network (CNN) for forecasting the wind power generation throughout Germany. Moreover, we propose the Total Adversarial Robustness Score (TARS), an evaluation metric for quantifying the robustness of regression models to targeted and semi-targeted adversarial attacks. It assesses the impact of attacks on the model's performance, as well as the extent to which the attacker's goal was achieved, by assigning a score between 0 (very vulnerable) and 1 (very robust). In our experiments, the LSTM forecasting model was fairly robust and achieved a TARS value of over 0.81 for all adversarial attacks investigated. The CNN forecasting model only achieved TARS values below 0.06 when trained ordinarily, and was thus very vulnerable. Yet, its robustness could be significantly improved by adversarial training, which always resulted in a TARS above 0.46.

摘要: 近年来，研究人员提出了各种用于风电功率预测的深度学习模型。这些模型比传统的机器学习算法或物理模型更准确地预测风电场或整个地区的风力发电量。然而，最新的研究表明，深度学习模型经常会被对抗性攻击所操纵。由于风力发电预测对现代电力系统的稳定性至关重要，因此保护它们免受这种威胁是很重要的。在这项工作中，我们调查了两种不同的预测模型对目标攻击、半目标攻击和非目标攻击的脆弱性。我们考虑了用于预测风电场发电量的长短期记忆(LSTM)网络和用于预测整个德国风电场发电量的卷积神经网络(CNN)。此外，我们还提出了总对抗稳健性分数(TARS)，这是一个量化回归模型对定向和半定向对抗攻击的稳健性的评估指标。它通过在0(非常脆弱)和1(非常健壮)之间分配分数来评估攻击对模型性能的影响，以及攻击者目标的实现程度。在我们的实验中，LSTM预测模型是相当健壮的，对于所有被调查的对抗性攻击，TARS值都超过了0.81。CNN预测模型在正常训练时只能达到低于0.06的TARS值，因此非常容易受到攻击。然而，它的稳健性可以通过对抗性训练显著提高，这总是导致TARS高于0.46。



## **18. Diffusion Denoised Smoothing for Certified and Adversarial Robust Out-Of-Distribution Detection**

基于扩散去噪平滑的认证和对抗稳健失配检测 cs.LG

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.14961v2) [paper-pdf](http://arxiv.org/pdf/2303.14961v2)

**Authors**: Nicola Franco, Daniel Korth, Jeanette Miriam Lorenz, Karsten Roscher, Stephan Guennemann

**Abstract**: As the use of machine learning continues to expand, the importance of ensuring its safety cannot be overstated. A key concern in this regard is the ability to identify whether a given sample is from the training distribution, or is an "Out-Of-Distribution" (OOD) sample. In addition, adversaries can manipulate OOD samples in ways that lead a classifier to make a confident prediction. In this study, we present a novel approach for certifying the robustness of OOD detection within a $\ell_2$-norm around the input, regardless of network architecture and without the need for specific components or additional training. Further, we improve current techniques for detecting adversarial attacks on OOD samples, while providing high levels of certified and adversarial robustness on in-distribution samples. The average of all OOD detection metrics on CIFAR10/100 shows an increase of $\sim 13 \% / 5\%$ relative to previous approaches.

摘要: 随着机器学习的使用不断扩大，确保其安全性的重要性怎么强调都不为过。这方面的一个关键问题是能否识别给定样本是来自训练分布，还是“超出分布”(OOD)样本。此外，攻击者还可以操纵OOD样本，从而使分类器做出可靠的预测。在这项研究中，我们提出了一种新的方法来证明OOD检测的稳健性在输入周围的$\ell_2$-范数内，而与网络体系结构无关，并且不需要特定的组件或额外的训练。此外，我们改进了当前检测OOD样本上的对抗性攻击的技术，同时在分发内样本上提供了高水平的认证和对抗性健壮性。与以前的方法相比，CIFAR10/100上所有OOD检测指标的平均值增加了$\sim 13/5$。



## **19. What Does the Gradient Tell When Attacking the Graph Structure**

当攻击图形结构时，渐变说明了什么 cs.LG

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2208.12815v2) [paper-pdf](http://arxiv.org/pdf/2208.12815v2)

**Authors**: Zihan Liu, Ge Wang, Yun Luo, Stan Z. Li

**Abstract**: Recent research has revealed that Graph Neural Networks (GNNs) are susceptible to adversarial attacks targeting the graph structure. A malicious attacker can manipulate a limited number of edges, given the training labels, to impair the victim model's performance. Previous empirical studies indicate that gradient-based attackers tend to add edges rather than remove them. In this paper, we present a theoretical demonstration revealing that attackers tend to increase inter-class edges due to the message passing mechanism of GNNs, which explains some previous empirical observations. By connecting dissimilar nodes, attackers can more effectively corrupt node features, making such attacks more advantageous. However, we demonstrate that the inherent smoothness of GNN's message passing tends to blur node dissimilarity in the feature space, leading to the loss of crucial information during the forward process. To address this issue, we propose a novel surrogate model with multi-level propagation that preserves the node dissimilarity information. This model parallelizes the propagation of unaggregated raw features and multi-hop aggregated features, while introducing batch normalization to enhance the dissimilarity in node representations and counteract the smoothness resulting from topological aggregation. Our experiments show significant improvement with our approach.Furthermore, both theoretical and experimental evidence suggest that adding inter-class edges constitutes an easily observable attack pattern. We propose an innovative attack loss that balances attack effectiveness and imperceptibility, sacrificing some attack effectiveness to attain greater imperceptibility. We also provide experiments to validate the compromise performance achieved through this attack loss.

摘要: 最近的研究表明，图神经网络(GNN)容易受到针对图结构的对手攻击。恶意攻击者可以在给定训练标签的情况下操纵有限数量的边，以损害受害者模型的性能。先前的经验研究表明，基于梯度的攻击者倾向于添加边缘，而不是移除它们。在本文中，我们给出了一个理论证明，揭示了由于GNN的消息传递机制，攻击者倾向于增加类间边缘，这解释了以前的一些经验观察。通过连接不同的节点，攻击者可以更有效地破坏节点功能，使此类攻击更具优势。然而，我们证明了GNN消息传递固有的平稳性往往会模糊特征空间中节点的差异，导致转发过程中关键信息的丢失。针对这一问题，我们提出了一种保留节点相异信息的多级传播代理模型。该模型并行传播未聚集的原始特征和多跳聚集的特征，同时引入批量归一化来增强节点表示的差异性，抵消拓扑聚集带来的光滑性。实验表明，该方法具有明显的改进效果，而且理论和实验结果都表明，添加类间边缘构成了一种容易观察到的攻击模式。我们提出了一种创新的攻击损失，它平衡了攻击有效性和不可感知性，牺牲了一些攻击有效性来获得更大的不可感知性。我们还提供了实验来验证通过这种攻击损失所获得的折衷性能。



## **20. A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion**

针对稳定扩散的无查询对抗性攻击的初步研究 cs.CV

The 3rd Workshop of Adversarial Machine Learning on Computer Vision:  Art of Robustness

**SubmitDate**: 2023-03-29    [abs](http://arxiv.org/abs/2303.16378v1) [paper-pdf](http://arxiv.org/pdf/2303.16378v1)

**Authors**: Haomin Zhuang, Yihua Zhang, Sijia Liu

**Abstract**: Despite the record-breaking performance in Text-to-Image (T2I) generation by Stable Diffusion, less research attention is paid to its adversarial robustness. In this work, we study the problem of adversarial attack generation for Stable Diffusion and ask if an adversarial text prompt can be obtained even in the absence of end-to-end model queries. We call the resulting problem 'query-free attack generation'. To resolve this problem, we show that the vulnerability of T2I models is rooted in the lack of robustness of text encoders, e.g., the CLIP text encoder used for attacking Stable Diffusion. Based on such insight, we propose both untargeted and targeted query-free attacks, where the former is built on the most influential dimensions in the text embedding space, which we call steerable key dimensions. By leveraging the proposed attacks, we empirically show that only a five-character perturbation to the text prompt is able to cause the significant content shift of synthesized images using Stable Diffusion. Moreover, we show that the proposed target attack can precisely steer the diffusion model to scrub the targeted image content without causing much change in untargeted image content.

摘要: 尽管稳定扩散在文本到图像(T2I)的生成中取得了创纪录的性能，但对其对抗健壮性的研究较少。在这项工作中，我们研究了稳定扩散的对抗性攻击生成问题，并询问即使在没有端到端模型查询的情况下，是否也能获得对抗性文本提示。我们称由此产生的问题为“无查询攻击生成”。为了解决这个问题，我们证明了T2I模型的脆弱性源于文本编码器缺乏健壮性，例如用于攻击稳定扩散的CLIP文本编码器。基于这样的见解，我们提出了无目标查询攻击和无目标查询攻击，前者建立在文本嵌入空间中最有影响力的维度上，我们称之为可引导的关键维度。通过利用提出的攻击，我们的经验表明，只有对文本提示的五个字符的扰动才能导致使用稳定扩散的合成图像的显著内容偏移。此外，我们还证明了所提出的目标攻击能够准确地引导扩散模型对目标图像内容进行擦除，而不会对非目标图像内容造成太大改变。



## **21. A Survey on Malware Detection with Graph Representation Learning**

基于图表示学习的恶意软件检测综述 cs.CR

Preprint, submitted to ACM Computing Surveys on March 2023. For any  suggestions or improvements, please contact me directly by e-mail

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.16004v1) [paper-pdf](http://arxiv.org/pdf/2303.16004v1)

**Authors**: Tristan Bilot, Nour El Madhoun, Khaldoun Al Agha, Anis Zouaoui

**Abstract**: Malware detection has become a major concern due to the increasing number and complexity of malware. Traditional detection methods based on signatures and heuristics are used for malware detection, but unfortunately, they suffer from poor generalization to unknown attacks and can be easily circumvented using obfuscation techniques. In recent years, Machine Learning (ML) and notably Deep Learning (DL) achieved impressive results in malware detection by learning useful representations from data and have become a solution preferred over traditional methods. More recently, the application of such techniques on graph-structured data has achieved state-of-the-art performance in various domains and demonstrates promising results in learning more robust representations from malware. Yet, no literature review focusing on graph-based deep learning for malware detection exists. In this survey, we provide an in-depth literature review to summarize and unify existing works under the common approaches and architectures. We notably demonstrate that Graph Neural Networks (GNNs) reach competitive results in learning robust embeddings from malware represented as expressive graph structures, leading to an efficient detection by downstream classifiers. This paper also reviews adversarial attacks that are utilized to fool graph-based detection methods. Challenges and future research directions are discussed at the end of the paper.

摘要: 由于恶意软件的数量和复杂性不断增加，恶意软件检测已成为一个主要问题。传统的基于签名和启发式的检测方法被用于恶意软件检测，但遗憾的是，它们对未知攻击的泛化能力较差，可以通过混淆技术轻松地绕过。近年来，机器学习(ML)和深度学习(DL)通过从数据中学习有用的表示，在恶意软件检测方面取得了令人印象深刻的结果，并成为一种比传统方法更受欢迎的解决方案。最近，这种技术在图结构数据上的应用已经在各个领域取得了最先进的性能，并在从恶意软件中学习更健壮的表示方面展示了良好的结果。然而，目前还没有关于基于图的深度学习用于恶意软件检测的文献综述。在这次调查中，我们提供了深入的文献回顾，以总结和统一在共同的方法和架构下的现有工作。值得注意的是，图神经网络(GNN)在学习表示为可表达图结构的恶意软件的健壮嵌入方面取得了具有竞争力的结果，从而导致了下游分类器的有效检测。本文还回顾了用于欺骗基于图的检测方法的对抗性攻击。在文章的最后，讨论了挑战和未来的研究方向。



## **22. TransAudio: Towards the Transferable Adversarial Audio Attack via Learning Contextualized Perturbations**

TransAudio：通过学习上下文扰动实现可转移的对抗性音频攻击 cs.SD

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15940v1) [paper-pdf](http://arxiv.org/pdf/2303.15940v1)

**Authors**: Qi Gege, Yuefeng Chen, Xiaofeng Mao, Yao Zhu, Binyuan Hui, Xiaodan Li, Rong Zhang, Hui Xue

**Abstract**: In a transfer-based attack against Automatic Speech Recognition (ASR) systems, attacks are unable to access the architecture and parameters of the target model. Existing attack methods are mostly investigated in voice assistant scenarios with restricted voice commands, prohibiting their applicability to more general ASR related applications. To tackle this challenge, we propose a novel contextualized attack with deletion, insertion, and substitution adversarial behaviors, namely TransAudio, which achieves arbitrary word-level attacks based on the proposed two-stage framework. To strengthen the attack transferability, we further introduce an audio score-matching optimization strategy to regularize the training process, which mitigates adversarial example over-fitting to the surrogate model. Extensive experiments and analysis demonstrate the effectiveness of TransAudio against open-source ASR models and commercial APIs.

摘要: 在针对自动语音识别(ASR)系统的基于传输的攻击中，攻击无法访问目标模型的体系结构和参数。现有的攻击方法大多是在语音助手受限制的语音命令场景中研究的，这使得它们不适用于更一般的ASR相关应用。为了应对这一挑战，我们提出了一种新的具有删除、插入和替换敌意行为的上下文攻击，即TransAudio，它基于所提出的两阶段框架实现了任意词级攻击。为了增强攻击的可转移性，我们进一步引入了音频得分匹配的优化策略来规范训练过程，从而缓解了对手例子对代理模型的过度拟合。大量的实验和分析证明了TransAudio相对于开源ASR模型和商业API的有效性。



## **23. Denoising Autoencoder-based Defensive Distillation as an Adversarial Robustness Algorithm**

基于自动编码去噪的防御蒸馏作为一种对抗健壮性算法 cs.LG

This paper have 4 pages, 3 figures and it is accepted at the Ada User  journal

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15901v1) [paper-pdf](http://arxiv.org/pdf/2303.15901v1)

**Authors**: Bakary Badjie, José Cecílio, António Casimiro

**Abstract**: Adversarial attacks significantly threaten the robustness of deep neural networks (DNNs). Despite the multiple defensive methods employed, they are nevertheless vulnerable to poison attacks, where attackers meddle with the initial training data. In order to defend DNNs against such adversarial attacks, this work proposes a novel method that combines the defensive distillation mechanism with a denoising autoencoder (DAE). This technique tries to lower the sensitivity of the distilled model to poison attacks by spotting and reconstructing poisonous adversarial inputs in the training data. We added carefully created adversarial samples to the initial training data to assess the proposed method's performance. Our experimental findings demonstrate that our method successfully identified and reconstructed the poisonous inputs while also considering enhancing the DNN's resilience. The proposed approach provides a potent and robust defense mechanism for DNNs in various applications where data poisoning attacks are a concern. Thus, the defensive distillation technique's limitation posed by poisonous adversarial attacks is overcome.

摘要: 敌意攻击严重威胁了深度神经网络(DNN)的健壮性。尽管采用了多种防御方法，但它们仍然容易受到毒药攻击，攻击者会干预初始训练数据。为了防止DNN受到这种恶意攻击，提出了一种新的方法，该方法将防御蒸馏机制与去噪自动编码器(DAE)相结合。这种技术试图通过在训练数据中发现和重建有毒的对抗性输入来降低提取模型对毒物攻击的敏感度。我们将精心创建的对抗性样本添加到初始训练数据中，以评估所提出的方法的性能。我们的实验结果表明，我们的方法成功地识别和重建了有毒输入，同时还考虑了增强DNN的弹性。所提出的方法为DNN在数据中毒攻击引起关注的各种应用中提供了一种有效和健壮的防御机制。从而克服了防御性蒸馏技术因恶意对抗性攻击而带来的局限性。



## **24. Machine-learned Adversarial Attacks against Fault Prediction Systems in Smart Electrical Grids**

智能电网中对故障预测系统的机器学习对抗性攻击 cs.CR

Accepted in AdvML@KDD'22

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.18136v1) [paper-pdf](http://arxiv.org/pdf/2303.18136v1)

**Authors**: Carmelo Ardito, Yashar Deldjoo, Tommaso Di Noia, Eugenio Di Sciascio, Fatemeh Nazary, Giovanni Servedio

**Abstract**: In smart electrical grids, fault detection tasks may have a high impact on society due to their economic and critical implications. In the recent years, numerous smart grid applications, such as defect detection and load forecasting, have embraced data-driven methodologies. The purpose of this study is to investigate the challenges associated with the security of machine learning (ML) applications in the smart grid scenario. Indeed, the robustness and security of these data-driven algorithms have not been extensively studied in relation to all power grid applications. We demonstrate first that the deep neural network method used in the smart grid is susceptible to adversarial perturbation. Then, we highlight how studies on fault localization and type classification illustrate the weaknesses of present ML algorithms in smart grids to various adversarial attacks

摘要: 在智能电网中，故障检测任务可能会对社会产生很大的影响，因为它们具有经济和关键意义。近年来，许多智能电网应用，如缺陷检测和负荷预测，都采用了数据驱动的方法。这项研究的目的是调查与智能电网场景中的机器学习(ML)应用程序的安全相关的挑战。事实上，这些数据驱动算法的健壮性和安全性并没有在所有的电网应用中得到广泛的研究。我们首先证明了智能电网中使用的深度神经网络方法容易受到对抗性扰动的影响。然后重点介绍了故障定位和类型分类的研究如何说明智能电网中现有最大似然算法在各种敌意攻击下的弱点



## **25. Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition**

面向物理人脸识别的有效对抗性纹理3D网格 cs.CV

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15818v1) [paper-pdf](http://arxiv.org/pdf/2303.15818v1)

**Authors**: Xiao Yang, Chang Liu, Longlong Xu, Yikai Wang, Yinpeng Dong, Ning Chen, Hang Su, Jun Zhu

**Abstract**: Face recognition is a prevailing authentication solution in numerous biometric applications. Physical adversarial attacks, as an important surrogate, can identify the weaknesses of face recognition systems and evaluate their robustness before deployed. However, most existing physical attacks are either detectable readily or ineffective against commercial recognition systems. The goal of this work is to develop a more reliable technique that can carry out an end-to-end evaluation of adversarial robustness for commercial systems. It requires that this technique can simultaneously deceive black-box recognition models and evade defensive mechanisms. To fulfill this, we design adversarial textured 3D meshes (AT3D) with an elaborate topology on a human face, which can be 3D-printed and pasted on the attacker's face to evade the defenses. However, the mesh-based optimization regime calculates gradients in high-dimensional mesh space, and can be trapped into local optima with unsatisfactory transferability. To deviate from the mesh-based space, we propose to perturb the low-dimensional coefficient space based on 3D Morphable Model, which significantly improves black-box transferability meanwhile enjoying faster search efficiency and better visual quality. Extensive experiments in digital and physical scenarios show that our method effectively explores the security vulnerabilities of multiple popular commercial services, including three recognition APIs, four anti-spoofing APIs, two prevailing mobile phones and two automated access control systems.

摘要: 人脸识别是众多生物识别应用中的主流身份验证解决方案。物理对抗攻击作为一种重要的替代手段，可以在部署前识别人脸识别系统的弱点并评估其健壮性。然而，大多数现有的物理攻击要么很容易被检测到，要么对商业识别系统无效。这项工作的目标是开发一种更可靠的技术，可以对商业系统的对手健壮性进行端到端的评估。这就要求该技术能够同时欺骗黑盒识别模型和规避防御机制。为了实现这一点，我们设计了对抗性纹理3D网格(AT3D)，在人脸上具有复杂的拓扑结构，可以3D打印并粘贴在攻击者的脸上以躲避防御。然而，基于网格的优化方法在高维网格空间中计算梯度，容易陷入局部最优，可移植性不理想。为了偏离基于网格的空间，我们提出了基于3D可变形模型的低维系数空间的扰动，在获得更快的搜索效率和更好的视觉质量的同时，显著提高了黑盒的可转移性。在数字和物理场景中的大量实验表明，我们的方法有效地探测了多个流行的商业服务的安全漏洞，包括三个识别API、四个反欺骗API、两个流行的手机和两个自动访问控制系统。



## **26. Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization**

基于令牌梯度正则化的视觉变换可转移敌意攻击 cs.CV

CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15754v1) [paper-pdf](http://arxiv.org/pdf/2303.15754v1)

**Authors**: Jianping Zhang, Yizhan Huang, Weibin Wu, Michael R. Lyu

**Abstract**: Vision transformers (ViTs) have been successfully deployed in a variety of computer vision tasks, but they are still vulnerable to adversarial samples. Transfer-based attacks use a local model to generate adversarial samples and directly transfer them to attack a target black-box model. The high efficiency of transfer-based attacks makes it a severe security threat to ViT-based applications. Therefore, it is vital to design effective transfer-based attacks to identify the deficiencies of ViTs beforehand in security-sensitive scenarios. Existing efforts generally focus on regularizing the input gradients to stabilize the updated direction of adversarial samples. However, the variance of the back-propagated gradients in intermediate blocks of ViTs may still be large, which may make the generated adversarial samples focus on some model-specific features and get stuck in poor local optima. To overcome the shortcomings of existing approaches, we propose the Token Gradient Regularization (TGR) method. According to the structural characteristics of ViTs, TGR reduces the variance of the back-propagated gradient in each internal block of ViTs in a token-wise manner and utilizes the regularized gradient to generate adversarial samples. Extensive experiments on attacking both ViTs and CNNs confirm the superiority of our approach. Notably, compared to the state-of-the-art transfer-based attacks, our TGR offers a performance improvement of 8.8% on average.

摘要: 视觉转换器(VITS)已经成功地应用于各种计算机视觉任务中，但它们仍然容易受到对手样本的攻击。基于转移的攻击使用局部模型来生成对抗性样本，并直接转移它们来攻击目标黑盒模型。基于传输的攻击的高效率使其对基于VIT的应用程序构成了严重的安全威胁。因此，设计有效的基于传输的攻击以在安全敏感的场景中预先识别VITS的缺陷是至关重要的。现有的努力一般侧重于使输入梯度正规化，以稳定对抗性样本的最新方向。然而，VITS中间块的反向传播梯度的方差可能仍然很大，这可能会使生成的对抗性样本集中在某些模型特定的特征上，陷入较差的局部最优。为了克服现有方法的不足，我们提出了令牌梯度正则化方法。根据VITS的结构特点，TGR以象征性的方式减小VITS各内部块反向传播梯度的方差，并利用正则化的梯度生成对抗性样本。在攻击VITS和CNN上的大量实验证实了该方法的优越性。值得注意的是，与最先进的基于传输的攻击相比，我们的TGR提供了8.8%的平均性能改进。



## **27. Improving the Transferability of Adversarial Samples by Path-Augmented Method**

利用路径扩展方法提高对抗性样本的可转移性 cs.CV

10 pages + appendix, CVPR 2023

**SubmitDate**: 2023-03-28    [abs](http://arxiv.org/abs/2303.15735v1) [paper-pdf](http://arxiv.org/pdf/2303.15735v1)

**Authors**: Jianping Zhang, Jen-tse Huang, Wenxuan Wang, Yichen Li, Weibin Wu, Xiaosen Wang, Yuxin Su, Michael R. Lyu

**Abstract**: Deep neural networks have achieved unprecedented success on diverse vision tasks. However, they are vulnerable to adversarial noise that is imperceptible to humans. This phenomenon negatively affects their deployment in real-world scenarios, especially security-related ones. To evaluate the robustness of a target model in practice, transfer-based attacks craft adversarial samples with a local model and have attracted increasing attention from researchers due to their high efficiency. The state-of-the-art transfer-based attacks are generally based on data augmentation, which typically augments multiple training images from a linear path when learning adversarial samples. However, such methods selected the image augmentation path heuristically and may augment images that are semantics-inconsistent with the target images, which harms the transferability of the generated adversarial samples. To overcome the pitfall, we propose the Path-Augmented Method (PAM). Specifically, PAM first constructs a candidate augmentation path pool. It then settles the employed augmentation paths during adversarial sample generation with greedy search. Furthermore, to avoid augmenting semantics-inconsistent images, we train a Semantics Predictor (SP) to constrain the length of the augmentation path. Extensive experiments confirm that PAM can achieve an improvement of over 4.8% on average compared with the state-of-the-art baselines in terms of the attack success rates.

摘要: 深度神经网络在不同的视觉任务上取得了前所未有的成功。然而，它们很容易受到人类察觉不到的对抗性噪音的影响。这种现象对它们在现实世界场景中的部署产生了负面影响，特别是与安全相关的场景。为了在实际应用中评估目标模型的稳健性，基于转移的攻击利用局部模型来构造对手样本，由于其高效性而受到越来越多的研究人员的关注。最先进的基于传输的攻击通常基于数据增强，当学习对抗性样本时，数据增强通常从线性路径增加多个训练图像。然而，这些方法对图像增强路径的选择是启发式的，可能会对与目标图像语义不一致的图像进行增强，从而损害了生成的对抗性样本的可转移性。为了克服这一缺陷，我们提出了路径扩展方法(PAM)。具体地，PAM首先构建候选扩展路径池。然后利用贪婪搜索解决对抗性样本生成过程中所采用的扩充路径。此外，为了避免增强语义不一致的图像，我们训练了一个语义预测器(SP)来约束增强路径的长度。广泛的实验证实，与最先进的基线相比，PAM在攻击成功率方面平均可以提高4.8%以上。



## **28. EMShepherd: Detecting Adversarial Samples via Side-channel Leakage**

EMShepherd：通过旁路泄漏检测敌方样本 cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15571v1) [paper-pdf](http://arxiv.org/pdf/2303.15571v1)

**Authors**: Ruyi Ding, Cheng Gongye, Siyue Wang, Aidong Ding, Yunsi Fei

**Abstract**: Deep Neural Networks (DNN) are vulnerable to adversarial perturbations-small changes crafted deliberately on the input to mislead the model for wrong predictions. Adversarial attacks have disastrous consequences for deep learning-empowered critical applications. Existing defense and detection techniques both require extensive knowledge of the model, testing inputs, and even execution details. They are not viable for general deep learning implementations where the model internal is unknown, a common 'black-box' scenario for model users. Inspired by the fact that electromagnetic (EM) emanations of a model inference are dependent on both operations and data and may contain footprints of different input classes, we propose a framework, EMShepherd, to capture EM traces of model execution, perform processing on traces and exploit them for adversarial detection. Only benign samples and their EM traces are used to train the adversarial detector: a set of EM classifiers and class-specific unsupervised anomaly detectors. When the victim model system is under attack by an adversarial example, the model execution will be different from executions for the known classes, and the EM trace will be different. We demonstrate that our air-gapped EMShepherd can effectively detect different adversarial attacks on a commonly used FPGA deep learning accelerator for both Fashion MNIST and CIFAR-10 datasets. It achieves a 100% detection rate on most types of adversarial samples, which is comparable to the state-of-the-art 'white-box' software-based detectors.

摘要: 深度神经网络(DNN)很容易受到对抗性扰动--故意在输入上精心设计的小变化，以误导模型进行错误预测。对抗性攻击会给深度学习支持的关键应用程序带来灾难性的后果。现有的防御和检测技术都需要对模型、测试输入甚至执行细节有广泛的了解。它们对于模型内部未知的一般深度学习实现是不可行的，对于模型用户来说，这是一个常见的“黑箱”场景。受模型推理的电磁辐射依赖于操作和数据并且可能包含不同输入类的足迹这一事实的启发，我们提出了一个框架EMShepherd，用于捕获模型执行的电磁跟踪，对跟踪进行处理，并利用它们进行对抗性检测。只有良性样本及其EM踪迹被用于训练对抗性检测器：一组EM分类器和特定类别的非监督异常检测器。当受害者模型系统受到敌意示例攻击时，模型执行将与已知类的执行不同，EM跟踪也将不同。我们证明了我们的空隙EMShepherd可以有效地检测到针对Fashion MNIST和CIFAR-10数据集的常用FPGA深度学习加速器上的不同对手攻击。它在大多数类型的对手样本上实现了100%的检测率，这可以与最先进的基于软件的白盒检测器相媲美。



## **29. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

掩码和恢复：使用掩码自动编码器在测试时进行盲后门保护 cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15564v1) [paper-pdf](http://arxiv.org/pdf/2303.15564v1)

**Authors**: Tao Sun, Lu Pang, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from the hard label predictions of a suspicious model. The heuristic trigger search in image space, however, is not scalable to complex triggers or high image resolution. We circumvent such barrier by leveraging generic image generation models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It uses the image structural similarity and label consistency between the test image and MAE restorations to detect possible triggers. The detection result is refined by considering the topology of triggers. We obtain a purified test image from restorations for making prediction. Our approach is blind to the model architectures, trigger patterns or image benignity. Extensive experiments on multiple datasets with different backdoor attacks validate its effectiveness and generalizability. Code is available at https://github.com/tsun/BDMAE.

摘要: 深度神经网络很容易受到后门攻击，在后门攻击中，对手通过使用特殊触发器覆盖图像来恶意操纵模型行为。现有的后门防御方法通常需要访问一些验证数据和模型参数，这在许多真实世界的应用中是不切实际的，例如当模型作为云服务提供时。在本文中，我们讨论了测试时的盲后门防御的实际任务，特别是对于黑盒模型。每个测试图像的真实标签都需要从可疑模型的硬标签预测中动态恢复。然而，图像空间中的启发式触发器搜索不能扩展到复杂的触发器或高图像分辨率。我们通过利用通用的图像生成模型来绕过这一障碍，并提出了一种基于掩蔽自动编码器的盲防框架(BDMAE)。它使用测试图像和MAE恢复图像之间的图像结构相似性和标签一致性来检测可能的触发因素。通过考虑触发器的拓扑结构，对检测结果进行了改进。我们从复原中获得一个净化的测试图像来进行预测。我们的方法对模型架构、触发模式或图像亲和性视而不见。在具有不同后门攻击的多个数据集上的大量实验验证了该算法的有效性和泛化能力。代码可在https://github.com/tsun/BDMAE.上找到



## **30. Intel TDX Demystified: A Top-Down Approach**

英特尔TDX揭秘：自上而下的方法 cs.CR

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15540v1) [paper-pdf](http://arxiv.org/pdf/2303.15540v1)

**Authors**: Pau-Chen Cheng, Wojciech Ozga, Enriquillo Valdez, Salman Ahmed, Zhongshu Gu, Hani Jamjoom, Hubertus Franke, James Bottomley

**Abstract**: Intel Trust Domain Extensions (TDX) is a new architectural extension in the 4th Generation Intel Xeon Scalable Processor that supports confidential computing. TDX allows the deployment of virtual machines in the Secure-Arbitration Mode (SEAM) with encrypted CPU state and memory, integrity protection, and remote attestation. TDX aims to enforce hardware-assisted isolation for virtual machines and minimize the attack surface exposed to host platforms, which are considered to be untrustworthy or adversarial in the confidential computing's new threat model. TDX can be leveraged by regulated industries or sensitive data holders to outsource their computations and data with end-to-end protection in public cloud infrastructure.   This paper aims to provide a comprehensive understanding of TDX to potential adopters, domain experts, and security researchers looking to leverage the technology for their own purposes. We adopt a top-down approach, starting with high-level security principles and moving to low-level technical details of TDX. Our analysis is based on publicly available documentation and source code, offering insights from security researchers outside of Intel.

摘要: 英特尔信任域扩展(TDX)是支持机密计算的第4代英特尔至强可扩展处理器中的新架构扩展。TDX允许在安全仲裁模式(SEAM)下部署具有加密的CPU状态和内存、完整性保护和远程证明的虚拟机。TDX旨在加强对虚拟机的硬件辅助隔离，并将暴露在主机平台上的攻击面降至最低，在机密计算的新威胁模型中，主机平台被认为是不可信任或对抗性的。受监管行业或敏感数据持有者可以利用TDX在公共云基础设施中提供端到端保护，以外包其计算和数据。本文旨在为潜在的采用者、领域专家和希望利用TDX技术实现其自身目的的安全研究人员提供对TDX的全面了解。我们采用自上而下的方法，从高级别的安全原则开始，转向TDX的低级别技术细节。我们的分析基于公开的文档和源代码，提供了英特尔以外的安全研究人员的见解。



## **31. Classifier Robustness Enhancement Via Test-Time Transformation**

利用测试时间变换增强分类器稳健性 cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15409v1) [paper-pdf](http://arxiv.org/pdf/2303.15409v1)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex Bronstein

**Abstract**: It has been recently discovered that adversarially trained classifiers exhibit an intriguing property, referred to as perceptually aligned gradients (PAG). PAG implies that the gradients of such classifiers possess a meaningful structure, aligned with human perception. Adversarial training is currently the best-known way to achieve classification robustness under adversarial attacks. The PAG property, however, has yet to be leveraged for further improving classifier robustness. In this work, we introduce Classifier Robustness Enhancement Via Test-Time Transformation (TETRA) -- a novel defense method that utilizes PAG, enhancing the performance of trained robust classifiers. Our method operates in two phases. First, it modifies the input image via a designated targeted adversarial attack into each of the dataset's classes. Then, it classifies the input image based on the distance to each of the modified instances, with the assumption that the shortest distance relates to the true class. We show that the proposed method achieves state-of-the-art results and validate our claim through extensive experiments on a variety of defense methods, classifier architectures, and datasets. We also empirically demonstrate that TETRA can boost the accuracy of any differentiable adversarial training classifier across a variety of attacks, including ones unseen at training. Specifically, applying TETRA leads to substantial improvement of up to $+23\%$, $+20\%$, and $+26\%$ on CIFAR10, CIFAR100, and ImageNet, respectively.

摘要: 最近发现，对抗性训练的分类器表现出一种有趣的特性，称为感知对齐梯度(PAG)。PAG暗示，这种量词的梯度具有一种有意义的结构，与人类的感知一致。对抗性训练是目前已知的在对抗性攻击下实现分类稳健性的最好方法。然而，PAG属性还有待于进一步提高分类器的健壮性。在这项工作中，我们引入了通过测试时间转换的分类器健壮性增强(TETRA)--一种利用PAG的新的防御方法，提高了训练的健壮分类器的性能。我们的方法分两个阶段进行。首先，它通过指定的目标对抗性攻击将输入图像修改为数据集的每个类。然后，在假设最短距离与真实类别相关的情况下，基于到每个修改实例的距离对输入图像进行分类。我们通过在各种防御方法、分类器架构和数据集上的大量实验，证明了所提出的方法取得了最先进的结果，并验证了我们的主张。我们还通过实验证明，TETRA可以提高任何可区分的对抗性训练分类器在各种攻击中的准确性，包括在训练中看不到的攻击。具体地说，应用TETRA后，CIFAR10、CIFAR100和ImageNet的性能分别提高了23美元、20美元和26美元。



## **32. Learning the Unlearnable: Adversarial Augmentations Suppress Unlearnable Example Attacks**

学习无法学习的：对抗性增强抑制无法学习的示例攻击 cs.LG

UEraser introduces adversarial augmentations to suppress unlearnable  example attacks and outperforms current defenses

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15127v1) [paper-pdf](http://arxiv.org/pdf/2303.15127v1)

**Authors**: Tianrui Qin, Xitong Gao, Juanjuan Zhao, Kejiang Ye, Cheng-Zhong Xu

**Abstract**: Unlearnable example attacks are data poisoning techniques that can be used to safeguard public data against unauthorized use for training deep learning models. These methods add stealthy perturbations to the original image, thereby making it difficult for deep learning models to learn from these training data effectively. Current research suggests that adversarial training can, to a certain degree, mitigate the impact of unlearnable example attacks, while common data augmentation methods are not effective against such poisons. Adversarial training, however, demands considerable computational resources and can result in non-trivial accuracy loss. In this paper, we introduce the UEraser method, which outperforms current defenses against different types of state-of-the-art unlearnable example attacks through a combination of effective data augmentation policies and loss-maximizing adversarial augmentations. In stark contrast to the current SOTA adversarial training methods, UEraser uses adversarial augmentations, which extends beyond the confines of $ \ell_p $ perturbation budget assumed by current unlearning attacks and defenses. It also helps to improve the model's generalization ability, thus protecting against accuracy loss. UEraser wipes out the unlearning effect with error-maximizing data augmentations, thus restoring trained model accuracies. Interestingly, UEraser-Lite, a fast variant without adversarial augmentations, is also highly effective in preserving clean accuracies. On challenging unlearnable CIFAR-10, CIFAR-100, SVHN, and ImageNet-subset datasets produced with various attacks, it achieves results that are comparable to those obtained during clean training. We also demonstrate its efficacy against possible adaptive attacks. Our code is open source and available to the deep learning community: https://github.com/lafeat/ueraser.

摘要: 无法学习的示例攻击是一种数据中毒技术，可用于保护公共数据免受未经授权的用于训练深度学习模型的使用。这些方法给原始图像增加了隐蔽的扰动，从而使得深度学习模型很难从这些训练数据中有效地学习。目前的研究表明，对抗性训练可以在一定程度上缓解不可学习的例子攻击的影响，而常用的数据增强方法对此类毒药并不有效。然而，对抗性训练需要相当大的计算资源，并且可能导致相当大的精度损失。在本文中，我们介绍了UEraser方法，它通过有效的数据增强策略和损失最大化的对手增强相结合，对不同类型的不可学习示例攻击的防御性能优于现有的防御方法。与目前的SOTA对抗性训练方法形成鲜明对比的是，UEraser使用对抗性增强，超出了当前遗忘攻击和防御假设的$\ell_p$扰动预算的范围。它还有助于提高模型的泛化能力，从而防止精度损失。UEraser通过误差最大化的数据增加消除了遗忘效应，从而恢复了训练的模型精度。有趣的是，UEraser-Lite，一个没有对抗性增强的快速变体，在保持干净的准确性方面也是非常有效的。在挑战各种攻击产生的难以学习的CIFAR-10、CIFAR-100、SVHN和ImageNet-Subset数据集上，它取得了与干净训练期间相当的结果。我们还展示了它对可能的自适应攻击的有效性。我们的代码是开源的，可供深度学习社区使用：https://github.com/lafeat/ueraser.



## **33. Among Us: Adversarially Robust Collaborative Perception by Consensus**

在我们中间：基于共识的相反的强健协作感知 cs.RO

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.09495v2) [paper-pdf](http://arxiv.org/pdf/2303.09495v2)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.

摘要: 多个机器人可以比个人更好地协作感知场景(例如，检测对象)，尽管在使用深度学习时很容易受到对抗性攻击。这可以通过对抗性防守来解决，但它的训练需要往往未知的攻击机制。不同的是，我们提出了ROBOSAC，一种新的基于采样的防御策略，可以推广到看不见的攻击者。我们的关键思想是，与个人感知相比，合作感知应该在结果中导致共识，而不是分歧。这导致了我们的假设和验证框架：对随机的队友子集进行协作和不协作的感知结果进行比较，直到达成共识。在这样的框架中，采样子集中更多的队友通常会带来更好的感知性能，但需要更长的采样时间来拒绝潜在的攻击者。因此，我们推导出需要多少次抽样试验才能确保没有攻击者的子集的期望大小，或者等价地，在给定的试验次数内可以成功抽样的子集的最大大小。我们在自主驾驶场景下的协同3D目标检测任务中验证了我们的方法。



## **34. Identifying Adversarially Attackable and Robust Samples**

识别恶意攻击和健壮样本 cs.LG

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.12896v2) [paper-pdf](http://arxiv.org/pdf/2301.12896v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attacks insert small, imperceptible perturbations to input samples that cause large, undesired changes to the output of deep learning models. Despite extensive research on generating adversarial attacks and building defense systems, there has been limited research on understanding adversarial attacks from an input-data perspective. This work introduces the notion of sample attackability, where we aim to identify samples that are most susceptible to adversarial attacks (attackable samples) and conversely also identify the least susceptible samples (robust samples). We propose a deep-learning-based method to detect the adversarially attackable and robust samples in an unseen dataset for an unseen target model. Experiments on standard image classification datasets enables us to assess the portability of the deep attackability detector across a range of architectures. We find that the deep attackability detector performs better than simple model uncertainty-based measures for identifying the attackable/robust samples. This suggests that uncertainty is an inadequate proxy for measuring sample distance to a decision boundary. In addition to better understanding adversarial attack theory, it is found that the ability to identify the adversarially attackable and robust samples has implications for improving the efficiency of sample-selection tasks, e.g. active learning in augmentation for adversarial training.

摘要: 对抗性攻击在输入样本中插入微小的、不可察觉的扰动，从而导致深度学习模型的输出发生巨大的、不希望看到的变化。尽管对生成对抗性攻击和建立防御系统进行了广泛的研究，但从输入数据的角度理解对抗性攻击的研究有限。这项工作引入了样本可攻击性的概念，其中我们的目标是识别最容易受到对手攻击的样本(可攻击样本)，反过来也识别最不敏感的样本(稳健样本)。我们提出了一种基于深度学习的方法来检测不可见目标模型中不可见数据集中的可攻击样本和稳健样本。在标准图像分类数据集上的实验使我们能够评估深度可攻击性检测器在一系列体系结构中的可移植性。我们发现，深度可攻击性检测器在识别可攻击/稳健样本方面比基于简单模型不确定性的度量方法表现得更好。这表明，不确定性不足以衡量样本到决策边界的距离。除了更好地理解敌意攻击理论外，研究发现，识别敌意可攻击和健壮样本的能力对于提高样本选择任务的效率也有意义，例如，在对抗性训练的增强中的主动学习。



## **35. Improving the Transferability of Adversarial Examples via Direction Tuning**

通过方向调整提高对抗性例句的可转移性 cs.CV

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.15109v1) [paper-pdf](http://arxiv.org/pdf/2303.15109v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: In the transfer-based adversarial attacks, adversarial examples are only generated by the surrogate models and achieve effective perturbation in the victim models. Although considerable efforts have been developed on improving the transferability of adversarial examples generated by transfer-based adversarial attacks, our investigation found that, the big deviation between the actual and steepest update directions of the current transfer-based adversarial attacks is caused by the large update step length, resulting in the generated adversarial examples can not converge well. However, directly reducing the update step length will lead to serious update oscillation so that the generated adversarial examples also can not achieve great transferability to the victim models. To address these issues, a novel transfer-based attack, namely direction tuning attack, is proposed to not only decrease the update deviation in the large step length, but also mitigate the update oscillation in the small sampling step length, thereby making the generated adversarial examples converge well to achieve great transferability on victim models. In addition, a network pruning method is proposed to smooth the decision boundary, thereby further decreasing the update oscillation and enhancing the transferability of the generated adversarial examples. The experiment results on ImageNet demonstrate that the average attack success rate (ASR) of the adversarial examples generated by our method can be improved from 87.9\% to 94.5\% on five victim models without defenses, and from 69.1\% to 76.2\% on eight advanced defense methods, in comparison with that of latest gradient-based attacks.

摘要: 在基于迁移的对抗性攻击中，对抗性实例仅由代理模型生成，并在受害者模型中实现有效的扰动。虽然在提高基于转移的对抗性攻击生成的对抗性样本的可转移性方面已经做了大量的工作，但我们的调查发现，当前基于转移的对抗性攻击的实际更新方向与最陡的更新方向之间存在较大的偏差，这是由于更新步长较大，导致生成的对抗性样本不能很好地收敛。但是，直接缩短更新步长会导致严重的更新振荡，使得生成的对抗性实例也不能很好地移植到受害者模型中。针对这些问题，提出了一种新的基于转移的攻击方法，即方向调整攻击，它不仅可以减小大步长时的更新偏差，而且可以缓解小采样步长时的更新振荡，从而使生成的敌意样本能够很好地收敛到受害者模型上，达到很好的可转移性。此外，还提出了一种网络剪枝方法来平滑决策边界，从而进一步减小更新振荡，增强生成的对抗性实例的可转移性。在ImageNet上的实验结果表明，与最新的基于梯度的攻击方法相比，该方法生成的攻击实例的平均攻击成功率(ASR)在5个无防御的受害者模型上可以从87.9提高到94.5，在8种高级防御方法上从69.1提高到76.2。



## **36. Improved Adversarial Training Through Adaptive Instance-wise Loss Smoothing**

通过自适应实例损失平滑改进对手训练 cs.CV

12 pages, work in submission

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14077v2) [paper-pdf](http://arxiv.org/pdf/2303.14077v2)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Deep neural networks can be easily fooled into making incorrect predictions through corruption of the input by adversarial perturbations: human-imperceptible artificial noise. So far adversarial training has been the most successful defense against such adversarial attacks. This work focuses on improving adversarial training to boost adversarial robustness. We first analyze, from an instance-wise perspective, how adversarial vulnerability evolves during adversarial training. We find that during training an overall reduction of adversarial loss is achieved by sacrificing a considerable proportion of training samples to be more vulnerable to adversarial attack, which results in an uneven distribution of adversarial vulnerability among data. Such "uneven vulnerability", is prevalent across several popular robust training methods and, more importantly, relates to overfitting in adversarial training. Motivated by this observation, we propose a new adversarial training method: Instance-adaptive Smoothness Enhanced Adversarial Training (ISEAT). It jointly smooths both input and weight loss landscapes in an adaptive, instance-specific, way to enhance robustness more for those samples with higher adversarial vulnerability. Extensive experiments demonstrate the superiority of our method over existing defense methods. Noticeably, our method, when combined with the latest data augmentation and semi-supervised learning techniques, achieves state-of-the-art robustness against $\ell_{\infty}$-norm constrained attacks on CIFAR10 of 59.32% for Wide ResNet34-10 without extra data, and 61.55% for Wide ResNet28-10 with extra data. Code is available at https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.

摘要: 深层神经网络很容易被欺骗，通过破坏对抗性扰动的输入做出错误的预测：人类无法察觉的人工噪声。到目前为止，对抗性训练一直是对这种对抗性攻击最成功的防御。这项工作的重点是改进对手训练，以提高对手的稳健性。我们首先从实例的角度分析在对抗性训练过程中对抗性脆弱性是如何演变的。我们发现，在训练过程中，通过牺牲相当大比例的训练样本来更容易受到对手攻击，从而总体上减少了对手的损失，这导致了对手脆弱性在数据中的不均匀分布。这种“脆弱性参差不齐”普遍存在于几种流行的健壮训练方法中，更重要的是与对抗性训练中的过度适应有关。基于这一观察结果，我们提出了一种新的对抗性训练方法：实例自适应平滑增强对抗性训练(ISEAT)。它以一种自适应的、特定于实例的方式联合平滑输入和减肥环境，以增强那些具有更高对手脆弱性的样本的健壮性。大量的实验证明了该方法相对于现有防御方法的优越性。值得注意的是，当我们的方法与最新的数据增强和半监督学习技术相结合时，对于针对CIFAR10的$-范数约束攻击，对于没有额外数据的宽ResNet34-10达到了59.32%，对于具有额外数据的宽ResNet28-10达到了61.55%。代码可在https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.上找到



## **37. CAT:Collaborative Adversarial Training**

CAT：协同对抗训练 cs.CV

Tech report

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2303.14922v1) [paper-pdf](http://arxiv.org/pdf/2303.14922v1)

**Authors**: Xingbin Liu, Huafeng Kuang, Xianming Lin, Yongjian Wu, Rongrong Ji

**Abstract**: Adversarial training can improve the robustness of neural networks. Previous methods focus on a single adversarial training strategy and do not consider the model property trained by different strategies. By revisiting the previous methods, we find different adversarial training methods have distinct robustness for sample instances. For example, a sample instance can be correctly classified by a model trained using standard adversarial training (AT) but not by a model trained using TRADES, and vice versa. Based on this observation, we propose a collaborative adversarial training framework to improve the robustness of neural networks. Specifically, we use different adversarial training methods to train robust models and let models interact with their knowledge during the training process. Collaborative Adversarial Training (CAT) can improve both robustness and accuracy. Extensive experiments on various networks and datasets validate the effectiveness of our method. CAT achieves state-of-the-art adversarial robustness without using any additional data on CIFAR-10 under the Auto-Attack benchmark. Code is available at https://github.com/liuxingbin/CAT.

摘要: 对抗性训练可以提高神经网络的健壮性。以往的方法侧重于单一的对抗性训练策略，没有考虑不同策略训练的模型性质。通过回顾以往的方法，我们发现不同的对抗性训练方法对样本实例具有不同的稳健性。例如，样本实例可以通过使用标准对手训练(AT)训练的模型来正确分类，但不可以通过使用交易训练的模型来正确分类，反之亦然。基于这一观察结果，我们提出了一个协同对抗训练框架来提高神经网络的健壮性。具体地说，我们使用不同的对抗性训练方法来训练健壮的模型，并在训练过程中让模型与他们的知识进行交互。协同对抗训练(CAT)可以同时提高鲁棒性和准确性。在不同网络和数据集上的大量实验验证了该方法的有效性。在自动攻击基准下，CAT无需使用CIFAR-10上的任何额外数据即可实现最先进的对手健壮性。代码可在https://github.com/liuxingbin/CAT.上找到



## **38. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

对抗性时空聚焦视频的高效稳健性评估 cs.CV

accepted by TPAMI2023

**SubmitDate**: 2023-03-27    [abs](http://arxiv.org/abs/2301.00896v2) [paper-pdf](http://arxiv.org/pdf/2301.00896v2)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.

摘要: 视频识别模型的对抗性健壮性评估由于其在安全关键任务中的广泛应用而引起了人们的关注。与图像相比，视频的维度要高得多，这在生成对抗性视频时带来了巨大的计算代价。这对于基于查询的黑盒攻击尤为严重，这种攻击通常使用威胁模型的梯度估计，高维将导致大量的查询。为了缓解这一问题，我们提出同时消除视频中的时间和空间冗余，在缩减的搜索空间上实现有效和高效的梯度估计，从而减少查询数量。为了实现这一思想，我们设计了一种新颖的对抗性时空聚焦(AstFocus)攻击，它从视频的帧间和帧内对同时聚焦的关键帧和关键区域进行攻击。AstFocus攻击基于协作多智能体强化学习(MAIL)框架。一个代理负责选择关键帧，另一个代理负责选择关键区域。这两个代理通过从黑盒威胁模型获得的共同奖励来联合训练，以执行合作预测。通过连续查询，缩小了由关键帧和关键区域组成的搜索空间，变得更加精确，整个查询次数比原始视频上的少。在四个主流视频识别模型和三个广泛使用的动作识别数据集上的大量实验表明，AstFocus攻击的性能优于SOTA方法，后者在愚弄率、查询次数、时间和扰动幅度方面都优于SOTA方法。



## **39. Don't be a Victim During a Pandemic! Analysing Security and Privacy Threats in Twitter During COVID-19**

不要在大流行期间成为受害者！新冠肺炎期间推特面临的安全和隐私威胁分析 cs.CR

Paper has been accepted for publication in IEEE Access. Currently  available on IEEE ACCESS early access (see DOI)

**SubmitDate**: 2023-03-26    [abs](http://arxiv.org/abs/2202.10543v2) [paper-pdf](http://arxiv.org/pdf/2202.10543v2)

**Authors**: Bibhas Sharma, Ishan Karunanayake, Rahat Masood, Muhammad Ikram

**Abstract**: There has been a huge spike in the usage of social media platforms during the COVID-19 lockdowns. These lockdown periods have resulted in a set of new cybercrimes, thereby allowing attackers to victimise social media users with a range of threats. This paper performs a large-scale study to investigate the impact of a pandemic and the lockdown periods on the security and privacy of social media users. We analyse 10.6 Million COVID-related tweets from 533 days of data crawling and investigate users' security and privacy behaviour in three different periods (i.e., before, during, and after the lockdown). Our study shows that users unintentionally share more personal identifiable information when writing about the pandemic situation (e.g., sharing nearby coronavirus testing locations) in their tweets. The privacy risk reaches 100% if a user posts three or more sensitive tweets about the pandemic. We investigate the number of suspicious domains shared on social media during different phases of the pandemic. Our analysis reveals an increase in the number of suspicious domains during the lockdown compared to other lockdown phases. We observe that IT, Search Engines, and Businesses are the top three categories that contain suspicious domains. Our analysis reveals that adversaries' strategies to instigate malicious activities change with the country's pandemic situation.

摘要: 在新冠肺炎被封锁期间，社交媒体平台的使用量大幅上升。这些封锁期导致了一系列新的网络犯罪，从而使攻击者能够通过一系列威胁来攻击社交媒体用户。本文进行了一项大规模的研究，以调查大流行和封锁期对社交媒体用户安全和隐私的影响。我们从533天的数据爬行中分析了1060万条与CoVID相关的推文，并调查了用户在三个不同时期(即封锁前、封锁期间和封锁后)的安全和隐私行为。我们的研究表明，当用户在他们的推特上写关于大流行情况的信息(例如，分享附近的冠状病毒检测地点)时，无意中分享了更多的个人可识别信息。如果用户发布三条或三条以上有关疫情的敏感推文，隐私风险将达到100%。我们调查了在疫情不同阶段在社交媒体上分享的可疑域名的数量。我们的分析显示，与其他锁定阶段相比，锁定期间可疑域名的数量有所增加。我们观察到，IT、搜索引擎和企业是包含可疑域名的前三大类别。我们的分析显示，对手煽动恶意活动的战略会随着该国疫情的变化而变化。



## **40. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

分布稳健多类分类及其在深度图像分类器中的应用 stat.ML

9 pages; Previously this version appeared as arXiv:2210.08198 which  was submitted as a new work by accident

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2109.12772v2) [paper-pdf](http://arxiv.org/pdf/2109.12772v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.

摘要: 针对多分类Logistic回归(MLR)模型，提出了一种分布式稳健优化(DRO)方法，该方法能够容忍数据中的离群点污染。DRO框架使用概率模糊集，该模糊集被定义为在Wasserstein度量意义上接近训练集的经验分布的分布球。我们将DRO公式松弛为一个正则化学习问题，其正则化子是系数矩阵的范数。我们为我们的模型的解建立了样本外性能保证，提供了关于正则化在控制预测误差中的作用的见解。我们将该方法应用于基于深度视觉转换器(VIT)的图像分类器对随机攻击和敌意攻击的稳健性。具体地说，使用MNIST和CIFAR-10数据集，我们证明了通过采用一种新的随机训练方法，与基线方法相比，测试错误率降低了83.5%，损失降低了91.3%。



## **41. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

分布稳健多类分类及其在深度图像分类器中的应用 cs.CV

This work was intended as a replacement of arXiv:2109.12772 and any  subsequent updates will appear there

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2210.08198v2) [paper-pdf](http://arxiv.org/pdf/2210.08198v2)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.

摘要: 针对多分类Logistic回归(MLR)模型，提出了一种分布式稳健优化(DRO)方法，该方法能够容忍数据中的离群点污染。DRO框架使用概率模糊集，该模糊集被定义为在Wasserstein度量意义上接近训练集的经验分布的分布球。我们将DRO公式松弛为一个正则化学习问题，其正则化子是系数矩阵的范数。我们为我们的模型的解建立了样本外性能保证，提供了关于正则化在控制预测误差中的作用的见解。我们将该方法应用于基于深度视觉转换器(VIT)的图像分类器对随机攻击和敌意攻击的稳健性。具体地说，使用MNIST和CIFAR-10数据集，我们证明了通过采用一种新的随机训练方法，与基线方法相比，测试错误率降低了83.5%，损失降低了91.3%。



## **42. AdvCheck: Characterizing Adversarial Examples via Local Gradient Checking**

AdvCheck：通过局部梯度检查来刻画敌意例子 cs.CR

26 pages

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.18131v1) [paper-pdf](http://arxiv.org/pdf/2303.18131v1)

**Authors**: Ruoxi Chen, Haibo Jin, Jinyin Chen, Haibin Zheng

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples, which may lead to catastrophe in security-critical domains. Numerous detection methods are proposed to characterize the feature uniqueness of adversarial examples, or to distinguish DNN's behavior activated by the adversarial examples. Detections based on features cannot handle adversarial examples with large perturbations. Besides, they require a large amount of specific adversarial examples. Another mainstream, model-based detections, which characterize input properties by model behaviors, suffer from heavy computation cost. To address the issues, we introduce the concept of local gradient, and reveal that adversarial examples have a quite larger bound of local gradient than the benign ones. Inspired by the observation, we leverage local gradient for detecting adversarial examples, and propose a general framework AdvCheck. Specifically, by calculating the local gradient from a few benign examples and noise-added misclassified examples to train a detector, adversarial examples and even misclassified natural inputs can be precisely distinguished from benign ones. Through extensive experiments, we have validated the AdvCheck's superior performance to the state-of-the-art (SOTA) baselines, with detection rate ($\sim \times 1.2$) on general adversarial attacks and ($\sim \times 1.4$) on misclassified natural inputs on average, with average 1/500 time cost. We also provide interpretable results for successful detection.

摘要: 深度神经网络(DNN)很容易受到敌意例子的攻击，这可能会导致安全关键域的灾难。为了刻画对抗性实例的特征唯一性，或区分由对抗性实例激活的DNN的行为，人们提出了许多检测方法。基于特征的检测不能处理具有大扰动的对抗性示例。此外，它们还需要大量具体的对抗性例子。另一种主流的基于模型的检测通过模型行为来表征输入属性，但存在计算成本高的问题。为了解决这些问题，我们引入了局部梯度的概念，并揭示了对抗性例子比良性例子具有更大的局部梯度界。受此启发，我们利用局部梯度来检测对抗性实例，并提出了一个通用的框架AdvCheck。具体地说，通过计算少数良性样本和含有噪声的错误分类样本的局部梯度来训练检测器，可以准确地区分对抗性样本甚至错误分类的自然输入。通过大量的实验，我们验证了AdvCheck相对于最新(SOTA)基线的优越性能，对一般对手攻击的平均检测率($\sim\x 1.2$)和对错误分类的自然输入的平均检测率($\sim\x 1.4$)，平均时间开销为1/500。我们还为成功的检测提供了可解释的结果。



## **43. STDLens: Model Hijacking-Resilient Federated Learning for Object Detection**

STDLens：用于目标检测的模型劫持-弹性联合学习 cs.CR

CVPR 2023. Source Code: https://github.com/git-disl/STDLens

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.11511v2) [paper-pdf](http://arxiv.org/pdf/2303.11511v2)

**Authors**: Ka-Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu

**Abstract**: Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates.

摘要: 联邦学习(FL)作为一种协作学习框架，在分布的客户群上训练基于深度学习的目标检测模型，已经越来越受欢迎。尽管有优势，但FL很容易受到模特劫持的攻击。攻击者可以通过在协作学习过程中仅使用少量受攻击的客户端植入特洛伊木马梯度来控制对象检测系统的不当行为。本文介绍了STDLens，一种保护FL免受此类攻击的原则性方法。我们首先调查了现有的缓解机制，并分析了它们由于梯度空间聚类分析的固有错误而导致的失败。基于这些见解，我们引入了一个三层取证框架来识别和排除特洛伊木马的梯度，并在FL过程中恢复性能。我们考虑了三种类型的自适应攻击，并证明了STDLens对高级攻击者的健壮性。大量的实验表明，STDLens能够保护FL免受不同模型的劫持攻击，并且在识别和去除特洛伊木马梯度方面优于现有的方法，具有明显更高的精度和更低的误检率。



## **44. Improving robustness of jet tagging algorithms with adversarial training: exploring the loss surface**

利用对抗性训练提高JET标记算法的稳健性：损失曲面的探索 hep-ex

5 pages, 2 figures; submitted to ACAT 2022 proceedings

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14511v1) [paper-pdf](http://arxiv.org/pdf/2303.14511v1)

**Authors**: Annika Stein

**Abstract**: In the field of high-energy physics, deep learning algorithms continue to gain in relevance and provide performance improvements over traditional methods, for example when identifying rare signals or finding complex patterns. From an analyst's perspective, obtaining highest possible performance is desirable, but recently, some attention has been shifted towards studying robustness of models to investigate how well these perform under slight distortions of input features. Especially for tasks that involve many (low-level) inputs, the application of deep neural networks brings new challenges. In the context of jet flavor tagging, adversarial attacks are used to probe a typical classifier's vulnerability and can be understood as a model for systematic uncertainties. A corresponding defense strategy, adversarial training, improves robustness, while maintaining high performance. Investigating the loss surface corresponding to the inputs and models in question reveals geometric interpretations of robustness, taking correlations into account.

摘要: 在高能物理领域，深度学习算法继续提高相关性，并提供比传统方法更好的性能，例如在识别罕见信号或发现复杂模式时。从分析师的角度来看，获得尽可能高的性能是可取的，但最近，一些注意力已经转移到研究模型的稳健性上，以调查这些模型在输入特征轻微扭曲的情况下表现如何。特别是对于涉及许多(低层)输入的任务，深度神经网络的应用带来了新的挑战。在喷气标签的背景下，对抗性攻击被用来探测典型分类器的脆弱性，并且可以被理解为系统不确定性的模型。一种相应的防御策略，对抗性训练，在保持高性能的同时提高了健壮性。研究与所讨论的输入和模型相对应的损失面，揭示了考虑相关性的稳健性的几何解释。



## **45. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

CgAT：中心引导的基于深度散列的对抗性训练 cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2204.10779v6) [paper-pdf](http://arxiv.org/pdf/2204.10779v6)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.

摘要: 深度哈希法以其高效、高效的特点在海量图像检索中得到了广泛应用。然而，深度哈希模型很容易受到敌意例子的攻击，因此有必要开发针对图像检索的对抗性防御方法。现有的解决方案由于使用弱对抗性样本进行训练，并且缺乏区分优化目标来学习稳健特征，使得防御性能受到限制。本文提出了一种基于最小-最大值的中心引导敌意训练算法，即CgAT，通过最坏的敌意例子来提高深度哈希网络的健壮性。具体地说，我们首先将中心代码定义为输入图像内容的语义区分代表，它保留了与正例的语义相似性和与反例的不相似性。我们证明了一个数学公式可以立即计算中心代码。在获得深度散列网络每次优化迭代的中心代码后，将其用于指导对抗性训练过程。一方面，CgAT通过最大化对抗性示例的哈希码与中心码之间的汉明距离来生成最差的对抗性示例作为扩充数据。另一方面，CgAT通过最小化到中心码的汉明距离来学习减轻对抗性样本的影响。在基准数据集上的大量实验表明，我们的对抗性训练算法在防御基于深度散列的检索的对抗性攻击方面是有效的。与当前最先进的防御方法相比，我们在Flickr-25K、NUS-wide和MS-CoCo上的防御性能分别平均提高了18.61、12.35和11.56。代码可在https://github.com/xunguangwang/CgAT.上获得



## **46. No more Reviewer #2: Subverting Automatic Paper-Reviewer Assignment using Adversarial Learning**

不再有审稿人#2：使用对抗性学习颠覆自动论文审稿人分配 cs.CR

Accepted at USENIX Security Symposium 2023

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14443v1) [paper-pdf](http://arxiv.org/pdf/2303.14443v1)

**Authors**: Thorsten Eisenhofer, Erwin Quiring, Jonas Möller, Doreen Riepel, Thorsten Holz, Konrad Rieck

**Abstract**: The number of papers submitted to academic conferences is steadily rising in many scientific disciplines. To handle this growth, systems for automatic paper-reviewer assignments are increasingly used during the reviewing process. These systems use statistical topic models to characterize the content of submissions and automate the assignment to reviewers. In this paper, we show that this automation can be manipulated using adversarial learning. We propose an attack that adapts a given paper so that it misleads the assignment and selects its own reviewers. Our attack is based on a novel optimization strategy that alternates between the feature space and problem space to realize unobtrusive changes to the paper. To evaluate the feasibility of our attack, we simulate the paper-reviewer assignment of an actual security conference (IEEE S&P) with 165 reviewers on the program committee. Our results show that we can successfully select and remove reviewers without access to the assignment system. Moreover, we demonstrate that the manipulated papers remain plausible and are often indistinguishable from benign submissions.

摘要: 在许多科学领域，提交给学术会议的论文数量正在稳步上升。为了应对这种增长，在审查过程中越来越多地使用自动分配论文审稿人的系统。这些系统使用统计主题模型来表征提交的内容，并自动分配给评审员。在本文中，我们证明了这种自动化可以通过对抗性学习来操纵。我们提出了一种攻击，该攻击改编一篇给定的论文，以便它误导作业并选择自己的审稿人。我们的攻击是基于一种新的优化策略，该策略在特征空间和问题空间之间交替使用，以实现对论文的不引人注目的更改。为了评估我们攻击的可行性，我们模拟了一个实际安全会议(IEEE S&P)的论文审稿人分配，项目委员会有165名审稿人。我们的结果表明，我们可以在不访问分配系统的情况下成功地选择和删除审阅者。此外，我们证明，被操纵的文件仍然可信，通常与良性提交的文件没有区别。



## **47. A User-Based Authentication and DoS Mitigation Scheme for Wearable Wireless Body Sensor Networks**

一种基于用户的可穿戴无线体感网络认证和DoS防御方案 cs.CR

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.14441v1) [paper-pdf](http://arxiv.org/pdf/2303.14441v1)

**Authors**: Nombulelo Zulu, Deon P. Du Plessis, Topside E. Mathonsi, Tshimangadzo M. Tshilongamulenzhe

**Abstract**: Wireless Body Sensor Networks (WBSNs) is one of the greatest growing technology for sensing and performing various tasks. The information transmitted in the WBSNs is vulnerable to cyber-attacks, therefore security is very important. Denial of Service (DoS) attacks are considered one of the major threats against WBSNs security. In DoS attacks, an adversary targets to degrade and shut down the efficient use of the network and disrupt the services in the network causing them inaccessible to its intended users. If sensitive information of patients in WBSNs, such as the medical history is accessed by unauthorized users, the patient may suffer much more than the disease itself, it may result in loss of life. This paper proposes a User-Based authentication scheme to mitigate DoS attacks in WBSNs. A five-phase User-Based authentication DoS mitigation scheme for WBSNs is designed by integrating Elliptic Curve Cryptography (ECC) with Rivest Cipher 4 (RC4) to ensure a strong authentication process that will only allow authorized users to access nodes on WBSNs.

摘要: 无线身体传感器网络(WBSNs)是目前发展最快的传感和执行各种任务的技术之一。无线传感器网络中传输的信息很容易受到网络攻击，因此安全性非常重要。拒绝服务(DoS)攻击被认为是对WBSNs安全的主要威胁之一。在DoS攻击中，对手的目标是降低和关闭网络的有效使用，并中断网络中的服务，使其目标用户无法访问这些服务。如果WBSNs中患者的敏感信息，如病历被未经授权的用户访问，患者可能会遭受比疾病本身更大的痛苦，可能会导致生命损失。提出了一种基于用户的身份认证方案来缓解无线传感器网络中的DoS攻击。通过将椭圆曲线密码体制(ECC)和Rivest密码4(RC4)相结合，设计了一种基于用户的五阶段无线传感器网络认证DoS缓解方案，以确保强认证过程只允许授权用户访问WBSNs上的节点。



## **48. Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation**

一致攻击：具身视觉导航的普遍对抗性扰动 cs.LG

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2206.05751v4) [paper-pdf](http://arxiv.org/pdf/2206.05751v4)

**Authors**: Chengyang Ying, You Qiaoben, Xinning Zhou, Hang Su, Wenbo Ding, Jianyong Ai

**Abstract**: Embodied agents in vision navigation coupled with deep neural networks have attracted increasing attention. However, deep neural networks have been shown vulnerable to malicious adversarial noises, which may potentially cause catastrophic failures in Embodied Vision Navigation. Among different adversarial noises, universal adversarial perturbations (UAP), i.e., a constant image-agnostic perturbation applied on every input frame of the agent, play a critical role in Embodied Vision Navigation since they are computation-efficient and application-practical during the attack. However, existing UAP methods ignore the system dynamics of Embodied Vision Navigation and might be sub-optimal. In order to extend UAP to the sequential decision setting, we formulate the disturbed environment under the universal noise $\delta$, as a $\delta$-disturbed Markov Decision Process ($\delta$-MDP). Based on the formulation, we analyze the properties of $\delta$-MDP and propose two novel Consistent Attack methods, named Reward UAP and Trajectory UAP, for attacking Embodied agents, which consider the dynamic of the MDP and calculate universal noises by estimating the disturbed distribution and the disturbed Q function. For various victim models, our Consistent Attack can cause a significant drop in their performance in the PointGoal task in Habitat with different datasets and different scenes. Extensive experimental results indicate that there exist serious potential risks for applying Embodied Vision Navigation methods to the real world.

摘要: 视觉导航中的具身智能体与深度神经网络相结合，越来越受到人们的关注。然而，深度神经网络已被证明容易受到恶意对抗性噪声的攻击，这可能会导致具身视觉导航中的灾难性故障。在不同的对抗噪声中，通用对抗扰动(UAP)，即在智能体的每一输入帧上施加的与图像无关的恒定扰动，在嵌入视觉导航中起着至关重要的作用，因为它们在攻击过程中具有计算效率和应用实用性。然而，现有的UAP方法忽略了体现视觉导航的系统动力学，可能是次优的。为了将UAP扩展到序贯决策环境，我们将普遍噪声下的扰动环境描述为一个$-扰动马尔可夫决策过程($-MDP)。在此基础上，分析了$Delta$-MDP的特性，提出了两种新的一致性攻击方法--报酬UAP和轨迹UAP，该方法考虑了MDP的动态特性，通过估计扰动分布和扰动Q函数来计算通用噪声。对于不同的受害者模型，我们的一致攻击会导致他们在不同数据集和不同场景下在Habit的PointGoal任务中的性能显著下降。大量的实验结果表明，将具身视觉导航方法应用于现实世界存在着严重的潜在风险。



## **49. Test-time Defense against Adversarial Attacks: Detection and Reconstruction of Adversarial Examples via Masked Autoencoder**

对抗性攻击的测试时间防御：基于屏蔽自动编码器的对抗性实例检测与重构 cs.CV

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2303.12848v2) [paper-pdf](http://arxiv.org/pdf/2303.12848v2)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Existing defense methods against adversarial attacks can be categorized into training time and test time defenses. Training time defense, i.e., adversarial training, requires a significant amount of extra time for training and is often not able to be generalized to unseen attacks. On the other hand, test time defense by test time weight adaptation requires access to perform gradient descent on (part of) the model weights, which could be infeasible for models with frozen weights. To address these challenges, we propose DRAM, a novel defense method to Detect and Reconstruct multiple types of Adversarial attacks via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a KS-test to detect adversarial attacks. Moreover, the MAE losses can be used to repair adversarial samples from unseen attack types. In this sense, DRAM neither requires model weight updates in test time nor augments the training set with more adversarial samples. Evaluating DRAM on the large-scale ImageNet data, we achieve the best detection rate of 82% on average on eight types of adversarial attacks compared with other detection baselines. For reconstruction, DRAM improves the robust accuracy by 6% ~ 41% for Standard ResNet50 and 3% ~ 8% for Robust ResNet50 compared with other self-supervision tasks, such as rotation prediction and contrastive learning.

摘要: 现有的对抗攻击防御方法可分为训练时间防御和测试时间防御。训练时间防守，即对抗性训练，需要大量的额外时间进行训练，通常不能概括为看不见的攻击。另一方面，通过测试时间权重自适应来保护测试时间需要访问对模型权重(部分)执行梯度下降的权限，这对于具有冻结权重的模型可能是不可行的。为了应对这些挑战，我们提出了一种新的防御方法DRAM，它通过掩蔽自动编码器(MAE)来检测和重建多种类型的对抗性攻击。我们演示了如何使用MAE损失来构建KS测试来检测对手攻击。此外，MAE损失可用于修复来自未知攻击类型的敌方样本。从这个意义上说，DRAM既不需要在测试时间更新模型权重，也不需要用更多的对抗性样本来扩充训练集。在大规模的ImageNet数据上对DRAM进行评估，与其他检测基线相比，对8种类型的对抗性攻击平均获得了82%的最佳检测率。在重建方面，与旋转预测和对比学习等其他自我监督任务相比，DRAM将标准ResNet50的稳健准确率提高了6%~41%，稳健ResNet50的稳健准确率提高了3%~8%。



## **50. WiFi Physical Layer Stays Awake and Responds When it Should Not**

WiFi物理层保持唤醒，并在不应唤醒时进行响应 cs.NI

12 pages

**SubmitDate**: 2023-03-25    [abs](http://arxiv.org/abs/2301.00269v2) [paper-pdf](http://arxiv.org/pdf/2301.00269v2)

**Authors**: Ali Abedi, Haofan Lu, Alex Chen, Charlie Liu, Omid Abari

**Abstract**: WiFi communication should be possible only between devices inside the same network. However, we find that all existing WiFi devices send back acknowledgments (ACK) to even fake packets received from unauthorized WiFi devices outside of their network. Moreover, we find that an unauthorized device can manipulate the power-saving mechanism of WiFi radios and keep them continuously awake by sending specific fake beacon frames to them. Our evaluation of over 5,000 devices from 186 vendors confirms that these are widespread issues. We believe these loopholes cannot be prevented, and hence they create privacy and security concerns. Finally, to show the importance of these issues and their consequences, we implement and demonstrate two attacks where an adversary performs battery drain and WiFi sensing attacks just using a tiny WiFi module which costs less than ten dollars.

摘要: WiFi通信应该只能在同一网络内的设备之间进行。然而，我们发现，所有现有的WiFi设备都会向从其网络外部的未经授权的WiFi设备接收的虚假数据包发送回确认(ACK)。此外，我们发现未经授权的设备可以操纵WiFi无线电的节电机制，并通过向其发送特定的虚假信标帧来保持其持续唤醒。我们对186家供应商的5,000多台设备进行的评估证实，这些问题普遍存在。我们认为这些漏洞是无法阻止的，因此它们会造成隐私和安全方面的问题。最后，为了说明这些问题的重要性及其后果，我们实现并演示了两个攻击，其中对手仅使用一个成本不到10美元的微小WiFi模块就可以执行电池耗尽攻击和WiFi传感攻击。



