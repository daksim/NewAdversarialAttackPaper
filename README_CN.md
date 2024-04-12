# Latest Adversarial Attack Papers
**update at 2024-04-12 09:24:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习一个通用的、可转移的对抗后缀生成模型，用于越狱既开放式又封闭式LLM cs.CL

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07921v1) [paper-pdf](http://arxiv.org/pdf/2404.07921v1)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **2. A Measurement of Genuine Tor Traces for Realistic Website Fingerprinting**

真实网站指纹识别的真实Tor痕迹测量 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07892v1) [paper-pdf](http://arxiv.org/pdf/2404.07892v1)

**Authors**: Rob Jansen, Ryan Wails, Aaron Johnson

**Abstract**: Website fingerprinting (WF) is a dangerous attack on web privacy because it enables an adversary to predict the website a user is visiting, despite the use of encryption, VPNs, or anonymizing networks such as Tor. Previous WF work almost exclusively uses synthetic datasets to evaluate the performance and estimate the feasibility of WF attacks despite evidence that synthetic data misrepresents the real world. In this paper we present GTT23, the first WF dataset of genuine Tor traces, which we obtain through a large-scale measurement of the Tor network. GTT23 represents real Tor user behavior better than any existing WF dataset, is larger than any existing WF dataset by at least an order of magnitude, and will help ground the future study of realistic WF attacks and defenses. In a detailed evaluation, we survey 25 WF datasets published over the last 15 years and compare their characteristics to those of GTT23. We discover common deficiencies of synthetic datasets that make them inferior to GTT23 for drawing meaningful conclusions about the effectiveness of WF attacks directed at real Tor users. We have made GTT23 available to promote reproducible research and to help inspire new directions for future work.

摘要: 网站指纹识别(WF)是对网络隐私的一种危险攻击，因为它使对手能够预测用户正在访问的网站，尽管使用了加密、VPN或匿名网络(如ToR)。以前的WF工作几乎完全使用合成数据集来评估WF攻击的性能和估计WF攻击的可行性，尽管有证据表明合成数据歪曲了真实世界。本文介绍了GTT23，这是我们通过对Tor网络的大规模测量而获得的第一个真实Tor痕迹的WF数据集。GTT23比任何现有的WF数据集更好地表示真实的ToR用户行为，比任何现有的WF数据集至少大一个数量级，并将有助于未来对现实WF攻击和防御的研究。在详细的评估中，我们调查了过去15年发布的25个WF数据集，并将它们的特征与GTT23的特征进行了比较。我们发现了合成数据集的共同缺陷，使其在针对真实Tor用户的WF攻击的有效性方面不如GTT23得出有意义的结论。我们已经提供了GTT23，以促进可重复的研究，并帮助启发未来工作的新方向。



## **3. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

多机器人目标跟踪的传感和通信危险区 cs.RO

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07880v1) [paper-pdf](http://arxiv.org/pdf/2404.07880v1)

**Authors**: Jiazhen Li, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.

摘要: 多机器人目标跟踪在环境监测、野火管理等不同场景中有着广泛的应用，这就要求多机器人系统在不确定和危险环境中的实际部署具有很强的鲁棒性。传统的方法往往只关注跟踪精度的性能，没有对环境进行建模和假设，而忽略了实际部署中可能导致系统故障的环境危害。为了应对这一挑战，我们研究了在具有不确定性的感知和通信攻击的对抗性环境中的多机器人目标跟踪。设计了避开不同危险区域的具体策略，提出了危险环境下的多智能体跟踪框架。我们对概率约束进行近似，并制定实用的优化策略来有效地应对计算挑战。我们在仿真中评估了我们提出的方法的性能，以展示机器人在不同的环境不确定性和风险置信度下调整其风险意识行为的能力。通过真实世界的机器人实验进一步验证了所提出的方法，其中一组无人机成功地跟踪了动态的地面机器人，同时意识到了传感和/或通信危险区域的风险。



## **4. LeapFrog: The Rowhammer Instruction Skip Attack**

LeapFrog：Rowhammer指令跳过攻击 cs.CR

Accepted at Hardware.io 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07878v1) [paper-pdf](http://arxiv.org/pdf/2404.07878v1)

**Authors**: Andrew Adiletta, Caner Tol, Berk Sunar

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats not only compromising data integrity but also the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The Leapfrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, re-positions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify Leapfrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first showcase this new attack vector through a practical demonstration on a TLS handshake client/server scenario, successfully inducing an instruction skip in a client application. We then demonstrate the attack on real-world code found in the wild, implementing an attack on OpenSSL.   Our findings extend the impact of Rowhammer attacks on control flow and contribute to the development of more robust defenses against these increasingly sophisticated threats.

摘要: 自成立以来，Rowhammer漏洞攻击已迅速演变为日益复杂的威胁，不仅危及数据完整性，还危及受害者进程的控制流完整性。然而，对于攻击者来说，识别易受攻击的目标(即Rowhammer小工具)、了解尝试的故障的结果并制定产生有用结果的攻击仍然是一项挑战。在本文中，我们提出了一种新的Rowhammer小工具，称为LeapFrog小工具，当它存在于受害者代码中时，允许攻击者破坏代码执行以绕过关键代码段(例如，身份验证逻辑、加密轮、安全协议中的填充)。当受害者代码在用户或内核堆栈中存储程序计数器(PC)值(例如，函数调用期间的返回地址)时，当被篡改时，将返回地址重新定位到绕过安全关键代码模式的位置时，LeapFrog小工具就会显现出来。这项研究还提出了一个识别LeapFrog小工具的系统过程。这种方法能够自动检测易受影响的目标并确定最佳攻击参数。我们首先通过TLS握手客户端/服务器场景的实际演示展示了这种新的攻击载体，成功地在客户端应用程序中诱导了指令跳过。然后，我们演示了对在野外发现的真实代码的攻击，实现了对OpenSSL的攻击。我们的发现扩大了Rowhammer攻击对控制流的影响，并有助于开发针对这些日益复杂的威胁的更强大的防御措施。



## **5. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

无小区大规模MIMO下行链路导频欺骗攻击：基于对手的视角 cs.IT

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2403.04435v2) [paper-pdf](http://arxiv.org/pdf/2403.04435v2)

**Authors**: Weiyang Xu, Ruiguang Wang, Yuan Zhang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.

摘要: 与蜂窝系统相比，无小区大规模多输入多输出(MMIMO)系统中的信道硬化效应不那么明显，因此有必要估计下行链路的有效信道增益以确保良好的性能。然而，下行训练无意中为敌对节点创造了发起试点欺骗攻击(PSA)的机会。首先，我们证明了敌意分布式接入点(AP)会严重降低可实现的下行链路速率。它们通过在上行链路训练阶段估计其对用户的信道，然后预编码并发送与合法AP在下行链路训练阶段使用的导频序列相同的导频序列来实现这一点。然后，通过严格推导每个用户可实现的下行链路速率的闭合形式表达式来研究下行链路PSA的影响。通过使用最小-最大准则来优化功率分配系数，从对抗性AP的角度最小化每用户可实现的最大下行传输速率。作为下行链路PSA的替代方案，敌意AP可以选择在下行链路数据传输阶段对随机干扰进行预编码，以便中断合法通信。在这种情况下，推导了可实现的下行链路速率，并开发了功率优化算法。我们给出了数值结果来展示下行PSA的有害影响，并比较了这两种类型的攻击的影响。



## **6. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

基于执行状态证明的联邦学习和差分隐私中毒预防 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.06721v2) [paper-pdf](http://arxiv.org/pdf/2404.06721v2)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.

摘要: 物联网驱动的分布式数据分析的兴起，加上对隐私的日益担忧，导致了对有效的隐私保护和联合数据收集/模型培训机制的需求。在过去的几年里，联邦学习(FL)和局部差异隐私(LDP)等方法被提出并引起了人们的广泛关注。然而，它们仍然有一个共同的局限性，即容易受到中毒攻击，在这些攻击中，危害边缘设备的对手提供伪造的(也称为。有毒)数据到聚合后端，破坏FL/LDP结果的完整性。在这项工作中，我们提出了一种基于物联网/嵌入式设备软件状态执行证明(PoSX)的新的安全概念来解决这一问题。为了实现PoSX的概念，我们设计了SLAPP：一种系统级的中毒预防方法。SLAPP利用嵌入式设备的商用安全功能--尤其是ARM TrustZoneM安全扩展--作为FL/LDP边缘设备例程的一部分，以可验证的方式将原始感测数据与其正确使用绑定在一起。因此，它为防止中毒提供了强有力的安全保障。我们的评估基于具有多个加密原语和数据收集方案的真实世界原型，展示了SLAPP的安全性和低开销。



## **7. Enhancing Network Intrusion Detection Performance using Generative Adversarial Networks**

利用生成对抗网络提高网络入侵检测性能 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07464v1) [paper-pdf](http://arxiv.org/pdf/2404.07464v1)

**Authors**: Xinxing Zhao, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Network intrusion detection systems (NIDS) play a pivotal role in safeguarding critical digital infrastructures against cyber threats. Machine learning-based detection models applied in NIDS are prevalent today. However, the effectiveness of these machine learning-based models is often limited by the evolving and sophisticated nature of intrusion techniques as well as the lack of diverse and updated training samples. In this research, a novel approach for enhancing the performance of an NIDS through the integration of Generative Adversarial Networks (GANs) is proposed. By harnessing the power of GANs in generating synthetic network traffic data that closely mimics real-world network behavior, we address a key challenge associated with NIDS training datasets, which is the data scarcity. Three distinct GAN models (Vanilla GAN, Wasserstein GAN and Conditional Tabular GAN) are implemented in this work to generate authentic network traffic patterns specifically tailored to represent the anomalous activity. We demonstrate how this synthetic data resampling technique can significantly improve the performance of the NIDS model for detecting such activity. By conducting comprehensive experiments using the CIC-IDS2017 benchmark dataset, augmented with GAN-generated data, we offer empirical evidence that shows the effectiveness of our proposed approach. Our findings show that the integration of GANs into NIDS can lead to enhancements in intrusion detection performance for attacks with limited training data, making it a promising avenue for bolstering the cybersecurity posture of organizations in an increasingly interconnected and vulnerable digital landscape.

摘要: 网络入侵检测系统在保护关键数字基础设施免受网络威胁方面发挥着举足轻重的作用。目前，基于机器学习的检测模型在网络入侵检测系统中的应用非常普遍。然而，这些基于机器学习的模型的有效性往往受到入侵技术不断发展和复杂的性质以及缺乏多样化和更新的训练样本的限制。在这项研究中，提出了一种通过整合生成性对抗网络(GANS)来提高网络入侵检测系统性能的新方法。通过利用Gans生成接近模拟真实网络行为的合成网络流量数据的能力，我们解决了与NIDS训练数据集相关的一个关键挑战，即数据稀缺性。本文实现了三种不同的GAN模型(Vanilla GAN、Wasserstein GAN和Conditional Tablular GAN)来生成真实的网络流量模式，该模式专门用于表示异常活动。我们演示了这种合成数据重采样技术如何显著提高NIDS模型检测此类活动的性能。通过使用CIC-IDS2017基准数据集和GaN生成的数据进行全面的实验，我们提供了经验证据，证明了我们所提出的方法的有效性。我们的研究结果表明，将GANS集成到网络入侵检测系统中可以增强对训练数据有限的攻击的入侵检测性能，使其成为在日益互联和脆弱的数字环境中支持组织网络安全态势的一种有前途的途径。



## **8. Privacy preserving layer partitioning for Deep Neural Network models**

深度神经网络模型的隐私保护层划分 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07437v1) [paper-pdf](http://arxiv.org/pdf/2404.07437v1)

**Authors**: Kishore Rajasekar, Randolph Loh, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: MLaaS (Machine Learning as a Service) has become popular in the cloud computing domain, allowing users to leverage cloud resources for running private inference of ML models on their data. However, ensuring user input privacy and secure inference execution is essential. One of the approaches to protect data privacy and integrity is to use Trusted Execution Environments (TEEs) by enabling execution of programs in secure hardware enclave. Using TEEs can introduce significant performance overhead due to the additional layers of encryption, decryption, security and integrity checks. This can lead to slower inference times compared to running on unprotected hardware. In our work, we enhance the runtime performance of ML models by introducing layer partitioning technique and offloading computations to GPU. The technique comprises two distinct partitions: one executed within the TEE, and the other carried out using a GPU accelerator. Layer partitioning exposes intermediate feature maps in the clear which can lead to reconstruction attacks to recover the input. We conduct experiments to demonstrate the effectiveness of our approach in protecting against input reconstruction attacks developed using trained conditional Generative Adversarial Network(c-GAN). The evaluation is performed on widely used models such as VGG-16, ResNet-50, and EfficientNetB0, using two datasets: ImageNet for Image classification and TON IoT dataset for cybersecurity attack detection.

摘要: MLaaS(机器学习即服务)在云计算领域变得流行起来，允许用户利用云资源对其数据运行ML模型的私有推理。然而，确保用户输入隐私和安全推理执行是必不可少的。保护数据隐私和完整性的方法之一是使用可信执行环境(TEE)，通过在安全硬件飞地中执行程序来实现。由于加密、解密、安全和完整性检查的附加层，使用TES可能会带来显著的性能开销。与在不受保护的硬件上运行相比，这可能会导致较慢的推断时间。在我们的工作中，我们通过引入层划分技术和将计算卸载到GPU来提高ML模型的运行时性能。该技术包括两个不同的分区：一个在TEE内执行，另一个使用GPU加速器执行。层划分将中间特征映射暴露在明文中，这可能导致重建攻击以恢复输入。我们进行了实验，以证明我们的方法在防止输入重构攻击方面的有效性，该攻击是使用训练的条件生成对抗网络(c-GAN)开发的。在VGG-16、ResNet-50和EfficientNetB0等广泛使用的模型上进行了评估，使用了两个数据集：用于图像分类的ImageNet和用于网络安全攻击检测的Ton IoT数据集。



## **9. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

黑盒神经排序模型的多粒度对抗攻击 cs.IR

Accepted by SIGIR2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.01574v2) [paper-pdf](http://arxiv.org/pdf/2404.01574v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word or sentence level, to target documents. However, limiting perturbations to a single level of granularity may reduce the flexibility of adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step build on the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.

摘要: 对抗性排序攻击因其在探测漏洞方面的成功，从而增强了神经排序模型的稳健性而受到越来越多的关注。传统的攻击方法在单个粒度(例如，单词或句子级别)上使用扰动以文档为目标。然而，将扰动限制在单一的粒度级别可能会降低对抗性示例的灵活性，从而降低攻击的潜在威胁。因此，我们专注于通过结合多粒度扰动来生成高质量的对抗性实例。实现这一目标需要处理组合爆炸问题，这需要确定跨所有可能级别的粒度、位置和文本片段的扰动的最佳组合。为了应对这一挑战，我们将多粒度的对抗性攻击转化为一个顺序的决策过程，其中下一攻击步骤中的扰动建立在当前攻击步骤中被扰动的文档之上。由于攻击过程只能访问最终状态，没有直接的中间信号，因此我们使用强化学习来执行多粒度攻击。在强化学习过程中，两个代理协作识别多粒度漏洞作为攻击目标，并将扰动候选组织成最终的扰动序列。实验结果表明，我们的攻击方法在攻击有效性和不可感知性方面都超过了主流基线。



## **10. Incremental Randomized Smoothing Certification**

增量随机平滑认证 cs.LG

ICLR 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2305.19521v2) [paper-pdf](http://arxiv.org/pdf/2305.19521v2)

**Authors**: Shubham Ugare, Tarun Suresh, Debangshu Banerjee, Gagandeep Singh, Sasa Misailovic

**Abstract**: Randomized smoothing-based certification is an effective approach for obtaining robustness certificates of deep neural networks (DNNs) against adversarial attacks. This method constructs a smoothed DNN model and certifies its robustness through statistical sampling, but it is computationally expensive, especially when certifying with a large number of samples. Furthermore, when the smoothed model is modified (e.g., quantized or pruned), certification guarantees may not hold for the modified DNN, and recertifying from scratch can be prohibitively expensive.   We present the first approach for incremental robustness certification for randomized smoothing, IRS. We show how to reuse the certification guarantees for the original smoothed model to certify an approximated model with very few samples. IRS significantly reduces the computational cost of certifying modified DNNs while maintaining strong robustness guarantees. We experimentally demonstrate the effectiveness of our approach, showing up to 3x certification speedup over the certification that applies randomized smoothing of the approximate model from scratch.

摘要: 基于随机平滑的认证是获得深层神经网络抗攻击健壮性证书的有效方法。该方法构造了一个平滑的DNN模型，并通过统计抽样验证了其稳健性，但其计算量很大，特别是在需要大量样本的情况下。此外，当修改平滑的模型(例如，量化或修剪)时，认证保证可能不适用于修改的DNN，并且从头开始重新认证可能昂贵得令人望而却步。我们提出了第一种随机平滑的增量式稳健性证明方法--IRS。我们展示了如何重用对原始平滑模型的证明保证来证明具有很少样本的近似模型。IRS在保持较强的健壮性保证的同时，显著降低了证明修改的DNN的计算代价。我们在实验中展示了我们方法的有效性，与从头开始应用随机平滑近似模型的认证相比，认证加速高达3倍。



## **11. Indoor Location Fingerprinting Privacy: A Comprehensive Survey**

室内位置指纹隐私：综合调查 cs.CR

Submitted to ACM Computing Surveys

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07345v1) [paper-pdf](http://arxiv.org/pdf/2404.07345v1)

**Authors**: Amir Fathalizadeh, Vahideh Moghtadaiee, Mina Alishahi

**Abstract**: The pervasive integration of Indoor Positioning Systems (IPS) arises from the limitations of Global Navigation Satellite Systems (GNSS) in indoor environments, leading to the widespread adoption of Location-Based Services (LBS). Specifically, indoor location fingerprinting employs diverse signal fingerprints from user devices, enabling precise location identification by Location Service Providers (LSP). Despite its broad applications across various domains, indoor location fingerprinting introduces a notable privacy risk, as both LSP and potential adversaries inherently have access to this sensitive information, compromising users' privacy. Consequently, concerns regarding privacy vulnerabilities in this context necessitate a focused exploration of privacy-preserving mechanisms. In response to these concerns, this survey presents a comprehensive review of Privacy-Preserving Mechanisms in Indoor Location Fingerprinting (ILFPPM) based on cryptographic, anonymization, differential privacy (DP), and federated learning (FL) techniques. We also propose a distinctive and novel grouping of privacy vulnerabilities, adversary and attack models, and available evaluation metrics specific to indoor location fingerprinting systems. Given the identified limitations and research gaps in this survey, we highlight numerous prospective opportunities for future investigation, aiming to motivate researchers interested in advancing this field. This survey serves as a valuable reference for researchers and provides a clear overview for those beyond this specific research domain.

摘要: 室内定位系统(IPS)的广泛集成源于全球导航卫星系统(GNSS)在室内环境中的局限性，导致基于位置的服务(LBS)的广泛采用。具体地说，室内位置指纹识别使用来自用户设备的不同信号指纹，从而实现位置服务提供商(LSP)的精确位置识别。尽管其广泛应用于各个领域，但室内位置指纹识别带来了显著的隐私风险，因为LSP和潜在的对手天生都可以访问这些敏感信息，从而危及用户的隐私。因此，在这种情况下，对隐私漏洞的担忧需要集中探索隐私保护机制。针对这些问题，本文基于密码学、匿名化、差分隐私(DP)和联合学习(FL)等技术，对室内位置指纹识别(ILFPPM)中的隐私保护机制进行了全面的综述。我们还提出了一种独特而新颖的隐私漏洞、对手和攻击模型的分组，以及针对室内位置指纹系统的可用评估指标。鉴于这项调查中确定的局限性和研究差距，我们强调了未来研究的许多预期机会，旨在激励有兴趣推进这一领域的研究人员。这项调查为研究人员提供了有价值的参考，并为这一特定研究领域以外的人提供了一个明确的概述。



## **12. Towards a Game-theoretic Understanding of Explanation-based Membership Inference Attacks**

基于博弈论的成员推断攻击的博弈理解 cs.AI

arXiv admin note: text overlap with arXiv:2202.02659

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07139v1) [paper-pdf](http://arxiv.org/pdf/2404.07139v1)

**Authors**: Kavita Kumari, Murtuza Jadliwala, Sumit Kumar Jha, Anindya Maiti

**Abstract**: Model explanations improve the transparency of black-box machine learning (ML) models and their decisions; however, they can also be exploited to carry out privacy threats such as membership inference attacks (MIA). Existing works have only analyzed MIA in a single "what if" interaction scenario between an adversary and the target ML model; thus, it does not discern the factors impacting the capabilities of an adversary in launching MIA in repeated interaction settings. Additionally, these works rely on assumptions about the adversary's knowledge of the target model's structure and, thus, do not guarantee the optimality of the predefined threshold required to distinguish the members from non-members. In this paper, we delve into the domain of explanation-based threshold attacks, where the adversary endeavors to carry out MIA attacks by leveraging the variance of explanations through iterative interactions with the system comprising of the target ML model and its corresponding explanation method. We model such interactions by employing a continuous-time stochastic signaling game framework. In our framework, an adversary plays a stopping game, interacting with the system (having imperfect information about the type of an adversary, i.e., honest or malicious) to obtain explanation variance information and computing an optimal threshold to determine the membership of a datapoint accurately. First, we propose a sound mathematical formulation to prove that such an optimal threshold exists, which can be used to launch MIA. Then, we characterize the conditions under which a unique Markov perfect equilibrium (or steady state) exists in this dynamic system. By means of a comprehensive set of simulations of the proposed game model, we assess different factors that can impact the capability of an adversary to launch MIA in such repeated interaction settings.

摘要: 模型解释提高了黑盒机器学习(ML)模型及其决策的透明度；然而，它们也可以被利用来实施隐私威胁，如成员推理攻击(MIA)。已有的研究只分析了敌方和目标ML模型之间的单一假设交互场景中的MIA，没有发现影响敌方在重复交互环境下发起MIA的能力的因素。此外，这些工作依赖于关于对手对目标模型结构的知识的假设，因此，不能保证区分成员和非成员所需的预定义阈值的最佳性。在本文中，我们深入研究了基于解释的门限攻击领域，即攻击者通过与目标ML模型及其相应解释方法组成的系统的迭代交互，利用解释的差异来努力实施MIA攻击。我们采用一个连续时间随机信号博弈框架对这种相互作用进行建模。在我们的框架中，对手进行停止博弈，与系统交互(拥有关于对手类型的不完善信息，即诚实或恶意)以获得解释差异信息，并计算最优阈值以准确确定数据点的成员资格。首先，我们提出了一个合理的数学公式来证明这样一个最优门限的存在，该最优门限可用于启动MIA。然后，我们刻画了该动力系统存在唯一的马尔可夫完全平衡(或稳态)的条件。通过对所提出的博弈模型的全面模拟，我们评估了在这种重复交互环境中影响对手发起MIA的能力的不同因素。



## **13. Adversarial purification for no-reference image-quality metrics: applicability study and new methods**

无参考图像质量度量的对抗纯化：适用性研究和新方法 cs.CV

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06957v1) [paper-pdf](http://arxiv.org/pdf/2404.06957v1)

**Authors**: Aleksandr Gushchin, Anna Chistyakova, Vladislav Minashkin, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Recently, the area of adversarial attacks on image quality metrics has begun to be explored, whereas the area of defences remains under-researched. In this study, we aim to cover that case and check the transferability of adversarial purification defences from image classifiers to IQA methods. In this paper, we apply several widespread attacks on IQA models and examine the success of the defences against them. The purification methodologies covered different preprocessing techniques, including geometrical transformations, compression, denoising, and modern neural network-based methods. Also, we address the challenge of assessing the efficacy of a defensive methodology by proposing ways to estimate output visual quality and the success of neutralizing attacks. Defences were tested against attack on three IQA metrics -- Linearity, MetaIQA and SPAQ. The code for attacks and defences is available at: (link is hidden for a blind review).

摘要: 最近，对图像质量指标的对抗性攻击领域已经开始探索，而防御领域的研究仍然不足。在这项研究中，我们的目标是涵盖这种情况，并检查对抗纯化防御从图像分类器到IQA方法的可转移性。在本文中，我们应用了几个广泛的攻击IQA模型和检查的成功防御他们。纯化方法涵盖了不同的预处理技术，包括几何变换、压缩、去噪和基于神经网络的现代方法。此外，我们解决了评估防御方法的有效性的挑战，提出了估计输出视觉质量和中和攻击的成功的方法。针对三个IQA指标-线性、MetaIQA和SPAQ-的攻击进行了防御测试。攻击和防御的代码可在：（隐藏链接以供盲目审查）。



## **14. Simpler becomes Harder: Do LLMs Exhibit a Coherent Behavior on Simplified Corpora?**

简单变得更难：LLM在简化语料库上表现出一致性行为吗？ cs.CL

Published at DeTermIt! Workshop at LREC-COLING 2024

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06838v1) [paper-pdf](http://arxiv.org/pdf/2404.06838v1)

**Authors**: Miriam Anschütz, Edoardo Mosca, Georg Groh

**Abstract**: Text simplification seeks to improve readability while retaining the original content and meaning. Our study investigates whether pre-trained classifiers also maintain such coherence by comparing their predictions on both original and simplified inputs. We conduct experiments using 11 pre-trained models, including BERT and OpenAI's GPT 3.5, across six datasets spanning three languages. Additionally, we conduct a detailed analysis of the correlation between prediction change rates and simplification types/strengths. Our findings reveal alarming inconsistencies across all languages and models. If not promptly addressed, simplified inputs can be easily exploited to craft zero-iteration model-agnostic adversarial attacks with success rates of up to 50%

摘要: 文本简化旨在提高可读性，同时保留原始内容和含义。我们的研究通过比较原始输入和简化输入的预测来研究预训练分类器是否也保持这种一致性。我们使用11个预训练模型进行了实验，包括BERT和OpenAI的GPT 3.5，跨越三种语言的六个数据集。此外，我们还对预测变化率和简化类型/强度之间的相关性进行了详细的分析。我们的发现揭示了所有语言和模型之间的惊人不一致性。如果没有及时解决，简化的输入很容易被用来制造零迭代模型不可知的对抗攻击，成功率高达50%。



## **15. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

MixedNUTS：通过非线性混合分类器实现免训练精度-鲁棒性平衡 cs.LG

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2402.02263v2) [paper-pdf](http://arxiv.org/pdf/2402.02263v2)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.

摘要: 对抗性的稳健性往往是以降低精度为代价的，这阻碍了稳健分类模型的实际应用。基于培训的更好权衡的解决方案受到与已经培训的高性能大型模型不兼容的限制，因此有必要探索无需培训的整体方法。我们观察到，稳健模型在正确预测中的信心比基于干净和敌对数据的不正确预测更有信心，我们推测，放大这种“良性置信度属性”可以在整体设置中调和准确性和稳健性。为了实现这一点，我们提出了一种无需训练的方法“MixedNUTS”，其中稳健分类器和标准非稳健分类器的输出逻辑通过只有三个参数的非线性变换来处理，并通过有效的算法进行优化。MixedNUTS然后将转换后的Logit转换为概率，并将它们混合为整体输出。在CIFAR-10、CIFAR-100和ImageNet数据集上，自定义强自适应攻击的实验结果表明，MixedNUTS的精确度和接近SOTA的稳健性都得到了极大的提高--它将CIFAR-100的干净精确度提高了7.86个点，而健壮精确度仅牺牲了0.87个点。



## **16. Logit Calibration and Feature Contrast for Robust Federated Learning on Non-IID Data**

非IID数据鲁棒联邦学习的Logit校正和特征对比 cs.LG

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06776v1) [paper-pdf](http://arxiv.org/pdf/2404.06776v1)

**Authors**: Yu Qiao, Chaoning Zhang, Apurba Adhikary, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed framework for collaborative model training on devices in edge networks. However, challenges arise due to vulnerability to adversarial examples (AEs) and the non-independent and identically distributed (non-IID) nature of data distribution among devices, hindering the deployment of adversarially robust and accurate learning models at the edge. While adversarial training (AT) is commonly acknowledged as an effective defense strategy against adversarial attacks in centralized training, we shed light on the adverse effects of directly applying AT in FL that can severely compromise accuracy, especially in non-IID challenges. Given this limitation, this paper proposes FatCC, which incorporates local logit \underline{C}alibration and global feature \underline{C}ontrast into the vanilla federated adversarial training (\underline{FAT}) process from both logit and feature perspectives. This approach can effectively enhance the federated system's robust accuracy (RA) and clean accuracy (CA). First, we propose logit calibration, where the logits are calibrated during local adversarial updates, thereby improving adversarial robustness. Second, FatCC introduces feature contrast, which involves a global alignment term that aligns each local representation with unbiased global features, thus further enhancing robustness and accuracy in federated adversarial environments. Extensive experiments across multiple datasets demonstrate that FatCC achieves comparable or superior performance gains in both CA and RA compared to other baselines.

摘要: 联合学习(FL)是一种保护隐私的分布式框架，用于边缘网络中设备上的协作模型训练。然而，由于易受对抗性示例(AE)的攻击，以及设备之间数据分布的非独立和相同分布(Non-IID)的性质，出现了挑战，阻碍了在边缘部署对抗性的健壮和准确的学习模型。虽然对抗训练(AT)被公认为是集中训练中对抗攻击的一种有效防御策略，但我们揭示了在外语教学中直接应用AT的不利影响，它会严重影响准确性，特别是在非IID挑战中。针对这一局限性，本文提出了FatCC，它从Logit和特征两个角度将局部Logit\Underline{C}校准和全局特征\Underline{C}对比引入到普通的联合对手训练(\Underline{FAT})过程中。该方法可以有效地提高联邦系统的鲁棒精度(RA)和清洁精度(CA)。首先，我们提出了LOGIT校准，其中LOGIT在本地对抗性更新期间被校准，从而提高了对抗性健壮性。其次，FatCC引入了特征对比度，它涉及一个全局对齐项，将每个局部表示与无偏的全局特征对齐，从而进一步增强了联合对抗环境中的稳健性和准确性。跨多个数据集的广泛实验表明，与其他基准相比，FatCC在CA和RA方面都获得了可比或更好的性能提升。



## **17. False Claims against Model Ownership Resolution**

针对模型所有权决议的虚假声明 cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2304.06607v7) [paper-pdf](http://arxiv.org/pdf/2304.06607v7)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we show that our false claim attacks always succeed in the MOR schemes that follow our generalization, including in a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的实证评估，我们表明我们的虚假声明攻击在遵循我们的推广的MOR方案中总是成功的，包括在真实世界的模型中：亚马逊的Rekognition API。



## **18. Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs**

三明治攻击：对LLM的多语言混合自适应攻击 cs.CR

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.07242v1) [paper-pdf](http://arxiv.org/pdf/2404.07242v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Large Language Models (LLMs) are increasingly being developed and applied, but their widespread use faces challenges. These include aligning LLMs' responses with human values to prevent harmful outputs, which is addressed through safety training methods. Even so, bad actors and malicious users have succeeded in attempts to manipulate the LLMs to generate misaligned responses for harmful questions such as methods to create a bomb in school labs, recipes for harmful drugs, and ways to evade privacy rights. Another challenge is the multilingual capabilities of LLMs, which enable the model to understand and respond in multiple languages. Consequently, attackers exploit the unbalanced pre-training datasets of LLMs in different languages and the comparatively lower model performance in low-resource languages than high-resource ones. As a result, attackers use a low-resource languages to intentionally manipulate the model to create harmful responses. Many of the similar attack vectors have been patched by model providers, making the LLMs more robust against language-based manipulation. In this paper, we introduce a new black-box attack vector called the \emph{Sandwich attack}: a multi-language mixture attack, which manipulates state-of-the-art LLMs into generating harmful and misaligned responses. Our experiments with five different models, namely Google's Bard, Gemini Pro, LLaMA-2-70-B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS, show that this attack vector can be used by adversaries to generate harmful responses and elicit misaligned responses from these models. By detailing both the mechanism and impact of the Sandwich attack, this paper aims to guide future research and development towards more secure and resilient LLMs, ensuring they serve the public good while minimizing potential for misuse.

摘要: 大型语言模型(LLM)的开发和应用越来越多，但它们的广泛使用面临着挑战。这些措施包括使LLMS的反应与人的价值观相一致，以防止有害的输出，这是通过安全培训方法解决的。尽管如此，不良行为者和恶意用户仍成功地操纵LLMS，以生成对有害问题的错误响应，这些问题包括在学校实验室制造炸弹的方法、有害药物的配方以及逃避隐私权的方法。另一个挑战是LLMS的多语言能力，这使得该模型能够理解并以多种语言响应。因此，攻击者利用不同语言的LLMS的不平衡的预训练数据集，以及低资源语言的模型性能相对较低的高资源语言。因此，攻击者使用低资源语言来故意操纵模型以创建有害的响应。许多类似的攻击载体已经由模型提供商修补，使LLM对基于语言的操纵更加健壮。本文介绍了一种新的黑盒攻击向量--夹心攻击：一种多语言混合攻击，它操纵最先进的LLM产生有害的和未对齐的响应。我们对谷歌的Bard、Gemini Pro、Llama-2-70-B-Chat、GPT-3.5-Turbo、GPT-4和Claude-3-opus这五个不同的模型进行的实验表明，该攻击向量可被攻击者用来生成有害响应并从这些模型中引发错误的响应。通过详细描述三明治攻击的机制和影响，本文旨在引导未来的研究和开发朝着更安全和更具弹性的方向发展，确保它们服务于公共利益，同时将滥用的可能性降至最低。



## **19. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\textit{LinkPrompt}$：基于XSLT语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.16432v3) [paper-pdf](http://arxiv.org/pdf/2403.16432v3)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo. The resource is available at $\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。该资源可在$\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.上获得



## **20. LRR: Language-Driven Resamplable Continuous Representation against Adversarial Tracking Attacks**

LRR：一种对抗跟踪攻击的可重采样连续表示方法 cs.CV

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06247v1) [paper-pdf](http://arxiv.org/pdf/2404.06247v1)

**Authors**: Jianlang Chen, Xuhong Ren, Qing Guo, Felix Juefei-Xu, Di Lin, Wei Feng, Lei Ma, Jianjun Zhao

**Abstract**: Visual object tracking plays a critical role in visual-based autonomous systems, as it aims to estimate the position and size of the object of interest within a live video. Despite significant progress made in this field, state-of-the-art (SOTA) trackers often fail when faced with adversarial perturbations in the incoming frames. This can lead to significant robustness and security issues when these trackers are deployed in the real world. To achieve high accuracy on both clean and adversarial data, we propose building a spatial-temporal continuous representation using the semantic text guidance of the object of interest. This novel continuous representation enables us to reconstruct incoming frames to maintain semantic and appearance consistency with the object of interest and its clean counterparts. As a result, our proposed method successfully defends against different SOTA adversarial tracking attacks while maintaining high accuracy on clean data. In particular, our method significantly increases tracking accuracy under adversarial attacks with around 90% relative improvement on UAV123, which is even higher than the accuracy on clean data.

摘要: 视觉对象跟踪在基于视觉的自主系统中起着至关重要的作用，因为它的目标是估计实时视频中感兴趣对象的位置和大小。尽管在这一领域取得了重大进展，但最先进的(SOTA)跟踪器在面对传入帧中的对抗性扰动时往往会失败。当这些跟踪器部署在现实世界中时，这可能会导致严重的健壮性和安全性问题。为了实现对干净数据和对抗性数据的高准确度，我们建议使用感兴趣对象的语义文本指导来构建时空连续表示。这种新颖的连续表示使我们能够重建进入的帧，以保持与感兴趣的对象及其干净的对应物在语义和外观上的一致性。因此，我们提出的方法成功地防御了不同的SOTA对手跟踪攻击，同时保持了对干净数据的高精度。特别是，我们的方法显著提高了对抗性攻击下的跟踪准确率，与UAV123相比，相对提高了90%左右，甚至高于在干净数据上的准确率。



## **21. Towards Robust Domain Generation Algorithm Classification**

鲁棒的领域生成算法分类 cs.CR

Accepted at ACM Asia Conference on Computer and Communications  Security (ASIA CCS 2024)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06236v1) [paper-pdf](http://arxiv.org/pdf/2404.06236v1)

**Authors**: Arthur Drichel, Marc Meyer, Ulrike Meyer

**Abstract**: In this work, we conduct a comprehensive study on the robustness of domain generation algorithm (DGA) classifiers. We implement 32 white-box attacks, 19 of which are very effective and induce a false-negative rate (FNR) of $\approx$ 100\% on unhardened classifiers. To defend the classifiers, we evaluate different hardening approaches and propose a novel training scheme that leverages adversarial latent space vectors and discretized adversarial domains to significantly improve robustness. In our study, we highlight a pitfall to avoid when hardening classifiers and uncover training biases that can be easily exploited by attackers to bypass detection, but which can be mitigated by adversarial training (AT). In our study, we do not observe any trade-off between robustness and performance, on the contrary, hardening improves a classifier's detection performance for known and unknown DGAs. We implement all attacks and defenses discussed in this paper as a standalone library, which we make publicly available to facilitate hardening of DGA classifiers: https://gitlab.com/rwth-itsec/robust-dga-detection

摘要: 在这项工作中，我们对域生成算法(DGA)分类器的稳健性进行了全面的研究。我们实现了32个白盒攻击，其中19个非常有效，并在未硬化的分类器上诱导了约100美元的假阴性率(FNR)。为了保护分类器，我们评估了不同的强化方法，并提出了一种新的训练方案，该方案利用对抗性潜在空间向量和离散化的对抗性领域来显著提高鲁棒性。在我们的研究中，我们强调了在硬化分类器和发现训练偏差时需要避免的陷阱，攻击者可以很容易地利用这些偏差来绕过检测，但可以通过对抗性训练(AT)来缓解。在我们的研究中，我们没有观察到稳健性和性能之间的任何权衡，相反，硬化提高了分类器对已知和未知DGA的检测性能。我们将本文讨论的所有攻击和防御作为一个独立库来实现，我们公开该库是为了促进DGA分类器的强化：https://gitlab.com/rwth-itsec/robust-dga-detection



## **22. FLEX: FLEXible Federated Learning Framework**

FLEX：Flexible联邦学习框架 cs.CR

Submitted to Information Fusion

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06127v1) [paper-pdf](http://arxiv.org/pdf/2404.06127v1)

**Authors**: Francisco Herrera, Daniel Jiménez-López, Alberto Argente-Garrido, Nuria Rodríguez-Barroso, Cristina Zuheros, Ignacio Aguilera-Martos, Beatriz Bello, Mario García-Márquez, M. Victoria Luzón

**Abstract**: In the realm of Artificial Intelligence (AI), the need for privacy and security in data processing has become paramount. As AI applications continue to expand, the collection and handling of sensitive data raise concerns about individual privacy protection. Federated Learning (FL) emerges as a promising solution to address these challenges by enabling decentralized model training on local devices, thus preserving data privacy. This paper introduces FLEX: a FLEXible Federated Learning Framework designed to provide maximum flexibility in FL research experiments. By offering customizable features for data distribution, privacy parameters, and communication strategies, FLEX empowers researchers to innovate and develop novel FL techniques. The framework also includes libraries for specific FL implementations including: (1) anomalies, (2) blockchain, (3) adversarial attacks and defences, (4) natural language processing and (5) decision trees, enhancing its versatility and applicability in various domains. Overall, FLEX represents a significant advancement in FL research, facilitating the development of robust and efficient FL applications.

摘要: 在人工智能(AI)领域，数据处理中对隐私和安全的需求已经变得至关重要。随着人工智能应用的不断扩大，敏感数据的收集和处理引发了对个人隐私保护的担忧。联合学习(FL)通过在本地设备上实现分散的模型训练，从而保护数据隐私，从而成为应对这些挑战的一种有前途的解决方案。本文介绍了FLEX：一个灵活的联邦学习框架，旨在为外语研究实验提供最大的灵活性。通过为数据分发、隐私参数和通信策略提供可定制的功能，FLEX使研究人员能够创新和开发新的FL技术。该框架还包括用于特定FL实现的库，包括：(1)异常、(2)区块链、(3)对抗性攻击和防御、(4)自然语言处理和(5)决策树，增强了其在各个领域的通用性和适用性。总体而言，FLEX代表着外语研究的重大进步，促进了强大而高效的外语应用程序的开发。



## **23. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

PeerAiD：从专业的同伴导师改善对抗蒸馏 cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.06668v2) [paper-pdf](http://arxiv.org/pdf/2403.06668v2)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.

摘要: 当神经网络应用于安全关键领域时，它的对抗健壮性是一个重要的问题。在这种情况下，对抗性蒸馏是一种很有前途的选择，它旨在提取教师网络的健壮性，以提高小型学生网络的健壮性。以前的工作预先训练教师网络，使其对针对自己的对抗性例子具有健壮性。然而，对抗性的例子取决于目标网络的参数。在对抗性提取过程中，固定的教师网络不可避免地降低了其对看不见的转移的对抗性范例的鲁棒性，这些例子针对的是学生网络的参数。我们建议PeerAiD使对等网络学习学生网络的对抗性例子，而不是针对自己的对抗性例子。PeerAiD是一种对抗性的升华，它同时训练对等网络和学生网络，使对等网络专门用于防御学生网络。我们观察到这种对等网络超过了预先训练的稳健教师网络对学生攻击的对手样本的稳健性。通过这种对等网络和对抗性蒸馏，PeerAiD实现了显著更高的学生网络的健壮性，AutoAttack(AA)准确率高达1.66%p，并使用ResNet-18和TinyImageNet数据集将学生网络的自然准确率提高到4.72%p。



## **24. Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**

Greedy-DiM：用于不合理有效面部形态的贪婪算法 cs.CV

Initial preprint. Under review

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06025v1) [paper-pdf](http://arxiv.org/pdf/2404.06025v1)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Varational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.

摘要: 变形攻击是对最先进的人脸识别(FR)系统的新威胁，该系统旨在创建包含多个身份的生物识别信息的单一图像。扩散变形(Dim)是最近提出的一种变形攻击，它已经在基于表示的变形攻击中获得了最先进的性能。然而，现有的关于DIMS的研究都没有利用DIMS的迭代性质，将DIM模型视为一个黑盒，将其视为与生成性对抗性网络(GAN)或变分自动编码器(VAE)没有区别的模型。针对DIM模型的迭代采样过程，我们提出了一种贪婪策略，在基于身份的启发式函数的指导下寻找最优步长。我们使用开源的SYN-MAD 2022竞赛数据集将我们提出的算法与其他十种最先进的变形算法进行了比较。我们发现我们提出的算法是不合理的有效的，愚弄了所有测试的FR系统，MMPMR为100%，比所有其他变形算法都要好。



## **25. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

使用Pre-Softmax评分的归因方法的一个漏洞 cs.LG

7 pages, 5 figures

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2307.03305v3) [paper-pdf](http://arxiv.org/pdf/2307.03305v3)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.

摘要: 我们讨论了一个漏洞，涉及一类属性方法用于解释卷积神经网络作为分类器的输出。众所周知，这种类型的网络容易受到对抗攻击，其中输入的不可察觉的扰动可能会改变模型的输出。相反，这里我们关注的是模型中的小修改可能对归因方法造成的影响，而不会改变模型输出。



## **26. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

基于自适应平滑的分类器精度-鲁棒性权衡 cs.LG

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2301.12554v4) [paper-pdf](http://arxiv.org/pdf/2301.12554v4)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.

摘要: 虽然先前的研究已经提出了太多的方法来构建稳健的神经分类器来对抗对手的健壮性，但实践者仍然不愿采用它们，因为它们具有不可接受的严重的干净准确性惩罚。本文通过混合标准分类器和稳健分类器的输出概率显著缓解了这种精度与稳健性的权衡，其中标准网络针对干净的精度进行了优化，而通常不是稳健的。研究表明，稳健的基分类器对正确样本和错误样本的置信度差异是这一改进的关键。除了提供直觉和经验证据外，我们还从理论上证明了混合分类器在现实假设下的稳健性。此外，我们将对抗性输入检测器引入混合网络，该混合网络自适应地调整两个基本模型的混合，从而进一步降低了实现稳健性的精度损失。这一灵活的方法被称为“自适应平滑”，可以与现有甚至未来的方法结合使用，以提高干净的准确性、健壮性或敌手检测。我们的经验评估考虑了强攻击方法，包括AutoAttack和自适应攻击。在CIFAR-100数据集上，我们的方法实现了85.21%的清洁准确率，同时保持了38.72%的$\ELL_\INFTY$-AutoAttaced($\epsilon=8/255$)精度，成为截至提交时在RobustBuchCIFAR-100基准上第二健壮的方法，同时与所有列出的模型相比，清洁准确率提高了10个百分点。实现我们方法的代码可以在https://github.com/Bai-YT/AdaptiveSmoothing.上找到



## **27. Quantum Adversarial Learning for Kernel Methods**

核方法的量子对抗学习 quant-ph

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05824v1) [paper-pdf](http://arxiv.org/pdf/2404.05824v1)

**Authors**: Giuseppe Montalbano, Leonardo Banchi

**Abstract**: We show that hybrid quantum classifiers based on quantum kernel methods and support vector machines are vulnerable against adversarial attacks, namely small engineered perturbations of the input data can deceive the classifier into predicting the wrong result. Nonetheless, we also show that simple defence strategies based on data augmentation with a few crafted perturbations can make the classifier robust against new attacks. Our results find applications in security-critical learning problems and in mitigating the effect of some forms of quantum noise, since the attacker can also be understood as part of the surrounding environment.

摘要: 我们表明，基于量子核方法和支持向量机的混合量子分类器对对抗攻击很脆弱，即输入数据的小工程扰动可以欺骗分类器预测错误的结果。尽管如此，我们也表明，简单的防御策略基于数据增强与一些精心设计的扰动可以使分类器强大的新攻击。我们的研究结果在安全关键的学习问题和减轻某些形式的量子噪声的影响方面找到了应用，因为攻击者也可以被理解为周围环境的一部分。



## **28. Case Study: Neural Network Malware Detection Verification for Feature and Image Datasets**

案例研究：特征和图像数据集的神经网络恶意软件检测验证 cs.CR

In International Conference On Formal Methods in Software  Engineering, 2024; (FormaliSE'24)

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05703v1) [paper-pdf](http://arxiv.org/pdf/2404.05703v1)

**Authors**: Preston K. Robinette, Diego Manzanas Lopez, Serena Serbinowska, Kevin Leach, Taylor T. Johnson

**Abstract**: Malware, or software designed with harmful intent, is an ever-evolving threat that can have drastic effects on both individuals and institutions. Neural network malware classification systems are key tools for combating these threats but are vulnerable to adversarial machine learning attacks. These attacks perturb input data to cause misclassification, bypassing protective systems. Existing defenses often rely on enhancing the training process, thereby increasing the model's robustness to these perturbations, which is quantified using verification. While training improvements are necessary, we propose focusing on the verification process used to evaluate improvements to training. As such, we present a case study that evaluates a novel verification domain that will help to ensure tangible safeguards against adversaries and provide a more reliable means of evaluating the robustness and effectiveness of anti-malware systems. To do so, we describe malware classification and two types of common malware datasets (feature and image datasets), demonstrate the certified robustness accuracy of malware classifiers using the Neural Network Verification (NNV) and Neural Network Enumeration (nnenum) tools, and outline the challenges and future considerations necessary for the improvement and refinement of the verification of malware classification. By evaluating this novel domain as a case study, we hope to increase its visibility, encourage further research and scrutiny, and ultimately enhance the resilience of digital systems against malicious attacks.

摘要: 恶意软件，即带有有害意图的软件，是一种不断演变的威胁，可能会对个人和机构产生严重影响。神经网络恶意软件分类系统是打击这些威胁的关键工具，但容易受到对抗性机器学习攻击。这些攻击会绕过保护系统，扰乱输入数据，导致错误分类。现有的防御通常依赖于加强训练过程，从而增加模型对这些扰动的稳健性，这是使用验证来量化的。虽然培训改进是必要的，但我们建议将重点放在用于评估培训改进情况的核查过程上。因此，我们提供了一个案例研究来评估一个新的验证域，该验证域将有助于确保针对攻击者的切实保护，并提供一种更可靠的方法来评估反恶意软件系统的健壮性和有效性。为此，我们描述了恶意软件分类和两种常见的恶意软件数据集(特征数据集和图像数据集)，使用神经网络验证(NNV)和神经网络枚举(Nnenum)工具证明了恶意软件分类器的健壮性准确性，并概述了改进和完善恶意软件分类验证所面临的挑战和未来需要考虑的问题。通过评估这一新领域作为案例研究，我们希望提高其可见度，鼓励进一步的研究和审查，并最终增强数字系统对恶意攻击的弹性。



## **29. David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**

David and Goliath：对QNN在深度边缘的攻击和防御的经验评估 cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05688v1) [paper-pdf](http://arxiv.org/pdf/2404.05688v1)

**Authors**: Miguel Costa, Sandro Pinto

**Abstract**: ML is shifting from the cloud to the edge. Edge computing reduces the surface exposing private data and enables reliable throughput guarantees in real-time applications. Of the panoply of devices deployed at the edge, resource-constrained MCUs, e.g., Arm Cortex-M, are more prevalent, orders of magnitude cheaper, and less power-hungry than application processors or GPUs. Thus, enabling intelligence at the deep edge is the zeitgeist, with researchers focusing on unveiling novel approaches to deploy ANNs on these constrained devices. Quantization is a well-established technique that has proved effective in enabling the deployment of neural networks on MCUs; however, it is still an open question to understand the robustness of QNNs in the face of adversarial examples.   To fill this gap, we empirically evaluate the effectiveness of attacks and defenses from (full-precision) ANNs on (constrained) QNNs. Our evaluation includes three QNNs targeting TinyML applications, ten attacks, and six defenses. With this study, we draw a set of interesting findings. First, quantization increases the point distance to the decision boundary and leads the gradient estimated by some attacks to explode or vanish. Second, quantization can act as a noise attenuator or amplifier, depending on the noise magnitude, and causes gradient misalignment. Regarding adversarial defenses, we conclude that input pre-processing defenses show impressive results on small perturbations; however, they fall short as the perturbation increases. At the same time, train-based defenses increase the average point distance to the decision boundary, which holds after quantization. However, we argue that train-based defenses still need to smooth the quantization-shift and gradient misalignment phenomenons to counteract adversarial example transferability to QNNs. All artifacts are open-sourced to enable independent validation of results.

摘要: ML正在从云端转移到边缘。边缘计算减少了暴露私有数据的表面，并在实时应用中实现了可靠的吞吐量保证。在部署在边缘的所有设备中，资源受限的MCU(例如ARM Cortex-M)比应用处理器或GPU更普遍、更便宜、耗电量更低。因此，在深层实现智能是时代的精神，研究人员专注于推出在这些受限设备上部署ANN的新方法。量化是一种成熟的技术，已被证明在MCU上部署神经网络是有效的；然而，面对敌对例子，理解QNN的稳健性仍然是一个悬而未决的问题。为了填补这一空白，我们从经验上评估了(全精度)人工神经网络对(受约束的)QNN的攻击和防御的有效性。我们的评估包括三个针对TinyML应用程序的QNN，十个攻击和六个防御。通过这项研究，我们得出了一系列有趣的发现。首先，量化增加了到决策边界的点距离，并导致某些攻击估计的梯度爆炸或消失。其次，量化可以充当噪声衰减器或放大器，这取决于噪声的大小，并导致梯度失调。对于对抗性防御，我们得出的结论是，输入预处理防御在小扰动下表现出令人印象深刻的结果；然而，随着扰动的增加，它们不能满足要求。同时，基于训练的防御增加了到决策边界的平均点距离，量化后该距离保持不变。然而，我们认为，基于训练的防御仍然需要平滑量化位移和梯度错位现象，以抵消向QNN的对抗性示例转移。所有构件都是开源的，以支持结果的独立验证。



## **30. Investigating the Impact of Quantization on Adversarial Robustness**

研究量化对对抗鲁棒性的影响 cs.LG

Accepted to ICLR 2024 Workshop PML4LRS

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05639v1) [paper-pdf](http://arxiv.org/pdf/2404.05639v1)

**Authors**: Qun Li, Yuan Meng, Chen Tang, Jiacheng Jiang, Zhi Wang

**Abstract**: Quantization is a promising technique for reducing the bit-width of deep models to improve their runtime performance and storage efficiency, and thus becomes a fundamental step for deployment. In real-world scenarios, quantized models are often faced with adversarial attacks which cause the model to make incorrect inferences by introducing slight perturbations. However, recent studies have paid less attention to the impact of quantization on the model robustness. More surprisingly, existing studies on this topic even present inconsistent conclusions, which prompted our in-depth investigation. In this paper, we conduct a first-time analysis of the impact of the quantization pipeline components that can incorporate robust optimization under the settings of Post-Training Quantization and Quantization-Aware Training. Through our detailed analysis, we discovered that this inconsistency arises from the use of different pipelines in different studies, specifically regarding whether robust optimization is performed and at which quantization stage it occurs. Our research findings contribute insights into deploying more secure and robust quantized networks, assisting practitioners in reference for scenarios with high-security requirements and limited resources.

摘要: 量化是一种很有前途的技术，可以减少深度模型的位宽，从而提高其运行时性能和存储效率，因此成为部署的基础步骤。在现实场景中，量化模型经常面临敌意攻击，通过引入微小的扰动，导致模型做出不正确的推断。然而，最近的研究较少关注量化对模型稳健性的影响。更令人惊讶的是，现有的研究甚至得出了不一致的结论，这促使我们进行了深入的调查。在本文中，我们首次分析了在训练后量化和量化感知训练的情况下，能够结合稳健优化的量化流水线组件的影响。通过我们的详细分析，我们发现这种不一致是由于在不同的研究中使用了不同的流水线，特别是关于是否进行了稳健优化以及它发生在哪个量化阶段。我们的研究成果有助于深入了解如何部署更安全、更强大的量化网络，帮助实践者在具有高安全性要求和资源有限的场景中进行参考。



## **31. SoK: Gradient Leakage in Federated Learning**

SoK：联邦学习中的梯度泄漏 cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05403v1) [paper-pdf](http://arxiv.org/pdf/2404.05403v1)

**Authors**: Jiacheng Du, Jiahui Hu, Zhibo Wang, Peng Sun, Neil Zhenqiang Gong, Kui Ren

**Abstract**: Federated learning (FL) enables collaborative model training among multiple clients without raw data exposure. However, recent studies have shown that clients' private training data can be reconstructed from the gradients they share in FL, known as gradient inversion attacks (GIAs). While GIAs have demonstrated effectiveness under \emph{ideal settings and auxiliary assumptions}, their actual efficacy against \emph{practical FL systems} remains under-explored. To address this gap, we conduct a comprehensive study on GIAs in this work. We start with a survey of GIAs that establishes a milestone to trace their evolution and develops a systematization to uncover their inherent threats. Specifically, we categorize the auxiliary assumptions used by existing GIAs based on their practical accessibility to potential adversaries. To facilitate deeper analysis, we highlight the challenges that GIAs face in practical FL systems from three perspectives: \textit{local training}, \textit{model}, and \textit{post-processing}. We then perform extensive theoretical and empirical evaluations of state-of-the-art GIAs across diverse settings, utilizing eight datasets and thirteen models. Our findings indicate that GIAs have inherent limitations when reconstructing data under practical local training settings. Furthermore, their efficacy is sensitive to the trained model, and even simple post-processing measures applied to gradients can be effective defenses. Overall, our work provides crucial insights into the limited effectiveness of GIAs in practical FL systems. By rectifying prior misconceptions, we hope to inspire more accurate and realistic investigations on this topic.

摘要: 联合学习(FL)实现了多个客户之间的协作模型培训，而不会暴露原始数据。然而，最近的研究表明，客户的私人训练数据可以从他们在FL中共享的梯度重建，称为梯度反转攻击(GIA)。虽然GIA已经在理想环境和辅助假设下证明了其有效性，但它们对实际FL系统的实际有效性仍未得到充分研究。为了弥补这一差距，我们在这项工作中对GIA进行了全面的研究。我们从对GIA的调查开始，建立了一个里程碑来跟踪它们的演变，并制定了一个系统化的方法来揭示它们的内在威胁。具体地说，我们根据现有GIA对潜在对手的实际可访问性对其使用的辅助假设进行分类。为了便于更深入的分析，我们从三个角度强调了GIA在实际外语系统中所面临的挑战：\textit{本地训练}、\textit{模型}和\textit{后处理}。然后，我们利用8个数据集和13个模型，在不同的环境中对最先进的GIA进行广泛的理论和经验评估。我们的发现表明，在实际的本地训练环境下，GIA在重建数据时存在固有的局限性。此外，它们的有效性对训练的模型很敏感，甚至对梯度应用简单的后处理措施也可以成为有效的防御措施。总体而言，我们的工作对GIA在实际外语系统中的有限有效性提供了至关重要的见解。通过纠正之前的误解，我们希望启发对这一主题的更准确和更现实的调查。



## **32. BruSLeAttack: A Query-Efficient Score-Based Black-Box Sparse Adversarial Attack**

BruSLeAttack：一种基于分数的查询高效黑盒稀疏对抗攻击 cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2024). Code is available at  https://brusliattack.github.io/

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05311v1) [paper-pdf](http://arxiv.org/pdf/2404.05311v1)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: We study the unique, less-well understood problem of generating sparse adversarial samples simply by observing the score-based replies to model queries. Sparse attacks aim to discover a minimum number-the l0 bounded-perturbations to model inputs to craft adversarial examples and misguide model decisions. But, in contrast to query-based dense attack counterparts against black-box models, constructing sparse adversarial perturbations, even when models serve confidence score information to queries in a score-based setting, is non-trivial. Because, such an attack leads to i) an NP-hard problem; and ii) a non-differentiable search space. We develop the BruSLeAttack-a new, faster (more query-efficient) Bayesian algorithm for the problem. We conduct extensive attack evaluations including an attack demonstration against a Machine Learning as a Service (MLaaS) offering exemplified by Google Cloud Vision and robustness testing of adversarial training regimes and a recent defense against black-box attacks. The proposed attack scales to achieve state-of-the-art attack success rates and query efficiency on standard computer vision tasks such as ImageNet across different model architectures. Our artefacts and DIY attack samples are available on GitHub. Importantly, our work facilitates faster evaluation of model vulnerabilities and raises our vigilance on the safety, security and reliability of deployed systems.

摘要: 我们简单地通过观察对模型查询的基于分数的回复来研究生成稀疏对抗性样本的独特的、较少被理解的问题。稀疏攻击的目的是发现最小数量的--10个有界的--扰动，以对输入进行建模，以制造敌意的例子并误导模型决策。但是，与基于查询的密集攻击对应的黑盒模型相比，构建稀疏的对抗性扰动，即使当模型在基于分数的设置中向查询提供置信度分数信息时，也不是微不足道的。因为，这样的攻击导致i)NP-Hard问题；以及ii)不可微搜索空间。我们开发了BruSLeAttack-一种新的、更快(查询效率更高)的贝叶斯算法。我们进行广泛的攻击评估，包括针对机器学习即服务(MLaaS)产品的攻击演示，例如Google Cloud Vision和对抗性训练机制的健壮性测试，以及最近针对黑盒攻击的防御。建议的攻击规模可跨不同的模型架构在标准计算机视觉任务(如ImageNet)上实现最先进的攻击成功率和查询效率。我们的手工艺品和DIY攻击样本可以在GitHub上找到。重要的是，我们的工作有助于更快地评估模型漏洞，并提高我们对已部署系统的安全性、安全性和可靠性的警惕。



## **33. Out-of-Distribution Data: An Acquaintance of Adversarial Examples -- A Survey**

非分布数据：对抗性实例的认识--一项调查 cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05219v1) [paper-pdf](http://arxiv.org/pdf/2404.05219v1)

**Authors**: Naveen Karunanayake, Ravin Gunawardena, Suranga Seneviratne, Sanjay Chawla

**Abstract**: Deep neural networks (DNNs) deployed in real-world applications can encounter out-of-distribution (OOD) data and adversarial examples. These represent distinct forms of distributional shifts that can significantly impact DNNs' reliability and robustness. Traditionally, research has addressed OOD detection and adversarial robustness as separate challenges. This survey focuses on the intersection of these two areas, examining how the research community has investigated them together. Consequently, we identify two key research directions: robust OOD detection and unified robustness. Robust OOD detection aims to differentiate between in-distribution (ID) data and OOD data, even when they are adversarially manipulated to deceive the OOD detector. Unified robustness seeks a single approach to make DNNs robust against both adversarial attacks and OOD inputs. Accordingly, first, we establish a taxonomy based on the concept of distributional shifts. This framework clarifies how robust OOD detection and unified robustness relate to other research areas addressing distributional shifts, such as OOD detection, open set recognition, and anomaly detection. Subsequently, we review existing work on robust OOD detection and unified robustness. Finally, we highlight the limitations of the existing work and propose promising research directions that explore adversarial and OOD inputs within a unified framework.

摘要: 在实际应用中部署的深度神经网络(DNN)可能会遇到分布外(OOD)数据和敌意示例。这些代表了不同形式的分布变化，可以显著影响DNN的可靠性和稳健性。传统上，研究将OOD检测和对手健壮性作为单独的挑战。这项调查聚焦于这两个领域的交集，考察了研究界是如何一起调查这两个领域的。因此，我们确定了两个关键的研究方向：稳健的面向对象检测和统一的稳健性。稳健的OOD检测旨在区分分布内(ID)数据和OOD数据，即使它们被相反地操纵以欺骗OOD检测器。统一健壮性寻求一种单一的方法来使DNN对对手攻击和OOD输入都具有健壮性。因此，首先，我们建立了基于分布移位概念的分类。该框架阐明了健壮的OOD检测和统一的健壮性如何与解决分布迁移的其他研究领域相关，例如OOD检测、开集识别和异常检测。随后，我们回顾了健壮性面向对象检测和统一健壮性方面的现有工作。最后，我们强调了现有工作的局限性，并提出了在统一框架内探索对抗性和OOD输入的有前途的研究方向。



## **34. Semantic Stealth: Adversarial Text Attacks on NLP Using Several Methods**

语义隐身：基于几种方法的NLP对抗性文本攻击 cs.CL

This report pertains to the Capstone Project done by Group 2 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 28 pages and it includes 10 tables. This is the preprint  which will be submitted to IEEE CONIT 2024 for review

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05159v1) [paper-pdf](http://arxiv.org/pdf/2404.05159v1)

**Authors**: Roopkatha Dey, Aivy Debnath, Sayak Kumar Dutta, Kaustav Ghosh, Arijit Mitra, Arghya Roy Chowdhury, Jaydip Sen

**Abstract**: In various real-world applications such as machine translation, sentiment analysis, and question answering, a pivotal role is played by NLP models, facilitating efficient communication and decision-making processes in domains ranging from healthcare to finance. However, a significant challenge is posed to the robustness of these natural language processing models by text adversarial attacks. These attacks involve the deliberate manipulation of input text to mislead the predictions of the model while maintaining human interpretability. Despite the remarkable performance achieved by state-of-the-art models like BERT in various natural language processing tasks, they are found to remain vulnerable to adversarial perturbations in the input text. In addressing the vulnerability of text classifiers to adversarial attacks, three distinct attack mechanisms are explored in this paper using the victim model BERT: BERT-on-BERT attack, PWWS attack, and Fraud Bargain's Attack (FBA). Leveraging the IMDB, AG News, and SST2 datasets, a thorough comparative analysis is conducted to assess the effectiveness of these attacks on the BERT classifier model. It is revealed by the analysis that PWWS emerges as the most potent adversary, consistently outperforming other methods across multiple evaluation scenarios, thereby emphasizing its efficacy in generating adversarial examples for text classification. Through comprehensive experimentation, the performance of these attacks is assessed and the findings indicate that the PWWS attack outperforms others, demonstrating lower runtime, higher accuracy, and favorable semantic similarity scores. The key insight of this paper lies in the assessment of the relative performances of three prevalent state-of-the-art attack mechanisms.

摘要: 在机器翻译、情感分析和问题回答等各种现实应用中，自然语言处理模型扮演着至关重要的角色，促进了从医疗保健到金融等领域的高效沟通和决策过程。然而，文本对抗攻击对这些自然语言处理模型的稳健性提出了重大挑战。这些攻击包括故意操纵输入文本以误导模型的预测，同时保持人类的可解释性。尽管像BERT这样的最先进的模型在各种自然语言处理任务中取得了显著的性能，但它们仍然容易受到输入文本中的对抗性干扰。针对文本分类器易受敌意攻击的问题，利用受害者模型BERT，探讨了三种不同的攻击机制：BERT-ON-BERT攻击、PWWS攻击和欺诈交易攻击(FBA)。利用IMDB、AG News和Sst2数据集，进行了全面的比较分析，以评估这些攻击对BERT分类器模型的有效性。分析表明，PWWS是最强大的对手，在多个评估场景中的表现一直优于其他方法，从而强调了它在生成用于文本分类的对抗性实例方面的有效性。通过综合实验，评估了这些攻击的性能，结果表明，PWWS攻击的性能优于其他攻击，表现出更低的运行时间、更高的准确率和良好的语义相似度得分。本文的重点在于对目前流行的三种攻击机制的相对性能进行评估。



## **35. Enabling Privacy-Preserving Cyber Threat Detection with Federated Learning**

利用联邦学习实现隐私保护网络威胁检测 cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05130v1) [paper-pdf](http://arxiv.org/pdf/2404.05130v1)

**Authors**: Yu Bi, Yekai Li, Xuan Feng, Xianghang Mi

**Abstract**: Despite achieving good performance and wide adoption, machine learning based security detection models (e.g., malware classifiers) are subject to concept drift and evasive evolution of attackers, which renders up-to-date threat data as a necessity. However, due to enforcement of various privacy protection regulations (e.g., GDPR), it is becoming increasingly challenging or even prohibitive for security vendors to collect individual-relevant and privacy-sensitive threat datasets, e.g., SMS spam/non-spam messages from mobile devices. To address such obstacles, this study systematically profiles the (in)feasibility of federated learning for privacy-preserving cyber threat detection in terms of effectiveness, byzantine resilience, and efficiency. This is made possible by the build-up of multiple threat datasets and threat detection models, and more importantly, the design of realistic and security-specific experiments.   We evaluate FL on two representative threat detection tasks, namely SMS spam detection and Android malware detection. It shows that FL-trained detection models can achieve a performance that is comparable to centrally trained counterparts. Also, most non-IID data distributions have either minor or negligible impact on the model performance, while a label-based non-IID distribution of a high extent can incur non-negligible fluctuation and delay in FL training. Then, under a realistic threat model, FL turns out to be adversary-resistant to attacks of both data poisoning and model poisoning. Particularly, the attacking impact of a practical data poisoning attack is no more than 0.14\% loss in model accuracy. Regarding FL efficiency, a bootstrapping strategy turns out to be effective to mitigate the training delay as observed in label-based non-IID scenarios.

摘要: 尽管获得了良好的性能和广泛的采用，基于机器学习的安全检测模型(例如恶意软件分类器)仍然受到攻击者概念漂移和回避演变的影响，这使得最新的威胁数据成为必要。然而，由于各种隐私保护法规的执行(例如，GDPR)，安全供应商收集与个人相关和隐私敏感的威胁数据集，例如来自移动设备的短信垃圾邮件/非垃圾邮件，正变得越来越具有挑战性，甚至令人望而却步。为了解决这些障碍，本研究从有效性、拜占庭复原力和效率三个方面系统地描述了联合学习用于隐私保护网络威胁检测的可行性。这是由于建立了多个威胁数据集和威胁检测模型，更重要的是，设计了现实的和特定于安全的实验。我们在两个有代表性的威胁检测任务上对FL进行了评估，即短信垃圾邮件检测和Android恶意软件检测。这表明，FL训练的检测模型可以获得与中央训练的同类模型相当的性能。此外，大多数非IID数据分布对模型性能的影响很小或可以忽略不计，而高范围的基于标签的非IID分布可能会在FL训练中引起不可忽略的波动和延迟。然后，在一个真实的威胁模型下，FL对数据中毒和模型中毒的攻击都具有抵抗能力。特别是，实际的数据中毒攻击对模型精度的影响不超过0.14。关于外语学习的效率，事实证明，在基于标签的非IID场景中，自举策略被证明是有效的减轻训练延迟。



## **36. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

隐藏在普通视野中：对脆弱患者人群的不可检测的对抗偏见攻击 cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.05713v3) [paper-pdf](http://arxiv.org/pdf/2402.05713v3)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.

摘要: 人工智能(AI)在放射学中的扩散揭示了深度学习(DL)模型的风险，加剧了对脆弱患者群体的临床偏见。虽然以前的文献集中于量化训练的DL模型所表现出的偏差，但针对人口统计目标的对DL模型的对抗性偏见攻击及其在临床环境中的应用仍然是医学成像领域中探索不足的研究领域。在这项工作中，我们证明了人口统计目标的标签中毒攻击可以在DL模型中引入不可检测的漏诊偏差。我们在多个性能指标和人口统计组(如性别、年龄及其相交的子组)上的结果表明，对抗性偏见攻击通过降低组模型性能而不影响整体模型性能，显示了对目标组中的偏见的高选择性。此外，我们的结果表明，对抗性偏差攻击导致有偏差的DL模型传播预测偏差，即使在使用外部数据集进行评估时也是如此。



## **37. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroIDBench：一个基于脑波的认证研究方法标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.08656v3) [paper-pdf](http://arxiv.org/pdf/2402.08656v3)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroIDBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroIDB边来研究文献中提出的浅层分类器和基于深度学习的方法，并测试多个会话的健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroIDBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法学实践进一步推进基于脑电波的身份验证。



## **38. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱陷阱可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2311.08268v4) [paper-pdf](http://arxiv.org/pdf/2311.08268v4)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **39. Provable Robustness Against a Union of $\ell_0$ Adversarial Attacks**

针对$\ell_0 $对抗攻击联盟的可证明鲁棒性 cs.LG

Accepted at AAAI 2024 -- Extended version including the supplementary  material

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2302.11628v4) [paper-pdf](http://arxiv.org/pdf/2302.11628v4)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Sparse or $\ell_0$ adversarial attacks arbitrarily perturb an unknown subset of the features. $\ell_0$ robustness analysis is particularly well-suited for heterogeneous (tabular) data where features have different types or scales. State-of-the-art $\ell_0$ certified defenses are based on randomized smoothing and apply to evasion attacks only. This paper proposes feature partition aggregation (FPA) -- a certified defense against the union of $\ell_0$ evasion, backdoor, and poisoning attacks. FPA generates its stronger robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Compared to state-of-the-art $\ell_0$ defenses, FPA is up to 3,000${\times}$ faster and provides larger median robustness guarantees (e.g., median certificates of 13 pixels over 10 for CIFAR10, 12 pixels over 10 for MNIST, 4 features over 1 for Weather, and 3 features over 1 for Ames), meaning FPA provides the additional dimensions of robustness essentially for free.

摘要: 稀疏或$\ell_0 $对抗攻击任意干扰特征的未知子集。$\ell_0$鲁棒性分析特别适合于特征具有不同类型或规模的异构（表格）数据。最先进的$\ell_0$认证防御基于随机平滑，仅适用于规避攻击。本文提出了特征分区聚合（FPA）--一种针对$\ell_0 $规避、后门和中毒攻击的认证防御方法。FPA通过一个子模型在不相交特征集上训练的集成来产生更强的鲁棒性保证。与最先进的$\ell_0$防御相比，FPA速度最高可达3，000 ${\times}$，并提供更大的中值鲁棒性保证（例如，CIFAR 10的中值证书为13个像素超过10，MNIST的中值证书为12个像素超过10，Weather的4个特征超过1，Ames的3个特征超过1），这意味着FPA基本上免费提供了额外的鲁棒性维度。



## **40. Data Poisoning Attacks on Off-Policy Policy Evaluation Methods**

非策略策略评估方法的数据中毒攻击 cs.LG

Accepted at UAI 2022

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04714v1) [paper-pdf](http://arxiv.org/pdf/2404.04714v1)

**Authors**: Elita Lobo, Harvineet Singh, Marek Petrik, Cynthia Rudin, Himabindu Lakkaraju

**Abstract**: Off-policy Evaluation (OPE) methods are a crucial tool for evaluating policies in high-stakes domains such as healthcare, where exploration is often infeasible, unethical, or expensive. However, the extent to which such methods can be trusted under adversarial threats to data quality is largely unexplored. In this work, we make the first attempt at investigating the sensitivity of OPE methods to marginal adversarial perturbations to the data. We design a generic data poisoning attack framework leveraging influence functions from robust statistics to carefully construct perturbations that maximize error in the policy value estimates. We carry out extensive experimentation with multiple healthcare and control datasets. Our results demonstrate that many existing OPE methods are highly prone to generating value estimates with large errors when subject to data poisoning attacks, even for small adversarial perturbations. These findings question the reliability of policy values derived using OPE methods and motivate the need for developing OPE methods that are statistically robust to train-time data poisoning attacks.

摘要: 非政策评估(OPE)方法是评估高风险领域(如医疗保健)政策的重要工具，这些领域的探索通常是不可行、不道德或昂贵的。然而，在数据质量受到敌对威胁的情况下，这种方法在多大程度上可以得到信任，这在很大程度上是未知的。在这项工作中，我们首次尝试研究OPE方法对数据的边缘对抗性扰动的敏感性。我们设计了一个通用的数据中毒攻击框架，利用稳健统计中的影响函数来仔细构造扰动，使策略值估计的误差最大化。我们对多个医疗保健和对照数据集进行了广泛的实验。我们的结果表明，许多现有的OPE方法在受到数据中毒攻击时，即使是在小的对抗性扰动下，也很容易产生误差较大的值估计。这些发现质疑使用OPE方法得出的策略值的可靠性，并促使人们需要开发在统计上对训练时间数据中毒攻击具有健壮性的OPE方法。



## **41. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队游戏：红色团队语言模型的游戏理论框架 cs.CL

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2310.00322v4) [paper-pdf](http://arxiv.org/pdf/2310.00322v4)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **42. CANEDERLI: On The Impact of Adversarial Training and Transferability on CAN Intrusion Detection Systems**

CANEDERLI：对抗训练和传输性对CAN入侵检测系统的影响 cs.CR

Accepted at WiseML 2024

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04648v1) [paper-pdf](http://arxiv.org/pdf/2404.04648v1)

**Authors**: Francesco Marchiori, Mauro Conti

**Abstract**: The growing integration of vehicles with external networks has led to a surge in attacks targeting their Controller Area Network (CAN) internal bus. As a countermeasure, various Intrusion Detection Systems (IDSs) have been suggested in the literature to prevent and mitigate these threats. With the increasing volume of data facilitated by the integration of Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication networks, most of these systems rely on data-driven approaches such as Machine Learning (ML) and Deep Learning (DL) models. However, these systems are susceptible to adversarial evasion attacks. While many researchers have explored this vulnerability, their studies often involve unrealistic assumptions, lack consideration for a realistic threat model, and fail to provide effective solutions.   In this paper, we present CANEDERLI (CAN Evasion Detection ResiLIence), a novel framework for securing CAN-based IDSs. Our system considers a realistic threat model and addresses the impact of adversarial attacks on DL-based detection systems. Our findings highlight strong transferability properties among diverse attack methodologies by considering multiple state-of-the-art attacks and model architectures. We analyze the impact of adversarial training in addressing this threat and propose an adaptive online adversarial training technique outclassing traditional fine-tuning methodologies with F1 scores up to 0.941. By making our framework publicly available, we aid practitioners and researchers in assessing the resilience of IDSs to a varied adversarial landscape.

摘要: 车辆与外部网络的日益集成导致了针对其控制器区域网络(CAN)内部总线的攻击激增。作为一种对策，文献中提出了各种入侵检测系统(入侵检测系统)来预防和缓解这些威胁。随着车辆到车辆(V2V)和车辆到基础设施(V2I)通信网络的集成促进了数据量的增加，这些系统中的大多数依赖于数据驱动的方法，如机器学习(ML)和深度学习(DL)模型。然而，这些系统容易受到对抗性逃避攻击。虽然许多研究人员探索了这一漏洞，但他们的研究往往涉及不切实际的假设，缺乏对现实威胁模型的考虑，无法提供有效的解决方案。本文提出了一种新的基于CAN的入侵检测系统安全框架CANEDERLI(CAN Elevation Detect Resilience)。我们的系统考虑了一个现实的威胁模型，并解决了对抗性攻击对基于DL的检测系统的影响。通过考虑多个最先进的攻击和模型体系结构，我们的研究结果突出了不同攻击方法之间的强大可转移性。我们分析了对抗训练在应对这一威胁方面的影响，并提出了一种自适应在线对抗训练技术，其F1得分高达0.941分，超过了传统的微调方法。通过将我们的框架公之于众，我们帮助从业者和研究人员评估入侵检测系统对不同对手环境的韧性。



## **43. Dynamic Graph Information Bottleneck**

动态图形信息瓶颈 cs.LG

Accepted by the research tracks of The Web Conference 2024 (WWW 2024)

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2402.06716v3) [paper-pdf](http://arxiv.org/pdf/2402.06716v3)

**Authors**: Haonan Yuan, Qingyun Sun, Xingcheng Fu, Cheng Ji, Jianxin Li

**Abstract**: Dynamic Graphs widely exist in the real world, which carry complicated spatial and temporal feature patterns, challenging their representation learning. Dynamic Graph Neural Networks (DGNNs) have shown impressive predictive abilities by exploiting the intrinsic dynamics. However, DGNNs exhibit limited robustness, prone to adversarial attacks. This paper presents the novel Dynamic Graph Information Bottleneck (DGIB) framework to learn robust and discriminative representations. Leveraged by the Information Bottleneck (IB) principle, we first propose the expected optimal representations should satisfy the Minimal-Sufficient-Consensual (MSC) Condition. To compress redundant as well as conserve meritorious information into latent representation, DGIB iteratively directs and refines the structural and feature information flow passing through graph snapshots. To meet the MSC Condition, we decompose the overall IB objectives into DGIB$_{MS}$ and DGIB$_C$, in which the DGIB$_{MS}$ channel aims to learn the minimal and sufficient representations, with the DGIB$_{MS}$ channel guarantees the predictive consensus. Extensive experiments on real-world and synthetic dynamic graph datasets demonstrate the superior robustness of DGIB against adversarial attacks compared with state-of-the-art baselines in the link prediction task. To the best of our knowledge, DGIB is the first work to learn robust representations of dynamic graphs grounded in the information-theoretic IB principle.

摘要: 动态图形广泛存在于现实世界中，携带着复杂的时空特征模式，对其表示学习提出了挑战。动态图神经网络(DGNN)利用其内在的动力学特性，表现出了令人印象深刻的预测能力。然而，DGNN表现出有限的健壮性，容易受到对抗性攻击。本文提出了一种新的动态图信息瓶颈(DGIB)框架来学习稳健和区分的表示。利用信息瓶颈(IB)原理，我们首先提出期望的最优表示应该满足最小-充分-一致(MSC)条件。为了压缩冗余信息并将有价值的信息保存到潜在表示中，DGIB迭代地指导和细化通过图快照传递的结构和特征信息流。为了满足MSC条件，我们将整体IB目标分解为DGIB${MS}$和DGIB$_C$，其中DGIB${MS}$通道旨在学习最小且充分的表示，而DGIB${MS}$通道保证预测共识。在真实世界和合成动态图数据集上的大量实验表明，与链接预测任务中的最新基线相比，DGIB对对手攻击具有更好的稳健性。据我们所知，DGIB是第一个基于信息论IB原理学习动态图的稳健表示的工作。



## **44. Goal-guided Generative Prompt Injection Attack on Large Language Models**

面向大型语言模型的目标引导生成式提示注入攻击 cs.CR

22 pages, 8 figures

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.07234v1) [paper-pdf](http://arxiv.org/pdf/2404.07234v1)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **45. Recovery from Adversarial Attacks in Cyber-physical Systems: Shallow, Deep and Exploratory Works**

网络物理系统中对抗性攻击的恢复：浅、深和探索性工作 eess.SY

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04472v1) [paper-pdf](http://arxiv.org/pdf/2404.04472v1)

**Authors**: Pengyuan Lu, Lin Zhang, Mengyu Liu, Kaustubh Sridhar, Fanxin Kong, Oleg Sokolsky, Insup Lee

**Abstract**: Cyber-physical systems (CPS) have experienced rapid growth in recent decades. However, like any other computer-based systems, malicious attacks evolve mutually, driving CPS to undesirable physical states and potentially causing catastrophes. Although the current state-of-the-art is well aware of this issue, the majority of researchers have not focused on CPS recovery, the procedure we defined as restoring a CPS's physical state back to a target condition under adversarial attacks. To call for attention on CPS recovery and identify existing efforts, we have surveyed a total of 30 relevant papers. We identify a major partition of the proposed recovery strategies: shallow recovery vs. deep recovery, where the former does not use a dedicated recovery controller while the latter does. Additionally, we surveyed exploratory research on topics that facilitate recovery. From these publications, we discuss the current state-of-the-art of CPS recovery, with respect to applications, attack type, attack surfaces and system dynamics. Then, we identify untouched sub-domains in this field and suggest possible future directions for researchers.

摘要: 近几十年来，网络物理系统(CP)经历了快速增长。然而，与任何其他基于计算机的系统一样，恶意攻击也会相互演化，将CP推向不受欢迎的物理状态，并可能造成灾难。虽然目前的研究水平已经很好地意识到了这一问题，但大多数研究人员并没有关注CPS的恢复，我们定义的CPS恢复过程是指在对手攻击下将CPS的物理状态恢复到目标状态。为了唤起人们对CPS恢复的关注，并确定现有的努力，我们总共调查了30篇相关论文。我们确定了建议的恢复策略的主要部分：浅恢复和深度恢复，前者不使用专用恢复控制器，而后者使用。此外，我们对促进恢复的主题进行了探索性研究。从这些出版物中，我们从应用程序、攻击类型、攻击面和系统动力学方面讨论了CPS恢复的最新技术。然后，我们确定了该领域中未触及的子域，并为研究人员提出了可能的未来方向。



## **46. Increased LLM Vulnerabilities from Fine-tuning and Quantization**

从微调和量化增加LLM漏洞 cs.CR

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04392v1) [paper-pdf](http://arxiv.org/pdf/2404.04392v1)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have become very popular and have found use cases in many domains, such as chatbots, auto-task completion agents, and much more. However, LLMs are vulnerable to different types of attacks, such as jailbreaking, prompt injection attacks, and privacy leakage attacks. Foundational LLMs undergo adversarial and alignment training to learn not to generate malicious and toxic content. For specialized use cases, these foundational LLMs are subjected to fine-tuning or quantization for better performance and efficiency. We examine the impact of downstream tasks such as fine-tuning and quantization on LLM vulnerability. We test foundation models like Mistral, Llama, MosaicML, and their fine-tuned versions. Our research shows that fine-tuning and quantization reduces jailbreak resistance significantly, leading to increased LLM vulnerabilities. Finally, we demonstrate the utility of external guardrails in reducing LLM vulnerabilities.

摘要: 大型语言模型（LLM）已经变得非常流行，并在许多领域找到了用例，如聊天机器人，自动任务完成代理等等。然而，LLM容易受到不同类型的攻击，例如越狱、即时注入攻击和隐私泄漏攻击。基础LLM接受对抗和对齐培训，学习不生成恶意和有毒内容。对于特殊的用例，这些基本的LLM需要经过微调或量化，以获得更好的性能和效率。我们研究了下游任务的影响，如微调和量化LLM脆弱性。我们测试了Mistral、Llama、MosaicML等基础模型，以及它们的微调版本。我们的研究表明，微调和量化显著降低了越狱阻力，导致LLM漏洞增加。最后，我们演示了外部防护措施在减少LLM漏洞方面的效用。



## **47. Compositional Estimation of Lipschitz Constants for Deep Neural Networks**

深度神经网络Lipschitz常数的合成估计 cs.LG

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04375v1) [paper-pdf](http://arxiv.org/pdf/2404.04375v1)

**Authors**: Yuezhu Xu, S. Sivaranjani

**Abstract**: The Lipschitz constant plays a crucial role in certifying the robustness of neural networks to input perturbations and adversarial attacks, as well as the stability and safety of systems with neural network controllers. Therefore, estimation of tight bounds on the Lipschitz constant of neural networks is a well-studied topic. However, typical approaches involve solving a large matrix verification problem, the computational cost of which grows significantly for deeper networks. In this letter, we provide a compositional approach to estimate Lipschitz constants for deep feedforward neural networks by obtaining an exact decomposition of the large matrix verification problem into smaller sub-problems. We further obtain a closed-form solution that applies to most common neural network activation functions, which will enable rapid robustness and stability certificates for neural networks deployed in online control settings. Finally, we demonstrate through numerical experiments that our approach provides a steep reduction in computation time while yielding Lipschitz bounds that are very close to those achieved by state-of-the-art approaches.

摘要: Lipschitz常数在证明神经网络对输入扰动和敌意攻击的稳健性以及具有神经网络控制器的系统的稳定性和安全性方面起着至关重要的作用。因此，神经网络Lipschitz常数的紧界估计是一个很好的研究课题。然而，典型的方法涉及解决大型矩阵验证问题，对于更深层次的网络，其计算成本显著增加。在这封信中，我们提供了一种组合方法来估计深度前馈神经网络的Lipschitz常数，方法是将大的矩阵验证问题精确地分解成更小的子问题。我们进一步得到了适用于大多数常见神经网络激活函数的封闭形式的解，这将为在线控制环境中部署的神经网络提供快速的健壮性和稳定性证书。最后，我们通过数值实验证明，我们的方法大大减少了计算时间，而得到的Lipschitz界与最先进的方法所达到的非常接近。



## **48. Dissecting Distribution Inference**

剖析分布推理 cs.LG

Accepted at SaTML 2023 (updated Yifu's email address)

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2212.07591v2) [paper-pdf](http://arxiv.org/pdf/2212.07591v2)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference

摘要: 分布推断攻击旨在推断用于训练机器学习模型的数据的统计特性。这些攻击有时威力惊人，但影响分布推断风险的因素并未得到很好的理解，已证明的攻击往往依赖于强大而不切实际的假设，例如完全了解训练环境，即使在假设的黑箱威胁场景中也是如此。为了提高对分布推断风险的理解，我们开发了一种新的黑盒攻击，该攻击在大多数情况下甚至比最著名的白盒攻击性能更好。使用这种新的攻击，我们评估了分布推断风险，同时放松了在黑盒访问下关于对手知识的各种假设，如已知的模型体系结构和仅标签访问。最后，我们评估了以前提出的防御措施的有效性，并引入了新的防御措施。我们发现，尽管基于噪声的防御似乎无效，但简单的重新采样防御可以非常有效。代码可在https://github.com/iamgroot42/dissecting_distribution_inference上找到



## **49. Evaluating Adversarial Robustness: A Comparison Of FGSM, Carlini-Wagner Attacks, And The Role of Distillation as Defense Mechanism**

评估对抗鲁棒性：FGSM、Carlini-Wagner攻击的比较以及蒸馏作为防御机制的作用 cs.CR

This report pertains to the Capstone Project done by Group 1 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 35 pages and it includes 15 figures and 10 tables. This  is the preprint which will be submitted to to an IEEE international  conference for review

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04245v1) [paper-pdf](http://arxiv.org/pdf/2404.04245v1)

**Authors**: Trilokesh Ranjan Sarkar, Nilanjan Das, Pralay Sankar Maitra, Bijoy Some, Ritwik Saha, Orijita Adhikary, Bishal Bose, Jaydip Sen

**Abstract**: This technical report delves into an in-depth exploration of adversarial attacks specifically targeted at Deep Neural Networks (DNNs) utilized for image classification. The study also investigates defense mechanisms aimed at bolstering the robustness of machine learning models. The research focuses on comprehending the ramifications of two prominent attack methodologies: the Fast Gradient Sign Method (FGSM) and the Carlini-Wagner (CW) approach. These attacks are examined concerning three pre-trained image classifiers: Resnext50_32x4d, DenseNet-201, and VGG-19, utilizing the Tiny-ImageNet dataset. Furthermore, the study proposes the robustness of defensive distillation as a defense mechanism to counter FGSM and CW attacks. This defense mechanism is evaluated using the CIFAR-10 dataset, where CNN models, specifically resnet101 and Resnext50_32x4d, serve as the teacher and student models, respectively. The proposed defensive distillation model exhibits effectiveness in thwarting attacks such as FGSM. However, it is noted to remain susceptible to more sophisticated techniques like the CW attack. The document presents a meticulous validation of the proposed scheme. It provides detailed and comprehensive results, elucidating the efficacy and limitations of the defense mechanisms employed. Through rigorous experimentation and analysis, the study offers insights into the dynamics of adversarial attacks on DNNs, as well as the effectiveness of defensive strategies in mitigating their impact.

摘要: 这份技术报告深入探讨了专门针对用于图像分类的深度神经网络(DNN)的对抗性攻击。该研究还调查了旨在增强机器学习模型稳健性的防御机制。研究重点在于理解两种主要的攻击方法：快速梯度符号法(FGSM)和卡里尼-瓦格纳(CW)方法。这些攻击涉及三个预先训练的图像分类器：Resnext50_32x4d、DenseNet-201和VGG-19，使用Tiny-ImageNet数据集。此外，研究还提出了防御蒸馏作为对抗FGSM和CW攻击的一种防御机制的健壮性。这一防御机制使用CIFAR-10数据集进行了评估，其中CNN模型，特别是resnet101和Resnext50_32x4d分别充当教师模型和学生模型。所提出的防御蒸馏模型在挫败FGSM等攻击方面表现出了有效性。然而，值得注意的是，它仍然容易受到更复杂的技术的影响，比如CW攻击。该文件对提议的方案进行了细致的验证。它提供了详细和全面的结果，阐明了所采用的防御机制的有效性和局限性。通过严格的实验和分析，这项研究提供了对DNN的对抗性攻击的动态以及防御策略在减轻其影响方面的有效性的见解。



## **50. On Inherent Adversarial Robustness of Active Vision Systems**

主动视觉系统的固有对抗鲁棒性 cs.CV

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.00185v2) [paper-pdf](http://arxiv.org/pdf/2404.00185v2)

**Authors**: Amitangshu Mukherjee, Timur Ibrayev, Kaushik Roy

**Abstract**: Current Deep Neural Networks are vulnerable to adversarial examples, which alter their predictions by adding carefully crafted noise. Since human eyes are robust to such inputs, it is possible that the vulnerability stems from the standard way of processing inputs in one shot by processing every pixel with the same importance. In contrast, neuroscience suggests that the human vision system can differentiate salient features by (1) switching between multiple fixation points (saccades) and (2) processing the surrounding with a non-uniform external resolution (foveation). In this work, we advocate that the integration of such active vision mechanisms into current deep learning systems can offer robustness benefits. Specifically, we empirically demonstrate the inherent robustness of two active vision methods - GFNet and FALcon - under a black box threat model. By learning and inferencing based on downsampled glimpses obtained from multiple distinct fixation points within an input, we show that these active methods achieve (2-3) times greater robustness compared to a standard passive convolutional network under state-of-the-art adversarial attacks. More importantly, we provide illustrative and interpretable visualization analysis that demonstrates how performing inference from distinct fixation points makes active vision methods less vulnerable to malicious inputs.

摘要: 当前的深度神经网络很容易受到敌意例子的影响，这些例子通过添加精心设计的噪声来改变它们的预测。由于人眼对这样的输入很健壮，这种漏洞可能源于一次处理输入的标准方式，即处理具有相同重要性的每个像素。相比之下，神经科学表明，人类的视觉系统可以通过(1)在多个注视点(眼跳)之间切换和(2)用非均匀的外部分辨率(中心凹)处理周围环境来区分显著特征。在这项工作中，我们主张将这种主动视觉机制集成到当前的深度学习系统中，可以提供健壮性优势。具体而言，我们通过实验验证了两种主动视觉方法--GFNet和Falcon--在黑匣子威胁模型下的内在稳健性。通过基于从输入内多个不同固定点获得的下采样一瞥的学习和推理，我们表明这些主动方法在最先进的对抗攻击下比标准的被动卷积网络获得(2-3)倍的健壮性。更重要的是，我们提供了说明性和可解释性的可视化分析，演示了如何从不同的注视点执行推理使主动视觉方法不太容易受到恶意输入的影响。



