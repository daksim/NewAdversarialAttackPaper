# Latest Adversarial Attack Papers
**update at 2025-02-14 10:56:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On the Importance of Backbone to the Adversarial Robustness of Object Detectors**

论主干对对象检测器对抗鲁棒性的重要性 cs.CV

Accepted by IEEE TIFS

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2305.17438v2) [paper-pdf](http://arxiv.org/pdf/2305.17438v2)

**Authors**: Xiao Li, Hang Chen, Xiaolin Hu

**Abstract**: Object detection is a critical component of various security-sensitive applications, such as autonomous driving and video surveillance. However, existing object detectors are vulnerable to adversarial attacks, which poses a significant challenge to their reliability and security. Through experiments, first, we found that existing works on improving the adversarial robustness of object detectors give a false sense of security. Second, we found that adversarially pre-trained backbone networks were essential for enhancing the adversarial robustness of object detectors. We then proposed a simple yet effective recipe for fast adversarial fine-tuning on object detectors with adversarially pre-trained backbones. Without any modifications to the structure of object detectors, our recipe achieved significantly better adversarial robustness than previous works. Finally, we explored the potential of different modern object detector designs for improving adversarial robustness with our recipe and demonstrated interesting findings, which inspired us to design state-of-the-art (SOTA) robust detectors. Our empirical results set a new milestone for adversarially robust object detection. Code and trained checkpoints are available at https://github.com/thu-ml/oddefense.

摘要: 目标检测是各种安全敏感应用的关键组件，例如自动驾驶和视频监控。然而，现有的目标探测器容易受到敌意攻击，这对其可靠性和安全性构成了巨大的挑战。通过实验，首先，我们发现现有的提高目标检测器对抗健壮性的工作给人一种错误的安全感。其次，我们发现对抗性预训练的骨干网络对于增强目标检测器的对抗性稳健性是必不可少的。然后，我们提出了一个简单但有效的配方，用于在具有对抗性预培训主干的对象探测器上进行快速对抗性微调。在不对对象检测器的结构进行任何修改的情况下，我们的配方比以前的工作获得了更好的对抗健壮性。最后，我们探索了不同的现代对象检测器设计在提高对手健壮性方面的潜力，并展示了有趣的发现，这启发了我们设计最先进的(SOTA)健壮性检测器。我们的实验结果为反向稳健的目标检测建立了一个新的里程碑。代码和训练有素的检查点可在https://github.com/thu-ml/oddefense.上找到



## **2. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

ADBM：用于可靠对抗净化的对抗扩散桥模型 cs.LG

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.00315v3) [paper-pdf](http://arxiv.org/pdf/2408.00315v3)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.

摘要: 最近，基于扩散的纯化（DiffPure）被认为是针对对抗性例子的有效防御方法。然而，我们发现直接使用原始预训练的扩散模型进行对抗性纯化的迪夫Pure是次优的。这是由于噪音净化性能和数据恢复质量之间固有的权衡。此外，现有的DistPure评估的可靠性值得怀疑，因为它们依赖于弱适应性攻击。在这项工作中，我们提出了一种新型的对抗扩散桥模型，称为ADBM。ADBM直接构建了从扩散的对抗数据到其原始干净示例的反向桥梁，增强了原始扩散模型的净化能力。通过各种场景的理论分析和实验验证，ADBM已被证明是一种卓越且强大的防御机制，为实际应用提供了巨大的前景。



## **3. Wasserstein distributional adversarial training for deep neural networks**

深度神经网络的Wasserstein分布式对抗训练 cs.LG

15 pages, 4 figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09352v1) [paper-pdf](http://arxiv.org/pdf/2502.09352v1)

**Authors**: Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

**Abstract**: Design of adversarial attacks for deep neural networks, as well as methods of adversarial training against them, are subject of intense research. In this paper, we propose methods to train against distributional attack threats, extending the TRADES method used for pointwise attacks. Our approach leverages recent contributions and relies on sensitivity analysis for Wasserstein distributionally robust optimization problems. We introduce an efficient fine-tuning method which can be deployed on a previously trained model. We test our methods on a range of pre-trained models on RobustBench. These experimental results demonstrate the additional training enhances Wasserstein distributional robustness, while maintaining original levels of pointwise robustness, even for already very successful networks. The improvements are less marked for models pre-trained using huge synthetic datasets of 20-100M images. However, remarkably, sometimes our methods are still able to improve their performance even when trained using only the original training dataset (50k images).

摘要: 深度神经网络的对抗性攻击的设计，以及针对它们的对抗性训练方法，都是深入研究的主题。在本文中，我们提出了针对分布式攻击威胁的训练方法，扩展了用于点式攻击的TRADS方法。我们的方法利用了最近的贡献，并依赖于对Wasserstein分布稳健优化问题的敏感度分析。我们介绍了一种高效的微调方法，该方法可以部署在先前训练的模型上。我们在一系列预先训练好的模型上对我们的方法进行了测试。这些实验结果表明，额外的训练增强了Wasserstein分布的健壮性，同时保持了原始的逐点健壮性，即使对于已经非常成功的网络也是如此。对于使用2000万至1亿张图像的大型合成数据集进行预训练的模型，改进效果不那么明显。然而，值得注意的是，有时我们的方法仍然能够提高它们的性能，即使只使用原始训练数据集(50k图像)进行训练。



## **4. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

LiSA：利用链接推荐通过子图注入攻击图神经网络 cs.LG

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09271v1) [paper-pdf](http://arxiv.org/pdf/2502.09271v1)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.

摘要: 图神经网络(GNN)在用图结构建模数据方面表现出了卓越的能力，但最近的研究表明，它们对对手攻击很敏感。传统的攻击方法依赖于操纵原始图形或添加到人工创建的节点的链接，在现实世界中往往被证明是不切实际的。在GNN系统中，引入一个孤立的子图来欺骗链接推荐器和节点分类器，提出了一种新的对抗性场景。具体地说，链接推荐器被误导提出目标受害节点与子图之间的链接，鼓励用户无意中建立连接，这将降低节点分类的准确性，从而促进攻击的成功。为了解决这一问题，我们提出了LISA框架，该框架采用双重代理模型和双层优化来同时满足两个对抗性目标。在真实数据集上的大量实验证明了该方法的有效性。



## **5. FLAME: Flexible LLM-Assisted Moderation Engine**

FLAME：灵活的LLM辅助审核引擎 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.

摘要: 大型语言模型(LLM)的快速发展给协调用户与模型的交互带来了巨大的挑战。虽然LLM显示出非凡的能力，但它们仍然容易受到对抗性攻击，特别是绕过内容安全措施的“越狱”技术。目前的内容审核系统主要依赖于输入提示过滤，已被证明是不够的，像N中最佳(Bon)越狱技术对流行的LLM的成功率达到80%或更高。在本文中，我们介绍了灵活的LLM辅助调节引擎(FLAME)：一种将焦点从输入过滤转移到输出调节的新方法。与分析用户查询的传统断路方法不同，FLAME评估模型响应，提供了几个关键优势：(1)训练和推理的计算效率，(2)增强了对Bon越狱攻击的抵抗，以及(3)通过可定制的主题过滤灵活地定义和更新安全标准。我们的实验表明，火焰系统的性能明显优于现有的慢化系统。例如，FLAME将GPT-40-mini和DeepSeek-v3中的攻击成功率降低了~9倍，同时保持了较低的计算开销。我们对各种LLM进行了综合评估，并针对最先进的越狱情况分析了发动机的效率。这项工作有助于开发更健壮和适应性更强的LLMS内容审核系统。



## **6. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

拉开帷幕：通过对比辅助网络的无监督对抗检测 cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09110v1) [paper-pdf](http://arxiv.org/pdf/2502.09110v1)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.

摘要: 深度学习模型广泛应用于安全关键应用中，但仍然容易受到对抗性攻击--可能会显著降低模型性能的不可察觉的扰动。传统的防御机制主要集中在增强模型的稳健性或独立检测敌方输入。在这项工作中，我们提出了一种基于对比辅助网络的无监督敌意检测方法(U-CAN)，以发现辅助特征表示中的对抗性行为，而不需要对抗性实例。U-CAN被嵌入到目标模型的选定中间层中。这些辅助网络包括投影层和基于ArcFace的线性层，改进了特征表示，以更有效地区分良性输入和敌意输入。在多个数据集(CIFAR-10、哺乳动物和ImageNet的子集)和体系结构(ResNet-50、VGG-16和VIT)上的综合实验表明，我们的方法超过了现有的无监督对手检测技术，在四种不同的攻击方法上获得了优越的F1分数。该框架为提高深度学习系统的安全性和可靠性提供了一种可扩展的有效解决方案。



## **7. Universal Adversarial Attack on Aligned Multimodal LLMs**

对对齐多模式LLM的普遍对抗攻击 cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.

摘要: 我们提出了一种针对多模式大型语言模型(LLMS)的通用对抗性攻击，该攻击利用单个优化图像覆盖跨不同查询甚至多个模型的对齐保障。通过视觉编码器和语言头部的反向传播，我们制作了一个合成图像，迫使模型使用有针对性的短语(例如，“当然，就是这里”)或其他不安全的内容做出响应--即使是有害的提示。在SafeBtch基准测试上的实验中，我们的方法获得了比现有基线显著更高的攻击成功率，包括纯文本通用提示(例如，在某些型号上高达93%)。我们通过同时在多个多模式LLM上进行训练和在看不见的体系结构上进行测试来进一步证明跨模型的可转移性。此外，我们的方法的一个多答案变体会产生听起来更自然(但仍然是恶意的)响应。这些发现突显了当前多模式联合的严重弱点，并呼吁进行更强大的对抗性防御。我们将在APACHE-2.0许可下发布代码和数据集。警告：本文中的多模式LLMS生成的某些内容可能会冒犯某些读者。



## **8. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

RL SA-PFL：隐私保护联邦学习中具有模型不一致性检测的鲁棒轻量级安全聚合 cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08989v1) [paper-pdf](http://arxiv.org/pdf/2502.08989v1)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.

摘要: 联合学习(FL)允许用户通过只共享本地模型来协作训练全局机器学习模型，而不会将他们的私有数据暴露给中央服务器。这种分布式学习在数据隐私至关重要的场景中特别有吸引力，它引起了工业界和学术界的大量关注。然而，研究揭示了FL中的隐私漏洞，攻击者可能会从共享的模型参数中推断敏感信息。本文提出了一种高效的基于掩码的安全聚合方案，该方案利用轻量级的密码原语来降低隐私风险。与现有方法相比，我们的方案有几个优点。首先，它只需要整个FL培训课程的单一设置阶段，大大减少了通信开销。其次，它利用中间服务器层和轻量级密钥协商方法，消除了用户到用户交互的需要，从而最大限度地减少了用户端开销。第三，该方案对用户退出具有很强的弹性，用户可以在任何FL轮加入。第四，它可以检测和防御恶意服务器活动，包括最近发现的模型不一致攻击。最后，我们的方案确保了半诚实和恶意环境下的安全性。我们提供安全分析来正式证明我们方法的健壮性。此外，我们还实现了我们方案的端到端原型。我们进行了全面的实验和比较，结果表明，它在通信和计算开销、功能和安全性方面都优于现有的解决方案。



## **9. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。代码发布地址为：https://github.com/jianshuod/Engorgio-prompt.



## **10. Siren Song: Manipulating Pose Estimation in XR Headsets Using Acoustic Attacks**

Siren Song：使用声学攻击操纵XR耳机中的姿势估计 cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08865v1) [paper-pdf](http://arxiv.org/pdf/2502.08865v1)

**Authors**: Zijian Huang, Yicheng Zhang, Sophie Chen, Nael Abu-Ghazaleh, Jiasi Chen

**Abstract**: Extended Reality (XR) experiences involve interactions between users, the real world, and virtual content. A key step to enable these experiences is the XR headset sensing and estimating the user's pose in order to accurately place and render virtual content in the real world. XR headsets use multiple sensors (e.g., cameras, inertial measurement unit) to perform pose estimation and improve its robustness, but this provides an attack surface for adversaries to interfere with the pose estimation process. In this paper, we create and study the effects of acoustic attacks that create false signals in the inertial measurement unit (IMU) on XR headsets, leading to adverse downstream effects on XR applications. We generate resonant acoustic signals on a HoloLens 2 and measure the resulting perturbations in the IMU readings, and also demonstrate both fine-grained and coarse attacks on the popular ORB-SLAM3 and an open-source XR system (ILLIXR). With the knowledge gleaned from attacking these open-source frameworks, we demonstrate four end-to-end proof-of-concept attacks on a HoloLens 2: manipulating user input, clickjacking, zone invasion, and denial of user interaction. Our experiments show that current commercial XR headsets are susceptible to acoustic attacks, raising concerns for their security.

摘要: 扩展现实(XR)体验涉及用户、真实世界和虚拟内容之间的交互。实现这些体验的关键一步是XR耳机感知和估计用户的姿势，以便在现实世界中准确放置和呈现虚拟内容。XR耳机使用多个传感器(如摄像头、惯性测量单元)来执行位姿估计并提高其稳健性，但这为对手提供了一个干扰位姿估计过程的攻击面。在本文中，我们创建和研究了在XR耳机上的惯性测量单元(IMU)中产生错误信号的声学攻击的影响，从而导致XR应用的不利下游影响。我们在HoloLens 2上生成共振声信号，并测量IMU读数中产生的扰动，还演示了对流行的Orb-SLAM3和开放源代码XR系统(ILLIXR)的细粒度和粗粒度攻击。利用从攻击这些开源框架中获得的知识，我们演示了针对HoloLens 2的四种端到端概念验证攻击：操纵用户输入、点击劫持、区域入侵和拒绝用户交互。我们的实验表明，目前的商用XR耳机容易受到声学攻击，这引发了人们对其安全性的担忧。



## **11. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

ASVspoof 5：使用众包语音设计、收集和验证用于欺骗、Deepfake和对抗性攻击检测的资源 eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.08857v1) [paper-pdf](http://arxiv.org/pdf/2502.08857v1)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.

摘要: ASVspoof5是一系列挑战中的第五版，这些挑战促进了对语音欺骗和深度假冒攻击的研究以及检测解决方案的设计。我们介绍了ASVspoof5数据库，它是以众包方式从不同声学条件下收集的数据生成的(参见。较早ASVspoof数据库的演播室质量数据)和来自约2,000名演讲者的数据(参见~100早)。该数据库包含32种不同的算法产生的攻击，也是众包的，并使用新的代理检测模型在不同程度上进行了优化。其中包括使用传统和当代文本到语音合成和语音转换模型的混合生成的攻击，以及首次纳入的对抗性攻击。ASVspoof 5协议包括七个说话人不相交的分区。它们包括用于训练不同攻击模型集的两个不同的分区，用于开发和评估代理检测模型的另外两个分区，以及组成ASVspoof5训练、开发和评估集的另外三个分区。从另外30,000个说话者收集的辅助数据集也可用于训练说话人编码者以实现攻击算法。这里还描述了使用一组自动说话人验证和欺骗/深度伪基线检测器对新的ASVspoof5数据库进行的实验验证。除了用于生成欺骗/深度假语音的协议和工具外，本文描述的资源现在都可以免费向社区提供，这些资源已经被2024年ASVspoof 5挑战赛的参与者使用。



## **12. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

LLM水印的理论基础框架：分布自适应方法 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.

摘要: 数字水印已经成为区分人工智能生成的文本和人类创建的文本的关键方法。在本文中，我们提出了一种新的大语言模型(LLMS)水印理论框架，该框架同时优化了水印方案和检测过程。我们的方法专注于最大化检测性能，同时保持对最坏情况下的类型I错误和文本失真的控制。我们将其刻画在水印可检测性和文本失真之间的基本权衡。重要的是，我们发现最优水印方案对LLM生成分布是自适应的。基于我们的理论见解，我们提出了一种高效的、与模型无关的、分布自适应的水印算法，该算法利用代理模型和Gumbel-max技巧。在Llama2-13B和Mistral-8$\x$70亿模型上进行的实验证实了该方法的有效性。此外，我们还研究了将健壮性融入到我们的框架中，为未来更有效地抵御对手攻击的水印系统铺平了道路。



## **13. Bankrupting DoS Attackers**

破产的拒绝服务攻击者 cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2205.08287v4) [paper-pdf](http://arxiv.org/pdf/2205.08287v4)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstract**: Can we make a denial-of-service attacker pay more than the server and honest clients? Consider a model where a server sees a stream of jobs sent by either honest clients or an adversary. The server sets a price for servicing each job with the aid of an estimator, which provides approximate statistical information about the distribution of previously occurring good jobs.   We describe and analyze pricing algorithms for the server under different models of synchrony, with total cost parameterized by the accuracy of the estimator. Given a reasonably accurate estimator, the algorithm's cost provably grows more slowly than the attacker's cost, as the attacker's cost grows large. Additionally, we prove a lower bound, showing that our pricing algorithm yields asymptotically tight results when the estimator is accurate within constant factors.

摘要: 我们能否让拒绝服务攻击者支付比服务器和诚实客户更多的费用？考虑一个模型，其中服务器看到诚实的客户或对手发送的作业流。服务器在估计器的帮助下为每个作业设定服务的价格，该估计器提供有关之前发生的好作业分布的大致统计信息。   我们描述和分析了不同同步模型下服务器的定价算法，总成本由估计器的准确性参数化。给定一个相当准确的估计器，可以证明，随着攻击者的成本变得很大，算法的成本增长速度比攻击者的成本慢。此外，我们证明了一个下界，表明当估计量在恒定因子内准确时，我们的定价算法会产生渐进紧的结果。



## **14. Extreme vulnerability to intruder attacks destabilizes network dynamics**

对入侵者攻击的极端脆弱性会破坏网络动态的稳定 nlin.AO

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08552v1) [paper-pdf](http://arxiv.org/pdf/2502.08552v1)

**Authors**: Amirhossein Nazerian, Sahand Tangerami, Malbor Asllani, David Phillips, Hernan Makse, Francesco Sorrentino

**Abstract**: Consensus, synchronization, formation control, and power grid balance are all examples of virtuous dynamical states that may arise in networks. Here we focus on how such states can be destabilized from a fundamental perspective; namely, we address the question of how one or a few intruder agents within an otherwise functioning network may compromise its dynamics. We show that a single adversarial node coupled via adversarial couplings to one or more other nodes is sufficient to destabilize the entire network, which we prove to be more efficient than targeting multiple nodes. Then, we show that concentrating the attack on a single low-indegree node induces the greatest instability, challenging the common assumption that hubs are the most critical nodes. This leads to a new characterization of the vulnerability of a node, which contrasts with previous work, and identifies low-indegree nodes (as opposed to the hubs) as the most vulnerable components of a network. Our results are derived for linear systems but hold true for nonlinear networks, including those described by the Kuramoto model. Finally, we derive scaling laws showing that larger networks are less susceptible, on average, to single-node attacks. Overall, these findings highlight an intrinsic vulnerability of technological systems such as autonomous networks, sensor networks, power grids, and the internet of things, which also extend to the realm of complex social and biological networks.

摘要: 共识、同步、队形控制和电网平衡都是网络中可能出现的良性动态状态的例子。在这里，我们从根本的角度关注如何破坏这种状态的稳定；也就是，我们解决了一个或几个入侵者代理在其他功能正常的网络中如何可能危及其动态的问题。我们证明了单个敌意节点通过对抗性耦合耦合到一个或多个其他节点足以破坏整个网络的稳定，我们证明了这比针对多个节点更有效。然后，我们证明了将攻击集中在单个低度节点上会导致最大的不稳定性，挑战了集线器是最关键节点的普遍假设。这导致了对节点脆弱性的新的表征，这与以前的工作形成了对比，并将低索引度节点(而不是集线器)识别为网络中最脆弱的组件。我们的结果适用于线性系统，但也适用于非线性网络，包括用Kuramoto模型描述的网络。最后，我们推导出了标度律，表明较大的网络平均而言不太容易受到单节点攻击。总体而言，这些发现突显了自主网络、传感器网络、电网和物联网等技术系统的内在脆弱性，这些系统也延伸到复杂的社会和生物网络领域。



## **15. The Impact of Logic Locking on Confidentiality: An Automated Evaluation**

逻辑锁定对保密性的影响：自动评估 cs.CR

8 pages, accepted at 26th International Symposium on Quality  Electronic Design (ISQED'25)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.01240v2) [paper-pdf](http://arxiv.org/pdf/2502.01240v2)

**Authors**: Lennart M. Reimann, Evgenii Rezunov, Dominik Germek, Luca Collini, Christian Pilato, Ramesh Karri, Rainer Leupers

**Abstract**: Logic locking secures hardware designs in untrusted foundries by incorporating key-driven gates to obscure the original blueprint. While this method safeguards the integrated circuit from malicious alterations during fabrication, its influence on data confidentiality during runtime has been ignored. In this study, we employ path sensitization to formally examine the impact of logic locking on confidentiality. By applying three representative logic locking mechanisms on open-source cryptographic benchmarks, we utilize an automatic test pattern generation framework to evaluate the effect of locking on cryptographic encryption keys and sensitive data signals. Our analysis reveals that logic locking can inadvertently cause sensitive data leakage when incorrect logic locking keys are used. We show that a single malicious logic locking key can expose over 70% of an encryption key. If an adversary gains control over other inputs, the entire encryption key can be compromised. This research uncovers a significant security vulnerability in logic locking and emphasizes the need for comprehensive security assessments that extend beyond key-recovery attacks.

摘要: 逻辑锁定通过结合钥匙驱动的门来模糊原始蓝图，从而保护不可信铸造厂中的硬件设计。虽然这种方法保护集成电路在制造过程中不受恶意更改，但它对运行时数据保密性的影响被忽略。在这项研究中，我们使用路径敏感化来形式化地检查逻辑锁定对机密性的影响。通过将三种典型的逻辑锁定机制应用于开源密码基准测试，我们利用一个自动测试模式生成框架来评估锁定对加密密钥和敏感数据信号的影响。我们的分析表明，当使用了错误的逻辑锁密钥时，逻辑锁可能会无意中导致敏感数据泄漏。我们发现，一个恶意的逻辑锁密钥可以暴露超过70%的加密密钥。如果对手获得了对其他输入的控制，则整个加密密钥可能会被泄露。这项研究揭示了逻辑锁定中的一个重大安全漏洞，并强调需要进行全面的安全评估，而不仅仅是密钥恢复攻击。



## **16. AdvSwap: Covert Adversarial Perturbation with High Frequency Info-swapping for Autonomous Driving Perception**

AdvSwap：隐性对抗扰动，通过高频信息交换实现自动驾驶感知 cs.CV

27th IEEE International Conference on Intelligent Transportation  Systems (ITSC)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08374v1) [paper-pdf](http://arxiv.org/pdf/2502.08374v1)

**Authors**: Yuanhao Huang, Qinfan Zhang, Jiandong Xing, Mengyue Cheng, Haiyang Yu, Yilong Ren, Xiao Xiong

**Abstract**: Perception module of Autonomous vehicles (AVs) are increasingly susceptible to be attacked, which exploit vulnerabilities in neural networks through adversarial inputs, thereby compromising the AI safety. Some researches focus on creating covert adversarial samples, but existing global noise techniques are detectable and difficult to deceive the human visual system. This paper introduces a novel adversarial attack method, AdvSwap, which creatively utilizes wavelet-based high-frequency information swapping to generate covert adversarial samples and fool the camera. AdvSwap employs invertible neural network for selective high-frequency information swapping, preserving both forward propagation and data integrity. The scheme effectively removes the original label data and incorporates the guidance image data, producing concealed and robust adversarial samples. Experimental evaluations and comparisons on the GTSRB and nuScenes datasets demonstrate that AdvSwap can make concealed attacks on common traffic targets. The generates adversarial samples are also difficult to perceive by humans and algorithms. Meanwhile, the method has strong attacking robustness and attacking transferability.

摘要: 自主车辆的感知模块越来越容易受到攻击，它通过敌意输入利用神经网络的脆弱性，从而危及人工智能的安全性。一些研究侧重于创建隐蔽的对抗性样本，但现有的全局噪声技术是可以检测的，很难欺骗人类的视觉系统。介绍了一种新颖的敌意攻击方法AdvSwp，它创造性地利用基于小波的高频信息交换来生成隐蔽的对抗性样本并愚弄摄像机。AdvSwp使用可逆神经网络进行选择性高频信息交换，同时保持前向传播和数据完整性。该方案有效地去除了原始的标签数据，并加入了制导图像数据，产生了隐藏的、健壮的对抗样本。在GTSRB和nuScenes数据集上的实验评估和比较表明，AdvSwp可以对常见的交通目标进行隐蔽攻击。生成的敌意样本也很难被人类和算法感知到。同时，该方法具有较强的攻击健壮性和攻击可转移性。



## **17. TASAR: Transfer-based Attack on Skeletal Action Recognition**

TASSAR：基于传输的对Skelty动作识别的攻击 cs.CV

Accepted in ICLR 2025

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.02483v5) [paper-pdf](http://arxiv.org/pdf/2409.02483v5)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xiaoshuai Hao, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequence data, as a widely employed representation of human actions, are crucial in Human Activity Recognition (HAR). Recently, adversarial attacks have been proposed in this area, which exposes potential security concerns, and more importantly provides a good tool for model robustness test. Within this research, transfer-based attack is an important tool as it mimics the real-world scenario where an attacker has no knowledge of the target model, but is under-explored in Skeleton-based HAR (S-HAR). Consequently, existing S-HAR attacks exhibit weak adversarial transferability and the reason remains largely unknown. In this paper, we investigate this phenomenon via the characterization of the loss function. We find that one prominent indicator of poor transferability is the low smoothness of the loss function. Led by this observation, we improve the transferability by properly smoothening the loss when computing the adversarial examples. This leads to the first Transfer-based Attack on Skeletal Action Recognition, TASAR. TASAR explores the smoothened model posterior of pre-trained surrogates, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike existing transfer-based methods which overlook the temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack, effectively disrupting the spatial-temporal coherence of S-HARs. For exhaustive evaluation, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense models. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the https://github.com/yunfengdiao/Skeleton-Robustness-Benchmark.

摘要: 骨骼序列数据作为人类行为的广泛表征，在人类活动识别(HAR)中起着至关重要的作用。近年来，该领域出现了对抗性攻击，暴露了潜在的安全隐患，更重要的是为模型稳健性测试提供了一个很好的工具。在这项研究中，基于传输的攻击是一个重要的工具，因为它模拟了攻击者不知道目标模型的真实场景，但在基于骨架的HAR(S-HAR)中没有得到充分的探索。因此，现有的S-哈尔进攻表现出较弱的对抗性转移能力，其原因在很大程度上尚不清楚。在本文中，我们通过对损失函数的刻画来研究这一现象。我们发现，可转移性差的一个显著指标是损失函数的低光滑性。在此基础上，我们在计算对抗性实例时，通过适当地平滑损失来提高可转移性。这导致了对骨骼动作识别的第一次基于转移的攻击，Tasar。Tasar提出了一种新的训练后双贝叶斯优化策略，实现了训练前的后验平滑模型。此外，与现有基于传输的方法忽略了序列内部的时间一致性不同，Tasar将运动动力学融入到贝叶斯攻击中，有效地破坏了S-HARS的时空一致性。为了进行详尽的评估，我们构建了第一个大规模稳健的S-HAR基准，包括7个S-HAR模型，10种攻击方法，3个S-HAR数据集和2个防御模型。广泛的结果证明了Tasar的优越性。我们的基准测试可以轻松地与https://github.com/yunfengdiao/Skeleton-Robustness-Benchmark.中提供的代码进行比较，以便将来进行研究



## **18. RIDA: A Robust Attack Framework on Incomplete Graphs**

RIDA：一个针对不完整图的稳健攻击框架 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2407.18170v3) [paper-pdf](http://arxiv.org/pdf/2407.18170v3)

**Authors**: Jianke Yu, Hanchen Wang, Chen Chen, Xiaoyang Wang, Lu Qin, Wenjie Zhang, Ying Zhang, Xijuan Liu

**Abstract**: Graph Neural Networks (GNNs) are vital in data science but are increasingly susceptible to adversarial attacks. To help researchers develop more robust GNN models, it's essential to focus on designing strong attack models as foundational benchmarks and guiding references. Among adversarial attacks, gray-box poisoning attacks are noteworthy due to their effectiveness and fewer constraints. These attacks exploit GNNs' need for retraining on updated data, thereby impacting their performance by perturbing these datasets. However, current research overlooks the real-world scenario of incomplete graphs. To address this gap, we introduce the Robust Incomplete Deep Attack Framework (RIDA). It is the first algorithm for robust gray-box poisoning attacks on incomplete graphs. The approach innovatively aggregates distant vertex information and ensures powerful data utilization. Extensive tests against 9 SOTA baselines on 3 real-world datasets demonstrate that RIDA's superiority in handling incompleteness and high attack performance on the incomplete graph.

摘要: 图神经网络(GNN)在数据科学中至关重要，但越来越容易受到对手攻击。为了帮助研究人员开发更健壮的GNN模型，有必要将重点放在设计强大的攻击模型作为基础基准和指导参考。在对抗性攻击中，灰箱中毒攻击由于其有效性和较少的约束而值得注意。这些攻击利用了GNN对更新数据进行再培训的需要，从而通过扰乱这些数据集来影响其性能。然而，目前的研究忽略了现实世界中不完整图形的场景。为了弥补这一差距，我们引入了健壮的不完全深度攻击框架(RIDA)。这是第一个针对不完备图的稳健灰盒中毒攻击的算法。该方法创新性地聚合了距离较远的顶点信息，并确保了强大的数据利用率。在3个真实世界数据集上对9个SOTA基线进行的广泛测试表明，RIDA在处理不完全图和在不完整图上的高攻击性能方面具有优势。



## **19. Time-based GNSS attack detection**

基于时间的全球导航卫星系统攻击检测 cs.CR

IEEE Transactions on Aerospace and Electronic Systems (Early Access)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.03868v2) [paper-pdf](http://arxiv.org/pdf/2502.03868v2)

**Authors**: Marco Spanghero, Panos Papadimitratos

**Abstract**: To safeguard Civilian Global Navigation Satellite Systems (GNSS) external information available to the platform encompassing the GNSS receiver can be used to detect attacks. Cross-checking the GNSS-provided time against alternative multiple trusted time sources can lead to attack detection aiming at controlling the GNSS receiver time. Leveraging external, network-connected secure time providers and onboard clock references, we achieve detection even under fine-grained time attacks. We provide an extensive evaluation of our multi-layered defense against adversaries mounting attacks against the GNSS receiver along with controlling the network link. We implement adversaries spanning from simplistic spoofers to advanced ones synchronized with the GNSS constellation. We demonstrate attack detection is possible in all tested cases (sharp discontinuity, smooth take-over, and coordinated network manipulation) without changes to the structure of the GNSS receiver. Leveraging the diversity of the reference time sources, detection of take-over time push as low as 150us is possible. Smooth take-overs forcing variations as low as 30ns are also detected based on on-board precision oscillators. The method (and thus the evaluation) is largely agnostic to the satellite constellation and the attacker type, making time-based data validation of GNSS information compatible with existing receivers and readily deployable.

摘要: 为保障民用全球导航卫星系统(全球导航卫星系统)的安全，包括全球导航卫星系统接收器在内的平台可获得的外部信息可用于检测攻击。将GNSS提供的时间与备选的多个可信时间源进行交叉检查，可以导致旨在控制GNSS接收器时间的攻击检测。利用外部、网络连接的安全时间提供程序和板载时钟参考，我们即使在细粒度的时间攻击下也能实现检测。我们提供了对我们的多层防御的广泛评估，以抵御对GNSS接收器发起攻击的对手以及控制网络链路的对手。我们实现了从简单的欺骗者到与GNSS星座同步的高级欺骗者的对手。我们演示了在所有测试情况下(急剧中断、平稳接管和协调网络操作)都可以进行攻击检测，而不需要改变GNSS接收器的结构。利用基准时间源的多样性，可以检测到低至150us的接管时间推进。基于机载精密振荡器，还可以检测到低至30 ns的平稳接管强迫变化。该方法(以及评估)在很大程度上与卫星星座和攻击者类型无关，使全球导航卫星系统信息的基于时间的数据验证与现有接收器兼容，并易于部署。



## **20. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.20002v3) [paper-pdf](http://arxiv.org/pdf/2409.20002v3)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型的广泛应用对其推理性能的优化提出了强烈的需求。目前用于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，而在很大程度上忽略了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道来自共享缓存和GPU内存分配，可以用来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了在传统计算系统中观察到的安全挑战，突显出迫切需要解决LLM服务基础设施中潜在的信息泄漏问题。在本文中，我们报告了一种新的攻击策略，旨在利用LLM部署中固有的计时侧通道，特别是针对广泛用于提高LLM推理性能的键值(KV)缓存和语义缓存。我们的方法利用计时测量和分类模型来检测缓存命中，允许对手以高精度推断私人提示。我们还提出了一种逐令牌搜索算法来高效地恢复缓存中的共享提示前缀，证明了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑盒测试的实验研究表明，这种隐私风险是完全现实的，具有显著的后果。我们的发现强调了强有力的缓解措施的必要性，以保护LLM系统免受此类新出现的威胁。



## **21. Safety at Scale: A Comprehensive Survey of Large Model Safety**

大规模安全性：大型车型安全性全面调查 cs.CR

47 pages, 3 figures, 11 tables GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.05206v2) [paper-pdf](http://arxiv.org/pdf/2502.05206v2)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.

摘要: 大型模型的快速发展，受到其通过大规模预训练而具有的非凡学习和泛化能力的推动，重塑了人工智能(AI)的版图。这些模型现在是广泛应用的基础，包括对话式人工智能、推荐系统、自动驾驶、内容生成、医疗诊断和科学发现。然而，它们的广泛部署也使它们面临重大的安全风险，引发了人们对健壮性、可靠性和道德影响的担忧。本调查系统地回顾了当前关于大模型的安全研究，包括视觉基础模型(VFM)、大语言模型(LLMS)、视觉语言预训练(VLP)模型、视觉语言模型(VLMS)、扩散模型(DM)和基于大模型的代理。我们的工作总结如下：(1)对这些模型的安全威胁进行了全面的分类，包括对抗性攻击、数据中毒、后门攻击、越狱和快速注入攻击、能量延迟攻击、数据和模型提取攻击以及新出现的特定于代理的威胁。(2)我们回顾了针对每种攻击类型提出的防御策略(如果可用)，并总结了安全研究常用的数据集和基准。(3)在此基础上，我们确定并讨论了大型模型安全方面的开放挑战，强调需要全面的安全评估、可扩展和有效的防御机制以及可持续的数据实践。更重要的是，我们强调了研究界和国际合作集体努力的必要性。我们的工作可以作为研究人员和从业者的有用参考，促进正在进行的全面防御系统和平台的开发，以保护人工智能模型。



## **22. Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach**

优化混合专家中的稳健性和准确性：双模型方法 cs.LG

10 pages, 3 figures, submitted to ICML 2025 (under review)

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.06832v2) [paper-pdf](http://arxiv.org/pdf/2502.06832v2)

**Authors**: Xu Zhang, Kaidi Xu, Ziqing Hu, Ren Wang

**Abstract**: Mixture of Experts (MoE) have shown remarkable success in leveraging specialized expert networks for complex machine learning tasks. However, their susceptibility to adversarial attacks presents a critical challenge for deployment in robust applications. This paper addresses the critical question of how to incorporate robustness into MoEs while maintaining high natural accuracy. We begin by analyzing the vulnerability of MoE components, finding that expert networks are notably more susceptible to adversarial attacks than the router. Based on this insight, we propose a targeted robust training technique that integrates a novel loss function to enhance the adversarial robustness of MoE, requiring only the robustification of one additional expert without compromising training or inference efficiency. Building on this, we introduce a dual-model strategy that linearly combines a standard MoE model with our robustified MoE model using a smoothing parameter. This approach allows for flexible control over the robustness-accuracy trade-off. We further provide theoretical foundations by deriving certified robustness bounds for both the single MoE and the dual-model. To push the boundaries of robustness and accuracy, we propose a novel joint training strategy JTDMoE for the dual-model. This joint training enhances both robustness and accuracy beyond what is achievable with separate models. Experimental results on CIFAR-10 and TinyImageNet datasets using ResNet18 and Vision Transformer (ViT) architectures demonstrate the effectiveness of our proposed methods.

摘要: 混合专家(MOE)在利用专门的专家网络完成复杂的机器学习任务方面取得了显着的成功。然而，它们对对抗性攻击的敏感性给部署在健壮的应用程序中带来了一个关键的挑战。本文讨论了如何在保持较高的自然精度的同时将稳健性融入到MOE中这一关键问题。我们首先分析MOE组件的漏洞，发现专家网络明显比路由器更容易受到对手攻击。基于这一观点，我们提出了一种有针对性的健壮训练技术，该技术结合了一种新的损失函数来增强MOE的对抗健壮性，只需要增加一名专家的健壮性，而不会影响训练或推理效率。在此基础上，我们引入了一种双模型策略，它使用一个平滑参数将标准MOE模型和我们的增强型MOE模型线性地组合在一起。这种方法允许灵活控制稳健性和精确度之间的权衡。通过推导出单模型和双模型的证明的稳健界，我们进一步提供了理论基础。为了突破稳健性和准确性的界限，我们提出了一种新的双模型联合训练策略JTDMoE。这种联合训练增强了健壮性和准确性，超过了单独模型所能达到的效果。在CIFAR-10和TinyImageNet数据集上使用ResNet18和Vision Transformer(VIT)架构的实验结果证明了所提方法的有效性。



## **23. Real-Time Privacy Risk Measurement with Privacy Tokens for Gradient Leakage**

使用隐私令牌进行实时隐私风险测量以应对梯度泄漏 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.02913v4) [paper-pdf](http://arxiv.org/pdf/2502.02913v4)

**Authors**: Jiayang Meng, Tao Huang, Hong Chen, Xin Shi, Qingyu Huang, Chen Hou

**Abstract**: The widespread deployment of deep learning models in privacy-sensitive domains has amplified concerns regarding privacy risks, particularly those stemming from gradient leakage during training. Current privacy assessments primarily rely on post-training attack simulations. However, these methods are inherently reactive, unable to encompass all potential attack scenarios, and often based on idealized adversarial assumptions. These limitations underscore the need for proactive approaches to privacy risk assessment during the training process. To address this gap, we propose the concept of privacy tokens, which are derived directly from private gradients during training. Privacy tokens encapsulate gradient features and, when combined with data features, offer valuable insights into the extent of private information leakage from training data, enabling real-time measurement of privacy risks without relying on adversarial attack simulations. Additionally, we employ Mutual Information (MI) as a robust metric to quantify the relationship between training data and gradients, providing precise and continuous assessments of privacy leakage throughout the training process. Extensive experiments validate our framework, demonstrating the effectiveness of privacy tokens and MI in identifying and quantifying privacy risks. This proactive approach marks a significant advancement in privacy monitoring, promoting the safer deployment of deep learning models in sensitive applications.

摘要: 深度学习模型在隐私敏感领域的广泛部署加剧了人们对隐私风险的担忧，特别是培训期间梯度泄漏造成的风险。目前的隐私评估主要依赖于训练后的攻击模拟。然而，这些方法本质上是被动的，无法涵盖所有潜在的攻击场景，并且通常基于理想化的对抗性假设。这些限制强调了在培训过程中对隐私风险评估采取积极主动的方法的必要性。为了弥补这一差距，我们提出了隐私令牌的概念，它直接从训练过程中的隐私梯度派生出来。隐私令牌封装了梯度特征，当与数据特征相结合时，可以提供对训练数据中私人信息泄漏程度的有价值的见解，从而能够实时测量隐私风险，而不需要依赖对抗性攻击模拟。此外，我们使用相互信息(MI)作为一个稳健的度量来量化训练数据和梯度之间的关系，在整个训练过程中提供对隐私泄露的准确和连续的评估。大量的实验验证了我们的框架，证明了隐私令牌和MI在识别和量化隐私风险方面的有效性。这种主动的方法标志着隐私监控方面的重大进步，促进了在敏感应用程序中更安全地部署深度学习模型。



## **24. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

分层自我暴露和补丁：越狱攻击防御的肯定代币缓解 cs.CR

14 pages, 4 figures, conference

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2501.02629v2) [paper-pdf](http://arxiv.org/pdf/2501.02629v2)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Meijun Gao, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak attacks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods. Our code is publicly available at: https://github.com/oyy2000/LayerAdvPatcher

摘要: 随着大型语言模型(LLM)越来越多地部署在各种应用中，包括聊天机器人助手和代码生成，使它们的行为符合安全和道德标准变得至关重要。然而，越狱攻击利用漏洞来引发意外或有害的输出，严重威胁到LLMS的安全。在本文中，我们介绍了Layer-AdvPatcher，这是一种新的方法，旨在通过一种遗忘策略来通过自增强数据集修补LLMS中的特定层来防御越狱攻击。我们的洞察是，某些层面(S)，在面对有害的提示时，往往会产生肯定的表征。通过识别这些层并恶意暴露它们以生成更多有害数据，人们可以了解它们固有的和不同的攻击漏洞。有了这些暴露，我们就可以“忘掉”这些问题，减少肯定令牌的影响，从而最大限度地减少越狱风险，同时保持模型对安全查询的响应完好无损。我们在两个模型、四个基准数据集和多个最先进的越狱攻击上进行了广泛的实验，以证明我们的方法的有效性。结果表明，与现有的防御方法相比，该框架降低了越狱攻击的危害性和攻击成功率，而不影响良性查询的有效性。我们的代码在以下网址公开提供：https://github.com/oyy2000/LayerAdvPatcher



## **25. Pseudorandom Permutations from Random Reversible Circuits**

随机可逆电路的伪随机排列 cs.CC

Merged with arXiv:2409.14614; for merged paper see arXiv:2502.07159.  A previous version of one of the merged components of this paper contained  candidate constructions of computationally pseudorandom permutations from  one-way functions. There was an error in the proof of security, and we have  withdrawn this result

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2404.14648v4) [paper-pdf](http://arxiv.org/pdf/2404.14648v4)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).

摘要: 我们研究了由可逆$3$位门($0，1^3$上的置换)构成的随机电路计算的$0，1^n上置换的伪随机性。我们的主要结果是，一个深度为$n\cot\tide{O}(k^2)$的随机电路，每一层由固定最近邻体系结构中的$\约n/3$随机门组成，产生几乎$k$方向的独立排列。主要的技术内容是证明了由单个随机的$3$比特最近邻门产生的$n$比特串的$k$-元组上的马尔可夫链至少有$1/n\cdot\tilde{O}(K)$。这比Gowers[Gowers96]的原始工作有所改进，Gowers[Gowers96]对一个随机门(具有非相邻输入)显示了$1/\mathm{pol}(n，k)$的差距；在随后的工作[HMMR05，BH08]中，在相同设置下将差距改进为$\Omega(1/n^2k)$。从密码学的角度来看，我们的结果可以看作是一种特别简单实用的分组密码构造，它提供了针对在几轮内访问$k$~输入输出对的攻击者的可证明的统计安全性。我们还证明了伪随机函数的伪随机置换的Luby-Rackoff构造可以用可逆电路实现。由此，我们在最小可逆电路大小问题(MRCSP)的复杂性方面取得了进展，表明在假设存在单向函数(OWF)的情况下，固定多项式大小的分组密码在计算上是安全的，可以抵抗任意多项式时间的攻击者。



## **26. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

MAA：针对视觉语言预训练模型的强力对抗攻击 cs.CV

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08079v1) [paper-pdf](http://arxiv.org/pdf/2502.08079v1)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.

摘要: 当前用于评估视觉语言预训练(VLP)模型在多模式任务中的稳健性的对抗性攻击存在可转移性有限的问题，其中针对特定模型的攻击往往难以在不同的模型上有效地泛化，从而限制了它们在更广泛地评估稳健性方面的有效性。这主要归因于过度依赖特定型号的特征和区域，特别是在图像模式方面。在本文中，我们提出了一种优雅而高效的方法，称为精细攻击(MAA)，它充分利用了个体样本的模型无关特性和脆弱性，从而增强了泛化能力，降低了模型依赖。MAA通过开发一种新的调整大小和滑动裁剪(RSCrop)技术，结合多粒度相似破坏(MGSD)策略，强调对抗性图像的细粒度优化。在不同的VLP模型、多个基准数据集和各种下游任务上的广泛实验表明，MAA显著增强了对抗性攻击的有效性和可转移性。我们进行了大量的性能研究，以深入了解各种型号配置的有效性，从而指导该领域的未来发展。



## **27. Cascading Bandits Robust to Adversarial Corruptions**

级联强盗对对抗性腐败的鲁棒性 cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08077v1) [paper-pdf](http://arxiv.org/pdf/2502.08077v1)

**Authors**: Jize Xie, Cheng Chen, Zhiyong Wang, Shuai Li

**Abstract**: Online learning to rank sequentially recommends a small list of items to users from a large candidate set and receives the users' click feedback. In many real-world scenarios, users browse the recommended list in order and click the first attractive item without checking the rest. Such behaviors are usually formulated as the cascade model. Many recent works study algorithms for cascading bandits, an online learning to rank framework in the cascade model. However, the performance of existing methods may drop significantly if part of the user feedback is adversarially corrupted (e.g., click fraud). In this work, we study how to resist adversarial corruptions in cascading bandits. We first formulate the ``\textit{Cascading Bandits with Adversarial Corruptions}" (CBAC) problem, which assumes that there is an adaptive adversary that may manipulate the user feedback. Then we propose two robust algorithms for this problem, which assume the corruption level is known and agnostic, respectively. We show that both algorithms can achieve logarithmic regret when the algorithm is not under attack, and the regret increases linearly with the corruption level. The experimental results also verify the robustness of our methods.

摘要: 在线学习排名按顺序从一个大的候选集合中向用户推荐一小部分项目，并接收用户的点击反馈。在许多现实世界的场景中，用户按顺序浏览推荐列表，然后点击第一个有吸引力的项目，而不检查其他项目。这类行为通常被表述为级联模型。最近的许多著作研究了级联强盗的算法，这是一个在线学习在级联模型中排名的框架。然而，如果部分用户反馈被相反地破坏(例如，点击欺诈)，则现有方法的性能可能显著下降。在这项工作中，我们研究了如何抵抗级联强盗中的对抗性腐败。我们首先提出了一个假设存在一个可以操纵用户反馈的自适应对手的``\textit(Cascading Bandits With Adversative Corruptions)问题(CBAC)。然后，我们针对这个问题提出了两种稳健的算法，分别假设腐败程度是已知的和不可知的。我们证明了这两种算法在算法没有受到攻击的情况下都能实现对数后悔，并且后悔程度随着腐败程度的增加而线性增加。实验结果也验证了我们方法的稳健性。



## **28. RESIST: Resilient Decentralized Learning Using Consensus Gradient Descent**

REIST：使用共识梯度下降的弹性去中心化学习 cs.LG

preprint of a journal paper; 100 pages and 17 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07977v1) [paper-pdf](http://arxiv.org/pdf/2502.07977v1)

**Authors**: Cheng Fang, Rishabh Dixit, Waheed U. Bajwa, Mert Gurbuzbalaban

**Abstract**: Empirical risk minimization (ERM) is a cornerstone of modern machine learning (ML), supported by advances in optimization theory that ensure efficient solutions with provable algorithmic convergence rates, which measure the speed at which optimization algorithms approach a solution, and statistical learning rates, which characterize how well the solution generalizes to unseen data. Privacy, memory, computational, and communications constraints increasingly necessitate data collection, processing, and storage across network-connected devices. In many applications, these networks operate in decentralized settings where a central server cannot be assumed, requiring decentralized ML algorithms that are both efficient and resilient. Decentralized learning, however, faces significant challenges, including an increased attack surface for adversarial interference during decentralized learning processes. This paper focuses on the man-in-the-middle (MITM) attack, which can cause models to deviate significantly from their intended ERM solutions. To address this challenge, we propose RESIST (Resilient dEcentralized learning using conSensus gradIent deScenT), an optimization algorithm designed to be robust against adversarially compromised communication links. RESIST achieves algorithmic and statistical convergence for strongly convex, Polyak-Lojasiewicz, and nonconvex ERM problems. Experimental results demonstrate the robustness and scalability of RESIST for real-world decentralized learning in adversarial environments.

摘要: 经验风险最小化(ERM)是现代机器学习(ML)的基石，得到了最优化理论的进步的支持，最优化理论的进步确保了有效解的可证明算法收敛速度(衡量优化算法逼近解的速度)和统计学习率(表征解对未知数据的泛化程度)。隐私、内存、计算和通信方面的限制要求跨网络连接的设备进行数据收集、处理和存储。在许多应用中，这些网络在不能假设中央服务器的去中心化设置中运行，这需要既高效又有弹性的去中心化ML算法。然而，去中心化学习面临重大挑战，包括在去中心化学习过程中，敌意干扰的攻击面增加。本文主要研究中间人(MITM)攻击，这种攻击会导致模型严重偏离其预期的ERM解决方案。为了应对这一挑战，我们提出了RESTECT(弹性分散学习使用共识梯度下降)，这是一种优化算法，旨在对相反的通信链路具有健壮性。对于强凸、Polyak-Lojasiewicz和非凸的ERM问题，Reat实现了算法和统计的收敛。实验结果表明，在对抗性环境下，该算法具有较好的鲁棒性和可扩展性。



## **29. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

EIA：针对多面手网络代理隐私泄露的环境注入攻击 cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2409.11295v4) [paper-pdf](http://arxiv.org/pdf/2409.11295v4)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' PII, which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites, a scenario that remains largely unexplored in the literature. In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request. Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments. We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70% ASR in stealing specific PII and 16% ASR for full user request. Additionally, by accessing the stealthiness and experimenting with a defensive system prompt, we indicate that EIA is hard to detect and mitigate. Notably, attacks that are not well adapted for a webpage can be detected via human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such supervision ineffective. Thus, we further discuss the defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.

摘要: 多面手网络代理在自主完成真实网站上的各种任务方面表现出了非凡的潜力，显著提高了人类的生产力。然而，预订机票等网络任务通常涉及用户的PII，如果网络代理意外地与受影响的网站交互，可能会面临潜在的隐私风险，这种情况在文献中基本上仍未探讨。在这项工作中，我们通过对对抗环境中通才网络代理的隐私风险进行第一次研究来缩小这一差距。首先，我们给出了一个现实的网站攻击威胁模型，其中我们考虑了两个敌对目标：窃取用户的特定PII或整个用户请求。然后，我们提出了一种新的攻击方法，称为环境注入攻击(EIA)。EIA注入恶意内容，旨在很好地适应代理运行的环境，我们的工作特别针对Web环境中的隐私场景实例化了EIA。我们从Mind2Web收集了177个动作步骤，涉及现实网站上不同的PII类别，并使用迄今最有能力的通才Web代理框架之一进行了实验。结果表明，EIA在窃取特定PII请求时获得了高达70%的ASR，对于完整的用户请求达到了16%的ASR。此外，通过访问隐蔽性和试验防御系统提示，我们表明EIA很难检测和缓解。值得注意的是，没有很好地适应网页的攻击可以通过人工检查来检测，这导致了我们关于安全性和自主性之间的权衡的讨论。然而，额外的攻击者的努力可能会使EIA无缝适应，使这种监督无效。因此，我们进一步讨论了网站部署前和部署后阶段的防御，而不依赖于人的监督，并呼吁更先进的防御策略。



## **30. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

Logicbreaks：理解基于规则的推理颠覆的框架 cs.AI

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2407.00075v3) [paper-pdf](http://arxiv.org/pdf/2407.00075v3)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.

摘要: 我们研究如何根据预算指定的规则颠覆大型语言模型（LLM）。我们首先将规则遵循形式化为命题Horn逻辑中的推理，这是一个数学系统，其中规则的形式为“如果$P$和$Q$，那么$R$”，对于某些命题$P$、$Q$和$R$。接下来，我们证明，尽管小型变压器可以忠实地遵循这些规则，但恶意制作的提示仍然会误导理论构建和从数据中学习的模型。此外，我们证明了LLM上的流行攻击算法可以找到对抗提示并诱导与我们的理论一致的注意力模式。我们新颖的基于逻辑的框架为在基于规则的环境中研究LLM提供了基础，从而能够对逻辑推理和越狱攻击等任务进行正式分析。



## **31. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

部分失去控制权下非线性系统的能量弹性 math.OC

20 pages, 1 figure

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07603v1) [paper-pdf](http://arxiv.org/pdf/2502.07603v1)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting considers only simple linear models and not general nonlinear dynamical systems. We first characterize the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Based on this approximation, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A simulation example on an academic nonlinear system demonstrates that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies.

摘要: 在本文中，我们通过研究由于执行器故障或通过对手攻击而遭受部分控制权威丧失的系统的所有输入所使用的增加的能量来量化非线性动力系统的弹性。为了量化能量的最大增长，我们引入了能量弹性度量的概念。在这种特殊情况下，以前的工作只考虑简单的线性模型，而不考虑一般的非线性动力系统。我们首先刻画了标称系统和故障系统中控制信号的平均值，这允许我们近似控制中的能量。然后，我们得到所有故障输入的故障系统的能量的最坏情况的近似值。基于这一近似，我们推导出了当一个执行器失去控制权时能量弹性度量的界。对一个学术非线性系统的仿真实例表明，尽管在获得控制能量时使用了近似，但该度量在量化系统的弹性方面是有用的，而不具有显著的保守性。



## **32. Efficient Image-to-Image Diffusion Classifier for Adversarial Robustness**

具有对抗鲁棒性的高效图像到图像扩散分类器 cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2408.08502v2) [paper-pdf](http://arxiv.org/pdf/2408.08502v2)

**Authors**: Hefei Mei, Minjing Dong, Chang Xu

**Abstract**: Diffusion models (DMs) have demonstrated great potential in the field of adversarial robustness, where DM-based defense methods can achieve superior defense capability without adversarial training. However, they all require huge computational costs due to the usage of large-scale pre-trained DMs, making it difficult to conduct full evaluation under strong attacks and compare with traditional CNN-based methods. Simply reducing the network size and timesteps in DMs could significantly harm the image generation quality, which invalidates previous frameworks. To alleviate this issue, we redesign the diffusion framework from generating high-quality images to predicting distinguishable image labels. Specifically, we employ an image translation framework to learn many-to-one mapping from input samples to designed orthogonal image labels. Based on this framework, we introduce an efficient Image-to-Image diffusion classifier with a pruned U-Net structure and reduced diffusion timesteps. Besides the framework, we redesign the optimization objective of DMs to fit the target of image classification, where a new classification loss is incorporated in the DM-based image translation framework to distinguish the generated label from those of other classes. We conduct sufficient evaluations of the proposed classifier under various attacks on popular benchmarks. Extensive experiments show that our method achieves better adversarial robustness with fewer computational costs than DM-based and CNN-based methods. The code is available at https://github.com/hfmei/IDC

摘要: 扩散模型在对抗鲁棒性领域显示出了巨大的潜力，基于扩散模型的防御方法可以在不需要对手训练的情况下获得优越的防御能力。然而，由于它们都需要使用大规模的预先训练的DM，因此它们都需要巨大的计算代价，这使得在强攻击下进行充分评估并与传统的基于CNN的方法进行比较是困难的。简单地减少DM中的网络大小和时间步长可能会严重损害映像生成质量，从而使之前的框架失效。为了缓解这个问题，我们重新设计了扩散框架，从生成高质量的图像到预测可区分的图像标签。具体地说，我们使用一个图像转换框架来学习从输入样本到设计的正交图像标签的多对一映射。基于该框架，我们提出了一种高效的图像到图像扩散分类器，该分类器具有修剪的U网结构和减少的扩散时间。除了该框架外，我们还重新设计了DMS的优化目标，以适应图像分类的目标，其中在基于DM的图像翻译框架中引入了新的分类损失，以区分生成的标签和其他类别的标签。在各种针对流行基准的攻击下，我们对提出的分类器进行了充分的评估。大量实验表明，与基于DM的方法和基于CNN的方法相比，我们的方法具有更好的对抗健壮性和更少的计算代价。代码可在https://github.com/hfmei/IDC上获得



## **33. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

RoMA：通过具有全局扰动和对抗一致性正规化的字节级对抗训练来实现稳健的恶意软件归因 cs.CR

13 pages, 4 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07492v1) [paper-pdf](http://arxiv.org/pdf/2502.07492v1)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.

摘要: 将APT(高级持续威胁)恶意软件归因于各自的组织对于威胁情报和网络安全至关重要。然而，聪明的对手往往隐藏自己的身份，使归因具有内在的对抗性。现有的基于机器学习的归因模型虽然有效，但仍然非常容易受到对手的攻击。例如，最先进的字节级模型MalConv在PGD(投影梯度下降)攻击下的准确率从90%以上下降到2%以下。将已有的基于梯度的恶意软件检测或图像处理的对抗性训练技术应用到恶意软件属性识别中，发现无论是稳健性还是训练效率都需要显著提高。为了解决这一问题，我们提出了一种新颖的单步对抗性训练方法，该方法结合全局扰动来生成增强的对抗性样本，并使用对抗性一致性正则化来提高表示质量和韧性。引入了一个新的APT恶意软件数据集AMG18，该数据集具有多样化的样本和真实的类别不平衡。大量实验表明，在对抗性稳健性(例如，在PGD攻击下达到80%以上的健壮性--是次佳方法的两倍多)和训练效率(例如，在准确率方面是次佳方法的两倍以上)方面，ROMA显著优于七种竞争方法，同时在非对抗性场景中保持了卓越的标准准确率。



## **34. Mining Power Destruction Attacks in the Presence of Petty-Compliant Mining Pools**

存在小兼容性采矿设备的情况下的采矿电力破坏攻击 cs.CR

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07410v1) [paper-pdf](http://arxiv.org/pdf/2502.07410v1)

**Authors**: Roozbeh Sarenche, Svetla Nikova, Bart Preneel

**Abstract**: Bitcoin's security relies on its Proof-of-Work consensus, where miners solve puzzles to propose blocks. The puzzle's difficulty is set by the difficulty adjustment mechanism (DAM), based on the network's available mining power. Attacks that destroy some portion of mining power can exploit the DAM to lower difficulty, making such attacks profitable. In this paper, we analyze three types of mining power destruction attacks in the presence of petty-compliant mining pools: selfish mining, bribery, and mining power distraction attacks. We analyze selfish mining while accounting for the distribution of mining power among pools, a factor often overlooked in the literature. Our findings indicate that selfish mining can be more destructive when the non-adversarial mining share is well distributed among pools. We also introduce a novel bribery attack, where the adversarial pool bribes petty-compliant pools to orphan others' blocks. For small pools, we demonstrate that the bribery attack can dominate strategies like selfish mining or undercutting. Lastly, we present the mining distraction attack, where the adversarial pool incentivizes petty-compliant pools to abandon Bitcoin's puzzle and mine for a simpler puzzle, thus wasting some part of their mining power. Similar to the previous attacks, this attack can lower the mining difficulty, but with the difference that it does not generate any evidence of mining power destruction, such as orphan blocks.

摘要: 比特币的安全性依赖于其工作证明共识，即矿工通过解决谜题来提出块。谜题的难度是由难度调整机制(大坝)根据网络的可用采矿力设定的。摧毁部分采矿力量的攻击可以利用大坝来降低难度，使此类攻击有利可图。在本文中，我们分析了三种类型的矿权破坏攻击：自私开采、贿赂和矿权分散攻击。我们分析了自私的采矿，同时考虑了矿权在不同矿池之间的分配，这是文献中经常被忽视的一个因素。我们的发现表明，当非对抗性的采矿份额在池中均匀分布时，自私开采可能会更具破坏性。我们还引入了一种新的贿赂攻击，在这种攻击中，对手池贿赂符合小额规则的池以孤立他人的块。对于较小的资金池，我们证明了贿赂攻击可以主导自私开采或偷工减料等策略。最后，我们提出了挖掘分心攻击，在这种攻击中，敌意的池激励遵守规则的池放弃比特币的谜题，转而使用更简单的谜题，从而浪费了他们的部分挖掘力。与之前的攻击类似，这次攻击可以降低采矿难度，但不同的是，它不会产生任何矿权破坏的证据，如孤儿区块。



## **35. Enhancing Security and Privacy in Federated Learning using Low-Dimensional Update Representation and Proximity-Based Defense**

使用低维更新表示和基于邻近度的防御增强联邦学习中的安全性和隐私 cs.CR

14 pages

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2405.18802v2) [paper-pdf](http://arxiv.org/pdf/2405.18802v2)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, particularly against curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{F}ederated \underline{L}earning with Low-Dimensional \underline{U}pdate \underline{R}epresentation and \underline{P}roximity-Based defense (FLURP), designed to address privacy preservation and resistance to Byzantine attacks in distributed learning environments. FLURP employs $\mathsf{LinfSample}$ method, enabling clients to compute the $l_{\infty}$ norm across sliding windows of updates, resulting in a Low-Dimensional Update Representation (LUR). Calculating the shared distance matrix among LURs, rather than updates, significantly reduces the overhead of Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and poisoned updates. Additionally, FLURP integrates a privacy-preserving proximity-based defense mechanism utilizing optimized SMPC protocols to minimize communication rounds. Our experiments demonstrate FLURP's effectiveness in countering Byzantine adversaries with low communication and runtime overhead. FLURP offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.

摘要: 联合学习(FL)是一种很有前途的隐私保护机器学习范例，允许数据所有者在保持数据本地化的同时协作训练模型。尽管有潜力，FL仍面临着与客户端和服务器的可信性相关的挑战，特别是在面对好奇或恶意的对手时。为了解决分布式学习环境中隐私保护和抵抗拜占庭攻击的问题，本文提出了一种新的框架-FURPFLURP使用$\mathsf{LinfSample}$方法，使客户端能够跨更新的滑动窗口计算$L_{\inty}$范数，从而产生低维更新表示(LUR)。计算LURs之间的共享距离矩阵，而不是更新，显著减少了安全多方计算(SMPC)的开销三个数量级，同时有效地区分了良性和有毒更新。此外，FLURP利用优化的SMPC协议集成了基于隐私保护的邻近防御机制，以最大限度地减少通信轮次。我们的实验证明了FLURP以较低的通信和运行时间开销对抗拜占庭攻击的有效性。FLURP为分布式环境中安全可靠的FL提供了一个可扩展的框架，促进了其在需要强大的数据管理和安全的场景中的应用。



## **36. CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models**

CAT：用于评估潜在扩散模型中保护性扰动稳健性的对比对抗训练 cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07225v1) [paper-pdf](http://arxiv.org/pdf/2502.07225v1)

**Authors**: Sen Peng, Mingyue Wang, Jianfei He, Jijia Yang, Xiaohua Jia

**Abstract**: Latent diffusion models have recently demonstrated superior capabilities in many downstream image synthesis tasks. However, customization of latent diffusion models using unauthorized data can severely compromise the privacy and intellectual property rights of data owners. Adversarial examples as protective perturbations have been developed to defend against unauthorized data usage by introducing imperceptible noise to customization samples, preventing diffusion models from effectively learning them. In this paper, we first reveal that the primary reason adversarial examples are effective as protective perturbations in latent diffusion models is the distortion of their latent representations, as demonstrated through qualitative and quantitative experiments. We then propose the Contrastive Adversarial Training (CAT) utilizing adapters as an adaptive attack against these protection methods, highlighting their lack of robustness. Extensive experiments demonstrate that our CAT method significantly reduces the effectiveness of protective perturbations in customization configurations, urging the community to reconsider and enhance the robustness of existing protective perturbation methods. Code is available at \hyperlink{here}{https://github.com/senp98/CAT}.

摘要: 最近，潜扩散模型在许多下游图像合成任务中表现出了优越的性能。然而，使用未经授权的数据定制潜在扩散模型可能会严重损害数据所有者的隐私和知识产权。作为保护性扰动的对抗性例子已经被开发出来，通过向定制样本引入不可察觉的噪声来防御未经授权的数据使用，从而防止扩散模型有效地学习它们。在这篇文章中，我们首先揭示了对抗性例子在潜在扩散模型中作为保护性扰动有效的主要原因是其潜在表示的扭曲，通过定性和定量实验证明了这一点。然后，我们提出了利用适配器作为对这些保护方法的自适应攻击的对比性对抗训练(CAT)，突出了它们的健壮性不足。大量的实验表明，我们的CAT方法显著降低了定制配置中保护扰动的有效性，促使社区重新考虑并增强现有保护扰动方法的健壮性。代码可在\hyperlink{here}{https://github.com/senp98/CAT}.上找到



## **37. LUNAR: LLM Unlearning via Neural Activation Redirection**

LUNAR：LLM通过神经激活重定向消除学习 cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.

摘要: 大型语言模型(LLM)受益于对越来越多的文本数据进行培训，但结果是，它们越来越多地招致泄露私人信息的风险。因此，有选择地从LLM中移除知识的能力是一种非常理想的能力。在本文中，我们提出了一种基于线性表征假设的去学习方法--LUNAR。LUNAR通过将未学习数据的表示重定向到触发模型表达其无法回答问题的固有能力的区域来运行。LUNAR实现了最先进的遗忘性能，同时显著增强了推理过程中未学习模型的可控性。具体地说，在各种基本型号的手枪数据集上，LUNAR在组合的“遗忘效能”和“模型效用”分数(“偏差分数”)上取得了2.9倍到11.7倍的改进。我们还通过定量分析和定性例子证明，月球在产生连贯的和上下文感知的响应方面具有优越的可控性，减轻了现有方法的不良副作用。此外，我们还证明了LUNAR对白盒攻击具有很强的健壮性，并且在处理真实场景(如处理顺序遗忘请求)方面具有很强的通用性。



## **38. SMAB: MAB based word Sensitivity Estimation Framework and its Applications in Adversarial Text Generation**

SMAB：基于MAB的词敏感度估计框架及其在对抗性文本生成中的应用 cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07101v1) [paper-pdf](http://arxiv.org/pdf/2502.07101v1)

**Authors**: Saurabh Kumar Pandey, Sachin Vashistha, Debrup Das, Somak Aditya, Monojit Choudhury

**Abstract**: To understand the complexity of sequence classification tasks, Hahn et al. (2021) proposed sensitivity as the number of disjoint subsets of the input sequence that can each be individually changed to change the output. Though effective, calculating sensitivity at scale using this framework is costly because of exponential time complexity. Therefore, we introduce a Sensitivity-based Multi-Armed Bandit framework (SMAB), which provides a scalable approach for calculating word-level local (sentence-level) and global (aggregated) sensitivities concerning an underlying text classifier for any dataset. We establish the effectiveness of our approach through various applications. We perform a case study on CHECKLIST generated sentiment analysis dataset where we show that our algorithm indeed captures intuitively high and low-sensitive words. Through experiments on multiple tasks and languages, we show that sensitivity can serve as a proxy for accuracy in the absence of gold data. Lastly, we show that guiding perturbation prompts using sensitivity values in adversarial example generation improves attack success rate by 15.58%, whereas using sensitivity as an additional reward in adversarial paraphrase generation gives a 12.00% improvement over SOTA approaches. Warning: Contains potentially offensive content.

摘要: 为了理解序列分类任务的复杂性，Hahn等人。(2021)提出的敏感度是输入序列的不相交子集的数目，每个子集都可以单独改变以改变输出。虽然有效，但由于指数时间复杂性，使用该框架在规模上计算敏感度代价高昂。因此，我们引入了一种基于敏感度的多臂Bandit框架(SMAB)，它提供了一种可扩展的方法来计算关于任何数据集的底层文本分类器的词级局部(句子级)和全局(聚合)敏感度。我们通过各种应用证明了我们方法的有效性。我们在核对表生成的情感分析数据集上进行了实例研究，结果表明，我们的算法确实能够直观地捕获高敏感度和低敏感度的词。通过对多个任务和语言的实验，我们发现，在没有GOLD数据的情况下，敏感度可以作为准确度的代理。最后，我们展示了在对抗性范例生成中使用敏感度来引导扰动提示将攻击成功率提高了15.58%，而在对抗性释义生成中使用敏感度作为额外奖励则比SOTA方法提高了12.00%。警告：包含潜在的攻击性内容。



## **39. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

Drop：通过联邦学习的知识蒸馏进行毒药稀释 cs.LG

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07011v1) [paper-pdf](http://arxiv.org/pdf/2502.07011v1)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.

摘要: 联合学习很容易受到敌意操纵，恶意客户端可以注入有毒更新来影响全局模型的行为。虽然现有的防御机制已经取得了显著的进展，但它们无法防御旨在根据不同的学习和攻击配置诱导有针对性的后门的对手。为了解决这一局限性，我们引入了Drop(基于蒸馏的毒化减少)，这是一种新型的防御机制，它将集群和活动跟踪技术与通过知识蒸馏从客户中提取良性行为相结合，以应对在联邦内操纵低数据毒化率和不同恶意客户比率的隐蔽对手。通过广泛的实验，与现有的防御相比，我们的方法在广泛的学习配置上表现出了卓越的稳健性。最后，在非IID客户端数据分发的挑战环境下，我们评估了现有的防御措施和我们的方法，并强调了在这种情况下设计弹性FL防御措施所面临的挑战。



## **40. Breaking Quantum Key Distributions under Quantum Switch-Based Attack**

基于量子交换机的攻击下破坏量子密钥分布 quant-ph

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06780v1) [paper-pdf](http://arxiv.org/pdf/2502.06780v1)

**Authors**: Sumit Nandi, Biswaranjan Panda, Pankaj Agrawal, Arun K Pati

**Abstract**: Quantum key distribution (QKD) enables secure key sharing between distant parties, with several protocols proven resilient against conventional eavesdropping strategies. Here, we introduce a new attack scenario where an eavesdropper, Eve, exploits a quantum switch using the indefinite causal order to intercept and manipulate quantum communication channel. Using multiple metrics such as the information gain, mutual information, and Bell violation, we demonstrate that the presence of a quantum switch significantly compromises QKD security. Our results highlight a previously overlooked vulnerability, emphasizing the need for countermeasures against quantum-controlled adversarial strategies.

摘要: 量子密钥分发（QKD）实现了远程方之间的安全密钥共享，几种协议被证明具有抵御传统窃听策略的能力。在这里，我们引入了一种新的攻击场景，其中窃听者Eve利用量子开关，使用不确定因果顺序来拦截和操纵量子通信通道。使用信息收益、互信息和Bell破坏等多种指标，我们证明量子交换机的存在会显着损害QKD安全性。我们的结果强调了以前被忽视的漏洞，强调了针对量子控制对抗策略的对策的必要性。



## **41. When Witnesses Defend: A Witness Graph Topological Layer for Adversarial Graph Learning**

当证人辩护时：对抗图学习的证人图布局层 cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2409.14161v3) [paper-pdf](http://arxiv.org/pdf/2409.14161v3)

**Authors**: Naheed Anjum Arafat, Debabrota Basu, Yulia Gel, Yuzhou Chen

**Abstract**: Capitalizing on the intuitive premise that shape characteristics are more robust to perturbations, we bridge adversarial graph learning with the emerging tools from computational topology, namely, persistent homology representations of graphs. We introduce the concept of witness complex to adversarial analysis on graphs, which allows us to focus only on the salient shape characteristics of graphs, yielded by the subset of the most essential nodes (i.e., landmarks), with minimal loss of topological information on the whole graph. The remaining nodes are then used as witnesses, governing which higher-order graph substructures are incorporated into the learning process. Armed with the witness mechanism, we design Witness Graph Topological Layer (WGTL), which systematically integrates both local and global topological graph feature representations, the impact of which is, in turn, automatically controlled by the robust regularized topological loss. Given the attacker's budget, we derive the important stability guarantees of both local and global topology encodings and the associated robust topological loss. We illustrate the versatility and efficiency of WGTL by its integration with five GNNs and three existing non-topological defense mechanisms. Our extensive experiments across six datasets demonstrate that WGTL boosts the robustness of GNNs across a range of perturbations and against a range of adversarial attacks. Our datasets and source codes are available at https://github.com/toggled/WGTL.

摘要: 基于形状特征对扰动的鲁棒性更强这一直观前提，我们利用计算拓扑学中的新兴工具，即图的持久同调表示，在对抗性图学习之间架起桥梁。我们将证人复合体的概念引入到图的对抗分析中，使得我们只关注图的显著形状特征，这些特征是由最重要的结点(即地标)的子集产生的，而整个图的拓扑信息损失最小。然后，剩余的节点被用作见证，控制哪些更高阶图的子结构被合并到学习过程中。结合见证机制，我们设计了见证图拓扑层(Witness Graph Topology Layer，WGTL)，该拓扑层系统地集成了局部拓扑图和全局拓扑图的特征表示，其影响由稳健的正则化拓扑损失自动控制。在给定攻击者预算的情况下，我们推导出了局部和全局拓扑编码的重要稳定性保证以及相关的稳健拓扑损失。我们通过与五个GNN和三个现有的非拓扑防御机制的集成来说明WGTL的通用性和有效性。我们在六个数据集上的广泛实验表明，WGTL增强了GNN在一系列扰动和一系列对手攻击下的健壮性。我们的数据集和源代码可在https://github.com/toggled/WGTL.上获得



## **42. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便即使在数百个步骤的微调之后，对手也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，在防篡改方面取得进展是可能的，为提高开放重量LLMS的安全性开辟了一条有希望的新途径。



## **43. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

探索音频编辑功能，以用户为中心的隐私防御基于大型语言模型（LLM）的情感推理攻击 cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.

摘要: 包括虚拟助理、视频会议平台和可穿戴设备在内的语音支持技术的迅速普及引发了人们对隐私的严重担忧，特别是关于从音频数据推断敏感情感信息的问题。现有的隐私保护方法往往会损害可用性和安全性，限制了它们在实际场景中的采用。本文介绍了一种新颖的、以用户为中心的方法，该方法利用熟悉的音频编辑技术，特别是音调和节奏操作，在不牺牲可用性的情况下保护情感隐私。通过分析Android和iOS平台上流行的音频编辑应用程序，我们发现这些功能广泛使用和使用。我们严格评估了它们对威胁模型的有效性，考虑了来自不同来源的对抗性攻击，包括深度神经网络(DNN)、大型语言模型(LLMS)和可逆性测试。我们在三个不同的数据集上进行的实验表明，音调和节奏操作有效地混淆了情感数据。此外，我们还探讨了轻量级设备上实施的设计原则，以确保跨各种设备和平台的广泛适用性。



## **44. LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

LIAR：利用推理时间对齐（N中最佳）以秒为单位越狱LLM cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.05232v2) [paper-pdf](http://arxiv.org/pdf/2412.05232v2)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Traditional jailbreaks have successfully exposed vulnerabilities in LLMs, primarily relying on discrete combinatorial optimization, while more recent methods focus on training LLMs to generate adversarial prompts. However, both approaches are computationally expensive and slow, often requiring significant resources to generate a single successful attack. We hypothesize that the inefficiency of these methods arises from an inadequate characterization of the jailbreak problem itself. To address this gap, we approach the jailbreak problem as an alignment problem, leading us to propose LIAR (Leveraging Inference time Alignment to jailbReak), a fast and efficient best-of-N approach tailored for jailbreak attacks. LIAR offers several key advantages: it eliminates the need for additional training, operates in a fully black-box setting, significantly reduces computational overhead, and produces more human-readable adversarial prompts while maintaining competitive attack success rates. Our results demonstrate that a best-of-N approach is a simple yet highly effective strategy for evaluating the robustness of aligned LLMs, achieving attack success rates (ASR) comparable to state-of-the-art methods while offering a 10x improvement in perplexity and a significant speedup in Time-to-Attack, reducing execution time from tens of hours to seconds. Additionally, We also provide sub-optimality guarantees for the proposed LIAR. Our work highlights the potential of efficient, alignment-based jailbreak strategies for assessing and stress-testing AI safety measures.

摘要: 传统的越狱已经成功地暴露了LLMS中的漏洞，主要依赖于离散组合优化，而最近的方法则专注于训练LLMS生成对抗性提示。然而，这两种方法在计算上都很昂贵且速度很慢，通常需要大量资源才能生成一次成功的攻击。我们假设，这些方法的低效源于对越狱问题本身的不充分描述。为了解决这一差距，我们将越狱问题视为一个对齐问题，导致我们提出了LIAR(利用推理时间对齐越狱)，这是一种为越狱攻击量身定做的快速高效的N中之最方法。Liar提供了几个关键优势：它消除了对额外培训的需要，在完全黑箱设置下运行，显著减少了计算开销，并在保持有竞争力的攻击成功率的同时生成更易读的对抗性提示。我们的结果表明，N中最佳方法是一种简单但高效的策略来评估对齐的LLM的健壮性，实现了与最先进方法相当的攻击成功率(ASR)，同时将困惑程度提高了10倍，攻击时间显著加快，执行时间从数十小时减少到数秒。此外，我们还为被提议的说谎者提供次最优保证。我们的工作突出了高效、基于路线的越狱战略在评估和压力测试人工智能安全措施方面的潜力。



## **45. Automatic ISA analysis for Secure Context Switching**

用于安全上下文切换的自动ISA分析 cs.OS

15 pages, 6 figures, 2 tables, 4 listings

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06609v1) [paper-pdf](http://arxiv.org/pdf/2502.06609v1)

**Authors**: Neelu S. Kalani, Thomas Bourgeat, Guerney D. H. Hunt, Wojciech Ozga

**Abstract**: Instruction set architectures are complex, with hundreds of registers and instructions that can modify dozens of them during execution, variably on each instance. Prose-style ISA specifications struggle to capture these intricacies of the ISAs, where often the important details about a single register are spread out across hundreds of pages of documentation. Ensuring that all ISA-state is swapped in context switch implementations of privileged software requires meticulous examination of these pages. This manual process is tedious and error-prone.   We propose a tool called Sailor that leverages machine-readable ISA specifications written in Sail to automate this task. Sailor determines the ISA-state necessary to swap during the context switch using the data collected from Sail and a novel algorithm to classify ISA-state as security-sensitive. Using Sailor's output, we identify three different classes of mishandled ISA-state across four open-source confidential computing systems. We further reveal five distinct security vulnerabilities that can be exploited using the mishandled ISA-state. This research exposes an often overlooked attack surface that stems from mishandled ISA-state, enabling unprivileged adversaries to exploit system vulnerabilities.

摘要: 指令集体系结构很复杂，有数百个寄存器和指令，这些寄存器和指令可以在执行期间修改数十个寄存器和指令，这些寄存器和指令在每个实例上都是可变的。散文式的ISA规范很难捕捉到ISA的这些错综复杂之处，其中关于单个寄存器的重要细节通常分布在数百页的文档中。要确保在特权软件的上下文切换实现中交换所有ISA状态，需要仔细检查这些页面。此手动过程繁琐且容易出错。我们提出了一个名为Sailor的工具，它利用用Sail编写的机器可读的ISA规范来自动执行这项任务。Silor使用从SAIL收集的数据和一种新的算法将ISA状态分类为安全敏感的，来确定在上下文切换期间交换所需的ISA状态。使用Sailor的输出，我们确定了四个开源机密计算系统中三种不同类别的处理不当的ISA-STATE。我们进一步揭示了五个不同的安全漏洞，可以使用处理不当的ISA状态来利用这些漏洞。这项研究揭露了一个经常被忽视的攻击面，这个攻击面源于处理不当的ISA状态，使没有特权的对手能够利用系统漏洞。



## **46. Krum Federated Chain (KFC): Using blockchain to defend against adversarial attacks in Federated Learning**

Krum Federated Chain（KFC）：使用区块链防御联邦学习中的对抗性攻击 cs.LG

Submitted to Neural Networks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06917v1) [paper-pdf](http://arxiv.org/pdf/2502.06917v1)

**Authors**: Mario García-Márquez, Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: Federated Learning presents a nascent approach to machine learning, enabling collaborative model training across decentralized devices while safeguarding data privacy. However, its distributed nature renders it susceptible to adversarial attacks. Integrating blockchain technology with Federated Learning offers a promising avenue to enhance security and integrity. In this paper, we tackle the potential of blockchain in defending Federated Learning against adversarial attacks. First, we test Proof of Federated Learning, a well known consensus mechanism designed ad-hoc to federated contexts, as a defense mechanism demonstrating its efficacy against Byzantine and backdoor attacks when at least one miner remains uncompromised. Second, we propose Krum Federated Chain, a novel defense strategy combining Krum and Proof of Federated Learning, valid to defend against any configuration of Byzantine or backdoor attacks, even when all miners are compromised. Our experiments conducted on image classification datasets validate the effectiveness of our proposed approaches.

摘要: 联合学习提供了一种新的机器学习方法，在保护数据隐私的同时，实现了跨分散设备的协作模型培训。然而，它的分布式特性使其容易受到对手的攻击。将区块链技术与联合学习相结合为增强安全性和完整性提供了一条很有前途的途径。在本文中，我们讨论了区块链在保护联邦学习免受对手攻击方面的潜力。首先，我们测试了联合学习的证据，这是一种著名的共识机制，专为联合环境而设计，作为一种防御机制，在至少一个矿工保持不受危害的情况下，展示了其对抗拜占庭和后门攻击的有效性。其次，我们提出了Krum联邦链，这是一种结合了Krum和联合学习证明的新型防御策略，即使在所有矿工都被攻破的情况下，也能有效地防御任何配置的拜占庭攻击或后门攻击。我们在图像分类数据集上进行的实验验证了所提方法的有效性。



## **47. Robust Watermarks Leak: Channel-Aware Feature Extraction Enables Adversarial Watermark Manipulation**

稳健的水印泄露：队列感知特征提取实现对抗性水印操纵 cs.CV

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06418v1) [paper-pdf](http://arxiv.org/pdf/2502.06418v1)

**Authors**: Zhongjie Ba, Yitao Zhang, Peng Cheng, Bin Gong, Xinyu Zhang, Qinglong Wang, Kui Ren

**Abstract**: Watermarking plays a key role in the provenance and detection of AI-generated content. While existing methods prioritize robustness against real-world distortions (e.g., JPEG compression and noise addition), we reveal a fundamental tradeoff: such robust watermarks inherently improve the redundancy of detectable patterns encoded into images, creating exploitable information leakage. To leverage this, we propose an attack framework that extracts leakage of watermark patterns through multi-channel feature learning using a pre-trained vision model. Unlike prior works requiring massive data or detector access, our method achieves both forgery and detection evasion with a single watermarked image. Extensive experiments demonstrate that our method achieves a 60\% success rate gain in detection evasion and 51\% improvement in forgery accuracy compared to state-of-the-art methods while maintaining visual fidelity. Our work exposes the robustness-stealthiness paradox: current "robust" watermarks sacrifice security for distortion resistance, providing insights for future watermark design.

摘要: 水印在人工智能生成的内容的来源和检测中起着关键作用。虽然现有的方法优先考虑对真实世界的扭曲(例如，JPEG压缩和噪声添加)的稳健性，但我们揭示了一个基本的权衡：这种健壮的水印内在地提高了编码到图像中的可检测模式的冗余性，造成了可利用的信息泄漏。为了利用这一点，我们提出了一种攻击框架，该框架使用预先训练的视觉模型通过多通道特征学习来提取水印模式的泄漏。与以往需要访问大量数据或检测器的工作不同，我们的方法实现了对单个水印图像的伪造和检测规避。大量实验表明，该方法在保持视觉保真度的前提下，与现有方法相比，检测规避成功率提高了60%，伪造准确率提高了51%。我们的工作揭示了健壮性-隐蔽性悖论：当前的“健壮性”水印以牺牲安全性为代价来抵抗失真，为未来的水印设计提供了见解。



## **48. Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection**

Adjesia作为增强图像分类和对象检测中黑匣子像素攻击的催化剂 cs.CV

Accepted as a poster at NeurIPS 2024

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07821v1) [paper-pdf](http://arxiv.org/pdf/2502.07821v1)

**Authors**: Dongsu Song, Daehwa Ko, Jay Hoon Jung

**Abstract**: It is well known that query-based attacks tend to have relatively higher success rates in adversarial black-box attacks. While research on black-box attacks is actively being conducted, relatively few studies have focused on pixel attacks that target only a limited number of pixels. In image classification, query-based pixel attacks often rely on patches, which heavily depend on randomness and neglect the fact that scattered pixels are more suitable for adversarial attacks. Moreover, to the best of our knowledge, query-based pixel attacks have not been explored in the field of object detection. To address these issues, we propose a novel pixel-based black-box attack called Remember and Forget Pixel Attack using Reinforcement Learning(RFPAR), consisting of two main components: the Remember and Forget processes. RFPAR mitigates randomness and avoids patch dependency by leveraging rewards generated through a one-step RL algorithm to perturb pixels. RFPAR effectively creates perturbed images that minimize the confidence scores while adhering to limited pixel constraints. Furthermore, we advance our proposed attack beyond image classification to object detection, where RFPAR reduces the confidence scores of detected objects to avoid detection. Experiments on the ImageNet-1K dataset for classification show that RFPAR outperformed state-of-the-art query-based pixel attacks. For object detection, using the MSCOCO dataset with YOLOv8 and DDQ, RFPAR demonstrates comparable mAP reduction to state-of-the-art query-based attack while requiring fewer query. Further experiments on the Argoverse dataset using YOLOv8 confirm that RFPAR effectively removed objects on a larger scale dataset. Our code is available at https://github.com/KAU-QuantumAILab/RFPAR.

摘要: 众所周知，在对抗性黑盒攻击中，基于查询的攻击往往具有相对较高的成功率。虽然对黑匣子攻击的研究正在积极进行，但相对较少的研究集中在仅针对有限数量的像素的像素攻击。在图像分类中，基于查询的像素攻击往往依赖于斑块，这严重依赖于随机性，而忽略了散乱像素更适合于对抗性攻击的事实。此外，据我们所知，基于查询的像素攻击在目标检测领域还没有被探索过。为了解决这些问题，我们提出了一种新的基于像素的黑盒攻击，称为使用强化学习的记住和忘记像素攻击(RFPAR)，该攻击由两个主要部分组成：记住和忘记过程。RFPAR通过利用一步RL算法产生的奖励来扰乱像素，从而减轻了随机性并避免了对补丁的依赖。RFPAR有效地创建扰动图像，使置信度分数最小化，同时遵守有限的像素约束。此外，我们将我们提出的攻击从图像分类推进到目标检测，其中RFPAR降低检测目标的置信度以避免被检测到。在ImageNet-1K数据集上的分类实验表明，RFPAR的性能优于最新的基于查询的像素攻击。在目标检测方面，使用带有YOLOv8和DDQ的MSCOCO数据集，RFPAR表现出与最先进的基于查询的攻击相当的MAP减少，同时需要的查询更少。使用YOLOv8在ArgoVerse数据集上的进一步实验证实，RFPAR有效地去除了较大规模数据集上的对象。我们的代码可以在https://github.com/KAU-QuantumAILab/RFPAR.上找到



## **49. Hyperparameters in Score-Based Membership Inference Attacks**

基于分数的成员推断攻击中的超参数 cs.LG

This work has been accepted for publication in the 3rd IEEE  Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final  version will be available on IEEE Xplore

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06374v1) [paper-pdf](http://arxiv.org/pdf/2502.06374v1)

**Authors**: Gauri Pradhan, Joonas Jälkö, Marlon Tobaben, Antti Honkela

**Abstract**: Membership Inference Attacks (MIAs) have emerged as a valuable framework for evaluating privacy leakage by machine learning models. Score-based MIAs are distinguished, in particular, by their ability to exploit the confidence scores that the model generates for particular inputs. Existing score-based MIAs implicitly assume that the adversary has access to the target model's hyperparameters, which can be used to train the shadow models for the attack. In this work, we demonstrate that the knowledge of target hyperparameters is not a prerequisite for MIA in the transfer learning setting. Based on this, we propose a novel approach to select the hyperparameters for training the shadow models for MIA when the attacker has no prior knowledge about them by matching the output distributions of target and shadow models. We demonstrate that using the new approach yields hyperparameters that lead to an attack near indistinguishable in performance from an attack that uses target hyperparameters to train the shadow models. Furthermore, we study the empirical privacy risk of unaccounted use of training data for hyperparameter optimization (HPO) in differentially private (DP) transfer learning. We find no statistically significant evidence that performing HPO using training data would increase vulnerability to MIA.

摘要: 成员关系推理攻击(MIA)已经成为机器学习模型评估隐私泄露的一个有价值的框架。基于分数的MIA的特别之处在于，它们能够利用模型为特定输入生成的置信度分数。现有的基于分数的MIA隐含地假设对手可以访问目标模型的超参数，这些超参数可以用于训练攻击的影子模型。在这项工作中，我们证明了在迁移学习环境下，目标超参数的知识不是MIA的先决条件。基于此，我们提出了一种新的方法，当攻击者对阴影模型一无所知时，通过匹配目标和阴影模型的输出分布来选择用于训练MIA阴影模型的超参数。我们证明，使用新方法产生的超参数导致的攻击在性能上与使用目标超参数来训练阴影模型的攻击几乎没有区别。此外，我们还研究了在差异私有(DP)迁移学习中使用训练数据进行超参数优化(HPO)时的经验隐私风险。我们没有发现有统计学意义的证据表明，使用训练数据执行HPO会增加MIA的易感性。



## **50. POEX: Understanding and Mitigating Policy Executable Jailbreak Attacks against Embodied AI**

POEX：了解和缓解针对被授权人工智能的政策可执行越狱攻击 cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.16633v2) [paper-pdf](http://arxiv.org/pdf/2412.16633v2)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: Embodied AI systems are rapidly evolving due to the integration of LLMs as planning modules, which transform complex instructions into executable policies. However, LLMs are vulnerable to jailbreak attacks, which can generate malicious content. This paper investigates the feasibility and rationale behind applying traditional LLM jailbreak attacks to EAI systems. We aim to answer three questions: (1) Do traditional LLM jailbreak attacks apply to EAI systems? (2) What challenges arise if they do not? and (3) How can we defend against EAI jailbreak attacks? To this end, we first measure existing LLM-based EAI systems using a newly constructed dataset, i.e., the Harmful-RLbench. Our study confirms that traditional LLM jailbreak attacks are not directly applicable to EAI systems and identifies two unique challenges. First, the harmful text does not necessarily constitute harmful policies. Second, even if harmful policies can be generated, they are not necessarily executable by the EAI systems, which limits the potential risk. To facilitate a more comprehensive security analysis, we refine and introduce POEX, a novel red teaming framework that optimizes adversarial suffixes to induce harmful yet executable policies against EAI systems. The design of POEX employs adversarial constraints, policy evaluators, and suffix optimization to ensure successful policy execution while evading safety detection inside an EAI system. Experiments on the real-world robotic arm and simulator using Harmful-RLbench demonstrate the efficacy, highlighting severe safety vulnerabilities and high transferability across models. Finally, we propose prompt-based and model-based defenses, achieving an 85% success rate in mitigating attacks and enhancing safety awareness in EAI systems. Our findings underscore the urgent need for robust security measures to ensure the safe deployment of EAI in critical applications.

摘要: 由于将LLM作为规划模块进行集成，将复杂的指令转换为可执行的策略，因此具体化人工智能系统正在快速发展。然而，LLMS容易受到越狱攻击，这可能会生成恶意内容。本文研究了将传统的LLM越狱攻击应用于EAI系统的可行性和基本原理。我们的目标是回答三个问题：(1)传统的LLM越狱攻击适用于EAI系统吗？(2)如果不适用，会带来什么挑战？以及(3)我们如何防御EAI越狱攻击？为此，我们首先使用一个新构建的数据集--有害RLbench来度量现有的基于LLM的EAI系统。我们的研究证实了传统的LLM越狱攻击不直接适用于EAI系统，并确定了两个独特的挑战。首先，有害的文本不一定构成有害的政策。其次，即使可以产生有害的政策，它们也不一定可以由EAI系统执行，这限制了潜在的风险。为了便于更全面的安全分析，我们改进并引入了POEX，这是一个新的红色团队框架，它优化了敌意后缀，以诱导针对EAI系统的有害但可执行的策略。POEX的设计采用对抗性约束、策略评估器和后缀优化，以确保成功执行策略，同时避免EAI系统内的安全检测。在真实世界的机械臂和模拟器上的实验证明了该方法的有效性，突出了严重的安全漏洞和高度的跨模型可移植性。最后，我们提出了基于提示和基于模型的防御，在缓解攻击和增强EAI系统的安全意识方面取得了85%的成功率。我们的发现强调了迫切需要强有力的安全措施，以确保在关键应用中安全地部署EAI。



