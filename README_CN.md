# Latest Adversarial Attack Papers
**update at 2025-01-06 09:59:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Familiarity-Based Open-Set Recognition Under Adversarial Attacks**

对抗性攻击下基于家族关系的开放集识别 cs.CV

Published in: Proceedings of the 6th Northern Lights Deep Learning  Conference (NLDL), PMLR 265, 2025

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2311.05006v2) [paper-pdf](http://arxiv.org/pdf/2311.05006v2)

**Authors**: Philip Enevoldsen, Christian Gundersen, Nico Lang, Serge Belongie, Christian Igel

**Abstract**: Open-set recognition (OSR), the identification of novel categories, can be a critical component when deploying classification models in real-world applications. Recent work has shown that familiarity-based scoring rules such as the Maximum Softmax Probability (MSP) or the Maximum Logit Score (MLS) are strong baselines when the closed-set accuracy is high. However, one of the potential weaknesses of familiarity-based OSR are adversarial attacks. Here, we study gradient-based adversarial attacks on familiarity scores for both types of attacks, False Familiarity and False Novelty attacks, and evaluate their effectiveness in informed and uninformed settings on TinyImageNet. Furthermore, we explore how novel and familiar samples react to adversarial attacks and formulate the adversarial reaction score as an alternative OSR scoring rule, which shows a high correlation with the MLS familiarity score.

摘要: 开放集识别（OSR）是新型类别的识别，在现实应用程序中部署分类模型时可能是一个关键组件。最近的工作表明，当闭集准确性较高时，基于熟悉度的评分规则，例如最大Softmax概率（MSP）或最大Logit得分（MLS）是强大的基线。然而，基于熟悉度的OSR的潜在弱点之一是对抗性攻击。在这里，我们研究了基于梯度的对抗攻击对两种攻击（假熟悉性和假新奇性攻击）的熟悉度分数的熟悉度分数，并在TinyImageNet上评估它们在知情和不知情环境中的有效性。此外，我们探索了新颖且熟悉的样本如何对对抗性攻击做出反应，并将对抗性反应评分制定为替代OSR评分规则，该规则与MLS熟悉度评分具有高度相关性。



## **2. SAP: Corrective Machine Unlearning with Scaled Activation Projection for Label Noise Robustness**

SAP：通过缩放激活投影纠正机器遗忘以实现标签噪音稳健性 cs.LG

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2403.08618v2) [paper-pdf](http://arxiv.org/pdf/2403.08618v2)

**Authors**: Sangamesh Kodge, Deepak Ravikumar, Gobinda Saha, Kaushik Roy

**Abstract**: Label corruption, where training samples are mislabeled due to non-expert annotation or adversarial attacks, significantly degrades model performance. Acquiring large, perfectly labeled datasets is costly, and retraining models from scratch is computationally expensive. To address this, we introduce Scaled Activation Projection (SAP), a novel SVD (Singular Value Decomposition)-based corrective machine unlearning algorithm. SAP mitigates label noise by identifying a small subset of trusted samples using cross-entropy loss and projecting model weights onto a clean activation space estimated using SVD on these trusted samples. This process suppresses the noise introduced in activations due to the mislabeled samples. In our experiments, we demonstrate SAP's effectiveness on synthetic noise with different settings and real-world label noise. SAP applied to the CIFAR dataset with 25% synthetic corruption show upto 6% generalization improvements. Additionally, SAP can improve the generalization over noise robust training approaches on CIFAR dataset by ~3.2% on average. Further, we observe generalization improvements of 2.31% for a Vision Transformer model trained on naturally corrupted Clothing1M.

摘要: 标签损坏，其中训练样本由于非专家注释或对抗性攻击而被错误标记，显著降低了模型的性能。获取标记完善的大型数据集的成本很高，从头开始重新训练模型的计算成本也很高。为了解决这个问题，我们引入了一种新的基于奇异值分解(SVD)的纠错机器遗忘算法--比例激活投影(SAP)。SAP通过使用交叉熵损失识别受信任样本的一小部分，并将模型权重投影到使用SVD对这些受信任样本估计的干净激活空间来减轻标签噪声。该过程抑制了由于错误标记的样本而在激活中引入的噪声。在我们的实验中，我们展示了SAP对不同设置的合成噪声和真实世界标签噪声的有效性。SAP应用于CIFAR数据集，具有25%的合成损坏，表现出高达6%的泛化改进。此外，SAP可以将CIFAR数据集上的泛化抗噪稳健训练方法平均提高约3.2%。此外，我们观察到在自然损坏的Clothing1M上训练的Vision Transformer模型的泛化性能提高了2.31%。



## **3. AIM: Additional Image Guided Generation of Transferable Adversarial Attacks**

目标：额外图像引导生成可转移对抗攻击 cs.CV

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01106v1) [paper-pdf](http://arxiv.org/pdf/2501.01106v1)

**Authors**: Teng Li, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Transferable adversarial examples highlight the vulnerability of deep neural networks (DNNs) to imperceptible perturbations across various real-world applications. While there have been notable advancements in untargeted transferable attacks, targeted transferable attacks remain a significant challenge. In this work, we focus on generative approaches for targeted transferable attacks. Current generative attacks focus on reducing overfitting to surrogate models and the source data domain, but they often overlook the importance of enhancing transferability through additional semantics. To address this issue, we introduce a novel plug-and-play module into the general generator architecture to enhance adversarial transferability. Specifically, we propose a \emph{Semantic Injection Module} (SIM) that utilizes the semantics contained in an additional guiding image to improve transferability. The guiding image provides a simple yet effective method to incorporate target semantics from the target class to create targeted and highly transferable attacks. Additionally, we propose new loss formulations that can integrate the semantic injection module more effectively for both targeted and untargeted attacks. We conduct comprehensive experiments under both targeted and untargeted attack settings to demonstrate the efficacy of our proposed approach.

摘要: 可转移的敌意例子突出了深度神经网络(DNN)在各种现实世界应用中对不可察觉的扰动的脆弱性。虽然在非定向转移攻击方面取得了显著进展，但定向转移攻击仍然是一个重大挑战。在这项工作中，我们关注的是针对目标可转移攻击的生成性方法。当前的生成性攻击侧重于减少对代理模型和源数据域的过度匹配，但它们往往忽略了通过额外的语义增强可转移性的重要性。为了解决这个问题，我们在通用生成器体系结构中引入了一种新的即插即用模块，以增强对抗的可转移性。具体地说，我们提出了一种语义注入模块(SIM)，它利用附加引导图像中包含的语义来提高可转移性。引导图像提供了一种简单而有效的方法来结合来自目标类的目标语义来创建有针对性的、高度可转移的攻击。此外，我们提出了新的损失公式，可以更有效地集成语义注入模块，无论是针对定向攻击还是非定向攻击。我们在目标攻击和非目标攻击环境下进行了全面的实验，以证明我们所提出的方法的有效性。



## **4. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

作为入侵者的基于图像的多模式模型：对基于视频的MLLM的可转移多模式攻击 cs.CV

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01042v1) [paper-pdf](http://arxiv.org/pdf/2501.01042v1)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.

摘要: 基于视频的多通道大语言模型(V-MLLM)在视频-文本多通道任务中表现出对敌意例子的脆弱性。然而，对抗性视频是否可以转移到看不见的模型上--这是现实世界中常见和实用的场景--仍未得到探索。在本文中，我们率先对对抗性视频样本在V-MLLMS上的可转移性进行了研究。我们发现，现有的对抗性攻击方法在应用于V-MLLMS的黑盒环境时面临着很大的局限性，我们将其归因于以下缺点：(1)对扰动视频特征缺乏泛化；(2)只关注稀疏关键帧；(3)未能整合多模信息。为了解决这些限制并加深对黑盒场景中V-MLLM漏洞的理解，我们引入了图像到视频MLLM(I2V-MLLM)攻击。在I2V-MLLM中，我们使用基于图像的多模式模型(IMM)作为代理模型来制作对抗性视频样本。多模式交互和时间信息被集成以扰乱潜在空间内的视频表示，提高了对抗性转移。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，该方法能够在多个视频-文本多模式任务的不同V-MLLMS之间生成具有较强可转移性的对抗性实例。与这些模型上的白盒攻击相比，我们的黑盒攻击(以BLIP-2为代理模型)取得了与之相当的性能，对于视频QA任务，MSVD-QA和MSRVTT-QA的平均攻击成功率分别为55.48%和58.26%。我们的代码将在接受后发布。



## **5. Detecting subtle cyberattacks on adaptive cruise control vehicles: A machine learning approach**

检测对自适应巡航控制车辆的微妙网络攻击：机器学习方法 cs.MA

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2310.17091v2) [paper-pdf](http://arxiv.org/pdf/2310.17091v2)

**Authors**: Tianyi Li, Mingfeng Shang, Shian Wang, Raphael Stern

**Abstract**: With the advent of vehicles equipped with advanced driver-assistance systems, such as adaptive cruise control (ACC) and other automated driving features, the potential for cyberattacks on these automated vehicles (AVs) has emerged. While overt attacks that force vehicles to collide may be easily identified, more insidious attacks, which only slightly alter driving behavior, can result in network-wide increases in congestion, fuel consumption, and even crash risk without being easily detected. To address the detection of such attacks, we first present a traffic model framework for three types of potential cyberattacks: malicious manipulation of vehicle control commands, false data injection attacks on sensor measurements, and denial-of-service (DoS) attacks. We then investigate the impacts of these attacks at both the individual vehicle (micro) and traffic flow (macro) levels. A novel generative adversarial network (GAN)-based anomaly detection model is proposed for real-time identification of such attacks using vehicle trajectory data. We provide numerical evidence {to demonstrate} the efficacy of our machine learning approach in detecting cyberattacks on ACC-equipped vehicles. The proposed method is compared against some recently proposed neural network models and observed to have higher accuracy in identifying anomalous driving behaviors of ACC vehicles.

摘要: 随着配备先进的驾驶员辅助系统的车辆的出现，如自适应巡航控制(ACC)和其他自动驾驶功能，针对这些自动车辆(AV)的网络攻击的可能性已经出现。虽然迫使车辆相撞的公开攻击可能很容易识别，但更隐蔽的攻击只会轻微改变驾驶行为，可能会导致网络范围内拥堵、燃油消耗甚至碰撞风险的增加，而不容易被发现。为了解决此类攻击的检测，我们首先提出了一个针对三种潜在网络攻击的流量模型框架：对车辆控制命令的恶意操纵、对传感器测量的虚假数据注入攻击和拒绝服务(DoS)攻击。然后，我们调查这些攻击在单个车辆(微观)和交通流量(宏观)两个层面上的影响。为利用车辆轨迹数据实时识别此类攻击，提出了一种基于产生式对抗网络(GAN)的异常检测模型。我们提供了数字证据，以证明我们的机器学习方法在检测针对配备了ACC的车辆的网络攻击方面的有效性。将该方法与最近提出的几种神经网络模型进行了比较，发现该方法在识别ACC车辆的异常驾驶行为方面具有更高的准确率。



## **6. Towards Adversarially Robust Deep Metric Learning**

迈向对抗稳健的深度度量学习 cs.LG

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01025v1) [paper-pdf](http://arxiv.org/pdf/2501.01025v1)

**Authors**: Xiaopeng Ke

**Abstract**: Deep Metric Learning (DML) has shown remarkable successes in many domains by taking advantage of powerful deep neural networks. Deep neural networks are prone to adversarial attacks and could be easily fooled by adversarial examples. The current progress on this robustness issue is mainly about deep classification models but pays little attention to DML models. Existing works fail to thoroughly inspect the robustness of DML and neglect an important DML scenario, the clustering-based inference. In this work, we first point out the robustness issue of DML models in clustering-based inference scenarios. We find that, for the clustering-based inference, existing defenses designed DML are unable to be reused and the adaptions of defenses designed for deep classification models cannot achieve satisfactory robustness performance. To alleviate the hazard of adversarial examples, we propose a new defense, the Ensemble Adversarial Training (EAT), which exploits ensemble learning and adversarial training. EAT promotes the diversity of the ensemble, encouraging each model in the ensemble to have different robustness features, and employs a self-transferring mechanism to make full use of the robustness statistics of the whole ensemble in the update of every single model. We evaluate the EAT method on three widely-used datasets with two popular model architectures. The results show that the proposed EAT method greatly outperforms the adaptions of defenses designed for deep classification models.

摘要: 深度度量学习(DML)利用了强大的深度神经网络，在许多领域都取得了显著的成功。深度神经网络容易受到对抗性攻击，很容易被对抗性例子愚弄。目前在这一稳健性问题上的研究进展主要集中在深度分类模型上，对DML模型的研究较少。现有的工作没有对DML的健壮性进行彻底的检验，并且忽略了DML的一个重要场景--基于聚类的推理。在这项工作中，我们首先指出了DML模型在基于聚类的推理场景中的健壮性问题。我们发现，对于基于聚类的推理，现有的防御设计的DML不能被重用，并且针对深度分类模型的防御的适应性不能达到令人满意的健壮性。为了减少对抗性例子的危害，我们提出了一种新的防御方法--集成对抗性训练(EAT)，它利用了集成学习和对抗性训练。EAT促进了系综的多样性，鼓励系综中的每个模型具有不同的稳健性特征，并采用自迁移机制在每个单个模型的更新中充分利用整个系综的稳健性统计。我们在三个广泛使用的数据集和两个流行的模型体系结构上对EAT方法进行了评估。结果表明，提出的EAT方法的性能大大优于针对深度分类模型设计的防御方法。



## **7. Region-Guided Attack on the Segment Anything Model (SAM)**

对分段任意模型（Sam）的区域引导攻击 cs.CV

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2411.02974v3) [paper-pdf](http://arxiv.org/pdf/2411.02974v3)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao

**Abstract**: The Segment Anything Model (SAM) is a cornerstone of image segmentation, demonstrating exceptional performance across various applications, particularly in autonomous driving and medical imaging, where precise segmentation is crucial. However, SAM is vulnerable to adversarial attacks that can significantly impair its functionality through minor input perturbations. Traditional techniques, such as FGSM and PGD, are often ineffective in segmentation tasks due to their reliance on global perturbations that overlook spatial nuances. Recent methods like Attack-SAM-K and UAD have begun to address these challenges, but they frequently depend on external cues and do not fully leverage the structural interdependencies within segmentation processes. This limitation underscores the need for a novel adversarial strategy that exploits the unique characteristics of segmentation tasks. In response, we introduce the Region-Guided Attack (RGA), designed specifically for SAM. RGA utilizes a Region-Guided Map (RGM) to manipulate segmented regions, enabling targeted perturbations that fragment large segments and expand smaller ones, resulting in erroneous outputs from SAM. Our experiments demonstrate that RGA achieves high success rates in both white-box and black-box scenarios, emphasizing the need for robust defenses against such sophisticated attacks. RGA not only reveals SAM's vulnerabilities but also lays the groundwork for developing more resilient defenses against adversarial threats in image segmentation.

摘要: Segment Anything Model(SAM)是图像分割的基石，在各种应用中表现出卓越的性能，特别是在自动驾驶和医学成像中，准确的分割至关重要。然而，SAM很容易受到对抗性攻击，这些攻击可能会通过微小的输入扰动显著损害其功能。传统的分割技术，如FGSM和PGD，在分割任务中往往是无效的，因为它们依赖于忽略空间细微差别的全局扰动。最近的方法，如攻击-SAM-K和UAD已经开始解决这些挑战，但它们经常依赖外部线索，并且没有充分利用分割过程中的结构相互依赖。这一限制强调了需要一种利用分段任务的独特特征的新的对抗性策略。作为回应，我们引入了专门为SAM设计的区域制导攻击(RGA)。RGA利用区域引导地图(RGM)来处理分割的区域，从而实现了将大片段分割并扩展小片段的有针对性的扰动，从而导致SAM的错误输出。我们的实验表明，RGA在白盒和黑盒场景中都取得了很高的成功率，强调了对这种复杂攻击的稳健防御的必要性。RGA不仅揭示了SAM的漏洞，而且为开发更具弹性的防御图像分割中的对抗性威胁奠定了基础。



## **8. Boosting Adversarial Transferability with Spatial Adversarial Alignment**

通过空间对抗对齐增强对抗可移植性 cs.CV

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01015v1) [paper-pdf](http://arxiv.org/pdf/2501.01015v1)

**Authors**: Zhaoyu Chen, Haijing Guo, Kaixun Jiang, Jiyuan Fu, Xinyu Zhou, Dingkang Yang, Hao Tang, Bo Li, Wenqiang Zhang

**Abstract**: Deep neural networks are vulnerable to adversarial examples that exhibit transferability across various models. Numerous approaches are proposed to enhance the transferability of adversarial examples, including advanced optimization, data augmentation, and model modifications. However, these methods still show limited transferability, particularly in cross-architecture scenarios, such as from CNN to ViT. To achieve high transferability, we propose a technique termed Spatial Adversarial Alignment (SAA), which employs an alignment loss and leverages a witness model to fine-tune the surrogate model. Specifically, SAA consists of two key parts: spatial-aware alignment and adversarial-aware alignment. First, we minimize the divergences of features between the two models in both global and local regions, facilitating spatial alignment. Second, we introduce a self-adversarial strategy that leverages adversarial examples to impose further constraints, aligning features from an adversarial perspective. Through this alignment, the surrogate model is trained to concentrate on the common features extracted by the witness model. This facilitates adversarial attacks on these shared features, thereby yielding perturbations that exhibit enhanced transferability. Extensive experiments on various architectures on ImageNet show that aligned surrogate models based on SAA can provide higher transferable adversarial examples, especially in cross-architecture attacks.

摘要: 深度神经网络很容易受到敌意例子的攻击，这些例子表现出跨各种模型的可转移性。人们提出了许多方法来提高对抗性例子的可转移性，包括高级优化、数据增强和模型修改。然而，这些方法仍然显示出有限的可转移性，特别是在跨架构的情况下，例如从CNN到VIT。为了实现高可转移性，我们提出了一种称为空间对抗对齐(SAA)的技术，该技术利用对齐损失并利用证人模型来微调代理模型。具体地说，SAA包括两个关键部分：空间感知对齐和对抗性对齐。首先，我们最小化了两个模型在全局和局部区域的特征差异，促进了空间对齐。第二，我们引入了一种自我对抗的策略，利用对抗的例子来施加进一步的限制，从对抗的角度对特征进行对齐。通过这种比对，代理模型被训练成专注于由证人模型提取的共同特征。这促进了对这些共享功能的敌意攻击，从而产生了表现出增强的可转移性的扰动。在ImageNet上的各种架构上的大量实验表明，基于SAA的对齐代理模型能够提供更高可转移性的对抗性实例，特别是在跨架构攻击中。



## **9. A Survey of Secure Semantic Communications**

安全语义通信综述 cs.CR

123 pages, 27 figures

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00842v1) [paper-pdf](http://arxiv.org/pdf/2501.00842v1)

**Authors**: Rui Meng, Song Gao, Dayu Fan, Haixiao Gao, Yining Wang, Xiaodong Xu, Bizhu Wang, Suyu Lv, Zhidi Zhang, Mengying Sun, Shujun Han, Chen Dong, Xiaofeng Tao, Ping Zhang

**Abstract**: Semantic communication (SemCom) is regarded as a promising and revolutionary technology in 6G, aiming to transcend the constraints of ``Shannon's trap" by filtering out redundant information and extracting the core of effective data. Compared to traditional communication paradigms, SemCom offers several notable advantages, such as reducing the burden on data transmission, enhancing network management efficiency, and optimizing resource allocation. Numerous researchers have extensively explored SemCom from various perspectives, including network architecture, theoretical analysis, potential technologies, and future applications. However, as SemCom continues to evolve, a multitude of security and privacy concerns have arisen, posing threats to the confidentiality, integrity, and availability of SemCom systems. This paper presents a comprehensive survey of the technologies that can be utilized to secure SemCom. Firstly, we elaborate on the entire life cycle of SemCom, which includes the model training, model transfer, and semantic information transmission phases. Then, we identify the security and privacy issues that emerge during these three stages. Furthermore, we summarize the techniques available to mitigate these security and privacy threats, including data cleaning, robust learning, defensive strategies against backdoor attacks, adversarial training, differential privacy, cryptography, blockchain technology, model compression, and physical-layer security. Lastly, this paper outlines future research directions to guide researchers in related fields.

摘要: 语义通信(SemCom)被认为是6G中一种很有前途的革命性技术，旨在通过过滤冗余信息和提取有效数据的核心来超越香农陷阱的限制。与传统的通信模式相比，SemCom具有一些显著的优势，如减轻数据传输负担，提高网络管理效率，优化资源配置。许多研究人员从不同的角度对SemCom进行了广泛的探索，包括网络体系结构、理论分析、潜在技术和未来应用。然而，随着SemCom的不断发展，出现了大量的安全和隐私问题，对SemCom系统的机密性、完整性和可用性构成了威胁。本文对可用于确保SemCom安全的技术进行了全面的综述。首先，详细阐述了SemCom的整个生命周期，包括模型训练、模型迁移、语义信息传递等阶段。然后，我们确定在这三个阶段中出现的安全和隐私问题。此外，我们还总结了可用于缓解这些安全和隐私威胁的技术，包括数据清理、稳健学习、针对后门攻击的防御策略、对抗性训练、差异隐私、密码学、区块链技术、模型压缩和物理层安全。最后，本文概述了未来的研究方向，以指导相关领域的研究人员。



## **10. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

基于大型语言模型的搜索引擎的对抗性攻击动态 cs.CL

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00745v1) [paper-pdf](http://arxiv.org/pdf/2501.00745v1)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.

摘要: 基于大型语言模型(LLM)的搜索引擎的日益集成已经改变了信息检索的格局。然而，这些系统容易受到对抗性攻击，特别是排名操纵攻击，攻击者精心编制网页内容来操纵LLM的排名并推广特定内容，从而获得相对于竞争对手的不公平优势。本文研究了排名操纵攻击的动态特性。我们将这个问题描述为一个无限重复的囚徒困境，其中多个参与者战略性地决定是合作还是攻击。我们分析了合作能够持续的条件，确定了影响玩家行为的关键因素，如攻击成本、折扣率、攻击成功率和触发策略。我们确定了系统动态中的引爆点，表明当参与者具有前瞻性时，合作更有可能持续下去。然而，从防御的角度来看，我们发现，矛盾的是，仅仅降低攻击成功的概率就可以在某些条件下激励攻击。此外，在某些情况下，为攻击成功率上限设定上限的防御措施可能被证明是徒劳的。这些见解突显了保护基于LLM的系统的复杂性。我们的工作为理解和缓解它们的漏洞提供了理论基础和实践见解，同时强调了自适应安全策略和深思熟虑的生态系统设计的重要性。



## **11. Everywhere Attack: Attacking Locally and Globally to Boost Targeted Transferability**

无处不在攻击：在本地和全球范围内攻击以提高目标可转移性 cs.CV

11 pages, 6 figures, 8 tables, accepted by 2025AAAI

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00707v1) [paper-pdf](http://arxiv.org/pdf/2501.00707v1)

**Authors**: Hui Zeng, Sanshuai Cui, Biwei Chen, Anjie Peng

**Abstract**: Adversarial examples' (AE) transferability refers to the phenomenon that AEs crafted with one surrogate model can also fool other models. Notwithstanding remarkable progress in untargeted transferability, its targeted counterpart remains challenging. This paper proposes an everywhere scheme to boost targeted transferability. Our idea is to attack a victim image both globally and locally. We aim to optimize 'an army of targets' in every local image region instead of the previous works that optimize a high-confidence target in the image. Specifically, we split a victim image into non-overlap blocks and jointly mount a targeted attack on each block. Such a strategy mitigates transfer failures caused by attention inconsistency between surrogate and victim models and thus results in stronger transferability. Our approach is method-agnostic, which means it can be easily combined with existing transferable attacks for even higher transferability. Extensive experiments on ImageNet demonstrate that the proposed approach universally improves the state-of-the-art targeted attacks by a clear margin, e.g., the transferability of the widely adopted Logit attack can be improved by 28.8%-300%.We also evaluate the crafted AEs on a real-world platform: Google Cloud Vision. Results further support the superiority of the proposed method.

摘要: 对抗性例子(AE)的可转移性是指用一个代理模型制作的AE也可以愚弄其他模型的现象。尽管在非目标可转让性方面取得了显著进展，但目标对应方仍然具有挑战性。本文提出了一种Everywhere方案来提高定向可转移性。我们的想法是在全球和当地攻击受害者形象。我们的目标是优化每个局部图像区域的目标大军，而不是以前在图像中优化高置信度目标的工作。具体地说，我们将受害者图像分割成不重叠的块，并联合对每个块发动有针对性的攻击。这样的策略减少了由于代理模型和受害者模型之间的注意不一致而导致的迁移失败，从而导致更强的迁移能力。我们的方法是与方法无关的，这意味着它可以很容易地与现有的可转移攻击相结合，从而获得更高的可转移性。在ImageNet上的大量实验表明，该方法在整体上明显改善了最新的定向攻击，例如，被广泛采用的Logit攻击的可转移性可以提高28.8%-300%。我们还在真实的Google Cloud Vision平台上对定制的攻击引擎进行了评估。结果进一步支持了该方法的优越性。



## **12. Privacy-Preserving Distributed Defense Framework for DC Microgrids Against Exponentially Unbounded False Data Injection Attacks**

针对指数无界虚假数据注入攻击的DC微电网保护隐私分布式防御框架 eess.SY

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2501.00588v1) [paper-pdf](http://arxiv.org/pdf/2501.00588v1)

**Authors**: Yi Zhang, Mohamadamin Rajabinezhad, Yichao Wang, Junbo Zhao, Shan Zuo

**Abstract**: This paper introduces a novel, fully distributed control framework for DC microgrids, enhancing resilience against exponentially unbounded false data injection (EU-FDI) attacks. Our framework features a consensus-based secondary control for each converter, effectively addressing these advanced threats. To further safeguard sensitive operational data, a privacy-preserving mechanism is incorporated into the control design, ensuring that critical information remains secure even under adversarial conditions. Rigorous Lyapunov stability analysis confirms the framework's ability to maintain critical DC microgrid operations like voltage regulation and load sharing under EU-FDI threats. The framework's practicality is validated through hardware-in-the-loop experiments, demonstrating its enhanced resilience and robust privacy protection against the complex challenges posed by quick variant FDI attacks.

摘要: 本文介绍了一种新颖的、完全分布式的DC微电网控制框架，增强了抵御指数无界虚假数据注入（EU-Direct）攻击的弹性。我们的框架为每个转换器提供了基于共识的二级控制，可以有效地解决这些高级威胁。为了进一步保护敏感的运营数据，控制设计中纳入了隐私保护机制，确保关键信息即使在敌对条件下也保持安全。严格的李亚普诺夫稳定性分析证实了该框架在欧盟外国直接投资威胁下维持关键的直流微电网运营的能力，例如电压调节和负载共享。该框架的实用性通过硬件在环实验得到了验证，展示了其增强的弹性和强大的隐私保护，以应对快速变体外国直接投资攻击带来的复杂挑战。



## **13. Extending XReason: Formal Explanations for Adversarial Detection**

扩展XReason：对抗性检测的正式解释 cs.AI

International Congress on Information and Communication Technology  (ICICT), Lecture Notes in Networks and Systems (LNNS), Springer, 2025

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2501.00537v1) [paper-pdf](http://arxiv.org/pdf/2501.00537v1)

**Authors**: Amira Jemaa, Adnan Rashid, Sofiene Tahar

**Abstract**: Explainable Artificial Intelligence (XAI) plays an important role in improving the transparency and reliability of complex machine learning models, especially in critical domains such as cybersecurity. Despite the prevalence of heuristic interpretation methods such as SHAP and LIME, these techniques often lack formal guarantees and may produce inconsistent local explanations. To fulfill this need, few tools have emerged that use formal methods to provide formal explanations. Among these, XReason uses a SAT solver to generate formal instance-level explanation for XGBoost models. In this paper, we extend the XReason tool to support LightGBM models as well as class-level explanations. Additionally, we implement a mechanism to generate and detect adversarial examples in XReason. We evaluate the efficiency and accuracy of our approach on the CICIDS-2017 dataset, a widely used benchmark for detecting network attacks.

摘要: 可解释人工智能（XAI）在提高复杂机器学习模型的透明度和可靠性方面发挥着重要作用，特别是在网络安全等关键领域。尽管SHAP和LIME等启发式解释方法盛行，但这些技术通常缺乏正式保证，并且可能会产生不一致的局部解释。为了满足这一需求，很少出现使用形式方法来提供形式解释的工具。其中，XReason使用SAT求解器来为XGBOP模型生成正式的实例级解释。在本文中，我们扩展了XReason工具以支持LightGBM模型以及类级解释。此外，我们还在XReason中实现了一种生成和检测对抗示例的机制。我们在CICIDS-2017数据集上评估了我们方法的效率和准确性，CICIDS-2017数据集是一个广泛使用的检测网络攻击的基准。



## **14. From Sands to Mansions: Simulating Full Attack Chain with LLM-Organized Knowledge**

从金沙到豪宅：利用法学硕士组织的知识模拟完整攻击链 cs.CR

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2407.16928v2) [paper-pdf](http://arxiv.org/pdf/2407.16928v2)

**Authors**: Lingzhi Wang, Zhenyuan Li, Zonghan Guo, Yi Jiang, Kyle Jung, Kedar Thiagarajan, Jiahui Wang, Zhengkai Wang, Emily Wei, Xiangmin Shen, Yan Chen

**Abstract**: Adversarial dynamics are intrinsic to the nature of offense and defense in cyberspace, with both attackers and defenders continuously evolving their technologies. Given the wide array of security products available, users often face challenges in selecting the most effective solutions. Furthermore, traditional benchmarks based on single-point attacks are increasingly inadequate, failing to accurately reflect the full range of attacker capabilities and falling short in properly evaluating the effectiveness of defense products. Automated multi-stage attack simulations offer a promising approach to enhance system evaluation efficiency and aid in analyzing the effectiveness of detection systems. However, simulating a full attack chain is complex and requires significant time and expertise from security professionals, facing several challenges, including limited coverage of attack techniques, a high level of required expertise, and a lack of execution detail. In this paper, we model automatic attack simulation as a planning problem. By using the Planning Domain Definition Language (PDDL) to formally describe the attack simulation problem, and combining domain knowledge of both the problem and the domain space, we enable the planning of attack paths through standardized, domain-independent planning algorithms. We explore the potential of Large Language Models (LLMs) to summarize and analyze knowledge from existing attack documentation and reports, facilitating automated attack planning. We introduce Aurora, a system that autonomously simulates full attack chains based on external attack tools and threat intelligence reports.

摘要: 对抗动态是网络空间进攻和防御的本质所固有的，攻击者和防御者都在不断地发展他们的技术。鉴于可用的安全产品种类繁多，用户在选择最有效的解决方案时经常面临挑战。此外，基于单点攻击的传统基准日益不足，无法准确反映攻击者的全方位能力，无法正确评估防御产品的有效性。自动多阶段攻击模拟为提高系统评估效率和辅助分析检测系统的有效性提供了一种很有前途的方法。然而，模拟完整的攻击链是复杂的，需要大量的时间和安全专业人员的专业知识，面临着几个挑战，包括攻击技术的覆盖范围有限，所需专业知识水平较高，以及缺乏执行细节。在本文中，我们将自动攻击模拟建模为一个规划问题。通过使用规划领域定义语言(PDDL)对攻击模拟问题进行形式化描述，并结合问题和领域空间的领域知识，我们能够通过标准化的、与领域无关的规划算法来规划攻击路径。我们探索大型语言模型(LLM)的潜力，以总结和分析现有攻击文档和报告中的知识，从而促进自动攻击规划。我们介绍了Aurora，这是一个基于外部攻击工具和威胁情报报告自主模拟完整攻击链的系统。



## **15. UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models**

UIB扩散：扩散模型的普遍不可感知后门攻击 cs.CR

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2412.11441v2) [paper-pdf](http://arxiv.org/pdf/2412.11441v2)

**Authors**: Yuning Han, Bingyin Zhao, Rui Chu, Feng Luo, Biplab Sikdar, Yingjie Lao

**Abstract**: Recent studies show that diffusion models (DMs) are vulnerable to backdoor attacks. Existing backdoor attacks impose unconcealed triggers (e.g., a gray box and eyeglasses) that contain evident patterns, rendering remarkable attack effects yet easy detection upon human inspection and defensive algorithms. While it is possible to improve stealthiness by reducing the strength of the backdoor, doing so can significantly compromise its generality and effectiveness. In this paper, we propose UIBDiffusion, the universal imperceptible backdoor attack for diffusion models, which allows us to achieve superior attack and generation performance while evading state-of-the-art defenses. We propose a novel trigger generation approach based on universal adversarial perturbations (UAPs) and reveal that such perturbations, which are initially devised for fooling pre-trained discriminative models, can be adapted as potent imperceptible backdoor triggers for DMs. We evaluate UIBDiffusion on multiple types of DMs with different kinds of samplers across various datasets and targets. Experimental results demonstrate that UIBDiffusion brings three advantages: 1) Universality, the imperceptible trigger is universal (i.e., image and model agnostic) where a single trigger is effective to any images and all diffusion models with different samplers; 2) Utility, it achieves comparable generation quality (e.g., FID) and even better attack success rate (i.e., ASR) at low poison rates compared to the prior works; and 3) Undetectability, UIBDiffusion is plausible to human perception and can bypass Elijah and TERD, the SOTA defenses against backdoors for DMs. We will release our backdoor triggers and code.

摘要: 最近的研究表明，扩散模型(DM)容易受到后门攻击。现有的后门攻击施加了包含明显模式的隐藏触发器(例如，灰色盒子和眼镜)，使攻击效果显著，但很容易检测到人工检查和防御算法。虽然可以通过降低后门的强度来提高隐蔽性，但这样做会显著影响后门的通用性和有效性。在本文中，我们提出了一种针对扩散模型的通用不可感知后门攻击--UIB扩散，它使我们能够在避开最先进的防御的同时获得优越的攻击和生成性能。我们提出了一种新的基于通用对抗性扰动(UAP)的触发器生成方法，并揭示了这种扰动最初是为愚弄预训练的区分模型而设计的，现在可以被改装成用于DM的有效的不可感知的后门触发器。我们使用不同类型的采样器在不同的数据集和目标上评估了UIB在多种类型的DM上的扩散。实验结果表明：1)通用性，隐形触发器具有普适性(即，图像和模型无关)，其中单个触发器对任何图像和具有不同采样器的所有扩散模型有效；2)实用性，它在较低的毒害率下获得了与已有工作相当的生成质量(例如，FID)和更好的攻击成功率(例如，ASR)；以及3)不可检测性，UIB扩散对人类感知是可信的，并且可以绕过Elijah和Terd，SOTA对DM的后门防御。我们将发布我们的后门触发器和代码。



## **16. MADE: Graph Backdoor Defense with Masked Unlearning**

MADE：带屏蔽的忘记学习的图形后门防御 cs.CR

15 pages, 10 figures

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2411.18648v2) [paper-pdf](http://arxiv.org/pdf/2411.18648v2)

**Authors**: Xiao Lin, Mingjie Li, Yisen Wang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant attention from researchers due to their outstanding performance in handling graph-related tasks, such as social network analysis, protein design, and so on. Despite their widespread application, recent research has demonstrated that GNNs are vulnerable to backdoor attacks, implemented by injecting triggers into the training datasets. Trained on the poisoned data, GNNs will predict target labels when attaching trigger patterns to inputs. This vulnerability poses significant security risks for applications of GNNs in sensitive domains, such as drug discovery. While there has been extensive research into backdoor defenses for images, strategies to safeguard GNNs against such attacks remain underdeveloped. Furthermore, we point out that conventional backdoor defense methods designed for images cannot work well when directly implemented on graph data. In this paper, we first analyze the key difference between image backdoor and graph backdoor attacks. Then we tackle the graph defense problem by presenting a novel approach called MADE, which devises an adversarial mask generation mechanism that selectively preserves clean sub-graphs and further leverages masks on edge weights to eliminate the influence of triggers effectively. Extensive experiments across various graph classification tasks demonstrate the effectiveness of MADE in significantly reducing the attack success rate (ASR) while maintaining a high classification accuracy.

摘要: 图神经网络(GNN)因其在处理社会网络分析、蛋白质设计等与图相关的任务方面的出色表现而受到研究人员的极大关注。尽管它们被广泛应用，但最近的研究表明，GNN很容易受到后门攻击，这些攻击是通过向训练数据集中注入触发器来实现的。在有毒数据上进行训练后，GNN将在将触发模式附加到输入时预测目标标签。这一漏洞给GNN在药物发现等敏感领域的应用带来了重大的安全风险。虽然已经对图像的后门防御进行了广泛的研究，但保护GNN免受此类攻击的战略仍然不发达。此外，我们还指出，传统的针对图像设计的后门防御方法在直接应用于图形数据时不能很好地工作。本文首先分析了图像后门攻击和图形后门攻击的主要区别。然后，我们提出了一种称为Made的新方法来解决图防御问题，该方法设计了一种对抗性掩码生成机制，该机制选择性地保留干净的子图，并进一步利用边权重的掩码来有效地消除触发因素的影响。在不同的图分类任务上的大量实验表明，Made在显著降低攻击成功率(ASR)的同时保持了较高的分类精度。



## **17. Adversarial Attack and Defense for LoRa Device Identification and Authentication via Deep Learning**

通过深度学习进行LoRa设备识别和认证的对抗性攻击和防御 cs.NI

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21164v1) [paper-pdf](http://arxiv.org/pdf/2412.21164v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek

**Abstract**: LoRa provides long-range, energy-efficient communications in Internet of Things (IoT) applications that rely on Low-Power Wide-Area Network (LPWAN) capabilities. Despite these merits, concerns persist regarding the security of LoRa networks, especially in situations where device identification and authentication are imperative to secure the reliable access to the LoRa networks. This paper explores a deep learning (DL) approach to tackle these concerns, focusing on two critical tasks, namely (i) identifying LoRa devices and (ii) classifying them to legitimate and rogue devices. Deep neural networks (DNNs), encompassing both convolutional and feedforward neural networks, are trained for these tasks using actual LoRa signal data. In this setting, the adversaries may spoof rogue LoRa signals through the kernel density estimation (KDE) method based on legitimate device signals that are received by the adversaries. Two cases are considered, (i) training two separate classifiers, one for each of the two tasks, and (ii) training a multi-task classifier for both tasks. The vulnerabilities of the resulting DNNs to manipulations in input samples are studied in form of untargeted and targeted adversarial attacks using the Fast Gradient Sign Method (FGSM). Individual and common perturbations are considered against single-task and multi-task classifiers for the LoRa signal analysis. To provide resilience against such attacks, a defense approach is presented by increasing the robustness of classifiers with adversarial training. Results quantify how vulnerable LoRa signal classification tasks are to adversarial attacks and emphasize the need to fortify IoT applications against these subtle yet effective threats.

摘要: LORA在依赖低功耗广域网络(LPWAN)功能的物联网(IoT)应用中提供远程、高能效通信。尽管有这些优点，人们仍然对LoRa网络的安全感到担忧，特别是在必须进行设备识别和认证才能确保可靠访问LoRa网络的情况下。本文探索了一种深度学习的方法来解决这些问题，重点关注两个关键任务，即(I)识别LoRa设备和(Ii)将它们分类为合法设备和流氓设备。深度神经网络(DNN)包括卷积神经网络和前馈神经网络，使用实际的LORA信号数据进行训练。在这种情况下，攻击者可以通过基于攻击者接收到的合法设备信号的核密度估计(KDE)方法来欺骗恶意LORA信号。考虑了两种情况，(I)训练两个单独的分类器，两个任务中的每一个一个，以及(Ii)为两个任务训练一个多任务分类器。使用快速梯度符号方法(FGSM)，以非目标攻击和目标攻击的形式研究了所得到的DNN对输入样本中的操纵的脆弱性。在LORA信号分析中，针对单任务和多任务分类器，考虑了单个和共同的扰动。为了提供对此类攻击的恢复能力，提出了一种通过对抗性训练提高分类器的稳健性的防御方法。结果量化了LORA信号分类任务面对对手攻击的脆弱性，并强调需要加强物联网应用程序以抵御这些微妙但有效的威胁。



## **18. BridgePure: Revealing the Fragility of Black-box Data Protection**

BridgePure：揭示黑匣子数据保护的脆弱性 cs.LG

26 pages,13 figures

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21061v1) [paper-pdf](http://arxiv.org/pdf/2412.21061v1)

**Authors**: Yihan Wang, Yiwei Lu, Xiao-Shan Gao, Gautam Kamath, Yaoliang Yu

**Abstract**: Availability attacks, or unlearnable examples, are defensive techniques that allow data owners to modify their datasets in ways that prevent unauthorized machine learning models from learning effectively while maintaining the data's intended functionality. It has led to the release of popular black-box tools for users to upload personal data and receive protected counterparts. In this work, we show such black-box protections can be substantially bypassed if a small set of unprotected in-distribution data is available. Specifically, an adversary can (1) easily acquire (unprotected, protected) pairs by querying the black-box protections with the unprotected dataset; and (2) train a diffusion bridge model to build a mapping. This mapping, termed BridgePure, can effectively remove the protection from any previously unseen data within the same distribution. Under this threat model, our method demonstrates superior purification performance on classification and style mimicry tasks, exposing critical vulnerabilities in black-box data protection.

摘要: 可用性攻击，或无法学习的例子，是一种防御性技术，允许数据所有者修改他们的数据集，以防止未经授权的机器学习模型有效学习，同时保持数据的预期功能。这导致了流行的黑盒工具的发布，用户可以上传个人数据并接收受保护的对应数据。在这项工作中，我们表明，如果有一小部分未受保护的分发中数据可用，则可以实质上绕过此类黑盒保护。具体地说，敌手可以(1)通过使用未受保护的数据集查询黑盒保护来容易地获得(未受保护的，受保护的)对；以及(2)训练扩散桥模型来建立映射。这种名为BridgePure的映射可以有效地消除对同一分发中任何以前不可见的数据的保护。在这种威胁模型下，我们的方法在分类和风格模仿任务上表现出了优越的净化性能，暴露了黑盒数据保护中的关键漏洞。



## **19. RobustBlack: Challenging Black-Box Adversarial Attacks on State-of-the-Art Defenses**

RobustBlack：对抗对最先进防御的黑匣子对抗攻击 cs.LG

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20987v1) [paper-pdf](http://arxiv.org/pdf/2412.20987v1)

**Authors**: Mohamed Djilani, Salah Ghamizi, Maxime Cordy

**Abstract**: Although adversarial robustness has been extensively studied in white-box settings, recent advances in black-box attacks (including transfer- and query-based approaches) are primarily benchmarked against weak defenses, leaving a significant gap in the evaluation of their effectiveness against more recent and moderate robust models (e.g., those featured in the Robustbench leaderboard). In this paper, we question this lack of attention from black-box attacks to robust models. We establish a framework to evaluate the effectiveness of recent black-box attacks against both top-performing and standard defense mechanisms, on the ImageNet dataset. Our empirical evaluation reveals the following key findings: (1) the most advanced black-box attacks struggle to succeed even against simple adversarially trained models; (2) robust models that are optimized to withstand strong white-box attacks, such as AutoAttack, also exhibits enhanced resilience against black-box attacks; and (3) robustness alignment between the surrogate models and the target model plays a key factor in the success rate of transfer-based attacks

摘要: 尽管在白盒环境中已经对对抗健壮性进行了广泛的研究，但黑盒攻击(包括基于转移和基于查询的方法)的最新进展主要是以弱防御为基准的，在评估其有效性方面与较新的中等健壮性模型(例如，罗布斯堡垒排行榜中的那些模型)相比存在很大差距。在这篇文章中，我们质疑这种从黑箱攻击到稳健模型的缺乏关注。我们建立了一个框架来评估最近针对ImageNet数据集上最高性能和标准防御机制的黑盒攻击的有效性。我们的经验评估揭示了以下关键发现：(1)最高级的黑盒攻击即使在简单的对抗性训练模型下也很难成功；(2)经过优化以抵抗强大的白盒攻击的健壮模型，例如AutoAttack，也表现出对黑盒攻击的更强的弹性；以及(3)代理模型和目标模型之间的健壮性对齐在基于转移的攻击的成功率中起着关键作用



## **20. GASLITEing the Retrieval: Exploring Vulnerabilities in Dense Embedding-based Search**

GASLITEING检索：探索基于密集嵌入的搜索中的漏洞 cs.CR

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20953v1) [paper-pdf](http://arxiv.org/pdf/2412.20953v1)

**Authors**: Matan Ben-Tov, Mahmood Sharif

**Abstract**: Dense embedding-based text retrieval$\unicode{x2013}$retrieval of relevant passages from corpora via deep learning encodings$\unicode{x2013}$has emerged as a powerful method attaining state-of-the-art search results and popularizing the use of Retrieval Augmented Generation (RAG). Still, like other search methods, embedding-based retrieval may be susceptible to search-engine optimization (SEO) attacks, where adversaries promote malicious content by introducing adversarial passages to corpora. To faithfully assess and gain insights into the susceptibility of such systems to SEO, this work proposes the GASLITE attack, a mathematically principled gradient-based search method for generating adversarial passages without relying on the corpus content or modifying the model. Notably, GASLITE's passages (1) carry adversary-chosen information while (2) achieving high retrieval ranking for a selected query distribution when inserted to corpora. We use GASLITE to extensively evaluate retrievers' robustness, testing nine advanced models under varied threat models, while focusing on realistic adversaries targeting queries on a specific concept (e.g., a public figure). We found GASLITE consistently outperformed baselines by $\geq$140% success rate, in all settings. Particularly, adversaries using GASLITE require minimal effort to manipulate search results$\unicode{x2013}$by injecting a negligible amount of adversarial passages ($\leq$0.0001% of the corpus), they could make them visible in the top-10 results for 61-100% of unseen concept-specific queries against most evaluated models. Inspecting variance in retrievers' robustness, we identify key factors that may contribute to models' susceptibility to SEO, including specific properties in the embedding space's geometry.

摘要: 基于密集嵌入的文本检索$\unicode{x2013}$通过深度学习编码从语料库中检索相关段落$\unicode{x2013}$已成为获得最先进的搜索结果并普及检索增强一代(RAG)的一种强大方法。尽管如此，像其他搜索方法一样，基于嵌入的检索可能容易受到搜索引擎优化(SEO)攻击，即对手通过向语料库引入敌意段落来推广恶意内容。为了忠实地评估和洞察这类系统对SEO的敏感性，本文提出了GASLITE攻击，这是一种基于数学原理的基于梯度的搜索方法，可以在不依赖语料库内容或修改模型的情况下生成对抗性段落。值得注意的是，GASLITE的段落(1)携带对手选择的信息，而(2)在插入到语料库时，对选定的查询分布实现了较高的检索排名。我们使用GASLITE来广泛地评估检索器的健壮性，在不同的威胁模型下测试了九个高级模型，同时专注于针对特定概念(例如，公众人物)的查询的现实对手。我们发现，在所有情况下，GASLITE的成功率都比基线高出140%。特别是，使用GASLITE的攻击者只需很少的努力就可以操纵搜索结果$\unicode{x2013}$通过注入微不足道的对抗性段落($\leq$0.0001%的语料库)，他们可以使它们在针对大多数评估模型的未见概念特定查询的前10个结果中可见。通过考察检索者稳健性的差异，我们确定了可能导致模型对SEO易感性的关键因素，包括嵌入空间几何结构中的特定属性。



## **21. Two Heads Are Better Than One: Averaging along Fine-Tuning to Improve Targeted Transferability**

两个头总比一个头好：利用微调来提高目标可转让性 cs.CV

9 pages, 6 figures, accepted by 2025ICASSP

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20807v1) [paper-pdf](http://arxiv.org/pdf/2412.20807v1)

**Authors**: Hui Zeng, Sanshuai Cui, Biwei Chen, Anjie Peng

**Abstract**: With much longer optimization time than that of untargeted attacks notwithstanding, the transferability of targeted attacks is still far from satisfactory. Recent studies reveal that fine-tuning an existing adversarial example (AE) in feature space can efficiently boost its targeted transferability. However, existing fine-tuning schemes only utilize the endpoint and ignore the valuable information in the fine-tuning trajectory. Noting that the vanilla fine-tuning trajectory tends to oscillate around the periphery of a flat region of the loss surface, we propose averaging over the fine-tuning trajectory to pull the crafted AE towards a more centered region. We compare the proposed method with existing fine-tuning schemes by integrating them with state-of-the-art targeted attacks in various attacking scenarios. Experimental results uphold the superiority of the proposed method in boosting targeted transferability. The code is available at github.com/zengh5/Avg_FT.

摘要: 尽管优化时间比非目标攻击长得多，但目标攻击的可转移性仍然远不能令人满意。最近的研究表明，微调特征空间中现有的对抗性示例（AE）可以有效地提高其目标可移植性。然而，现有的微调方案只利用端点而忽略了微调轨迹中的有价值信息。注意到普通微调轨迹往往会围绕损失表面平坦区域的外围振荡，我们建议对微调轨迹进行平均，以将精心制作的AE拉向更中心的区域。我们通过将所提出的方法与现有的微调方案与各种攻击场景中的最先进的定向攻击相结合，将其与现有的微调方案进行比较。实验结果证实了所提出的方法在提高目标可移植性方面的优越性。该代码可在github.com/zengh5/Avg_FT上获取。



## **22. DV-FSR: A Dual-View Target Attack Framework for Federated Sequential Recommendation**

DV-FSR：一种用于联合顺序推荐的双视图目标攻击框架 cs.CR

I am requesting the withdrawal of my paper due to identified errors  that require significant revision

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2409.07500v2) [paper-pdf](http://arxiv.org/pdf/2409.07500v2)

**Authors**: Qitao Qin, Yucong Luo, Mingyue Cheng, Qingyang Mao, Chenyi Lei

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation (FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models.

摘要: 联邦推荐(FedRec)通过支持个性化模型的分散训练来保护用户隐私，但这种体系结构天生就容易受到敌意攻击。出于商业和社会影响的考虑，对FedRec系统中的目标攻击进行了重要的研究。然而，这些工作在很大程度上忽略了推荐模型的差异化稳健性。此外，我们的实验结果表明，现有的定向攻击方法在联邦顺序推荐(FSR)任务中只能取得有限的效果。在此基础上，我们重点研究了FSR中的目标攻击，并提出了一种新的DualView攻击框架DV-FSR。该攻击方法独特地结合了基于采样的显式策略和基于对比学习的隐式梯度策略来协调攻击。此外，我们在FSR中引入了一种针对目标攻击的特定防御机制，旨在评估我们提出的攻击方法的缓解效果。在典型的序列模型上进行了大量的实验，验证了该方法的有效性。



## **23. Sample Correlation for Fingerprinting Deep Face Recognition**

指纹深度人脸识别的样本相关性 cs.CV

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20768v1) [paper-pdf](http://arxiv.org/pdf/2412.20768v1)

**Authors**: Jiyang Guan, Jian Liang, Yanbo Wang, Ran He

**Abstract**: Face recognition has witnessed remarkable advancements in recent years, thanks to the development of deep learning techniques.However, an off-the-shelf face recognition model as a commercial service could be stolen by model stealing attacks, posing great threats to the rights of the model owner.Model fingerprinting, as a model stealing detection method, aims to verify whether a suspect model is stolen from the victim model, gaining more and more attention nowadays.Previous methods always utilize transferable adversarial examples as the model fingerprint, but this method is known to be sensitive to adversarial defense and transfer learning techniques.To address this issue, we consider the pairwise relationship between samples instead and propose a novel yet simple model stealing detection method based on SAmple Correlation (SAC).Specifically, we present SAC-JC that selects JPEG compressed samples as model inputs and calculates the correlation matrix among their model outputs.Extensive results validate that SAC successfully defends against various model stealing attacks in deep face recognition, encompassing face verification and face emotion recognition, exhibiting the highest performance in terms of AUC, p-value and F1 score.Furthermore, we extend our evaluation of SAC-JC to object recognition datasets including Tiny-ImageNet and CIFAR10, which also demonstrates the superior performance of SAC-JC to previous methods.The code will be available at \url{https://github.com/guanjiyang/SAC_JC}.

摘要: 近年来，随着深度学习技术的发展，人脸识别技术取得了显著的进步。然而，现有的人脸识别模型作为一种商业服务，可能会被模型窃取攻击窃取，这对模型所有者的权利构成了极大的威胁。模型指纹作为一种模型窃取检测方法，旨在验证嫌疑人模型是否从受害者模型中被盗，受到越来越多的关注。以前的方法通常使用可转移的对手样本作为模型指纹，但这种方法对攻击防御和转移学习技术非常敏感。针对这一问题，提出了一种基于模型指纹的人脸识别方法考虑了样本之间的成对关系，提出了一种新颖而简单的基于样本相关的模型窃取检测方法(SAC-JC)。具体地，我们选择JPEG压缩样本作为模型输入，并计算模型输出之间的相关矩阵。大量的实验结果验证了SAC-JC在深度人脸识别中成功地抵抗了各种模型窃取攻击，包括人脸验证和人脸情感识别，在AUC、p值和F1得分方面表现出最高的性能。此外，我们将SAC-JC的评估扩展到目标识别数据集，包括微小图像网和CIFAR10，这也展示了SAC-JC相对于以前方法的卓越性能。代码将在\url{https://github.com/guanjiyang/SAC_JC}.



## **24. Enhancing Privacy in Federated Learning through Quantum Teleportation Integration**

通过量子隐形传输集成增强联邦学习中的隐私 quant-ph

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20762v1) [paper-pdf](http://arxiv.org/pdf/2412.20762v1)

**Authors**: Koffka Khan

**Abstract**: Federated learning enables collaborative model training across multiple clients without sharing raw data, thereby enhancing privacy. However, the exchange of model updates can still expose sensitive information. Quantum teleportation, a process that transfers quantum states between distant locations without physical transmission of the particles themselves, has recently been implemented in real-world networks. This position paper explores the potential of integrating quantum teleportation into federated learning frameworks to bolster privacy. By leveraging quantum entanglement and the no-cloning theorem, quantum teleportation ensures that data remains secure during transmission, as any eavesdropping attempt would be detectable. We propose a novel architecture where quantum teleportation facilitates the secure exchange of model parameters and gradients among clients and servers. This integration aims to mitigate risks associated with data leakage and adversarial attacks inherent in classical federated learning setups. We also discuss the practical challenges of implementing such a system, including the current limitations of quantum network infrastructure and the need for hybrid quantum-classical protocols. Our analysis suggests that, despite these challenges, the convergence of quantum communication technologies and federated learning presents a promising avenue for achieving unprecedented levels of privacy in distributed machine learning.

摘要: 联合学习实现了跨多个客户的协作模型培训，而无需共享原始数据，从而增强了隐私。然而，模型更新的交换仍然可能暴露敏感信息。量子隐形传态是一种在遥远的位置之间传输量子态的过程，而不需要粒子本身的物理传输，最近已经在现实世界的网络中实现。这份立场文件探索了将量子隐形传态整合到联邦学习框架中以保护隐私的潜力。通过利用量子纠缠和不可克隆定理，量子隐形传态确保了数据在传输过程中保持安全，因为任何窃听企图都是可以检测到的。我们提出了一种新的体系结构，其中量子隐形传态促进了客户端和服务器之间模型参数和梯度的安全交换。这种集成旨在降低与传统联合学习设置中固有的数据泄露和对抗性攻击相关的风险。我们还讨论了实现这样一个系统的实际挑战，包括目前量子网络基础设施的限制以及对混合量子经典协议的需求。我们的分析表明，尽管存在这些挑战，但量子通信技术和联邦学习的融合为在分布式机器学习中实现前所未有的隐私水平提供了一条有希望的途径。



## **25. Unsupervised dense retrieval with conterfactual contrastive learning**

具有反事实对比学习的无监督密集检索 cs.IR

arXiv admin note: text overlap with arXiv:2107.07773 by other authors

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20756v1) [paper-pdf](http://arxiv.org/pdf/2412.20756v1)

**Authors**: Haitian Chen, Qingyao Ai, Xiao Wang, Yiqun Liu, Fen Lin, Qin Liu

**Abstract**: Efficiently retrieving a concise set of candidates from a large document corpus remains a pivotal challenge in Information Retrieval (IR). Neural retrieval models, particularly dense retrieval models built with transformers and pretrained language models, have been popular due to their superior performance. However, criticisms have also been raised on their lack of explainability and vulnerability to adversarial attacks. In response to these challenges, we propose to improve the robustness of dense retrieval models by enhancing their sensitivity of fine-graned relevance signals. A model achieving sensitivity in this context should exhibit high variances when documents' key passages determining their relevance to queries have been modified, while maintaining low variances for other changes in irrelevant passages. This sensitivity allows a dense retrieval model to produce robust results with respect to attacks that try to promote documents without actually increasing their relevance. It also makes it possible to analyze which part of a document is actually relevant to a query, and thus improve the explainability of the retrieval model. Motivated by causality and counterfactual analysis, we propose a series of counterfactual regularization methods based on game theory and unsupervised learning with counterfactual passages. Experiments show that, our method can extract key passages without reliance on the passage-level relevance annotations. Moreover, the regularized dense retrieval models exhibit heightened robustness against adversarial attacks, surpassing the state-of-the-art anti-attack methods.

摘要: 从大型文档语料库中高效地检索一组简明的候选对象仍然是信息检索(IR)中的一个关键挑战。神经检索模型，特别是用转换器构建的密集检索模型和预先训练的语言模型，由于其优越的性能而受到广泛的欢迎。然而，也有人批评说，它们缺乏可解释性，容易受到对手攻击。为了应对这些挑战，我们提出了通过提高密集检索模型对细粒度关联信号的敏感度来提高其稳健性。在这种情况下实现敏感性的模型应该在确定其与查询的相关性的文档的关键段落被修改时表现出高方差，同时保持对不相关段落中的其他变化的低方差。这种敏感度使得密集检索模型能够针对试图提升文档而不实际增加其相关性的攻击产生稳健的结果。它还可以分析文档的哪个部分实际上与查询相关，从而提高检索模型的可解释性。在因果关系和反事实分析的启发下，我们提出了一系列基于博弈论和带有反事实段落的无监督学习的反事实正则化方法。实验表明，我们的方法可以在不依赖于段落级关联标注的情况下提取关键段落。此外，正则化的密集检索模型表现出对对手攻击的高度稳健性，超过了最先进的反攻击方法。



## **26. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

下一代物联网的图形神经网络：最近的进展和开放的挑战 cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20634v1) [paper-pdf](http://arxiv.org/pdf/2412.20634v1)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.

摘要: 图形神经网络(GNN)已经成为下一代网络中优化和管理物联网(IoT)复杂性的关键工具。这项调查全面探讨了如何在6G物联网环境中利用GNN，并通过一系列开放问题重点介绍了关键挑战和机遇。我们首先探讨GNN范例以及节点、边和图级任务在解决无线网络问题中的作用，并强调GNN克服传统优化方法局限性的能力。本指南提高了各种下一代(NG)物联网场景中的问题解决效率。接下来，我们将详细讨论GNN在先进的NG使能技术中的应用，包括大规模MIMO、可重构智能表面、卫星、太赫兹、移动边缘计算(MEC)和超可靠低延迟通信(URLLC)。然后，我们深入研究对抗性攻击带来的挑战，深入了解保护基于GNN的NG-IoT网络的防御机制。接下来，我们研究如何将GNN与未来的技术相结合，如综合传感和通信(ISAC)、卫星-空中-地面-海洋综合网络(SAGSIN)和量子计算。我们的发现突出了GNN在提高NG-IoT系统内的效率、可扩展性和安全性方面的变革潜力，为未来的发展铺平了道路。最后，我们提出了一套设计指南，以促进为下一代物联网应用定制的高效、可扩展和安全的GNN模型的开发。



## **27. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

ErasableMass：针对黑匣子人脸识别模型的稳健且可擦除的隐私保护方案 cs.CV

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.17038v3) [paper-pdf](http://arxiv.org/pdf/2412.17038v3)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Jiacheng Deng, Ziyi Liu

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.

摘要: 虽然人脸识别(FR)模型在人脸验证和识别方面带来了显著的便利，但它们也给公众带来了巨大的隐私风险。现有的人脸隐私保护方案通常采用对抗性的例子来干扰FR模型的人脸验证。然而，这些方案往往对黑盒FR模型的可转移性较弱，并且永久性地破坏了不能满足取证和认证等授权操作要求的可识别信息。为了解决这些局限性，我们提出了一种针对黑盒FR模型的健壮且可擦除的隐私保护方案--可擦除掩码。具体地说，通过重新考虑代理FR模型之间的内在联系，ErasableMASK引入了一种新的元辅助攻击，该攻击通过学习稳定平衡的优化策略中的更多一般特征来提高黑盒的可转移性。它还提供了一种扰动消除机制，支持在不降低图像质量的情况下消除受保护人脸的语义扰动。为了进一步提高性能，ErasableMASK采用了课程学习策略来缓解对抗性攻击和扰动擦除之间的优化冲突。在CelebA-HQ和FFHQ数据集上的广泛实验表明，可擦除掩码在可转移性方面达到了最先进的性能，在商业FR系统中平均达到72%以上的置信度。此外，可擦除掩模还表现出出色的扰动擦除性能，擦除成功率达到90%以上。



## **28. Real-time Fake News from Adversarial Feedback**

来自对抗反馈的实时假新闻 cs.CL

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2410.14651v2) [paper-pdf](http://arxiv.org/pdf/2410.14651v2)

**Authors**: Sanxing Chen, Yukun Huang, Bhuwan Dhingra

**Abstract**: We show that existing evaluations for fake news detection based on conventional sources, such as claims on fact-checking websites, result in high accuracies over time for LLM-based detectors -- even after their knowledge cutoffs. This suggests that recent popular fake news from such sources can be easily detected due to pre-training and retrieval corpus contamination or increasingly salient shallow patterns. Instead, we argue that a proper fake news detection dataset should test a model's ability to reason factually about the current world by retrieving and reading related evidence. To this end, we develop a novel pipeline that leverages natural language feedback from a RAG-based detector to iteratively modify real-time news into deceptive fake news that challenges LLMs. Our iterative rewrite decreases the binary classification ROC-AUC by an absolute 17.5 percent for a strong RAG-based GPT-4o detector. Our experiments reveal the important role of RAG in both detecting and generating fake news, as retrieval-free LLM detectors are vulnerable to unseen events and adversarial attacks, while feedback from RAG detection helps discover more deceitful patterns in fake news.

摘要: 我们表明，现有的基于传统来源的假新闻检测评估，例如在事实核查网站上的声明，导致基于LLM的检测器随着时间的推移而获得高精度-即使在他们的知识中断之后。这表明，由于预训练和检索语料库的污染或日益突出的浅层模式，来自这些来源的最近流行的假新闻可以很容易地被检测出来。相反，我们认为，一个适当的假新闻检测数据集应该通过检索和阅读相关证据来测试模型对当前世界进行事实推理的能力。为此，我们开发了一种新的流水线，利用来自基于RAG的检测器的自然语言反馈来迭代地将实时新闻修改为挑战LLMS的欺骗性假新闻。对于一个强的基于RAG的GPT-40检测器，我们的迭代重写使二进制分类ROC-AUC绝对减少了17.5%。我们的实验揭示了RAG在检测和生成假新闻中的重要作用，因为免检索LLM检测器容易受到不可见事件和对手攻击的攻击，而RAG检测的反馈有助于在假新闻中发现更多的欺骗性模式。



## **29. On Adversarial Robustness of Language Models in Transfer Learning**

迁移学习中语言模型的对抗鲁棒性 cs.CL

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2501.00066v1) [paper-pdf](http://arxiv.org/pdf/2501.00066v1)

**Authors**: Bohdan Turbal, Anastasiia Mazur, Jiaxu Zhao, Mykola Pechenizkiy

**Abstract**: We investigate the adversarial robustness of LLMs in transfer learning scenarios. Through comprehensive experiments on multiple datasets (MBIB Hate Speech, MBIB Political Bias, MBIB Gender Bias) and various model architectures (BERT, RoBERTa, GPT-2, Gemma, Phi), we reveal that transfer learning, while improving standard performance metrics, often leads to increased vulnerability to adversarial attacks. Our findings demonstrate that larger models exhibit greater resilience to this phenomenon, suggesting a complex interplay between model size, architecture, and adaptation methods. Our work highlights the crucial need for considering adversarial robustness in transfer learning scenarios and provides insights into maintaining model security without compromising performance. These findings have significant implications for the development and deployment of LLMs in real-world applications where both performance and robustness are paramount.

摘要: 我们研究了迁移学习场景中LLM的对抗稳健性。通过对多个数据集（MBIB仇恨言论、MBIB政治偏见、MBIB性别偏见）和各种模型架构（BERT、RoBERTa、GPT-2、Gemma、Phi）的全面实验，我们揭示了迁移学习在提高标准性能指标的同时，往往会导致对抗性攻击的脆弱性增加。我们的研究结果表明，较大的模型对这种现象表现出更大的弹性，这表明模型大小、架构和适应方法之间存在复杂的相互作用。我们的工作强调了在迁移学习场景中考虑对抗稳健性的迫切需要，并提供了在不影响性能的情况下维护模型安全性的见解。这些发现对于在性能和稳健性都至关重要的现实应用程序中开发和部署LLM具有重大影响。



## **30. Optimal and Feasible Contextuality-based Randomness Generation**

最佳可行的基于上下文的随机生成 quant-ph

21 pages, 8 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20126v1) [paper-pdf](http://arxiv.org/pdf/2412.20126v1)

**Authors**: Yuan Liu, Ravishankar Ramanathan

**Abstract**: Semi-device-independent (SDI) randomness generation protocols based on Kochen-Specker contextuality offer the attractive features of compact devices, high rates, and ease of experimental implementation over fully device-independent (DI) protocols. Here, we investigate this paradigm and derive four results to improve the state-of-art. Firstly, we introduce a family of simple, experimentally feasible orthogonality graphs (measurement compatibility structures) for which the maximum violation of the corresponding non-contextuality inequalities allows to certify the maximum amount of $\log_2 d$ bits from a qu$d$it system with projective measurements for $d \geq 3$. We analytically derive the Lovasz theta and fractional packing number for this graph family, and thereby prove their utility for optimal randomness generation in both randomness expansion and amplification tasks. Secondly, a central additional assumption in contextuality-based protocols over fully DI ones, is that the measurements are repeatable and satisfy an intended compatibility structure. We frame a relaxation of this condition in terms of $\epsilon$-orthogonality graphs for a parameter $\epsilon > 0$, and derive quantum correlations that allow to certify randomness for arbitrary relaxation $\epsilon \in [0,1)$. Thirdly, it is well known that a single qubit is non-contextual, i.e., the qubit correlations can be explained by a non-contextual hidden variable (NCHV) model. We show however that a single qubit is \textit{almost} contextual, in that there exist qubit correlations that cannot be explained by $\epsilon$-ontologically faithful NCHV models for small $\epsilon > 0$. Finally, we point out possible attacks by quantum and general consistent (non-signalling) adversaries for certain classes of contextuality tests over and above those considered in DI scenarios.

摘要: 基于Kochen-specker上下文的半设备无关(SDI)随机性生成协议具有设备紧凑、速率高、易于实验实现等特点，优于完全设备无关(DI)协议。在这里，我们研究了这一范式，并得出了四个结果来提高最新水平。首先，我们引入了一族简单的，实验上可行的正交图(测量相容结构)，对于它，对应的非上下文不等式的最大违反允许证明具有$d\geq 3$的射影测量的Qu$d$it系统的最大$\log_2 d$比特的数量。我们解析地推导了这个图族的Lovaszθ和分数填充数，从而证明了它们在随机性扩展和放大任务中的最优随机性生成方面的有效性。其次，与完全依赖注入协议相比，基于上下文的协议的一个核心附加假设是测量是可重复的，并且满足预期的兼容性结构。对于参数$\epsilon>0$，我们用$-epsilon$-正交图表示这一条件的松弛，并推导出量子关联，它允许证明[0，1]$中任意松弛的随机性。第三，众所周知，单个量子比特是非上下文相关的，即量子比特之间的关联可以用非上下文隐藏变量(NCHV)模型来解释。然而，我们证明了单个量子比特是与上下文相关的，这是因为存在着不能用$epsilon>0$的本体论忠实的NCHV模型来解释的量子比特关联。最后，我们指出了量子和一般一致(非信令)对手对某些类别的上下文测试的可能攻击，这些测试超出了DI场景中考虑的测试。



## **31. On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs**

传统漏洞评分系统对LLM对抗性攻击的有效性 cs.CR

101 pages, 3 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20087v1) [paper-pdf](http://arxiv.org/pdf/2412.20087v1)

**Authors**: Atmane Ayoub Mansour Bahar, Ahmad Samer Wazan

**Abstract**: This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics.   This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks.   This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs.

摘要: 这项研究考察了通用漏洞评分系统(CVSS)等已建立的漏洞度量在评估针对大型语言模型(LLMS)的攻击时的有效性，重点是对抗性攻击(AA)。这项研究探讨了一般和特定指标因素在确定脆弱性得分方面的影响，为这些指标的潜在增强提供了新的视角。本研究采用定量的方法，计算并比较了56种对抗性攻击下的LLMS脆弱性得分的变异系数。这些攻击来自各种研究论文，通过在线数据库获得，使用多种漏洞指标进行评估。得分通过三个不同的LLM评估的值的平均值来确定。结果表明，现有的评分系统产生的脆弱性分数在不同攻击之间的差异很小，这表明许多度量因素不足以评估对LLM的对抗性攻击。对于特定于上下文的因素或具有预定义值集的因素尤其如此，例如CVSS中的那些因素。这些发现支持这样一种假设，即当前的脆弱性指标，特别是那些具有刚性值的指标，在评估LLM上的AA方面是有限的，这突显了开发针对此类攻击量身定做的更灵活、更通用的指标的必要性。这项研究对已建立的脆弱性度量的有效性和适用性进行了新的分析，特别是在针对大型语言模型的对抗性攻击的背景下，这两种攻击在最近几年都得到了极大的关注。通过广泛的测试和计算，这项研究强调了这些指标的局限性，并为改进和完善专门为低土地管理定制的脆弱性评估框架开辟了新的途径。



## **32. LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models**

LLM-Virus：对大型语言模型的进化越狱攻击 cs.CR

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2501.00055v1) [paper-pdf](http://arxiv.org/pdf/2501.00055v1)

**Authors**: Miao Yu, Junfeng Fang, Yingjie Zhou, Xing Fan, Kun Wang, Shirui Pan, Qingsong Wen

**Abstract**: While safety-aligned large language models (LLMs) are increasingly used as the cornerstone for powerful systems such as multi-agent frameworks to solve complex real-world problems, they still suffer from potential adversarial queries, such as jailbreak attacks, which attempt to induce harmful content. Researching attack methods allows us to better understand the limitations of LLM and make trade-offs between helpfulness and safety. However, existing jailbreak attacks are primarily based on opaque optimization techniques (e.g. token-level gradient descent) and heuristic search methods like LLM refinement, which fall short in terms of transparency, transferability, and computational cost. In light of these limitations, we draw inspiration from the evolution and infection processes of biological viruses and propose LLM-Virus, a jailbreak attack method based on evolutionary algorithm, termed evolutionary jailbreak. LLM-Virus treats jailbreak attacks as both an evolutionary and transfer learning problem, utilizing LLMs as heuristic evolutionary operators to ensure high attack efficiency, transferability, and low time cost. Our experimental results on multiple safety benchmarks show that LLM-Virus achieves competitive or even superior performance compared to existing attack methods.

摘要: 尽管与安全一致的大型语言模型(LLM)越来越多地被用作多代理框架等强大系统的基石，以解决复杂的现实世界问题，但它们仍面临潜在的对抗性查询，例如试图诱导有害内容的越狱攻击。研究攻击方法可以让我们更好地了解LLM的局限性，并在有效性和安全性之间进行权衡。然而，现有的越狱攻击主要基于不透明的优化技术(如令牌级梯度下降)和启发式搜索方法，如LLM求精，这些方法在透明度、可转移性和计算成本方面都存在不足。针对这些局限性，我们从生物病毒的进化和感染过程中得到启发，提出了一种基于进化算法的越狱攻击方法LLM-Virus，称为进化越狱。LLM-Virus将越狱攻击视为一个进化和转移学习问题，利用LLM作为启发式进化算子，以确保高攻击效率、可转移性和低时间开销。我们在多个安全基准上的实验结果表明，与现有的攻击方法相比，LLM-Virus具有相当甚至更好的性能。



## **33. B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions**

B-AVIBench：评估黑匣子对抗视觉指令上大型视觉语言模型的鲁棒性 cs.CV

Accepted by IEEE Transactions on Information Forensics & Security

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2403.09346v2) [paper-pdf](http://arxiv.org/pdf/2403.09346v2)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Nanning Zheng, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in responding well to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 316K B-AVIs encompassing five categories of multimodal capabilities (ten tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. B-AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against B-AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark are available at https://github.com/zhanghao5201/B-AVIBench.

摘要: 大型视觉语言模型(LVLM)在很好地响应用户的视觉指令方面取得了重大进展。但是，这些包含图像和文本的说明很容易受到有意和无意的攻击。尽管LVLMS对这类威胁的稳健性至关重要，但目前在这一领域的研究仍然有限。为了弥补这一差距，我们引入了B-AVIB边框架，该框架旨在分析LVLMS在面对各种黑盒对抗性视觉指令(B-AVI)时的健壮性，包括四种类型的基于图像的B-AVI、10种类型的基于文本的B-AVI和九种类型的内容偏见B-AVI(如性别、暴力、文化和种族偏见等)。我们生成了316k B-AVI，包括五类多模式能力(十项任务)和内容偏见。然后，我们对14个开源LVLM进行了全面的评估，以评估它们的性能。B-AVIBtch也可作为从业者评估LVLMS对B-AVIS的稳健性的便捷工具。我们的发现和广泛的实验结果揭示了LVLMS的漏洞，并突出表明即使在GeminiProVision和GPT-4V等先进的闭源LVLM中也存在固有偏差。这凸显了增强LVLM的健壮性、安全性和公平性的重要性。源代码和基准测试可在https://github.com/zhanghao5201/B-AVIBench.上获得



## **34. A Robust Adversarial Ensemble with Causal (Feature Interaction) Interpretations for Image Classification**

具有因果（特征相互作用）解释的图像分类鲁棒对抗集成 cs.CV

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20025v1) [paper-pdf](http://arxiv.org/pdf/2412.20025v1)

**Authors**: Chunheng Zhao, Pierluigi Pisu, Gurcan Comert, Negash Begashaw, Varghese Vaidyan, Nina Christine Hubig

**Abstract**: Deep learning-based discriminative classifiers, despite their remarkable success, remain vulnerable to adversarial examples that can mislead model predictions. While adversarial training can enhance robustness, it fails to address the intrinsic vulnerability stemming from the opaque nature of these black-box models. We present a deep ensemble model that combines discriminative features with generative models to achieve both high accuracy and adversarial robustness. Our approach integrates a bottom-level pre-trained discriminative network for feature extraction with a top-level generative classification network that models adversarial input distributions through a deep latent variable model. Using variational Bayes, our model achieves superior robustness against white-box adversarial attacks without adversarial training. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate our model's superior adversarial robustness. Through evaluations using counterfactual metrics and feature interaction-based metrics, we establish correlations between model interpretability and adversarial robustness. Additionally, preliminary results on Tiny-ImageNet validate our approach's scalability to more complex datasets, offering a practical solution for developing robust image classification models.

摘要: 基于深度学习的判别分类器尽管取得了显著的成功，但仍然容易受到可能误导模型预测的对抗性例子的影响。虽然对抗性训练可以增强稳健性，但它无法解决这些黑箱模型的不透明性质所产生的内在脆弱性。我们提出了一种深度集成模型，该模型结合了区分特征和生成模型，以实现高准确率和对抗健壮性。我们的方法结合了用于特征提取的底层预训练判别网络和顶层生成性分类网络，该网络通过深度潜变量模型对对抗性输入分布进行建模。使用变分贝叶斯，我们的模型在不需要对抗性训练的情况下，对白盒对抗性攻击获得了优越的稳健性。在CIFAR-10和CIFAR-100上的大量实验证明了我们的模型具有优越的对抗鲁棒性。通过使用反事实度量和基于特征交互的度量进行评估，我们建立了模型可解释性和对抗健壮性之间的关联。此外，在Tiny-ImageNet上的初步结果验证了我们的方法对更复杂的数据集的可扩展性，为开发健壮的图像分类模型提供了一个实用的解决方案。



## **35. Adversarial Robustness for Deep Learning-based Wildfire Detection Models**

基于深度学习的野火检测模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20006v1) [paper-pdf](http://arxiv.org/pdf/2412.20006v1)

**Authors**: Ryo Ide, Lei Yang

**Abstract**: Smoke detection using Deep Neural Networks (DNNs) is an effective approach for early wildfire detection. However, because smoke is temporally and spatially anomalous, there are limitations in collecting sufficient training data. This raises overfitting and bias concerns in existing DNN-based wildfire detection models. Thus, we introduce WARP (Wildfire Adversarial Robustness Procedure), the first model-agnostic framework for evaluating the adversarial robustness of DNN-based wildfire detection models. WARP addresses limitations in smoke image diversity using global and local adversarial attack methods. The global attack method uses image-contextualized Gaussian noise, while the local attack method uses patch noise injection, tailored to address critical aspects of wildfire detection. Leveraging WARP's model-agnostic capabilities, we assess the adversarial robustness of real-time Convolutional Neural Networks (CNNs) and Transformers. The analysis revealed valuable insights into the models' limitations. Specifically, the global attack method demonstrates that the Transformer model has more than 70\% precision degradation than the CNN against global noise. In contrast, the local attack method shows that both models are susceptible to cloud image injections when detecting smoke-positive instances, suggesting a need for model improvements through data augmentation. WARP's comprehensive robustness analysis contributed to the development of wildfire-specific data augmentation strategies, marking a step toward practicality.

摘要: 基于深度神经网络的烟雾检测是野火早期检测的一种有效方法。然而，由于烟雾在时间和空间上都是反常的，收集足够的训练数据是有局限性的。这在现有的基于DNN的野火检测模型中引发了过度拟合和偏差的担忧。因此，我们引入了WARP(Wildfire对抗稳健性过程)，这是第一个模型不可知的框架，用于评估基于DNN的野火检测模型的对抗稳健性。WARP使用全球和局部对抗性攻击方法解决烟雾图像多样性方面的限制。全局攻击方法使用与图像相关的高斯噪声，而局部攻击方法使用补丁噪声注入，该方法针对野火检测的关键方面进行了量身定做。利用WARP的模型不可知能力，我们评估了实时卷积神经网络(CNN)和变形金刚的对抗健壮性。分析揭示了对模型局限性的有价值的见解。具体来说，全局攻击方法表明，Transformer模型在抗全局噪声方面比CNN模型有70%以上的精度下降。相比之下，本地攻击方法表明，当检测到烟雾阳性实例时，这两个模型都容易受到云图注入的影响，这表明需要通过数据增强来改进模型。WARP的全面稳健性分析有助于开发特定于野火的数据增强策略，标志着朝着实用化迈出了一步。



## **36. Standard-Deviation-Inspired Regularization for Improving Adversarial Robustness**

标准偏差启发的规范化提高对抗稳健性 cs.LG

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19947v1) [paper-pdf](http://arxiv.org/pdf/2412.19947v1)

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) has been demonstrated to improve the robustness of deep neural networks (DNNs) against adversarial attacks. AT is a min-max optimization procedure where in adversarial examples are generated to train a more robust DNN. The inner maximization step of AT increases the losses of inputs with respect to their actual classes. The outer minimization involves minimizing the losses on the adversarial examples obtained from the inner maximization. This work proposes a standard-deviation-inspired (SDI) regularization term to improve adversarial robustness and generalization. We argue that the inner maximization in AT is similar to minimizing a modified standard deviation of the model's output probabilities. Moreover, we suggest that maximizing this modified standard deviation can complement the outer minimization of the AT framework. To support our argument, we experimentally show that the SDI measure can be used to craft adversarial examples. Additionally, we demonstrate that combining the SDI regularization term with existing AT variants enhances the robustness of DNNs against stronger attacks, such as CW and Auto-attack, and improves generalization.

摘要: 对抗训练(AT)已被证明可以提高深度神经网络(DNN)对对抗攻击的稳健性。AT是一种最小-最大优化过程，其中在对抗性例子中生成训练更健壮的DNN。AT的内部最大化步骤增加了输入相对于其实际类别的损失。外极小化包括最小化由内极大化得到的对抗性例子的损失。该工作提出了一种标准差启发(SDI)正则化项来提高对手的稳健性和泛化能力。我们认为AT中的内极大化类似于最小化模型输出概率的修正标准差。此外，我们认为最大化这个修正的标准差可以补充AT框架的外极小化。为了支持我们的论点，我们通过实验证明SDI测量可以用来制作对抗性的例子。此外，我们还证明了将SDI正则化项与现有的AT变体相结合，增强了DNN对更强的攻击(如CW和Auto-Attack)的健壮性，并提高了泛化能力。



## **37. A High Dimensional Statistical Model for Adversarial Training: Geometry and Trade-Offs**

对抗训练的多维统计模型：几何结构和权衡 stat.ML

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2402.05674v3) [paper-pdf](http://arxiv.org/pdf/2402.05674v3)

**Authors**: Kasimir Tanner, Matteo Vilucchio, Bruno Loureiro, Florent Krzakala

**Abstract**: This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses for a Block Feature Model. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. We show that the the presence of multiple different feature types is crucial to the high sample complexity performances of adversarial training. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust features during training, identifying a uniform protection as an inherently effective defence mechanism.

摘要: 该工作研究了高维环境下基于差值的线性分类器的对抗性训练，其中维度$d$和数据点数目$n$以固定的比率$\α=n/d$发散。我们引入了一个易于处理的数学模型，其中可以研究数据和敌意攻击者几何之间的相互作用，同时捕获在对抗性健壮性文献中观察到的核心现象学。我们的主要理论贡献是在块特征模型的一般凸和非增加损失下，给出了对抗性经验风险最小化充分统计量的精确渐近描述。我们的结果使我们能够准确地描述数据中的哪些方向与更高的泛化/稳健性权衡相关，如稳健性和有用性度量所定义的那样。结果表明，多个不同特征类型的存在对对抗性训练的高样本复杂度性能至关重要。特别是，我们揭示了方向的存在，这些方向可以在不影响准确性的情况下得到辩护。最后，我们展示了在训练过程中防御非健壮特征的优势，确定了统一保护作为一种内在有效的防御机制。



## **38. Enhancing Adversarial Robustness of Deep Neural Networks Through Supervised Contrastive Learning**

通过监督对比学习增强深度神经网络的对抗鲁棒性 cs.LG

8 pages, 11 figures

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19747v1) [paper-pdf](http://arxiv.org/pdf/2412.19747v1)

**Authors**: Longwei Wang, Navid Nayyem, Abdullah Rakin

**Abstract**: Adversarial attacks exploit the vulnerabilities of convolutional neural networks by introducing imperceptible perturbations that lead to misclassifications, exposing weaknesses in feature representations and decision boundaries. This paper presents a novel framework combining supervised contrastive learning and margin-based contrastive loss to enhance adversarial robustness. Supervised contrastive learning improves the structure of the feature space by clustering embeddings of samples within the same class and separating those from different classes. Margin-based contrastive loss, inspired by support vector machines, enforces explicit constraints to create robust decision boundaries with well-defined margins. Experiments on the CIFAR-100 dataset with a ResNet-18 backbone demonstrate robustness performance improvements in adversarial accuracy under Fast Gradient Sign Method attacks.

摘要: 对抗性攻击通过引入难以感知的扰动来利用卷积神经网络的漏洞，从而导致错误分类，暴露特征表示和决策边界的弱点。本文提出了一种新颖的框架，将监督对比学习和基于边缘的对比损失相结合，以增强对抗鲁棒性。监督对比学习通过对同一类内的样本嵌入进行聚集并将来自不同类的样本嵌入分离，来改善特征空间的结构。基于利润的对比损失受支持向量机的启发，强制执行显式约束，以创建具有明确定义利润的稳健决策边界。在具有ResNet-18主干的CIFAR-100数据集上进行的实验表明，在快速梯度符号法攻击下，对抗准确性的鲁棒性性能有所提高。



## **39. Gröbner Basis Cryptanalysis of Ciminion and Hydra**

格罗布纳基础对西米尼恩和海德拉的密码分析 cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2405.05040v3) [paper-pdf](http://arxiv.org/pdf/2405.05040v3)

**Authors**: Matthias Johann Steiner

**Abstract**: Ciminion and Hydra are two recently introduced symmetric key Pseudo-Random Functions for Multi-Party Computation applications. For efficiency both primitives utilize quadratic permutations at round level. Therefore, polynomial system solving-based attacks pose a serious threat to these primitives. For Ciminion, we construct a quadratic degree reverse lexicographic (DRL) Gr\"obner basis for the iterated polynomial model via linear transformations. With the Gr\"obner basis we can simplify cryptanalysis since we do not need to impose genericity assumptions anymore to derive complexity estimations. For Hydra, with the help of a computer algebra program like SageMath we construct a DRL Gr\"obner basis for the iterated model via linear transformations and a linear change of coordinates. In the Hydra proposal it was claimed that $r_\mathcal{H} = 31$ rounds are sufficient to provide $128$ bits of security against Gr\"obner basis attacks for an ideal adversary with $\omega = 2$. However, via our Hydra Gr\"obner basis standard term order conversion to a lexicographic (LEX) Gr\"obner basis requires just $126$ bits with $\omega = 2$. Moreover, via a dedicated polynomial system solving technique up to $r_\mathcal{H} = 33$ rounds can be attacked below $128$ bits for an ideal adversary.

摘要: Ciminion和Hydra是最近推出的两个用于多方计算应用的对称密钥伪随机函数。为了提高效率，两个基元都在循环水平上使用二次置换。因此，基于多项式系统求解的攻击对这些原语构成了严重威胁。对于Ciminion，我们通过线性变换为迭代多项式模型构造了一个二次逆词典(DRL)Gr‘obner基，利用这个Gr’obner基，我们可以简化密码分析，因为我们不再需要强加一般性假设来推导复杂性估计。对于Hydra，借助于SageMath这样的计算机代数程序，我们通过线性变换和线性坐标变化，为迭代模型构造了一个DRL Grobner基.在Hydra的方案中，声称$r_\mathcal{H}=31$轮足以为$omega=2$的理想对手提供$128$bit的Gr‘obner基攻击安全.然而，通过我们的Hydra Gr\“obner基础”将标准术语顺序转换为词典(Lex)Gr\“obner基础只需要$126$位，且$\omega=2$。此外，通过一种专门的多项式系统求解技术，高达$r_\数学{H}=33$的轮数可以被攻击到低于$128$比特的理想对手。



## **40. Attribution for Enhanced Explanation with Transferable Adversarial eXploration**

可转移对抗性探索增强解释的归因 cs.AI

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19523v1) [paper-pdf](http://arxiv.org/pdf/2412.19523v1)

**Authors**: Zhiyu Zhu, Jiayu Zhang, Zhibo Jin, Huaming Chen, Jianlong Zhou, Fang Chen

**Abstract**: The interpretability of deep neural networks is crucial for understanding model decisions in various applications, including computer vision. AttEXplore++, an advanced framework built upon AttEXplore, enhances attribution by incorporating transferable adversarial attack methods such as MIG and GRA, significantly improving the accuracy and robustness of model explanations. We conduct extensive experiments on five models, including CNNs (Inception-v3, ResNet-50, VGG16) and vision transformers (MaxViT-T, ViT-B/16), using the ImageNet dataset. Our method achieves an average performance improvement of 7.57\% over AttEXplore and 32.62\% compared to other state-of-the-art interpretability algorithms. Using insertion and deletion scores as evaluation metrics, we show that adversarial transferability plays a vital role in enhancing attribution results. Furthermore, we explore the impact of randomness, perturbation rate, noise amplitude, and diversity probability on attribution performance, demonstrating that AttEXplore++ provides more stable and reliable explanations across various models. We release our code at: https://anonymous.4open.science/r/ATTEXPLOREP-8435/

摘要: 深度神经网络的可解释性对于理解包括计算机视觉在内的各种应用中的模型决策至关重要。AttEXplore++是建立在AttEXplore基础上的高级框架，通过整合可转移的对抗性攻击方法(如MIG和GRA)来增强属性，显著提高模型解释的准确性和健壮性。我们使用ImageNet数据集，在五个模型上进行了广泛的实验，包括CNN(初始-v3，ResNet-50，VGG16)和视觉转换器(MaxViT-T，Vit-B/16)。与AttEXplore算法相比，该方法的性能平均提高了7.57倍，与其他最先进的可解释性算法相比，平均性能提高了32.62倍。以插入和删除分数作为评价指标，我们发现对抗性转移对提高归因结果起着至关重要的作用。此外，我们探讨了随机性、扰动率、噪声幅度和多样性概率对归因性能的影响，证明了AttEXplore++在各种模型中提供了更稳定和可靠的解释。我们的代码发布地址为：https://anonymous.4open.science/r/ATTEXPLOREP-8435/



## **41. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19394v1) [paper-pdf](http://arxiv.org/pdf/2412.19394v1)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is accessible at https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。该代码可在https://github.com/jianshuod/Engorgio-prompt.上访问



## **42. Quantum-Inspired Weight-Constrained Neural Network: Reducing Variable Numbers by 100x Compared to Standard Neural Networks**

量子启发的权重约束神经网络：与标准神经网络相比将变量数减少100倍 quant-ph

13 pages, 5 figures. Comments are welcome

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19355v1) [paper-pdf](http://arxiv.org/pdf/2412.19355v1)

**Authors**: Shaozhi Li, M Sabbir Salek, Binayyak Roy, Yao Wang, Mashrur Chowdhury

**Abstract**: Although quantum machine learning has shown great promise, the practical application of quantum computers remains constrained in the noisy intermediate-scale quantum era. To take advantage of quantum machine learning, we investigate the underlying mathematical principles of these quantum models and adapt them to classical machine learning frameworks. Specifically, we develop a classical weight-constrained neural network that generates weights based on quantum-inspired insights. We find that this approach can reduce the number of variables in a classical neural network by a factor of 135 while preserving its learnability. In addition, we develop a dropout method to enhance the robustness of quantum machine learning models, which are highly susceptible to adversarial attacks. This technique can also be applied to improve the adversarial resilience of the classical weight-constrained neural network, which is essential for industry applications, such as self-driving vehicles. Our work offers a novel approach to reduce the complexity of large classical neural networks, addressing a critical challenge in machine learning.

摘要: 尽管量子机器学习显示出了巨大的前景，但在嘈杂的中等规模量子时代，量子计算机的实际应用仍然受到限制。为了利用量子机器学习的优势，我们研究了这些量子模型的基本数学原理，并将它们适应于经典的机器学习框架。具体地说，我们开发了一个经典的权重约束神经网络，它基于量子启发的见解生成权重。我们发现，这种方法可以将经典神经网络中的变量数量减少135倍，同时保持其可学习性。此外，我们开发了一种丢弃方法来增强量子机器学习模型的健壮性，这些模型对对手攻击非常敏感。该技术还可以用于提高经典的权值约束神经网络的对抗能力，这对于自动驾驶汽车等工业应用是必不可少的。我们的工作提供了一种新的方法来降低大型经典神经网络的复杂性，解决了机器学习中的一个关键挑战。



## **43. Federated Hybrid Training and Self-Adversarial Distillation: Towards Robust Edge Networks**

联合混合训练和自对抗蒸馏：迈向稳健的边缘网络 cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19354v1) [paper-pdf](http://arxiv.org/pdf/2412.19354v1)

**Authors**: Yu Qiao, Apurba Adhikary, Kitae Kim, Eui-Nam Huh, Zhu Han, Choong Seon Hong

**Abstract**: Federated learning (FL) is a distributed training technology that enhances data privacy in mobile edge networks by allowing data owners to collaborate without transmitting raw data to the edge server. However, data heterogeneity and adversarial attacks pose challenges to develop an unbiased and robust global model for edge deployment. To address this, we propose Federated hyBrid Adversarial training and self-adversarial disTillation (FedBAT), a new framework designed to improve both robustness and generalization of the global model. FedBAT seamlessly integrates hybrid adversarial training and self-adversarial distillation into the conventional FL framework from data augmentation and feature distillation perspectives. From a data augmentation perspective, we propose hybrid adversarial training to defend against adversarial attacks by balancing accuracy and robustness through a weighted combination of standard and adversarial training. From a feature distillation perspective, we introduce a novel augmentation-invariant adversarial distillation method that aligns local adversarial features of augmented images with their corresponding unbiased global clean features. This alignment can effectively mitigate bias from data heterogeneity while enhancing both the robustness and generalization of the global model. Extensive experimental results across multiple datasets demonstrate that FedBAT yields comparable or superior performance gains in improving robustness while maintaining accuracy compared to several baselines.

摘要: 联合学习(FL)是一种分布式训练技术，它允许数据所有者在不向边缘服务器传输原始数据的情况下进行协作，从而增强移动边缘网络中的数据隐私。然而，数据异构性和对抗性攻击给开发无偏见和健壮的全球EDGE部署模型带来了挑战。为了解决这一问题，我们提出了联邦混合对抗训练和自我对抗蒸馏(FedBAT)，这是一个新的框架，旨在提高全局模型的健壮性和泛化能力。FedBAT从数据增强和特征提取的角度，将混合对抗性训练和自我对抗性提炼无缝地集成到传统的FL框架中。从数据增强的角度，我们提出了混合对抗性训练，通过标准训练和对抗性训练的加权组合来平衡精确度和稳健性来防御对抗性攻击。从特征提取的角度，我们提出了一种新的增强不变对抗提取方法，该方法将增强图像的局部对抗特征与其相应的无偏全局清洁特征对齐。这种对齐可以有效地减少数据异质性带来的偏差，同时增强全局模型的稳健性和泛化能力。在多个数据集上的广泛实验结果表明，与几个基线相比，FedBAT在提高稳健性同时保持准确性方面获得了类似或更好的性能收益。



## **44. xSRL: Safety-Aware Explainable Reinforcement Learning -- Safety as a Product of Explainability**

xSRL：安全意识的可解释强化学习--安全作为可解释性的产物 cs.AI

Accepted to 24th International Conference on Autonomous Agents and  Multiagent Systems (AAMAS 2025)

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19311v1) [paper-pdf](http://arxiv.org/pdf/2412.19311v1)

**Authors**: Risal Shahriar Shefin, Md Asifur Rahman, Thai Le, Sarra Alqahtani

**Abstract**: Reinforcement learning (RL) has shown great promise in simulated environments, such as games, where failures have minimal consequences. However, the deployment of RL agents in real-world systems such as autonomous vehicles, robotics, UAVs, and medical devices demands a higher level of safety and transparency, particularly when facing adversarial threats. Safe RL algorithms have been developed to address these concerns by optimizing both task performance and safety constraints. However, errors are inevitable, and when they occur, it is essential that the RL agents can also explain their actions to human operators. This makes trust in the safety mechanisms of RL systems crucial for effective deployment. Explainability plays a key role in building this trust by providing clear, actionable insights into the agent's decision-making process, ensuring that safety-critical decisions are well understood. While machine learning (ML) has seen significant advances in interpretability and visualization, explainability methods for RL remain limited. Current tools fail to address the dynamic, sequential nature of RL and its needs to balance task performance with safety constraints over time. The re-purposing of traditional ML methods, such as saliency maps, is inadequate for safety-critical RL applications where mistakes can result in severe consequences. To bridge this gap, we propose xSRL, a framework that integrates both local and global explanations to provide a comprehensive understanding of RL agents' behavior. xSRL also enables developers to identify policy vulnerabilities through adversarial attacks, offering tools to debug and patch agents without retraining. Our experiments and user studies demonstrate xSRL's effectiveness in increasing safety in RL systems, making them more reliable and trustworthy for real-world deployment. Code is available at https://github.com/risal-shefin/xSRL.

摘要: 强化学习(RL)在模拟环境中显示了巨大的前景，例如游戏，在这些环境中，失败的后果最小。然而，在自动驾驶车辆、机器人、无人机和医疗设备等现实世界系统中部署RL代理需要更高水平的安全性和透明度，特别是在面临对手威胁的情况下。安全RL算法已经被开发出来，通过优化任务性能和安全约束来解决这些问题。然而，错误是不可避免的，当它们发生时，RL代理也可以向人类操作员解释他们的行为是至关重要的。这使得对RL系统安全机制的信任对有效部署至关重要。可解释性在建立这种信任方面发挥了关键作用，它为代理人的决策过程提供了清晰、可操作的见解，确保了对安全至关重要的决策得到很好的理解。虽然机器学习(ML)在可解释性和可视化方面取得了重大进展，但用于RL的可解释性方法仍然有限。目前的工具不能解决RL的动态、连续的性质，以及它需要随着时间的推移平衡任务性能和安全约束。传统ML方法的再利用，如显著图，对于安全关键的RL应用是不够的，因为错误可能会导致严重的后果。为了弥合这一差距，我们提出了xSRL，一个整合了局部和全局解释的框架，以提供对RL代理行为的全面理解。XSRL还使开发人员能够通过对抗性攻击识别策略漏洞，提供工具来调试和修补代理，而无需重新培训。我们的实验和用户研究证明了xSRL在提高RL系统安全性方面的有效性，使它们在现实世界的部署中更加可靠和值得信赖。代码可在https://github.com/risal-shefin/xSRL.上找到



## **45. Game-Theoretically Secure Distributed Protocols for Fair Allocation in Coalitional Games**

联盟游戏中公平分配的游戏理论安全分布式协议 cs.GT

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19192v1) [paper-pdf](http://arxiv.org/pdf/2412.19192v1)

**Authors**: T-H. Hubert Chan, Qipeng Kuang, Quan Xue

**Abstract**: We consider game-theoretically secure distributed protocols for coalition games that approximate the Shapley value with small multiplicative error. Since all known existing approximation algorithms for the Shapley value are randomized, it is a challenge to design efficient distributed protocols among mutually distrusted players when there is no central authority to generate unbiased randomness. The game-theoretic notion of maximin security has been proposed to offer guarantees to an honest player's reward even if all other players are susceptible to an adversary.   Permutation sampling is often used in approximation algorithms for the Shapley value. A previous work in 1994 by Zlotkin et al. proposed a simple constant-round distributed permutation generation protocol based on commitment scheme, but it is vulnerable to rushing attacks. The protocol, however, can detect such attacks.   In this work, we model the limited resources of an adversary by a violation budget that determines how many times it can perform such detectable attacks. Therefore, by repeating the number of permutation samples, an honest player's reward can be guaranteed to be close to its Shapley value. We explore both high probability and expected maximin security. We obtain an upper bound on the number of permutation samples for high probability maximin security, even with an unknown violation budget. Furthermore, we establish a matching lower bound for the weaker notion of expected maximin security in specific permutation generation protocols. We have also performed experiments on both synthetic and real data to empirically verify our results.

摘要: 我们考虑在小乘法误差下近似Shapley值的联盟博弈的博弈论安全分布式协议。由于所有已知的Shapley值的近似算法都是随机化的，在没有中央权威机构来产生无偏随机性的情况下，在相互不信任的参与者之间设计有效的分布式协议是一个挑战。博弈论的最大限度安全的概念被提出，以保证诚实的玩家的回报，即使所有其他玩家都容易受到对手的影响。在Shapley值的近似算法中，通常使用置换采样。Zlotkin等人在1994年进行的前一项工作。提出了一种简单的基于承诺方案的恒轮分布式置换生成协议，但该协议容易受到冲刺攻击。然而，该协议可以检测到此类攻击。在这项工作中，我们通过违规预算来模拟对手的有限资源，该预算决定了对手可以执行这种可检测到的攻击的次数。因此，通过重复排列样本的数量，可以保证诚实玩家的奖励接近其Shapley值。我们探讨了高概率安全性和期望最大安全性。我们得到了高概率最大化安全性的置换样本数目的上界，即使在未知的违规预算下也是如此。此外，对于特定置换生成协议中较弱的期望最大安全性概念，我们建立了匹配的下界。我们还在合成数据和真实数据上进行了实验，以经验地验证我们的结果。



## **46. TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity**

TSCheater：通过视觉相似性生成高质量的西藏对抗文本 cs.CL

Camera-Ready Version; Accepted at ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.02371v3) [paper-pdf](http://arxiv.org/pdf/2412.02371v3)

**Authors**: Xi Cao, Quzong Gesang, Yuan Sun, Nuo Qun, Tashi Nyima

**Abstract**: Language models based on deep neural networks are vulnerable to textual adversarial attacks. While rich-resource languages like English are receiving focused attention, Tibetan, a cross-border language, is gradually being studied due to its abundant ancient literature and critical language strategy. Currently, there are several Tibetan adversarial text generation methods, but they do not fully consider the textual features of Tibetan script and overestimate the quality of generated adversarial texts. To address this issue, we propose a novel Tibetan adversarial text generation method called TSCheater, which considers the characteristic of Tibetan encoding and the feature that visually similar syllables have similar semantics. This method can also be transferred to other abugidas, such as Devanagari script. We utilize a self-constructed Tibetan syllable visual similarity database called TSVSDB to generate substitution candidates and adopt a greedy algorithm-based scoring mechanism to determine substitution order. After that, we conduct the method on eight victim language models. Experimentally, TSCheater outperforms existing methods in attack effectiveness, perturbation magnitude, semantic similarity, visual similarity, and human acceptance. Finally, we construct the first Tibetan adversarial robustness evaluation benchmark called AdvTS, which is generated by existing methods and proofread by humans.

摘要: 基于深度神经网络的语言模型容易受到文本攻击。在英语等资源丰富的语言受到关注的同时，藏语这一跨境语言也因其丰富的古代文献和批评的语言策略而逐渐被研究。目前，有几种藏文对抗性文本生成方法，但它们没有充分考虑藏文的文本特征，高估了生成的对抗性文本的质量。针对这一问题，我们提出了一种新的藏文对抗性文本生成方法TSCheater，该方法考虑了藏文编码的特点和视觉上相似音节具有相似语义的特点。这种方法也可以移植到其他ABUGIDAS，如天成文书。利用自行构建的藏文音节视觉相似度数据库TSVSDB生成替换候选，并采用基于贪婪算法的评分机制确定替换顺序。之后，我们在八个受害者语言模型上进行了该方法。实验结果表明，TSCheater在攻击效果、扰动幅度、语义相似度、视觉相似度和人类接受度等方面均优于现有方法。最后，我们构建了第一个藏文对手健壮性评估基准ADVTS，该基准由现有方法生成并由人工校对。



## **47. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Model**

迪夫补丁：使用扩散模型生成可定制的对抗补丁 cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.01440v2) [paper-pdf](http://arxiv.org/pdf/2412.01440v2)

**Authors**: Zhixiang Wang, Guangnan Ye, Xiaosen Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can easily allow individuals to evade person detectors. However, most existing adversarial patch generation methods prioritize attack effectiveness over stealthiness, resulting in patches that are aesthetically unpleasing. Although existing methods using generative adversarial networks or diffusion models can produce more natural-looking patches, they often struggle to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these challenges, we propose a novel diffusion-based customizable patch generation framework termed DiffPatch, specifically tailored for creating naturalistic and customizable adversarial patches. Our approach enables users to utilize a reference image as the source, rather than starting from random noise, and incorporates masks to craft naturalistic patches of various shapes, not limited to squares. To prevent the original semantics from being lost during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Notably, while maintaining a natural appearance, our method achieves a comparable attack performance to state-of-the-art non-naturalistic patches when using similarly sized attacks. Using DiffPatch, we have created a physical adversarial T-shirt dataset, AdvPatch-1K, specifically targeting YOLOv5s. This dataset includes over a thousand images across diverse scenarios, validating the effectiveness of our attack in real-world environments. Moreover, it provides a valuable resource for future research.

摘要: 衣服上印有敌意的物理补丁可以很容易地让个人躲避个人探测器。然而，大多数现有的对抗性补丁生成方法将攻击效率置于隐蔽性之上，导致生成的补丁在美学上令人不快。虽然现有的方法使用生成性对抗网络或扩散模型可以产生看起来更自然的补丁，但它们往往难以平衡隐蔽性和攻击有效性，并且缺乏用户定制的灵活性。为了应对这些挑战，我们提出了一种新的基于扩散的可定制补丁生成框架DiffPatch，该框架专门用于创建自然的和可定制的对抗性补丁。我们的方法使用户能够利用参考图像作为源，而不是从随机噪声开始，并结合蒙版来制作各种形状的自然斑块，而不限于正方形。为了避免在扩散过程中丢失原始语义，我们使用空文本反转将随机噪声样本映射到单一输入图像，并通过不完全扩散优化(IDO)生成斑块。值得注意的是，在保持自然外观的同时，我们的方法在使用类似大小的攻击时，实现了与最先进的非自然主义补丁相当的攻击性能。使用DiffPatch，我们已经创建了一个物理对手T恤数据集AdvPatch-1K，专门针对YOLOv5。该数据集包括1000多张不同场景的图像，验证了我们的攻击在真实环境中的有效性。此外，它还为今后的研究提供了宝贵的资源。



## **48. Provable Robust Saliency-based Explanations**

可证明的稳健基于显着性的解释 cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2212.14106v4) [paper-pdf](http://arxiv.org/pdf/2212.14106v4)

**Authors**: Chao Chen, Chenghua Guo, Rufeng Chen, Guixiang Ma, Ming Zeng, Xiangwen Liao, Xi Zhang, Sihong Xie

**Abstract**: To foster trust in machine learning models, explanations must be faithful and stable for consistent insights. Existing relevant works rely on the $\ell_p$ distance for stability assessment, which diverges from human perception. Besides, existing adversarial training (AT) associated with intensive computations may lead to an arms race. To address these challenges, we introduce a novel metric to assess the stability of top-$k$ salient features. We introduce R2ET which trains for stable explanation by efficient and effective regularizer, and analyze R2ET by multi-objective optimization to prove numerical and statistical stability of explanations. Moreover, theoretical connections between R2ET and certified robustness justify R2ET's stability in all attacks. Extensive experiments across various data modalities and model architectures show that R2ET achieves superior stability against stealthy attacks, and generalizes effectively across different explanation methods.

摘要: 为了促进对机器学习模型的信任，解释必须忠实且稳定，以获得一致的见解。现有的相关作品依赖于$\ell_p$距离进行稳定性评估，这与人类的感知存在分歧。此外，与密集计算相关的现有对抗训练（AT）可能会导致军备竞赛。为了应对这些挑战，我们引入了一种新颖的指标来评估顶级$k$显着特征的稳定性。我们引入R2 ET，通过高效且有效的正规化器训练稳定的解释，并通过多目标优化分析R2 ET，以证明解释的数字和统计稳定性。此外，R2 ET和认证稳健性之间的理论联系证明了R2 ET在所有攻击中的稳定性。跨各种数据模式和模型架构的广泛实验表明，R2 ET针对隐形攻击实现了卓越的稳定性，并在不同的解释方法中有效推广。



## **49. Imperceptible Adversarial Attacks on Point Clouds Guided by Point-to-Surface Field**

点到表面场引导下的点云不可感知的对抗攻击 cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19015v1) [paper-pdf](http://arxiv.org/pdf/2412.19015v1)

**Authors**: Keke Tang, Weiyao Ke, Weilong Peng, Xiaofei Wang, Ziyong Du, Zhize Wu, Peican Zhu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds are crucial for assessing and improving the adversarial robustness of 3D deep learning models. Traditional solutions strictly limit point displacement during attacks, making it challenging to balance imperceptibility with adversarial effectiveness. In this paper, we attribute the inadequate imperceptibility of adversarial attacks on point clouds to deviations from the underlying surface. To address this, we introduce a novel point-to-surface (P2S) field that adjusts adversarial perturbation directions by dragging points back to their original underlying surface. Specifically, we use a denoising network to learn the gradient field of the logarithmic density function encoding the shape's surface, and apply a distance-aware adjustment to perturbation directions during attacks, thereby enhancing imperceptibility. Extensive experiments show that adversarial attacks guided by our P2S field are more imperceptible, outperforming state-of-the-art methods.

摘要: 对点云的对抗攻击对于评估和提高3D深度学习模型的对抗稳健性至关重要。传统的解决方案严格限制攻击期间的点位移，使得平衡不可感知性与对抗有效性变得具有挑战性。在本文中，我们将点云对抗攻击的不可感知性不足归因于与底层表面的偏差。为了解决这个问题，我们引入了一种新型的点到面（P2 S）场，该场通过将点拖回其原始底层表面来调整对抗扰动方向。具体来说，我们使用去噪网络来学习编码形状表面的log密度函数的梯度场，并在攻击期间对扰动方向进行距离感知调整，从而增强不可感知性。大量实验表明，由我们的P2S领域引导的对抗攻击更难以察觉，性能优于最先进的方法。



## **50. Bridging Interpretability and Robustness Using LIME-Guided Model Refinement**

使用LIME引导的模型细化来弥合可解释性和鲁棒性 cs.LG

10 pages, 15 figures

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18952v1) [paper-pdf](http://arxiv.org/pdf/2412.18952v1)

**Authors**: Navid Nayyem, Abdullah Rakin, Longwei Wang

**Abstract**: This paper explores the intricate relationship between interpretability and robustness in deep learning models. Despite their remarkable performance across various tasks, deep learning models often exhibit critical vulnerabilities, including susceptibility to adversarial attacks, over-reliance on spurious correlations, and a lack of transparency in their decision-making processes. To address these limitations, we propose a novel framework that leverages Local Interpretable Model-Agnostic Explanations (LIME) to systematically enhance model robustness. By identifying and mitigating the influence of irrelevant or misleading features, our approach iteratively refines the model, penalizing reliance on these features during training. Empirical evaluations on multiple benchmark datasets demonstrate that LIME-guided refinement not only improves interpretability but also significantly enhances resistance to adversarial perturbations and generalization to out-of-distribution data.

摘要: 本文探讨了深度学习模型中可解释性和稳健性之间的复杂关系。尽管深度学习模型在各种任务中表现出色，但它们往往表现出严重的漏洞，包括容易受到对抗攻击、过度依赖虚假相关性以及决策过程缺乏透明度。为了解决这些限制，我们提出了一种新颖的框架，该框架利用本地可解释模型不可知解释（LIME）来系统性地增强模型稳健性。通过识别和减轻不相关或误导性特征的影响，我们的方法迭代地完善模型，惩罚训练期间对这些特征的依赖。对多个基准数据集的经验评估表明，LIME引导的细化不仅提高了可解释性，而且显着增强了对对抗性扰动的抵抗力和对非分布数据的概括。



