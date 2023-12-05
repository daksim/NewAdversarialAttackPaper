# Latest Adversarial Attack Papers
**update at 2023-12-05 21:17:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整的定向攻击 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型（LVLM）已经证明了它们在图像理解和响应生成方面令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性示例的攻击。在本文中，我们制定了一个新的和实用的灰盒攻击的情况下，对手只能访问受害者LVLM的视觉编码器，而不知道其提示（这往往是专有的服务提供商和不公开）和其底层的大语言模型（LLM）。这种实际设置对针对性对抗攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种防御调整的有针对性的攻击（称为指令TA），以提供具有高可转移性的LVLM上的有针对性的对抗攻击。首先，我们利用一个公共的文本到图像生成模型来“反转”目标响应到目标图像，并采用GPT-4从目标响应中推断出合理的指令$\boldsymbol{p}^\prime$。然后，我们形成一个本地代理模型（与受害者LVLM共享相同的视觉编码器）来提取对抗图像示例和目标图像的防御感知特征，并最小化这两个特征之间的距离以优化对抗示例。为了进一步提高可移植性，我们增加了指令$\boldsymbol{p}^\prime$与从LLM解释的指令。大量的实验表明，我们提出的方法在有针对性的攻击性能和可移植性的优越性。



## **2. Two-stage optimized unified adversarial patch for attacking visible-infrared cross-modal detectors in the physical world**

用于攻击物理世界中可见光-红外交叉模式探测器的两阶段优化统一对抗补丁 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01789v1) [paper-pdf](http://arxiv.org/pdf/2312.01789v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Currently, many studies have addressed security concerns related to visible and infrared detectors independently. In practical scenarios, utilizing cross-modal detectors for tasks proves more reliable than relying on single-modal detectors. Despite this, there is a lack of comprehensive security evaluations for cross-modal detectors. While existing research has explored the feasibility of attacks against cross-modal detectors, the implementation of a robust attack remains unaddressed. This work introduces the Two-stage Optimized Unified Adversarial Patch (TOUAP) designed for performing attacks against visible-infrared cross-modal detectors in real-world, black-box settings. The TOUAP employs a two-stage optimization process: firstly, PSO optimizes an irregular polygonal infrared patch to attack the infrared detector; secondly, the color QR code is optimized, and the shape information of the infrared patch from the first stage is used as a mask. The resulting irregular polygon visible modal patch executes an attack on the visible detector. Through extensive experiments conducted in both digital and physical environments, we validate the effectiveness and robustness of the proposed method. As the TOUAP surpasses baseline performance, we advocate for its widespread attention.

摘要: 目前，许多研究已经独立地解决了与可见光和红外探测器相关的安全问题。在实际场景中，使用跨模式检测器执行任务被证明比依赖单模式检测器更可靠。尽管如此，目前还缺乏对跨模式探测器的全面安全评估。虽然现有的研究已经探索了针对跨模式检测器的攻击的可行性，但健壮攻击的实现仍然没有得到解决。这项工作介绍了两阶段优化的统一对抗补丁(TOUAP)，设计用于在现实世界的黑匣子环境中执行对可见光-红外交叉模式探测器的攻击。TOUAP算法采用两阶段优化过程：首先，粒子群算法对攻击红外探测器的不规则多边形红外贴片进行优化；其次，对颜色二维码进行优化，并将第一阶段得到的红外贴片的形状信息作为掩码。所得到的不规则多边形可见模式面片对可见检测器执行攻击。通过在数字和物理环境中进行的大量实验，验证了该方法的有效性和稳健性。由于TOUAP超过了基线表现，我们主张广泛关注它。



## **3. Malicious Lateral Movement in 5G Core With Network Slicing And Its Detection**

基于网络分片的5G核心恶意侧移及其检测 cs.CR

Accepted for publication in the Proceedings of IEEE ITNAC-2023

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01681v1) [paper-pdf](http://arxiv.org/pdf/2312.01681v1)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: 5G networks are susceptible to cyber attacks due to reasons such as implementation issues and vulnerabilities in 3GPP standard specifications. In this work, we propose lateral movement strategies in a 5G Core (5GC) with network slicing enabled, as part of a larger attack campaign by well-resourced adversaries such as APT groups. Further, we present 5GLatte, a system to detect such malicious lateral movement. 5GLatte operates on a host-container access graph built using host/NF container logs collected from the 5GC. Paths inferred from the access graph are scored based on selected filtering criteria and subsequently presented as input to a threshold-based anomaly detection algorithm to reveal malicious lateral movement paths. We evaluate 5GLatte on a dataset containing attack campaigns (based on MITRE ATT&CK and FiGHT frameworks) launched in a 5G test environment which shows that compared to other lateral movement detectors based on state-of-the-art, it can achieve higher true positive rates with similar false positive rates.

摘要: 由于3GPP标准规范中的实现问题和漏洞等原因，5G网络容易受到网络攻击。在这项工作中，我们提出了启用网络切片的5G核心(5GC)中的横向移动策略，作为资源丰富的对手(如APT组)更大攻击活动的一部分。此外，我们还提出了5G Latte，这是一个检测此类恶意横向移动的系统。5GLatte在使用从5GC收集的主机/NF容器日志构建的主机-容器访问图上运行。根据选择的过滤标准对从访问图推断的路径进行评分，并随后将其作为基于阈值的异常检测算法的输入，以揭示恶意的横向移动路径。我们在包含5G测试环境中发起的攻击活动(基于MITRE ATT&CK和Fight框架)的数据集上对5G Latte进行了评估，结果表明，与其他基于最新技术的侧向运动检测器相比，它可以在相似的误警率下获得更高的真阳性率。



## **4. Adversarial Medical Image with Hierarchical Feature Hiding**

基于分层特征隐藏的对抗性医学图像 eess.IV

Our code is available at  \url{https://github.com/qsyao/Hierarchical_Feature_Constraint}

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01679v1) [paper-pdf](http://arxiv.org/pdf/2312.01679v1)

**Authors**: Qingsong Yao, Zecheng He, Yuexiang Li, Yi Lin, Kai Ma, Yefeng Zheng, S. Kevin Zhou

**Abstract**: Deep learning based methods for medical images can be easily compromised by adversarial examples (AEs), posing a great security flaw in clinical decision-making. It has been discovered that conventional adversarial attacks like PGD which optimize the classification logits, are easy to distinguish in the feature space, resulting in accurate reactive defenses. To better understand this phenomenon and reassess the reliability of the reactive defenses for medical AEs, we thoroughly investigate the characteristic of conventional medical AEs. Specifically, we first theoretically prove that conventional adversarial attacks change the outputs by continuously optimizing vulnerable features in a fixed direction, thereby leading to outlier representations in the feature space. Then, a stress test is conducted to reveal the vulnerability of medical images, by comparing with natural images. Interestingly, this vulnerability is a double-edged sword, which can be exploited to hide AEs. We then propose a simple-yet-effective hierarchical feature constraint (HFC), a novel add-on to conventional white-box attacks, which assists to hide the adversarial feature in the target feature distribution. The proposed method is evaluated on three medical datasets, both 2D and 3D, with different modalities. The experimental results demonstrate the superiority of HFC, \emph{i.e.,} it bypasses an array of state-of-the-art adversarial medical AE detectors more efficiently than competing adaptive attacks, which reveals the deficiencies of medical reactive defense and allows to develop more robust defenses in future.

摘要: 基于深度学习的医学图像处理方法容易受到对抗性实例的攻击，在临床决策中存在很大的安全缺陷。已经发现，像PGD这样的传统对抗性攻击优化了分类逻辑，在特征空间中很容易区分，从而产生准确的反应性防御。为了更好地理解这一现象，并重新评估医用AEs反应性防御的可靠性，我们深入研究了传统医用AEs的特点。具体地说，我们首先从理论上证明了传统的对抗性攻击通过在固定方向上不断优化易受攻击的特征来改变输出，从而导致特征空间中的孤立点表示。然后，通过与自然图像的比较，进行了压力测试，揭示了医学图像的脆弱性。有趣的是，这个漏洞是一把双刃剑，可以被利用来隐藏AE。然后，我们提出了一种简单有效的层次特征约束(HFC)，这是对传统白盒攻击的一种新的补充，它帮助隐藏目标特征分布中的对抗性特征。该方法在三个医学数据集上进行了评估，包括2D和3D，使用不同的模式。实验结果证明了HFC的优越性，即它比竞争的自适应攻击更有效地绕过了一系列最先进的对抗性医学AE检测器，这揭示了医学反应性防御的不足，并为未来开发更健壮的防御奠定了基础。



## **5. The Queen's Guard: A Secure Enforcement of Fine-grained Access Control In Distributed Data Analytics Platforms**

女王卫队：分布式数据分析平台中细粒度访问控制的安全执行 cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2106.13123v4) [paper-pdf](http://arxiv.org/pdf/2106.13123v4)

**Authors**: Fahad Shaon, Sazzadur Rahaman, Murat Kantarcioglu

**Abstract**: Distributed data analytics platforms (i.e., Apache Spark, Hadoop) provide high-level APIs to programmatically write analytics tasks that are run distributedly in multiple computing nodes. The design of these frameworks was primarily motivated by performance and usability. Thus, the security takes a back seat. Consequently, they do not inherently support fine-grained access control or offer any plugin mechanism to enable it, making them risky to be used in multi-tier organizational settings.   There have been attempts to build "add-on" solutions to enable fine-grained access control for distributed data analytics platforms. In this paper, first, we show that straightforward enforcement of ``add-on'' access control is insecure under adversarial code execution. Specifically, we show that an attacker can abuse platform-provided APIs to evade access controls without leaving any traces. Second, we designed a two-layered (i.e., proactive and reactive) defense system to protect against API abuses. On submission of a user code, our proactive security layer statically screens it to find potential attack signatures prior to its execution. The reactive security layer employs code instrumentation-based runtime checks and sandboxed execution to throttle any exploits at runtime. Next, we propose a new fine-grained access control framework with an enhanced policy language that supports map and filter primitives. Finally, we build a system named SecureDL with our new access control framework and defense system on top of Apache Spark, which ensures secure access control policy enforcement under adversaries capable of executing code.   To the best of our knowledge, this is the first fine-grained attribute-based access control framework for distributed data analytics platforms that is secure against platform API abuse attacks. Performance evaluation showed that the overhead due to added security is low.

摘要: 分布式数据分析平台(即，ApacheSpark、Hadoop)提供高级API，以编程方式编写在多个计算节点上分布式运行的分析任务。这些框架的设计主要是出于性能和可用性的考虑。因此，安全措施就退居次要地位了。因此，它们本身并不支持细粒度的访问控制，也不提供任何插件机制来启用它，这使得它们在多层组织设置中使用存在风险。已经有人尝试构建“附加”解决方案来实现分布式数据分析平台的细粒度访问控制。在这篇文章中，我们首先证明了直接实施“附加”访问控制在恶意代码执行下是不安全的。具体地说，我们展示了攻击者可以滥用平台提供的API来逃避访问控制，而不会留下任何痕迹。其次，我们设计了一个双层(即主动和被动)防御体系，以防止API滥用。在提交用户代码时，我们的主动安全层会在代码执行之前对其进行静态筛选，以发现潜在的攻击特征。反应式安全层采用基于代码检测的运行时检查和沙箱执行，以在运行时遏制任何利用漏洞。接下来，我们提出了一种新的细粒度访问控制框架，该框架具有增强的策略语言，支持映射和过滤原语。最后，我们使用新的访问控制框架和防御系统在ApacheSpark之上构建了一个名为SecureDL的系统，确保了在能够执行代码的攻击者的情况下安全地执行访问控制策略。据我们所知，这是第一个针对分布式数据分析平台的细粒度基于属性的访问控制框架，可以安全地抵御平台API滥用攻击。性能评估表明，由于增加安全性而产生的开销很低。



## **6. Robust Evaluation of Diffusion-Based Adversarial Purification**

基于扩散的对抗净化算法的稳健性评价 cs.CV

Accepted by ICCV 2023, oral presentation. Code is available at  https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2303.09051v3) [paper-pdf](http://arxiv.org/pdf/2303.09051v3)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.

摘要: 我们质疑目前基于扩散的净化方法的评价实践。基于扩散的净化方法旨在在测试时从输入数据点中去除对抗效应。由于训练和测试之间的分离，该方法作为对抗训练的替代方案越来越受到关注。众所周知的白盒攻击经常被用来衡量纯化的鲁棒性。然而，目前还不清楚这些攻击是否对基于扩散的净化最有效，因为这些攻击通常是针对对抗训练而定制的。我们分析了当前的实践，并提供了一个新的准则来衡量对抗攻击的净化方法的鲁棒性。基于我们的分析，我们进一步提出了一个新的净化策略，提高鲁棒性相比，目前基于扩散的净化方法。



## **7. Exploring Adversarial Robustness of LiDAR-Camera Fusion Model in Autonomous Driving**

自动驾驶中LiDAR-Camera融合模型的对抗稳健性研究 cs.RO

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01468v1) [paper-pdf](http://arxiv.org/pdf/2312.01468v1)

**Authors**: Bo Yang, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji

**Abstract**: Our study assesses the adversarial robustness of LiDAR-camera fusion models in 3D object detection. We introduce an attack technique that, by simply adding a limited number of physically constrained adversarial points above a car, can make the car undetectable by the fusion model. Experimental results reveal that even without changes to the image data channel, the fusion model can be deceived solely by manipulating the LiDAR data channel. This finding raises safety concerns in the field of autonomous driving. Further, we explore how the quantity of adversarial points, the distance between the front-near car and the LiDAR-equipped car, and various angular factors affect the attack success rate. We believe our research can contribute to the understanding of multi-sensor robustness, offering insights and guidance to enhance the safety of autonomous driving.

摘要: 我们的研究评估了LiDAR-相机融合模型在3D目标检测中的对抗健壮性。我们介绍了一种攻击技术，只需在汽车上方添加有限数量的物理约束对手点，就可以使汽车无法被融合模型检测到。实验结果表明，即使在不改变图像数据通道的情况下，仅通过操纵LiDAR数据通道也可以欺骗融合模型。这一发现引发了自动驾驶领域的安全担忧。在此基础上，进一步探讨了攻击点的数量、前近车与装备激光雷达的车之间的距离以及各种角度因素对攻击成功率的影响。我们相信，我们的研究将有助于理解多传感器的稳健性，为提高自动驾驶的安全性提供见解和指导。



## **8. Evaluating the Security of Satellite Systems**

评估卫星系统的安全性 cs.CR

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01330v1) [paper-pdf](http://arxiv.org/pdf/2312.01330v1)

**Authors**: Roy Peled, Eran Aizikovich, Edan Habler, Yuval Elovici, Asaf Shabtai

**Abstract**: Satellite systems are facing an ever-increasing amount of cybersecurity threats as their role in communications, navigation, and other services expands. Recent papers have examined attacks targeting satellites and space systems; however, they did not comprehensively analyze the threats to satellites and systematically identify adversarial techniques across the attack lifecycle. This paper presents a comprehensive taxonomy of adversarial tactics, techniques, and procedures explicitly targeting LEO satellites. First, we analyze the space ecosystem including the ground, space, Communication, and user segments, highlighting their architectures, functions, and vulnerabilities. Then, we examine the threat landscape, including adversary types, and capabilities, and survey historical and recent attacks such as jamming, spoofing, and supply chain. Finally, we propose a novel extension of the MITRE ATT&CK framework to categorize satellite attack techniques across the adversary lifecycle from reconnaissance to impact. The taxonomy is demonstrated by modeling high-profile incidents, including the Viasat attack that disrupted Ukraine's communications. The taxonomy provides the foundation for the development of defenses against emerging cyber risks to space assets. The proposed threat model will advance research in the space domain and contribute to the security of the space domain against sophisticated attacks.

摘要: 随着卫星系统在通信、导航和其他服务中的作用不断扩大，它们正面临着越来越多的网络安全威胁。最近的论文审查了针对卫星和空间系统的攻击；然而，它们没有全面分析对卫星的威胁，并在整个攻击生命周期中系统地确定对抗技术。本文对明确针对低轨卫星的对抗战术、技术和程序进行了全面的分类。首先，我们分析了空间生态系统，包括地面、空间、通信和用户部分，重点介绍了它们的架构、功能和漏洞。然后，我们检查威胁环境，包括对手类型和能力，并调查历史和最近的攻击，如干扰、欺骗和供应链。最后，我们提出了一种新的扩展MITRE ATT&CK框架，用于对从侦察到影响的整个敌方生命周期的卫星攻击技术进行分类。通过对备受瞩目的事件进行建模，包括中断乌克兰通信的Viasat袭击，展示了这一分类。该分类为开发针对空间资产新出现的网络风险的防御措施提供了基础。拟议的威胁模型将推动空间领域的研究，并有助于空间领域免受复杂攻击的安全。



## **9. Rethinking PGD Attack: Is Sign Function Necessary?**

对PGD攻击的再思考：是否需要符号功能？ cs.LG

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01260v1) [paper-pdf](http://arxiv.org/pdf/2312.01260v1)

**Authors**: Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang

**Abstract**: Neural networks have demonstrated success in various domains, yet their performance can be significantly degraded by even a small input perturbation. Consequently, the construction of such perturbations, known as adversarial attacks, has gained significant attention, many of which fall within "white-box" scenarios where we have full access to the neural network. Existing attack algorithms, such as the projected gradient descent (PGD), commonly take the sign function on the raw gradient before updating adversarial inputs, thereby neglecting gradient magnitude information. In this paper, we present a theoretical analysis of how such sign-based update algorithm influences step-wise attack performance, as well as its caveat. We also interpret why previous attempts of directly using raw gradients failed. Based on that, we further propose a new raw gradient descent (RGD) algorithm that eliminates the use of sign. Specifically, we convert the constrained optimization problem into an unconstrained one, by introducing a new hidden variable of non-clipped perturbation that can move beyond the constraint. The effectiveness of the proposed RGD algorithm has been demonstrated extensively in experiments, outperforming PGD and other competitors in various settings, without incurring any additional computational overhead. The codes is available in https://github.com/JunjieYang97/RGD.

摘要: 神经网络已经在各个领域取得了成功，但即使是很小的输入扰动也会显著降低其性能。因此，这种被称为对抗性攻击的扰动的构造得到了极大的关注，其中许多都属于我们可以完全访问神经网络的“白盒”情景。现有的攻击算法，如投影梯度下降(PGD)算法，通常在更新敌方输入之前对原始梯度取符号函数，从而忽略了梯度大小信息。本文从理论上分析了这种基于符号的更新算法对分步攻击性能的影响，并给出了相应的警告。我们还解释了为什么以前直接使用原始梯度的尝试失败了。在此基础上，进一步提出了一种新的原始梯度下降(RGD)算法，该算法省去了符号的使用。具体地说，我们通过引入一个可以超越约束的非剪裁扰动的新的隐变量，将约束优化问题转化为无约束优化问题。所提出的RGD算法的有效性已经在实验中得到了广泛的证明，在不引起任何额外计算开销的情况下，在不同环境下的性能优于PGD和其他竞争对手。这些代码可以在https://github.com/JunjieYang97/RGD.中找到



## **10. Look Closer to Your Enemy: Learning to Attack via Teacher-Student Mimicking**

走近你的敌人：通过师生模仿学习攻击 cs.CV

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2207.13381v4) [paper-pdf](http://arxiv.org/pdf/2207.13381v4)

**Authors**: Mingjie Wang, Jianxiong Guo, Sirui Li, Dingwen Xiao, Zhiqing Tang

**Abstract**: Deep neural networks have significantly advanced person re-identification (ReID) applications in the realm of the industrial internet, yet they remain vulnerable. Thus, it is crucial to study the robustness of ReID systems, as there are risks of adversaries using these vulnerabilities to compromise industrial surveillance systems. Current adversarial methods focus on generating attack samples using misclassification feedback from victim models (VMs), neglecting VM's cognitive processes. We seek to address this by producing authentic ReID attack instances through VM cognition decryption. This approach boasts advantages like better transferability to open-set ReID tests, easier VM misdirection, and enhanced creation of realistic and undetectable assault images. However, the task of deciphering the cognitive mechanism in VM is widely considered to be a formidable challenge. In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE (Look Closer to Your Enemy), to generate adversarial query images. Specifically, LCYE first distills VM's knowledge via teacher-student memory mimicking the proxy task. This knowledge prior serves as an unambiguous cryptographic token, encapsulating elements deemed indispensable and plausible by the VM, with the intent of facilitating precise adversarial misdirection. Further, benefiting from the multiple opposing task framework of LCYE, we investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. The source code can be found at https://github.com/MingjieWang0606/LCYE-attack_reid.

摘要: 深度神经网络在工业互联网领域显著推进了个人重新识别(ReID)的应用，但它们仍然很脆弱。因此，研究REID系统的健壮性是至关重要的，因为存在攻击者利用这些漏洞来危害工业监控系统的风险。目前的攻击方法主要是利用受害者模型的错误分类反馈来生成攻击样本，而忽略了受害者模型的认知过程。我们试图通过VM认知解密生成真实的Reid攻击实例来解决这个问题。这种方法拥有更好的可移植到开放设置的Reid测试，更容易的VM误导，以及增强的真实和不可检测的攻击图像的创建等优势。然而，破译VM的认知机制被广泛认为是一项艰巨的挑战。在本文中，我们提出了一种新的隐蔽和可控的Reid攻击基线，LCYE(Look Closer To Your Enemy)，用于生成敌意查询图像。具体地说，LCYE首先通过模仿代理任务的师生记忆提取VM的知识。该知识先验用作明确的加密令牌，封装了被VM认为必不可少且看似可信的元素，目的是促进精确的对抗性误导。此外，借鉴LCYE的多重对立任务框架，我们从对抗性攻击的角度考察了Reid模型的可解释性和泛化，包括跨域适应、跨模型共识和在线学习过程。在四个Reid基准测试上的大量实验表明，我们的方法在白盒、黑盒和目标攻击中的性能远远优于其他最先进的攻击者。源代码可以在https://github.com/MingjieWang0606/LCYE-attack_reid.上找到



## **11. FRAUDability: Estimating Users' Susceptibility to Financial Fraud Using Adversarial Machine Learning**

欺诈性：使用对抗性机器学习估计用户对金融欺诈的敏感性 cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.01200v1) [paper-pdf](http://arxiv.org/pdf/2312.01200v1)

**Authors**: Chen Doytshman, Satoru Momiyama, Inderjeet Singh, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, financial fraud detection systems have become very efficient at detecting fraud, which is a major threat faced by e-commerce platforms. Such systems often include machine learning-based algorithms aimed at detecting and reporting fraudulent activity. In this paper, we examine the application of adversarial learning based ranking techniques in the fraud detection domain and propose FRAUDability, a method for the estimation of a financial fraud detection system's performance for every user. We are motivated by the assumption that "not all users are created equal" -- while some users are well protected by fraud detection algorithms, others tend to pose a challenge to such systems. The proposed method produces scores, namely "fraudability scores," which are numerical estimations of a fraud detection system's ability to detect financial fraud for a specific user, given his/her unique activity in the financial system. Our fraudability scores enable those tasked with defending users in a financial platform to focus their attention and resources on users with high fraudability scores to better protect them. We validate our method using a real e-commerce platform's dataset and demonstrate the application of fraudability scores from the attacker's perspective, on the platform, and more specifically, on the fraud detection systems used by the e-commerce enterprise. We show that the scores can also help attackers increase their financial profit by 54%, by engaging solely with users with high fraudability scores, avoiding those users whose spending habits enable more accurate fraud detection.

摘要: 近年来，金融欺诈检测系统在检测欺诈方面变得非常高效，这是电子商务平台面临的一大威胁。这类系统通常包括旨在检测和报告欺诈活动的基于机器学习的算法。本文研究了基于对抗性学习的排序技术在欺诈检测领域中的应用，并提出了一种估计每个用户的金融欺诈检测系统性能的方法FRAUDability。我们的动机是这样一个假设：“并非所有用户都是生而平等的”--虽然一些用户受到欺诈检测算法的良好保护，但其他用户往往会对这样的系统构成挑战。建议的方法产生分数，即“欺诈性分数”，这是对欺诈检测系统在给定特定用户在金融系统中的独特活动的情况下为其检测金融欺诈的能力的数字估计。我们的欺诈性得分使那些在金融平台中负责保护用户的人能够将他们的注意力和资源集中在欺诈性得分较高的用户上，以更好地保护他们。我们使用一个真实的电子商务平台的数据集来验证我们的方法，并从攻击者的角度演示了欺诈性分数在平台上的应用，更具体地说，在电子商务企业使用的欺诈检测系统上的应用。我们发现，这些分数还可以帮助攻击者增加54%的经济利润，只与欺诈性分数较高的用户打交道，避开那些消费习惯能够更准确地检测欺诈的用户。



## **12. Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions**

图神经网络的对抗性训练：陷阱、解决方案和新方向 cs.LG

Published as a conference paper at NeurIPS 2023

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2306.15427v2) [paper-pdf](http://arxiv.org/pdf/2306.15427v2)

**Authors**: Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

**Abstract**: Despite its success in the image domain, adversarial training did not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that more flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.

摘要: 尽管它在图像领域取得了成功，但对抗性训练(目前还没有)成为图神经网络(GNN)对抗图结构扰动的有效防御。在固定对抗性训练的过程中，(1)我们展示并克服了以前工作中采用的图学习设置的基本理论和实践限制；(2)我们揭示了基于可学习图扩散的更灵活的GNN能够适应对抗性扰动，而学习的消息传递方案自然是可解释的；(3)我们引入了针对结构扰动的第一次攻击，虽然一次针对多个节点，但能够处理全局(图级)和局部(节点级)约束。包括这些贡献，我们证明了对抗性训练是对对抗性结构扰动的一种最先进的防御。



## **13. Scrappy: SeCure Rate Assuring Protocol with PrivacY**

Scarppy：带隐私的安全速率保证协议 cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00989v1) [paper-pdf](http://arxiv.org/pdf/2312.00989v1)

**Authors**: Kosei Akama, Yoshimichi Nakatsuka, Masaaki Sato, Keisuke Uehara

**Abstract**: Preventing abusive activities caused by adversaries accessing online services at a rate exceeding that expected by websites has become an ever-increasing problem. CAPTCHAs and SMS authentication are widely used to provide a solution by implementing rate limiting, although they are becoming less effective, and some are considered privacy-invasive. In light of this, many studies have proposed better rate-limiting systems that protect the privacy of legitimate users while blocking malicious actors. However, they suffer from one or more shortcomings: (1) assume trust in the underlying hardware and (2) are vulnerable to side-channel attacks. Motivated by the aforementioned issues, this paper proposes Scrappy: SeCure Rate Assuring Protocol with PrivacY. Scrappy allows clients to generate unforgeable yet unlinkable rate-assuring proofs, which provides the server with cryptographic guarantees that the client is not misbehaving. We design Scrappy using a combination of DAA and hardware security devices. Scrappy is implemented over three types of devices, including one that can immediately be deployed in the real world. Our baseline evaluation shows that the end-to-end latency of Scrappy is minimal, taking only 0.32 seconds, and uses only 679 bytes of bandwidth when transferring necessary data. We also conduct an extensive security evaluation, showing that the rate-limiting capability of Scrappy is unaffected even if the hardware security device is compromised.

摘要: 防止对手以超出网站预期的速度访问在线服务导致的滥用活动已成为一个日益严重的问题。验证码和短信身份验证被广泛用于通过实施速率限制来提供解决方案，尽管它们正在变得不那么有效，而且有些被认为是侵犯隐私的。有鉴于此，许多研究提出了更好的速率限制系统，在保护合法用户隐私的同时阻止恶意行为者。然而，它们有一个或多个缺点：(1)信任底层硬件；(2)容易受到旁路攻击。基于上述问题，本文提出了Scrppy：带隐私的安全速率保证协议。Screppy允许客户端生成不可伪造但不可链接的费率保证证据，这为服务器提供了客户端没有行为不端的加密保证。我们使用DAA和硬件安全设备的组合来设计Screppy。Scrppy在三种类型的设备上实现，其中一种可以立即部署到现实世界中。我们的基线评估表明，Scarppy的端到端延迟最小，仅需0.32秒，在传输必要的数据时仅占用679字节的带宽。我们还进行了广泛的安全评估，表明即使硬件安全设备被攻破，Screppy的限速能力也不受影响。



## **14. Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification**

数据驱动离线签名验证的深度生成攻击及对策 cs.CV

10 pages, 7 figures, 1 table, Signature verification, Deep generative  models, attacks, generative attack explainability, data-driven verification  system

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00987v1) [paper-pdf](http://arxiv.org/pdf/2312.00987v1)

**Authors**: An Ngo, MinhPhuong Cao, Rajesh Kumar

**Abstract**: While previous studies have explored attacks via random, simple, and skilled forgeries, generative attacks have received limited attention in the data-driven signature verification (DASV) process. Thus, this paper explores the impact of generative attacks on DASV and proposes practical and interpretable countermeasures. We investigate the power of two prominent Deep Generative Models (DGMs), Variational Auto-encoders (VAE) and Conditional Generative Adversarial Networks (CGAN), on their ability to generate signatures that would successfully deceive DASV. Additionally, we evaluate the quality of generated images using the Structural Similarity Index measure (SSIM) and use the same to explain the attack's success. Finally, we propose countermeasures that effectively reduce the impact of deep generative attacks on DASV.   We first generated six synthetic datasets from three benchmark offline-signature datasets viz. CEDAR, BHSig260- Bengali, and BHSig260-Hindi using VAE and CGAN. Then, we built baseline DASVs using Xception, ResNet152V2, and DenseNet201. These DASVs achieved average (over the three datasets) False Accept Rates (FARs) of 2.55%, 3.17%, and 1.06%, respectively. Then, we attacked these baselines using the synthetic datasets. The VAE-generated signatures increased average FARs to 10.4%, 10.1%, and 7.5%, while CGAN-generated signatures to 32.5%, 30%, and 26.1%. The variation in the effectiveness of attack for VAE and CGAN was investigated further and explained by a strong (rho = -0.86) negative correlation between FARs and SSIMs. We created another set of synthetic datasets and used the same to retrain the DASVs. The retained baseline showed significant robustness to random, skilled, and generative attacks as the FARs shrank to less than 1% on average. The findings underscore the importance of studying generative attacks and potential countermeasures for DASV.

摘要: 虽然以前的研究探索了通过随机、简单和熟练的伪造进行的攻击，但在数据驱动的签名验证(DASV)过程中，生成性攻击受到的关注有限。因此，本文探讨了产生性攻击对DASV的影响，并提出了切实可行的、可解释的对策。我们研究了两个重要的深度生成模型(DGM)--变分自动编码器(VAE)和条件生成对抗网络(CGAN)--对它们生成成功欺骗DASV的签名的能力的影响。此外，我们使用结构相似性指数(SSIM)来评估生成的图像的质量，并用它来解释攻击的成功。最后，提出了有效降低深度生成性攻击对DASV影响的对策。我们首先从三个基准离线签名数据集生成六个合成数据集，即。雪松、BHSig260-孟加拉语和BHSig260-印地语，使用VAE和CGAN。然后，我们使用Xception、ResNet152V2和DenseNet201构建了基准DASV。这些DASV在三个数据集上的平均错误接受率(FAR)分别为2.55%、3.17%和1.06%。然后，我们使用合成数据集攻击这些基线。VAE生成的签名将平均FAR提高到10.4%、10.1%和7.5%，而CGAN生成的签名增加到32.5%、30%和26.1%。对VAE和CGAN攻击效能的变化进行了进一步的研究，并用FARS和SSIM之间的强负相关(Rho=-0.86)来解释。我们创建了另一组合成数据集，并使用相同的数据集重新训练DASV。保留的基线对随机、熟练和生成性攻击显示出显著的稳健性，因为FAR平均收缩到不到1%。这些发现强调了研究DASV的生成性攻击和潜在对策的重要性。



## **15. Stealing the Decoding Algorithms of Language Models**

窃取语言模型的译码算法 cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2303.04729v4) [paper-pdf](http://arxiv.org/pdf/2303.04729v4)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2, GPT-3 and GPT-Neo. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.

摘要: 从现代语言模型(LM)生成文本的一个关键组件是解码算法的选择和调整。这些算法确定如何从LM生成的内部概率分布生成文本。选择解码算法和调整其超参数的过程需要大量的时间、人工和计算，还需要广泛的人工评估。因此，这种译码算法的恒等式和超参数被认为对它们的所有者非常有价值。在这项工作中，我们首次证明，具有典型API访问权限的攻击者可以以非常低的金钱成本窃取其解码算法的类型和超参数。我们的攻击对文本生成API中使用的流行LMS有效，包括GPT-2、GPT-3和GPT-Neo。我们证明了只需几美元即可窃取此类信息的可行性，例如，对于GPT-3的四个版本，仅需$0.8$、$1$、$4$和$40$。



## **16. Adversarial Attacks and Defenses on 3D Point Cloud Classification: A Survey**

三维点云分类的对抗性攻击与防御研究综述 cs.CV

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2307.00309v2) [paper-pdf](http://arxiv.org/pdf/2307.00309v2)

**Authors**: Hanieh Naderi, Ivan V. Bajić

**Abstract**: Deep learning has successfully solved a wide range of tasks in 2D vision as a dominant AI technique. Recently, deep learning on 3D point clouds is becoming increasingly popular for addressing various tasks in this field. Despite remarkable achievements, deep learning algorithms are vulnerable to adversarial attacks. These attacks are imperceptible to the human eye but can easily fool deep neural networks in the testing and deployment stage. To encourage future research, this survey summarizes the current progress on adversarial attack and defense techniques on point cloud classification.This paper first introduces the principles and characteristics of adversarial attacks and summarizes and analyzes adversarial example generation methods in recent years. Additionally, it provides an overview of defense strategies, organized into data-focused and model-focused methods. Finally, it presents several current challenges and potential future research directions in this domain.

摘要: 深度学习作为一种占主导地位的人工智能技术，已经成功地解决了2D视觉中的一系列任务。近年来，针对三维点云的深度学习成为解决该领域各种问题的热门方法。尽管深度学习算法取得了令人瞩目的成就，但它仍然容易受到对手的攻击。这些攻击是人眼看不见的，但在测试和部署阶段很容易就能愚弄深度神经网络。本文首先介绍了对抗性攻击的原理和特点，并对近年来的对抗性实例生成方法进行了总结和分析。此外，它还概述了防御策略，并将其组织为以数据为中心和以模型为中心的方法。最后提出了该领域目前面临的几个挑战和潜在的研究方向。



## **17. A Unified Approach to Interpreting and Boosting Adversarial Transferability**

一种统一的解释和提高对抗性转移能力的方法 cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2010.04055v2) [paper-pdf](http://arxiv.org/pdf/2010.04055v2)

**Authors**: Xin Wang, Jie Ren, Shuyun Lin, Xiangming Zhu, Yisen Wang, Quanshi Zhang

**Abstract**: In this paper, we use the interaction inside adversarial perturbations to explain and boost the adversarial transferability. We discover and prove the negative correlation between the adversarial transferability and the interaction inside adversarial perturbations. The negative correlation is further verified through different DNNs with various inputs. Moreover, this negative correlation can be regarded as a unified perspective to understand current transferability-boosting methods. To this end, we prove that some classic methods of enhancing the transferability essentially decease interactions inside adversarial perturbations. Based on this, we propose to directly penalize interactions during the attacking process, which significantly improves the adversarial transferability.

摘要: 在本文中，我们使用对抗性扰动内部的相互作用来解释和增强对抗性转移。我们发现并证明了对抗性转移与对抗性扰动中的相互作用之间的负相关关系。通过具有不同输入的不同DNN进一步验证了负相关性。此外，这种负相关性可以被视为理解当前可转让性提升方法的统一视角。为此，我们证明了一些增强可转移性的经典方法本质上减少了对抗性扰动中的相互作用。基于此，我们提出了直接惩罚攻击过程中的交互，这显著提高了对手的可转移性。



## **18. PyraTrans: Learning Attention-Enriched Multi-Scale Pyramid Network from Pre-Trained Transformers for Effective Malicious URL Detection**

金字塔：从预先训练的变形金字塔中学习注意力丰富的多尺度金字塔网络，以实现有效的恶意URL检测 cs.CR

12 pages, 7 figures

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00508v1) [paper-pdf](http://arxiv.org/pdf/2312.00508v1)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Detecting malicious URLs is a crucial aspect of web search and mining, significantly impacting internet security. Though advancements in machine learning have improved the effectiveness of detection methods, these methods still face significant challenges in their capacity to generalize and their resilience against evolving threats. In this paper, we propose PyraTrans, an approach that combines the strengths of pretrained Transformers and pyramid feature learning for improving malicious URL detection. We implement PyraTrans by leveraging a pretrained CharBERT as the base and augmenting it with 3 connected feature modules: 1) The Encoder Feature Extraction module, which extracts representations from each encoder layer of CharBERT to obtain multi-order features; 2) The Multi-Scale Feature Learning Module, which captures multi-scale local contextual insights and aggregate information across different layer-levels; and 3) The Pyramid Spatial Attention Module, which learns hierarchical and spatial feature attentions, highlighting critical classification signals while reducing noise. The proposed approach addresses the limitations of the Transformer in local feature learning and spatial awareness, and enabling us to extract multi-order, multi-scale URL feature representations with enhanced attentional focus. PyraTrans is evaluated using 4 benchmark datasets, where it demonstrated significant advancements over prior baseline methods. Particularly, on the imbalanced dataset, our method, with just 10% of the data for training, the TPR is 3.3-6.5 times and the F1-score is 2.9-4.5 times that of the baseline. Our approach also demonstrates robustness against adversarial attacks. Codes and data are available at https://github.com/Alixyvtte/PyraTrans.

摘要: 检测恶意URL是Web搜索和挖掘的一个重要方面，对互联网安全产生重大影响。尽管机器学习的进步提高了检测方法的有效性，但这些方法在泛化能力和应对不断变化的威胁的能力方面仍然面临着重大挑战。在本文中，我们提出了PyraTrans，这是一种结合了预训练的Transformers和金字塔特征学习的优势的方法，用于改进恶意URL检测。我们通过利用预训练的CharBERT作为基础并使用3个连接的特征模块对其进行增强来实现PyraTrans：1）编码器特征提取模块，其从CharBERT的每个编码器层提取表示以获得多阶特征; 2）多尺度特征学习模块，其捕获多尺度局部上下文洞察并跨不同层次聚合信息;以及3）金字塔空间注意力模块，其学习分层和空间特征注意力，突出关键分类信号，同时减少噪声。该方法解决了Transformer在局部特征学习和空间感知方面的局限性，使我们能够提取多阶、多尺度的URL特征表示，并增强注意力。PyraTrans使用4个基准数据集进行评估，在这些基准数据集上，它比以前的基线方法有了显着的进步。特别是，在不平衡数据集上，我们的方法，只有10%的数据用于训练，TPR是基线的3.3-6.5倍，F1分数是基线的2.9-4.5倍。我们的方法也证明了对抗性攻击的鲁棒性。代码和数据可在https://github.com/Alixyvtte/PyraTrans上获得。



## **19. Unleashing Cheapfakes through Trojan Plugins of Large Language Models**

通过大型语言模型特洛伊木马插件释放Cheapfake cs.CR

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00374v1) [paper-pdf](http://arxiv.org/pdf/2312.00374v1)

**Authors**: Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.

摘要: 开源的大型语言模型(LLM)最近越来越受欢迎，因为它们的性能可以与专有的LLM相媲美。为了高效地完成领域专门化任务，可以使用低级别适配器对开源LLM进行提炼，而无需使用昂贵的加速器。然而，是否可以利用低阶适配器来控制LLM仍然是未知的。为了弥补这一漏洞，我们演示了受感染的适配器可以在特定触发下诱导LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击方法，磨光攻击和融合攻击，它们比以前的方法有所改进。波兰德使用LLM增强的释义来抛光基准有毒数据集。相比之下，在没有数据集的情况下，Fusion利用过度中毒的程序来转换良性适配器。我们的实验验证了我们的攻击提供了比基线更高的攻击效率，并且为了吸引下载的目的，保留或提高了适配器的实用性。最后，我们提供了两个案例研究来演示特洛伊木马适配器可以导致LLM驱动的自主代理执行意外脚本或发送钓鱼电子邮件。我们的新型攻击首次通过特洛伊木马插件的镜头研究了LLM的供应链威胁。



## **20. Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense**

具有信息增益的贝叶斯学习被证明是强健对抗防御的风险界限 cs.LG

Published at ICML 2022. Code is available at  https://github.com/baogiadoan/IG-BNN

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.02003v2) [paper-pdf](http://arxiv.org/pdf/2212.02003v2)

**Authors**: Bao Gia Doan, Ehsan Abbasnejad, Javen Qinfeng Shi, Damith C. Ranasinghe

**Abstract**: We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the adversarial learning approach for approximating the multi-modal posterior distribution of a Bayesian model can lead to mode collapse; consequently, the model's achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step toward a basis for a principled method of adversarially training BNNs. Our model demonstrate significantly improved robustness--up to 20%--compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 datasets.

摘要: 提出了一种新的学习深度神经网络模型的算法，该模型具有较强的抗攻击能力。以前的算法表明，反向训练的贝叶斯神经网络(BNN)提供了更好的鲁棒性。我们认识到，用对抗性学习方法来逼近贝叶斯模型的多模式后验分布可能会导致模式崩溃，因此，该模型在稳健性和性能方面的成就是次优的。相反，我们首先提出防止模式崩溃，以更好地逼近多模式的后验分布。其次，基于健壮模型应该忽略扰动而只考虑输入的信息内容的直觉，我们概念化和制定了一个信息增益目标来衡量和强制从良性和对抗性训练实例中学习到的信息相似。重要的是。我们证明并证明了最小化信息收益目标使对手风险接近于传统的经验风险。我们相信，我们的努力为对抗性训练BNN的原则性方法奠定了基础。在CIFAR-10和STL-10数据集上，与对抗性训练和ADV-BNN相比，在具有0.035失真的PGD攻击下，我们的模型表现出了高达20%的健壮性。



## **21. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

虚假数据注入攻击下的大规模网络安全防御：一种攻击检测调度方法 eess.SY

14 pages, 11 figures

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.05500v3) [paper-pdf](http://arxiv.org/pdf/2212.05500v3)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large-scale networks, communication links between nodes are easily injected with false data by adversaries. This paper proposes a novel security defense strategy from the perspective of attack detection scheduling to ensure the security of the network. Based on the proposed strategy, each sensor can directly exclude suspicious sensors from its neighboring set. First, the problem of selecting suspicious sensors is formulated as a combinatorial optimization problem, which is non-deterministic polynomial-time hard (NP-hard). To solve this problem, the original function is transformed into a submodular function. Then, we propose a distributed attack detection scheduling algorithm based on the sequential submodular optimization theory, which incorporates \emph{expert problem} to better utilize historical information to guide the sensor selection task at the current moment. For different attack strategies, theoretical results show that the average optimization rate of the proposed algorithm has a lower bound, and the error expectation for any subset is bounded. In addition, under two kinds of insecurity conditions, the proposed algorithm can guarantee the security of the entire network from the perspective of the augmented estimation error. Finally, the effectiveness of the proposed method is verified by the numerical simulation and practical experiment.

摘要: 在大规模网络中，节点之间的通信链路很容易被对手注入虚假数据。本文从攻击检测调度的角度提出了一种新的安全防御策略，以确保网络的安全。基于该策略，每个传感器可以直接从其相邻集合中排除可疑传感器。首先，将可疑传感器的选择问题描述为一个非确定多项式时间难(NP-Hard)的组合优化问题。为了解决这一问题，将原函数转化为子模函数。在此基础上，提出了一种基于序贯子模优化理论的分布式攻击检测调度算法，该算法结合专家问题，更好地利用历史信息指导当前时刻的传感器选择任务。理论结果表明，对于不同的攻击策略，该算法的平均最优率有一个下界，并且对任何子集的误差期望都是有界的。另外，在两种不安全情况下，从估计误差增大的角度来看，该算法可以保证整个网络的安全性。最后，通过数值仿真和实际实验验证了该方法的有效性。



## **22. SPAM: Secure & Private Aircraft Management**

垃圾邮件：安全和私人飞机管理 cs.CR

6 pages

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00245v1) [paper-pdf](http://arxiv.org/pdf/2312.00245v1)

**Authors**: Yaman Jandali, Nojan Sheybani, Farinaz Koushanfar

**Abstract**: With the rising use of aircrafts for operations ranging from disaster-relief to warfare, there is a growing risk of adversarial attacks. Malicious entities often only require the location of the aircraft for these attacks. Current satellite-aircraft communication and tracking protocols put aircrafts at risk if the satellite is compromised, due to computation being done in plaintext. In this work, we present \texttt{SPAM}, a private, secure, and accurate system that allows satellites to efficiently manage and maintain tracking angles for aircraft fleets without learning aircrafts' locations. \texttt{SPAM} is built upon multi-party computation and zero-knowledge proofs to guarantee privacy and high efficiency. While catered towards aircrafts, \texttt{SPAM}'s zero-knowledge fleet management can be easily extended to the IoT, with very little overhead.

摘要: 随着越来越多的飞机被用于从救灾到战争的各种行动，发生对抗性攻击的风险越来越大。恶意实体通常只需要飞机的位置就可以进行这些攻击。目前的卫星-飞机通信和跟踪协议使飞机面临风险，如果卫星受到威胁，因为计算是以明文进行的。在这项工作中，我们介绍了\exttt{Spam}，这是一个私人、安全和准确的系统，允许卫星在不了解飞机位置的情况下高效地管理和维护飞机机队的跟踪角度。\exttt{Spam}基于多方计算和零知识证明，保证隐私和高效。在迎合飞机的同时，S的零知识机队管理可以很容易地扩展到物联网，而开销很小。



## **23. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

忽略这个标题和HackAprompt：通过全球规模的即时黑客竞争暴露LLMs的系统漏洞 cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16119v2) [paper-pdf](http://arxiv.org/pdf/2311.16119v2)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.

摘要: 大型语言模型(LLM)部署在具有直接用户参与的交互上下文中，例如聊天机器人和写作助手。这些部署容易受到即时注入和越狱(统称为即时黑客)的攻击，在这些情况下，模型被操纵以忽略其原始指令并遵循潜在的恶意指令。尽管被广泛认为是一个重大的安全威胁，但缺乏关于即时黑客攻击的大规模资源和量化研究。为了弥补这一漏洞，我们发起了一场全球即时黑客竞赛，允许自由形式的人工输入攻击。我们在三个最先进的LLM上获得了600K+的对抗性提示。我们描述了数据集，这从经验上验证了当前的LLM确实可以通过即时黑客来操纵。我们还提出了对抗性提示类型的全面分类本体。



## **24. Optimal Attack and Defense for Reinforcement Learning**

强化学习的最优攻击与防御 cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00198v1) [paper-pdf](http://arxiv.org/pdf/2312.00198v1)

**Authors**: Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie

**Abstract**: To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.

摘要: 为了确保强化学习(RL)在实际系统中的有效性，确保它们对噪声和对手攻击具有健壮性是至关重要的。在对抗性RL中，外部攻击者有权操纵受害者代理与环境的交互。我们研究了所有类型的在线操纵攻击，包括(I)状态攻击，(Ii)观察攻击(它是感知状态攻击的推广)，(Iii)动作攻击，和(Iv)奖励攻击。我们展示了攻击者设计最大化自身期望回报的隐形攻击的问题，这通常对应于最小化受害者的价值，被马尔可夫决策过程(MDP)捕获，我们称之为元MDP，因为它不是真正的环境，而是由攻击交互引起的更高级别的环境。我们证明了攻击者可以通过在多项式时间内进行规划或使用标准RL技术以多项式样本复杂性学习来获得最优攻击。我们认为，受害者的最优防御策略可以归结为一个随机Stackelberg博弈的解，它可以进一步简化为一个部分可观测的基于回合的随机博弈(POTBSG)。攻击者和受害者都不会从偏离各自的最优策略中受益，因此这样的解决方案是真正可靠的。虽然防御问题是NP难的，但我们证明了在许多情况下，最优马尔可夫防御可以在多项式时间(样本复杂性)内计算(学习)。



## **25. Fool the Hydra: Adversarial Attacks against Multi-view Object Detection Systems**

愚弄九头蛇：针对多视点目标检测系统的对抗性攻击 cs.CV

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00173v1) [paper-pdf](http://arxiv.org/pdf/2312.00173v1)

**Authors**: Bilel Tarchoun, Quazi Mishkatul Alam, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Adversarial patches exemplify the tangible manifestation of the threat posed by adversarial attacks on Machine Learning (ML) models in real-world scenarios. Robustness against these attacks is of the utmost importance when designing computer vision applications, especially for safety-critical domains such as CCTV systems. In most practical situations, monitoring open spaces requires multi-view systems to overcome acquisition challenges such as occlusion handling. Multiview object systems are able to combine data from multiple views, and reach reliable detection results even in difficult environments. Despite its importance in real-world vision applications, the vulnerability of multiview systems to adversarial patches is not sufficiently investigated. In this paper, we raise the following question: Does the increased performance and information sharing across views offer as a by-product robustness to adversarial patches? We first conduct a preliminary analysis showing promising robustness against off-the-shelf adversarial patches, even in an extreme setting where we consider patches applied to all views by all persons in Wildtrack benchmark. However, we challenged this observation by proposing two new attacks: (i) In the first attack, targeting a multiview CNN, we maximize the global loss by proposing gradient projection to the different views and aggregating the obtained local gradients. (ii) In the second attack, we focus on a Transformer-based multiview framework. In addition to the focal loss, we also maximize the transformer-specific loss by dissipating its attention blocks. Our results show a large degradation in the detection performance of victim multiview systems with our first patch attack reaching an attack success rate of 73% , while our second proposed attack reduced the performance of its target detector by 62%

摘要: 对抗性补丁例证了现实世界场景中对抗性攻击对机器学习(ML)模型构成的威胁的有形表现。在设计计算机视觉应用程序时，对这些攻击的健壮性至关重要，特别是对于安全关键领域，如闭路电视系统。在大多数实际情况下，监控开放空间需要多视角系统来克服采集挑战，如遮挡处理。多视点目标系统能够组合来自多个视点的数据，即使在困难的环境中也能达到可靠的检测结果。尽管多视点系统在现实世界的视觉应用中具有重要的意义，但多视点系统对敌意补丁的脆弱性尚未得到充分的研究。在这篇文章中，我们提出了以下问题：增加的性能和跨视图的信息共享是否作为副产品提供了对对手补丁的健壮性？我们首先进行了初步分析，展示了对现成的敌意补丁具有良好的稳健性，即使在我们认为WildTrack基准测试中所有人的所有视图都应用了补丁的极端环境中也是如此。然而，我们通过提出两个新的攻击来挑战这一观察结果：(I)在第一个攻击中，针对多视点CNN，我们通过向不同的视点提出梯度投影并聚合获得的局部梯度来最大化全局损失。(Ii)在第二个攻击中，我们重点介绍了基于Transformer的多视图框架。除了焦点损耗，我们还通过分散其注意力块来最大化特定于变压器的损耗。结果表明，第一次补丁攻击使受害者多视角系统的检测性能大幅下降，攻击成功率达到73%，而第二次补丁攻击使目标检测器的性能下降了62%



## **26. Universal Backdoor Attacks**

通用后门攻击 cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00157v1) [paper-pdf](http://arxiv.org/pdf/2312.00157v1)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset.

摘要: 网络抓取的数据集很容易受到数据中毒的影响，在训练过程中，数据中毒可以用于回溯深度图像分类器。由于在大型数据集上进行训练的成本很高，因此一个模型只需训练一次，就可以多次重复使用。与对抗性示例不同，后门攻击通常针对特定类，而不是模型学习到的任何类。人们可能会认为，通过天真的攻击组合以许多类别为目标会极大地增加毒物样本的数量。我们证明这不一定是真的，而且更有效，普遍存在的数据中毒攻击允许在毒物样本少量增加的情况下控制从任何源类到任何目标类的误分类。我们的想法是生成模型可以学习的具有显著特征的触发器。我们制作的触发器利用了一种我们称为类间毒药可转移性的现象，即从一个类学习触发器使模型更容易学习其他类的触发器。我们通过控制多达6,000个类的模型来展示我们的通用后门攻击的有效性和健壮性，而只毒化了0.15%的训练数据集。



## **27. On the Adversarial Robustness of Graph Contrastive Learning Methods**

图对比学习方法的对抗鲁棒性研究 cs.LG

Accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop  (NeurIPS GLFrontiers 2023)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17853v2) [paper-pdf](http://arxiv.org/pdf/2311.17853v2)

**Authors**: Filippo Guerranti, Zinuo Yi, Anna Starovoit, Rafiq Kamel, Simon Geisler, Stephan Günnemann

**Abstract**: Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.

摘要: 对比学习(CL)已经成为一种强大的框架，用于以自我监督的方式学习图像和文本的表示，同时增强模型对对手攻击的稳健性。最近，研究人员将对比学习的原理扩展到图结构的数据，从而诞生了图对比学习(GCL)领域。然而，GCL方法在对抗稳健性方面是否能提供与其在图像和文本领域的对应方法相同的优势仍然是一个悬而未决的问题。在本文中，我们介绍了一个全面的健壮性评估协议，以评估GCL模型的健壮性。我们使这些模型受到针对图结构的自适应对抗性攻击，特别是在逃避场景中。我们使用不同的真实数据集和攻击策略来评估节点和图分类任务。通过我们的工作，我们旨在为GCL方法的稳健性提供见解，并希望为未来潜在的研究方向开辟道路。



## **28. Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs**

基于CDI感知Gans的无线信号分类器的对抗性攻击与防御 cs.IT

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18820v1) [paper-pdf](http://arxiv.org/pdf/2311.18820v1)

**Authors**: Sujata Sinha, Alkan Soysal

**Abstract**: We introduce a Channel Distribution Information (CDI)-aware Generative Adversarial Network (GAN), designed to address the unique challenges of adversarial attacks in wireless communication systems. The generator in this CDI-aware GAN maps random input noise to the feature space, generating perturbations intended to deceive a target modulation classifier. Its discriminators play a dual role: one enforces that the perturbations follow a Gaussian distribution, making them indistinguishable from Gaussian noise, while the other ensures these perturbations account for realistic channel effects and resemble no-channel perturbations.   Our proposed CDI-aware GAN can be used as an attacker and a defender. In attack scenarios, the CDI-aware GAN demonstrates its prowess by generating robust adversarial perturbations that effectively deceive the target classifier, outperforming known methods. Furthermore, CDI-aware GAN as a defender significantly improves the target classifier's resilience against adversarial attacks.

摘要: 我们介绍了一个信道分布信息（CDI）感知的生成对抗网络（GAN），旨在解决无线通信系统中对抗攻击的独特挑战。该CDI感知GAN中的生成器将随机输入噪声映射到特征空间，生成旨在欺骗目标调制分类器的扰动。它的鉴别器扮演着双重角色：一个强制扰动遵循高斯分布，使它们无法与高斯噪声区分开来，而另一个则确保这些扰动考虑到现实的信道效应，并类似于非信道扰动。   我们提出的CDI感知GAN可以用作攻击者和防御者。在攻击场景中，CDI-aware GAN通过生成强大的对抗性扰动来证明其实力，这些扰动有效地欺骗了目标分类器，优于已知方法。此外，CDI-aware GAN作为防御者显着提高了目标分类器对对抗性攻击的弹性。



## **29. Improving the Robustness of Quantized Deep Neural Networks to White-Box Attacks using Stochastic Quantization and Information-Theoretic Ensemble Training**

利用随机量化和信息论集成训练提高量化深度神经网络对白盒攻击的稳健性 cs.CV

9 pages, 9 figures, 4 tables

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00105v1) [paper-pdf](http://arxiv.org/pdf/2312.00105v1)

**Authors**: Saurabh Farkya, Aswin Raghavan, Avi Ziskind

**Abstract**: Most real-world applications that employ deep neural networks (DNNs) quantize them to low precision to reduce the compute needs. We present a method to improve the robustness of quantized DNNs to white-box adversarial attacks. We first tackle the limitation of deterministic quantization to fixed ``bins'' by introducing a differentiable Stochastic Quantizer (SQ). We explore the hypothesis that different quantizations may collectively be more robust than each quantized DNN. We formulate a training objective to encourage different quantized DNNs to learn different representations of the input image. The training objective captures diversity and accuracy via mutual information between ensemble members. Through experimentation, we demonstrate substantial improvement in robustness against $L_\infty$ attacks even if the attacker is allowed to backpropagate through SQ (e.g., > 50\% accuracy to PGD(5/255) on CIFAR10 without adversarial training), compared to vanilla DNNs as well as existing ensembles of quantized DNNs. We extend the method to detect attacks and generate robustness profiles in the adversarial information plane (AIP), towards a unified analysis of different threat models by correlating the MI and accuracy.

摘要: 大多数使用深度神经网络(DNN)的现实世界应用程序将它们量化到低精度，以减少计算需求。提出了一种提高量化DNN对白盒攻击的鲁棒性的方法。我们首先通过引入一种可微随机量化器(SQ)来解决确定性量化对固定‘箱’的限制。我们探索了这样的假设，即不同的量化可能共同比每个量化的DNN更稳健。我们制定了一个训练目标，以鼓励不同的量化DNN学习输入图像的不同表示。训练目标通过集合成员之间的互信息来捕捉多样性和准确性。通过实验表明，与普通的DNN和现有的量化DNN集成相比，即使允许攻击者通过SQ反向传播(例如，在CIFAR10上对PGD(5/255)的准确率>50\%)，我们也表现出对$L_INFTY$攻击的鲁棒性显著提高。我们将该方法扩展到检测攻击并在对抗信息平面(AIP)中生成健壮性配置文件，通过关联MI和准确性来统一分析不同的威胁模型。



## **30. Differentiable JPEG: The Devil is in the Details**

与众不同的JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2309.06978v3) [paper-pdf](http://arxiv.org/pdf/2309.06978v3)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **31. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

不可察觉和可转移对抗性攻击的扩散模型 cs.CV

Code Page: https://github.com/WindVChen/DiffAttack. In Paper Version  v2, we incorporate more discussions and experiments

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2305.08192v2) [paper-pdf](http://arxiv.org/pdf/2305.08192v2)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in the latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an implicit recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into the adversarial attack field. Extensive experiments on various model structures, datasets, and defense methods have demonstrated the superiority of our attack over the existing attack methods.

摘要: 许多现有的对抗性攻击在图像RGB空间上产生$L_p$-范数扰动。尽管在可转移性和攻击成功率方面取得了一些成就，但制作的对抗性例子很容易被人眼察觉。对于视觉不可感知性，最近的一些工作探索了没有$L_p$-范数约束的无限攻击，但缺乏攻击黑盒模型的可转移性。在这项工作中，我们提出了一种新的不可察觉和可转移的攻击，利用扩散模型的生成性和区分性。具体地说，我们不是在像素空间中直接操作，而是在扩散模型的潜在空间中制造扰动。与设计良好的内容保持结构相结合，我们可以生成嵌入语义线索的人类不敏感的扰动。为了获得更好的可转移性，我们通过将扩散模型的注意力从目标区域转移开，进一步欺骗了可以被视为隐式识别代理的扩散模型。据我们所知，我们提出的DiffAttack方法首次将扩散模型引入到对抗性攻击领域。在各种模型结构、数据集和防御方法上的广泛实验证明了该攻击相对于现有攻击方法的优越性。



## **32. Data-Agnostic Model Poisoning against Federated Learning: A Graph Autoencoder Approach**

数据不可知模型毒化联合学习：一种图自动编码器方法 cs.LG

15 pages, 10 figures, submitted to IEEE Transactions on Information  Forensics and Security (TIFS)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18498v1) [paper-pdf](http://arxiv.org/pdf/2311.18498v1)

**Authors**: Kai Li, Jingjing Zheng, Xin Yuan, Wei Ni, Ozgur B. Akan, H. Vincent Poor

**Abstract**: This paper proposes a novel, data-agnostic, model poisoning attack on Federated Learning (FL), by designing a new adversarial graph autoencoder (GAE)-based framework. The attack requires no knowledge of FL training data and achieves both effectiveness and undetectability. By listening to the benign local models and the global model, the attacker extracts the graph structural correlations among the benign local models and the training data features substantiating the models. The attacker then adversarially regenerates the graph structural correlations while maximizing the FL training loss, and subsequently generates malicious local models using the adversarial graph structure and the training data features of the benign ones. A new algorithm is designed to iteratively train the malicious local models using GAE and sub-gradient descent. The convergence of FL under attack is rigorously proved, with a considerably large optimality gap. Experiments show that the FL accuracy drops gradually under the proposed attack and existing defense mechanisms fail to detect it. The attack can give rise to an infection across all benign devices, making it a serious threat to FL.

摘要: 通过设计一种新的基于对抗性图自动编码器(GAE)的框架，提出了一种针对联邦学习(FL)的数据不可知的模型中毒攻击。该攻击不需要了解FL训练数据，并且实现了有效性和不可检测性。通过监听良性局部模型和全局模型，攻击者提取良性局部模型和证实模型的训练数据特征之间的图结构相关性。然后攻击者在最大化FL训练损失的同时对抗性地重新生成图结构相关性，并随后利用对抗性图结构和良性图结构的训练数据特征来生成恶意局部模型。设计了一种利用GAE和次梯度下降迭代训练恶意局部模型的新算法。严格地证明了FL在攻击下的收敛，但存在相当大的最优性差距。实验表明，在所提出的攻击下，FL的准确率逐渐下降，现有的防御机制无法检测到它。这种攻击可以引起对所有良性设备的感染，使其成为对FL的严重威胁。



## **33. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

走向更安全的生成性语言模型：安全风险、评估和改进的综述 cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.

摘要: 随着产生式大型模型能力的进步，安全问题在其输出中变得更加明显。为了确保人工智能生态系统的可持续增长，必须对相关安全风险进行全面评估和细化。本调查提出了与大型模型相关的安全研究框架，描绘了安全风险的图景以及安全评估和改进方法。我们首先介绍广泛关注的安全问题，然后深入研究大型模型的安全评估方法，包括基于偏好的测试、对抗性攻击方法、问题检测和其他高级评估方法。此外，我们还探讨了从培训到部署增强大型模型安全性的策略，重点介绍了构建大型模型的每个阶段的前沿安全方法。最后，我们讨论了向更负责任的人工智能发展的核心挑战，包括安全机制的可解释性、持续的安全问题和针对恶意攻击的健壮性。通过这次调查，我们旨在为安全研究人员提供明确的技术指导，并鼓励进一步研究大型模型的安全性。



## **34. On the Robustness of Decision-Focused Learning**

决策聚焦学习的稳健性研究 cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16487v2) [paper-pdf](http://arxiv.org/pdf/2311.16487v2)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.

摘要: 聚焦决策学习(DFL)是一种新兴的学习范式，它解决了训练机器学习(ML)模型来预测不完全优化问题的缺失参数的任务，其中缺失的参数被预测。DFL通过集成预测和优化任务，在端到端系统中训练ML模型，提供更好的训练和测试目标的一致性。DFL已经显示出了很大的潜力，并拥有在许多现实世界应用程序中彻底改变决策的能力。然而，人们对这些模型在对抗性攻击下的性能知之甚少。我们采用了十种独特的DFL方法，并对它们在两种针对预测-然后优化问题设置的明显集中的攻击下的性能进行了基准测试。我们的研究提出了这样的假设，即模型的稳健性与其在不偏离地面事实标签的情况下找到导致最优决策的预测的能力高度相关。此外，我们还提供了对如何针对违反这一条件的模型的洞察，并展示了这些模型如何根据在其训练周期结束时实现的最优化而做出不同的反应。



## **35. Improving the Robustness of Transformer-based Large Language Models with Dynamic Attention**

利用动态注意提高基于Transformer的大语言模型的健壮性 cs.CL

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17400v2) [paper-pdf](http://arxiv.org/pdf/2311.17400v2)

**Authors**: Lujia Shen, Yuwen Pu, Shouling Ji, Changjiang Li, Xuhong Zhang, Chunpeng Ge, Ting Wang

**Abstract**: Transformer-based models, such as BERT and GPT, have been widely adopted in natural language processing (NLP) due to their exceptional performance. However, recent studies show their vulnerability to textual adversarial attacks where the model's output can be misled by intentionally manipulating the text inputs. Despite various methods that have been proposed to enhance the model's robustness and mitigate this vulnerability, many require heavy consumption resources (e.g., adversarial training) or only provide limited protection (e.g., defensive dropout). In this paper, we propose a novel method called dynamic attention, tailored for the transformer architecture, to enhance the inherent robustness of the model itself against various adversarial attacks. Our method requires no downstream task knowledge and does not incur additional costs. The proposed dynamic attention consists of two modules: (I) attention rectification, which masks or weakens the attention value of the chosen tokens, and (ii) dynamic modeling, which dynamically builds the set of candidate tokens. Extensive experiments demonstrate that dynamic attention significantly mitigates the impact of adversarial attacks, improving up to 33\% better performance than previous methods against widely-used adversarial attacks. The model-level design of dynamic attention enables it to be easily combined with other defense methods (e.g., adversarial training) to further enhance the model's robustness. Furthermore, we demonstrate that dynamic attention preserves the state-of-the-art robustness space of the original model compared to other dynamic modeling methods.

摘要: 基于转换器的模型，如BERT和GPT，由于其卓越的性能，已被广泛采用在自然语言处理（NLP）中。然而，最近的研究表明，它们容易受到文本对抗攻击，其中模型的输出可能会被故意操纵文本输入所误导。尽管已经提出了各种方法来增强模型的鲁棒性并减轻这种脆弱性，但许多方法需要大量消耗资源（例如，对抗训练）或仅提供有限的保护（例如，防御性辍学）。在本文中，我们提出了一种称为动态注意力的新方法，为Transformer架构量身定制，以增强模型本身对各种对抗性攻击的固有鲁棒性。我们的方法不需要下游任务知识，也不会产生额外的成本。所提出的动态注意力包括两个模块：（I）注意力矫正，它掩盖或削弱所选标记的注意力值，以及（ii）动态建模，它动态地构建候选标记集。大量的实验表明，动态注意力显着减轻了对抗性攻击的影响，提高了高达33%的性能比以前的方法对广泛使用的对抗性攻击。动态注意力的模型级设计使其能够轻松地与其他防御方法（例如，对抗训练），以进一步增强模型的鲁棒性。此外，我们证明了动态注意力保持了最先进的鲁棒性空间的原始模型相比，其他动态建模方法。



## **36. Effective Backdoor Mitigation Depends on the Pre-training Objective**

有效的后门缓解取决于培训前的目标 cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.14948v2) [paper-pdf](http://arxiv.org/pdf/2311.14948v2)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.

摘要: 尽管当代机器学习(ML)模型具有先进的能力，但它们仍然容易受到对手和后门攻击。此漏洞在实际部署中尤其令人担忧，在实际部署中，受危害的模型可能会在关键情况下表现出不可预测的行为。为训练前的多模式模型收集来自互联网的海量数据集的普遍做法加剧了这种风险，因为这些数据集可能有后门。已经提出了各种技术来减轻这些模型中回溯的影响，例如CleanCLIP，这是当前最先进的方法。在这项工作中，我们证明了CleanCLIP在缓解后门方面的有效性高度依赖于在模型预培训期间使用的特定目标。我们观察到，较强的培训前目标与较难消除后门行为相关。我们通过在两个由300万(CC3M)和600万(CC6M)数据点组成的大型数据集上训练多模模型，在不同的预训练目标下，然后使用CleanCLIP去除毒物来证明这一点。我们发现，当使用更强的预培训目标时，即使进行了广泛的超参数调整，CleanCLIP也是无效的。我们的发现强调了ML从业者的关键考虑，他们使用大规模的网络管理数据对模型进行预培训，并担心潜在的后门威胁。值得注意的是，我们的结果表明，简单的预培训目标更容易有效地移除后门。对于寻求在使用更强的预培训目标和针对后门攻击的安全性之间进行权衡的从业者来说，这一见解至关重要。



## **37. AnonPSI: An Anonymity Assessment Framework for PSI**

AnonPSI：一种面向PSI的匿名评估框架 cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18118v1) [paper-pdf](http://arxiv.org/pdf/2311.18118v1)

**Authors**: Bo Jiang, Jian Du, Qiang Yan

**Abstract**: Private Set Intersection (PSI) is a widely used protocol that enables two parties to securely compute a function over the intersected part of their shared datasets and has been a significant research focus over the years. However, recent studies have highlighted its vulnerability to Set Membership Inference Attacks (SMIA), where an adversary might deduce an individual's membership by invoking multiple PSI protocols. This presents a considerable risk, even in the most stringent versions of PSI, which only return the cardinality of the intersection. This paper explores the evaluation of anonymity within the PSI context. Initially, we highlight the reasons why existing works fall short in measuring privacy leakage, and subsequently propose two attack strategies that address these deficiencies. Furthermore, we provide theoretical guarantees on the performance of our proposed methods. In addition to these, we illustrate how the integration of auxiliary information, such as the sum of payloads associated with members of the intersection (PSI-SUM), can enhance attack efficiency. We conducted a comprehensive performance evaluation of various attack strategies proposed utilizing two real datasets. Our findings indicate that the methods we propose markedly enhance attack efficiency when contrasted with previous research endeavors. {The effective attacking implies that depending solely on existing PSI protocols may not provide an adequate level of privacy assurance. It is recommended to combine privacy-enhancing technologies synergistically to enhance privacy protection even further.

摘要: 私有集合交集(PSI)是一种广泛使用的协议，它使双方能够安全地在其共享数据集的相交部分上计算函数，多年来一直是一个重要的研究热点。然而，最近的研究强调了它对集合成员推理攻击(SMIA)的脆弱性，在这种攻击中，攻击者可以通过调用多个PSI协议来推断个人的成员资格。这带来了相当大的风险，即使在最严格的PSI版本中也是如此，它只返回交集的基数。本文探讨了PSI环境下的匿名性评估。首先，我们强调了现有作品在测量隐私泄露方面不足的原因，并随后提出了两种攻击策略来解决这些不足。此外，我们还为我们所提出的方法的性能提供了理论保证。除此之外，我们还说明了辅助信息的集成，例如与交集成员相关的有效负载之和(PSI-SUM)如何提高攻击效率。我们利用两个真实数据集对提出的各种攻击策略进行了全面的性能评估。我们的研究结果表明，与以前的研究相比，我们提出的方法显着提高了攻击效率。{有效的攻击意味着，仅依赖现有的PSI协议可能无法提供足够级别的隐私保障。建议将增强隐私的技术协同结合，以进一步加强隐私保护。



## **38. Improving Faithfulness for Vision Transformers**

提高视觉变形金刚的忠诚度 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17983v1) [paper-pdf](http://arxiv.org/pdf/2311.17983v1)

**Authors**: Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance for various vision tasks. One reason behind the success lies in their ability to provide plausible innate explanations for the behavior of neural architectures. However, ViTs suffer from issues with explanation faithfulness, as their focal points are fragile to adversarial attacks and can be easily changed with even slight perturbations on the input image. In this paper, we propose a rigorous approach to mitigate these issues by introducing Faithful ViTs (FViTs). Briefly speaking, an FViT should have the following two properties: (1) The top-$k$ indices of its self-attention vector should remain mostly unchanged under input perturbation, indicating stable explanations; (2) The prediction distribution should be robust to perturbations. To achieve this, we propose a new method called Denoised Diffusion Smoothing (DDS), which adopts randomized smoothing and diffusion-based denoising. We theoretically prove that processing ViTs directly with DDS can turn them into FViTs. We also show that Gaussian noise is nearly optimal for both $\ell_2$ and $\ell_\infty$-norm cases. Finally, we demonstrate the effectiveness of our approach through comprehensive experiments and evaluations. Specifically, we compare our FViTs with other baselines through visual interpretation and robustness accuracy under adversarial attacks. Results show that FViTs are more robust against adversarial attacks while maintaining the explainability of attention, indicating higher faithfulness.

摘要: 视觉转换器（ViTs）在各种视觉任务中实现了最先进的性能。成功背后的一个原因在于他们能够为神经结构的行为提供合理的先天解释。然而，ViTs存在解释忠实性的问题，因为它们的焦点对于对抗性攻击是脆弱的，并且可以很容易地通过输入图像上的轻微扰动而改变。在本文中，我们提出了一个严格的方法来减轻这些问题，通过引入忠实的ViTs（FViTs）。简而言之，FViT应该具有以下两个性质：（1）其自注意向量的前k$索引在输入扰动下应该保持基本不变，表明稳定的解释;（2）预测分布应该对扰动具有鲁棒性。为了实现这一点，我们提出了一种新的方法称为去噪扩散平滑（DDS），它采用随机平滑和基于扩散的去噪。我们从理论上证明了直接用DDS处理ViT可以将它们转化为FViT。我们还表明，高斯噪声是近最佳的$\ell_2$和$\ell_\infty$-范数的情况下。最后，我们证明了我们的方法的有效性，通过全面的实验和评估。具体来说，我们通过视觉解释和对抗性攻击下的鲁棒性准确性将我们的FViT与其他基线进行比较。结果表明，FViTs对对抗性攻击更鲁棒，同时保持注意力的可解释性，表明更高的忠诚度。



## **39. SenTest: Evaluating Robustness of Sentence Encoders**

SenTest：评估句子编码器的健壮性 cs.CL

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17722v1) [paper-pdf](http://arxiv.org/pdf/2311.17722v1)

**Authors**: Tanmay Chavan, Shantanu Patankar, Aditya Kane, Omkar Gokhale, Geetanjali Kale, Raviraj Joshi

**Abstract**: Contrastive learning has proven to be an effective method for pre-training models using weakly labeled data in the vision domain. Sentence transformers are the NLP counterparts to this architecture, and have been growing in popularity due to their rich and effective sentence representations. Having effective sentence representations is paramount in multiple tasks, such as information retrieval, retrieval augmented generation (RAG), and sentence comparison. Keeping in mind the deployability factor of transformers, evaluating the robustness of sentence transformers is of utmost importance. This work focuses on evaluating the robustness of the sentence encoders. We employ several adversarial attacks to evaluate its robustness. This system uses character-level attacks in the form of random character substitution, word-level attacks in the form of synonym replacement, and sentence-level attacks in the form of intra-sentence word order shuffling. The results of the experiments strongly undermine the robustness of sentence encoders. The models produce significantly different predictions as well as embeddings on perturbed datasets. The accuracy of the models can fall up to 15 percent on perturbed datasets as compared to unperturbed datasets. Furthermore, the experiments demonstrate that these embeddings does capture the semantic and syntactic structure (sentence order) of sentences. However, existing supervised classification strategies fail to leverage this information, and merely function as n-gram detectors.

摘要: 对比学习已被证明是在视觉领域使用弱标记数据进行预训练模型的一种有效方法。句子转换器是这种体系结构的NLP对应物，由于其丰富而有效的句子表示形式而越来越受欢迎。在信息检索、检索增强生成(RAG)和句子比较等多项任务中，拥有有效的句子表征是至关重要的。考虑到转换器的可部署性因素，评估语句转换器的健壮性至关重要。这项工作的重点是评估句子编码器的健壮性。我们使用了几种对抗性攻击来评估它的健壮性。该系统使用了以随机字符替换形式的字符级攻击、以同义词替换形式的词级攻击和以句内语序洗牌形式的句子级攻击。实验结果严重削弱了句子编码器的健壮性。这些模型产生了显著不同的预测以及对扰动数据集的嵌入。与未受干扰的数据集相比，该模型在扰动数据集上的准确率最高可下降15%。此外，实验表明，这些嵌入确实捕捉到了句子的语义和句法结构(句序)。然而，现有的监督分类策略不能利用这些信息，而仅仅起到n元语法检测器的作用。



## **40. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。我们的代码可通过以下链接公开获得：https://github.com/arobey1/smooth-llm.



## **41. Natural & Adversarial Bokeh Rendering via Circle-of-Confusion Predictive Network**

基于混淆环预测网络的自然与对抗性Bokeh绘制 cs.CV

11 pages, accepted by TMM

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2111.12971v3) [paper-pdf](http://arxiv.org/pdf/2111.12971v3)

**Authors**: Yihao Huang, Felix Juefei-Xu, Qing Guo, Geguang Pu, Yang Liu

**Abstract**: Bokeh effect is a natural shallow depth-of-field phenomenon that blurs the out-of-focus part in photography. In recent years, a series of works have proposed automatic and realistic bokeh rendering methods for artistic and aesthetic purposes. They usually employ cutting-edge data-driven deep generative networks with complex training strategies and network architectures. However, these works neglect that the bokeh effect, as a real phenomenon, can inevitably affect the subsequent visual intelligent tasks like recognition, and their data-driven nature prevents them from studying the influence of bokeh-related physical parameters (i.e., depth-of-the-field) on the intelligent tasks. To fill this gap, we study a totally new problem, i.e., natural & adversarial bokeh rendering, which consists of two objectives: rendering realistic and natural bokeh and fooling the visual perception models (i.e., bokeh-based adversarial attack). To this end, beyond the pure data-driven solution, we propose a hybrid alternative by taking the respective advantages of data-driven and physical-aware methods. Specifically, we propose the circle-of-confusion predictive network (CoCNet) by taking the all-in-focus image and depth image as inputs to estimate circle-of-confusion parameters for each pixel, which are employed to render the final image through a well-known physical model of bokeh. With the hybrid solution, our method could achieve more realistic rendering results with the naive training strategy and a much lighter network.

摘要: 波克效应是一种自然的浅景深现象，它会模糊摄影中的失焦部分。近年来，出于艺术和审美的目的，一系列作品提出了自动和逼真的bokeh绘制方法。他们通常使用尖端的数据驱动的深度生成网络，具有复杂的训练策略和网络架构。然而，这些工作忽略了波克效应作为一种真实的现象，不可避免地会影响后续的视觉智能任务，如识别，其数据驱动的性质阻碍了他们研究波克相关的物理参数(即景深)对智能任务的影响。为了填补这一空白，我们研究了一个全新的问题，即自然和对抗性的bokeh绘制，它包括两个目标：渲染逼真的自然bokeh和愚弄视觉感知模型(即基于bokeh的对抗性攻击)。为此，除了纯数据驱动的解决方案之外，我们还提出了一种结合数据驱动和物理感知方法各自优势的混合替代方案。具体地说，我们提出了混淆圈预测网络(CoCNet)，它以全焦图像和深度图像作为输入来估计每个像素的混淆圈参数，并利用这些参数通过一个著名的Bokeh物理模型来呈现最终的图像。使用混合方法，我们的方法可以在简单的训练策略和更轻的网络环境下获得更逼真的渲染结果。



## **42. Query-Relevant Images Jailbreak Large Multi-Modal Models**

与查询相关的图像越狱大型多模式模型 cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

摘要: 警告：本文包含有害语言和图片的例子，建议读者自行决定。围绕大型语言模型(LLM)的安全问题已经得到了广泛的研究，但大型多模式模型(LMM)的安全性仍未得到充分研究。在我们的研究中，我们提出了一种新的视觉提示攻击，利用与查询相关的图像来越狱开源的LMM。我们的方法从一个由扩散模型生成的图像和另一个基于从恶意查询中提取的关键字将文本显示为排版的图像创建合成图像。我们表明，即使所使用的大型语言模型安全地对齐，LLM也可以很容易地被我们的方法攻击。为了评估这一漏洞在开源LMM中的程度，我们使用我们提出的攻击技术编制了一个包含13个场景的大量数据集，总共有5,040个文本-图像对。我们使用这个数据集对12个尖端的LMM进行了评估，表明了现有的多模式模型在对抗攻击时的脆弱性。这一发现强调了需要共同努力，加强和改进开放源码LMM的安全措施，以防范潜在的恶意利用。该资源位于\href{此HTTPS URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **43. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

去极化噪声下的量子神经网络：白盒攻击与防御探索 quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17458v1) [paper-pdf](http://arxiv.org/pdf/2311.17458v1)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.

摘要: 利用量子力学的独特性质，量子机器学习(QML)有望在传统系统达到其边界的地方实现计算突破和丰富视角。然而，与经典机器学习类似，QML也不能幸免于对手攻击。量子对抗性机器学习已成为突出QML模型在面对对抗性特制特征向量时的弱点的工具。深入到这个领域，我们的探索揭示了去极化噪声和对手稳健性之间的相互作用。虽然之前的结果通过去极化噪声增强了对抗威胁的稳健性，但我们的发现描绘了一幅不同的图景。有趣的是，添加去极化噪声会中断为多类分类场景提供进一步稳健性的效果。综合我们的发现，我们用一个多类分类器进行了实验，该分类器在基于门的量子模拟器上进行了相反的训练，进一步阐明了这种意想不到的行为。



## **44. Group-wise Sparse and Explainable Adversarial Attacks**

群组稀疏和可解释的对抗性攻击 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17434v1) [paper-pdf](http://arxiv.org/pdf/2311.17434v1)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, typically regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs than previously anticipated. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. In this paper, we tackle this challenge by presenting an algorithm that simultaneously generates group-wise sparse attacks within semantically meaningful areas of an image. In each iteration, the core operation of our algorithm involves the optimization of a quasinorm adversarial loss. This optimization is achieved by employing the $1/2$-quasinorm proximal operator for some iterations, a method tailored for nonconvex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2$-norm regularization applied to perturbation magnitudes. We rigorously evaluate the efficacy of our novel attack in both targeted and non-targeted attack scenarios, on CIFAR-10 and ImageNet datasets. When compared to state-of-the-art methods, our attack consistently results in a remarkable increase in group-wise sparsity, e.g., an increase of $48.12\%$ on CIFAR-10 and $40.78\%$ on ImageNet (average case, targeted attack), all while maintaining lower perturbation magnitudes. Notably, this performance is complemented by a significantly faster computation time and a $100\%$ attack success rate.

摘要: 稀疏敌意攻击通过最小的像素扰动欺骗深度神经网络(DNN)，通常由$\ell_0$范数正则化。最近的努力已经用结构稀疏性正则化规则取代了这一规范，例如核集团规范，以制定群组稀疏对抗性攻击。因此，由此产生的扰动是可以解释的，并具有重要的实际意义，揭示了DNN比之前预期的更大的脆弱性。然而，精心设计这样的攻击构成了一个优化挑战，因为它涉及到计算非凸目标内的像素组的规范。在本文中，我们通过提出一种算法来解决这一挑战，该算法可以在图像的语义有意义的区域内同时生成分组稀疏攻击。在每一次迭代中，我们算法的核心操作都涉及到对一个拟正态对抗性损失的优化。这种优化是通过使用$1/2$-拟正态逼近算子进行一些迭代实现的，这是一种为非凸规划量身定做的方法。随后，算法过渡到投影的内斯特罗夫加速梯度下降，并对摄动幅度应用$2范数正则化。我们在CIFAR-10和ImageNet数据集上严格评估了我们的新型攻击在目标攻击和非目标攻击场景中的有效性。与最先进的攻击方法相比，我们的攻击始终导致组稀疏性的显著增加，例如，在CIFAR-10上增加了48.12美元，在ImageNet(平均情况下，有针对性的攻击)上增加了40.78美元，所有这些都保持了较低的扰动幅度。值得注意的是，这一性能得到了显著更快的计算时间和100美元的攻击成功率的补充。



## **45. Enhancing Adversarial Attacks: The Similar Target Method**

加强对抗性攻击：相似靶法 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2308.10743v3) [paper-pdf](http://arxiv.org/pdf/2308.10743v3)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.

摘要: 深度神经网络很容易受到敌意例子的攻击，这对模型的应用构成了威胁，并引发了安全担忧。对抗性例子的一个耐人寻味的特点是它们具有很强的可转移性。已经提出了几种提高可转移性的方法，包括已经证明其有效性的集合攻击。然而，以前的方法只是对模型集成的对数、概率或损失进行平均，缺乏对模型集成如何以及为什么显著提高可转移性的全面分析。本文提出了一种类似的目标攻击方法--相似目标~(ST)。通过提高每个模型梯度之间的余弦相似度，我们的方法将优化方向正则化以同时攻击所有代理模型。实践证明，该策略提高了泛化能力。在ImageNet上的实验结果验证了该方法在提高对手可转移性方面的有效性。我们的方法在18个区分分类器和对抗性训练的模型上优于最先进的攻击者。



## **46. RADAP: A Robust and Adaptive Defense Against Diverse Adversarial Patches on Face Recognition**

RADAP：一种对人脸识别中不同敌意补丁的稳健自适应防御 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17339v1) [paper-pdf](http://arxiv.org/pdf/2311.17339v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) systems powered by deep learning have become widely used in various applications. However, they are vulnerable to adversarial attacks, especially those based on local adversarial patches that can be physically applied to real-world objects. In this paper, we propose RADAP, a robust and adaptive defense mechanism against diverse adversarial patches in both closed-set and open-set FR systems. RADAP employs innovative techniques, such as FCutout and F-patch, which use Fourier space sampling masks to improve the occlusion robustness of the FR model and the performance of the patch segmenter. Moreover, we introduce an edge-aware binary cross-entropy (EBCE) loss function to enhance the accuracy of patch detection. We also present the split and fill (SAF) strategy, which is designed to counter the vulnerability of the patch segmenter to complete white-box adaptive attacks. We conduct comprehensive experiments to validate the effectiveness of RADAP, which shows significant improvements in defense performance against various adversarial patches, while maintaining clean accuracy higher than that of the undefended Vanilla model.

摘要: 由深度学习驱动的人脸识别（FR）系统已广泛用于各种应用中。然而，它们容易受到对抗性攻击，特别是那些基于本地对抗补丁的攻击，这些补丁可以物理地应用于现实世界的对象。在本文中，我们提出了RADAP，一个强大的和自适应的防御机制，对不同的敌对补丁在闭集和开集FR系统。RADAP采用创新的技术，如FCutout和F-patch，使用傅立叶空间采样掩码来提高FR模型的遮挡鲁棒性和补丁分割器的性能。此外，我们引入了一个边缘感知的二进制交叉熵（EBCE）损失函数，以提高补丁检测的准确性。我们还提出了分裂和填充（SAF）的策略，这是为了对付补丁分割器的脆弱性，以完成白盒自适应攻击。我们进行了全面的实验来验证RADAP的有效性，该实验显示了对各种对抗性补丁的防御性能的显着改善，同时保持了比不设防的Vanilla模型更高的准确性。



## **47. NeRFTAP: Enhancing Transferability of Adversarial Patches on Face Recognition using Neural Radiance Fields**

NeRFTAP：利用神经辐射场增强人脸识别中敌方补丁的可转移性 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17332v1) [paper-pdf](http://arxiv.org/pdf/2311.17332v1)

**Authors**: Xiaoliang Liu, Furao Shen, Feng Han, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) technology plays a crucial role in various applications, but its vulnerability to adversarial attacks poses significant security concerns. Existing research primarily focuses on transferability to different FR models, overlooking the direct transferability to victim's face images, which is a practical threat in real-world scenarios. In this study, we propose a novel adversarial attack method that considers both the transferability to the FR model and the victim's face image, called NeRFTAP. Leveraging NeRF-based 3D-GAN, we generate new view face images for the source and target subjects to enhance transferability of adversarial patches. We introduce a style consistency loss to ensure the visual similarity between the adversarial UV map and the target UV map under a 0-1 mask, enhancing the effectiveness and naturalness of the generated adversarial face images. Extensive experiments and evaluations on various FR models demonstrate the superiority of our approach over existing attack techniques. Our work provides valuable insights for enhancing the robustness of FR systems in practical adversarial settings.

摘要: 人脸识别(FR)技术在各种应用中扮演着至关重要的角色，但其对对手攻击的脆弱性引发了重大的安全问题。现有的研究主要集中在对不同FR模型的可转移性，而忽略了对受害者面部图像的直接可转移性，这在现实世界场景中是一种实际威胁。在这项研究中，我们提出了一种新的对抗性攻击方法，它同时考虑了对FR模型的可转换性和受害者的面部图像，称为NeRFTAP。利用基于神经网络的3D-GAN算法，为源对象和目标对象生成新的视角人脸图像，以增强对抗性补丁的可转移性。通过引入风格一致性损失，在0-1掩码下保证了敌方UV图与目标UV图的视觉相似性，增强了生成的敌方人脸图像的有效性和自然性。在各种FR模型上的广泛实验和评估证明了该方法相对于现有攻击技术的优越性。我们的工作为增强FR系统在实际对抗环境中的稳健性提供了有价值的见解。



## **48. Content-based Unrestricted Adversarial Attack**

基于内容的无限制对抗性攻击 cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2305.10665v2) [paper-pdf](http://arxiv.org/pdf/2305.10665v2)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.

摘要: 不受限制的对抗性攻击通常会操纵图像的语义内容(例如，颜色或纹理)，以创建既有效又逼真的对抗性示例，展示它们以隐蔽和成功的方式欺骗人类感知和深层神经网络的能力。然而，目前的作品往往牺牲不受限制的程度，主观地选择一些图像内容来保证不受限制的对抗性例子的照片真实感，这限制了其攻击性能。为了保证对抗性实例的真实感，提高攻击性能，我们提出了一种新的无限制攻击框架，称为基于内容的无限对抗性攻击。通过利用表示自然图像的低维流形，我们将图像映射到流形上，并沿着其相反的方向进行优化。因此，在该框架下，我们实现了基于稳定扩散的对抗性内容攻击，并且可以生成具有多种对抗性内容的高可转移性的无限制对抗性实例。广泛的实验和可视化证明了蚁群算法的有效性，特别是在正常训练的模型和防御方法上，平均分别超过最先进的攻击13.3%-50.4%和16.8%-48.0%。



## **49. Advancing Attack-Resilient Scheduling of Integrated Energy Systems with Demand Response via Deep Reinforcement Learning**

基于深度强化学习的需求响应集成能源系统攻击弹性调度 eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17941v1) [paper-pdf](http://arxiv.org/pdf/2311.17941v1)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, a state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10\% improvement in economic performance under cyber-attack scenarios.

摘要: 多能流优化调度是利用可再生能源、提高综合能源系统稳定性和经济性的有效方法。然而，工业企业稳定的需求供应面临着挑战，这些挑战来自资源和负载带来的不确定性，以及采用先进信息和通信技术的网络攻击的影响越来越大。针对这些挑战，提出了一种基于状态对抗性深度强化学习(DRL)的集成需求响应(IDR)支持的IES的无模型弹性调度方法。该方法设计了一个IDR程序来研究电-气-热柔性负荷的相互作用能力。此外，状态对抗马尔可夫决策过程(SA-MDP)模型刻画了网络攻击下IES的能量调度问题。为了缓解网络攻击对调度策略的影响，提出了状态对抗性软行动者-批评者(SA-SAC)算法。仿真结果表明，该方法能够很好地处理资源和负荷带来的不确定性，减轻网络攻击对调度策略的影响，保证各种能源的稳定需求。此外，该方法还表现出了对网络攻击的恢复能力。与原有的软演员-批评者(SAC)算法相比，该算法在网络攻击场景下的经济性能提高了10%。



## **50. Scalable Extraction of Training Data from (Production) Language Models**

从(产生式)语言模型中可伸缩地提取训练数据 cs.LG

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17035v1) [paper-pdf](http://arxiv.org/pdf/2311.17035v1)

**Authors**: Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee

**Abstract**: This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. Existing techniques from the literature suffice to attack unaligned models; in order to attack the aligned ChatGPT, we develop a new divergence attack that causes the model to diverge from its chatbot-style generations and emit training data at a rate 150x higher than when behaving properly. Our methods show practical attacks can recover far more data than previously thought, and reveal that current alignment techniques do not eliminate memorization.

摘要: 本文研究了可提取记忆：对手可以通过查询机器学习模型有效提取的训练数据，而无需事先了解训练数据集。我们发现，攻击者可以从Pythia或GPT-Neo等开源语言模型、LLaMA或Falcon等半开放模型以及ChatGPT等封闭模型中提取千兆字节的训练数据。文献中的现有技术足以攻击未对齐的模型;为了攻击对齐的ChatGPT，我们开发了一种新的发散攻击，导致模型从其聊天机器人风格的生成中发散，并以比正常行为高150倍的速度发出训练数据。我们的方法表明，实际的攻击可以恢复比以前认为的更多的数据，并揭示了目前的对齐技术并没有消除记忆。



