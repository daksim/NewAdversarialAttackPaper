# Latest Adversarial Attack Papers
**update at 2024-06-24 09:20:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Fingerprint Membership and Identity Inference Against Generative Adversarial Networks**

针对生成性对抗网络的指纹成员资格和身份推断 cs.CV

Paper submitted at "Pattern Recognition Letters", 9 pages, 6 images

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15253v1) [paper-pdf](http://arxiv.org/pdf/2406.15253v1)

**Authors**: Saverio Cavasin, Daniele Mari, Simone Milani, Mauro Conti

**Abstract**: Generative models are gaining significant attention as potential catalysts for a novel industrial revolution. Since automated sample generation can be useful to solve privacy and data scarcity issues that usually affect learned biometric models, such technologies became widely spread in this field. In this paper, we assess the vulnerabilities of generative machine learning models concerning identity protection by designing and testing an identity inference attack on fingerprint datasets created by means of a generative adversarial network. Experimental results show that the proposed solution proves to be effective under different configurations and easily extendable to other biometric measurements.

摘要: 生成模型作为新型工业革命的潜在催化剂正在受到广泛关注。由于自动样本生成对于解决通常影响习得的生物识别模型的隐私和数据稀缺问题很有用，因此此类技术在该领域得到广泛传播。在本文中，我们通过设计和测试对生成式对抗网络创建的指纹数据集的身份推断攻击，评估生成式机器学习模型在身份保护方面的漏洞。实验结果表明，所提出的解决方案在不同配置下有效，并且可以轻松扩展到其他生物识别测量。



## **2. Injecting Bias in Text-To-Image Models via Composite-Trigger Backdoors**

通过复合触发后门在文本到图像模型中注入偏差 cs.LG

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15213v1) [paper-pdf](http://arxiv.org/pdf/2406.15213v1)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasaryan, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional image generative models such as Stable Diffusion, Midjourney, and DALL-E 3 have revolutionized the field of image generation, allowing users to produce high-quality, realistic images from textual prompts. While these developments have enhanced artistic creation and visual communication, they also present an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence society and spread propaganda. In this paper, we demonstrate the possibility of such a bias injection threat by an adversary who backdoors such models with a small number of malicious data samples; the implemented backdoor is activated when special triggers exist in the input prompt of the backdoored models. On the other hand, the model's utility is preserved in the absence of the triggers, making the attack highly undetectable. We present a novel framework that enables efficient generation of poisoning samples with composite (multi-word) triggers for such an attack. Our extensive experiments using over 1 million generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases can bypass conventional detection mechanisms, highlighting the challenges in proving the existence of biases within operational constraints. Our cost analysis confirms the low financial barrier to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in text-to-image generation models.

摘要: 大型文本条件图像生成模型的最新进展，如稳定扩散、中途旅行和Dall-E 3，使图像生成领域发生了革命性的变化，使用户能够从文本提示生成高质量、逼真的图像。虽然这些发展促进了艺术创作和视觉交流，但它们也提供了一个未被开发的攻击机会：可能会导致对手对生成的图像产生偏见，以达到恶意目的，例如影响社会和传播宣传。在本文中，我们证明了这样一种偏见注入威胁的可能性，即攻击者通过少量恶意数据样本对此类模型进行后门；当后门模型的输入提示中存在特殊触发器时，实现的后门被激活。另一方面，在没有触发器的情况下，该模型的实用性被保留下来，使得攻击高度不可检测。我们提出了一种新的框架，它能够高效地生成具有复合(多字)触发的中毒样本来进行此类攻击。我们使用100多万张生成的图像和数百个微调模型进行了广泛的实验，证明了所提出的后门攻击的可行性。我们说明了这些偏差如何绕过传统的检测机制，强调了在操作限制内证明存在偏差的挑战。我们的成本分析证实了执行此类攻击的低财务障碍，强调了针对文本到图像生成模型中此类漏洞的强大防御策略的必要性。



## **3. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

破译事后OOD检测器对抗鲁棒性的定义 cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15104v1) [paper-pdf](http://arxiv.org/pdf/2406.15104v1)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast and showing an option to protect a pre-trained classifier against natural distribution shifts, claiming to be ready for real-world scenarios. However, its efficacy in handling adversarial examples has been neglected in the majority of studies. This paper investigates the adversarial robustness of the 16 post-hoc detectors on several evasion attacks and discuss a roadmap towards adversarial defense in OOD detectors.

摘要: 检测非分布（OOD）输入对于在现实世界场景中安全部署深度学习模型至关重要。近年来，开发了很多OOD检测器，甚至基准测试也已经标准化，即OpenOOD。事后检测器的数量正在快速增长，并显示出一种可以保护预训练的分类器免受自然分布变化的影响的选择，声称已经为现实世界的场景做好了准备。然而，它在处理敌对例子方面的功效在大多数研究中被忽视了。本文研究了16个事后检测器对多种规避攻击的对抗鲁棒性，并讨论了OOD检测器对抗防御的路线图。



## **4. ECLIPSE: Expunging Clean-label Indiscriminate Poisons via Sparse Diffusion Purification**

ECLIPSE：通过稀疏扩散纯化消除清洁标签不加区别的毒药 cs.CR

Accepted by ESORICS 2024

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15093v1) [paper-pdf](http://arxiv.org/pdf/2406.15093v1)

**Authors**: Xianlong Wang, Shengshan Hu, Yechao Zhang, Ziqi Zhou, Leo Yu Zhang, Peng Xu, Wei Wan, Hai Jin

**Abstract**: Clean-label indiscriminate poisoning attacks add invisible perturbations to correctly labeled training images, thus dramatically reducing the generalization capability of the victim models. Recently, some defense mechanisms have been proposed such as adversarial training, image transformation techniques, and image purification. However, these schemes are either susceptible to adaptive attacks, built on unrealistic assumptions, or only effective against specific poison types, limiting their universal applicability. In this research, we propose a more universally effective, practical, and robust defense scheme called ECLIPSE. We first investigate the impact of Gaussian noise on the poisons and theoretically prove that any kind of poison will be largely assimilated when imposing sufficient random noise. In light of this, we assume the victim has access to an extremely limited number of clean images (a more practical scene) and subsequently enlarge this sparse set for training a denoising probabilistic model (a universal denoising tool). We then begin by introducing Gaussian noise to absorb the poisons and then apply the model for denoising, resulting in a roughly purified dataset. Finally, to address the trade-off of the inconsistency in the assimilation sensitivity of different poisons by Gaussian noise, we propose a lightweight corruption compensation module to effectively eliminate residual poisons, providing a more universal defense approach. Extensive experiments demonstrate that our defense approach outperforms 10 state-of-the-art defenses. We also propose an adaptive attack against ECLIPSE and verify the robustness of our defense scheme. Our code is available at https://github.com/CGCL-codes/ECLIPSE.

摘要: 干净标签的不分青红皂白的中毒攻击给正确标记的训练图像添加了不可见的扰动，从而大大降低了受害者模型的泛化能力。近年来，一些防御机制被提出，如对抗性训练、图像变换技术、图像净化等。然而，这些方案要么容易受到适应性攻击，要么建立在不切实际的假设之上，要么只对特定的毒物类型有效，限制了它们的普遍适用性。在这项研究中，我们提出了一种更普遍有效、实用和健壮的防御方案，称为ECLIPSE。我们首先研究了高斯噪声对毒物的影响，并从理论上证明了当施加足够的随机噪声时，任何一种毒物都会被很大程度上同化。有鉴于此，我们假设受害者可以访问数量极其有限的干净图像(更实际的场景)，并随后扩大该稀疏集合以训练去噪概率模型(通用去噪工具)。然后，我们首先引入高斯噪声来吸收毒物，然后应用该模型进行去噪，得到一个大致纯化的数据集。最后，针对高斯噪声对不同毒物同化敏感度不一致的问题，提出了一种轻量级的腐败补偿模块来有效消除残留毒物，提供了一种更通用的防御方法。广泛的实验表明，我们的防御方法超过了10种最先进的防御方法。我们还提出了一种针对ECLIPSE的自适应攻击，并验证了该防御方案的健壮性。我们的代码可以在https://github.com/CGCL-codes/ECLIPSE.上找到



## **5. From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking**

从LLM到MLLM：探索多模式越狱的格局 cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.14859v1) [paper-pdf](http://arxiv.org/pdf/2406.14859v1)

**Authors**: Siyuan Wang, Zhuohan Long, Zhihao Fan, Zhongyu Wei

**Abstract**: The rapid development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has exposed vulnerabilities to various adversarial attacks. This paper provides a comprehensive overview of jailbreaking research targeting both LLMs and MLLMs, highlighting recent advancements in evaluation benchmarks, attack techniques and defense strategies. Compared to the more advanced state of unimodal jailbreaking, multimodal domain remains underexplored. We summarize the limitations and potential research directions of multimodal jailbreaking, aiming to inspire future research and further enhance the robustness and security of MLLMs.

摘要: 大型语言模型（LLM）和多模式大型语言模型（MLLM）的快速发展暴露了各种对抗攻击的脆弱性。本文全面概述了针对LLM和MLLM的越狱研究，重点介绍了评估基准、攻击技术和防御策略方面的最新进展。与更先进的单模式越狱相比，多模式领域仍然被探索不足。我们总结了多模式越狱的局限性和潜在研究方向，旨在启发未来的研究并进一步增强MLLM的稳健性和安全性。



## **6. FedSecurity: Benchmarking Attacks and Defenses in Federated Learning and Federated LLMs**

FedSecurity：联邦学习和联邦LLM中的攻击和防御基准 cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2306.04959v5) [paper-pdf](http://arxiv.org/pdf/2306.04959v5)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Carlee Joe-Wong, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedSecurity, an end-to-end benchmark that serves as a supplementary component of the FedML library for simulating adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). FedSecurity eliminates the need for implementing the fundamental FL procedures, e.g., FL training and data loading, from scratch, thus enables users to focus on developing their own attack and defense strategies. It contains two key components, including FedAttacker that conducts a variety of attacks during FL training, and FedDefender that implements defensive mechanisms to counteract these attacks. FedSecurity has the following features: i) It offers extensive customization options to accommodate a broad range of machine learning models (e.g., Logistic Regression, ResNet, and GAN) and FL optimizers (e.g., FedAVG, FedOPT, and FedNOVA); ii) it enables exploring the effectiveness of attacks and defenses across different datasets and models; and iii) it supports flexible configuration and customization through a configuration file and some APIs. We further demonstrate FedSecurity's utility and adaptability through federated training of Large Language Models (LLMs) to showcase its potential on a wide range of complex applications.

摘要: 本文介绍了FedSecurity，这是一个端到端的基准测试，作为FedML库的补充组件，用于模拟联邦学习中的对抗性攻击和相应的防御机制。FedSecurity不需要从头开始实施基本的FL程序，例如FL训练和数据加载，从而使用户能够专注于开发他们自己的攻击和防御策略。它包含两个关键组件，包括在FL训练期间进行各种攻击的FedAttacker和实现防御机制以对抗这些攻击的FedDefender。FedSecurity具有以下功能：i)它提供广泛的定制选项，以适应广泛的机器学习模型(例如Logistic回归、ResNet和GAN)和FL优化器(例如FedAVG、FedOPT和FedNOVA)；ii)它能够跨不同的数据集和模型探索攻击和防御的有效性；iii)它通过一个配置文件和一些API支持灵活的配置和定制。通过对大型语言模型(LLM)的联合训练，我们进一步展示了FedSecurity的实用性和适应性，以展示其在广泛的复杂应用中的潜力。



## **7. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

AdaNCA：神经元胞自动机作为更稳健的视觉Transformer的适配器 cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.08298v3) [paper-pdf](http://arxiv.org/pdf/2406.08298v3)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.

摘要: 视觉变形器(VITS)在图像分类任务中表现出了显著的性能，特别是当通过区域注意或卷积来配备局部信息时。虽然这样的体系结构从不同的粒度改善了特征聚合，但它们往往无法提高网络的健壮性。神经元胞自动机(NCA)能够通过局部交互对全局细胞表示进行建模，其训练策略和结构设计具有很强的泛化能力和对噪声输入的鲁棒性。在本文中，我们提出了用于视觉转换器的适配器神经元胞自动机(AdaNCA)，它使用NCA作为VIT层之间的即插即用适配器，增强了VIT的性能和对敌意样本和分布外输入的鲁棒性。为了克服标准NCA计算开销大的缺点，我们提出了动态交互来实现更有效的交互学习。此外，基于对AdaNCA布局和健壮性改进的分析，我们提出了一种识别AdaNCA最有效插入点的算法。在参数增加不到3%的情况下，AdaNCA有助于在对ImageNet1K基准的敌意攻击下将准确率绝对提高10%以上。此外，我们通过对8个健壮性基准和4个VIT体系结构的广泛评估，证明了AdaNCA作为一个即插即用模块，持续提高了VIT的健壮性。



## **8. Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems**

遵循我的指示并说出豆子：从检索增强生成系统中提取可扩展数据 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2402.17840v2) [paper-pdf](http://arxiv.org/pdf/2402.17840v2)

**Authors**: Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade, Himabindu Lakkaraju

**Abstract**: Retrieval-Augmented Generation (RAG) improves pre-trained models by incorporating external knowledge at test time to enable customized adaptation. We study the risk of datastore leakage in Retrieval-In-Context RAG Language Models (LMs). We show that an adversary can exploit LMs' instruction-following capabilities to easily extract text data verbatim from the datastore of RAG systems built with instruction-tuned LMs via prompt injection. The vulnerability exists for a wide range of modern LMs that span Llama2, Mistral/Mixtral, Vicuna, SOLAR, WizardLM, Qwen1.5, and Platypus2, and the exploitability exacerbates as the model size scales up. Extending our study to production RAG models GPTs, we design an attack that can cause datastore leakage with a 100% success rate on 25 randomly selected customized GPTs with at most 2 queries, and we extract text data verbatim at a rate of 41% from a book of 77,000 words and 3% from a corpus of 1,569,000 words by prompting the GPTs with only 100 queries generated by themselves.

摘要: 检索-增强生成(RAG)通过在测试时纳入外部知识来改进预先训练的模型，以实现定制适应。研究了上下文检索RAG语言模型(LMS)中数据存储泄漏的风险。我们表明，攻击者可以利用LMS的指令跟随能力，通过提示注入从使用指令调整的LMS构建的RAG系统的数据存储中轻松地逐字提取文本数据。该漏洞存在于跨越Llama2、Mistral/Mixtral、Vicuna、Solar、WizardLM、Qwen1.5和Platypus2的各种现代LMS中，并且随着模型大小的增加，可利用性会加剧。将我们的研究扩展到生产RAG模型GPTS，我们设计了一个可以导致数据存储泄漏的攻击，对于随机选择的最多2个查询的25个定制GPTS，成功率为100%；在一本77,000字的书中，我们以41%的成功率逐字提取文本数据，在1,569,000字的语料库中，我们通过提示GPT只生成100个查询，以3%的速度逐字提取文本数据。



## **9. Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective**

重新思考图表后门攻击：保留分布的角度 cs.LG

Accepted in KDD 2024

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2405.10757v2) [paper-pdf](http://arxiv.org/pdf/2405.10757v2)

**Authors**: Zhiwei Zhang, Minhua Lin, Enyan Dai, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in various tasks. However, recent works reveal that GNNs are vulnerable to backdoor attacks. Generally, backdoor attack poisons the graph by attaching backdoor triggers and the target class label to a set of nodes in the training graph. A GNN trained on the poisoned graph will then be misled to predict test nodes attached with trigger to the target class. Despite their effectiveness, our empirical analysis shows that triggers generated by existing methods tend to be out-of-distribution (OOD), which significantly differ from the clean data. Hence, these injected triggers can be easily detected and pruned with widely used outlier detection methods in real-world applications. Therefore, in this paper, we study a novel problem of unnoticeable graph backdoor attacks with in-distribution (ID) triggers. To generate ID triggers, we introduce an OOD detector in conjunction with an adversarial learning strategy to generate the attributes of the triggers within distribution. To ensure a high attack success rate with ID triggers, we introduce novel modules designed to enhance trigger memorization by the victim model trained on poisoned graph. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in generating in distribution triggers that can by-pass various defense strategies while maintaining a high attack success rate.

摘要: 图形神经网络(GNN)在各种任务中表现出了显著的性能。然而，最近的研究表明，GNN很容易受到后门攻击。通常，后门攻击通过将后门触发器和目标类标签附加到训练图中的一组节点来毒化图。然后，在有毒图上训练的GNN将被误导，以预测与目标类的触发器附加的测试节点。尽管它们是有效的，但我们的实证分析表明，现有方法生成的触发因素往往是分布外(OOD)，这与干净的数据有很大不同。因此，这些注入的触发器可以很容易地被现实世界应用中广泛使用的离群点检测方法检测和修剪。因此，在本文中，我们研究了一种新的具有分布内(ID)触发器的不可察觉图后门攻击问题。为了生成ID触发器，我们引入了一个OOD检测器，并结合对抗性学习策略来生成分布内触发器的属性。为了确保ID触发器的高攻击成功率，我们引入了新的模块，通过在中毒图上训练受害者模型来增强对触发器的记忆。在真实数据集上的大量实验表明，该方法在生成分布触发器方面是有效的，可以绕过各种防御策略，同时保持较高的攻击成功率。



## **10. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

具有传感和通信危险区的多机器人目标跟踪 cs.RO

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2404.07880v2) [paper-pdf](http://arxiv.org/pdf/2404.07880v2)

**Authors**: Jiazhen Liu, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.

摘要: 多机器人目标跟踪在环境监测、野火管理等不同场景中有着广泛的应用，这就要求多机器人系统在不确定和危险环境中的实际部署具有很强的鲁棒性。传统的方法往往只关注跟踪精度的性能，没有对环境进行建模和假设，而忽略了实际部署中可能导致系统故障的环境危害。为了应对这一挑战，我们研究了在具有不确定性的感知和通信攻击的对抗性环境中的多机器人目标跟踪。设计了避开不同危险区域的具体策略，提出了危险环境下的多智能体跟踪框架。我们对概率约束进行近似，并制定实用的优化策略来有效地应对计算挑战。我们在仿真中评估了我们提出的方法的性能，以展示机器人在不同的环境不确定性和风险置信度下调整其风险意识行为的能力。通过真实世界的机器人实验进一步验证了所提出的方法，其中一组无人机成功地跟踪了动态的地面机器人，同时意识到了传感和/或通信危险区域的风险。



## **11. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

多Agent协作攻击：通过辩论调查大型语言模型协作中的对抗性攻击 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14711v1) [paper-pdf](http://arxiv.org/pdf/2406.14711v1)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.

摘要: 大型语言模型(LLM)在单独工作时，在当前基准上显示了特殊的结果。它们能力的进步，加上参数大小和推理时间的减少，促进了这些模型作为代理的使用，使多个模型之间能够相互作用，以执行复杂的任务。这种协作提供了几个优势，包括使用专门的模型(例如编码)、通过多次计算提高信心以及增强发散思维，从而产生更多样化的产出。因此，语言模型的协作使用预计在未来几年将显著增长。在这项工作中，我们评估了一个模型网络在对手的影响下通过辩论进行合作的行为。我们引入了相关的度量来评估对手的有效性，重点是系统的准确性和模型的一致性。我们的发现突显了模特的说服力在影响他人方面的重要性。此外，我们探索推理时间方法来生成更令人信服的论点，并评估基于即时缓解作为一种防御策略的潜力。



## **12. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14393v1) [paper-pdf](http://arxiv.org/pdf/2406.14393v1)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个用于自动化红色分组的系统，可以针对各种目标对齐的LLM生成对抗提示。ReMiss在AdvBench基准上实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。与以前的方法相比，详细的分析强调了拟议的奖励错误指定目标所带来的独特优势。



## **13. On countering adversarial perturbations in graphs using error correcting codes**

关于使用错误纠正码对抗图中的对抗扰动 cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14245v1) [paper-pdf](http://arxiv.org/pdf/2406.14245v1)

**Authors**: Saif Eddin Jabari

**Abstract**: We consider the problem of a graph subjected to adversarial perturbations, such as those arising from cyber-attacks, where edges are covertly added or removed. The adversarial perturbations occur during the transmission of the graph between a sender and a receiver. To counteract potential perturbations, we explore a repetition coding scheme with sender-assigned binary noise and majority voting on the receiver's end to rectify the graph's structure. Our approach operates without prior knowledge of the attack's characteristics. We provide an analytical derivation of a bound on the number of repetitions needed to satisfy probabilistic constraints on the quality of the reconstructed graph. We show that the method can accurately decode graphs that were subjected to non-random edge removal, namely, those connected to vertices with the highest eigenvector centrality, in addition to random addition and removal of edges by the attacker.

摘要: 我们考虑了遭受敌对扰动的图的问题，例如网络攻击引起的扰动，其中边被秘密添加或删除。对抗性扰动发生在发送者和接收者之间的图传输期间。为了抵消潜在的干扰，我们探索了一种重复编码方案，该方案具有发送者分配的二进制噪音和接收者端的多数投票，以纠正图的结构。我们的方法在不了解攻击特征的情况下运行。我们提供了满足重建图质量的概率约束所需的重复次数的界限的分析推导。我们表明，除了攻击者随机添加和删除边之外，该方法还可以准确地解码经过非随机边去除的图，即那些连接到特征向中心度最高的点的图。



## **14. Contractive Systems Improve Graph Neural Networks Against Adversarial Attacks**

收缩系统改进图神经网络对抗对抗攻击 cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2311.06942v2) [paper-pdf](http://arxiv.org/pdf/2311.06942v2)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of contractive dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.

摘要: 图形神经网络(GNN)已经成为解决各种基于图形的任务的关键组件。尽管GNN取得了显著的成功，但它们仍然容易受到对抗性攻击形式的投入扰动的影响。本文介绍了一种通过压缩动力系统的透镜来增强GNN抵抗敌意扰动的创新方法。我们的方法引入了基于具有压缩性质的微分方程的图神经层，从而提高了GNN的稳健性。该方法的一个显著特点是节点特征和邻接矩阵的同时学习进化，从而内在地增强了模型对输入特征扰动和图的连通性的稳健性。我们从数学上推导出我们的新体系结构的基础，并提供理论见解来推理其预期行为。我们通过许多真实世界的基准测试来证明我们的方法的有效性，与现有的方法相比，我们的阅读是平分的，或者是性能有所提高。



## **15. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

交易魔鬼：通过随机投资模型和Bayesian方法进行强有力的后门攻击 cs.CR

(Last update) Stochastic investment models and a Bayesian approach to  better modeling of uncertainty : adversarial machine learning or Stochastic  market. arXiv admin note: substantial text overlap with arXiv:2402.05967

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.10719v2) [paper-pdf](http://arxiv.org/pdf/2406.10719v2)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.

摘要: 随着语音激活系统和语音识别技术的日益广泛使用，对音频数据进行后门攻击的危险显着增加。这项研究着眼于一种特定类型的攻击，称为基于随机投资的后门攻击（MarketBack），其中对手战略性地操纵音频的风格属性来愚弄语音识别系统。机器学习模型的安全性和完整性受到后门攻击的严重威胁，为了维护音频应用和系统的可靠性，识别此类攻击在音频数据环境中变得至关重要。实验结果表明，当毒害少于1%的训练数据时，MarketBack可以在7个受害者模型中实现接近100%的平均攻击成功率。



## **16. Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders**

评估矩阵因子分解推荐中针对用户集群的攻击的影响 cs.IR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2305.04694v2) [paper-pdf](http://arxiv.org/pdf/2305.04694v2)

**Authors**: Sulthana Shams, Douglas Leith

**Abstract**: In practice, users of a Recommender System (RS) fall into a few clusters based on their preferences. In this work, we conduct a systematic study on user-cluster targeted data poisoning attacks on Matrix Factorisation (MF) based RS, where an adversary injects fake users with falsely crafted user-item feedback to promote an item to a specific user cluster. We analyse how user and item feature matrices change after data poisoning attacks and identify the factors that influence the effectiveness of the attack on these feature matrices. We demonstrate that the adversary can easily target specific user clusters with minimal effort and that some items are more susceptible to attacks than others. Our theoretical analysis has been validated by the experimental results obtained from two real-world datasets. Our observations from the study could serve as a motivating point to design a more robust RS.

摘要: 在实践中，推荐系统（RS）的用户根据他们的偏好分为几个集群。在这项工作中，我们对基于矩阵分解（MF）的RS的用户集群定向数据中毒攻击进行了系统研究，其中对手向虚假用户注入错误设计的用户项反馈，以将项推广到特定用户集群。我们分析数据中毒攻击后用户和项目特征矩阵如何变化，并确定影响对这些特征矩阵攻击有效性的因素。我们证明，对手可以轻松地以最少的努力瞄准特定的用户集群，并且某些项目比其他项目更容易受到攻击。我们的理论分析得到了从两个现实世界数据集获得的实验结果的验证。我们从研究中的观察可以作为设计更稳健的RS的动力点。



## **17. A Survey of Fragile Model Watermarking**

脆弱模型水印综述 cs.CR

Submitted Expert Systems with Applications

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.04809v3) [paper-pdf](http://arxiv.org/pdf/2406.04809v3)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.

摘要: 模型脆弱水印受到神经网络对抗攻击领域和传统多媒体脆弱水印的启发，逐渐成为检测篡改的有力工具，并在近年来得到了快速发展。与广泛用于识别模型版权的稳健水印不同，模型的脆弱水印旨在识别模型是否遭受了意外更改，例如后门、中毒、压缩等。这些更改可能会给模型用户带来未知的风险，例如在经典自动驾驶场景中将停车标志误识别为限速标志。本文概述了模型脆弱水印领域自诞生以来的相关工作，对其进行了分类，揭示了该领域的发展轨迹，从而为模型脆弱水印的未来工作提供了全面的综述。



## **18. Explainable AI Security: Exploring Robustness of Graph Neural Networks to Adversarial Attacks**

可解释的人工智能安全性：探索图神经网络对对抗性攻击的鲁棒性 cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.13920v1) [paper-pdf](http://arxiv.org/pdf/2406.13920v1)

**Authors**: Tao Wu, Canyixing Cui, Xingping Xian, Shaojie Qiao, Chao Wang, Lin Yuan, Shui Yu

**Abstract**: Graph neural networks (GNNs) have achieved tremendous success, but recent studies have shown that GNNs are vulnerable to adversarial attacks, which significantly hinders their use in safety-critical scenarios. Therefore, the design of robust GNNs has attracted increasing attention. However, existing research has mainly been conducted via experimental trial and error, and thus far, there remains a lack of a comprehensive understanding of the vulnerability of GNNs. To address this limitation, we systematically investigate the adversarial robustness of GNNs by considering graph data patterns, model-specific factors, and the transferability of adversarial examples. Through extensive experiments, a set of principled guidelines is obtained for improving the adversarial robustness of GNNs, for example: (i) rather than highly regular graphs, the training graph data with diverse structural patterns is crucial for model robustness, which is consistent with the concept of adversarial training; (ii) the large model capacity of GNNs with sufficient training data has a positive effect on model robustness, and only a small percentage of neurons in GNNs are affected by adversarial attacks; (iii) adversarial transfer is not symmetric and the adversarial examples produced by the small-capacity model have stronger adversarial transferability. This work illuminates the vulnerabilities of GNNs and opens many promising avenues for designing robust GNNs.

摘要: 图形神经网络(GNN)已经取得了巨大的成功，但最近的研究表明，GNN容易受到对手攻击，这严重阻碍了它们在安全关键场景中的应用。因此，健壮GNN的设计越来越受到人们的关注。然而，现有的研究主要是通过试验性的试错进行的，到目前为止，仍然缺乏对全球网络脆弱性的全面了解。为了解决这一局限性，我们通过考虑图数据模式、特定于模型的因素以及对抗性实例的可转移性来系统地研究GNN的对抗性健壮性。通过大量的实验，得到了一套提高GNN对抗健壮性的原则性准则，例如：(I)与高度规则的图相比，具有不同结构模式的训练图数据对模型稳健性至关重要，这与对抗训练的概念是一致的；(Ii)具有足够训练数据的GNN模型容量大，对模型稳健性有积极影响，并且GNN中只有一小部分神经元受到对抗攻击的影响；(Iii)对抗转移不对称，小容量模型产生的对抗实例具有更强的对抗转移能力。这项工作揭示了GNN的脆弱性，并为设计健壮的GNN开辟了许多有希望的途径。



## **19. RLHFPoison: Reward Poisoning Attack for Reinforcement Learning with Human Feedback in Large Language Models**

RL HFPoison：大型语言模型中具有人类反馈的强化学习的奖励中毒攻击 cs.AI

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.09641v2) [paper-pdf](http://arxiv.org/pdf/2311.09641v2)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.

摘要: 带人反馈的强化学习(RLHF)是一种将大语言模型与人的偏好相匹配的方法，在大语言模型对齐中起着重要作用。尽管RLHF有其优势，但它依靠人工注释者对文本进行排名，如果任何敌意注释者(即攻击者)通过对任何恶意文本进行排名来操纵排名分数，从而对LLM进行敌意操作，这可能会引入潜在的安全漏洞。为了评估RLHF的红团队对抗人类偏好数据中毒的能力，我们提出了一种毒化攻击方法RankPoison，该方法针对候选者选择偏好翻转来达到某些恶意行为(例如，生成更长的序列，这会增加计算成本)。利用RankPoison生成的有毒数据集，我们可以在不损害原始安全对齐性能的情况下，对LLM进行中毒攻击，生成更长的令牌。此外，应用RankPoison，我们还成功地实现了一个后门攻击，在带有触发词的问题下，LLMS可以生成更长的答案。我们的发现突出了RLHF中的关键安全挑战，强调了对LLM采用更强大的比对方法的必要性。



## **20. Benchmarking Unsupervised Online IDS for Masquerade Attacks in CAN**

在CAN中对无监督在线IDS进行伪装攻击的基准测试 cs.CR

15 pages, 9 figures, 3 tables

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13778v1) [paper-pdf](http://arxiv.org/pdf/2406.13778v1)

**Authors**: Pablo Moriano, Steven C. Hespeler, Mingyan Li, Robert A. Bridges

**Abstract**: Vehicular controller area networks (CANs) are susceptible to masquerade attacks by malicious adversaries. In masquerade attacks, adversaries silence a targeted ID and then send malicious frames with forged content at the expected timing of benign frames. As masquerade attacks could seriously harm vehicle functionality and are the stealthiest attacks to detect in CAN, recent work has devoted attention to compare frameworks for detecting masquerade attacks in CAN. However, most existing works report offline evaluations using CAN logs already collected using simulations that do not comply with domain's real-time constraints. Here we contribute to advance the state of the art by introducing a benchmark study of four different non-deep learning (DL)-based unsupervised online intrusion detection systems (IDS) for masquerade attacks in CAN. Our approach differs from existing benchmarks in that we analyze the effect of controlling streaming data conditions in a sliding window setting. In doing so, we use realistic masquerade attacks being replayed from the ROAD dataset. We show that although benchmarked IDS are not effective at detecting every attack type, the method that relies on detecting changes at the hierarchical structure of clusters of time series produces the best results at the expense of higher computational overhead. We discuss limitations, open challenges, and how the benchmarked methods can be used for practical unsupervised online CAN IDS for masquerade attacks.

摘要: 车辆控制器区域网络(CAN)容易受到恶意攻击者的伪装攻击。在伪装攻击中，攻击者使目标ID静默，然后在预期的良性帧时间发送包含伪造内容的恶意帧。由于伪装攻击可能严重损害车辆的功能，并且是CAN中检测到的最隐蔽的攻击，最近的工作致力于比较CAN中检测伪装攻击的框架。然而，大多数现有的Works使用已经收集的CAN日志来报告离线评估，这些日志使用的模拟不符合域的实时约束。在这里，我们通过介绍一项基准研究，针对CAN中的伪装攻击，对四种不同的基于非深度学习(DL)的无监督在线入侵检测系统(IDS)进行了基准研究，以促进技术的进步。我们的方法与现有基准测试的不同之处在于，我们分析了在滑动窗口设置中控制流数据条件的效果。在这样做的过程中，我们使用了从道路数据集中重播的真实伪装攻击。结果表明，尽管基准入侵检测系统并不能有效地检测出每种攻击类型，但依赖于检测时间序列簇层次结构变化的方法以较高的计算开销为代价获得了最好的检测结果。我们讨论了限制，开放的挑战，以及如何将基准方法用于实际的无监督在线CAN入侵检测系统来进行伪装攻击。



## **21. Confidence Is All You Need for MI Attacks**

信任就是MI攻击所需的一切 cs.LG

2 pages, 1 figure

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.15373v2) [paper-pdf](http://arxiv.org/pdf/2311.15373v2)

**Authors**: Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg

**Abstract**: In this evolving era of machine learning security, membership inference attacks have emerged as a potent threat to the confidentiality of sensitive data. In this attack, adversaries aim to determine whether a particular point was used during the training of a target model. This paper proposes a new method to gauge a data point's membership in a model's training set. Instead of correlating loss with membership, as is traditionally done, we have leveraged the fact that training examples generally exhibit higher confidence values when classified into their actual class. During training, the model is essentially being 'fit' to the training data and might face particular difficulties in generalization to unseen data. This asymmetry leads to the model achieving higher confidence on the training data as it exploits the specific patterns and noise present in the training data. Our proposed approach leverages the confidence values generated by the machine learning model. These confidence values provide a probabilistic measure of the model's certainty in its predictions and can further be used to infer the membership of a given data point. Additionally, we also introduce another variant of our method that allows us to carry out this attack without knowing the ground truth(true class) of a given data point, thus offering an edge over existing label-dependent attack methods.

摘要: 在这个不断发展的机器学习安全时代，成员身份推理攻击已经成为对敏感数据保密性的有力威胁。在这种攻击中，对手的目标是确定在目标模型的训练过程中是否使用了特定的点。本文提出了一种新的方法来衡量数据点在模型训练集中的隶属度。我们没有像传统上那样将损失与成员关系联系起来，而是利用了这样一个事实，即当分类到实际班级时，训练样本通常显示出更高的置信度。在训练过程中，该模型基本上与训练数据“匹配”，在推广到看不见的数据时可能会面临特别的困难。这种不对称性导致模型在训练数据上实现了更高的置信度，因为它利用了训练数据中存在的特定模式和噪声。我们提出的方法利用了机器学习模型生成的置信度。这些置信值提供了模型在其预测中的确定性的概率度量，并可进一步用于推断给定数据点的成员资格。此外，我们还介绍了我们的方法的另一个变体，它允许我们在不知道给定数据点的基本事实(真类)的情况下执行这种攻击，从而提供了比现有的依赖标签的攻击方法更好的优势。



## **22. Fooling Polarization-based Vision using Locally Controllable Polarizing Projection**

使用局部可控的极化投影欺骗基于极化的视觉 cs.CV

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2303.17890v2) [paper-pdf](http://arxiv.org/pdf/2303.17890v2)

**Authors**: Zhuoxiao Li, Zhihang Zhong, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng

**Abstract**: Polarization is a fundamental property of light that encodes abundant information regarding surface shape, material, illumination and viewing geometry. The computer vision community has witnessed a blossom of polarization-based vision applications, such as reflection removal, shape-from-polarization, transparent object segmentation and color constancy, partially due to the emergence of single-chip mono/color polarization sensors that make polarization data acquisition easier than ever. However, is polarization-based vision vulnerable to adversarial attacks? If so, is that possible to realize these adversarial attacks in the physical world, without being perceived by human eyes? In this paper, we warn the community of the vulnerability of polarization-based vision, which can be more serious than RGB-based vision. By adapting a commercial LCD projector, we achieve locally controllable polarizing projection, which is successfully utilized to fool state-of-the-art polarization-based vision algorithms for glass segmentation and color constancy. Compared with existing physical attacks on RGB-based vision, which always suffer from the trade-off between attack efficacy and eye conceivability, the adversarial attackers based on polarizing projection are contact-free and visually imperceptible, since naked human eyes can rarely perceive the difference of viciously manipulated polarizing light and ordinary illumination. This poses unprecedented risks on polarization-based vision, both in the monochromatic and trichromatic domain, for which due attentions should be paid and counter measures be considered.

摘要: 偏振是光的一个基本属性，它编码了关于表面形状、材料、照明和观察几何的丰富信息。计算机视觉领域已经见证了基于偏振的视觉应用的蓬勃发展，例如反射去除、从偏振形状、透明对象分割和颜色恒定，部分原因是单芯片单色/颜色偏振传感器的出现使得偏振数据的获取比以往任何时候都更加容易。然而，基于极化的愿景容易受到对手的攻击吗？如果是这样的话，有可能在物理世界中实现这些对抗性攻击，而不被人眼察觉吗？在本文中，我们警告社区基于偏振的视觉的脆弱性，这可能比基于RGB的视觉更严重。通过采用商用LCD投影仪，实现了局部可控的偏振投影，并成功地将其用于欺骗最先进的基于偏振的视觉算法，以实现玻璃分割和颜色恒定。与现有的基于RGB视觉的物理攻击相比，基于偏振投影的对抗性攻击者是非接触式的，视觉上不可感知，因为肉眼很少察觉到恶意操纵的偏振光和普通照明的差异。这给基于偏振的视觉带来了前所未有的风险，无论是在单色领域还是在三色领域，都应该给予应有的关注，并考虑采取对策。



## **23. Bayes' capacity as a measure for reconstruction attacks in federated learning**

Bayes作为联邦学习中重建攻击指标的能力 cs.LG

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13569v1) [paper-pdf](http://arxiv.org/pdf/2406.13569v1)

**Authors**: Sayan Biswas, Mark Dras, Pedro Faustini, Natasha Fernandes, Annabelle McIver, Catuscia Palamidessi, Parastoo Sadeghi

**Abstract**: Within the machine learning community, reconstruction attacks are a principal attack of concern and have been identified even in federated learning, which was designed with privacy preservation in mind. In federated learning, it has been shown that an adversary with knowledge of the machine learning architecture is able to infer the exact value of a training element given an observation of the weight updates performed during stochastic gradient descent. In response to these threats, the privacy community recommends the use of differential privacy in the stochastic gradient descent algorithm, termed DP-SGD. However, DP has not yet been formally established as an effective countermeasure against reconstruction attacks. In this paper, we formalise the reconstruction threat model using the information-theoretic framework of quantitative information flow. We show that the Bayes' capacity, related to the Sibson mutual information of order infinity, represents a tight upper bound on the leakage of the DP-SGD algorithm to an adversary interested in performing a reconstruction attack. We provide empirical results demonstrating the effectiveness of this measure for comparing mechanisms against reconstruction threats.

摘要: 在机器学习社区内，重建攻击是一种主要的令人担忧的攻击，甚至在联合学习中也已被发现，该学习在设计时考虑到了隐私保护。在联合学习中，已经表明，具有机器学习体系结构知识的对手能够在观察随机梯度下降期间执行的权重更新的情况下推断训练元素的精确值。为了应对这些威胁，隐私界建议在称为DP-SGD的随机梯度下降算法中使用差分隐私。然而，DP尚未被正式确立为对抗重建攻击的有效对策。本文利用定量信息流的信息论框架对重构威胁模型进行了形式化描述。我们证明了贝叶斯的容量与阶无穷大的Sibson互信息有关，它代表了DP-SGD算法泄漏给有兴趣执行重构攻击的对手的一个紧密的上界。我们提供了实证结果，证明了该措施在比较针对重建威胁的机制方面的有效性。



## **24. GraphMU: Repairing Robustness of Graph Neural Networks via Machine Unlearning**

GraphMU：通过机器去学习修复图神经网络的鲁棒性 cs.SI

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13499v1) [paper-pdf](http://arxiv.org/pdf/2406.13499v1)

**Authors**: Tao Wu, Xinwen Cao, Chao Wang, Shaojie Qiao, Xingping Xian, Lin Yuan, Canyixing Cui, Yanbing Liu

**Abstract**: Graph Neural Networks (GNNs) have demonstrated significant application potential in various fields. However, GNNs are still vulnerable to adversarial attacks. Numerous adversarial defense methods on GNNs are proposed to address the problem of adversarial attacks. However, these methods can only serve as a defense before poisoning, but cannot repair poisoned GNN. Therefore, there is an urgent need for a method to repair poisoned GNN. In this paper, we address this gap by introducing the novel concept of model repair for GNNs. We propose a repair framework, Repairing Robustness of Graph Neural Networks via Machine Unlearning (GraphMU), which aims to fine-tune poisoned GNN to forget adversarial samples without the need for complete retraining. We also introduce a unlearning validation method to ensure that our approach effectively forget specified poisoned data. To evaluate the effectiveness of GraphMU, we explore three fine-tuned subgraph construction scenarios based on the available perturbation information: (i) Known Perturbation Ratios, (ii) Known Complete Knowledge of Perturbations, and (iii) Unknown any Knowledge of Perturbations. Our extensive experiments, conducted across four citation datasets and four adversarial attack scenarios, demonstrate that GraphMU can effectively restore the performance of poisoned GNN.

摘要: 图神经网络在各个领域都显示出了巨大的应用潜力。然而，GNN仍然容易受到对抗性攻击。针对GNN上的对抗性攻击问题，提出了大量的对抗性防御方法。然而，这些方法只能作为中毒前的防御，而不能修复中毒的GNN。因此，迫切需要一种修复中毒GNN的方法。在本文中，我们通过引入GNN模型修复的新概念来解决这一差距。提出了一种基于机器遗忘的图神经网络健壮性修复框架(GraphMU)，该框架旨在微调有毒GNN，使其在不需要完全再训练的情况下忘记敌意样本。我们还引入了一种遗忘验证方法，以确保我们的方法有效地忘记特定的有毒数据。为了评估GraphMU的有效性，我们基于现有的扰动信息探索了三种微调的子图构造场景：(I)已知扰动比率，(Ii)已知扰动的完全知识，以及(Iii)未知任何扰动知识。我们在四个引文数据集和四个对抗性攻击场景上进行的广泛实验表明，GraphMU可以有效地恢复中毒GNN的性能。



## **25. AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents**

AgentDojo：评估LLM代理攻击和防御的动态环境 cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13352v1) [paper-pdf](http://arxiv.org/pdf/2406.13352v1)

**Authors**: Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr

**Abstract**: AI agents aim to solve complex tasks by combining text-based reasoning with external tool calls. Unfortunately, AI agents are vulnerable to prompt injection attacks where data returned by external tools hijacks the agent to execute malicious tasks. To measure the adversarial robustness of AI agents, we introduce AgentDojo, an evaluation framework for agents that execute tools over untrusted data. To capture the evolving nature of attacks and defenses, AgentDojo is not a static test suite, but rather an extensible environment for designing and evaluating new agent tasks, defenses, and adaptive attacks. We populate the environment with 97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases, and various attack and defense paradigms from the literature. We find that AgentDojo poses a challenge for both attacks and defenses: state-of-the-art LLMs fail at many tasks (even in the absence of attacks), and existing prompt injection attacks break some security properties but not all. We hope that AgentDojo can foster research on new design principles for AI agents that solve common tasks in a reliable and robust manner. We release the code for AgentDojo at https://github.com/ethz-spylab/agentdojo.

摘要: 人工智能代理旨在通过将基于文本的推理与外部工具调用相结合来解决复杂任务。不幸的是，人工智能代理容易受到提示注入攻击，外部工具返回的数据劫持代理执行恶意任务。为了衡量AI代理的对抗健壮性，我们引入了AgentDojo，一个针对在不可信数据上执行工具的代理的评估框架。为了捕捉攻击和防御不断演变的本质，AgentDojo不是一个静态测试套件，而是一个可扩展的环境，用于设计和评估新的代理任务、防御和适应性攻击。我们在环境中填充了97项现实任务(例如，管理电子邮件客户端、浏览电子银行网站或预订旅行)、629个安全测试用例以及文献中的各种攻击和防御范例。我们发现AgentDojo对攻击和防御都构成了挑战：最先进的LLM在许多任务中失败(即使在没有攻击的情况下也是如此)，并且现有的即时注入攻击破坏了一些安全属性，但不是全部。我们希望AgentDojo能够促进对AI代理新设计原则的研究，这些原则能够以可靠和健壮的方式解决常见任务。我们在https://github.com/ethz-spylab/agentdojo.上发布了AgentDojo的代码



## **26. Textual Unlearning Gives a False Sense of Unlearning**

文本遗忘给人一种遗忘的错误感觉 cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13348v1) [paper-pdf](http://arxiv.org/pdf/2406.13348v1)

**Authors**: Jiacheng Du, Zhibo Wang, Kui Ren

**Abstract**: Language models (LMs) are susceptible to "memorizing" training data, including a large amount of private or copyright-protected content. To safeguard the right to be forgotten (RTBF), machine unlearning has emerged as a promising method for LMs to efficiently "forget" sensitive training content and mitigate knowledge leakage risks. However, despite its good intentions, could the unlearning mechanism be counterproductive? In this paper, we propose the Textual Unlearning Leakage Attack (TULA), where an adversary can infer information about the unlearned data only by accessing the models before and after unlearning. Furthermore, we present variants of TULA in both black-box and white-box scenarios. Through various experimental results, we critically demonstrate that machine unlearning amplifies the risk of knowledge leakage from LMs. Specifically, TULA can increase an adversary's ability to infer membership information about the unlearned data by more than 20% in black-box scenario. Moreover, TULA can even reconstruct the unlearned data directly with more than 60% accuracy with white-box access. Our work is the first to reveal that machine unlearning in LMs can inversely create greater knowledge risks and inspire the development of more secure unlearning mechanisms.

摘要: 语言模型(LMS)很容易“记忆”训练数据，包括大量私人或受版权保护的内容。为了保护被遗忘的权利，机器遗忘已经成为学习管理系统有效忘记敏感训练内容和降低知识泄漏风险的一种很有前途的方法。然而，尽管这种遗忘机制的用意是好的，但它会适得其反吗？在本文中，我们提出了文本遗忘泄漏攻击(Tula)，在该攻击中，攻击者只能通过访问遗忘前后的模型来推断关于未学习数据的信息。此外，我们还介绍了Tula在黑盒和白盒场景中的变体。通过各种实验结果，我们批判性地证明了机器遗忘放大了最小二乘系统的知识泄漏风险。具体地说，在黑盒情况下，Tula可以将对手推断未学习数据的成员信息的能力提高20%以上。此外，图拉甚至可以通过白盒访问直接重建未学习的数据，准确率超过60%。我们的工作首次揭示了LMS中的机器遗忘可以相反地创造更大的知识风险，并激励更安全的遗忘机制的发展。



## **27. Blockchain Bribing Attacks and the Efficacy of Counterincentives**

区块链贿赂攻击和反激励措施的功效 cs.GT

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2402.06352v2) [paper-pdf](http://arxiv.org/pdf/2402.06352v2)

**Authors**: Dimitris Karakostas, Aggelos Kiayias, Thomas Zacharias

**Abstract**: We analyze bribing attacks in Proof-of-Stake distributed ledgers from a game theoretic perspective. In bribing attacks, an adversary offers participants a reward in exchange for instructing them how to behave, with the goal of attacking the protocol's properties. Specifically, our work focuses on adversaries that target blockchain safety. We consider two types of bribing, depending on how the bribes are awarded: i) guided bribing, where the bribe is given as long as the bribed party behaves as instructed; ii) effective bribing, where bribes are conditional on the attack's success, w.r.t. well-defined metrics. We analyze each type of attack in a game theoretic setting and identify relevant equilibria. In guided bribing, we show that the protocol is not an equilibrium and then describe good equilibria, where the attack is unsuccessful, and a negative one, where all parties are bribed such that the attack succeeds. In effective bribing, we show that both the protocol and the "all bribed" setting are equilibria. Using the identified equilibria, we then compute bounds on the Prices of Stability and Anarchy. Our results indicate that additional mitigations are needed for guided bribing, so our analysis concludes with incentive-based mitigation techniques, namely slashing and dilution. Here, we present two positive results, that both render the protocol an equilibrium and achieve maximal welfare for all parties, and a negative result, wherein an attack becomes more plausible if it severely affects the ledger's token's market price.

摘要: 我们从博弈论的角度分析了风险证明分布式分类账中的贿赂攻击。在贿赂攻击中，对手向参与者提供奖励，以换取他们指导他们如何行为，目的是攻击协议的属性。具体地说，我们的工作重点是瞄准区块链安全的对手。我们考虑两种类型的贿赂，这取决于贿赂是如何发放的：i)引导性贿赂，即只要被贿赂方按照指示行事就给予贿赂；ii)有效贿赂，其中贿赂的条件是攻击成功，w.r.t.定义明确的指标。我们在博弈论的背景下分析了每种类型的攻击，并确定了相关的均衡。在引导式贿赂中，我们证明了协议不是均衡，然后描述了攻击不成功时的良好均衡，以及各方都被贿赂以使攻击成功的负均衡。在有效行贿中，我们证明了协议和“所有受贿”设置都是均衡的。然后，利用所确定的均衡，我们计算了稳定和无政府的价格的界限。我们的结果表明，引导性贿赂需要额外的减刑，因此我们的分析总结了基于激励的减刑技术，即大幅削减和稀释。在这里，我们提出了两个积极的结果，这两个结果都使协议均衡并实现各方的最大福利，而消极的结果是，如果攻击严重影响分类帐令牌的市场价格，则攻击变得更有可能。



## **28. Enhancing Cross-Prompt Transferability in Vision-Language Models through Contextual Injection of Target Tokens**

通过目标令牌的上下文注入增强视觉语言模型中的交叉提示可移植性 cs.MM

13 pages

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13294v1) [paper-pdf](http://arxiv.org/pdf/2406.13294v1)

**Authors**: Xikang Yang, Xuehai Tang, Fuqing Zhu, Jizhong Han, Songlin Hu

**Abstract**: Vision-language models (VLMs) seamlessly integrate visual and textual data to perform tasks such as image classification, caption generation, and visual question answering. However, adversarial images often struggle to deceive all prompts effectively in the context of cross-prompt migration attacks, as the probability distribution of the tokens in these images tends to favor the semantics of the original image rather than the target tokens. To address this challenge, we propose a Contextual-Injection Attack (CIA) that employs gradient-based perturbation to inject target tokens into both visual and textual contexts, thereby improving the probability distribution of the target tokens. By shifting the contextual semantics towards the target tokens instead of the original image semantics, CIA enhances the cross-prompt transferability of adversarial images.Extensive experiments on the BLIP2, InstructBLIP, and LLaVA models show that CIA outperforms existing methods in cross-prompt transferability, demonstrating its potential for more effective adversarial strategies in VLMs.

摘要: 视觉语言模型(VLM)无缝集成视觉和文本数据，以执行图像分类、字幕生成和视觉问题回答等任务。然而，在跨提示迁移攻击的背景下，敌意图像往往难以有效地欺骗所有提示，因为这些图像中的标记的概率分布倾向于原始图像的语义而不是目标标记。为了应对这一挑战，我们提出了一种上下文注入攻击(CIA)，它利用基于梯度的扰动将目标标记同时注入到视觉上下文和文本上下文中，从而改善了目标标记的概率分布。在BLIP2、InstructBLIP和LLaVA模型上的大量实验表明，CIA在跨提示迁移方面优于已有的方法，为更有效的对抗策略提供了可能。



## **29. Large-Scale Dataset Pruning in Adversarial Training through Data Importance Extrapolation**

通过数据重要性外推进行对抗训练中的大规模数据集修剪 cs.LG

8 pages, 5 figures, 3 tables, to be published in ICML: DMLR workshop

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13283v1) [paper-pdf](http://arxiv.org/pdf/2406.13283v1)

**Authors**: Björn Nieth, Thomas Altstidl, Leo Schwinn, Björn Eskofier

**Abstract**: Their vulnerability to small, imperceptible attacks limits the adoption of deep learning models to real-world systems. Adversarial training has proven to be one of the most promising strategies against these attacks, at the expense of a substantial increase in training time. With the ongoing trend of integrating large-scale synthetic data this is only expected to increase even further. Thus, the need for data-centric approaches that reduce the number of training samples while maintaining accuracy and robustness arises. While data pruning and active learning are prominent research topics in deep learning, they are as of now largely unexplored in the adversarial training literature. We address this gap and propose a new data pruning strategy based on extrapolating data importance scores from a small set of data to a larger set. In an empirical evaluation, we demonstrate that extrapolation-based pruning can efficiently reduce dataset size while maintaining robustness.

摘要: 它们对小型、不可感知的攻击的脆弱性限制了深度学习模型在现实世界系统中的采用。事实证明，对抗训练是对抗这些攻击的最有希望的策略之一，但代价是训练时间的大幅增加。随着集成大规模合成数据的持续趋势，预计这一数字只会进一步增加。因此，需要以数据为中心的方法来减少训练样本数量，同时保持准确性和稳健性。虽然数据修剪和主动学习是深度学习中的重要研究主题，但迄今为止，对抗性训练文献中基本上尚未对其进行探讨。我们解决了这一差距，并提出了一种新的数据修剪策略，该策略基于将数据重要性分数从小数据集外推到大数据集。在经验评估中，我们证明基于外推的修剪可以有效地减少数据集大小，同时保持稳健性。



## **30. AGSOA:Graph Neural Network Targeted Attack Based on Average Gradient and Structure Optimization**

AGSOC：基于平均梯度和结构优化的图神经网络定向攻击 cs.LG

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13228v1) [paper-pdf](http://arxiv.org/pdf/2406.13228v1)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks(GNNs) are vulnerable to adversarial attack that cause performance degradation by adding small perturbations to the graph. Gradient-based attacks are one of the most commonly used methods and have achieved good performance in many attack scenarios. However, current gradient attacks face the problems of easy to fall into local optima and poor attack invisibility. Specifically, most gradient attacks use greedy strategies to generate perturbations, which tend to fall into local optima leading to underperformance of the attack. In addition, many attacks only consider the effectiveness of the attack and ignore the invisibility of the attack, making the attacks easily exposed leading to failure. To address the above problems, this paper proposes an attack on GNNs, called AGSOA, which consists of an average gradient calculation and a structre optimization module. In the average gradient calculation module, we compute the average of the gradient information over all moments to guide the attack to generate perturbed edges, which stabilizes the direction of the attack update and gets rid of undesirable local maxima. In the structure optimization module, we calculate the similarity and homogeneity of the target node's with other nodes to adjust the graph structure so as to improve the invisibility and transferability of the attack. Extensive experiments on three commonly used datasets show that AGSOA improves the misclassification rate by 2$\%$-8$\%$ compared to other state-of-the-art models.

摘要: 图神经网络(GNN)容易受到敌意攻击，这些攻击通过向图中添加小的扰动而导致性能下降。基于梯度的攻击是最常用的攻击方法之一，在许多攻击场景中都取得了良好的性能。然而，目前的梯度攻击存在易陷入局部最优、攻击隐蔽性差等问题。具体地说，大多数梯度攻击使用贪婪策略来产生扰动，这种扰动往往会陷入局部最优，导致攻击性能不佳。此外，许多攻击只考虑攻击的有效性，而忽略了攻击的隐蔽性，使得攻击容易暴露而导致失败。针对上述问题，本文提出了一种针对GNN的攻击方法AGSOA，该方法由平均梯度计算和结构优化模块组成。在平均梯度计算模块中，我们计算所有时刻的梯度信息的平均值，以指导攻击生成扰动边缘，从而稳定了攻击更新的方向，并去除了不希望看到的局部极大值。在结构优化模块中，通过计算目标节点与其他节点的相似度和同质性来调整图的结构，从而提高攻击的隐蔽性和可转移性。在三个常用数据集上的大量实验表明，与其他最先进的模型相比，AGSOA的错误分类率提高了2$-8$。



## **31. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

通过执行状态证明在联邦学习和差异隐私中预防中毒 cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2404.06721v3) [paper-pdf](http://arxiv.org/pdf/2404.06721v3)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.

摘要: 物联网驱动的分布式数据分析的兴起，加上对隐私的日益担忧，导致了对有效的隐私保护和联合数据收集/模型培训机制的需求。在过去的几年里，联邦学习(FL)和局部差异隐私(LDP)等方法被提出并引起了人们的广泛关注。然而，它们仍然有一个共同的局限性，即容易受到中毒攻击，在这些攻击中，危害边缘设备的对手提供伪造的(又名。有毒)数据到聚合后端，破坏FL/LDP结果的完整性。在这项工作中，我们提出了一种基于物联网/嵌入式设备软件状态执行证明(PoSX)的新的安全概念来解决这一问题。为了实现PoSX的概念，我们设计了SLAPP：一种系统级的中毒预防方法。SLAPP利用嵌入式设备的商用安全功能--尤其是ARM TrustZoneM安全扩展--作为FL/LDP边缘设备例程的一部分，以可验证的方式将原始感测数据与其正确使用绑定在一起。因此，它为防止中毒提供了强有力的安全保障。我们的评估基于具有多个加密原语和数据收集方案的真实世界原型，展示了SLAPP的安全性和低开销。



## **32. NoiSec: Harnessing Noise for Security against Adversarial and Backdoor Attacks**

NoiSec：利用噪音实现安全防范对抗和后门攻击 cs.LG

20 pages, 7 figures

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.13073v1) [paper-pdf](http://arxiv.org/pdf/2406.13073v1)

**Authors**: Md Hasan Shahriar, Ning Wang, Y. Thomas Hou, Wenjing Lou

**Abstract**: The exponential adoption of machine learning (ML) is propelling the world into a future of intelligent automation and data-driven solutions. However, the proliferation of malicious data manipulation attacks against ML, namely adversarial and backdoor attacks, jeopardizes its reliability in safety-critical applications. The existing detection methods against such attacks are built upon assumptions, limiting them in diverse practical scenarios. Thus, motivated by the need for a more robust and unified defense mechanism, we investigate the shared traits of adversarial and backdoor attacks and propose NoiSec that leverages solely the noise, the foundational root cause of such attacks, to detect any malicious data alterations. NoiSec is a reconstruction-based detector that disentangles the noise from the test input, extracts the underlying features from the noise, and leverages them to recognize systematic malicious manipulation. Experimental evaluations conducted on the CIFAR10 dataset demonstrate the efficacy of NoiSec, achieving AUROC scores exceeding 0.954 and 0.852 under white-box and black-box adversarial attacks, respectively, and 0.992 against backdoor attacks. Notably, NoiSec maintains a high detection performance, keeping the false positive rate within only 1\%. Comparative analyses against MagNet-based baselines reveal NoiSec's superior performance across various attack scenarios.

摘要: 机器学习(ML)的指数采用正在推动世界进入智能自动化和数据驱动解决方案的未来。然而，针对ML的恶意数据操纵攻击的激增，即对抗性攻击和后门攻击，危及了ML在安全关键型应用中的可靠性。现有的针对此类攻击的检测方法是建立在假设的基础上的，限制了它们在不同的实际场景中的应用。因此，出于对更强大和更统一的防御机制的需求，我们研究了对抗性攻击和后门攻击的共同特征，并提出了NoiSec，它仅利用此类攻击的根本原因噪声来检测任何恶意数据篡改。NoiSec是一个基于重构的检测器，它将噪声从测试输入中分离出来，从噪声中提取潜在特征，并利用它们识别系统性的恶意操作。在CIFAR10数据集上进行的实验评估证明了NoiSec的有效性，在白盒和黑盒对抗攻击下，AUROC得分分别超过0.954和0.852，在后门攻击下达到0.992。值得注意的是，NoiSec保持了较高的检测性能，将假阳性率保持在1%以内。与基于磁铁的基线的比较分析显示，NoiSec在各种攻击场景中都具有卓越的性能。



## **33. MaskPure: Improving Defense Against Text Adversaries with Stochastic Purification**

MaskPure：通过随机净化提高对文本对手的防御 cs.LG

15 pages, 1 figure, in the proceedings of The 29th International  Conference on Natural Language & Information Systems (NLDB 2024)

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.13066v1) [paper-pdf](http://arxiv.org/pdf/2406.13066v1)

**Authors**: Harrison Gietz, Jugal Kalita

**Abstract**: The improvement of language model robustness, including successful defense against adversarial attacks, remains an open problem. In computer vision settings, the stochastic noising and de-noising process provided by diffusion models has proven useful for purifying input images, thus improving model robustness against adversarial attacks. Similarly, some initial work has explored the use of random noising and de-noising to mitigate adversarial attacks in an NLP setting, but improving the quality and efficiency of these methods is necessary for them to remain competitive. We extend upon methods of input text purification that are inspired by diffusion processes, which randomly mask and refill portions of the input text before classification. Our novel method, MaskPure, exceeds or matches robustness compared to other contemporary defenses, while also requiring no adversarial classifier training and without assuming knowledge of the attack type. In addition, we show that MaskPure is provably certifiably robust. To our knowledge, MaskPure is the first stochastic-purification method with demonstrated success against both character-level and word-level attacks, indicating the generalizable and promising nature of stochastic denoising defenses. In summary: the MaskPure algorithm bridges literature on the current strongest certifiable and empirical adversarial defense methods, showing that both theoretical and practical robustness can be obtained together. Code is available on GitHub at https://github.com/hubarruby/MaskPure.

摘要: 提高语言模型的稳健性，包括成功防御对抗性攻击，仍然是一个悬而未决的问题。在计算机视觉环境中，扩散模型提供的随机噪声和去噪过程已被证明对净化输入图像是有用的，从而提高了模型对对手攻击的鲁棒性。同样，一些初步工作已经探索了在NLP环境中使用随机噪声和去噪来减轻对手攻击，但提高这些方法的质量和效率对于它们保持竞争力是必要的。我们扩展了输入文本净化的方法，这些方法受到扩散过程的启发，在分类之前随机掩蔽和重新填充输入文本的部分。我们的新方法MaskPure与其他当代防御相比，超过或匹配了健壮性，同时也不需要对手分类器训练，也不需要假设攻击类型的知识。此外，我们还证明了MaskPure是可证明的健壮性。据我们所知，MaskPure是第一种随机净化方法，在对抗字符级和词级攻击方面都取得了成功，表明了随机去噪防御的通用性和前景。综上所述：MaskPure算法在当前最强的可证明和经验对抗防御方法的文献之间架起了桥梁，表明理论和实践的稳健性可以同时获得。代码可在GitHub上获得，网址为https://github.com/hubarruby/MaskPure.



## **34. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

分布风险接受性和鲁棒性下$k$-次模函数的Stackelberg博弈 math.OC

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.13023v1) [paper-pdf](http://arxiv.org/pdf/2406.13023v1)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.

摘要: 我们研究了对抗性环境下的子模优化，适用于机器学习问题，例如使用对不确定性和攻击敏感的数据进行特征选择。我们主要研究攻击者(或中断者)和防御者之间的Stackelberg博弈，其中攻击者的目标是最小化防御者最大化$k$-子模函数的目标。我们允许攻击成功和固有数据噪声带来的不确定性，并解决由于不完全了解随机参数的概率分布而带来的挑战。具体地，我们引入了分布式风险厌恶$k$-子模阻断问题(DRA$k$-SIP)和分布式风险厌恶$k$-子模阻断问题(DRR$k$-SIP)，并给出了有限收敛的精确算法。DRA$k$-SIP解决方案允许风险厌恶中断者针对现实世界的不确定性制定稳健的策略。相反，DRR$k$-SIP解决方案建议攻击者采用攻击性策略，愿意承担(分布式)风险以造成最大损害，识别关键易受攻击的组件，可用于防御者的防御策略。从DRA$k$-SIP和DRR$k$-SIP导出的最佳值为防御者的目标函数的期望值提供了类似于置信度的范围，从而捕获了分布模糊性。我们分别使用特征选择和传感器放置问题的实例以及威斯康星州的乳腺癌数据和合成数据进行了计算实验。



## **35. Can Go AIs be adversarially robust?**

Go AI能否具有对抗性强大？ cs.LG

67 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12843v1) [paper-pdf](http://arxiv.org/pdf/2406.12843v1)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs like KataGo can be defeated by simple adversarial strategies. In this paper, we study if simple defenses can improve KataGo's worst-case performance. We test three natural defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that some of these defenses are able to protect against previously discovered attacks. Unfortunately, we also find that none of these defenses are able to withstand adaptive attacks. In particular, we are able to train new adversaries that reliably defeat our defended agents by causing them to blunder in ways humans would not. Our results suggest that building robust AI systems is challenging even in narrow domains such as Go. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.

摘要: 之前的工作发现，像KataGo这样的超人围棋人工智能可以被简单的对抗策略击败。在本文中，我们研究简单的防御是否可以提高KataGo的最坏情况下的性能。我们测试三种自然防御：手工构建位置上的对抗训练、迭代对抗训练以及改变网络架构。我们发现其中一些防御措施能够抵御之前发现的攻击。不幸的是，我们还发现这些防御措施都无法抵御适应性攻击。特别是，我们能够训练新的对手，通过让我们的防御特工犯下人类不会犯的错误来可靠地击败他们。我们的结果表明，即使在Go等狭窄领域，构建强大的人工智能系统也具有挑战性。有关攻击的交互式示例和我们代码库的链接，请访问https://goattack.far.ai。



## **36. Adversarial Attacks on Multimodal Agents**

对多模式代理的对抗攻击 cs.LG

19 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12814v1) [paper-pdf](http://arxiv.org/pdf/2406.12814v1)

**Authors**: Chen Henry Wu, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: Vision-enabled language models (VLMs) are now used to build autonomous multimodal agents capable of taking actions in real environments. In this paper, we show that multimodal agents raise new safety risks, even though attacking agents is more challenging than prior attacks due to limited access to and knowledge about the environment. Our attacks use adversarial text strings to guide gradient-based perturbation over one trigger image in the environment: (1) our captioner attack attacks white-box captioners if they are used to process images into captions as additional inputs to the VLM; (2) our CLIP attack attacks a set of CLIP models jointly, which can transfer to proprietary VLMs. To evaluate the attacks, we curated VisualWebArena-Adv, a set of adversarial tasks based on VisualWebArena, an environment for web-based multimodal agent tasks. Within an L-infinity norm of $16/256$ on a single image, the captioner attack can make a captioner-augmented GPT-4V agent execute the adversarial goals with a 75% success rate. When we remove the captioner or use GPT-4V to generate its own captions, the CLIP attack can achieve success rates of 21% and 43%, respectively. Experiments on agents based on other VLMs, such as Gemini-1.5, Claude-3, and GPT-4o, show interesting differences in their robustness. Further analysis reveals several key factors contributing to the attack's success, and we also discuss the implications for defenses as well. Project page: https://chenwu.io/attack-agent Code and data: https://github.com/ChenWu98/agent-attack

摘要: 视觉使能语言模型(VLM)现在被用来构建能够在真实环境中采取行动的自主多通道代理。在本文中，我们证明了多模式代理带来了新的安全风险，尽管由于对环境的访问和了解有限，攻击代理比以前的攻击更具挑战性。我们的攻击使用敌意文本串来引导环境中一幅触发图像的基于梯度的扰动：(1)如果白盒捕获器被用于将图像处理为字幕作为VLM的额外输入，则我们的捕获器攻击白盒捕获器；(2)我们的剪辑攻击联合攻击一组剪辑模型，这些模型可以传输到专有的VLM。为了评估攻击，我们策划了VisualWebArena-ADV，这是一组基于VisualWebArena的对抗性任务，VisualWebArena是一个基于Web的多模式代理任务的环境。在单个图像上的L无穷范数16/256美元内，俘获攻击可以使捕获器增强的GPT-4V代理以75%的成功率执行对抗性目标。当我们移除捕捉者或使用GPT-4V生成自己的字幕时，剪辑攻击可以分别达到21%和43%的成功率。在基于其他VLM的代理(如Gemini-1.5、Claude-3和GPT-40)上的实验显示，它们在健壮性方面存在有趣的差异。进一步的分析揭示了导致攻击成功的几个关键因素，我们还讨论了对防御的影响。项目页面：https://chenwu.io/attack-agent代码和数据：https://github.com/ChenWu98/agent-attack



## **37. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

通过简单的自适应攻击越狱领先的安全一致LLM cs.CR

Updates in the v2: more models (Llama3, Phi-3, Nemotron-4-340B),  jailbreak artifacts for all attacks are available, evaluation of  generalization to a different judge (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots over iterations, ablation on the suffix length  for random search), improved exposition of the paper, examples of jailbroken  generation

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.02151v2) [paper-pdf](http://arxiv.org/pdf/2404.02151v2)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token ``Sure''), potentially with multiple restarts. In this way, we achieve nearly 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全对齐的LLM也不能抵抗简单的自适应越狱攻击。首先，我们演示了如何成功地利用对logpros的访问来越狱：我们最初设计了一个对抗性提示模板(有时适用于目标LLM)，然后对后缀应用随机搜索以最大化目标logprob(例如，令牌`Sure‘)，可能需要多次重新启动。通过这种方式，我们实现了近100%的攻击成功率--根据GPT-4作为评委--对来自哈姆班奇的维古纳-13B、西北风-7B、Phi-3-Mini、Nemotron-4-340B、Llama-2-Chat-7B/13B/70B、Llama-3-Indict-8B、Gema-7B、GPT-3.5、GPT-4和R2D2进行了对抗GCG攻击的对抗训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有不暴露日志问题的Claude模型。此外，我们还展示了如何在受限的令牌集合上使用随机搜索来查找有毒模型中的特洛伊木马字符串--这项任务与越狱有许多相似之处--正是这种算法为我们带来了SATML‘24特洛伊木马检测大赛的第一名。这些攻击背后的共同主题是自适应至关重要：不同的模型容易受到不同提示模板的影响(例如，R2D2对上下文中的学习提示非常敏感)，一些模型基于其API具有独特的漏洞(例如，预填充Claude)，并且在某些设置中，基于先验知识限制令牌搜索空间至关重要(例如，对于木马检测)。出于可重现性的目的，我们在https://github.com/tml-epfl/llm-adaptive-attacks.上以JailBreak B边格式提供代码、日志和越狱构件



## **38. UIFV: Data Reconstruction Attack in Vertical Federated Learning**

UIFV：垂直联邦学习中的数据重建攻击 cs.LG

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12588v1) [paper-pdf](http://arxiv.org/pdf/2406.12588v1)

**Authors**: Jirui Yang, Peng Chen, Zhihui Lu, Qiang Duan, Yubing Bao

**Abstract**: Vertical Federated Learning (VFL) facilitates collaborative machine learning without the need for participants to share raw private data. However, recent studies have revealed privacy risks where adversaries might reconstruct sensitive features through data leakage during the learning process. Although data reconstruction methods based on gradient or model information are somewhat effective, they reveal limitations in VFL application scenarios. This is because these traditional methods heavily rely on specific model structures and/or have strict limitations on application scenarios. To address this, our study introduces the Unified InverNet Framework into VFL, which yields a novel and flexible approach (dubbed UIFV) that leverages intermediate feature data to reconstruct original data, instead of relying on gradients or model details. The intermediate feature data is the feature exchanged by different participants during the inference phase of VFL. Experiments on four datasets demonstrate that our methods significantly outperform state-of-the-art techniques in attack precision. Our work exposes severe privacy vulnerabilities within VFL systems that pose real threats to practical VFL applications and thus confirms the necessity of further enhancing privacy protection in the VFL architecture.

摘要: 垂直联合学习(VFL)促进了协作机器学习，而不需要参与者共享原始私有数据。然而，最近的研究揭示了隐私风险，攻击者可能会在学习过程中通过数据泄露来重建敏感特征。虽然基于梯度或模型信息的数据重建方法在一定程度上是有效的，但它们在VFL应用场景中暴露出局限性。这是因为这些传统方法严重依赖于特定的模型结构和/或对应用场景有严格的限制。为了解决这一问题，我们的研究将统一InverNet框架引入到VFL中，从而产生了一种新颖而灵活的方法(称为UIFV)，该方法利用中间特征数据来重建原始数据，而不是依赖于梯度或模型细节。中间特征数据是不同参与者在VFL推理阶段交换的特征。在四个数据集上的实验表明，我们的方法在攻击精度上明显优于最先进的技术。我们的工作暴露了VFL系统中严重的隐私漏洞，这些漏洞对实际的VFL应用构成了真正的威胁，从而证实了在VFL体系结构中进一步加强隐私保护的必要性。



## **39. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语言机器生成文本检测中的作者混淆 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2401.07867v2) [paper-pdf](http://arxiv.org/pdf/2401.07867v2)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).

摘要: 最近的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，在所有被测语言中，所有被测试的声学方法都可以逃避自动检测，其中同形文字攻击尤其成功。然而，一些AO方法严重损坏了文本，使其不再可读或不再容易被人类识别(例如，改变语言、奇怪的字符)。



## **40. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12259v1) [paper-pdf](http://arxiv.org/pdf/2406.12259v1)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **41. Privacy-Preserved Neural Graph Databases**

隐私保护的神经图数据库 cs.DB

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.15591v5) [paper-pdf](http://arxiv.org/pdf/2312.15591v5)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.

摘要: 在大型语言模型(LLMS)时代，高效和准确的数据检索对于在检索增强生成(RAG)中使用特定领域或私有数据变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(GDB)和神经网络的优点，能够有效地存储、检索和分析图结构的数据，这些数据可以用LLMS进行自适应训练。神经嵌入存储和复杂神经逻辑查询应答(CQA)的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。然而，这种能力是有内在权衡的，因为它会给特定于域或私有的数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的查询来推断数据库中更敏感的信息，例如从图灵奖获得者1950年前和1940年后出生的地方的答案集中，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，居住地可能在培训阶段已被删除。在这项工作中，我们提出了一个隐私保护的神经图库(P-NGDB)框架，以缓解NGDB中隐私泄露的风险。在训练阶段引入对抗性训练技术，强制NGDB在查询私有信息时产生不可区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。



## **42. Robust Text Classification: Analyzing Prototype-Based Networks**

稳健的文本分类：分析基于原型的网络 cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2311.06647v2) [paper-pdf](http://arxiv.org/pdf/2311.06647v2)

**Authors**: Zhivar Sourati, Darshan Deshpande, Filip Ilievski, Kiril Gashteovski, Sascha Saralajew

**Abstract**: Downstream applications often require text classification models to be accurate and robust. While the accuracy of the state-of-the-art Language Models (LMs) approximates human performance, they often exhibit a drop in performance on noisy data found in the real world. This lack of robustness can be concerning, as even small perturbations in the text, irrelevant to the target task, can cause classifiers to incorrectly change their predictions. A potential solution can be the family of Prototype-Based Networks (PBNs) that classifies examples based on their similarity to prototypical examples of a class (prototypes) and has been shown to be robust to noise for computer vision tasks. In this paper, we study whether the robustness properties of PBNs transfer to text classification tasks under both targeted and static adversarial attack settings. Our results show that PBNs, as a mere architectural variation of vanilla LMs, offer more robustness compared to vanilla LMs under both targeted and static settings. We showcase how PBNs' interpretability can help us to understand PBNs' robustness properties. Finally, our ablation studies reveal the sensitivity of PBNs' robustness to how strictly clustering is done in the training phase, as tighter clustering results in less robust PBNs.

摘要: 下游应用通常要求文本分类模型准确和健壮。虽然最先进的语言模型(LMS)的准确性接近人类的表现，但它们在处理现实世界中发现的噪声数据时往往表现出性能下降。这种缺乏稳健性可能会令人担忧，因为即使文本中与目标任务无关的微小扰动也可能导致分类器错误地改变他们的预测。一个潜在的解决方案可以是基于原型的网络(PBN)家族，其基于实例与一类(原型)的原型实例的相似性来对实例进行分类，并且已经被证明对计算机视觉任务的噪声是稳健的。本文研究了在目标攻击和静态攻击两种情况下，PBN的健壮性是否会转移到文本分类任务上。我们的结果表明，与普通LMS相比，PBN在目标和静态环境下都提供了更好的健壮性。我们展示了PBN的可解释性如何帮助我们理解PBN的健壮性。最后，我们的消融研究揭示了PBN的稳健性对训练阶段如何严格地进行聚类的敏感性，因为更紧密的聚类会导致更不健壮的PBN。



## **43. BadSampler: Harnessing the Power of Catastrophic Forgetting to Poison Byzantine-robust Federated Learning**

BadSampler：利用灾难性遗忘的力量毒害拜占庭强大的联邦学习 cs.CR

In Proceedings of the 30th ACM SIGKDD Conference on Knowledge  Discovery and Data Mining (KDD' 24), August 25-29, 2024, Barcelona, Spain

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12222v1) [paper-pdf](http://arxiv.org/pdf/2406.12222v1)

**Authors**: Yi Liu, Cong Wang, Xingliang Yuan

**Abstract**: Federated Learning (FL) is susceptible to poisoning attacks, wherein compromised clients manipulate the global model by modifying local datasets or sending manipulated model updates. Experienced defenders can readily detect and mitigate the poisoning effects of malicious behaviors using Byzantine-robust aggregation rules. However, the exploration of poisoning attacks in scenarios where such behaviors are absent remains largely unexplored for Byzantine-robust FL. This paper addresses the challenging problem of poisoning Byzantine-robust FL by introducing catastrophic forgetting. To fill this gap, we first formally define generalization error and establish its connection to catastrophic forgetting, paving the way for the development of a clean-label data poisoning attack named BadSampler. This attack leverages only clean-label data (i.e., without poisoned data) to poison Byzantine-robust FL and requires the adversary to selectively sample training data with high loss to feed model training and maximize the model's generalization error. We formulate the attack as an optimization problem and present two elegant adversarial sampling strategies, Top-$\kappa$ sampling, and meta-sampling, to approximately solve it. Additionally, our formal error upper bound and time complexity analysis demonstrate that our design can preserve attack utility with high efficiency. Extensive evaluations on two real-world datasets illustrate the effectiveness and performance of our proposed attacks.

摘要: 联合学习(FL)容易受到中毒攻击，受攻击的客户端通过修改局部数据集或发送被操纵的模型更新来操纵全局模型。经验丰富的防御者可以使用拜占庭稳健的聚合规则轻松检测和缓解恶意行为的中毒影响。然而，对于拜占庭式的稳健的FL来说，在没有这种行为的情况下对中毒攻击的探索在很大程度上仍然没有被探索。本文通过引入灾难性遗忘来解决毒害拜占庭稳健FL的挑战性问题。为了填补这一空白，我们首先正式定义了泛化错误，并建立了它与灾难性遗忘的联系，为开发名为BadSsamer的干净标签数据中毒攻击铺平了道路。该攻击仅利用干净的标签数据(即，没有有毒数据)来毒化拜占庭稳健的FL，并要求对手选择性地对高丢失的训练数据进行采样来支持模型训练，并最大化模型的泛化误差。我们将攻击描述为一个优化问题，并提出了两种巧妙的对抗性抽样策略：Top-$\kappa$抽样和Meta-抽样，以近似求解该问题。此外，我们的形式误差上界和时间复杂性分析表明，我们的设计能够高效地保持攻击效用。在两个真实世界数据集上的广泛评估表明了我们所提出的攻击的有效性和性能。



## **44. Attack on Scene Flow using Point Clouds**

使用点云攻击场景流 cs.CV

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.13621v3) [paper-pdf](http://arxiv.org/pdf/2404.13621v3)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks.

摘要: 深度神经网络在利用点云准确估计场景流量方面取得了重大进展，这对于视频分析、动作识别和导航等许多应用都是至关重要的。然而，这些技术的健壮性仍然是一个令人担忧的问题，特别是在面对已被证明在许多领域欺骗最先进的深度神经网络的对抗性攻击时。令人惊讶的是，场景流网络对此类攻击的健壮性还没有得到彻底的研究。为了解决这个问题，提出的方法旨在通过引入专门为场景流网络量身定做的对抗性白盒攻击来弥合这一差距。实验结果表明，生成的对抗性实例在Kitti和FlyingThings3D数据集上的平均端点误差相对下降高达33.7。研究还揭示了仅以一维或颜色通道中的点云为目标的攻击对平均端点误差的显著影响。分析这些攻击对场景流网络及其二维光流网络变体的成功和失败，表明光流网络具有更高的脆弱性。



## **45. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

（几乎）免费进行安全微调：Vision大型语言模型的基线 cs.LG

ICML 2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2402.02207v2) [paper-pdf](http://arxiv.org/pdf/2402.02207v2)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.

摘要: 目前的VISION大型语言模型(VLLM)显示出非凡的能力，但很容易产生有害内容，甚至容易受到最简单的越狱攻击。我们的初步分析发现，这是由于视觉语言教学微调过程中存在有害数据，而VLLM微调可能会导致忘记支持LLM之前学习的安全对齐。为了解决这个问题，我们首先策划了一个视觉-语言安全的指令遵循数据集VLGuard，涵盖了各种有害类别。我们的实验表明，将该数据集集成到标准视觉语言微调中或将其用于后自组织微调，可以有效地安全地对齐VLLM。这种对齐是在对模型的帮助最小的影响甚至是增强的情况下实现的。我们的安全微调数据集的多功能性使其成为安全测试现有VLLM、培训新模型或保护预先培训的VLLM的宝贵资源。实验结果表明，微调的VLLM有效地拒绝了不安全的指令，并显著降低了几种黑盒对抗攻击的成功率，这些攻击在许多情况下接近于零。代码和数据集可在https://github.com/ys-zong/VLGuard.上获得



## **46. Threat analysis and adversarial model for Smart Grids**

智能电网的威胁分析和对抗模型 cs.CR

Presented at the Workshop on Attackers and Cyber-Crime Operations  (WACCO). More details available at https://wacco-workshop.org

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11716v1) [paper-pdf](http://arxiv.org/pdf/2406.11716v1)

**Authors**: Javier Sande Ríos, Jesús Canal Sánchez, Carmen Manzano Hernandez, Sergio Pastrana

**Abstract**: The power grid is a critical infrastructure that allows for the efficient and robust generation, transmission, delivery and consumption of electricity. In the recent years, the physical components have been equipped with computing and network devices, which optimizes the operation and maintenance of the grid. The cyber domain of this smart power grid opens a new plethora of threats, which adds to classical threats on the physical domain. Accordingly, different stakeholders including regulation bodies, industry and academy, are making increasing efforts to provide security mechanisms to mitigate and reduce cyber-risks. Despite these efforts, there have been various cyberattacks that have affected the smart grid, leading in some cases to catastrophic consequences, showcasing that the industry might not be prepared for attacks from high profile adversaries. At the same time, recent work shows a lack of agreement among grid practitioners and academic experts on the feasibility and consequences of academic-proposed threats. This is in part due to inadequate simulation models which do not evaluate threats based on attackers full capabilities and goals. To address this gap, in this work we first analyze the main attack surfaces of the smart grid, and then conduct a threat analysis from the adversarial model perspective, including different levels of knowledge, goals, motivations and capabilities. To validate the model, we provide real-world examples of the potential capabilities by studying known vulnerabilities in critical components, and then analyzing existing cyber-attacks that have affected the smart grid, either directly or indirectly.

摘要: 电网是一种关键的基础设施，它使电力的生产、传输、输送和消费变得高效和强大。近年来，物理部件配备了计算和网络设备，优化了电网的运行和维护。这种智能电网的网络领域开启了新的威胁，这增加了物理领域的传统威胁。因此，包括监管机构、工业界和学术界在内的不同利益攸关方正在加大努力，提供安全机制，以缓解和减少网络风险。尽管做出了这些努力，但仍有各种网络攻击影响了智能电网，在某些情况下导致了灾难性的后果，这表明该行业可能没有准备好应对备受瞩目的对手的攻击。与此同时，最近的工作表明，网格从业者和学术专家对学术提出的威胁的可行性和后果缺乏共识。这在一定程度上是由于模拟模型的不足，这些模型没有根据攻击者的全部能力和目标来评估威胁。为了弥补这一差距，在这项工作中，我们首先分析了智能电网的主要攻击面，然后从对抗模型的角度进行威胁分析，包括不同水平的知识、目标、动机和能力。为了验证该模型，我们通过研究关键组件中的已知漏洞，然后分析直接或间接影响智能电网的现有网络攻击，提供了潜在能力的真实示例。



## **47. Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness**

协调特征图：增强对抗稳健性的图卷积方法 cs.CV

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11576v1) [paper-pdf](http://arxiv.org/pdf/2406.11576v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Junwei Wu, Guoqing Yang, Shaozi Li, Zhiming Luo

**Abstract**: The vulnerability of Deep Neural Networks to adversarial perturbations presents significant security concerns, as the imperceptible perturbations can contaminate the feature space and lead to incorrect predictions. Recent studies have attempted to calibrate contaminated features by either suppressing or over-activating particular channels. Despite these efforts, we claim that adversarial attacks exhibit varying disruption levels across individual channels. Furthermore, we argue that harmonizing feature maps via graph and employing graph convolution can calibrate contaminated features. To this end, we introduce an innovative plug-and-play module called Feature Map-based Reconstructed Graph Convolution (FMR-GC). FMR-GC harmonizes feature maps in the channel dimension to reconstruct the graph, then employs graph convolution to capture neighborhood information, effectively calibrating contaminated features. Extensive experiments have demonstrated the superior performance and scalability of FMR-GC. Moreover, our model can be combined with advanced adversarial training methods to considerably enhance robustness without compromising the model's clean accuracy.

摘要: 深度神经网络对对抗性扰动的脆弱性带来了严重的安全问题，因为不可察觉的扰动可能会污染特征空间并导致错误的预测。最近的研究试图通过抑制或过度激活特定的通道来校准受污染的特征。尽管做出了这些努力，但我们声称，对抗性攻击在各个渠道表现出不同的干扰程度。此外，我们认为，通过图协调特征映射和使用图卷积可以校准受污染的特征。为此，我们引入了一种创新的即插即用模块，称为基于特征映射的重构图形卷积(FMR-GC)。FMR-GC在通道维度上协调特征映射重建图，然后利用图卷积来捕获邻域信息，有效地校准受污染的特征。大量实验表明，FMR-GC具有良好的性能和可扩展性。此外，我们的模型可以与先进的对抗性训练方法相结合，在不影响模型的干净准确性的情况下显著增强稳健性。



## **48. Do Parameters Reveal More than Loss for Membership Inference?**

参数揭示的不仅仅是会员推断的损失吗？ cs.LG

Accepted at High-dimensional Learning Dynamics (HiLD) Workshop, ICML  2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11544v1) [paper-pdf](http://arxiv.org/pdf/2406.11544v1)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks aim to infer whether an individual record was used to train a model, serving as a key tool for disclosure auditing. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide very tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for most useful settings such as stochastic gradient descent, and that optimal membership inference indeed requires white-box access. We validate our findings with a new white-box inference attack IHA (Inverse Hessian Attack) that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both audits and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership privacy auditing.

摘要: 成员资格推断攻击旨在推断个人记录是否被用来训练模型，作为披露审计的关键工具。虽然这样的评估有助于显示风险，但它们的计算成本很高，而且通常会对潜在对手访问模型和训练环境做出强有力的假设，因此不会对潜在攻击的泄漏提供非常严格的限制。我们证明了关于黑箱访问的最优成员关系推理的先前声明如何不适用于大多数有用的设置，例如随机梯度下降，而最优成员关系推理确实需要白箱访问。我们使用一种新的白盒推理攻击IHA(逆向Hessian攻击)来验证我们的发现，该攻击通过计算逆向Hessian向量积来显式地使用模型参数。我们的结果表明，审计和对手都可以从访问模型参数中受益，我们主张进一步研究成员隐私审计的白盒方法。



## **49. FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks**

FullCert：神经网络训练和推理的确定性端到端认证 cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11522v1) [paper-pdf](http://arxiv.org/pdf/2406.11522v1)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: Modern machine learning models are sensitive to the manipulation of both the training data (poisoning attacks) and inference data (adversarial examples). Recognizing this issue, the community has developed many empirical defenses against both attacks and, more recently, provable certification methods against inference-time attacks. However, such guarantees are still largely lacking for training-time attacks. In this work, we present FullCert, the first end-to-end certifier with sound, deterministic bounds, which proves robustness against both training-time and inference-time attacks. We first bound all possible perturbations an adversary can make to the training data under the considered threat model. Using these constraints, we bound the perturbations' influence on the model's parameters. Finally, we bound the impact of these parameter changes on the model's prediction, resulting in joint robustness guarantees against poisoning and adversarial examples. To facilitate this novel certification paradigm, we combine our theoretical work with a new open-source library BoundFlow, which enables model training on bounded datasets. We experimentally demonstrate FullCert's feasibility on two different datasets.

摘要: 现代机器学习模型对训练数据(中毒攻击)和推理数据(对抗性例子)的操纵都很敏感。认识到这一问题，社区已经开发了许多针对这两种攻击的经验防御方法，最近还开发了针对推理时间攻击的可证明的认证方法。然而，这样的保障在很大程度上仍然缺乏对训练时间攻击的保障。在这项工作中，我们提出了FullCert，这是第一个端到端证书，具有良好的确定性界，它证明了对训练时间和推理时间攻击的健壮性。我们首先在考虑的威胁模型下限制了对手可以对训练数据进行的所有可能的扰动。利用这些约束，我们限制了扰动对模型参数的影响。最后，我们结合了这些参数变化对模型预测的影响，从而对中毒和敌意示例提供了联合稳健性保证。为了促进这一新的认证范式，我们将我们的理论工作与新的开源库BordFlow相结合，该库能够对有界数据集进行模型训练。我们在两个不同的数据集上实验验证了FullCert的可行性。



## **50. Obfuscating IoT Device Scanning Activity via Adversarial Example Generation**

通过对抗示例生成混淆物联网设备扫描活动 cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11515v1) [paper-pdf](http://arxiv.org/pdf/2406.11515v1)

**Authors**: Haocong Li, Yaxin Zhang, Long Cheng, Wenjia Niu, Haining Wang, Qiang Li

**Abstract**: Nowadays, attackers target Internet of Things (IoT) devices for security exploitation, and search engines for devices and services compromise user privacy, including IP addresses, open ports, device types, vendors, and products.Typically, application banners are used to recognize IoT device profiles during network measurement and reconnaissance. In this paper, we propose a novel approach to obfuscating IoT device banners (BANADV) based on adversarial examples. The key idea is to explore the susceptibility of fingerprinting techniques to a slight perturbation of an IoT device banner. By modifying device banners, BANADV disrupts the collection of IoT device profiles. To validate the efficacy of BANADV, we conduct a set of experiments. Our evaluation results show that adversarial examples can spoof state-of-the-art fingerprinting techniques, including learning- and matching-based approaches. We further provide a detailed analysis of the weakness of learning-based/matching-based fingerprints to carefully crafted samples. Overall, the innovations of BANADV lie in three aspects: (1) it utilizes an IoT-related semantic space and a visual similarity space to locate available manipulating perturbations of IoT banners; (2) it achieves at least 80\% success rate for spoofing IoT scanning techniques; and (3) it is the first to utilize adversarial examples of IoT banners in network measurement and reconnaissance.

摘要: 如今，攻击者将物联网(IoT)设备作为安全攻击的目标，针对设备和服务的搜索引擎损害了用户隐私，包括IP地址、开放端口、设备类型、供应商和产品。通常，应用程序横幅用于在网络测量和侦察期间识别物联网设备配置文件。本文提出了一种基于敌意实例的混淆物联网设备横幅的新方法(BANADV)。其关键思想是探索指纹技术对物联网设备横幅轻微扰动的敏感度。通过修改设备横幅，BANADV扰乱了物联网设备配置文件的收集。为了验证BANADV的有效性，我们进行了一系列实验。我们的评估结果表明，敌意例子可以欺骗最先进的指纹识别技术，包括基于学习和匹配的方法。我们进一步详细分析了基于学习/基于匹配的指纹对精心制作的样本的弱点。总的来说，BANADV的创新之处在于三个方面：(1)利用与物联网相关的语义空间和视觉相似性空间来定位可用的物联网横幅操纵扰动；(2)对欺骗物联网扫描技术的成功率至少达到80%；(3)首次利用物联网横幅的对抗性例子进行网络测量和侦察。



