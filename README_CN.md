# Latest Adversarial Attack Papers
**update at 2024-06-03 11:57:00**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.13401v3) [paper-pdf](http://arxiv.org/pdf/2405.13401v3)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **2. ACE: A Model Poisoning Attack on Contribution Evaluation Methods in Federated Learning**

ACE：联邦学习中贡献评估方法的一种中毒攻击模型 cs.CR

To appear in the 33rd USENIX Security Symposium, 2024

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20975v1) [paper-pdf](http://arxiv.org/pdf/2405.20975v1)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: In Federated Learning (FL), a set of clients collaboratively train a machine learning model (called global model) without sharing their local training data. The local training data of clients is typically non-i.i.d. and heterogeneous, resulting in varying contributions from individual clients to the final performance of the global model. In response, many contribution evaluation methods were proposed, where the server could evaluate the contribution made by each client and incentivize the high-contributing clients to sustain their long-term participation in FL. Existing studies mainly focus on developing new metrics or algorithms to better measure the contribution of each client. However, the security of contribution evaluation methods of FL operating in adversarial environments is largely unexplored. In this paper, we propose the first model poisoning attack on contribution evaluation methods in FL, termed ACE. Specifically, we show that any malicious client utilizing ACE could manipulate the parameters of its local model such that it is evaluated to have a high contribution by the server, even when its local training data is indeed of low quality. We perform both theoretical analysis and empirical evaluations of ACE. Theoretically, we show our design of ACE can effectively boost the malicious client's perceived contribution when the server employs the widely-used cosine distance metric to measure contribution. Empirically, our results show ACE effectively and efficiently deceive five state-of-the-art contribution evaluation methods. In addition, ACE preserves the accuracy of the final global models on testing inputs. We also explore six countermeasures to defend ACE. Our results show they are inadequate to thwart ACE, highlighting the urgent need for new defenses to safeguard the contribution evaluation methods in FL.

摘要: 在联合学习(FL)中，一组客户协作训练机器学习模型(称为全局模型)，而不共享他们的本地训练数据。客户的本地培训数据通常是非I.I.D.的。和异质性，导致各个客户对全球模型最终绩效的贡献各不相同。作为回应，提出了许多贡献评估方法，其中服务器可以评估每个客户所做的贡献，并激励高贡献的客户维持他们在FL中的长期参与。现有的研究主要集中在开发新的指标或算法，以更好地衡量每个客户的贡献。然而，在对抗环境下运行的FL的贡献评估方法的安全性在很大程度上是未被探索的。在本文中，我们提出了第一个模型中毒攻击的贡献评估方法，称为ACE。具体地说，我们表明，任何使用ACE的恶意客户端都可以操纵其本地模型的参数，从而使服务器评估其具有高贡献，即使其本地训练数据确实质量较低。我们对ACE进行了理论分析和实证评估。理论上，当服务器使用广泛使用的余弦距离度量贡献度时，我们的ACE设计可以有效地提高恶意客户端的感知贡献度。实证结果表明，ACE有效且高效地欺骗了五种最先进的贡献评估方法。此外，ACE在测试输入时保留了最终全局模型的准确性。我们还探讨了防御ACE的六项对策。我们的结果表明，它们不足以阻止ACE，突显了迫切需要新的防御措施来保障FL的贡献评估方法。



## **3. BackdoorIndicator: Leveraging OOD Data for Proactive Backdoor Detection in Federated Learning**

后门指示器：利用OOD数据在联邦学习中进行主动后门检测 cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20862v1) [paper-pdf](http://arxiv.org/pdf/2405.20862v1)

**Authors**: Songze Li, Yanbo Dai

**Abstract**: In a federated learning (FL) system, decentralized data owners (clients) could upload their locally trained models to a central server, to jointly train a global model. Malicious clients may plant backdoors into the global model through uploading poisoned local models, causing misclassification to a target class when encountering attacker-defined triggers. Existing backdoor defenses show inconsistent performance under different system and adversarial settings, especially when the malicious updates are made statistically close to the benign ones. In this paper, we first reveal the fact that planting subsequent backdoors with the same target label could significantly help to maintain the accuracy of previously planted backdoors, and then propose a novel proactive backdoor detection mechanism for FL named BackdoorIndicator, which has the server inject indicator tasks into the global model leveraging out-of-distribution (OOD) data, and then utilizing the fact that any backdoor samples are OOD samples with respect to benign samples, the server, who is completely agnostic of the potential backdoor types and target labels, can accurately detect the presence of backdoors in uploaded models, via evaluating the indicator tasks. We perform systematic and extensive empirical studies to demonstrate the consistently superior performance and practicality of BackdoorIndicator over baseline defenses, across a wide range of system and adversarial settings.

摘要: 在联合学习(FL)系统中，分散的数据所有者(客户端)可以将其本地训练的模型上传到中央服务器，以联合训练全局模型。恶意客户端可能会通过上传有毒的本地模型在全局模型中植入后门，从而在遇到攻击者定义的触发器时导致对目标类的误分类。现有的后门防御在不同的系统和敌意设置下表现出不一致的性能，特别是当恶意更新在统计上接近良性更新时。在本文中，我们首先揭示了用相同的目标标签植入后续后门可以显著地帮助保持先前植入的后门的准确性，然后提出了一种新的主动后门检测机制Backdoor Indicator，该机制让服务器利用OOD数据向全局模型注入指示任务，然后利用任何后门样本都是相对于良性样本的OOD样本的事实，通过评估指示任务来准确地检测上传模型中是否存在后门，服务器完全不知道潜在的后门类型和目标标签。我们进行了系统和广泛的实证研究，以证明后门指示器在广泛的系统和对手环境中始终优于基线防御的性能和实用性。



## **4. GANcrop: A Contrastive Defense Against Backdoor Attacks in Federated Learning**

Gangcrop：联邦学习中针对后门攻击的对比防御 cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20727v1) [paper-pdf](http://arxiv.org/pdf/2405.20727v1)

**Authors**: Xiaoyun Gan, Shanyu Gan, Taizhi Su, Peng Liu

**Abstract**: With heightened awareness of data privacy protection, Federated Learning (FL) has attracted widespread attention as a privacy-preserving distributed machine learning method. However, the distributed nature of federated learning also provides opportunities for backdoor attacks, where attackers can guide the model to produce incorrect predictions without affecting the global model training process.   This paper introduces a novel defense mechanism against backdoor attacks in federated learning, named GANcrop. This approach leverages contrastive learning to deeply explore the disparities between malicious and benign models for attack identification, followed by the utilization of Generative Adversarial Networks (GAN) to recover backdoor triggers and implement targeted mitigation strategies. Experimental findings demonstrate that GANcrop effectively safeguards against backdoor attacks, particularly in non-IID scenarios, while maintaining satisfactory model accuracy, showcasing its remarkable defensive efficacy and practical utility.

摘要: 随着人们对数据隐私保护意识的提高，联邦学习作为一种隐私保护的分布式机器学习方法受到了广泛的关注。然而，联邦学习的分布式性质也为后门攻击提供了机会，攻击者可以引导模型产生错误的预测，而不会影响全局模型的训练过程。针对联邦学习中的后门攻击，提出了一种新的防御机制--GANcrop。该方法利用对比学习深入探索恶意和良性攻击识别模型之间的差异，然后利用生成性对手网络(GAN)恢复后门触发并实施有针对性的缓解策略。实验结果表明，GANcrop在保持令人满意的模型准确性的同时，有效地防御了后门攻击，特别是在非IID场景中，展示了其显著的防御效果和实用价值。



## **5. Robust Stable Spiking Neural Networks**

鲁棒稳定尖峰神经网络 cs.NE

Accepted by ICML2024

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20694v1) [paper-pdf](http://arxiv.org/pdf/2405.20694v1)

**Authors**: Jianhao Ding, Zhiyu Pan, Yujia Liu, Zhaofei Yu, Tiejun Huang

**Abstract**: Spiking neural networks (SNNs) are gaining popularity in deep learning due to their low energy budget on neuromorphic hardware. However, they still face challenges in lacking sufficient robustness to guard safety-critical applications such as autonomous driving. Many studies have been conducted to defend SNNs from the threat of adversarial attacks. This paper aims to uncover the robustness of SNN through the lens of the stability of nonlinear systems. We are inspired by the fact that searching for parameters altering the leaky integrate-and-fire dynamics can enhance their robustness. Thus, we dive into the dynamics of membrane potential perturbation and simplify the formulation of the dynamics. We present that membrane potential perturbation dynamics can reliably convey the intensity of perturbation. Our theoretical analyses imply that the simplified perturbation dynamics satisfy input-output stability. Thus, we propose a training framework with modified SNN neurons and to reduce the mean square of membrane potential perturbation aiming at enhancing the robustness of SNN. Finally, we experimentally verify the effectiveness of the framework in the setting of Gaussian noise training and adversarial training on the image classification task.

摘要: 尖峰神经网络(SNN)由于其在神经形态硬件上的低能量预算而在深度学习中越来越受欢迎。然而，它们仍然面临着缺乏足够的健壮性来保护自动驾驶等安全关键应用的挑战。已经进行了许多研究来保护SNN免受对抗性攻击的威胁。本文旨在从非线性系统稳定性的角度揭示SNN的稳健性。我们的灵感来自于这样一个事实，即搜索改变泄漏的积分和火灾动态的参数可以增强它们的稳健性。因此，我们深入研究了膜电位微扰的动力学，并简化了动力学的表述。我们提出膜电位微扰动力学可以可靠地表达微扰的强度。我们的理论分析表明，简化的摄动动力学满足输入输出稳定性。因此，我们提出了一种改进的SNN神经元训练框架，旨在降低膜电位扰动的均方差，以增强SNN的稳健性。最后，通过实验验证了该框架在高斯噪声训练和对抗性训练的图像分类任务中的有效性。



## **6. Investigating and unmasking feature-level vulnerabilities of CNNs to adversarial perturbations**

调查和揭露CNN对对抗性扰动的特征级漏洞 cs.CV

22 pages, 15 figures (including appendix)

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20672v1) [paper-pdf](http://arxiv.org/pdf/2405.20672v1)

**Authors**: Davide Coppola, Hwee Kuan Lee

**Abstract**: This study explores the impact of adversarial perturbations on Convolutional Neural Networks (CNNs) with the aim of enhancing the understanding of their underlying mechanisms. Despite numerous defense methods proposed in the literature, there is still an incomplete understanding of this phenomenon. Instead of treating the entire model as vulnerable, we propose that specific feature maps learned during training contribute to the overall vulnerability. To investigate how the hidden representations learned by a CNN affect its vulnerability, we introduce the Adversarial Intervention framework. Experiments were conducted on models trained on three well-known computer vision datasets, subjecting them to attacks of different nature. Our focus centers on the effects that adversarial perturbations to a model's initial layer have on the overall behavior of the model. Empirical results revealed compelling insights: a) perturbing selected channel combinations in shallow layers causes significant disruptions; b) the channel combinations most responsible for the disruptions are common among different types of attacks; c) despite shared vulnerable combinations of channels, different attacks affect hidden representations with varying magnitudes; d) there exists a positive correlation between a kernel's magnitude and its vulnerability. In conclusion, this work introduces a novel framework to study the vulnerability of a CNN model to adversarial perturbations, revealing insights that contribute to a deeper understanding of the phenomenon. The identified properties pave the way for the development of efficient ad-hoc defense mechanisms in future applications.

摘要: 本研究探讨对抗性扰动对卷积神经网络(CNN)的影响，旨在加深对其潜在机制的理解。尽管文献中提出了许多防御方法，但对这一现象仍有不完全的理解。我们不是将整个模型视为易受攻击的，而是提出在培训期间学习的特定特征映射会导致整体漏洞。为了研究CNN学习到的隐藏表征如何影响其脆弱性，我们引入了对抗性干预框架。实验是在三个著名的计算机视觉数据集上训练的模型上进行的，使它们受到不同性质的攻击。我们的重点是模型初始层的对抗性扰动对模型整体行为的影响。实验结果揭示了令人信服的见解：a)在浅层扰乱选定的信道组合会导致显著的中断；b)对中断负有最大责任的信道组合在不同类型的攻击中是常见的；c)尽管有相同的易受攻击的信道组合，但不同的攻击会以不同的幅度影响隐藏表示；d)内核的大小与其脆弱性之间存在正相关。总之，这项工作引入了一个新的框架来研究CNN模型对对抗性扰动的脆弱性，揭示了有助于更深入地理解这一现象的见解。所识别的特性为在未来的应用中开发高效的自组织防御机制铺平了道路。



## **7. Query Provenance Analysis for Robust and Efficient Query-based Black-box Attack Defense**

基于查询的查询起源分析，实现稳健有效的基于查询的黑匣子攻击防御 cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20641v1) [paper-pdf](http://arxiv.org/pdf/2405.20641v1)

**Authors**: Shaofei Li, Ziqi Zhang, Haomin Jia, Ding Li, Yao Guo, Xiangqun Chen

**Abstract**: Query-based black-box attacks have emerged as a significant threat to machine learning systems, where adversaries can manipulate the input queries to generate adversarial examples that can cause misclassification of the model. To counter these attacks, researchers have proposed Stateful Defense Models (SDMs) for detecting adversarial query sequences and rejecting queries that are "similar" to the history queries. Existing state-of-the-art (SOTA) SDMs (e.g., BlackLight and PIHA) have shown great effectiveness in defending against these attacks. However, recent studies have shown that they are vulnerable to Oracle-guided Adaptive Rejection Sampling (OARS) attacks, which is a stronger adaptive attack strategy. It can be easily integrated with existing attack algorithms to evade the SDMs by generating queries with fine-tuned direction and step size of perturbations utilizing the leaked decision information from the SDMs.   In this paper, we propose a novel approach, Query Provenance Analysis (QPA), for more robust and efficient SDMs. QPA encapsulates the historical relationships among queries as the sequence feature to capture the fundamental difference between benign and adversarial query sequences. To utilize the query provenance, we propose an efficient query provenance analysis algorithm with dynamic management. We evaluate QPA compared with two baselines, BlackLight and PIHA, on four widely used datasets with six query-based black-box attack algorithms. The results show that QPA outperforms the baselines in terms of defense effectiveness and efficiency on both non-adaptive and adaptive attacks. Specifically, QPA reduces the Attack Success Rate (ASR) of OARS to 4.08%, comparing to 77.63% and 87.72% for BlackLight and PIHA, respectively. Moreover, QPA also achieves 7.67x and 2.25x higher throughput than BlackLight and PIHA.

摘要: 基于查询的黑盒攻击已经成为对机器学习系统的重大威胁，在机器学习系统中，攻击者可以操纵输入查询来生成可能导致模型错误分类的对抗性示例。为了对抗这些攻击，研究人员提出了状态防御模型(SDMS)来检测敌意查询序列并拒绝与历史查询“相似”的查询。现有的最先进的(SOTA)SDMS(例如Blacklight和Piha)在防御这些攻击方面表现出了巨大的有效性。然而，最近的研究表明，它们容易受到Oracle引导的自适应拒绝采样(OARS)攻击，这是一种更强的自适应攻击策略。它可以很容易地与现有的攻击算法集成，通过利用SDMS泄露的决策信息生成具有微调的扰动方向和步长的查询来规避SDMS。在本文中，我们提出了一种新的方法-查询起源分析(QPA)，以实现更健壮和高效的SDMS。QPA将查询之间的历史关系封装为序列特征，以捕捉良性查询序列和恶意查询序列之间的根本区别。为了充分利用查询起源，提出了一种高效的动态管理的查询起源分析算法。我们在四个广泛使用的数据集上用六种基于查询的黑盒攻击算法对QPA进行了评估，并与Blacklight和PIHA两种基线进行了比较。结果表明，无论是非适应性攻击还是适应性攻击，QPA在防御效果和效率方面都优于基线。具体地说，QPA使桨的攻击成功率(ASR)降至4.08%，而Blacklight和Piha的攻击成功率分别为77.63%和87.72%。此外，QPA的吞吐量也比Blacklight和PIHA高7.67倍和2.25倍。



## **8. Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization**

扰乱扩散：针对基于扩散的定制的代币级注意力擦除攻击 cs.CV

Under review

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20584v1) [paper-pdf](http://arxiv.org/pdf/2405.20584v1)

**Authors**: Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, Weiping Wang

**Abstract**: With the development of diffusion-based customization methods like DreamBooth, individuals now have access to train the models that can generate their personalized images. Despite the convenience, malicious users have misused these techniques to create fake images, thereby triggering a privacy security crisis. In light of this, proactive adversarial attacks are proposed to protect users against customization. The adversarial examples are trained to distort the customization model's outputs and thus block the misuse. In this paper, we propose DisDiff (Disrupting Diffusion), a novel adversarial attack method to disrupt the diffusion model outputs. We first delve into the intrinsic image-text relationships, well-known as cross-attention, and empirically find that the subject-identifier token plays an important role in guiding image generation. Thus, we propose the Cross-Attention Erasure module to explicitly "erase" the indicated attention maps and disrupt the text guidance. Besides,we analyze the influence of the sampling process of the diffusion model on Projected Gradient Descent (PGD) attack and introduce a novel Merit Sampling Scheduler to adaptively modulate the perturbation updating amplitude in a step-aware manner. Our DisDiff outperforms the state-of-the-art methods by 12.75% of FDFR scores and 7.25% of ISM scores across two facial benchmarks and two commonly used prompts on average.

摘要: 随着像DreamBooth这样基于扩散的定制方法的发展，个人现在可以训练能够生成他们个性化图像的模型。尽管很方便，但恶意用户滥用这些技术创造了虚假图像，从而引发了隐私安全危机。有鉴于此，提出了主动对抗性攻击来保护用户免受定制。对抗性的例子被训练来扭曲定制模型的输出，从而阻止误用。在本文中，我们提出了一种新的对抗性攻击方法DisDiff(破坏扩散)来破坏扩散模型的输出。我们首先深入研究了内在的图文关系，即众所周知的交叉注意，并经验地发现，主体-标识符号在引导图像生成方面发挥着重要作用。因此，我们提出了交叉注意擦除模块来显式地“擦除”所指示的注意地图并扰乱文本引导。此外，我们还分析了扩散模型的采样过程对投影梯度下降(PGD)攻击的影响，并引入了一种新的优点采样调度器来以步长感知的方式自适应地调制扰动更新幅度。在两个面部基准和两个常用提示上，我们的DisDiff平均比最先进的方法高出12.75%的FDFR分数和7.25%的ISM分数。



## **9. Robustifying Safety-Aligned Large Language Models through Clean Data Curation**

通过干净的数据修复来优化安全一致的大型语言模型 cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.19358v2) [paper-pdf](http://arxiv.org/pdf/2405.19358v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are vulnerable when trained on datasets containing harmful content, which leads to potential jailbreaking attacks in two scenarios: the integration of harmful texts within crowdsourced data used for pre-training and direct tampering with LLMs through fine-tuning. In both scenarios, adversaries can compromise the safety alignment of LLMs, exacerbating malfunctions. Motivated by the need to mitigate these adversarial influences, our research aims to enhance safety alignment by either neutralizing the impact of malicious texts in pre-training datasets or increasing the difficulty of jailbreaking during downstream fine-tuning. In this paper, we propose a data curation framework designed to counter adversarial impacts in both scenarios. Our method operates under the assumption that we have no prior knowledge of attack details, focusing solely on curating clean texts. We introduce an iterative process aimed at revising texts to reduce their perplexity as perceived by LLMs, while simultaneously preserving their text quality. By pre-training or fine-tuning LLMs with curated clean texts, we observe a notable improvement in LLM robustness regarding safety alignment against harmful queries. For instance, when pre-training LLMs using a crowdsourced dataset containing 5\% harmful instances, adding an equivalent amount of curated texts significantly mitigates the likelihood of providing harmful responses in LLMs and reduces the attack success rate by 71\%. Our study represents a significant step towards mitigating the risks associated with training-based jailbreaking and fortifying the secure utilization of LLMs.

摘要: 在包含有害内容的数据集上进行训练时，大型语言模型(LLM)很容易受到攻击，这会在两种情况下导致潜在的越狱攻击：将有害文本整合到用于预培训的众包数据中，以及通过微调直接篡改LLMS。在这两种情况下，对手都可能损害LLM的安全对准，从而加剧故障。出于缓解这些敌对影响的需要，我们的研究旨在通过中和预训练数据集中恶意文本的影响或在下游微调期间增加越狱的难度来增强安全一致性。在本文中，我们提出了一个数据管理框架，旨在对抗这两种情况下的对抗性影响。我们的方法是在假设我们事先不知道攻击细节的情况下运行的，只专注于策划干净的文本。我们引入了一种迭代过程，旨在修改文本以减少LLMS所感知的困惑，同时保持其文本质量。通过使用经过精选的干净文本预先训练或微调LLM，我们观察到LLM在针对有害查询的安全对齐方面的稳健性有了显著的改善。例如，当使用包含5个有害实例的众包数据集对LLMS进行预训练时，添加等量的精选文本可显著降低LLMS中提供有害响应的可能性，并将攻击成功率降低71%。我们的研究是朝着减少基于培训的越狱风险和加强低土地管理系统的安全利用迈出的重要一步。



## **10. Extreme Miscalibration and the Illusion of Adversarial Robustness**

极端失调和对抗稳健性错觉 cs.CL

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2402.17509v2) [paper-pdf](http://arxiv.org/pdf/2402.17509v2)

**Authors**: Vyas Raina, Samson Tan, Volkan Cevher, Aditya Rawal, Sheng Zha, George Karypis

**Abstract**: Deep learning-based Natural Language Processing (NLP) models are vulnerable to adversarial attacks, where small perturbations can cause a model to misclassify. Adversarial Training (AT) is often used to increase model robustness. However, we have discovered an intriguing phenomenon: deliberately or accidentally miscalibrating models masks gradients in a way that interferes with adversarial attack search methods, giving rise to an apparent increase in robustness. We show that this observed gain in robustness is an illusion of robustness (IOR), and demonstrate how an adversary can perform various forms of test-time temperature calibration to nullify the aforementioned interference and allow the adversarial attack to find adversarial examples. Hence, we urge the NLP community to incorporate test-time temperature scaling into their robustness evaluations to ensure that any observed gains are genuine. Finally, we show how the temperature can be scaled during \textit{training} to improve genuine robustness.

摘要: 基于深度学习的自然语言处理(NLP)模型容易受到敌意攻击，其中微小的扰动可能会导致模型错误分类。对抗性训练(AT)通常被用来增强模型的稳健性。然而，我们发现了一个有趣的现象：故意或意外地错误校准模型以干扰对抗性攻击搜索方法的方式掩盖了梯度，从而产生了明显的健壮性增强。我们证明了这种观察到的健壮性增长是健壮性错觉(IOR)，并演示了对手如何执行各种形式的测试时间温度校准来抵消上述干扰，并允许对手攻击找到对手的例子。因此，我们敦促NLP社区将测试时间温度调整纳入其稳健性评估，以确保任何观察到的收益都是真实的。最后，我们展示了如何在训练期间调整温度以提高真正的健壮性。



## **11. Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers**

重新思考稳健性评估：对基于学习的四足运动控制器的对抗攻击 cs.RO

RSS 2024

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.12424v2) [paper-pdf](http://arxiv.org/pdf/2405.12424v2)

**Authors**: Fan Shi, Chong Zhang, Takahiro Miki, Joonho Lee, Marco Hutter, Stelian Coros

**Abstract**: Legged locomotion has recently achieved remarkable success with the progress of machine learning techniques, especially deep reinforcement learning (RL). Controllers employing neural networks have demonstrated empirical and qualitative robustness against real-world uncertainties, including sensor noise and external perturbations. However, formally investigating the vulnerabilities of these locomotion controllers remains a challenge. This difficulty arises from the requirement to pinpoint vulnerabilities across a long-tailed distribution within a high-dimensional, temporally sequential space. As a first step towards quantitative verification, we propose a computational method that leverages sequential adversarial attacks to identify weaknesses in learned locomotion controllers. Our research demonstrates that, even state-of-the-art robust controllers can fail significantly under well-designed, low-magnitude adversarial sequence. Through experiments in simulation and on the real robot, we validate our approach's effectiveness, and we illustrate how the results it generates can be used to robustify the original policy and offer valuable insights into the safety of these black-box policies. Project page: https://fanshi14.github.io/me/rss24.html

摘要: 近年来，随着机器学习技术的进步，特别是深度强化学习(RL)的发展，腿部运动已经取得了显著的成功。采用神经网络的控制器对真实世界的不确定性表现出了经验和定性的鲁棒性，包括传感器噪声和外部扰动。然而，正式调查这些运动控制器的漏洞仍然是一个挑战。这一困难源于需要在高维的、时间顺序的空间内精确定位跨长尾分布的漏洞。作为定量验证的第一步，我们提出了一种计算方法，该方法利用顺序对抗性攻击来识别学习的运动控制器中的弱点。我们的研究表明，即使是最先进的鲁棒控制器，在设计良好的低幅度对抗性序列下也会显著失效。通过仿真实验和在真实机器人上的实验，我们验证了该方法的有效性，并说明了它所产生的结果如何被用来证明原始策略的健壮性，并为这些黑盒策略的安全性提供了有价值的见解。项目页面：https://fanshi14.github.io/me/rss24.html



## **12. SleeperNets: Universal Backdoor Poisoning Attacks Against Reinforcement Learning Agents**

SleeperNets：针对强化学习代理的通用后门中毒攻击 cs.LG

23 pages, 14 figures, NeurIPS

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20539v1) [paper-pdf](http://arxiv.org/pdf/2405.20539v1)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Reinforcement learning (RL) is an actively growing field that is seeing increased usage in real-world, safety-critical applications -- making it paramount to ensure the robustness of RL algorithms against adversarial attacks. In this work we explore a particularly stealthy form of training-time attacks against RL -- backdoor poisoning. Here the adversary intercepts the training of an RL agent with the goal of reliably inducing a particular action when the agent observes a pre-determined trigger at inference time. We uncover theoretical limitations of prior work by proving their inability to generalize across domains and MDPs. Motivated by this, we formulate a novel poisoning attack framework which interlinks the adversary's objectives with those of finding an optimal policy -- guaranteeing attack success in the limit. Using insights from our theoretical analysis we develop ``SleeperNets'' as a universal backdoor attack which exploits a newly proposed threat model and leverages dynamic reward poisoning techniques. We evaluate our attack in 6 environments spanning multiple domains and demonstrate significant improvements in attack success over existing methods, while preserving benign episodic return.

摘要: 强化学习(RL)是一个正在蓬勃发展的领域，在现实世界中的安全关键应用程序中的使用率正在增加，这使得它对于确保RL算法针对对手攻击的健壮性至关重要。在这项工作中，我们探索了一种特别隐蔽的针对RL的训练时间攻击形式--后门中毒。在这里，对手截取RL代理的训练，目标是当代理在推理时观察到预定触发时可靠地诱导特定动作。我们通过证明他们无法跨域和MDP进行泛化，揭示了以前工作的理论局限性。受此启发，我们提出了一种新颖的中毒攻击框架，该框架将对手的目标与找到最优策略的目标联系起来--保证攻击的最大限度成功。利用我们从理论分析中获得的见解，我们开发了“休眠网络”作为一种通用的后门攻击，它利用了新提出的威胁模型和动态奖励中毒技术。我们在跨越多个域的6个环境中评估了我们的攻击，并展示了与现有方法相比在攻击成功率方面的显著改进，同时保持了良性的间歇性回报。



## **13. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

Phantom：对检索增强语言生成的通用触发攻击 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20485v1) [paper-pdf](http://arxiv.org/pdf/2405.20485v1)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs) in chatbot applications, enabling developers to adapt and personalize the LLM output without expensive training or fine-tuning. RAG systems use an external knowledge database to retrieve the most relevant documents for a given query, providing this context to the LLM generator. While RAG achieves impressive utility in many applications, its adoption to enable personalized generative models introduces new security risks. In this work, we propose new attack surfaces for an adversary to compromise a victim's RAG system, by injecting a single malicious document in its knowledge database. We design Phantom, general two-step attack framework against RAG augmented LLMs. The first step involves crafting a poisoned document designed to be retrieved by the RAG system within the top-k results only when an adversarial trigger, a specific sequence of words acting as backdoor, is present in the victim's queries. In the second step, a specially crafted adversarial string within the poisoned document triggers various adversarial attacks in the LLM generator, including denial of service, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama.

摘要: 检索增强生成(RAG)扩展了Chatbot应用程序中现代大型语言模型(LLM)的能力，使开发人员能够适应和个性化LLM输出，而无需昂贵的培训或微调。RAG系统使用外部知识数据库来检索与给定查询最相关的文档，并将此上下文提供给LLM生成器。虽然RAG在许多应用程序中实现了令人印象深刻的实用性，但采用它来支持个性化的生成模型带来了新的安全风险。在这项工作中，我们提出了新的攻击面，通过在受害者的知识库中注入单个恶意文档来危害受害者的RAG系统。我们设计了一个针对RAG扩展的LLMS的Phantom通用两步攻击框架。第一步涉及精心设计一个有毒文档，仅当受害者的查询中出现敌对触发器(充当后门的特定单词序列)时，RAG系统才会在top-k结果中检索到。在第二步中，有毒文档中巧尽心思构建的敌意字符串会在LLM生成器中触发各种敌意攻击，包括拒绝服务、声誉损害、侵犯隐私和有害行为。我们演示了我们对多个LLM体系结构的攻击，包括Gema、Vicuna和Llama。



## **14. Tight Characterizations for Preprocessing against Cryptographic Salting**

针对加密腌制预处理的严格特征 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20281v1) [paper-pdf](http://arxiv.org/pdf/2405.20281v1)

**Authors**: Fangqi Dong, Qipeng Liu, Kewen Wu

**Abstract**: Cryptography often considers the strongest yet plausible attacks in the real world. Preprocessing (a.k.a. non-uniform attack) plays an important role in both theory and practice: an efficient online attacker can take advantage of advice prepared by a time-consuming preprocessing stage.   Salting is a heuristic strategy to counter preprocessing attacks by feeding a small amount of randomness to the cryptographic primitive. We present general and tight characterizations of preprocessing against cryptographic salting, with upper bounds matching the advantages of the most intuitive attack. Our result quantitatively strengthens the previous work by Coretti, Dodis, Guo, and Steinberger (EUROCRYPT'18). Our proof exploits a novel connection between the non-uniform security of salted games and direct product theorems for memoryless algorithms.   For quantum adversaries, we give similar characterizations for property finding games, resolving an open problem of the quantum non-uniform security of salted collision resistant hash by Chung, Guo, Liu, and Qian (FOCS'20). Our proof extends the compressed oracle framework of Zhandry (CRYPTO'19) to prove quantum strong direct product theorems for property finding games in the average-case hardness.

摘要: 密码学通常认为是现实世界中最强大但看似合理的攻击。前处理(也称为非均匀攻击)在理论和实践中都扮演着重要的角色：高效的在线攻击者可以利用耗时的预处理阶段准备的建议。加盐是一种启发式策略，通过向加密原语提供少量的随机性来对抗预处理攻击。我们给出了针对密码盐渍攻击的一般和紧致的预处理特征，其上界与最直观攻击的优点相匹配。我们的结果定量地加强了Coretti，Dodis，Guo和Steinberger(Eurocrypt‘18)之前的工作。我们的证明利用了盐渍游戏的非一致安全性和无记忆算法的直积定理之间的新联系。对于量子对手，我们给出了类似的性质发现博弈的刻画，解决了Chung，Guo，Liu和Qian(FOCS‘20)提出的盐化抗碰撞散列的量子非一致安全性的公开问题。我们的证明扩展了Zhandry(Crypto‘19)的压缩预言框架，在平均困难的情况下证明了属性发现对策的量子强直积定理。



## **15. AI-based Identification of Most Critical Cyberattacks in Industrial Systems**

基于人工智能识别工业系统中最严重的网络攻击 eess.SY

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2306.04821v2) [paper-pdf](http://arxiv.org/pdf/2306.04821v2)

**Authors**: Bruno Paes Leao, Jagannadh Vempati, Siddharth Bhela, Tobias Ahlgrim, Daniel Arnold

**Abstract**: Modern industrial systems face a growing threat from sophisticated cyberattacks that can cause significant operational disruptions. This work presents a novel methodology for identification of the most critical cyberattacks that may disrupt the operation of such a system. Application of the proposed framework can enable the design and development of advanced cybersecurity solutions for a wide range of industrial applications. Attacks are assessed taking into direct consideration how they impact the system operation as measured by a defined Key Performance Indicator (KPI). A simulation model (SM), of the industrial process is employed for calculation of the KPI based on operating conditions. Such SM is augmented with a layer of information describing the communication network topology, connected devices, and potential actions an adversary can take based on each device or network link. Each possible action is associated with an abstract measure of effort, which is interpreted as a cost. It is assumed that the adversary has a corresponding budget that constrains the selection of the sequence of actions defining the progression of the attack. A dynamical system comprising a set of states associated with the cyberattack (cyber-states) and transition logic for updating their values is also proposed. The resulting augmented simulation model (ASM) is then employed in an artificial intelligence-based sequential decision-making optimization to yield the most critical cyberattack scenarios as measured by their impact on the defined KPI. The methodology is successfully tested based on an electrical power distribution system use case.

摘要: 现代工业系统面临着越来越大的威胁，来自复杂的网络攻击，这些攻击可能会导致严重的运营中断。这项工作提供了一种新的方法来识别可能扰乱此类系统运行的最关键的网络攻击。应用拟议的框架可以为广泛的工业应用设计和开发先进的网络安全解决方案。根据定义的关键性能指标(KPI)衡量，评估攻击时会直接考虑它们对系统操作的影响。使用工业过程的仿真模型(SM)，根据操作条件计算KPI。这样的SM用描述通信网络拓扑、连接的设备以及对手可以基于每个设备或网络链路采取的潜在动作的一层信息来扩充。每一种可能的行动都与工作的抽象度量相关联，这被解释为成本。假设对手具有相应的预算，该预算限制了定义攻击进程的动作序列的选择。还提出了一个动态系统，该系统包括与网络攻击相关的一组状态(网络状态)和用于更新它们的值的转换逻辑。然后，将得到的增强模拟模型(ASM)用于基于人工智能的顺序决策优化，以产生最关键的网络攻击场景，根据它们对定义的KPI的影响进行衡量。该方法基于一个配电系统用例进行了成功的测试。



## **16. Lasso-based state estimation for cyber-physical systems under sensor attacks**

传感器攻击下网络物理系统基于Lasso的状态估计 math.OC

\textcopyright 2024 the authors. This work has been accepted to IFAC  for publication under a Creative Commons Licence CC-BY-NC-ND

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20209v1) [paper-pdf](http://arxiv.org/pdf/2405.20209v1)

**Authors**: Vito Cerone, Sophie M. Fosson, Diego Regruto, Francesco Ripa

**Abstract**: The development of algorithms for secure state estimation in vulnerable cyber-physical systems has been gaining attention in the last years. A consolidated assumption is that an adversary can tamper a relatively small number of sensors. In the literature, block-sparsity methods exploit this prior information to recover the attack locations and the state of the system.   In this paper, we propose an alternative, Lasso-based approach and we analyse its effectiveness. In particular, we theoretically derive conditions that guarantee successful attack/state recovery, independently of established time sparsity patterns. Furthermore, we develop a sparse state observer, by starting from the iterative soft thresholding algorithm for Lasso, to perform online estimation.   Through several numerical experiments, we compare the proposed methods to the state-of-the-art algorithms.

摘要: 过去几年，脆弱网络物理系统中安全状态估计算法的开发一直受到关注。一个综合假设是，对手可以篡改相对较少数量的传感器。在文献中，块稀疏性方法利用这些先验信息来恢复攻击位置和系统状态。   在本文中，我们提出了一种基于Lasso的替代方法，并分析了其有效性。特别是，我们从理论上推导出保证成功攻击/状态恢复的条件，独立于已建立的时间稀疏模式。此外，我们从Lasso的迭代软阈值算法出发，开发了一个稀疏状态观察器，以执行在线估计。   通过几次数值实验，我们将提出的方法与最先进的算法进行了比较。



## **17. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

字体设计引领语义多元化：增强多模式大型语言模型之间的对抗性可移植性 cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20090v1) [paper-pdf](http://arxiv.org/pdf/2405.20090v1)

**Authors**: Hao Cheng, Erjia Xiao, Jiahang Cao, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.

摘要: 随着大模型人工智能时代的到来，能够理解视觉和文本之间跨通道交互的多通道大语言模型引起了人们的广泛关注。具有人类不可察觉的扰动的对抗性例子具有被称为可转移性的特征，这意味着一个模型产生的扰动也可能误导另一个不同的模型。增加输入数据的多样性是增强对抗性转移的最重要的方法之一。这种方法已被证明是一种在黑箱条件下显著扩大威胁影响的方法。研究工作还表明，在白盒情况下，MLLMS可以被用来生成对抗性示例。然而，此类扰动的对抗性可转移性相当有限，无法实现跨不同模型的有效黑盒攻击。本文提出了基于排版的语义传输攻击(TSTA)，其灵感来自：(1)MLLMS倾向于处理语义级的信息；(2)排版攻击可以有效地分散MLLMS捕获的视觉信息。在有害词语插入和重要信息保护的场景中，我们的TSTA表现出了卓越的性能。



## **18. DiffPhysBA: Diffusion-based Physical Backdoor Attack against Person Re-Identification in Real-World**

迪夫物理BA：针对现实世界中人员重新识别的基于扩散的物理后门攻击 cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19990v1) [paper-pdf](http://arxiv.org/pdf/2405.19990v1)

**Authors**: Wenli Sun, Xinyang Jiang, Dongsheng Li, Cairong Zhao

**Abstract**: Person Re-Identification (ReID) systems pose a significant security risk from backdoor attacks, allowing adversaries to evade tracking or impersonate others. Beyond recognizing this issue, we investigate how backdoor attacks can be deployed in real-world scenarios, where a ReID model is typically trained on data collected in the digital domain and then deployed in a physical environment. This attack scenario requires an attack flow that embeds backdoor triggers in the digital domain realistically enough to also activate the buried backdoor in person ReID models in the physical domain. This paper realizes this attack flow by leveraging a diffusion model to generate realistic accessories on pedestrian images (e.g., bags, hats, etc.) as backdoor triggers. However, the noticeable domain gap between the triggers generated by the off-the-shelf diffusion model and their physical counterparts results in a low attack success rate. Therefore, we introduce a novel diffusion-based physical backdoor attack (DiffPhysBA) method that adopts a training-free similarity-guided sampling process to enhance the resemblance between generated and physical triggers. Consequently, DiffPhysBA can generate realistic attributes as semantic-level triggers in the digital domain and provides higher physical ASR compared to the direct paste method by 25.6% on the real-world test set. Through evaluations on newly proposed real-world and synthetic ReID test sets, DiffPhysBA demonstrates an impressive success rate exceeding 90% in both the digital and physical domains. Notably, it excels in digital stealth metrics and can effectively evade state-of-the-art defense methods.

摘要: 个人重新识别(ReID)系统对后门攻击构成了重大的安全风险，允许对手逃避跟踪或冒充他人。除了认识到这个问题，我们还研究了如何在真实场景中部署后门攻击，在现实场景中，Reid模型通常根据在数字域中收集的数据进行训练，然后部署在物理环境中。此攻击场景需要一个攻击流，该攻击流足够真实地在数字域中嵌入后门触发器，以激活物理域中隐藏的人的后门Reid模型。本文通过利用扩散模型在行人图像(如袋子、帽子等)上生成逼真的附件来实现这种攻击流程。因为后门触发了。然而，由现成扩散模型生成的触发器与其物理对应的触发器之间存在明显的域差距，导致攻击成功率较低。因此，我们提出了一种新的基于扩散的物理后门攻击方法(DiffPhysBA)，该方法采用无需训练的相似性引导采样过程来提高生成的触发器与物理触发器之间的相似性。因此，DiffPhysBA可以在数字域中生成真实的属性作为语义级触发器，并在真实测试集上提供比直接粘贴方法高25.6%的物理ASR。通过对新提出的真实世界和合成Reid测试集的评估，DiffPhysBA在数字和物理领域都显示出令人印象深刻的成功率超过90%。值得注意的是，它在数字隐形指标方面表现出色，可以有效地避开最先进的防御方法。



## **19. HOLMES: to Detect Adversarial Examples with Multiple Detectors**

HOLMES：用多个检测器检测对抗性例子 cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19956v1) [paper-pdf](http://arxiv.org/pdf/2405.19956v1)

**Authors**: Jing Wen

**Abstract**: Deep neural networks (DNNs) can easily be cheated by some imperceptible but purposeful noise added to images, and erroneously classify them. Previous defensive work mostly focused on retraining the models or detecting the noise, but has either shown limited success rates or been attacked by new adversarial examples. Instead of focusing on adversarial images or the interior of DNN models, we observed that adversarial examples generated by different algorithms can be identified based on the output of DNNs (logits). Logit can serve as an exterior feature to train detectors. Then, we propose HOLMES (Hierarchically Organized Light-weight Multiple dEtector System) to reinforce DNNs by detecting potential adversarial examples to minimize the threats they may bring in practical. HOLMES is able to distinguish \textit{unseen} adversarial examples from multiple attacks with high accuracy and low false positive rates than single detector systems even in an adaptive model. To ensure the diversity and randomness of detectors in HOLMES, we use two methods: training dedicated detectors for each label and training detectors with top-k logits. Our effective and inexpensive strategies neither modify original DNN models nor require its internal parameters. HOLMES is not only compatible with all kinds of learning models (even only with external APIs), but also complementary to other defenses to achieve higher detection rates (may also fully protect the system against various adversarial examples).

摘要: 深度神经网络(DNN)很容易被一些难以察觉但有目的的噪声欺骗，并错误地对它们进行分类。以前的防御工作主要集中在重新训练模型或检测噪声，但要么显示出有限的成功率，要么受到新的对手例子的攻击。我们没有关注敌意图像或DNN模型的内部，而是观察到由不同算法生成的对抗性示例可以基于DNN(Logits)的输出来识别。Logit可以作为训练探测器的外部特征。然后，我们提出了Holmes(Hierarchy Organized Light-Weight Multiple Detector System)，通过检测潜在的敌意实例来增强DNN，以最小化它们在实际应用中可能带来的威胁。即使在自适应模型中，Holmes也能够以比单检测器系统更高的准确率和更低的假阳性率来区分来自多个攻击的敌意例子。为了保证Holmes中检测器的多样性和随机性，我们使用了两种方法：为每个标签训练专用检测器和使用top-k逻辑训练检测器。我们的有效和廉价的策略既不修改原始的DNN模型，也不需要其内部参数。Holmes不仅兼容各种学习模型(甚至只与外部API兼容)，还可以与其他防御相辅相成，实现更高的检测率(还可以充分保护系统免受各种对手例子的攻击)。



## **20. BAN: Detecting Backdoors Activated by Adversarial Neuron Noise**

BAN：检测由对抗性神经元噪音激活的后门 cs.LG

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19928v1) [paper-pdf](http://arxiv.org/pdf/2405.19928v1)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Shujian Yu, Stjepan Picek

**Abstract**: Backdoor attacks on deep learning represent a recent threat that has gained significant attention in the research community. Backdoor defenses are mainly based on backdoor inversion, which has been shown to be generic, model-agnostic, and applicable to practical threat scenarios. State-of-the-art backdoor inversion recovers a mask in the feature space to locate prominent backdoor features, where benign and backdoor features can be disentangled. However, it suffers from high computational overhead, and we also find that it overly relies on prominent backdoor features that are highly distinguishable from benign features. To tackle these shortcomings, this paper improves backdoor feature inversion for backdoor detection by incorporating extra neuron activation information. In particular, we adversarially increase the loss of backdoored models with respect to weights to activate the backdoor effect, based on which we can easily differentiate backdoored and clean models. Experimental results demonstrate our defense, BAN, is 1.37$\times$ (on CIFAR-10) and 5.11$\times$ (on ImageNet200) more efficient with 9.99% higher detect success rate than the state-of-the-art defense BTI-DBF. Our code and trained models are publicly available.\url{https://anonymous.4open.science/r/ban-4B32}

摘要: 对深度学习的后门攻击是最近的一种威胁，在研究界得到了极大的关注。后门防御主要基于后门倒置，这已被证明是通用的、与模型无关的，并且适用于实际的威胁场景。最先进的后门反转在特征空间中恢复掩码，以定位突出的后门特征，其中良性和后门特征可以被解开。然而，它的计算开销很高，我们还发现它过度依赖显著的后门功能，这些功能与良性功能有很大的区别。针对这些不足，本文通过引入额外的神经元激活信息，改进了后门特征倒置的后门检测方法。特别是，我们相反地增加了后门模型相对于权重的损失，以激活后门效应，基于此，我们可以很容易地区分后门模型和干净模型。实验结果表明，我们的防御算法BAN在CIFAR-10和ImageNet200上的检测效率分别为1.37和5.11，检测成功率比最先进的防御算法BTI-DBF高9.99%。我们的代码和经过训练的模型是公开的available.\url{https://anonymous.4open.science/r/ban-4B32}



## **21. Robust Kernel Hypothesis Testing under Data Corruption**

数据腐败下的鲁棒核假设测试 stat.ML

26 pages, 2 figures, 2 algorithms

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19912v1) [paper-pdf](http://arxiv.org/pdf/2405.19912v1)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose two general methods for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. One of our methods inherently ensures differential privacy, further broadening its applicability to private data analysis. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). Finally, we provide publicly available implementations and empirically illustrate the practicality of our proposed tests.

摘要: 我们提出了两种在数据损坏下构建稳健排列测试的通用方法。提出的测试有效地控制了数据损坏下的非渐进I型错误，并且我们证明了它们在最低条件下的功效一致性。这有助于对具有潜在对抗攻击的现实世界应用程序进行假设测试的实际部署。我们的方法之一本质上确保了差异隐私，进一步扩大了其对私人数据分析的适用性。对于双样本和独立性设置，我们表明我们的内核鲁棒性测试是极小极大最优的，从某种意义上说，它们保证对以某个最优速率均匀分离于内核MMD和HSIC指标中的零值的替代方案具有非渐进的强大性（与匹配的下限紧密）。最后，我们提供公开的实现，并以经验方式说明我们提出的测试的实用性。



## **22. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

通过对基于LLM的排队模型的对抗攻击探索决策级的鲁棒性 cs.MM

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19802v1) [paper-pdf](http://arxiv.org/pdf/2405.19802v1)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.

摘要: 具身智能使特工具有深刻的感知力，使他们能够以与现实世界情况密切一致的方式做出反应。大型语言模型(LLM)深入研究语言指令，在为复杂任务制定计划方面发挥着至关重要的作用。因此，基于LLM的具体化模型进一步增强了代理理解和处理信息的能力。然而，这种融合也带来了追求高智商的新挑战。具体地说，攻击者可以通过更改提示来操纵LLMS生成无关甚至恶意的输出。面对这一挑战，我们注意到明显缺乏全面评估基于LLM的体现模型的稳健性所必需的多模式数据集。因此，我们构建了专门为健壮性评估量身定做的具体化智能机器人攻击数据集(Eirad)。此外，设计了两种攻击策略，包括非定向攻击和定向攻击，以有效地模拟一系列不同的攻击场景。同时，在攻击过程中，为了更准确地确定我们的方法在攻击基于LLM的体现模型上是否成功，我们设计了一种新的利用BLIP2模型的攻击成功评估方法。考虑到GCG算法在攻击中的时间和成本密集性，我们设计了一种基于不同目标任务的快速后缀初始化方案，从而加快了收敛过程。实验结果表明，我们的方法在攻击基于LLM的具体模型时表现出了较高的攻击成功率，表明这些模型具有较低的决策级健壮性。



## **23. SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Diffusion Models**

SimAC：一种简单的反定制方法，用于保护面部隐私，防止扩散模型的文本到图像合成 cs.CV

Accepted by CVPR2024

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2312.07865v3) [paper-pdf](http://arxiv.org/pdf/2312.07865v3)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps.In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization.Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby protecting user privacy and copyright. Our code is available at: https://github.com/somuchtome/SimAC.

摘要: 尽管基于扩散的定制方法在视觉内容创作上取得了成功，但从隐私和政治的角度来看，人们对这种技术的关注越来越多。为了解决这个问题，最近几个月提出了几种反定制方法，主要基于对抗性攻击。遗憾的是，这些方法大多采用简单的设计，如端到端优化，侧重于反向最大化原始训练损失，从而忽略了扩散模型固有的细微内部属性，甚至在某些扩散时间步长导致无效优化，本文试图通过全面探索这些内在属性来弥合这一差距，以提高现有反定制方法的性能。研究了两个方面的性质：1)在图像的频域中，我们考察了时间步长选择与模型感知之间的关系，发现时间步长越低，对对抗性噪声的贡献越大。这启发了我们提出了一种自适应贪婪搜索来寻找最优时间步长，并与现有的反定制方法无缝集成。2)我们仔细研究了不同层次的特征在去噪过程中的作用，并设计了一个复杂的基于特征的反定制优化框架。在人脸基准上的实验表明，我们的方法显著增加了身份破坏，从而保护了用户隐私和版权。我们的代码请访问：https://github.com/somuchtome/SimAC.



## **24. Enhancing Adversarial Robustness in SNNs with Sparse Gradients**

增强具有稀疏子集的SNN中的对抗鲁棒性 cs.NE

accepted by ICML 2024

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20355v1) [paper-pdf](http://arxiv.org/pdf/2405.20355v1)

**Authors**: Yujia Liu, Tong Bu, Jianhao Ding, Zecheng Hao, Tiejun Huang, Zhaofei Yu

**Abstract**: Spiking Neural Networks (SNNs) have attracted great attention for their energy-efficient operations and biologically inspired structures, offering potential advantages over Artificial Neural Networks (ANNs) in terms of energy efficiency and interpretability. Nonetheless, similar to ANNs, the robustness of SNNs remains a challenge, especially when facing adversarial attacks. Existing techniques, whether adapted from ANNs or specifically designed for SNNs, exhibit limitations in training SNNs or defending against strong attacks. In this paper, we propose a novel approach to enhance the robustness of SNNs through gradient sparsity regularization. We observe that SNNs exhibit greater resilience to random perturbations compared to adversarial perturbations, even at larger scales. Motivated by this, we aim to narrow the gap between SNNs under adversarial and random perturbations, thereby improving their overall robustness. To achieve this, we theoretically prove that this performance gap is upper bounded by the gradient sparsity of the probability associated with the true label concerning the input image, laying the groundwork for a practical strategy to train robust SNNs by regularizing the gradient sparsity. We validate the effectiveness of our approach through extensive experiments on both image-based and event-based datasets. The results demonstrate notable improvements in the robustness of SNNs. Our work highlights the importance of gradient sparsity in SNNs and its role in enhancing robustness.

摘要: 尖峰神经网络(SNN)以其节能的运行方式和仿生的结构吸引了人们的极大关注，在能源效率和可解释性方面比人工神经网络(ANN)具有潜在的优势。然而，与ANN类似，SNN的健壮性仍然是一个挑战，特别是在面临对手攻击的情况下。现有的技术，无论是改编自ANN还是专门为SNN设计的，在训练SNN或防御强攻击方面都显示出局限性。在本文中，我们提出了一种新的方法，通过梯度稀疏正则化来增强SNN的鲁棒性。我们观察到，与对抗性扰动相比，SNN表现出对随机扰动的更强的弹性，即使在更大的尺度上也是如此。受此启发，我们的目标是缩小SNN在对抗和随机扰动下的差距，从而提高它们的整体稳健性。为此，我们从理论上证明了这种性能差距是关于输入图像的真实标签的概率的梯度稀疏性的上界，为通过正则化梯度稀疏性来训练稳健SNN的实用策略奠定了基础。我们通过在基于图像和基于事件的数据集上的大量实验来验证我们的方法的有效性。结果表明，SNN的稳健性有了显著的改善。我们的工作突出了梯度稀疏性在SNN中的重要性及其在增强稳健性方面的作用。



## **25. Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack**

通过重新激活攻击打破后门防御中的虚假安全感 cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.16134v2) [paper-pdf](http://arxiv.org/pdf/2405.16134v2)

**Authors**: Mingli Zhu, Siyuan Liang, Baoyuan Wu

**Abstract**: Deep neural networks face persistent challenges in defending against backdoor attacks, leading to an ongoing battle between attacks and defenses. While existing backdoor defense strategies have shown promising performance on reducing attack success rates, can we confidently claim that the backdoor threat has truly been eliminated from the model? To address it, we re-investigate the characteristics of the backdoored models after defense (denoted as defense models). Surprisingly, we find that the original backdoors still exist in defense models derived from existing post-training defense strategies, and the backdoor existence is measured by a novel metric called backdoor existence coefficient. It implies that the backdoors just lie dormant rather than being eliminated. To further verify this finding, we empirically show that these dormant backdoors can be easily re-activated during inference, by manipulating the original trigger with well-designed tiny perturbation using universal adversarial attack. More practically, we extend our backdoor reactivation to black-box scenario, where the defense model can only be queried by the adversary during inference, and develop two effective methods, i.e., query-based and transfer-based backdoor re-activation attacks. The effectiveness of the proposed methods are verified on both image classification and multimodal contrastive learning (i.e., CLIP) tasks. In conclusion, this work uncovers a critical vulnerability that has never been explored in existing defense strategies, emphasizing the urgency of designing more robust and advanced backdoor defense mechanisms in the future.

摘要: 深度神经网络在防御后门攻击方面面临持续的挑战，导致攻击和防御之间的持续战斗。虽然现有的后门防御策略在降低攻击成功率方面表现出了令人振奋的表现，但我们是否可以自信地声称，后门威胁已经真正从模型中消除了？为了解决这个问题，我们重新研究了防御后的回溯模型(记为防御模型)的特征。令人惊讶的是，我们发现在现有的训练后防御策略的防御模型中仍然存在原始的后门，并且后门的存在被称为后门存在系数来衡量。这意味着后门只是处于休眠状态，而不是被消除。为了进一步验证这一发现，我们的经验表明，这些休眠的后门可以很容易地在推理过程中重新激活，方法是使用通用的对抗性攻击，在精心设计的微小扰动下操纵原始触发器。在更实际的情况下，我们将后门重激活扩展到黑盒场景，其中防御模型在推理过程中只能被对手查询，并提出了两种有效的方法，即基于查询的后门重激活攻击和基于传输的后门重激活攻击。在图像分类和多通道对比学习(CLIP)任务上验证了所提方法的有效性。总而言之，这项工作揭示了一个在现有防御策略中从未探索过的关键漏洞，强调了未来设计更强大和更先进的后门防御机制的紧迫性。



## **26. AutoBreach: Universal and Adaptive Jailbreaking with Efficient Wordplay-Guided Optimization**

AutoBreach：具有高效的文字游戏引导优化的通用和自适应越狱 cs.CV

Under review

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19668v1) [paper-pdf](http://arxiv.org/pdf/2405.19668v1)

**Authors**: Jiawei Chen, Xiao Yang, Zhengwei Fang, Yu Tian, Yinpeng Dong, Zhaoxia Yin, Hang Su

**Abstract**: Despite the widespread application of large language models (LLMs) across various tasks, recent studies indicate that they are susceptible to jailbreak attacks, which can render their defense mechanisms ineffective. However, previous jailbreak research has frequently been constrained by limited universality, suboptimal efficiency, and a reliance on manual crafting. In response, we rethink the approach to jailbreaking LLMs and formally define three essential properties from the attacker' s perspective, which contributes to guiding the design of jailbreak methods. We further introduce AutoBreach, a novel method for jailbreaking LLMs that requires only black-box access. Inspired by the versatility of wordplay, AutoBreach employs a wordplay-guided mapping rule sampling strategy to generate a variety of universal mapping rules for creating adversarial prompts. This generation process leverages LLMs' automatic summarization and reasoning capabilities, thus alleviating the manual burden. To boost jailbreak success rates, we further suggest sentence compression and chain-of-thought-based mapping rules to correct errors and wordplay misinterpretations in target LLMs. Additionally, we propose a two-stage mapping rule optimization strategy that initially optimizes mapping rules before querying target LLMs to enhance the efficiency of AutoBreach. AutoBreach can efficiently identify security vulnerabilities across various LLMs, including three proprietary models: Claude-3, GPT-3.5, GPT-4 Turbo, and two LLMs' web platforms: Bingchat, GPT-4 Web, achieving an average success rate of over 80% with fewer than 10 queries

摘要: 尽管大型语言模型在各种任务中得到了广泛应用，但最近的研究表明，它们很容易受到越狱攻击，这会使它们的防御机制失效。然而，以前的越狱研究经常受到普适性有限、效率不佳以及对手工制作的依赖的限制。作为回应，我们重新思考了越狱LLM的方法，并从攻击者S的角度正式定义了三个基本性质，这有助于指导越狱方法的设计。我们进一步介绍了AutoBReach，这是一种新的越狱LLMS方法，只需要黑盒访问。受文字游戏的多样性启发，AutoBReach采用了文字游戏指导的映射规则采样策略，生成了各种通用的映射规则，用于创建对抗性提示。这一生成过程利用了LLMS的自动摘要和推理能力，从而减轻了手动负担。为了提高越狱成功率，我们进一步建议句子压缩和基于思想链的映射规则来纠正目标LLM中的错误和文字游戏误解。此外，我们还提出了一种两阶段映射规则优化策略，在查询目标LLM之前对映射规则进行初始优化，以提高AutoBReach的效率。AutoBReach可以高效地识别各种LLMS的安全漏洞，包括三种专有模型：Claude-3、GPT-3.5、GPT-4 Turbo，以及两种LLMS的Web平台：Bingchat、GPT-4 Web，平均成功率超过80%，查询次数不到10次



## **27. Evaluating the Effectiveness and Robustness of Visual Similarity-based Phishing Detection Models**

评估基于视觉相似性的网络钓鱼检测模型的有效性和稳健性 cs.CR

12 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19598v1) [paper-pdf](http://arxiv.org/pdf/2405.19598v1)

**Authors**: Fujiao Ji, Kiho Lee, Hyungjoon Koo, Wenhao You, Euijin Choo, Hyoungshick Kim, Doowon Kim

**Abstract**: Phishing attacks pose a significant threat to Internet users, with cybercriminals elaborately replicating the visual appearance of legitimate websites to deceive victims. Visual similarity-based detection systems have emerged as an effective countermeasure, but their effectiveness and robustness in real-world scenarios have been unexplored. In this paper, we comprehensively scrutinize and evaluate state-of-the-art visual similarity-based anti-phishing models using a large-scale dataset of 450K real-world phishing websites. Our analysis reveals that while certain models maintain high accuracy, others exhibit notably lower performance than results on curated datasets, highlighting the importance of real-world evaluation. In addition, we observe the real-world tactic of manipulating visual components that phishing attackers employ to circumvent the detection systems. To assess the resilience of existing models against adversarial attacks and robustness, we apply visible and perturbation-based manipulations to website logos, which adversaries typically target. We then evaluate the models' robustness in handling these adversarial samples. Our findings reveal vulnerabilities in several models, emphasizing the need for more robust visual similarity techniques capable of withstanding sophisticated evasion attempts. We provide actionable insights for enhancing the security of phishing defense systems, encouraging proactive actions. To the best of our knowledge, this work represents the first large-scale, systematic evaluation of visual similarity-based models for phishing detection in real-world settings, necessitating the development of more effective and robust defenses.

摘要: 网络钓鱼攻击对互联网用户构成重大威胁，网络犯罪分子精心复制合法网站的视觉外观来欺骗受害者。基于视觉相似性的检测系统已经成为一种有效的对策，但其在现实世界场景中的有效性和稳健性还没有得到探索。在本文中，我们使用450K真实网络钓鱼网站的大规模数据集，对基于视觉相似性的反网络钓鱼模型进行了全面的审查和评估。我们的分析表明，虽然某些模型保持了高精度，但其他模型的性能明显低于精选数据集的结果，这突显了现实世界评估的重要性。此外，我们还观察到了网络钓鱼攻击者用来绕过检测系统的操纵视觉组件的真实策略。为了评估现有模型对对手攻击的弹性和稳健性，我们对网站徽标应用了可见的和基于扰动的操纵，这是对手通常的目标。然后，我们评估了模型在处理这些对抗性样本时的稳健性。我们的发现揭示了几个模型中的漏洞，强调了需要更强大的视觉相似性技术，能够抵御复杂的逃避尝试。我们为增强网络钓鱼防御系统的安全性提供了可操作的见解，鼓励采取主动行动。据我们所知，这项工作是第一次大规模、系统地评估现实世界中基于视觉相似性的网络钓鱼检测模型，这就需要开发更有效和更强大的防御措施。



## **28. AI Risk Management Should Incorporate Both Safety and Security**

人工智能风险管理应兼顾安全与保障 cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19524v1) [paper-pdf](http://arxiv.org/pdf/2405.19524v1)

**Authors**: Xiangyu Qi, Yangsibo Huang, Yi Zeng, Edoardo Debenedetti, Jonas Geiping, Luxi He, Kaixuan Huang, Udari Madhushani, Vikash Sehwag, Weijia Shi, Boyi Wei, Tinghao Xie, Danqi Chen, Pin-Yu Chen, Jeffrey Ding, Ruoxi Jia, Jiaqi Ma, Arvind Narayanan, Weijie J Su, Mengdi Wang, Chaowei Xiao, Bo Li, Dawn Song, Peter Henderson, Prateek Mittal

**Abstract**: The exposure of security vulnerabilities in safety-aligned language models, e.g., susceptibility to adversarial attacks, has shed light on the intricate interplay between AI safety and AI security. Although the two disciplines now come together under the overarching goal of AI risk management, they have historically evolved separately, giving rise to differing perspectives. Therefore, in this paper, we advocate that stakeholders in AI risk management should be aware of the nuances, synergies, and interplay between safety and security, and unambiguously take into account the perspectives of both disciplines in order to devise mostly effective and holistic risk mitigation approaches. Unfortunately, this vision is often obfuscated, as the definitions of the basic concepts of "safety" and "security" themselves are often inconsistent and lack consensus across communities. With AI risk management being increasingly cross-disciplinary, this issue is particularly salient. In light of this conceptual challenge, we introduce a unified reference framework to clarify the differences and interplay between AI safety and AI security, aiming to facilitate a shared understanding and effective collaboration across communities.

摘要: 安全一致的语言模型中安全漏洞的暴露，例如对对手攻击的敏感性，揭示了人工智能安全和人工智能安全之间复杂的相互作用。尽管这两个学科现在在人工智能风险管理的总体目标下走到了一起，但它们在历史上是分开发展的，产生了不同的观点。因此，在本文中，我们主张人工智能风险管理的利益相关者应该意识到安全和安保之间的细微差别、协同效应和相互作用，并毫不含糊地考虑这两个学科的观点，以便设计出最有效和全面的风险缓解方法。不幸的是，这一愿景经常被混淆，因为“安全”和“安保”这两个基本概念本身的定义往往不一致，而且在各社区之间缺乏共识。随着人工智能风险管理日益跨学科，这个问题尤为突出。针对这一概念挑战，我们引入了一个统一的参考框架，以澄清人工智能安全和人工智能安全之间的差异和相互作用，旨在促进社区之间的共同理解和有效合作。



## **29. IOI: Invisible One-Iteration Adversarial Attack on No-Reference Image- and Video-Quality Metrics**

IOI：对无参考图像和视频质量收件箱的隐形单迭代对抗攻击 eess.IV

Accepted to ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2403.05955v2) [paper-pdf](http://arxiv.org/pdf/2403.05955v2)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: No-reference image- and video-quality metrics are widely used in video processing benchmarks. The robustness of learning-based metrics under video attacks has not been widely studied. In addition to having success, attacks that can be employed in video processing benchmarks must be fast and imperceptible. This paper introduces an Invisible One-Iteration (IOI) adversarial attack on no reference image and video quality metrics. We compared our method alongside eight prior approaches using image and video datasets via objective and subjective tests. Our method exhibited superior visual quality across various attacked metric architectures while maintaining comparable attack success and speed. We made the code available on GitHub: https://github.com/katiashh/ioi-attack.

摘要: 无参考图像和视频质量指标广泛用于视频处理基准测试中。基于学习的指标在视频攻击下的稳健性尚未得到广泛研究。除了取得成功之外，可以用于视频处理基准的攻击还必须快速且不可察觉。本文介绍了针对无参考图像和视频质量指标的隐形单迭代（IOI）对抗攻击。我们通过客观和主观测试将我们的方法与使用图像和视频数据集的八种先前方法进行了比较。我们的方法在各种受攻击的指标架构中表现出卓越的视觉质量，同时保持相当的攻击成功率和速度。我们在GitHub上提供了该代码：https://github.com/katiashh/ioi-attack。



## **30. Diffusion Policy Attacker: Crafting Adversarial Attacks for Diffusion-based Policies**

扩散政策攻击者：为基于扩散的政策设计对抗攻击 cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19424v1) [paper-pdf](http://arxiv.org/pdf/2405.19424v1)

**Authors**: Yipu Chen, Haotian Xue, Yongxin Chen

**Abstract**: Diffusion models (DMs) have emerged as a promising approach for behavior cloning (BC). Diffusion policies (DP) based on DMs have elevated BC performance to new heights, demonstrating robust efficacy across diverse tasks, coupled with their inherent flexibility and ease of implementation. Despite the increasing adoption of DP as a foundation for policy generation, the critical issue of safety remains largely unexplored. While previous attempts have targeted deep policy networks, DP used diffusion models as the policy network, making it ineffective to be attacked using previous methods because of its chained structure and randomness injected. In this paper, we undertake a comprehensive examination of DP safety concerns by introducing adversarial scenarios, encompassing offline and online attacks, and global and patch-based attacks. We propose DP-Attacker, a suite of algorithms that can craft effective adversarial attacks across all aforementioned scenarios. We conduct attacks on pre-trained diffusion policies across various manipulation tasks. Through extensive experiments, we demonstrate that DP-Attacker has the capability to significantly decrease the success rate of DP for all scenarios. Particularly in offline scenarios, DP-Attacker can generate highly transferable perturbations applicable to all frames. Furthermore, we illustrate the creation of adversarial physical patches that, when applied to the environment, effectively deceive the model. Video results are put in: https://sites.google.com/view/diffusion-policy-attacker.

摘要: 扩散模型(DM)已经成为行为克隆(BC)的一种有前途的方法。基于DM的扩散策略(DP)将BC性能提升到了新的高度，展示了跨不同任务的强大效率，以及其固有的灵活性和实施简便性。尽管越来越多地采用DP作为政策制定的基础，但安全这一关键问题在很大程度上仍未得到探索。虽然以前的尝试都是针对深层策略网络的，但是DP使用扩散模型作为策略网络，由于其链式结构和注入的随机性，使得使用以前的方法攻击是无效的。在本文中，我们通过引入对抗性场景，包括离线和在线攻击，以及全局攻击和基于补丁的攻击，对DP安全问题进行了全面的检查。我们提出了DP-攻击者，这是一套算法，可以在上述所有场景中创建有效的对抗性攻击。我们在各种操作任务中对预先训练的扩散策略进行攻击。通过大量的实验，我们证明了DP-攻击者在所有场景下都有能力显著降低DP的成功率。特别是在离线情况下，DP-攻击者可以生成适用于所有帧的高度可传递的扰动。此外，我们还说明了敌意物理补丁的创建，当应用到环境中时，这些补丁有效地欺骗了模型。视频结果放入：https://sites.google.com/view/diffusion-policy-attacker.



## **31. ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning**

ConceptPrune：通过熟练的神经元修剪在扩散模型中进行概念编辑 cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19237v1) [paper-pdf](http://arxiv.org/pdf/2405.19237v1)

**Authors**: Ruchika Chavhan, Da Li, Timothy Hospedales

**Abstract**: While large-scale text-to-image diffusion models have demonstrated impressive image-generation capabilities, there are significant concerns about their potential misuse for generating unsafe content, violating copyright, and perpetuating societal biases. Recently, the text-to-image generation community has begun addressing these concerns by editing or unlearning undesired concepts from pre-trained models. However, these methods often involve data-intensive and inefficient fine-tuning or utilize various forms of token remapping, rendering them susceptible to adversarial jailbreaks. In this paper, we present a simple and effective training-free approach, ConceptPrune, wherein we first identify critical regions within pre-trained models responsible for generating undesirable concepts, thereby facilitating straightforward concept unlearning via weight pruning. Experiments across a range of concepts including artistic styles, nudity, object erasure, and gender debiasing demonstrate that target concepts can be efficiently erased by pruning a tiny fraction, approximately 0.12% of total weights, enabling multi-concept erasure and robustness against various white-box and black-box adversarial attacks.

摘要: 虽然大规模的文本到图像传播模型显示了令人印象深刻的图像生成能力，但人们非常担心它们可能被滥用来生成不安全的内容、侵犯版权和永久存在社会偏见。最近，文本到图像生成社区已经开始通过编辑或不学习预先训练的模型中不需要的概念来解决这些问题。然而，这些方法往往涉及数据密集型和低效的微调或利用各种形式的令牌重新映射，使得它们容易受到对抗性越狱的影响。在本文中，我们提出了一种简单而有效的免训练方法ConceptPrune，其中我们首先在预先训练的模型中识别负责产生不想要的概念的关键区域，从而通过权重剪枝来促进直接的概念遗忘。在艺术风格、裸体、对象擦除和性别去偏向等一系列概念上的实验表明，目标概念可以通过修剪极小的部分(约占总权重的0.12%)来有效擦除，从而实现多概念擦除和对各种白盒和黑盒对抗攻击的健壮性。



## **32. Gone but Not Forgotten: Improved Benchmarks for Machine Unlearning**

消失但未被遗忘：机器取消学习的改进基准 cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19211v1) [paper-pdf](http://arxiv.org/pdf/2405.19211v1)

**Authors**: Keltin Grimes, Collin Abidi, Cole Frank, Shannon Gallagher

**Abstract**: Machine learning models are vulnerable to adversarial attacks, including attacks that leak information about the model's training data. There has recently been an increase in interest about how to best address privacy concerns, especially in the presence of data-removal requests. Machine unlearning algorithms aim to efficiently update trained models to comply with data deletion requests while maintaining performance and without having to resort to retraining the model from scratch, a costly endeavor. Several algorithms in the machine unlearning literature demonstrate some level of privacy gains, but they are often evaluated only on rudimentary membership inference attacks, which do not represent realistic threats. In this paper we describe and propose alternative evaluation methods for three key shortcomings in the current evaluation of unlearning algorithms. We show the utility of our alternative evaluations via a series of experiments of state-of-the-art unlearning algorithms on different computer vision datasets, presenting a more detailed picture of the state of the field.

摘要: 机器学习模型容易受到敌意攻击，包括泄露模型训练数据信息的攻击。最近，人们对如何最好地解决隐私问题的兴趣有所增加，特别是在存在数据删除请求的情况下。机器遗忘算法的目标是高效地更新训练好的模型，以符合数据删除请求，同时保持性能，而不必求助于从头开始重新训练模型，这是一项代价高昂的工作。机器遗忘文献中的几个算法展示了一定程度的隐私收益，但它们通常只在基本的成员关系推理攻击上进行评估，而这些攻击并不代表现实的威胁。本文针对当前遗忘算法评价中的三个关键缺陷，描述并提出了可供选择的评价方法。我们通过在不同的计算机视觉数据集上进行一系列最先进的遗忘算法的实验，展示了我们的替代评估的实用性，呈现了该领域状态的更详细的图景。



## **33. Model Agnostic Defense against Adversarial Patch Attacks on Object Detection in Unmanned Aerial Vehicles**

无人机目标检测对抗补丁攻击的模型不可知防御 cs.CV

submitted to IROS 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19179v1) [paper-pdf](http://arxiv.org/pdf/2405.19179v1)

**Authors**: Saurabh Pathak, Samridha Shrestha, Abdelrahman AlMahmoud

**Abstract**: Object detection forms a key component in Unmanned Aerial Vehicles (UAVs) for completing high-level tasks that depend on the awareness of objects on the ground from an aerial perspective. In that scenario, adversarial patch attacks on an onboard object detector can severely impair the performance of upstream tasks. This paper proposes a novel model-agnostic defense mechanism against the threat of adversarial patch attacks in the context of UAV-based object detection. We formulate adversarial patch defense as an occlusion removal task. The proposed defense method can neutralize adversarial patches located on objects of interest, without exposure to adversarial patches during training. Our lightweight single-stage defense approach allows us to maintain a model-agnostic nature, that once deployed does not require to be updated in response to changes in the object detection pipeline. The evaluations in digital and physical domains show the feasibility of our method for deployment in UAV object detection pipelines, by significantly decreasing the Attack Success Ratio without incurring significant processing costs. As a result, the proposed defense solution can improve the reliability of object detection for UAVs.

摘要: 目标检测是无人机(UAV)完成高层次任务的关键组成部分，它依赖于从空中角度对地面目标的感知。在这种情况下，对机载对象探测器的敌意补丁攻击可能会严重影响上游任务的性能。针对无人机目标检测中的敌意补丁攻击威胁，提出了一种新的模型不可知防御机制。我们将对抗性补丁防御定义为一种遮挡消除任务。所提出的防御方法可以中和位于感兴趣对象上的对抗性补丁，而不会在训练期间暴露于对抗性补丁。我们的轻量级单级防御方法允许我们保持与模型无关的性质，一旦部署，就不需要更新以响应对象检测管道的变化。数字和物理领域的评估表明，该方法在不产生显著处理成本的情况下，显著降低了攻击成功率，从而在无人机目标检测管道中部署的可行性。结果表明，所提出的防御方案能够提高无人机目标检测的可靠性。



## **34. Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness**

引入自适应持续对抗训练（ACAT）以增强ML稳健性 cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2403.10461v2) [paper-pdf](http://arxiv.org/pdf/2403.10461v2)

**Authors**: Mohamed elShehaby, Aditya Kotha, Ashraf Matrawy

**Abstract**: Adversarial training enhances the robustness of Machine Learning (ML) models against adversarial attacks. However, obtaining labeled training and adversarial training data in network/cybersecurity domains is challenging and costly. Therefore, this letter introduces Adaptive Continuous Adversarial Training (ACAT), a method that integrates adversarial training samples into the model during continuous learning sessions using real-world detected adversarial data. Experimental results with a SPAM detection dataset demonstrate that ACAT reduces the time required for adversarial sample detection compared to traditional processes. Moreover, the accuracy of the under-attack ML-based SPAM filter increased from 69% to over 88% after just three retraining sessions.

摘要: 对抗性训练增强了机器学习（ML）模型对抗对抗性攻击的鲁棒性。然而，在网络/网络安全领域中获得标记训练和对抗训练数据具有挑战性且成本高昂。因此，这封信引入了自适应连续对抗训练（ACAT），这是一种使用现实世界检测到的对抗数据在连续学习会话期间将对抗训练样本集成到模型中的方法。SPAM检测数据集的实验结果表明，与传统过程相比，ACAT减少了对抗性样本检测所需的时间。此外，仅经过三次再培训后，受攻击的基于ML的垃圾邮件过滤器的准确性就从69%提高到了88%以上。



## **35. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior**

通过函数先验引导的Bayesian优化进行高效的黑匣子对抗攻击 cs.LG

ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19098v1) [paper-pdf](http://arxiv.org/pdf/2405.19098v1)

**Authors**: Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, Jun Zhu

**Abstract**: This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.

摘要: 研究了一种具有挑战性的黑盒对抗性攻击，其目的是通过只使用模型的输出反馈来输入查询来生成针对黑盒模型的对抗性实例。以前的一些方法通过将代理白盒模型的梯度融入到基于查询的攻击中来提高查询效率，这是因为攻击具有对抗性。然而，局部化的梯度信息不足，使得这些方法仍然是查询密集型的。本文提出了一种先验引导的贝叶斯优化算法(P-BO)，该算法利用代理模型作为黑盒对抗攻击的全局先验函数。由于代理模型包含了丰富的黑盒模型的先验信息，P-BO用一个高斯过程对攻击目标进行建模，其均值函数被初始化为代理模型的损失。我们对遗憾界的理论分析表明，坏的先验可能会影响P-BO的性能。因此，我们进一步提出了一种自适应积分策略，通过最小化遗憾界来自动调整函数先验上的系数。在图像分类器和大型视觉语言模型上的大量实验表明，与最先进的黑盒攻击相比，该算法在减少查询和提高攻击成功率方面具有优势。代码可在https://github.com/yibo-miao/PBO-Attack.上找到



## **36. New perspectives on the optimal placement of detectors for suicide bombers using metaheuristics**

使用元启发法研究自杀式炸弹袭击者探测器最佳放置的新观点 cs.NE

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19060v1) [paper-pdf](http://arxiv.org/pdf/2405.19060v1)

**Authors**: Carlos Cotta, José E. Gallardo

**Abstract**: We consider an operational model of suicide bombing attacks -- an increasingly prevalent form of terrorism -- against specific targets, and the use of protective countermeasures based on the deployment of detectors over the area under threat. These detectors have to be carefully located in order to minimize the expected number of casualties or the economic damage suffered, resulting in a hard optimization problem for which different metaheuristics have been proposed. Rather than assuming random decisions by the attacker, the problem is approached by considering different models of the latter, whereby he takes informed decisions on which objective must be targeted and through which path it has to be reached based on knowledge on the importance or value of the objectives or on the defensive strategy of the defender (a scenario that can be regarded as an adversarial game). We consider four different algorithms, namely a greedy heuristic, a hill climber, tabu search and an evolutionary algorithm, and study their performance on a broad collection of problem instances trying to resemble different realistic settings such as a coastal area, a modern urban area, and the historic core of an old town. It is shown that the adversarial scenario is harder for all techniques, and that the evolutionary algorithm seems to adapt better to the complexity of the resulting search landscape.

摘要: 我们考虑一种针对具体目标的自杀式爆炸袭击--一种日益普遍的恐怖主义形式--的行动模式，以及在受威胁地区部署探测器的基础上使用保护性对策。这些探测器必须小心地放置，以使预期的伤亡人数或遭受的经济损失最小化，这导致了一个困难的优化问题，对于这个问题，已经提出了不同的元启发式算法。不是假定攻击者的随机决定，而是通过考虑后者的不同模型来处理问题，根据对目标的重要性或价值的了解或防守者的防守策略(可以被视为对抗性游戏的场景)，他根据对目标的重要性或价值的了解，做出关于哪个目标必须成为目标以及必须通过哪条路径到达的明智决定。我们考虑了四种不同的算法，即贪婪启发式算法、爬山者算法、禁忌搜索算法和进化算法，并研究了它们在大量问题实例上的性能，这些问题实例试图类似于不同的现实环境，如沿海地区、现代城市地区和古镇的历史核心。结果表明，对于所有技术来说，对抗性场景都更加困难，而进化算法似乎更好地适应了结果搜索环境的复杂性。



## **37. Verifiably Robust Conformal Prediction**

可验证鲁棒性保形预测 cs.LO

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18942v1) [paper-pdf](http://arxiv.org/pdf/2405.18942v1)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces \emph{VRCP (Verifiably Robust Conformal Prediction)}, a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.

摘要: 保角预测是一种流行的不确定性量化方法，它假设训练和测试数据是可交换的，提供了无分布的、统计上有效的预测集。在这种情况下，CP的预测集保证以用户指定的概率覆盖(未知)真实测试输出。然而，当数据受到对抗性攻击时，这一保证就会被违反，这往往会导致覆盖范围的重大损失。最近，已经提出了几种在这种情况下恢复CP担保的方法。这些方法利用随机平滑的变化来产生保守集合，这些保守集合考虑了对抗性扰动的影响。然而，它们的局限性在于它们只支持$^2$有界的扰动和分类任务。介绍了一种新的基于神经网络验证方法的抗攻击覆盖恢复框架--可验证稳健共形预测(VRCP)。我们的VRCP方法是第一个支持以任意范数为界的扰动，包括$^1$，$^2$，$^inty$，以及回归任务。我们在深度强化学习环境下的图像分类任务(CIFAR10、CIFAR100和TinyImageNet)和回归任务上对我们的方法进行了评估和比较。在任何情况下，VRCP都达到了名义覆盖率以上，并产生了比SOTA更有效和更有信息量的预测区域。



## **38. Proactive Load-Shaping Strategies with Privacy-Cost Trade-offs in Residential Households based on Deep Reinforcement Learning**

基于深度强化学习的住宅家庭具有隐私成本权衡的主动负载塑造策略 eess.SY

7 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18888v1) [paper-pdf](http://arxiv.org/pdf/2405.18888v1)

**Authors**: Ruichang Zhang, Youcheng Sun, Mustafa A. Mustafa

**Abstract**: Smart meters play a crucial role in enhancing energy management and efficiency, but they raise significant privacy concerns by potentially revealing detailed user behaviors through energy consumption patterns. Recent scholarly efforts have focused on developing battery-aided load-shaping techniques to protect user privacy while balancing costs. This paper proposes a novel deep reinforcement learning-based load-shaping algorithm (PLS-DQN) designed to protect user privacy by proactively creating artificial load signatures that mislead potential attackers. We evaluate our proposed algorithm against a non-intrusive load monitoring (NILM) adversary. The results demonstrate that our approach not only effectively conceals real energy usage patterns but also outperforms state-of-the-art methods in enhancing user privacy while maintaining cost efficiency.

摘要: 智能电表在提高能源管理和效率方面发挥着至关重要的作用，但它们可能通过能源消耗模式揭示详细的用户行为，从而引发了严重的隐私问题。最近的学术工作重点是开发电池辅助负载整形技术，以保护用户隐私，同时平衡成本。本文提出了一种新型的基于深度强化学习的负载整形算法（PLS-DQN），旨在通过主动创建误导潜在攻击者的人工负载签名来保护用户隐私。我们针对非侵入性负载监控（NILM）对手评估了我们提出的算法。结果表明，我们的方法不仅有效地隐藏了真实的能源使用模式，而且在增强用户隐私的同时保持成本效率方面优于最先进的方法。



## **39. Enhancing Security and Privacy in Federated Learning using Update Digests and Voting-Based Defense**

使用更新摘要和基于投票的防御增强联邦学习中的安全性和隐私 cs.CR

14 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18802v1) [paper-pdf](http://arxiv.org/pdf/2405.18802v1)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, especially in the presence of curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{\textbf{F}}ederated \underline{\textbf{L}}earning with \underline{\textbf{U}}pdate \underline{\textbf{D}}igest (FLUD), which addresses the critical issues of privacy preservation and resistance to Byzantine attacks within distributed learning environments. FLUD utilizes an innovative approach, the $\mathsf{LinfSample}$ method, allowing clients to compute the $l_{\infty}$ norm across sliding windows of updates as an update digest. This digest enables the server to calculate a shared distance matrix, significantly reducing the overhead associated with Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and malicious updates. Additionally, FLUD integrates a privacy-preserving, voting-based defense mechanism that employs optimized SMPC protocols to minimize communication rounds. Our comprehensive experiments demonstrate FLUD's effectiveness in countering Byzantine adversaries while incurring low communication and runtime overhead. FLUD offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.

摘要: 联合学习(FL)是一种很有前途的隐私保护机器学习范例，允许数据所有者在保持数据本地化的同时协作训练模型。尽管有潜力，FL仍面临着与客户端和服务器的可信性相关的挑战，特别是在存在好奇或恶意对手的情况下。针对分布式学习环境中隐私保护和抵抗拜占庭攻击的关键问题，本文提出了一种新的框架-.Flud使用了一种创新的方法，即$\mathsf{LinfSample}$方法，允许客户端跨滑动更新窗口计算$L_{\infty}$范数作为更新摘要。此摘要使服务器能够计算共享距离矩阵，从而将与安全多方计算(SMPC)相关的开销显著降低三个数量级，同时有效区分良性更新和恶意更新。此外，Flud集成了隐私保护、基于投票的防御机制，该机制采用优化的SMPC协议来最大限度地减少通信轮次。我们的综合实验证明了Flud在对抗拜占庭对手方面的有效性，同时产生了较低的通信和运行时间开销。Flud为分布式环境中的安全可靠FL提供了一个可扩展的框架，促进了其在需要强大的数据管理和安全的场景中的应用。



## **40. MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training**

MIST：通过授权不变子空间训练防御成员推断攻击 cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2311.00919v2) [paper-pdf](http://arxiv.org/pdf/2311.00919v2)

**Authors**: Jiacheng Li, Ninghui Li, Bruno Ribeiro

**Abstract**: In Member Inference (MI) attacks, the adversary try to determine whether an instance is used to train a machine learning (ML) model. MI attacks are a major privacy concern when using private data to train ML models. Most MI attacks in the literature take advantage of the fact that ML models are trained to fit the training data well, and thus have very low loss on training instances. Most defenses against MI attacks therefore try to make the model fit the training data less well. Doing so, however, generally results in lower accuracy. We observe that training instances have different degrees of vulnerability to MI attacks. Most instances will have low loss even when not included in training. For these instances, the model can fit them well without concerns of MI attacks. An effective defense only needs to (possibly implicitly) identify instances that are vulnerable to MI attacks and avoids overfitting them. A major challenge is how to achieve such an effect in an efficient training process. Leveraging two distinct recent advancements in representation learning: counterfactually-invariant representations and subspace learning methods, we introduce a novel Membership-Invariant Subspace Training (MIST) method to defend against MI attacks. MIST avoids overfitting the vulnerable instances without significant impact on other instances. We have conducted extensive experimental studies, comparing MIST with various other state-of-the-art (SOTA) MI defenses against several SOTA MI attacks. We find that MIST outperforms other defenses while resulting in minimal reduction in testing accuracy.

摘要: 在成员推理(MI)攻击中，对手试图确定是否使用实例来训练机器学习(ML)模型。在使用私有数据训练ML模型时，MI攻击是一个主要的隐私问题。文献中的大多数MI攻击都利用了ML模型经过训练以很好地拟合训练数据的事实，因此在训练实例上的损失非常低。因此，大多数针对MI攻击的防御措施都试图使模型不太适合训练数据。然而，这样做通常会导致精度降低。我们观察到，训练实例对MI攻击具有不同程度的脆弱性。即使不包括在培训中，大多数实例的损失也很低。对于这些实例，该模型可以很好地对它们进行拟合，而无需担心MI攻击。有效的防御只需要(可能是隐式地)识别易受MI攻击的实例，并避免过度匹配它们。一个主要的挑战是如何在有效的培训过程中达到这样的效果。利用表示学习的两个不同的最新进展：反事实不变表示和子空间学习方法，我们引入了一种新的成员不变子空间训练(MIST)方法来防御MI攻击。MIST避免对易受攻击的实例过度拟合，而不会对其他实例产生重大影响。我们进行了广泛的实验研究，将MIST与其他各种最先进的(SOTA)MI防御系统进行了比较，以抵御几种SOTA MI攻击。我们发现，MIST的性能优于其他防御系统，同时对测试精度的影响也很小。



## **41. Leveraging Many-To-Many Relationships for Defending Against Visual-Language Adversarial Attacks**

利用多对多关系防御视觉语言对抗攻击 cs.CV

Under review

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18770v1) [paper-pdf](http://arxiv.org/pdf/2405.18770v1)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos

**Abstract**: Recent studies have revealed that vision-language (VL) models are vulnerable to adversarial attacks for image-text retrieval (ITR). However, existing defense strategies for VL models primarily focus on zero-shot image classification, which do not consider the simultaneous manipulation of image and text, as well as the inherent many-to-many (N:N) nature of ITR, where a single image can be described in numerous ways, and vice versa. To this end, this paper studies defense strategies against adversarial attacks on VL models for ITR for the first time. Particularly, we focus on how to leverage the N:N relationship in ITR to enhance adversarial robustness. We found that, although adversarial training easily overfits to specific one-to-one (1:1) image-text pairs in the train data, diverse augmentation techniques to create one-to-many (1:N) / many-to-one (N:1) image-text pairs can significantly improve adversarial robustness in VL models. Additionally, we show that the alignment of the augmented image-text pairs is crucial for the effectiveness of the defense strategy, and that inappropriate augmentations can even degrade the model's performance. Based on these findings, we propose a novel defense strategy that leverages the N:N relationship in ITR, which effectively generates diverse yet highly-aligned N:N pairs using basic augmentations and generative model-based augmentations. This work provides a novel perspective on defending against adversarial attacks in VL tasks and opens up new research directions for future work.

摘要: 最近的研究表明，视觉语言(VL)模型容易受到图像文本检索(ITR)的敌意攻击。然而，现有的VL模型防御策略主要集中在零镜头图像分类，没有考虑图像和文本的同时操作，以及ITR固有的多对多(N：N)性质，其中单个图像可以以多种方式描述，反之亦然。为此，本文首次研究了针对ITR VL模型的对抗攻击防御策略。特别是，我们关注如何利用ITR中的N：N关系来增强对手的健壮性。我们发现，尽管对抗性训练很容易超过训练数据中特定的一对一(1：1)图文对，但创建一对多(1：N)/多对一(N：1)图文对的各种增强技术可以显著提高VL模型中的对抗性健壮性。此外，我们还证明了增强图文对的对齐对于防御策略的有效性是至关重要的，并且不适当的增强甚至会降低模型的性能。基于这些发现，我们提出了一种新的防御策略，该策略利用ITR中的N：N关系，使用基本增强和基于生成性模型的增强有效地生成各种但高度一致的N：N对。这项工作为虚拟学习任务中对抗攻击的防御提供了一个新的视角，并为未来的工作开辟了新的研究方向。



## **42. Genshin: General Shield for Natural Language Processing with Large Language Models**

Genshin：具有大型语言模型的自然语言处理的通用盾牌 cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18741v1) [paper-pdf](http://arxiv.org/pdf/2405.18741v1)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.

摘要: 像ChatGPT、Gemini或Llama这样的大型语言模型(LLM)最近已经成为趋势，在无数领域展示了相当大的先进性和泛化能力。然而，LLM创建了一个更大的黑匣子，加剧了不透明度，可解释性仅限于几种方法。LLMS本质上的不确定性和不透明性限制了它们在高风险领域的应用，如金融欺诈、网络钓鱼等。目前的方法主要依赖于传统的文本分类和后验可解释算法，攻击者可能会创建通用的对抗性样本来破坏系统的防御，迫使用户在效率和健壮性之间做出权衡。为了解决这个问题，我们提出了一种新颖的级联框架Genshin(General Shield For Natural Language Processing With Large Language Models)，利用LLMS作为防御性的一次性插件。与大多数试图将文本转换为新的或结构化的文本的LLMS应用程序不同，Genshin使用LLMS将文本恢复到其原始状态。Genshin的目标是将LLM的泛化能力、中值模型的区分性和简单模型的可解释性结合起来。我们在情感分析和垃圾邮件检测任务上的实验表明，现有的中值模型存在致命缺陷，并且在LLMS的恢复能力上取得了令人振奋的结果，证明了Genshin是有效的和高效的。在我们的消融研究中，我们发现了几个有趣的观察结果。利用LLM Defender，一个源自第四范式的工具，我们在NLP的第三范式中复制了Bert的15%最优掩蔽率结果。此外，当使用LLM作为潜在的敌意工具时，攻击者能够执行几乎在语义上无损的有效攻击。



## **43. Security--Throughput Tradeoff of Nakamoto Consensus under Bandwidth Constraints**

安全性--带宽限制下中本共识的短期权衡 cs.CR

ACM Conference on Computer and Communications Security 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2303.09113v3) [paper-pdf](http://arxiv.org/pdf/2303.09113v3)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: For Nakamoto's longest-chain consensus protocol, whose proof-of-work (PoW) and proof-of-stake (PoS) variants power major blockchains such as Bitcoin and Cardano, we revisit the classic problem of the security-performance tradeoff: Given a network of nodes with limited capacities, against what fraction of adversary power is Nakamoto consensus (NC) secure for a given block production rate? State-of-the-art analyses of Nakamoto's protocol fail to answer this question because their bounded-delay model does not capture realistic constraints such as limited communication- and computation-resources. We develop a new analysis technique to prove a refined security-performance tradeoff for PoW Nakamoto consensus in a bounded-bandwidth model. In this model, we show that, in contrast to the classic bounded-delay model, Nakamoto's private attack is no longer the worst attack, and a new attack strategy we call the teasing strategy, that exploits the network congestion caused by limited bandwidth, is strictly worse. In PoS, equivocating blocks can exacerbate congestion, making the traditional PoS Nakamoto consensus protocol insecure except at very low block production rates. To counter such equivocation spamming, we present a variant of the PoS NC protocol we call Blanking NC (BlaNC), which achieves the same resilience as PoW NC.

摘要: 对于Nakamoto的最长链共识协议，其工作证明(PoW)和风险证明(Pos)变体为比特币和Cardano等主要区块链提供支持，我们重温了安全与性能权衡的经典问题：给定一个容量有限的节点网络，对于给定的块生产率，相对于对手力量的多少部分，Nakamoto共识(NC)是安全的？对Nakamoto协议的最新分析未能回答这个问题，因为他们的有限延迟模型没有捕捉到现实的约束，如有限的通信和计算资源。我们开发了一种新的分析技术来证明在有限带宽模型中PoW Nakamoto共识的改进的安全与性能权衡。在该模型中，我们证明了与经典的有限延迟模型相比，Nakamoto的私有攻击不再是最糟糕的攻击，而一种新的攻击策略--利用有限带宽造成的网络拥塞--严格地更差。在PoS中，模棱两可的块会加剧拥塞，使得传统的PoS Nakamoto共识协议不安全，除非在非常低的块生产率下。为了应对这种模棱两可的垃圾邮件，我们提出了一种POS NC协议的变体，我们称之为BLANC(BLANC)，它实现了与POW NC相同的弹性。



## **44. Watermarking Counterfactual Explanations**

水印反事实解释 cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18671v1) [paper-pdf](http://arxiv.org/pdf/2405.18671v1)

**Authors**: Hangzhi Guo, Amulya Yadav

**Abstract**: The field of Explainable Artificial Intelligence (XAI) focuses on techniques for providing explanations to end-users about the decision-making processes that underlie modern-day machine learning (ML) models. Within the vast universe of XAI techniques, counterfactual (CF) explanations are often preferred by end-users as they help explain the predictions of ML models by providing an easy-to-understand & actionable recourse (or contrastive) case to individual end-users who are adversely impacted by predicted outcomes. However, recent studies have shown significant security concerns with using CF explanations in real-world applications; in particular, malicious adversaries can exploit CF explanations to perform query-efficient model extraction attacks on proprietary ML models. In this paper, we propose a model-agnostic watermarking framework (for adding watermarks to CF explanations) that can be leveraged to detect unauthorized model extraction attacks (which rely on the watermarked CF explanations). Our novel framework solves a bi-level optimization problem to embed an indistinguishable watermark into the generated CF explanation such that any future model extraction attacks that rely on these watermarked CF explanations can be detected using a null hypothesis significance testing (NHST) scheme, while ensuring that these embedded watermarks do not compromise the quality of the generated CF explanations. We evaluate this framework's performance across a diverse set of real-world datasets, CF explanation methods, and model extraction techniques, and show that our watermarking detection system can be used to accurately identify extracted ML models that are trained using the watermarked CF explanations. Our work paves the way for the secure adoption of CF explanations in real-world applications.

摘要: 可解释人工智能(XAI)领域的重点是向最终用户提供关于现代机器学习(ML)模型基础的决策过程的解释的技术。在XAI技术的浩瀚宇宙中，反事实(CF)解释往往受到最终用户的青睐，因为它们通过向受到预测结果不利影响的个人最终用户提供易于理解和可操作的资源(或对比)案例来帮助解释ML模型的预测。然而，最近的研究表明，在现实世界的应用程序中使用CF解释存在严重的安全问题；特别是，恶意攻击者可以利用CF解释对专有ML模型执行查询高效的模型提取攻击。在本文中，我们提出了一个模型无关的水印框架(用于在CF解释中添加水印)，该框架可用于检测未经授权的模型提取攻击(依赖于带水印的CF解释)。我们的新框架解决了一个双层优化问题，将不可区分的水印嵌入到生成的CF解释中，使得依赖于这些带水印的CF解释的任何未来模型提取攻击可以使用零假设显著性检验(NHST)方案来检测，同时确保这些嵌入的水印不会损害生成的CF解释的质量。我们在一组不同的真实数据集、CF解释方法和模型提取技术上对该框架的性能进行了评估，并表明我们的水印检测系统可以用于准确识别提取的ML模型，这些模型是使用带水印的CF解释训练的。我们的工作为在实际应用中安全地采用CF解释铺平了道路。



## **45. PureEBM: Universal Poison Purification via Mid-Run Dynamics of Energy-Based Models**

PureEBM：通过基于能量的模型的中期动力学进行通用毒物净化 cs.LG

arXiv admin note: substantial text overlap with arXiv:2405.18627

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.19376v1) [paper-pdf](http://arxiv.org/pdf/2405.19376v1)

**Authors**: Omead Pooladzandi, Jeffrey Jiang, Sunay Bhat, Gregory Pottie

**Abstract**: Data poisoning attacks pose a significant threat to the integrity of machine learning models by leading to misclassification of target distribution test data by injecting adversarial examples during training. Existing state-of-the-art (SoTA) defense methods suffer from a variety of limitations, such as significantly reduced generalization performance, specificity to particular attack types and classifiers, and significant overhead during training, making them impractical or limited for real-world applications. In response to this challenge, we introduce a universal data purification method that defends naturally trained classifiers from malicious white-, gray-, and black-box image poisons by applying a universal stochastic preprocessing step $\Psi_{T}(x)$, realized by iterative Langevin sampling of a convergent Energy Based Model (EBM) initialized with an image $x.$ Mid-run dynamics of $\Psi_{T}(x)$ purify poison information with minimal impact on features important to the generalization of a classifier network. We show that the contrastive learning process of EBMs allows them to remain universal purifiers, even in the presence of poisoned EBM training data, and to achieve SoTA defense on leading triggered poison Narcissus and triggerless poisons Gradient Matching and Bullseye Polytope. This work is a subset of a larger framework introduced in PureGen with a more detailed focus on EBM purification and poison defense.

摘要: 数据中毒攻击通过在训练过程中注入对抗性的例子，导致目标分布测试数据的错误分类，对机器学习模型的完整性构成了严重威胁。现有的最先进的(SOTA)防御方法受到各种限制，例如显著降低的泛化性能，对特定攻击类型和分类器的特异性，以及训练过程中的巨大开销，使得它们不切实际或限制了现实世界的应用。为了应对这一挑战，我们引入了一种通用的数据净化方法，通过应用通用随机预处理步骤$\psi_{T}(X)$来保护自然训练的分类器免受白盒、灰盒和黑盒图像的毒害，该步骤通过对以图像$x初始化的基于收敛能量的模型(EBM)的迭代朗之万式采样来实现。$\psi_{T}(X)$的中期运行动态$\psi_{T}(X)$净化有毒信息，同时最大限度地减少对分类器网络推广重要特征的影响。我们证明了EBM的对比学习过程使它们即使在有毒的EBM训练数据存在的情况下也能保持通用的净化器，并实现了对领先的触发毒物水仙和无触发毒物的梯度匹配和Bullseye多态的SOTA防御。这项工作是PureGen中引入的更大框架的子集，更详细地关注EBM纯化和毒物防御。



## **46. PureGen: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics**

PureGen：通过生成模型动力学进行训练时毒物防御的通用数据净化 cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18627v1) [paper-pdf](http://arxiv.org/pdf/2405.18627v1)

**Authors**: Sunay Bhat, Jeffrey Jiang, Omead Pooladzandi, Alexander Branch, Gregory Pottie

**Abstract**: Train-time data poisoning attacks threaten machine learning models by introducing adversarial examples during training, leading to misclassification. Current defense methods often reduce generalization performance, are attack-specific, and impose significant training overhead. To address this, we introduce a set of universal data purification methods using a stochastic transform, $\Psi(x)$, realized via iterative Langevin dynamics of Energy-Based Models (EBMs), Denoising Diffusion Probabilistic Models (DDPMs), or both. These approaches purify poisoned data with minimal impact on classifier generalization. Our specially trained EBMs and DDPMs provide state-of-the-art defense against various attacks (including Narcissus, Bullseye Polytope, Gradient Matching) on CIFAR-10, Tiny-ImageNet, and CINIC-10, without needing attack or classifier-specific information. We discuss performance trade-offs and show that our methods remain highly effective even with poisoned or distributionally shifted generative model training data.

摘要: 训练时间数据中毒攻击通过在训练过程中引入对抗性示例来威胁机器学习模型，导致错误分类。当前的防御方法通常会降低泛化性能，针对特定攻击，并且会带来显著的训练开销。为了解决这个问题，我们介绍了一套通用的数据净化方法，它使用随机变换$\Psi(X)$，通过基于能量的模型的迭代朗之万动力学(EBMS)或去噪扩散概率模型(DDPM)实现，或者两者兼而有之。这些方法净化有毒数据，对分类器泛化的影响最小。我们经过专门训练的EBM和DDPM提供针对CIFAR-10、Tiny-ImageNet和CINIC-10的各种攻击(包括水仙攻击、Bullseye多面体攻击、梯度匹配攻击)的最先进防御，而不需要攻击或特定于分类器的信息。我们讨论了性能权衡，并表明我们的方法仍然非常有效，即使在有毒或分布转移的生成性模型训练数据的情况下。



## **47. Wavelet-Based Image Tokenizer for Vision Transformers**

用于视觉变形者的基于微波的图像代币化器 cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18616v1) [paper-pdf](http://arxiv.org/pdf/2405.18616v1)

**Authors**: Zhenhai Zhu, Radu Soricut

**Abstract**: Non-overlapping patch-wise convolution is the default image tokenizer for all state-of-the-art vision Transformer (ViT) models. Even though many ViT variants have been proposed to improve its efficiency and accuracy, little research on improving the image tokenizer itself has been reported in the literature. In this paper, we propose a new image tokenizer based on wavelet transformation. We show that ViT models with the new tokenizer achieve both higher training throughput and better top-1 precision for the ImageNet validation set. We present a theoretical analysis on why the proposed tokenizer improves the training throughput without any change to ViT model architecture. Our analysis suggests that the new tokenizer can effectively handle high-resolution images and is naturally resistant to adversarial attack. Furthermore, the proposed image tokenizer offers a fresh perspective on important new research directions for ViT-based model design, such as image tokens on a non-uniform grid for image understanding.

摘要: 非重叠的面片卷积是所有最先进的视觉转换器(VIT)模型的默认图像标记器。尽管已经提出了许多VIT变体来提高其效率和准确性，但文献中关于改进图像标记器本身的研究很少。提出了一种新的基于小波变换的图像标记器。我们表明，使用新的标记器的VIT模型在ImageNet验证集上实现了更高的训练吞吐量和更好的TOP-1精度。我们从理论上分析了为什么在不改变VIT模型结构的情况下，所提出的记号器提高了训练吞吐量。我们的分析表明，新的标记器可以有效地处理高分辨率图像，并且具有天然的抵抗对手攻击的能力。此外，所提出的图像标记器为基于VIT的模型设计提供了新的研究方向，例如用于图像理解的非均匀网格上的图像标记物。



## **48. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

潘多拉的白盒：大型语言模型中的精确训练数据检测和提取 cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.17012v2) [paper-pdf](http://arxiv.org/pdf/2402.17012v2)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model which leverages recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Taken together, these results represent the strongest existing privacy attacks against both pretrained and fine-tuned LLMs for MIAs and training data extraction, which are of independent scientific interest and have important practical implications for LLM security, privacy, and copyright issues.

摘要: 在本文中，我们开发了针对大型语言模型(LLM)的最先进的隐私攻击，其中对该模型具有一定访问权限的对手试图了解一些关于潜在训练数据的信息。我们的主要结果是针对预先训练的LLM的新成员推理攻击(MIA)，其性能比基线攻击高数百倍，并且管道显示超过50%(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑了不同程度的访问底层模型、预训练和微调数据，以及MIA和训练数据提取。对于预训练数据，我们提出了两个新的MIA：一个有监督的神经网络分类器，它基于(降维)模型梯度来预测训练数据的成员资格，以及这种攻击的一个变体，它只需要Logit访问模型，利用了最近在LLMS上的模型窃取工作。据我们所知，这是第一个明确纳入模型窃取信息的MIA。这两种攻击都超过了现有的黑盒基线，我们的监督攻击缩小了针对LLMS的MIA攻击成功与针对其他机器学习模型的已知最强攻击之间的差距。在微调中，我们发现基于基本模型和微调模型之间的损失比率的简单攻击能够获得近乎完美的MIA性能；然后，我们利用我们的MIA从微调的Pythia和Llama模型中提取很大一部分微调数据集。综上所述，这些结果代表了针对用于MIA和训练数据提取的预先训练和微调的LLM的现有最强隐私攻击，这些攻击具有独立的科学意义，并对LLM的安全、隐私和版权问题具有重要的实践意义。



## **49. Unleashing the potential of prompt engineering: a comprehensive review**

释放即时工程的潜力：全面回顾 cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2310.14735v3) [paper-pdf](http://arxiv.org/pdf/2310.14735v3)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review explores the transformative potential of prompt engineering within the realm of large language models (LLMs) and multimodal language models (MMLMs). The development of AI, from its inception in the 1950s to the emergence of neural networks and deep learning architectures, has culminated in sophisticated LLMs like GPT-4 and BERT, as well as MMLMs like DALL-E and CLIP. These models have revolutionized tasks in diverse fields such as workplace automation, healthcare, and education. Prompt engineering emerges as a crucial technique to maximize the utility and accuracy of these models. This paper delves into both foundational and advanced methodologies of prompt engineering, including techniques like Chain of Thought, Self-consistency, and Generated Knowledge, which significantly enhance model performance. Additionally, it examines the integration of multimodal data through innovative approaches such as Multi-modal Prompt Learning (MaPLe), Conditional Prompt Learning, and Context Optimization. Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review underscores the pivotal role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的综述探索了快速工程在大型语言模型(LLM)和多模式语言模型(MMLM)领域中的变革潜力。人工智能从20世纪50年代开始发展到神经网络和深度学习体系结构的出现，最终出现了GPT-4和BERT等复杂的LLM，以及Dall-E和CLIP等MMLM。这些模式使工作场所自动化、医疗保健和教育等不同领域的任务发生了革命性变化。为了最大限度地提高这些模型的实用性和准确性，快速工程技术应运而生。本文深入研究了即时工程的基础和高级方法，包括思想链、自我一致性和生成知识等技术，这些技术显著提高了模型的性能。此外，它还通过多模式快速学习(Maple)、条件性快速学习和上下文优化等创新方法研究了多模式数据的集成。对这一讨论至关重要的是人工智能安全方面，特别是利用即时工程中的漏洞进行的对抗性攻击。对缓解这些风险和增强模型稳健性的策略进行了彻底的回顾。对快速方法的评估通过主观和客观两个指标进行，确保对其有效性进行稳健的分析。这篇综述强调了快速工程在推进人工智能能力方面的关键作用，为未来的研究和应用提供了一个结构化的框架。



## **50. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

通过特定层的编辑保护大型语言模型免受越狱攻击 cs.AI

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18166v1) [paper-pdf](http://arxiv.org/pdf/2405.18166v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.

摘要: 大型语言模型(LLM)正越来越多地被广泛地应用于现实世界中。尽管它们的表现令人印象深刻，但最近的研究表明，即使在通过从人类反馈的强化学习或监督微调进行调整时，LLM仍容易受到故意设计的敌意提示的攻击。虽然现有的防御方法侧重于检测有害提示或通过各种手段减少有害响应的可能性，但基于LLMS的内部机制来防御LLMS的越狱攻击在很大程度上仍未被探索。在这项工作中，我们研究了LLMS对有害提示的响应，并提出了一种新的防御方法-.通过LED，我们揭示了LLMS的早期层之间存在着几个关键的安全层。然后，我们展示了将这些安全层(以及一些选定的附加层)与选定目标层的解码安全响应重新对准可以显著提高LLM对抗越狱攻击的对准。在各种LLM(如Llama2、Mistral)上的广泛实验表明，LED是有效的，它可以有效防御越狱攻击，同时保持对良性提示的性能。我们的代码可在\url{https://github.com/ledllm/ledllm}.



