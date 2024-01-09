# Latest Adversarial Attack Papers
**update at 2024-01-09 15:08:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Vulnerabilities Unveiled: Adversarially Attacking a Multimodal Vision Language Model for Pathology Imaging**

漏洞揭开面纱：对病理成像多模式视觉语言模型的敌意攻击 eess.IV

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.02565v2) [paper-pdf](http://arxiv.org/pdf/2401.02565v2)

**Authors**: Jai Prakash Veerla, Poojitha Thota, Partha Sai Guttikonda, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the dynamic landscape of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted adversarial conditions. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial attacks to intentionally induce misclassifications. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models.

摘要: 在医学人工智能的动态场景中，本研究探索了视觉语言基础模型-病理语言图像预训练(PLIP)模型在有针对性的对抗条件下的脆弱性。利用Kather Colon数据集，其中包含9种组织类型的7,180张H&E图像，我们的调查使用了投影梯度下降(PGD)对抗性攻击来故意诱导错误分类。结果显示，PLIP操纵预测的成功率为100%，突显出其易受对手干扰的影响。对抗性例子的定性分析深入到了可解释性的挑战，揭示了对抗性操纵导致的预测的细微变化。这些发现为医学成像中视觉语言模型的可解释性、领域适应性和可信性提供了重要的见解。该研究强调，迫切需要强大的防御措施，以确保人工智能模型的可靠性。



## **2. The Impact of Adversarial Node Placement in Decentralized Federated Learning Networks**

分布式联合学习网络中对抗性节点放置的影响 cs.CR

Submitted to ICC 2024 conference

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2311.07946v2) [paper-pdf](http://arxiv.org/pdf/2311.07946v2)

**Authors**: Adam Piaseczny, Eric Ruzomberka, Rohit Parasnis, Christopher G. Brinton

**Abstract**: As Federated Learning (FL) grows in popularity, new decentralized frameworks are becoming widespread. These frameworks leverage the benefits of decentralized environments to enable fast and energy-efficient inter-device communication. However, this growing popularity also intensifies the need for robust security measures. While existing research has explored various aspects of FL security, the role of adversarial node placement in decentralized networks remains largely unexplored. This paper addresses this gap by analyzing the performance of decentralized FL for various adversarial placement strategies when adversaries can jointly coordinate their placement within a network. We establish two baseline strategies for placing adversarial node: random placement and network centrality-based placement. Building on this foundation, we propose a novel attack algorithm that prioritizes adversarial spread over adversarial centrality by maximizing the average network distance between adversaries. We show that the new attack algorithm significantly impacts key performance metrics such as testing accuracy, outperforming the baseline frameworks by between 9% and 66.5% for the considered setups. Our findings provide valuable insights into the vulnerabilities of decentralized FL systems, setting the stage for future research aimed at developing more secure and robust decentralized FL frameworks.

摘要: 随着联邦学习(FL)的流行，新的去中心化框架正在变得广泛。这些框架利用分散环境的优势，实现快速、节能的设备间通信。然而，这种日益增长的人气也加剧了采取强有力的安全措施的必要性。虽然现有的研究已经探索了FL安全的各个方面，但敌意节点放置在分散网络中的作用在很大程度上仍未被探索。本文通过分析当对手可以在一个网络内联合协调他们的放置时，分散的FL在不同的对手放置策略下的性能来解决这一差距。我们建立了两种放置敌意节点的基线策略：随机放置和基于网络中心性的放置。在此基础上，我们提出了一种新的攻击算法，该算法通过最大化对手之间的平均网络距离来优先考虑对手的传播而不是对手的中心。我们发现，新的攻击算法显著影响了测试准确率等关键性能指标，在所考虑的设置下，性能比基准框架高出9%到66.5%。我们的发现对去中心化FL系统的脆弱性提供了有价值的见解，为未来旨在开发更安全和健壮的去中心化FL框架的研究奠定了基础。



## **3. Logits Poisoning Attack in Federated Distillation**

联合蒸馏中的洛吉斯中毒攻击 cs.LG

13 pages, 3 figures, 5 tables

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.03685v1) [paper-pdf](http://arxiv.org/pdf/2401.03685v1)

**Authors**: Yuhan Tang, Zhiyuan Wu, Bo Gao, Tian Wen, Yuwei Wang, Sheng Sun

**Abstract**: Federated Distillation (FD) is a novel and promising distributed machine learning paradigm, where knowledge distillation is leveraged to facilitate a more efficient and flexible cross-device knowledge transfer in federated learning. By optimizing local models with knowledge distillation, FD circumvents the necessity of uploading large-scale model parameters to the central server, simultaneously preserving the raw data on local clients. Despite the growing popularity of FD, there is a noticeable gap in previous works concerning the exploration of poisoning attacks within this framework. This can lead to a scant understanding of the vulnerabilities to potential adversarial actions. To this end, we introduce FDLA, a poisoning attack method tailored for FD. FDLA manipulates logit communications in FD, aiming to significantly degrade model performance on clients through misleading the discrimination of private samples. Through extensive simulation experiments across a variety of datasets, attack scenarios, and FD configurations, we demonstrate that LPA effectively compromises client model accuracy, outperforming established baseline algorithms in this regard. Our findings underscore the critical need for robust defense mechanisms in FD settings to mitigate such adversarial threats.

摘要: 联合蒸馏(FD)是一种新颖的、有前途的分布式机器学习范式，其中利用知识蒸馏来促进联合学习中更高效、更灵活的跨设备知识转移。通过知识提炼优化本地模型，FD避免了将大规模模型参数上传到中心服务器的需要，同时将原始数据保存在本地客户端。尽管FD越来越受欢迎，但在这一框架内关于中毒攻击的探索存在着明显的空白。这可能导致对潜在敌对行动的脆弱性缺乏了解。为此，我们引入了FDLA，一种为FD量身定做的中毒攻击方法。FDLA操纵FD中的Logit通信，旨在通过误导私人样本的区分来显著降低客户的模型性能。通过对各种数据集、攻击场景和FD配置进行广泛的模拟实验，我们证明了LPA有效地折衷了客户端模型的准确性，在这方面优于已建立的基准算法。我们的发现强调了在FD环境中迫切需要强大的防御机制来缓解这种对抗性威胁。



## **4. Assessing the Influence of Different Types of Probing on Adversarial Decision-Making in a Deception Game**

在欺骗游戏中评估不同类型的探测对对抗性决策的影响 cs.CR

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2310.10662v3) [paper-pdf](http://arxiv.org/pdf/2310.10662v3)

**Authors**: Md Abu Sayed, Mohammad Ariful Islam Khan, Bryant A Allsup, Joshua Zamora, Palvi Aggarwal

**Abstract**: Deception, which includes leading cyber-attackers astray with false information, has shown to be an effective method of thwarting cyber-attacks. There has been little investigation of the effect of probing action costs on adversarial decision-making, despite earlier studies on deception in cybersecurity focusing primarily on variables like network size and the percentage of honeypots utilized in games. Understanding human decision-making when prompted with choices of various costs is essential in many areas such as in cyber security. In this paper, we will use a deception game (DG) to examine different costs of probing on adversarial decisions. To achieve this we utilized an IBLT model and a delayed feedback mechanism to mimic knowledge of human actions. Our results were taken from an even split of deception and no deception to compare each influence. It was concluded that probing was slightly taken less as the cost of probing increased. The proportion of attacks stayed relatively the same as the cost of probing increased. Although a constant cost led to a slight decrease in attacks. Overall, our results concluded that the different probing costs do not have an impact on the proportion of attacks whereas it had a slightly noticeable impact on the proportion of probing.

摘要: 欺骗，包括用虚假信息误导网络攻击者，已被证明是挫败网络攻击的有效方法。关于探测行动成本对对抗性决策的影响几乎没有调查，尽管早期关于网络安全欺骗的研究主要集中在网络规模和游戏中使用的蜜罐百分比等变量上。在网络安全等许多领域，理解人类在面临各种成本选择时的决策至关重要。在本文中，我们将使用欺骗游戏（DG）来研究对抗性决策的不同探测成本。为了实现这一目标，我们利用IBLT模型和延迟反馈机制来模仿人类行为的知识。我们的结果是从欺骗和不欺骗的平均分裂中得出的，以比较每种影响。结果表明，随着探测成本的增加，探测的使用率略有下降。攻击的比例保持相对不变，而探测的成本则增加。虽然恒定的成本导致攻击略有下降。总的来说，我们的研究结果得出结论，不同的探测成本不会对攻击的比例产生影响，而对探测的比例有轻微的影响。



## **5. AdvSQLi: Generating Adversarial SQL Injections against Real-world WAF-as-a-service**

AdvSQLi：针对真实世界的WAF-as-a-Service生成敌意SQL注入 cs.CR

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.02615v2) [paper-pdf](http://arxiv.org/pdf/2401.02615v2)

**Authors**: Zhenqing Qu, Xiang Ling, Ting Wang, Xiang Chen, Shouling Ji, Chunming Wu

**Abstract**: As the first defensive layer that attacks would hit, the web application firewall (WAF) plays an indispensable role in defending against malicious web attacks like SQL injection (SQLi). With the development of cloud computing, WAF-as-a-service, as one kind of Security-as-a-service, has been proposed to facilitate the deployment, configuration, and update of WAFs in the cloud. Despite its tremendous popularity, the security vulnerabilities of WAF-as-a-service are still largely unknown, which is highly concerning given its massive usage. In this paper, we propose a general and extendable attack framework, namely AdvSQLi, in which a minimal series of transformations are performed on the hierarchical tree representation of the original SQLi payload, such that the generated SQLi payloads can not only bypass WAF-as-a-service under black-box settings but also keep the same functionality and maliciousness as the original payload. With AdvSQLi, we make it feasible to inspect and understand the security vulnerabilities of WAFs automatically, helping vendors make products more secure.   To evaluate the attack effectiveness and efficiency of AdvSQLi, we first employ two public datasets to generate adversarial SQLi payloads, leading to a maximum attack success rate of 100% against state-of-the-art ML-based SQLi detectors. Furthermore, to demonstrate the immediate security threats caused by AdvSQLi, we evaluate the attack effectiveness against 7 WAF-as-a-service solutions from mainstream vendors and find all of them are vulnerable to AdvSQLi. For instance, AdvSQLi achieves an attack success rate of over 79% against the F5 WAF. Through in-depth analysis of the evaluation results, we further condense out several general yet severe flaws of these vendors that cannot be easily patched.

摘要: Web应用防火墙(WAF)作为攻击攻击的第一防御层，在防御SQL注入(SQLI)等恶意Web攻击中发挥着不可或缺的作用。随着云计算的发展，WAF-as-a-Service作为安全即服务的一种，被提出以方便WAF在云中的部署、配置和更新。尽管网站管家非常受欢迎，但它的安全漏洞仍然很大程度上是未知的，这是高度关注的，因为它的大量使用。本文提出了一个通用的、可扩展的攻击框架AdvSQLI，该框架对原始SQLI负载的层次树表示进行最小一系列的变换，使得生成的SQLI负载不仅可以在黑盒环境下绕过WAF-as-a-Service，而且保持了与原始负载相同的功能和恶意。有了AdvSQLi，我们就可以自动检查和了解无线局域网的安全漏洞，帮助供应商让产品更安全。为了评估AdvSQLI的攻击效果和效率，我们首先使用两个公开的数据集来生成对抗性的SQLI有效负载，导致对最先进的基于ML的SQLI检测器的最大攻击成功率为100%。此外，为了展示AdvSQLi带来的直接安全威胁，我们对来自主流厂商的7个网站管家即服务解决方案进行了攻击有效性评估，发现它们都容易受到AdvSQLi的攻击。例如，AdvSQLi对F5 WAF的攻击成功率超过79%。通过对评估结果的深入分析，我们进一步提炼出了这些供应商普遍存在的几个不容易修补的严重缺陷。



## **6. Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience**

提升防御：在对抗性训练和模型复原力水印之间架起桥梁 cs.LG

Accepted at DAI Workshop, AAAI 2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2312.14260v2) [paper-pdf](http://arxiv.org/pdf/2312.14260v2)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Machine learning models are being used in an increasing number of critical applications; thus, securing their integrity and ownership is critical. Recent studies observed that adversarial training and watermarking have a conflicting interaction. This work introduces a novel framework to integrate adversarial training with watermarking techniques to fortify against evasion attacks and provide confident model verification in case of intellectual property theft. We use adversarial training together with adversarial watermarks to train a robust watermarked model. The key intuition is to use a higher perturbation budget to generate adversarial watermarks compared to the budget used for adversarial training, thus avoiding conflict. We use the MNIST and Fashion-MNIST datasets to evaluate our proposed technique on various model stealing attacks. The results obtained consistently outperform the existing baseline in terms of robustness performance and further prove the resilience of this defense against pruning and fine-tuning removal attacks.

摘要: 机器学习模型正在越来越多的关键应用中使用；因此，确保它们的完整性和所有权至关重要。最近的研究发现，对抗性训练和水印之间存在相互冲突的作用。这项工作引入了一种新的框架，将对抗性训练与水印技术相结合，以加强对逃避攻击的防御，并在知识产权被盗的情况下提供可信的模型验证。我们使用对抗性训练和对抗性水印相结合的方法来训练一个健壮的水印模型。关键的直觉是，与用于对抗性训练的预算相比，使用更高的扰动预算来生成对抗性水印，从而避免冲突。我们使用MNIST和Fashion-MNIST数据集来评估我们提出的针对各种模型窃取攻击的技术。得到的结果在稳健性性能方面始终优于现有的基线，并进一步证明了该防御措施对剪枝和微调删除攻击的弹性。



## **7. ROIC-DM: Robust Text Inference and Classification via Diffusion Model**

ROIC-DM：基于扩散模型的鲁棒文本推理与分类 cs.CL

aaai2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.03514v1) [paper-pdf](http://arxiv.org/pdf/2401.03514v1)

**Authors**: Shilong Yuan, Wei Yuan, Tieke HE

**Abstract**: While language models have made many milestones in text inference and classification tasks, they remain susceptible to adversarial attacks that can lead to unforeseen outcomes. Existing works alleviate this problem by equipping language models with defense patches. However, these defense strategies often rely on impractical assumptions or entail substantial sacrifices in model performance. Consequently, enhancing the resilience of the target model using such defense mechanisms is a formidable challenge. This paper introduces an innovative model for robust text inference and classification, built upon diffusion models (ROIC-DM). Benefiting from its training involving denoising stages, ROIC-DM inherently exhibits greater robustness compared to conventional language models. Moreover, ROIC-DM can attain comparable, and in some cases, superior performance to language models, by effectively incorporating them as advisory components. Extensive experiments conducted with several strong textual adversarial attacks on three datasets demonstrate that (1) ROIC-DM outperforms traditional language models in robustness, even when the latter are fortified with advanced defense mechanisms; (2) ROIC-DM can achieve comparable and even better performance than traditional language models by using them as advisors.

摘要: 虽然语言模型在文本推理和分类任务中取得了许多里程碑，但它们仍然容易受到可能导致不可预见结果的对抗性攻击。现有的工作通过为语言模型配备防御补丁来缓解这个问题。然而，这些防御策略往往依赖于不切实际的假设，或者需要在模型性能上做出实质性的牺牲。因此，使用这种防御机制提高目标模型的弹性是一个巨大的挑战。本文介绍了一种基于扩散模型的稳健文本推理和分类模型(ROIC-DM)。由于其训练涉及去噪阶段，ROIC-DM固有地表现出比传统语言模型更强的稳健性。此外，通过有效地将语言模型合并为咨询组件，ROIC-DM可以获得与语言模型相当的、甚至在某些情况下优于语言模型的性能。在三个数据集上进行的大量实验表明：(1)ROIC-DM在稳健性方面优于传统的语言模型，即使后者有先进的防御机制；(2)ROIC-DM可以获得与传统语言模型相当甚至更好的性能，通过使用它们作为顾问。



## **8. Data-Driven Subsampling in the Presence of an Adversarial Actor**

对抗性参与者在场情况下的数据驱动子抽样 cs.LG

Accepted for publication at ICMLCN 2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.03488v1) [paper-pdf](http://arxiv.org/pdf/2401.03488v1)

**Authors**: Abu Shafin Mohammad Mahdee Jameel, Ahmed P. Mohamed, Jinho Yi, Aly El Gamal, Akshay Malhotra

**Abstract**: Deep learning based automatic modulation classification (AMC) has received significant attention owing to its potential applications in both military and civilian use cases. Recently, data-driven subsampling techniques have been utilized to overcome the challenges associated with computational complexity and training time for AMC. Beyond these direct advantages of data-driven subsampling, these methods also have regularizing properties that may improve the adversarial robustness of the modulation classifier. In this paper, we investigate the effects of an adversarial attack on an AMC system that employs deep learning models both for AMC and for subsampling. Our analysis shows that subsampling itself is an effective deterrent to adversarial attacks. We also uncover the most efficient subsampling strategy when an adversarial attack on both the classifier and the subsampler is anticipated.

摘要: 基于深度学习的自动调制分类(AMC)因其在军事和民用方面的潜在应用而受到广泛关注。最近，数据驱动的子采样技术已经被用来克服与AMC的计算复杂性和训练时间相关的挑战。除了数据驱动的子采样的这些直接优势之外，这些方法还具有可提高调制分类器的对抗性的正则化特性。在本文中，我们研究了敌意攻击对AMC系统的影响，该系统对AMC和子采样都采用了深度学习模型。我们的分析表明，二次抽样本身就是一种有效的对抗攻击的威慑。当分类器和子采样器都受到敌意攻击时，我们还发现了最有效的子采样策略。



## **9. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey**

自然语言处理中的标记修改攻击：综述 cs.CL

Version 3: edited and expanded

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2103.00676v3) [paper-pdf](http://arxiv.org/pdf/2103.00676v3)

**Authors**: Tom Roth, Yansong Gao, Alsharif Abuadbba, Surya Nepal, Wei Liu

**Abstract**: Many adversarial attacks target natural language processing systems, most of which succeed through modifying the individual tokens of a document. Despite the apparent uniqueness of each of these attacks, fundamentally they are simply a distinct configuration of four components: a goal function, allowable transformations, a search method, and constraints. In this survey, we systematically present the different components used throughout the literature, using an attack-independent framework which allows for easy comparison and categorisation of components. Our work aims to serve as a comprehensive guide for newcomers to the field and to spark targeted research into refining the individual attack components.

摘要: 许多敌意攻击的目标是自然语言处理系统，其中大多数攻击是通过修改文档的单个令牌来成功的。尽管这些攻击看起来都是独一无二的，但从根本上说，它们只是四个组件的不同配置：目标函数、允许的转换、搜索方法和约束。在本调查中，我们系统地介绍了整个文献中使用的不同组件，使用了与攻击无关的框架，该框架允许对组件进行轻松的比较和分类。我们的工作旨在为该领域的新手提供全面的指导，并引发有针对性的研究，以完善个别攻击组件。



## **10. Affinity Uncertainty-based Hard Negative Mining in Graph Contrastive Learning**

图对比学习中基于亲和度不确定性的硬否定挖掘 cs.LG

Accepted to TNNLS

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2301.13340v2) [paper-pdf](http://arxiv.org/pdf/2301.13340v2)

**Authors**: Chaoxi Niu, Guansong Pang, Ling Chen

**Abstract**: Hard negative mining has shown effective in enhancing self-supervised contrastive learning (CL) on diverse data types, including graph CL (GCL). The existing hardness-aware CL methods typically treat negative instances that are most similar to the anchor instance as hard negatives, which helps improve the CL performance, especially on image data. However, this approach often fails to identify the hard negatives but leads to many false negatives on graph data. This is mainly due to that the learned graph representations are not sufficiently discriminative due to oversmooth representations and/or non-independent and identically distributed (non-i.i.d.) issues in graph data. To tackle this problem, this article proposes a novel approach that builds a discriminative model on collective affinity information (i.e., two sets of pairwise affinities between the negative instances and the anchor instance) to mine hard negatives in GCL. In particular, the proposed approach evaluates how confident/uncertain the discriminative model is about the affinity of each negative instance to an anchor instance to determine its hardness weight relative to the anchor instance. This uncertainty information is then incorporated into the existing GCL loss functions via a weighting term to enhance their performance. The enhanced GCL is theoretically grounded that the resulting GCL loss is equivalent to a triplet loss with an adaptive margin being exponentially proportional to the learned uncertainty of each negative instance. Extensive experiments on ten graph datasets show that our approach does the following: 1) consistently enhances different state-of-the-art (SOTA) GCL methods in both graph and node classification tasks and 2) significantly improves their robustness against adversarial attacks. Code is available at https://github.com/mala-lab/AUGCL.

摘要: 硬负挖掘在增强包括图对比学习(GCL)在内的不同数据类型上的自我监督对比学习(CL)方面表现出了有效的效果。现有的硬度感知CL方法通常将与锚实例最相似的否定实例视为硬否定，这有助于提高CL的性能，特别是在图像数据上。然而，这种方法往往不能识别硬否定，而是导致图数据上的许多假否定。这主要是因为由于过度光滑的表示和/或非独立且同分布的(非I.I.D.)，学习的图表示不具有足够的区分性。图形数据中的问题。针对这一问题，本文提出了一种基于群体亲和力信息的判别模型(即否定实例和锚定实例之间的两组成对亲和度)来挖掘GCL中的硬否定。特别地，该方法评估判别模型关于每个否定实例对锚实例的亲和度的置信度/不确定性，以确定其相对于锚实例的硬度权重。然后，这种不确定性信息通过加权项被合并到现有的GCL损失函数中，以提高它们的性能。增强型GCL的理论基础是，由此产生的GCL损失等同于三重损失，其自适应裕度与每个负实例的学习不确定性成指数正比。在10个图数据集上的大量实验表明，我们的方法做了以下工作：1)在图和节点分类任务中一致地提高了不同的最新技术(SOTA)GCL方法；2)显著提高了它们对对手攻击的健壮性。代码可在https://github.com/mala-lab/AUGCL.上找到



## **11. Data-Dependent Stability Analysis of Adversarial Training**

对抗性训练的数据依赖稳定性分析 cs.LG

**SubmitDate**: 2024-01-06    [abs](http://arxiv.org/abs/2401.03156v1) [paper-pdf](http://arxiv.org/pdf/2401.03156v1)

**Authors**: Yihan Wang, Shuang Liu, Xiao-Shan Gao

**Abstract**: Stability analysis is an essential aspect of studying the generalization ability of deep learning, as it involves deriving generalization bounds for stochastic gradient descent-based training algorithms. Adversarial training is the most widely used defense against adversarial example attacks. However, previous generalization bounds for adversarial training have not included information regarding the data distribution. In this paper, we fill this gap by providing generalization bounds for stochastic gradient descent-based adversarial training that incorporate data distribution information. We utilize the concepts of on-average stability and high-order approximate Lipschitz conditions to examine how changes in data distribution and adversarial budget can affect robust generalization gaps. Our derived generalization bounds for both convex and non-convex losses are at least as good as the uniform stability-based counterparts which do not include data distribution information. Furthermore, our findings demonstrate how distribution shifts from data poisoning attacks can impact robust generalization.

摘要: 稳定性分析是研究深度学习泛化能力的一个重要方面，因为它涉及到推导基于随机梯度下降的训练算法的泛化界限。对抗性训练是对抗对抗性范例攻击最广泛使用的防御手段。然而，以前的对抗性训练的泛化界限没有包括关于数据分布的信息。在本文中，我们通过提供包含数据分布信息的基于随机梯度下降的对抗性训练的泛化界限来填补这一空白。我们利用平均稳定性和高阶近似Lipschitz条件的概念来检验数据分布和对抗性预算的变化如何影响稳健的泛化差距。我们推导出的凸损失和非凸损失的泛化界至少与基于一致稳定性的相应界一样好，后者不包括数据分布信息。此外，我们的发现表明，数据中毒攻击的分布变化如何影响健壮的泛化。



## **12. Transferable Learned Image Compression-Resistant Adversarial Perturbations**

可转移的学习图像抗压缩对抗扰动 cs.CV

Accepted as poster at Data Compression Conference 2024 (DCC 2024)

**SubmitDate**: 2024-01-06    [abs](http://arxiv.org/abs/2401.03115v1) [paper-pdf](http://arxiv.org/pdf/2401.03115v1)

**Authors**: Yang Sui, Zhuohang Li, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Zhenzhong Chen

**Abstract**: Adversarial attacks can readily disrupt the image classification system, revealing the vulnerability of DNN-based recognition tasks. While existing adversarial perturbations are primarily applied to uncompressed images or compressed images by the traditional image compression method, i.e., JPEG, limited studies have investigated the robustness of models for image classification in the context of DNN-based image compression. With the rapid evolution of advanced image compression, DNN-based learned image compression has emerged as the promising approach for transmitting images in many security-critical applications, such as cloud-based face recognition and autonomous driving, due to its superior performance over traditional compression. Therefore, there is a pressing need to fully investigate the robustness of a classification system post-processed by learned image compression. To bridge this research gap, we explore the adversarial attack on a new pipeline that targets image classification models that utilize learned image compressors as pre-processing modules. Furthermore, to enhance the transferability of perturbations across various quality levels and architectures of learned image compression models, we introduce a saliency score-based sampling method to enable the fast generation of transferable perturbation. Extensive experiments with popular attack methods demonstrate the enhanced transferability of our proposed method when attacking images that have been post-processed with different learned image compression models.

摘要: 敌意攻击很容易破坏图像分类系统，暴露了基于DNN的识别任务的脆弱性。虽然现有的对抗性扰动主要应用于未压缩图像或使用传统图像压缩方法(即JPEG)压缩的图像，但在基于DNN的图像压缩环境下，已有有限的研究调查了图像分类模型的稳健性。随着先进图像压缩技术的迅速发展，基于DNN的学习图像压缩技术以其优于传统压缩的性能，在基于云的人脸识别和自动驾驶等安全关键应用中成为一种很有前途的图像传输方法。因此，迫切需要充分研究学习图像压缩后处理的分类系统的稳健性。为了弥补这一研究空白，我们探索了一种新的管道上的敌意攻击，该管道的目标是使用学习的图像压缩器作为预处理模块的图像分类模型。此外，为了增强扰动在学习图像压缩模型的不同质量水平和体系结构上的可转移性，我们引入了基于显著分数的采样方法来快速生成可转移的扰动。对常用攻击方法的大量实验表明，当攻击经过不同学习图像压缩模型后处理的图像时，所提出的方法具有更强的可转移性。



## **13. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning**

乐透：联合学习中对抗敌意服务器的安全参与者选择 cs.CR

20 pages, 14 figures

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02880v1) [paper-pdf](http://arxiv.org/pdf/2401.02880v1)

**Authors**: Zhifeng Jiang, Peng Ye, Shiqi He, Wei Wang, Ruichuan Chen, Bo Li

**Abstract**: In Federated Learning (FL), common privacy-preserving technologies, such as secure aggregation and distributed differential privacy, rely on the critical assumption of an honest majority among participants to withstand various attacks. In practice, however, servers are not always trusted, and an adversarial server can strategically select compromised clients to create a dishonest majority, thereby undermining the system's security guarantees. In this paper, we present Lotto, an FL system that addresses this fundamental, yet underexplored issue by providing secure participant selection against an adversarial server. Lotto supports two selection algorithms: random and informed. To ensure random selection without a trusted server, Lotto enables each client to autonomously determine their participation using verifiable randomness. For informed selection, which is more vulnerable to manipulation, Lotto approximates the algorithm by employing random selection within a refined client pool. Our theoretical analysis shows that Lotto effectively restricts the number of server-selected compromised clients, thus ensuring an honest majority among participants. Large-scale experiments further reveal that Lotto achieves time-to-accuracy performance comparable to that of insecure selection methods, indicating a low computational overhead for secure selection.

摘要: 在联邦学习(FL)中，常见的隐私保护技术，如安全聚合和分布式差异隐私，依赖于参与者之间诚实多数的关键假设来抵御各种攻击。然而，在实践中，服务器并不总是可信的，敌意服务器可以策略性地选择受攻击的客户端来制造不诚实的多数，从而破坏系统的安全保证。在本文中，我们提出了乐透，一个FL系统，解决了这个基本的，但探索不足的问题，通过提供安全的参与者选择对抗敌对的服务器。乐透支持两种选择算法：随机和通知。为了确保在没有可信服务器的情况下随机选择，乐透使每个客户端能够使用可验证的随机性自主确定他们的参与。对于更容易受到操纵的知情选择，乐透通过在改进的客户机池中使用随机选择来近似算法。我们的理论分析表明，乐透有效地限制了服务器选择的受攻击客户端的数量，从而确保了参与者中诚实的多数。大规模实验进一步表明，乐透算法的时间精度性能与非安全选择方法相当，表明安全选择方法具有较低的计算开销。



## **14. PromptBench: A Unified Library for Evaluation of Large Language Models**

PromptBitch：大型语言模型评估的统一库 cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2312.07910v2) [paper-pdf](http://arxiv.org/pdf/2312.07910v2)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型(LLM)的评估对于评估其性能和降低潜在的安全风险至关重要。在本文中，我们介绍了一个用于评估LLMS的统一库PromptBitch.它由几个易于研究人员使用和扩展的关键组件组成：即时构建、即时工程、数据集和模型加载、对抗性即时攻击、动态评估协议和分析工具。PromptBitch是一个开放的、通用的、灵活的研究代码库，可以在创建新的基准、部署下游应用程序和设计新的评估协议方面促进原创研究。该代码可在https://github.com/microsoft/promptbench上获得，并将继续受到支持。



## **15. Enhancing targeted transferability via feature space fine-tuning**

通过特征空间微调增强目标可转移性 cs.CV

9 pages, 10 figures, accepted by 2024ICASSP

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02727v1) [paper-pdf](http://arxiv.org/pdf/2401.02727v1)

**Authors**: Hui Zeng, Biwei Chen, Anjie Peng

**Abstract**: Adversarial examples (AEs) have been extensively studied due to their potential for privacy protection and inspiring robust neural networks. However, making a targeted AE transferable across unknown models remains challenging. In this paper, to alleviate the overfitting dilemma common in an AE crafted by existing simple iterative attacks, we propose fine-tuning it in the feature space. Specifically, starting with an AE generated by a baseline attack, we encourage the features that contribute to the target class and discourage the features that contribute to the original class in a middle layer of the source model. Extensive experiments demonstrate that only a few iterations of fine-tuning can boost existing attacks in terms of targeted transferability nontrivially and universally. Our results also verify that the simple iterative attacks can yield comparable or even better transferability than the resource-intensive methods, which rely on training target-specific classifiers or generators with additional data. The code is available at: github.com/zengh5/TA_feature_FT.

摘要: 对抗性例子(AEs)由于其在隐私保护和激发健壮神经网络方面的潜力而被广泛研究。然而，使有针对性的AE可以在未知模型之间转移仍然具有挑战性。在本文中，为了缓解现有简单迭代攻击所产生的AE中普遍存在的过适应困境，我们提出了在特征空间中对其进行微调。具体地说，从基线攻击生成的AE开始，我们鼓励对目标类有贡献的特征，而不鼓励在源模型的中间层中对原始类有贡献的特征。广泛的实验表明，只有几次迭代的微调才能在目标可转移性、非平凡和普遍方面增强现有攻击。我们的结果还验证了简单的迭代攻击可以产生与资源密集型方法相当甚至更好的可转移性，后者依赖于用额外的数据来训练特定于目标的分类器或生成器。代码可在以下网址获得：githorb.com/zengh5/TA_Feature_FT。



## **16. Calibration Attack: A Framework For Adversarial Attacks Targeting Calibration**

校准攻击：一种针对校准的对抗性攻击框架 cs.LG

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02718v1) [paper-pdf](http://arxiv.org/pdf/2401.02718v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu, Hongyu Guo

**Abstract**: We introduce a new framework of adversarial attacks, named calibration attacks, in which the attacks are generated and organized to trap victim models to be miscalibrated without altering their original accuracy, hence seriously endangering the trustworthiness of the models and any decision-making based on their confidence scores. Specifically, we identify four novel forms of calibration attacks: underconfidence attacks, overconfidence attacks, maximum miscalibration attacks, and random confidence attacks, in both the black-box and white-box setups. We then test these new attacks on typical victim models with comprehensive datasets, demonstrating that even with a relatively low number of queries, the attacks can create significant calibration mistakes. We further provide detailed analyses to understand different aspects of calibration attacks. Building on that, we investigate the effectiveness of widely used adversarial defences and calibration methods against these types of attacks, which then inspires us to devise two novel defences against such calibration attacks.

摘要: 我们引入了一种新的对抗性攻击框架，称为校准攻击，在不改变受害者模型原有精度的情况下，生成和组织攻击以捕获需要错误校准的受害者模型，从而严重危及模型的可信性和任何基于其置信度的决策。具体地说，我们识别了四种新的校准攻击形式：在黑盒和白盒设置中的低信任度攻击、过度自信攻击、最大误校准度攻击和随机信任攻击。然后，我们使用全面的数据集在典型的受害者模型上测试这些新攻击，表明即使查询次数相对较少，攻击也可能造成严重的校准错误。我们进一步提供了详细的分析，以了解校准攻击的不同方面。在此基础上，我们研究了广泛使用的对抗性防御和校准方法对这类攻击的有效性，这促使我们设计了两种针对此类校准攻击的新防御方法。



## **17. Game Theory for Adversarial Attacks and Defenses**

对抗性攻防的博弈论 cs.LG

With the agreement of my coauthors, I would like to withdraw the  manuscript "Game Theory for Adversarial Attacks and Defenses". Some  experimental procedures were not included in the manuscript, which makes a  part of important claims not meaningful

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2110.06166v4) [paper-pdf](http://arxiv.org/pdf/2110.06166v4)

**Authors**: Shorya Sharma

**Abstract**: Adversarial attacks can generate adversarial inputs by applying small but intentionally worst-case perturbations to samples from the dataset, which leads to even state-of-the-art deep neural networks outputting incorrect answers with high confidence. Hence, some adversarial defense techniques are developed to improve the security and robustness of the models and avoid them being attacked. Gradually, a game-like competition between attackers and defenders formed, in which both players would attempt to play their best strategies against each other while maximizing their own payoffs. To solve the game, each player would choose an optimal strategy against the opponent based on the prediction of the opponent's strategy choice. In this work, we are on the defensive side to apply game-theoretic approaches on defending against attacks. We use two randomization methods, random initialization and stochastic activation pruning, to create diversity of networks. Furthermore, we use one denoising technique, super resolution, to improve models' robustness by preprocessing images before attacks. Our experimental results indicate that those three methods can effectively improve the robustness of deep-learning neural networks.

摘要: 对抗性攻击可以通过对数据集中的样本应用小的但有意的最坏情况扰动来生成对抗性输入，这甚至导致最先进的深度神经网络以高置信度输出不正确的答案。因此，一些对抗性防御技术被开发出来，以提高模型的安全性和稳健性，避免模型受到攻击。逐渐地，攻击者和防守者之间形成了一种游戏式的竞争，双方都试图在最大化自己收益的同时，发挥自己最好的策略。为了解决博弈，每个参与者都会根据对手的策略选择的预测来选择一个最优策略来对抗对手。在这项工作中，我们处于守势，将博弈论方法应用于防御攻击。我们使用两种随机化方法，随机初始化和随机激活剪枝，以创建网络的多样性。此外，我们使用了超分辨率去噪技术，通过在攻击前对图像进行预处理来提高模型的稳健性。实验结果表明，这三种方法都能有效地提高深度学习神经网络的鲁棒性。



## **18. Secure Control of Connected and Automated Vehicles Using Trust-Aware Robust Event-Triggered Control Barrier Functions**

使用信任感知的健壮事件触发控制屏障功能实现互联和自动化车辆的安全控制 eess.SY

arXiv admin note: substantial text overlap with arXiv:2305.16818

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02306v2) [paper-pdf](http://arxiv.org/pdf/2401.02306v2)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Akua Dickson, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: We address the security of a network of Connected and Automated Vehicles (CAVs) cooperating to safely navigate through a conflict area (e.g., traffic intersections, merging roadways, roundabouts). Previous studies have shown that such a network can be targeted by adversarial attacks causing traffic jams or safety violations ending in collisions. We focus on attacks targeting the V2X communication network used to share vehicle data and consider as well uncertainties due to noise in sensor measurements and communication channels. To combat these, motivated by recent work on the safe control of CAVs, we propose a trust-aware robust event-triggered decentralized control and coordination framework that can provably guarantee safety. We maintain a trust metric for each vehicle in the network computed based on their behavior and used to balance the tradeoff between conservativeness (when deeming every vehicle as untrustworthy) and guaranteed safety and security. It is important to highlight that our framework is invariant to the specific choice of the trust framework. Based on this framework, we propose an attack detection and mitigation scheme which has twofold benefits: (i) the trust framework is immune to false positives, and (ii) it provably guarantees safety against false positive cases. We use extensive simulations (in SUMO and CARLA) to validate the theoretical guarantees and demonstrate the efficacy of our proposed scheme to detect and mitigate adversarial attacks.

摘要: 我们致力于解决互联和自动化车辆(CAV)网络的安全问题，这些车辆通过协作安全地通过冲突区域(例如，交通路口、合并道路、环形交叉路口)。以前的研究表明，这样的网络可以成为导致交通拥堵或以碰撞结束的安全违规行为的对抗性攻击的目标。我们专注于针对用于共享车辆数据的V2X通信网络的攻击，并考虑由于传感器测量和通信通道中的噪声而产生的不确定性。为了应对这些问题，基于最近在CAV安全控制方面的工作，我们提出了一个信任感知的、健壮的、事件触发的分布式控制和协调框架，该框架能够有效地保证安全。我们为网络中的每辆车维护一个基于其行为计算的信任度量，用于平衡保守性(当认为每辆车不值得信任时)与保证的安全和保障之间的权衡。必须强调的是，我们的框架与信任框架的具体选择是不变的。基于该框架，我们提出了一种攻击检测和缓解方案，该方案具有两个优点：(I)信任框架不受误报的影响；(Ii)它可证明地保证了对误报情况的安全性。我们使用大量的仿真(在相扑和CALA中)来验证理论上的保证，并展示了我们所提出的方案在检测和缓解敌意攻击方面的有效性。



## **19. MalModel: Hiding Malicious Payload in Mobile Deep Learning Models with Black-box Backdoor Attack**

MalModel：利用黑盒后门攻击隐藏移动深度学习模型中的恶意负载 cs.CR

Due to the limitation "The abstract field cannot be longer than 1,920  characters", the abstract here is shorter than that in the PDF file

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02659v1) [paper-pdf](http://arxiv.org/pdf/2401.02659v1)

**Authors**: Jiayi Hua, Kailong Wang, Meizhen Wang, Guangdong Bai, Xiapu Luo, Haoyu Wang

**Abstract**: Mobile malware has become one of the most critical security threats in the era of ubiquitous mobile computing. Despite the intensive efforts from security experts to counteract it, recent years have still witnessed a rapid growth of identified malware samples. This could be partly attributed to the newly-emerged technologies that may constantly open up under-studied attack surfaces for the adversaries. One typical example is the recently-developed mobile machine learning (ML) framework that enables storing and running deep learning (DL) models on mobile devices. Despite obvious advantages, this new feature also inadvertently introduces potential vulnerabilities (e.g., on-device models may be modified for malicious purposes). In this work, we propose a method to generate or transform mobile malware by hiding the malicious payloads inside the parameters of deep learning models, based on a strategy that considers four factors (layer type, layer number, layer coverage and the number of bytes to replace). Utilizing the proposed method, we can run malware in DL mobile applications covertly with little impact on the model performance (i.e., as little as 0.4% drop in accuracy and at most 39ms latency overhead).

摘要: 在移动计算无处不在的时代，移动恶意软件已经成为最关键的安全威胁之一。尽管安全专家做出了密集的努力来对抗它，但近年来识别出的恶意软件样本仍然快速增长。这在一定程度上可以归因于新出现的技术，这些技术可能会不断为对手打开研究不足的攻击面。一个典型的例子是最近开发的移动机器学习(ML)框架，该框架允许在移动设备上存储和运行深度学习(DL)模型。尽管有明显的优势，但这一新功能也在不经意间引入了潜在的漏洞(例如，设备上的模型可能会被恶意修改)。在这项工作中，我们提出了一种通过将恶意负载隐藏在深度学习模型的参数中来生成或转换移动恶意软件的方法，该方法考虑了四个因素(层类型、层数、层覆盖率和替换字节数)。利用该方法，我们可以在不影响模型性能的情况下，隐蔽地在下行移动应用程序中运行恶意软件(即，准确率仅下降0.4%，延迟开销最多39ms)。



## **20. A Random Ensemble of Encrypted models for Enhancing Robustness against Adversarial Examples**

一种增强对抗性样本鲁棒性的加密模型随机包围算法 cs.CR

4 pages

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02633v1) [paper-pdf](http://arxiv.org/pdf/2401.02633v1)

**Authors**: Ryota Iijima, Sayaka Shiota, Hitoshi Kiya

**Abstract**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, which means AEs generated for a source model can fool another black-box model (target model) with a non-trivial probability. In previous studies, it was confirmed that the vision transformer (ViT) is more robust against the property of adversarial transferability than convolutional neural network (CNN) models such as ConvMixer, and moreover encrypted ViT is more robust than ViT without any encryption. In this article, we propose a random ensemble of encrypted ViT models to achieve much more robust models. In experiments, the proposed scheme is verified to be more robust against not only black-box attacks but also white-box ones than convention methods.

摘要: 众所周知，深度神经网络（DNN）容易受到对抗性示例（AE）的影响。此外，AE具有对抗性可转移性，这意味着为源模型生成的AE可以以非平凡的概率欺骗另一个黑盒模型（目标模型）。在以前的研究中，已经证实视觉Transformer（ViT）比卷积神经网络（CNN）模型（如ConvMixer）更鲁棒，并且加密的ViT比没有任何加密的ViT更鲁棒。在这篇文章中，我们提出了一个加密ViT模型的随机集成，以实现更强大的模型。实验结果表明，该方案不仅对黑盒攻击具有较好的鲁棒性，而且对白盒攻击也具有较好的鲁棒性。



## **21. A Practical Survey on Emerging Threats from AI-driven Voice Attacks: How Vulnerable are Commercial Voice Control Systems?**

人工智能驱动的语音攻击新威胁的实践调查：商业语音控制系统有多脆弱？ cs.CR

14 pages

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.06010v2) [paper-pdf](http://arxiv.org/pdf/2312.06010v2)

**Authors**: Yuanda Wang, Qiben Yan, Nikolay Ivanov, Xun Chen

**Abstract**: The emergence of Artificial Intelligence (AI)-driven audio attacks has revealed new security vulnerabilities in voice control systems. While researchers have introduced a multitude of attack strategies targeting voice control systems (VCS), the continual advancements of VCS have diminished the impact of many such attacks. Recognizing this dynamic landscape, our study endeavors to comprehensively assess the resilience of commercial voice control systems against a spectrum of malicious audio attacks. Through extensive experimentation, we evaluate six prominent attack techniques across a collection of voice control interfaces and devices. Contrary to prevailing narratives, our results suggest that commercial voice control systems exhibit enhanced resistance to existing threats. Particularly, our research highlights the ineffectiveness of white-box attacks in black-box scenarios. Furthermore, the adversaries encounter substantial obstacles in obtaining precise gradient estimations during query-based interactions with commercial systems, such as Apple Siri and Samsung Bixby. Meanwhile, we find that current defense strategies are not completely immune to advanced attacks. Our findings contribute valuable insights for enhancing defense mechanisms in VCS. Through this survey, we aim to raise awareness within the academic community about the security concerns of VCS and advocate for continued research in this crucial area.

摘要: 人工智能(AI)驱动的音频攻击的出现揭示了语音控制系统中的新安全漏洞。虽然研究人员针对语音控制系统(VCS)引入了多种攻击策略，但VCS的不断进步削弱了许多此类攻击的影响。认识到这种动态格局，我们的研究努力全面评估商业语音控制系统对一系列恶意音频攻击的弹性。通过广泛的实验，我们评估了一系列语音控制接口和设备上的六种重要攻击技术。与流行的说法相反，我们的结果表明，商业语音控制系统对现有威胁的抵抗力增强了。特别是，我们的研究突出了白盒攻击在黑盒场景中的无效。此外，在与Apple Siri和Samsung Bixby等商业系统进行基于查询的交互期间，对手在获得精确的梯度估计方面遇到了巨大的障碍。与此同时，我们发现，当前的防御策略并不能完全免受高级攻击的影响。我们的发现为增强VCS的防御机制提供了有价值的见解。通过这项调查，我们旨在提高学术界对VCS安全问题的认识，并倡导在这一关键领域继续研究。



## **22. Evasive Hardware Trojan through Adversarial Power Trace**

利用敌意权力追踪规避硬件木马 cs.CR

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2401.02342v1) [paper-pdf](http://arxiv.org/pdf/2401.02342v1)

**Authors**: Behnam Omidi, Khaled N. Khasawneh, Ihsen Alouani

**Abstract**: The globalization of the Integrated Circuit (IC) supply chain, driven by time-to-market and cost considerations, has made ICs vulnerable to hardware Trojans (HTs). Against this threat, a promising approach is to use Machine Learning (ML)-based side-channel analysis, which has the advantage of being a non-intrusive method, along with efficiently detecting HTs under golden chip-free settings. In this paper, we question the trustworthiness of ML-based HT detection via side-channel analysis. We introduce a HT obfuscation (HTO) approach to allow HTs to bypass this detection method. Rather than theoretically misleading the model by simulated adversarial traces, a key aspect of our approach is the design and implementation of adversarial noise as part of the circuitry, alongside the HT. We detail HTO methodologies for ASICs and FPGAs, and evaluate our approach using TrustHub benchmark. Interestingly, we found that HTO can be implemented with only a single transistor for ASIC designs to generate adversarial power traces that can fool the defense with 100% efficiency. We also efficiently implemented our approach on a Spartan 6 Xilinx FPGA using 2 different variants: (i) DSP slices-based, and (ii) ring-oscillator-based design. Additionally, we assess the efficiency of countermeasures like spectral domain analysis, and we show that an adaptive attacker can still design evasive HTOs by constraining the design with a spectral noise budget. In addition, while adversarial training (AT) offers higher protection against evasive HTs, AT models suffer from a considerable utility loss, potentially rendering them unsuitable for such security application. We believe this research represents a significant step in understanding and exploiting ML vulnerabilities in a hardware security context, and we make all resources and designs openly available online: https://dev.d18uu4lqwhbmka.amplifyapp.com

摘要: 在上市时间和成本考虑的推动下，集成电路(IC)供应链的全球化使IC容易受到硬件特洛伊木马(HTS)的攻击。针对这一威胁，一种很有前途的方法是使用基于机器学习(ML)的旁路分析，它的优势是非侵入性方法，以及在无芯片设置下有效地检测HTS。在本文中，我们通过旁路分析来质疑基于ML的HT检测的可信性。我们引入了一种HT混淆(HTO)方法来允许HTS绕过这种检测方法。我们方法的一个关键方面不是通过模拟的对抗性痕迹在理论上误导模型，而是设计和实现对抗性噪声作为电路的一部分，与HT一起。我们详细介绍了ASIC和FPGA的HTO方法，并使用TrustHub Benchmark对我们的方法进行了评估。有趣的是，我们发现，对于ASIC设计，HTO可以只用一个晶体管来实现，以产生对抗性的功率痕迹，可以100%的效率欺骗防御。我们还使用两种不同的变型在Spartan 6 Xilinx FPGA上高效地实现了我们的方法：(I)基于DSP片的设计，和(Ii)基于环形振荡器的设计。此外，我们评估了像谱域分析这样的对策的效率，并且我们证明了自适应攻击者仍然可以通过用频谱噪声预算来约束设计来设计规避HTO。此外，尽管对抗性训练(AT)提供了对规避HTS的更高保护，但AT模型遭受了相当大的实用损失，潜在地使它们不适合这种安全应用。我们相信，这项研究代表着在了解和利用硬件安全环境中的ML漏洞方面迈出了重要的一步，我们将所有资源和设计在网上公开提供：https://dev.d18uu4lqwhbmka.amplifyapp.com



## **23. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

用于虚假新闻检测的对抗性数据中毒：如何使模型在不修改目标新闻的情况下对其进行错误分类 cs.LG

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.15228v2) [paper-pdf](http://arxiv.org/pdf/2312.15228v2)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccini, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.

摘要: 假新闻检测模型对打击虚假信息至关重要，但可以通过对抗性攻击来操纵。在这份立场文件中，我们分析了攻击者如何在不能操纵原始目标新闻的情况下，损害在线学习检测器在特定新闻内容上的性能。在某些情况下，例如社交网络，攻击者无法完全控制所有信息，这种情况确实很有可能发生。因此，我们展示了攻击者如何潜在地将中毒数据引入训练数据以操纵在线学习方法的行为。我们的初步发现揭示了基于复杂性和攻击类型的Logistic回归模型的不同易感性。



## **24. Attacks in Adversarial Machine Learning: A Systematic Survey from the Life-cycle Perspective**

对抗性机器学习中的攻击：基于生命周期的系统综述 cs.LG

35 pages, 4 figures, 10 tables, 313 reference papers

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2302.09457v2) [paper-pdf](http://arxiv.org/pdf/2302.09457v2)

**Authors**: Baoyuan Wu, Zihao Zhu, Li Liu, Qingshan Liu, Zhaofeng He, Siwei Lyu

**Abstract**: Adversarial machine learning (AML) studies the adversarial phenomenon of machine learning, which may make inconsistent or unexpected predictions with humans. Some paradigms have been recently developed to explore this adversarial phenomenon occurring at different stages of a machine learning system, such as backdoor attack occurring at the pre-training, in-training and inference stage; weight attack occurring at the post-training, deployment and inference stage; adversarial attack occurring at the inference stage. However, although these adversarial paradigms share a common goal, their developments are almost independent, and there is still no big picture of AML. In this work, we aim to provide a unified perspective to the AML community to systematically review the overall progress of this field. We firstly provide a general definition about AML, and then propose a unified mathematical framework to covering existing attack paradigms. According to the proposed unified framework, we build a full taxonomy to systematically categorize and review existing representative methods for each paradigm. Besides, using this unified framework, it is easy to figure out the connections and differences among different attack paradigms, which may inspire future researchers to develop more advanced attack paradigms. Finally, to facilitate the viewing of the built taxonomy and the related literature in adversarial machine learning, we further provide a website, \ie, \url{http://adversarial-ml.com}, where the taxonomies and literature will be continuously updated.

摘要: 对抗性机器学习(AML)研究机器学习中的对抗性现象，它可能会做出与人类不一致或意想不到的预测。最近已经发展了一些范例来探索这种发生在机器学习系统的不同阶段的对抗性现象，例如发生在预训练、训练中和推理阶段的后门攻击；发生在训练后、部署和推理阶段的权重攻击；发生在推理阶段的对抗性攻击。然而，尽管这些对抗性范式有着共同的目标，但它们的发展几乎是独立的，仍然没有AML的大图景。在这项工作中，我们旨在为AML社区提供一个统一的视角，以便系统地审查这一领域的整体进展。我们首先给出了AML的一般定义，然后提出了一个统一的数学框架来覆盖现有的攻击范型。根据提出的统一框架，我们建立了一个完整的分类法，对每个范式的现有代表性方法进行了系统的分类和审查。此外，使用这个统一的框架，可以很容易地找出不同攻击范式之间的联系和区别，这可能会启发未来的研究人员开发更高级的攻击范式。最后，为了便于在对抗性机器学习中查看所建立的分类和相关文献，我们还提供了一个网站，\即\url{http://adversarial-ml.com}，，在那里分类和文献将不断更新。



## **25. Fast Certification of Vision-Language Models Using Incremental Randomized Smoothing**

基于增量随机平滑的视觉语言模型快速认证 cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2311.09024v2) [paper-pdf](http://arxiv.org/pdf/2311.09024v2)

**Authors**: A K Nirala, A Joshi, C Hegde, S Sarkar

**Abstract**: A key benefit of deep vision-language models such as CLIP is that they enable zero-shot open vocabulary classification; the user has the ability to define novel class labels via natural language prompts at inference time. However, while CLIP-based zero-shot classifiers have demonstrated competitive performance across a range of domain shifts, they remain highly vulnerable to adversarial attacks. Therefore, ensuring the robustness of such models is crucial for their reliable deployment in the wild.   In this work, we introduce Open Vocabulary Certification (OVC), a fast certification method designed for open-vocabulary models like CLIP via randomized smoothing techniques. Given a base "training" set of prompts and their corresponding certified CLIP classifiers, OVC relies on the observation that a classifier with a novel prompt can be viewed as a perturbed version of nearby classifiers in the base training set. Therefore, OVC can rapidly certify the novel classifier using a variation of incremental randomized smoothing. By using a caching trick, we achieve approximately two orders of magnitude acceleration in the certification process for novel prompts. To achieve further (heuristic) speedups, OVC approximates the embedding space at a given input using a multivariate normal distribution bypassing the need for sampling via forward passes through the vision backbone. We demonstrate the effectiveness of OVC on through experimental evaluation using multiple vision-language backbones on the CIFAR-10 and ImageNet test datasets.

摘要: 深度视觉语言模型（如CLIP）的一个关键好处是它们能够实现零触发开放词汇分类;用户能够在推理时通过自然语言提示定义新的类标签。然而，尽管基于CLIP的零触发分类器在一系列领域转移中表现出了竞争力，但它们仍然非常容易受到对抗性攻击。因此，确保这些模型的鲁棒性对于它们在野外的可靠部署至关重要。   在这项工作中，我们引入了开放词汇认证（OVC），这是一种通过随机平滑技术为CLIP等开放词汇模型设计的快速认证方法。给定一个基本的“训练”提示集及其相应的认证CLIP分类器，OVC依赖于这样的观察，即具有新提示的分类器可以被视为基本训练集中附近分类器的扰动版本。因此，OVC可以使用增量随机平滑的变化来快速地认证新的分类器。通过使用缓存技巧，我们实现了大约两个数量级的加速认证过程中的新提示。为了实现进一步的（启发式）加速，OVC使用多变量正态分布来近似给定输入的嵌入空间，从而绕过了通过视觉骨干的前向传递进行采样的需要。我们通过在CIFAR-10和ImageNet测试数据集上使用多个视觉语言主干进行实验评估，证明了OVC的有效性。



## **26. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整的定向攻击 cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.01886v2) [paper-pdf](http://arxiv.org/pdf/2312.01886v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。本文提出了一种新颖实用的灰盒攻击方案，即攻击者只能访问受害者LVLM的可视编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大型语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(InstructTA)，对具有高可转移性的LVLMS进行定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高可转移性，我们用转译自LLM的指令扩充了指令$\boldSymbol{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。



## **27. DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification**

DiffAttack：针对基于扩散的对抗性净化的逃避攻击 cs.CR

Accepted to NeurIPS 2023

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2311.16124v2) [paper-pdf](http://arxiv.org/pdf/2311.16124v2)

**Authors**: Mintong Kang, Dawn Song, Bo Li

**Abstract**: Diffusion-based purification defenses leverage diffusion models to remove crafted perturbations of adversarial examples and achieve state-of-the-art robustness. Recent studies show that even advanced attacks cannot break such defenses effectively, since the purification process induces an extremely deep computational graph which poses the potential problem of gradient obfuscation, high memory cost, and unbounded randomness. In this paper, we propose a unified framework DiffAttack to perform effective and efficient attacks against diffusion-based purification defenses, including both DDPM and score-based approaches. In particular, we propose a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation to tackle the problem of vanishing/exploding gradients. We also provide a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient backpropagation. We validate the attack effectiveness of DiffAttack compared with existing adaptive attacks on CIFAR-10 and ImageNet. We show that DiffAttack decreases the robust accuracy of models compared with SOTA attacks by over 20% on CIFAR-10 under $\ell_\infty$ attack $(\epsilon=8/255)$, and over 10% on ImageNet under $\ell_\infty$ attack $(\epsilon=4/255)$. We conduct a series of ablations studies, and we find 1) DiffAttack with the deviated-reconstruction loss added over uniformly sampled time steps is more effective than that added over only initial/final steps, and 2) diffusion-based purification with a moderate diffusion length is more robust under DiffAttack.

摘要: 基于扩散的净化防御利用扩散模型来消除对抗性示例的精心设计的扰动，并实现最先进的健壮性。最近的研究表明，即使是高级攻击也不能有效地打破这种防御，因为净化过程会产生非常深的计算图，这会带来梯度混淆、高存储成本和无限随机性的潜在问题。在本文中，我们提出了一个统一的框架DiffAttack来执行对基于扩散的净化防御的有效和高效的攻击，包括DDPM和基于分数的方法。特别是，我们提出了中间扩散步骤的偏差重建损失，以导致不准确的密度梯度估计，以解决梯度消失/爆炸的问题。我们还提出了一种分段向前-向后算法，这导致了内存效率较高的梯度反向传播。与已有的针对CIFAR-10和ImageNet的自适应攻击相比，验证了DiffAttack的攻击有效性。我们发现，与SOTA攻击相比，DiffAttack在$\ell_\inty$攻击$(\epsilon=8/255)$下对CIFAR-10的鲁棒性准确率降低了20%以上，在ImageNet上在$\ell_\inty$攻击$(\epsilon=4/255)$下降低了10%以上。我们进行了一系列的烧蚀研究，我们发现1)DiffAttack在均匀采样的时间步长上添加偏差重建损耗比仅在初始/最终步骤上添加的DiffAttack更有效，2)在DiffAttack下，具有适度扩散长度的基于扩散的净化更稳健。



## **28. CBD: A Certified Backdoor Detector Based on Local Dominant Probability**

CBD：一种基于局部支配概率的认证后门检测器 cs.LG

Accepted to NeurIPS 2023

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2310.17498v2) [paper-pdf](http://arxiv.org/pdf/2310.17498v2)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.

摘要: 后门攻击是深度神经网络的常见威胁。在测试过程中，嵌入后门触发器的样本将被后门模型错误分类为对手目标，而没有后门触发器的样本将被正确分类。在本文中，我们提出了第一个认证的后门检测器(CBD)，它基于我们提出的统计局部主导概率的一种新的、可调整的共形预测方案。对于被检查的任何分类器，CBD提供了1)检测推理，2)保证攻击对于相同的分类域是可检测的条件，以及3)错误阳性率的概率上界。我们的理论结果表明，触发对测试时间噪声更具弹性并且扰动幅度更小的攻击更有可能被检测到。此外，我们在四个基准数据集上进行了广泛的实验，考虑了不同的后门类型，如BadNet、CB和Blend。CBD实现了与最先进的检测器相当甚至更高的检测精度，此外，它还提供检测认证。值得注意的是，对于攻击成功率超过90%的随机扰动触发的后门攻击，CBD在GTSRB、SVHN、CIFAR-10和TinyImageNet四个基准数据集上分别获得了100(98)、100(84)、98(98)和72(40)的经验(认证)检测真阳性，假阳性率较低。



## **29. DeepTaster: Adversarial Perturbation-Based Fingerprinting to Identify Proprietary Dataset Use in Deep Neural Networks**

DeepTaster：基于对抗性扰动的指纹识别在深度神经网络中使用的专有数据集 cs.CR

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2211.13535v2) [paper-pdf](http://arxiv.org/pdf/2211.13535v2)

**Authors**: Seonhye Park, Alsharif Abuadbba, Shuo Wang, Kristen Moore, Yansong Gao, Hyoungshick Kim, Surya Nepal

**Abstract**: Training deep neural networks (DNNs) requires large datasets and powerful computing resources, which has led some owners to restrict redistribution without permission. Watermarking techniques that embed confidential data into DNNs have been used to protect ownership, but these can degrade model performance and are vulnerable to watermark removal attacks. Recently, DeepJudge was introduced as an alternative approach to measuring the similarity between a suspect and a victim model. While DeepJudge shows promise in addressing the shortcomings of watermarking, it primarily addresses situations where the suspect model copies the victim's architecture. In this study, we introduce DeepTaster, a novel DNN fingerprinting technique, to address scenarios where a victim's data is unlawfully used to build a suspect model. DeepTaster can effectively identify such DNN model theft attacks, even when the suspect model's architecture deviates from the victim's. To accomplish this, DeepTaster generates adversarial images with perturbations, transforms them into the Fourier frequency domain, and uses these transformed images to identify the dataset used in a suspect model. The underlying premise is that adversarial images can capture the unique characteristics of DNNs built with a specific dataset. To demonstrate the effectiveness of DeepTaster, we evaluated the effectiveness of DeepTaster by assessing its detection accuracy on three datasets (CIFAR10, MNIST, and Tiny-ImageNet) across three model architectures (ResNet18, VGG16, and DenseNet161). We conducted experiments under various attack scenarios, including transfer learning, pruning, fine-tuning, and data augmentation. Specifically, in the Multi-Architecture Attack scenario, DeepTaster was able to identify all the stolen cases across all datasets, while DeepJudge failed to detect any of the cases.

摘要: 训练深度神经网络(DNN)需要大数据集和强大的计算资源，这导致一些所有者在未经许可的情况下限制重新分发。将机密数据嵌入到DNN中的水印技术已被用于保护所有权，但这些技术会降低模型性能，并且容易受到水印移除攻击。最近，深度法官被引入作为衡量嫌疑人和受害者模型之间相似性的另一种方法。虽然DeepJustice在解决水印的缺点方面表现出了希望，但它主要解决了可疑模型复制受害者的体系结构的情况。在这项研究中，我们引入了DeepTaster，一种新的DNN指纹识别技术，以解决受害者的数据被非法用于建立嫌疑人模型的情况。DeepTaster可以有效地识别这种DNN模型盗窃攻击，即使当可疑模型的架构与受害者的架构不同时也是如此。为了实现这一点，DeepTaster生成带有扰动的对抗性图像，将它们转换到傅立叶频域，并使用这些转换后的图像来识别可疑模型中使用的数据集。其基本前提是敌意图像可以捕捉到用特定数据集构建的DNN的独特特征。为了证明DeepTaster的有效性，我们通过评估DeepTaster在三个模型体系结构(ResNet18、VGG16和DenseNet161)的三个数据集(CIFAR10、MNIST和Tiny-ImageNet)上的检测精度来评估DeepTaster的有效性。我们在不同的攻击场景下进行了实验，包括迁移学习、剪枝、微调和数据增强。具体地说，在多架构攻击场景中，DeepTaster能够识别所有数据集的所有被盗案例，而DeepJustice未能检测到任何案例。



## **30. Integrated Cyber-Physical Resiliency for Power Grids under IoT-Enabled Dynamic Botnet Attacks**

物联网动态僵尸网络攻击下电网的综合网络物理弹性 eess.SY

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01963v1) [paper-pdf](http://arxiv.org/pdf/2401.01963v1)

**Authors**: Yuhan Zhao, Juntao Chen, Quanyan Zhu

**Abstract**: The wide adoption of Internet of Things (IoT)-enabled energy devices improves the quality of life, but simultaneously, it enlarges the attack surface of the power grid system. The adversary can gain illegitimate control of a large number of these devices and use them as a means to compromise the physical grid operation, a mechanism known as the IoT botnet attack. This paper aims to improve the resiliency of cyber-physical power grids to such attacks. Specifically, we use an epidemic model to understand the dynamic botnet formation, which facilitates the assessment of the cyber layer vulnerability of the grid. The attacker aims to exploit this vulnerability to enable a successful physical compromise, while the system operator's goal is to ensure a normal operation of the grid by mitigating cyber risks. We develop a cross-layer game-theoretic framework for strategic decision-making to enhance cyber-physical grid resiliency. The cyber-layer game guides the system operator on how to defend against the botnet attacker as the first layer of defense, while the dynamic game strategy at the physical layer further counteracts the adversarial behavior in real time for improved physical resilience. A number of case studies on the IEEE-39 bus system are used to corroborate the devised approach.

摘要: 物联网(IoT)能源设备的广泛应用在提高生活质量的同时，也扩大了电网系统的攻击面。敌手可以非法控制大量这样的设备，并将它们用作危害物理电网操作的手段，这种机制称为物联网僵尸网络攻击。本文旨在提高网络物理电网对此类攻击的恢复能力。具体地说，我们使用流行病模型来理解僵尸网络的动态形成，这有助于评估网格的网络层脆弱性。攻击者的目标是利用该漏洞进行成功的物理危害，而系统运营商的目标是通过降低网络风险来确保电网的正常运行。我们开发了一个跨层博弈论的战略决策框架，以增强网络物理网格的弹性。网络层游戏指导系统操作员如何防御僵尸网络攻击者，作为第一层防御，而物理层的动态游戏策略进一步实时抵消敌对行为，以提高身体弹性。在IEEE-39母线系统上的一些案例研究被用来证实所设计的方法。



## **31. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

从网络威胁情报报告中挖掘时态攻击模式 cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.

摘要: 防御网络攻击需要从业者对高级别的对手行为进行操作。网络威胁情报（CTI）报告对过去的网络攻击事件描述了恶意行为的时间链。为了避免重复的网络攻击事件，从业人员必须主动识别和防御重复的行动链-我们称之为时间攻击模式。自动挖掘行为之间的模式提供了关于过去网络攻击的对手行为的结构化和可操作的信息。本文的目标是通过从网络威胁情报报告中挖掘时间攻击模式，帮助安全从业人员优先考虑和主动防御网络攻击。为此，我们提出了ChronoCTI，这是一个自动化的管道，用于从过去网络攻击的网络威胁情报（CTI）报告中挖掘时间攻击模式。为了构建ChronoCTI，我们构建了时间攻击模式的真实数据集，并应用了最先进的大型语言模型、自然语言处理和机器学习技术。我们将ChronoCTI应用于一组713份CTI报告，其中我们确定了124种时间攻击模式-我们将其分为9种模式类别。我们发现，最普遍的模式类别是欺骗受害者用户执行恶意代码发起攻击，然后绕过受害者网络中的反恶意软件系统。根据观察到的模式，我们建议组织对用户进行网络安全最佳实践培训，引入功能有限的不可变操作系统，并实施多用户身份验证。此外，我们提倡从业人员利用ChronoCTI的自动挖掘功能，并针对反复出现的攻击模式设计对策。



## **32. Attackers reveal their arsenal: An investigation of adversarial techniques in CTI reports**

攻击者展示他们的武器库：CTI报告中的对抗性技术调查 cs.CR

This version is submitted to ACM Transactions on Privacy and  Security. This version is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01865v1) [paper-pdf](http://arxiv.org/pdf/2401.01865v1)

**Authors**: Md Rayhanur Rahman, Setu Kumar Basak, Rezvan Mahdavi Hezaveh, Laurie Williams

**Abstract**: Context: Cybersecurity vendors often publish cyber threat intelligence (CTI) reports, referring to the written artifacts on technical and forensic analysis of the techniques used by the malware in APT attacks. Objective: The goal of this research is to inform cybersecurity practitioners about how adversaries form cyberattacks through an analysis of adversarial techniques documented in cyberthreat intelligence reports. Dataset: We use 594 adversarial techniques cataloged in MITRE ATT\&CK. We systematically construct a set of 667 CTI reports that MITRE ATT\&CK used as citations in the descriptions of the cataloged adversarial techniques. Methodology: We analyze the frequency and trend of adversarial techniques, followed by a qualitative analysis of the implementation of techniques. Next, we perform association rule mining to identify pairs of techniques recurring in APT attacks. We then perform qualitative analysis to identify the underlying relations among the techniques in the recurring pairs. Findings: The set of 667 CTI reports documents 10,370 techniques in total, and we identify 19 prevalent techniques accounting for 37.3\% of documented techniques. We also identify 425 statistically significant recurring pairs and seven types of relations among the techniques in these pairs. The top three among the seven relationships suggest that techniques used by the malware inter-relate with one another in terms of (a) abusing or affecting the same system assets, (b) executing in sequences, and (c) overlapping in their implementations. Overall, the study quantifies how adversaries leverage techniques through malware in APT attacks based on publicly reported documents. We advocate organizations prioritize their defense against the identified prevalent techniques and actively hunt for potential malicious intrusion based on the identified pairs of techniques.

摘要: 背景：网络安全供应商经常发布网络威胁情报（CTI）报告，指的是对恶意软件在APT攻击中使用的技术进行技术和取证分析的书面工件。目的：这项研究的目标是通过分析网络威胁情报报告中记录的对抗技术，告知网络安全从业人员对手如何形成网络攻击。数据集：我们使用了MITRE ATT\&CK中收录的594种对抗技术。我们系统地构建了一组667个CTI报告，MITRE ATT\&CK在描述编目对抗技术时用作引文。方法：我们分析了对抗性技术的频率和趋势，然后对技术的实施进行了定性分析。接下来，我们进行关联规则挖掘，以确定对APT攻击中经常出现的技术。然后，我们进行定性分析，以确定潜在的关系之间的技术在循环对。发现：667份CTI报告共记录了10,370项技术，其中19项技术占记录技术的37.3%。我们还确定了425个统计上显着的经常性对和七种类型的技术之间的关系，在这些对。七种关系中的前三种表明恶意软件使用的技术在以下方面相互关联：（a）滥用或影响相同的系统资产，（b）按顺序执行，以及（c）在实现中重叠。总的来说，该研究量化了攻击者如何根据公开报告的文档在APT攻击中通过恶意软件利用技术。我们提倡组织优先考虑对已识别的流行技术的防御，并根据已识别的技术对积极寻找潜在的恶意入侵。



## **33. Locally Differentially Private Embedding Models in Distributed Fraud Prevention Systems**

分布式防骗系统中的局部差分私有嵌入模型 cs.CR

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.02450v1) [paper-pdf](http://arxiv.org/pdf/2401.02450v1)

**Authors**: Iker Perez, Jason Wong, Piotr Skalski, Stuart Burrell, Richard Mortier, Derek McAuley, David Sutton

**Abstract**: Global financial crime activity is driving demand for machine learning solutions in fraud prevention. However, prevention systems are commonly serviced to financial institutions in isolation, and few provisions exist for data sharing due to fears of unintentional leaks and adversarial attacks. Collaborative learning advances in finance are rare, and it is hard to find real-world insights derived from privacy-preserving data processing systems. In this paper, we present a collaborative deep learning framework for fraud prevention, designed from a privacy standpoint, and awarded at the recent PETs Prize Challenges. We leverage latent embedded representations of varied-length transaction sequences, along with local differential privacy, in order to construct a data release mechanism which can securely inform externally hosted fraud and anomaly detection models. We assess our contribution on two distributed data sets donated by large payment networks, and demonstrate robustness to popular inference-time attacks, along with utility-privacy trade-offs analogous to published work in alternative application domains.

摘要: 全球金融犯罪活动正在推动对预防欺诈的机器学习解决方案的需求。然而，预防系统通常是单独向金融机构提供服务的，由于担心无意泄露和对抗性攻击，几乎没有关于数据共享的规定。金融领域的协作学习进展非常罕见，而且很难从保护隐私的数据处理系统中找到现实世界的见解。在这篇文章中，我们提出了一个协作式深度学习框架，用于预防欺诈，从隐私的角度设计，并在最近的PETS奖挑战中获奖。我们利用可变长度交易序列的潜在嵌入表示以及本地差异隐私来构建一种数据发布机制，该机制可以安全地通知外部托管的欺诈和异常检测模型。我们评估了我们在大型支付网络捐赠的两个分布式数据集上的贡献，并展示了对流行的推理时间攻击的健壮性，以及类似于在替代应用领域发布的工作的实用隐私权衡。



## **34. Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement**

基于注意力精化的抗补丁攻击的稳健语义分割 cs.CV

30 pages, 3 figures, 12 tables

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01750v1) [paper-pdf](http://arxiv.org/pdf/2401.01750v1)

**Authors**: Zheng Yuan, Jie Zhang, Yude Wang, Shiguang Shan, Xilin Chen

**Abstract**: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones. However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch. To address this issue, in this paper, we propose a Robust Attention Mechanism (RAM) to improve the robustness of the semantic segmentation model, which can notably relieve the vulnerability against patch-based attacks. Compared to the vallina attention mechanism, RAM introduces two novel modules called Max Attention Suppression and Random Attention Dropout, both of which aim to refine the attention matrix and limit the influence of a single adversarial patch on the semantic segmentation results of other positions. Extensive experiments demonstrate the effectiveness of our RAM to improve the robustness of semantic segmentation models against various patch-based attack methods under different attack settings.

摘要: 近年来，注意机制在各种视觉任务中被证明是有效的。在语义分割任务中，注意力机制被应用于各种方法，包括卷积神经网络(CNN)和视觉转换器(VIT)作为骨干的情况。然而，我们观察到注意机制很容易受到基于补丁的对抗性攻击。通过对有效接受场的分析，我们将其归因于全球注意带来的广泛接受场可能导致对抗性斑块的传播。针对这一问题，本文提出了一种健壮的注意力机制(RAM)来提高语义分割模型的健壮性，该机制可以显著缓解语义分割模型对基于补丁攻击的脆弱性。与Vallina注意机制相比，RAM引入了两个新的模块：最大注意抑制和随机注意丢弃，这两个模块的目的都是为了细化注意矩阵，并限制单个敌意补丁对其他位置语义分割结果的影响。大量实验表明，在不同的攻击环境下，该算法能够有效地提高语义分割模型对各种基于补丁的攻击方法的稳健性。



## **35. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

用于自动说话人确认的空中对抗扰动神经重放模拟器的初步研究 cs.SD

Accepted in ICASSP 2024

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2310.05354v4) [paper-pdf](http://arxiv.org/pdf/2310.05354v4)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.

摘要: 在过去的几年里，深度学习发展了自动说话人确认(ASV)。虽然众所周知，基于深度学习的ASV系统在数字访问中容易受到敌意攻击，但在涉及重播过程(即空中重播)的物理访问环境中，很少有关于对抗性攻击的研究。空中攻击包括扬声器、麦克风和影响声波移动的重放环境。我们的初步实验证实，重放过程会影响空中攻击性能的有效性。本研究对利用神经重放模拟器来提高空中对抗攻击的稳健性进行了初步的研究。这是通过使用神经波形合成器来模拟在估计对抗性扰动时的重播过程来实现的。在ASVspoof2019数据集上进行的实验证实，神经重放模拟器可以显著提高空中对抗性攻击的成功率。这引起了人们对物理访问应用中说话人验证的对抗性攻击的关注。



## **36. Will 6G be Semantic Communications? Opportunities and Challenges from Task Oriented and Secure Communications to Integrated Sensing**

6G会成为语义通信吗？从任务导向和安全通信到集成传感的机遇和挑战 cs.NI

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01531v1) [paper-pdf](http://arxiv.org/pdf/2401.01531v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Aylin Yener, Sennur Ulukus

**Abstract**: This paper explores opportunities and challenges of task (goal)-oriented and semantic communications for next-generation (NextG) communication networks through the integration of multi-task learning. This approach employs deep neural networks representing a dedicated encoder at the transmitter and multiple task-specific decoders at the receiver, collectively trained to handle diverse tasks including semantic information preservation, source input reconstruction, and integrated sensing and communications. To extend the applicability from point-to-point links to multi-receiver settings, we envision the deployment of decoders at various receivers, where decentralized learning addresses the challenges of communication load and privacy concerns, leveraging federated learning techniques that distribute model updates across decentralized nodes. However, the efficacy of this approach is contingent on the robustness of the employed deep learning models. We scrutinize potential vulnerabilities stemming from adversarial attacks during both training and testing phases. These attacks aim to manipulate both the inputs at the encoder at the transmitter and the signals received over the air on the receiver side, highlighting the importance of fortifying semantic communications against potential multi-domain exploits. Overall, the joint and robust design of task-oriented communications, semantic communications, and integrated sensing and communications in a multi-task learning framework emerges as the key enabler for context-aware, resource-efficient, and secure communications ultimately needed in NextG network systems.

摘要: 通过整合多任务学习，探索面向任务(目标)和语义通信的下一代通信网络的机遇和挑战。这种方法采用深度神经网络，在发送端代表一个专用编码器，在接收端代表多个特定于任务的解码器，共同训练以处理包括语义信息保存、源输入重建以及集成传感和通信在内的各种任务。为了将适用性从点对点链路扩展到多接收器设置，我们设想在不同的接收器上部署解码器，其中分散学习利用跨分散节点分发模型更新的联合学习技术来解决通信负载和隐私问题的挑战。然而，这种方法的有效性取决于所采用的深度学习模型的稳健性。我们在培训和测试阶段仔细检查来自对抗性攻击的潜在漏洞。这些攻击的目的是同时操纵发送器编码器的输入和接收器端通过空中接收的信号，突显加强语义通信以抵御潜在的多域利用的重要性。总体而言，面向任务的通信、语义通信以及多任务学习框架中的集成传感和通信的联合稳健设计成为下一代网络系统最终需要的情景感知、资源高效和安全通信的关键推动因素。



## **37. JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example**

JMA：一种构造近似最优目标对抗实例的通用算法 cs.LG

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01199v1) [paper-pdf](http://arxiv.org/pdf/2401.01199v1)

**Authors**: Benedetta Tondi, Wei Guo, Mauro Barni

**Abstract**: Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, we propose a more general, theoretically sound, targeted attack that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm provides an optimal solution to a linearized version of the adversarial example problem originally introduced by Szegedy et al. \cite{szegedy2013intriguing}. The experiments we carried out confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, the JMA attack is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in a complex multilabel classification scenario with 20 labels, a capability that is out of reach of all the attacks proposed so far. As a further advantage, the JMA attack usually requires very few iterations, thus resulting more efficient than existing methods.

摘要: 到目前为止，大多数针对深度学习分类器提出的针对性对抗性示例的方法都是非常次优的，并且通常依赖于增加目标类的可能性，因此隐含地专注于独热编码设置。在本文中，我们提出了一个更一般的，理论上健全的，有针对性的攻击，采取最小化的雅可比诱导马氏距离（JMA）项，考虑到努力（在输入空间）所需的潜在空间表示的输入样本在给定的方向移动。最小化是解决利用沃尔夫对偶定理，减少问题的解决方案的非负最小二乘（NNLS）问题。所提出的算法提供了一个最佳的解决方案，最初介绍了Szegedy等人的对抗性的例子问题的线性化版本。\cite{szegedy 2013 intriguing}。我们进行的实验证实了所提出的攻击，这是被证明是有效的各种各样的输出编码方案下的一般性。值得注意的是，JMA攻击在多标签分类场景中也是有效的，能够在具有20个标签的复杂多标签分类场景中诱导多达一半标签的有针对性的修改，这是迄今为止提出的所有攻击都无法达到的能力。作为另一个优点，JMA攻击通常需要很少的迭代，因此比现有方法更有效。



## **38. Dual Teacher Knowledge Distillation with Domain Alignment for Face Anti-spoofing**

基于领域对齐的双教师知识提取人脸反欺骗算法 cs.CV

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01102v1) [paper-pdf](http://arxiv.org/pdf/2401.01102v1)

**Authors**: Zhe Kong, Wentian Zhang, Tao Wang, Kaihao Zhang, Yuexiang Li, Xiaoying Tang, Wenhan Luo

**Abstract**: Face recognition systems have raised concerns due to their vulnerability to different presentation attacks, and system security has become an increasingly critical concern. Although many face anti-spoofing (FAS) methods perform well in intra-dataset scenarios, their generalization remains a challenge. To address this issue, some methods adopt domain adversarial training (DAT) to extract domain-invariant features. However, the competition between the encoder and the domain discriminator can cause the network to be difficult to train and converge. In this paper, we propose a domain adversarial attack (DAA) method to mitigate the training instability problem by adding perturbations to the input images, which makes them indistinguishable across domains and enables domain alignment. Moreover, since models trained on limited data and types of attacks cannot generalize well to unknown attacks, we propose a dual perceptual and generative knowledge distillation framework for face anti-spoofing that utilizes pre-trained face-related models containing rich face priors. Specifically, we adopt two different face-related models as teachers to transfer knowledge to the target student model. The pre-trained teacher models are not from the task of face anti-spoofing but from perceptual and generative tasks, respectively, which implicitly augment the data. By combining both DAA and dual-teacher knowledge distillation, we develop a dual teacher knowledge distillation with domain alignment framework (DTDA) for face anti-spoofing. The advantage of our proposed method has been verified through extensive ablation studies and comparison with state-of-the-art methods on public datasets across multiple protocols.

摘要: 人脸识别系统由于易受不同表现形式的攻击而引起了人们的关注，系统安全已经成为一个越来越关键的问题。尽管许多Face反欺骗(FAS)方法在数据集内场景中执行得很好，但它们的泛化仍然是一个挑战。为了解决这一问题，一些方法采用领域对抗训练(DAT)来提取领域不变特征。然而，编码器和域鉴别器之间的竞争会导致网络难以训练和收敛。在本文中，我们提出了一种域对抗攻击(DAA)方法，通过在输入图像中添加扰动来缓解训练不稳定性问题，从而使输入图像无法跨域区分并实现域对齐。此外，由于在有限数据和攻击类型上训练的模型不能很好地推广到未知攻击，我们提出了一种双重感知和生成的人脸反欺骗知识蒸馏框架，该框架利用包含丰富人脸先验的预先训练的人脸相关模型。具体地说，我们采用了两种不同的面子相关模式作为教师向目标学生模式传递知识。预先训练的教师模型不是来自面子反恶搞任务，而是来自知觉任务和生成性任务，这两个任务分别内隐地增加了数据。将DAA和双师知识提炼相结合，提出了一种双师知识提炼的领域对齐框架(DTDA)，用于人脸防欺骗。我们提出的方法的优势已经通过广泛的消融研究以及在多个协议的公共数据集上与最新方法的比较得到了验证。



## **39. Imperio: Language-Guided Backdoor Attacks for Arbitrary Model Control**

Imperio：语言引导的任意模型控制后门攻击 cs.CR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01085v1) [paper-pdf](http://arxiv.org/pdf/2401.01085v1)

**Authors**: Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: Revolutionized by the transformer architecture, natural language processing (NLP) has received unprecedented attention. While advancements in NLP models have led to extensive research into their backdoor vulnerabilities, the potential for these advancements to introduce new backdoor threats remains unexplored. This paper proposes Imperio, which harnesses the language understanding capabilities of NLP models to enrich backdoor attacks. Imperio provides a new model control experience. It empowers the adversary to control the victim model with arbitrary output through language-guided instructions. This is achieved using a language model to fuel a conditional trigger generator, with optimizations designed to extend its language understanding capabilities to backdoor instruction interpretation and execution. Our experiments across three datasets, five attacks, and nine defenses confirm Imperio's effectiveness. It can produce contextually adaptive triggers from text descriptions and control the victim model with desired outputs, even in scenarios not encountered during training. The attack maintains a high success rate across complex datasets without compromising the accuracy of clean inputs and also exhibits resilience against representative defenses. The source code is available at \url{https://khchow.com/Imperio}.

摘要: 自然语言处理(NLP)受到了变压器体系结构的革命性变革，受到了前所未有的关注。虽然NLP模型的进步导致了对其后门漏洞的广泛研究，但这些进步引入新的后门威胁的潜力仍未被发掘。本文提出了Imperio，它利用NLP模型的语言理解能力来丰富后门攻击。Imperio提供了全新的模型控制体验。它使攻击者能够通过语言引导的指令控制具有任意输出的受害者模型。这是通过使用语言模型来为条件触发生成器提供燃料来实现的，优化旨在将其语言理解能力扩展到后门指令解释和执行。我们在三个数据集、五个攻击和九个防御系统上的实验证实了Imperio的有效性。它可以从文本描述中产生上下文自适应触发，并用所需的输出控制受害者模型，即使在培训期间没有遇到的情况下也是如此。该攻击在复杂的数据集上保持了高的成功率，而不会影响干净输入的准确性，并且对典型的防御系统也表现出了韧性。源代码可在\url{https://khchow.com/Imperio}.



## **40. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

安全和性能，为什么不能两者兼而有之呢？针对AI软件部署异构性攻击的双目标优化模型压缩 cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.

摘要: 人工智能(AI)软件中的深度学习模型的规模正在迅速增长，阻碍了在资源受限的设备(如智能手机)上的大规模部署。为了缓解这个问题，人工智能软件压缩起到了至关重要的作用，其目标是在保持高性能的同时压缩模型大小。然而，大模型中的固有缺陷可能会被压缩的模型继承。这样的缺陷很容易被攻击者利用，因为压缩模型通常部署在大量设备中，而没有足够的保护。在本文中，我们旨在从安全-性能联合优化的角度解决安全模型压缩问题。具体地说，受软件工程中测试驱动开发(TDD)范式的启发，我们提出了一个称为SafeCompress的测试驱动稀疏训练框架。通过将攻击机制模拟为安全测试，SafeCompress可以按照动态稀疏训练范式自动将大模型压缩为小模型。然后，考虑到两种典型的异构性攻击机制，即黑盒成员关系推理攻击和白盒成员关系推理攻击，我们开发了两个具体的实例：BMIA-SafeCompress和WMIA-SafeCompress。此外，我们实现了另一个实例MMIA-SafeCompress，通过扩展SafeCompress来防御对手同时进行黑盒和白盒成员推理攻击的情况。我们在计算机视觉和自然语言处理任务的五个数据集上进行了广泛的实验。结果表明，该框架具有较好的通用性和有效性。我们还讨论了如何使SafeCompress适应除成员推理攻击之外的其他攻击，展示了SafeCompress的灵活性。



## **41. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

基于引导扩散的视觉感知推荐系统中的对抗性项目提升 cs.IR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2312.15826v3) [paper-pdf](http://arxiv.org/pdf/2312.15826v3)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Guanhua Ye, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.

摘要: 视觉感知推荐系统在视觉元素对用户潜在偏好的推断有重要作用的领域得到了广泛的应用。虽然加入视觉信息有望提高推荐的准确性和缓解冷启动问题，但必须指出的是，纳入物品图像可能会带来重大的安全挑战。一些已有的研究表明，物品提供者可以通过构建对抗性图像来操纵物品曝光率。然而，这些工作并不能揭示视觉感知推荐系统的真正弱点，因为(1)生成的敌意图像明显失真，使得人类很容易发现它们；(2)攻击的有效性在某些场景下是不一致的，甚至无效的。为了揭示视觉感知推荐系统在面对敌意图像时的真正弱点，提出了一种新的攻击方法--IPDGI(Item Promotion By Diffumation Generated Image)。具体地说，IPDGI使用引导扩散模型来生成敌意样本，旨在欺骗视觉感知的推荐系统。利用扩散模型精确模拟良性图像的分布，生成的对抗性图像与原始图像具有较高的保真度，保证了IPDGI的隐蔽性。为了验证我们提出的方法的有效性，我们在两个常用的电子商务推荐数据集(Amazon Beauty和Amazon Baby)上进行了广泛的实验，并使用几个典型的视觉感知推荐系统进行了实验。实验结果表明，我们的攻击方法在提升长尾(即不受欢迎)项的性能和生成对抗性图像的质量方面都有显著的提高。



## **42. Passive Inference Attacks on Split Learning via Adversarial Regularization**

基于对抗性正则化的分裂学习被动推理攻击 cs.CR

17 pages, 20 figures

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2310.10483v3) [paper-pdf](http://arxiv.org/pdf/2310.10483v3)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.

摘要: 分裂学习(Split Learning，SL)已成为传统联合学习的一种实用有效的替代方案。虽然以前攻击SL的尝试通常依赖于过于强烈的假设或目标明确、易于利用的模型，但我们寻求开发更实际的攻击。我们介绍了SDAR，这是一种针对SL的新型攻击框架，具有诚实但好奇的服务器。SDAR利用辅助数据和对抗性正则化学习客户私有模型的可解码模拟器，该模拟器可以有效地推断客户在香草SL下的私有特征，以及U形SL下的特征和标签。我们在两种配置下都进行了大量的实验，以验证我们提出的攻击的有效性。值得注意的是，在具有挑战性但实用的场景中，现有的被动攻击难以有效地重建客户端的私有数据，SDAR始终实现与主动攻击相当的攻击性能。在CIFAR-10上，在7的深度分裂水平上，SDAR实现了私有特征重建，在普通SL和U形SL上的均方误差都小于0.025，在U形背景下获得了98%以上的标签推理准确率，而现有的攻击无法产生非平凡的结果。



## **43. Channel Reciprocity Attacks Using Intelligent Surfaces with Non-Diagonal Phase Shifts**

基于非对角相移智能表面的信道互易攻击 eess.SP

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2309.11665v2) [paper-pdf](http://arxiv.org/pdf/2309.11665v2)

**Authors**: Haoyu Wang, Zhu Han, A. Lee Swindlehurst

**Abstract**: While reconfigurable intelligent surface (RIS) technology has been shown to provide numerous benefits to wireless systems, in the hands of an adversary such technology can also be used to disrupt communication links. This paper describes and analyzes an RIS-based attack on multi-antenna wireless systems that operate in time-division duplex mode under the assumption of channel reciprocity. In particular, we show how an RIS with a non-diagonal (ND) phase shift matrix (referred to here as an ND-RIS) can be deployed to maliciously break the channel reciprocity and hence degrade the downlink network performance. Such an attack is entirely passive and difficult to detect and counteract. We provide a theoretical analysis of the degradation in the sum ergodic rate that results when an arbitrary malicious ND-RIS is deployed and design an approach based on the genetic algorithm for optimizing the ND structure under partial knowledge of the available channel state information. Our simulation results validate the analysis and demonstrate that an ND-RIS channel reciprocity attack can dramatically reduce the downlink throughput.

摘要: 虽然可重构智能表面(RIS)技术已被证明为无线系统提供了许多好处，但在对手手中，这种技术也可能被用来中断通信链路。在信道互易性的假设下，描述和分析了一种基于RIS的对时分双工多天线无线系统的攻击。特别是，我们展示了如何部署具有非对角线(ND)相移矩阵的RIS(这里称为ND-RIS)来恶意破坏信道互易性，从而降低下行链路网络的性能。这种攻击完全是被动的，很难发现和反击。从理论上分析了任意恶意ND-RIS部署后导致的和遍历率的下降，并设计了一种在部分已知信道状态信息的情况下基于遗传算法的ND结构优化方法。我们的仿真结果验证了分析，并证明了ND-RIS信道互惠攻击可以显著降低下行链路吞吐量。



## **44. Is It Possible to Backdoor Face Forgery Detection with Natural Triggers?**

是否有可能使用自然触发器进行后门人脸伪造检测？ cs.CV

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00414v1) [paper-pdf](http://arxiv.org/pdf/2401.00414v1)

**Authors**: Xiaoxuan Han, Songlin Yang, Wei Wang, Ziwen He, Jing Dong

**Abstract**: Deep neural networks have significantly improved the performance of face forgery detection models in discriminating Artificial Intelligent Generated Content (AIGC). However, their security is significantly threatened by the injection of triggers during model training (i.e., backdoor attacks). Although existing backdoor defenses and manual data selection can mitigate those using human-eye-sensitive triggers, such as patches or adversarial noises, the more challenging natural backdoor triggers remain insufficiently researched. To further investigate natural triggers, we propose a novel analysis-by-synthesis backdoor attack against face forgery detection models, which embeds natural triggers in the latent space. We thoroughly study such backdoor vulnerability from two perspectives: (1) Model Discrimination (Optimization-Based Trigger): we adopt a substitute detection model and find the trigger by minimizing the cross-entropy loss; (2) Data Distribution (Custom Trigger): we manipulate the uncommon facial attributes in the long-tailed distribution to generate poisoned samples without the supervision from detection models. Furthermore, to completely evaluate the detection models towards the latest AIGC, we utilize both state-of-the-art StyleGAN and Stable Diffusion for trigger generation. Finally, these backdoor triggers introduce specific semantic features to the generated poisoned samples (e.g., skin textures and smile), which are more natural and robust. Extensive experiments show that our method is superior from three levels: (1) Attack Success Rate: ours achieves a high attack success rate (over 99%) and incurs a small model accuracy drop (below 0.2%) with a low poisoning rate (less than 3%); (2) Backdoor Defense: ours shows better robust performance when faced with existing backdoor defense methods; (3) Human Inspection: ours is less human-eye-sensitive from a comprehensive user study.

摘要: 深度神经网络显著提高了人脸伪造检测模型在区分人工智能生成内容（AIGC）方面的性能。然而，它们的安全性受到模型训练期间触发器注入的严重威胁（即，后门攻击）。虽然现有的后门防御和手动数据选择可以减轻那些使用人眼敏感触发器的攻击，例如补丁或对抗性噪声，但更具挑战性的自然后门触发器仍然没有得到充分的研究。为了进一步研究自然触发器，我们提出了一种新的分析合成后门攻击人脸伪造检测模型，它嵌入在潜在空间的自然触发器。我们从两个角度深入研究了这种后门漏洞：（1）模型识别（基于优化的触发器）：我们采用替代检测模型，通过最小化交叉熵损失来找到触发器;（2）数据分布（自定义触发器）：我们操纵长尾分布中的不常见面部属性来生成中毒样本，而无需检测模型的监督。此外，为了全面评估最新AIGC的检测模型，我们利用最先进的StyleGAN和Stable Diffusion进行触发生成。最后，这些后门触发器将特定的语义特征引入所生成的中毒样本（例如，皮肤纹理和微笑），这是更自然和强大的。大量的实验表明，我们的方法从三个层面上来说是优越的：（1）攻击成功率：我们的方法达到了很高的攻击成功率（超过99%），并导致模型精度下降（低于0.2%），中毒率低（不到3%）;（2）后门防御：当面对现有的后门防御方法时，我们表现出更好的鲁棒性能;（3）人工检测：从全面的用户研究来看，我们的眼睛不太敏感。



## **45. Dictionary Attack on IMU-based Gait Authentication**

基于IMU的步态认证字典攻击 cs.CR

12 pages, 9 figures, accepted at AISec23 colocated with ACM CCS,  November 30, 2023, Copenhagen, Denmark

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2309.11766v2) [paper-pdf](http://arxiv.org/pdf/2309.11766v2)

**Authors**: Rajesh Kumar, Can Isik, Chilukuri K. Mohan

**Abstract**: We present a novel adversarial model for authentication systems that use gait patterns recorded by the inertial measurement unit (IMU) built into smartphones. The attack idea is inspired by and named after the concept of a dictionary attack on knowledge (PIN or password) based authentication systems. In particular, this work investigates whether it is possible to build a dictionary of IMUGait patterns and use it to launch an attack or find an imitator who can actively reproduce IMUGait patterns that match the target's IMUGait pattern. Nine physically and demographically diverse individuals walked at various levels of four predefined controllable and adaptable gait factors (speed, step length, step width, and thigh-lift), producing 178 unique IMUGait patterns. Each pattern attacked a wide variety of user authentication models. The deeper analysis of error rates (before and after the attack) challenges the belief that authentication systems based on IMUGait patterns are the most difficult to spoof; further research is needed on adversarial models and associated countermeasures.

摘要: 我们提出了一种新的敌意认证系统模型，该模型使用智能手机内置的惯性测量单元(IMU)记录的步态模式。该攻击思想的灵感来自于对基于知识(PIN或密码)的身份验证系统的字典攻击的概念，并以此命名。特别是，这项工作调查是否有可能建立一个IMUGait图案词典，并使用它来发动攻击，或者找到一个模仿者，他可以主动复制与目标的IMUGait图案匹配的IMUGait图案。九个身体和人口统计学上不同的人在四个预定义的可控和可适应步态因素(速度、步长、步宽和大腿抬起)的不同水平上行走，产生了178个独特的IMU步态模式。每种模式都攻击了各种各样的用户身份验证模型。对错误率的深入分析(攻击前和攻击后)挑战了基于IMUGait模式的认证系统最难欺骗的观点；需要对敌意模型和相关对策进行进一步研究。



## **46. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

禁忌事实：骆驼2号中相互竞争的目标的调查 cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023; (v3:  clarified experimental details)

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2312.08793v3) [paper-pdf](http://arxiv.org/pdf/2312.08793v3)

**Authors**: Tony T. Wang, Miles Wang, Kaivalya Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .

摘要: 低收入国家经常面临相互竞争的压力(例如，有益与无害)。为了理解模型如何解决此类冲突，我们研究了关于禁止事实任务的Llama-2-Chat模型。具体地说，我们指示骆驼2号如实完成事实回忆声明，同时禁止它说出正确的答案。这经常使模型给出错误的答案。我们将Llama-2分解成1000多个成分，并根据它们对阻止正确答案的作用程度对每个成分进行排名。我们发现，总共大约35个组件就足以可靠地实现完全抑制行为。然而，这些组件具有相当大的异构性，许多组件使用错误的启发式方法进行操作。我们发现，其中一个启发式攻击可以通过手动设计的对抗性攻击来利用，我们称之为加利福尼亚州攻击。我们的结果突出了一些阻碍成功解释高级ML系统的障碍。项目网站为https://forbiddenfacts.github.io。



## **47. Explainability-Driven Leaf Disease Classification using Adversarial Training and Knowledge Distillation**

基于对抗性训练和知识提炼的可解释性叶部病害分类 cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00334v1) [paper-pdf](http://arxiv.org/pdf/2401.00334v1)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.

摘要: 这项工作的重点是植物叶病分类，并探讨了三个关键方面：对抗性训练，模型可解释性和模型压缩。通过对抗性训练增强了模型对对抗性攻击的鲁棒性，即使在存在威胁的情况下也能确保准确的分类。利用可解释性技术，我们可以深入了解模型的决策过程，提高信任度和透明度。此外，我们探索模型压缩技术，以优化计算效率，同时保持分类性能。通过我们的实验，我们确定在基准数据集上，鲁棒性可以是分类准确性的代价，对于常规测试，性能降低3%-20%，对于对抗性攻击测试，性能提高50%-70%。我们还证明，学生模型的计算效率可以提高15-25倍，性能略有下降，提取更复杂模型的知识。



## **48. Unraveling the Connections between Privacy and Certified Robustness in Federated Learning Against Poisoning Attacks**

解开联合学习抗中毒攻击中隐私与认证稳健性之间的联系 cs.CR

ACM CCS 2023

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2209.04030v3) [paper-pdf](http://arxiv.org/pdf/2209.04030v3)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Qinbin Li, Arash Nourian, Sanmi Koyejo, Bo Li

**Abstract**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As local training data comes from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is usually trained in a differentially private way (DPFL). Thus, in this paper, we ask: What are the underlying connections between differential privacy and certified robustness in FL against poisoning attacks? Can we leverage the innate privacy property of DPFL to provide certified robustness for FL? Can we further improve the privacy of FL to improve such robustness certification? We first investigate both user-level and instance-level privacy of FL and provide formal privacy analysis to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack inefficacy for DPFL on both user and instance levels. Theoretically, we provide the certified robustness of DPFL based on both criteria given a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of poisoning attacks on different datasets. We find that increasing the level of privacy protection in DPFL results in stronger certified attack inefficacy; however, it does not necessarily lead to a stronger certified prediction. Thus, achieving the optimal certified prediction requires a proper balance between privacy and utility loss.

摘要: 联合学习(FL)提供了一种有效的范例来联合训练利用来自分布式用户的数据的全局模型。由于本地训练数据来自可能不值得信任的不同用户，多项研究表明，FL容易受到中毒攻击。同时，为了保护本地用户的隐私，FL通常会以一种不同的私人方式进行培训(DPFL)。因此，在这篇文章中，我们问：区别隐私和FL对中毒攻击的认证健壮性之间有什么潜在的联系？我们能否利用DPFL与生俱来的隐私属性为FL提供经过认证的健壮性？我们能否进一步改善FL的隐私，以提高这种健壮性认证？我们首先对FL的用户级和实例级隐私进行了研究，并提供了形式化的隐私分析，以实现改进的实例级隐私。然后，我们提供了两个健壮性认证标准：DPFL在用户和实例级别上的认证预测和认证攻击无效。理论上，在给定有限数量的敌意用户或实例的情况下，我们基于这两个标准提供了DPFL的证明的健壮性。在经验上，我们在不同数据集的一系列中毒攻击下进行了广泛的实验来验证我们的理论。我们发现，增加DPFL中的隐私保护级别会导致更强的认证攻击无效；然而，这并不一定会导致更强的认证预测。因此，要实现最佳验证预测，需要在隐私和效用损失之间取得适当的平衡。



## **49. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

基于骨架的图卷积神经网络鲁棒性的傅立叶分析 cs.CV

18 pages, 13 figures

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2305.17939v2) [paper-pdf](http://arxiv.org/pdf/2305.17939v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.

摘要: 利用傅立叶分析，我们研究了基于骨架的动作识别的图卷积神经网络(GCNS)的稳健性和脆弱性。我们采用联合傅里叶变换(JFT)，即图傅里叶变换(GFT)和离散傅立叶变换(DFT)的组合，来检验经过对抗性训练的GCNS对敌意攻击和常见腐败的健壮性。在NTU RGB+D数据集上的实验结果表明，对抗性训练不会在对抗性攻击和低频扰动之间引入稳健性权衡，而这通常发生在基于卷积神经网络的图像分类中。这一发现表明，在基于骨架的动作识别中，对抗性训练是一种增强对对抗性攻击和常见腐败的稳健性的实用方法。此外，我们发现傅立叶方法不能解释对骨骼部分遮挡破坏的脆弱性，这突出了它的局限性。这些发现扩展了我们对GCNS健壮性的理解，潜在地指导了基于骨骼的动作识别的更健壮的学习方法的发展。



## **50. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

ReMAV：自动车辆发现可能故障事件的奖励模型 cs.AI

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2308.14550v2) [paper-pdf](http://arxiv.org/pdf/2308.14550v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known to be vulnerable to various adversarial attacks, compromising vehicle safety and posing a risk to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found to be less confident. In this paper, we propose a black-box testing framework ReMAV that uses offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds to find the probability of failure events. To this end, we introduce a three-step methodology which i) uses offline state action pairs of any autonomous vehicle under test, ii) builds an abstract behavior representation using our designed reward modeling technique to analyze states with uncertain driving decisions, and iii) uses a disturbance model for minimal perturbation attacks where the driving decisions are less confident. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the standard autonomous vehicle performs well. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single- and multi-agent interactions. Our experiment shows an increase in 35, 23, 48, and 50% in the occurrences of vehicle collision, road object collision, pedestrian collision, and offroad steering events, respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We compare ReMAV with two baselines and show that ReMAV demonstrates significantly better effectiveness in generating failure events compared to the baselines in all evaluation metrics.

摘要: 自动驾驶汽车是一种先进的驾驶系统，众所周知，它容易受到各种对抗性攻击，危及车辆安全，并对其他道路使用者构成风险。与其通过与环境互动来积极训练复杂的对手，不如首先智能地找到搜索空间，并将搜索空间缩小到那些自动驾驶汽车被发现不那么自信的状态。在本文中，我们提出了一个黑盒测试框架ReMAV，该框架首先使用离线轨迹来分析自动驾驶车辆的现有行为，并确定合适的阈值来发现故障事件的概率。为此，我们介绍了一种三步法，即i)使用任何被测自动驾驶车辆的离线状态动作对，ii)使用我们设计的奖励建模技术来建立抽象的行为表示来分析具有不确定驾驶决策的状态，以及iii)使用扰动模型来对驾驶决策不太可信的最小扰动攻击进行分析。我们的奖励建模技术有助于创建行为表示，使我们能够突出显示可能存在不确定行为的区域，即使标准自动驾驶汽车表现良好。我们在高保真的城市驾驶环境中使用三种不同的驾驶场景进行实验，其中包含单代理和多代理交互。我们的实验表明，被测试的自动驾驶汽车的车辆碰撞、道路物体碰撞、行人碰撞和越野转向事件的发生率分别增加了35%、23%、48%和50%，表明故障事件显著增加。我们比较了ReMAV和两个基线，结果表明，在所有评估指标中，ReMAV在生成故障事件方面都比基线表现出了更好的有效性。



