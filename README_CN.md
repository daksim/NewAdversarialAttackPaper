# Latest Adversarial Attack Papers
**update at 2023-05-10 10:07:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improving Adversarial Transferability via Intermediate-level Perturbation Decay**

通过中层扰动衰减提高对手的可转换性 cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2304.13410v2) [paper-pdf](http://arxiv.org/pdf/2304.13410v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

摘要: 中级攻击试图按照对抗性方向彻底扰乱特征表示，在制作可转移的对抗性示例方面表现出了良好的性能。现有的这类方法通常分为两个不同的阶段，首先需要确定一个方向导轨，然后放大中层摄动在该方向导轨上的标量投影。所得到的扰动在特征空间中不可避免地偏离了导引，本文揭示了这种偏离可能导致次优攻击。为了解决这个问题，我们开发了一种新的中级方法，该方法在单个优化阶段内创建对抗性示例。特别是，所提出的方法，称为中层扰动衰变(ILPD)，它鼓励中层扰动朝着有效的对抗性方向发展，同时具有较大的幅度。通过深入讨论，验证了该方法的有效性。实验结果表明，在ImageNet(平均+10.07%)和CIFAR-10(平均+3.88%)上攻击各种受害者模型时，该算法的性能明显优于最新的攻击模型。我们的代码在https://github.com/qizhangli/ILPD-attack.



## **2. Turning Privacy-preserving Mechanisms against Federated Learning**

将隐私保护机制转向联合学习 cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05355v1) [paper-pdf](http://arxiv.org/pdf/2305.05355v1)

**Authors**: Marco Arazzi, Mauro Conti, Antonino Nocera, Stjepan Picek

**Abstract**: Recently, researchers have successfully employed Graph Neural Networks (GNNs) to build enhanced recommender systems due to their capability to learn patterns from the interaction between involved entities. In addition, previous studies have investigated federated learning as the main solution to enable a native privacy-preserving mechanism for the construction of global GNN models without collecting sensitive data into a single computation unit. Still, privacy issues may arise as the analysis of local model updates produced by the federated clients can return information related to sensitive local data. For this reason, experts proposed solutions that combine federated learning with Differential Privacy strategies and community-driven approaches, which involve combining data from neighbor clients to make the individual local updates less dependent on local sensitive data. In this paper, we identify a crucial security flaw in such a configuration, and we design an attack capable of deceiving state-of-the-art defenses for federated learning. The proposed attack includes two operating modes, the first one focusing on convergence inhibition (Adversarial Mode), and the second one aiming at building a deceptive rating injection on the global federated model (Backdoor Mode). The experimental results show the effectiveness of our attack in both its modes, returning on average 60% performance detriment in all the tests on Adversarial Mode and fully effective backdoors in 93% of cases for the tests performed on Backdoor Mode.

摘要: 最近，研究人员已经成功地使用图神经网络(GNN)来构建增强的推荐系统，这是因为它们能够从相关实体之间的交互中学习模式。此外，以前的研究已经将联合学习作为主要解决方案，以实现在不将敏感数据收集到单个计算单元的情况下构建全局GNN模型的本地隐私保护机制。尽管如此，隐私问题可能会出现，因为对联合客户端生成的本地模型更新的分析可能会返回与敏感本地数据相关的信息。为此，专家们提出了将联合学习与差异隐私策略和社区驱动方法相结合的解决方案，其中包括合并来自邻居客户端的数据，以减少个别本地更新对本地敏感数据的依赖。在本文中，我们确定了这种配置中的一个关键安全漏洞，并设计了一个能够欺骗联邦学习的最新防御的攻击。提出的攻击包括两种工作模式，第一种集中在收敛抑制(对抗性模式)，第二种旨在在全球联邦模型上建立欺骗性评级注入(后门模式)。实验结果表明，我们的攻击在两种模式下都是有效的，在对抗性模式下的所有测试中平均返回60%的性能损失，在后门模式上执行的测试中，93%的情况下完全有效的后门程序。



## **3. Data Protection and Security Issues With Network Error Logging**

网络错误记录的数据保护和安全问题 cs.CR

Accepted for SECRYPT'23

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05343v1) [paper-pdf](http://arxiv.org/pdf/2305.05343v1)

**Authors**: Libor Polčák, Kamil Jeřábek

**Abstract**: Network Error Logging helps web server operators detect operational problems in real-time to provide fast and reliable services. This paper analyses Network Error Logging from two angles. Firstly, this paper overviews Network Error Logging from the data protection view. The ePrivacy Directive requires consent for non-essential access to the end devices. Nevertheless, the Network Error Logging design does not allow limiting the tracking to consenting users. Other issues lay in GDPR requirements for transparency and the obligations in the contract between controllers and processors of personal data. Secondly, this paper explains Network Error Logging exploitations to deploy long-time trackers to the victim devices. Even though users should be able to disable Network Error Logging, it is not clear how to do so. Web server operators can mitigate the attack by configuring servers to preventively remove policies that adversaries might have added.

摘要: 网络错误记录帮助Web服务器操作员实时检测运行问题，以提供快速可靠的服务。本文从两个角度对网络错误记录进行了分析。首先，本文从数据保护的角度对网络错误记录进行了综述。电子隐私指令要求对终端设备进行非必要访问的同意。然而，网络错误记录设计不允许将跟踪限制到同意的用户。其他问题包括GDPR对透明度的要求以及个人数据管制员和处理者之间合同中的义务。其次，本文解释了利用网络错误记录漏洞将长期跟踪器部署到受攻击设备。尽管用户应该能够禁用网络错误记录，但不清楚如何这样做。Web服务器运营商可以通过配置服务器以预防性地删除攻击者可能添加的策略来缓解攻击。



## **4. Attack Named Entity Recognition by Entity Boundary Interference**

利用实体边界干扰攻击命名实体识别 cs.CL

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05253v1) [paper-pdf](http://arxiv.org/pdf/2305.05253v1)

**Authors**: Yifei Yang, Hongqiu Wu, Hai Zhao

**Abstract**: Named Entity Recognition (NER) is a cornerstone NLP task while its robustness has been given little attention. This paper rethinks the principles of NER attacks derived from sentence classification, as they can easily violate the label consistency between the original and adversarial NER examples. This is due to the fine-grained nature of NER, as even minor word changes in the sentence can result in the emergence or mutation of any entities, resulting in invalid adversarial examples. To this end, we propose a novel one-word modification NER attack based on a key insight, NER models are always vulnerable to the boundary position of an entity to make their decision. We thus strategically insert a new boundary into the sentence and trigger the Entity Boundary Interference that the victim model makes the wrong prediction either on this boundary word or on other words in the sentence. We call this attack Virtual Boundary Attack (ViBA), which is shown to be remarkably effective when attacking both English and Chinese models with a 70%-90% attack success rate on state-of-the-art language models (e.g. RoBERTa, DeBERTa) and also significantly faster than previous methods.

摘要: 命名实体识别(NER)是一项基础性的自然语言处理任务，但其健壮性却鲜有人关注。本文重新思考了基于句子分类的NER攻击的原理，因为它们很容易破坏原始例子和对抗性例子之间的标签一致性。这是由于NER的细粒度性质，因为即使句子中的微小单词变化也可能导致任何实体的出现或突变，从而导致无效的对抗性例子。为此，我们提出了一种新颖的基于关键洞察力的单字修正NER攻击，NER模型总是容易受到实体边界位置的影响而做出决策。因此，我们策略性地在句子中插入新的边界，并触发实体边界干扰，即受害者模型对句子中的该边界词或其他词做出错误预测。我们将这种攻击称为虚拟边界攻击(VIBA)，该攻击在对最先进的语言模型(如Roberta，DeBERTa)的攻击成功率为70%-90%的情况下，对英文和中文模型的攻击都非常有效，而且攻击速度也明显快于以前的方法。



## **5. Generating Phishing Attacks using ChatGPT**

使用ChatGPT生成网络钓鱼攻击 cs.CR

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05133v1) [paper-pdf](http://arxiv.org/pdf/2305.05133v1)

**Authors**: Sayak Saha Roy, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The ability of ChatGPT to generate human-like responses and understand context has made it a popular tool for conversational agents, content creation, data analysis, and research and innovation. However, its effectiveness and ease of accessibility makes it a prime target for generating malicious content, such as phishing attacks, that can put users at risk. In this work, we identify several malicious prompts that can be provided to ChatGPT to generate functional phishing websites. Through an iterative approach, we find that these phishing websites can be made to imitate popular brands and emulate several evasive tactics that have been known to avoid detection by anti-phishing entities. These attacks can be generated using vanilla ChatGPT without the need of any prior adversarial exploits (jailbreaking).

摘要: ChatGPT生成类似人类的响应并理解上下文的能力使其成为对话代理、内容创建、数据分析以及研究和创新的流行工具。然而，它的有效性和易访问性使其成为生成恶意内容的主要目标，例如网络钓鱼攻击，这些内容可能会将用户置于风险之中。在这项工作中，我们识别了几个可以提供给ChatGPT以生成功能性钓鱼网站的恶意提示。通过迭代的方法，我们发现可以让这些钓鱼网站模仿流行品牌，并模仿几种已知的规避策略，以避免被反钓鱼实体发现。这些攻击可以使用普通的ChatGPT生成，而不需要任何先前的对抗性攻击(越狱)。



## **6. Communication-Robust Multi-Agent Learning by Adaptable Auxiliary Multi-Agent Adversary Generation**

基于自适应辅助多智能体对手生成的通信健壮多智能体学习 cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05116v1) [paper-pdf](http://arxiv.org/pdf/2305.05116v1)

**Authors**: Lei Yuan, Feng Chen, Zhongzhang Zhang, Yang Yu

**Abstract**: Communication can promote coordination in cooperative Multi-Agent Reinforcement Learning (MARL). Nowadays, existing works mainly focus on improving the communication efficiency of agents, neglecting that real-world communication is much more challenging as there may exist noise or potential attackers. Thus the robustness of the communication-based policies becomes an emergent and severe issue that needs more exploration. In this paper, we posit that the ego system trained with auxiliary adversaries may handle this limitation and propose an adaptable method of Multi-Agent Auxiliary Adversaries Generation for robust Communication, dubbed MA3C, to obtain a robust communication-based policy. In specific, we introduce a novel message-attacking approach that models the learning of the auxiliary attacker as a cooperative problem under a shared goal to minimize the coordination ability of the ego system, with which every information channel may suffer from distinct message attacks. Furthermore, as naive adversarial training may impede the generalization ability of the ego system, we design an attacker population generation approach based on evolutionary learning. Finally, the ego system is paired with an attacker population and then alternatively trained against the continuously evolving attackers to improve its robustness, meaning that both the ego system and the attackers are adaptable. Extensive experiments on multiple benchmarks indicate that our proposed MA3C provides comparable or better robustness and generalization ability than other baselines.

摘要: 在协作多智能体强化学习(MAIL)中，通信可以促进协作。目前，已有的研究主要集中在提高智能体的通信效率上，而忽略了现实世界中可能存在噪声或潜在攻击者的情况下，通信更具挑战性。因此，基于通信的策略的健壮性成为一个迫切而严峻的问题，需要进一步探讨。在本文中，我们假设用辅助对手训练的EGO系统可以处理这一局限性，并提出了一种用于稳健通信的自适应多智能体辅助对手生成方法MA3C，以获得基于通信的健壮策略。具体地说，我们引入了一种新的消息攻击方法，将辅助攻击者的学习建模为一个共享目标下的合作问题，以最小化EGO系统的协调能力，在这种情况下，每个信息通道都可能遭受不同的消息攻击。此外，由于天真的对抗性训练可能会阻碍EGO系统的泛化能力，我们设计了一种基于进化学习的攻击种群生成方法。最后，EGO系统与攻击者群体配对，然后交替地针对不断进化的攻击者进行训练，以提高其健壮性，这意味着EGO系统和攻击者都是自适应的。在多个基准上的大量实验表明，我们提出的MA3C提供了与其他基准相当或更好的稳健性和泛化能力。



## **7. Escaping saddle points in zeroth-order optimization: the power of two-point estimators**

零阶最优化中的鞍点逃逸：两点估计的能力 math.OC

To appear at ICML 2023

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2209.13555v3) [paper-pdf](http://arxiv.org/pdf/2209.13555v3)

**Authors**: Zhaolin Ren, Yujie Tang, Na Li

**Abstract**: Two-point zeroth order methods are important in many applications of zeroth-order optimization, such as robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem may be high-dimensional and/or time-varying. Most problems in these applications are nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \leq m \leq d$) function evaluations per iteration can not only find $\epsilon$-second order stationary points polynomially fast, but do so using only $\tilde{O}\left(\frac{d}{m\epsilon^{2}\bar{\psi}}\right)$ function evaluations, where $\bar{\psi} \geq \tilde{\Omega}\left(\sqrt{\epsilon}\right)$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property.

摘要: 两点零阶方法在零阶优化的许多应用中都是重要的，例如机器人、风电场、电力系统、在线优化以及深层神经网络中对黑盒攻击的对抗鲁棒性，这些问题可能是高维的和/或时变的。这些应用中的大多数问题都是非凸的，并且包含鞍点。虽然已有的工作表明，利用每次迭代的$\Omega(D)$函数赋值(其中$d$表示问题的维度)的零级方法可以有效地逃离鞍点，但基于两点估计的零级方法是否能够逃离鞍点仍然是一个悬而未决的问题。本文证明了，通过在每次迭代中加入适当的各向同性扰动，基于每一次迭代的$2m$(对于任意$1\leq m\leq d$)函数求值的零阶算法不仅可以多项式地快速地找到$-二阶驻点，而且只使用$\tilde{O}\left(\frac{d}{m\epsilon^{2}\bar{\psi}}\right)$函数求值，其中，$\bar{\psi}\geq\tilde{\omega}\Left(\Sqrt{\epsilon}\right)$是捕获感兴趣函数展现严格鞍形属性的程度的参数。



## **8. Less is More: Removing Text-regions Improves CLIP Training Efficiency and Robustness**

少即是多：去除文本区域可以提高剪辑训练的效率和健壮性 cs.CV

10 pages, 8 figures

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05095v1) [paper-pdf](http://arxiv.org/pdf/2305.05095v1)

**Authors**: Liangliang Cao, Bowen Zhang, Chen Chen, Yinfei Yang, Xianzhi Du, Wencong Zhang, Zhiyun Lu, Yantao Zheng

**Abstract**: The CLIP (Contrastive Language-Image Pre-training) model and its variants are becoming the de facto backbone in many applications. However, training a CLIP model from hundreds of millions of image-text pairs can be prohibitively expensive. Furthermore, the conventional CLIP model doesn't differentiate between the visual semantics and meaning of text regions embedded in images. This can lead to non-robustness when the text in the embedded region doesn't match the image's visual appearance. In this paper, we discuss two effective approaches to improve the efficiency and robustness of CLIP training: (1) augmenting the training dataset while maintaining the same number of optimization steps, and (2) filtering out samples that contain text regions in the image. By doing so, we significantly improve the classification and retrieval accuracy on public benchmarks like ImageNet and CoCo. Filtering out images with text regions also protects the model from typographic attacks. To verify this, we build a new dataset named ImageNet with Adversarial Text Regions (ImageNet-Attr). Our filter-based CLIP model demonstrates a top-1 accuracy of 68.78\%, outperforming previous models whose accuracy was all below 50\%.

摘要: CLIP(对比语言-图像预训练)模型及其变体正在成为许多应用中事实上的支柱。然而，从数以亿计的图像-文本对中训练剪辑模型的成本可能高得令人望而却步。此外，传统的剪辑模型不区分嵌入在图像中的文本区域的视觉语义和含义。当嵌入区域中的文本与图像的视觉外观不匹配时，这可能会导致不稳定。在本文中，我们讨论了两种有效的方法来提高剪辑训练的效率和稳健性：(1)在保持相同的优化步数的情况下扩大训练数据集；(2)过滤掉图像中包含文本区域的样本。通过这样做，我们在ImageNet和CoCo等公共基准上显著提高了分类和检索的准确性。过滤掉带有文本区域的图像还可以保护模型免受排版攻击。为了验证这一点，我们构建了一个名为ImageNet的新数据集，其中包含敌对文本区域(ImageNet-Attr)。我们的基于过滤器的剪辑模型的TOP-1精度为68.78\%，超过了以前的精度都在50\%以下的模型。



## **9. Distributed Detection over Blockchain-aided Internet of Things in the Presence of Attacks**

存在攻击的区块链辅助物联网分布式检测 cs.CR

16 pages, 4 figures. This work has been submitted to the IEEE TIFS

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05070v1) [paper-pdf](http://arxiv.org/pdf/2305.05070v1)

**Authors**: Yiming Jiang, Jiangfan Zhang

**Abstract**: Distributed detection over a blockchain-aided Internet of Things (BIoT) network in the presence of attacks is considered, where the integrated blockchain is employed to secure data exchanges over the BIoT as well as data storage at the agents of the BIoT. We consider a general adversary model where attackers jointly exploit the vulnerability of IoT devices and that of the blockchain employed in the BIoT. The optimal attacking strategy which minimizes the Kullback-Leibler divergence is pursued. It can be shown that this optimization problem is nonconvex, and hence it is generally intractable to find the globally optimal solution to such a problem. To overcome this issue, we first propose a relaxation method that can convert the original nonconvex optimization problem into a convex optimization problem, and then the analytic expression for the optimal solution to the relaxed convex optimization problem is derived. The optimal value of the relaxed convex optimization problem provides a detection performance guarantee for the BIoT in the presence of attacks. In addition, we develop a coordinate descent algorithm which is based on a capped water-filling method to solve the relaxed convex optimization problem, and moreover, we show that the convergence of the proposed coordinate descent algorithm can be guaranteed.

摘要: 考虑了在存在攻击的情况下通过区块链辅助的物联网(Biot)网络进行分布式检测，其中使用集成的区块链来保护Biot上的数据交换以及Biot代理处的数据存储。我们考虑了一个一般的对手模型，在该模型中，攻击者联合利用物联网设备和Biot中采用的区块链的漏洞。寻求使Kullback-Leibler发散最小的最优攻击策略。可以看出，该优化问题是非凸的，因此寻找此类问题的全局最优解通常是困难的。为了克服这个问题，我们首先提出了一种松弛方法，可以将原来的非凸优化问题转化为凸优化问题，然后推导出松弛凸优化问题的最优解的解析表达式。松弛凸优化问题的最优值为Biot在存在攻击时的检测性能提供了保证。此外，我们还提出了一种基于封顶注水方法的坐标下降算法来求解松弛凸优化问题，并证明了该算法的收敛是有保证的。



## **10. A Survey on AI/ML-Driven Intrusion and Misbehavior Detection in Networked Autonomous Systems: Techniques, Challenges and Opportunities**

AI/ML驱动的网络自治系统入侵与行为检测研究综述：技术、挑战与机遇 cs.NI

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05040v1) [paper-pdf](http://arxiv.org/pdf/2305.05040v1)

**Authors**: Opeyemi Ajibuwa, Bechir Hamdaoui, Attila A. Yavuz

**Abstract**: AI/ML-based intrusion detection systems (IDSs) and misbehavior detection systems (MDSs) have shown great potential in identifying anomalies in the network traffic of networked autonomous systems. Despite the vast research efforts, practical deployments of such systems in the real world have been limited. Although the safety-critical nature of autonomous systems and the vulnerability of learning-based techniques to adversarial attacks are among the potential reasons, the lack of objective evaluation and feasibility assessment metrics is one key reason behind the limited adoption of these systems in practical settings. This survey aims to address the aforementioned limitation by presenting an in-depth analysis of AI/ML-based IDSs/MDSs and establishing baseline metrics relevant to networked autonomous systems. Furthermore, this work thoroughly surveys recent studies in this domain, highlighting the evaluation metrics and gaps in the current literature. It also presents key findings derived from our analysis of the surveyed papers and proposes guidelines for providing AI/ML-based IDS/MDS solution approaches suitable for vehicular network applications. Our work provides researchers and practitioners with the needed tools to evaluate the feasibility of AI/ML-based IDS/MDS techniques in real-world settings, with the aim of facilitating the practical adoption of such techniques in emerging autonomous vehicular systems.

摘要: 基于AI/ML的入侵检测系统和行为异常检测系统在识别网络自治系统的网络流量异常方面显示出了巨大的潜力。尽管付出了巨大的研究努力，但这类系统在现实世界中的实际部署一直有限。虽然自主系统的安全关键性质和基于学习的技术对对手攻击的脆弱性是潜在原因之一，但缺乏客观评估和可行性评估衡量标准是这些系统在实际环境中采用有限的一个关键原因。本调查旨在通过深入分析基于AI/ML的入侵检测系统/分布式检测系统并建立与联网自治系统相关的基线度量来解决上述限制。此外，这项工作深入地综述了这一领域的最新研究，突出了当前文献中的评价指标和差距。它还介绍了我们对调查论文的分析得出的主要发现，并提出了适用于车载网络应用的基于AI/ML的入侵检测/MDS解决方案方法的指导方针。我们的工作为研究人员和实践者提供了必要的工具来评估基于AI/ML的入侵检测/MDS技术在现实世界中的可行性，目的是促进此类技术在新兴的自主车辆系统中的实际采用。



## **11. White-Box Multi-Objective Adversarial Attack on Dialogue Generation**

对话生成的白盒多目标对抗性攻击 cs.CL

ACL 2023 main conference long paper

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.03655v2) [paper-pdf](http://arxiv.org/pdf/2305.03655v2)

**Authors**: Yufei Li, Zexin Li, Yingfan Gao, Cong Liu

**Abstract**: Pre-trained transformers are popular in state-of-the-art dialogue generation (DG) systems. Such language models are, however, vulnerable to various adversarial samples as studied in traditional tasks such as text classification, which inspires our curiosity about their robustness in DG systems. One main challenge of attacking DG models is that perturbations on the current sentence can hardly degrade the response accuracy because the unchanged chat histories are also considered for decision-making. Instead of merely pursuing pitfalls of performance metrics such as BLEU, ROUGE, we observe that crafting adversarial samples to force longer generation outputs benefits attack effectiveness -- the generated responses are typically irrelevant, lengthy, and repetitive. To this end, we propose a white-box multi-objective attack method called DGSlow. Specifically, DGSlow balances two objectives -- generation accuracy and length, via a gradient-based multi-objective optimizer and applies an adaptive searching mechanism to iteratively craft adversarial samples with only a few modifications. Comprehensive experiments on four benchmark datasets demonstrate that DGSlow could significantly degrade state-of-the-art DG models with a higher success rate than traditional accuracy-based methods. Besides, our crafted sentences also exhibit strong transferability in attacking other models.

摘要: 预先培训的变压器在最先进的对话生成(DG)系统中很受欢迎。然而，这类语言模型容易受到文本分类等传统任务中研究的各种对抗性样本的影响，这引发了我们对它们在DG系统中的健壮性的好奇。攻击DG模型的一个主要挑战是，当前句子的扰动几乎不会降低响应精度，因为没有变化的聊天历史也被考虑用于决策。而不是仅仅追求性能指标的陷阱，如BLEU，Rouge，我们观察到精心制作敌意样本来迫使更长的世代输出有利于攻击效率-生成的响应通常是无关的、冗长的和重复的。为此，我们提出了一种白盒多目标攻击方法DGSlow。具体地说，DGSlow通过基于梯度的多目标优化器来平衡生成精度和长度这两个目标，并应用自适应搜索机制来迭代地创建只需少量修改的对抗性样本。在四个基准数据集上的综合实验表明，DGSlow可以显著降低最新的DG模型，并且比传统的基于精度的方法具有更高的成功率。此外，我们制作的句子在攻击其他模型时也表现出很强的可转移性。



## **12. Understanding Noise-Augmented Training for Randomized Smoothing**

理解随机平滑的噪声增强训练 cs.LG

Transactions on Machine Learning Research, 2023

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04746v1) [paper-pdf](http://arxiv.org/pdf/2305.04746v1)

**Authors**: Ambar Pal, Jeremias Sulam

**Abstract**: Randomized smoothing is a technique for providing provable robustness guarantees against adversarial attacks while making minimal assumptions about a classifier. This method relies on taking a majority vote of any base classifier over multiple noise-perturbed inputs to obtain a smoothed classifier, and it remains the tool of choice to certify deep and complex neural network models. Nonetheless, non-trivial performance of such smoothed classifier crucially depends on the base model being trained on noise-augmented data, i.e., on a smoothed input distribution. While widely adopted in practice, it is still unclear how this noisy training of the base classifier precisely affects the risk of the robust smoothed classifier, leading to heuristics and tricks that are poorly understood. In this work we analyze these trade-offs theoretically in a binary classification setting, proving that these common observations are not universal. We show that, without making stronger distributional assumptions, no benefit can be expected from predictors trained with noise-augmentation, and we further characterize distributions where such benefit is obtained. Our analysis has direct implications to the practical deployment of randomized smoothing, and we illustrate some of these via experiments on CIFAR-10 and MNIST, as well as on synthetic datasets.

摘要: 随机化平滑是一种在对分类器做出最小假设的同时提供针对对手攻击的可证明的稳健性保证的技术。这种方法依赖于在多个受噪声干扰的输入上取得任何基本分类器的多数票来获得平滑的分类器，并且它仍然是验证深度和复杂神经网络模型的首选工具。尽管如此，这种平滑分类器的非平凡性能关键取决于基本模型是基于噪声增强的数据来训练的，即基于平滑的输入分布。虽然在实践中被广泛采用，但仍然不清楚基分类器的这种噪声训练如何准确地影响稳健平滑分类器的风险，从而导致启发式算法和技巧鲜为人知。在这项工作中，我们在二进制分类的背景下从理论上分析了这些权衡，证明了这些常见的观察结果并不普遍。我们证明，如果不做更强的分布假设，则不能期望从经过噪声增强训练的预报器中获益，并且我们进一步刻画了获得这种益处的分布。我们的分析对随机平滑的实际部署有直接的影响，我们通过在CIFAR-10和MNIST上以及在合成数据集上的实验来说明其中的一些。



## **13. Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders**

在矩阵分解推荐器中评估用户聚类定向攻击的影响 cs.IR

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04694v1) [paper-pdf](http://arxiv.org/pdf/2305.04694v1)

**Authors**: Sulthana Shams, Douglas Leith

**Abstract**: In practice, users of a Recommender System (RS) fall into a few clusters based on their preferences. In this work, we conduct a systematic study on user-cluster targeted data poisoning attacks on Matrix Factorisation (MF) based RS, where an adversary injects fake users with falsely crafted user-item feedback to promote an item to a specific user cluster. We analyse how user and item feature matrices change after data poisoning attacks and identify the factors that influence the effectiveness of the attack on these feature matrices. We demonstrate that the adversary can easily target specific user clusters with minimal effort and that some items are more susceptible to attacks than others. Our theoretical analysis has been validated by the experimental results obtained from two real-world datasets. Our observations from the study could serve as a motivating point to design a more robust RS.

摘要: 在实践中，推荐系统(RS)的用户根据他们的偏好分为几个集群。在这项工作中，我们对基于矩阵分解(MF)的RS中针对用户簇的定向数据中毒攻击进行了系统的研究。在该攻击中，敌手向虚假用户注入虚假的用户项反馈，以将项推送到特定的用户簇。我们分析了数据中毒攻击后用户和项目特征矩阵的变化，并确定了影响这些特征矩阵攻击有效性的因素。我们证明了敌手可以很容易地以最小的努力瞄准特定的用户集群，并且一些项目比其他项目更容易受到攻击。两个真实数据集的实验结果验证了我们的理论分析。我们从这项研究中观察到的结果可以作为设计更健壮的RS的激励点。



## **14. StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning**

StyleAdv：跨域少发学习的元式对抗性训练 cs.CV

accepted by CVPR 2023

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2302.09309v2) [paper-pdf](http://arxiv.org/pdf/2302.09309v2)

**Authors**: Yuqian Fu, Yu Xie, Yanwei Fu, Yu-Gang Jiang

**Abstract**: Cross-Domain Few-Shot Learning (CD-FSL) is a recently emerging task that tackles few-shot learning across different domains. It aims at transferring prior knowledge learned on the source dataset to novel target datasets. The CD-FSL task is especially challenged by the huge domain gap between different datasets. Critically, such a domain gap actually comes from the changes of visual styles, and wave-SAN empirically shows that spanning the style distribution of the source data helps alleviate this issue. However, wave-SAN simply swaps styles of two images. Such a vanilla operation makes the generated styles ``real'' and ``easy'', which still fall into the original set of the source styles. Thus, inspired by vanilla adversarial learning, a novel model-agnostic meta Style Adversarial training (StyleAdv) method together with a novel style adversarial attack method is proposed for CD-FSL. Particularly, our style attack method synthesizes both ``virtual'' and ``hard'' adversarial styles for model training. This is achieved by perturbing the original style with the signed style gradients. By continually attacking styles and forcing the model to recognize these challenging adversarial styles, our model is gradually robust to the visual styles, thus boosting the generalization ability for novel target datasets. Besides the typical CNN-based backbone, we also employ our StyleAdv method on large-scale pretrained vision transformer. Extensive experiments conducted on eight various target datasets show the effectiveness of our method. Whether built upon ResNet or ViT, we achieve the new state of the art for CD-FSL. Code is available at https://github.com/lovelyqian/StyleAdv-CDFSL.

摘要: 跨域少镜头学习(CD-FSL)是近年来出现的一项研究课题，旨在解决不同领域的少镜头学习问题。它的目的是将在源数据集上学习的先验知识转移到新的目标数据集。CD-FSL任务尤其受到不同数据集之间巨大的域差距的挑战。关键的是，这样的领域差距实际上来自视觉风格的变化，而WAVE-SAN经验表明，跨越源数据的风格分布有助于缓解这一问题。然而，WAVE-SAN只是简单地交换两个图像的样式。这样的普通操作使生成的样式“真实”和“容易”，它们仍然属于原始的源样式集。因此，受传统对抗学习的启发，提出了一种新的模型--不可知元风格对抗训练方法(StyleAdv)和一种新风格的对抗攻击方法。具体地说，我们的风格攻击方法为模型训练综合了“虚拟”和“硬”两种对抗性风格。这是通过用签名的样式渐变扰乱原始样式来实现的。通过不断攻击风格并迫使模型识别这些具有挑战性的对抗性风格，我们的模型逐渐对视觉风格具有健壮性，从而增强了对新目标数据集的泛化能力。除了典型的基于CNN的主干，我们还将我们的StyleAdv方法应用于大规模的预训练视觉转换器。在8个不同的目标数据集上进行的大量实验表明了该方法的有效性。无论是建立在ResNet还是VIT之上，我们都达到了CD-FSL的最新技术水平。代码可在https://github.com/lovelyqian/StyleAdv-CDFSL.上找到



## **15. Recent Advances in Reliable Deep Graph Learning: Inherent Noise, Distribution Shift, and Adversarial Attack**

可靠深度图学习的最新进展：固有噪声、分布漂移和敌意攻击 cs.LG

Preprint. 9 pages, 2 figures

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2202.07114v2) [paper-pdf](http://arxiv.org/pdf/2202.07114v2)

**Authors**: Jintang Li, Bingzhe Wu, Chengbin Hou, Guoji Fu, Yatao Bian, Liang Chen, Junzhou Huang, Zibin Zheng

**Abstract**: Deep graph learning (DGL) has achieved remarkable progress in both business and scientific areas ranging from finance and e-commerce to drug and advanced material discovery. Despite the progress, applying DGL to real-world applications faces a series of reliability threats including inherent noise, distribution shift, and adversarial attacks. This survey aims to provide a comprehensive review of recent advances for improving the reliability of DGL algorithms against the above threats. In contrast to prior related surveys which mainly focus on adversarial attacks and defense, our survey covers more reliability-related aspects of DGL, i.e., inherent noise and distribution shift. Additionally, we discuss the relationships among above aspects and highlight some important issues to be explored in future research.

摘要: 深度图学习(DGL)在从金融和电子商务到药物和先进材料发现的商业和科学领域都取得了显着的进展。尽管取得了进展，但将DGL应用于现实世界的应用程序面临着一系列可靠性威胁，包括固有的噪声、分布偏移和对手攻击。本调查旨在全面回顾在提高DGL算法针对上述威胁的可靠性方面的最新进展。与以往主要关注对抗性攻击和防御的相关调查不同，我们的调查涵盖了DGL更多与可靠性相关的方面，即固有噪声和分布漂移。此外，我们还讨论了上述几个方面之间的关系，并指出了未来研究中需要探索的一些重要问题。



## **16. Toward Adversarial Training on Contextualized Language Representation**

语境化语言表征的对抗性训练 cs.CL

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04557v1) [paper-pdf](http://arxiv.org/pdf/2305.04557v1)

**Authors**: Hongqiu Wu, Yongxiang Liu, Hanwen Shi, Hai Zhao, Min Zhang

**Abstract**: Beyond the success story of adversarial training (AT) in the recent text domain on top of pre-trained language models (PLMs), our empirical study showcases the inconsistent gains from AT on some tasks, e.g. commonsense reasoning, named entity recognition. This paper investigates AT from the perspective of the contextualized language representation outputted by PLM encoders. We find the current AT attacks lean to generate sub-optimal adversarial examples that can fool the decoder part but have a minor effect on the encoder. However, we find it necessary to effectively deviate the latter one to allow AT to gain. Based on the observation, we propose simple yet effective \textit{Contextualized representation-Adversarial Training} (CreAT), in which the attack is explicitly optimized to deviate the contextualized representation of the encoder. It allows a global optimization of adversarial examples that can fool the entire model. We also find CreAT gives rise to a better direction to optimize the adversarial examples, to let them less sensitive to hyperparameters. Compared to AT, CreAT produces consistent performance gains on a wider range of tasks and is proven to be more effective for language pre-training where only the encoder part is kept for downstream tasks. We achieve the new state-of-the-art performances on a series of challenging benchmarks, e.g. AdvGLUE (59.1 $ \rightarrow $ 61.1), HellaSWAG (93.0 $ \rightarrow $ 94.9), ANLI (68.1 $ \rightarrow $ 69.3).

摘要: 除了最近文本领域在预训练语言模型(PLM)之上的对抗性训练(AT)的成功案例外，我们的实证研究还展示了在一些任务上，例如常识推理、命名实体识别，对抗性训练(AT)的不一致收获。本文从PLM编码者输出的语境化语言表征的角度对自动翻译进行研究。我们发现，当前的AT攻击倾向于生成次优的对抗性示例，这些示例可以愚弄解码器部分，但对编码器的影响很小。然而，我们发现有必要有效地偏离后者，以使AT获得收益。在此基础上，我们提出了简单而有效的文本化表示-对抗性训练(CREAT)，其中攻击被显式地优化以偏离编码者的上下文表示。它允许对可以愚弄整个模型的对抗性例子进行全局优化。我们还发现CREAT给出了一个更好的方向来优化对抗性例子，让它们对超参数不那么敏感。与AT相比，CREAT在更广泛的任务范围内产生了一致的性能提升，并且被证明在语言预训练中更有效，因为只有编码器部分被保留用于后续任务。我们在一系列具有挑战性的基准上实现了新的最先进的表现，例如AdvGLUE(59.1$\right tarrow$61.1)，HellaSWAG(93.0$\right tarrow$94.9)，Anli(68.1$\right tarrow$69.3)。



## **17. Privacy-preserving Adversarial Facial Features**

保护隐私的敌意面部特征 cs.CV

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05391v1) [paper-pdf](http://arxiv.org/pdf/2305.05391v1)

**Authors**: Zhibo Wang, He Wang, Shuaifan Jin, Wenwen Zhang, Jiahui Hu, Yan Wang, Peng Sun, Wei Yuan, Kaixin Liu, Kui Ren

**Abstract**: Face recognition service providers protect face privacy by extracting compact and discriminative facial features (representations) from images, and storing the facial features for real-time recognition. However, such features can still be exploited to recover the appearance of the original face by building a reconstruction network. Although several privacy-preserving methods have been proposed, the enhancement of face privacy protection is at the expense of accuracy degradation. In this paper, we propose an adversarial features-based face privacy protection (AdvFace) approach to generate privacy-preserving adversarial features, which can disrupt the mapping from adversarial features to facial images to defend against reconstruction attacks. To this end, we design a shadow model which simulates the attackers' behavior to capture the mapping function from facial features to images and generate adversarial latent noise to disrupt the mapping. The adversarial features rather than the original features are stored in the server's database to prevent leaked features from exposing facial information. Moreover, the AdvFace requires no changes to the face recognition network and can be implemented as a privacy-enhancing plugin in deployed face recognition systems. Extensive experimental results demonstrate that AdvFace outperforms the state-of-the-art face privacy-preserving methods in defending against reconstruction attacks while maintaining face recognition accuracy.

摘要: 人脸识别服务提供商通过从图像中提取紧凑和区别性的面部特征(表示)，并存储面部特征以供实时识别，从而保护面部隐私。然而，这些特征仍然可以通过构建重建网络来恢复原始人脸的外观。虽然已经提出了几种隐私保护方法，但增强人脸隐私保护的代价是准确性下降。本文提出了一种基于对抗特征的人脸隐私保护方法(AdvFace)来生成隐私保护的对抗特征，该方法可以破坏对抗特征到人脸图像的映射，从而防止重构攻击。为此，我们设计了一种模拟攻击者行为的阴影模型来捕捉人脸特征到图像的映射函数，并产生对抗性的潜在噪声来破坏映射。敌意特征而不是原始特征存储在服务器的数据库中，以防止泄露的特征暴露面部信息。此外，AdvFace不需要改变人脸识别网络，可以作为已部署的人脸识别系统中的隐私增强插件来实现。大量的实验结果表明，AdvFace在抵抗重构攻击的同时，在保持人脸识别准确率方面优于最先进的人脸隐私保护方法。



## **18. Attack-SAM: Towards Attacking Segment Anything Model With Adversarial Examples**

攻击-SAM：以对手为例的攻击分段Anything模型 cs.CV

The first work to attack Segment Anything Model with adversarial  examples

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.00866v2) [paper-pdf](http://arxiv.org/pdf/2305.00866v2)

**Authors**: Chenshuang Zhang, Chaoning Zhang, Taegoo Kang, Donghun Kim, Sung-Ho Bae, In So Kweon

**Abstract**: Segment Anything Model (SAM) has attracted significant attention recently, due to its impressive performance on various downstream tasks in a zero-short manner. Computer vision (CV) area might follow the natural language processing (NLP) area to embark on a path from task-specific vision models toward foundation models. However, deep vision models are widely recognized as vulnerable to adversarial examples, which fool the model to make wrong predictions with imperceptible perturbation. Such vulnerability to adversarial attacks causes serious concerns when applying deep models to security-sensitive applications. Therefore, it is critical to know whether the vision foundation model SAM can also be fooled by adversarial attacks. To the best of our knowledge, our work is the first of its kind to conduct a comprehensive investigation on how to attack SAM with adversarial examples. With the basic attack goal set to mask removal, we investigate the adversarial robustness of SAM in the full white-box setting and transfer-based black-box settings. Beyond the basic goal of mask removal, we further investigate and find that it is possible to generate any desired mask by the adversarial attack.

摘要: 分段任意模型(SAM)最近受到了极大的关注，因为它在各种下游任务上以零-短的方式表现出令人印象深刻的性能。计算机视觉(CV)领域可能会跟随自然语言处理(NLP)领域，走上一条从特定于任务的视觉模型到基础模型的道路。然而，深度视觉模型被广泛认为容易受到敌意例子的影响，这些例子愚弄了模型，使其在不知不觉中做出了错误的预测。在将深度模型应用于安全敏感应用程序时，此类易受敌意攻击的漏洞会引起严重关注。因此，了解VISION基础模型SAM是否也会被对抗性攻击愚弄是至关重要的。据我们所知，我们的工作是第一次对如何用对抗性例子攻击SAM进行全面调查。以去除掩码为基本攻击目标，研究了SAM在完全白盒设置和基于传输的黑盒设置下的对抗健壮性。除了掩码去除的基本目标之外，我们进一步研究发现，通过对抗性攻击可以产生任何想要的掩码。



## **19. Location Privacy Threats and Protections in Future Vehicular Networks: A Comprehensive Review**

未来车载网络中位置隐私的威胁与保护综述 cs.CR

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04503v1) [paper-pdf](http://arxiv.org/pdf/2305.04503v1)

**Authors**: Baihe Ma, Xu Wang, Xiaojie Lin, Yanna Jiang, Caijun Sun, Zhe Wang, Guangsheng Yu, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: Location privacy is critical in vehicular networks, where drivers' trajectories and personal information can be exposed, allowing adversaries to launch data and physical attacks that threaten drivers' safety and personal security. This survey reviews comprehensively different localization techniques, including widely used ones like sensing infrastructure-based, optical vision-based, and cellular radio-based localization, and identifies inadequately addressed location privacy concerns. We classify Location Privacy Preserving Mechanisms (LPPMs) into user-side, server-side, and user-server-interface-based, and evaluate their effectiveness. Our analysis shows that the user-server-interface-based LPPMs have received insufficient attention in the literature, despite their paramount importance in vehicular networks. Further, we examine methods for balancing data utility and privacy protection for existing LPPMs in vehicular networks and highlight emerging challenges from future upper-layer location privacy attacks, wireless technologies, and network convergences. By providing insights into the relationship between localization techniques and location privacy, and evaluating the effectiveness of different LPPMs, this survey can help inform the development of future LPPMs in vehicular networks.

摘要: 位置隐私在车载网络中至关重要，在车载网络中，司机的轨迹和个人信息可能会被暴露，从而允许对手发动威胁司机安全和人身安全的数据和物理攻击。这项调查全面回顾了不同的定位技术，包括广泛使用的基于传感基础设施的定位技术、基于光学视觉的定位技术和基于蜂窝无线电的定位技术，并确定了没有充分解决位置隐私问题的问题。我们将位置隐私保护机制(LPPM)分为基于用户端、基于服务器端和基于用户-服务器接口，并对其有效性进行了评估。我们的分析表明，基于用户-服务器接口的LPPM在文献中没有得到足够的关注，尽管它们在车载网络中非常重要。此外，我们还研究了在车载网络中平衡现有LPPM的数据效用和隐私保护的方法，并强调了未来上层位置隐私攻击、无线技术和网络融合带来的新挑战。通过深入了解定位技术和位置隐私之间的关系，以及评估不同LPPM的有效性，这项调查有助于为未来LPPM在车载网络中的发展提供信息。



## **20. Adversarial Examples Detection with Enhanced Image Difference Features based on Local Histogram Equalization**

基于局部直方图均衡的增强图像差分特征敌例检测 cs.CV

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04436v1) [paper-pdf](http://arxiv.org/pdf/2305.04436v1)

**Authors**: Zhaoxia Yin, Shaowei Zhu, Hang Su, Jianteng Peng, Wanli Lyu, Bin Luo

**Abstract**: Deep Neural Networks (DNNs) have recently made significant progress in many fields. However, studies have shown that DNNs are vulnerable to adversarial examples, where imperceptible perturbations can greatly mislead DNNs even if the full underlying model parameters are not accessible. Various defense methods have been proposed, such as feature compression and gradient masking. However, numerous studies have proven that previous methods create detection or defense against certain attacks, which renders the method ineffective in the face of the latest unknown attack methods. The invisibility of adversarial perturbations is one of the evaluation indicators for adversarial example attacks, which also means that the difference in the local correlation of high-frequency information in adversarial examples and normal examples can be used as an effective feature to distinguish the two. Therefore, we propose an adversarial example detection framework based on a high-frequency information enhancement strategy, which can effectively extract and amplify the feature differences between adversarial examples and normal examples. Experimental results show that the feature augmentation module can be combined with existing detection models in a modular way under this framework. Improve the detector's performance and reduce the deployment cost without modifying the existing detection model.

摘要: 深度神经网络(DNN)近年来在许多领域都取得了重大进展。然而，研究表明，DNN很容易受到敌意例子的影响，在这些例子中，即使无法获得完整的底层模型参数，不可察觉的扰动也会极大地误导DNN。人们已经提出了各种防御方法，如特征压缩和梯度掩蔽。然而，大量的研究已经证明，以前的方法是针对某些攻击进行检测或防御，这使得该方法在面对最新的未知攻击方法时效率低下。对抗性扰动的不可见性是对抗性范例攻击的评价指标之一，这也意味着对抗性范例与正常范例高频信息局部相关性的差异可以作为区分两者的有效特征。因此，我们提出了一种基于高频信息增强策略的对抗性范例检测框架，能够有效地提取和放大对抗性范例与正常范例之间的特征差异。实验结果表明，在该框架下，特征增强模块可以与现有的检测模型以模块化的方式结合起来。在不修改现有检测模型的情况下，提高了检测器的性能，降低了部署成本。



## **21. Are Synonym Substitution Attacks Really Synonym Substitution Attacks?**

同义词替换攻击真的是同义词替换攻击吗？ cs.CL

Findings in ACL 2023. Major revisions compared with previous versions  are made to incorporate the reviewers' suggestions. The modifications made  are listed in Appendix A

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2210.02844v3) [paper-pdf](http://arxiv.org/pdf/2210.02844v3)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: In this paper, we explore the following question: Are synonym substitution attacks really synonym substitution attacks (SSAs)? We approach this question by examining how SSAs replace words in the original sentence and show that there are still unresolved obstacles that make current SSAs generate invalid adversarial samples. We reveal that four widely used word substitution methods generate a large fraction of invalid substitution words that are ungrammatical or do not preserve the original sentence's semantics. Next, we show that the semantic and grammatical constraints used in SSAs for detecting invalid word replacements are highly insufficient in detecting invalid adversarial samples.

摘要: 本文探讨了以下问题：同义词替换攻击真的是同义词替换攻击吗？我们通过审查SSA如何替换原始句子中的单词来处理这个问题，并表明仍然存在尚未解决的障碍，使当前的SSA生成无效的对抗性样本。我们发现，四种广泛使用的词替换方法产生了很大一部分无效替换词，这些词不符合语法或没有保留原始句子的语义。其次，我们证明了SSA中用于检测无效单词替换的语义和语法约束在检测无效对抗性样本方面严重不足。



## **22. Reactive Perturbation Defocusing for Textual Adversarial Defense**

文本对抗防御中的反应性扰动散焦 cs.CL

**SubmitDate**: 2023-05-06    [abs](http://arxiv.org/abs/2305.04067v1) [paper-pdf](http://arxiv.org/pdf/2305.04067v1)

**Authors**: Heng Yang, Ke Li

**Abstract**: Recent studies have shown that large pre-trained language models are vulnerable to adversarial attacks. Existing methods attempt to reconstruct the adversarial examples. However, these methods usually have limited performance in defense against adversarial examples, while also negatively impacting the performance on natural examples. To overcome this problem, we propose a method called Reactive Perturbation Defocusing (RPD). RPD uses an adversarial detector to identify adversarial examples and reduce false defenses on natural examples. Instead of reconstructing the adversaries, RPD injects safe perturbations into adversarial examples to distract the objective models from the malicious perturbations. Our experiments on three datasets, two objective models, and various adversarial attacks show that our proposed framework successfully repairs up to approximately 97% of correctly identified adversarial examples with only about a 2% performance decrease on natural examples. We also provide a demo of adversarial detection and repair based on our work.

摘要: 最近的研究表明，大型预先训练的语言模型容易受到对抗性攻击。现有的方法试图重建对抗性的例子。然而，这些方法通常在防御对抗性实例时性能有限，同时也对自然实例的性能产生负面影响。为了克服这个问题，我们提出了一种称为反应性微扰散焦(RPD)的方法。RPD使用对抗性检测器来识别对抗性实例，并减少对自然实例的错误防御。RPD不是重构对手，而是将安全扰动注入到敌意实例中，以分散目标模型对恶意扰动的注意力。我们在三个数据集、两个目标模型和各种对抗性攻击上的实验表明，我们的框架成功地修复了大约97%的正确识别的对抗性例子，而对自然例子的性能只有大约2%的下降。我们还提供了一个基于我们的工作的对抗性检测和修复的演示。



## **23. TPC: Transformation-Specific Smoothing for Point Cloud Models**

TPC：点云模型的变换特定平滑 cs.CV

Accepted as a conference paper at ICML 2022

**SubmitDate**: 2023-05-06    [abs](http://arxiv.org/abs/2201.12733v5) [paper-pdf](http://arxiv.org/pdf/2201.12733v5)

**Authors**: Wenda Chu, Linyi Li, Bo Li

**Abstract**: Point cloud models with neural network architectures have achieved great success and have been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable to adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into three categories: additive (e.g., shearing), composable (e.g., rotation), and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for all categories respectively. We then specify unique certification protocols for a range of specific semantic transformations and their compositions. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within 20$^\circ$) from 20.3$\%$ to 83.8$\%$. Codes and models are available at https://github.com/chuwd19/Point-Cloud-Smoothing.

摘要: 神经网络结构的点云模型已经取得了巨大的成功，并在安全关键应用中得到了广泛的应用，例如自动车辆中基于激光雷达的识别系统。然而，这类模型被证明容易受到敌意攻击，这些攻击旨在应用诸如旋转和缩减等隐蔽的语义转换来误导模型预测。在本文中，我们提出了一个特定于变换的平滑框架TPC，它为点云模型抵抗语义变换攻击提供了紧凑和可伸缩的健壮性保证。我们首先将常见的3D变换分为三类：加性变换(如剪切)、可合成变换(如旋转变换)和间接合成变换(如锥化变换)，并分别针对所有类别提出了通用的健壮性认证策略。然后，我们为一系列特定的语义转换及其组合指定唯一的认证协议。对几种常见的3D变换进行的大量实验表明，TPC的性能明显优于最先进的技术。例如，我们的框架将z轴(在20$^\cic$内)抗扭曲变换的认证精度从20.3$\$提高到83.8$\$。有关代码和型号，请访问https://github.com/chuwd19/Point-Cloud-Smoothing.



## **24. Towards Prompt-robust Face Privacy Protection via Adversarial Decoupling Augmentation Framework**

基于对抗性解耦增强框架的快速稳健人脸隐私保护 cs.CV

8 pages, 6 figures

**SubmitDate**: 2023-05-06    [abs](http://arxiv.org/abs/2305.03980v1) [paper-pdf](http://arxiv.org/pdf/2305.03980v1)

**Authors**: Ruijia Wu, Yuhang Wang, Huafeng Shi, Zhipeng Yu, Yichao Wu, Ding Liang

**Abstract**: Denoising diffusion models have shown remarkable potential in various generation tasks. The open-source large-scale text-to-image model, Stable Diffusion, becomes prevalent as it can generate realistic artistic or facial images with personalization through fine-tuning on a limited number of new samples. However, this has raised privacy concerns as adversaries can acquire facial images online and fine-tune text-to-image models for malicious editing, leading to baseless scandals, defamation, and disruption to victims' lives. Prior research efforts have focused on deriving adversarial loss from conventional training processes for facial privacy protection through adversarial perturbations. However, existing algorithms face two issues: 1) they neglect the image-text fusion module, which is the vital module of text-to-image diffusion models, and 2) their defensive performance is unstable against different attacker prompts. In this paper, we propose the Adversarial Decoupling Augmentation Framework (ADAF), addressing these issues by targeting the image-text fusion module to enhance the defensive performance of facial privacy protection algorithms. ADAF introduces multi-level text-related augmentations for defense stability against various attacker prompts. Concretely, considering the vision, text, and common unit space, we propose Vision-Adversarial Loss, Prompt-Robust Augmentation, and Attention-Decoupling Loss. Extensive experiments on CelebA-HQ and VGGFace2 demonstrate ADAF's promising performance, surpassing existing algorithms.

摘要: 去噪扩散模型在各种发电任务中显示出显著的潜力。开源的大规模文本到图像模型稳定扩散，因为它可以通过对有限数量的新样本进行微调来生成具有个性化的逼真艺术或面部图像，因此变得流行起来。然而，这引发了隐私问题，因为攻击者可以在线获取面部图像，并微调文本到图像的模型以进行恶意编辑，从而导致毫无根据的丑闻、诽谤和对受害者生活的破坏。以前的研究工作集中在通过对抗性扰动从传统的面部隐私保护训练过程中获得对抗性损失。然而，现有的算法面临着两个问题：1)它们忽略了文本到图像扩散模型中的关键模块--图文融合模块；2)它们对不同的攻击者提示的防御性能不稳定。在本文中，我们提出了对抗性解耦增强框架(ADAF)，通过针对图文融合模块来解决这些问题，以增强人脸隐私保护算法的防御性能。ADAF引入了与文本相关的多级别增强，以针对各种攻击者提示提供防御稳定性。具体地说，考虑到视觉、文本和公共单元空间，我们提出了视觉对抗性损失、即时稳健增强和注意分离损失。在CelebA-HQ和VGGFace2上的大量实验证明了ADAF的良好性能，超过了现有的算法。



## **25. Beyond the Model: Data Pre-processing Attack to Deep Learning Models in Android Apps**

超越模型：Android应用程序中对深度学习模型的数据预处理攻击 cs.CR

Accepted to AsiaCCS WorkShop on Secure and Trustworthy Deep Learning  Systems (SecTL 2023)

**SubmitDate**: 2023-05-06    [abs](http://arxiv.org/abs/2305.03963v1) [paper-pdf](http://arxiv.org/pdf/2305.03963v1)

**Authors**: Ye Sang, Yujin Huang, Shuo Huang, Helei Cui

**Abstract**: The increasing popularity of deep learning (DL) models and the advantages of computing, including low latency and bandwidth savings on smartphones, have led to the emergence of intelligent mobile applications, also known as DL apps, in recent years. However, this technological development has also given rise to several security concerns, including adversarial examples, model stealing, and data poisoning issues. Existing works on attacks and countermeasures for on-device DL models have primarily focused on the models themselves. However, scant attention has been paid to the impact of data processing disturbance on the model inference. This knowledge disparity highlights the need for additional research to fully comprehend and address security issues related to data processing for on-device models. In this paper, we introduce a data processing-based attacks against real-world DL apps. In particular, our attack could influence the performance and latency of the model without affecting the operation of a DL app. To demonstrate the effectiveness of our attack, we carry out an empirical study on 517 real-world DL apps collected from Google Play. Among 320 apps utilizing MLkit, we find that 81.56\% of them can be successfully attacked.   The results emphasize the importance of DL app developers being aware of and taking actions to secure on-device models from the perspective of data processing.

摘要: 近年来，深度学习模型的日益流行以及计算的优势，包括智能手机上的低延迟和带宽节省，导致了智能移动应用程序的出现，也被称为深度学习应用程序。然而，这种技术的发展也引起了一些安全问题，包括对抗性例子、模型窃取和数据中毒问题。现有的针对设备上DL模型的攻击和对策的研究主要集中在模型本身。然而，数据处理干扰对模型推理的影响还没有引起足够的重视。这种知识差距突出表明，需要进行更多的研究，以充分理解和解决与设备上模型的数据处理相关的安全问题。在本文中，我们介绍了一种基于数据处理的针对现实世界数字图书馆应用程序的攻击。特别是，我们的攻击可能会影响模型的性能和延迟，而不会影响DL应用的操作。为了证明我们攻击的有效性，我们对从Google Play收集的517个真实数字图书馆应用程序进行了实证研究。在320个使用MLkit的应用中，我们发现81.56%的应用可以被成功攻击。这些结果强调了数字图书馆应用程序开发人员从数据处理的角度意识到并采取行动保护设备上模型的重要性。



## **26. Evading Watermark based Detection of AI-Generated Content**

基于规避水印的人工智能生成内容检测 cs.LG

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03807v1) [paper-pdf](http://arxiv.org/pdf/2305.03807v1)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model -- such as DALL-E, Stable Diffusion, and ChatGPT -- can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process an AI-generated watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed AI-generated image evades detection while maintaining its visual quality. We demonstrate the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to the AI-generated images and thus better maintain their visual quality than existing popular image post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work demonstrates the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new detection methods.

摘要: 生成性AI模型--如Dall-E、稳定扩散和ChatGPT--可以生成极其逼真的内容，对信息的真实性提出了越来越大的挑战。为了应对这些挑战，水印被用来检测人工智能生成的内容。具体地说，水印在发布之前被嵌入到人工智能生成的内容中。如果可以从内容中解码类似的水印，则该内容被检测为人工智能生成的内容。在这项工作中，我们对这种基于水印的人工智能内容检测的稳健性进行了系统的研究。我们专注于人工智能生成的图像。我们的工作表明，攻击者可以通过在人工智能生成的水印图像上添加一个人类无法察觉的小扰动来对其进行后处理，从而在保持其视觉质量的同时逃避检测。我们从理论上和经验上证明了我们的攻击的有效性。此外，为了逃避检测，我们的对抗性后处理方法对人工智能生成的图像添加了更小的扰动，从而比现有的流行的图像后处理方法，如JPEG压缩、高斯模糊和亮度/对比度，更好地保持了它们的视觉质量。我们的工作证明了现有基于水印的人工智能生成内容检测的不足，突出了对新检测方法的迫切需求。



## **27. Verifiable Learning for Robust Tree Ensembles**

用于稳健树集成的可验证学习 cs.LG

17 pages, 3 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03626v1) [paper-pdf](http://arxiv.org/pdf/2305.03626v1)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on publicly available datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, while incurring in just a relatively small loss of accuracy in the non-adversarial setting.

摘要: 验证机器学习模型在测试时对逃避攻击的稳健性是一个重要的研究问题。不幸的是，以前的工作确定了这个问题对于决策树集成来说是NP-Hard的，因此对于特定的输入必然是棘手的。在本文中，我们识别了一类受限的决策树集成，称为大分布集成，它允许安全验证算法在多项式时间内运行。然后，我们提出了一种新的方法，称为可验证学习，它主张训练这样的受限模型类，这些模型类适合于有效的验证。我们通过设计一种新的训练算法，从标记数据中自动学习大规模决策树集成，从而在多项式时间内实现其安全性验证，从而展示了这种思想的好处。在公开可用的数据集上的实验结果证实，使用我们的算法训练的大范围集成可以在几秒钟内使用标准商业硬件进行验证。此外，大范围的合奏比传统的合奏更能抵御逃避攻击，而在非对抗性的设置中，只会造成相对较小的准确性损失。



## **28. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection**

与您签约的目标不同：使用间接提示注入损害真实世界的LLM集成应用程序 cs.CR

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.12173v2) [paper-pdf](http://arxiv.org/pdf/2302.12173v2)

**Authors**: Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into various applications. The functionalities of recent LLMs can be flexibly modulated via natural language prompts. This renders them susceptible to targeted adversarial prompting, e.g., Prompt Injection (PI) attacks enable attackers to override original instructions and employed controls. So far, it was assumed that the user is directly prompting the LLM. But, what if it is not the user prompting? We argue that LLM-Integrated Applications blur the line between data and instructions. We reveal new attack vectors, using Indirect Prompt Injection, that enable adversaries to remotely (without a direct interface) exploit LLM-integrated applications by strategically injecting prompts into data likely to be retrieved. We derive a comprehensive taxonomy from a computer security perspective to systematically investigate impacts and vulnerabilities, including data theft, worming, information ecosystem contamination, and other novel security risks. We demonstrate our attacks' practical viability against both real-world systems, such as Bing's GPT-4 powered Chat and code-completion engines, and synthetic applications built on GPT-4. We show how processing retrieved prompts can act as arbitrary code execution, manipulate the application's functionality, and control how and if other APIs are called. Despite the increasing integration and reliance on LLMs, effective mitigations of these emerging threats are currently lacking. By raising awareness of these vulnerabilities and providing key insights into their implications, we aim to promote the safe and responsible deployment of these powerful models and the development of robust defenses that protect users and systems from potential attacks.

摘要: 大型语言模型(LLM)越来越多地被集成到各种应用程序中。最近的LLMS的功能可以通过自然语言提示灵活地调节。这使得它们容易受到有针对性的对抗性提示，例如，提示注入(PI)攻击使攻击者能够覆盖原始指令和采用的控制。到目前为止，假设用户是在直接提示LLM。但是，如果不是用户提示呢？我们认为，LLM集成应用程序模糊了数据和指令之间的界限。我们使用间接提示注入揭示了新的攻击载体，使攻击者能够通过战略性地向可能被检索的数据注入提示来远程(无需直接接口)利用LLM集成的应用程序。我们从计算机安全的角度得出了一个全面的分类，以系统地调查影响和漏洞，包括数据窃取、蠕虫、信息生态系统污染和其他新的安全风险。我们展示了我们的攻击在现实世界系统上的实际可行性，例如Bing的GPT-4聊天和代码完成引擎，以及基于GPT-4的合成应用程序。我们展示了处理检索到的提示如何充当任意代码执行、操纵应用程序的功能以及控制调用其他API的方式和是否调用。尽管对小岛屿发展中国家的整合和依赖日益增加，但目前缺乏对这些新出现的威胁的有效缓解。通过提高对这些漏洞的认识并提供对其影响的关键见解，我们的目标是促进安全和负责任地部署这些强大的模型，并开发强大的防御措施，以保护用户和系统免受潜在攻击。



## **29. Exploring the Connection between Robust and Generative Models**

探索健壮性模型和生成性模型之间的联系 cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2304.04033v2) [paper-pdf](http://arxiv.org/pdf/2304.04033v2)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.

摘要: 我们提供了一项研究，将经过对抗性训练(AT)训练的稳健区分分类器与基于能量的模型(EBM)形式的生成性建模相结合。我们通过分解判别分类器的损失来做到这一点，并表明判别模型也知道输入数据的密度。虽然一个普遍的假设是敌对点离开了输入数据的流形，但我们的研究发现，令人惊讶的是，在隐藏在判别分类器中的生成模型下，输入空间中的非目标对抗性点很可能在EBM中具有低能量。我们提出了两个证据：非目标攻击的可能性甚至比自然数据更高，并且随着攻击强度的增加，它们的可能性也会增加。这使我们能够轻松地检测到它们，并创建一种名为高能PGD的新型攻击，它愚弄了分类器，但具有与数据集相似的能量。



## **30. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

融合Top-1分解特征的Logit提高对手的可转移性 cs.CV

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.01361v2) [paper-pdf](http://arxiv.org/pdf/2305.01361v2)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in the middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial examples. Our extensive experimental results verify the effectiveness of our proposed method, which can be easily integrated into various baselines to significantly enhance the transferability of adversarial samples for disturbing normally trained CNNs and advanced defense strategies. The source code of this study is available at \textcolor{blue}{\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}{Link}}.

摘要: 最近的研究表明，深度神经网络非常容易受到敌意样本的攻击，这些样本具有很高的可传递性，可以用来攻击其他未知的黑盒模型。为了提高对抗性样本的可转移性，已经提出了几种基于特征的对抗性攻击方法来破坏中间层神经元的激活。然而，当前最先进的基于特征的攻击方法通常需要额外的计算成本来估计神经元的重要性。为了应对这一挑战，我们提出了一种基于奇异值分解(SVD)的特征级攻击方法。我们的方法是受到这样的发现的启发，即与从中间层特征分解的较大奇异值相关的特征向量具有更好的泛化和注意特性。具体地说，我们通过保留分解后的Top-1奇异值关联特征来计算输出逻辑，然后将其与原始逻辑相结合来优化对抗性实例，从而进行攻击。大量的实验结果验证了该方法的有效性，该方法可以很容易地集成到不同的基线中，显著提高对手样本干扰正常训练的CNN和高级防御策略的可转移性。这项研究的源代码可在\textcolor{blue}{\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}{Link}}.上获得



## **31. Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks**

具有自动复制/粘贴攻击的深度神经网络的诊断 cs.LG

Best paper award at the NeurIPS 2022 ML Safety Workshop --  https://neurips2022.mlsafety.org/

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2211.10024v3) [paper-pdf](http://arxiv.org/pdf/2211.10024v3)

**Authors**: Stephen Casper, Kaivalya Hariharan, Dylan Hadfield-Menell

**Abstract**: This paper considers the problem of helping humans exercise scalable oversight over deep neural networks (DNNs). Adversarial examples can be useful by helping to reveal weaknesses in DNNs, but they can be difficult to interpret or draw actionable conclusions from. Some previous works have proposed using human-interpretable adversarial attacks including copy/paste attacks in which one natural image pasted into another causes an unexpected misclassification. We build on these with two contributions. First, we introduce Search for Natural Adversarial Features Using Embeddings (SNAFUE) which offers a fully automated method for finding copy/paste attacks. Second, we use SNAFUE to red team an ImageNet classifier. We reproduce copy/paste attacks from previous works and find hundreds of other easily-describable vulnerabilities, all without a human in the loop. Code is available at https://github.com/thestephencasper/snafue

摘要: 本文研究了帮助人类对深度神经网络进行可扩展监督的问题。对抗性的例子可以通过帮助揭示DNN中的弱点而有用，但它们可能很难解释或从中得出可操作的结论。以前的一些工作已经提出使用人类可解释的对抗性攻击，包括复制/粘贴攻击，在这种攻击中，一幅自然图像粘贴到另一幅图像中会导致意外的错误分类。我们在这些基础上做出了两项贡献。首先，我们介绍了使用嵌入搜索自然对抗性特征(SNAFUE)，它提供了一种全自动的方法来发现复制/粘贴攻击。其次，我们使用SNAFUE对ImageNet分类器进行分组。我们复制了以前的作品中的复制/粘贴攻击，并发现了数百个其他容易描述的漏洞，所有这些都没有人参与。代码可在https://github.com/thestephencasper/snafue上找到



## **32. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

基于鲁棒性感知CoReset选择的高效对抗性对比学习 cs.LG

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.03857v3) [paper-pdf](http://arxiv.org/pdf/2302.03857v3)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS.

摘要: 对抗性对比学习(ACL)不需要昂贵的数据标注，但输出了一种稳健的表示，可以抵抗对抗性攻击，并适用于广泛的下游任务。然而，ACL需要大量的运行时间来生成所有训练数据的对抗性变体，这限制了其在大数据集上的可扩展性。为了提高访问控制列表的速度，提出了一种健壮性感知的核心重置选择(RCS)方法。RCS不需要标签信息，并且搜索最小化表示分歧的信息子集，表示分歧是自然数据和它们的虚拟对抗性变体之间的表示距离。通过遍历所有可能子集的RCS的香草解在计算上是令人望而却步的。因此，我们从理论上将RCS问题转化为子模最大化的代理问题，其中贪婪搜索是原问题的最优性保证的有效解。实验结果表明，RCS在不影响健壮性和可转移性的前提下，可以大幅度地提高ACL的速度。值得注意的是，据我们所知，我们是第一个在大规模ImageNet-1K数据集上高效地进行ACL的人，通过RCS获得了有效的健壮表示。



## **33. Single Node Injection Label Specificity Attack on Graph Neural Networks via Reinforcement Learning**

基于强化学习的图神经网络单节点注入标签专用性攻击 cs.LG

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02901v1) [paper-pdf](http://arxiv.org/pdf/2305.02901v1)

**Authors**: Dayuan Chen, Jian Zhang, Yuqian Lv, Jinhuan Wang, Hongjie Ni, Shanqing Yu, Zhen Wang, Qi Xuan

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various real-world applications. However, recent studies highlight the vulnerability of GNNs to malicious perturbations. Previous adversaries primarily focus on graph modifications or node injections to existing graphs, yielding promising results but with notable limitations. Graph modification attack~(GMA) requires manipulation of the original graph, which is often impractical, while graph injection attack~(GIA) necessitates training a surrogate model in the black-box setting, leading to significant performance degradation due to divergence between the surrogate architecture and the actual victim model. Furthermore, most methods concentrate on a single attack goal and lack a generalizable adversary to develop distinct attack strategies for diverse goals, thus limiting precise control over victim model behavior in real-world scenarios. To address these issues, we present a gradient-free generalizable adversary that injects a single malicious node to manipulate the classification result of a target node in the black-box evasion setting. We propose Gradient-free Generalizable Single Node Injection Attack, namely G$^2$-SNIA, a reinforcement learning framework employing Proximal Policy Optimization. By directly querying the victim model, G$^2$-SNIA learns patterns from exploration to achieve diverse attack goals with extremely limited attack budgets. Through comprehensive experiments over three acknowledged benchmark datasets and four prominent GNNs in the most challenging and realistic scenario, we demonstrate the superior performance of our proposed G$^2$-SNIA over the existing state-of-the-art baselines. Moreover, by comparing G$^2$-SNIA with multiple white-box evasion baselines, we confirm its capacity to generate solutions comparable to those of the best adversaries.

摘要: 图神经网络(GNN)在各种实际应用中取得了显著的成功。然而，最近的研究强调了GNN对恶意扰动的脆弱性。以前的对手主要集中在修改图或向现有图注入节点，产生了有希望的结果，但具有显著的局限性。图修改攻击~(GMA)需要对原始图进行操作，这往往是不切实际的，而图注入攻击~(GIA)需要在黑盒环境下训练代理模型，由于代理体系结构与实际受害者模型之间的差异，导致性能显著下降。此外，大多数方法集中在单个攻击目标上，缺乏一个可概括的对手来针对不同的目标制定不同的攻击策略，从而限制了对现实场景中受害者模型行为的精确控制。为了解决这些问题，我们提出了一种无梯度泛化攻击，在黑盒规避环境下注入单个恶意节点来操纵目标节点的分类结果。本文提出了一种无梯度泛化单节点注入攻击，即G$^2$-SNIA，这是一种基于近邻策略优化的强化学习框架。通过直接查询受害者模型，G$^2$-SNIA从探索中学习模式，以极其有限的攻击预算实现不同的攻击目标。通过在最具挑战性和最现实的场景中对三个公认的基准数据集和四个重要的GNN进行全面的实验，我们证明了我们提出的G$^2$-SNIA比现有的最先进的基线具有更好的性能。此外，通过将G$^2$-SNIA与多个白盒规避基线进行比较，我们证实了它产生与最好的对手相当的解的能力。



## **34. IMAP: Intrinsically Motivated Adversarial Policy**

IMAP：内在动机的对抗政策 cs.LG

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02605v1) [paper-pdf](http://arxiv.org/pdf/2305.02605v1)

**Authors**: Xiang Zheng, Xingjun Ma, Shengjie Wang, Xinyu Wang, Chao Shen, Cong Wang

**Abstract**: Reinforcement learning (RL) agents are known to be vulnerable to evasion attacks during deployment. In single-agent environments, attackers can inject imperceptible perturbations on the policy or value network's inputs or outputs; in multi-agent environments, attackers can control an adversarial opponent to indirectly influence the victim's observation. Adversarial policies offer a promising solution to craft such attacks. Still, current approaches either require perfect or partial knowledge of the victim policy or suffer from sample inefficiency due to the sparsity of task-related rewards. To overcome these limitations, we propose the Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box evasion attacks in single- and multi-agent environments without any knowledge of the victim policy. IMAP uses four intrinsic objectives based on state coverage, policy coverage, risk, and policy divergence to encourage exploration and discover stronger attacking skills. We also design a novel Bias-Reduction (BR) method to boost IMAP further. Our experiments demonstrate the effectiveness of these intrinsic objectives and BR in improving adversarial policy learning in the black-box setting against multiple types of victim agents in various single- and multi-agent MuJoCo environments. Notably, our IMAP reduces the performance of the state-of-the-art robust WocaR-PPO agents by 34\%-54\% and achieves a SOTA attacking success rate of 83.91\% in the two-player zero-sum game YouShallNotPass.

摘要: 强化学习(RL)代理在部署过程中容易受到逃避攻击。在单智能体环境中，攻击者可以在策略或价值网络的输入或输出上注入不可察觉的扰动；在多智能体环境中，攻击者可以控制对手来间接影响受害者的观察。对抗性政策为精心策划此类攻击提供了一个前景看好的解决方案。尽管如此，目前的方法要么需要完全或部分了解受害者政策，要么由于与任务相关的奖励稀少而导致样本效率低下。为了克服这些局限性，我们提出了本质动机对抗策略(IMAP)，用于在不知道受害者策略的情况下，在单代理和多代理环境中进行有效的黑盒逃避攻击。IMAP使用基于州覆盖、政策覆盖、风险和政策分歧的四个内在目标来鼓励探索和发现更强大的攻击技能。我们还设计了一种新的偏置减少(BR)方法来进一步提高IMAP。我们的实验证明了这些内在目标和BR在黑箱环境中针对各种单代理和多代理MuJoCo环境中多种受害者代理的对抗策略学习的有效性。值得注意的是，我们的IMAP使最先进的健壮WocaR-PPO代理的性能降低了34-54，并在两人零和游戏YouShallNotPass中实现了83.91\%的SOTA攻击成功率。



## **35. Madvex: Instrumentation-based Adversarial Attacks on Machine Learning Malware Detection**

MAdvex：机器学习恶意软件检测中基于工具的敌意攻击 cs.CR

20 pages. To be published in The 20th Conference on Detection of  Intrusions and Malware & Vulnerability Assessment (DIMVA 2023)

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02559v1) [paper-pdf](http://arxiv.org/pdf/2305.02559v1)

**Authors**: Nils Loose, Felix Mächtle, Claudius Pott, Volodymyr Bezsmertnyi, Thomas Eisenbarth

**Abstract**: WebAssembly (Wasm) is a low-level binary format for web applications, which has found widespread adoption due to its improved performance and compatibility with existing software. However, the popularity of Wasm has also led to its exploitation for malicious purposes, such as cryptojacking, where malicious actors use a victim's computing resources to mine cryptocurrencies without their consent. To counteract this threat, machine learning-based detection methods aiming to identify cryptojacking activities within Wasm code have emerged. It is well-known that neural networks are susceptible to adversarial attacks, where inputs to a classifier are perturbed with minimal changes that result in a crass misclassification. While applying changes in image classification is easy, manipulating binaries in an automated fashion to evade malware classification without changing functionality is non-trivial. In this work, we propose a new approach to include adversarial examples in the code section of binaries via instrumentation. The introduced gadgets allow for the inclusion of arbitrary bytes, enabling efficient adversarial attacks that reliably bypass state-of-the-art machine learning classifiers such as the CNN-based Minos recently proposed at NDSS 2021. We analyze the cost and reliability of instrumentation-based adversarial example generation and show that the approach works reliably at minimal size and performance overheads.

摘要: WebAssembly(WASM)是一种用于Web应用程序的低级二进制格式，由于其改进的性能和与现有软件的兼容性而被广泛采用。然而，WASM的流行也导致了对其进行恶意攻击，例如加密劫持，即恶意行为者在未经受害者同意的情况下使用受害者的计算资源来挖掘加密货币。为了应对这种威胁，出现了基于机器学习的检测方法，旨在识别WASM代码中的加密劫持活动。众所周知，神经网络容易受到敌意攻击，在这种攻击中，分类器的输入会受到干扰，只需进行极小的更改，就会导致粗略的错误分类。虽然在图像分类中应用更改很容易，但在不更改功能的情况下以自动方式操作二进制文件来规避恶意软件分类并不是一件容易的事情。在这项工作中，我们提出了一种新的方法，通过插装将对抗性示例包括在二进制文件的代码部分中。引入的小工具允许包含任意字节，从而实现了高效的对抗性攻击，可靠地绕过了最先进的机器学习分类器，例如最近在NDSS 2021上提出的基于CNN的Minos。我们分析了基于插桩的对抗性实例生成的代价和可靠性，结果表明该方法在最小规模和最小性能开销的情况下能够可靠地工作。



## **36. Detecting Adversarial Faces Using Only Real Face Self-Perturbations**

仅利用真实人脸自扰动检测敌方人脸 cs.CV

IJCAI2023

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2304.11359v2) [paper-pdf](http://arxiv.org/pdf/2304.11359v2)

**Authors**: Qian Wang, Yongqin Xian, Hefei Ling, Jinyuan Zhang, Xiaorui Lin, Ping Li, Jiazhong Chen, Ning Yu

**Abstract**: Adversarial attacks aim to disturb the functionality of a target system by adding specific noise to the input samples, bringing potential threats to security and robustness when applied to facial recognition systems. Although existing defense techniques achieve high accuracy in detecting some specific adversarial faces (adv-faces), new attack methods especially GAN-based attacks with completely different noise patterns circumvent them and reach a higher attack success rate. Even worse, existing techniques require attack data before implementing the defense, making it impractical to defend newly emerging attacks that are unseen to defenders. In this paper, we investigate the intrinsic generality of adv-faces and propose to generate pseudo adv-faces by perturbing real faces with three heuristically designed noise patterns. We are the first to train an adv-face detector using only real faces and their self-perturbations, agnostic to victim facial recognition systems, and agnostic to unseen attacks. By regarding adv-faces as out-of-distribution data, we then naturally introduce a novel cascaded system for adv-face detection, which consists of training data self-perturbations, decision boundary regularization, and a max-pooling-based binary classifier focusing on abnormal local color aberrations. Experiments conducted on LFW and CelebA-HQ datasets with eight gradient-based and two GAN-based attacks validate that our method generalizes to a variety of unseen adversarial attacks.

摘要: 敌意攻击的目的是通过在输入样本中添加特定的噪声来扰乱目标系统的功能，从而在应用于面部识别系统时给安全性和健壮性带来潜在威胁。虽然现有的防御技术对某些特定的敌方人脸(adv-Faces)的检测准确率很高，但新的攻击方法，特别是具有完全不同噪声模式的基于GAN的攻击，可以绕过它们，达到更高的攻击成功率。更糟糕的是，现有技术在实施防御之前需要攻击数据，因此防御防御者看不到的新出现的攻击是不切实际的。在本文中，我们研究了广告人脸的内在共性，并提出了通过用三种启发式设计的噪声模式来扰动真实人脸来生成伪广告人脸的方法。我们是第一个只使用真实人脸及其自身扰动来训练Adv-Face检测器的人，对受害者面部识别系统不可知，对看不见的攻击不可知。通过将广告人脸看作非分布数据，我们自然地提出了一种新的级联广告人脸检测系统，该系统包括训练数据自扰动、决策边界正则化和基于最大池的二进制分类器，该分类器针对异常局部颜色偏差。在LFW和CelebA-HQ数据集上对8个基于梯度的攻击和2个基于GAN的攻击进行了实验，验证了我们的方法适用于各种看不见的对抗性攻击。



## **37. On the Security Risks of Knowledge Graph Reasoning**

论知识图推理的安全风险 cs.CR

In proceedings of USENIX Security'23. Codes:  https://github.com/HarrialX/security-risk-KG-reasoning

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.02383v1) [paper-pdf](http://arxiv.org/pdf/2305.02383v1)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Xiapu Luo, Xusheng Xiao, Fenglong Ma, Ting Wang

**Abstract**: Knowledge graph reasoning (KGR) -- answering complex logical queries over large knowledge graphs -- represents an important artificial intelligence task, entailing a range of applications (e.g., cyber threat hunting). However, despite its surging popularity, the potential security risks of KGR are largely unexplored, which is concerning, given the increasing use of such capability in security-critical domains.   This work represents a solid initial step towards bridging the striking gap. We systematize the security threats to KGR according to the adversary's objectives, knowledge, and attack vectors. Further, we present ROAR, a new class of attacks that instantiate a variety of such threats. Through empirical evaluation in representative use cases (e.g., medical decision support, cyber threat hunting, and commonsense reasoning), we demonstrate that ROAR is highly effective to mislead KGR to suggest pre-defined answers for target queries, yet with negligible impact on non-target ones. Finally, we explore potential countermeasures against ROAR, including filtering of potentially poisoning knowledge and training with adversarially augmented queries, which leads to several promising research directions.

摘要: [TencentCloudSDKException] code:ClientNetworkError message:HTTPSConnectionPool(host='tmt.tencentcloudapi.com', port=443): Max retries exceeded with url: / (Caused by ProxyError('Cannot connect to proxy.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))) requestId:None



## **38. Privacy-Preserving Federated Recurrent Neural Networks**

隐私保护的联邦递归神经网络 cs.CR

Accepted for publication at the 23rd Privacy Enhancing Technologies  Symposium (PETS 2023)

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2207.13947v2) [paper-pdf](http://arxiv.org/pdf/2207.13947v2)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstract**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a cross-silo federated learning setting by relying on multiparty homomorphic encryption. RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates federated learning attacks that target the gradients under a passive-adversary threat model. We propose a packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, RHODE provides several clipping approximations for performing gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.

摘要: 我们提出了一种新的系统Rhode，它依靠多方同态加密在跨竖井的联合学习环境中实现对递归神经网络(RNN)的隐私保护训练和预测。Rhode保留了训练数据、模型和预测数据的机密性；它缓解了被动对手威胁模型下针对梯度的联合学习攻击。为了更好地利用加密环境下的单指令多数据(SIMD)运算，我们提出了一种多维打包方案。通过多维包装，Rhode能够并行高效地处理一批样品。为了避免爆炸渐变问题，Rhode提供了几种用于在加密情况下执行渐变裁剪的裁剪近似。我们的实验表明，对于数据持有者之间的同质和异质数据分布，Rhode模型的性能与非安全解决方案相似。我们的实验评估表明，Rhode与数据持有者数量和时间步数成线性关系，分别与RNN的特征数和隐含单元数成亚线性和次二次关系。据我们所知，Rhode是第一个在联合学习环境中加密的、为RNN及其变体的训练提供构建块的系统。



## **39. New Adversarial Image Detection Based on Sentiment Analysis**

一种新的基于情感分析的对抗性图像检测 cs.CR

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.03173v1) [paper-pdf](http://arxiv.org/pdf/2305.03173v1)

**Authors**: Yulong Wang, Tianxiang Li, Shenghong Li, Xin Yuan, Wei Ni

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial examples, while adversarial attack models, e.g., DeepFool, are on the rise and outrunning adversarial example detection techniques. This paper presents a new adversarial example detector that outperforms state-of-the-art detectors in identifying the latest adversarial attacks on image datasets. Specifically, we propose to use sentiment analysis for adversarial example detection, qualified by the progressively manifesting impact of an adversarial perturbation on the hidden-layer feature maps of a DNN under attack. Accordingly, we design a modularized embedding layer with the minimum learnable parameters to embed the hidden-layer feature maps into word vectors and assemble sentences ready for sentiment analysis. Extensive experiments demonstrate that the new detector consistently surpasses the state-of-the-art detection algorithms in detecting the latest attacks launched against ResNet and Inception neutral networks on the CIFAR-10, CIFAR-100 and SVHN datasets. The detector only has about 2 million parameters, and takes shorter than 4.6 milliseconds to detect an adversarial example generated by the latest attack models using a Tesla K80 GPU card.

摘要: 深度神经网络(DNN)容易受到敌意实例的影响，而DeepFool等敌意攻击模型正在兴起，并且超越了敌意实例检测技术。本文提出了一种新的敌意实例检测器，它在识别图像数据集上的最新敌意攻击方面优于最新的检测器。具体地说，我们建议使用情感分析来检测敌意示例，以敌意扰动对DNN在攻击下的隐含层特征映射的逐渐显现的影响为条件。相应地，我们设计了一个具有最小学习参数的模块化嵌入层，将隐含层特征映射嵌入到词向量中，并组装句子，为情感分析做好准备。大量的实验表明，新的检测器在检测针对CIFAR-10、CIFAR-100和SVHN数据集上的ResNet和Inception神经网络的最新攻击时，始终优于最新的检测算法。该检测器只有大约200万个参数，只需不到4.6毫秒就可以检测到使用特斯拉K80 GPU卡的最新攻击模型生成的对抗性示例。



## **40. Text Adversarial Purification as Defense against Adversarial Attacks**

文本对抗性净化作为对抗攻击的防御 cs.CL

Accepted by ACL2023 main conference

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2203.14207v2) [paper-pdf](http://arxiv.org/pdf/2203.14207v2)

**Authors**: Linyang Li, Demin Song, Xipeng Qiu

**Abstract**: Adversarial purification is a successful defense mechanism against adversarial attacks without requiring knowledge of the form of the incoming attack. Generally, adversarial purification aims to remove the adversarial perturbations therefore can make correct predictions based on the recovered clean samples. Despite the success of adversarial purification in the computer vision field that incorporates generative models such as energy-based models and diffusion models, using purification as a defense strategy against textual adversarial attacks is rarely explored. In this work, we introduce a novel adversarial purification method that focuses on defending against textual adversarial attacks. With the help of language models, we can inject noise by masking input texts and reconstructing the masked texts based on the masked language models. In this way, we construct an adversarial purification process for textual models against the most widely used word-substitution adversarial attacks. We test our proposed adversarial purification method on several strong adversarial attack methods including Textfooler and BERT-Attack and experimental results indicate that the purification algorithm can successfully defend against strong word-substitution attacks.

摘要: 对抗性净化是一种成功的防御对抗性攻击的机制，不需要知道即将到来的攻击的形式。通常，对抗性净化的目的是消除对抗性扰动，因此可以基于恢复的干净样本做出正确的预测。尽管对抗性净化在计算机视觉领域取得了成功，它结合了生成模型，如基于能量的模型和扩散模型，但将净化作为一种防御文本对抗攻击的策略却很少被探索。在这项工作中，我们介绍了一种新的对抗净化方法，该方法专注于防御文本对抗攻击。在语言模型的帮助下，我们可以通过对输入文本进行掩蔽并基于掩蔽的语言模型重建被掩蔽的文本来注入噪声。通过这种方式，我们为文本模型构建了对抗最广泛使用的词替换对抗攻击的对抗净化过程。我们在几种强对抗性攻击方法上对我们提出的对抗性净化方法进行了测试，实验结果表明，该净化算法能够成功地防御强词替换攻击。



## **41. Adversarial Neon Beam: Robust Physical-World Adversarial Attack to DNNs**

对抗性霓虹灯：对DNN的强大物理世界对抗性攻击 cs.CV

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2204.00853v2) [paper-pdf](http://arxiv.org/pdf/2204.00853v2)

**Authors**: Chengyin Hu, Kalibinuer Tiliwalidi

**Abstract**: In the physical world, light affects the performance of deep neural networks. Nowadays, many products based on deep neural network have been put into daily life. There are few researches on the effect of light on the performance of deep neural network models. However, the adversarial perturbations generated by light may have extremely dangerous effects on these systems. In this work, we propose an attack method called adversarial neon beam (AdvNB), which can execute the physical attack by obtaining the physical parameters of adversarial neon beams with very few queries. Experiments show that our algorithm can achieve advanced attack effect in both digital test and physical test. In the digital environment, 99.3% attack success rate was achieved, and in the physical environment, 100% attack success rate was achieved. Compared with the most advanced physical attack methods, our method can achieve better physical perturbation concealment. In addition, by analyzing the experimental data, we reveal some new phenomena brought about by the adversarial neon beam attack.

摘要: 在物理世界中，光线会影响深度神经网络的性能。如今，许多基于深度神经网络的产品已经进入日常生活。关于光照对深度神经网络模型性能影响的研究很少。然而，光产生的对抗性扰动可能会对这些系统产生极其危险的影响。在这项工作中，我们提出了一种称为对抗性霓虹束的攻击方法(AdvNB)，该方法只需很少的查询就可以获得对抗性霓虹束的物理参数来执行物理攻击。实验表明，该算法在数字测试和物理测试中均能达到较好的攻击效果。在数字环境下，攻击成功率达到99.3%，在物理环境下，攻击成功率达到100%。与最先进的物理攻击方法相比，我们的方法可以实现更好的物理扰动隐藏。此外，通过对实验数据的分析，揭示了对抗性霓虹束攻击带来的一些新现象。



## **42. Can Large Language Models Be an Alternative to Human Evaluations?**

大型语言模型能替代人类评估吗？ cs.CL

ACL 2023 main conference paper. Main content: 10 pages (including  limitations). Appendix: 13 pages

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.01937v1) [paper-pdf](http://arxiv.org/pdf/2305.01937v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: Human evaluation is indispensable and inevitable for assessing the quality of texts generated by machine learning models or written by humans. However, human evaluation is very difficult to reproduce and its quality is notoriously unstable, hindering fair comparisons among different natural language processing (NLP) models and algorithms. Recently, large language models (LLMs) have demonstrated exceptional performance on unseen tasks when only the task instructions are provided. In this paper, we explore if such an ability of the LLMs can be used as an alternative to human evaluation. We present the LLMs with the exact same instructions, samples to be evaluated, and questions used to conduct human evaluation, and then ask the LLMs to generate responses to those questions; we dub this LLM evaluation. We use human evaluation and LLM evaluation to evaluate the texts in two NLP tasks: open-ended story generation and adversarial attacks. We show that the result of LLM evaluation is consistent with the results obtained by expert human evaluation: the texts rated higher by human experts are also rated higher by the LLMs. We also find that the results of LLM evaluation are stable over different formatting of the task instructions and the sampling algorithm used to generate the answer. We are the first to show the potential of using LLMs to assess the quality of texts and discuss the limitations and ethical considerations of LLM evaluation.

摘要: 对于机器学习模型生成的文本或由人类编写的文本的质量进行评估时，人工评估是必不可少的，也是不可避免的。然而，人类的评价很难重现，而且其质量是出了名的不稳定，阻碍了不同自然语言处理(NLP)模型和算法之间的公平比较。最近，大型语言模型(LLM)在只提供任务指令的情况下，在看不见的任务上表现出了出色的性能。在本文中，我们探索LLMS的这种能力是否可以用作人类评估的替代方案。我们向LLMS提供完全相同的指令、要评估的样本和用于进行人工评估的问题，然后要求LLMS对这些问题做出回应；我们将这种LLM评估称为LLM评估。我们使用人类评价和LLM评价来评价两个自然语言处理任务中的文本：开放式故事生成和对抗性攻击。结果表明，LLM评价的结果与专家人类评价的结果是一致的：人类专家评得越高的文本，LLMS也会给出更高的评价。我们还发现，在不同的任务指令格式和用于生成答案的采样算法下，LLM评估的结果是稳定的。我们首先展示了使用LLMS评估文本质量的潜力，并讨论了LLM评估的局限性和伦理考虑。



## **43. Towards Imperceptible Document Manipulations against Neural Ranking Models**

针对神经排序模型的不可感知文档操作 cs.IR

Accepted to Findings of ACL 2023

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.01860v1) [paper-pdf](http://arxiv.org/pdf/2305.01860v1)

**Authors**: Xuanang Chen, Ben He, Zheng Ye, Le Sun, Yingfei Sun

**Abstract**: Adversarial attacks have gained traction in order to identify potential vulnerabilities in neural ranking models (NRMs), but current attack methods often introduce grammatical errors, nonsensical expressions, or incoherent text fragments, which can be easily detected. Additionally, current methods rely heavily on the use of a well-imitated surrogate NRM to guarantee the attack effect, which makes them difficult to use in practice. To address these issues, we propose a framework called Imperceptible DocumEnt Manipulation (IDEM) to produce adversarial documents that are less noticeable to both algorithms and humans. IDEM instructs a well-established generative language model, such as BART, to generate connection sentences without introducing easy-to-detect errors, and employs a separate position-wise merging strategy to balance relevance and coherence of the perturbed text. Experimental results on the popular MS MARCO benchmark demonstrate that IDEM can outperform strong baselines while preserving fluency and correctness of the target documents as evidenced by automatic and human evaluations. Furthermore, the separation of adversarial text generation from the surrogate NRM makes IDEM more robust and less affected by the quality of the surrogate NRM.

摘要: 为了识别神经排名模型(NRM)中的潜在漏洞，对抗性攻击已经获得了吸引力，但目前的攻击方法通常会引入语法错误、无意义的表达或不连贯的文本片段，这些都很容易被检测到。此外，目前的方法严重依赖于使用一个模仿得很好的代理NRM来保证攻击效果，这使得它们在实践中很难使用。为了解决这些问题，我们提出了一个称为不可感知文档操作(IDEM)的框架，以生成算法和人类都不太注意的敌意文档。Idem指导一个成熟的生成语言模型，如BART，在不引入容易检测的错误的情况下生成连接句子，并采用单独的位置合并策略来平衡扰动文本的关联性和连贯性。在流行的MS Marco基准上的实验结果表明，IDEM的性能优于强基线，同时保持了目标文档的流畅性和正确性，这一点得到了自动和人工评估的证明。此外，将敌意文本生成从代理NRM中分离出来，使得IDEM更健壮，受代理NRM质量的影响更小。



## **44. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

高分：使用生成模型对对抗性扰动进行全局稳健性评估 cs.LG

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2304.09875v2) [paper-pdf](http://arxiv.org/pdf/2304.09875v2)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.

摘要: 目前关于对抗稳健性的研究主要集中在从一组数据样本中聚集局部稳健性结果来评估和排序不同的模型。然而，局部统计可能不能很好地代表潜在未知数据分布的真实全局稳健性。为了应对这一挑战，本文首次尝试提出了一种新的框架，称为Great Score，用于利用产生式模型评估对抗扰动的全局稳健性。在形式上，高分具有全球统计的物理意义，该统计捕获来自生成模型的所有样本的平均经认证的防攻击扰动水平。对于有限样本评价，我们还得到了样本复杂度和样本均值与真均值之差的概率保证。Great Score有几个优点：(1)使用Great Score进行健壮性评估是高效的，并且可以扩展到大型模型，因为它避免了运行对抗性攻击的需要。特别是，与基于攻击的模型排名相比，我们表现出了高度的相关性和显著的降低了计算开销。艾尔2021年)。(2)生成模型的使用有利于未知数据分布的近似。在我们对不同生成对抗网络(GANS)的消融研究中，我们观察到全局健壮性评估与GANS质量之间的一致性。(3)Great Score可以用于隐私敏感的黑盒模型的远程审计，我们在几种在线人脸识别服务上的健壮性评估证明了这一点。



## **45. Sentiment Perception Adversarial Attacks on Neural Machine Translation Systems**

神经机器翻译系统中情感感知的敌意攻击 cs.CL

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01437v1) [paper-pdf](http://arxiv.org/pdf/2305.01437v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: With the advent of deep learning methods, Neural Machine Translation (NMT) systems have become increasingly powerful. However, deep learning based systems are susceptible to adversarial attacks, where imperceptible changes to the input can cause undesirable changes at the output of the system. To date there has been little work investigating adversarial attacks on sequence-to-sequence systems, such as NMT models. Previous work in NMT has examined attacks with the aim of introducing target phrases in the output sequence. In this work, adversarial attacks for NMT systems are explored from an output perception perspective. Thus the aim of an attack is to change the perception of the output sequence, without altering the perception of the input sequence. For example, an adversary may distort the sentiment of translated reviews to have an exaggerated positive sentiment. In practice it is challenging to run extensive human perception experiments, so a proxy deep-learning classifier applied to the NMT output is used to measure perception changes. Experiments demonstrate that the sentiment perception of NMT systems' output sequences can be changed significantly.

摘要: 随着深度学习方法的出现，神经机器翻译(NMT)系统变得越来越强大。然而，基于深度学习的系统很容易受到对抗性攻击，在这种攻击中，输入的不可察觉的变化可能会导致系统输出的不希望看到的变化。到目前为止，很少有人研究针对序列到序列系统的对抗性攻击，例如NMT模型。NMT之前的工作已经检查了攻击，目的是在输出序列中引入目标短语。本文从输出感知的角度研究了NMT系统的敌意攻击问题。因此，攻击的目的是改变对输出序列的感知，而不改变对输入序列的感知。例如，对手可能会扭曲翻译后的评论的情绪，使其具有夸大的积极情绪。在实践中，进行广泛的人类感知实验是具有挑战性的，因此将代理深度学习分类器应用于NMT输出来测量感知变化。实验表明，NMT系统输出序列的情感感知可以发生显著变化。



## **46. Improving adversarial robustness by putting more regularizations on less robust samples**

通过对健壮性较差的样本进行更多的正则化来提高对手的稳健性 stat.ML

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2206.03353v3) [paper-pdf](http://arxiv.org/pdf/2206.03353v3)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstract**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.

摘要: 对抗性训练是为了提高对抗攻击的稳健性，因为它很容易产生人类无法察觉的数据扰动来欺骗给定的深度神经网络。在本文中，我们提出了一种新的对抗性训练算法，该算法在理论上动机良好，在经验上优于其他现有的算法。与现有的正则化算法相比，该算法的一个新特点是对易受敌意攻击的数据进行了更多的正则化。理论上，我们的算法可以理解为最小化正则化经验风险的算法，该正则化经验风险是由新导出的稳健风险上界引起的。数值实验表明，我们提出的算法同时提高了泛化(例题准确率)和稳健性(对抗性攻击准确率)，达到了最好的性能。



## **47. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过样式转换愚弄视频分类系统 cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2203.16000v3) [paper-pdf](http://arxiv.org/pdf/2203.16000v3)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在有针对性的攻击中，还考虑了目标类置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后使用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。此外，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以愚弄视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **48. Exposing Fine-Grained Adversarial Vulnerability of Face Anti-Spoofing Models**

暴露Face反欺骗模型的细粒度攻击漏洞 cs.CV

Accepted by IEEE/CVF Conference on Computer Vision and Pattern  Recognition (CVPR) Workshop, 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2205.14851v3) [paper-pdf](http://arxiv.org/pdf/2205.14851v3)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Face anti-spoofing aims to discriminate the spoofing face images (e.g., printed photos) from live ones. However, adversarial examples greatly challenge its credibility, where adding some perturbation noise can easily change the predictions. Previous works conducted adversarial attack methods to evaluate the face anti-spoofing performance without any fine-grained analysis that which model architecture or auxiliary feature is vulnerable to the adversary. To handle this problem, we propose a novel framework to expose the fine-grained adversarial vulnerability of the face anti-spoofing models, which consists of a multitask module and a semantic feature augmentation (SFA) module. The multitask module can obtain different semantic features for further evaluation, but only attacking these semantic features fails to reflect the discrimination-related vulnerability. We then design the SFA module to introduce the data distribution prior for more discrimination-related gradient directions for generating adversarial examples. Comprehensive experiments show that SFA module increases the attack success rate by nearly 40$\%$ on average. We conduct this fine-grained adversarial analysis on different annotations, geometric maps, and backbone networks (e.g., Resnet network). These fine-grained adversarial examples can be used for selecting robust backbone networks and auxiliary features. They also can be used for adversarial training, which makes it practical to further improve the accuracy and robustness of the face anti-spoofing models.

摘要: 人脸反欺骗的目的是区分伪造的人脸图像(如打印的照片)和活的人脸图像。然而，对抗性的例子极大地挑战了它的可信度，在那里添加一些扰动噪声很容易改变预测。以往的工作采用对抗性攻击的方法来评估人脸的反欺骗性能，没有任何细粒度的分析来确定哪个模型、架构或辅助特征容易受到对手的攻击。为了解决这个问题，我们提出了一种新的框架来暴露人脸反欺骗模型的细粒度攻击漏洞，该框架由多任务模块和语义特征增强(SFA)模块组成。多任务模块可以获得不同的语义特征用于进一步的评估，但仅攻击这些语义特征并不能反映与歧视相关的脆弱性。然后，我们设计了SFA模块来引入数据分布，以获得更多与区分相关的梯度方向，以生成对抗性示例。综合实验表明，SFA模块的攻击成功率平均提高了近40美元。我们在不同的注释、几何地图和骨干网络(例如RESNET网络)上进行了这种细粒度的对抗性分析。这些细粒度的对抗性实例可用于选择健壮的主干网络和辅助特征。它们还可以用于对抗性训练，从而进一步提高人脸反欺骗模型的准确性和稳健性。



## **49. Stratified Adversarial Robustness with Rejection**

具有拒绝的分层对抗健壮性 cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01139v1) [paper-pdf](http://arxiv.org/pdf/2305.01139v1)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.

摘要: 最近，对抗性地训练具有拒绝选项的分类器(也称为选择性分类器)以增强对抗性健壮性是一种新的兴趣。虽然拒绝在许多应用中可能会导致成本，但现有研究通常将零成本与拒绝扰动输入联系在一起，这可能导致拒绝许多可以正确分类的轻微扰动输入。在这项工作中，我们研究了分层拒绝环境下的具有拒绝的对抗性鲁棒分类，其中拒绝代价由拒绝损失函数来建模，拒绝损失函数在扰动幅度上单调地不增加。我们从理论上分析了分层拒绝的设置，并提出了一种新的防御方法--基于一致预测拒绝的对抗训练(CPR)--来构建一个健壮的选择性分类器。在图像数据集上的实验表明，该方法在强自适应攻击下的性能明显优于已有方法。例如，在CIFAR-10上，CPR在看得见和看不见的攻击下都将总的稳健损失(针对不同的拒绝损失)减少了至少7.3%。



## **50. Randomized Reversible Gate-Based Obfuscation for Secured Compilation of Quantum Circuit**

基于随机可逆门的量子电路安全编译混淆算法 quant-ph

11 pages, 12 figures, conference

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01133v1) [paper-pdf](http://arxiv.org/pdf/2305.01133v1)

**Authors**: Subrata Das, Swaroop Ghosh

**Abstract**: The success of quantum circuits in providing reliable outcomes for a given problem depends on the gate count and depth in near-term noisy quantum computers. Quantum circuit compilers that decompose high-level gates to native gates of the hardware and optimize the circuit play a key role in quantum computing. However, the quality and time complexity of the optimization process can vary significantly especially for practically relevant large-scale quantum circuits. As a result, third-party (often less-trusted/untrusted) compilers have emerged, claiming to provide better and faster optimization of complex quantum circuits than so-called trusted compilers. However, untrusted compilers can pose severe security risks, such as the theft of sensitive intellectual property (IP) embedded within the quantum circuit. We propose an obfuscation technique for quantum circuits using randomized reversible gates to protect them from such attacks during compilation. The idea is to insert a small random circuit into the original circuit and send it to the untrusted compiler. Since the circuit function is corrupted, the adversary may get incorrect IP. However, the user may also get incorrect output post-compilation. To circumvent this issue, we concatenate the inverse of the random circuit in the compiled circuit to recover the original functionality. We demonstrate the practicality of our method by conducting exhaustive experiments on a set of benchmark circuits and measuring the quality of obfuscation by calculating the Total Variation Distance (TVD) metric. Our method achieves TVD of up to 1.92 and performs at least 2X better than a previously reported obfuscation method. We also propose a novel adversarial reverse engineering (RE) approach and show that the proposed obfuscation is resilient against RE attacks. The proposed technique introduces minimal degradation in fidelity (~1% to ~3% on average).

摘要: 量子电路在为给定问题提供可靠结果方面的成功取决于近期嘈杂的量子计算机中的门数量和深度。量子电路编译器将高层门分解为硬件的本机门，并对电路进行优化，在量子计算中发挥着关键作用。然而，优化过程的质量和时间复杂性可能会有很大的变化，特别是对于实际相关的大规模量子电路。因此，第三方(通常不太可信/不可信)编译器应运而生，声称比所谓的可信编译器提供更好、更快的复杂量子电路优化。然而，不可信的编译器可能会带来严重的安全风险，例如嵌入量子电路中的敏感知识产权(IP)被窃取。我们提出了一种量子电路的混淆技术，该技术使用随机化的可逆门来保护它们在编译过程中免受此类攻击。其想法是在原始电路中插入一个小的随机电路，并将其发送给不可信的编译器。由于电路功能被破坏，对手可能获得错误的IP。但是，用户也可能在编译后得到不正确的输出。为了避免这个问题，我们将编译电路中的随机电路的逆连接起来，以恢复原来的功能。我们在一组基准电路上进行了详尽的实验，并通过计算总变化距离(TVD)度量来衡量混淆质量，从而证明了该方法的实用性。我们的方法获得了高达1.92的TVD，并且比先前报道的混淆方法的性能至少提高了2倍。我们还提出了一种新的对抗性逆向工程(RE)方法，并证明了该方法对逆向工程攻击具有较强的抵抗力。提出的技术在保真度方面引入了最小的降级(平均~1%到~3%)。



