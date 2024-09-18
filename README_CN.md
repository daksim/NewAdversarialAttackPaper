# Latest Adversarial Attack Papers
**update at 2024-09-18 09:36:32**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

EIA：针对多面手网络代理隐私泄露的环境注入攻击 cs.CR

24 pages

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11295v1) [paper-pdf](http://arxiv.org/pdf/2409.11295v1)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have evolved rapidly and demonstrated remarkable potential. However, there are unprecedented safety risks associated with these them, which are nearly unexplored so far. In this work, we aim to narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a threat model that discusses the adversarial targets, constraints, and attack scenarios. Particularly, we consider two types of adversarial targets: stealing users' specific personally identifiable information (PII) or stealing the entire user request. To achieve these objectives, we propose a novel attack method, termed Environmental Injection Attack (EIA). This attack injects malicious content designed to adapt well to different environments where the agents operate, causing them to perform unintended actions. This work instantiates EIA specifically for the privacy scenario. It inserts malicious web elements alongside persuasive instructions that mislead web agents into leaking private information, and can further leverage CSS and JavaScript features to remain stealthy. We collect 177 actions steps that involve diverse PII categories on realistic websites from the Mind2Web dataset, and conduct extensive experiments using one of the most capable generalist web agent frameworks to date, SeeAct. The results demonstrate that EIA achieves up to 70% ASR in stealing users' specific PII. Stealing full user requests is more challenging, but a relaxed version of EIA can still achieve 16% ASR. Despite these concerning results, it is important to note that the attack can still be detectable through careful human inspection, highlighting a trade-off between high autonomy and security. This leads to our detailed discussion on the efficacy of EIA under different levels of human supervision as well as implications on defenses for generalist web agents.

摘要: 多面手网络代理发展迅速，并显示出非凡的潜力。然而，它们存在着前所未有的安全风险，到目前为止几乎没有人探索过。在这项工作中，我们旨在通过对对抗环境中通才网络代理的隐私风险进行第一次研究来缩小这一差距。首先，我们提出了一个威胁模型，该模型讨论了对抗性目标、约束和攻击场景。具体地说，我们考虑了两种类型的对抗目标：窃取用户特定的个人身份信息(PII)或窃取整个用户请求。为了实现这些目标，我们提出了一种新的攻击方法，称为环境注入攻击(EIA)。此攻击注入恶意内容，旨在很好地适应代理程序运行的不同环境，导致它们执行意外操作。这项工作专门为隐私场景实例化了EIA。它将恶意的网络元素与具有说服力的指令一起插入，误导网络代理泄露私人信息，并可以进一步利用CSS和JavaScript功能来保持隐蔽性。我们从Mind2Web数据集中收集了177个动作步骤，涉及现实网站上的不同PII类别，并使用迄今最有能力的通用Web代理框架之一SeeAct进行了广泛的实验。结果表明，在窃取用户特定PII时，EIA的ASR高达70%。窃取完整的用户请求更具挑战性，但宽松版本的EIA仍可实现16%的ASR。尽管有这些令人担忧的结果，但必须指出的是，通过仔细的人工检查仍然可以检测到攻击，这突显了高度自治和安全之间的权衡。这导致了我们详细讨论了在不同级别的人类监督下的EIA的有效性，以及对多面手网络代理的防御的影响。



## **2. Backdoor Attacks in Peer-to-Peer Federated Learning**

点对点联邦学习中的后门攻击 cs.LG

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2301.09732v4) [paper-pdf](http://arxiv.org/pdf/2301.09732v4)

**Authors**: Georgios Syros, Gokberk Yar, Simona Boboila, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that successfully mitigates the backdoor attacks, without an impact on model accuracy.

摘要: 大多数机器学习应用程序依赖于集中的学习过程，从而打开了暴露其训练数据集的风险。虽然联合学习(FL)在一定程度上缓解了这些隐私风险，但它依赖于可信的聚合服务器来训练共享的全局模型。近年来，基于对等联合学习(P2P-to-Peer Federated Learning，简称P2PFL)的新型分布式学习体系结构在保密性和可靠性方面都具有优势。尽管如此，他们在训练期间对中毒攻击的抵抗力还没有得到调查。本文提出了一种新的针对P2P PFL的后门攻击，利用结构图的性质来选择恶意节点，在保持隐蔽性的同时获得较高的攻击成功率。我们在各种现实条件下评估我们的攻击，包括多个图拓扑、有限的网络敌意可见性以及具有非IID数据的客户端。最后，我们指出了现有防御方案的局限性，并设计了一种新的防御方案，在不影响模型精度的情况下，成功地缓解了后门攻击。



## **3. A Survey of Machine Unlearning**

机器学习研究 cs.LG

extend the survey with more recent published work and add more  discussions

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2209.02299v6) [paper-pdf](http://arxiv.org/pdf/2209.02299v6)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Zhao Ren, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 今天，计算机系统保存着大量的个人数据。然而，尽管如此丰富的数据使人工智能，特别是机器学习(ML)取得了突破，但它的存在可能会对用户隐私构成威胁，并可能削弱人类与人工智能之间的信任纽带。最近的法规现在要求，根据请求，必须从计算机系统和ML模型中删除关于用户的私人信息，即“被遗忘权”)。虽然从后端数据库中删除数据应该是直接的，但在人工智能上下文中这是不够的，因为ML模型经常‘记住’旧数据。当代针对训练模型的对抗性攻击已经证明，我们可以学习到一个实例或一个属性是否属于训练数据。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。因此，本文致力于对机器遗忘的概念、场景、方法和应用进行全面的考察。具体地说，作为尖端研究的类别集合，本文背后的目的是为寻求介绍机器遗忘及其公式、设计标准、移除请求、算法和应用的研究人员和从业者提供全面的资源。此外，我们的目标是强调关键的发现、当前的趋势和新的研究领域，这些领域还没有使用机器遗忘，但可以从中受益匪浅。我们希望这项调查对ML研究人员和那些寻求创新隐私技术的人来说是一个有价值的资源。我们的资源可在https://github.com/tamlhp/awesome-machine-unlearning.上公开获取



## **4. Remote Keylogging Attacks in Multi-user VR Applications**

多用户VR应用程序中的远程键盘记录攻击 cs.CR

Accepted for Usenix 2024

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2405.14036v2) [paper-pdf](http://arxiv.org/pdf/2405.14036v2)

**Authors**: Zihao Su, Kunlin Cai, Reuben Beeler, Lukas Dresel, Allan Garcia, Ilya Grishchenko, Yuan Tian, Christopher Kruegel, Giovanni Vigna

**Abstract**: As Virtual Reality (VR) applications grow in popularity, they have bridged distances and brought users closer together. However, with this growth, there have been increasing concerns about security and privacy, especially related to the motion data used to create immersive experiences. In this study, we highlight a significant security threat in multi-user VR applications, which are applications that allow multiple users to interact with each other in the same virtual space. Specifically, we propose a remote attack that utilizes the avatar rendering information collected from an adversary's game clients to extract user-typed secrets like credit card information, passwords, or private conversations. We do this by (1) extracting motion data from network packets, and (2) mapping motion data to keystroke entries. We conducted a user study to verify the attack's effectiveness, in which our attack successfully inferred 97.62% of the keystrokes. Besides, we performed an additional experiment to underline that our attack is practical, confirming its effectiveness even when (1) there are multiple users in a room, and (2) the attacker cannot see the victims. Moreover, we replicated our proposed attack on four applications to demonstrate the generalizability of the attack. Lastly, we proposed a defense against the attack, which has been implemented by major players in the VR industry. These results underscore the severity of the vulnerability and its potential impact on millions of VR social platform users.

摘要: 随着虚拟现实(VR)应用越来越受欢迎，它们弥合了距离，拉近了用户之间的距离。然而，随着这种增长，人们对安全和隐私的担忧也越来越多，特别是与用于创建身临其境体验的运动数据有关。在这项研究中，我们强调了多用户VR应用中的一个重大安全威胁，即允许多个用户在同一虚拟空间中相互交互的应用。具体地说，我们提出了一种远程攻击，它利用从对手的游戏客户端收集的化身渲染信息来提取用户键入的秘密，如信用卡信息、密码或私人对话。我们通过(1)从网络分组中提取运动数据，以及(2)将运动数据映射到击键条目来实现这一点。我们进行了用户研究来验证攻击的有效性，其中我们的攻击成功推断了97.62%的击键。此外，我们还执行了一个额外的实验，以强调我们的攻击是实用的，即使在(1)一个房间有多个用户，以及(2)攻击者看不到受害者的情况下，也证实了它的有效性。此外，我们在四个应用程序上复制了我们提出的攻击，以证明该攻击的泛化能力。最后，我们提出了针对攻击的防御方案，并已被VR行业的主要参与者实施。这些结果突显了该漏洞的严重性及其对数百万VR社交平台用户的潜在影响。



## **5. An Anti-disguise Authentication System Using the First Impression of Avatar in Metaverse**

利用虚拟宇宙阿凡达第一印象的反伪装认证系统 cs.CR

19 pages, 16 figures

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.10850v1) [paper-pdf](http://arxiv.org/pdf/2409.10850v1)

**Authors**: Zhenyong Zhang, Kedi Yang, Youliang Tian, Jianfeng Ma

**Abstract**: Metaverse is a vast virtual world parallel to the physical world, where the user acts as an avatar to enjoy various services that break through the temporal and spatial limitations of the physical world. Metaverse allows users to create arbitrary digital appearances as their own avatars by which an adversary may disguise his/her avatar to fraud others. In this paper, we propose an anti-disguise authentication method that draws on the idea of the first impression from the physical world to recognize an old friend. Specifically, the first meeting scenario in the metaverse is stored and recalled to help the authentication between avatars. To prevent the adversary from replacing and forging the first impression, we construct a chameleon-based signcryption mechanism and design a ciphertext authentication protocol to ensure the public verifiability of encrypted identities. The security analysis shows that the proposed signcryption mechanism meets not only the security requirement but also the public verifiability. Besides, the ciphertext authentication protocol has the capability of defending against the replacing and forging attacks on the first impression. Extensive experiments show that the proposed avatar authentication system is able to achieve anti-disguise authentication at a low storage consumption on the blockchain.

摘要: Metverse是一个与物理世界平行的广阔虚拟世界，用户在其中扮演化身，享受各种突破物理世界时空限制的服务。Metverse允许用户创建任意的数字外观作为他们自己的化身，对手可以利用这些化身来伪装他/她的化身以欺骗他人。在本文中，我们提出了一种反伪装认证方法，该方法借鉴了物理世界第一印象的思想来识别老朋友。具体地说，存储和调用虚拟世界中的第一个会议场景，以帮助在化身之间进行身份验证。为了防止攻击者替换和伪造第一印象，我们构造了一个基于变色龙的签密机制，并设计了一个密文认证协议来确保加密身份的公开可验证性。安全性分析表明，该签密机制不仅满足安全性要求，而且具有公开可验证性。此外，密文认证协议还具有抵抗替换攻击和伪造第一印象攻击的能力。大量实验表明，所提出的头像认证系统能够在区块链上以较低的存储消耗实现反伪装认证。



## **6. Weak Superimposed Codes of Improved Asymptotic Rate and Their Randomized Construction**

改进渐进率的弱叠加码及其随机构造 cs.IT

6 pages, accepted for presentation at the 2022 IEEE International  Symposium on Information Theory (ISIT)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.10511v1) [paper-pdf](http://arxiv.org/pdf/2409.10511v1)

**Authors**: Yu Tsunoda, Yuichiro Fujiwara

**Abstract**: Weak superimposed codes are combinatorial structures related closely to generalized cover-free families, superimposed codes, and disjunct matrices in that they are only required to satisfy similar but less stringent conditions. This class of codes may also be seen as a stricter variant of what are known as locally thin families in combinatorics. Originally, weak superimposed codes were introduced in the context of multimedia content protection against illegal distribution of copies under the assumption that a coalition of malicious users may employ the averaging attack with adversarial noise. As in many other kinds of codes in information theory, it is of interest and importance in the study of weak superimposed codes to find the highest achievable rate in the asymptotic regime and give an efficient construction that produces an infinite sequence of codes that achieve it. Here, we prove a tighter lower bound than the sharpest known one on the rate of optimal weak superimposed codes and give a polynomial-time randomized construction algorithm for codes that asymptotically attain our improved bound with high probability. Our probabilistic approach is versatile and applicable to many other related codes and arrays.

摘要: 弱叠加码是一种与广义无覆盖族、叠加码和析取矩阵密切相关的组合结构，它们只需要满足相似但不那么严格的条件。这类代码也可以被视为组合数学中所知的局部瘦族的更严格的变体。最初，在防止非法分发副本的多媒体内容保护的环境中引入了弱叠加代码，假设恶意用户的联盟可能采用带有对抗性噪声的平均攻击。正如信息论中的许多其他类型的码一样，在弱重叠码的研究中，寻找渐近状态下的最高可达速率并给出一种有效的构造以产生实现它的无限序列是很有意义和重要的。在这里，我们证明了最优弱叠加码码率的一个比已知的最强下界更紧的下界，并给出了一个多项式时间的随机构造算法，它以很高的概率渐近地达到我们的改进界。我们的概率方法是通用的，并适用于许多其他相关的代码和数组。



## **7. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

通过查询高效抽样攻击评估大型语言模型中生物医学知识的稳健性 cs.CL

28 pages incl. appendix, updated version

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.10527v2) [paper-pdf](http://arxiv.org/pdf/2402.10527v2)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。了解高风险和知识密集型任务中的模型脆弱性对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性实例(即对抗性实体)，这引发了人们对高风险和专门领域中预先训练和精细调整的LLM知识稳健性的潜在影响的问题。我们研究了使用类型一致的实体替换作为收集具有生物医学知识的10亿参数LLM的对抗性实体的模板。为此，我们提出了一种基于加权距离加权抽样的嵌入空间攻击方法，以较低的查询预算和可控的覆盖率来评估他们的生物医学知识的稳健性。与基于随机抽样和黑盒梯度引导搜索的方法相比，我们的方法具有良好的查询效率和伸缩性，并在生物医学问答中的对抗性干扰项生成中得到了验证。随后的失效模式分析揭示了攻击面上具有不同特征的两种对抗实体的机制，我们表明实体替换攻击可以操纵令人信服的Shapley值解释，在这种情况下，这种解释变得具有欺骗性。我们的方法补充了对大容量模型的标准评估，结果突出了领域知识在LLMS中的脆性。



## **8. Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey**

自主网络运营的深度强化学习：调查 cs.LG

89 pages, 14 figures, 4 tables

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2310.07745v2) [paper-pdf](http://arxiv.org/pdf/2310.07745v2)

**Authors**: Gregory Palmer, Chris Parry, Daniel J. B. Harrold, Chris Willis

**Abstract**: The rapid increase in the number of cyber-attacks in recent years raises the need for principled methods for defending networks against malicious actors. Deep reinforcement learning (DRL) has emerged as a promising approach for mitigating these attacks. However, while DRL has shown much potential for cyber defence, numerous challenges must be overcome before DRL can be applied to autonomous cyber operations (ACO) at scale. Principled methods are required for environments that confront learners with very high-dimensional state spaces, large multi-discrete action spaces, and adversarial learning. Recent works have reported success in solving these problems individually. There have also been impressive engineering efforts towards solving all three for real-time strategy games. However, applying DRL to the full ACO problem remains an open challenge. Here, we survey the relevant DRL literature and conceptualize an idealised ACO-DRL agent. We provide: i.) A summary of the domain properties that define the ACO problem; ii.) A comprehensive comparison of current ACO environments used for benchmarking DRL approaches; iii.) An overview of state-of-the-art approaches for scaling DRL to domains that confront learners with the curse of dimensionality, and; iv.) A survey and critique of current methods for limiting the exploitability of agents within adversarial settings from the perspective of ACO. We conclude with open research questions that we hope will motivate future directions for researchers and practitioners working on ACO.

摘要: 近年来，网络攻击数量的迅速增加增加了对保护网络免受恶意行为侵害的原则性方法的需求。深度强化学习(DRL)已成为缓解这些攻击的一种很有前途的方法。然而，尽管DRL在网络防御方面显示出了很大的潜力，但在DRL能够大规模应用于自主网络作战(ACO)之前，必须克服许多挑战。对于学习者面对高维状态空间、大的多离散动作空间和对抗性学习的环境，需要有原则性的方法。最近的研究报告成功地单独解决了这些问题。也有令人印象深刻的工程努力，以解决所有这三个实时战略游戏。然而，将DRL应用于整个蚁群优化问题仍然是一个开放的挑战。在这里，我们回顾了相关的DRL文献，并概念化了一个理想的ACO-DRL试剂。我们提供：i.)定义ACO问题的域属性摘要；ii.)对当前用于基准DRL方法的ACO环境进行了全面比较；三.)概述将DRL扩展到学习者面临维度诅咒的领域的最新方法，以及；i.)从蚁群算法的角度对当前在对抗性环境中限制代理的可利用性的方法进行了调查和评论。我们以开放的研究问题结束，我们希望这些问题将激励从事ACO工作的研究人员和从业者未来的方向。



## **9. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

机器对抗RAG：用阻止器文档干扰检索增强生成 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.05870v2) [paper-pdf](http://arxiv.org/pdf/2406.05870v2)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database, then generating an answer by applying an LLM to the retrieved documents. We demonstrate that RAG systems that operate on databases with untrusted content are vulnerable to a new class of denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and result in the RAG system not answering this query - ostensibly because it lacks the information or because the answer is unsafe.   We describe and measure the efficacy of several methods for generating blocker documents, including a new method based on black-box optimization. This method (1) does not rely on instruction injection, (2) does not require the adversary to know the embedding or LLM used by the target RAG system, and (3) does not use an auxiliary LLM to generate blocker documents.   We evaluate jamming attacks on several LLMs and embeddings and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.

摘要: 检索-增强生成(RAG)系统通过从知识数据库中检索相关文档，然后通过将LLM应用于所检索的文档来生成答案来响应查询。我们证明，在含有不可信内容的数据库上运行的RAG系统容易受到一种新的拒绝服务攻击，我们称之为干扰。敌手可以向数据库添加一个“拦截器”文档，该文档将响应于特定查询而被检索，并导致RAG系统不回答该查询--表面上是因为它缺乏信息或因为答案不安全。我们描述并测试了几种生成拦截器文档的方法的有效性，其中包括一种基于黑盒优化的新方法。该方法(1)不依赖于指令注入，(2)不要求对手知道目标RAG系统使用的嵌入或LLM，以及(3)不使用辅助LLM来生成拦截器文档。我们评估了几个LLM和嵌入上的干扰攻击，并证明了现有的LLM安全度量没有捕捉到它们对干扰的脆弱性。然后我们讨论针对拦截器文档的防御。



## **10. Towards Evaluating the Robustness of Visual State Space Models**

评估视觉状态空间模型的稳健性 cs.CV

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.09407v2) [paper-pdf](http://arxiv.org/pdf/2406.09407v2)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Vision State Space Models (VSSMs), a novel architecture that combines the strengths of recurrent neural networks and latent variable models, have demonstrated remarkable performance in visual perception tasks by efficiently capturing long-range dependencies and modeling complex visual dynamics. However, their robustness under natural and adversarial perturbations remains a critical concern. In this work, we present a comprehensive evaluation of VSSMs' robustness under various perturbation scenarios, including occlusions, image structure, common corruptions, and adversarial attacks, and compare their performance to well-established architectures such as transformers and Convolutional Neural Networks. Furthermore, we investigate the resilience of VSSMs to object-background compositional changes on sophisticated benchmarks designed to test model performance in complex visual scenes. We also assess their robustness on object detection and segmentation tasks using corrupted datasets that mimic real-world scenarios. To gain a deeper understanding of VSSMs' adversarial robustness, we conduct a frequency-based analysis of adversarial attacks, evaluating their performance against low-frequency and high-frequency perturbations. Our findings highlight the strengths and limitations of VSSMs in handling complex visual corruptions, offering valuable insights for future research. Our code and models will be available at https://github.com/HashmatShadab/MambaRobustness.

摘要: 视觉状态空间模型(VSSMS)是一种结合了递归神经网络和潜变量模型优点的新型结构，通过有效地捕捉长距离依赖关系和建模复杂的视觉动力学，在视觉感知任务中表现出了显著的性能。然而，它们在自然和对抗性扰动下的稳健性仍然是一个严重的问题。在这项工作中，我们对VSSM在各种扰动场景下的健壮性进行了全面的评估，包括遮挡、图像结构、常见的腐败和敌对攻击，并将它们的性能与成熟的架构，如变压器和卷积神经网络进行了比较。此外，我们在复杂的基准测试中考察了VSSM对对象-背景成分变化的弹性，该基准旨在测试复杂视觉场景中的模型性能。我们还使用模拟真实世界场景的损坏数据集评估了它们在对象检测和分割任务中的稳健性。为了更深入地了解VSSM的对抗稳健性，我们对对抗攻击进行了基于频率的分析，评估了它们对低频和高频扰动的性能。我们的发现突出了VSSM在处理复杂视觉腐败方面的优势和局限性，为未来的研究提供了有价值的见解。我们的代码和模型将在https://github.com/HashmatShadab/MambaRobustness.上提供



## **11. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.10071v1) [paper-pdf](http://arxiv.org/pdf/2409.10071v1)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].

摘要: 在安全关键环境中部署具体化导航代理引起了人们对它们在深层神经网络上易受敌意攻击的担忧。然而，由于从数字世界向物理世界过渡的挑战，现有的攻击方法往往缺乏实用性，而现有的针对目标检测的物理攻击无法达到多视角的有效性和自然性。为了解决这一问题，我们提出了一种实用的具身导航攻击方法，通过将具有可学习纹理和不透明度的敌意补丁附加到对象上。具体地说，为了确保不同视点的有效性，我们采用了一种基于对象感知采样的多视点优化策略，该策略利用导航模型的反馈来优化面片的纹理。为了使面片不易被人察觉，我们引入了一种两阶段不透明度优化机制，在纹理优化后对不透明度进行细化。实验结果表明，我们的对抗性补丁使导航成功率降低了约40%，在实用性、有效性和自然性方面都优于以往的方法。代码可从以下网址获得：[https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



## **12. Multi-agent Attacks for Black-box Social Recommendations**

针对黑匣子社交推荐的多代理攻击 cs.SI

Accepted by ACM TOIS

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2311.07127v4) [paper-pdf](http://arxiv.org/pdf/2311.07127v4)

**Authors**: Shijie Wang, Wenqi Fan, Xiao-yong Wei, Xiaowei Mei, Shanru Lin, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks (GNNs) in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on argeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework MultiAttack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.

摘要: 在线社交网络的兴起促进了社交推荐系统的发展，社交推荐系统整合了社会关系，以增强用户的决策过程。随着图神经网络(GNN)在学习节点表示方面的巨大成功，基于GNN的社交推荐被广泛研究以同时建模用户-项目交互和用户-用户社会关系。尽管它们取得了巨大的成功，但最近的研究表明，这些先进的推荐系统非常容易受到对手攻击，攻击者可以注入精心设计的虚假用户配置文件来破坏推荐性能。虽然现有的研究主要集中在香草推荐系统上为推广目标项而进行的有针对性的攻击，但在黑盒场景下的社交推荐中，降低整体预测性能的非目标攻击的研究较少。为了对社交推荐系统进行无针对性的攻击，攻击者可以为虚假用户构建恶意的社交关系，以提高攻击性能。然而，社交关系和项目简介的协调对于攻击黑箱社交推荐是具有挑战性的。为了解决这一局限性，我们首先进行了几项初步研究，以证明跨社区联系和冷启动项目在降低推荐性能方面的有效性。具体地说，我们提出了一种基于多智能体强化学习的新型框架MultiAttack，用于协调冷启动项目配置文件的生成和跨社区社会关系的生成，以对黑盒社交推荐进行无针对性的攻击。在各种真实数据集上的综合实验证明了我们提出的攻击框架在黑盒环境下的有效性。



## **13. Towards Adversarial Robustness And Backdoor Mitigation in SSL**

SSL中的对抗稳健性和后门缓解 cs.CV

8 pages, 2 figures

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2403.15918v3) [paper-pdf](http://arxiv.org/pdf/2403.15918v3)

**Authors**: Aryan Satpathy, Nilaksh Singh, Dhruva Rajwade, Somesh Kumar

**Abstract**: Self-Supervised Learning (SSL) has shown great promise in learning representations from unlabeled data. The power of learning representations without the need for human annotations has made SSL a widely used technique in real-world problems. However, SSL methods have recently been shown to be vulnerable to backdoor attacks, where the learned model can be exploited by adversaries to manipulate the learned representations, either through tampering the training data distribution, or via modifying the model itself. This work aims to address defending against backdoor attacks in SSL, where the adversary has access to a realistic fraction of the SSL training data, and no access to the model. We use novel methods that are computationally efficient as well as generalizable across different problem settings. We also investigate the adversarial robustness of SSL models when trained with our method, and show insights into increased robustness in SSL via frequency domain augmentations. We demonstrate the effectiveness of our method on a variety of SSL benchmarks, and show that our method is able to mitigate backdoor attacks while maintaining high performance on downstream tasks. Code for our work is available at github.com/Aryan-Satpathy/Backdoor

摘要: 自监督学习(SSL)在从未标记数据中学习表示方面显示出巨大的前景。无需人工注释即可学习表示的能力使SSL成为实际问题中广泛使用的技术。然而，最近已证明SSL方法容易受到后门攻击，攻击者可以通过篡改训练数据分发或通过修改模型本身来利用学习的模型来操纵学习的表示。这项工作旨在解决针对SSL中的后门攻击的防御，在这种攻击中，攻击者可以访问真实的一小部分SSL训练数据，而不能访问模型。我们使用新的方法，这些方法在计算上是有效的，并且可以在不同的问题设置中推广。我们还研究了使用我们的方法训练的SSL模型的对抗稳健性，并展示了通过频域增强来增强SSL的稳健性的见解。我们在不同的SSL基准测试上证明了我们的方法的有效性，并表明我们的方法能够在保持下游任务的高性能的同时减少后门攻击。我们工作的代码可在githorb.com/aryan-Satthy/Backdoor上找到



## **14. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

交易魔鬼：通过随机投资模型和Bayesian方法进行强有力的后门攻击 cs.CR

(Last update!, a constructive comment from arxiv led to this latest  update ) Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market.  arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this  link to the paper by : Orson Mengara)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.10719v4) [paper-pdf](http://arxiv.org/pdf/2406.10719v4)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.

摘要: 随着语音激活系统和语音识别技术的日益广泛使用，对音频数据进行后门攻击的危险显着增加。这项研究着眼于一种特定类型的攻击，称为基于随机投资的后门攻击（MarketBack），其中对手战略性地操纵音频的风格属性来愚弄语音识别系统。机器学习模型的安全性和完整性受到后门攻击的严重威胁，为了维护音频应用和系统的可靠性，识别此类攻击在音频数据环境中变得至关重要。实验结果表明，当毒害少于1%的训练数据时，MarketBack可以在7个受害者模型中实现接近100%的平均攻击成功率。



## **15. Exact Recovery Guarantees for Parameterized Non-linear System Identification Problem under Adversarial Attacks**

对抗攻击下参数化非线性系统识别问题的精确恢复保证 math.OC

33 pages

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.00276v2) [paper-pdf](http://arxiv.org/pdf/2409.00276v2)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo D. Sontag

**Abstract**: In this work, we study the system identification problem for parameterized non-linear systems using basis functions under adversarial attacks. Motivated by the LASSO-type estimators, we analyze the exact recovery property of a non-smooth estimator, which is generated by solving an embedded $\ell_1$-loss minimization problem. First, we derive necessary and sufficient conditions for the well-specifiedness of the estimator and the uniqueness of global solutions to the underlying optimization problem. Next, we provide exact recovery guarantees for the estimator under two different scenarios of boundedness and Lipschitz continuity of the basis functions. The non-asymptotic exact recovery is guaranteed with high probability, even when there are more severely corrupted data than clean data. Finally, we numerically illustrate the validity of our theory. This is the first study on the sample complexity analysis of a non-smooth estimator for the non-linear system identification problem.

摘要: 在这项工作中，我们研究了对抗攻击下使用基函数的参数化非线性系统的系统识别问题。受LANSO型估计器的激励，我们分析了非光滑估计器的精确恢复性质，该估计器是通过解决嵌入的$\ell_1 $-损失最小化问题而生成的。首先，我们推导出估计量的良好指定性和基本优化问题的全局解的唯一性的充要条件。接下来，我们在基函数的有界性和Lipschitz连续性两种不同场景下为估计器提供精确的恢复保证。即使存在比干净数据更严重的损坏数据，也能以高概率保证非渐进精确恢复。最后，我们用数字说明了我们理论的有效性。这是首次对非线性系统识别问题的非光滑估计器的样本复杂性分析进行研究。



## **16. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

LLM Whisperer：对LLM偏见回应的不起眼攻击 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.04755v2) [paper-pdf](http://arxiv.org/pdf/2406.04755v2)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.

摘要: 为大型语言模型(LLM)编写有效的提示可能是不直观和繁琐的。作为回应，优化或建议提示的服务应运而生。虽然这类服务可以减少用户的工作，但它们也带来了风险：提示提供商可能会巧妙地操纵提示，以产生严重偏见的LLM响应。在这项工作中，我们表明，提示中微妙的同义词替换可以增加LLMS提到目标概念(例如，品牌、政党、国家)的可能性(差异高达78%)。我们通过一项用户研究证实了我们的观察结果，表明我们受到敌意干扰的提示1)与人类未更改的提示无法区分，2)推动LLMS更频繁地推荐目标概念，3)使用户更有可能注意到目标概念，所有这些都不会引起怀疑。这种攻击的实用性有可能破坏用户的自主性。在其他措施中，我们建议实施警告，以防止使用来自不受信任方的提示。



## **17. Revisiting Physical-World Adversarial Attack on Traffic Sign Recognition: A Commercial Systems Perspective**

重新审视对交通标志识别的物理世界对抗攻击：商业系统的角度 cs.CR

Accepted by NDSS 2025

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09860v1) [paper-pdf](http://arxiv.org/pdf/2409.09860v1)

**Authors**: Ningfei Wang, Shaoyuan Xie, Takami Sato, Yunpeng Luo, Kaidi Xu, Qi Alfred Chen

**Abstract**: Traffic Sign Recognition (TSR) is crucial for safe and correct driving automation. Recent works revealed a general vulnerability of TSR models to physical-world adversarial attacks, which can be low-cost, highly deployable, and capable of causing severe attack effects such as hiding a critical traffic sign or spoofing a fake one. However, so far existing works generally only considered evaluating the attack effects on academic TSR models, leaving the impacts of such attacks on real-world commercial TSR systems largely unclear. In this paper, we conduct the first large-scale measurement of physical-world adversarial attacks against commercial TSR systems. Our testing results reveal that it is possible for existing attack works from academia to have highly reliable (100\%) attack success against certain commercial TSR system functionality, but such attack capabilities are not generalizable, leading to much lower-than-expected attack success rates overall. We find that one potential major factor is a spatial memorization design that commonly exists in today's commercial TSR systems. We design new attack success metrics that can mathematically model the impacts of such design on the TSR system-level attack success, and use them to revisit existing attacks. Through these efforts, we uncover 7 novel observations, some of which directly challenge the observations or claims in prior works due to the introduction of the new metrics.

摘要: 交通标志识别(TSR)对于安全、正确的驾驶自动化至关重要。最近的工作揭示了TSR模型对物理世界对抗性攻击的普遍脆弱性，这些攻击可以是低成本的，高度可部署的，并且能够造成严重的攻击效果，例如隐藏关键交通标志或欺骗假交通标志。然而，到目前为止，现有的工作一般只考虑评估攻击对学术TSR模型的影响，而对现实世界商业TSR系统的影响很大程度上是未知的。在本文中，我们首次进行了针对商业TSR系统的物理世界对抗性攻击的大规模测量。我们的测试结果表明，学术界现有的攻击工作有可能对某些商用TSR系统功能具有高可靠性(100%)的攻击成功，但这种攻击能力不是通用的，导致总体攻击成功率远低于预期。我们发现一个潜在的主要因素是空间记忆设计，这种设计普遍存在于今天的商业TSR系统中。我们设计了新的攻击成功度量，可以对这种设计对TSR系统级攻击成功的影响进行数学建模，并使用它们来重新审视现有的攻击。通过这些努力，我们发现了7个新颖的观察结果，其中一些由于新度量的引入直接挑战了先前工作中的观察或主张。



## **18. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

对抗环境中的联邦学习：网络安全中的测试床设计和毒害韧性 cs.CR

7 pages, 4 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09794v1) [paper-pdf](http://arxiv.org/pdf/2409.09794v1)

**Authors**: Hao Jian Huang, Bekzod Iskandarov, Mizanur Rahman, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, we demonstrate the testbed's capabilities in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. Our results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.

摘要: 本文介绍了联邦学习(FL)测试平台的设计与实现，重点研究了其在网络安全中的应用，并对其抗中毒攻击的能力进行了评估。联合学习允许多个客户协作培训全球模型，同时保持他们的数据分散，满足数据隐私和安全的关键需求，特别是在网络安全等敏感领域。我们的试验台使用Flower框架构建，促进了各种FL框架的实验，评估了它们的性能、可扩展性和集成简易性。通过一个联合入侵检测系统的案例研究，我们展示了测试床在检测异常和保护关键基础设施方面的能力，而不会暴露敏感的网络数据。针对模型和数据完整性的全面中毒测试，评估系统在对抗条件下的健壮性。我们的结果表明，尽管联合学习增强了数据隐私和分布式学习，但它仍然容易受到中毒攻击，必须加以缓解，以确保其在现实世界应用中的可靠性。



## **19. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

展望未来：通过揭露对抗性合同来防止DeFi攻击 cs.CR

21 pages, 7 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2401.07261v3) [paper-pdf](http://arxiv.org/pdf/2401.07261v3)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: Decentralized Finance (DeFi) incidents stemming from the exploitation of smart contract vulnerabilities have culminated in financial damages exceeding 3 billion US dollars. Existing defense mechanisms typically focus on detecting and reacting to malicious transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively.   Based on the fact that most attack logic rely on deploying one or more intermediate smart contracts as supporting components to the exploitation of victim contracts, in this paper, we propose a new direction for detecting DeFi attacks that focuses on identifying adversarial contracts instead of adversarial transactions. Our approach allows us to leverage common attack patterns, code semantics and intrinsic characteristics found in malicious smart contracts to build the LookAhead system based on Machine Learning (ML) classifiers and a transformer model that is able to effectively distinguish adversarial contracts from benign ones, and make just-in-time predictions of potential zero-day attacks. Our contributions are three-fold: First, we construct a comprehensive dataset consisting of features extracted and constructed from recent contracts deployed on the Ethereum and BSC blockchains. Secondly, we design a condensed representation of smart contract programs called Pruned Semantic-Control Flow Tokenization (PSCFT) and use it to train a combination of ML models that understand the behaviour of malicious codes based on function calls, control flows and other pattern-conforming features. Lastly, we provide the complete implementation of LookAhead and the evaluation of its performance metrics for detecting adversarial contracts.

摘要: 因利用智能合同漏洞而引发的去中心化金融(Defi)事件已造成超过30亿美元的经济损失。现有的防御机制通常专注于检测和响应攻击者执行的针对受害者合同的恶意交易。然而，随着私人交易池的出现，交易直接发送给矿工，而不是首先出现在公共记忆池中，当前的检测工具在有效识别攻击活动方面面临重大挑战。基于大多数攻击逻辑依赖于部署一个或多个中间智能合约作为攻击受害者合约的支持组件的事实，本文提出了一种新的检测Defi攻击的方向，该方向侧重于识别对手合约而不是对手交易。我们的方法允许我们利用恶意智能合同中发现的常见攻击模式、代码语义和内在特征来构建基于机器学习(ML)分类器和转换器模型的前瞻性系统，该系统能够有效区分敌意合同和良性合同，并及时预测潜在的零日攻击。我们的贡献有三个方面：首先，我们构建了一个全面的数据集，其中包含从Etherum和BSC区块链上部署的最近合同中提取和构建的特征。其次，我们设计了智能合同程序的精简表示，称为剪枝语义控制流令牌化(PSCFT)，并使用它来训练ML模型的组合，这些模型基于函数调用、控制流和其他符合模式的特征来理解恶意代码的行为。最后，我们给出了LookHead的完整实现，并对其用于检测敌对合同的性能度量进行了评估。



## **20. Real-world Adversarial Defense against Patch Attacks based on Diffusion Model**

基于扩散模型的现实世界补丁攻击对抗防御 cs.CV

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2409.09406v1) [paper-pdf](http://arxiv.org/pdf/2409.09406v1)

**Authors**: Xingxing Wei, Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su

**Abstract**: Adversarial patches present significant challenges to the robustness of deep learning models, making the development of effective defenses become critical for real-world applications. This paper introduces DIFFender, a novel DIFfusion-based DeFender framework that leverages the power of a text-guided diffusion model to counter adversarial patch attacks. At the core of our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which enables the diffusion model to accurately detect and locate adversarial patches by analyzing distributional anomalies. DIFFender seamlessly integrates the tasks of patch localization and restoration within a unified diffusion model framework, enhancing defense efficacy through their close interaction. Additionally, DIFFender employs an efficient few-shot prompt-tuning algorithm, facilitating the adaptation of the pre-trained diffusion model to defense tasks without the need for extensive retraining. Our comprehensive evaluation, covering image classification and face recognition tasks, as well as real-world scenarios, demonstrates DIFFender's robust performance against adversarial attacks. The framework's versatility and generalizability across various settings, classifiers, and attack methodologies mark a significant advancement in adversarial patch defense strategies. Except for the popular visible domain, we have identified another advantage of DIFFender: its capability to easily expand into the infrared domain. Consequently, we demonstrate the good flexibility of DIFFender, which can defend against both infrared and visible adversarial patch attacks alternatively using a universal defense framework.

摘要: 对抗性补丁对深度学习模型的健壮性提出了重大挑战，使得有效防御的发展成为现实世界应用的关键。本文介绍了DIFFender，一个新的基于扩散的防御框架，它利用文本引导的扩散模型的能力来对抗敌意补丁攻击。该方法的核心是发现对抗性异常感知(AAP)现象，使扩散模型能够通过分析分布异常来准确地检测和定位对抗性补丁。DIFFender在一个统一的扩散模型框架内无缝集成了补丁定位和恢复任务，通过它们的密切交互提高了防御效率。此外，DIFFender采用了一种高效的少镜头即时调整算法，便于将预先训练的扩散模型适应于防御任务，而不需要进行广泛的再训练。我们的综合评估涵盖了图像分类和人脸识别任务，以及真实世界的场景，证明了DIFFender在对抗对手攻击方面的强大性能。该框架在各种设置、分类器和攻击方法上的多功能性和通用性标志着对抗性补丁防御策略的重大进步。除了流行的可见光领域外，我们还发现了DIFFender的另一个优势：它可以很容易地扩展到红外线领域。因此，我们展示了DIFFender良好的灵活性，它可以使用通用的防御框架交替防御红外和可见光对手补丁攻击。



## **21. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

针对潜行对手的遗憾最佳防御：系统级方法 eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2407.18448v2) [paper-pdf](http://arxiv.org/pdf/2407.18448v2)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems rely heavily on real-world data obtained through system outputs. However, these outputs can be compromised by system faults and malicious attacks, distorting critical system information needed for secure and reliable operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that make a linear system robust against stealthy attacks, including both sensor and actuator attacks. Specifically, we present (a) a convex optimization-based system metric to quantify the regret under the worst-case stealthy attack (the difference between actual performance and optimal performance with hindsight of the attack), which adapts and improves upon the $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of (a) in system-level parameterization, enabling localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem equivalent to the optimization of (b), which can be solved using convex rank minimization methods. We also present numerical simulations that demonstrate the effectiveness of our proposed framework.

摘要: 机器人、航空航天和计算机物理系统的现代控制设计在很大程度上依赖于通过系统输出获得的真实数据。但是，这些输出可能会受到系统故障和恶意攻击的影响，从而扭曲安全可靠运行所需的关键系统信息。在本文中，我们介绍了一种新的后悔最优控制框架，用于设计控制器，使线性系统对隐身攻击具有鲁棒性，包括传感器和执行器攻击。具体地说，我们提出了(A)基于凸优化的系统度量来量化在最坏情况下的隐蔽攻击(实际性能和最优性能之间的差值)下的遗憾，该度量适应并改进了在存在隐形对手的情况下的数学{H}_2和$\数学{H}_1，(B)用于最小化(A)在系统级参数化中的遗憾的优化问题，使得在大规模系统中实现局部和分布式实现，以及(C)等同于(B)的优化的秩约束优化问题，该问题可以使用凸秩化方法来求解。我们还给出了数值模拟，证明了我们所提出的框架的有效性。



## **22. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

迈向弹性和高效的法学硕士：效率、绩效和对抗稳健性的比较研究 cs.CL

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.04585v3) [paper-pdf](http://arxiv.org/pdf/2408.04585v3)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs and conduct extensive experiments on three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.

摘要: 随着大型语言模型的实际应用需求的增加，人们已经开发了许多注意力高效的模型来平衡性能和计算成本。然而，这些模型的对抗性稳健性仍然没有得到充分的研究。在这项工作中，我们设计了一个框架来研究LLMS的效率、性能和对抗健壮性之间的权衡，并利用GLUE和AdvGLUE数据集在三个不同复杂度和效率的重要模型上进行了广泛的实验--Transformer++、门控线性注意(GLA)Transformer和MatMul-Free LM。AdvGLUE数据集使用旨在挑战模型稳健性的对抗性样本扩展了GLUE数据集。我们的结果表明，虽然GLA Transformer和MatMul-Free LM在粘合任务上的准确率略低，但在不同攻击级别上，它们在AdvGLUE任务上表现出比Transformer++更高的效率和更好的健壮性或相对较高的稳健性。这些发现突出了简化体系结构在效率、性能和对手攻击健壮性之间实现引人注目的平衡的潜力，为资源约束和对抗攻击的弹性至关重要的应用程序提供了宝贵的见解。



## **23. Tamper-Resistant Safeguards for Open-Weight LLMs**

开放重量LLM的防篡改保障措施 cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.00761v3) [paper-pdf](http://arxiv.org/pdf/2408.00761v3)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.

摘要: 大型语言模型(LLM)功能的快速发展引起了人们对其潜在恶意使用的广泛关注。开放重量LLM提出了独特的挑战，因为现有的保障措施缺乏对篡改模型权重的篡改攻击的稳健性。例如，最近的研究表明，通过几个步骤的微调，就可以很容易地消除拒绝和遗忘的保障措施。这些漏洞需要新的方法来实现安全释放未加重量的低密度脂蛋白。我们开发了一种名为TAR的方法，用于在开放重量的LLM中构建防篡改保护措施，以便对手即使在数千个步骤的微调之后也无法移除这些保护措施。在广泛的评估和红团队分析中，我们发现我们的方法在保持良性性能的同时大大提高了防篡改能力。我们的结果表明，防篡改是一个容易解决的问题，为提高开重LLMS的安全性开辟了一条很有前途的新途径。



## **24. Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization**

通过异常对抗示例规范化消除灾难性过度匹配 cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2404.08154v2) [paper-pdf](http://arxiv.org/pdf/2404.08154v2)

**Authors**: Runqi Lin, Chaojian Yu, Tongliang Liu

**Abstract**: Single-step adversarial training (SSAT) has demonstrated the potential to achieve both efficiency and robustness. However, SSAT suffers from catastrophic overfitting (CO), a phenomenon that leads to a severely distorted classifier, making it vulnerable to multi-step adversarial attacks. In this work, we observe that some adversarial examples generated on the SSAT-trained network exhibit anomalous behaviour, that is, although these training samples are generated by the inner maximization process, their associated loss decreases instead, which we named abnormal adversarial examples (AAEs). Upon further analysis, we discover a close relationship between AAEs and classifier distortion, as both the number and outputs of AAEs undergo a significant variation with the onset of CO. Given this observation, we re-examine the SSAT process and uncover that before the occurrence of CO, the classifier already displayed a slight distortion, indicated by the presence of few AAEs. Furthermore, the classifier directly optimizing these AAEs will accelerate its distortion, and correspondingly, the variation of AAEs will sharply increase as a result. In such a vicious circle, the classifier rapidly becomes highly distorted and manifests as CO within a few iterations. These observations motivate us to eliminate CO by hindering the generation of AAEs. Specifically, we design a novel method, termed Abnormal Adversarial Examples Regularization (AAER), which explicitly regularizes the variation of AAEs to hinder the classifier from becoming distorted. Extensive experiments demonstrate that our method can effectively eliminate CO and further boost adversarial robustness with negligible additional computational overhead.

摘要: 单步对抗训练(SSAT)已经证明了实现效率和稳健性的潜力。然而，SSAT存在灾难性过匹配(CO)，这一现象导致分类器严重失真，使其容易受到多步骤对抗性攻击。在这项工作中，我们观察到在SSAT训练的网络上产生的一些对抗性样本表现出异常行为，即这些训练样本虽然是由内部最大化过程产生的，但其关联损失反而减少，我们称之为异常对抗性样本(AAES)。通过进一步的分析，我们发现AAEs与分类器失真之间有密切的关系，因为AAEs的数量和输出都随着CO的开始而发生显著的变化。鉴于这一观察，我们重新检查SSAT过程并发现，在CO发生之前，分类器已经显示出轻微的失真，这表明存在很少的AAE。而且，直接对这些AAEs进行优化的分类器会加速AAEs的失真，相应地，AAEs的变化量也会急剧增加。在这样的恶性循环中，分类器迅速变得高度失真，并在几次迭代内表现为CO。这些观察结果促使我们通过阻碍AAEs的产生来消除CO。具体地说，我们设计了一种新的方法，称为异常对抗实例正则化(AAER)，它显式地规则化AAE的变化，以防止分类器变得失真。大量的实验表明，该方法可以有效地消除CO，并在几乎不增加计算开销的情况下进一步提高对手攻击的健壮性。



## **25. Layer-Aware Analysis of Catastrophic Overfitting: Revealing the Pseudo-Robust Shortcut Dependency**

灾难性过度匹配的分层感知分析：揭示伪稳健的预设依赖 cs.LG

Accepted by ICML 2024

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2405.16262v2) [paper-pdf](http://arxiv.org/pdf/2405.16262v2)

**Authors**: Runqi Lin, Chaojian Yu, Bo Han, Hang Su, Tongliang Liu

**Abstract**: Catastrophic overfitting (CO) presents a significant challenge in single-step adversarial training (AT), manifesting as highly distorted deep neural networks (DNNs) that are vulnerable to multi-step adversarial attacks. However, the underlying factors that lead to the distortion of decision boundaries remain unclear. In this work, we delve into the specific changes within different DNN layers and discover that during CO, the former layers are more susceptible, experiencing earlier and greater distortion, while the latter layers show relative insensitivity. Our analysis further reveals that this increased sensitivity in former layers stems from the formation of pseudo-robust shortcuts, which alone can impeccably defend against single-step adversarial attacks but bypass genuine-robust learning, resulting in distorted decision boundaries. Eliminating these shortcuts can partially restore robustness in DNNs from the CO state, thereby verifying that dependence on them triggers the occurrence of CO. This understanding motivates us to implement adaptive weight perturbations across different layers to hinder the generation of pseudo-robust shortcuts, consequently mitigating CO. Extensive experiments demonstrate that our proposed method, Layer-Aware Adversarial Weight Perturbation (LAP), can effectively prevent CO and further enhance robustness.

摘要: 灾难性过拟合(CO)是单步对抗训练(AT)中的一个重大挑战，表现为高度扭曲的深度神经网络(DNN)，容易受到多步对抗攻击。然而，导致决策边界扭曲的潜在因素仍然不清楚。在这项工作中，我们深入研究了不同DNN层内的具体变化，发现在CO过程中，前一层更容易受到影响，经历更早和更大的失真，而后一层表现出相对不敏感。我们的分析进一步表明，前几层敏感度的增加源于伪稳健捷径的形成，这些捷径可以无懈可击地防御单步对手攻击，但绕过了真正的稳健学习，导致决策边界扭曲。消除这些捷径可以从CO状态部分恢复DNN的稳健性，从而验证对它们的依赖是否触发了CO的发生。这种理解促使我们在不同的层上实现自适应的权重扰动，以阻止伪稳健捷径的生成，从而减少CO。大量实验表明，本文提出的层感知对抗性权重扰动(LAP)方法能够有效地防止CO，并进一步增强了鲁棒性。



## **26. Cybersecurity Software Tool Evaluation Using a 'Perfect' Network Model**

使用“完美”网络模型的网络安全软件工具评估 cs.CR

The U.S. federal sponsor has requested that we not include funding  acknowledgement for this publication

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.09175v1) [paper-pdf](http://arxiv.org/pdf/2409.09175v1)

**Authors**: Jeremy Straub

**Abstract**: Cybersecurity software tool evaluation is difficult due to the inherently adversarial nature of the field. A penetration testing (or offensive) tool must be tested against a viable defensive adversary and a defensive tool must, similarly, be tested against a viable offensive adversary. Characterizing the tool's performance inherently depends on the quality of the adversary, which can vary from test to test. This paper proposes the use of a 'perfect' network, representing computing systems, a network and the attack pathways through it as a methodology to use for testing cybersecurity decision-making tools. This facilitates testing by providing a known and consistent standard for comparison. It also allows testing to include researcher-selected levels of error, noise and uncertainty to evaluate cybersecurity tools under these experimental conditions.

摘要: 由于该领域固有的对抗性，网络安全软件工具评估很困难。渗透测试（或进攻性）工具必须针对可行的防御对手进行测试，同样，防御工具也必须针对可行的进攻性对手进行测试。描述工具的性能本质上取决于对手的质量，而对手的质量可能因测试而异。本文建议使用“完美”网络，代表计算系统、网络和通过它的攻击路径，作为用于测试网络安全决策工具的方法论。这通过提供已知且一致的比较标准来促进测试。它还允许测试包括研究人员选择的错误、噪音和不确定性水平，以在这些实验条件下评估网络安全工具。



## **27. Clean Label Attacks against SLU Systems**

针对SL U系统的干净标签攻击 cs.CR

Accepted at IEEE SLT 2024

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08985v1) [paper-pdf](http://arxiv.org/pdf/2409.08985v1)

**Authors**: Henry Li Xinyuan, Sonal Joshi, Thomas Thebaud, Jesus Villalba, Najim Dehak, Sanjeev Khudanpur

**Abstract**: Poisoning backdoor attacks involve an adversary manipulating the training data to induce certain behaviors in the victim model by inserting a trigger in the signal at inference time. We adapted clean label backdoor (CLBD)-data poisoning attacks, which do not modify the training labels, on state-of-the-art speech recognition models that support/perform a Spoken Language Understanding task, achieving 99.8% attack success rate by poisoning 10% of the training data. We analyzed how varying the signal-strength of the poison, percent of samples poisoned, and choice of trigger impact the attack. We also found that CLBD attacks are most successful when applied to training samples that are inherently hard for a proxy model. Using this strategy, we achieved an attack success rate of 99.3% by poisoning a meager 1.5% of the training data. Finally, we applied two previously developed defenses against gradient-based attacks, and found that they attain mixed success against poisoning.

摘要: 中毒后门攻击涉及对手操纵训练数据，通过在推理时在信号中插入触发器来诱导受害者模型中的某些行为。我们在支持/执行口语理解任务的最先进语音识别模型上采用了干净标签后门（CLBD）-数据中毒攻击，其不会修改训练标签，通过毒害10%的训练数据，实现了99.8%的攻击成功率。我们分析了毒物的信号强度、中毒样本的百分比以及触发器的选择如何影响攻击。我们还发现，CLBD攻击在应用于对于代理模型来说本质上很难的训练样本时最为成功。使用该策略，我们通过毒害可怜的1.5%的训练数据，实现了99.3%的攻击成功率。最后，我们应用了两种之前开发的针对基于梯度的攻击的防御措施，并发现它们在对抗中毒方面取得了好坏参半的成功。



## **28. XSub: Explanation-Driven Adversarial Attack against Blackbox Classifiers via Feature Substitution**

XSub：通过特征替代对黑匣子分类器的描述驱动的对抗攻击 cs.LG

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08919v1) [paper-pdf](http://arxiv.org/pdf/2409.08919v1)

**Authors**: Kiana Vu, Phung Lai, Truc Nguyen

**Abstract**: Despite its significant benefits in enhancing the transparency and trustworthiness of artificial intelligence (AI) systems, explainable AI (XAI) has yet to reach its full potential in real-world applications. One key challenge is that XAI can unintentionally provide adversaries with insights into black-box models, inevitably increasing their vulnerability to various attacks. In this paper, we develop a novel explanation-driven adversarial attack against black-box classifiers based on feature substitution, called XSub. The key idea of XSub is to strategically replace important features (identified via XAI) in the original sample with corresponding important features from a "golden sample" of a different label, thereby increasing the likelihood of the model misclassifying the perturbed sample. The degree of feature substitution is adjustable, allowing us to control how much of the original samples information is replaced. This flexibility effectively balances a trade-off between the attacks effectiveness and its stealthiness. XSub is also highly cost-effective in that the number of required queries to the prediction model and the explanation model in conducting the attack is in O(1). In addition, XSub can be easily extended to launch backdoor attacks in case the attacker has access to the models training data. Our evaluation demonstrates that XSub is not only effective and stealthy but also cost-effective, enabling its application across a wide range of AI models.

摘要: 尽管可解释人工智能(XAI)在提高人工智能(AI)系统的透明度和可信性方面具有显著优势，但它在现实世界的应用中尚未充分发挥其潜力。一个关键的挑战是，XAI可能会无意中向对手提供对黑盒模型的洞察，从而不可避免地增加他们对各种攻击的脆弱性。本文提出了一种新的基于特征替换的解释驱动的对抗性黑盒分类器攻击方法XSub。XSub的关键思想是策略性地将原始样本中的重要特征(通过XAI识别)替换为来自不同标签的“黄金样本”的相应重要特征，从而增加模型错误分类扰动样本的可能性。特征替换的程度是可调的，允许我们控制原始样本信息的替换量。这种灵活性有效地平衡了攻击的有效性和隐蔽性之间的权衡。XSub还具有很高的性价比，因为在进行攻击时，对预测模型和解释模型所需的查询数量为O(1)。此外，XSub可以很容易地扩展为在攻击者有权访问模型训练数据的情况下发动后门攻击。我们的评估表明，XSub不仅有效和隐身，而且性价比高，使其能够在广泛的人工智能模型中应用。



## **29. Are Existing Road Design Guidelines Suitable for Autonomous Vehicles?**

现有的道路设计指南适合自动驾驶车辆吗？ cs.CV

Currently under review by IEEE Transactions on Software Engineering  (TSE)

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.10562v1) [paper-pdf](http://arxiv.org/pdf/2409.10562v1)

**Authors**: Yang Sun, Christopher M. Poskitt, Jun Sun

**Abstract**: The emergence of Autonomous Vehicles (AVs) has spurred research into testing the resilience of their perception systems, i.e. to ensure they are not susceptible to making critical misjudgements. It is important that they are tested not only with respect to other vehicles on the road, but also those objects placed on the roadside. Trash bins, billboards, and greenery are all examples of such objects, typically placed according to guidelines that were developed for the human visual system, and which may not align perfectly with the needs of AVs. Existing tests, however, usually focus on adversarial objects with conspicuous shapes/patches, that are ultimately unrealistic given their unnatural appearances and the need for white box knowledge. In this work, we introduce a black box attack on the perception systems of AVs, in which the objective is to create realistic adversarial scenarios (i.e. satisfying road design guidelines) by manipulating the positions of common roadside objects, and without resorting to `unnatural' adversarial patches. In particular, we propose TrashFuzz , a fuzzing algorithm to find scenarios in which the placement of these objects leads to substantial misperceptions by the AV -- such as mistaking a traffic light's colour -- with overall the goal of causing it to violate traffic laws. To ensure the realism of these scenarios, they must satisfy several rules encoding regulatory guidelines about the placement of objects on public streets. We implemented and evaluated these attacks for the Apollo, finding that TrashFuzz induced it into violating 15 out of 24 different traffic laws.

摘要: 自动驾驶汽车(AVs)的出现促使人们研究测试其感知系统的弹性，即确保它们不容易做出关键的误判。重要的是，不仅要对道路上的其他车辆进行测试，还要对放置在路边的那些物体进行测试。垃圾桶、广告牌和绿色植物都是这种物体的例子，通常是根据为人类视觉系统开发的指导方针放置的，可能不能完全符合AVs的需求。然而，现有的测试通常集中在具有明显形状/补丁的对抗性对象上，考虑到它们不自然的外观和对白盒知识的需要，这些最终是不现实的。在这项工作中，我们引入了一种针对自动驾驶系统感知系统的黑盒攻击，其目的是通过操纵常见路旁对象的位置来创建现实的对抗性场景(即满足道路设计准则)，而不求助于不自然的对抗性补丁。特别是，我们提出了TrashFuzz，这是一种模糊算法，用于查找这些对象的放置导致AV产生重大误解的场景--例如错误地识别红绿灯的颜色--总体目标是导致它违反交通法规。为了确保这些场景的真实性，它们必须满足几项规则，这些规则编码了关于在公共街道上放置物体的监管指南。我们为阿波罗实施并评估了这些攻击，发现TrashFuzz导致它违反了24项不同交通法规中的15项。



## **30. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

仔细研究GAN先验：利用中间功能进行增强模型倒置攻击 cs.CV

ECCV 2024

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2407.13863v4) [paper-pdf](http://arxiv.org/pdf/2407.13863v4)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI

摘要: 模型反转(MI)攻击的目的是利用输出信息从已发布的模型中重建隐私敏感的训练数据，这引起了人们对深度神经网络(DNN)安全性的广泛关注。生成性对抗网络(GANS)的最新进展为MI攻击的性能改进做出了重要贡献，因为它们能够生成高保真和适当语义的真实图像。然而，以往的MI攻击只在GaN先验的潜在空间中泄露隐私信息，限制了它们的语义提取和跨多个目标模型和数据集的可传输性。为了解决这一挑战，我们提出了一种新的方法，中间特征增强的生成性模型反转(IF-GMI)，它分解GaN结构并利用中间块之间的特征。这允许我们将优化空间从潜在代码扩展到具有增强表达能力的中间功能。为了防止GaN先验数据产生不真实的图像，我们在优化过程中应用了L1球约束。在多个基准测试上的实验表明，我们的方法显著优于以前的方法，并在各种设置下获得了最先进的结果，特别是在分布外(OOD)的情况下。我们的代码请访问：https://github.com/final-solution/IF-GMI



## **31. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

保护人工智能代理：开发和分析安全架构 cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.03793v2) [paper-pdf](http://arxiv.org/pdf/2409.03793v2)

**Authors**: Ishaan Domkundwar, Mukunda N S, Ishaan Bhola

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.

摘要: 人工智能代理，特别是由大型语言模型驱动的，在需要精确度和效率的各种应用中展示了非凡的能力。然而，这些代理伴随着固有的风险，包括潜在的不安全或有偏见的行动，易受对手攻击，缺乏透明度，以及产生幻觉的倾向。随着人工智能代理在该行业的关键部门变得越来越普遍，实施有效的安全协议变得越来越重要。本文讨论了人工智能系统中安全措施的迫切需要，特别是与人类团队协作的系统。我们提出并评估了三个框架来增强AI代理系统中的安全协议：LLM驱动的输入输出过滤器、集成在系统中的安全代理以及嵌入安全检查的基于分级委托的系统。我们的方法涉及实现这些框架并针对一组不安全的代理用例对它们进行测试，提供对它们在降低与AI代理部署相关的风险方面的有效性的全面评估。我们的结论是，这些框架可以显著加强AI代理系统的安全性和安全性，将潜在的有害行为或输出降至最低。我们的工作有助于持续努力创建安全可靠的人工智能应用程序，特别是在自动化操作中，并为开发强大的护栏提供基础，以确保在现实世界的应用程序中负责任地使用人工智能代理。



## **32. Towards Efficient Transferable Preemptive Adversarial Defense**

迈向高效的可转让先发制人的对抗防御 cs.CR

Under Review

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2407.15524v2) [paper-pdf](http://arxiv.org/pdf/2407.15524v2)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, we have devised a proactive strategy for "attacking" the medias before it is attacked by the third party, so that when the protected medias are further attacked, the adversarial perturbations are automatically neutralized. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks. The proposed methodology will be made available on GitHub.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对不起眼的扰动(即对抗性攻击)的敏感性而变得不可信任。攻击者可能会利用这种敏感性来操纵预测。为了防御此类攻击，我们制定了一种主动策略，在媒体受到第三方攻击之前对其进行“攻击”，以便当受保护的媒体进一步受到攻击时，对手的干扰会自动被中和。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。我们还设计了我们所知的第一个有效的白盒自适应恢复攻击，并证明了我们的防御策略添加的保护是不可逆转的，除非主干模型、算法和设置完全受损。这项工作为主动防御对抗性攻击提供了新的方向。拟议的方法将在GitHub上提供。



## **33. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

h4 rm3l：LLM安全评估的可组合越狱攻击的动态基准 cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2408.04811v2) [paper-pdf](http://arxiv.org/pdf/2408.04811v2)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.

摘要: 大型语言模型(LLM)的安全性仍然是一个严重的问题，因为缺乏系统地评估它们抵抗产生有害内容的能力的适当基准。以前的自动化红色团队的努力包括静态的或模板化的非法请求集和对抗性提示，鉴于越狱攻击不断演变和可组合的性质，这些提示的效用有限。我们提出了一种新的可组合越狱攻击的动态基准，以超越静态数据集和攻击和危害的分类。我们的方法由三个组件组成，统称为h4rm3l：(1)特定于领域的语言，它将越狱攻击形式化地表达为参数化提示转换原语的组合；(2)基于盗贼的少发程序合成算法，它生成经过优化的新型攻击，以穿透目标黑盒LLM的安全过滤器；以及(3)使用前两个组件的开源自动红队软件。我们使用h4rm3l生成了一个2656个成功的新型越狱攻击的数据集，目标是6个最先进的开源和专有LLM。我们的几个合成攻击比以前报道的更有效，在Claude-3-haiku和GPT4-o等Sota封闭语言模型上的攻击成功率超过90%。通过以统一的形式表示生成越狱攻击的数据集，h4rm3l实现了可重现的基准测试和自动化的红团队，有助于了解LLM的安全限制，并支持在日益集成LLM的世界中开发强大的防御措施。警告：本文和相关研究文章包含冒犯性和潜在令人不安的提示和模型生成的内容。



## **34. Adversarial Attacks and Defenses on Text-to-Image Diffusion Models: A Survey**

文本到图像扩散模型的对抗性攻击和防御：综述 cs.CR

Accepted for Information Fusion. Related benchmarks and codes are  available at \url{https://github.com/datar001/Awesome-AD-on-T2IDM}

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2407.15861v2) [paper-pdf](http://arxiv.org/pdf/2407.15861v2)

**Authors**: Chenyu Zhang, Mingwang Hu, Wenhui Li, Lanjun Wang

**Abstract**: Recently, the text-to-image diffusion model has gained considerable attention from the community due to its exceptional image generation capability. A representative model, Stable Diffusion, amassed more than 10 million users within just two months of its release. This surge in popularity has facilitated studies on the robustness and safety of the model, leading to the proposal of various adversarial attack methods. Simultaneously, there has been a marked increase in research focused on defense methods to improve the robustness and safety of these models. In this survey, we provide a comprehensive review of the literature on adversarial attacks and defenses targeting text-to-image diffusion models. We begin with an overview of text-to-image diffusion models, followed by an introduction to a taxonomy of adversarial attacks and an in-depth review of existing attack methods. We then present a detailed analysis of current defense methods that improve model robustness and safety. Finally, we discuss ongoing challenges and explore promising future research directions. For a complete list of the adversarial attack and defense methods covered in this survey, please refer to our curated repository at https://github.com/datar001/Awesome-AD-on-T2IDM.

摘要: 近年来，文本到图像扩散模型以其卓越的图像生成能力受到了社会各界的广泛关注。一个有代表性的模式--稳定扩散--在发布后的短短两个月内就积累了超过1000万用户。这种受欢迎程度的激增促进了对该模型的健壮性和安全性的研究，导致了各种对抗性攻击方法的提出。与此同时，集中在防御方法以提高这些模型的稳健性和安全性的研究也有了显著的增加。在这项调查中，我们提供了一个全面的文献回顾的对抗性攻击和防御目标的文本到图像扩散模型。我们首先概述文本到图像的扩散模型，然后介绍对抗性攻击的分类，并深入回顾现有的攻击方法。然后，我们详细分析了当前提高模型健壮性和安全性的防御方法。最后，我们讨论了正在进行的挑战，并探索了未来的研究方向。有关本调查中涵盖的对抗性攻击和防御方法的完整列表，请参阅我们的精选知识库，网址为https://github.com/datar001/Awesome-AD-on-T2IDM.



## **35. Exploiting Supervised Poison Vulnerability to Strengthen Self-Supervised Defense**

利用监督毒物漏洞加强自我监督防御 cs.CV

28 pages, 5 figures

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08509v1) [paper-pdf](http://arxiv.org/pdf/2409.08509v1)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Yi Huang, Adams Kong

**Abstract**: Availability poisons exploit supervised learning (SL) algorithms by introducing class-related shortcut features in images such that models trained on poisoned data are useless for real-world datasets. Self-supervised learning (SSL), which utilizes augmentations to learn instance discrimination, is regarded as a strong defense against poisoned data. However, by extending the study of SSL across multiple poisons on the CIFAR-10 and ImageNet-100 datasets, we demonstrate that it often performs poorly, far below that of training on clean data. Leveraging the vulnerability of SL to poison attacks, we introduce adversarial training (AT) on SL to obfuscate poison features and guide robust feature learning for SSL. Our proposed defense, designated VESPR (Vulnerability Exploitation of Supervised Poisoning for Robust SSL), surpasses the performance of six previous defenses across seven popular availability poisons. VESPR displays superior performance over all previous defenses, boosting the minimum and average ImageNet-100 test accuracies of poisoned models by 16% and 9%, respectively. Through analysis and ablation studies, we elucidate the mechanisms by which VESPR learns robust class features.

摘要: 可用性毒药通过在图像中引入与类相关的快捷特征来利用监督学习(SL)算法，使得在有毒数据上训练的模型对真实世界的数据集毫无用处。自监督学习(SSL)利用扩充来学习实例区分，被认为是对抗有毒数据的强大防御。然而，通过在CIFAR-10和ImageNet-100数据集上扩展对SSL的研究，我们证明它的表现经常很差，远远低于对干净数据的培训。利用SL对毒物攻击的脆弱性，我们在SL上引入对抗性训练(AT)来混淆毒物特征，并指导针对SSL的稳健特征学习。我们建议的防御措施，命名为VeSpR(漏洞利用监管毒药的稳健SSL)，超过了七种流行的可用性毒药的六个以前的防御措施的性能。VeSpR表现出比所有以前的防御系统更优越的性能，将中毒模型的最低和平均ImageNet-100测试精度分别提高了16%和9%。通过分析和消融研究，我们阐明了VeSpR学习健壮类别特征的机制。



## **36. Sub-graph Based Diffusion Model for Link Prediction**

基于子图的链路预测扩散模型 cs.LG

17 pages, 3 figures

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.08487v1) [paper-pdf](http://arxiv.org/pdf/2409.08487v1)

**Authors**: Hang Li, Wei Jin, Geri Skenderi, Harry Shomer, Wenzhuo Tang, Wenqi Fan, Jiliang Tang

**Abstract**: Denoising Diffusion Probabilistic Models (DDPMs) represent a contemporary class of generative models with exceptional qualities in both synthesis and maximizing the data likelihood. These models work by traversing a forward Markov Chain where data is perturbed, followed by a reverse process where a neural network learns to undo the perturbations and recover the original data. There have been increasing efforts exploring the applications of DDPMs in the graph domain. However, most of them have focused on the generative perspective. In this paper, we aim to build a novel generative model for link prediction. In particular, we treat link prediction between a pair of nodes as a conditional likelihood estimation of its enclosing sub-graph. With a dedicated design to decompose the likelihood estimation process via the Bayesian formula, we are able to separate the estimation of sub-graph structure and its node features. Such designs allow our model to simultaneously enjoy the advantages of inductive learning and the strong generalization capability. Remarkably, comprehensive experiments across various datasets validate that our proposed method presents numerous advantages: (1) transferability across datasets without retraining, (2) promising generalization on limited training data, and (3) robustness against graph adversarial attacks.

摘要: 去噪扩散概率模型(DDPM)代表了当代一类生成性模型，它在综合和最大化数据似然方面都具有优异的品质。这些模型的工作原理是遍历一个正向马尔可夫链，其中数据受到扰动，然后是一个反向过程，神经网络学习取消扰动并恢复原始数据。人们越来越努力地探索DDPM在图论领域中的应用。然而，他们中的大多数都关注于生成视角。在本文中，我们的目标是建立一种新的链接预测产生式模型。特别地，我们将节点对之间的链接预测视为对其包围子图的条件似然估计。通过专门设计通过贝叶斯公式分解似然估计过程，我们能够将子图结构的估计与其节点特征分离。这样的设计使得我们的模型同时具有归纳学习的优点和较强的泛化能力。值得注意的是，在不同数据集上的综合实验验证了我们所提出的方法具有许多优点：(1)无需重新训练即可跨数据集转移；(2)在有限的训练数据上具有良好的泛化能力；(3)对图攻击具有较强的鲁棒性。



## **37. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

评估大型语言模型的对抗稳健性：实证研究 cs.CL

Oral presentation at KDD 2024 GenAI Evaluation workshop

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2405.02764v2) [paper-pdf](http://arxiv.org/pdf/2405.02764v2)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但其对抗攻击的稳健性仍然是一个关键问题。我们提出了一种新颖的白盒式攻击方法，该方法暴露了领先开源LLM（包括Llama、OPT和T5）中的漏洞。我们评估了模型大小、结构和微调策略对其抵抗对抗性扰动的影响。我们对五种不同文本分类任务的全面评估为LLM稳健性建立了新基准。这项研究的结果对于LLM在现实世界应用程序中的可靠部署具有深远的影响，并有助于发展值得信赖的人工智能系统。



## **38. Safety of Linear Systems under Severe Sensor Attacks**

严重传感器攻击下线性系统的安全性 eess.SY

To appear at CDC 2024

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08413v1) [paper-pdf](http://arxiv.org/pdf/2409.08413v1)

**Authors**: Xiao Tan, Pio Ong, Paulo Tabuada, Aaron D. Ames

**Abstract**: Cyber-physical systems can be subject to sensor attacks, e.g., sensor spoofing, leading to unsafe behaviors. This paper addresses this problem in the context of linear systems when an omniscient attacker can spoof several system sensors at will. In this adversarial environment, existing results have derived necessary and sufficient conditions under which the state estimation problem has a unique solution. In this work, we consider a severe attacking scenario when such conditions do not hold. To deal with potential state estimation uncertainty, we derive an exact characterization of the set of all possible state estimates. Using the framework of control barrier functions, we propose design principles for system safety in offline and online phases. For the offline phase, we derive conditions on safe sets for all possible sensor attacks that may be encountered during system deployment. For the online phase, with past system measurements collected, a quadratic program-based safety filter is proposed to enforce system safety. A 2D-vehicle example is used to illustrate the theoretical results.

摘要: 网络物理系统可能会受到传感器攻击，例如传感器欺骗，从而导致不安全行为。本文在线性系统的背景下讨论了当无所不知的攻击者可以任意欺骗多个系统传感器时的问题。在这种对抗性的环境下，已有的结果得到了状态估计问题有唯一解的充要条件。在这项工作中，我们考虑了这样的条件不成立时的严重攻击场景。为了处理潜在的状态估计不确定性，我们给出了所有可能的状态估计集合的精确刻画。利用控制屏障功能框架，提出了离线和在线两个阶段的系统安全设计原则。对于离线阶段，我们推导出系统部署期间可能遇到的所有可能的传感器攻击的安全集上的条件。对于在线阶段，通过收集系统过去的测量数据，提出了一种基于二次规划的安全过滤器来加强系统的安全性。文中以一辆二维车为例对理论结果进行了验证。



## **39. LoRID: Low-Rank Iterative Diffusion for Adversarial Purification**

LoDID：对抗净化的低等级迭代扩散 cs.LG

LA-UR-24-28834

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08255v1) [paper-pdf](http://arxiv.org/pdf/2409.08255v1)

**Authors**: Geigh Zollicoffer, Minh Vu, Ben Nebgen, Juan Castorena, Boian Alexandrov, Manish Bhattarai

**Abstract**: This work presents an information-theoretic examination of diffusion-based purification methods, the state-of-the-art adversarial defenses that utilize diffusion models to remove malicious perturbations in adversarial examples. By theoretically characterizing the inherent purification errors associated with the Markov-based diffusion purifications, we introduce LoRID, a novel Low-Rank Iterative Diffusion purification method designed to remove adversarial perturbation with low intrinsic purification errors. LoRID centers around a multi-stage purification process that leverages multiple rounds of diffusion-denoising loops at the early time-steps of the diffusion models, and the integration of Tucker decomposition, an extension of matrix factorization, to remove adversarial noise at high-noise regimes. Consequently, LoRID increases the effective diffusion time-steps and overcomes strong adversarial attacks, achieving superior robustness performance in CIFAR-10/100, CelebA-HQ, and ImageNet datasets under both white-box and black-box settings.

摘要: 这项工作提出了一种基于扩散的净化方法的信息论检验，这种方法是一种最先进的对抗防御方法，它利用扩散模型来消除对抗例子中的恶意扰动。通过对基于马尔可夫扩散净化的固有净化误差进行理论分析，提出了一种新的低阶迭代扩散净化方法LoRID，旨在以较低的固有净化误差去除对抗性扰动。LoRID以多阶段净化过程为中心，在扩散模型的早期步骤利用多轮扩散去噪循环，并整合Tucker分解(矩阵因式分解的扩展)，以在高噪声区域消除对抗性噪声。因此，LoRID增加了有效的扩散时间步长，并克服了强大的对手攻击，在CIFAR-10/100、CelebA-HQ和ImageNet数据集上实现了在白盒和黑盒设置下的卓越鲁棒性性能。



## **40. High-Frequency Anti-DreamBooth: Robust Defense Against Image Synthesis**

高频反DreamBooth：针对图像合成的强大防御 cs.CV

ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08167v1) [paper-pdf](http://arxiv.org/pdf/2409.08167v1)

**Authors**: Takuto Onikubo, Yusuke Matsui

**Abstract**: Recently, text-to-image generative models have been misused to create unauthorized malicious images of individuals, posing a growing social problem. Previous solutions, such as Anti-DreamBooth, add adversarial noise to images to protect them from being used as training data for malicious generation. However, we found that the adversarial noise can be removed by adversarial purification methods such as DiffPure. Therefore, we propose a new adversarial attack method that adds strong perturbation on the high-frequency areas of images to make it more robust to adversarial purification. Our experiment showed that the adversarial images retained noise even after adversarial purification, hindering malicious image generation.

摘要: 最近，文本到图像的生成模型被滥用来创建未经授权的恶意个人图像，造成了日益严重的社会问题。以前的解决方案（例如Anti-DreamBooth）会向图像添加对抗性噪音，以保护它们不被用作恶意生成的训练数据。然而，我们发现对抗性噪音可以通过迪夫Pure等对抗性净化方法去除。因此，我们提出了一种新的对抗性攻击方法，该方法在图像的高频区域添加强扰动，使其对对抗性净化更稳健。我们的实验表明，即使在对抗净化之后，对抗图像也会保留噪音，从而阻碍了恶意图像的生成。



## **41. Unleashing Worms and Extracting Data: Escalating the Outcome of Attacks against RAG-based Inference in Scale and Severity Using Jailbreaking**

释放蠕虫和提取数据：使用越狱从规模和严重性上升级针对基于RAG的推理的攻击结果 cs.CR

for Github, see  https://github.com/StavC/UnleashingWorms-ExtractingData

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08045v1) [paper-pdf](http://arxiv.org/pdf/2409.08045v1)

**Authors**: Stav Cohen, Ron Bitton, Ben Nassi

**Abstract**: In this paper, we show that with the ability to jailbreak a GenAI model, attackers can escalate the outcome of attacks against RAG-based GenAI-powered applications in severity and scale. In the first part of the paper, we show that attackers can escalate RAG membership inference attacks and RAG entity extraction attacks to RAG documents extraction attacks, forcing a more severe outcome compared to existing attacks. We evaluate the results obtained from three extraction methods, the influence of the type and the size of five embeddings algorithms employed, the size of the provided context, and the GenAI engine. We show that attackers can extract 80%-99.8% of the data stored in the database used by the RAG of a Q&A chatbot. In the second part of the paper, we show that attackers can escalate the scale of RAG data poisoning attacks from compromising a single GenAI-powered application to compromising the entire GenAI ecosystem, forcing a greater scale of damage. This is done by crafting an adversarial self-replicating prompt that triggers a chain reaction of a computer worm within the ecosystem and forces each affected application to perform a malicious activity and compromise the RAG of additional applications. We evaluate the performance of the worm in creating a chain of confidential data extraction about users within a GenAI ecosystem of GenAI-powered email assistants and analyze how the performance of the worm is affected by the size of the context, the adversarial self-replicating prompt used, the type and size of the embeddings algorithm employed, and the number of hops in the propagation. Finally, we review and analyze guardrails to protect RAG-based inference and discuss the tradeoffs.

摘要: 在本文中，我们展示了通过越狱GenAI模型的能力，攻击者可以在严重性和规模上升级对基于RAG的GenAI支持的应用程序的攻击结果。在论文的第一部分，我们证明了攻击者可以将RAG成员关系推理攻击和RAG实体提取攻击升级为RAG文档提取攻击，从而迫使出现比现有攻击更严重的后果。我们评估了三种提取方法获得的结果，所采用的五种嵌入算法的类型和大小、所提供的上下文的大小以及GenAI引擎的影响。我们表明，攻击者可以提取80%-99.8%的数据存储在数据库中的问答聊天机器人使用的RAG。在本文的第二部分中，我们展示了攻击者可以将RAG数据中毒攻击的规模从危害单个GenAI支持的应用程序升级到危害整个GenAI生态系统，从而迫使更大规模的破坏。这是通过精心编制一个敌意的自我复制提示来实现的，该提示会在生态系统中触发计算机蠕虫的连锁反应，并迫使每个受影响的应用程序执行恶意活动，并危及其他应用程序的安全。我们评估了蠕虫在由GenAI支持的电子邮件助手组成的GenAI生态系统中创建关于用户的机密数据提取链的性能，并分析了上下文大小、使用的敌意自复制提示、所使用的嵌入算法的类型和大小以及传播中的跳数对蠕虫性能的影响。最后，我们回顾和分析了保护基于RAG的推理的护栏，并讨论了其权衡。



## **42. Detecting and Defending Against Adversarial Attacks on Automatic Speech Recognition via Diffusion Models**

利用扩散模型检测和防御自动语音识别中的对抗攻击 eess.AS

Under review at ICASSP 2025

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07936v1) [paper-pdf](http://arxiv.org/pdf/2409.07936v1)

**Authors**: Nikolai L. Kühne, Astrid H. F. Kitchen, Marie S. Jensen, Mikkel S. L. Brøndt, Martin Gonzalez, Christophe Biscio, Zheng-Hua Tan

**Abstract**: Automatic speech recognition (ASR) systems are known to be vulnerable to adversarial attacks. This paper addresses detection and defence against targeted white-box attacks on speech signals for ASR systems. While existing work has utilised diffusion models (DMs) to purify adversarial examples, achieving state-of-the-art results in keyword spotting tasks, their effectiveness for more complex tasks such as sentence-level ASR remains unexplored. Additionally, the impact of the number of forward diffusion steps on performance is not well understood. In this paper, we systematically investigate the use of DMs for defending against adversarial attacks on sentences and examine the effect of varying forward diffusion steps. Through comprehensive experiments on the Mozilla Common Voice dataset, we demonstrate that two forward diffusion steps can completely defend against adversarial attacks on sentences. Moreover, we introduce a novel, training-free approach for detecting adversarial attacks by leveraging a pre-trained DM. Our experimental results show that this method can detect adversarial attacks with high accuracy.

摘要: 众所周知，自动语音识别(ASR)系统容易受到对手攻击。本文研究了ASR系统中语音信号白盒攻击的检测和防御。虽然现有的工作已经利用扩散模型(DM)来净化对抗性例子，在关键字识别任务中取得了最先进的结果，但它们对更复杂任务(如句子级ASR)的有效性仍未被探索。此外，前向扩散步数对性能的影响还不是很清楚。在这篇文章中，我们系统地研究了DMS在防御对抗性句子攻击中的应用，并考察了不同的前向扩散步骤的效果。通过在Mozilla公共语音数据集上的综合实验，我们证明了两个前向扩散步骤可以完全防御针对句子的敌意攻击。此外，我们引入了一种新的、无需训练的方法来利用预先训练的DM来检测对抗性攻击。实验结果表明，该方法具有较高的检测准确率。



## **43. What Matters to Enhance Traffic Rule Compliance of Imitation Learning for End-to-End Autonomous Driving**

增强端到端自动驾驶模仿学习的交通规则合规性的重要性 cs.CV

14 pages, 3 figures

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2309.07808v3) [paper-pdf](http://arxiv.org/pdf/2309.07808v3)

**Authors**: Hongkuan Zhou, Wei Cao, Aifen Sui, Zhenshan Bing

**Abstract**: End-to-end autonomous driving, where the entire driving pipeline is replaced with a single neural network, has recently gained research attention because of its simpler structure and faster inference time. Despite this appealing approach largely reducing the complexity in the driving pipeline, it also leads to safety issues because the trained policy is not always compliant with the traffic rules. In this paper, we proposed P-CSG, a penalty-based imitation learning approach with contrastive-based cross semantics generation sensor fusion technologies to increase the overall performance of end-to-end autonomous driving. In this method, we introduce three penalties - red light, stop sign, and curvature speed penalty to make the agent more sensitive to traffic rules. The proposed cross semantics generation helps to align the shared information of different input modalities. We assessed our model's performance using the CARLA Leaderboard - Town 05 Long Benchmark and Longest6 Benchmark, achieving 8.5% and 2.0% driving score improvement compared to the baselines. Furthermore, we conducted robustness evaluations against adversarial attacks like FGSM and Dot attacks, revealing a substantial increase in robustness compared to other baseline models. More detailed information can be found at https://hk-zh.github.io/p-csg-plus.

摘要: 端到端自动驾驶，即整个驾驶管道被单一的神经网络取代，由于其结构更简单，推理时间更快，最近得到了研究人员的关注。尽管这种吸引人的方法极大地降低了驾驶管道的复杂性，但它也导致了安全问题，因为经过培训的政策并不总是符合交通规则。为了提高端到端自主驾驶的整体性能，本文提出了一种基于惩罚的模仿学习方法P-CSG，并结合基于对比的交叉语义生成传感器融合技术。在该方法中，我们引入了三种惩罚--红灯、停车标志和曲率速度惩罚，以使智能体对交通规则更加敏感。提出的交叉语义生成有助于对齐不同输入通道的共享信息。我们使用Carla Leaderboard-town 05 Long基准和Longest6基准评估了我们的模型的性能，与基准相比，驾驶分数分别提高了8.5%和2.0%。此外，我们对FGSM和Dot攻击等对手攻击进行了健壮性评估，显示出与其他基线模型相比，健壮性有了显著的提高。欲了解更多详细信息，请访问https://hk-zh.github.io/p-csg-plus.。



## **44. A Spatiotemporal Stealthy Backdoor Attack against Cooperative Multi-Agent Deep Reinforcement Learning**

针对协作多智能体深度强化学习的时空隐形后门攻击 cs.AI

6 pages, IEEE Globecom 2024

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07775v1) [paper-pdf](http://arxiv.org/pdf/2409.07775v1)

**Authors**: Yinbo Yu, Saihao Yan, Jiajia Liu

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform abnormal actions leading to failures or malicious goals. However, existing proposed backdoors suffer from several issues, e.g., fixed visual trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor attack against c-MADRL, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the attack duration. This method can guarantee the stealthiness and practicality of injected backdoors. Secondly, we hack the original reward function of the backdoored agent via reward reverse and unilateral guidance during training to ensure its adverse influence on the entire team. We evaluate our backdoor attacks on two classic c-MADRL algorithms VDN and QMIX, in a popular c-MADRL environment SMAC. The experimental results demonstrate that our backdoor attacks are able to reach a high attack success rate (91.6\%) while maintaining a low clean performance variance rate (3.7\%).

摘要: 最近的研究表明，协作多智能体深度强化学习(c-MADRL)受到后门攻击的威胁。一旦观察到后门触发器，它将执行导致失败或恶意目标的异常操作。然而，现有的拟议后门受到几个问题的困扰，例如，固定的视觉触发模式缺乏隐蔽性，后门被额外的网络训练或激活，或者所有代理都被后门。为此，本文提出了一种新的针对c-MADRL的后门攻击，通过在单个代理中嵌入后门来攻击整个多代理团队。首先，引入敌方时空行为模式作为后门触发，而不是人工注入固定的视觉模式或即时状态，并控制攻击持续时间。该方法可以保证注入后门的隐蔽性和实用性。其次，在训练过程中，通过奖励反转和单边指导，破解背靠背代理人原有的奖励功能，以确保其对整个团队的不利影响。我们在一个流行的c-MADRL环境SMAC中评估了我们对两个经典c-MADRL算法VDN和QMIX的后门攻击。实验结果表明，我们的后门攻击能够在保持较低的干净性能变异率(3.7)的同时达到较高的攻击成功率(91.6)。



## **45. Attack End-to-End Autonomous Driving through Module-Wise Noise**

通过模块化噪音攻击端到端自动驾驶 cs.LG

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07706v1) [paper-pdf](http://arxiv.org/pdf/2409.07706v1)

**Authors**: Lu Wang, Tianyuan Zhang, Yikai Han, Muyang Fang, Ting Jin, Jiaqi Kang

**Abstract**: With recent breakthroughs in deep neural networks, numerous tasks within autonomous driving have exhibited remarkable performance. However, deep learning models are susceptible to adversarial attacks, presenting significant security risks to autonomous driving systems. Presently, end-to-end architectures have emerged as the predominant solution for autonomous driving, owing to their collaborative nature across different tasks. Yet, the implications of adversarial attacks on such models remain relatively unexplored. In this paper, we conduct comprehensive adversarial security research on the modular end-to-end autonomous driving model for the first time. We thoroughly consider the potential vulnerabilities in the model inference process and design a universal attack scheme through module-wise noise injection. We conduct large-scale experiments on the full-stack autonomous driving model and demonstrate that our attack method outperforms previous attack methods. We trust that our research will offer fresh insights into ensuring the safety and reliability of autonomous driving systems.

摘要: 随着最近深度神经网络的突破，自动驾驶中的许多任务表现出了显著的性能。然而，深度学习模型容易受到对抗性攻击，给自动驾驶系统带来了巨大的安全风险。目前，端到端架构已经成为自动驾驶的主要解决方案，因为它们跨不同任务的协作性质。然而，对抗性攻击对这类模型的影响仍相对未被探索。本文首次对模块化端到端自主驾驶模型进行了全面的对抗性安全研究。我们充分考虑了模型推理过程中的潜在漏洞，并通过模块化噪声注入设计了一种通用的攻击方案。我们在全栈自主驾驶模型上进行了大规模的实验，证明了我们的攻击方法比以前的攻击方法要好。我们相信，我们的研究将为确保自动驾驶系统的安全和可靠性提供新的见解。



## **46. A Training Rate and Survival Heuristic for Inference and Robustness Evaluation (TRASHFIRE)**

推理和稳健性评估的训练率和生存启发式（TRASHFIRE） cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2401.13751v2) [paper-pdf](http://arxiv.org/pdf/2401.13751v2)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Machine learning models -- deep neural networks in particular -- have performed remarkably well on benchmark datasets across a wide variety of domains. However, the ease of finding adversarial counter-examples remains a persistent problem when training times are measured in hours or days and the time needed to find a successful adversarial counter-example is measured in seconds. Much work has gone into generating and defending against these adversarial counter-examples, however the relative costs of attacks and defences are rarely discussed. Additionally, machine learning research is almost entirely guided by test/train metrics, but these would require billions of samples to meet industry standards. The present work addresses the problem of understanding and predicting how particular model hyper-parameters influence the performance of a model in the presence of an adversary. The proposed approach uses survival models, worst-case examples, and a cost-aware analysis to precisely and accurately reject a particular model change during routine model training procedures rather than relying on real-world deployment, expensive formal verification methods, or accurate simulations of very complicated systems (\textit{e.g.}, digitally recreating every part of a car or a plane). Through an evaluation of many pre-processing techniques, adversarial counter-examples, and neural network configurations, the conclusion is that deeper models do offer marginal gains in survival times compared to more shallow counterparts. However, we show that those gains are driven more by the model inference time than inherent robustness properties. Using the proposed methodology, we show that ResNet is hopelessly insecure against even the simplest of white box attacks.

摘要: 机器学习模型--尤其是深度神经网络--在各种领域的基准数据集上表现得非常好。然而，当训练时间以小时或天衡量，而找到成功的对抗性反例所需的时间以秒衡量时，寻找对抗性反例的容易程度仍然是一个持久的问题。在生成和防御这些对抗性反例方面已经做了很多工作，然而攻击和防御的相对成本很少被讨论。此外，机器学习研究几乎完全由测试/训练指标指导，但这些指标需要数十亿个样本才能满足行业标准。目前的工作解决的问题是理解和预测特定的模型超参数如何在对手存在的情况下影响模型的性能。该方法使用生存模型、最坏情况示例和成本意识分析，在常规模型训练过程中准确和准确地拒绝特定的模型更改，而不是依赖于真实世界的部署、昂贵的形式验证方法或非常复杂的系统的准确模拟(例如，以数字方式重建汽车或飞机的每个部件)。通过对许多预处理技术、对抗性反例和神经网络配置的评估，结论是，与较浅的模型相比，较深的模型确实提供了生存时间的边际收益。然而，我们表明，这些收益更多地是由模型推理时间驱动的，而不是固有的稳健性。使用提出的方法，我们表明ResNet对于即使是最简单的白盒攻击也是无可救药的不安全的。



## **47. A Cost-Aware Approach to Adversarial Robustness in Neural Networks**

神经网络中对抗鲁棒性的一种具有成本意识的方法 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07609v1) [paper-pdf](http://arxiv.org/pdf/2409.07609v1)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Considering the growing prominence of production-level AI and the threat of adversarial attacks that can evade a model at run-time, evaluating the robustness of models to these evasion attacks is of critical importance. Additionally, testing model changes likely means deploying the models to (e.g. a car or a medical imaging device), or a drone to see how it affects performance, making un-tested changes a public problem that reduces development speed, increases cost of development, and makes it difficult (if not impossible) to parse cause from effect. In this work, we used survival analysis as a cloud-native, time-efficient and precise method for predicting model performance in the presence of adversarial noise. For neural networks in particular, the relationships between the learning rate, batch size, training time, convergence time, and deployment cost are highly complex, so researchers generally rely on benchmark datasets to assess the ability of a model to generalize beyond the training data. To address this, we propose using accelerated failure time models to measure the effect of hardware choice, batch size, number of epochs, and test-set accuracy by using adversarial attacks to induce failures on a reference model architecture before deploying the model to the real world. We evaluate several GPU types and use the Tree Parzen Estimator to maximize model robustness and minimize model run-time simultaneously. This provides a way to evaluate the model and optimise it in a single step, while simultaneously allowing us to model the effect of model parameters on training time, prediction time, and accuracy. Using this technique, we demonstrate that newer, more-powerful hardware does decrease the training time, but with a monetary and power cost that far outpaces the marginal gains in accuracy.

摘要: 考虑到产生级人工智能的日益突出以及运行时可以逃避模型的对抗性攻击的威胁，评估模型对这些逃避攻击的稳健性至关重要。此外，测试模型更改可能意味着将模型部署到(例如，汽车或医疗成像设备)或无人机，以了解它如何影响性能，使未经测试的更改成为一个公共问题，从而降低开发速度、增加开发成本，并使分析原因和结果变得困难(如果不是不可能的话)。在这项工作中，我们使用生存分析作为一种云本地的、时间高效和精确的方法来预测存在对抗性噪声存在的模型性能。尤其对于神经网络，学习速度、批次大小、训练时间、收敛时间和部署成本之间的关系非常复杂，因此研究人员通常依赖基准数据集来评估模型在训练数据之外的泛化能力。为了解决这个问题，我们建议在将参考模型部署到现实世界之前，使用加速故障时间模型来衡量硬件选择、批量大小、历元数和测试集精度的影响，方法是使用对抗性攻击在参考模型体系结构上诱导故障。我们评估了几种类型的GPU，并使用Tree Parzen Estimator来最大化模型的健壮性，同时最小化模型的运行时间。这提供了一种在单个步骤中评估和优化模型的方法，同时允许我们对模型参数对训练时间、预测时间和精度的影响进行建模。使用这项技术，我们证明了更新的、功能更强大的硬件确实减少了训练时间，但在金钱和电力成本上远远超过了精度的边际收益。



## **48. Resilient Graph Neural Networks: A Coupled Dynamical Systems Approach**

弹性图神经网络：一种耦合动态系统方法 cs.LG

ECAI 2024

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2311.06942v3) [paper-pdf](http://arxiv.org/pdf/2311.06942v3)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of coupled dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.

摘要: 图形神经网络(GNN)已经成为解决各种基于图形的任务的关键组件。尽管GNN取得了显著的成功，但它们仍然容易受到对抗性攻击形式的投入扰动的影响。本文介绍了一种通过耦合动力系统的透镜来增强GNN抵抗敌意扰动的创新方法。我们的方法引入了基于具有压缩性质的微分方程的图神经层，从而提高了GNN的稳健性。该方法的一个显著特点是节点特征和邻接矩阵的同时学习进化，从而内在地增强了模型对输入特征扰动和图的连通性的稳健性。我们从数学上推导出我们的新体系结构的基础，并提供理论见解来推理其预期行为。我们通过许多真实世界的基准测试来证明我们的方法的有效性，与现有的方法相比，我们的阅读是平分的，或者是性能有所提高。



## **49. Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS**

引入扰动能力评分（PS）增强ML-NIDS对抗规避攻击的鲁棒性 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07448v1) [paper-pdf](http://arxiv.org/pdf/2409.07448v1)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: This paper proposes a novel Perturb-ability Score (PS) that can be used to identify Network Intrusion Detection Systems (NIDS) features that can be easily manipulated by attackers in the problem-space. We demonstrate that using PS to select only non-perturb-able features for ML-based NIDS maintains detection performance while enhancing robustness against adversarial attacks.

摘要: 本文提出了一种新颖的扰动能力评分（PS），可用于识别攻击者在问题空间中容易操纵的网络入侵检测系统（NIDS）特征。我们证明，使用PS仅为基于ML的NIDS选择不可扰动的特征可以保持检测性能，同时增强针对对抗性攻击的鲁棒性。



## **50. Enhancing adversarial robustness in Natural Language Inference using explanations**

使用解释增强自然语言推理中的对抗稳健性 cs.CL

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07423v1) [paper-pdf](http://arxiv.org/pdf/2409.07423v1)

**Authors**: Alexandros Koulakos, Maria Lymperaiou, Giorgos Filandrianos, Giorgos Stamou

**Abstract**: The surge of state-of-the-art Transformer-based models has undoubtedly pushed the limits of NLP model performance, excelling in a variety of tasks. We cast the spotlight on the underexplored task of Natural Language Inference (NLI), since models trained on popular well-suited datasets are susceptible to adversarial attacks, allowing subtle input interventions to mislead the model. In this work, we validate the usage of natural language explanation as a model-agnostic defence strategy through extensive experimentation: only by fine-tuning a classifier on the explanation rather than premise-hypothesis inputs, robustness under various adversarial attacks is achieved in comparison to explanation-free baselines. Moreover, since there is no standard strategy of testing the semantic validity of the generated explanations, we research the correlation of widely used language generation metrics with human perception, in order for them to serve as a proxy towards robust NLI models. Our approach is resource-efficient and reproducible without significant computational limitations.

摘要: 最先进的基于变形金刚的模型的激增无疑已经突破了NLP模型的性能极限，在各种任务中表现出色。我们将注意力集中在自然语言推理(NLI)这一未被探索的任务上，因为在流行的匹配良好的数据集上训练的模型容易受到对抗性攻击，允许微妙的输入干预误导模型。在这项工作中，我们通过广泛的实验验证了自然语言解释作为一种与模型无关的防御策略的使用：只有通过微调解释而不是前提假设输入的分类器，才能实现与无解释基线相比在各种对手攻击下的健壮性。此外，由于没有标准的策略来测试生成的解释的语义有效性，我们研究了广泛使用的语言生成度量与人类感知的相关性，以便它们能够作为稳健的NLI模型的代理。我们的方法是资源高效和可重复性的，没有明显的计算限制。



