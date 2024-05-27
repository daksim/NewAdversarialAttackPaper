# Latest Adversarial Attack Papers
**update at 2024-05-27 09:25:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models as Covert Channels... a Systematic Analysis**

$$\mathBF{L ' 2\csot M = C ' 2}$$大型语言模型作为隐蔽通道.了系统的分析 cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15652v1) [paper-pdf](http://arxiv.org/pdf/2405.15652v1)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity in the last few years due to their performance in diverse tasks such as translation, prediction, or content generation. At the same time, the research community has shown that LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship resistant communication?   In this paper, we explore the capabilities of open-source LLM-based covert channels. We approach this problem from the experimental side by empirically measuring the security vs. capacity of the open-source LLM model (Llama-7B) to assess how well it performs as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, which depend on message length and model entropy, we also show that the chance for an adversary to detect covert communication is low. To ensure that our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.

摘要: 大型语言模型(LLM)在过去几年中因其在翻译、预测或内容生成等各种任务中的性能而广受欢迎。与此同时，研究界的研究表明，LLMS容易受到各种攻击，但也可以提高各种系统的安全性。然而，除了实现更安全的系统外，开源LLM作为隐蔽文本分发在促进抗审查通信方面的表现如何？在这篇文章中，我们探索了基于LLM的开源隐蔽通道的能力。我们从实验方面解决这个问题，通过经验测量开放源码LLM模型(LLAMA-7B)的安全性与容量，以评估它作为隐蔽通道的表现如何。虽然我们的结果表明，这种信道不太可能获得较高的实际比特率，这取决于消息长度和模型熵，但我们也表明，攻击者检测到隐蔽通信的机会很低。为了确保我们的结果能够以最小的努力作为一般参考，我们采用了一个概念上简单而简明的方案，并且只假设公共模型。



## **2. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15589v1) [paper-pdf](http://arxiv.org/pdf/2405.15589v1)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们在不同家族(Gema，Phi3，Mistral，Zephy)和不同尺度(2B，3.8B，7B)的四个模型上的实验评估表明，这两种算法在保持实用性的同时，显著提高了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **3. Rethinking Independent Cross-Entropy Loss For Graph-Structured Data**

重新思考图结构数据的独立交叉熵损失 cs.LG

20 pages, 4 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15564v1) [paper-pdf](http://arxiv.org/pdf/2405.15564v1)

**Authors**: Rui Miao, Kaixiong Zhou, Yili Wang, Ninghao Liu, Ying Wang, Xin Wang

**Abstract**: Graph neural networks (GNNs) have exhibited prominent performance in learning graph-structured data. Considering node classification task, based on the i.i.d assumption among node labels, the traditional supervised learning simply sums up cross-entropy losses of the independent training nodes and applies the average loss to optimize GNNs' weights. But different from other data formats, the nodes are naturally connected. It is found that the independent distribution modeling of node labels restricts GNNs' capability to generalize over the entire graph and defend adversarial attacks. In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. We learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data-label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. The extensive experiments demonstrate that our joint-cluster supervised learning can effectively bolster GNNs' node classification accuracy. Furthermore, being benefited from the reference signals which may be free from spiteful interference, our learning paradigm significantly protects the node classification from being affected by the adversarial attack.

摘要: 图神经网络(GNN)在学习图结构数据方面表现出了突出的性能。考虑到节点分类任务，传统的监督学习基于节点标签间的I.I.D假设，简单地总结独立训练节点的交叉熵损失，并利用平均损失来优化GNN的权值。但与其他数据格式不同的是，节点是自然相连的。研究发现，节点标签的独立分布建模限制了GNN在整个图上的泛化能力和防御敌意攻击的能力。在这项工作中，我们提出了一种新的框架，称为联合聚类监督学习，以建模每个节点与其对应的簇的联合分布。我们学习节点和簇标签在表示条件下的联合分布，并用获得的联合损失来训练GNN。这样，从本地簇中提取的数据标签参考信号明确地增强了对目标节点的区分能力。大量实验表明，我们的联合聚类监督学习能够有效地提高GNN的节点分类精度。此外，得益于参考信号可能不会受到恶意干扰，我们的学习范例显著地保护了节点分类不受恶意攻击的影响。



## **4. AuthNet: Neural Network with Integrated Authentication Logic**

AuthNet：具有集成认证逻辑的神经网络 cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15426v1) [paper-pdf](http://arxiv.org/pdf/2405.15426v1)

**Authors**: Yuling Cai, Fan Xiang, Guozhu Meng, Yinzhi Cao, Kai Chen

**Abstract**: Model stealing, i.e., unauthorized access and exfiltration of deep learning models, has become one of the major threats. Proprietary models may be protected by access controls and encryption. However, in reality, these measures can be compromised due to system breaches, query-based model extraction or a disgruntled insider. Security hardening of neural networks is also suffering from limits, for example, model watermarking is passive, cannot prevent the occurrence of piracy and not robust against transformations. To this end, we propose a native authentication mechanism, called AuthNet, which integrates authentication logic as part of the model without any additional structures. Our key insight is to reuse redundant neurons with low activation and embed authentication bits in an intermediate layer, called a gate layer. Then, AuthNet fine-tunes the layers after the gate layer to embed authentication logic so that only inputs with special secret key can trigger the correct logic of AuthNet. It exhibits two intuitive advantages. It provides the last line of defense, i.e., even being exfiltrated, the model is not usable as the adversary cannot generate valid inputs without the key. Moreover, the authentication logic is difficult to inspect and identify given millions or billions of neurons in the model. We theoretically demonstrate the high sensitivity of AuthNet to the secret key and its high confusion for unauthorized samples. AuthNet is compatible with any convolutional neural network, where our extensive evaluations show that AuthNet successfully achieves the goal in rejecting unauthenticated users (whose average accuracy drops to 22.03%) with a trivial accuracy decrease (1.18% on average) for legitimate users, and is robust against model transformation and adaptive attacks.

摘要: 模型窃取，即对深度学习模型的未经授权的访问和渗出，已成为主要威胁之一。专有型号可以通过访问控制和加密进行保护。然而，在现实中，这些措施可能会由于系统违规、基于查询的模型提取或心怀不满的内部人员而受到损害。神经网络的安全加固也面临着局限性，例如模型水印是被动的，不能防止盗版的发生，对变换的鲁棒性不强。为此，我们提出了一种原生认证机制，称为AuthNet，它将认证逻辑整合为模型的一部分，而不需要任何额外的结构。我们的关键见解是重用低激活的冗余神经元，并在称为门层的中间层中嵌入身份验证比特。然后，AuthNet对门层之后的层进行微调，嵌入认证逻辑，只有使用特殊密钥的输入才能触发AuthNet的正确逻辑。它表现出两个直观的优势。它提供了最后一道防线，即即使被渗透，该模型也是不可用的，因为没有密钥，对手无法生成有效的输入。此外，在模型中存在数百万或数十亿个神经元的情况下，验证逻辑很难检查和识别。我们从理论上证明了AuthNet对密钥的高度敏感性和对未经授权的样本的高度混淆。AuthNet兼容任何卷积神经网络，广泛的测试表明，AuthNet成功地达到了拒绝未认证用户的目标(其平均准确率下降到22.03%)，合法用户的准确率略有下降(平均1.18%)，并且对模型变换和自适应攻击具有健壮性。



## **5. Lost in the Averages: A New Specific Setup to Evaluate Membership Inference Attacks Against Machine Learning Models**

迷失在空中：一种新的特定设置来评估针对机器学习模型的成员推断攻击 cs.LG

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15423v1) [paper-pdf](http://arxiv.org/pdf/2405.15423v1)

**Authors**: Florent Guépin, Nataša Krčo, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Membership Inference Attacks (MIAs) are widely used to evaluate the propensity of a machine learning (ML) model to memorize an individual record and the privacy risk releasing the model poses. MIAs are commonly evaluated similarly to ML models: the MIA is performed on a test set of models trained on datasets unseen during training, which are sampled from a larger pool, $D_{eval}$. The MIA is evaluated across all datasets in this test set, and is thus evaluated across the distribution of samples from $D_{eval}$. While this was a natural extension of ML evaluation to MIAs, recent work has shown that a record's risk heavily depends on its specific dataset. For example, outliers are particularly vulnerable, yet an outlier in one dataset may not be one in another. The sources of randomness currently used to evaluate MIAs may thus lead to inaccurate individual privacy risk estimates. We propose a new, specific evaluation setup for MIAs against ML models, using weight initialization as the sole source of randomness. This allows us to accurately evaluate the risk associated with the release of a model trained on a specific dataset. Using SOTA MIAs, we empirically show that the risk estimates given by the current setup lead to many records being misclassified as low risk. We derive theoretical results which, combined with empirical evidence, suggest that the risk calculated in the current setup is an average of the risks specific to each sampled dataset, validating our use of weight initialization as the only source of randomness. Finally, we consider an MIA with a stronger adversary leveraging information about the target dataset to infer membership. Taken together, our results show that current MIA evaluation is averaging the risk across datasets leading to inaccurate risk estimates, and the risk posed by attacks leveraging information about the target dataset to be potentially underestimated.

摘要: 成员关系推断攻击(MIA)被广泛用于评估机器学习(ML)模型记忆个人记录的倾向和释放模型带来的隐私风险。MIA的评估通常类似于ML模型：MIA是在训练过程中看不到的数据集上训练的一组测试模型上执行的，这些数据集是从更大的池$D_{val}$中采样的。在该测试集中的所有数据集上评估MIA，因此在$D_{val}$中的样本分布上评估MIA。虽然这是ML评估对MIA的自然扩展，但最近的研究表明，记录的风险在很大程度上取决于其特定的数据集。例如，异常值特别容易受到攻击，但一个数据集中的异常值可能不是另一个数据集中的异常值。因此，目前用于评估MIA的随机性来源可能会导致不准确的个人隐私风险估计。我们提出了一种针对ML模型的MIA的新的、特定的评估设置，使用权重初始化作为唯一的随机性来源。这使我们能够准确地评估与发布针对特定数据集训练的模型相关的风险。使用SOTA MIA，我们的经验表明，当前设置给出的风险估计导致许多记录被错误归类为低风险。我们得出的理论结果，与经验证据相结合，表明在当前设置中计算的风险是每个样本数据集特定的风险的平均值，验证了我们使用权重初始化作为随机性的唯一来源。最后，我们考虑一个具有更强对手的MIA，该对手利用有关目标数据集的信息来推断成员资格。综上所述，我们的结果表明，当前的MIA评估正在平均跨数据集的风险，从而导致不准确的风险估计，而利用有关目标数据集的信息的攻击所构成的风险可能被低估。



## **6. Decaf: Data Distribution Decompose Attack against Federated Learning**

Decaf：数据分布分解对联邦学习的攻击 cs.LG

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15316v1) [paper-pdf](http://arxiv.org/pdf/2405.15316v1)

**Authors**: Zhiyang Dai, Chunyi Zhou, Anmin Fu

**Abstract**: In contrast to prevalent Federated Learning (FL) privacy inference techniques such as generative adversarial networks attacks, membership inference attacks, property inference attacks, and model inversion attacks, we devise an innovative privacy threat: the Data Distribution Decompose Attack on FL, termed Decaf. This attack enables an honest-but-curious FL server to meticulously profile the proportion of each class owned by the victim FL user, divulging sensitive information like local market item distribution and business competitiveness. The crux of Decaf lies in the profound observation that the magnitude of local model gradient changes closely mirrors the underlying data distribution, including the proportion of each class. Decaf addresses two crucial challenges: accurately identify the missing/null class(es) given by any victim user as a premise and then quantify the precise relationship between gradient changes and each remaining non-null class. Notably, Decaf operates stealthily, rendering it entirely passive and undetectable to victim users regarding the infringement of their data distribution privacy. Experimental validation on five benchmark datasets (MNIST, FASHION-MNIST, CIFAR-10, FER-2013, and SkinCancer) employing diverse model architectures, including customized convolutional networks, standardized VGG16, and ResNet18, demonstrates Decaf's efficacy. Results indicate its ability to accurately decompose local user data distribution, regardless of whether it is IID or non-IID distributed. Specifically, the dissimilarity measured using $L_{\infty}$ distance between the distribution decomposed by Decaf and ground truth is consistently below 5\% when no null classes exist. Moreover, Decaf achieves 100\% accuracy in determining any victim user's null classes, validated through formal proof.

摘要: 针对目前流行的联合学习隐私推理技术，如生成性对抗网络攻击、成员关系推理攻击、属性推理攻击、模型反转攻击等，本文提出了一种新颖的联合学习隐私威胁方法：数据分布分解攻击。这种攻击使诚实但好奇的FL服务器能够仔细分析受害FL用户拥有的每个类别的比例，泄露当地市场项目分布和商业竞争力等敏感信息。DECF的关键在于深刻地观察到局部模型梯度的大小变化密切地反映了底层数据的分布，包括每一类的比例。Decaf解决了两个关键挑战：准确识别任何受害者用户给出的缺失/空类作为前提，然后量化梯度变化与每个剩余非空类之间的精确关系。值得注意的是，Decaf的操作是秘密的，使其完全被动，受害者用户无法检测到他们的数据分发隐私受到侵犯。在五个基准数据集(MNIST、FORM-MNIST、CIFAR-10、FER-2013和皮肤癌)上进行了实验验证，这些数据集采用了不同的模型架构，包括定制卷积网络、标准化VGG16和ResNet18，验证了Decaf的有效性。结果表明，它能够准确分解本地用户数据分布，无论它是IID分布的还是非IID分布的。具体地说，当不存在零类时，用DECF分解的分布与地面真实数据之间的$L距离测量的差异始终小于5。此外，通过形式化证明，该算法在确定受害用户的空类时达到了100%的准确率。



## **7. Robust Diffusion Models for Adversarial Purification**

对抗性净化的鲁棒扩散模型 cs.CV

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2403.16067v2) [paper-pdf](http://arxiv.org/pdf/2403.16067v2)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted on CIFAR-10, CIFAR-100 and ImageNet to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.

摘要: 基于扩散模型(DM)的对抗净化(AP)已被证明是对抗训练(AT)最有效的替代方法。然而，这些方法忽略了这样一个事实，即预先训练的扩散模型本身对对手攻击也不是很健壮。此外，扩散过程容易破坏语义信息，生成高质量的图像，但反向处理后的图像与原始输入图像完全不同，导致标准精度下降。为了克服这些问题，一个自然的想法是利用对抗性训练策略来重新训练或微调预先训练的扩散模型，这在计算上是令人望而却步的。我们提出了一种新的具有对抗性指导的稳健逆向过程，它独立于给定的预先训练的DM，并且避免了对DM的重新训练或微调。这种健壮的指导不仅可以确保生成保持更多语义内容的纯化实例，还可以第一次缓解DM的准确性和健壮性之间的权衡，这也为基于DM的AP提供了对新攻击的有效适应能力。在CIFAR-10、CIFAR-100和ImageNet上的大量实验表明，该方法达到了最先进的结果，并对不同的攻击表现出了泛化能力。



## **8. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

18 pages, 13 figures, 4 tables

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.13401v2) [paper-pdf](http://arxiv.org/pdf/2405.13401v2)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **9. Adversarial Attacks on Hidden Tasks in Multi-Task Learning**

多任务学习中隐藏任务的对抗攻击 cs.LG

14 pages, 6 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15244v1) [paper-pdf](http://arxiv.org/pdf/2405.15244v1)

**Authors**: Yu Zhe, Rei Nagaike, Daiki Nishiyama, Kazuto Fukuchi, Jun Sakuma

**Abstract**: Deep learning models are susceptible to adversarial attacks, where slight perturbations to input data lead to misclassification. Adversarial attacks become increasingly effective with access to information about the targeted classifier. In the context of multi-task learning, where a single model learns multiple tasks simultaneously, attackers may aim to exploit vulnerabilities in specific tasks with limited information. This paper investigates the feasibility of attacking hidden tasks within multi-task classifiers, where model access regarding the hidden target task and labeled data for the hidden target task are not available, but model access regarding the non-target tasks is available. We propose a novel adversarial attack method that leverages knowledge from non-target tasks and the shared backbone network of the multi-task model to force the model to forget knowledge related to the target task. Experimental results on CelebA and DeepFashion datasets demonstrate the effectiveness of our method in degrading the accuracy of hidden tasks while preserving the performance of visible tasks, contributing to the understanding of adversarial vulnerabilities in multi-task classifiers.

摘要: 深度学习模型容易受到敌意攻击，在这种攻击中，输入数据的轻微扰动会导致错误分类。通过访问有关目标分类器的信息，对抗性攻击变得越来越有效。在多任务学习的情况下，单个模型同时学习多个任务，攻击者可能会利用有限信息的特定任务中的漏洞进行攻击。研究了在多任务分类器中攻击隐藏任务的可行性，其中隐藏目标任务的模型访问和隐藏目标任务的标记数据不可用，而非目标任务的模型访问可用。我们提出了一种新的对抗性攻击方法，该方法利用来自非目标任务的知识和多任务模型的共享骨干网络来迫使模型忘记与目标任务相关的知识。在CelebA和DeepFashion数据集上的实验结果表明，该方法在保持可见任务性能的同时，降低了隐藏任务的准确率，有助于理解多任务分类器中的对抗性弱点。



## **10. Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models**

扩散模型中鲁棒概念擦除的对抗训练防御性取消学习 cs.CV

Codes are available at https://github.com/OPTML-Group/AdvUnlearn

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15234v1) [paper-pdf](http://arxiv.org/pdf/2405.15234v1)

**Authors**: Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, Sijia Liu

**Abstract**: Diffusion models (DMs) have achieved remarkable success in text-to-image generation, but they also pose safety risks, such as the potential generation of harmful content and copyright violations. The techniques of machine unlearning, also known as concept erasing, have been developed to address these risks. However, these techniques remain vulnerable to adversarial prompt attacks, which can prompt DMs post-unlearning to regenerate undesired images containing concepts (such as nudity) meant to be erased. This work aims to enhance the robustness of concept erasing by integrating the principle of adversarial training (AT) into machine unlearning, resulting in the robust unlearning framework referred to as AdvUnlearn. However, achieving this effectively and efficiently is highly nontrivial. First, we find that a straightforward implementation of AT compromises DMs' image generation quality post-unlearning. To address this, we develop a utility-retaining regularization on an additional retain set, optimizing the trade-off between concept erasure robustness and model utility in AdvUnlearn. Moreover, we identify the text encoder as a more suitable module for robustification compared to UNet, ensuring unlearning effectiveness. And the acquired text encoder can serve as a plug-and-play robust unlearner for various DM types. Empirically, we perform extensive experiments to demonstrate the robustness advantage of AdvUnlearn across various DM unlearning scenarios, including the erasure of nudity, objects, and style concepts. In addition to robustness, AdvUnlearn also achieves a balanced tradeoff with model utility. To our knowledge, this is the first work to systematically explore robust DM unlearning through AT, setting it apart from existing methods that overlook robustness in concept erasing. Codes are available at: https://github.com/OPTML-Group/AdvUnlearn

摘要: 扩散模型(DM)在文本到图像的生成方面取得了显著的成功，但它们也带来了安全风险，如可能生成有害内容和侵犯版权。机器遗忘技术，也被称为概念擦除，就是为了解决这些风险而开发的。然而，这些技术仍然容易受到敌意的即时攻击，这可能会促使忘记后的DM重新生成包含要擦除的概念(如裸体)的不需要的图像。这项工作旨在通过将对抗性训练(AT)的原理整合到机器遗忘中来增强概念删除的稳健性，从而产生健壮的遗忘框架，称为AdvUnLearning。然而，有效和高效地实现这一点并不是微不足道的。首先，我们发现AT的直接实现损害了DM在遗忘后的图像生成质量。为了解决这个问题，我们在一个额外的保留集上开发了效用保留正则化，优化了AdvUnLearning中概念删除健壮性和模型实用之间的权衡。此外，我们认为文本编码器是一个更适合于粗暴的模块，与联合国教科文组织相比，确保了遗忘的有效性。并且所获得的文本编码器可以作为各种DM类型的即插即用鲁棒去学习器。经验性地，我们进行了大量的实验来展示AdvUnLearning在各种DM遗忘场景中的健壮性优势，包括对裸体、物体和风格概念的删除。除了健壮性之外，AdvUnLearning还实现了与模型实用程序之间的平衡。据我们所知，这是第一个通过AT系统地探索稳健的DM遗忘的工作，区别于现有的忽略概念删除中的稳健性的方法。代码可在以下网址获得：https://github.com/OPTML-Group/AdvUnlearn



## **11. TrojanForge: Adversarial Hardware Trojan Examples with Reinforcement Learning**

TrojanForge：具有强化学习的对抗性硬件木马示例 cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15184v1) [paper-pdf](http://arxiv.org/pdf/2405.15184v1)

**Authors**: Amin Sarihi, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: The Hardware Trojan (HT) problem can be thought of as a continuous game between attackers and defenders, each striving to outsmart the other by leveraging any available means for an advantage. Machine Learning (ML) has recently been key in advancing HT research. Various novel techniques, such as Reinforcement Learning (RL) and Graph Neural Networks (GNNs), have shown HT insertion and detection capabilities. HT insertion with ML techniques, specifically, has seen a spike in research activity due to the shortcomings of conventional HT benchmarks and the inherent human design bias that occurs when we create them. This work continues this innovation by presenting a tool called "TrojanForge", capable of generating HT adversarial examples that defeat HT detectors; demonstrating the capabilities of GAN-like adversarial tools for automatic HT insertion. We introduce an RL environment where the RL insertion agent interacts with HT detectors in an insertion-detection loop where the agent collects rewards based on its success in bypassing HT detectors. Our results show that this process leads to inserted HTs that evade various HT detectors, achieving high attack success percentages. This tool provides insight into why HT insertion fails in some instances and how we can leverage this knowledge in defense.

摘要: 硬件特洛伊木马(HT)问题可以被认为是攻击者和防御者之间的一场持续的游戏，双方都在努力利用任何可用的手段来获取优势，以智胜对方。近年来，机器学习(ML)已成为推进HT研究的关键。各种新的技术，如强化学习(RL)和图形神经网络(GNNS)，已经显示出HT插入和检测能力。特别是，由于传统的HT基准的缺点以及我们创建它们时固有的人为设计偏差，使用ML技术插入HT的研究活动出现了激增。这项工作通过提供一个名为“TrojanForge”的工具来继续这一创新，该工具能够生成击败HT检测器的HT对抗示例；展示了类GaN对抗工具自动插入HT的能力。我们引入了一种RL环境，在该环境中，RL插入剂与HT检测器在插入检测环路中相互作用，其中代理根据其成功绕过HT检测器来收取奖励。我们的结果表明，这一过程导致插入的HTS避开了各种HT检测器，获得了高攻击成功率。这个工具提供了一些关于为什么在某些情况下HT插入失败以及我们如何在防御中利用这一知识的洞察。



## **12. Are You Copying My Prompt? Protecting the Copyright of Vision Prompt for VPaaS via Watermark**

你正在卸载我的提示吗？通过水印保护VPaSaaS Vision Promise的版权 cs.CR

11 pages, 7 figures,

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15161v1) [paper-pdf](http://arxiv.org/pdf/2405.15161v1)

**Authors**: Huali Ren, Anli Yan, Chong-zhi Gao, Hongyang Yan, Zhenxin Zhang, Jin Li

**Abstract**: Visual Prompt Learning (VPL) differs from traditional fine-tuning methods in reducing significant resource consumption by avoiding updating pre-trained model parameters. Instead, it focuses on learning an input perturbation, a visual prompt, added to downstream task data for making predictions. Since learning generalizable prompts requires expert design and creation, which is technically demanding and time-consuming in the optimization process, developers of Visual Prompts as a Service (VPaaS) have emerged. These developers profit by providing well-crafted prompts to authorized customers. However, a significant drawback is that prompts can be easily copied and redistributed, threatening the intellectual property of VPaaS developers. Hence, there is an urgent need for technology to protect the rights of VPaaS developers. To this end, we present a method named \textbf{WVPrompt} that employs visual prompt watermarking in a black-box way. WVPrompt consists of two parts: prompt watermarking and prompt verification. Specifically, it utilizes a poison-only backdoor attack method to embed a watermark into the prompt and then employs a hypothesis-testing approach for remote verification of prompt ownership. Extensive experiments have been conducted on three well-known benchmark datasets using three popular pre-trained models: RN50, BIT-M, and Instagram. The experimental results demonstrate that WVPrompt is efficient, harmless, and robust to various adversarial operations.

摘要: 视觉提示学习(VPL)不同于传统的微调方法，它通过避免更新预先训练的模型参数来减少显着的资源消耗。相反，它专注于学习输入扰动，即添加到下游任务数据中进行预测的视觉提示。由于学习可概括的提示需要专家设计和创建，这在优化过程中技术要求高且耗时，因此出现了可视化提示即服务(VPaaS)的开发人员。这些开发人员通过向授权客户提供精心设计的提示来获利。然而，一个明显的缺点是提示很容易被复制和重新分发，威胁到VPaaS开发人员的知识产权。因此，迫切需要技术来保护VPaaS开发人员的权利。为此，我们提出了一种以黑盒方式使用视觉提示水印的方法，命名为\extbf{WVPrompt}。WVPrompt由两部分组成：提示水印和提示验证。具体地说，它利用仅有毒的后门攻击方法在提示中嵌入水印，然后使用假设检验方法对提示所有权进行远程验证。在三个著名的基准数据集上进行了广泛的实验，使用三个流行的预训练模型：RN50、BIT-M和Instagram。实验结果表明，WVPrompt是一种高效、无害、对各种敌意操作具有健壮性的算法。



## **13. Quantum Public-Key Encryption with Tamper-Resilient Public Keys from One-Way Functions**

使用来自单向函数的防篡改公钥的量子公钥加密 quant-ph

47pages

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2304.01800v4) [paper-pdf](http://arxiv.org/pdf/2304.01800v4)

**Authors**: Fuyuki Kitagawa, Tomoyuki Morimae, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We construct quantum public-key encryption from one-way functions. In our construction, public keys are quantum, but ciphertexts are classical. Quantum public-key encryption from one-way functions (or weaker primitives such as pseudorandom function-like states) are also proposed in some recent works [Morimae-Yamakawa, eprint:2022/1336; Coladangelo, eprint:2023/282; Barooti-Grilo-Malavolta-Sattath-Vu-Walter, eprint:2023/877]. However, they have a huge drawback: they are secure only when quantum public keys can be transmitted to the sender (who runs the encryption algorithm) without being tampered with by the adversary, which seems to require unsatisfactory physical setup assumptions such as secure quantum channels. Our construction is free from such a drawback: it guarantees the secrecy of the encrypted messages even if we assume only unauthenticated quantum channels. Thus, the encryption is done with adversarially tampered quantum public keys. Our construction is the first quantum public-key encryption that achieves the goal of classical public-key encryption, namely, to establish secure communication over insecure channels, based only on one-way functions. Moreover, we show a generic compiler to upgrade security against chosen plaintext attacks (CPA security) into security against chosen ciphertext attacks (CCA security) only using one-way functions. As a result, we obtain CCA secure quantum public-key encryption based only on one-way functions.

摘要: 我们用单向函数构造量子公钥加密。在我们的构造中，公钥是量子的，但密文是经典的。最近的一些工作也提出了基于单向函数(或更弱的基元，如伪随机函数类状态)的量子公钥加密[Morimae-Yamakawa，ePrint：2022/1336；Coladangelo，ePrint：2023/282；Barooti-Grilo-Malavolta-Sattath-Vu-Walter，ePrint：2023/877]。然而，它们有一个巨大的缺点：只有当量子公钥可以传输给发送者(运行加密算法)而不被对手篡改时，它们才是安全的，这似乎需要不令人满意的物理设置假设，如安全量子通道。我们的构造没有这样的缺点：它保证了加密消息的保密性，即使我们假设只有未经验证的量子通道。因此，加密是用恶意篡改的量子公钥完成的。我们的构造是第一个量子公钥加密，它实现了经典公钥加密的目标，即仅基于单向函数在不安全的通道上建立安全通信。此外，我们还给出了一个通用编译器，该编译器仅使用单向函数将针对选择明文攻击的安全性(CPA安全性)升级为针对选择密文攻击(CCA安全性)的安全性。由此，我们得到了仅基于单向函数的CCA安全量子公钥加密。



## **14. Universal Robustness via Median Randomized Smoothing for Real-World Super-Resolution**

通过随机中位数平滑实现现实世界超分辨率的普遍鲁棒性 eess.IV

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14934v1) [paper-pdf](http://arxiv.org/pdf/2405.14934v1)

**Authors**: Zakariya Chaouai, Mohamed Tamaazousti

**Abstract**: Most of the recent literature on image Super-Resolution (SR) can be classified into two main approaches. The first one involves learning a corruption model tailored to a specific dataset, aiming to mimic the noise and corruption in low-resolution images, such as sensor noise. However, this approach is data-specific, tends to lack adaptability, and its accuracy diminishes when faced with unseen types of image corruptions. A second and more recent approach, referred to as Robust Super-Resolution (RSR), proposes to improve real-world SR by harnessing the generalization capabilities of a model by making it robust to adversarial attacks. To delve further into this second approach, our paper explores the universality of various methods for enhancing the robustness of deep learning SR models. In other words, we inquire: "Which robustness method exhibits the highest degree of adaptability when dealing with a wide range of adversarial attacks ?". Our extensive experimentation on both synthetic and real-world images empirically demonstrates that median randomized smoothing (MRS) is more general in terms of robustness compared to adversarial learning techniques, which tend to focus on specific types of attacks. Furthermore, as expected, we also illustrate that the proposed universal robust method enables the SR model to handle standard corruptions more effectively, such as blur and Gaussian noise, and notably, corruptions naturally present in real-world images. These results support the significance of shifting the paradigm in the development of real-world SR methods towards RSR, especially via MRS.

摘要: 最近关于图像超分辨率(SR)的大多数文献可以分为两种主要方法。第一个是学习针对特定数据集定制的损坏模型，旨在模拟低分辨率图像中的噪声和损坏，例如传感器噪声。然而，这种方法是特定于数据的，往往缺乏适应性，当面临不可见的图像损坏类型时，其准确性会降低。第二种更新的方法被称为稳健超分辨率(RSR)，它建议通过利用模型的泛化能力来提高真实世界的SR，使其对对手攻击具有健壮性。为了进一步研究第二种方法，我们探索了增强深度学习随机共振模型稳健性的各种方法的普适性。换句话说，我们的问题是：“在处理范围广泛的对抗性攻击时，哪种健壮性方法表现出最高的适应性？”我们在合成图像和真实图像上的广泛实验经验表明，与倾向于关注特定类型攻击的对抗性学习技术相比，中值随机平滑(MRS)在稳健性方面更具一般性。此外，正如预期的那样，我们还说明了所提出的通用稳健方法使SR模型能够更有效地处理标准的损坏，例如模糊和高斯噪声，特别是真实世界图像中自然存在的损坏。这些结果支持了在现实世界SR方法的发展中将范式转变为RSR的意义，特别是通过MRS。



## **15. Overcoming the Challenges of Batch Normalization in Federated Learning**

克服联邦学习中批量规范化的挑战 cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14670v1) [paper-pdf](http://arxiv.org/pdf/2405.14670v1)

**Authors**: Rachid Guerraoui, Rafael Pinot, Geovani Rizk, John Stephan, François Taiani

**Abstract**: Batch normalization has proven to be a very beneficial mechanism to accelerate the training and improve the accuracy of deep neural networks in centralized environments. Yet, the scheme faces significant challenges in federated learning, especially under high data heterogeneity. Essentially, the main challenges arise from external covariate shifts and inconsistent statistics across clients. We introduce in this paper Federated BatchNorm (FBN), a novel scheme that restores the benefits of batch normalization in federated learning. Essentially, FBN ensures that the batch normalization during training is consistent with what would be achieved in a centralized execution, hence preserving the distribution of the data, and providing running statistics that accurately approximate the global statistics. FBN thereby reduces the external covariate shift and matches the evaluation performance of the centralized setting. We also show that, with a slight increase in complexity, we can robustify FBN to mitigate erroneous statistics and potentially adversarial attacks.

摘要: 批量归一化已被证明是一种非常有益的机制，可以在集中式环境下加快训练速度，提高深度神经网络的精度。然而，该方案在联合学习方面面临着巨大的挑战，特别是在数据高度异构性的情况下。从本质上讲，主要挑战来自外部协变量变化和客户之间不一致的统计数据。本文介绍了联邦BatchNorm(FBN)算法，它恢复了联合学习中批量归一化的优点。本质上，FBN确保训练期间的批归一化与在集中执行中实现的一致，从而保持数据的分布，并提供准确接近全局统计的运行统计。因此，FBN减少了外部协变量漂移，并且与集中式设置的评估性能匹配。我们还表明，在复杂性略有增加的情况下，我们可以粗暴地使用FBN来减少错误的统计数据和潜在的对抗性攻击。



## **16. A New Formulation for Zeroth-Order Optimization of Adversarial EXEmples in Malware Detection**

恶意软件检测中对抗实例零阶优化的新公式 cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14519v1) [paper-pdf](http://arxiv.org/pdf/2405.14519v1)

**Authors**: Marco Rando, Luca Demetrio, Lorenzo Rosasco, Fabio Roli

**Abstract**: Machine learning malware detectors are vulnerable to adversarial EXEmples, i.e. carefully-crafted Windows programs tailored to evade detection. Unlike other adversarial problems, attacks in this context must be functionality-preserving, a constraint which is challenging to address. As a consequence heuristic algorithms are typically used, that inject new content, either randomly-picked or harvested from legitimate programs. In this paper, we show how learning malware detectors can be cast within a zeroth-order optimization framework which allows to incorporate functionality-preserving manipulations. This permits the deployment of sound and efficient gradient-free optimization algorithms, which come with theoretical guarantees and allow for minimal hyper-parameters tuning. As a by-product, we propose and study ZEXE, a novel zero-order attack against Windows malware detection. Compared to state-of-the-art techniques, ZEXE provides drastic improvement in the evasion rate, while reducing to less than one third the size of the injected content.

摘要: 机器学习恶意软件检测器容易受到敌意例子的攻击，即精心设计的Windows程序，旨在逃避检测。与其他对抗性问题不同，这种情况下的攻击必须保留功能，这是一个具有挑战性的限制。因此，通常使用启发式算法，注入新内容，无论是随机挑选的还是从合法程序中获取的。在这篇文章中，我们展示了如何在零阶优化框架内转换学习恶意软件检测器，该框架允许合并保留功能的操作。这允许部署合理和高效的无梯度优化算法，这些算法具有理论上的保证，并允许最小限度的超参数调整。作为副产品，我们提出并研究了一种针对Windows恶意软件检测的新型零序攻击方法ZEXE。与最先进的技术相比，ZEXE在逃逸率方面有了很大的改进，同时将注入内容的大小减少到不到三分之一。



## **17. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

SIFER：调查恶意软件检测管道的性能和稳健性 cs.CR

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14478v1) [paper-pdf](http://arxiv.org/pdf/2405.14478v1)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyse samples; and (iii) analysing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we propose SLIFER, a novel Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples resistance to analysis, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER leveraging content-injections attacks, and we show that, counter-intuitively, attacks are blocked more by YARA rules than dynamic analysis due to byte artifacts created while optimizing the adversarial strategy.

摘要: 作为数十年研究的结果，Windows恶意软件检测是通过大量技术实现的。然而，学术界--追求在检测率和低虚警方面的最佳表现--与现实世界场景的要求之间存在着持续的不匹配。特别是，学术界专注于在单个或集成模型中结合静态和动态分析，陷入了几个陷阱，如(I)触发动态分析而不考虑其所需的计算负担；(Ii)丢弃无法分析的样本；以及(Iii)分析针对敌意攻击的稳健性，而不考虑恶意软件检测器补充了更多的非机器学习组件。因此，在本文中，我们提出了一种新颖的Windows恶意软件检测流水线Slifer，它顺序地利用静态和动态分析，在一个模块触发警报时立即中断计算，仅在需要时才需要动态分析。与最新技术相反，我们调查了如何处理抗拒分析的样本，显示了它们对性能的影响程度，得出的结论是，最好将它们标记为合法，不要大幅增加错误警报。最后，我们利用内容注入攻击对Slifer进行了健壮性评估，并且我们表明，与直觉相反，由于优化对抗策略时产生的字节伪影，攻击更多地被Yara规则阻止而不是动态分析。



## **18. BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks**

BadPart：针对像素式回归任务的统一黑匣子对抗补丁攻击 cs.CV

Paper accepted at ICML 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.00924v2) [paper-pdf](http://arxiv.org/pdf/2404.00924v2)

**Authors**: Zhiyuan Cheng, Zhaoyi Liu, Tengda Guo, Shiwei Feng, Dongfang Liu, Mingjie Tang, Xiangyu Zhang

**Abstract**: Pixel-wise regression tasks (e.g., monocular depth estimation (MDE) and optical flow estimation (OFE)) have been widely involved in our daily life in applications like autonomous driving, augmented reality and video composition. Although certain applications are security-critical or bear societal significance, the adversarial robustness of such models are not sufficiently studied, especially in the black-box scenario. In this work, we introduce the first unified black-box adversarial patch attack framework against pixel-wise regression tasks, aiming to identify the vulnerabilities of these models under query-based black-box attacks. We propose a novel square-based adversarial patch optimization framework and employ probabilistic square sampling and score-based gradient estimation techniques to generate the patch effectively and efficiently, overcoming the scalability problem of previous black-box patch attacks. Our attack prototype, named BadPart, is evaluated on both MDE and OFE tasks, utilizing a total of 7 models. BadPart surpasses 3 baseline methods in terms of both attack performance and efficiency. We also apply BadPart on the Google online service for portrait depth estimation, causing 43.5% relative distance error with 50K queries. State-of-the-art (SOTA) countermeasures cannot defend our attack effectively.

摘要: 像素级回归任务(如单目深度估计(MDE)和光流估计(OFE))在自动驾驶、增强现实和视频合成等应用中广泛应用于我们的日常生活中。虽然某些应用是安全关键的或具有社会意义的，但这些模型的对抗健壮性没有得到充分的研究，特别是在黑盒场景中。在这项工作中，我们引入了第一个针对像素回归任务的统一黑盒对抗性补丁攻击框架，旨在识别这些模型在基于查询的黑盒攻击下的脆弱性。提出了一种新的基于平方的对抗性补丁优化框架，并利用概率平方采样和基于分数的梯度估计技术有效地生成了补丁，克服了以往黑盒补丁攻击的可扩展性问题。我们的攻击原型名为BadPart，在MDE和OFE任务上进行了评估，总共使用了7个模型。BadPart在攻击性能和效率方面都超过了3种基线方法。我们还将BadPart应用于Google在线服务上进行人像深度估计，在50K查询中导致了43.5%的相对距离误差。最先进的(SOTA)对策不能有效地防御我们的攻击。



## **19. Self-playing Adversarial Language Game Enhances LLM Reasoning**

自玩对抗语言游戏增强LLM推理 cs.CL

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.10642v2) [paper-pdf](http://arxiv.org/pdf/2404.10642v2)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by self-play in this adversarial language game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is at https://github.com/Linear95/SPAG.

摘要: 我们探索了在一个名为对抗性禁忌的两人对抗性语言游戏中，大语言模型(LLM)的自我发挥训练过程。在这个游戏中，攻击者和防御者围绕一个只有攻击者才能看到的目标单词进行交流。攻击者的目的是诱导防御者无意识地说出目标词，而防御者则试图从攻击者的话语中推断出目标词。要赢得这场比赛，双方都应该有足够的目标词知识和高级推理能力，以便在这种信息储备的对话中进行推理和表达。因此，我们好奇的是，在这场对抗性语言游戏(SPAG)中，LLMS的推理能力能否通过自我游戏进一步增强。带着这个目标，我们选择了几个开源的LLM，让每个LLM扮演攻击者的角色，并在广泛的目标词上扮演自己的防御者。通过对游戏结果的强化学习，我们观察到LLMS的性能在广泛的推理基准上一致提高。此外，迭代地采用这种自我发挥过程可以不断提升LLMS的推理能力。代码在https://github.com/Linear95/SPAG.



## **20. Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds**

Eidos：高效、不可感知的对抗性3D点云 cs.CV

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14210v1) [paper-pdf](http://arxiv.org/pdf/2405.14210v1)

**Authors**: Hanwei Zhang, Luo Cheng, Qisong He, Wei Huang, Renjue Li, Ronan Sicre, Xiaowei Huang, Holger Hermanns, Lijun Zhang

**Abstract**: Classification of 3D point clouds is a challenging machine learning (ML) task with important real-world applications in a spectrum from autonomous driving and robot-assisted surgery to earth observation from low orbit. As with other ML tasks, classification models are notoriously brittle in the presence of adversarial attacks. These are rooted in imperceptible changes to inputs with the effect that a seemingly well-trained model ends up misclassifying the input. This paper adds to the understanding of adversarial attacks by presenting Eidos, a framework providing Efficient Imperceptible aDversarial attacks on 3D pOint cloudS. Eidos supports a diverse set of imperceptibility metrics. It employs an iterative, two-step procedure to identify optimal adversarial examples, thereby enabling a runtime-imperceptibility trade-off. We provide empirical evidence relative to several popular 3D point cloud classification models and several established 3D attack methods, showing Eidos' superiority with respect to efficiency as well as imperceptibility.

摘要: 三维点云的分类是一项具有挑战性的机器学习(ML)任务，在从自动驾驶和机器人辅助手术到低轨道对地观测等一系列实际应用中具有重要的应用。与其他ML任务一样，分类模型在存在对抗性攻击时是出了名的脆弱。这些问题根源于对投入的潜移默化的改变，其结果是，一个看似训练有素的模型最终会错误地对投入进行分类。本文通过介绍EIDOS来加深对敌意攻击的理解，EIDOS是一种在3D点云上提供高效的隐形攻击的框架。Eidos支持一组不同的不可感知性指标。它使用迭代的两步过程来确定最佳对抗性示例，从而实现了运行时不可感知性的权衡。我们提供了与几种流行的三维点云分类模型和几种已建立的三维攻击方法相关的经验证据，表明了Eidos在效率和不可感知性方面的优势。



## **21. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

S-Eval：用于大型语言模型基准安全评估的自动和自适应测试生成 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14191v1) [paper-pdf](http://arxiv.org/pdf/2405.14191v1)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of a LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200, 000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.

摘要: 大型语言模型因其革命性的能力而获得了相当大的关注。然而，人们也越来越担心它们的安全影响，这使得在模型部署之前迫切需要对LLMS进行全面的安全评估。在这项工作中，我们提出了一种新的全面、多维、开放式的安全评价基准S-EVAL。S-EVAL的核心是一种新颖的基于LLM的测试提示自动生成和选择框架，该框架训练一名测试专家，结合一系列测试选择策略，自动构建用于安全评估的高质量测试用例集。这一过程自动化的关键是一种新颖的专家安全评论LLm Mc，它能够量化LLm响应的风险分数，并另外产生风险标签和解释。此外，生成过程还遵循了精心设计的四个不同级别的风险分类，涵盖了令人关注的全面和多维度的安全风险。在此基础上，我们系统地构建了一个新的大规模的低层管理系统安全评估基准，该基准由22万条评估提示组成，其中包括2万条基本风险提示(中文10000条，英文10000条)和来自10种流行的对抗性指令攻击的20万条相应的攻击提示。此外，考虑到LLM的快速演化和伴随的安全威胁，S-EVAL可以灵活配置和调整，以包括新的风险、攻击和模型。S-EVAL在20个流行和有代表性的低成本模型上进行了广泛的评估。结果证实，与现有基准相比，S-EVAL能够更好地反映和告知低成本机械的安全风险。我们还探讨了参数尺度、语言环境和解码参数对评估的影响，为评估LLMS的安全性提供了一种系统的方法。



## **22. Certified Robustness against Sparse Adversarial Perturbations via Data Localization**

通过数据本地化认证针对稀疏对抗扰动的鲁棒性 cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14176v1) [paper-pdf](http://arxiv.org/pdf/2405.14176v1)

**Authors**: Ambar Pal, René Vidal, Jeremias Sulam

**Abstract**: Recent work in adversarial robustness suggests that natural data distributions are localized, i.e., they place high probability in small volume regions of the input space, and that this property can be utilized for designing classifiers with improved robustness guarantees for $\ell_2$-bounded perturbations. Yet, it is still unclear if this observation holds true for more general metrics. In this work, we extend this theory to $\ell_0$-bounded adversarial perturbations, where the attacker can modify a few pixels of the image but is unrestricted in the magnitude of perturbation, and we show necessary and sufficient conditions for the existence of $\ell_0$-robust classifiers. Theoretical certification approaches in this regime essentially employ voting over a large ensemble of classifiers. Such procedures are combinatorial and expensive or require complicated certification techniques. In contrast, a simple classifier emerges from our theory, dubbed Box-NN, which naturally incorporates the geometry of the problem and improves upon the current state-of-the-art in certified robustness against sparse attacks for the MNIST and Fashion-MNIST datasets.

摘要: 最近在对抗性稳健性方面的工作表明，自然数据分布是局部化的，即它们将高概率放置在输入空间的小体积区域，并且这一性质可以用于设计具有更好的对$\ell_2有界扰动的稳健性保证的分类器。然而，目前仍不清楚这一观察结果是否适用于更普遍的指标。在这项工作中，我们将这一理论推广到$\ell_0$有界的对抗性扰动，其中攻击者可以修改图像的几个像素，但扰动的大小是不受限制的，我们给出了$\ell_0$-稳健分类器存在的充要条件。这一制度中的理论认证方法实质上是对大量分类器进行投票。这种程序既复杂又昂贵，或者需要复杂的认证技术。相反，从我们的理论中出现了一个简单的分类器，称为Box-NN，它自然地结合了问题的几何结构，并在MNIST和Fashion-MNIST数据集经过验证的针对稀疏攻击的健壮性方面改进了当前的最先进水平。



## **23. Towards Transferable Attacks Against Vision-LLMs in Autonomous Driving with Typography**

通过印刷术在自动驾驶中针对视觉LLM的可转移攻击 cs.CV

12 pages, 5 tables, 5 figures, work in progress

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14169v1) [paper-pdf](http://arxiv.org/pdf/2405.14169v1)

**Authors**: Nhat Chung, Sensen Gao, Tuan-Anh Vu, Jie Zhang, Aishan Liu, Yun Lin, Jin Song Dong, Qing Guo

**Abstract**: Vision-Large-Language-Models (Vision-LLMs) are increasingly being integrated into autonomous driving (AD) systems due to their advanced visual-language reasoning capabilities, targeting the perception, prediction, planning, and control mechanisms. However, Vision-LLMs have demonstrated susceptibilities against various types of adversarial attacks, which would compromise their reliability and safety. To further explore the risk in AD systems and the transferability of practical threats, we propose to leverage typographic attacks against AD systems relying on the decision-making capabilities of Vision-LLMs. Different from the few existing works developing general datasets of typographic attacks, this paper focuses on realistic traffic scenarios where these attacks can be deployed, on their potential effects on the decision-making autonomy, and on the practical ways in which these attacks can be physically presented. To achieve the above goals, we first propose a dataset-agnostic framework for automatically generating false answers that can mislead Vision-LLMs' reasoning. Then, we present a linguistic augmentation scheme that facilitates attacks at image-level and region-level reasoning, and we extend it with attack patterns against multiple reasoning tasks simultaneously. Based on these, we conduct a study on how these attacks can be realized in physical traffic scenarios. Through our empirical study, we evaluate the effectiveness, transferability, and realizability of typographic attacks in traffic scenes. Our findings demonstrate particular harmfulness of the typographic attacks against existing Vision-LLMs (e.g., LLaVA, Qwen-VL, VILA, and Imp), thereby raising community awareness of vulnerabilities when incorporating such models into AD systems. We will release our source code upon acceptance.

摘要: 视觉大语言模型(Vision-LLMS)由于其先进的视觉语言推理能力，针对感知、预测、规划和控制机制，越来越多地被集成到自动驾驶(AD)系统中。然而，Vision-LLM已经显示出对各种类型的对抗性攻击的敏感性，这将损害它们的可靠性和安全性。为了进一步探索AD系统中的风险和实际威胁的可转移性，我们建议依靠Vision-LLMS的决策能力来利用排版攻击AD系统。与现有开发排版攻击通用数据集的少数工作不同，本文关注的是可以部署这些攻击的现实交通场景，它们对决策自主性的潜在影响，以及这些攻击可以物理呈现的实际方式。为了实现上述目标，我们首先提出了一个与数据集无关的框架，用于自动生成可能误导Vision-LLMS推理的错误答案。在此基础上，提出了一种支持图像级和区域级推理攻击的语言扩充方案，并将其扩展为同时针对多个推理任务的攻击模式。在此基础上，对如何在物理流量场景下实现这些攻击进行了研究。通过我们的实证研究，我们评估了交通场景中排版攻击的有效性、可转移性和可实现性。我们的发现证明了针对现有Vision-LLM(例如LLaVA、Qwen-VL、Vila和IMP)的排版攻击的特殊危害性，从而提高了社区对将此类模型整合到AD系统中时的漏洞的认识。我们将在接受后公布我们的源代码。



## **24. Carefully Blending Adversarial Training and Purification Improves Adversarial Robustness**

仔细混合对抗训练和净化提高对抗稳健性 cs.CV

21 pages, 1 figure, 15 tables

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2306.06081v4) [paper-pdf](http://arxiv.org/pdf/2306.06081v4)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a synergistic robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the same adversarially-trained model, and an aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of strong adaptive attacks, across different image datasets, shows that CARSO is able to defend itself against adaptive end-to-end white-box attacks devised for stochastic defences. Paying a modest clean accuracy toll, our method improves by a significant margin the state-of-the-art for CIFAR-10, CIFAR-100, and TinyImageNet-200 $\ell_\infty$ robust classification accuracy against AutoAttack. Code, and instructions to obtain pre-trained models are available at https://github.com/emaballarin/CARSO .

摘要: 在这项工作中，我们提出了一种新的用于图像分类的对抗性防御机制-CASO-以协同增强鲁棒性的方式融合了对抗性训练和对抗性净化的范例。该方法建立在对抗性训练的分类器之上，并学习将其与潜在扰动输入相关联的内部表示映射到试探性干净重构的分布上。来自这种分布的多个样本被相同的对抗性训练模型分类，其输出的聚集最终构成感兴趣的稳健预测。基于不同图像数据集的强自适应攻击基准的实验评估表明，CARSO能够抵抗针对随机防御而设计的自适应端到端白盒攻击。付出适度的干净精度代价，我们的方法显著提高了CIFAR-10、CIFAR-100和TinyImageNet-200$\ell_\inty$相对于AutoAttack的稳健分类精度。代码，以及获取预先训练的模型的说明，请访问https://github.com/emaballarin/CARSO。



## **25. Breaking Free: How to Hack Safety Guardrails in Black-Box Diffusion Models!**

挣脱束缚：如何破解黑匣子扩散模型中的安全护栏！ cs.CV

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2402.04699v2) [paper-pdf](http://arxiv.org/pdf/2402.04699v2)

**Authors**: Shashank Kotyan, Po-Yuan Mao, Pin-Yu Chen, Danilo Vasconcellos Vargas

**Abstract**: Deep neural networks can be exploited using natural adversarial samples, which do not impact human perception. Current approaches often rely on deep neural networks' white-box nature to generate these adversarial samples or synthetically alter the distribution of adversarial samples compared to the training distribution. In contrast, we propose EvoSeed, a novel evolutionary strategy-based algorithmic framework for generating photo-realistic natural adversarial samples. Our EvoSeed framework uses auxiliary Conditional Diffusion and Classifier models to operate in a black-box setting. We employ CMA-ES to optimize the search for an initial seed vector, which, when processed by the Conditional Diffusion Model, results in the natural adversarial sample misclassified by the Classifier Model. Experiments show that generated adversarial images are of high image quality, raising concerns about generating harmful content bypassing safety classifiers. Our research opens new avenues to understanding the limitations of current safety mechanisms and the risk of plausible attacks against classifier systems using image generation. Project Website can be accessed at: https://shashankkotyan.github.io/EvoSeed.

摘要: 深度神经网络可以使用自然的对抗性样本来开发，这些样本不会影响人类的感知。目前的方法往往依赖于深度神经网络的白盒性质来生成这些对抗性样本，或者综合改变对抗性样本相对于训练分布的分布。相反，我们提出了EvoSeed，一个新的基于进化策略的算法框架，用于生成照片级的自然对抗性样本。我们的EvoSeed框架使用辅助的条件扩散和分类器模型在黑盒环境中运行。我们使用CMA-ES来优化初始种子向量的搜索，当该初始种子向量被条件扩散模型处理时，导致自然对抗性样本被分类器模型误分类。实验表明，生成的敌意图像具有高图像质量，这引发了人们对绕过安全分类器生成有害内容的担忧。我们的研究为理解当前安全机制的局限性以及使用图像生成对分类器系统进行可信攻击的风险开辟了新的途径。项目网站可访问：https://shashankkotyan.github.io/EvoSeed.



## **26. Nearly Tight Black-Box Auditing of Differentially Private Machine Learning**

差异私有机器学习的近乎严格的黑匣子审计 cs.CR

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14106v1) [paper-pdf](http://arxiv.org/pdf/2405.14106v1)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Emiliano De Cristofaro

**Abstract**: This paper presents a nearly tight audit of the Differentially Private Stochastic Gradient Descent (DP-SGD) algorithm in the black-box model. Our auditing procedure empirically estimates the privacy leakage from DP-SGD using membership inference attacks; unlike prior work, the estimates are appreciably close to the theoretical DP bounds. The main intuition is to craft worst-case initial model parameters, as DP-SGD's privacy analysis is agnostic to the choice of the initial model parameters. For models trained with theoretical $\varepsilon=10.0$ on MNIST and CIFAR-10, our auditing procedure yields empirical estimates of $7.21$ and $6.95$, respectively, on 1,000-record samples and $6.48$ and $4.96$ on the full datasets. By contrast, previous work achieved tight audits only in stronger (i.e., less realistic) white-box models that allow the adversary to access the model's inner parameters and insert arbitrary gradients. Our auditing procedure can be used to detect bugs and DP violations more easily and offers valuable insight into how the privacy analysis of DP-SGD can be further improved.

摘要: 本文对黑盒模型中的差分私有随机梯度下降(DP-SGD)算法进行了严格的审计。我们的审计过程使用成员推理攻击经验地估计了DP-SGD的隐私泄漏；与以前的工作不同，该估计相当接近理论DP界。主要的直觉是制作最坏情况的初始模型参数，因为DP-SGD的隐私分析与初始模型参数的选择无关。对于在MNIST和CIFAR-10上用理论上的$varepsilon=10.0$训练的模型，我们的审计程序在1,000个记录样本上产生的经验估计值分别为7.21美元和6.95美元，在完整数据集上产生的经验估计值分别为6.48美元和4.96美元。相比之下，以前的工作只在更强大(即不太现实)的白盒模型中实现了严格的审计，这些白盒模型允许对手访问模型的内部参数并插入任意渐变。我们的审计程序可用于更轻松地检测错误和DP违规，并为如何进一步改进DP-SGD的隐私分析提供宝贵的见解。



## **27. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

视觉感知推荐系统上的对抗项目推广 cs.IR

Accepted by TOIS 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2312.15826v4) [paper-pdf](http://arxiv.org/pdf/2312.15826v4)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Guanhua Ye, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.

摘要: 视觉感知推荐系统在视觉元素对用户潜在偏好的推断有重要作用的领域得到了广泛的应用。虽然加入视觉信息有望提高推荐的准确性和缓解冷启动问题，但必须指出的是，纳入物品图像可能会带来重大的安全挑战。一些已有的研究表明，物品提供者可以通过构建对抗性图像来操纵物品曝光率。然而，这些工作并不能揭示视觉感知推荐系统的真正弱点，因为(1)生成的敌意图像明显失真，使得人类很容易发现它们；(2)攻击的有效性在某些场景下是不一致的，甚至无效的。为了揭示视觉感知推荐系统在面对敌意图像时的真正弱点，提出了一种新的攻击方法--IPDGI(Item Promotion By Diffumation Generated Image)。具体地说，IPDGI使用引导扩散模型来生成敌意样本，旨在欺骗视觉感知的推荐系统。利用扩散模型精确模拟良性图像的分布，生成的对抗性图像与原始图像具有较高的保真度，保证了IPDGI的隐蔽性。为了验证我们提出的方法的有效性，我们在两个常用的电子商务推荐数据集(Amazon Beauty和Amazon Baby)上进行了广泛的实验，并使用几个典型的视觉感知推荐系统进行了实验。实验结果表明，我们的攻击方法在提升长尾(即不受欢迎)项的性能和生成对抗性图像的质量方面都有显著的提高。



## **28. Learning to Transform Dynamically for Better Adversarial Transferability**

学习动态转型以获得更好的对抗可移植性 cs.CV

accepted as a poster in CVPR 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14077v1) [paper-pdf](http://arxiv.org/pdf/2405.14077v1)

**Authors**: Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, Chenliang Xu

**Abstract**: Adversarial examples, crafted by adding perturbations imperceptible to humans, can deceive neural networks. Recent studies identify the adversarial transferability across various models, \textit{i.e.}, the cross-model attack ability of adversarial samples. To enhance such adversarial transferability, existing input transformation-based methods diversify input data with transformation augmentation. However, their effectiveness is limited by the finite number of available transformations. In our study, we introduce a novel approach named Learning to Transform (L2T). L2T increases the diversity of transformed images by selecting the optimal combination of operations from a pool of candidates, consequently improving adversarial transferability. We conceptualize the selection of optimal transformation combinations as a trajectory optimization problem and employ a reinforcement learning strategy to effectively solve the problem. Comprehensive experiments on the ImageNet dataset, as well as practical tests with Google Vision and GPT-4V, reveal that L2T surpasses current methodologies in enhancing adversarial transferability, thereby confirming its effectiveness and practical significance. The code is available at https://github.com/RongyiZhu/L2T.

摘要: 通过添加人类察觉不到的扰动而精心制作的对抗性例子可以欺骗神经网络。最近的研究发现了各种模型之间的对抗性转移，即对抗性样本的跨模型攻击能力。为了增强这种对抗性的可转移性，现有的基于输入变换的方法通过变换增强来使输入数据多样化。然而，它们的有效性受到可用变换数量有限的限制。在我们的研究中，我们引入了一种名为学习转化(L2T)的新方法。L2T通过从候选集合中选择最优的操作组合来增加变换图像的多样性，从而提高了对抗性转移。我们将最优变换组合的选择概念化为一个轨迹优化问题，并采用强化学习策略来有效地解决该问题。在ImageNet数据集上的综合实验以及与Google Vision和GPT-4V的实际测试表明，L2T在增强对抗性可转移性方面优于现有方法，从而证实了其有效性和现实意义。代码可在https://github.com/RongyiZhu/L2T.上获得



## **29. Remote Keylogging Attacks in Multi-user VR Applications**

多用户VR应用程序中的远程键盘记录攻击 cs.CR

Accepted for Usenix 2024

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14036v1) [paper-pdf](http://arxiv.org/pdf/2405.14036v1)

**Authors**: Zihao Su, Kunlin Cai, Reuben Beeler, Lukas Dresel, Allan Garcia, Ilya Grishchenko, Yuan Tian, Christopher Kruegel, Giovanni Vigna

**Abstract**: As Virtual Reality (VR) applications grow in popularity, they have bridged distances and brought users closer together. However, with this growth, there have been increasing concerns about security and privacy, especially related to the motion data used to create immersive experiences. In this study, we highlight a significant security threat in multi-user VR applications, which are applications that allow multiple users to interact with each other in the same virtual space. Specifically, we propose a remote attack that utilizes the avatar rendering information collected from an adversary's game clients to extract user-typed secrets like credit card information, passwords, or private conversations. We do this by (1) extracting motion data from network packets, and (2) mapping motion data to keystroke entries. We conducted a user study to verify the attack's effectiveness, in which our attack successfully inferred 97.62% of the keystrokes. Besides, we performed an additional experiment to underline that our attack is practical, confirming its effectiveness even when (1) there are multiple users in a room, and (2) the attacker cannot see the victims. Moreover, we replicated our proposed attack on four applications to demonstrate the generalizability of the attack. These results underscore the severity of the vulnerability and its potential impact on millions of VR social platform users.

摘要: 随着虚拟现实(VR)应用越来越受欢迎，它们弥合了距离，拉近了用户之间的距离。然而，随着这种增长，人们对安全和隐私的担忧也越来越多，特别是与用于创建身临其境体验的运动数据有关。在这项研究中，我们强调了多用户VR应用中的一个重大安全威胁，即允许多个用户在同一虚拟空间中相互交互的应用。具体地说，我们提出了一种远程攻击，它利用从对手的游戏客户端收集的化身渲染信息来提取用户键入的秘密，如信用卡信息、密码或私人对话。我们通过(1)从网络分组中提取运动数据，以及(2)将运动数据映射到击键条目来实现这一点。我们进行了用户研究来验证攻击的有效性，其中我们的攻击成功推断了97.62%的击键。此外，我们还执行了一个额外的实验，以强调我们的攻击是实用的，即使在(1)一个房间有多个用户，以及(2)攻击者看不到受害者的情况下，也证实了它的有效性。此外，我们在四个应用程序上复制了我们提出的攻击，以证明该攻击的泛化能力。这些结果突显了该漏洞的严重性及其对数百万VR社交平台用户的潜在影响。



## **30. Adversarial Training of Two-Layer Polynomial and ReLU Activation Networks via Convex Optimization**

通过凸优化进行两层多边形和ReLU激活网络的对抗训练 cs.LG

6 pages, 4 figures

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14033v1) [paper-pdf](http://arxiv.org/pdf/2405.14033v1)

**Authors**: Daniel Kuelbs, Sanjay Lall, Mert Pilanci

**Abstract**: Training neural networks which are robust to adversarial attacks remains an important problem in deep learning, especially as heavily overparameterized models are adopted in safety-critical settings. Drawing from recent work which reformulates the training problems for two-layer ReLU and polynomial activation networks as convex programs, we devise a convex semidefinite program (SDP) for adversarial training of polynomial activation networks via the S-procedure. We also derive a convex SDP to compute the minimum distance from a correctly classified example to the decision boundary of a polynomial activation network. Adversarial training for two-layer ReLU activation networks has been explored in the literature, but, in contrast to prior work, we present a scalable approach which is compatible with standard machine libraries and GPU acceleration. The adversarial training SDP for polynomial activation networks leads to large increases in robust test accuracy against $\ell^\infty$ attacks on the Breast Cancer Wisconsin dataset from the UCI Machine Learning Repository. For two-layer ReLU networks, we leverage our scalable implementation to retrain the final two fully connected layers of a Pre-Activation ResNet-18 model on the CIFAR-10 dataset. Our 'robustified' model achieves higher clean and robust test accuracies than the same architecture trained with sharpness-aware minimization.

摘要: 训练对敌意攻击稳健的神经网络仍然是深度学习中的一个重要问题，特别是在安全关键环境中采用严重过度参数模型的情况下。借鉴最近将两层RELU和多项式激活网络的训练问题转化为凸规划的工作，我们设计了一个用于多项式激活网络对抗训练的凸半定规划(SDP)，该规划采用S过程。我们还推导了一个凸SDP来计算一个正确分类的例子到多项式激活网络的决策边界的最小距离。两层REU激活网络的对抗性训练已经在文献中得到了探索，但与以前的工作不同的是，我们提出了一种与标准机器库和GPU加速兼容的可扩展方法。多项式激活网络的对抗性训练SDP导致了针对来自UCI机器学习储存库的威斯康星州乳腺癌数据集的$\ell^\inty$攻击的稳健测试精度的大幅提高。对于两层RELU网络，我们利用我们的可扩展实施在CIFAR-10数据集上重新训练预激活ResNet-18模型的最后两个完全连接的层。与采用锐度感知最小化训练的相同架构相比，我们的“强健”模型实现了更高的清洁和健壮的测试精度。



## **31. WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response**

WordGame：通过查询和响应中的同时混淆高效且有效的LLM越狱 cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14023v1) [paper-pdf](http://arxiv.org/pdf/2405.14023v1)

**Authors**: Tianrong Zhang, Bochuan Cao, Yuanpu Cao, Lu Lin, Prasenjit Mitra, Jinghui Chen

**Abstract**: The recent breakthrough in large language models (LLMs) such as ChatGPT has revolutionized production processes at an unprecedented pace. Alongside this progress also comes mounting concerns about LLMs' susceptibility to jailbreaking attacks, which leads to the generation of harmful or unsafe content. While safety alignment measures have been implemented in LLMs to mitigate existing jailbreak attempts and force them to become increasingly complicated, it is still far from perfect. In this paper, we analyze the common pattern of the current safety alignment and show that it is possible to exploit such patterns for jailbreaking attacks by simultaneous obfuscation in queries and responses. Specifically, we propose WordGame attack, which replaces malicious words with word games to break down the adversarial intent of a query and encourage benign content regarding the games to precede the anticipated harmful content in the response, creating a context that is hardly covered by any corpus used for safety alignment. Extensive experiments demonstrate that WordGame attack can break the guardrails of the current leading proprietary and open-source LLMs, including the latest Claude-3, GPT-4, and Llama-3 models. Further ablation studies on such simultaneous obfuscation in query and response provide evidence of the merits of the attack strategy beyond an individual attack.

摘要: 最近ChatGPT等大型语言模型(LLM)的突破以前所未有的速度彻底改变了生产流程。在取得这一进展的同时，人们也越来越担心LLMS容易受到越狱攻击，这会导致产生有害或不安全的内容。虽然LLMS已经实施了安全调整措施，以减轻现有的越狱企图，并迫使它们变得越来越复杂，但它仍然远未达到完美。在本文中，我们分析了当前安全对齐的常见模式，并证明了通过在查询和响应中同时混淆来利用这种模式来进行越狱攻击是可能的。具体地说，我们提出了文字游戏攻击，它用文字游戏取代恶意单词，以打破查询的敌对意图，并鼓励与游戏有关的良性内容在响应中位于预期的有害内容之前，从而创造出任何用于安全对齐的语料库几乎无法覆盖的上下文。大量实验表明，文字游戏攻击可以打破目前领先的专有和开源LLM的屏障，包括最新的Claude-3、GPT-4和Llama-3型号。对查询和响应中的这种同时混淆的进一步消融研究提供了除单个攻击之外的攻击策略的优点的证据。



## **32. LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate**

LookHere：具有定向注意力的视觉变形者概括和推断 cs.CV

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13985v1) [paper-pdf](http://arxiv.org/pdf/2405.13985v1)

**Authors**: Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green

**Abstract**: High-resolution images offer more information about scenes that can improve model accuracy. However, the dominant model architecture in computer vision, the vision transformer (ViT), cannot effectively leverage larger images without finetuning -- ViTs poorly extrapolate to more patches at test time, although transformers offer sequence length flexibility. We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating.   We propose a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks. Our novel method, called LookHere, provides translation-equivariance, ensures attention head diversity, and limits the distribution shift that attention heads face when extrapolating. We demonstrate that LookHere improves performance on classification (avg. 1.6%), against adversarial attack (avg. 5.4%), and decreases calibration error (avg. 1.5%) -- on ImageNet without extrapolation. With extrapolation, LookHere outperforms the current SoTA position encoding method, 2D-RoPE, by 21.7% on ImageNet when trained at $224^2$ px and tested at $1024^2$ px. Additionally, we release a high-resolution test set to improve the evaluation of high-resolution image classifiers, called ImageNet-HR.

摘要: 高分辨率图像提供了有关场景的更多信息，可以提高模型精度。然而，计算机视觉中占主导地位的模型体系结构视觉转换器(VIT)在没有精细调整的情况下无法有效地利用更大的图像-VIT在测试时很难外推到更多的补丁，尽管转换器提供了序列长度的灵活性。我们将这一缺陷归因于目前的补丁位置编码方法，这些方法在外推时会产生分布偏移。我们提出了一种替代普通VITS的位置编码的方法，它使用2D注意掩码将注意力头部限制在指向不同方向的固定视野中。我们的新方法LookHere提供了平移等差性，确保了注意力头部的多样性，并限制了注意力头部在外推时面临的分布变化。我们证明LookHere提高了分类性能(平均1.6%)，抗对手攻击(Avg.5.4%)，降低了校准误差(平均1.5%)--在ImageNet上，没有外推。通过外推，LookHere在ImageNet上以$224^2$px进行训练并以$1024^2$px进行测试时，在ImageNet上的性能比当前的SOTA位置编码方法2D-ROPE高21.7%。此外，我们还发布了一个高分辨率测试集来改进高分辨率图像分类器的评估，称为ImageNet-HR。



## **33. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

对抗攻击下的不确定性校准认证 cs.LG

11 pages main paper, appendix included

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13922v1) [paper-pdf](http://arxiv.org/pdf/2405.13922v1)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.

摘要: 由于众所周知，神经分类器对改变其准确性的对抗性扰动敏感，因此\textit{认证方法}的开发是为了提供可证明的保证其预测对此类扰动的不敏感性。此外，在安全关键应用中，分类器置信度的频率主义解释（也称为模型校准）可能至关重要。该属性可以通过Brier评分或预期的校准误差来测量。我们表明，攻击可能会严重损害校准，因此建议将经过认证的校准作为对抗性扰动下校准的最坏情况界限。具体来说，我们通过对预期校准误差求解混合整数程序来产生Brier分数的分析界限和近似界限。最后，我们提出了新颖的校准攻击，并演示了它们如何通过\textit{对抗校准训练}来改进模型校准。



## **34. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2305.17000v4) [paper-pdf](http://arxiv.org/pdf/2305.17000v4)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **35. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

L-AutoDA：利用大型语言模型进行自动化基于决策的对抗性攻击 cs.CR

Camera ready version for GECCO'24 workshop

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2401.15335v2) [paper-pdf](http://arxiv.org/pdf/2401.15335v2)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.

摘要: 在快速发展的机器学习领域，敌意攻击对模型的健壮性和安全性提出了重大挑战。基于决策的攻击，只需要对模型的决策进行反馈，而不需要详细的概率或分数，特别隐蔽，很难防御。本文介绍了L-AUTODA(基于大语言模型的自动决策对抗性攻击)，它是一种利用大语言模型的生成能力来自动化攻击设计的新方法。通过在进化框架中迭代地与LLM交互，L自动DA无需太多人力即可自动高效地设计竞争攻击算法。我们在CIFAR-10数据集上验证了L-AutoDA的有效性，在成功率和计算效率上都比基线方法有了显著的提高。我们的发现强调了语言模型作为对抗性攻击生成工具的潜力，并强调了开发健壮的人工智能系统的新途径。



## **36. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

ALI-DPFL：具有自适应本地迭代的差异化私人联邦学习 cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2308.10457v9) [paper-pdf](http://arxiv.org/pdf/2308.10457v9)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/cheng-t/ALI-DPFL.

摘要: 联合学习(FL)是一种分布式机器学习技术，通过共享训练参数而不是原始数据，允许在多个设备或组织之间进行模型训练。然而，攻击者仍然可以通过对这些训练参数的推理攻击(例如差异攻击)来推断个人信息。因此，差分隐私(DP)被广泛应用于FL中以防止此类攻击。我们考虑在资源受限的情况下进行不同的私有联合学习，其中隐私预算和通信回合都受到限制。通过对收敛的理论分析，我们可以找到任意两个连续全局更新之间客户端的最优局部DPSGD迭代次数。在此基础上，设计了一种基于自适应局部迭代的差分私有联邦学习算法(ALI-DPFL)。我们在MNIST、FashionMNIST和Cifar10数据集上测试了我们的算法，并在资源受限的情况下展示了比以前的工作更好的性能。代码可在https://github.com/cheng-t/ALI-DPFL.上找到



## **37. Adversarial Training via Adaptive Knowledge Amalgamation of an Ensemble of Teachers**

通过教师群体的适应性知识融合进行对抗性培训 cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13324v1) [paper-pdf](http://arxiv.org/pdf/2405.13324v1)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Adversarial training (AT) is a popular method for training robust deep neural networks (DNNs) against adversarial attacks. Yet, AT suffers from two shortcomings: (i) the robustness of DNNs trained by AT is highly intertwined with the size of the DNNs, posing challenges in achieving robustness in smaller models; and (ii) the adversarial samples employed during the AT process exhibit poor generalization, leaving DNNs vulnerable to unforeseen attack types. To address these dual challenges, this paper introduces adversarial training via adaptive knowledge amalgamation of an ensemble of teachers (AT-AKA). In particular, we generate a diverse set of adversarial samples as the inputs to an ensemble of teachers; and then, we adaptively amalgamate the logtis of these teachers to train a generalized-robust student. Through comprehensive experiments, we illustrate the superior efficacy of AT-AKA over existing AT methods and adversarial robustness distillation techniques against cutting-edge attacks, including AutoAttack.

摘要: 对抗性训练(AT)是训练稳健的深层神经网络(DNN)抵抗敌意攻击的一种常用方法。然而，AT存在两个缺点：(I)由AT训练的DNN的健壮性与DNN的大小高度交织在一起，这给在较小模型中实现健壮性带来了挑战；(Ii)在AT过程中使用的敌意样本表现出较差的泛化能力，使得DNN容易受到不可预见的攻击类型的影响。为了应对这些双重挑战，本文引入了教师自适应知识融合的对抗性训练(AT-AKA)。特别是，我们生成了一组不同的对抗性样本作为教师集成的输入，然后，我们自适应地合并这些教师的逻辑，以训练一个普遍健壮的学生。通过综合实验，证明了AT-AKA算法对包括AutoAttack在内的前沿攻击具有优于现有的AT方法和对手健壮性蒸馏技术的有效性。



## **38. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

无框模型水印容易受到黑匣子删除攻击 cs.CV

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.09863v2) [paper-pdf](http://arxiv.org/pdf/2405.09863v2)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.

摘要: 无盒模型水印是一种新兴的保护深度学习模型知识产权的技术，尤其是用于低层图像处理任务的模型。已有的工作在几个方面验证和改进了它的有效性。然而，在本文中，我们揭示了无盒模型水印容易受到移除攻击，即使在真实世界的威胁模型下，受保护的模型和水印抽取器都在黑盒中。在此背景下，我们开展了三个方面的研究。1)我们开发了一种萃取器-梯度引导(EGG)去除器，并在仅使用RELU激活的情况下展示了其有效性。2)更一般地，对于未知的提取者，我们利用对抗性攻击，并基于估计的梯度来设计鸡蛋去除器。3)在抽取器不可访问的最严格条件下，基于一组私有代理模型设计了一个可转移的抽取器。在所有情况下，所提出的去除器都可以在保持处理图像质量的情况下成功地去除嵌入的水印，并且我们还证明了鸡蛋去除器甚至可以替换水印。大量的实验结果验证了所提出的攻击方法的有效性和泛化能力，揭示了现有去盒方法的弱点，需要进一步研究。



## **39. Cloud-based XAI Services for Assessing Open Repository Models Under Adversarial Attacks**

基于云的XAI服务，用于评估对抗性攻击下的开放存储库模型 cs.CR

Accepted by IEEE International Conference on Software Services  Engineering (SSE) 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2401.12261v3) [paper-pdf](http://arxiv.org/pdf/2401.12261v3)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: The opacity of AI models necessitates both validation and evaluation before their integration into services. To investigate these models, explainable AI (XAI) employs methods that elucidate the relationship between input features and output predictions. The operations of XAI extend beyond the execution of a single algorithm, involving a series of activities that include preprocessing data, adjusting XAI to align with model parameters, invoking the model to generate predictions, and summarizing the XAI results. Adversarial attacks are well-known threats that aim to mislead AI models. The assessment complexity, especially for XAI, increases when open-source AI models are subject to adversarial attacks, due to various combinations. To automate the numerous entities and tasks involved in XAI-based assessments, we propose a cloud-based service framework that encapsulates computing components as microservices and organizes assessment tasks into pipelines. The current XAI tools are not inherently service-oriented. This framework also integrates open XAI tool libraries as part of the pipeline composition. We demonstrate the application of XAI services for assessing five quality attributes of AI models: (1) computational cost, (2) performance, (3) robustness, (4) explanation deviation, and (5) explanation resilience across computer vision and tabular cases. The service framework generates aggregated analysis that showcases the quality attributes for more than a hundred combination scenarios.

摘要: 人工智能模型的不透明性需要在将其集成到服务中之前进行验证和评估。为了研究这些模型，可解释人工智能(XAI)使用了一些方法来阐明输入特征和输出预测之间的关系。XAI的操作超出了单一算法的执行范围，涉及一系列活动，包括数据预处理、调整XAI以与模型参数保持一致、调用模型以生成预测，以及汇总XAI结果。对抗性攻击是众所周知的旨在误导人工智能模型的威胁。当开源AI模型由于各种组合而受到对抗性攻击时，评估的复杂性，特别是对XAI来说，会增加。为了自动化基于XAI的评估中涉及的众多实体和任务，我们提出了一个基于云的服务框架，该框架将计算组件封装为微服务，并将评估任务组织到管道中。当前的XAI工具本身并不是面向服务的。该框架还集成了开放的XAI工具库，作为管道组合的一部分。我们展示了XAI服务在评估人工智能模型的五个质量属性中的应用：(1)计算成本，(2)性能，(3)稳健性，(4)解释偏差，(5)跨计算机视觉和表格案例的解释弹性。服务框架生成聚合分析，展示一百多个组合场景的质量属性。



## **40. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不精确的遗忘需要更仔细的评估，以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.01218v3) [paper-pdf](http://arxiv.org/pdf/2403.01218v3)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their "U-MIA" counterparts). We propose a categorization of existing U-MIAs into "population U-MIAs", where the same attacker is instantiated for all examples, and "per-example U-MIAs", where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高昂成本使得开发忘却学习的技术变得越来越受欢迎。这些技术寻求消除训练示例的影响，而不必从头开始重新训练模型。直观地说，一旦模型取消学习，与该模型交互的对手应该不再能够判断未学习的示例是否包括在该模型的训练集中。在隐私文献中，这被称为成员关系推断。在这项工作中，我们讨论了成员关系推理攻击(MIA)对遗忘环境的适应(导致它们的U-MIA对应)。我们建议将现有的U-MIA分类为“群体U-MIA”，其中针对所有示例实例化相同的攻击者，以及“每示例U-MIA”，其中针对每个示例实例化一个专门的攻击者。我们表明，后一类，其中攻击者根据每个被攻击的例子定制其成员预测，明显更强。事实上，我们的结果表明，遗忘文献中常用的U-MIA高估了现有遗忘技术在视觉和语言模型上提供的隐私保护。我们的调查显示，不同示例对每个示例的U-MIA的脆弱性存在很大差异。事实上，几种忘记算法降低了我们希望忘记的一些(但不是所有)示例的脆弱性，但代价是增加了其他示例的脆弱性。值得注意的是，我们发现，由于遗忘，其余训练样本的隐私保护可能会恶化。我们还讨论了使用现有的遗忘方案平等地保护所有例子的基本困难，因为例子被遗忘的比率不同。我们证明，根据不同的例子调整遗忘停止标准的天真尝试无法缓解这些问题。



## **41. Adversarial Attacks and Defenses in Automated Control Systems: A Comprehensive Benchmark**

自动控制系统中的对抗性攻击和防御：全面的基准 cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.13502v3) [paper-pdf](http://arxiv.org/pdf/2403.13502v3)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.

摘要: 将机器学习集成到自动化控制系统（ACS）中，增强了工业过程管理中的决策。这些技术在工业中广泛采用的局限性之一是神经网络容易受到对抗攻击。本研究探索了使用田纳西州伊士曼Process数据集在ACS中部署深度学习模型进行故障诊断的威胁。通过评估具有不同架构的三个神经网络，我们将它们置于六种类型的对抗攻击中，并探索五种不同的防御方法。我们的结果凸显了模型对对抗样本的强烈脆弱性以及防御策略的不同有效性。我们还提出了一种通过结合多种防御方法的新型保护方法，并证明了其有效性。这项研究为确保ACS内的机器学习提供了多项见解，确保工业流程中的稳健故障诊断。



## **42. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

重新思考面部识别系统的漏洞：从实践的角度 cs.CR

19 pages

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12786v1) [paper-pdf](http://arxiv.org/pdf/2405.12786v1)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.

摘要: 人脸识别系统(FRS)越来越多地集成到包括监控和用户身份验证在内的关键应用中，突显了它们在现代安全系统中的关键作用。最近的研究发现，FRS对对抗性攻击(例如，对抗性补丁攻击)和后门攻击(例如，训练数据中毒)的脆弱性，引起了人们对其可靠性和可信性的严重担忧。以往的研究主要集中于传统的对抗性攻击或后门攻击，忽略了此类威胁的资源密集型或特权操纵性，从而限制了它们的实用通用性、隐蔽性、普遍性和健壮性。相应地，在本文中，我们通过用户研究和初步探索，深入研究了FRS的固有漏洞。通过利用这些漏洞，我们确定了一种新型的攻击，即面部识别后门攻击，称为FIBA，它揭示了对FRS的一个潜在的更具破坏性的威胁：注册阶段的后门攻击。FIBA绕过了传统攻击的限制，允许任何使用特定触发器的攻击者绕过这些系统，从而实现广泛的破坏。这意味着在将单个有毒示例插入数据库后，相应的触发器将成为任何攻击者欺骗FRS的通用密钥。该策略实质上是通过在注册阶段发起攻击来挑战传统攻击，通过毒化特征数据库而不是训练数据来极大地改变威胁格局。



## **43. Generative AI and Large Language Models for Cyber Security: All Insights You Need**

网络安全的生成性人工智能和大型语言模型：您需要的所有见解 cs.CR

50 pages, 8 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12750v1) [paper-pdf](http://arxiv.org/pdf/2405.12750v1)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.

摘要: 本文通过生成式人工智能和大型语言模型(LLMS)对网络安全的未来进行了全面的回顾。我们探索了LLM在不同领域的应用，包括硬件设计安全、入侵检测、软件工程、设计验证、网络威胁情报、恶意软件检测和网络钓鱼检测。我们概述了LLM的演化和现状，重点介绍了GPT-4、GPT-3.5、Mixtral-8x7B、BERT、Falcon2和Llama等模型的进展。我们的分析扩展到LLM漏洞，如快速注入、不安全的输出处理、数据中毒、DDoS攻击和敌意指令。我们深入研究缓解策略以保护这些模型，提供对潜在攻击场景和预防技术的全面了解。此外，我们评估了42个LLM模型在网络安全知识和硬件安全方面的性能，突出了它们的优势和劣势。我们为LLM培训和测试彻底评估网络安全数据集，涵盖从数据创建到使用的整个生命周期，并为未来的研究确定差距。此外，我们还回顾了利用LLMS的新策略，包括半二次量化(HQQ)、带人反馈的强化学习(RLHF)、直接偏好优化(DPO)、量化低阶适配器(QLoRA)和检索增强生成(RAG)。这些见解旨在增强实时网络安全防御，并提高LLM应用程序在威胁检测和响应方面的复杂性。我们的论文为将低成本管理系统整合到未来的网络安全框架中提供了一个基础性的理解和战略方向，强调创新和稳健的模型部署，以防范不断演变的网络威胁。



## **44. A GAN-Based Data Poisoning Attack Against Federated Learning Systems and Its Countermeasure**

针对联邦学习系统的基于GAN的数据中毒攻击及其对策 cs.CR

18 pages, 16 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.11440v2) [paper-pdf](http://arxiv.org/pdf/2405.11440v2)

**Authors**: Wei Sun, Bo Gao, Ke Xiong, Yuwei Wang

**Abstract**: As a distributed machine learning paradigm, federated learning (FL) is collaboratively carried out on privately owned datasets but without direct data access. Although the original intention is to allay data privacy concerns, "available but not visible" data in FL potentially brings new security threats, particularly poisoning attacks that target such "not visible" local data. Initial attempts have been made to conduct data poisoning attacks against FL systems, but cannot be fully successful due to their high chance of causing statistical anomalies. To unleash the potential for truly "invisible" attacks and build a more deterrent threat model, in this paper, a new data poisoning attack model named VagueGAN is proposed, which can generate seemingly legitimate but noisy poisoned data by untraditionally taking advantage of generative adversarial network (GAN) variants. Capable of manipulating the quality of poisoned data on demand, VagueGAN enables to trade-off attack effectiveness and stealthiness. Furthermore, a cost-effective countermeasure named Model Consistency-Based Defense (MCD) is proposed to identify GAN-poisoned data or models after finding out the consistency of GAN outputs. Extensive experiments on multiple datasets indicate that our attack method is generally much more stealthy as well as more effective in degrading FL performance with low complexity. Our defense method is also shown to be more competent in identifying GAN-poisoned data or models. The source codes are publicly available at \href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.

摘要: 作为一种分布式机器学习范式，联合学习(FL)是在私有数据集上协作进行的，但不需要直接访问数据。虽然初衷是为了缓解数据隐私问题，但FL中的“可用但不可见”数据可能会带来新的安全威胁，特别是针对此类“不可见”本地数据的中毒攻击。已经进行了针对FL系统的数据中毒攻击的初步尝试，但由于造成统计异常的可能性很高，因此不能完全成功。为了释放真正隐形攻击的可能性，建立更具威慑力的威胁模型，提出了一种新的数据中毒攻击模型VagueGAN，该模型通过非传统地利用生成性对手网络(GAN)变体来生成看似合法但含有噪声的有毒数据。VagueGAN能够按需操纵有毒数据的质量，从而能够在攻击效率和隐蔽性之间进行权衡。此外，还提出了一种基于模型一致性防御(MCD)的高性价比对策，用于在发现GaN输出的一致性之后识别GaN中毒数据或模型。在多个数据集上的大量实验表明，我们的攻击方法通常更隐蔽，并且在降低复杂度的情况下更有效地降低了FL性能。我们的防御方法也被证明在识别GaN中毒数据或模型方面更有能力。源代码可在\href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.上公开获取



## **45. How to Train a Backdoor-Robust Model on a Poisoned Dataset without Auxiliary Data?**

如何在没有辅助数据的情况下在中毒数据集中训练后门稳健模型？ cs.CR

13 pages, under review

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12719v1) [paper-pdf](http://arxiv.org/pdf/2405.12719v1)

**Authors**: Yuwen Pu, Jiahao Chen, Chunyi Zhou, Zhou Feng, Qingming Li, Chunqiang Hu, Shouling Ji

**Abstract**: Backdoor attacks have attracted wide attention from academia and industry due to their great security threat to deep neural networks (DNN). Most of the existing methods propose to conduct backdoor attacks by poisoning the training dataset with different strategies, so it's critical to identify the poisoned samples and then train a clean model on the unreliable dataset in the context of defending backdoor attacks. Although numerous backdoor countermeasure researches are proposed, their inherent weaknesses render them limited in practical scenarios, such as the requirement of enough clean samples, unstable defense performance under various attack conditions, poor defense performance against adaptive attacks, and so on.Therefore, in this paper, we are committed to overcome the above limitations and propose a more practical backdoor defense method. Concretely, we first explore the inherent relationship between the potential perturbations and the backdoor trigger, and the theoretical analysis and experimental results demonstrate that the poisoned samples perform more robustness to perturbation than the clean ones. Then, based on our key explorations, we introduce AdvrBD, an Adversarial perturbation-based and robust Backdoor Defense framework, which can effectively identify the poisoned samples and train a clean model on the poisoned dataset. Constructively, our AdvrBD eliminates the requirement for any clean samples or knowledge about the poisoned dataset (e.g., poisoning ratio), which significantly improves the practicality in real-world scenarios.

摘要: 后门攻击因其对深度神经网络(DNN)的巨大安全威胁而受到学术界和工业界的广泛关注。现有的大多数方法都是通过对训练数据集使用不同的策略进行中毒来进行后门攻击，因此在防御后门攻击的背景下，识别中毒样本并在不可靠的数据集上训练一个干净的模型是至关重要的。虽然已经提出了大量的后门对抗研究，但其固有的缺陷使其在实际场景中受到限制，如需要足够的清洁样本，在各种攻击条件下的防御性能不稳定，对自适应攻击的防御性能较差等，因此，本文致力于克服上述局限性，提出一种更实用的后门防御方法。具体地说，我们首先探讨了潜在扰动和后门触发之间的内在联系，理论分析和实验结果表明，中毒样本比干净样本对扰动具有更强的鲁棒性。然后，在重点探索的基础上，提出了一种基于对抗性扰动的健壮后门防御框架AdvrBD，它可以有效地识别有毒样本，并在有毒数据集上训练一个干净的模型。建设性地说，我们的AdvrBD不需要任何干净的样本或关于有毒数据集的知识(例如，投毒率)，这显著提高了现实世界场景中的实用性。



## **46. Robust Classification via a Single Diffusion Model**

通过单一扩散模型的稳健分类 cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2305.15241v2) [paper-pdf](http://arxiv.org/pdf/2305.15241v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Diffusion models have been applied to improve adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, this paper proposes Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. RDC first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood estimated by the diffusion model through Bayes' theorem. To further reduce the computational cost, we propose a new diffusion backbone called multi-head diffusion and develop efficient sampling strategies. As RDC does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $75.67\%$ robust accuracy against various $\ell_\infty$ norm-bounded adaptive attacks with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+4.77\%$. The results highlight the potential of generative classifiers by employing pre-trained diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers. Code is available at \url{https://github.com/huanranchen/DiffusionClassifier}.

摘要: 扩散模型已被应用于通过净化对抗性噪声或生成用于对抗性训练的真实数据来提高图像分类器的对抗性鲁棒性。然而，基于扩散的净化方法可以通过更强的自适应攻击来规避，而对抗性训练在看不见的威胁下表现不佳，显示出这些方法不可避免的局限性。为了更好地利用扩散模型的表达能力，提出了稳健扩散分类器(RDC)，它是一种生成式分类器，由预先训练的扩散模型构造而成，具有相反的鲁棒性。RDC首先最大化给定输入的数据似然，然后利用扩散模型通过贝叶斯定理估计的条件似然来预测优化输入的类别概率。为了进一步降低计算成本，我们提出了一种新的扩散骨干，称为多头扩散，并开发了高效的采样策略。由于RDC不需要关于特定对手攻击的培训，我们证明了防御多个看不见的威胁更具普遍性。特别是，在CIFAR-10上，RDC对各种有界自适应攻击的稳健准确率达到了75.67美元，其中$epsilon_INFTY=8/255$，比以前最先进的对抗性训练模型高出$+4.77$。与通常研究的判别分类器相比，这些结果突出了生成式分类器通过使用预训练的扩散模型来提高对抗稳健性的潜力。代码可在\url{https://github.com/huanranchen/DiffusionClassifier}.上找到



## **47. EmInspector: Combating Backdoor Attacks in Federated Self-Supervised Learning Through Embedding Inspection**

EmInspector：通过嵌入检查打击联邦自我监督学习中的后门攻击 cs.CR

18 pages, 12 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13080v1) [paper-pdf](http://arxiv.org/pdf/2405.13080v1)

**Authors**: Yuwen Qian, Shuchi Wu, Kang Wei, Ming Ding, Di Xiao, Tao Xiang, Chuan Ma, Song Guo

**Abstract**: Federated self-supervised learning (FSSL) has recently emerged as a promising paradigm that enables the exploitation of clients' vast amounts of unlabeled data while preserving data privacy. While FSSL offers advantages, its susceptibility to backdoor attacks, a concern identified in traditional federated supervised learning (FSL), has not been investigated. To fill the research gap, we undertake a comprehensive investigation into a backdoor attack paradigm, where unscrupulous clients conspire to manipulate the global model, revealing the vulnerability of FSSL to such attacks. In FSL, backdoor attacks typically build a direct association between the backdoor trigger and the target label. In contrast, in FSSL, backdoor attacks aim to alter the global model's representation for images containing the attacker's specified trigger pattern in favor of the attacker's intended target class, which is less straightforward. In this sense, we demonstrate that existing defenses are insufficient to mitigate the investigated backdoor attacks in FSSL, thus finding an effective defense mechanism is urgent. To tackle this issue, we dive into the fundamental mechanism of backdoor attacks on FSSL, proposing the Embedding Inspector (EmInspector) that detects malicious clients by inspecting the embedding space of local models. In particular, EmInspector assesses the similarity of embeddings from different local models using a small set of inspection images (e.g., ten images of CIFAR100) without specific requirements on sample distribution or labels. We discover that embeddings from backdoored models tend to cluster together in the embedding space for a given inspection image. Evaluation results show that EmInspector can effectively mitigate backdoor attacks on FSSL across various adversary settings. Our code is avaliable at https://github.com/ShuchiWu/EmInspector.

摘要: 联合自我监督学习(FSSL)最近成为一种很有前途的范例，它能够在保护数据隐私的同时利用客户的大量未标记数据。虽然FSSL提供了优势，但它对后门攻击的敏感性尚未得到调查，这是传统联邦监督学习(FSL)中发现的一个问题。为了填补这一研究空白，我们对后门攻击范式进行了全面调查，在这种范式中，不择手段的客户合谋操纵全球模型，揭示了FSSL对此类攻击的脆弱性。在FSL中，后门攻击通常在后门触发器和目标标签之间建立直接关联。相比之下，在FSSL中，后门攻击的目标是更改包含攻击者指定的触发模式的图像的全局模型表示，以支持攻击者预期的目标类，这不是那么直接。从这个意义上说，我们证明了现有的防御措施不足以缓解FSSL中被调查的后门攻击，因此迫切需要找到一种有效的防御机制。为了解决这个问题，我们深入研究了FSSL后门攻击的基本机制，提出了嵌入检查器(EmInspector)，它通过检查本地模型的嵌入空间来检测恶意客户端。特别是，EmInspector使用一小组检查图像(例如，CIFAR100的10张图像)来评估来自不同本地模型的嵌入的相似性，而对样本分布或标签没有特定要求。我们发现，对于给定的检查图像，来自后置模型的嵌入往往在嵌入空间中聚集在一起。评估结果表明，EmInspector能够有效地缓解各种敌方环境下对FSSL的后门攻击。我们的代码可在https://github.com/ShuchiWu/EmInspector.上获得



## **48. Fully Randomized Pointers**

完全随机指针 cs.CR

24 pages, 3 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12513v1) [paper-pdf](http://arxiv.org/pdf/2405.12513v1)

**Authors**: Gregory J. Duck, Sai Dhawal Phaye, Roland H. C. Yap, Trevor E. Carlson

**Abstract**: Software security continues to be a critical concern for programs implemented in low-level programming languages such as C and C++. Many defenses have been proposed in the current literature, each with different trade-offs including performance, compatibility, and attack resistance. One general class of defense is pointer randomization or authentication, where invalid object access (e.g., memory errors) is obfuscated or denied. Many defenses rely on the program termination (e.g., crashing) to abort attacks, with the implicit assumption that an adversary cannot "brute force" the defense with multiple attack attempts. However, such assumptions do not always hold, such as hardware speculative execution attacks or network servers configured to restart on error. In such cases, we argue that most existing defenses provide only weak effective security.   In this paper, we propose Fully Randomized Pointers (FRP) as a stronger memory error defense that is resistant to even brute force attacks. The key idea is to fully randomize pointer bits -- as much as possible while also preserving binary compatibility -- rendering the relationships between pointers highly unpredictable. Furthermore, the very high degree of randomization renders brute force attacks impractical -- providing strong effective security compared to existing work. We design a new FRP encoding that is: (1) compatible with existing binary code (without recompilation); (2) decoupled from the underlying object layout; and (3) can be efficiently decoded on-the-fly to the underlying memory address. We prototype FRP in the form of a software implementation (BlueFat) to test security and compatibility, and a proof-of-concept hardware implementation (GreenFat) to evaluate performance. We show that FRP is secure, practical, and compatible at the binary level, while a hardware implementation can achieve low performance overheads (<10%).

摘要: 对于使用低级编程语言(如C和C++)实现的程序来说，软件安全性仍然是一个关键问题。在当前的文献中已经提出了许多防御措施，每种防御措施都具有不同的权衡，包括性能、兼容性和抗攻击能力。一种常见的防御类别是指针随机化或身份验证，在这种情况下，无效对象访问(例如，内存错误)被混淆或拒绝。许多防御依赖于程序终止(例如崩溃)来中止攻击，并隐含地假设对手不能通过多次攻击尝试来“野蛮地强迫”防御。然而，这样的假设并不总是成立的，例如硬件推测性执行攻击或配置为在出错时重新启动的网络服务器。在这种情况下，我们认为，大多数现有的防御措施只提供了薄弱的有效安全。在本文中，我们提出了完全随机化指针(FRP)作为一种更强的内存错误防御机制，它甚至可以抵抗暴力攻击。其关键思想是完全随机化指针位--尽可能多地同时保持二进制兼容性--使指针之间的关系高度不可预测。此外，非常高的随机性使暴力攻击变得不切实际--与现有工作相比，提供了强大的有效安全性。我们设计了一种新的FRP编码：(1)与现有的二进制代码兼容(无需重新编译)；(2)与底层对象布局解耦；(3)可以高效地动态解码到底层内存地址。我们以软件实现(BlueFat)的形式构建FRP原型以测试安全性和兼容性，并以概念验证硬件实现(GreenFat)的形式评估性能。我们证明了FRP是安全的、实用的和二进制级兼容的，而硬件实现可以获得较低的性能开销(<10%)。



## **49. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

GPT-4通过自我解释以近乎完美的成功越狱 cs.CR

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13077v1) [paper-pdf](http://arxiv.org/pdf/2405.13077v1)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find IRIS achieves jailbreak success rates of 98% on GPT-4 and 92% on GPT-4 Turbo in under 7 queries. It significantly outperforms prior approaches in automatic, black-box and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.

摘要: 越狱研究对于测试和理解大型语言模型(LLM)的安全和安保问题具有重要价值。在本文中，我们介绍了迭代精化诱导的自越狱(IRIS)，这是一种新的方法，它利用LLMS的反射能力来实现只访问黑盒的越狱。与以前的方法不同，IRIS通过将单一模型用作攻击者和目标来简化越狱过程。这种方法首先通过自我解释迭代地精炼对抗性提示，这对于确保即使是排列良好的LLM也遵守对抗性指令至关重要。然后，虹膜给出精致的提示，对产量进行评级并提高产量，以增加其危害性。我们发现IRIS在GPT-4上的越狱成功率为98%，在GPT-4 Turbo上的越狱成功率为92%。它在自动、黑盒和可解释越狱方面显著优于现有方法，同时需要的查询大大减少，从而建立了可解释越狱方法的新标准。



## **50. Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers**

重新思考稳健性评估：对基于学习的四足运动控制器的对抗攻击 cs.RO

RSS 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12424v1) [paper-pdf](http://arxiv.org/pdf/2405.12424v1)

**Authors**: Fan Shi, Chong Zhang, Takahiro Miki, Joonho Lee, Marco Hutter, Stelian Coros

**Abstract**: Legged locomotion has recently achieved remarkable success with the progress of machine learning techniques, especially deep reinforcement learning (RL). Controllers employing neural networks have demonstrated empirical and qualitative robustness against real-world uncertainties, including sensor noise and external perturbations. However, formally investigating the vulnerabilities of these locomotion controllers remains a challenge. This difficulty arises from the requirement to pinpoint vulnerabilities across a long-tailed distribution within a high-dimensional, temporally sequential space. As a first step towards quantitative verification, we propose a computational method that leverages sequential adversarial attacks to identify weaknesses in learned locomotion controllers. Our research demonstrates that, even state-of-the-art robust controllers can fail significantly under well-designed, low-magnitude adversarial sequence. Through experiments in simulation and on the real robot, we validate our approach's effectiveness, and we illustrate how the results it generates can be used to robustify the original policy and offer valuable insights into the safety of these black-box policies.

摘要: 近年来，随着机器学习技术的进步，特别是深度强化学习(RL)的发展，腿部运动已经取得了显著的成功。采用神经网络的控制器对真实世界的不确定性表现出了经验和定性的鲁棒性，包括传感器噪声和外部扰动。然而，正式调查这些运动控制器的漏洞仍然是一个挑战。这一困难源于需要在高维的、时间顺序的空间内精确定位跨长尾分布的漏洞。作为定量验证的第一步，我们提出了一种计算方法，该方法利用顺序对抗性攻击来识别学习的运动控制器中的弱点。我们的研究表明，即使是最先进的鲁棒控制器，在设计良好的低幅度对抗性序列下也会显著失效。通过仿真实验和在真实机器人上的实验，我们验证了该方法的有效性，并说明了它所产生的结果如何被用来证明原始策略的健壮性，并为这些黑盒策略的安全性提供了有价值的见解。



