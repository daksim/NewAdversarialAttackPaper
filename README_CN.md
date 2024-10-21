# Latest Adversarial Attack Papers
**update at 2024-10-21 09:53:25**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

通过交叉Wasserstein Balls进行分布和反向稳健逻辑回归 math.OC

33 pages, 3 color figures, under review at a conference

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2407.13625v2) [paper-pdf](http://arxiv.org/pdf/2407.13625v2)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Adversarially robust optimization (ARO) has become the de facto standard for training models to defend against adversarial attacks during testing. However, despite their robustness, these models often suffer from severe overfitting. To mitigate this issue, several successful approaches have been proposed, including replacing the empirical distribution in training with: (i) a worst-case distribution within an ambiguity set, leading to a distributionally robust (DR) counterpart of ARO; or (ii) a mixture of the empirical distribution with one derived from an auxiliary dataset (e.g., synthetic, external, or out-of-domain). Building on the first approach, we explore the Wasserstein DR counterpart of ARO for logistic regression and show it admits a tractable convex optimization reformulation. Adopting the second approach, we enhance the DR framework by intersecting its ambiguity set with one constructed from an auxiliary dataset, which yields significant improvements when the Wasserstein distance between the data-generating and auxiliary distributions can be estimated. We analyze the resulting optimization problem, develop efficient solutions, and show that our method outperforms benchmark approaches on standard datasets.

摘要: 对抗性稳健优化(ARO)已经成为训练模型在测试过程中防御对手攻击的事实上的标准。然而，尽管这些模型具有稳健性，但它们往往存在严重的过度拟合问题。为了缓解这个问题，已经提出了几种成功的方法，包括将训练中的经验分布替换为：(I)模糊集中的最差情况分布，导致ARO的分布稳健(DR)对应；或(Ii)经验分布与来自辅助数据集(例如，合成的、外部的或域外的)的经验分布的混合。在第一种方法的基础上，我们探索了用于Logistic回归的Wasserstein DR对应的ARO，并证明了它允许一个易于处理的凸优化改写。采用第二种方法，我们通过将DR框架的歧义集与从辅助数据集构造的歧义集相交来增强DR框架，当可以估计数据生成和辅助分布之间的Wasserstein距离时，这将产生显著的改进。我们分析了由此产生的优化问题，开发了有效的解决方案，并证明了我们的方法在标准数据集上的性能优于基准方法。



## **2. Explainable Graph Neural Networks Under Fire**

受攻击的可解释图神经网络 cs.LG

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2406.06417v2) [paper-pdf](http://arxiv.org/pdf/2406.06417v2)

**Authors**: Zhong Li, Simon Geisler, Yuhang Wang, Stephan Günnemann, Matthijs van Leeuwen

**Abstract**: Predictions made by graph neural networks (GNNs) usually lack interpretability due to their complex computational behavior and the abstract nature of graphs. In an attempt to tackle this, many GNN explanation methods have emerged. Their goal is to explain a model's predictions and thereby obtain trust when GNN models are deployed in decision critical applications. Most GNN explanation methods work in a post-hoc manner and provide explanations in the form of a small subset of important edges and/or nodes. In this paper we demonstrate that these explanations can unfortunately not be trusted, as common GNN explanation methods turn out to be highly susceptible to adversarial perturbations. That is, even small perturbations of the original graph structure that preserve the model's predictions may yield drastically different explanations. This calls into question the trustworthiness and practical utility of post-hoc explanation methods for GNNs. To be able to attack GNN explanation models, we devise a novel attack method dubbed \textit{GXAttack}, the first \textit{optimization-based} adversarial white-box attack method for post-hoc GNN explanations under such settings. Due to the devastating effectiveness of our attack, we call for an adversarial evaluation of future GNN explainers to demonstrate their robustness. For reproducibility, our code is available via GitHub.

摘要: 由于图的复杂的计算行为和图的抽象性质，图神经网络(GNN)的预测通常缺乏可解释性。为了解决这个问题，出现了许多GNN解释方法。他们的目标是解释模型的预测，从而在决策关键应用程序中部署GNN模型时获得信任。大多数GNN解释方法以后自组织的方式工作，并以重要边和/或节点的小子集的形式提供解释。在本文中，我们证明了不幸的是，这些解释不能被信任，因为常见的GNN解释方法被证明非常容易受到对抗性扰动的影响。也就是说，即使是对原始图表结构的微小扰动，保留了模型的预测，也可能产生截然不同的解释。这使人们对特别解释GNN的方法的可信性和实用性产生了疑问。为了能够攻击GNN解释模型，我们设计了一种新的攻击方法，称为文本{GXAttack}，这是第一个在这种情况下针对后自组织GNN解释的对抗性白盒攻击方法。由于我们的攻击具有毁灭性的效果，我们呼吁对未来的GNN解释器进行对抗性评估，以证明其健壮性。为了重现性，我们的代码可以通过GitHub获得。



## **3. JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework**

JAILJEDGE：具有多智能体增强解释评估框架的全面越狱法官基准 cs.CL

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.12855v2) [paper-pdf](http://arxiv.org/pdf/2410.12855v2)

**Authors**: Fan Liu, Yue Feng, Zhao Xu, Lixin Su, Xinyu Ma, Dawei Yin, Hao Liu

**Abstract**: Despite advancements in enhancing LLM safety against jailbreak attacks, evaluating LLM defenses remains a challenge, with current methods often lacking explainability and generalization to complex scenarios, leading to incomplete assessments (e.g., direct judgment without reasoning, low F1 score of GPT-4 in complex cases, bias in multilingual scenarios). To address this, we present JAILJUDGE, a comprehensive benchmark featuring diverse risk scenarios, including synthetic, adversarial, in-the-wild, and multilingual prompts, along with high-quality human-annotated datasets. The JAILJUDGE dataset includes over 35k+ instruction-tune data with reasoning explainability and JAILJUDGETEST, a 4.5k+ labeled set for risk scenarios, and a 6k+ multilingual set across ten languages. To enhance evaluation with explicit reasoning, we propose the JailJudge MultiAgent framework, which enables explainable, fine-grained scoring (1 to 10). This framework supports the construction of instruction-tuning ground truth and facilitates the development of JAILJUDGE Guard, an end-to-end judge model that provides reasoning and eliminates API costs. Additionally, we introduce JailBoost, an attacker-agnostic attack enhancer, and GuardShield, a moderation defense, both leveraging JAILJUDGE Guard. Our experiments demonstrate the state-of-the-art performance of JailJudge methods (JailJudge MultiAgent, JAILJUDGE Guard) across diverse models (e.g., GPT-4, Llama-Guard) and zero-shot scenarios. JailBoost and GuardShield significantly improve jailbreak attack and defense tasks under zero-shot settings, with JailBoost enhancing performance by 29.24% and GuardShield reducing defense ASR from 40.46% to 0.15%.

摘要: 尽管在增强LLM针对越狱攻击的安全性方面取得了进展，但评估LLM的防御措施仍然是一项挑战，当前的方法往往缺乏对复杂情景的解释性和通用性，导致评估不完整(例如，没有推理的直接判断、复杂案件中GPT-4的F1低分、多语言情景中的偏见)。为了解决这个问题，我们提出了JAILJUDGE，这是一个全面的基准，具有各种风险场景，包括合成提示、对抗性提示、野外提示和多语言提示，以及高质量的人工注释数据集。JAILJUDGE数据集包括超过35k+的具有推理可解释性的指令调优数据和JAILJUDGETEST，一个用于风险情景的4.5k+标签集，以及一个跨越10种语言的6k+多语种集。为了增强显式推理的评估，我们提出了JailJustice多代理框架，该框架支持可解释的、细粒度的评分(1到10)。该框架支持构建指令调优基础真理，并促进JAILJUDGE Guard的开发，JAILJUDGE Guard是一种提供推理并消除API成本的端到端判断模型。此外，我们还介绍了JailBoost和GuardShield，JailBoost是一种与攻击者无关的攻击增强器，GuardShield是一种适度防御，两者都利用JAILJUDGE Guard。我们的实验展示了JailJustice方法(JailJustice多代理，JAILJUDGE Guard)在不同模型(例如，GPT-4，Llama-Guard)和零射击场景中的最先进性能。JailBoost和GuardShield显著改善了零射击设置下的越狱攻防任务，JailBoost将性能提升了29.24%，GuardShield将防御ASR从40.46%降低到0.15%。



## **4. DMGNN: Detecting and Mitigating Backdoor Attacks in Graph Neural Networks**

DMGNN：检测和缓解图神经网络中的后门攻击 cs.CR

12 pages, 8 figures

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14105v1) [paper-pdf](http://arxiv.org/pdf/2410.14105v1)

**Authors**: Hao Sui, Bing Chen, Jiale Zhang, Chengcheng Zhu, Di Wu, Qinghua Lu, Guodong Long

**Abstract**: Recent studies have revealed that GNNs are highly susceptible to multiple adversarial attacks. Among these, graph backdoor attacks pose one of the most prominent threats, where attackers cause models to misclassify by learning the backdoored features with injected triggers and modified target labels during the training phase. Based on the features of the triggers, these attacks can be categorized into out-of-distribution (OOD) and in-distribution (ID) graph backdoor attacks, triggers with notable differences from the clean sample feature distributions constitute OOD backdoor attacks, whereas the triggers in ID backdoor attacks are nearly identical to the clean sample feature distributions. Existing methods can successfully defend against OOD backdoor attacks by comparing the feature distribution of triggers and clean samples but fail to mitigate stealthy ID backdoor attacks. Due to the lack of proper supervision signals, the main task accuracy is negatively affected in defending against ID backdoor attacks. To bridge this gap, we propose DMGNN against OOD and ID graph backdoor attacks that can powerfully eliminate stealthiness to guarantee defense effectiveness and improve the model performance. Specifically, DMGNN can easily identify the hidden ID and OOD triggers via predicting label transitions based on counterfactual explanation. To further filter the diversity of generated explainable graphs and erase the influence of the trigger features, we present a reverse sampling pruning method to screen and discard the triggers directly on the data level. Extensive experimental evaluations on open graph datasets demonstrate that DMGNN far outperforms the state-of-the-art (SOTA) defense methods, reducing the attack success rate to 5% with almost negligible degradation in model performance (within 3.5%).

摘要: 最近的研究表明，GNN非常容易受到多种对抗性攻击。其中，图的后门攻击构成了最突出的威胁之一，攻击者在训练阶段通过学习带有注入触发器和修改的目标标签的后置特征来导致模型错误分类。根据触发器的特征，这些攻击可以分为分布外(OOD)和分布内(ID)图后门攻击，与干净样本特征分布有显著差异的触发器构成了OOD后门攻击，而ID后门攻击中的触发器与干净样本特征分布几乎相同。现有的方法可以通过比较触发器和干净样本的特征分布来成功防御OOD后门攻击，但无法缓解隐蔽的ID后门攻击。由于缺乏适当的监管信号，在防御ID后门攻击时，主要任务的准确性受到了负面影响。为了弥补这一差距，我们提出了针对OOD和ID图后门攻击的DMGNN，能够有效地消除隐蔽性，保证防御效果，提高模型性能。具体地说，DMGNN可以通过基于反事实解释的标签转换预测来轻松识别隐藏的ID和OOD触发器。为了进一步过滤生成的可解释图的多样性，消除触发器特征的影响，我们提出了一种反向采样剪枝方法，直接在数据层面上对触发器进行筛选和丢弃。在开放图数据集上的大量实验评估表明，DMGNN的性能远远优于最新的防御方法(SOTA)，将攻击成功率降低到5%，而模型性能的下降几乎可以忽略不计(在3.5%以内)。



## **5. MMAD-Purify: A Precision-Optimized Framework for Efficient and Scalable Multi-Modal Attacks**

MMAD-Puriify：一个精确优化的框架，用于高效且可扩展的多模式攻击 cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.14089v1) [paper-pdf](http://arxiv.org/pdf/2410.14089v1)

**Authors**: Xinxin Liu, Zhongliang Guo, Siyuan Huang, Chun Pong Lau

**Abstract**: Neural networks have achieved remarkable performance across a wide range of tasks, yet they remain susceptible to adversarial perturbations, which pose significant risks in safety-critical applications. With the rise of multimodality, diffusion models have emerged as powerful tools not only for generative tasks but also for various applications such as image editing, inpainting, and super-resolution. However, these models still lack robustness due to limited research on attacking them to enhance their resilience. Traditional attack techniques, such as gradient-based adversarial attacks and diffusion model-based methods, are hindered by computational inefficiencies and scalability issues due to their iterative nature. To address these challenges, we introduce an innovative framework that leverages the distilled backbone of diffusion models and incorporates a precision-optimized noise predictor to enhance the effectiveness of our attack framework. This approach not only enhances the attack's potency but also significantly reduces computational costs. Our framework provides a cutting-edge solution for multi-modal adversarial attacks, ensuring reduced latency and the generation of high-fidelity adversarial examples with superior success rates. Furthermore, we demonstrate that our framework achieves outstanding transferability and robustness against purification defenses, outperforming existing gradient-based attack models in both effectiveness and efficiency.

摘要: 神经网络在广泛的任务中取得了显著的性能，但它们仍然容易受到对抗性扰动的影响，这些扰动在安全关键的应用中构成了巨大的风险。随着多通道技术的兴起，扩散模型已经成为一种强大的工具，不仅可用于生成性任务，还可用于各种应用，如图像编辑、修复和超分辨率。然而，由于对攻击它们以增强其弹性的研究有限，这些模型仍然缺乏稳健性。传统的攻击技术，如基于梯度的对抗性攻击和基于扩散模型的方法，由于其迭代的性质而受到计算效率和可扩展性问题的阻碍。为了应对这些挑战，我们引入了一个创新的框架，它利用了扩散模型的精炼主干，并结合了一个经过精度优化的噪声预测器来增强我们攻击框架的有效性。这种方法不仅增强了攻击的威力，而且显著降低了计算成本。我们的框架为多模式对抗性攻击提供了一种尖端的解决方案，确保减少延迟并生成具有卓越成功率的高保真对抗性示例。此外，我们还证明了我们的框架实现了出色的可转移性和对净化防御的健壮性，在有效性和效率上都优于现有的基于梯度的攻击模型。



## **6. Uncovering Attacks and Defenses in Secure Aggregation for Federated Deep Learning**

揭露联邦深度学习安全聚合中的攻击和防御 cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.09676v2) [paper-pdf](http://arxiv.org/pdf/2410.09676v2)

**Authors**: Yiwei Zhang, Rouzbeh Behnia, Attila A. Yavuz, Reza Ebrahimi, Elisa Bertino

**Abstract**: Federated learning enables the collaborative learning of a global model on diverse data, preserving data locality and eliminating the need to transfer user data to a central server. However, data privacy remains vulnerable, as attacks can target user training data by exploiting the updates sent by users during each learning iteration. Secure aggregation protocols are designed to mask/encrypt user updates and enable a central server to aggregate the masked information. MicroSecAgg (PoPETS 2024) proposes a single server secure aggregation protocol that aims to mitigate the high communication complexity of the existing approaches by enabling a one-time setup of the secret to be re-used in multiple training iterations. In this paper, we identify a security flaw in the MicroSecAgg that undermines its privacy guarantees. We detail the security flaw and our attack, demonstrating how an adversary can exploit predictable masking values to compromise user privacy. Our findings highlight the critical need for enhanced security measures in secure aggregation protocols, particularly the implementation of dynamic and unpredictable masking strategies. We propose potential countermeasures to mitigate these vulnerabilities and ensure robust privacy protection in the secure aggregation frameworks.

摘要: 联合学习实现了对不同数据的全球模型的协作学习，保留了数据的局部性，消除了将用户数据传输到中央服务器的需要。然而，数据隐私仍然很容易受到攻击，因为攻击可以通过利用用户在每次学习迭代期间发送的更新来攻击用户训练数据。安全聚合协议旨在屏蔽/加密用户更新，并使中央服务器能够聚合屏蔽的信息。MicroSecAgg(PoPETS 2024)提出了一种单服务器安全聚合协议，该协议旨在通过允许一次性设置秘密以在多次训练迭代中重复使用来缓解现有方法的高度通信复杂性。在本文中，我们发现了MicroSecAgg中的一个安全漏洞，该漏洞破坏了其隐私保障。我们详细介绍了安全漏洞和我们的攻击，展示了对手如何利用可预测的掩蔽值来危害用户隐私。我们的发现强调了在安全聚合协议中增强安全措施的迫切需要，特别是实施动态和不可预测的掩蔽策略。我们提出了潜在的对策来缓解这些漏洞，并确保在安全聚合框架中提供强大的隐私保护。



## **7. Adversarial Inception for Bounded Backdoor Poisoning in Deep Reinforcement Learning**

深度强化学习中有界后门中毒的对抗性初始 cs.LG

10 pages, 5 figures, ICLR 2025

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13995v1) [paper-pdf](http://arxiv.org/pdf/2410.13995v1)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Recent works have demonstrated the vulnerability of Deep Reinforcement Learning (DRL) algorithms against training-time, backdoor poisoning attacks. These attacks induce pre-determined, adversarial behavior in the agent upon observing a fixed trigger during deployment while allowing the agent to solve its intended task during training. Prior attacks rely on arbitrarily large perturbations to the agent's rewards to achieve both of these objectives - leaving them open to detection. Thus, in this work, we propose a new class of backdoor attacks against DRL which achieve state of the art performance while minimally altering the agent's rewards. These ``inception'' attacks train the agent to associate the targeted adversarial behavior with high returns by inducing a disjunction between the agent's chosen action and the true action executed in the environment during training. We formally define these attacks and prove they can achieve both adversarial objectives. We then devise an online inception attack which significantly out-performs prior attacks under bounded reward constraints.

摘要: 最近的工作证明了深度强化学习(DRL)算法在抵抗训练时间、后门中毒攻击时的脆弱性。这些攻击在部署期间观察到固定触发器时，会在代理中诱导预先确定的对抗性行为，同时允许代理在培训期间解决其预期任务。以前的攻击依赖于对代理报酬的任意大扰动来实现这两个目标--使它们容易被检测到。因此，在这项工作中，我们提出了一类新的针对DRL的后门攻击，它在最大限度地改变代理的报酬的同时实现了最先进的性能。这些“初始”攻击训练代理将目标对抗性行为与高回报相关联，方法是在培训期间诱导代理选择的动作与在环境中执行的真实动作之间的脱节。我们正式定义了这些攻击，并证明它们可以达到两个对抗性目标。然后，我们设计了一个在线初始攻击，在有限制的报酬约束下，它的性能明显优于先前的攻击。



## **8. Trojan Prompt Attacks on Graph Neural Networks**

图神经网络的特洛伊提示攻击 cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13974v1) [paper-pdf](http://arxiv.org/pdf/2410.13974v1)

**Authors**: Minhua Lin, Zhiwei Zhang, Enyan Dai, Zongyu Wu, Yilong Wang, Xiang Zhang, Suhang Wang

**Abstract**: Graph Prompt Learning (GPL) has been introduced as a promising approach that uses prompts to adapt pre-trained GNN models to specific downstream tasks without requiring fine-tuning of the entire model. Despite the advantages of GPL, little attention has been given to its vulnerability to backdoor attacks, where an adversary can manipulate the model's behavior by embedding hidden triggers. Existing graph backdoor attacks rely on modifying model parameters during training, but this approach is impractical in GPL as GNN encoder parameters are frozen after pre-training. Moreover, downstream users may fine-tune their own task models on clean datasets, further complicating the attack. In this paper, we propose TGPA, a backdoor attack framework designed specifically for GPL. TGPA injects backdoors into graph prompts without modifying pre-trained GNN encoders and ensures high attack success rates and clean accuracy. To address the challenge of model fine-tuning by users, we introduce a finetuning-resistant poisoning approach that maintains the effectiveness of the backdoor even after downstream model adjustments. Extensive experiments on multiple datasets under various settings demonstrate the effectiveness of TGPA in compromising GPL models with fixed GNN encoders.

摘要: 图形提示学习(GPL)是一种很有前途的方法，它使用提示来使预先训练的GNN模型适应特定的下游任务，而不需要对整个模型进行微调。尽管GPL具有优势，但很少有人注意到它在后门攻击中的脆弱性，在后门攻击中，对手可以通过嵌入隐藏的触发器来操纵模型的行为。现有的图后门攻击依赖于在训练过程中修改模型参数，但这种方法在GPL中是不可行的，因为GNN编码器参数在预训练后被冻结。此外，下游用户可能会在干净的数据集上微调自己的任务模型，从而使攻击进一步复杂化。在本文中，我们提出了一个专门为GPL设计的后门攻击框架TGPA。TGPA在不修改预先训练的GNN编码器的情况下，将后门注入图形提示，并确保高攻击成功率和干净的准确性。为了解决用户对模型微调的挑战，我们引入了一种抗微调的中毒方法，即使在下游模型调整后仍保持后门的有效性。在不同设置下的多个数据集上的大量实验证明了TGPA在折衷具有固定GNN编码器的GPL模型方面的有效性。



## **9. Multi-style conversion for semantic segmentation of lesions in fundus images by adversarial attacks**

对抗性攻击下的视网膜图像病变语义分割的多种类型转换 cs.CV

preprint

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13822v1) [paper-pdf](http://arxiv.org/pdf/2410.13822v1)

**Authors**: Clément Playout, Renaud Duval, Marie Carole Boucher, Farida Cheriet

**Abstract**: The diagnosis of diabetic retinopathy, which relies on fundus images, faces challenges in achieving transparency and interpretability when using a global classification approach. However, segmentation-based databases are significantly more expensive to acquire and combining them is often problematic. This paper introduces a novel method, termed adversarial style conversion, to address the lack of standardization in annotation styles across diverse databases. By training a single architecture on combined databases, the model spontaneously modifies its segmentation style depending on the input, demonstrating the ability to convert among different labeling styles. The proposed methodology adds a linear probe to detect dataset origin based on encoder features and employs adversarial attacks to condition the model's segmentation style. Results indicate significant qualitative and quantitative through dataset combination, offering avenues for improved model generalization, uncertainty estimation and continuous interpolation between annotation styles. Our approach enables training a segmentation model with diverse databases while controlling and leveraging annotation styles for improved retinopathy diagnosis.

摘要: 糖尿病视网膜病变的诊断依赖于眼底图像，在使用全球分类方法时面临着实现透明度和可解释性的挑战。然而，基于细分的数据库的获取成本要高得多，而将它们结合起来往往是有问题的。本文介绍了一种称为对抗性风格转换的新方法，以解决跨不同数据库的注释风格缺乏标准化的问题。通过在组合数据库上训练单个体系结构，该模型根据输入自发地修改其分割样式，展示了在不同标签样式之间进行转换的能力。该方法在编码特征的基础上增加了一个线性探测器来检测数据集的来源，并使用对抗性攻击来调节模型的分割风格。结果表明，通过数据集的组合具有显著的定性和定量，为改进模型泛化、不确定性估计和注记样式之间的连续内插提供了途径。我们的方法能够使用不同的数据库来训练分割模型，同时控制和利用注释样式来改进视网膜病变诊断。



## **10. Persistent Pre-Training Poisoning of LLMs**

LLM训练前持续中毒 cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13722v1) [paper-pdf](http://arxiv.org/pdf/2410.13722v1)

**Authors**: Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric Michael Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito

**Abstract**: Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web. Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets. Our work evaluates for the first time whether language models can also be compromised during pre-training, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B). Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.

摘要: 大型语言模型是在未经精选的文本数据集上预先训练的，这些数据集由从Web上刮来的数万亿个标记组成。先前的工作表明：(1)网络刮来的预训练数据集实际上可能会被恶意行为者毒化；(2)攻击者在毒化微调数据集后可能会危害语言模型。我们的工作首次评估了语言模型在预训练期间是否也会被破坏，重点是在模型被微调为有帮助和无害的聊天机器人后(即在SFT和DPO之后)，预训练攻击的持久性。我们从头开始预先训练一系列LLM，以衡量潜在中毒对手在四种不同攻击目标(拒绝服务、信念操纵、越狱和即时盗窃)下的影响，并跨越广泛的模型大小(从600M到7B)。我们的主要结果是，只有0.1%的模型训练前数据集的中毒足以使四分之三的攻击在训练后可测量地持续存在。此外，像拒绝服务这样的简单攻击在培训后持续存在，投毒率仅为0.001%。



## **11. Optimal MEV Extraction Using Absolute Commitments**

使用绝对承诺的最佳MEV提取 cs.GT

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13624v1) [paper-pdf](http://arxiv.org/pdf/2410.13624v1)

**Authors**: Daji Landis, Nikolaj I. Schwartzbach

**Abstract**: We propose a new, more potent attack on decentralized exchanges. This attack leverages absolute commitments, which are commitments that can condition on the strategies made by other agents. This attack allows an adversary to charge monopoly prices by committing to undercut those other miners that refuse to charge an even higher fee. This allows the miner to extract the maximum possible price from the user, potentially through side channels that evade the inefficiencies and fees usually incurred. This is considerably more efficient than the prevailing strategy of `sandwich attacks', wherein the adversary induces and profits from fluctuations in the market price to the detriment of users. The attack we propose can, in principle, be realized by the irrevocable and self-executing nature of smart contracts, which are readily available on many major blockchains. Thus, the attack could potentially be used against a decentralized exchange and could drastically reduce the utility of the affected exchange.

摘要: 我们提出了对去中心化交易所的一种新的、更有力的攻击。这种攻击利用了绝对承诺，这是可以以其他代理人的战略为条件的承诺。这种攻击允许对手通过承诺以低于其他拒绝收取更高费用的矿商的价格来收取垄断价格。这使得矿商可以从用户那里获取最大可能的价格，可能是通过旁路渠道来规避通常产生的低效和费用。这比目前流行的“三明治攻击”策略要有效得多，在“三明治攻击”策略中，对手从市场价格的波动中诱导并获利，损害了用户的利益。原则上，我们提出的攻击可以通过智能合约的不可撤销和自动执行的性质来实现，这些合约在许多主要区块链上都很容易获得。因此，该攻击可能被用来攻击分散的交易所，并可能极大地降低受影响交易所的效用。



## **12. Transformer-Based Approaches for Sensor-Based Human Activity Recognition: Opportunities and Challenges**

基于传感器的人类活动识别方法：机遇与挑战 cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13605v1) [paper-pdf](http://arxiv.org/pdf/2410.13605v1)

**Authors**: Clayton Souza Leite, Henry Mauranen, Aziza Zhanabatyrova, Yu Xiao

**Abstract**: Transformers have excelled in natural language processing and computer vision, paving their way to sensor-based Human Activity Recognition (HAR). Previous studies show that transformers outperform their counterparts exclusively when they harness abundant data or employ compute-intensive optimization algorithms. However, neither of these scenarios is viable in sensor-based HAR due to the scarcity of data in this field and the frequent need to perform training and inference on resource-constrained devices. Our extensive investigation into various implementations of transformer-based versus non-transformer-based HAR using wearable sensors, encompassing more than 500 experiments, corroborates these concerns. We observe that transformer-based solutions pose higher computational demands, consistently yield inferior performance, and experience significant performance degradation when quantized to accommodate resource-constrained devices. Additionally, transformers demonstrate lower robustness to adversarial attacks, posing a potential threat to user trust in HAR.

摘要: 变形金刚在自然语言处理和计算机视觉方面表现出色，为基于传感器的人类活动识别(HAR)铺平了道路。以前的研究表明，只有当变压器利用大量数据或采用计算密集型优化算法时，它们的性能才会优于同行。然而，这两种情况在基于传感器的HAR中都是不可行的，因为该领域的数据稀缺，并且经常需要在资源受限的设备上执行训练和推理。我们对使用可穿戴传感器的基于变压器的HAR和基于非变压器的HAR的各种实现进行了广泛的调查，包括500多个实验，证实了这些担忧。我们观察到，基于变压器的解决方案提出了更高的计算要求，始终产生较差的性能，并且当量化以适应资源受限的设备时，性能会显著下降。此外，变压器对敌意攻击表现出较低的稳健性，对用户对HAR的信任构成了潜在威胁。



## **13. Adversarial Exposure Attack on Diabetic Retinopathy Imagery Grading**

糖尿病视网膜病变图像分级的对抗暴露攻击 cs.CV

13 pages, 7 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2009.09231v2) [paper-pdf](http://arxiv.org/pdf/2009.09231v2)

**Authors**: Yupeng Cheng, Qing Guo, Felix Juefei-Xu, Huazhu Fu, Shang-Wei Lin, Weisi Lin

**Abstract**: Diabetic Retinopathy (DR) is a leading cause of vision loss around the world. To help diagnose it, numerous cutting-edge works have built powerful deep neural networks (DNNs) to automatically grade DR via retinal fundus images (RFIs). However, RFIs are commonly affected by camera exposure issues that may lead to incorrect grades. The mis-graded results can potentially pose high risks to an aggravation of the condition. In this paper, we study this problem from the viewpoint of adversarial attacks. We identify and introduce a novel solution to an entirely new task, termed as adversarial exposure attack, which is able to produce natural exposure images and mislead the state-of-the-art DNNs. We validate our proposed method on a real-world public DR dataset with three DNNs, e.g., ResNet50, MobileNet, and EfficientNet, demonstrating that our method achieves high image quality and success rate in transferring the attacks. Our method reveals the potential threats to DNN-based automatic DR grading and would benefit the development of exposure-robust DR grading methods in the future.

摘要: 糖尿病视网膜病变(DR)是世界范围内导致视力丧失的主要原因。为了帮助诊断，许多尖端工作已经建立了强大的深层神经网络(DNN)，通过视网膜眼底图像(RFI)自动对DR进行分级。然而，RFI通常会受到相机曝光问题的影响，这可能会导致不正确的分数。错误评级的结果可能会给病情恶化带来很高的风险。本文从对抗性攻击的角度对这一问题进行了研究。我们发现并引入了一种全新的解决方案，称为对抗性曝光攻击，它能够产生自然曝光图像并误导最先进的DNN。我们在一个真实的公共灾难恢复数据集上验证了我们的方法，并用ResNet50、MobileNet和EfficientNet三个DNN进行了验证，结果表明我们的方法达到了较高的图像质量和攻击转移的成功率。我们的方法揭示了基于DNN的DR自动分级面临的潜在威胁，并将有助于未来开发具有曝光性的DR分级方法。



## **14. Bias in the Mirror : Are LLMs opinions robust to their own adversarial attacks ?**

镜子中的偏见：LLM的观点是否能抵御自己的对抗攻击？ cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13517v1) [paper-pdf](http://arxiv.org/pdf/2410.13517v1)

**Authors**: Virgile Rennard, Christos Xypolopoulos, Michalis Vazirgiannis

**Abstract**: Large language models (LLMs) inherit biases from their training data and alignment processes, influencing their responses in subtle ways. While many studies have examined these biases, little work has explored their robustness during interactions. In this paper, we introduce a novel approach where two instances of an LLM engage in self-debate, arguing opposing viewpoints to persuade a neutral version of the model. Through this, we evaluate how firmly biases hold and whether models are susceptible to reinforcing misinformation or shifting to harmful viewpoints. Our experiments span multiple LLMs of varying sizes, origins, and languages, providing deeper insights into bias persistence and flexibility across linguistic and cultural contexts.

摘要: 大型语言模型（LLM）从其训练数据和对齐过程中继承了偏差，以微妙的方式影响其响应。虽然许多研究已经检查了这些偏差，但很少有工作探索它们在互动过程中的稳健性。在本文中，我们引入了一种新颖的方法，其中两个LLM实例进行自我辩论，争论相反的观点以说服模型的中立版本。通过此，我们评估偏见的存在程度，以及模型是否容易强化错误信息或转向有害观点。我们的实验跨越了不同规模、起源和语言的多个LLM，为跨语言和文化背景的偏见持续性和灵活性提供了更深入的见解。



## **15. MirrorCheck: Efficient Adversarial Defense for Vision-Language Models**

收件箱检查：视觉语言模型的有效对抗防御 cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2406.09250v2) [paper-pdf](http://arxiv.org/pdf/2406.09250v2)

**Authors**: Samar Fares, Klea Ziu, Toluwani Aremu, Nikita Durasov, Martin Takáč, Pascal Fua, Karthik Nandakumar, Ivan Laptev

**Abstract**: Vision-Language Models (VLMs) are becoming increasingly vulnerable to adversarial attacks as various novel attack strategies are being proposed against these models. While existing defenses excel in unimodal contexts, they currently fall short in safeguarding VLMs against adversarial threats. To mitigate this vulnerability, we propose a novel, yet elegantly simple approach for detecting adversarial samples in VLMs. Our method leverages Text-to-Image (T2I) models to generate images based on captions produced by target VLMs. Subsequently, we calculate the similarities of the embeddings of both input and generated images in the feature space to identify adversarial samples. Empirical evaluations conducted on different datasets validate the efficacy of our approach, outperforming baseline methods adapted from image classification domains. Furthermore, we extend our methodology to classification tasks, showcasing its adaptability and model-agnostic nature. Theoretical analyses and empirical findings also show the resilience of our approach against adaptive attacks, positioning it as an excellent defense mechanism for real-world deployment against adversarial threats.

摘要: 随着针对视觉语言模型的各种新的攻击策略的提出，视觉语言模型正变得越来越容易受到对手攻击。虽然现有的防御系统在单峰环境中表现出色，但它们目前在保护VLM免受对手威胁方面存在不足。为了缓解这一漏洞，我们提出了一种新颖而又非常简单的方法来检测VLM中的敌意样本。我们的方法利用文本到图像(T2I)模型来生成基于目标VLM生成的字幕的图像。随后，我们计算输入图像和生成图像在特征空间中的嵌入相似度，以识别敌意样本。在不同的数据集上进行的经验评估验证了我们方法的有效性，优于适用于图像分类领域的基线方法。此外，我们将我们的方法扩展到分类任务，展示了其适应性和模型不可知性。理论分析和经验结果也表明了我们的方法对适应性攻击的弹性，将其定位为针对对抗性威胁的真实部署的优秀防御机制。



## **16. Byzantine-Resilient Output Optimization of Multiagent via Self-Triggered Hybrid Detection Approach**

通过自触发混合检测方法实现MultiAgent的抗拜占庭输出优化 eess.SY

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13454v1) [paper-pdf](http://arxiv.org/pdf/2410.13454v1)

**Authors**: Chenhang Yan, Liping Yan, Yuezu Lv, Bolei Dong, Yuanqing Xia

**Abstract**: How to achieve precise distributed optimization despite unknown attacks, especially the Byzantine attacks, is one of the critical challenges for multiagent systems. This paper addresses a distributed resilient optimization for linear heterogeneous multi-agent systems faced with adversarial threats. We establish a framework aimed at realizing resilient optimization for continuous-time systems by incorporating a novel self-triggered hybrid detection approach. The proposed hybrid detection approach is able to identify attacks on neighbors using both error thresholds and triggering intervals, thereby optimizing the balance between effective attack detection and the reduction of excessive communication triggers. Through using an edge-based adaptive self-triggered approach, each agent can receive its neighbors' information and determine whether these information is valid. If any neighbor prove invalid, each normal agent will isolate that neighbor by disconnecting communication along that specific edge. Importantly, our adaptive algorithm guarantees the accuracy of the optimization solution even when an agent is isolated by its neighbors.

摘要: 如何在未知攻击，特别是拜占庭攻击的情况下实现精确的分布式优化，是多智能体系统面临的关键挑战之一。针对面临敌意威胁的线性异质多智能体系统，提出了一种分布式弹性优化算法。通过引入一种新的自触发混合检测方法，我们建立了一个框架，旨在实现连续时间系统的弹性优化。提出的混合检测方法能够同时使用错误阈值和触发间隔来识别对邻居的攻击，从而优化了有效的攻击检测和减少过度通信触发之间的平衡。通过使用基于边缘的自适应自触发方法，每个代理可以接收邻居的信息并判断这些信息是否有效。如果任何邻居被证明是无效，则每个正常代理将通过断开沿该特定边缘的通信来隔离该邻居。重要的是，我们的自适应算法保证了最优解的准确性，即使在代理被邻居隔离的情况下也是如此。



## **17. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.02240v3) [paper-pdf](http://arxiv.org/pdf/2410.02240v3)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统容易受到敌意攻击。不受限制的对抗性攻击通常操纵图像的语义内容(例如，颜色或纹理)以创建既有效又逼真的对抗性示例。最近的工作利用扩散逆过程将图像映射到潜在空间，在潜在空间中通过引入扰动来操纵高级语义。然而，它们往往会在去噪输出中造成严重的语义扭曲，并导致效率低下。在这项研究中，我们提出了一种新的框架，称为语义一致的无限对抗攻击(SCA)，它使用一种反转方法来提取编辑友好的噪声映射，并利用多模式大语言模型(MLLM)在整个过程中提供语义指导。在MLLM提供丰富语义信息的条件下，使用一系列编辑友好的噪声图对每个步骤进行DDPM去噪处理，并利用DPM Solver++加速这一过程，从而实现高效的语义一致性采样。与现有的方法相比，我们的框架能够高效地生成对抗性的例子，这些例子表现出最小的可识别的语义变化。因此，我们首次引入了语义一致的对抗性例子(SCAE)。广泛的实验和可视化已经证明了SCA的高效率，特别是在平均速度上是最先进的攻击的12倍。我们的研究可以进一步引起人们对多媒体信息安全的关注。



## **18. Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning**

打破链条：解开多跳知识遗忘中的链接 cs.CL

16 pages, 5 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13274v1) [paper-pdf](http://arxiv.org/pdf/2410.13274v1)

**Authors**: Minseok Choi, ChaeHun Park, Dohyun Lee, Jaegul Choo

**Abstract**: Large language models (LLMs) serve as giant information stores, often including personal or copyrighted data, and retraining them from scratch is not a viable option. This has led to the development of various fast, approximate unlearning techniques to selectively remove knowledge from LLMs. Prior research has largely focused on minimizing the probabilities of specific token sequences by reversing the language modeling objective. However, these methods still leave LLMs vulnerable to adversarial attacks that exploit indirect references. In this work, we examine the limitations of current unlearning techniques in effectively erasing a particular type of indirect prompt: multi-hop queries. Our findings reveal that existing methods fail to completely remove multi-hop knowledge when one of the intermediate hops is unlearned. To address this issue, we propose MUNCH, a simple uncertainty-based approach that breaks down multi-hop queries into subquestions and leverages the uncertainty of the unlearned model in final decision-making. Empirical results demonstrate the effectiveness of our framework, and MUNCH can be easily integrated with existing unlearning techniques, making it a flexible and useful solution for enhancing unlearning processes.

摘要: 大型语言模型(LLM)充当了巨大的信息存储，通常包括个人或受版权保护的数据，从零开始对它们进行再培训并不是一个可行的选择。这导致了各种快速、近似的遗忘技术的发展，以选择性地从LLM中移除知识。以前的研究主要集中在通过颠倒语言建模目标来最小化特定标记序列的概率。然而，这些方法仍然使LLM容易受到利用间接引用的敌意攻击。在这项工作中，我们检查了当前遗忘技术在有效消除一种特定类型的间接提示：多跳查询方面的局限性。我们的发现表明，当中间跳之一未被学习时，现有方法无法完全消除多跳知识。为了解决这个问题，我们提出了Munch，一种简单的基于不确定性的方法，将多跳查询分解为子问题，并在最终决策中利用未学习模型的不确定性。实验结果表明我们的框架是有效的，而且Munch可以很容易地与现有的遗忘技术相结合，使其成为一种灵活而有用的解决方案来增强遗忘过程。



## **19. SPIN: Self-Supervised Prompt INjection**

旋转：自我监督的即时注射 cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13236v1) [paper-pdf](http://arxiv.org/pdf/2410.13236v1)

**Authors**: Leon Zhou, Junfeng Yang, Chengzhi Mao

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of important applications, yet their safety and reliability remain as major concerns. Various adversarial and jailbreak attacks have been proposed to bypass the safety alignment and cause the model to produce harmful responses. We introduce Self-supervised Prompt INjection (SPIN) which can detect and reverse these various attacks on LLMs. As our self-supervised prompt defense is done at inference-time, it is also compatible with existing alignment and adds an additional layer of safety for defense. Our benchmarks demonstrate that our system can reduce the attack success rate by up to 87.9%, while maintaining the performance on benign user requests. In addition, we discuss the situation of an adaptive attacker and show that our method is still resilient against attackers who are aware of our defense.

摘要: 大型语言模型（LLM）越来越多地用于各种重要应用程序，但其安全性和可靠性仍然是主要问题。人们提出了各种对抗和越狱攻击来绕过安全一致并导致模型产生有害响应。我们引入了自我监督提示注入（SPIN），它可以检测和逆转对LLM的各种攻击。由于我们的自我监督即时防御是在推理时完成的，因此它也与现有的对齐兼容，并为防御增加了额外的安全层。我们的基准测试表明，我们的系统可以将攻击成功率降低高达87.9%，同时保持良性用户请求的性能。此外，我们还讨论了自适应攻击者的情况，并表明我们的方法对于意识到我们防御的攻击者仍然具有弹性。



## **20. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

多模式大型语言模型中检测越狱的跨模式信息检查 cs.CL

12 pages, 9 figures, EMNLP 2024 Findings

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2407.21659v4) [paper-pdf](http://arxiv.org/pdf/2407.21659v4)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.

摘要: 多通道大语言模型扩展了多通道大语言模型对多通道信息的理解能力，在许多以视觉为中心的任务中取得了显著的性能。尽管如此，最近的研究表明，这些模型容易受到越狱攻击，越狱攻击指的是一种利用技术，恶意用户可以破坏目标模型的安全对齐，并生成误导性和有害的答案。这种潜在的威胁既是由LLM固有的漏洞造成的，也是由视觉输入引入的更大的攻击范围造成的。为了提高MLMS抵御越狱攻击的安全性，研究人员开发了各种防御技术。然而，这些方法要么需要修改模型的内部结构，要么在推理阶段需要大量的计算资源。多式联运信息是一把双刃剑。虽然它增加了攻击的风险，但它也提供了额外的数据，可以加强安全措施。受此启发，我们提出了跨模式信息检测器(Cider)，这是一种即插即用的越狱检测器，旨在利用有害查询和敌意图像之间的跨模式相似性来识别恶意扰动的图像输入。苹果酒不依赖于目标MLLM，并且需要较少的计算成本。大量的实验结果证明了苹果酒的有效性和效率，以及它对白盒和黑盒MLLMS的可转换性。



## **21. Golyadkin's Torment: Doppelgängers and Adversarial Vulnerability**

戈利亚德金的折磨：分身和对抗脆弱性 cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13193v1) [paper-pdf](http://arxiv.org/pdf/2410.13193v1)

**Authors**: George I. Kamberov

**Abstract**: Many machine learning (ML) classifiers are claimed to outperform humans, but they still make mistakes that humans do not. The most notorious examples of such mistakes are adversarial visual metamers. This paper aims to define and investigate the phenomenon of adversarial Doppelgangers (AD), which includes adversarial visual metamers, and to compare the performance and robustness of ML classifiers to human performance.   We find that AD are inputs that are close to each other with respect to a perceptual metric defined in this paper. AD are qualitatively different from the usual adversarial examples. The vast majority of classifiers are vulnerable to AD and robustness-accuracy trade-offs may not improve them. Some classification problems may not admit any AD robust classifiers because the underlying classes are ambiguous. We provide criteria that can be used to determine whether a classification problem is well defined or not; describe the structure and attributes of an AD-robust classifier; introduce and explore the notions of conceptual entropy and regions of conceptual ambiguity for classifiers that are vulnerable to AD attacks, along with methods to bound the AD fooling rate of an attack. We define the notion of classifiers that exhibit hypersensitive behavior, that is, classifiers whose only mistakes are adversarial Doppelgangers. Improving the AD robustness of hyper-sensitive classifiers is equivalent to improving accuracy. We identify conditions guaranteeing that all classifiers with sufficiently high accuracy are hyper-sensitive.   Our findings are aimed at significant improvements in the reliability and security of machine learning systems.

摘要: 许多机器学习(ML)分类器声称比人类性能更好，但他们仍然会犯人类不会犯的错误。这类错误最臭名昭著的例子是对抗性的视觉异构体。本文旨在定义和研究包括对抗性视觉异构体在内的对抗性二重体(AD)现象，并将ML分类器的性能和稳健性与人类的性能进行比较。我们发现AD是相对于本文定义的知觉度量彼此接近的输入。广告在性质上不同于通常的对抗性例子。绝大多数分类器容易受到AD的影响，稳健性和准确性之间的权衡可能不会改善它们。一些分类问题可能不允许任何AD稳健分类器，因为底层类是不明确的。我们提供了可用于判断分类问题是否定义良好的标准；描述了AD稳健分类器的结构和属性；引入和探索了易受AD攻击的分类器的概念熵和概念歧义区域的概念，以及限制攻击的AD愚弄率的方法。我们定义了表现出超敏感行为的量词的概念，也就是说，量词的唯一错误是对抗性的双重词。提高超敏感分类器的AD稳健性等同于提高准确率。我们确定了保证所有具有足够高精度的分类器都是超敏感的条件。我们的发现旨在显著提高机器学习系统的可靠性和安全性。



## **22. Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability**

模型供应链中毒：通过嵌入不可分割性对预训练模型进行后门 cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2401.15883v2) [paper-pdf](http://arxiv.org/pdf/2401.15883v2)

**Authors**: Hao Wang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang, Tao Xiang

**Abstract**: Pre-trained models (PTMs) are widely adopted across various downstream tasks in the machine learning supply chain. Adopting untrustworthy PTMs introduces significant security risks, where adversaries can poison the model supply chain by embedding hidden malicious behaviors (backdoors) into PTMs. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. This makes it challenging for the backdoors to persist and propagate through the supply chain. In this paper, we propose a novel and severer backdoor attack, TransTroj, which enables the backdoors embedded in PTMs to efficiently transfer in the model supply chain. In particular, we first formalize this attack as an indistinguishability problem between poisoned and clean samples in the embedding space. We decompose embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that our method significantly outperforms SOTA task-agnostic backdoor attacks -- achieving nearly 100\% attack success rate on most downstream tasks -- and demonstrates robustness under various system settings. Our findings underscore the urgent need to secure the model supply chain against such transferable backdoor attacks. The code is available at https://github.com/haowang-cqu/TransTroj .

摘要: 预训练模型(PTM)广泛应用于机器学习供应链中的各种下游任务。采用不可信的PTMS会带来严重的安全风险，攻击者可以通过在PTMS中嵌入隐藏的恶意行为(后门)来毒化模型供应链。然而，现有的对PTMS的后门攻击只能实现部分任务无关，并且嵌入的后门在微调过程中很容易被擦除。这使得后门在供应链中的持续和传播变得具有挑战性。在本文中，我们提出了一种新的更严重的后门攻击，TransTroj，它使得嵌入PTMS的后门能够在模型供应链中有效地转移。特别地，我们首先将这种攻击形式化为嵌入空间中有毒样本和干净样本之间的不可区分问题。我们将嵌入不可区分性分解为攻击前后的不可区分性，表示攻击前后中毒嵌入和参考嵌入的相似性。然后，我们提出了一种两阶段优化方法，分别对触发者和受害者PTM进行优化，以达到嵌入不可区分的目的。我们在四个PTM和六个下游任务上对TransTroj进行了评估。实验结果表明，我们的方法显著优于SOTA任务无关的后门攻击--在大多数下游任务上获得近100%的攻击成功率--并在不同的系统设置下表现出健壮性。我们的发现突显出，迫切需要确保模型供应链免受这种可转移的后门攻击。代码可在https://github.com/haowang-cqu/TransTroj上获得。



## **23. Data Defenses Against Large Language Models**

数据防御大型语言模型 cs.CL

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13138v1) [paper-pdf](http://arxiv.org/pdf/2410.13138v1)

**Authors**: William Agnew, Harry H. Jiang, Cella Sum, Maarten Sap, Sauvik Das

**Abstract**: Large language models excel at performing inference over text to extract information, summarize information, or generate additional text. These inference capabilities are implicated in a variety of ethical harms spanning surveillance, labor displacement, and IP/copyright theft. While many policy, legal, and technical mitigations have been proposed to counteract these harms, these mitigations typically require cooperation from institutions that move slower than technical advances (i.e., governments) or that have few incentives to act to counteract these harms (i.e., the corporations that create and profit from these LLMs). In this paper, we define and build "data defenses" -- a novel strategy that directly empowers data owners to block LLMs from performing inference on their data. We create data defenses by developing a method to automatically generate adversarial prompt injections that, when added to input text, significantly reduce the ability of LLMs to accurately infer personally identifying information about the subject of the input text or to use copyrighted text in inference. We examine the ethics of enabling such direct resistance to LLM inference, and argue that making data defenses that resist and subvert LLMs enables the realization of important values such as data ownership, data sovereignty, and democratic control over AI systems. We verify that our data defenses are cheap and fast to generate, work on the latest commercial and open-source LLMs, resistance to countermeasures, and are robust to several different attack settings. Finally, we consider the security implications of LLM data defenses and outline several future research directions in this area. Our code is available at https://github.com/wagnew3/LLMDataDefenses and a tool for using our defenses to protect text against LLM inference is at https://wagnew3.github.io/LLM-Data-Defenses/.

摘要: 大型语言模型擅长对文本执行推理，以提取信息、汇总信息或生成附加文本。这些推理能力牵涉到各种道德危害，包括监控、劳动力转移和知识产权/版权盗窃。虽然已经提出了许多政策、法律和技术缓解措施来抵消这些危害，但这些缓解措施通常需要行动速度慢于技术进步的机构(即政府)或几乎没有采取行动抵消这些危害的动机的机构(即创造这些低成本管理并从中获利的公司)的合作。在本文中，我们定义并构建了“数据防御”--一种新的策略，它直接授权数据所有者阻止LLM对其数据执行推理。我们通过开发一种自动生成对抗性提示注入的方法来创建数据防御，当这些注入添加到输入文本时，显著降低了LLMS准确推断关于输入文本主题的个人识别信息或在推理中使用受版权保护的文本的能力。我们审查了允许这种直接抵抗LLM推理的伦理，并认为，制定抵抗和颠覆LLM的数据防御能够实现重要的价值，如数据所有权、数据主权和对人工智能系统的民主控制。我们验证了我们的数据防御是廉价和快速生成的，在最新的商业和开源LLM上工作，对对策的抵抗力，以及对几种不同攻击设置的健壮性。最后，我们考虑了LLM数据防御的安全含义，并概述了该领域未来的几个研究方向。我们的代码可在https://github.com/wagnew3/LLMDataDefenses上获得，使用我们的防御措施保护文本免受LLm推断的工具可在https://wagnew3.github.io/LLM-Data-Defenses/.上获得



## **24. Degraded Polygons Raise Fundamental Questions of Neural Network Perception**

退化的多边形提出了神经网络感知的基本问题 cs.CV

Accepted as a conference paper to NeurIPS 2023 (Datasets & Benchmarks  Track)

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2306.04955v2) [paper-pdf](http://arxiv.org/pdf/2306.04955v2)

**Authors**: Leonard Tang, Dan Ley

**Abstract**: It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deep learning vision models suffer in a variety of settings that humans capably handle. In light of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images under degradation, first introduced over 30 years ago in the Recognition-by-Components theory of human vision. Specifically, we study the performance and behavior of neural networks on the seemingly simple task of classifying regular polygons at varying orders of degradation along their perimeters. To this end, we implement the Automated Shape Recoverability Test for rapidly generating large-scale datasets of perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity of neural networks to recognize and recover such degraded shapes when initialized with different priors. Ultimately, we find that neural networks' behavior on this simple task conflicts with human behavior, raising a fundamental question of the robustness and learning capabilities of modern computer vision models.

摘要: 众所周知，现代计算机视觉系统经常表现出与人类不一致的行为：从敌意攻击到图像损坏，深度学习视觉模型在人类能够处理的各种环境中受到影响。鉴于这些现象，我们在这里介绍了另一个研究人机视觉鸿沟的正交视角。我们回顾了30多年前在人类视觉的成分识别理论中首次引入的恢复退化图像的任务。具体地说，我们研究了神经网络在一项看似简单的任务中的性能和行为，该任务是对沿其周长以不同降级顺序的规则多边形进行分类。为此，我们实现了自动形状可恢复性测试，用于快速生成周长退化的规则多边形的大规模数据集，使历史上手动创建图像可恢复性实验的过程现代化。然后，我们研究了当用不同的先验进行初始化时，神经网络识别和恢复这些退化形状的能力。最终，我们发现神经网络在这一简单任务中的行为与人类行为相冲突，这引发了现代计算机视觉模型的稳健性和学习能力的根本问题。



## **25. Hiding-in-Plain-Sight (HiPS) Attack on CLIP for Targetted Object Removal from Images**

对CLIP进行隐藏式视线（HiPS）攻击，以从图像中删除目标对象 cs.LG

Published in the 3rd Workshop on New Frontiers in Adversarial Machine  Learning at NeurIPS 2024. 10 pages, 7 figures, 3 tables

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.13010v1) [paper-pdf](http://arxiv.org/pdf/2410.13010v1)

**Authors**: Arka Daw, Megan Hong-Thanh Chung, Maria Mahbub, Amir Sadovnik

**Abstract**: Machine learning models are known to be vulnerable to adversarial attacks, but traditional attacks have mostly focused on single-modalities. With the rise of large multi-modal models (LMMs) like CLIP, which combine vision and language capabilities, new vulnerabilities have emerged. However, prior work in multimodal targeted attacks aim to completely change the model's output to what the adversary wants. In many realistic scenarios, an adversary might seek to make only subtle modifications to the output, so that the changes go unnoticed by downstream models or even by humans. We introduce Hiding-in-Plain-Sight (HiPS) attacks, a novel class of adversarial attacks that subtly modifies model predictions by selectively concealing target object(s), as if the target object was absent from the scene. We propose two HiPS attack variants, HiPS-cls and HiPS-cap, and demonstrate their effectiveness in transferring to downstream image captioning models, such as CLIP-Cap, for targeted object removal from image captions.

摘要: 众所周知，机器学习模型容易受到对抗性攻击，但传统攻击大多集中在单一模式上。随着像CLIP这样结合了视觉和语言能力的大型多模式模型(LMM)的兴起，出现了新的漏洞。然而，以前在多模式定向攻击方面的工作旨在将模型的输出完全改变为对手想要的。在许多现实场景中，对手可能只寻求对输出进行微妙的修改，这样下游模型甚至人类都不会注意到这些变化。介绍了一种新型的对抗性攻击--视线隐藏攻击(HIPS)，它通过选择性地隐藏目标对象(S)来巧妙地修改模型预测，就好像目标对象不在场景中一样。我们提出了两个HIPS攻击变体，HIPS-CLS和HIPS-CAP，并证明了它们在转移到下游图像字幕模型(如CLIP-Cap)以从图像字幕中去除目标对象方面的有效性。



## **26. TMI! Finetuned Models Leak Private Information from their Pretraining Data**

TMI！Finetuned模型从其预训练数据中泄露私人信息 cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2306.01181v3) [paper-pdf](http://arxiv.org/pdf/2306.01181v3)

**Authors**: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman

**Abstract**: Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for $\textit{privacy}$ in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, $\textbf{TMI}$, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate $\textbf{TMI}$ on both vision and natural language tasks across multiple transfer learning settings, including finetuning with differential privacy. Through our evaluation, we find that $\textbf{TMI}$ can successfully infer membership of pretraining examples using query access to the finetuned model. An open-source implementation of $\textbf{TMI}$ can be found on GitHub: https://github.com/johnmath/tmi-pets24.

摘要: 迁移学习已经成为机器学习中一种越来越流行的技术，作为一种利用为一个任务训练的预先训练的模型来帮助构建相关任务的精调模型的一种方式。这种范例在机器学习中尤其流行，在机器学习中，预先训练的模型被认为是公共的，只有用于精细调整的数据被认为是敏感的。然而，有理由相信，用于预培训的数据仍然敏感，因此了解精调模型泄露了多少关于预培训数据的信息是至关重要的。在这项工作中，我们提出了一个新的成员资格推理威胁模型，其中对手只能访问精调的模型，并希望推断预训练数据的成员资格。为了实现这种威胁模型，我们实现了一种新的基于元分类器的攻击，它利用已记忆的预训练样本对下游任务预测的影响。我们在多个迁移学习环境中对视觉和自然语言任务进行了评估，包括在不同隐私条件下的精细调整。通过我们的评估，我们发现$\textbf{TMI}$可以成功地使用对精调模型的查询访问来推断预训练样本的隶属度。$\Textbf{tmi}$的开源实现可在giHub上找到：https://github.com/johnmath/tmi-pets24.



## **27. Adversarial Training of Two-Layer Polynomial and ReLU Activation Networks via Convex Optimization**

通过凸优化进行两层多边形和ReLU激活网络的对抗训练 cs.LG

17 pages, 2 figures. Added a proof of the main theorem in the  appendix. Expanded numerical results section. Added references

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2405.14033v2) [paper-pdf](http://arxiv.org/pdf/2405.14033v2)

**Authors**: Daniel Kuelbs, Sanjay Lall, Mert Pilanci

**Abstract**: Training neural networks which are robust to adversarial attacks remains an important problem in deep learning, especially as heavily overparameterized models are adopted in safety-critical settings. Drawing from recent work which reformulates the training problems for two-layer ReLU and polynomial activation networks as convex programs, we devise a convex semidefinite program (SDP) for adversarial training of two-layer polynomial activation networks and prove that the convex SDP achieves the same globally optimal solution as its nonconvex counterpart. The convex SDP is observed to improve robust test accuracy against $\ell_\infty$ attacks relative to the original convex training formulation on multiple datasets. Additionally, we present scalable implementations of adversarial training for two-layer polynomial and ReLU networks which are compatible with standard machine learning libraries and GPU acceleration. Leveraging these implementations, we retrain the final two fully connected layers of a Pre-Activation ResNet-18 model on the CIFAR-10 dataset with both polynomial and ReLU activations. The two `robustified' models achieve significantly higher robust test accuracies against $\ell_\infty$ attacks than a Pre-Activation ResNet-18 model trained with sharpness-aware minimization, demonstrating the practical utility of convex adversarial training on large-scale problems.

摘要: 训练对敌意攻击稳健的神经网络仍然是深度学习中的一个重要问题，特别是在安全关键环境中采用严重过度参数模型的情况下。借鉴最近将两层RELU和多项式激活网络的训练问题转化为凸规划的工作，我们设计了一个用于两层多项式激活网络对抗训练的凸半定规划(SDP)，并证明了凸SDP与非凸SDP具有相同的全局最优解。与原始的凸训练公式相比，凸SDP在多个数据集上提高了对$\ell_\inty$攻击的稳健测试精度。此外，我们还提出了与标准机器学习库和GPU加速兼容的两层多项式网络和RELU网络的对抗性训练的可扩展实现。利用这些实现，我们在CIFAR-10数据集上重新训练激活前ResNet-18模型的最后两个完全连接层，同时使用多项式和RELU激活。与激活前采用锐度感知最小化训练的ResNet-18模型相比，这两个“强壮化”模型对攻击具有显著更高的稳健测试精度，表明了凸对抗性训练在大规模问题上的实用价值。



## **28. Unitary Multi-Margin BERT for Robust Natural Language Processing**

用于鲁棒自然语言处理的个位多余量BERT cs.CL

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12759v1) [paper-pdf](http://arxiv.org/pdf/2410.12759v1)

**Authors**: Hao-Yuan Chang, Kang L. Wang

**Abstract**: Recent developments in adversarial attacks on deep learning leave many mission-critical natural language processing (NLP) systems at risk of exploitation. To address the lack of computationally efficient adversarial defense methods, this paper reports a novel, universal technique that drastically improves the robustness of Bidirectional Encoder Representations from Transformers (BERT) by combining the unitary weights with the multi-margin loss. We discover that the marriage of these two simple ideas amplifies the protection against malicious interference. Our model, the unitary multi-margin BERT (UniBERT), boosts post-attack classification accuracies significantly by 5.3% to 73.8% while maintaining competitive pre-attack accuracies. Furthermore, the pre-attack and post-attack accuracy tradeoff can be adjusted via a single scalar parameter to best fit the design requirements for the target applications.

摘要: 深度学习对抗性攻击的最新发展使许多任务关键型自然语言处理（NLP）系统面临被利用的风险。为了解决缺乏计算高效的对抗性防御方法的问题，本文报告了一种新颖的通用技术，该技术通过将单位权重与多裕度损失相结合，大幅提高了来自变形器的双向编码器表示（BERT）的鲁棒性。我们发现这两个简单想法的结合增强了对恶意干扰的保护。我们的模型，即单一多裕度BERT（UniBERT），将攻击后分类准确性显着提高了5.3%至73.8%，同时保持有竞争力的攻击前准确性。此外，攻击前和攻击后的准确性权衡可以通过单个纯量参数进行调整，以最好地满足目标应用程序的设计要求。



## **29. ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection**

ToBlend：与LLM集合进行令牌级混合以攻击AI生成的文本检测 cs.CL

Submitted to ARR Oct-2024 Cycle

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.11167v2) [paper-pdf](http://arxiv.org/pdf/2402.11167v2)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An

**Abstract**: The robustness of AI-content detection models against sophisticated adversarial strategies, such as paraphrasing or word switching, is a rising concern in natural language generation (NLG) applications. This study proposes ToBlend, a novel token-level ensemble text generation method to challenge the robustness of current AI-content detection approaches by utilizing multiple sets of candidate generative large language models (LLMs). By randomly sampling token(s) from candidate LLMs sets, we find ToBlend significantly drops the performance of most mainstream AI-content detection methods. We evaluate the text quality produced under different ToBlend settings based on annotations from experienced human experts. We proposed a fine-tuned Llama3.1 model to distinguish the ToBlend generated text more accurately. Our findings underscore our proposed text generation approach's great potential in deceiving and improving detection models. Our datasets, codes, and annotations are open-sourced.

摘要: 人工智能内容检测模型对重述或单词切换等复杂对抗策略的稳健性是自然语言生成（NLG）应用程序中日益关注的问题。这项研究提出了ToBlend，这是一种新型的代币级集成文本生成方法，通过利用多组候选生成式大型语言模型（LLM）来挑战当前人工智能内容检测方法的稳健性。通过从候选LLM集中随机采样令牌，我们发现ToBlend显着降低了大多数主流AI内容检测方法的性能。我们根据经验丰富的人类专家的注释来评估不同ToBlend设置下生成的文本质量。我们提出了一个微调的Llama3.1模型，以更准确地区分ToBlend生成的文本。我们的发现强调了我们提出的文本生成方法在欺骗和改进检测模型方面的巨大潜力。我们的数据集、代码和注释是开源的。



## **30. Low-Rank Adversarial PGD Attack**

低级对抗性PVD攻击 cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12607v1) [paper-pdf](http://arxiv.org/pdf/2410.12607v1)

**Authors**: Dayana Savostianova, Emanuele Zangrando, Francesco Tudisco

**Abstract**: Adversarial attacks on deep neural network models have seen rapid development and are extensively used to study the stability of these networks. Among various adversarial strategies, Projected Gradient Descent (PGD) is a widely adopted method in computer vision due to its effectiveness and quick implementation, making it suitable for adversarial training. In this work, we observe that in many cases, the perturbations computed using PGD predominantly affect only a portion of the singular value spectrum of the original image, suggesting that these perturbations are approximately low-rank. Motivated by this observation, we propose a variation of PGD that efficiently computes a low-rank attack. We extensively validate our method on a range of standard models as well as robust models that have undergone adversarial training. Our analysis indicates that the proposed low-rank PGD can be effectively used in adversarial training due to its straightforward and fast implementation coupled with competitive performance. Notably, we find that low-rank PGD often performs comparably to, and sometimes even outperforms, the traditional full-rank PGD attack, while using significantly less memory.

摘要: 针对深度神经网络模型的对抗性攻击得到了迅速的发展，并被广泛地用于研究这些网络的稳定性。在各种对抗策略中，投影梯度下降(PGD)方法因其有效性和快速实现而被广泛应用于计算机视觉中，适合于对抗训练。在这项工作中，我们观察到在许多情况下，使用PGD计算的扰动只影响原始图像的奇异值频谱的一部分，这表明这些扰动是近似低阶的。基于这一观察结果，我们提出了一种有效计算低等级攻击的PGD的变体。我们在一系列标准模型以及经过对抗性训练的稳健模型上广泛验证了我们的方法。我们的分析表明，由于其简单、快速的实现以及具有竞争力的性能，所提出的低等级PGD可以有效地用于对抗性训练。值得注意的是，我们发现低级PGD攻击的性能通常与传统的全级PGD攻击相当，有时甚至超过传统的全级PGD攻击，而使用的内存要少得多。



## **31. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

针对视觉语言预训练模型的高效且有效的通用对抗攻击 cs.CV

11 pages

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.11639v2) [paper-pdf](http://arxiv.org/pdf/2410.11639v2)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.

摘要: 视觉-语言预训练模型是在大规模图文对上训练的，已被广泛应用于各种下游视觉与语言(V+L)任务。这种广泛的采用引起了人们对它们易受对手攻击的担忧。非通用对抗性攻击虽然有效，但对于实时在线应用程序来说往往是不切实际的，因为它们对每个数据实例的计算要求很高。最近，通用对抗扰动(UAP)被引入作为解决方案，但现有的基于生成器的UAP方法非常耗时。为了克服这一局限性，我们提出了一种基于直接优化的UAP方法，称为DO-UAP，它在保持高攻击性能的同时显著减少了资源消耗。具体地说，我们探讨了多峰损失设计的必要性，并介绍了一种有用的数据增强策略。在三个基准VLP数据集、六个流行的VLP模型和三个经典下游任务上的广泛实验证明了DO-UAP的效率和有效性。具体地说，我们的方法大大减少了23倍的时间消耗，同时实现了更好的攻击性能。



## **32. A Proactive Decoy Selection Scheme for Cyber Deception using MITRE ATT&CK**

使用MITRE ATT & CK的网络欺骗主动诱饵选择方案 cs.CR

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2404.12783v3) [paper-pdf](http://arxiv.org/pdf/2404.12783v3)

**Authors**: Marco Zambianco, Claudio Facchinetti, Domenico Siracusa

**Abstract**: Cyber deception allows compensating the late response of defenders countermeasures to the ever evolving tactics, techniques, and procedures (TTPs) of attackers. This proactive defense strategy employs decoys resembling legitimate system components to lure stealthy attackers within the defender environment, slowing and/or denying the accomplishment of their goals. In this regard, the selection of decoys that can expose the techniques used by malicious users plays a central role to incentivize their engagement. However, this is a difficult task to achieve in practice, since it requires an accurate and realistic modeling of the attacker capabilities and his possible targets. In this work, we tackle this challenge and we design a decoy selection scheme that is supported by an adversarial modeling based on empirical observation of real-world attackers. We take advantage of a domain-specific threat modelling language using MITRE ATT&CK framework as source of attacker TTPs targeting enterprise systems. In detail, we extract the information about the execution preconditions of each technique as well as its possible effects on the environment to generate attack graphs modeling the adversary capabilities. Based on this, we formulate a graph partition problem that minimizes the number of decoys detecting a corresponding number of techniques employed in various attack paths directed to specific targets. We compare our optimization-based decoy selection approach against several benchmark schemes that ignore the preconditions between the various attack steps. Results reveal that the proposed scheme provides the highest interception rate of attack paths using the lowest amount of decoys.

摘要: 网络欺骗可以补偿防御者对攻击者不断演变的战术、技术和程序(TTP)的反应迟缓。这种主动防御策略使用类似于合法系统组件的诱饵，在防御者环境中引诱隐形攻击者，减缓和/或拒绝他们目标的实现。在这方面，选择能够揭露恶意用户使用的技术的诱饵对激励他们的参与起着核心作用。然而，这在实践中是一项困难的任务，因为它需要对攻击者的能力及其可能的目标进行准确和现实的建模。在这项工作中，我们解决了这一挑战，并设计了一个诱饵选择方案，该方案由基于对真实世界攻击者的经验观察的对抗性建模来支持。我们利用一种特定于领域的威胁建模语言，使用MITRE ATT&CK框架作为针对企业系统的攻击者TTP的来源。详细地，我们提取了关于每种技术的执行前提及其对环境的可能影响的信息，以生成模拟对手能力的攻击图。在此基础上，我们提出了一个图划分问题，该问题最小化了在针对特定目标的各种攻击路径中检测到相应数量的技术的诱饵数量。我们将我们基于优化的诱饵选择方法与几个忽略攻击步骤之间的前提条件的基准方案进行了比较。结果表明，该方案以最少的诱饵获得了最高的攻击路径拦截率。



## **33. Query Provenance Analysis: Efficient and Robust Defense against Query-based Black-box Attacks**

查询来源分析：针对基于查询的黑匣子攻击的高效而稳健的防御 cs.CR

The final version of this paper is going to appear in IEEE Symposium  on Security and Privacy 2025

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2405.20641v2) [paper-pdf](http://arxiv.org/pdf/2405.20641v2)

**Authors**: Shaofei Li, Ziqi Zhang, Haomin Jia, Ding Li, Yao Guo, Xiangqun Chen

**Abstract**: Query-based black-box attacks have emerged as a significant threat to machine learning systems, where adversaries can manipulate the input queries to generate adversarial examples that can cause misclassification of the model. To counter these attacks, researchers have proposed Stateful Defense Models (SDMs) for detecting adversarial query sequences and rejecting queries that are "similar" to the history queries. Existing state-of-the-art (SOTA) SDMs (e.g., BlackLight and PIHA) have shown great effectiveness in defending against these attacks. However, recent studies have shown that they are vulnerable to Oracle-guided Adaptive Rejection Sampling (OARS) attacks, which is a stronger adaptive attack strategy. It can be easily integrated with existing attack algorithms to evade the SDMs by generating queries with fine-tuned direction and step size of perturbations utilizing the leaked decision information from the SDMs.   In this paper, we propose a novel approach, Query Provenance Analysis (QPA), for more robust and efficient SDMs. QPA encapsulates the historical relationships among queries as the sequence feature to capture the fundamental difference between benign and adversarial query sequences. To utilize the query provenance, we propose an efficient query provenance analysis algorithm with dynamic management. We evaluate QPA compared with two baselines, BlackLight and PIHA, on four widely used datasets with six query-based black-box attack algorithms. The results show that QPA outperforms the baselines in terms of defense effectiveness and efficiency on both non-adaptive and adaptive attacks. Specifically, QPA reduces the Attack Success Rate (ASR) of OARS to 4.08%, comparing to 77.63% and 87.72% for BlackLight and PIHA, respectively. Moreover, QPA also achieves 7.67x and 2.25x higher throughput than BlackLight and PIHA.

摘要: 基于查询的黑盒攻击已经成为对机器学习系统的重大威胁，在机器学习系统中，攻击者可以操纵输入查询来生成可能导致模型错误分类的对抗性示例。为了对抗这些攻击，研究人员提出了状态防御模型(SDMS)来检测敌意查询序列并拒绝与历史查询“相似”的查询。现有的最先进的(SOTA)SDMS(例如Blacklight和Piha)在防御这些攻击方面表现出了巨大的有效性。然而，最近的研究表明，它们容易受到Oracle引导的自适应拒绝采样(OARS)攻击，这是一种更强的自适应攻击策略。它可以很容易地与现有的攻击算法集成，通过利用SDMS泄露的决策信息生成具有微调的扰动方向和步长的查询来规避SDMS。在本文中，我们提出了一种新的方法-查询起源分析(QPA)，以实现更健壮和高效的SDMS。QPA将查询之间的历史关系封装为序列特征，以捕捉良性查询序列和恶意查询序列之间的根本区别。为了充分利用查询起源，提出了一种高效的动态管理的查询起源分析算法。我们在四个广泛使用的数据集上用六种基于查询的黑盒攻击算法对QPA进行了评估，并与Blacklight和PIHA两种基线进行了比较。结果表明，无论是非适应性攻击还是适应性攻击，QPA在防御效果和效率方面都优于基线。具体地说，QPA使桨的攻击成功率(ASR)降至4.08%，而Blacklight和Piha的攻击成功率分别为77.63%和87.72%。此外，QPA的吞吐量也比Blacklight和PIHA高7.67倍和2.25倍。



## **34. Perseus: Leveraging Common Data Patterns with Curriculum Learning for More Robust Graph Neural Networks**

英仙座：利用常见数据模式与课程学习来实现更稳健的图神经网络 cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12425v1) [paper-pdf](http://arxiv.org/pdf/2410.12425v1)

**Authors**: Kaiwen Xia, Huijun Wu, Duanyu Li, Min Xie, Ruibo Wang, Wenzhe Zhang

**Abstract**: Graph Neural Networks (GNNs) excel at handling graph data but remain vulnerable to adversarial attacks. Existing defense methods typically rely on assumptions like graph sparsity and homophily to either preprocess the graph or guide structure learning. However, preprocessing methods often struggle to accurately distinguish between normal edges and adversarial perturbations, leading to suboptimal results due to the loss of valuable edge information. Robust graph neural network models train directly on graph data affected by adversarial perturbations, without preprocessing. This can cause the model to get stuck in poor local optima, negatively affecting its performance. To address these challenges, we propose Perseus, a novel adversarial defense method based on curriculum learning. Perseus assesses edge difficulty using global homophily and applies a curriculum learning strategy to adjust the learning order, guiding the model to learn the full graph structure while adaptively focusing on common data patterns. This approach mitigates the impact of adversarial perturbations. Experiments show that models trained with Perseus achieve superior performance and are significantly more robust to adversarial attacks.

摘要: 图形神经网络(GNN)擅长处理图形数据，但仍然容易受到对手的攻击。现有的防御方法通常依赖于图的稀疏性和同质性等假设来对图进行预处理或指导结构学习。然而，由于丢失了有价值的边缘信息，预处理方法往往难以准确区分正常边缘和对抗性扰动，导致结果不是最优的。健壮的图神经网络模型直接在受对抗性扰动影响的图数据上训练，不需要进行预处理。这可能会导致模型陷入较差的局部最优，从而对其性能产生负面影响。为了应对这些挑战，我们提出了一种新的基于课程学习的对抗性防御方法Perseus。Perseus使用全局同质性来评估边缘难度，并应用课程学习策略来调整学习顺序，引导模型学习完整的图形结构，同时自适应地专注于常见的数据模式。这种方法减轻了对抗性扰动的影响。实验表明，使用Perseus训练的模型取得了更好的性能，并且对对手攻击具有更强的鲁棒性。



## **35. DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in Frequency Domain**

DART：通过频域生成幅度混合提高对抗鲁棒性 cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2410.12307v1) [paper-pdf](http://arxiv.org/pdf/2410.12307v1)

**Authors**: Fengpeng Li, Kemou Li, Haiwei Wu, Jinyu Tian, Jiantao Zhou

**Abstract**: To protect deep neural networks (DNNs) from adversarial attacks, adversarial training (AT) is developed by incorporating adversarial examples (AEs) into model training. Recent studies show that adversarial attacks disproportionately impact the patterns within the phase of the sample's frequency spectrum -- typically containing crucial semantic information -- more than those in the amplitude, resulting in the model's erroneous categorization of AEs. We find that, by mixing the amplitude of training samples' frequency spectrum with those of distractor images for AT, the model can be guided to focus on phase patterns unaffected by adversarial perturbations. As a result, the model's robustness can be improved. Unfortunately, it is still challenging to select appropriate distractor images, which should mix the amplitude without affecting the phase patterns. To this end, in this paper, we propose an optimized Adversarial Amplitude Generator (AAG) to achieve a better tradeoff between improving the model's robustness and retaining phase patterns. Based on this generator, together with an efficient AE production procedure, we design a new Dual Adversarial Training (DAT) strategy. Experiments on various datasets show that our proposed DAT leads to significantly improved robustness against diverse adversarial attacks.

摘要: 为了保护深层神经网络(DNN)免受对抗性攻击，将对抗性实例(AES)融入到模型训练中，提出了对抗性训练(AT)方法。最近的研究表明，对抗性攻击对样本频谱阶段内的模式--通常包含关键语义信息--的影响不成比例地大于幅度内的模式，导致模型对AE的错误分类。我们发现，通过将训练样本频谱的幅度与AT的分心图像的频谱幅度混合，该模型可以被引导到不受对抗性扰动影响的相位模式。从而提高了模型的稳健性。不幸的是，选择合适的干扰图像仍然具有挑战性，它应该在不影响相位模式的情况下混合幅度。为此，在本文中，我们提出了一种优化的对抗性幅度产生器(AAG)，以在提高模型的稳健性和保持相位模式之间实现更好的折衷。基于该生成器，结合高效的AE生成过程，我们设计了一种新的双重对抗性训练(DAT)策略。在不同数据集上的实验表明，我们提出的DAT能够显著提高对不同对手攻击的稳健性。



## **36. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

MixedNUTS：通过非线性混合分类器实现免训练的准确性-稳健性平衡 cs.LG

**SubmitDate**: 2024-10-16    [abs](http://arxiv.org/abs/2402.02263v5) [paper-pdf](http://arxiv.org/pdf/2402.02263v5)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding real-life applications of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.

摘要: 对抗性的稳健性往往是以降低精度为代价的，这阻碍了稳健分类模型的实际应用。基于培训的更好权衡的解决方案受到与已经培训的高性能大型模型不兼容的限制，因此有必要探索无需培训的整体方法。我们观察到，稳健模型在正确预测中的信心比基于干净和敌对数据的不正确预测更有信心，我们推测，放大这种“良性置信度属性”可以在整体设置中调和准确性和稳健性。为了实现这一点，我们提出了一种无需训练的方法“MixedNUTS”，其中稳健分类器和标准非稳健分类器的输出逻辑通过只有三个参数的非线性变换来处理，并通过有效的算法进行优化。MixedNUTS然后将转换后的Logit转换为概率，并将它们混合为整体输出。在CIFAR-10、CIFAR-100和ImageNet数据集上，自定义强自适应攻击的实验结果表明，MixedNUTS的精确度和接近SOTA的稳健性都得到了极大的提高--它将CIFAR-100的干净精确度提高了7.86个点，而健壮精确度仅牺牲了0.87个点。



## **37. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

GPT-4通过自我解释以近乎完美的成功越狱 cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.13077v2) [paper-pdf](http://arxiv.org/pdf/2405.13077v2)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find that IRIS achieves jailbreak success rates of 98% on GPT-4, 92% on GPT-4 Turbo, and 94% on Llama-3.1-70B in under 7 queries. It significantly outperforms prior approaches in automatic, black-box, and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.

摘要: 越狱研究对于测试和理解大型语言模型(LLM)的安全和安保问题具有重要价值。在本文中，我们介绍了迭代精化诱导的自越狱(IRIS)，这是一种新的方法，它利用LLMS的反射能力来实现只访问黑盒的越狱。与以前的方法不同，IRIS通过将单一模型用作攻击者和目标来简化越狱过程。这种方法首先通过自我解释迭代地精炼对抗性提示，这对于确保即使是排列良好的LLM也遵守对抗性指令至关重要。然后，虹膜给出精致的提示，对产量进行评级并提高产量，以增加其危害性。我们发现，IRIS在GPT-4上的越狱成功率为98%，在GPT-4Turbo上的越狱成功率为92%，在Llama-3.1-70B上的越狱成功率为94%。它在自动、黑盒和可解释越狱方面显著优于现有方法，同时需要的查询大大减少，从而建立了可解释越狱方法的新标准。



## **38. Taking off the Rose-Tinted Glasses: A Critical Look at Adversarial ML Through the Lens of Evasion Attacks**

摘下玫瑰色眼镜：从逃避攻击的角度批判性地审视对抗性ML cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.12076v1) [paper-pdf](http://arxiv.org/pdf/2410.12076v1)

**Authors**: Kevin Eykholt, Farhan Ahmed, Pratik Vaishnavi, Amir Rahmati

**Abstract**: The vulnerability of machine learning models in adversarial scenarios has garnered significant interest in the academic community over the past decade, resulting in a myriad of attacks and defenses. However, while the community appears to be overtly successful in devising new attacks across new contexts, the development of defenses has stalled. After a decade of research, we appear no closer to securing AI applications beyond additional training. Despite a lack of effective mitigations, AI development and its incorporation into existing systems charge full speed ahead with the rise of generative AI and large language models. Will our ineffectiveness in developing solutions to adversarial threats further extend to these new technologies?   In this paper, we argue that overly permissive attack and overly restrictive defensive threat models have hampered defense development in the ML domain. Through the lens of adversarial evasion attacks against neural networks, we critically examine common attack assumptions, such as the ability to bypass any defense not explicitly built into the model. We argue that these flawed assumptions, seen as reasonable by the community based on paper acceptance, have encouraged the development of adversarial attacks that map poorly to real-world scenarios. In turn, new defenses evaluated against these very attacks are inadvertently required to be almost perfect and incorporated as part of the model. But do they need to? In practice, machine learning models are deployed as a small component of a larger system. We analyze adversarial machine learning from a system security perspective rather than an AI perspective and its implications for emerging AI paradigms.

摘要: 在过去的十年里，机器学习模型在对抗性场景中的脆弱性引起了学术界的极大兴趣，导致了无数的攻击和防御。然而，尽管该社区似乎公开成功地在新的背景下设计了新的攻击，但防御的发展却停滞不前。经过十年的研究，除了额外的培训外，我们似乎并没有更进一步地确保人工智能应用的安全。尽管缺乏有效的缓解措施，但随着生成性人工智能和大型语言模型的崛起，人工智能的发展及其与现有系统的结合全速前进。我们在开发对抗威胁的解决方案方面的无效性是否会进一步延伸到这些新技术？在本文中，我们认为过度允许的攻击和过度受限的防御威胁模型阻碍了ML领域的防御发展。通过针对神经网络的对抗性逃避攻击的镜头，我们批判性地检查了常见的攻击假设，例如绕过未显式构建在模型中的任何防御的能力。我们认为，这些有缺陷的假设被社区视为基于论文接受的合理假设，鼓励了与现实世界情景映射不佳的对抗性攻击的发展。反过来，针对这些攻击评估的新防御措施被无意中要求近乎完美，并作为模型的一部分纳入。但他们真的需要这样做吗？在实践中，机器学习模型被部署为更大系统的一个小组件。我们从系统安全的角度分析对抗性机器学习，而不是从人工智能的角度分析它对新兴人工智能范例的影响。



## **39. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **40. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

Phantom：对检索增强语言生成的通用触发攻击 cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".

摘要: 检索增强生成(RAG)通过锚定、调整和个性化对最相关的知识源的响应来扩展现代大型语言模型(LLMS)的能力。它在聊天机器人应用程序中特别有用，允许开发人员定制LLM输出，而无需昂贵的再培训。尽管RAG系统在各种应用中具有重要的实用价值，但它带来了新的安全风险。在这项工作中，我们提出了新的攻击向量，允许攻击者将单个恶意文档注入RAG系统的知识库，并发动后门中毒攻击。我们设计了Phantom，这是一个针对RAG系统的通用两阶段优化框架，它手工制作了一个恶意中毒文档，导致模型输出中的完整性破坏。首先，文档被构建为仅在受害者的查询中出现特定的令牌触发序列时才检索。其次，通过精心设计的敌意文本进一步优化了文档，这些文本在LLM输出上诱导了各种敌意目标，包括拒绝回答、声誉损害、侵犯隐私和有害行为。我们演示了我们对多个LLM体系结构的攻击，包括Gema、Vicuna和Llama，并表明它们可以传输到GPT-3.5Turbo和GPT-4。最后，我们成功地对NVIDIA的黑匣子生产RAG系统“与腾讯通聊天”进行了幻影攻击。



## **41. Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor**

通过注入主动防御后门来缓解后门攻击 cs.CR

Accepted by NeurIPS 2024. 32 pages, 7 figures, 28 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.16112v2) [paper-pdf](http://arxiv.org/pdf/2405.16112v2)

**Authors**: Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Data-poisoning backdoor attacks are serious security threats to machine learning models, where an adversary can manipulate the training dataset to inject backdoors into models. In this paper, we focus on in-training backdoor defense, aiming to train a clean model even when the dataset may be potentially poisoned. Unlike most existing methods that primarily detect and remove/unlearn suspicious samples to mitigate malicious backdoor attacks, we propose a novel defense approach called PDB (Proactive Defensive Backdoor). Specifically, PDB leverages the home-field advantage of defenders by proactively injecting a defensive backdoor into the model during training. Taking advantage of controlling the training process, the defensive backdoor is designed to suppress the malicious backdoor effectively while remaining secret to attackers. In addition, we introduce a reversible mapping to determine the defensive target label. During inference, PDB embeds a defensive trigger in the inputs and reverses the model's prediction, suppressing malicious backdoor and ensuring the model's utility on the original task. Experimental results across various datasets and models demonstrate that our approach achieves state-of-the-art defense performance against a wide range of backdoor attacks. The code is available at https://github.com/shawkui/Proactive_Defensive_Backdoor.

摘要: 数据中毒后门攻击是对机器学习模型的严重安全威胁，攻击者可以操纵训练数据集向模型注入后门。在本文中，我们将重点放在训练中的后门防御上，目的是训练一个干净的模型，即使数据集可能被毒化。不同于现有的大多数方法主要是检测和删除/取消学习可疑样本来缓解恶意后门攻击，我们提出了一种称为主动防御后门的新防御方法。具体地说，PDB通过在训练期间主动向模型中注入防守后门来利用后卫的主场优势。利用控制训练过程的优势，防御性后门被设计成在对攻击者保密的同时有效地抑制恶意后门。此外，我们还引入了一种可逆映射来确定防御目标标签。在推理过程中，PDB在输入中嵌入一个防御触发器，逆转模型的预测，抑制恶意后门，确保模型在原始任务上的实用性。在不同的数据集和模型上的实验结果表明，我们的方法在抵抗广泛的后门攻击时获得了最先进的防御性能。代码可在https://github.com/shawkui/Proactive_Defensive_Backdoor.上获得



## **42. Security of and by Generative AI platforms**

生成性人工智能平台的安全性 cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.13899v1) [paper-pdf](http://arxiv.org/pdf/2410.13899v1)

**Authors**: Hari Hayagreevan, Souvik Khamaru

**Abstract**: This whitepaper highlights the dual importance of securing generative AI (genAI) platforms and leveraging genAI for cybersecurity. As genAI technologies proliferate, their misuse poses significant risks, including data breaches, model tampering, and malicious content generation. Securing these platforms is critical to protect sensitive data, ensure model integrity, and prevent adversarial attacks. Simultaneously, genAI presents opportunities for enhancing security by automating threat detection, vulnerability analysis, and incident response. The whitepaper explores strategies for robust security frameworks around genAI systems, while also showcasing how genAI can empower organizations to anticipate, detect, and mitigate sophisticated cyber threats.

摘要: 本白皮书强调了保护生成性人工智能（genAI）平台和利用genAI实现网络安全的双重重要性。随着genAI技术的激增，它们的滥用带来了重大风险，包括数据泄露、模型篡改和恶意内容生成。保护这些平台的安全对于保护敏感数据、确保模型完整性和防止对抗性攻击至关重要。同时，genAI通过自动化威胁检测、漏洞分析和事件响应来增强安全性。该白皮书探讨了围绕genAI系统的强大安全框架的策略，同时还展示了genAI如何使组织能够预测、检测和缓解复杂的网络威胁。



## **43. GSE: Group-wise Sparse and Explainable Adversarial Attacks**

GSE：分组稀疏和可解释的对抗性攻击 cs.CV

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2311.17434v2) [paper-pdf](http://arxiv.org/pdf/2311.17434v2)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2-$quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2-$norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved explainability, and a $100\%$ attack success rate.

摘要: 稀疏敌意攻击通过最小的像素扰动来欺骗深度神经网络(DNN)，这种扰动通常由$\ell_0$范数来正则化。最近的努力已经用结构稀疏性正则化规则取代了这一规范，例如核集团规范，以制定群组稀疏对抗性攻击。因此，由此产生的扰动是可以解释的，并具有重要的实际意义，揭示了DNN更大的脆弱性。然而，精心设计这样的攻击构成了一个优化挑战，因为它涉及到计算非凸目标内的像素组的规范。我们通过提出一个两阶段算法来解决这个问题，该算法在图像的语义有意义的区域内生成分组稀疏攻击。首先，我们使用为非凸规划量身定做的$1/2-$拟正态近似算子来优化拟正态对抗性损失。随后，算法过渡到投影的内斯特罗夫加速梯度下降，并对摄动幅度应用$2-$范数正则化。在CIFAR-10和ImageNet数据集上的严格评估表明，组内稀疏性显著增加，例如，CIFAR-10上的稀疏度为50.9美元，ImageNet上的稀疏度为38.4美元(平均案例，有针对性的攻击)。伴随着这种性能改进的是显著更快的计算时间、更好的可解释性以及$100\$攻击成功率。



## **44. Information Importance-Aware Defense against Adversarial Attack for Automatic Modulation Classification:An XAI-Based Approach**

自动调制分类的信息重要性感知对抗攻击防御：一种基于XAI的方法 eess.SP

Accepted by WCSP 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11608v1) [paper-pdf](http://arxiv.org/pdf/2410.11608v1)

**Authors**: Jingchun Wang, Peihao Dong, Fuhui Zhou, Qihui Wu

**Abstract**: Deep learning (DL) has significantly improved automatic modulation classification (AMC) by leveraging neural networks as the feature extractor.However, as the DL-based AMC becomes increasingly widespread, it is faced with the severe secure issue from various adversarial attacks. Existing defense methods often suffer from the high computational cost, intractable parameter tuning, and insufficient robustness.This paper proposes an eXplainable artificial intelligence (XAI) defense approach, which uncovers the negative information caused by the adversarial attack through measuring the importance of input features based on the SHapley Additive exPlanations (SHAP).By properly removing the negative information in adversarial samples and then fine-tuning(FT) the model, the impact of the attacks on the classification result can be mitigated.Experimental results demonstrate that the proposed SHAP-FT improves the classification performance of the model by 15%-20% under different attack levels,which not only enhances model robustness against various attack levels but also reduces the resource consumption, validating its effectiveness in safeguarding communication networks.

摘要: 深度学习利用神经网络作为特征提取工具，极大地改善了自动调制分类算法的性能，但随着基于深度学习的自动调制分类算法的应用越来越广泛，它也面临着严峻的安全问题。针对现有防御方法计算量大、参数整定困难、鲁棒性不足等问题，提出了一种基于Shapley附加解释(Shap)的可解释人工智能(XAI)防御方法。该方法基于Shapley附加解释(Shap)度量输入特征的重要性来揭示敌方攻击带来的负面信息，通过适当去除敌方样本中的负面信息并对模型进行微调，可以缓解攻击对分类结果的影响。实验结果表明，在不同攻击级别下，Shap-FT使模型的分类性能提高了15%-20%。这不仅增强了模型对各种攻击级别的健壮性，而且降低了资源消耗，验证了其在保护通信网络方面的有效性。



## **45. RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation**

RAUCA：通过稳健而准确的伪装生成对车辆检测器的新型物理对抗攻击 cs.CV

12 pages. In Proceedings of the Forty-first International Conference  on Machine Learning (ICML), Vienna, Austria, July 21-27, 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2402.15853v2) [paper-pdf](http://arxiv.org/pdf/2402.15853v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle, resulting in suboptimal attack performance. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, Neural Renderer Plus (NRP), which can accurately project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA consistently outperforms existing methods in both simulation and real-world settings.

摘要: 对抗伪装是一种广泛使用的针对车辆探测器的物理攻击，具有多视点攻击性能的优势。一种有希望的方法包括使用可微神经呈现器通过梯度反向传播来促进对抗性伪装优化。然而，现有的方法往往难以捕捉渲染过程中的环境特征，或者生成能够精确映射到目标车辆的对抗性纹理，导致攻击性能次优。此外，这些方法忽略了不同的天气条件，降低了在不同天气情况下产生的伪装效果。为了应对这些挑战，我们提出了一种健壮而准确的伪装生成方法，即Ruca。Ruca的核心是一个新的神经渲染组件-神经渲染器Plus(NRP)，它可以准确地投影车辆纹理，并渲染具有照明和天气等环境特征的图像。此外，我们还集成了一个用于伪装生成的多天气数据集，利用NRP来增强攻击的健壮性。在六个流行的目标探测器上的实验结果表明，无论是在模拟环境中还是在现实世界中，Ruca的性能都一致优于现有的方法。



## **46. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

破译混乱：通过对抗性提示翻译增强越狱攻击 cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.

摘要: 自动对抗性提示生成在越狱安全对齐的大型语言模型(LLM)方面取得了显着的成功。现有的基于梯度的攻击虽然在越狱白盒LLM中表现出出色的性能，但往往会产生外观混乱的乱码对抗性提示。这些对抗性提示很难转移到其他LLM上，阻碍了它们在攻击未知受害者模型时的表现。在本文中，我们首次深入研究了混淆的对抗性提示中所蕴含的语义，并提出了一种新的方法，将它们“翻译”成连贯的、人类可读的自然语言对抗性提示。通过这种方式，我们可以有效地发现触发模型漏洞的语义信息，并毫不含糊地将其传递给受害者模型，而不会忽视隐藏在乱码文本中的对抗性信息，以增强越狱攻击。它还提供了一种新的方法来发现有效的越狱提示设计，促进了对越狱攻击的理解。实验结果表明，我们的方法显著提高了对各种安全对齐LLM的越狱攻击成功率，并且远远超过了最新的技术水平。在最多10个查询的情况下，我们的方法在HarmBch上攻击包括GPT和Claude-3系列在内的7个商业闭源LLM，平均攻击成功率为81.8%。我们的方法对AdvBtch上的Llama-2-Chat模型的攻击成功率也达到了90%以上，尽管它们对越狱攻击具有出色的抵抗力。代码：https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **47. On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning**

关于测试时间适应的对抗风险：对现实测试时间数据中毒的调查 cs.LG

19 pages, 4 figures, 8 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.04682v2) [paper-pdf](http://arxiv.org/pdf/2410.04682v2)

**Authors**: Yongyi Su, Yushu Li, Nanqing Liu, Kui Jia, Xulei Yang, Chuan-Sheng Foo, Xun Xu

**Abstract**: Test-time adaptation (TTA) updates the model weights during the inference stage using testing data to enhance generalization. However, this practice exposes TTA to adversarial risks. Existing studies have shown that when TTA is updated with crafted adversarial test samples, also known as test-time poisoned data, the performance on benign samples can deteriorate. Nonetheless, the perceived adversarial risk may be overstated if the poisoned data is generated under overly strong assumptions. In this work, we first review realistic assumptions for test-time data poisoning, including white-box versus grey-box attacks, access to benign data, attack budget, and more. We then propose an effective and realistic attack method that better produces poisoned samples without access to benign samples, and derive an effective in-distribution attack objective. We also design two TTA-aware attack objectives. Our benchmarks of existing attack methods reveal that the TTA methods are more robust than previously believed. In addition, we analyze effective defense strategies to help develop adversarially robust TTA methods.

摘要: 测试时间自适应(TTA)在推理阶段使用测试数据更新模型权重，以增强泛化能力。然而，这种做法使TTA面临对抗性风险。现有研究表明，当使用精心编制的对抗性测试样本(也称为测试时间中毒数据)更新TTA时，良性样本的性能可能会恶化。然而，如果有毒数据是在过于强烈的假设下产生的，那么感知到的对抗性风险可能被夸大了。在这项工作中，我们首先回顾测试时间数据中毒的现实假设，包括白盒攻击与灰盒攻击、对良性数据的访问、攻击预算等。然后，我们提出了一种有效且现实的攻击方法，在不访问良性样本的情况下更好地产生有毒样本，并推导出有效的分布内攻击目标。我们还设计了两个TTA感知攻击目标。我们对现有攻击方法的基准测试表明，TTA方法比之前认为的更健壮。此外，我们分析了有效的防御策略，以帮助开发对抗健壮的TTA方法。



## **48. BRC20 Pinning Attack**

BRRC 20钉扎攻击 cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11295v1) [paper-pdf](http://arxiv.org/pdf/2410.11295v1)

**Authors**: Minfeng Qi, Qin Wang, Zhipeng Wang, Lin Zhong, Tianqing Zhu, Shiping Chen, William Knottenbelt

**Abstract**: BRC20 tokens are a type of non-fungible asset on the Bitcoin network. They allow users to embed customized content within Bitcoin satoshis. The related token frenzy has reached a market size of USD 3,650b over the past year (2023Q3-2024Q3). However, this intuitive design has not undergone serious security scrutiny.   We present the first in-depth analysis of the BRC20 transfer mechanism and identify a critical attack vector. A typical BRC20 transfer involves two bundled on-chain transactions with different fee levels: the first (i.e., Tx1) with a lower fee inscribes the transfer request, while the second (i.e., Tx2) with a higher fee finalizes the actual transfer. We find that an adversary can exploit this by sending a manipulated fee transaction (falling between the two fee levels), which allows Tx1 to be processed while Tx2 remains pinned in the mempool. This locks the BRC20 liquidity and disrupts normal transfers for users. We term this BRC20 pinning attack.   Our attack exposes an inherent design flaw that can be applied to 90+% inscription-based tokens within the Bitcoin ecosystem.   We also conducted the attack on Binance's ORDI hot wallet (the most prevalent BRC20 token and the most active wallet), resulting in a temporary suspension of ORDI withdrawals on Binance for 3.5 hours, which were shortly resumed after our communication.

摘要: BRC20代币是比特币网络上的一种不可替代资产。它们允许用户在比特币Satoshis中嵌入定制内容。在过去的一年里(2023Q3-2024Q3)，相关的代币狂潮已经达到了3.65万亿美元的市场规模。然而，这种直观的设计并没有经过严格的安全审查。我们首次深入分析了BRC20的传输机制，并确定了一个关键的攻击载体。典型的BRC20转移涉及两个不同费用水平的捆绑链上交易：第一个费用较低的(即TX1)记录转移请求，而第二个(即Tx2)费用较高的完成实际转移。我们发现，对手可以通过发送被操纵的费用事务(介于两个费用水平之间)来利用这一点，这允许在Tx1被处理的同时Tx2仍然被固定在内存池中。这锁定了BRC20的流动性，并扰乱了用户的正常转账。我们称之为BRC20钉住攻击。我们的攻击暴露了一个固有的设计缺陷，该缺陷可以应用于比特币生态系统中90%以上的铭文令牌。我们还对Binance的Ordi热钱包(最流行的BRC20代币和最活跃的钱包)进行了攻击，导致Binance上的Ordi提款暂时暂停3.5小时，并在我们沟通后不久恢复。



## **49. Cognitive Overload Attack:Prompt Injection for Long Context**

认知过载攻击：长上下文的提示注入 cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.

摘要: 大型语言模型(LLM)已经显示出在不需要明确的再培训的情况下执行跨领域任务的显著能力。这种被称为情景学习(ICL)的能力虽然令人印象深刻，但会使LLM暴露在各种对抗性提示和越狱之下，这些提示和越狱操作经过安全培训的LLM产生不需要的或有害的输出。在这篇文章中，我们提出了一种新的解释，从认知神经科学的角度，通过将人类认知中的学习与ICL相提并论，对LLMS中的ICL做出了新的解释。我们将认知负荷理论的原理应用到LLMS中，并实证验证了与人类认知类似，LLMS也存在认知过载，即认知加工需求超过模型的可用能力，从而导致潜在错误。此外，我们演示了攻击者如何通过故意设计的提示来利用ICL来越狱LLM，这些提示会导致LLM上的认知过载，从而危及LLMS的安全机制。我们通过制作不同的认知过载提示对该威胁模型进行了实证验证，结果表明，GPT-4、Claude-3.5十四行诗、Claude-3 opus、Llama-3-70B-Indict、Gemini-1.0-Pro和Gemini-1.5-Pro等高级模型可以成功越狱，攻击成功率高达99.99%。我们的发现突显了低土地管理制度的严重脆弱性，并强调了制定强有力的保障措施的紧迫性。我们建议将认知负荷理论的见解融入到LLMS的设计和评估中，以更好地预测和减轻对手攻击的风险。通过扩大我们的实验以涵盖更广泛的模型，并通过突出LLMS ICL中的漏洞，我们的目标是确保开发出更安全、更可靠的人工智能系统。



## **50. A Formal Framework for Assessing and Mitigating Emergent Security Risks in Generative AI Models: Bridging Theory and Dynamic Risk Mitigation**

评估和缓解生成人工智能模型中紧急安全风险的正式框架：桥梁理论和动态风险缓解 cs.CR

This paper was accepted in NeurIPS 2024 workshop on Red Teaming  GenAI: What can we learn with Adversaries?

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.13897v1) [paper-pdf](http://arxiv.org/pdf/2410.13897v1)

**Authors**: Aviral Srivastava, Sourav Panda

**Abstract**: As generative AI systems, including large language models (LLMs) and diffusion models, advance rapidly, their growing adoption has led to new and complex security risks often overlooked in traditional AI risk assessment frameworks. This paper introduces a novel formal framework for categorizing and mitigating these emergent security risks by integrating adaptive, real-time monitoring, and dynamic risk mitigation strategies tailored to generative models' unique vulnerabilities. We identify previously under-explored risks, including latent space exploitation, multi-modal cross-attack vectors, and feedback-loop-induced model degradation. Our framework employs a layered approach, incorporating anomaly detection, continuous red-teaming, and real-time adversarial simulation to mitigate these risks. We focus on formal verification methods to ensure model robustness and scalability in the face of evolving threats. Though theoretical, this work sets the stage for future empirical validation by establishing a detailed methodology and metrics for evaluating the performance of risk mitigation strategies in generative AI systems. This framework addresses existing gaps in AI safety, offering a comprehensive road map for future research and implementation.

摘要: 随着包括大型语言模型(LLM)和扩散模型在内的产生式AI系统的快速发展，它们的日益采用导致了传统AI风险评估框架中经常被忽视的新的复杂安全风险。本文介绍了一种新的形式化框架，通过集成针对生成式模型的独特漏洞定制的自适应、实时监控和动态风险缓解策略，对这些紧急安全风险进行分类和缓解。我们识别了以前未被充分开发的风险，包括潜在空间开发、多模式交叉攻击向量和反馈环导致的模型退化。我们的框架采用了分层的方法，结合了异常检测、持续的红色团队和实时对手模拟来降低这些风险。我们专注于形式化的验证方法，以确保模型在面对不断变化的威胁时的健壮性和可伸缩性。虽然这项工作是理论上的，但通过建立一种详细的方法和指标来评估生成性人工智能系统中风险缓解策略的性能，这项工作为未来的经验验证奠定了基础。这一框架解决了人工智能安全方面的现有差距，为未来的研究和实施提供了全面的路线图。



