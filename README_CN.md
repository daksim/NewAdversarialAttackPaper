# Latest Adversarial Attack Papers
**update at 2023-11-02 10:26:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks**

自动机器翻译度量在对抗性攻击下的稳健性测试 cs.CL

Accepted in Findings of EMNLP 2023

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00508v1) [paper-pdf](http://arxiv.org/pdf/2311.00508v1)

**Authors**: Yichen Huang, Timothy Baldwin

**Abstract**: We investigate MT evaluation metric performance on adversarially-synthesized texts, to shed light on metric robustness. We experiment with word- and character-level attacks on three popular machine translation metrics: BERTScore, BLEURT, and COMET. Our human experiments validate that automatic metrics tend to overpenalize adversarially-degraded translations. We also identify inconsistencies in BERTScore ratings, where it judges the original sentence and the adversarially-degraded one as similar, while judging the degraded translation as notably worse than the original with respect to the reference. We identify patterns of brittleness that motivate more robust metric development.

摘要: 我们研究了机器翻译在恶意合成文本上的评估度量性能，以阐明度量的稳健性。我们在三个流行的机器翻译指标上进行了单词和字符级别的攻击：BERTScore、BLEURT和Comet。我们的人类实验证实，自动度量往往会过度惩罚对抗性降级的翻译。我们还发现了BERTScore评级中的不一致之处，即它判断原始句子和反面降级的句子相似，而判断降级的翻译在引用方面明显比原始句子差。我们确定了激励更稳健的度量开发的脆性模式。



## **2. Improving Robustness for Vision Transformer with a Simple Dynamic Scanning Augmentation**

一种简单的动态扫描增强方法提高视觉变压器的稳健性 cs.CV

Accepted in Neurocomputing

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00441v1) [paper-pdf](http://arxiv.org/pdf/2311.00441v1)

**Authors**: Shashank Kotyan, Danilo Vasconcellos Vargas

**Abstract**: Vision Transformer (ViT) has demonstrated promising performance in computer vision tasks, comparable to state-of-the-art neural networks. Yet, this new type of deep neural network architecture is vulnerable to adversarial attacks limiting its capabilities in terms of robustness. This article presents a novel contribution aimed at further improving the accuracy and robustness of ViT, particularly in the face of adversarial attacks. We propose an augmentation technique called `Dynamic Scanning Augmentation' that leverages dynamic input sequences to adaptively focus on different patches, thereby maintaining performance and robustness. Our detailed investigations reveal that this adaptability to the input sequence induces significant changes in the attention mechanism of ViT, even for the same image. We introduce four variations of Dynamic Scanning Augmentation, outperforming ViT in terms of both robustness to adversarial attacks and accuracy against natural images, with one variant showing comparable results. By integrating our augmentation technique, we observe a substantial increase in ViT's robustness, improving it from $17\%$ to $92\%$ measured across different types of adversarial attacks. These findings, together with other comprehensive tests, indicate that Dynamic Scanning Augmentation enhances accuracy and robustness by promoting a more adaptive type of attention. In conclusion, this work contributes to the ongoing research on Vision Transformers by introducing Dynamic Scanning Augmentation as a technique for improving the accuracy and robustness of ViT. The observed results highlight the potential of this approach in advancing computer vision tasks and merit further exploration in future studies.

摘要: 视觉转换器(VIT)在计算机视觉任务中表现出与最先进的神经网络相媲美的良好性能。然而，这种新型的深度神经网络结构容易受到对手攻击，从而限制了其健壮性方面的能力。本文提出了一项新的贡献，旨在进一步提高VIT的准确性和稳健性，特别是在面对对手攻击的情况下。我们提出了一种称为动态扫描增强的增强技术，该技术利用动态输入序列自适应地聚焦于不同的补丁，从而保持了性能和稳健性。我们的详细研究表明，这种对输入序列的适应性导致了VIT注意机制的显著变化，即使对相同的图像也是如此。我们介绍了四种动态扫描增强算法，在对敌意攻击的稳健性和对自然图像的准确性方面都优于VIT，其中一种算法的结果与之相当。通过集成我们的增强技术，我们观察到VIT的健壮性有了很大的提高，在不同类型的对抗性攻击中测量到的VIT的健壮性从17美元提高到92美元。这些发现与其他综合性测试一起表明，动态扫描增强通过促进更适应类型的注意来提高准确性和稳健性。总之，这项工作通过引入动态扫描增强作为一种提高VIT准确性和稳健性的技术，为正在进行的视觉转换器研究做出了贡献。观察到的结果突出了这种方法在推进计算机视觉任务方面的潜力，值得在未来的研究中进一步探索。



## **3. NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks**

NEO-KD：基于知识蒸馏的健壮多出口神经网络对抗性训练 cs.LG

10 pages, 4 figures, accepted by 37th Conference on Neural  Information Processing Systems (NeurIPS 2023)

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00428v1) [paper-pdf](http://arxiv.org/pdf/2311.00428v1)

**Authors**: Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon

**Abstract**: While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.

摘要: 虽然多出口神经网络被认为是通过早期出口进行有效推理的一种有前途的解决方案，但对抗对手攻击仍然是一个具有挑战性的问题。在多出口网络中，由于不同子模型之间的高度依赖，针对特定出口的敌意例子不仅降低了目标出口的性能，而且同时降低了所有其他出口的性能。这使得多出口网络非常容易受到简单的对抗性攻击。在本文中，我们提出了NEO-KD，一种基于知识蒸馏的对抗性训练策略，基于两个关键贡献来应对这一根本挑战。NEO-KD首先采用邻居知识提炼的方法，引导对抗性实例的输出趋向于干净数据的邻居出口的集成输出。NEO-KD还采用了出口方式的正交知识提取，以减少不同子模型之间的对抗性转移。其结果是显著提高了对对手攻击的健壮性。在不同数据集/模型上的实验结果表明，与依赖于现有的对抗性训练或多出口网络的知识提取技术的基线相比，该方法在减少计算开销的情况下获得了最好的对抗性准确率。



## **4. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

自适应神经网络的动态感知敌意攻击 cs.CV

arXiv admin note: text overlap with arXiv:2112.09428

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2210.08159v3) [paper-pdf](http://arxiv.org/pdf/2210.08159v3)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods. Code is available at https://github.com/antao97/LGM.

摘要: 本文研究了自适应神经网络的动态感知对抗攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络架构在整个攻击过程中是固定的。然而，这种假设并不适用于许多最近提出的自适应神经网络，自适应地停用不必要的执行单元的基础上的输入，以提高计算效率。这导致了严重的滞后梯度问题，使得在当前步骤学习的攻击由于之后的架构更改而无效。为了解决这个问题，我们提出了一个领先的梯度方法（LGM），并显示滞后梯度的显着影响。更具体地说，我们重新制定的梯度要知道潜在的动态变化的网络架构，使学习的攻击更好地“领导”的下一步比动态不知道的方法时，网络架构动态变化。针对2D图像和3D点云的代表性自适应神经网络的广泛实验表明，与动态无意识攻击方法相比，我们的LGM实现了令人印象深刻的对抗性攻击性能。代码可在https://github.com/antao97/LGM上获得。



## **5. LFAA: Crafting Transferable Targeted Adversarial Examples with Low-Frequency Perturbations**

LFAA：制作具有低频扰动的可转移目标对抗性实例 cs.CV

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2310.20175v2) [paper-pdf](http://arxiv.org/pdf/2310.20175v2)

**Authors**: Kunyu Wang, Juluan Shi, Wenxuan Wang

**Abstract**: Deep neural networks are susceptible to adversarial attacks, which pose a significant threat to their security and reliability in real-world applications. The most notable adversarial attacks are transfer-based attacks, where an adversary crafts an adversarial example to fool one model, which can also fool other models. While previous research has made progress in improving the transferability of untargeted adversarial examples, the generation of targeted adversarial examples that can transfer between models remains a challenging task. In this work, we present a novel approach to generate transferable targeted adversarial examples by exploiting the vulnerability of deep neural networks to perturbations on high-frequency components of images. We observe that replacing the high-frequency component of an image with that of another image can mislead deep models, motivating us to craft perturbations containing high-frequency information to achieve targeted attacks. To this end, we propose a method called Low-Frequency Adversarial Attack (\name), which trains a conditional generator to generate targeted adversarial perturbations that are then added to the low-frequency component of the image. Extensive experiments on ImageNet demonstrate that our proposed approach significantly outperforms state-of-the-art methods, improving targeted attack success rates by a margin from 3.2\% to 15.5\%.

摘要: 深度神经网络容易受到敌意攻击，这对其在实际应用中的安全性和可靠性构成了严重威胁。最著名的对抗性攻击是基于传输的攻击，在这种攻击中，对手编造一个对抗性的例子来愚弄一个模型，这也可以愚弄其他模型。虽然以前的研究在提高非目标对抗性实例的可转移性方面取得了进展，但生成可以在模型之间转移的目标对抗性实例仍然是一项具有挑战性的任务。在这项工作中，我们提出了一种新的方法，通过利用深层神经网络对图像高频分量扰动的脆弱性来生成可转移的目标对抗性样本。我们观察到，用另一幅图像的高频分量替换另一幅图像的高频分量会误导深层模型，促使我们精心设计包含高频信息的扰动，以实现有针对性的攻击。为此，我们提出了一种称为低频对抗性攻击(\NAME)的方法，它训练一个条件生成器来生成目标对抗性扰动，然后将这些扰动添加到图像的低频分量中。在ImageNet上的大量实验表明，我们提出的方法明显优于最先进的方法，将目标攻击成功率从3.2%提高到15.5%。



## **6. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

Magmaw：基于机器学习的无线通信系统的通道无关敌意攻击 cs.CR

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00207v1) [paper-pdf](http://arxiv.org/pdf/2311.00207v1)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer components, and wireless domain constraints. This paper proposes Magmaw, the first black-box attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on ML-based downstream applications. The resilience of the attack to the existing widely used defense methods of adversarial training and perturbation signal subtraction is experimentally verified. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of the defense mechanisms. Surprisingly, Magmaw is also effective against encrypted communication channels and conventional communications.

摘要: 机器学习(ML)通过合并端到端无线通信系统的所有物理层块，在实现联合收发器优化方面发挥了重要作用。虽然已经有一些针对基于ML的无线系统的对抗性攻击，但现有的方法不能提供包括源数据的多模态、公共物理层组件和无线域约束在内的全面视角。本文提出了Magmaw，这是第一个能够对无线信道上传输的任何多模式信号产生通用对抗性扰动的黑盒攻击方法。我们进一步介绍了针对基于ML的下游应用程序的对抗性攻击的新目标。实验验证了该攻击对现有广泛使用的对抗性训练和扰动信号减法防御方法的抗攻击能力。对于概念验证评估，我们使用软件定义的无线电系统构建了一个实时无线攻击平台。实验结果表明，即使在存在防御机制的情况下，Magmaw也会导致性能显著下降。令人惊讶的是，Magmaw对加密通信渠道和传统通信也很有效。



## **7. Robust Safety Classifier for Large Language Models: Adversarial Prompt Shield**

用于大型语言模型的稳健安全分类器：对抗性提示盾 cs.CL

11 pages, 2 figures

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2311.00172v1) [paper-pdf](http://arxiv.org/pdf/2311.00172v1)

**Authors**: Jinhwa Kim, Ali Derakhshan, Ian G. Harris

**Abstract**: Large Language Models' safety remains a critical concern due to their vulnerability to adversarial attacks, which can prompt these systems to produce harmful responses. In the heart of these systems lies a safety classifier, a computational model trained to discern and mitigate potentially harmful, offensive, or unethical outputs. However, contemporary safety classifiers, despite their potential, often fail when exposed to inputs infused with adversarial noise. In response, our study introduces the Adversarial Prompt Shield (APS), a lightweight model that excels in detection accuracy and demonstrates resilience against adversarial prompts. Additionally, we propose novel strategies for autonomously generating adversarial training datasets, named Bot Adversarial Noisy Dialogue (BAND) datasets. These datasets are designed to fortify the safety classifier's robustness, and we investigate the consequences of incorporating adversarial examples into the training process. Through evaluations involving Large Language Models, we demonstrate that our classifier has the potential to decrease the attack success rate resulting from adversarial attacks by up to 60%. This advancement paves the way for the next generation of more reliable and resilient conversational agents.

摘要: 大型语言模型的安全性仍然是一个关键问题，因为它们容易受到对抗性攻击，这可能会促使这些系统产生有害的响应。这些系统的核心是安全分类器，这是一个经过训练的计算模型，用于识别和减少潜在的有害、攻击性或不道德的输出。然而，尽管现代安全分类器具有潜力，但在接触到充满对抗性噪音的输入时，它们往往会失败。作为回应，我们的研究引入了对抗性提示盾牌(APS)，这是一种轻量级模型，在检测准确性方面表现出色，并对对抗性提示表现出韧性。此外，我们还提出了自主生成对抗性训练数据集的新策略，称为僵尸对抗性噪声对话(BAND)数据集。这些数据集是为了加强安全分类器的稳健性而设计的，我们调查了将对抗性例子纳入训练过程的后果。通过对大型语言模型的评估，我们证明了我们的分类器具有将对抗性攻击导致的攻击成功率降低高达60%的潜力。这一进步为下一代更可靠、更具弹性的对话代理铺平了道路。



## **8. Amoeba: Circumventing ML-supported Network Censorship via Adversarial Reinforcement Learning**

阿米巴：通过对抗性强化学习绕过ML支持的网络审查 cs.CR

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20469v1) [paper-pdf](http://arxiv.org/pdf/2310.20469v1)

**Authors**: Haoyu Liu, Alec F. Diallo, Paul Patras

**Abstract**: Embedding covert streams into a cover channel is a common approach to circumventing Internet censorship, due to censors' inability to examine encrypted information in otherwise permitted protocols (Skype, HTTPS, etc.). However, recent advances in machine learning (ML) enable detecting a range of anti-censorship systems by learning distinct statistical patterns hidden in traffic flows. Therefore, designing obfuscation solutions able to generate traffic that is statistically similar to innocuous network activity, in order to deceive ML-based classifiers at line speed, is difficult.   In this paper, we formulate a practical adversarial attack strategy against flow classifiers as a method for circumventing censorship. Specifically, we cast the problem of finding adversarial flows that will be misclassified as a sequence generation task, which we solve with Amoeba, a novel reinforcement learning algorithm that we design. Amoeba works by interacting with censoring classifiers without any knowledge of their model structure, but by crafting packets and observing the classifiers' decisions, in order to guide the sequence generation process. Our experiments using data collected from two popular anti-censorship systems demonstrate that Amoeba can effectively shape adversarial flows that have on average 94% attack success rate against a range of ML algorithms. In addition, we show that these adversarial flows are robust in different network environments and possess transferability across various ML models, meaning that once trained against one, our agent can subvert other censoring classifiers without retraining.

摘要: 将隐蔽流嵌入覆盖频道是绕过互联网审查的常见方法，因为审查者无法检查以其他允许的协议(Skype、HTTPS等)加密的信息。然而，机器学习(ML)的最新进展使人们能够通过学习隐藏在交通流中的不同统计模式来检测一系列反审查系统。因此，为了在线速下欺骗基于ML的分类器，设计能够产生统计上类似于无害网络活动的流量的混淆解决方案是困难的。在本文中，我们制定了一种实用的对流分类器的对抗性攻击策略，作为一种规避审查的方法。具体地说，我们将发现被错误分类的敌意流的问题转化为序列生成任务，我们使用我们设计的一种新的强化学习算法Amoeba来解决这个问题。变形虫的工作原理是在不了解其模型结构的情况下与审查分类器交互，而是通过精心制作包并观察分类器的决策，以指导序列生成过程。我们使用从两个流行的反审查系统收集的数据进行的实验表明，变形虫能够有效地塑造敌意流，对一系列ML算法的攻击成功率平均为94%。此外，我们还证明了这些敌意流在不同的网络环境中是健壮的，并且具有跨不同ML模型的可转移性，这意味着一旦针对一个模型进行训练，我们的代理就可以颠覆其他审查分类器而不需要重新训练。



## **9. On Extracting Specialized Code Abilities from Large Language Models: A Feasibility Study**

从大型语言模型中提取专业代码能力的可行性研究 cs.SE

13 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2303.03012v4) [paper-pdf](http://arxiv.org/pdf/2303.03012v4)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, Yang Liu

**Abstract**: Recent advances in large language models (LLMs) significantly boost their usage in software engineering. However, training a well-performing LLM demands a substantial workforce for data collection and annotation. Moreover, training datasets may be proprietary or partially open, and the process often requires a costly GPU cluster. The intellectual property value of commercial LLMs makes them attractive targets for imitation attacks, but creating an imitation model with comparable parameters still incurs high costs. This motivates us to explore a practical and novel direction: slicing commercial black-box LLMs using medium-sized backbone models. In this paper, we explore the feasibility of launching imitation attacks on LLMs to extract their specialized code abilities, such as"code synthesis" and "code translation." We systematically investigate the effectiveness of launching code ability extraction attacks under different code-related tasks with multiple query schemes, including zero-shot, in-context, and Chain-of-Thought. We also design response checks to refine the outputs, leading to an effective imitation training process. Our results show promising outcomes, demonstrating that with a reasonable number of queries, attackers can train a medium-sized backbone model to replicate specialized code behaviors similar to the target LLMs. We summarize our findings and insights to help researchers better understand the threats posed by imitation attacks, including revealing a practical attack surface for generating adversarial code examples against LLMs.

摘要: 大型语言模型（LLM）的最新进展显着提高了它们在软件工程中的使用。然而，培训一个表现良好的LLM需要大量的劳动力进行数据收集和注释。此外，训练数据集可能是专有的或部分开放的，并且该过程通常需要昂贵的GPU集群。商业LLM的知识产权价值使其成为模仿攻击的有吸引力的目标，但创建具有可比参数的模仿模型仍然会产生高成本。这促使我们探索一个实用而新颖的方向：使用中型骨干模型切片商业黑盒LLM。在本文中，我们探讨了对LLM发起模仿攻击以提取其特殊代码能力的可行性，如“代码合成”和“代码翻译”。“我们系统地研究了在不同的代码相关任务下使用多种查询方案（包括零射击，上下文和思想链）发起代码能力提取攻击的有效性。我们还设计了响应检查来完善输出，从而实现有效的模仿训练过程。我们的研究结果显示了有希望的结果，表明通过合理数量的查询，攻击者可以训练一个中等规模的骨干模型来复制类似于目标LLM的专门代码行为。我们总结了我们的发现和见解，以帮助研究人员更好地理解模仿攻击所带来的威胁，包括揭示一个实际的攻击面，用于生成针对LLM的对抗性代码示例。



## **10. Robust nonparametric regression based on deep ReLU neural networks**

基于深度回归神经网络的稳健非参数回归 stat.ME

40 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20294v1) [paper-pdf](http://arxiv.org/pdf/2310.20294v1)

**Authors**: Juntong Chen

**Abstract**: In this paper, we consider robust nonparametric regression using deep neural networks with ReLU activation function. While several existing theoretically justified methods are geared towards robustness against identical heavy-tailed noise distributions, the rise of adversarial attacks has emphasized the importance of safeguarding estimation procedures against systematic contamination. We approach this statistical issue by shifting our focus towards estimating conditional distributions. To address it robustly, we introduce a novel estimation procedure based on $\ell$-estimation. Under a mild model assumption, we establish general non-asymptotic risk bounds for the resulting estimators, showcasing their robustness against contamination, outliers, and model misspecification. We then delve into the application of our approach using deep ReLU neural networks. When the model is well-specified and the regression function belongs to an $\alpha$-H\"older class, employing $\ell$-type estimation on suitable networks enables the resulting estimators to achieve the minimax optimal rate of convergence. Additionally, we demonstrate that deep $\ell$-type estimators can circumvent the curse of dimensionality by assuming the regression function closely resembles the composition of several H\"older functions. To attain this, new deep fully-connected ReLU neural networks have been designed to approximate this composition class. This approximation result can be of independent interest.

摘要: 本文考虑基于RELU激活函数的深度神经网络的稳健非参数回归问题。虽然现有的几种理论上合理的方法针对相同的重尾噪声分布具有稳健性，但对抗性攻击的兴起强调了保护估计过程免受系统污染的重要性。我们通过将我们的重点转移到估计条件分布来处理这个统计问题。为了更好地解决这个问题，我们引入了一种新的估计方法--估计。在温和的模型假设下，我们为所得到的估计量建立了一般的非渐近风险界，展示了它们对污染、异常值和模型错误指定的稳健性。然后，我们使用深度RELU神经网络深入研究我们的方法的应用。当模型被很好地描述并且回归函数属于$-α$-H“老类时，在适当的网络上采用$-型估计使得所得到的估计量能够达到极小极大最优收敛速度.此外，我们还证明了通过假设回归函数非常类似于几个H‘-老函数的组合，深$-型估计可以绕过维度诅咒.为了实现这一点，设计了新的深度全连接RELU神经网络来逼近这一组成类。这种近似结果可能具有独立的意义。



## **11. CFDP: Common Frequency Domain Pruning**

CFDP：公共频域修剪 cs.CV

CVPR ECV 2023 Accepted Paper

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.04147v2) [paper-pdf](http://arxiv.org/pdf/2306.04147v2)

**Authors**: Samir Khaki, Weihan Luo

**Abstract**: As the saying goes, sometimes less is more -- and when it comes to neural networks, that couldn't be more true. Enter pruning, the art of selectively trimming away unnecessary parts of a network to create a more streamlined, efficient architecture. In this paper, we introduce a novel end-to-end pipeline for model pruning via the frequency domain. This work aims to shed light on the interoperability of intermediate model outputs and their significance beyond the spatial domain. Our method, dubbed Common Frequency Domain Pruning (CFDP) aims to extrapolate common frequency characteristics defined over the feature maps to rank the individual channels of a layer based on their level of importance in learning the representation. By harnessing the power of CFDP, we have achieved state-of-the-art results on CIFAR-10 with GoogLeNet reaching an accuracy of 95.25%, that is, +0.2% from the original model. We also outperform all benchmarks and match the original model's performance on ImageNet, using only 55% of the trainable parameters and 60% of the FLOPs. In addition to notable performances, models produced via CFDP exhibit robustness to a variety of configurations including pruning from untrained neural architectures, and resistance to adversarial attacks. The implementation code can be found at https://github.com/Skhaki18/CFDP.

摘要: 俗话说，有时候少就是多--当谈到神经网络时，这是最正确的。修剪是有选择地修剪网络中不必要的部分，以创建更精简、更高效的架构的艺术。本文介绍了一种新的基于频域的端到端模型剪枝流水线。这项工作旨在阐明中间模式输出的互操作性及其在空间领域之外的意义。我们的方法被称为公共频域修剪(CFDP)，目的是外推在特征映射上定义的公共频率特征，以基于它们在学习表示的重要性级别来对层的各个通道进行排名。通过利用CFDP的力量，我们在CIFAR-10上取得了最先进的结果，GoogLeNet达到了95.25%的准确率，即比原始模型+0.2%。我们在ImageNet上的性能也超过了所有基准测试，与原始模型的性能相当，只使用了55%的可训练参数和60%的失败。除了显著的性能外，通过CFDP产生的模型还表现出对各种配置的健壮性，包括从未经训练的神经体系结构中进行修剪，以及对对手攻击的抵抗。实现代码可在https://github.com/Skhaki18/CFDP.上找到



## **12. BERT Lost Patience Won't Be Robust to Adversarial Slowdown**

伯特失去了耐心，不会对对手的减速表现得很健壮 cs.LG

Accepted to NeurIPS 2023 [Poster]

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.19152v2) [paper-pdf](http://arxiv.org/pdf/2310.19152v2)

**Authors**: Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong

**Abstract**: In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models. Our code is available at: https://github.com/ztcoalson/WAFFLE

摘要: 在本文中，我们系统地评估了多出口语言模型对对抗减速的稳健性。为了审计它们的健壮性，我们设计了一种减速攻击，该攻击绕过提前退出点生成自然的对抗性文本。我们使用由此产生的华夫饼攻击作为工具，使用针对对抗性放缓的GLUE基准对三种多退出机制进行了全面评估。然后，我们展示了我们的攻击显著降低了白盒和黑盒设置下的三种方法所提供的计算节省。一种机制越复杂，就越容易受到对抗性放缓的影响。我们还对受干扰的文本输入执行语言分析，识别我们的攻击产生的常见扰动模式，并将它们与标准的对抗性文本攻击进行比较。此外，我们还证明了对抗性训练不能有效地抵抗我们的减速攻击，但是使用会话模型(如ChatGPT)的输入净化可以有效地去除扰动。这一结果表明，未来需要开展工作来开发高效而稳健的多退出模型。我们的代码请访问：https://github.com/ztcoalson/WAFFLE



## **13. Is Robustness Transferable across Languages in Multilingual Neural Machine Translation?**

在多语言神经机器翻译中，健壮性可以跨语言传递吗？ cs.AI

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20162v1) [paper-pdf](http://arxiv.org/pdf/2310.20162v1)

**Authors**: Leiyu Pan, Supryadi, Deyi Xiong

**Abstract**: Robustness, the ability of models to maintain performance in the face of perturbations, is critical for developing reliable NLP systems. Recent studies have shown promising results in improving the robustness of models through adversarial training and data augmentation. However, in machine translation, most of these studies have focused on bilingual machine translation with a single translation direction. In this paper, we investigate the transferability of robustness across different languages in multilingual neural machine translation. We propose a robustness transfer analysis protocol and conduct a series of experiments. In particular, we use character-, word-, and multi-level noises to attack the specific translation direction of the multilingual neural machine translation model and evaluate the robustness of other translation directions. Our findings demonstrate that the robustness gained in one translation direction can indeed transfer to other translation directions. Additionally, we empirically find scenarios where robustness to character-level noise and word-level noise is more likely to transfer.

摘要: 鲁棒性，即模型在面对扰动时保持性能的能力，对于开发可靠的NLP系统至关重要。最近的研究表明，通过对抗性训练和数据增强，在提高模型的鲁棒性方面取得了可喜的成果。然而，在机器翻译方面，这些研究大多集中在双语机器翻译，翻译方向单一。本文研究了多语言神经机器翻译中鲁棒性在不同语言间的可移植性。我们提出了一个鲁棒性的传输分析协议，并进行了一系列的实验。特别是，我们使用字符，单词和多级噪声来攻击多语言神经机器翻译模型的特定翻译方向，并评估其他翻译方向的鲁棒性。我们的研究结果表明，在一个平移方向上获得的鲁棒性确实可以转移到其他平移方向。此外，我们根据经验发现，对字符级噪声和单词级噪声的鲁棒性更有可能转移。



## **14. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

TrojLLM：一种针对大型语言模型的黑盒木马提示攻击 cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.06815v3) [paper-pdf](http://arxiv.org/pdf/2306.06815v3)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.

摘要: 大型语言模型(LLM)正逐渐被用作各种应用的机器学习服务和接口工具。然而，LLMS的安全影响，特别是与对抗性攻击和特洛伊木马攻击有关的影响，仍然没有得到充分的研究。在本文中，我们提出了一个自动黑盒框架TrojLLM，它可以有效地生成通用的、隐蔽的触发器。当这些触发器被合并到输入数据中时，LLMS的输出可能被恶意操纵。此外，该框架还支持在离散提示中嵌入特洛伊木马，增强了触发器攻击的整体有效性和精确度。具体地说，我们提出了一种触发器发现算法，通过使用少量数据样本查询受害者基于LLM的API来为各种输入生成通用触发器。此外，我们引入了一种新的渐进式特洛伊木马中毒算法，旨在生成中毒提示，从而在不同的模型中保持有效性和可转移性。我们的实验和结果表明，TrojLLM能够在包括GPT-3.5和GPT-4在内的真实黑盒LLMAPI中有效地将特洛伊木马程序插入到文本提示中，同时在干净的测试集上保持出色的性能。我们的工作揭示了当前模型中的潜在安全风险，并提供了一种潜在的防御方法。TrojLLm的源代码可在https://github.com/UCF-ML-Research/TrojLLM.上找到



## **15. Exploring Geometry of Blind Spots in Vision Models**

视觉模型中盲点几何的探索 cs.CV

25 pages, 20 figures, Accepted at NeurIPS 2023 (spotlight)

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19889v1) [paper-pdf](http://arxiv.org/pdf/2310.19889v1)

**Authors**: Sriram Balasubramanian, Gaurang Sriramanan, Vinu Sankar Sadasivan, Soheil Feizi

**Abstract**: Despite the remarkable success of deep neural networks in a myriad of settings, several works have demonstrated their overwhelming sensitivity to near-imperceptible perturbations, known as adversarial attacks. On the other hand, prior works have also observed that deep networks can be under-sensitive, wherein large-magnitude perturbations in input space do not induce appreciable changes to network activations. In this work, we study in detail the phenomenon of under-sensitivity in vision models such as CNNs and Transformers, and present techniques to study the geometry and extent of "equi-confidence" level sets of such networks. We propose a Level Set Traversal algorithm that iteratively explores regions of high confidence with respect to the input space using orthogonal components of the local gradients. Given a source image, we use this algorithm to identify inputs that lie in the same equi-confidence level set as the source image despite being perceptually similar to arbitrary images from other classes. We further observe that the source image is linearly connected by a high-confidence path to these inputs, uncovering a star-like structure for level sets of deep networks. Furthermore, we attempt to identify and estimate the extent of these connected higher-dimensional regions over which the model maintains a high degree of confidence. The code for this project is publicly available at https://github.com/SriramB-98/blindspots-neurips-sub

摘要: 尽管深度神经网络在各种环境中取得了显著的成功，但有几项工作已经证明了它们对几乎不可察觉的扰动--即所谓的对抗性攻击--具有压倒性的敏感性。另一方面，以前的工作也观察到深层网络可能是欠敏感的，其中输入空间中的大幅度扰动不会引起网络激活的明显变化。在这项工作中，我们详细地研究了CNN和Transformers等视觉模型中的欠敏感性现象，并提出了研究这类网络的等置信度水平集的几何和程度的技术。我们提出了一种水平集遍历算法，该算法使用局部梯度的正交分量迭代地探索相对于输入空间的高置信度区域。在给定源图像的情况下，我们使用该算法来识别与源图像处于相同等置信度集合中的输入，尽管这些输入在感知上与来自其他类别的任意图像相似。我们进一步观察到，源图像通过高置信度路径线性连接到这些输入，揭示了深度网络水平集的星状结构。此外，我们试图识别和估计这些连通的高维区域的范围，在这些区域上，模型保持高度的置信度。该项目的代码可在https://github.com/SriramB-98/blindspots-neurips-sub上公开获得



## **16. Adversarial Attacks and Defenses in Large Language Models: Old and New Threats**

大型语言模型中的对抗性攻击和防御：旧威胁和新威胁 cs.AI

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19737v1) [paper-pdf](http://arxiv.org/pdf/2310.19737v1)

**Authors**: Leo Schwinn, David Dobre, Stephan Günnemann, Gauthier Gidel

**Abstract**: Over the past decade, there has been extensive research aimed at enhancing the robustness of neural networks, yet this problem remains vastly unsolved. Here, one major impediment has been the overestimation of the robustness of new defense approaches due to faulty defense evaluations. Flawed robustness evaluations necessitate rectifications in subsequent works, dangerously slowing down the research and providing a false sense of security. In this context, we will face substantial challenges associated with an impending adversarial arms race in natural language processing, specifically with closed-source Large Language Models (LLMs), such as ChatGPT, Google Bard, or Anthropic's Claude. We provide a first set of prerequisites to improve the robustness assessment of new approaches and reduce the amount of faulty evaluations. Additionally, we identify embedding space attacks on LLMs as another viable threat model for the purposes of generating malicious content in open-sourced models. Finally, we demonstrate on a recently proposed defense that, without LLM-specific best practices in place, it is easy to overestimate the robustness of a new approach.

摘要: 在过去的十年里，已经有大量的研究旨在增强神经网络的健壮性，但这个问题仍然远远没有解决。在这里，一个主要的障碍是由于错误的防御评估而高估了新的防御方法的稳健性。有缺陷的稳健性评估需要在后续工作中进行更正，这会危险地减缓研究速度，并提供一种错误的安全感。在这种情况下，我们将面临与自然语言处理领域即将到来的对抗性军备竞赛相关的重大挑战，特别是与封闭源代码的大型语言模型(LLM)相关的挑战，如ChatGPT、Google Bard或Anthropic的Claude。我们提供了第一组先决条件来改进新方法的稳健性评估，并减少错误评估的数量。此外，我们将在LLM上嵌入空间攻击作为另一种可行的威胁模型，目的是在开源模型中生成恶意内容。最后，我们在最近提出的一项防御中演示了，如果没有特定于LLM的最佳实践，很容易高估新方法的健壮性。



## **17. Differentially Private Reward Estimation with Preference Feedback**

带偏好反馈的差分私人报酬估计 cs.LG

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19733v1) [paper-pdf](http://arxiv.org/pdf/2310.19733v1)

**Authors**: Sayak Ray Chowdhury, Xingyu Zhou, Nagarajan Natarajan

**Abstract**: Learning from preference-based feedback has recently gained considerable traction as a promising approach to align generative models with human interests. Instead of relying on numerical rewards, the generative models are trained using reinforcement learning with human feedback (RLHF). These approaches first solicit feedback from human labelers typically in the form of pairwise comparisons between two possible actions, then estimate a reward model using these comparisons, and finally employ a policy based on the estimated reward model. An adversarial attack in any step of the above pipeline might reveal private and sensitive information of human labelers. In this work, we adopt the notion of label differential privacy (DP) and focus on the problem of reward estimation from preference-based feedback while protecting privacy of each individual labelers. Specifically, we consider the parametric Bradley-Terry-Luce (BTL) model for such pairwise comparison feedback involving a latent reward parameter $\theta^* \in \mathbb{R}^d$. Within a standard minimax estimation framework, we provide tight upper and lower bounds on the error in estimating $\theta^*$ under both local and central models of DP. We show, for a given privacy budget $\epsilon$ and number of samples $n$, that the additional cost to ensure label-DP under local model is $\Theta \big(\frac{1}{ e^\epsilon-1}\sqrt{\frac{d}{n}}\big)$, while it is $\Theta\big(\frac{\text{poly}(d)}{\epsilon n} \big)$ under the weaker central model. We perform simulations on synthetic data that corroborate these theoretical results.

摘要: 从基于偏好的反馈中学习最近获得了相当大的吸引力，因为它是使生成性模型与人类兴趣保持一致的一种有前途的方法。生成模型不依赖于数字奖励，而是使用带人类反馈的强化学习(RLHF)进行训练。这些方法首先征求人类标记者的反馈，通常是两个可能的动作之间的成对比较的形式，然后使用这些比较来估计奖励模型，最后采用基于估计的奖励模型的策略。在上述管道的任何步骤中的对抗性攻击都可能泄露人类标签员的私人和敏感信息。在这项工作中，我们采用了标签差异隐私(DP)的概念，并重点研究了基于偏好的反馈中的奖励估计问题，同时保护了每个标签者的隐私。具体地说，我们考虑了这类两两比较反馈的参数Bradley-Terry-Luce(BTL)模型，该模型在Mathbb{R}^d$中含有潜在的报酬参数。在一个标准的极大极小估计框架内，我们给出了在DP的局部和中心模型下的估计误差的严格上界和下界。我们证明了，对于给定的隐私预算$\epsilon$和样本数$n$，在局部模型下确保Label-DP的额外成本是$\theta\BIG(\FRAC{1}{e^\epsilon-1}\Sqrt{\FRAC{d}{n}}\BIG)$，而在较弱的中心模型下是$\Theta\BIG(\FRAC{\Text{pol}(D)}{\epsilon n}\BIG)$。我们对合成数据进行了模拟，证实了这些理论结果。



## **18. Iris: Dynamic Privacy Preserving Search in Structured Peer-to-Peer Networks**

IRIS：结构化对等网络中的动态隐私保护搜索 cs.CR

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19634v1) [paper-pdf](http://arxiv.org/pdf/2310.19634v1)

**Authors**: Angeliki Aktypi, Kasper Rasmussen

**Abstract**: In structured peer-to-peer networks like Chord, the users manage to retrieve the information they seek by asking other nodes from the network for the information they search. Revealing to other nodes the search target makes structured peer-to-peer networks unsuitable for applications that demand query privacy, i.e., hiding the query's target from the intermediate nodes that take part in the routing. This paper studies the query privacy of structured P2P networks, particularly the Chord protocol.   We initially observe that already proposed privacy notions, such as $k$-anonymity, do not allow us to reason about the privacy guarantees of a query in Chord in the presence of a strong adversary. Thus, we introduce a new privacy notion that we call $(\alpha,\delta)$-privacy that allows us to evaluate the privacy guarantees even when considering the worst-case scenario regarding an attacker's background knowledge.   We then design Iris, an algorithm that allows a requester to conceal the target of a query in Chord from the intermediate nodes that take part in the routing. Iris achieves that by having the requester query for other than the target addresses so as reaching each one of them allows the requester to get closer to the target address.   We perform a security analysis of the proposed algorithm, based on the privacy notion we introduce. We also develop a prototype of the algorithm in Matlab and evaluate its performance. Our analysis proves Iris to be $(\alpha,\delta)$-private while introducing a modest performance overhead.

摘要: 在像Chord这样的结构化对等网络中，用户通过向网络中的其他节点请求他们搜索的信息来设法检索他们寻找的信息。向其他节点透露搜索目标使得结构化对等网络不适合于要求查询隐私的应用，即向参与路由的中间节点隐藏查询的目标。本文研究了结构化P2P网络中的查询隐私问题，特别是Chord协议。我们最初观察到，已经提出的隐私概念，如$k$-匿名性，不允许我们在强大对手在场的情况下推理Chord查询的隐私保证。因此，我们引入了一个新的隐私概念，我们称之为$(\α，\Delta)$-隐私，它允许我们评估隐私保证，即使考虑到关于攻击者背景知识的最坏情况。然后我们设计了IRIS算法，该算法允许请求者在Chord中向参与路由的中间节点隐藏查询的目标。IRIS通过让请求者查询目标地址之外的其他地址来实现这一点，以便到达每个目标地址允许请求者更接近目标地址。基于我们引入的隐私概念，我们对提出的算法进行了安全性分析。我们还在MatLab中开发了该算法的原型，并对其性能进行了评估。我们的分析证明了Iris是$(\Alpha，\Delta)$-私有的，同时引入了适度的性能开销。



## **19. Testing Robustness Against Unforeseen Adversaries**

测试针对不可预见的对手的健壮性 cs.LG

Datasets available at  https://github.com/centerforaisafety/adversarial-corruptions

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/1908.08016v4) [paper-pdf](http://arxiv.org/pdf/1908.08016v4)

**Authors**: Max Kaufmann, Daniel Kang, Yi Sun, Steven Basart, Xuwang Yin, Mantas Mazeika, Akul Arora, Adam Dziedzic, Franziska Boenisch, Tom Brown, Jacob Steinhardt, Dan Hendrycks

**Abstract**: Adversarial robustness research primarily focuses on L_p perturbations, and most defenses are developed with identical training-time and test-time adversaries. However, in real-world applications developers are unlikely to have access to the full range of attacks or corruptions their system will face. Furthermore, worst-case inputs are likely to be diverse and need not be constrained to the L_p ball. To narrow in on this discrepancy between research and reality we introduce ImageNet-UA, a framework for evaluating model robustness against a range of unforeseen adversaries, including eighteen new non-L_p attacks. To perform well on ImageNet-UA, defenses must overcome a generalization gap and be robust to a diverse attacks not encountered during training. In extensive experiments, we find that existing robustness measures do not capture unforeseen robustness, that standard robustness techniques are beat by alternative training strategies, and that novel methods can improve unforeseen robustness. We present ImageNet-UA as a useful tool for the community for improving the worst-case behavior of machine learning systems.

摘要: 对抗稳健性的研究主要集中在L_p扰动上，而且大多数防御都是在训练时间和测试时间相同的情况下发展起来的。然而，在现实世界的应用程序中，开发人员不太可能获得他们的系统将面临的所有攻击或破坏的权限。此外，最糟糕的输入可能是多样化的，不需要被限制在L_p球上。为了缩小研究与实际之间的差距，我们引入了ImageNet-UA框架，用于评估模型对一系列不可预见的对手的健壮性，包括18个新的非L_p攻击。为了在ImageNet-UA上表现良好，防御必须克服泛化差距，并对训练中未遇到的各种攻击保持健壮。在大量的实验中，我们发现现有的稳健性度量没有捕捉到不可预见的稳健性，标准的稳健性技术被替代的训练策略所击败，新的方法可以改善不可预见的稳健性。我们将ImageNet-UA作为一个有用的工具，供社区改进机器学习系统的最坏情况行为。



## **20. A Generative Framework for Low-Cost Result Validation of Outsourced Machine Learning Tasks**

外包机器学习任务低成本结果验证的产生式框架 cs.CR

16 pages, 11 figures

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2304.00083v3) [paper-pdf](http://arxiv.org/pdf/2304.00083v3)

**Authors**: Abhinav Kumar, Miguel A. Guirao Aguilera, Reza Tourani, Satyajayant Misra

**Abstract**: The growing popularity of Machine Learning (ML) has led to its deployment in various sensitive domains, which has resulted in significant research focused on ML security and privacy. However, in some applications, such as autonomous driving, integrity verification of the outsourced ML workload is more critical--a facet that has not received much attention. Existing solutions, such as multi-party computation and proof-based systems, impose significant computation overhead, which makes them unfit for real-time applications. We propose Fides, a novel framework for real-time validation of outsourced ML workloads. Fides features a novel and efficient distillation technique--Greedy Distillation Transfer Learning--that dynamically distills and fine-tunes a space and compute-efficient verification model for verifying the corresponding service model while running inside a trusted execution environment. Fides features a client-side attack detection model that uses statistical analysis and divergence measurements to identify, with a high likelihood, if the service model is under attack. Fides also offers a re-classification functionality that predicts the original class whenever an attack is identified. We devised a generative adversarial network framework for training the attack detection and re-classification models. The evaluation shows that Fides achieves an accuracy of up to 98% for attack detection and 94% for re-classification.

摘要: 机器学习(ML)的日益流行导致了它在各种敏感领域的部署，这导致了对ML安全和隐私的大量研究。然而，在一些应用中，例如自动驾驶，外包的ML工作负载的完整性验证更关键--这一方面没有得到太多关注。现有的解决方案，如多方计算和基于证明的系统，带来了巨大的计算开销，这使得它们不适合实时应用。我们提出了一种新的实时验证外包ML工作负载的框架FIDS。FIDS的特点是一种新颖而高效的蒸馏技术--贪婪蒸馏转移学习--它动态地提取和微调空间和计算效率高的验证模型，以便在可信执行环境中运行时验证相应的服务模型。FIDS具有客户端攻击检测模型，该模型使用统计分析和分歧测量来识别服务模型是否受到攻击的可能性很高。FIDS还提供了重新分类功能，该功能可以在识别攻击时预测原始类别。我们设计了一个生成式对抗性网络框架来训练攻击检测和重分类模型。评估表明，FIDS对攻击检测的准确率高达98%，对重分类的准确率高达94%。



## **21. Generated Distributions Are All You Need for Membership Inference Attacks Against Generative Models**

生成的分布是针对生成模型的成员关系推断攻击所需的全部 cs.CR

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19410v1) [paper-pdf](http://arxiv.org/pdf/2310.19410v1)

**Authors**: Minxing Zhang, Ning Yu, Rui Wen, Michael Backes, Yang Zhang

**Abstract**: Generative models have demonstrated revolutionary success in various visual creation tasks, but in the meantime, they have been exposed to the threat of leaking private information of their training data. Several membership inference attacks (MIAs) have been proposed to exhibit the privacy vulnerability of generative models by classifying a query image as a training dataset member or nonmember. However, these attacks suffer from major limitations, such as requiring shadow models and white-box access, and either ignoring or only focusing on the unique property of diffusion models, which block their generalization to multiple generative models. In contrast, we propose the first generalized membership inference attack against a variety of generative models such as generative adversarial networks, [variational] autoencoders, implicit functions, and the emerging diffusion models. We leverage only generated distributions from target generators and auxiliary non-member datasets, therefore regarding target generators as black boxes and agnostic to their architectures or application scenarios. Experiments validate that all the generative models are vulnerable to our attack. For instance, our work achieves attack AUC $>0.99$ against DDPM, DDIM, and FastDPM trained on CIFAR-10 and CelebA. And the attack against VQGAN, LDM (for the text-conditional generation), and LIIF achieves AUC $>0.90.$ As a result, we appeal to our community to be aware of such privacy leakage risks when designing and publishing generative models.

摘要: 生成模型在各种视觉创建任务中取得了革命性的成功，但与此同时，它们也面临着泄露训练数据的私人信息的威胁。已经提出了几种成员推断攻击（MIA），通过将查询图像分类为训练数据集成员或非成员来展示生成模型的隐私漏洞。然而，这些攻击受到主要限制，例如需要阴影模型和白盒访问，并且忽略或仅关注扩散模型的独特属性，这阻碍了它们对多个生成模型的推广。相比之下，我们提出了针对各种生成模型的第一个广义隶属推理攻击，例如生成对抗网络，[变分]自编码器，隐函数和新兴的扩散模型。我们只利用目标生成器和辅助非成员数据集生成的分布，因此将目标生成器视为黑盒，对其架构或应用场景不可知。实验证明，所有的生成模型都容易受到我们的攻击。例如，我们的工作实现了攻击AUC $>0.99$对DDPM，DDIM，和快速训练CIFAR-10和CelebA。对VQGAN、LDM（用于文本条件生成）和LIIF的攻击AUC $>0.90。因此，我们呼吁我们的社区在设计和发布生成模型时要意识到这种隐私泄露风险。



## **22. Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective**

平衡、失衡和再平衡：从极小极大博弈的角度理解稳健的过度匹配 cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19360v1) [paper-pdf](http://arxiv.org/pdf/2310.19360v1)

**Authors**: Yifei Wang, Liangchen Li, Jiansheng Yang, Zhouchen Lin, Yisen Wang

**Abstract**: Adversarial Training (AT) has become arguably the state-of-the-art algorithm for extracting robust features. However, researchers recently notice that AT suffers from severe robust overfitting problems, particularly after learning rate (LR) decay. In this paper, we explain this phenomenon by viewing adversarial training as a dynamic minimax game between the model trainer and the attacker. Specifically, we analyze how LR decay breaks the balance between the minimax game by empowering the trainer with a stronger memorization ability, and show such imbalance induces robust overfitting as a result of memorizing non-robust features. We validate this understanding with extensive experiments, and provide a holistic view of robust overfitting from the dynamics of both the two game players. This understanding further inspires us to alleviate robust overfitting by rebalancing the two players by either regularizing the trainer's capacity or improving the attack strength. Experiments show that the proposed ReBalanced Adversarial Training (ReBAT) can attain good robustness and does not suffer from robust overfitting even after very long training. Code is available at https://github.com/PKU-ML/ReBAT.

摘要: 对抗性训练(AT)已经成为提取稳健特征的最先进的算法。然而，研究人员最近注意到，AT存在严重的稳健过适应问题，特别是在学习率(LR)衰减之后。在本文中，我们通过将对抗性训练视为模型训练者和攻击者之间的动态极大极小博弈来解释这一现象。具体地说，我们分析了LR衰减是如何通过赋予训练者更强的记忆能力来打破极小极大博弈之间的平衡的，并表明这种不平衡导致了由于记忆非稳健特征而导致的稳健过适应。我们通过广泛的实验验证了这一理解，并从两个游戏玩家的动态提供了稳健过适应的整体视图。这一理解进一步激励我们通过调整教练的能力或提高进攻强度来重新平衡两名球员，以缓解健壮的过度适应。实验表明，提出的再平衡对抗性训练算法(REBAT)具有较好的鲁棒性，即使经过很长时间的训练也不会出现鲁棒性过强的问题。代码可在https://github.com/PKU-ML/ReBAT.上找到



## **23. Label-Only Model Inversion Attacks via Knowledge Transfer**

基于知识转移的仅标签模型反转攻击 cs.LG

Accepted to Neurips 2023. The first two authors contributed equally

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19342v1) [paper-pdf](http://arxiv.org/pdf/2310.19342v1)

**Authors**: Ngoc-Bao Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man Cheung

**Abstract**: In a model inversion (MI) attack, an adversary abuses access to a machine learning (ML) model to infer and reconstruct private training data. Remarkable progress has been made in the white-box and black-box setups, where the adversary has access to the complete model or the model's soft output respectively. However, there is very limited study in the most challenging but practically important setup: Label-only MI attacks, where the adversary only has access to the model's predicted label (hard label) without confidence scores nor any other model information.   In this work, we propose LOKT, a novel approach for label-only MI attacks. Our idea is based on transfer of knowledge from the opaque target model to surrogate models. Subsequently, using these surrogate models, our approach can harness advanced white-box attacks. We propose knowledge transfer based on generative modelling, and introduce a new model, Target model-assisted ACGAN (T-ACGAN), for effective knowledge transfer. Our method casts the challenging label-only MI into the more tractable white-box setup. We provide analysis to support that surrogate models based on our approach serve as effective proxies for the target model for MI. Our experiments show that our method significantly outperforms existing SOTA Label-only MI attack by more than 15% across all MI benchmarks. Furthermore, our method compares favorably in terms of query budget. Our study highlights rising privacy threats for ML models even when minimal information (i.e., hard labels) is exposed. Our study highlights rising privacy threats for ML models even when minimal information (i.e., hard labels) is exposed. Our code, demo, models and reconstructed data are available at our project page: https://ngoc-nguyen-0.github.io/lokt/

摘要: 在模型反转(MI)攻击中，对手滥用对机器学习(ML)模型的访问来推断和重建私人训练数据。白盒和黑盒的设置已经取得了显著的进展，对手可以分别访问完整的模型或模型的软输出。然而，在最具挑战性但实际重要的设置中的研究非常有限：仅标签MI攻击，其中对手只能访问模型的预测标签(硬标签)，而没有置信度分数或任何其他模型信息。在这项工作中，我们提出了一种新的仅标签MI攻击方法LOKT。我们的想法是基于从不透明的目标模型到代理模型的知识转移。随后，使用这些代理模型，我们的方法可以利用高级白盒攻击。提出了基于产生式建模的知识转移模型，并提出了一种新的知识转移模型--目标模型辅助ACGAN(T-ACGAN)，以实现有效的知识转移。我们的方法将具有挑战性的仅标签MI投射到更容易处理的白盒设置中。我们提供了分析以支持基于我们的方法的代理模型作为MI的目标模型的有效代理。我们的实验表明，在所有的MI基准测试中，我们的方法比现有的SOTA仅标签MI攻击的性能高出15%以上。此外，在查询预算方面，我们的方法是比较有利的。我们的研究强调，即使在最小信息(即硬标签)被曝光的情况下，ML模型面临的隐私威胁也在上升。我们的研究强调，即使在最小信息(即硬标签)被曝光的情况下，ML模型面临的隐私威胁也在上升。我们的代码、演示、模型和重建数据可在我们的项目页面上获得：https://ngoc-nguyen-0.github.io/lokt/



## **24. Treatment Learning Causal Transformer for Noisy Image Classification**

用于含噪图像分类的治疗学习因果变换 cs.CV

Accepted to IEEE WACV 2023. The first version was finished in May  2018

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2203.15529v2) [paper-pdf](http://arxiv.org/pdf/2203.15529v2)

**Authors**: Chao-Han Huck Yang, I-Te Danny Hung, Yi-Chieh Liu, Pin-Yu Chen

**Abstract**: Current top-notch deep learning (DL) based vision models are primarily based on exploring and exploiting the inherent correlations between training data samples and their associated labels. However, a known practical challenge is their degraded performance against "noisy" data, induced by different circumstances such as spurious correlations, irrelevant contexts, domain shift, and adversarial attacks. In this work, we incorporate this binary information of "existence of noise" as treatment into image classification tasks to improve prediction accuracy by jointly estimating their treatment effects. Motivated from causal variational inference, we propose a transformer-based architecture, Treatment Learning Causal Transformer (TLT), that uses a latent generative model to estimate robust feature representations from current observational input for noise image classification. Depending on the estimated noise level (modeled as a binary treatment factor), TLT assigns the corresponding inference network trained by the designed causal loss for prediction. We also create new noisy image datasets incorporating a wide range of noise factors (e.g., object masking, style transfer, and adversarial perturbation) for performance benchmarking. The superior performance of TLT in noisy image classification is further validated by several refutation evaluation metrics. As a by-product, TLT also improves visual salience methods for perceiving noisy images.

摘要: 目前基于深度学习的视觉模型主要是基于探索和利用训练数据样本及其关联标签之间的内在相关性。然而，一个已知的实际挑战是，它们针对不同环境(如伪相关性、无关上下文、域转移和敌意攻击)引起的抗噪声数据的性能下降。在这项工作中，我们将“噪声的存在”这一二值信息作为处理，引入到图像分类任务中，通过联合估计它们的处理效果来提高预测精度。受因果变分推理的启发，我们提出了一种基于变换的结构，处理学习因果转换器(TLT)，它使用一个潜在的生成模型来估计当前观测输入的稳健特征表示，用于噪声图像分类。根据估计的噪声水平(建模为二进制处理因子)，TLT分配由设计的因果损失训练的相应的推理网络用于预测。我们还创建了新的噪声图像数据集，其中包含了广泛的噪声因素(例如，对象掩蔽、样式转移和对抗性扰动)，用于性能基准测试。几种反驳评价指标进一步验证了TLT在含噪图像分类中的优越性能。作为一个副产品，TLT还改进了感知噪声图像的视觉显著方法。



## **25. A Black-Box Approach to Post-Quantum Zero-Knowledge in Constant Rounds**

后量子零知识在常数轮中的黑箱方法 quant-ph

Fixed a minor technical issue (see Footnote 17 in page 21) and  improved the proof of Claim 4.5. (10/30/2023)

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2011.02670v4) [paper-pdf](http://arxiv.org/pdf/2011.02670v4)

**Authors**: Nai-Hui Chia, Kai-Min Chung, Takashi Yamakawa

**Abstract**: In a recent seminal work, Bitansky and Shmueli (STOC '20) gave the first construction of a constant round zero-knowledge argument for NP secure against quantum attacks. However, their construction has several drawbacks compared to the classical counterparts. Specifically, their construction only achieves computational soundness, requires strong assumptions of quantum hardness of learning with errors (QLWE assumption) and the existence of quantum fully homomorphic encryption (QFHE), and relies on non-black-box simulation. In this paper, we resolve these issues at the cost of weakening the notion of zero-knowledge to what is called $\epsilon$-zero-knowledge. Concretely, we construct the following protocols:   - We construct a constant round interactive proof for NP that satisfies statistical soundness and black-box $\epsilon$-zero-knowledge against quantum attacks assuming the existence of collapsing hash functions, which is a quantum counterpart of collision-resistant hash functions. Interestingly, this construction is just an adapted version of the classical protocol by Goldreich and Kahan (JoC '96) though the proof of $\epsilon$-zero-knowledge property against quantum adversaries requires novel ideas.   - We construct a constant round interactive argument for NP that satisfies computational soundness and black-box $\epsilon$-zero-knowledge against quantum attacks only assuming the existence of post-quantum one-way functions.   At the heart of our results is a new quantum rewinding technique that enables a simulator to extract a committed message of a malicious verifier while simulating verifier's internal state in an appropriate sense.

摘要: 在最近的一项开创性工作中，Bitansky和Shmueli(STOEC‘20)首次构造了NP安全抵抗量子攻击的常量轮零知识论点。然而，与经典的对应结构相比，它们的构造有几个缺陷。具体地说，它们的构造只实现了计算的可靠性，需要有错误学习的量子硬性(QLWE假设)和量子完全同态加密(QFHE)的存在的强假设，并且依赖于非黑盒模拟。在本文中，我们解决这些问题的代价是将零知识的概念弱化为所谓的$-零知识。具体地说，我们构造了如下协议：-我们构造了一个关于NP的常量轮次交互证明，它满足统计可靠性和黑盒-抗量子攻击的零知识，假设存在可折叠的哈希函数，它是抗碰撞哈希函数的量子对应。有趣的是，这种构造只是Goldreich和Kahan(Joc‘96)经典协议的改编版本，尽管针对量子对手的$\epsilon$零知识性质的证明需要新的想法。-我们为NP构造了一个常数轮交互论证，它仅在后量子单向函数存在的情况下才满足计算可靠性和黑盒$\epsilon$-抗量子攻击零知识。我们结果的核心是一种新的量子倒带技术，该技术使模拟器能够在适当意义上模拟验证器的内部状态的同时提取恶意验证器的承诺消息。



## **26. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

从聊天机器人到网络钓鱼机器人？--防止使用ChatGPT、Google Bard和Claude创建的网络钓鱼诈骗 cs.CR

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19181v1) [paper-pdf](http://arxiv.org/pdf/2310.19181v1)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs - ChatGPT (GPT 3.5 Turbo), GPT 4, Claude and Bard to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing emails and websites that can convincingly imitate well-known brands, and also deploy a range of evasive tactics for the latter to elude detection mechanisms employed by anti-phishing systems. Notably, these attacks can be generated using unmodified, or "vanilla," versions of these LLMs, without requiring any prior adversarial exploits such as jailbreaking. As a countermeasure, we build a BERT based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content attaining an accuracy of 97\% for phishing website prompts, and 94\% for phishing email prompts.

摘要: 大型语言模型(LLM)的高级功能使其在从会话代理和内容创建到数据分析、研究和创新的各种应用程序中具有无价的价值。然而，它们的有效性和可访问性也使它们容易被滥用来生成恶意内容，包括网络钓鱼攻击。这项研究探索了四种流行的商用LLM-ChatGPT(GPT 3.5 Turbo)、GPT 4、Claude和Bard使用一系列恶意提示生成功能性网络钓鱼攻击的可能性。我们发现，这些LLM既可以生成钓鱼电子邮件，也可以生成能够令人信服地模仿知名品牌的网站，并为后者部署一系列规避策略，以躲避反钓鱼系统使用的检测机制。值得注意的是，这些攻击可以使用这些LLM的未经修改或“普通”版本来生成，而不需要任何先前的对抗性攻击，如越狱。作为对策，我们构建了一个基于ERT的自动检测工具，用于早期检测恶意提示，以防止LLMS生成钓鱼内容，对钓鱼网站提示的准确率达到97%，对钓鱼电子邮件提示的准确率达到94%。



## **27. Robustifying Language Models with Test-Time Adaptation**

基于测试时间自适应的模糊语言模型 cs.CL

8 Pages 2 Figures Submitted to ICLR Workshop

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19177v1) [paper-pdf](http://arxiv.org/pdf/2310.19177v1)

**Authors**: Noah Thomas McDermott, Junfeng Yang, Chengzhi Mao

**Abstract**: Large-scale language models achieved state-of-the-art performance over a number of language tasks. However, they fail on adversarial language examples, which are sentences optimized to fool the language models but with similar semantic meanings for humans. While prior work focuses on making the language model robust at training time, retraining for robustness is often unrealistic for large-scale foundation models. Instead, we propose to make the language models robust at test time. By dynamically adapting the input sentence with predictions from masked words, we show that we can reverse many language adversarial attacks. Since our approach does not require any training, it works for novel tasks at test time and can adapt to novel adversarial corruptions. Visualizations and empirical results on two popular sentence classification datasets demonstrate that our method can repair adversarial language attacks over 65% o

摘要: 大规模语言模型在许多语言任务上实现了最先进的性能。然而，他们在对抗性语言例子上失败了，这些例子是为愚弄语言模型而优化的句子，但对人类来说具有相似的语义。虽然以前的工作重点是在训练时使语言模型具有健壮性，但对于大规模的基础模型来说，进行健壮性的再培训往往是不现实的。相反，我们建议在测试时使语言模型健壮。通过动态调整输入句子和掩蔽词的预测，我们证明了我们可以逆转许多语言对手攻击。由于我们的方法不需要任何训练，它在测试时适用于新的任务，并且可以适应新的对抗性腐败。在两个常用的句子分类数据集上的可视化和实验结果表明，该方法可以修复65%以上的敌意语言攻击



## **28. RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Active Data Manipulation**

RAIFLE：基于交互的主动数据操作联邦学习重构攻击 cs.CR

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19163v1) [paper-pdf](http://arxiv.org/pdf/2310.19163v1)

**Authors**: Dzung Pham, Shreyas Kulkarni, Amir Houmansadr

**Abstract**: Federated learning (FL) has recently emerged as a privacy-preserving approach for machine learning in domains that rely on user interactions, particularly recommender systems (RS) and online learning to rank (OLTR). While there has been substantial research on the privacy of traditional FL, little attention has been paid to studying the privacy properties of these interaction-based FL (IFL) systems. In this work, we show that IFL can introduce unique challenges concerning user privacy, particularly when the central server has knowledge and control over the items that users interact with. Specifically, we demonstrate the threat of reconstructing user interactions by presenting RAIFLE, a general optimization-based reconstruction attack framework customized for IFL. RAIFLE employs Active Data Manipulation (ADM), a novel attack technique unique to IFL, where the server actively manipulates the training features of the items to induce adversarial behaviors in the local FL updates. We show that RAIFLE is more impactful than existing FL privacy attacks in the IFL context, and describe how it can undermine privacy defenses like secure aggregation and private information retrieval. Based on our findings, we propose and discuss countermeasure guidelines to mitigate our attack in the context of federated RS/OLTR specifically and IFL more broadly.

摘要: 最近，联合学习(FL)作为一种隐私保护方法在依赖用户交互的领域中出现，特别是推荐系统(RS)和在线学习排名(OLTR)。虽然已经有大量的研究对传统外语的隐私进行了研究，但很少有人关注这些基于交互的外语系统的隐私特性。在这项工作中，我们展示了IFL可以引入关于用户隐私的独特挑战，特别是当中央服务器知道并控制用户交互的项目时。具体地说，我们通过提出一种为IFL定制的基于优化的通用重建攻击框架RAIFLE来演示重建用户交互的威胁。RAIFLE采用主动数据操纵技术(ADM)，这是IFL独有的一种新的攻击技术，其中服务器主动操纵项目的训练特征，以在本地FL更新中诱导对抗行为。我们证明了RAIFLE在IFL环境中比现有的FL隐私攻击更有效，并描述了它如何破坏安全聚合和私人信息检索等隐私防御。基于我们的发现，我们提出并讨论了对策指导方针，以在联邦RS/OLTR以及更广泛的IFL的背景下减轻我们的攻击。



## **29. Poisoning Retrieval Corpora by Injecting Adversarial Passages**

注入敌对通道毒化检索语料库 cs.CL

EMNLP 2023. Our code is available at  https://github.com/princeton-nlp/corpus-poisoning

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19156v1) [paper-pdf](http://arxiv.org/pdf/2310.19156v1)

**Authors**: Zexuan Zhong, Ziqing Huang, Alexander Wettig, Danqi Chen

**Abstract**: Dense retrievers have achieved state-of-the-art performance in various information retrieval tasks, but to what extent can they be safely deployed in real-world applications? In this work, we propose a novel attack for dense retrieval systems in which a malicious user generates a small number of adversarial passages by perturbing discrete tokens to maximize similarity with a provided set of training queries. When these adversarial passages are inserted into a large retrieval corpus, we show that this attack is highly effective in fooling these systems to retrieve them for queries that were not seen by the attacker. More surprisingly, these adversarial passages can directly generalize to out-of-domain queries and corpora with a high success attack rate -- for instance, we find that 50 generated passages optimized on Natural Questions can mislead >94% of questions posed in financial documents or online forums. We also benchmark and compare a range of state-of-the-art dense retrievers, both unsupervised and supervised. Although different systems exhibit varying levels of vulnerability, we show they can all be successfully attacked by injecting up to 500 passages, a small fraction compared to a retrieval corpus of millions of passages.

摘要: 密集检索器已经在各种信息检索任务中实现了最先进的性能，但它们在多大程度上可以安全地部署在现实世界的应用程序中？在这项工作中，我们提出了一种新的针对密集检索系统的攻击，其中恶意用户通过扰乱离散标记来最大化与给定训练查询集的相似度来生成少量对抗性段落。当这些对抗性段落被插入到大型检索语料库中时，我们表明这种攻击在欺骗这些系统检索攻击者看不到的查询方面非常有效。更令人惊讶的是，这些对抗性段落可以直接概括为域外查询和具有高成功率的语料库--例如，我们发现针对自然问题优化的50个生成段落可以误导金融文档或在线论坛中提出的问题的94%以上。我们还对一系列最先进的密集检索器进行了基准测试和比较，包括无监督和有监督的。尽管不同的系统表现出不同程度的漏洞，但我们表明，通过注入多达500个段落，它们都可以被成功攻击，与数百万个段落的检索语料库相比，这是一个很小的比例。



## **30. Alignment with human representations supports robust few-shot learning**

与人类表达方式保持一致，支持可靠的少数几次学习 cs.LG

Spotlight at NeurIPS 2023

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2301.11990v3) [paper-pdf](http://arxiv.org/pdf/2301.11990v3)

**Authors**: Ilia Sucholutsky, Thomas L. Griffiths

**Abstract**: Should we care whether AI systems have representations of the world that are similar to those of humans? We provide an information-theoretic analysis that suggests that there should be a U-shaped relationship between the degree of representational alignment with humans and performance on few-shot learning tasks. We confirm this prediction empirically, finding such a relationship in an analysis of the performance of 491 computer vision models. We also show that highly-aligned models are more robust to both natural adversarial attacks and domain shifts. Our results suggest that human-alignment is often a sufficient, but not necessary, condition for models to make effective use of limited data, be robust, and generalize well.

摘要: 我们是否应该关心人工智能系统是否具有与人类相似的世界表示法？我们提供的信息论分析表明，与人类的表征一致性程度与在少数几次学习任务上的表现之间应该存在U型关系。我们在对491个计算机视觉模型的性能分析中发现了这种关系，并从经验上证实了这一预测。我们还表明，高度对齐的模型对自然对抗性攻击和域转移都具有更强的鲁棒性。我们的结果表明，人的一致性通常是模型有效利用有限数据、健壮性和泛化良好的充分条件，但不是必要条件。



## **31. Boosting Decision-Based Black-Box Adversarial Attack with Gradient Priors**

增强基于决策的梯度先验黑箱对抗攻击 cs.LG

Accepted by IJCAI 2023

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19038v1) [paper-pdf](http://arxiv.org/pdf/2310.19038v1)

**Authors**: Han Liu, Xingshuo Huang, Xiaotong Zhang, Qimai Li, Fenglong Ma, Wei Wang, Hongyang Chen, Hong Yu, Xianchao Zhang

**Abstract**: Decision-based methods have shown to be effective in black-box adversarial attacks, as they can obtain satisfactory performance and only require to access the final model prediction. Gradient estimation is a critical step in black-box adversarial attacks, as it will directly affect the query efficiency. Recent works have attempted to utilize gradient priors to facilitate score-based methods to obtain better results. However, these gradient priors still suffer from the edge gradient discrepancy issue and the successive iteration gradient direction issue, thus are difficult to simply extend to decision-based methods. In this paper, we propose a novel Decision-based Black-box Attack framework with Gradient Priors (DBA-GP), which seamlessly integrates the data-dependent gradient prior and time-dependent prior into the gradient estimation procedure. First, by leveraging the joint bilateral filter to deal with each random perturbation, DBA-GP can guarantee that the generated perturbations in edge locations are hardly smoothed, i.e., alleviating the edge gradient discrepancy, thus remaining the characteristics of the original image as much as possible. Second, by utilizing a new gradient updating strategy to automatically adjust the successive iteration gradient direction, DBA-GP can accelerate the convergence speed, thus improving the query efficiency. Extensive experiments have demonstrated that the proposed method outperforms other strong baselines significantly.

摘要: 基于决策的方法在黑盒对抗攻击中是有效的，因为它们可以获得令人满意的性能，并且只需要访问最终的模型预测。在黑盒对抗性攻击中，梯度估计是一个关键步骤，它直接影响查询效率。最近的工作试图利用梯度先验来促进基于分数的方法以获得更好的结果。然而，这些梯度先验仍然存在边缘梯度差异问题和逐次迭代梯度方向问题，因此很难简单地推广到基于决策的方法。本文提出了一种新的基于决策的梯度先验黑盒攻击框架(DBA-GP)，该框架将数据依赖的梯度先验和时间依赖的先验无缝地结合到梯度估计过程中。首先，DBA-GP利用联合双边滤波来处理每个随机扰动，可以保证边缘位置产生的扰动很难被平滑，即缓解了边缘梯度的差异，从而尽可能地保持了原始图像的特征。其次，DBA-GP采用一种新的梯度更新策略自动调整迭代梯度方向，加快了收敛速度，从而提高了查询效率。大量实验表明，该方法的性能明显优于其他强基线方法。



## **32. Attacks on Online Learners: a Teacher-Student Analysis**

对网络学习者的攻击：一种师生分析 stat.ML

19 pages, 10 figures

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2305.11132v2) [paper-pdf](http://arxiv.org/pdf/2305.11132v2)

**Authors**: Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

**Abstract**: Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.

摘要: 众所周知，机器学习模型容易受到对抗性攻击：对数据的微小特别扰动可能会灾难性地改变模型预测。虽然有大量文献研究了测试时间攻击预先训练的模型的案例，但到目前为止，在线学习环境中的重要攻击案例很少受到关注。在这项工作中，我们使用控制理论的观点来研究攻击者可能扰乱数据标签以操纵在线学习者的学习动态的场景。我们在教师-学生系统中对该问题进行了理论分析，考虑了不同的攻击策略，得到了简单线性学习者稳态的解析结果。这些结果使我们能够证明，当攻击强度超过临界阈值时，学习者的准确率会发生不连续的转变。然后，我们使用真实数据对具有复杂架构的学习者进行了实证研究，证实了我们的理论分析的真知灼见。我们的发现表明，贪婪攻击可以非常有效，特别是当数据流以小批量传输时。



## **33. Blacksmith: Fast Adversarial Training of Vision Transformers via a Mixture of Single-step and Multi-step Methods**

铁匠：通过混合单步和多步方法对视觉变形金刚进行快速对抗性训练 cs.CV

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.18975v1) [paper-pdf](http://arxiv.org/pdf/2310.18975v1)

**Authors**: Mahdi Salmani, Alireza Dehghanpour Farashah, Mohammad Azizmalayeri, Mahdi Amiri, Navid Eslami, Mohammad Taghi Manzuri, Mohammad Hossein Rohban

**Abstract**: Despite the remarkable success achieved by deep learning algorithms in various domains, such as computer vision, they remain vulnerable to adversarial perturbations. Adversarial Training (AT) stands out as one of the most effective solutions to address this issue; however, single-step AT can lead to Catastrophic Overfitting (CO). This scenario occurs when the adversarially trained network suddenly loses robustness against multi-step attacks like Projected Gradient Descent (PGD). Although several approaches have been proposed to address this problem in Convolutional Neural Networks (CNNs), we found out that they do not perform well when applied to Vision Transformers (ViTs). In this paper, we propose Blacksmith, a novel training strategy to overcome the CO problem, specifically in ViTs. Our approach utilizes either of PGD-2 or Fast Gradient Sign Method (FGSM) randomly in a mini-batch during the adversarial training of the neural network. This will increase the diversity of our training attacks, which could potentially mitigate the CO issue. To manage the increased training time resulting from this combination, we craft the PGD-2 attack based on only the first half of the layers, while FGSM is applied end-to-end. Through our experiments, we demonstrate that our novel method effectively prevents CO, achieves PGD-2 level performance, and outperforms other existing techniques including N-FGSM, which is the state-of-the-art method in fast training for CNNs.

摘要: 尽管深度学习算法在计算机视觉等各个领域取得了显著的成功，但它们仍然容易受到对抗性扰动的影响。对抗训练（AT）是解决这个问题的最有效的解决方案之一;然而，单步AT可能导致灾难性过度拟合（CO）。这种情况发生在对抗训练的网络突然失去对投影梯度下降（PGD）等多步攻击的鲁棒性时。虽然已经提出了几种方法来解决卷积神经网络（CNN）中的这个问题，但我们发现它们在应用于视觉变换器（ViTs）时表现不佳。在本文中，我们提出了铁匠，一种新的训练策略，以克服CO的问题，特别是在ViTs。我们的方法利用PGD-2或快速梯度符号法（FGSM）随机在一个小批量在对抗训练的神经网络。这将增加我们训练攻击的多样性，这可能会缓解CO问题。为了管理由于这种组合而增加的训练时间，我们仅基于前半层制作PGD-2攻击，而FGSM是端到端应用的。通过我们的实验，我们证明了我们的新方法有效地防止了CO，实现了PGD-2级性能，并且优于其他现有技术，包括N-FGSM，这是CNN快速训练的最先进方法。



## **34. Label Poisoning is All You Need**

标签中毒就是你所需要的 cs.LG

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.18933v1) [paper-pdf](http://arxiv.org/pdf/2310.18933v1)

**Authors**: Rishi D. Jha, Jonathan Hayase, Sewoong Oh

**Abstract**: In a backdoor attack, an adversary injects corrupted data into a model's training dataset in order to gain control over its predictions on images with a specific attacker-defined trigger. A typical corrupted training example requires altering both the image, by applying the trigger, and the label. Models trained on clean images, therefore, were considered safe from backdoor attacks. However, in some common machine learning scenarios, the training labels are provided by potentially malicious third-parties. This includes crowd-sourced annotation and knowledge distillation. We, hence, investigate a fundamental question: can we launch a successful backdoor attack by only corrupting labels? We introduce a novel approach to design label-only backdoor attacks, which we call FLIP, and demonstrate its strengths on three datasets (CIFAR-10, CIFAR-100, and Tiny-ImageNet) and four architectures (ResNet-32, ResNet-18, VGG-19, and Vision Transformer). With only 2% of CIFAR-10 labels corrupted, FLIP achieves a near-perfect attack success rate of 99.4% while suffering only a 1.8% drop in the clean test accuracy. Our approach builds upon the recent advances in trajectory matching, originally introduced for dataset distillation.

摘要: 在后门攻击中，对手将被破坏的数据注入模型的训练数据集中，以便通过特定的攻击者定义的触发器来控制其对图像的预测。典型的被破坏的训练示例需要通过应用触发器和标签来改变图像。因此，接受过干净形象培训的模特被认为是安全的，不会受到后门攻击。然而，在一些常见的机器学习场景中，训练标签是由潜在的恶意第三方提供的。这包括众包注释和知识提炼。因此，我们调查了一个根本问题：我们能否仅通过腐败标签就能发动成功的后门攻击？我们介绍了一种新的设计仅标签后门攻击的方法，我们称之为翻转，并在三个数据集(CIFAR-10、CIFAR-100和Tiny-ImageNet)和四个体系结构(ResNet-32、ResNet-18、VGG-19和Vision Transformer)上展示了它的优势。在只有2%的CIFAR-10标签被损坏的情况下，Flip实现了近乎完美的攻击成功率99.4%，而干净的测试精度仅下降了1.8%。我们的方法建立在轨迹匹配方面的最新进展之上，最初是为了数据集蒸馏而引入的。



## **35. Reliable learning in challenging environments**

在充满挑战的环境中可靠地学习 cs.LG

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2304.03370v2) [paper-pdf](http://arxiv.org/pdf/2304.03370v2)

**Authors**: Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

**Abstract**: The problem of designing learners that provide guarantees that their predictions are provably correct is of increasing importance in machine learning. However, learning theoretic guarantees have only been considered in very specific settings. In this work, we consider the design and analysis of reliable learners in challenging test-time environments as encountered in modern machine learning problems: namely `adversarial' test-time attacks (in several variations) and `natural' distribution shifts. In this work, we provide a reliable learner with provably optimal guarantees in such settings. We discuss computationally feasible implementations of the learner and further show that our algorithm achieves strong positive performance guarantees on several natural examples: for example, linear separators under log-concave distributions or smooth boundary classifiers under smooth probability distributions.

摘要: 在机器学习中，设计学习器以保证他们的预测被证明是正确的问题变得越来越重要。然而，学习理论的保证只在非常具体的环境中被考虑过。在这项工作中，我们考虑了在具有挑战性的测试时间环境中的可靠学习者的设计和分析，就像现代机器学习问题中遇到的那样：即“对抗性”测试时间攻击(在几个变体中)和“自然”分布偏移。在这项工作中，我们为可靠的学习者提供了在这样的环境下可证明是最优的保证。我们讨论了学习器在计算上可行的实现，并进一步证明了我们的算法在几个自然例子上获得了很强的正性能保证：例如，对数凹分布下的线性分离器或光滑概率分布下的光滑边界分类器。



## **36. Trust, but Verify: Robust Image Segmentation using Deep Learning**

信任，但要验证：使用深度学习的稳健图像分割 cs.CV

5 Pages, 8 Figures, conference

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.16999v2) [paper-pdf](http://arxiv.org/pdf/2310.16999v2)

**Authors**: Fahim Ahmed Zaman, Xiaodong Wu, Weiyu Xu, Milan Sonka, Raghuraman Mudumbai

**Abstract**: We describe a method for verifying the output of a deep neural network for medical image segmentation that is robust to several classes of random as well as worst-case perturbations i.e. adversarial attacks. This method is based on a general approach recently developed by the authors called "Trust, but Verify" wherein an auxiliary verification network produces predictions about certain masked features in the input image using the segmentation as an input. A well-designed auxiliary network will produce high-quality predictions when the input segmentations are accurate, but will produce low-quality predictions when the segmentations are incorrect. Checking the predictions of such a network with the original image allows us to detect bad segmentations. However, to ensure the verification method is truly robust, we need a method for checking the quality of the predictions that does not itself rely on a black-box neural network. Indeed, we show that previous methods for segmentation evaluation that do use deep neural regression networks are vulnerable to false negatives i.e. can inaccurately label bad segmentations as good. We describe the design of a verification network that avoids such vulnerability and present results to demonstrate its robustness compared to previous methods.

摘要: 我们描述了一种用于医学图像分割的深度神经网络输出的验证方法，该方法对几类随机和最坏情况的扰动，即对抗性攻击具有鲁棒性。该方法基于作者最近开发的一种被称为“信任，但验证”的通用方法，其中辅助验证网络使用分割作为输入来产生关于输入图像中的某些被屏蔽特征的预测。一个设计良好的辅助网络在输入分割准确时会产生高质量的预测，但当分割不正确时会产生低质量的预测。用原始图像检查这种网络的预测可以让我们检测到错误的分割。然而，为了确保验证方法真正稳健，我们需要一种方法来检查预测的质量，该方法本身不依赖于黑盒神经网络。事实上，我们表明，以前使用深度神经回归网络的分割评估方法很容易出现假阴性，即可能不准确地将不良分割标记为良好分割。我们描述了一个避免这种漏洞的验证网络的设计，并给出了与以前方法相比的结果来证明它的健壮性。



## **37. On the Exploitability of Instruction Tuning**

论指令调优的可开发性 cs.CR

NeurIPS 2023 camera-ready (21 pages, 10 figures)

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2306.17194v2) [paper-pdf](http://arxiv.org/pdf/2306.17194v2)

**Authors**: Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein

**Abstract**: Instruction tuning is an effective technique to align large language models (LLMs) with human intents. In this work, we investigate how an adversary can exploit instruction tuning by injecting specific instruction-following examples into the training data that intentionally changes the model's behavior. For example, an adversary can achieve content injection by injecting training examples that mention target content and eliciting such behavior from downstream models. To achieve this goal, we propose \textit{AutoPoison}, an automated data poisoning pipeline. It naturally and coherently incorporates versatile attack goals into poisoned data with the help of an oracle LLM. We showcase two example attacks: content injection and over-refusal attacks, each aiming to induce a specific exploitable behavior. We quantify and benchmark the strength and the stealthiness of our data poisoning scheme. Our results show that AutoPoison allows an adversary to change a model's behavior by poisoning only a small fraction of data while maintaining a high level of stealthiness in the poisoned examples. We hope our work sheds light on how data quality affects the behavior of instruction-tuned models and raises awareness of the importance of data quality for responsible deployments of LLMs. Code is available at \url{https://github.com/azshue/AutoPoison}.

摘要: 指令调优是一种将大型语言模型（LLM）与人类意图对齐的有效技术。在这项工作中，我们研究了对手如何通过将特定的预防性后续示例注入到训练数据中来利用指令调优，从而故意改变模型的行为。例如，攻击者可以通过注入提到目标内容的训练示例并从下游模型中引出此类行为来实现内容注入。为了实现这一目标，我们提出了一个自动化的数据中毒管道。它在Oracle LLM的帮助下，自然而连贯地将多功能攻击目标合并到中毒数据中。我们展示了两个示例攻击：内容注入和过度拒绝攻击，每一个都旨在诱导特定的可利用行为。我们量化和基准的强度和隐秘性，我们的数据中毒计划。我们的研究结果表明，AutoPoison允许对手通过仅毒害一小部分数据来改变模型的行为，同时在中毒的示例中保持高水平的隐蔽性。我们希望我们的工作能够阐明数据质量如何影响预警调整模型的行为，并提高人们对数据质量对于负责任的LLM部署的重要性的认识。代码可以在\url{https：//github.com/azshue/AutoPoison}上找到。



## **38. Purify++: Improving Diffusion-Purification with Advanced Diffusion Models and Control of Randomness**

Purify++：利用先进的扩散模型和随机性控制改进扩散净化 cs.LG

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18762v1) [paper-pdf](http://arxiv.org/pdf/2310.18762v1)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks can mislead neural network classifiers. The defense against adversarial attacks is important for AI safety. Adversarial purification is a family of approaches that defend adversarial attacks with suitable pre-processing. Diffusion models have been shown to be effective for adversarial purification. Despite their success, many aspects of diffusion purification still remain unexplored. In this paper, we investigate and improve upon three limiting designs of diffusion purification: the use of an improved diffusion model, advanced numerical simulation techniques, and optimal control of randomness. Based on our findings, we propose Purify++, a new diffusion purification algorithm that is now the state-of-the-art purification method against several adversarial attacks. Our work presents a systematic exploration of the limits of diffusion purification methods.

摘要: 对抗性攻击会误导神经网络分类器。对抗性攻击的防御对于AI安全非常重要。对抗性净化是一系列通过适当的预处理来防御对抗性攻击的方法。扩散模型已被证明是有效的对抗净化。尽管他们的成功，扩散纯化的许多方面仍然是未开发的。在本文中，我们调查和改进后的三个限制设计的扩散净化：使用一个改进的扩散模型，先进的数值模拟技术，和最优控制的随机性。基于我们的研究结果，我们提出了Purify++，这是一种新的扩散净化算法，现在是对抗几种对抗性攻击的最先进的净化方法。我们的工作提出了一个系统的探索扩散净化方法的局限性。



## **39. PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization**

对抗性泛化的PAC-贝叶斯谱归一化界 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.06182v2) [paper-pdf](http://arxiv.org/pdf/2310.06182v2)

**Authors**: Jiancong Xiao, Ruoyu Sun, Zhi- Quan Luo

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.

摘要: 深度神经网络(DNN)很容易受到敌意攻击。实验发现，对抗性健壮性泛化对于建立抵抗对抗性攻击的防御算法至关重要。因此，研究健壮性泛化的理论保障是很有意义的。本文基于PAC-Bayes方法(Neyshabur等人，2017年)，重点研究基于范数的复杂性。主要的挑战在于将关键成分(标准设置中的权重扰动范围)扩展到稳健设置。现有的尝试严重依赖于额外的强有力的假设，导致了宽松的界限。在这篇文章中，我们解决了这个问题，并为DNN提供了一个谱归一化的鲁棒泛化上界。与现有的边界相比，我们的边界有两个显著的优点：第一，它不依赖于额外的假设。其次，它相当紧凑，与标准泛化的界限一致。因此，我们的结果为理解健壮性概括提供了一个不同的视角：以前的研究中显示的标准和健壮性概化界限之间的不匹配项并不是导致健壮性概括较差的原因。相反，这些差异完全是由于数学问题。最后，我们将主要结果推广到对抗一般非EELL_p$攻击和其他神经网络结构的健壮性。



## **40. Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models**

重温ImageNet的对抗性训练：跨威胁模型的架构、训练和泛化 cs.CV

Accepted at NeurIPS 2023

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2303.01870v2) [paper-pdf](http://arxiv.org/pdf/2303.01870v2)

**Authors**: Naman D Singh, Francesco Croce, Matthias Hein

**Abstract**: While adversarial training has been extensively studied for ResNet architectures and low resolution datasets like CIFAR, much less is known for ImageNet. Given the recent debate about whether transformers are more robust than convnets, we revisit adversarial training on ImageNet comparing ViTs and ConvNeXts. Extensive experiments show that minor changes in architecture, most notably replacing PatchStem with ConvStem, and training scheme have a significant impact on the achieved robustness. These changes not only increase robustness in the seen $\ell_\infty$-threat model, but even more so improve generalization to unseen $\ell_1/\ell_2$-attacks. Our modified ConvNeXt, ConvNeXt + ConvStem, yields the most robust $\ell_\infty$-models across different ranges of model parameters and FLOPs, while our ViT + ConvStem yields the best generalization to unseen threat models.

摘要: 虽然针对ResNet架构和CIFAR等低分辨率数据集的对抗性训练已被广泛研究，但对ImageNet的研究要少得多。鉴于最近关于变压器是否比凸轮更健壮的辩论，我们回顾了ImageNet上的对抗性培训，比较了VITS和ConvNeXts。大量的实验表明，体系结构中的微小变化，最明显的是用ConvStem替换PatchStem，以及训练方案对实现的健壮性有重大影响。这些更改不仅增加了已见的$\ell_\inty$-威胁模型中的健壮性，更重要的是改进了对未见的$\ell_1/\ell_2$-攻击的泛化。我们改进的ConvNeXt，ConvNeXt+ConvStem，在不同的模型参数和失败范围内产生最健壮的$\ell_\inty$-模型，而我们的Vit+ConvStem产生对未知威胁模型的最佳概括。



## **41. Enhancing Adversarial Robustness via Score-Based Optimization**

通过基于得分的优化增强对抗鲁棒性 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2307.04333v3) [paper-pdf](http://arxiv.org/pdf/2307.04333v3)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.

摘要: 对抗性攻击有可能通过引入轻微的扰动来误导深度神经网络分类器。开发能够缓解这些攻击影响的算法，对于确保人工智能的安全使用至关重要。最近的研究表明，基于分数的扩散模型在对抗防御中是有效的。然而，现有的基于扩散的防御依赖于对扩散模型的逆随机微分方程的顺序模拟，这在计算上效率低下，并且产生次优结果。在本文中，我们提出了一种新的对抗防御方案ScoreOpt，该方案在测试时优化对手样本，在基于分数的先验的指导下，朝着原始干净数据的方向进行优化。我们在包括CIFAR10、CIFAR100和ImageNet在内的多个数据集上进行了全面的实验。实验结果表明，该方法在稳健性和推理速度上均优于现有的对抗性防御方法。



## **42. Training Socially Aligned Language Models on Simulated Social Interactions**

关于模拟社会互动的社会一致性语言模型的培训 cs.CL

Code, data, and models can be downloaded via  https://github.com/agi-templar/Stable-Alignment

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2305.16960v3) [paper-pdf](http://arxiv.org/pdf/2305.16960v3)

**Authors**: Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Denny Zhou, Andrew M. Dai, Diyi Yang, Soroush Vosoughi

**Abstract**: Social alignment in AI systems aims to ensure that these models behave according to established societal values. However, unlike humans, who derive consensus on value judgments through social interaction, current language models (LMs) are trained to rigidly replicate their training corpus in isolation, leading to subpar generalization in unfamiliar scenarios and vulnerability to adversarial attacks. This work presents a novel training paradigm that permits LMs to learn from simulated social interactions. In comparison to existing methodologies, our approach is considerably more scalable and efficient, demonstrating superior performance in alignment benchmarks and human evaluations. This paradigm shift in the training of LMs brings us a step closer to developing AI systems that can robustly and accurately reflect societal norms and values.

摘要: 人工智能系统中的社会一致性旨在确保这些模型的行为符合既定的社会价值观。然而，与通过社交互动就价值判断达成共识的人类不同，当前的语言模型(LMS)被训练成孤立地僵硬地复制他们的训练语料库，导致在不熟悉的场景中的泛化能力不佳，并且容易受到对手攻击。这项工作提出了一种新的训练范式，允许LMS从模拟的社会互动中学习。与现有方法相比，我们的方法可伸缩性更强，效率更高，在比对基准和人工评估方面表现出卓越的性能。LMS培训的这种范式转变使我们离开发能够有力而准确地反映社会规范和价值观的人工智能系统又近了一步。



## **43. Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots**

设置陷阱：通过蜜罐捕获和击败预先训练的语言模型中的后门 cs.LG

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18633v1) [paper-pdf](http://arxiv.org/pdf/2310.18633v1)

**Authors**: Ruixiang Tang, Jiayi Yuan, Yiming Li, Zirui Liu, Rui Chen, Xia Hu

**Abstract**: In the field of natural language processing, the prevalent approach involves fine-tuning pretrained language models (PLMs) using local samples. Recent research has exposed the susceptibility of PLMs to backdoor attacks, wherein the adversaries can embed malicious prediction behaviors by manipulating a few training samples. In this study, our objective is to develop a backdoor-resistant tuning procedure that yields a backdoor-free model, no matter whether the fine-tuning dataset contains poisoned samples. To this end, we propose and integrate a honeypot module into the original PLM, specifically designed to absorb backdoor information exclusively. Our design is motivated by the observation that lower-layer representations in PLMs carry sufficient backdoor features while carrying minimal information about the original tasks. Consequently, we can impose penalties on the information acquired by the honeypot module to inhibit backdoor creation during the fine-tuning process of the stem network. Comprehensive experiments conducted on benchmark datasets substantiate the effectiveness and robustness of our defensive strategy. Notably, these results indicate a substantial reduction in the attack success rate ranging from 10\% to 40\% when compared to prior state-of-the-art methods.

摘要: 在自然语言处理领域，流行的方法包括使用本地样本微调预训练语言模型(PLM)。最近的研究揭示了PLM对后门攻击的敏感性，其中攻击者可以通过操纵少量训练样本来嵌入恶意预测行为。在这项研究中，我们的目标是开发一种防止后门的调整过程，无论微调数据集是否包含有毒样本，都可以产生一个没有后门的模型。为此，我们提出并集成了一个蜜罐模块到原来的PLM中，专门为吸收后门信息而专门设计。我们的设计动机是观察到PLM中的低层表示具有足够的后门功能，同时携带关于原始任务的最少信息。因此，我们可以对蜜罐模块获取的信息施加惩罚，以阻止在茎网络的微调过程中创建后门。在基准数据集上进行的全面实验证明了我们的防御策略的有效性和健壮性。值得注意的是，这些结果表明，与以前最先进的方法相比，攻击成功率大幅降低，从10%到40%不等。



## **44. Where have you been? A Study of Privacy Risk for Point-of-Interest Recommendation**

你去哪儿了？兴趣点推荐中的隐私风险研究 cs.LG

26 pages

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18606v1) [paper-pdf](http://arxiv.org/pdf/2310.18606v1)

**Authors**: Kunlin Cai, Jinghuai Zhang, Will Shand, Zhiqing Hong, Guang Wang, Desheng Zhang, Jianfeng Chi, Yuan Tian

**Abstract**: As location-based services (LBS) have grown in popularity, the collection of human mobility data has become increasingly extensive to build machine learning (ML) models offering enhanced convenience to LBS users. However, the convenience comes with the risk of privacy leakage since this type of data might contain sensitive information related to user identities, such as home/work locations. Prior work focuses on protecting mobility data privacy during transmission or prior to release, lacking the privacy risk evaluation of mobility data-based ML models. To better understand and quantify the privacy leakage in mobility data-based ML models, we design a privacy attack suite containing data extraction and membership inference attacks tailored for point-of-interest (POI) recommendation models, one of the most widely used mobility data-based ML models. These attacks in our attack suite assume different adversary knowledge and aim to extract different types of sensitive information from mobility data, providing a holistic privacy risk assessment for POI recommendation models. Our experimental evaluation using two real-world mobility datasets demonstrates that current POI recommendation models are vulnerable to our attacks. We also present unique findings to understand what types of mobility data are more susceptible to privacy attacks. Finally, we evaluate defenses against these attacks and highlight future directions and challenges.

摘要: 随着基于位置的服务(LBS)越来越流行，人类移动性数据的收集已经变得越来越广泛，以建立机器学习(ML)模型，为LBS用户提供更好的便利。然而，随之而来的是隐私泄露的风险，因为这种类型的数据可能包含与用户身份相关的敏感信息，如家庭/工作地点。以往的工作主要集中在移动数据传输过程中或发布前的隐私保护上，缺乏对基于移动数据的ML模型的隐私风险评估。为了更好地理解和量化基于移动数据的ML模型中的隐私泄漏，我们设计了一个隐私攻击套件，其中包含针对兴趣点(POI)推荐模型的数据提取和成员关系推理攻击，该模型是应用最广泛的基于移动数据的ML模型之一。我们攻击套件中的这些攻击假设了不同的对手知识，旨在从移动数据中提取不同类型的敏感信息，为POI推荐模型提供全面的隐私风险评估。我们使用两个真实的移动数据集进行的实验评估表明，当前的POI推荐模型容易受到我们的攻击。我们还提出了独特的发现，以了解哪些类型的移动数据更容易受到隐私攻击。最后，我们评估了针对这些攻击的防御措施，并强调了未来的方向和挑战。



## **45. Large Language Models Are Better Adversaries: Exploring Generative Clean-Label Backdoor Attacks Against Text Classifiers**

大型语言模型是更好的对手：探索针对文本分类器的生成性干净标签后门攻击 cs.LG

Accepted at EMNLP 2023 Findings

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18603v1) [paper-pdf](http://arxiv.org/pdf/2310.18603v1)

**Authors**: Wencong You, Zayd Hammoudeh, Daniel Lowd

**Abstract**: Backdoor attacks manipulate model predictions by inserting innocuous triggers into training and test data. We focus on more realistic and more challenging clean-label attacks where the adversarial training examples are correctly labeled. Our attack, LLMBkd, leverages language models to automatically insert diverse style-based triggers into texts. We also propose a poison selection technique to improve the effectiveness of both LLMBkd as well as existing textual backdoor attacks. Lastly, we describe REACT, a baseline defense to mitigate backdoor attacks via antidote training examples. Our evaluations demonstrate LLMBkd's effectiveness and efficiency, where we consistently achieve high attack success rates across a wide range of styles with little effort and no model training.

摘要: 后门攻击通过在训练和测试数据中插入无害的触发器来操纵模型预测。我们专注于更现实、更具挑战性的干净标签攻击，其中对抗性训练示例被正确标记。我们的攻击LLMBkd利用语言模型自动将不同的基于样式的触发器插入到文本中。我们还提出了一种毒物选择技术来提高LLMBkd和现有文本后门攻击的有效性。最后，我们描述了Reaction，一种通过解毒剂训练示例来减少后门攻击的基线防御。我们的评估证明了LLMBkd的有效性和效率，在这种情况下，我们几乎不费力气，也没有模型训练，就能在各种风格中始终获得高攻击成功率。



## **46. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

在评估转会对抗性攻击方面的良好做法 cs.CR

An extended version can be found at arXiv:2310.11850. Code and a list  of categorized attacks are available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2211.09565v3) [paper-pdf](http://arxiv.org/pdf/2211.09565v3)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of this field is difficult to assess due to two common limitations in existing evaluations. First, different methods are often not systematically and fairly evaluated in a one-to-one comparison. Second, only transferability is evaluated but another key attack property, stealthiness, is largely overlooked. In this work, we design good practices to address these limitations, and we present the first comprehensive evaluation of transfer attacks, covering 23 representative attacks against 9 defenses on ImageNet. In particular, we propose to categorize existing attacks into five categories, which enables our systematic category-wise analyses. These analyses lead to new findings that even challenge existing knowledge and also help determine the optimal attack hyperparameters for our attack-wise comprehensive evaluation. We also pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Overall, our new insights into transferability and stealthiness lead to actionable good practices for future evaluations.

摘要: 在现实世界的黑盒场景中，传输敌意攻击会引发严重的安全问题。然而，由于现有评价中的两个共同限制，这一领域的实际进展很难评估。首先，不同的方法往往不能在一对一的比较中得到系统和公平的评估。其次，只评估了可转移性，但另一个关键攻击属性--隐蔽性在很大程度上被忽视了。在这项工作中，我们设计了良好的实践来解决这些限制，并提出了第一个全面的传输攻击评估，涵盖了23个典型攻击对9个防御ImageNet。特别是，我们建议将现有攻击分为五类，这使得我们能够进行系统的分类分析。这些分析导致了新的发现，甚至挑战了现有的知识，并有助于为我们的攻击智能综合评估确定最佳攻击超参数。我们还特别关注隐蔽性，采用了不同的隐蔽性度量标准，并研究了新的、更细粒度的特征。总体而言，我们对可转移性和隐蔽性的新见解为未来的评估提供了可操作的良好做法。



## **47. A General Framework for Robust G-Invariance in G-Equivariant Networks**

G-等变网络中稳健G-不变性的一个通用框架 cs.LG

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18564v1) [paper-pdf](http://arxiv.org/pdf/2310.18564v1)

**Authors**: Sophia Sanborn, Nina Miolane

**Abstract**: We introduce a general method for achieving robust group-invariance in group-equivariant convolutional neural networks ($G$-CNNs), which we call the $G$-triple-correlation ($G$-TC) layer. The approach leverages the theory of the triple-correlation on groups, which is the unique, lowest-degree polynomial invariant map that is also complete. Many commonly used invariant maps - such as the max - are incomplete: they remove both group and signal structure. A complete invariant, by contrast, removes only the variation due to the actions of the group, while preserving all information about the structure of the signal. The completeness of the triple correlation endows the $G$-TC layer with strong robustness, which can be observed in its resistance to invariance-based adversarial attacks. In addition, we observe that it yields measurable improvements in classification accuracy over standard Max $G$-Pooling in $G$-CNN architectures. We provide a general and efficient implementation of the method for any discretized group, which requires only a table defining the group's product structure. We demonstrate the benefits of this method for $G$-CNNs defined on both commutative and non-commutative groups - $SO(2)$, $O(2)$, $SO(3)$, and $O(3)$ (discretized as the cyclic $C8$, dihedral $D16$, chiral octahedral $O$ and full octahedral $O_h$ groups) - acting on $\mathbb{R}^2$ and $\mathbb{R}^3$ on both $G$-MNIST and $G$-ModelNet10 datasets.

摘要: 介绍了一种在群等变卷积神经网络($G$-CNN)中实现稳健群不变性的一般方法，我们称之为$G$-三相关($G$-TC)层。该方法利用了群上的三重相关理论，这是唯一的、也是完全的最低次多项式不变映射。许多常用的不变量映射--例如最大不变量映射--是不完整的：它们既去掉了组结构又去掉了信号结构。相比之下，完全不变量只删除由于群的作用而引起的变化，同时保留关于信号结构的所有信息。三重相关性的完备性赋予$G$-TC层很强的健壮性，这可以从它对基于不变性的对手攻击的抵抗中观察到。此外，我们观察到它在分类准确率方面比标准的MAX$G$-在$G$-CNN架构中合并产生了显著的改进。我们为任何离散化的组提供了该方法的通用和有效的实现，它只需要一个定义组的产品结构的表。我们证明了这种方法对于定义在交换群和非交换群上的$G$-CNN的好处--$SO(2)$、$O(2)$、$SO(3)$和$O(3)$(离散为循环$C8$、二面体$D16$、手征八面体$O$和全八面体$O_h$群)-作用于$G$-MNIST和$G$-ModelNetdata集上的$\mathbb{R}^2$和$\mathbb{R}^3$。



## **48. Understanding and Improving Ensemble Adversarial Defense**

认识和完善整体对抗防御 cs.LG

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18477v1) [paper-pdf](http://arxiv.org/pdf/2310.18477v1)

**Authors**: Yian Deng, Tingting Mu

**Abstract**: The strategy of ensemble has become popular in adversarial defense, which trains multiple base classifiers to defend against adversarial attacks in a cooperative manner. Despite the empirical success, theoretical explanations on why an ensemble of adversarially trained classifiers is more robust than single ones remain unclear. To fill in this gap, we develop a new error theory dedicated to understanding ensemble adversarial defense, demonstrating a provable 0-1 loss reduction on challenging sample sets in an adversarial defense scenario. Guided by this theory, we propose an effective approach to improve ensemble adversarial defense, named interactive global adversarial training (iGAT). The proposal includes (1) a probabilistic distributing rule that selectively allocates to different base classifiers adversarial examples that are globally challenging to the ensemble, and (2) a regularization term to rescue the severest weaknesses of the base classifiers. Being tested over various existing ensemble adversarial defense techniques, iGAT is capable of boosting their performance by increases up to 17% evaluated using CIFAR10 and CIFAR100 datasets under both white-box and black-box attacks.

摘要: 在对抗性防御中，集成策略已经成为一种流行的策略，它训练多个基分类器以协作的方式防御对抗性攻击。尽管取得了经验上的成功，但关于为什么对抗性训练的分类器集合比单个分类器更稳健的理论解释仍然不清楚。为了填补这一空白，我们发展了一种新的错误理论，致力于理解集成对抗性防御，展示了在对抗性防御场景中挑战样本集上可证明的0-1损失减少。在此理论指导下，我们提出了一种提高集成对抗能力的有效方法，即交互式全局对抗训练(IGAT)。该方案包括(1)概率分布规则，它选择性地将对集成具有全局挑战性的对抗性实例分配给不同的基分类器；(2)正则化项以弥补基分类器最严重的弱点。在对各种现有的集成对抗防御技术进行测试后，iGAT能够将其性能提高17%，在白盒和黑盒攻击下使用CIFAR10和CIFAR100数据集进行评估。



## **49. LipSim: A Provably Robust Perceptual Similarity Metric**

LipSim：一种可证明的稳健感知相似性度量 cs.CV

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18274v1) [paper-pdf](http://arxiv.org/pdf/2310.18274v1)

**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Farshad Khorrami, Siddharth Garg

**Abstract**: Recent years have seen growing interest in developing and applying perceptual similarity metrics. Research has shown the superiority of perceptual metrics over pixel-wise metrics in aligning with human perception and serving as a proxy for the human visual system. On the other hand, as perceptual metrics rely on neural networks, there is a growing concern regarding their resilience, given the established vulnerability of neural networks to adversarial attacks. It is indeed logical to infer that perceptual metrics may inherit both the strengths and shortcomings of neural networks. In this work, we demonstrate the vulnerability of state-of-the-art perceptual similarity metrics based on an ensemble of ViT-based feature extractors to adversarial attacks. We then propose a framework to train a robust perceptual similarity metric called LipSim (Lipschitz Similarity Metric) with provable guarantees. By leveraging 1-Lipschitz neural networks as the backbone, LipSim provides guarded areas around each data point and certificates for all perturbations within an $\ell_2$ ball. Finally, a comprehensive set of experiments shows the performance of LipSim in terms of natural and certified scores and on the image retrieval application. The code is available at https://github.com/SaraGhazanfari/LipSim.

摘要: 近年来，人们对开发和应用感知相似性度量的兴趣与日俱增。研究表明，与像素度量相比，感知度量在与人类感知和作为人类视觉系统的代理方面具有优势。另一方面，由于感知指标依赖于神经网络，鉴于神经网络对对手攻击的公认脆弱性，人们越来越担心其弹性。推断感知指标可能继承了神经网络的长处和短处，这确实是合乎逻辑的。在这项工作中，我们展示了基于基于VIT的特征提取集合的最新感知相似性度量在对抗攻击中的脆弱性。然后，我们提出了一个框架来训练一个健壮的感知相似性度量，称为LipSim(Lipschitz相似性度量)，并具有可证明的保证。通过利用1-Lipschitz神经网络作为主干，LipSim在每个数据点周围提供保护区域，并为$\ell_2$球内的所有扰动提供证书。最后，一组全面的实验显示了LipSim在自然分数和认证分数以及图像检索应用方面的性能。代码可在https://github.com/SaraGhazanfari/LipSim.上获得



## **50. $α$-Mutual Information: A Tunable Privacy Measure for Privacy Protection in Data Sharing**

$α$-互信息：数据共享中隐私保护的可调隐私度量 cs.LG

2023 22nd IEEE International Conference on Machine Learning and  Applications (ICMLA)

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18241v1) [paper-pdf](http://arxiv.org/pdf/2310.18241v1)

**Authors**: MirHamed Jafarzadeh Asl, Mohammadhadi Shateri, Fabrice Labeau

**Abstract**: This paper adopts Arimoto's $\alpha$-Mutual Information as a tunable privacy measure, in a privacy-preserving data release setting that aims to prevent disclosing private data to adversaries. By fine-tuning the privacy metric, we demonstrate that our approach yields superior models that effectively thwart attackers across various performance dimensions. We formulate a general distortion-based mechanism that manipulates the original data to offer privacy protection. The distortion metrics are determined according to the data structure of a specific experiment. We confront the problem expressed in the formulation by employing a general adversarial deep learning framework that consists of a releaser and an adversary, trained with opposite goals. This study conducts empirical experiments on images and time-series data to verify the functionality of $\alpha$-Mutual Information. We evaluate the privacy-utility trade-off of customized models and compare them to mutual information as the baseline measure. Finally, we analyze the consequence of an attacker's access to side information about private data and witness that adapting the privacy measure results in a more refined model than the state-of-the-art in terms of resiliency against side information.

摘要: 本文采用Arimoto的$\alpha$-互信息作为一个可调的隐私措施，在隐私保护的数据发布设置，旨在防止泄露私人数据给对手。通过微调的隐私度量，我们证明了我们的方法产生优越的模型，有效地阻止攻击者在各个性能方面。我们制定了一个通用的扭曲为基础的机制，操纵原始数据提供隐私保护。失真度量根据特定实验的数据结构来确定。我们通过采用一个一般的对抗性深度学习框架来面对公式中表达的问题，该框架由一个攻击者和一个对手组成，他们以相反的目标进行训练。本研究针对影像与时间序列资料进行实证实验，以验证$\alpha$-Mutual Information的功能。我们评估了定制模型的隐私效用权衡，并将其与作为基线测量的互信息进行比较。最后，我们分析了攻击者访问有关私有数据的边信息的后果，并证明了在对边信息的弹性方面，调整隐私措施会导致比最先进的模型更精细的模型。



