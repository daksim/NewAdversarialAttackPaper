# Latest Adversarial Attack Papers
**update at 2024-10-28 09:35:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Robust Thompson Sampling Algorithms Against Reward Poisoning Attacks**

针对奖励中毒攻击的稳健Thompson抽样算法 cs.LG

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19705v1) [paper-pdf](http://arxiv.org/pdf/2410.19705v1)

**Authors**: Yinglun Xu, Zhiwei Wang, Gagandeep Singh

**Abstract**: Thompson sampling is one of the most popular learning algorithms for online sequential decision-making problems and has rich real-world applications. However, current Thompson sampling algorithms are limited by the assumption that the rewards received are uncorrupted, which may not be true in real-world applications where adversarial reward poisoning exists. To make Thompson sampling more reliable, we want to make it robust against adversarial reward poisoning. The main challenge is that one can no longer compute the actual posteriors for the true reward, as the agent can only observe the rewards after corruption. In this work, we solve this problem by computing pseudo-posteriors that are less likely to be manipulated by the attack. We propose robust algorithms based on Thompson sampling for the popular stochastic and contextual linear bandit settings in both cases where the agent is aware or unaware of the budget of the attacker. We theoretically show that our algorithms guarantee near-optimal regret under any attack strategy.

摘要: Thompson抽样是在线序贯决策问题中最常用的学习算法之一，在现实世界中有着广泛的应用。然而，当前的Thompson采样算法受到接收到的奖励是未被破坏的假设的限制，这在存在对抗性奖励中毒的现实应用中可能是不成立的。为了使汤普森抽样更可靠，我们希望使其对对手奖励中毒具有健壮性。主要的挑战是，人们不再能计算出真正报酬的实际后遗症，因为代理人只能观察腐败后的报酬。在这项工作中，我们通过计算不太可能被攻击操纵的伪后验来解决这个问题。对于常见的随机和上下文线性盗贼设置，我们提出了基于Thompson采样的稳健算法，在代理知道和不知道攻击者预算的两种情况下都是如此。我们从理论上证明了我们的算法在任何攻击策略下都能保证近似最优的错误。



## **2. A constrained optimization approach to improve robustness of neural networks**

提高神经网络鲁棒性的约束优化方法 cs.LG

29 pages, 4 figures, 5 tables

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2409.13770v2) [paper-pdf](http://arxiv.org/pdf/2409.13770v2)

**Authors**: Shudian Zhao, Jan Kronqvist

**Abstract**: In this paper, we present a novel nonlinear programming-based approach to fine-tune pre-trained neural networks to improve robustness against adversarial attacks while maintaining high accuracy on clean data. Our method introduces adversary-correction constraints to ensure correct classification of adversarial data and minimizes changes to the model parameters. We propose an efficient cutting-plane-based algorithm to iteratively solve the large-scale nonconvex optimization problem by approximating the feasible region through polyhedral cuts and balancing between robustness and accuracy. Computational experiments on standard datasets such as MNIST and CIFAR10 demonstrate that the proposed approach significantly improves robustness, even with a very small set of adversarial data, while maintaining minimal impact on accuracy.

摘要: 在本文中，我们提出了一种新型的基于非线性规划的方法来微调预训练的神经网络，以提高针对对抗性攻击的鲁棒性，同时保持干净数据的高准确性。我们的方法引入了对抗修正约束，以确保对抗数据的正确分类，并最大限度地减少对模型参数的更改。我们提出了一种高效的基于切割平面的算法，通过通过多边形切割逼近可行区域并平衡鲁棒性和准确性来迭代解决大规模非凸优化问题。对MNIST和CIFAR 10等标准数据集的计算实验表明，即使使用非常小的对抗数据集，所提出的方法也能显着提高稳健性，同时保持对准确性的影响最小。



## **3. Detecting adversarial attacks on random samples**

检测对随机样本的对抗攻击 math.PR

title changed; introduction expanded; new results about spherical  attacks

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2408.06166v2) [paper-pdf](http://arxiv.org/pdf/2408.06166v2)

**Authors**: Gleb Smirnov

**Abstract**: This paper studies the problem of detecting adversarial perturbations in a sequence of observations. Given a data sample $X_1, \ldots, X_n$ drawn from a standard normal distribution, an adversary, after observing the sample, can perturb each observation by a fixed magnitude or leave it unchanged. We explore the relationship between the perturbation magnitude, the sparsity of the perturbation, and the detectability of the adversary's actions, establishing precise thresholds for when detection becomes impossible.

摘要: 本文研究了在观察序列中检测对抗性扰动的问题。给定从标准正态分布中提取的数据样本$X_1，\ldots，X_n$，对手在观察样本后可以以固定幅度扰乱每个观察或保持其不变。我们探索了扰动幅度、扰动的稀疏性和对手行为的可检测性之间的关系，为何时检测变得不可能建立精确的阈值。



## **4. Corpus Poisoning via Approximate Greedy Gradient Descent**

通过近似贪婪梯度下降来中毒 cs.IR

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2406.05087v2) [paper-pdf](http://arxiv.org/pdf/2406.05087v2)

**Authors**: Jinyan Su, Preslav Nakov, Claire Cardie

**Abstract**: Dense retrievers are widely used in information retrieval and have also been successfully extended to other knowledge intensive areas such as language models, e.g., Retrieval-Augmented Generation (RAG) systems. Unfortunately, they have recently been shown to be vulnerable to corpus poisoning attacks in which a malicious user injects a small fraction of adversarial passages into the retrieval corpus to trick the system into returning these passages among the top-ranked results for a broad set of user queries. Further study is needed to understand the extent to which these attacks could limit the deployment of dense retrievers in real-world applications. In this work, we propose Approximate Greedy Gradient Descent (AGGD), a new attack on dense retrieval systems based on the widely used HotFlip method for efficiently generating adversarial passages. We demonstrate that AGGD can select a higher quality set of token-level perturbations than HotFlip by replacing its random token sampling with a more structured search. Experimentally, we show that our method achieves a high attack success rate on several datasets and using several retrievers, and can generalize to unseen queries and new domains. Notably, our method is extremely effective in attacking the ANCE retrieval model, achieving attack success rates that are 15.24\% and 17.44\% higher on the NQ and MS MARCO datasets, respectively, compared to HotFlip. Additionally, we demonstrate AGGD's potential to replace HotFlip in other adversarial attacks, such as knowledge poisoning of RAG systems.

摘要: 密集检索器被广泛应用于信息检索，也被成功地扩展到其他知识密集型领域，例如语言模型，例如检索-增强生成(RAG)系统。不幸的是，它们最近被证明容易受到语料库中毒攻击，在这种攻击中，恶意用户将一小部分对抗性段落注入检索语料库，以欺骗系统返回针对广泛的用户查询集合的排名靠前的结果中的这些段落。需要进一步的研究来了解这些攻击在多大程度上会限制密集检索器在现实世界应用中的部署。在这项工作中，我们提出了近似贪婪梯度下降(AGGD)，一种新的攻击密集检索系统的基础上，广泛使用的HotFlip方法，以有效地生成敌意段落。我们证明，通过用更结构化的搜索取代随机令牌抽样，AGGD可以选择比HotFlip更高质量的令牌级扰动集。实验表明，我们的方法在多个数据集和多个检索器上取得了很高的攻击成功率，并且可以推广到未知的查询和新的领域。值得注意的是，我们的方法在攻击ANCE检索模型方面非常有效，在NQ和MS Marco数据集上的攻击成功率分别比HotFlip高15.24和17.44。此外，我们还展示了AGGD在其他对抗性攻击中取代HotFlip的潜力，例如RAG系统的知识中毒。



## **5. Adversarial Attacks on Large Language Models Using Regularized Relaxation**

使用正规松弛对大型语言模型的对抗攻击 cs.LG

8 pages, 6 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19160v1) [paper-pdf](http://arxiv.org/pdf/2410.19160v1)

**Authors**: Samuel Jacob Chacko, Sajib Biswas, Chashi Mahiul Islam, Fatema Tabassum Liza, Xiuwen Liu

**Abstract**: As powerful Large Language Models (LLMs) are now widely used for numerous practical applications, their safety is of critical importance. While alignment techniques have significantly improved overall safety, LLMs remain vulnerable to carefully crafted adversarial inputs. Consequently, adversarial attack methods are extensively used to study and understand these vulnerabilities. However, current attack methods face significant limitations. Those relying on optimizing discrete tokens suffer from limited efficiency, while continuous optimization techniques fail to generate valid tokens from the model's vocabulary, rendering them impractical for real-world applications. In this paper, we propose a novel technique for adversarial attacks that overcomes these limitations by leveraging regularized gradients with continuous optimization methods. Our approach is two orders of magnitude faster than the state-of-the-art greedy coordinate gradient-based method, significantly improving the attack success rate on aligned language models. Moreover, it generates valid tokens, addressing a fundamental limitation of existing continuous optimization methods. We demonstrate the effectiveness of our attack on five state-of-the-art LLMs using four datasets.

摘要: 随着强大的大型语言模型(LLM)在众多实际应用中的广泛应用，它们的安全性至关重要。虽然对齐技术显著提高了总体安全性，但LLM仍然容易受到精心设计的敌方输入的影响。因此，对抗性攻击方法被广泛用于研究和理解这些漏洞。然而，目前的攻击方法面临着很大的局限性。那些依赖于优化离散令牌的人效率有限，而连续优化技术无法从模型的词汇表中生成有效的令牌，这使得它们在现实世界中的应用不切实际。在本文中，我们提出了一种新的对抗性攻击技术，通过利用正则化的梯度和连续优化方法来克服这些局限性。我们的方法比最先进的贪婪坐标梯度方法快两个数量级，显著提高了对齐语言模型的攻击成功率。此外，它还生成有效的令牌，解决了现有连续优化方法的一个基本限制。我们使用四个数据集演示了我们对五个最先进的LLM的攻击的有效性。



## **6. Provably Robust Watermarks for Open-Source Language Models**

开源语言模型的可证明稳健的水印 cs.CR

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18861v1) [paper-pdf](http://arxiv.org/pdf/2410.18861v1)

**Authors**: Miranda Christ, Sam Gunn, Tal Malkin, Mariana Raykova

**Abstract**: The recent explosion of high-quality language models has necessitated new methods for identifying AI-generated text. Watermarking is a leading solution and could prove to be an essential tool in the age of generative AI. Existing approaches embed watermarks at inference and crucially rely on the large language model (LLM) specification and parameters being secret, which makes them inapplicable to the open-source setting. In this work, we introduce the first watermarking scheme for open-source LLMs. Our scheme works by modifying the parameters of the model, but the watermark can be detected from just the outputs of the model. Perhaps surprisingly, we prove that our watermarks are unremovable under certain assumptions about the adversary's knowledge. To demonstrate the behavior of our construction under concrete parameter instantiations, we present experimental results with OPT-6.7B and OPT-1.3B. We demonstrate robustness to both token substitution and perturbation of the model parameters. We find that the stronger of these attacks, the model-perturbation attack, requires deteriorating the quality score to 0 out of 100 in order to bring the detection rate down to 50%.

摘要: 最近高质量语言模型的爆炸性增长需要新的方法来识别人工智能生成的文本。水印是一种领先的解决方案，可能会被证明是生成性人工智能时代的重要工具。现有的方法在推理时嵌入水印，重要的是依赖于大型语言模型(LLM)规范和参数是保密的，这使得它们不适用于开源环境。在这项工作中，我们介绍了第一个用于开源LLMS的水印方案。我们的方案通过修改模型的参数来工作，但仅从模型的输出就可以检测到水印。也许令人惊讶的是，我们证明了我们的水印在关于对手知识的某些假设下是不可移除的。为了演示我们的构造在混凝土参数实例化下的行为，我们给出了使用OPT-6.7B和OPT-1.3B的实验结果。我们证明了对令牌替换和模型参数摄动的稳健性。我们发现，在这些攻击中，较强的模型扰动攻击需要将质量分数恶化到0分(满分100分)，才能将检测率降至50%。



## **7. Rethinking Randomized Smoothing from the Perspective of Scalability**

从可扩展性的角度重新思考随机平滑 cs.LG

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2312.12608v2) [paper-pdf](http://arxiv.org/pdf/2312.12608v2)

**Authors**: Anupriya Kumari, Devansh Bhardwaj, Sukrit Jindal

**Abstract**: Machine learning models have demonstrated remarkable success across diverse domains but remain vulnerable to adversarial attacks. Empirical defense mechanisms often fail, as new attacks constantly emerge, rendering existing defenses obsolete, shifting the focus to certification-based defenses. Randomized smoothing has emerged as a promising technique among notable advancements. This study reviews the theoretical foundations and empirical effectiveness of randomized smoothing and its derivatives in verifying machine learning classifiers from a perspective of scalability. We provide an in-depth exploration of the fundamental concepts underlying randomized smoothing, highlighting its theoretical guarantees in certifying robustness against adversarial perturbations and discuss the challenges of existing methodologies.

摘要: 机器学习模型在不同领域取得了显着的成功，但仍然容易受到对抗攻击。经验防御机制经常会失败，因为新的攻击不断出现，使现有的防御变得过时，重点转向基于认证的防御。随机平滑已成为一种值得注意的进步中一种有前途的技术。本研究从可扩展性的角度审查了随机平滑及其衍生品在验证机器学习分类器方面的理论基础和经验有效性。我们对随机平滑的基本概念进行了深入探索，强调了其在证明对抗性扰动稳健性方面的理论保证，并讨论了现有方法论的挑战。



## **8. GADT: Enhancing Transferable Adversarial Attacks through Gradient-guided Adversarial Data Transformation**

GADT：通过用户引导的对抗数据转换增强可转移对抗攻击 cs.AI

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18648v1) [paper-pdf](http://arxiv.org/pdf/2410.18648v1)

**Authors**: Yating Ma, Xiaogang Xu, Liming Fang, Zhe Liu

**Abstract**: Current Transferable Adversarial Examples (TAE) are primarily generated by adding Adversarial Noise (AN). Recent studies emphasize the importance of optimizing Data Augmentation (DA) parameters along with AN, which poses a greater threat to real-world AI applications. However, existing DA-based strategies often struggle to find optimal solutions due to the challenging DA search procedure without proper guidance. In this work, we propose a novel DA-based attack algorithm, GADT. GADT identifies suitable DA parameters through iterative antagonism and uses posterior estimates to update AN based on these parameters. We uniquely employ a differentiable DA operation library to identify adversarial DA parameters and introduce a new loss function as a metric during DA optimization. This loss term enhances adversarial effects while preserving the original image content, maintaining attack crypticity. Extensive experiments on public datasets with various networks demonstrate that GADT can be integrated with existing transferable attack methods, updating their DA parameters effectively while retaining their AN formulation strategies. Furthermore, GADT can be utilized in other black-box attack scenarios, e.g., query-based attacks, offering a new avenue to enhance attacks on real-world AI applications in both research and industrial contexts.

摘要: 目前的可转移对抗性实例(TAE)主要是通过添加对抗性噪声(AN)来生成的。最近的研究强调了优化数据增强(DA)参数和AN的重要性，这对现实世界的AI应用构成了更大的威胁。然而，现有的基于DA的策略往往难以找到最优解决方案，因为DA搜索程序在没有适当指导的情况下具有挑战性。在本文中，我们提出了一种新的基于DA的攻击算法GADT。GADT通过迭代拮抗来识别合适的DA参数，并基于这些参数使用后验估计来更新AN。我们独特地使用了一个可微的DA运算库来识别对抗性DA参数，并在DA优化过程中引入了一个新的损失函数作为度量。这种损失项增强了对抗效果，同时保留了原始图像内容，保持了攻击的密码性。在具有不同网络的公共数据集上的大量实验表明，GADT可以与现有的可转移攻击方法相集成，在保持其AN制定策略的同时有效地更新它们的DA参数。此外，GADT还可以用于其他黑盒攻击场景，例如基于查询的攻击，为在研究和工业环境中增强对现实世界人工智能应用的攻击提供了一种新的途径。



## **9. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

18 pages

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18469v1) [paper-pdf](http://arxiv.org/pdf/2410.18469v1)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99% ASR on GPT-3.5 and 49% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它表现出很强的攻击可转换性，尽管只在Llama3上进行了优化，但在GPT-3.5上实现了99%的ASR，在GPT-4上实现了49%的ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。我们的代码请访问：https://github.com/SunChungEn/ADV-LLM



## **10. Effects of Scale on Language Model Robustness**

规模对语言模型稳健性的影响 cs.LG

36 pages; updated to include new results and analysis

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2407.18213v3) [paper-pdf](http://arxiv.org/pdf/2407.18213v3)

**Authors**: Nikolaus Howe, Ian McKenzie, Oskar Hollinsworth, Michał Zajac, Tom Tseng, Aaron Tucker, Pierre-Luc Bacon, Adam Gleave

**Abstract**: Language models exhibit scaling laws, whereby increasing model and dataset size yields predictable decreases in negative log likelihood, unlocking a dazzling array of capabilities. This phenomenon spurs many companies to train ever larger models in pursuit of ever improved performance. Yet, these models are vulnerable to adversarial inputs such as ``jailbreaks'' and prompt injections that induce models to perform undesired behaviors, posing a growing risk as models become more capable. Prior work indicates that computer vision models become more robust with model and data scaling, raising the question: does language model robustness also improve with scale?   We study this question empirically in the classification setting, finding that without explicit defense training, larger models tend to be modestly more robust on most tasks, though the effect is not reliable. Even with the advantage conferred by scale, undefended models remain easy to attack in absolute terms, and we thus turn our attention to explicitly training models for adversarial robustness, which we show to be a much more compute-efficient defense than scaling model size alone. In this setting, we also observe that adversarially trained larger models generalize faster and better to modified attacks not seen during training when compared with smaller models. Finally, we analyze the offense/defense balance of increasing compute, finding parity in some settings and an advantage for offense in others, suggesting that adversarial training alone is not sufficient to solve robustness, even at greater model scales.

摘要: 语言模型表现出伸缩规律，借此增加模型和数据集大小会导致负对数可能性的可预测下降，从而释放出一系列令人眼花缭乱的功能。这一现象促使许多公司培养出越来越大的车型，以追求越来越好的性能。然而，这些模型容易受到敌意输入的影响，例如“越狱”和促使模型执行不受欢迎的行为的提示注入，随着模型变得更有能力，构成越来越大的风险。先前的工作表明，随着模型和数据的缩放，计算机视觉模型变得更加健壮，这引发了一个问题：语言模型的健壮性是否也随着规模的扩大而提高？我们在分类环境下对这个问题进行了实证研究，发现在没有明确的防御训练的情况下，较大的模型在大多数任务上倾向于稍微更稳健，尽管效果并不可靠。即使具有规模所赋予的优势，非防御模型在绝对意义上仍然很容易受到攻击，因此我们将注意力转向显式训练模型以实现对抗健壮性，我们表明这是一种比单独缩放模型大小更高效的防御方法。在这种情况下，我们还观察到，与较小的模型相比，经过相反训练的较大模型对训练期间未见的修改攻击的泛化更快、更好。最后，我们分析了增加计算、在某些情况下找到平等性以及在其他情况下进攻的优势的进攻/防守平衡，表明仅有对抗性训练不足以解决健壮性问题，即使在更大的模型尺度上也是如此。



## **11. Backdoor in Seconds: Unlocking Vulnerabilities in Large Pre-trained Models via Model Editing**

秒内后门：通过模型编辑解锁大型预训练模型中的漏洞 cs.AI

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18267v1) [paper-pdf](http://arxiv.org/pdf/2410.18267v1)

**Authors**: Dongliang Guo, Mengxuan Hu, Zihan Guan, Junfeng Guo, Thomas Hartvigsen, Sheng Li

**Abstract**: Large pre-trained models have achieved notable success across a range of downstream tasks. However, recent research shows that a type of adversarial attack ($\textit{i.e.,}$ backdoor attack) can manipulate the behavior of machine learning models through contaminating their training dataset, posing significant threat in the real-world application of large pre-trained model, especially for those customized models. Therefore, addressing the unique challenges for exploring vulnerability of pre-trained models is of paramount importance. Through empirical studies on the capability for performing backdoor attack in large pre-trained models ($\textit{e.g.,}$ ViT), we find the following unique challenges of attacking large pre-trained models: 1) the inability to manipulate or even access large training datasets, and 2) the substantial computational resources required for training or fine-tuning these models. To address these challenges, we establish new standards for an effective and feasible backdoor attack in the context of large pre-trained models. In line with these standards, we introduce our EDT model, an \textbf{E}fficient, \textbf{D}ata-free, \textbf{T}raining-free backdoor attack method. Inspired by model editing techniques, EDT injects an editing-based lightweight codebook into the backdoor of large pre-trained models, which replaces the embedding of the poisoned image with the target image without poisoning the training dataset or training the victim model. Our experiments, conducted across various pre-trained models such as ViT, CLIP, BLIP, and stable diffusion, and on downstream tasks including image classification, image captioning, and image generation, demonstrate the effectiveness of our method. Our code is available in the supplementary material.

摘要: 大型预先培训的模型在一系列下游任务中取得了显着的成功。然而，最近的研究表明，一种对抗性攻击(即后门攻击)可以通过污染机器学习模型的训练数据集来操纵它们的行为，这对大型预训练模型的实际应用构成了巨大的威胁，特别是对那些定制的模型。因此，解决探索预先训练模型的脆弱性的独特挑战是至关重要的。通过对大型预训练模型执行后门攻击能力的实证研究，我们发现攻击大型预训练模型面临以下独特的挑战：1)无法操纵甚至访问大型训练数据集；2)训练或微调这些模型所需的大量计算资源。为了应对这些挑战，我们在大型预训练模型的背景下为有效和可行的后门攻击建立了新的标准。根据这些标准，我们介绍了我们的EDT模型，一种高效、无ATA、无雨的后门攻击方法。受模型编辑技术的启发，EDT将基于编辑的轻量级码本注入大型预训练模型的后门，在不毒化训练数据集或训练受害者模型的情况下，将有毒图像的嵌入替换为目标图像。我们在VIT、CLIP、BIP和稳定扩散等各种预先训练的模型上进行的实验，以及在图像分类、图像字幕和图像生成等下游任务上进行的实验，证明了该方法的有效性。我们的代码可以在补充材料中找到。



## **12. Advancing NLP Security by Leveraging LLMs as Adversarial Engines**

通过利用LLC作为对抗引擎来提高NLP安全性 cs.AI

5 pages

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18215v1) [paper-pdf](http://arxiv.org/pdf/2410.18215v1)

**Authors**: Sudarshan Srinivasan, Maria Mahbub, Amir Sadovnik

**Abstract**: This position paper proposes a novel approach to advancing NLP security by leveraging Large Language Models (LLMs) as engines for generating diverse adversarial attacks. Building upon recent work demonstrating LLMs' effectiveness in creating word-level adversarial examples, we argue for expanding this concept to encompass a broader range of attack types, including adversarial patches, universal perturbations, and targeted attacks. We posit that LLMs' sophisticated language understanding and generation capabilities can produce more effective, semantically coherent, and human-like adversarial examples across various domains and classifier architectures. This paradigm shift in adversarial NLP has far-reaching implications, potentially enhancing model robustness, uncovering new vulnerabilities, and driving innovation in defense mechanisms. By exploring this new frontier, we aim to contribute to the development of more secure, reliable, and trustworthy NLP systems for critical applications.

摘要: 这份立场文件提出了一种新颖的方法，通过利用大型语言模型（LLM）作为生成多样化对抗攻击的引擎来提高NLP安全性。在最近展示LLM在创建单词级对抗性示例方面有效性的工作的基础上，我们主张扩展这一概念以涵盖更广泛的攻击类型，包括对抗性补丁、普遍扰动和有针对性的攻击。我们证实，LLM复杂的语言理解和生成能力可以在各个领域和分类器架构中生成更有效、语义一致且类人的对抗性示例。对抗性NLP的这种范式转变具有深远的影响，可能会增强模型稳健性、发现新的漏洞并推动防御机制的创新。通过探索这一新领域，我们的目标是为关键应用开发更安全、可靠和值得信赖的NLP系统做出贡献。



## **13. Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks**

了解多语言LLM对微调攻击的脆弱性 cs.CL

14 pages, 6 figures, 7 tables

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18210v1) [paper-pdf](http://arxiv.org/pdf/2410.18210v1)

**Authors**: Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.

摘要: 最近大型语言模型(LLM)的进步引发了人们对其安全性的广泛担忧。最近的工作表明，通过使用一些恶意选择的指令跟随示例，即微调攻击，可以很容易地删除LLM的安全对齐。我们进一步了解多语言LLM中的微调攻击。我们首先发现了微调攻击的跨语言泛化：使用几个恶意选择的一种语言的指令跟随示例，多语言LLM也很容易被攻破(例如，多语言LLM无法拒绝其他语言的有害提示)。基于这一发现，我们假设安全相关信息是语言不可知的，并提出了一种在模型参数空间中识别安全相关信息的新方法--安全信息本地化。通过SIL语言，我们验证了这一假设，发现在微调攻击中只改变20%的权重参数就可以破坏所有语言的安全对齐。此外，我们为替代路径假说提供了证据，证明了冻结安全相关参数为什么不能防止微调攻击，并证明了我们的攻击向量仍然可以越狱适应新语言的LLM。



## **14. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

保障是一把双刃剑：对大型语言模型的拒绝服务攻击 cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02916v2) [paper-pdf](http://arxiv.org/pdf/2410.02916v2)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.

摘要: 安全是大型语言模型(LLM)在开放部署时最关心的问题。为此，保障措施旨在通过安全调整或护栏机制，强制以合乎道德和负责任的方式使用LLMS。然而，我们发现恶意攻击者可以利用安全措施的误报，即欺骗安全措施模型错误地阻止安全内容，从而导致对LLMS的新的拒绝服务(DoS)攻击。具体地说，通过软件或对用户客户端软件的网络钓鱼攻击，攻击者将一个看似无害的简短对抗性提示插入到配置文件中的用户提示模板中；因此，该提示出现在最终用户请求中，在用户界面中不可见，并且很难识别。通过设计一个利用梯度和注意力信息的优化过程，我们的攻击可以自动生成看似安全的敌意提示，大约只有30个字符，普遍阻止Llama Guard 3上超过97%的用户请求。该攻击提供了一个新的维度来评估LLM安全措施，从根本上不同于传统的越狱。



## **15. Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection**

探索CLIP用于人工智能生成图像检测的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2407.19553v2) [paper-pdf](http://arxiv.org/pdf/2407.19553v2)

**Authors**: Vincenzo De Rosa, Fabrizio Guillaro, Giovanni Poggi, Davide Cozzolino, Luisa Verdoliva

**Abstract**: In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer (ViT) backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.

摘要: 近年来，已经提出了许多法医检测器来检测人工智能生成的图像，并防止将其用于恶意目的。卷积神经网络(CNN)长期以来一直是这一领域的主导结构，也一直是研究的热点。然而，最近提出的基于变形金刚的检测器已经被证明与基于CNN的检测器相媲美，甚至优于基于CNN的检测器，特别是在泛化方面。本文研究了人工智能图像检测器的抗攻击能力，重点研究了基于对比语言图像预训练(CLIP)的方法，并与基于CNN的方法进行了性能比较。我们研究了在不同条件下对不同敌意攻击的鲁棒性，并对数值结果和频域模式进行了分析。基于剪辑的检测器被发现像基于CNN的检测器一样容易受到白盒攻击。然而，攻击并不容易在基于CNN的方法和基于剪辑的方法之间转移。对抗性噪声模式在频域中的不同分布也证实了这一点。总体而言，这一分析为法医探测器的特性提供了新的见解，有助于制定更有效的战略。



## **16. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02240v4) [paper-pdf](http://arxiv.org/pdf/2410.02240v4)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统容易受到敌意攻击。不受限制的对抗性攻击通常操纵图像的语义内容(例如，颜色或纹理)以创建既有效又逼真的对抗性示例。最近的工作利用扩散逆过程将图像映射到潜在空间，在潜在空间中通过引入扰动来操纵高级语义。然而，它们往往会在去噪输出中造成严重的语义扭曲，并导致效率低下。在这项研究中，我们提出了一种新的框架，称为语义一致的无限对抗攻击(SCA)，它使用一种反转方法来提取编辑友好的噪声映射，并利用多模式大语言模型(MLLM)在整个过程中提供语义指导。在MLLM提供丰富语义信息的条件下，使用一系列编辑友好的噪声图对每个步骤进行DDPM去噪处理，并利用DPM Solver++加速这一过程，从而实现高效的语义一致性采样。与现有的方法相比，我们的框架能够高效地生成对抗性的例子，这些例子表现出最小的可识别的语义变化。因此，我们首次引入了语义一致的对抗性例子(SCAE)。广泛的实验和可视化已经证明了SCA的高效率，特别是在平均速度上是最先进的攻击的12倍。我们的研究可以进一步引起人们对多媒体信息安全的关注。



## **17. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

插槽：通过图强化学习进行源驱动APT检测 cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.17910v1) [paper-pdf](http://arxiv.org/pdf/2410.17910v1)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zijian Zhang, Zhengzi Xu, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.

摘要: 高级持续性威胁(APT)是复杂的网络攻击，其特征是能够在受害者系统内长时间保持不被检测到，旨在渗漏敏感数据或中断操作。现有的检测方法往往难以有效地识别这些复杂的威胁，难以构建便于防御的攻击链，或者难以抵抗对抗性攻击。为了克服这些挑战，我们提出了一种基于起源图和图强化学习的高级APT检测方法SLOT。Slot擅长通过起源图挖掘发现系统行为之间的多层次隐藏关系，如因果关系、上下文关系和间接关系。通过开创图强化学习的集成，时隙动态适应新的用户活动和不断演变的攻击策略，增强了对对手攻击的弹性。此外，SLOT根据检测到的攻击利用分簇算法自动构建攻击链，提供准确的攻击路径识别，便于制定防御策略。对真实数据集的评估表明，在APT检测中，Slot具有出色的准确性、效率、适应性和稳健性，大多数指标都超过了最先进的方法。此外，为评估SLOT在支持APT防御方面的有效性而进行的案例研究进一步证明，它是一种实用和可靠的网络安全保护工具。



## **18. Gradient-based Jailbreak Images for Multimodal Fusion Models**

多模式融合模型的基于对象的越狱图像 cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.03489v2) [paper-pdf](http://arxiv.org/pdf/2410.03489v2)

**Authors**: Javier Rando, Hannah Korevaar, Erik Brinkman, Ivan Evtimov, Florian Tramèr

**Abstract**: Augmenting language models with image inputs may enable more effective jailbreak attacks through continuous optimization, unlike text inputs that require discrete optimization. However, new multimodal fusion models tokenize all input modalities using non-differentiable functions, which hinders straightforward attacks. In this work, we introduce the notion of a tokenizer shortcut that approximates tokenization with a continuous function and enables continuous optimization. We use tokenizer shortcuts to create the first end-to-end gradient image attacks against multimodal fusion models. We evaluate our attacks on Chameleon models and obtain jailbreak images that elicit harmful information for 72.5% of prompts. Jailbreak images outperform text jailbreaks optimized with the same objective and require 3x lower compute budget to optimize 50x more input tokens. Finally, we find that representation engineering defenses, like Circuit Breakers, trained only on text attacks can effectively transfer to adversarial image inputs.

摘要: 与需要离散优化的文本输入不同，使用图像输入增强语言模型可能会通过持续优化实现更有效的越狱攻击。然而，新的多模式融合模型使用不可微函数来标记化所有输入模式，这阻碍了直接攻击。在这项工作中，我们引入了标记器捷径的概念，它近似于连续函数的标记化，并使连续优化成为可能。我们使用标记器快捷键创建了第一个针对多模式融合模型的端到端梯度图像攻击。我们评估我们对变色龙模型的攻击，并获得72.5%的提示中引发有害信息的越狱图像。越狱图像的性能优于针对相同目标进行优化的文本越狱，并且需要将计算预算降低3倍才能优化50倍以上的输入令牌。最后，我们发现，表示工程防御，如断路器，只接受文本攻击训练，可以有效地转移到对抗性图像输入。



## **19. STBA: Towards Evaluating the Robustness of DNNs for Query-Limited Black-box Scenario**

STBA：评估DNN在查询受限黑匣子场景中的稳健性 cs.CV

Accepted by T-MM

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2404.00362v2) [paper-pdf](http://arxiv.org/pdf/2404.00362v2)

**Authors**: Renyang Liu, Kwok-Yan Lam, Wei Zhou, Sixing Wu, Jun Zhao, Dongting Hu, Mingming Gong

**Abstract**: Many attack techniques have been proposed to explore the vulnerability of DNNs and further help to improve their robustness. Despite the significant progress made recently, existing black-box attack methods still suffer from unsatisfactory performance due to the vast number of queries needed to optimize desired perturbations. Besides, the other critical challenge is that adversarial examples built in a noise-adding manner are abnormal and struggle to successfully attack robust models, whose robustness is enhanced by adversarial training against small perturbations. There is no doubt that these two issues mentioned above will significantly increase the risk of exposure and result in a failure to dig deeply into the vulnerability of DNNs. Hence, it is necessary to evaluate DNNs' fragility sufficiently under query-limited settings in a non-additional way. In this paper, we propose the Spatial Transform Black-box Attack (STBA), a novel framework to craft formidable adversarial examples in the query-limited scenario. Specifically, STBA introduces a flow field to the high-frequency part of clean images to generate adversarial examples and adopts the following two processes to enhance their naturalness and significantly improve the query efficiency: a) we apply an estimated flow field to the high-frequency part of clean images to generate adversarial examples instead of introducing external noise to the benign image, and b) we leverage an efficient gradient estimation method based on a batch of samples to optimize such an ideal flow field under query-limited settings. Compared to existing score-based black-box baselines, extensive experiments indicated that STBA could effectively improve the imperceptibility of the adversarial examples and remarkably boost the attack success rate under query-limited settings.

摘要: 已经提出了许多攻击技术来探索DNN的脆弱性，并进一步帮助提高它们的健壮性。尽管最近取得了显著的进展，但现有的黑盒攻击方法仍然存在性能不佳的问题，这是因为需要大量的查询来优化期望的扰动。此外，另一个关键的挑战是，以添加噪声的方式构建的对抗性样本是不正常的，并且难以成功地攻击健壮模型，而健壮模型的健壮性通过对抗小扰动的对抗性训练来增强。毫无疑问，上述两个问题将大大增加暴露的风险，并导致无法深入挖掘DNN的脆弱性。因此，有必要以一种非额外的方式充分评估DNN在查询受限设置下的脆弱性。在本文中，我们提出了空间变换黑盒攻击(STBA)，这是一个新的框架，可以在查询受限的情况下创建强大的对手示例。具体地说，STBA在清洁图像的高频部分引入了流场来生成对抗性实例，并采用了以下两个过程来增强其自然性，显著提高了查询效率：a)将估计的流场应用于干净图像的高频部分来生成对抗性实例，而不是在良性图像中引入外部噪声；b)在查询受限的情况下，利用一种基于批量样本的高效梯度估计方法来优化这样的理想流场。大量实验表明，与已有的基于分数的黑盒基线相比，STBA能够有效地提高对抗性实例的隐蔽性，显著提高查询受限环境下的攻击成功率。



## **20. DIP-Watermark: A Double Identity Protection Method Based on Robust Adversarial Watermark**

DIP-水印：一种基于鲁棒对抗水印的双重身份保护方法 cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2404.14693v2) [paper-pdf](http://arxiv.org/pdf/2404.14693v2)

**Authors**: Yunming Zhang, Dengpan Ye, Caiyun Xie, Sipeng Shen, Ziyi Liu, Jiacheng Deng, Long Tang

**Abstract**: The wide deployment of Face Recognition (FR) systems poses privacy risks. One countermeasure is adversarial attack, deceiving unauthorized malicious FR, but it also disrupts regular identity verification of trusted authorizers, exacerbating the potential threat of identity impersonation. To address this, we propose the first double identity protection scheme based on traceable adversarial watermarking, termed DIP-Watermark. DIP-Watermark employs a one-time watermark embedding to deceive unauthorized FR models and allows authorizers to perform identity verification by extracting the watermark. Specifically, we propose an information-guided adversarial attack against FR models. The encoder embeds an identity-specific watermark into the deep feature space of the carrier, guiding recognizable features of the image to deviate from the source identity. We further adopt a collaborative meta-optimization strategy compatible with sub-tasks, which regularizes the joint optimization direction of the encoder and decoder. This strategy enhances the representation of universal carrier features, mitigating multi-objective optimization conflicts in watermarking. Experiments confirm that DIP-Watermark achieves significant attack success rates and traceability accuracy on state-of-the-art FR models, exhibiting remarkable robustness that outperforms the existing privacy protection methods using adversarial attacks and deep watermarking, or simple combinations of the two. Our work potentially opens up new insights into proactive protection for FR privacy.

摘要: 人脸识别(FR)系统的广泛部署带来了隐私风险。一种对策是对抗性攻击，欺骗未经授权的恶意FR，但它也扰乱了可信授权者的常规身份验证，加剧了身份冒充的潜在威胁。为了解决这一问题，我们提出了第一个基于可追踪对抗水印的双重身份保护方案，称为DIP-水印。DIP-水印算法采用一次性水印嵌入的方法来欺骗未经授权的FR模型，并允许授权者通过提取水印来进行身份验证。具体地说，我们提出了一种针对FR模型的信息制导的对抗性攻击。编码器将特定于身份的水印嵌入到载体的深层特征空间中，引导图像的可识别特征偏离源身份。进一步采用了和子任务兼容的协作元优化策略，规范了编解码器的联合优化方向。该策略增强了对通用载体特征的表示，缓解了水印中的多目标优化冲突。实验证实，DIP-水印在最先进的FR模型上获得了显著的攻击成功率和可追踪性准确性，表现出显著的稳健性，其性能优于现有的使用对抗性攻击和深度水印的隐私保护方法，或两者的简单组合。我们的工作可能为主动保护FR隐私打开新的洞察力。



## **21. IBGP: Imperfect Byzantine Generals Problem for Zero-Shot Robustness in Communicative Multi-Agent Systems**

IBGP：通信多智能体系统中零攻击鲁棒性的不完美拜占庭将军问题 cs.MA

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.16237v2) [paper-pdf](http://arxiv.org/pdf/2410.16237v2)

**Authors**: Yihuan Mao, Yipeng Kang, Peilun Li, Ning Zhang, Wei Xu, Chongjie Zhang

**Abstract**: As large language model (LLM) agents increasingly integrate into our infrastructure, their robust coordination and message synchronization become vital. The Byzantine Generals Problem (BGP) is a critical model for constructing resilient multi-agent systems (MAS) under adversarial attacks. It describes a scenario where malicious agents with unknown identities exist in the system-situations that, in our context, could result from LLM agents' hallucinations or external attacks. In BGP, the objective of the entire system is to reach a consensus on the action to be taken. Traditional BGP requires global consensus among all agents; however, in practical scenarios, global consensus is not always necessary and can even be inefficient. Therefore, there is a pressing need to explore a refined version of BGP that aligns with the local coordination patterns observed in MAS. We refer to this refined version as Imperfect BGP (IBGP) in our research, aiming to address this discrepancy. To tackle this issue, we propose a framework that leverages consensus protocols within general MAS settings, providing provable resilience against communication attacks and adaptability to changing environments, as validated by empirical results. Additionally, we present a case study in a sensor network environment to illustrate the practical application of our protocol.

摘要: 随着大型语言模型(LLM)代理越来越多地集成到我们的基础设施中，它们强大的协调和消息同步变得至关重要。拜占庭将军问题(BGP)是在对抗攻击下构造具有弹性的多智能体系统(MAS)的重要模型。它描述了一种系统中存在身份未知的恶意代理的情况--在我们的上下文中，这种情况可能是由于LLM代理的幻觉或外部攻击造成的。在BGP中，整个系统的目标是就要采取的行动达成共识。传统的BGP需要在所有代理之间达成全局共识；然而，在实际场景中，全局共识并不总是必要的，甚至可能效率低下。因此，迫切需要探索一种与MAS中观察到的局部协调模式相一致的BGP改进版本。在我们的研究中，我们将这种精炼版本称为不完美BGP(IBGP)，旨在解决这一差异。为了解决这个问题，我们提出了一个框架，它在一般的MAS环境中利用共识协议，提供对通信攻击的可证明的弹性和对不断变化的环境的适应性，实验结果验证了这一点。此外，我们还给出了一个传感器网络环境下的案例研究，以说明该协议的实际应用。



## **22. The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations**

终极组合：通过编写数据增强增强对抗性示例的可移植性 cs.CV

Accepted by AISec'24

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2312.11309v2) [paper-pdf](http://arxiv.org/pdf/2312.11309v2)

**Authors**: Zebin Yun, Achi-Or Weingarten, Eyal Ronen, Mahmood Sharif

**Abstract**: To help adversarial examples generalize from surrogate machine-learning (ML) models to targets, certain transferability-based black-box evasion attacks incorporate data augmentations (e.g., random resizing). Yet, prior work has explored limited augmentations and their composition. To fill the gap, we systematically studied how data augmentation affects transferability. Specifically, we explored 46 augmentation techniques originally proposed to help ML models generalize to unseen benign samples, and assessed how they impact transferability, when applied individually or composed. Performing exhaustive search on a small subset of augmentation techniques and genetic search on all techniques, we identified augmentation combinations that help promote transferability. Extensive experiments with the ImageNet and CIFAR-10 datasets and 18 models showed that simple color-space augmentations (e.g., color to greyscale) attain high transferability when combined with standard augmentations. Furthermore, we discovered that composing augmentations impacts transferability mostly monotonically (i.e., more augmentations $\rightarrow$ $\ge$transferability). We also found that the best composition significantly outperformed the state of the art (e.g., 91.8% vs. $\le$82.5% average transferability to adversarially trained targets on ImageNet). Lastly, our theoretical analysis, backed by empirical evidence, intuitively explains why certain augmentations promote transferability.

摘要: 为了帮助敌意例子从代理机器学习(ML)模型推广到目标，某些基于可转移性的黑盒逃避攻击结合了数据增强(例如，随机调整大小)。然而，先前的工作探索了有限的增强及其组成。为了填补这一空白，我们系统地研究了数据扩充如何影响可转移性。具体地说，我们探索了最初提出的46种增强技术，这些技术最初是为了帮助ML模型推广到看不见的良性样本，并评估了它们在单独应用或组合时如何影响可转移性。我们对一小部分增强技术进行了穷举搜索，并对所有技术进行了遗传搜索，确定了有助于提高可转移性的增强组合。在ImageNet和CIFAR-10数据集和18个模型上的广泛实验表明，简单的颜色空间增强(例如，从颜色到灰度)在与标准增强相结合时获得了高可转移性。此外，我们还发现，组合增词对可转移性的影响主要是单调的(即，更多的增词对可转移性的影响)。我们还发现，最好的组合大大超过了最先进的水平(例如，91.8%对ImageNet上接受过相反训练的目标的平均可转移率为82.5%)。最后，我们的理论分析得到了经验证据的支持，直观地解释了为什么某些扩大促进了可转移性。



## **23. Diffusion Models are Certifiably Robust Classifiers**

扩散模型是可证明稳健的分类器 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2402.02316v3) [paper-pdf](http://arxiv.org/pdf/2402.02316v3)

**Authors**: Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: Generative learning, recognized for its effective modeling of data distributions, offers inherent advantages in handling out-of-distribution instances, especially for enhancing robustness to adversarial attacks. Among these, diffusion classifiers, utilizing powerful diffusion models, have demonstrated superior empirical robustness. However, a comprehensive theoretical understanding of their robustness is still lacking, raising concerns about their vulnerability to stronger future attacks. In this study, we prove that diffusion classifiers possess $O(1)$ Lipschitzness, and establish their certified robustness, demonstrating their inherent resilience. To achieve non-constant Lipschitzness, thereby obtaining much tighter certified robustness, we generalize diffusion classifiers to classify Gaussian-corrupted data. This involves deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. Experimental results show the superior certified robustness of these Noised Diffusion Classifiers (NDCs). Notably, we achieve over 80% and 70% certified robustness on CIFAR-10 under adversarial perturbations with \(\ell_2\) norms less than 0.25 and 0.5, respectively, using a single off-the-shelf diffusion model without any additional data.

摘要: 生成性学习以其对数据分布的有效建模而被公认，在处理分布外实例方面提供了固有的优势，特别是在增强对对手攻击的稳健性方面。其中，扩散分类器利用了强大的扩散模型，表现出了优越的经验稳健性。然而，对它们的健壮性仍然缺乏全面的理论理解，这引发了人们对它们在未来更强大的攻击中的脆弱性的担忧。在这项研究中，我们证明了扩散分类器具有$O(1)$Lipschitz性，并建立了它们被证明的稳健性，证明了它们的内在弹性。为了实现非常数的Lipschitz性，从而获得更紧密的认证稳健性，我们推广了扩散分类器来分类受高斯污染的数据。这涉及到推导这些分布的证据下界(ELBO)，使用ELBO近似似然性，以及通过贝叶斯定理计算分类概率。实验结果表明，这些带噪扩散分类器(NDC)具有良好的鲁棒性。值得注意的是，在没有任何额外数据的情况下，我们使用单个现成的扩散模型，在对抗性扰动下分别获得了超过80%和70%的CIFAR-10的认证鲁棒性。



## **24. A provable initialization and robust clustering method for general mixture models**

一般混合模型的可证明初始化和鲁棒性集群方法 math.ST

51 pages, corrected typos, updated structures and results are  improved

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2401.05574v3) [paper-pdf](http://arxiv.org/pdf/2401.05574v3)

**Authors**: Soham Jana, Jianqing Fan, Sanjeev Kulkarni

**Abstract**: Clustering is a fundamental tool in statistical machine learning in the presence of heterogeneous data. Most recent results focus primarily on optimal mislabeling guarantees when data are distributed around centroids with sub-Gaussian errors. Yet, the restrictive sub-Gaussian model is often invalid in practice since various real-world applications exhibit heavy tail distributions around the centroids or suffer from possible adversarial attacks that call for robust clustering with a robust data-driven initialization. In this paper, we present initialization and subsequent clustering methods that provably guarantee near-optimal mislabeling for general mixture models when the number of clusters and data dimensions are finite. We first introduce a hybrid clustering technique with a novel multivariate trimmed mean type centroid estimate to produce mislabeling guarantees under a weak initialization condition for general error distributions around the centroids. A matching lower bound is derived, up to factors depending on the number of clusters. In addition, our approach also produces similar mislabeling guarantees even in the presence of adversarial outliers. Our results reduce to the sub-Gaussian case in finite dimensions when errors follow sub-Gaussian distributions. To solve the problem thoroughly, we also present novel data-driven robust initialization techniques and show that, with probabilities approaching one, these initial centroid estimates are sufficiently good for the subsequent clustering algorithm to achieve the optimal mislabeling rates. Furthermore, we demonstrate that the Lloyd algorithm is suboptimal for more than two clusters even when errors are Gaussian and for two clusters when error distributions have heavy tails. Both simulated data and real data examples further support our robust initialization procedure and clustering algorithm.

摘要: 在存在异质数据的情况下，聚类是统计机器学习的基本工具。最新的结果主要集中在当数据分布在具有亚高斯误差的质心上时的最优错误标记保证。然而，受限的亚高斯模型在实践中往往是无效的，因为各种现实世界的应用程序在质心周围显示出沉重的尾部分布，或者遭受可能的敌意攻击，这些攻击需要使用健壮的数据驱动的初始化来进行健壮的聚类。在这篇文章中，我们提出了初始化和后续的聚类方法，当聚类数目和数据维度有限时，这些方法可证明保证了一般混合模型的近似最优错误标记。我们首先引入了一种新的多变量修剪平均型质心估计的混合聚类技术，以在弱初始化条件下对质心周围的一般误差分布产生误标记保证。得到了一个匹配的下限，最高可达取决于簇数的系数。此外，我们的方法还产生了类似的错误标签保证，即使在存在敌对异常值的情况下也是如此。当误差服从亚高斯分布时，我们的结果简化为有限维的亚高斯情形。为了彻底解决这个问题，我们还提出了新的数据驱动的稳健初始化技术，并证明了在概率接近于1的情况下，这些初始质心估计足以使后续的聚类算法获得最优的误标率。此外，我们证明了Lloyd算法对于两个以上的簇，即使在误差为高斯分布时也是次优的，对于两个簇，当误差分布具有重尾时，Lloyd算法也是次优的。模拟数据和真实数据的例子进一步支持了我们健壮的初始化过程和聚类算法。



## **25. Detecting Adversarial Examples**

检测对抗示例 cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17442v1) [paper-pdf](http://arxiv.org/pdf/2410.17442v1)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Deep Neural Networks (DNNs) have been shown to be vulnerable to adversarial examples. While numerous successful adversarial attacks have been proposed, defenses against these attacks remain relatively understudied. Existing defense approaches either focus on negating the effects of perturbations caused by the attacks to restore the DNNs' original predictions or use a secondary model to detect adversarial examples. However, these methods often become ineffective due to the continuous advancements in attack techniques. We propose a novel universal and lightweight method to detect adversarial examples by analyzing the layer outputs of DNNs. Through theoretical justification and extensive experiments, we demonstrate that our detection method is highly effective, compatible with any DNN architecture, and applicable across different domains, such as image, video, and audio.

摘要: 深度神经网络（DNN）已被证明容易受到对抗性示例的影响。虽然已经提出了许多成功的对抗性攻击，但针对这些攻击的防御研究仍然相对不足。现有的防御方法要么专注于抵消攻击引起的扰动的影响，以恢复DNN的原始预测，要么使用二级模型来检测对抗性示例。然而，由于攻击技术的不断进步，这些方法往往变得无效。我们提出了一种新颖的通用和轻量级方法来通过分析DNN的层输出来检测对抗性示例。通过理论论证和大量实验，我们证明我们的检测方法非常有效，与任何DNN架构兼容，并且适用于图像、视频和音频等不同领域。



## **26. Meta Stackelberg Game: Robust Federated Learning against Adaptive and Mixed Poisoning Attacks**

Meta Stackelberg博弈：针对自适应和混合中毒攻击的鲁棒联邦学习 cs.LG

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17431v1) [paper-pdf](http://arxiv.org/pdf/2410.17431v1)

**Authors**: Tao Li, Henger Li, Yunian Pan, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Federated learning (FL) is susceptible to a range of security threats. Although various defense mechanisms have been proposed, they are typically non-adaptive and tailored to specific types of attacks, leaving them insufficient in the face of multiple uncertain, unknown, and adaptive attacks employing diverse strategies. This work formulates adversarial federated learning under a mixture of various attacks as a Bayesian Stackelberg Markov game, based on which we propose the meta-Stackelberg defense composed of pre-training and online adaptation. {The gist is to simulate strong attack behavior using reinforcement learning (RL-based attacks) in pre-training and then design meta-RL-based defense to combat diverse and adaptive attacks.} We develop an efficient meta-learning approach to solve the game, leading to a robust and adaptive FL defense. Theoretically, our meta-learning algorithm, meta-Stackelberg learning, provably converges to the first-order $\varepsilon$-meta-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations with $O(\varepsilon^{-4})$ samples per iteration. Experiments show that our meta-Stackelberg framework performs superbly against strong model poisoning and backdoor attacks of uncertain and unknown types.

摘要: 联合学习(FL)容易受到一系列安全威胁的影响。虽然已经提出了各种防御机制，但它们通常是非适应性的，并且针对特定类型的攻击量身定做，使得它们在面对采用不同策略的多种不确定、未知和适应性攻击时显得力不从心。该工作将多种攻击混合情况下的对抗性联邦学习描述为贝叶斯Stackelberg马尔可夫博弈，在此基础上提出了由预训练和在线适应组成的元Stackelberg防御。{重点是在预训练中使用强化学习(RL-Based Attack)来模拟强攻击行为，然后设计基于元RL的防御来对抗多样化和自适应的攻击。}我们开发了一种高效的元学习方法来解决游戏问题，从而实现了健壮和自适应的FL防御。理论上，我们的元学习算法Meta-Stackelberg学习在$O(varepsilon^{-2})$梯度迭代中以$O(varepsilon^{-4})$个样本证明收敛到一阶$varepsilon$亚平衡点。实验表明，我们的Meta-Stackelberg框架对强模型中毒和不确定和未知类型的后门攻击具有很好的性能。



## **27. AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents**

AdvWeb：对TLR驱动的Web代理的可控黑匣子攻击 cs.CR

15 pages

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17401v1) [paper-pdf](http://arxiv.org/pdf/2410.17401v1)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Vision Language Models (VLMs) have revolutionized the creation of generalist web agents, empowering them to autonomously complete diverse tasks on real-world websites, thereby boosting human efficiency and productivity. However, despite their remarkable capabilities, the safety and security of these agents against malicious attacks remain critically underexplored, raising significant concerns about their safe deployment. To uncover and exploit such vulnerabilities in web agents, we provide AdvWeb, a novel black-box attack framework designed against web agents. AdvWeb trains an adversarial prompter model that generates and injects adversarial prompts into web pages, misleading web agents into executing targeted adversarial actions such as inappropriate stock purchases or incorrect bank transactions, actions that could lead to severe real-world consequences. With only black-box access to the web agent, we train and optimize the adversarial prompter model using DPO, leveraging both successful and failed attack strings against the target agent. Unlike prior approaches, our adversarial string injection maintains stealth and control: (1) the appearance of the website remains unchanged before and after the attack, making it nearly impossible for users to detect tampering, and (2) attackers can modify specific substrings within the generated adversarial string to seamlessly change the attack objective (e.g., purchasing stocks from a different company), enhancing attack flexibility and efficiency. We conduct extensive evaluations, demonstrating that AdvWeb achieves high success rates in attacking SOTA GPT-4V-based VLM agent across various web tasks. Our findings expose critical vulnerabilities in current LLM/VLM-based agents, emphasizing the urgent need for developing more reliable web agents and effective defenses. Our code and data are available at https://ai-secure.github.io/AdvWeb/ .

摘要: 视觉语言模型(VLM)彻底改变了多面手Web代理的创建，使其能够在现实世界的网站上自主完成各种任务，从而提高了人类的效率和生产力。然而，尽管这些代理具有非凡的能力，但其抵御恶意攻击的安全性和安全性仍然严重不足，这引发了人们对其安全部署的严重担忧。为了发现和利用Web代理中的此类漏洞，我们提供了AdvWeb，这是一个针对Web代理设计的新型黑盒攻击框架。AdvWeb训练一种对抗性提示器模型，该模型生成对抗性提示并将其注入网页，误导网络代理执行有针对性的对抗性行动，如不适当的股票购买或不正确的银行交易，这些行动可能会导致严重的现实世界后果。在只有黑盒访问Web代理的情况下，我们使用DPO训练和优化对抗性提示器模型，利用针对目标代理的成功和失败的攻击字符串。与以前的方法不同，我们的敌意字符串注入保持了隐蔽性和可控性：(1)攻击前后网站的外观保持不变，使得用户几乎不可能检测到篡改；(2)攻击者可以修改生成的敌意字符串中的特定子字符串，以无缝更改攻击目标(例如，从不同公司购买股票)，从而增强攻击的灵活性和效率。我们进行了广泛的评估，表明AdvWeb在各种Web任务中攻击基于Sota GPT-4V的VLM代理取得了很高的成功率。我们的发现暴露了当前基于LLM/VLM的代理的严重漏洞，强调了开发更可靠的网络代理和有效防御的迫切需要。我们的代码和数据可在https://ai-secure.github.io/AdvWeb/上获得。



## **28. Learning to Poison Large Language Models During Instruction Tuning**

学习在指令调优期间毒害大型语言模型 cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2402.13459v2) [paper-pdf](http://arxiv.org/pdf/2402.13459v2)

**Authors**: Yao Qiang, Xiangyu Zhou, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during the instruction tuning of LLMs and emphasizes the necessity of safeguarding LLMs against data poisoning attacks.

摘要: 大型语言模型的出现在语言处理和推理能力方面取得了显著的成就。尽管取得了进步，但LLM仍面临数据中毒攻击的漏洞，即对手在训练数据中插入后门触发器，以恶意目的操纵输出。这项工作通过设计一种新的数据中毒攻击来进一步识别LLMS中的额外安全风险，该攻击专为利用指令调优过程而定制。我们提出了一种新的梯度引导后门触发学习(GBTL)算法来高效地识别敌意触发，在保证内容完整性的同时确保了传统防御的检测。通过对各种任务的实验验证，包括情感分析、领域生成和问题回答，我们的中毒策略在牺牲各种LLMS的输出方面表现出了很高的成功率。针对数据中毒攻击，我们进一步提出了两种防御策略，包括上下文中学习(ICL)和连续学习(CL)，它们有效地纠正了LLM的行为，显著降低了性能下降。我们的工作突出了在LLMS的指令调优过程中存在的重大安全风险，并强调了保护LLMS免受数据中毒攻击的必要性。



## **29. Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods**

上下文感知即时调优：用对抗方法推进上下文学习 cs.CL

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17222v1) [paper-pdf](http://arxiv.org/pdf/2410.17222v1)

**Authors**: Tsachi Blau, Moshe Kimhi, Yonatan Belinkov, Alexander Bronstein, Chaim Baskin

**Abstract**: Fine-tuning Large Language Models (LLMs) typically involves updating at least a few billions of parameters. A more parameter-efficient approach is Prompt Tuning (PT), which updates only a few learnable tokens, and differently, In-Context Learning (ICL) adapts the model to a new task by simply including examples in the input without any training. When applying optimization-based methods, such as fine-tuning and PT for few-shot learning, the model is specifically adapted to the small set of training examples, whereas ICL leaves the model unchanged. This distinction makes traditional learning methods more prone to overfitting; in contrast, ICL is less sensitive to the few-shot scenario. While ICL is not prone to overfitting, it does not fully extract the information that exists in the training examples. This work introduces Context-aware Prompt Tuning (CPT), a method inspired by ICL, PT, and adversarial attacks. We build on the ICL strategy of concatenating examples before the input, but we extend this by PT-like learning, refining the context embedding through iterative optimization to extract deeper insights from the training examples. We carefully modify specific context tokens, considering the unique structure of input and output formats. Inspired by adversarial attacks, we adjust the input based on the labels present in the context, focusing on minimizing, rather than maximizing, the loss. Moreover, we apply a projected gradient descent algorithm to keep token embeddings close to their original values, under the assumption that the user-provided data is inherently valuable. Our method has been shown to achieve superior accuracy across multiple classification tasks using various LLM models.

摘要: 微调大型语言模型(LLM)通常需要更新至少数十亿个参数。一种参数效率更高的方法是即时调整(PT)，它只更新几个可学习的令牌，而不同的是，上下文中学习(ICL)通过在输入中简单地包括示例来使模型适应新任务，而不需要任何训练。当应用基于优化的方法时，例如微调和PT用于少镜头学习，该模型特别适合于小的训练样本集，而ICL保持模型不变。这种区别使得传统的学习方法更容易过度适应；相比之下，ICL对少数几次机会的情景不那么敏感。虽然ICL不容易过度拟合，但它没有完全提取训练示例中存在的信息。这项工作引入了上下文感知提示调优(CPT)，这是一种受ICL、PT和对手攻击启发的方法。我们建立在输入之前连接示例的ICL策略之上，但我们通过类似PT的学习来扩展这一策略，通过迭代优化来优化上下文嵌入，以从训练示例中提取更深层次的见解。考虑到输入和输出格式的独特结构，我们仔细修改了特定的上下文令牌。受到对抗性攻击的启发，我们根据上下文中存在的标签调整输入，重点是最小化而不是最大化损失。此外，在假设用户提供的数据具有内在价值的前提下，我们应用投影梯度下降算法来保持令牌嵌入接近其原始值。我们的方法已经被证明在使用各种LLM模型的多个分类任务中获得了更高的准确率。



## **30. Remote Timing Attacks on Efficient Language Model Inference**

对高效语言模型推理的远程计时攻击 cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17175v1) [paper-pdf](http://arxiv.org/pdf/2410.17175v1)

**Authors**: Nicholas Carlini, Milad Nasr

**Abstract**: Scaling up language models has significantly increased their capabilities. But larger models are slower models, and so there is now an extensive body of work (e.g., speculative sampling or parallel decoding) that improves the (average case) efficiency of language model generation. But these techniques introduce data-dependent timing characteristics. We show it is possible to exploit these timing differences to mount a timing attack. By monitoring the (encrypted) network traffic between a victim user and a remote language model, we can learn information about the content of messages by noting when responses are faster or slower. With complete black-box access, on open source systems we show how it is possible to learn the topic of a user's conversation (e.g., medical advice vs. coding assistance) with 90%+ precision, and on production systems like OpenAI's ChatGPT and Anthropic's Claude we can distinguish between specific messages or infer the user's language. We further show that an active adversary can leverage a boosting attack to recover PII placed in messages (e.g., phone numbers or credit card numbers) for open source systems. We conclude with potential defenses and directions for future work.

摘要: 扩大语言模型的规模显著提高了它们的能力。但较大的模型是较慢的模型，因此现在有大量的工作(例如，推测采样或并行解码)来提高语言模型生成的(平均情况)效率。但这些技术引入了依赖于数据的时序特性。我们证明了利用这些时序差异来发动时序攻击是可能的。通过监视受害者用户和远程语言模型之间的(加密的)网络流量，我们可以通过记录响应的速度或速度来了解有关消息内容的信息。通过完全的黑盒访问，在开源系统上，我们展示了如何能够以90%以上的精度学习用户对话的主题(例如，医疗建议与编码帮助)，而在OpenAI的ChatGPT和Anthropic的Claude等生产系统上，我们可以区分特定的消息或推断用户的语言。我们进一步表明，活跃的敌手可以利用助推攻击来恢复放置在开源系统的消息(例如电话号码或信用卡号码)中的PII。最后，我们提出了可能的防御措施和未来工作的方向。



## **31. FDINet: Protecting against DNN Model Extraction via Feature Distortion Index**

FDINet：通过特征失真指数防止DNN模型提取 cs.CR

Accepted to IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2306.11338v3) [paper-pdf](http://arxiv.org/pdf/2306.11338v3)

**Authors**: Hongwei Yao, Zheng Li, Haiqin Weng, Feng Xue, Zhan Qin, Kui Ren

**Abstract**: Machine Learning as a Service (MLaaS) platforms have gained popularity due to their accessibility, cost-efficiency, scalability, and rapid development capabilities. However, recent research has highlighted the vulnerability of cloud-based models in MLaaS to model extraction attacks. In this paper, we introduce FDINET, a novel defense mechanism that leverages the feature distribution of deep neural network (DNN) models. Concretely, by analyzing the feature distribution from the adversary's queries, we reveal that the feature distribution of these queries deviates from that of the model's training set. Based on this key observation, we propose Feature Distortion Index (FDI), a metric designed to quantitatively measure the feature distribution deviation of received queries. The proposed FDINET utilizes FDI to train a binary detector and exploits FDI similarity to identify colluding adversaries from distributed extraction attacks. We conduct extensive experiments to evaluate FDINET against six state-of-the-art extraction attacks on four benchmark datasets and four popular model architectures. Empirical results demonstrate the following findings FDINET proves to be highly effective in detecting model extraction, achieving a 100% detection accuracy on DFME and DaST. FDINET is highly efficient, using just 50 queries to raise an extraction alarm with an average confidence of 96.08% for GTSRB. FDINET exhibits the capability to identify colluding adversaries with an accuracy exceeding 91%. Additionally, it demonstrates the ability to detect two types of adaptive attacks.

摘要: 机器学习即服务(MLaaS)平台因其可访问性、成本效益、可扩展性和快速开发能力而广受欢迎。然而，最近的研究突显了MLaaS中基于云的模型对提取攻击的脆弱性。在本文中，我们介绍了FDINET，一种利用深度神经网络(DNN)模型特征分布的新型防御机制。具体地说，通过分析对手查询的特征分布，我们发现这些查询的特征分布偏离了模型训练集的特征分布。基于这一关键观察，我们提出了特征失真指数(FDI)，这是一种用来定量衡量所接收查询的特征分布偏差的度量。FDINET利用FDI来训练二进制检测器，并利用FDI的相似性从分布式抽取攻击中识别合谋对手。我们在四个基准数据集和四个流行的模型体系结构上进行了广泛的实验，以评估FDINET对六种最先进的提取攻击的攻击。实验结果表明，FDINET在检测模型提取方面具有很高的效率，在DFME和DAST上的检测准确率达到100%。FDINET的效率很高，仅使用50个查询就可以发出提取警报，GTSRB的平均置信度为96.08%。FDINET显示出识别串通对手的能力，准确率超过91%。此外，它还演示了检测两种类型的自适应攻击的能力。



## **32. Adversarial Challenges in Network Intrusion Detection Systems: Research Insights and Future Prospects**

网络入侵检测系统中的对抗挑战：研究见解和未来前景 cs.CR

35 pages

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2409.18736v3) [paper-pdf](http://arxiv.org/pdf/2409.18736v3)

**Authors**: Sabrine Ennaji, Fabio De Gaspari, Dorjan Hitaj, Alicia Kbidi, Luigi V. Mancini

**Abstract**: Machine learning has brought significant advances in cybersecurity, particularly in the development of Intrusion Detection Systems (IDS). These improvements are mainly attributed to the ability of machine learning algorithms to identify complex relationships between features and effectively generalize to unseen data. Deep neural networks, in particular, contributed to this progress by enabling the analysis of large amounts of training data, significantly enhancing detection performance. However, machine learning models remain vulnerable to adversarial attacks, where carefully crafted input data can mislead the model into making incorrect predictions. While adversarial threats in unstructured data, such as images and text, have been extensively studied, their impact on structured data like network traffic is less explored. This survey aims to address this gap by providing a comprehensive review of machine learning-based Network Intrusion Detection Systems (NIDS) and thoroughly analyzing their susceptibility to adversarial attacks. We critically examine existing research in NIDS, highlighting key trends, strengths, and limitations, while identifying areas that require further exploration. Additionally, we discuss emerging challenges in the field and offer insights for the development of more robust and resilient NIDS. In summary, this paper enhances the understanding of adversarial attacks and defenses in NIDS and guide future research in improving the robustness of machine learning models in cybersecurity applications.

摘要: 机器学习在网络安全方面带来了重大进展，特别是在入侵检测系统(入侵检测系统)的开发方面。这些改进主要归功于机器学习算法能够识别特征之间的复杂关系，并有效地概括到看不见的数据。尤其是深度神经网络，通过能够分析大量训练数据，大大提高了检测性能，从而促进了这一进展。然而，机器学习模型仍然容易受到敌意攻击，精心设计的输入数据可能会误导模型做出错误的预测。尽管图像和文本等非结构化数据中的敌意威胁已被广泛研究，但它们对网络流量等结构化数据的影响却鲜有人探讨。这项调查旨在通过对基于机器学习的网络入侵检测系统(NID)的全面审查来解决这一差距，并彻底分析它们对对手攻击的敏感性。我们批判性地检查NID中的现有研究，强调主要趋势、优势和局限性，同时确定需要进一步探索的领域。此外，我们还讨论了该领域新出现的挑战，并为开发更强大和更具弹性的网络入侵检测系统提供了见解。综上所述，本文加深了对网络入侵检测系统中对抗性攻击和防御的理解，并指导了未来在提高机器学习模型在网络安全应用中的稳健性方面的研究。



## **33. A Self-Organizing Clustering System for Unsupervised Distribution Shift Detection**

用于无监督分布漂移检测的自组织集群系统 cs.LG

Revised version of the accepted manuscript to IJCNN'2024. Main  corrections were in Section 2.2 and Section 3.3. In Section 2.2 was corrected  expression (3), and in Section 3.3 in the definition of the elements of the  matrix $D$ it was a typo where $\phi(x)$ was written instead of $x$

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2404.16656v2) [paper-pdf](http://arxiv.org/pdf/2404.16656v2)

**Authors**: Sebastián Basterrech, Line Clemmensen, Gerardo Rubino

**Abstract**: Modeling non-stationary data is a challenging problem in the field of continual learning, and data distribution shifts may result in negative consequences on the performance of a machine learning model. Classic learning tools are often vulnerable to perturbations of the input covariates, and are sensitive to outliers and noise, and some tools are based on rigid algebraic assumptions. Distribution shifts are frequently occurring due to changes in raw materials for production, seasonality, a different user base, or even adversarial attacks. Therefore, there is a need for more effective distribution shift detection techniques. In this work, we propose a continual learning framework for monitoring and detecting distribution changes. We explore the problem in a latent space generated by a bio-inspired self-organizing clustering and statistical aspects of the latent space. In particular, we investigate the projections made by two topology-preserving maps: the Self-Organizing Map and the Scale Invariant Map. Our method can be applied in both a supervised and an unsupervised context. We construct the assessment of changes in the data distribution as a comparison of Gaussian signals, making the proposed method fast and robust. We compare it to other unsupervised techniques, specifically Principal Component Analysis (PCA) and Kernel-PCA. Our comparison involves conducting experiments using sequences of images (based on MNIST and injected shifts with adversarial samples), chemical sensor measurements, and the environmental variable related to ozone levels. The empirical study reveals the potential of the proposed approach.

摘要: 非平稳数据建模是持续学习领域中的一个具有挑战性的问题，而数据分布的变化可能会对机器学习模型的性能造成负面影响。传统的学习工具往往容易受到输入协变量的扰动，对异常值和噪声敏感，而且一些工具是基于严格的代数假设的。由于生产原材料的变化、季节性、不同的用户群，甚至是对抗性的攻击，分销转变经常发生。因此，需要更有效的分布移位检测技术。在这项工作中，我们提出了一个持续学习框架，用于监测和检测分布变化。我们在由生物启发的自组织聚类和统计方面产生的潜在空间中探索这一问题。特别地，我们研究了两种保持拓扑的映射：自组织映射和比例不变映射。我们的方法可以应用于有监督和无监督的环境中。我们构造了对数据分布变化的评估作为对高斯信号的比较，使得所提出的方法快速且稳健。我们将其与其他非监督技术，特别是主成分分析(PCA)和核主成分分析(Kernel-PCA)进行了比较。我们的比较包括使用图像序列(基于MNIST和带有对抗性样本的注入位移)、化学传感器测量以及与臭氧水平相关的环境变量进行实验。实证研究揭示了该方法的潜力。



## **34. Test-time Adversarial Defense with Opposite Adversarial Path and High Attack Time Cost**

具有相反对抗路径和高攻击时间成本的测试时对抗防御 cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16805v1) [paper-pdf](http://arxiv.org/pdf/2410.16805v1)

**Authors**: Cheng-Han Yeh, Kuanchun Yu, Chun-Shien Lu

**Abstract**: Deep learning models are known to be vulnerable to adversarial attacks by injecting sophisticated designed perturbations to input data. Training-time defenses still exhibit a significant performance gap between natural accuracy and robust accuracy. In this paper, we investigate a new test-time adversarial defense method via diffusion-based recovery along opposite adversarial paths (OAPs). We present a purifier that can be plugged into a pre-trained model to resist adversarial attacks. Different from prior arts, the key idea is excessive denoising or purification by integrating the opposite adversarial direction with reverse diffusion to push the input image further toward the opposite adversarial direction. For the first time, we also exemplify the pitfall of conducting AutoAttack (Rand) for diffusion-based defense methods. Through the lens of time complexity, we examine the trade-off between the effectiveness of adaptive attack and its computation complexity against our defense. Experimental evaluation along with time cost analysis verifies the effectiveness of the proposed method.

摘要: 众所周知，深度学习模型通过向输入数据注入复杂的设计扰动而容易受到对手攻击。训练时间防守在自然准确度和稳健准确度之间仍然表现出显著的性能差距。本文研究了一种新的测试时间对抗防御方法--基于扩散的对抗性路径恢复(OAP)方法。我们提出了一种净化器，它可以插入到预先训练的模型中，以抵抗对手的攻击。与现有技术不同的是，其关键思想是通过将相反的对抗性方向与反向扩散相结合来过度去噪或净化，以将输入图像进一步推向相反的对抗性方向。我们还第一次举例说明了对基于扩散的防御方法进行自动攻击(Rand)的陷阱。通过时间复杂性的视角，我们考察了自适应攻击的有效性和计算复杂性与我们的防御之间的权衡。实验评估和时间代价分析验证了该方法的有效性。



## **35. Evaluating the Effectiveness of Attack-Agnostic Features for Morphing Attack Detection**

评估攻击不可知特征对变形攻击检测的有效性 cs.CV

Published in the 2024 IEEE International Joint Conference on  Biometrics (IJCB)

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16802v1) [paper-pdf](http://arxiv.org/pdf/2410.16802v1)

**Authors**: Laurent Colbois, Sébastien Marcel

**Abstract**: Morphing attacks have diversified significantly over the past years, with new methods based on generative adversarial networks (GANs) and diffusion models posing substantial threats to face recognition systems. Recent research has demonstrated the effectiveness of features extracted from large vision models pretrained on bonafide data only (attack-agnostic features) for detecting deep generative images. Building on this, we investigate the potential of these image representations for morphing attack detection (MAD). We develop supervised detectors by training a simple binary linear SVM on the extracted features and one-class detectors by modeling the distribution of bonafide features with a Gaussian Mixture Model (GMM). Our method is evaluated across a comprehensive set of attacks and various scenarios, including generalization to unseen attacks, different source datasets, and print-scan data. Our results indicate that attack-agnostic features can effectively detect morphing attacks, outperforming traditional supervised and one-class detectors from the literature in most scenarios. Additionally, we provide insights into the strengths and limitations of each considered representation and discuss potential future research directions to further enhance the robustness and generalizability of our approach.

摘要: 在过去的几年里，变形攻击已经显著多样化，基于生成性对抗网络(GANS)和扩散模型的新方法对人脸识别系统构成了巨大的威胁。最近的研究已经证明了从大的视觉模型中提取的特征(攻击无关特征)对于检测深度生成图像是有效的。在此基础上，我们研究了这些图像表示在变形攻击检测(MAD)中的潜力。我们通过对提取的特征训练一个简单的二元线性支持向量机来开发监督检测器，并通过使用高斯混合模型(GMM)对真实特征的分布进行建模来构建一类检测器。我们的方法在一组全面的攻击和各种场景中进行了评估，包括对看不见的攻击、不同源数据集和打印扫描数据的泛化。我们的结果表明，攻击不可知特征可以有效地检测变形攻击，在大多数情况下性能优于传统的监督和一类检测器。此外，我们提供了对每个考虑的表示的优点和局限性的见解，并讨论了潜在的未来研究方向，以进一步增强我们的方法的健壮性和普适性。



## **36. Imprompter: Tricking LLM Agents into Improper Tool Use**

入侵者：诱骗LLM代理人使用不当工具 cs.CR

website: https://imprompter.ai code:  https://github.com/Reapor-Yurnero/imprompter v2 changelog: add new results to  Table 3, correct several typos

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.14923v2) [paper-pdf](http://arxiv.org/pdf/2410.14923v2)

**Authors**: Xiaohan Fu, Shuheng Li, Zihan Wang, Yihao Liu, Rajesh K. Gupta, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Model (LLM) Agents are an emerging computing paradigm that blends generative machine learning with tools such as code interpreters, web browsing, email, and more generally, external resources. These agent-based systems represent an emerging shift in personal computing. We contribute to the security foundations of agent-based systems and surface a new class of automatically computed obfuscated adversarial prompt attacks that violate the confidentiality and integrity of user resources connected to an LLM agent. We show how prompt optimization techniques can find such prompts automatically given the weights of a model. We demonstrate that such attacks transfer to production-level agents. For example, we show an information exfiltration attack on Mistral's LeChat agent that analyzes a user's conversation, picks out personally identifiable information, and formats it into a valid markdown command that results in leaking that data to the attacker's server. This attack shows a nearly 80% success rate in an end-to-end evaluation. We conduct a range of experiments to characterize the efficacy of these attacks and find that they reliably work on emerging agent-based systems like Mistral's LeChat, ChatGLM, and Meta's Llama. These attacks are multimodal, and we show variants in the text-only and image domains.

摘要: 大型语言模型(LLM)代理是一种新兴的计算范例，它将生成式机器学习与代码解释器、Web浏览、电子邮件以及更一般的外部资源等工具相结合。这些基于代理的系统代表着个人计算领域正在发生的转变。我们为基于代理的系统的安全基础做出了贡献，并提出了一类新的自动计算的混淆对抗性提示攻击，这些攻击违反了连接到LLM代理的用户资源的机密性和完整性。我们展示了提示优化技术如何在给定模型权重的情况下自动找到这样的提示。我们证明了这种攻击会转移到生产级代理。例如，我们展示了对Mistral的Lechat代理的信息外泄攻击，该攻击分析用户的对话，挑选出个人身份信息，并将其格式化为有效的标记命令，从而导致该数据泄漏到攻击者的服务器。该攻击在端到端评估中显示了近80%的成功率。我们进行了一系列实验来表征这些攻击的有效性，并发现它们在新兴的基于代理的系统上可靠地工作，如Mistral的Lechat、ChatGLM和Meta的Llama。这些攻击是多模式的，我们展示了纯文本和图像领域的变体。



## **37. (Quantum) Indifferentiability and Pre-Computation**

（量子）不可微性和预计算 quant-ph

24 pages

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16595v1) [paper-pdf](http://arxiv.org/pdf/2410.16595v1)

**Authors**: Joseph Carolan, Alexander Poremba, Mark Zhandry

**Abstract**: Indifferentiability is a popular cryptographic paradigm for analyzing the security of ideal objects -- both in a classical as well as in a quantum world. It is typically stated in the form of a composable and simulation-based definition, and captures what it means for a construction (e.g., a cryptographic hash function) to be ``as good as'' an ideal object (e.g., a random oracle). Despite its strength, indifferentiability is not known to offer security against pre-processing attacks in which the adversary gains access to (classical or quantum) advice that is relevant to the particular construction. In this work, we show that indifferentiability is (generically) insufficient for capturing pre-computation. To accommodate this shortcoming, we propose a strengthening of indifferentiability which is not only composable but also takes arbitrary pre-computation into account. As an application, we show that the one-round sponge is indifferentiable (with pre-computation) from a random oracle. This yields the first (and tight) classical/quantum space-time trade-off for one-round sponge inversion.

摘要: 不可微性是一种流行的密码学范式，用于分析理想对象的安全性--无论是在经典还是在量子世界中。它通常以可组合的和基于模拟的定义的形式来陈述，并且捕捉其对于构造(例如，密码散列函数)与理想对象(例如，随机预言)一样好意味着什么。尽管不可微性很强大，但它并不能提供针对预处理攻击的安全性，在这种攻击中，对手可以获得与特定结构相关的(经典或量子)建议。在这项工作中，我们证明了不可微性(一般)不足以捕捉预计算。为了弥补这一不足，我们提出了一种增强的不可微性，它不仅是可合成的，而且还考虑了任意的预计算。作为应用，我们证明了单轮海绵与随机预言是不可区分的(具有预计算)。这为单轮海绵反转产生了第一个(也是严格的)经典/量子时空权衡。



## **38. Conflict-Aware Adversarial Training**

预算意识对抗培训 cs.LG

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16579v1) [paper-pdf](http://arxiv.org/pdf/2410.16579v1)

**Authors**: Zhiyu Xue, Haohan Wang, Yao Qin, Ramtin Pedarsani

**Abstract**: Adversarial training is the most effective method to obtain adversarial robustness for deep neural networks by directly involving adversarial samples in the training procedure. To obtain an accurate and robust model, the weighted-average method is applied to optimize standard loss and adversarial loss simultaneously. In this paper, we argue that the weighted-average method does not provide the best tradeoff for the standard performance and adversarial robustness. We argue that the failure of the weighted-average method is due to the conflict between the gradients derived from standard and adversarial loss, and further demonstrate such a conflict increases with attack budget theoretically and practically. To alleviate this problem, we propose a new trade-off paradigm for adversarial training with a conflict-aware factor for the convex combination of standard and adversarial loss, named \textbf{Conflict-Aware Adversarial Training~(CA-AT)}. Comprehensive experimental results show that CA-AT consistently offers a superior trade-off between standard performance and adversarial robustness under the settings of adversarial training from scratch and parameter-efficient finetuning.

摘要: 对抗性训练通过在训练过程中直接涉及对抗性样本，是获得深层神经网络对抗性稳健性的最有效方法。为了得到准确和稳健的模型，采用加权平均法同时优化标准损失和对抗性损失。在本文中，我们认为加权平均方法没有在标准性能和对手健壮性之间提供最佳的折衷。我们认为，加权平均方法的失败是由于标准损失和对抗性损失的梯度之间的冲突，并进一步从理论和实践上证明了这种冲突随着攻击预算的增加而增加。为了缓解这一问题，我们提出了一种新的具有冲突感知因子的对抗性训练范式-.综合实验结果表明，在对抗性训练从头开始和参数高效精调的情况下，CA-AT一致地在标准性能和对抗性健壮性之间提供了优越的折衷。



## **39. SleeperNets: Universal Backdoor Poisoning Attacks Against Reinforcement Learning Agents**

SleeperNets：针对强化学习代理的通用后门中毒攻击 cs.LG

23 pages, 14 figures, NeurIPS

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2405.20539v2) [paper-pdf](http://arxiv.org/pdf/2405.20539v2)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Reinforcement learning (RL) is an actively growing field that is seeing increased usage in real-world, safety-critical applications -- making it paramount to ensure the robustness of RL algorithms against adversarial attacks. In this work we explore a particularly stealthy form of training-time attacks against RL -- backdoor poisoning. Here the adversary intercepts the training of an RL agent with the goal of reliably inducing a particular action when the agent observes a pre-determined trigger at inference time. We uncover theoretical limitations of prior work by proving their inability to generalize across domains and MDPs. Motivated by this, we formulate a novel poisoning attack framework which interlinks the adversary's objectives with those of finding an optimal policy -- guaranteeing attack success in the limit. Using insights from our theoretical analysis we develop ``SleeperNets'' as a universal backdoor attack which exploits a newly proposed threat model and leverages dynamic reward poisoning techniques. We evaluate our attack in 6 environments spanning multiple domains and demonstrate significant improvements in attack success over existing methods, while preserving benign episodic return.

摘要: 强化学习(RL)是一个正在蓬勃发展的领域，在现实世界中的安全关键应用程序中的使用率正在增加，这使得它对于确保RL算法针对对手攻击的健壮性至关重要。在这项工作中，我们探索了一种特别隐蔽的针对RL的训练时间攻击形式--后门中毒。在这里，对手截取RL代理的训练，目标是当代理在推理时观察到预定触发时可靠地诱导特定动作。我们通过证明他们无法跨域和MDP进行泛化，揭示了以前工作的理论局限性。受此启发，我们提出了一种新颖的中毒攻击框架，该框架将对手的目标与找到最优策略的目标联系起来--保证攻击的最大限度成功。利用我们从理论分析中获得的见解，我们开发了“休眠网络”作为一种通用的后门攻击，它利用了新提出的威胁模型和动态奖励中毒技术。我们在跨越多个域的6个环境中评估了我们的攻击，并展示了与现有方法相比在攻击成功率方面的显著改进，同时保持了良性的间歇性回报。



## **40. Adversarial Inception for Bounded Backdoor Poisoning in Deep Reinforcement Learning**

深度强化学习中有界后门中毒的对抗性初始 cs.LG

10 pages, 5 figures, ICLR 2025

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.13995v2) [paper-pdf](http://arxiv.org/pdf/2410.13995v2)

**Authors**: Ethan Rathbun, Christopher Amato, Alina Oprea

**Abstract**: Recent works have demonstrated the vulnerability of Deep Reinforcement Learning (DRL) algorithms against training-time, backdoor poisoning attacks. These attacks induce pre-determined, adversarial behavior in the agent upon observing a fixed trigger during deployment while allowing the agent to solve its intended task during training. Prior attacks rely on arbitrarily large perturbations to the agent's rewards to achieve both of these objectives - leaving them open to detection. Thus, in this work, we propose a new class of backdoor attacks against DRL which achieve state of the art performance while minimally altering the agent's rewards. These "inception" attacks train the agent to associate the targeted adversarial behavior with high returns by inducing a disjunction between the agent's chosen action and the true action executed in the environment during training. We formally define these attacks and prove they can achieve both adversarial objectives. We then devise an online inception attack which significantly out-performs prior attacks under bounded reward constraints.

摘要: 最近的工作证明了深度强化学习(DRL)算法在抵抗训练时间、后门中毒攻击时的脆弱性。这些攻击在部署期间观察到固定触发器时，会在代理中诱导预先确定的对抗性行为，同时允许代理在培训期间解决其预期任务。以前的攻击依赖于对代理报酬的任意大扰动来实现这两个目标--使它们容易被检测到。因此，在这项工作中，我们提出了一类新的针对DRL的后门攻击，它在最大限度地改变代理的报酬的同时实现了最先进的性能。这些“初始”攻击训练代理将目标对抗性行为与高回报相关联，方法是在培训期间诱导代理选择的动作与在环境中执行的真实动作之间的脱节。我们正式定义了这些攻击，并证明它们可以达到两个对抗性目标。然后，我们设计了一个在线初始攻击，在有限制的报酬约束下，它的性能明显优于先前的攻击。



## **41. A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns**

具有传染性的越狱麻烦制造者扰乱诚实城镇 cs.CL

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16155v1) [paper-pdf](http://arxiv.org/pdf/2410.16155v1)

**Authors**: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.

摘要: 随着大型语言模型的发展，它们作为智能体被广泛应用于各个领域。代理的一个关键组件是内存，它存储重要信息，但容易受到越狱攻击。现有的研究主要集中在单代理攻击和共享内存攻击上。然而，现实世界中的场景通常涉及独立的内存。本文提出了Troublemaker Make Chaos in Honest town(TMCHT)任务，这是一个大规模、多代理、多拓扑、基于文本的攻击评估框架。TMCHT涉及一个攻击者代理试图误导整个代理社会。我们确定了多智能体攻击中的两个主要挑战：(1)非完全图结构，(2)大规模系统。我们将这些挑战归因于一种我们称之为毒性消失的现象。针对这些问题，我们提出了一种对抗性复制传染越狱(ARCJ)方法，通过优化检索后缀使中毒样本更容易检索，并优化复制后缀使中毒样本具有传染能力。我们在TMCHT中展示了我们的方法的优越性，在线路拓扑、星形拓扑和100-代理设置方面分别有23.51%、18.95%和52.93%的改进。鼓励社会各界关注多智能体系统的安全性。



## **42. On the Geometry of Regularization in Adversarial Training: High-Dimensional Asymptotics and Generalization Bounds**

对抗训练中规则化的几何学：多维渐进学和概括界限 stat.ML

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16073v1) [paper-pdf](http://arxiv.org/pdf/2410.16073v1)

**Authors**: Matteo Vilucchio, Nikolaos Tsilivis, Bruno Loureiro, Julia Kempe

**Abstract**: Regularization, whether explicit in terms of a penalty in the loss or implicit in the choice of algorithm, is a cornerstone of modern machine learning. Indeed, controlling the complexity of the model class is particularly important when data is scarce, noisy or contaminated, as it translates a statistical belief on the underlying structure of the data. This work investigates the question of how to choose the regularization norm $\lVert \cdot \rVert$ in the context of high-dimensional adversarial training for binary classification. To this end, we first derive an exact asymptotic description of the robust, regularized empirical risk minimizer for various types of adversarial attacks and regularization norms (including non-$\ell_p$ norms). We complement this analysis with a uniform convergence analysis, deriving bounds on the Rademacher Complexity for this class of problems. Leveraging our theoretical results, we quantitatively characterize the relationship between perturbation size and the optimal choice of $\lVert \cdot \rVert$, confirming the intuition that, in the data scarce regime, the type of regularization becomes increasingly important for adversarial training as perturbations grow in size.

摘要: 正则化，无论是明确的损失惩罚，还是隐含的算法选择，都是现代机器学习的基石。事实上，当数据稀缺、有噪声或受到污染时，控制模型类的复杂性尤其重要，因为它转化为对数据底层结构的统计信念。本文研究了在高维对抗性训练环境下如何选择正则化范数$\lVert\CDOT\rVert$的问题。为此，我们首先给出了各种类型的对抗性攻击和正则化范数(包括非正则范数)下的稳健正则化经验风险最小化的精确渐近描述。我们用一致收敛分析来补充这一分析，得到了这类问题的Rademacher复杂性的界。利用我们的理论结果，我们定量地刻画了扰动大小与最优选择$\lVert\rVert$之间的关系，证实了这样一种直觉，即在数据稀缺的情况下，随着扰动的大小，正则化类型对于对抗性训练变得越来越重要。



## **43. A Differentially Private Energy Trading Mechanism Approaching Social Optimum**

差异化的私人能源交易机制接近社会最优 cs.GT

11 pages, 8 figures

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.04787v2) [paper-pdf](http://arxiv.org/pdf/2410.04787v2)

**Authors**: Yuji Cao, Yue Chen

**Abstract**: This paper proposes a differentially private energy trading mechanism for prosumers in peer-to-peer (P2P) markets, offering provable privacy guarantees while approaching the Nash equilibrium with nearly socially optimal efficiency. We first model the P2P energy trading as a (generalized) Nash game and prove the vulnerability of traditional distributed algorithms to privacy attacks through an adversarial inference model. To address this challenge, we develop a privacy-preserving Nash equilibrium seeking algorithm incorporating carefully calibrated Laplacian noise. We prove that the proposed algorithm achieves $\epsilon$-differential privacy while converging in expectation to the Nash equilibrium with a suitable stepsize. Numerical experiments are conducted to evaluate the algorithm's robustness against privacy attacks, convergence behavior, and optimality compared to the non-private solution. Results demonstrate that our mechanism effectively protects prosumers' sensitive information while maintaining near-optimal market outcomes, offering a practical approach for privacy-preserving coordination in P2P markets.

摘要: 提出了一种在P2P市场中为消费者提供差异化私有能源交易机制，在提供可证明的隐私保证的同时，以接近社会最优效率的方式逼近纳什均衡。我们首先将P2P能量交易建模为(广义)Nash博弈，并通过一个对抗性推理模型证明了传统分布式算法对隐私攻击的脆弱性。为了应对这一挑战，我们开发了一种隐私保护的纳什均衡搜索算法，其中包含了仔细校准的拉普拉斯噪声。我们证明了该算法在以适当的步长期望收敛到纳什均衡的同时，实现了$epsilon$-差分隐私保护。与非私有解相比，数值实验评估了该算法对隐私攻击的健壮性、收敛行为和最优性。结果表明，该机制有效地保护了消费者的敏感信息，同时保持了接近最优的市场结果，为P2P市场中的隐私保护协调提供了一种实用的方法。



## **44. Model Mimic Attack: Knowledge Distillation for Provably Transferable Adversarial Examples**

模型模仿攻击：可证明可转移的对抗性示例的知识提炼 cs.LG

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15889v1) [paper-pdf](http://arxiv.org/pdf/2410.15889v1)

**Authors**: Kirill Lukyanov, Andrew Perminov, Denis Turdakov, Mikhail Pautov

**Abstract**: The vulnerability of artificial neural networks to adversarial perturbations in the black-box setting is widely studied in the literature. The majority of attack methods to construct these perturbations suffer from an impractically large number of queries required to find an adversarial example. In this work, we focus on knowledge distillation as an approach to conduct transfer-based black-box adversarial attacks and propose an iterative training of the surrogate model on an expanding dataset. This work is the first, to our knowledge, to provide provable guarantees on the success of knowledge distillation-based attack on classification neural networks: we prove that if the student model has enough learning capabilities, the attack on the teacher model is guaranteed to be found within the finite number of distillation iterations.

摘要: 文献中广泛研究了人工神经网络在黑匣子环境中对对抗性扰动的脆弱性。构建这些扰动的大多数攻击方法都面临着寻找对抗性示例所需的大量查询的问题。在这项工作中，我们重点关注知识蒸馏，作为一种进行基于传输的黑匣子对抗攻击的方法，并提出在扩展数据集上对代理模型进行迭代训练。据我们所知，这项工作是第一次为基于知识蒸馏的分类神经网络攻击的成功提供可证明的保证：我们证明，如果学生模型具有足够的学习能力，那么对教师模型的攻击就保证在有限数量的蒸馏迭代内被发现。



## **45. Vulnerabilities in Machine Learning-Based Voice Disorder Detection Systems**

基于机器学习的语音障碍检测系统中的漏洞 cs.CR

7 pages, 17 figures, accepted for 16th IEEE INTERNATIONAL WORKSHOP ON  INFORMATION FORENSICS AND SECURITY (WIFS) 2024

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16341v1) [paper-pdf](http://arxiv.org/pdf/2410.16341v1)

**Authors**: Gianpaolo Perelli, Andrea Panzino, Roberto Casula, Marco Micheletto, Giulia Orrù, Gian Luca Marcialis

**Abstract**: The impact of voice disorders is becoming more widely acknowledged as a public health issue. Several machine learning-based classifiers with the potential to identify disorders have been used in recent studies to differentiate between normal and pathological voices and sounds. In this paper, we focus on analyzing the vulnerabilities of these systems by exploring the possibility of attacks that can reverse classification and compromise their reliability. Given the critical nature of personal health information, understanding which types of attacks are effective is a necessary first step toward improving the security of such systems. Starting from the original audios, we implement various attack methods, including adversarial, evasion, and pitching techniques, and evaluate how state-of-the-art disorder detection models respond to them. Our findings identify the most effective attack strategies, underscoring the need to address these vulnerabilities in machine-learning systems used in the healthcare domain.

摘要: 嗓音障碍的影响正越来越广泛地被认为是一个公共卫生问题。在最近的研究中，一些基于机器学习的分类器被用于区分正常和病理的声音和声音，这些分类器具有识别疾病的潜力。在本文中，我们通过探索攻击的可能性来分析这些系统的漏洞，这些攻击可以颠倒分类并损害其可靠性。鉴于个人健康信息的关键性质，了解哪些类型的攻击是有效的，这是提高此类系统安全性的必要第一步。从原始音频开始，我们实现了各种攻击方法，包括对抗性、躲避和投球技术，并评估了最先进的障碍检测模型如何响应它们。我们的发现确定了最有效的攻击策略，强调了解决医疗保健领域使用的机器学习系统中的这些漏洞的必要性。



## **46. NetSafe: Exploring the Topological Safety of Multi-agent Networks**

NetSafe：探索多代理网络的布局安全 cs.MA

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15686v1) [paper-pdf](http://arxiv.org/pdf/2410.15686v1)

**Authors**: Miao Yu, Shilong Wang, Guibin Zhang, Junyuan Mao, Chenlong Yin, Qijiong Liu, Qingsong Wen, Kun Wang, Yang Wang

**Abstract**: Large language models (LLMs) have empowered nodes within multi-agent networks with intelligence, showing growing applications in both academia and industry. However, how to prevent these networks from generating malicious information remains unexplored with previous research on single LLM's safety be challenging to transfer. In this paper, we focus on the safety of multi-agent networks from a topological perspective, investigating which topological properties contribute to safer networks. To this end, we propose a general framework, NetSafe along with an iterative RelCom interaction to unify existing diverse LLM-based agent frameworks, laying the foundation for generalized topological safety research. We identify several critical phenomena when multi-agent networks are exposed to attacks involving misinformation, bias, and harmful information, termed as Agent Hallucination and Aggregation Safety. Furthermore, we find that highly connected networks are more susceptible to the spread of adversarial attacks, with task performance in a Star Graph Topology decreasing by 29.7%. Besides, our proposed static metrics aligned more closely with real-world dynamic evaluations than traditional graph-theoretic metrics, indicating that networks with greater average distances from attackers exhibit enhanced safety. In conclusion, our work introduces a new topological perspective on the safety of LLM-based multi-agent networks and discovers several unreported phenomena, paving the way for future research to explore the safety of such networks.

摘要: 大型语言模型(LLM)已经为多代理网络中的节点赋予了智能，显示出在学术界和工业中日益增长的应用。然而，如何防止这些网络产生恶意信息还没有被探索，以往关于单个LLM的安全传输的研究是具有挑战性的。本文从拓扑学的角度研究了多智能体网络的安全性，研究了哪些拓扑性质有助于网络的安全。为此，我们提出了一个通用的框架NetSafe以及一个迭代的RelCom交互来统一现有的各种基于LLM的代理框架，为广义拓扑安全研究奠定了基础。我们确定了当多智能体网络暴露于涉及错误信息、偏见和有害信息的攻击时的几个关键现象，称为智能体幻觉和聚集安全。此外，我们发现，高连接网络更容易受到敌意攻击的传播，星图拓扑中的任务性能下降了29.7%。此外，我们提出的静态度量比传统的图论度量更接近真实世界的动态评估，这表明离攻击者的平均距离越大的网络表现出更高的安全性。综上所述，我们的工作为基于LLM的多智能体网络的安全性引入了一种新的拓扑观，并发现了一些未被报道的现象，为进一步研究此类网络的安全性铺平了道路。



## **47. Patrol Security Game: Defending Against Adversary with Freedom in Attack Timing, Location, and Duration**

巡逻安全游戏：在攻击时间、地点和持续时间上自由防御对手 cs.AI

Under review of TCPS

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15600v1) [paper-pdf](http://arxiv.org/pdf/2410.15600v1)

**Authors**: Hao-Tsung Yang, Ting-Kai Weng, Ting-Yu Chang, Kin Sum Liu, Shan Lin, Jie Gao, Shih-Yu Tsai

**Abstract**: We explored the Patrol Security Game (PSG), a robotic patrolling problem modeled as an extensive-form Stackelberg game, where the attacker determines the timing, location, and duration of their attack. Our objective is to devise a patrolling schedule with an infinite time horizon that minimizes the attacker's payoff. We demonstrated that PSG can be transformed into a combinatorial minimax problem with a closed-form objective function. By constraining the defender's strategy to a time-homogeneous first-order Markov chain (i.e., the patroller's next move depends solely on their current location), we proved that the optimal solution in cases of zero penalty involves either minimizing the expected hitting time or return time, depending on the attacker model, and that these solutions can be computed efficiently. Additionally, we observed that increasing the randomness in the patrol schedule reduces the attacker's expected payoff in high-penalty cases. However, the minimax problem becomes non-convex in other scenarios. To address this, we formulated a bi-criteria optimization problem incorporating two objectives: expected maximum reward and entropy. We proposed three graph-based algorithms and one deep reinforcement learning model, designed to efficiently balance the trade-off between these two objectives. Notably, the third algorithm can identify the optimal deterministic patrol schedule, though its runtime grows exponentially with the number of patrol spots. Experimental results validate the effectiveness and scalability of our solutions, demonstrating that our approaches outperform state-of-the-art baselines on both synthetic and real-world crime datasets.

摘要: 我们探索了巡逻安全游戏(PSG)，这是一个机器人巡逻问题，被建模为一个广泛形式的Stackelberg游戏，攻击者决定他们攻击的时间、位置和持续时间。我们的目标是设计一个无限时间范围的巡逻计划，将攻击者的收益降至最低。我们证明了PSG可以转化为目标函数为闭合形式的组合极大极小问题。通过将防御者的策略约束为时间齐次的一阶马尔可夫链(即，巡逻者的下一步行动仅取决于他们的当前位置)，我们证明了零惩罚情况下的最优解包括最小化期望命中时间或返回时间，这取决于攻击者的模型，并且这些解可以有效地计算出来。此外，我们观察到，增加巡逻计划的随机性会降低攻击者在高惩罚情况下的预期回报。然而，在其他情况下，极小极大问题变得非凸。为了解决这个问题，我们提出了一个包含两个目标的双目标优化问题：期望最大回报和期望最大熵。我们提出了三个基于图的算法和一个深度强化学习模型，旨在有效地平衡这两个目标之间的平衡。值得注意的是，第三种算法可以确定最优的确定性巡逻计划，尽管其运行时间随着巡逻地点的数量呈指数增长。实验结果验证了我们的解决方案的有效性和可扩展性，表明我们的方法在合成和真实世界犯罪数据集上的性能都优于最先进的基线。



## **48. TrojanForge: Generating Adversarial Hardware Trojan Examples with Reinforcement Learning**

TrojanForge：通过强化学习生成对抗性硬件特洛伊示例 cs.CR

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2405.15184v2) [paper-pdf](http://arxiv.org/pdf/2405.15184v2)

**Authors**: Amin Sarihi, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: The Hardware Trojan (HT) problem can be thought of as a continuous game between attackers and defenders, each striving to outsmart the other by leveraging any available means for an advantage. Machine Learning (ML) has recently played a key role in advancing HT research. Various novel techniques, such as Reinforcement Learning (RL) and Graph Neural Networks (GNNs), have shown HT insertion and detection capabilities. HT insertion with ML techniques, specifically, has seen a spike in research activity due to the shortcomings of conventional HT benchmarks and the inherent human design bias that occurs when we create them. This work continues this innovation by presenting a tool called TrojanForge, capable of generating HT adversarial examples that defeat HT detectors; demonstrating the capabilities of GAN-like adversarial tools for automatic HT insertion. We introduce an RL environment where the RL insertion agent interacts with HT detectors in an insertion-detection loop where the agent collects rewards based on its success in bypassing HT detectors. Our results show that this process helps inserted HTs evade various HT detectors, achieving high attack success percentages. This tool provides insight into why HT insertion fails in some instances and how we can leverage this knowledge in defense.

摘要: 硬件特洛伊木马(HT)问题可以被认为是攻击者和防御者之间的一场持续的游戏，双方都在努力利用任何可用的手段来获取优势，以智胜对方。近年来，机器学习(ML)在推进HT研究中发挥了关键作用。各种新的技术，如强化学习(RL)和图形神经网络(GNNS)，已经显示出HT插入和检测能力。特别是，由于传统的HT基准的缺点以及我们创建它们时固有的人为设计偏差，使用ML技术插入HT的研究活动出现了激增。这项工作通过提供一个名为TrojanForge的工具来继续这一创新，该工具能够生成击败HT检测器的HT对抗示例；展示了GAN类对抗工具自动插入HT的能力。我们引入了一种RL环境，在该环境中，RL插入剂与HT检测器在插入检测环路中相互作用，其中代理根据其成功绕过HT检测器来收取奖励。我们的结果表明，这个过程帮助插入的HTS避开了各种HT探测器，获得了高攻击成功率。这个工具提供了一些关于为什么在某些情况下HT插入失败以及我们如何在防御中利用这一知识的洞察。



## **49. BRC20 Pinning Attack**

BRRC 20钉扎攻击 cs.CR

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.11295v2) [paper-pdf](http://arxiv.org/pdf/2410.11295v2)

**Authors**: Minfeng Qi, Qin Wang, Zhipeng Wang, Lin Zhong, Tianqing Zhu, Shiping Chen, William Knottenbelt

**Abstract**: BRC20 tokens are a type of non-fungible asset on the Bitcoin network. They allow users to embed customized content within Bitcoin satoshis. The related token frenzy has reached a market size of US$2,650b over the past year (2023Q3-2024Q3). However, this intuitive design has not undergone serious security scrutiny.   We present the first in-depth analysis of the BRC20 transfer mechanism and identify a critical attack vector. A typical BRC20 transfer involves two bundled on-chain transactions with different fee levels: the first (i.e., Tx1) with a lower fee inscribes the transfer request, while the second (i.e., Tx2) with a higher fee finalizes the actual transfer. We find that an adversary can exploit this by sending a manipulated fee transaction (falling between the two fee levels), which allows Tx1 to be processed while Tx2 remains pinned in the mempool. This locks the BRC20 liquidity and disrupts normal transfers for users. We term this BRC20 pinning attack.   Our attack exposes an inherent design flaw that can be applied to 90+% inscription-based tokens within the Bitcoin ecosystem.   We also conducted the attack on Binance's ORDI hot wallet (the most prevalent BRC20 token and the most active wallet), resulting in a temporary suspension of ORDI withdrawals on Binance for 3.5 hours, which were shortly resumed after our communication.

摘要: BRC20代币是比特币网络上的一种不可替代资产。它们允许用户在比特币Satoshis中嵌入定制内容。在过去的一年里(2023Q3-2024Q3)，相关的代币狂潮已经达到了2.65万亿美元的市场规模。然而，这种直观的设计并没有经过严格的安全审查。我们首次深入分析了BRC20的传输机制，并确定了一个关键的攻击载体。典型的BRC20转移涉及两个不同费用水平的捆绑链上交易：第一个费用较低的(即TX1)记录转移请求，而第二个(即Tx2)费用较高的完成实际转移。我们发现，对手可以通过发送被操纵的费用事务(介于两个费用水平之间)来利用这一点，这允许在Tx1被处理的同时Tx2仍然被固定在内存池中。这锁定了BRC20的流动性，并扰乱了用户的正常转账。我们称之为BRC20钉住攻击。我们的攻击暴露了一个固有的设计缺陷，该缺陷可以应用于比特币生态系统中90%以上的铭文令牌。我们还对Binance的Ordi热钱包(最流行的BRC20代币和最活跃的钱包)进行了攻击，导致Binance上的Ordi提款暂时暂停3.5小时，并在我们沟通后不久恢复。



## **50. Revisit, Extend, and Enhance Hessian-Free Influence Functions**

重新审视、扩展和增强无黑森影响力功能 cs.LG

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2405.17490v2) [paper-pdf](http://arxiv.org/pdf/2405.17490v2)

**Authors**: Ziao Yang, Han Yue, Jian Chen, Hongfu Liu

**Abstract**: Influence functions serve as crucial tools for assessing sample influence in model interpretation, subset training set selection, noisy label detection, and more. By employing the first-order Taylor extension, influence functions can estimate sample influence without the need for expensive model retraining. However, applying influence functions directly to deep models presents challenges, primarily due to the non-convex nature of the loss function and the large size of model parameters. This difficulty not only makes computing the inverse of the Hessian matrix costly but also renders it non-existent in some cases. Various approaches, including matrix decomposition, have been explored to expedite and approximate the inversion of the Hessian matrix, with the aim of making influence functions applicable to deep models. In this paper, we revisit a specific, albeit naive, yet effective approximation method known as TracIn. This method substitutes the inverse of the Hessian matrix with an identity matrix. We provide deeper insights into why this simple approximation method performs well. Furthermore, we extend its applications beyond measuring model utility to include considerations of fairness and robustness. Finally, we enhance TracIn through an ensemble strategy. To validate its effectiveness, we conduct experiments on synthetic data and extensive evaluations on noisy label detection, sample selection for large language model fine-tuning, and defense against adversarial attacks.

摘要: 影响函数在模型解释、子集训练集选择、噪声标签检测等方面用作评估样本影响的重要工具。通过使用一阶泰勒扩展，影响函数可以估计样本影响，而不需要昂贵的模型重新训练。然而，直接将影响函数应用于深层模型会带来挑战，这主要是由于损失函数的非凸性和模型参数的大尺寸。这一困难不仅使计算海森矩阵的逆的成本高昂，而且在某些情况下使其不存在。已经探索了各种方法，包括矩阵分解，以加快和近似海森矩阵的求逆，目的是使影响函数适用于深层模式。在这篇文章中，我们回顾了一种特定的，尽管很幼稚，但有效的近似方法，称为TracIn。该方法用单位矩阵代替海森矩阵的逆。我们对为什么这种简单的近似方法表现良好提供了更深层次的见解。此外，我们将它的应用扩展到测量模型效用之外，包括对公平性和稳健性的考虑。最后，我们通过集成策略增强了TracIn。为了验证其有效性，我们在合成数据上进行了实验，并在噪声标签检测、大语言模型微调的样本选择以及对对手攻击的防御方面进行了广泛的评估。



