# Latest Adversarial Attack Papers
**update at 2024-04-20 09:32:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. KDk: A Defense Mechanism Against Label Inference Attacks in Vertical Federated Learning**

KDk：垂直联邦学习中针对标签推理攻击的防御机制 cs.LG

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12369v1) [paper-pdf](http://arxiv.org/pdf/2404.12369v1)

**Authors**: Marco Arazzi, Serena Nicolazzo, Antonino Nocera

**Abstract**: Vertical Federated Learning (VFL) is a category of Federated Learning in which models are trained collaboratively among parties with vertically partitioned data. Typically, in a VFL scenario, the labels of the samples are kept private from all the parties except for the aggregating server, that is the label owner. Nevertheless, recent works discovered that by exploiting gradient information returned by the server to bottom models, with the knowledge of only a small set of auxiliary labels on a very limited subset of training data points, an adversary can infer the private labels. These attacks are known as label inference attacks in VFL. In our work, we propose a novel framework called KDk, that combines Knowledge Distillation and k-anonymity to provide a defense mechanism against potential label inference attacks in a VFL scenario. Through an exhaustive experimental campaign we demonstrate that by applying our approach, the performance of the analyzed label inference attacks decreases consistently, even by more than 60%, maintaining the accuracy of the whole VFL almost unaltered.

摘要: 垂直联合学习(VFL)是联合学习的一个类别，其中模型在具有垂直分割数据的各方之间协作训练。通常，在VFL场景中，除了聚合服务器(即标签所有者)之外，样本的标签对所有各方都是私有的。然而，最近的工作发现，通过利用服务器返回到底层模型的梯度信息，只要知道非常有限的训练数据点子集上的一小部分辅助标签，对手就可以推断出私有标签。这些攻击在VFL中被称为标签推理攻击。在我们的工作中，我们提出了一种新的框架KDK，它结合了知识蒸馏和k-匿名性来提供一种防御VFL场景中潜在的标签推理攻击的机制。通过详尽的实验证明，应用我们的方法，分析的标签推理攻击的性能持续下降，甚至下降了60%以上，几乎保持了整个VFL的准确率不变。



## **2. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV-28 K：评估多模式大型语言模型对抗越狱攻击的稳健性的基准 cs.CR

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.03027v2) [paper-pdf](http://arxiv.org/pdf/2404.03027v2)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **3. Transferability Ranking of Adversarial Examples**

对抗性示例的可转让性排名 cs.LG

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2208.10878v2) [paper-pdf](http://arxiv.org/pdf/2208.10878v2)

**Authors**: Mosh Levy, Guy Amit, Yuval Elovici, Yisroel Mirsky

**Abstract**: Adversarial transferability in black-box scenarios presents a unique challenge: while attackers can employ surrogate models to craft adversarial examples, they lack assurance on whether these examples will successfully compromise the target model. Until now, the prevalent method to ascertain success has been trial and error-testing crafted samples directly on the victim model. This approach, however, risks detection with every attempt, forcing attackers to either perfect their first try or face exposure. Our paper introduces a ranking strategy that refines the transfer attack process, enabling the attacker to estimate the likelihood of success without repeated trials on the victim's system. By leveraging a set of diverse surrogate models, our method can predict transferability of adversarial examples. This strategy can be used to either select the best sample to use in an attack or the best perturbation to apply to a specific sample. Using our strategy, we were able to raise the transferability of adversarial examples from a mere 20% - akin to random selection-up to near upper-bound levels, with some scenarios even witnessing a 100% success rate. This substantial improvement not only sheds light on the shared susceptibilities across diverse architectures but also demonstrates that attackers can forego the detectable trial-and-error tactics raising increasing the threat of surrogate-based attacks.

摘要: 黑盒场景中的对抗性可转移性是一个独特的挑战：虽然攻击者可以使用代理模型来制作对抗性示例，但他们无法确定这些示例是否会成功危害目标模型。到目前为止，确定成功的流行方法一直是试验和错误测试-直接在受害者模型上测试精心制作的样本。然而，这种方法每次尝试都有被检测的风险，迫使攻击者要么完善他们的第一次尝试，要么面临曝光。本文介绍了一种改进传输攻击过程的排序策略，使攻击者能够估计成功的可能性，而不需要在受害者的系统上重复试验。通过利用一组不同的代理模型，我们的方法可以预测对抗性例子的可转移性。此策略可用于选择用于攻击的最佳样本或应用于特定样本的最佳扰动。使用我们的策略，我们能够将对抗性例子的可转移性从仅仅20%-类似于随机选择-提高到接近上限的水平，有些场景甚至见证了100%的成功率。这一重大改进不仅揭示了不同体系结构之间的共同漏洞，还表明攻击者可以放弃可检测到的试错策略，从而增加了基于代理的攻击的威胁。



## **4. Struggle with Adversarial Defense? Try Diffusion**

与对抗性防御作斗争？尝试扩散 cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.08273v2) [paper-pdf](http://arxiv.org/pdf/2404.08273v2)

**Authors**: Yujie Li, Yanbin Wang, Haitao Xu, Bin Liu, Jianguo Sun, Zhenhao Guo, Wenrui Ma

**Abstract**: Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_{\infty}$ norm-bounded perturbations and 86.05% against $l_{2}$ norm-bounded perturbations, respectively, with $\epsilon=0.05$.

摘要: 对抗性攻击通过引入微妙的扰动来导致错误分类。近年来，扩散模型被应用到图像分类器中，通过对抗性训练或净化对抗性噪声来提高对抗性稳健性。然而，基于扩散的对抗性训练往往会遇到收敛挑战和较高的计算开销。此外，基于扩散的净化不可避免地会导致数据转移，并被认为容易受到更强的适应性攻击。为了解决这些问题，我们提出了真值最大化扩散分类器(TMDC)，这是一种生成式贝叶斯分类器，它建立在预先训练的扩散模型和贝叶斯定理的基础上。与数据驱动的分类器不同，TMDC在贝叶斯原理的指导下，利用扩散模型的条件似然来确定输入图像的类别概率，从而避免了数据迁移的影响和对抗性训练的限制。此外，为了增强TMDC对更强大的对手攻击的韧性，我们提出了一种扩散分类器的优化策略。该策略包括在扰动数据集上对扩散模型进行后训练，以地面真实标签为条件，引导扩散模型学习数据分布，并最大化地面真实标签下的似然。该方法在CIFAR10数据集上取得了较好的抗重白盒攻击和强自适应攻击的性能。具体地说，TMDC对$L范数有界摄动和L范数有界摄动的稳健精度分别为82.81%和86.05%，其中$epsilon=0.05$。



## **5. Advancing the Robustness of Large Language Models through Self-Denoised Smoothing**

通过自去噪平滑提高大型语言模型的鲁棒性 cs.CL

Accepted by NAACL 2024. Jiabao, Bairu, Zhen, Guanhua contributed  equally. This is an updated version of the paper: arXiv:2307.07171

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12274v1) [paper-pdf](http://arxiv.org/pdf/2404.12274v1)

**Authors**: Jiabao Ji, Bairu Hou, Zhen Zhang, Guanhua Zhang, Wenqi Fan, Qing Li, Yang Zhang, Gaowen Liu, Sijia Liu, Shiyu Chang

**Abstract**: Although large language models (LLMs) have achieved significant success, their vulnerability to adversarial perturbations, including recent jailbreak attacks, has raised considerable concerns. However, the increasing size of these models and their limited access make improving their robustness a challenging task. Among various defense strategies, randomized smoothing has shown great potential for LLMs, as it does not require full access to the model's parameters or fine-tuning via adversarial training. However, randomized smoothing involves adding noise to the input before model prediction, and the final model's robustness largely depends on the model's performance on these noise corrupted data. Its effectiveness is often limited by the model's sub-optimal performance on noisy data. To address this issue, we propose to leverage the multitasking nature of LLMs to first denoise the noisy inputs and then to make predictions based on these denoised versions. We call this procedure self-denoised smoothing. Unlike previous denoised smoothing techniques in computer vision, which require training a separate model to enhance the robustness of LLMs, our method offers significantly better efficiency and flexibility. Our experimental results indicate that our method surpasses existing methods in both empirical and certified robustness in defending against adversarial attacks for both downstream tasks and human alignments (i.e., jailbreak attacks). Our code is publicly available at https://github.com/UCSB-NLP-Chang/SelfDenoise

摘要: 虽然大型语言模型(LLM)已经取得了巨大的成功，但它们在对抗扰动中的脆弱性，包括最近的越狱攻击，已经引起了相当大的关注。然而，这些模型的规模越来越大，访问范围有限，因此提高它们的稳健性是一项具有挑战性的任务。在各种防御策略中，随机平滑在LLMS中显示出巨大的潜力，因为它不需要完全获取模型参数或通过对抗性训练进行微调。然而，随机平滑涉及在模型预测之前向输入添加噪声，而最终模型的稳健性在很大程度上取决于模型对这些噪声污染数据的性能。它的有效性往往受到模型在噪声数据上的次优性能的限制。为了解决这个问题，我们建议利用LLMS的多任务特性来首先对噪声输入进行去噪，然后基于这些去噪版本进行预测。我们称这一过程为自去噪平滑。与计算机视觉中以前的去噪平滑技术不同，我们的方法提供了更好的效率和灵活性，需要训练单独的模型来增强LLMS的鲁棒性。我们的实验结果表明，我们的方法在抵抗下游任务和人类对齐(即越狱攻击)的对手攻击方面，无论是经验上还是经过验证的稳健性都优于现有方法。我们的代码在https://github.com/UCSB-NLP-Chang/SelfDenoise上公开提供



## **6. Efficiently Adversarial Examples Generation for Visual-Language Models under Targeted Transfer Scenarios using Diffusion Models**

使用扩散模型高效生成目标迁移场景下视觉语言模型的对抗性示例 cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.10335v2) [paper-pdf](http://arxiv.org/pdf/2404.10335v2)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Qing Guo

**Abstract**: Targeted transfer-based attacks involving adversarial examples pose a significant threat to large visual-language models (VLMs). However, the state-of-the-art (SOTA) transfer-based attacks incur high costs due to excessive iteration counts. Furthermore, the generated adversarial examples exhibit pronounced adversarial noise and demonstrate limited efficacy in evading defense methods such as DiffPure. To address these issues, inspired by score matching, we introduce AdvDiffVLM, which utilizes diffusion models to generate natural, unrestricted adversarial examples. Specifically, AdvDiffVLM employs Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring the adversarial examples produced contain natural adversarial semantics and thus possess enhanced transferability. Simultaneously, to enhance the quality of adversarial examples further, we employ the GradCAM-guided Mask method to disperse adversarial semantics throughout the image, rather than concentrating them in a specific area. Experimental results demonstrate that our method achieves a speedup ranging from 10X to 30X compared to existing transfer-based attack methods, while maintaining superior quality of adversarial examples. Additionally, the generated adversarial examples possess strong transferability and exhibit increased robustness against adversarial defense methods. Notably, AdvDiffVLM can successfully attack commercial VLMs, including GPT-4V, in a black-box manner.

摘要: 涉及对抗性例子的基于目标转移的攻击对大型视觉语言模型(VLM)构成了重大威胁。然而，最先进的(SOTA)基于传输的攻击由于迭代次数过多而导致高昂的成本。此外，生成的对抗性示例显示出明显的对抗性噪声，并且在躲避DiffPure等防御方法方面表现出有限的有效性。为了解决这些问题，受分数匹配的启发，我们引入了AdvDiffVLM，它利用扩散模型来生成自然的、不受限制的对抗性示例。具体地说，AdvDiffVLM在扩散模型的逆向生成过程中使用自适应集成梯度估计来修正分数，确保生成的对抗性实例包含自然对抗性语义，从而具有增强的可转移性。同时，为了进一步提高对抗性实例的质量，我们使用了GradCAM引导的掩码方法，将对抗性语义分散在整个图像中，而不是集中在特定的区域。实验结果表明，与现有的基于传输的攻击方法相比，该方法在保持较好的对抗性实例质量的同时，获得了10倍到30倍的加速比。此外，生成的对抗性实例具有很强的可移植性，并且对对抗性防御方法表现出更强的稳健性。值得注意的是，AdvDiffVLM可以以黑盒方式成功攻击商业VLM，包括GPT-4V。



## **7. Fortify the Guardian, Not the Treasure: Resilient Adversarial Detectors**

强化守护者，而不是宝藏：弹性对抗探测器 cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12120v1) [paper-pdf](http://arxiv.org/pdf/2404.12120v1)

**Authors**: Raz Lapid, Almog Dubin, Moshe Sipper

**Abstract**: This paper presents RADAR-Robust Adversarial Detection via Adversarial Retraining-an approach designed to enhance the robustness of adversarial detectors against adaptive attacks, while maintaining classifier performance. An adaptive attack is one where the attacker is aware of the defenses and adapts their strategy accordingly. Our proposed method leverages adversarial training to reinforce the ability to detect attacks, without compromising clean accuracy. During the training phase, we integrate into the dataset adversarial examples, which were optimized to fool both the classifier and the adversarial detector, enabling the adversarial detector to learn and adapt to potential attack scenarios. Experimental evaluations on the CIFAR-10 and SVHN datasets demonstrate that our proposed algorithm significantly improves a detector's ability to accurately identify adaptive adversarial attacks -- without sacrificing clean accuracy.

摘要: 本文提出了RADART--通过对抗重训练的鲁棒对抗检测--一种旨在增强对抗检测器对抗自适应攻击的鲁棒性的方法，同时保持分类器性能。自适应攻击是攻击者意识到防御并相应调整策略的攻击。我们提出的方法利用对抗性训练来加强检测攻击的能力，而不会损害准确性。在训练阶段，我们将对抗性示例集成到数据集中，这些示例经过优化以愚弄分类器和对抗性检测器，使对抗性检测器能够学习和适应潜在的攻击场景。对CIFAR-10和SVHN数据集的实验评估表明，我们提出的算法显着提高了检测器准确识别自适应对抗攻击的能力，而不会牺牲清晰的准确性。



## **8. Enhance Robustness of Language Models Against Variation Attack through Graph Integration**

通过图集成增强语言模型抗变异攻击的鲁棒性 cs.CL

12 pages, 4 figures, accepted by COLING 2024

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12014v1) [paper-pdf](http://arxiv.org/pdf/2404.12014v1)

**Authors**: Zi Xiong, Lizhi Qing, Yangyang Kang, Jiawei Liu, Hongsong Li, Changlong Sun, Xiaozhong Liu, Wei Lu

**Abstract**: The widespread use of pre-trained language models (PLMs) in natural language processing (NLP) has greatly improved performance outcomes. However, these models' vulnerability to adversarial attacks (e.g., camouflaged hints from drug dealers), particularly in the Chinese language with its rich character diversity/variation and complex structures, hatches vital apprehension. In this study, we propose a novel method, CHinese vAriatioN Graph Enhancement (CHANGE), to increase the robustness of PLMs against character variation attacks in Chinese content. CHANGE presents a novel approach for incorporating a Chinese character variation graph into the PLMs. Through designing different supplementary tasks utilizing the graph structure, CHANGE essentially enhances PLMs' interpretation of adversarially manipulated text. Experiments conducted in a multitude of NLP tasks show that CHANGE outperforms current language models in combating against adversarial attacks and serves as a valuable contribution to robust language model research. These findings contribute to the groundwork on robust language models and highlight the substantial potential of graph-guided pre-training strategies for real-world applications.

摘要: 预训练语言模型(PLM)在自然语言处理(NLP)中的广泛使用极大地改善了性能结果。然而，这些模型易受敌意攻击(例如，来自毒贩的伪装暗示)，特别是在具有丰富字符多样性/多样性和复杂结构的中文中，孵化了至关重要的担忧。在这项研究中，我们提出了一种新的方法-中文变异图增强(CHANGE)，以提高PLM对中文内容中字符变异攻击的稳健性。Change提出了一种新的方法，将汉字变异图结合到PLM中。通过利用图形结构设计不同的补充任务，Change实质上增强了PLM对对抗性操纵的文本的理解。在大量自然语言处理任务中进行的实验表明，CHANGE在抵抗敌意攻击方面优于现有的语言模型，是对健壮语言模型研究的有价值的贡献。这些发现有助于为健壮的语言模型奠定基础，并突出了图形引导的预训练策略在现实世界应用中的巨大潜力。



## **9. Exploring DNN Robustness Against Adversarial Attacks Using Approximate Multipliers**

使用近似乘数探索DNN对抗对抗攻击的鲁棒性 cs.LG

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11665v1) [paper-pdf](http://arxiv.org/pdf/2404.11665v1)

**Authors**: Mohammad Javad Askarizadeh, Ebrahim Farahmand, Jorge Castro-Godinez, Ali Mahani, Laura Cabrera-Quiros, Carlos Salazar-Garcia

**Abstract**: Deep Neural Networks (DNNs) have advanced in many real-world applications, such as healthcare and autonomous driving. However, their high computational complexity and vulnerability to adversarial attacks are ongoing challenges. In this letter, approximate multipliers are used to explore DNN robustness improvement against adversarial attacks. By uniformly replacing accurate multipliers for state-of-the-art approximate ones in DNN layer models, we explore the DNNs robustness against various adversarial attacks in a feasible time. Results show up to 7% accuracy drop due to approximations when no attack is present while improving robust accuracy up to 10% when attacks applied.

摘要: 深度神经网络（DNN）在许多现实世界应用中取得了进步，例如医疗保健和自动驾驶。然而，它们的高计算复杂性和对对抗攻击的脆弱性是持续的挑战。在这封信中，使用近似乘数来探索DNN针对对抗攻击的鲁棒性改进。通过在DNN层模型中统一替换最先进的近似乘数，我们探索DNN在可行的时间内对抗各种对抗攻击的鲁棒性。结果显示，当不存在攻击时，由于逼近，准确性会下降高达7%，而当应用攻击时，鲁棒性准确性会提高高达10%。



## **10. Towards White Box Deep Learning**

走向白盒深度学习 cs.LG

16 pages, 12 figures, independent research, v5 changes: Expanded  Abstract and Related Work section; minor wording improvements

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2403.09863v5) [paper-pdf](http://arxiv.org/pdf/2403.09863v5)

**Authors**: Maciej Satkiewicz

**Abstract**: Deep neural networks learn fragile "shortcut" features, rendering them difficult to interpret (black box) and vulnerable to adversarial attacks. This paper proposes semantic features as a general architectural solution to this problem. The main idea is to make features locality-sensitive in the adequate semantic topology of the domain, thus introducing a strong regularization. The proof of concept network is lightweight, inherently interpretable and achieves almost human-level adversarial test metrics - with no adversarial training! These results and the general nature of the approach warrant further research on semantic features. The code is available at https://github.com/314-Foundation/white-box-nn

摘要: 深度神经网络学习脆弱的“捷径”特征，使其难以解释（黑匣子）并且容易受到对抗性攻击。本文提出了语义特征作为该问题的通用架构解决方案。主要思想是使特征在域的充分语义布局中对局部敏感，从而引入强正规化。概念验证网络是轻量级的，本质上是可解释的，并且实现了几乎人类水平的对抗测试指标-无需对抗训练！这些结果和该方法的一般性质值得对语义特征进行进一步研究。该代码可在https://github.com/314-Foundation/white-box-nn上获取



## **11. Towards Reliable Empirical Machine Unlearning Evaluation: A Game-Theoretic View**

迈向可靠的经验机器无学习评估：游戏理论的观点 cs.LG

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11577v1) [paper-pdf](http://arxiv.org/pdf/2404.11577v1)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning is the process of updating machine learning models to remove the information of specific training data samples, in order to comply with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics that lack reliability. Specifically, we propose a game-theoretic framework that formalizes the evaluation process as a game between unlearning algorithms and MIA adversaries, measuring the data removal efficacy of unlearning algorithms by the capability of the MIA adversaries. Through careful design of the game, we demonstrate that the natural evaluation metric induced from the game enjoys provable guarantees that the existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient algorithm to estimate the evaluation metric induced from the game, and demonstrate its effectiveness through both theoretical analysis and empirical experiments. This work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.

摘要: 机器遗忘是更新机器学习模型以删除特定训练数据样本的信息的过程，以遵守允许个人请求删除其个人数据的数据保护法规。尽管最近有许多遗忘算法的发展，但对这些算法的可靠评估仍然是一个开放的研究问题。在这项工作中，我们专注于基于成员关系推理攻击(MIA)的评估，这是评估遗忘算法最常见的方法之一，并解决了现有评估指标缺乏可靠性的各种缺陷。具体地说，我们提出了一个博弈论框架，将评估过程形式化为遗忘算法与MIA对手之间的博弈，通过MIA对手的能力来衡量遗忘算法的数据去除效率。通过对游戏的精心设计，我们证明了由游戏产生的自然评价指标享有现有评价指标不能满足的可证明保证。在此基础上，提出了一种实用高效的评估指标估计算法，并通过理论分析和实验验证了该算法的有效性。这项工作提供了一种新颖而可靠的方法来对遗忘算法进行经验评估，为开发更有效的遗忘技术铺平了道路。



## **12. GenFighter: A Generative and Evolutive Textual Attack Removal**

GenFighter：生成性和进化性的文本攻击删除 cs.LG

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11538v1) [paper-pdf](http://arxiv.org/pdf/2404.11538v1)

**Authors**: Md Athikul Islam, Edoardo Serra, Sushil Jajodia

**Abstract**: Adversarial attacks pose significant challenges to deep neural networks (DNNs) such as Transformer models in natural language processing (NLP). This paper introduces a novel defense strategy, called GenFighter, which enhances adversarial robustness by learning and reasoning on the training classification distribution. GenFighter identifies potentially malicious instances deviating from the distribution, transforms them into semantically equivalent instances aligned with the training data, and employs ensemble techniques for a unified and robust response. By conducting extensive experiments, we show that GenFighter outperforms state-of-the-art defenses in accuracy under attack and attack success rate metrics. Additionally, it requires a high number of queries per attack, making the attack more challenging in real scenarios. The ablation study shows that our approach integrates transfer learning, a generative/evolutive procedure, and an ensemble method, providing an effective defense against NLP adversarial attacks.

摘要: 对抗性攻击对深度神经网络(DNN)提出了巨大的挑战，例如自然语言处理(NLP)中的Transformer模型。本文介绍了一种新的防御策略GenFighter，该策略通过对训练分类分布进行学习和推理来增强对手的健壮性。GenFighter识别偏离分布的潜在恶意实例，将它们转换为与训练数据一致的语义等价实例，并采用集成技术实现统一和强大的响应。通过大量的实验，我们证明了GenFighter在攻击准确率和攻击成功率指标上都优于最先进的防御系统。此外，每次攻击都需要大量的查询，这使得攻击在真实场景中更具挑战性。消融研究表明，我们的方法集成了迁移学习、生成/进化过程和集成方法，提供了对NLP对手攻击的有效防御。



## **13. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

TransLinkGuard：保护Transformer模型，防止边缘部署中的模型窃取 cs.CR

arXiv admin note: text overlap with arXiv:2310.07152 by other authors

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11121v1) [paper-pdf](http://arxiv.org/pdf/2404.11121v1)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.

摘要: 专有的大型语言模型(LLM)已广泛应用于各种场景。此外，出于效率和隐私的原因，在边缘设备上部署LLM是一种趋势。然而，专有LLMS的边缘部署带来了新的安全挑战：边缘部署的模型暴露为用户可访问的白盒，使对手能够进行有效的模型窃取(MS)攻击。不幸的是，现有的防御机制未能提供有效的保护。具体地说，我们确定了现有方法无法同时满足的四个关键保护性质：(1)在物理复制模型后保持保护；(2)在请求级授权模型访问；(3)保护运行时逆向工程；(4)以可忽略的运行时开销实现高安全性。为了解决上述问题，我们提出了一种针对边缘设备上的模型窃取的即插即用模型保护方法TransLinkGuard。TransLinkGuard的核心部分是驻留在安全环境中的轻量级授权模块，例如TEE。授权模块可以基于其输入对每个请求进行新的授权。大量实验表明，TransLinkGuard实现了与黑盒安全保证相同的安全保护，而开销可以忽略不计。



## **14. ToDA: Target-oriented Diffusion Attacker against Recommendation System**

ToDA：针对推荐系统的目标导向扩散攻击者 cs.CR

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2401.12578v2) [paper-pdf](http://arxiv.org/pdf/2401.12578v2)

**Authors**: Xiaohao Liu, Zhulin Tao, Ting Jiang, He Chang, Yunshan Ma, Xianglin Huang, Xiang Wang

**Abstract**: Recommendation systems (RS) have become indispensable tools for web services to address information overload, thus enhancing user experiences and bolstering platforms' revenues. However, with their increasing ubiquity, security concerns have also emerged. As the public accessibility of RS, they are susceptible to specific malicious attacks where adversaries can manipulate user profiles, leading to biased recommendations. Recent research often integrates additional modules using generative models to craft these deceptive user profiles, ensuring them are imperceptible while causing the intended harm. Albeit their efficacy, these models face challenges of unstable training and the exploration-exploitation dilemma, which can lead to suboptimal results. In this paper, we pioneer to investigate the potential of diffusion models (DMs), for shilling attacks. Specifically, we propose a novel Target-oriented Diffusion Attack model (ToDA). It incorporates a pre-trained autoencoder that transforms user profiles into a high dimensional space, paired with a Latent Diffusion Attacker (LDA)-the core component of ToDA. LDA introduces noise into the profiles within this latent space, adeptly steering the approximation towards targeted items through cross-attention mechanisms. The global horizon, implemented by a bipartite graph, is involved in LDA and derived from the encoded user profile feature. This makes LDA possible to extend the generation outwards the on-processing user feature itself, and bridges the gap between diffused user features and target item features. Extensive experiments compared to several SOTA baselines demonstrate ToDA's effectiveness. Specific studies exploit the elaborative design of ToDA and underscore the potency of advanced generative models in such contexts.

摘要: 推荐系统(RS)已经成为Web服务解决信息过载的不可或缺的工具，从而增强了用户体验并增加了平台的收入。然而，随着它们越来越普遍，安全问题也出现了。由于RS的公共可访问性，它们容易受到特定的恶意攻击，攻击者可以操纵用户配置文件，导致有偏见的推荐。最近的研究经常使用生成性模型集成额外的模块来制作这些欺骗性的用户配置文件，确保它们在造成预期伤害的同时是不可察觉的。尽管这些模式很有效，但它们面临着训练不稳定和勘探-开采困境的挑战，这可能导致不太理想的结果。在本文中，我们率先研究了扩散模型(DM)对先令攻击的可能性。具体来说，我们提出了一种新的面向目标的扩散攻击模型(Toda)。它结合了一个预先训练的自动编码器，可以将用户配置文件转换到高维空间，并与Toda的核心组件潜在扩散攻击者(LDA)配对。LDA将噪声引入到这个潜在空间内的轮廓中，通过交叉注意机制熟练地将近似引导到目标项目。全局地平线由二部图实现，涉及LDA，并从编码的用户简档特征中派生出来。这使得LDA有可能将生成向外扩展到正在处理的用户功能本身，并弥合扩散的用户功能和目标项目功能之间的差距。与几个SOTA基线相比的广泛实验证明了Toda的有效性。具体的研究利用了户田的精心设计，并强调了高级生成模式在这种背景下的效力。



## **15. Design for Trust utilizing Rareness Reduction**

利用减少稀有性的信任设计 cs.CR

37th International Conference on VLSI Design, 2024

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2302.08984v2) [paper-pdf](http://arxiv.org/pdf/2302.08984v2)

**Authors**: Aruna Jayasena, Prabhat Mishra

**Abstract**: Increasing design complexity and reduced time-to-market have motivated manufacturers to outsource some parts of the System-on-Chip (SoC) design flow to third-party vendors. This provides an opportunity for attackers to introduce hardware Trojans by constructing stealthy triggers consisting of rare events (e.g., rare signals, states, and transitions). There are promising test generation-based hardware Trojan detection techniques that rely on the activation of rare events. In this paper, we investigate rareness reduction as a design-for-trust solution to make it harder for an adversary to hide Trojans (easier for Trojan detection). Specifically, we analyze different avenues to reduce the potential rare trigger cases, including design diversity and area optimization. While there is a good understanding of the relationship between area, power, energy, and performance, this research provides a better insight into the dependency between area and security. Our experimental evaluation demonstrates that area reduction leads to a reduction in rareness. It also reveals that reducing rareness leads to faster Trojan detection as well as improved coverage by Trojan detection methods.

摘要: 不断增加的设计复杂性和缩短的上市时间促使制造商将片上系统(SoC)设计流程的某些部分外包给第三方供应商。这为攻击者提供了通过构建由罕见事件(例如罕见信号、状态和转换)组成的隐形触发器来引入硬件特洛伊木马程序的机会。有一些很有前途的基于测试生成的硬件特洛伊木马检测技术，它们依赖于罕见事件的激活。在这篇文章中，我们研究了稀有性减少作为一种信任设计解决方案，使对手更难隐藏特洛伊木马(更容易检测木马)。具体地说，我们分析了减少潜在罕见触发情况的不同途径，包括设计多样性和面积优化。虽然对面积、功率、能量和性能之间的关系有了很好的理解，但这项研究提供了对面积和安全之间的依赖关系的更好的洞察。我们的实验评估表明，面积减少会导致稀有性的减少。它还揭示了减少稀有性会导致木马检测的速度更快，并提高了木马检测方法的覆盖率。



## **16. Fooling Contrastive Language-Image Pre-trained Models with CLIPMasterPrints**

使用CLIPMasterPrint愚弄对比图像预训练模型 cs.CV

This work was supported by a research grant (40575) from VILLUM  FONDEN

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2307.03798v3) [paper-pdf](http://arxiv.org/pdf/2307.03798v3)

**Authors**: Matthias Freiberger, Peter Kun, Christian Igel, Anders Sundnes Løvlie, Sebastian Risi

**Abstract**: Models leveraging both visual and textual data such as Contrastive Language-Image Pre-training (CLIP), are the backbone of many recent advances in artificial intelligence. In this work, we show that despite their versatility, such models are vulnerable to what we refer to as fooling master images. Fooling master images are capable of maximizing the confidence score of a CLIP model for a significant number of widely varying prompts, while being either unrecognizable or unrelated to the attacked prompts for humans. The existence of such images is problematic as it could be used by bad actors to maliciously interfere with CLIP-trained image retrieval models in production with comparably small effort as a single image can attack many different prompts. We demonstrate how fooling master images for CLIP (CLIPMasterPrints) can be mined using stochastic gradient descent, projected gradient descent, or blackbox optimization. Contrary to many common adversarial attacks, the blackbox optimization approach allows us to mine CLIPMasterPrints even when the weights of the model are not accessible. We investigate the properties of the mined images, and find that images trained on a small number of image captions generalize to a much larger number of semantically related captions. We evaluate possible mitigation strategies, where we increase the robustness of the model and introduce an approach to automatically detect CLIPMasterPrints to sanitize the input of vulnerable models. Finally, we find that vulnerability to CLIPMasterPrints is related to a modality gap in contrastive pre-trained multi-modal networks. Code available at https://github.com/matfrei/CLIPMasterPrints.

摘要: 利用视觉和文本数据的模型，如对比语言-图像预训练(CLIP)，是人工智能许多最新进展的支柱。在这项工作中，我们表明，尽管这些模型具有多功能性，但它们很容易受到我们所说的愚弄主图像的攻击。愚弄主图像能够针对大量差异很大的提示最大化剪辑模型的置信度分数，同时对人类来说要么无法识别，要么与被攻击的提示无关。这种图像的存在是有问题的，因为它可能被不良行为者用来恶意干扰生产中经过剪辑训练的图像检索模型，而工作量相对较小，因为一张图像可以攻击许多不同的提示。我们演示了如何使用随机梯度下降、投影梯度下降或黑盒优化来挖掘CLIP(CLIPMasterPrints)的愚弄主图像。与许多常见的对抗性攻击相反，黑盒优化方法允许我们在模型的权重不可访问的情况下挖掘CLIPMasterPrint。我们研究了挖掘出的图像的属性，发现在少量图像字幕上训练的图像概括为大量语义相关的字幕。我们评估了可能的缓解策略，其中我们增加了模型的健壮性，并引入了一种自动检测CLIPMasterPrints的方法来清理易受攻击的模型的输入。最后，我们发现CLIPMasterPrints的漏洞与对比预训练多通道网络中的通道缺口有关。代码可在https://github.com/matfrei/CLIPMasterPrints.上找到



## **17. Self-playing Adversarial Language Game Enhances LLM Reasoning**

自玩对抗语言游戏增强LLM推理 cs.CL

Preprint

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10642v1) [paper-pdf](http://arxiv.org/pdf/2404.10642v1)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate with respect to a target word only visible to the attacker. The attacker aims to induce the defender to utter the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Play in this Adversarial language Game (SPAG). With this goal, we let LLMs act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performance uniformly improves on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLM's reasoning ability. The code is at https://github.com/Linear95/SPAG.

摘要: 我们探索了在一个名为对抗性禁忌的两人对抗性语言游戏中，大语言模型(LLM)的自我发挥训练过程。在这个游戏中，攻击者和防御者就只有攻击者才能看到的目标单词进行交流。攻击者的目的是诱导防御者无意识地说出目标词，而防御者则试图从攻击者的话语中推断出目标词。要赢得这场比赛，双方都应该有足够的目标词知识和高级推理能力，以便在这种信息储备的对话中进行推理和表达。因此，我们好奇在这场对抗性语言游戏(SPAG)中，LLMS的推理能力能否通过自我游戏进一步增强。有了这个目标，我们让LLMS扮演攻击者的角色，并在广泛的目标词上扮演自己的防御者。通过对游戏结果的强化学习，我们观察到LLMS在广泛的推理基准上的性能一致提高。此外，迭代地采用这种自我发挥过程可以不断提升LLM的推理能力。代码在https://github.com/Linear95/SPAG.



## **18. The Critical Node Game**

关键节点游戏 math.OC

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2303.05961v2) [paper-pdf](http://arxiv.org/pdf/2303.05961v2)

**Authors**: Gabriele Dragotto, Amine Boukhtouta, Andrea Lodi, Mehdi Taobane

**Abstract**: In this work, we introduce a game-theoretic model that assesses the cyber-security risk of cloud networks and informs security experts on the optimal security strategies. Our approach combines game theory, combinatorial optimization, and cyber-security and aims to minimize the unexpected network disruptions caused by malicious cyber-attacks under uncertainty. Methodologically, we introduce the critical node game, a simultaneous and non-cooperative attacker-defender game where each player solves a combinatorial optimization problem parametrized in the variables of the other player. Each player simultaneously commits to a defensive (or attacking) strategy with limited knowledge about the choices of their adversary. We provide a realistic model for the critical node game and propose an algorithm to compute its stable solutions, i.e., its Nash equilibria. Practically, our approach enables security experts to assess the security posture of the cloud network and dynamically adapt the level of cyber-protection deployed on the network. We provide a detailed analysis of a real-world cloud network and demonstrate the efficacy of our approach through extensive computational tests.

摘要: 在这项工作中，我们引入了一个博弈论模型来评估云网络的网络安全风险，并向安全专家提供最优安全策略。我们的方法结合了博弈论、组合优化和网络安全，旨在将不确定情况下恶意网络攻击造成的意外网络中断降至最低。在方法上，我们引入了关键节点博弈，这是一个同时的、非合作的攻防博弈，其中每个参与者求解一个组合优化问题，该问题被参数化为另一个参与者的变量。每个球员都同时致力于防守(或进攻)策略，但对对手的选择知之甚少。我们为关键节点博弈提供了一个现实的模型，并提出了一个算法来计算它的稳定解，即它的纳什均衡。实际上，我们的方法使安全专家能够评估云网络的安全态势，并动态调整网络上部署的网络保护级别。我们对一个真实的云网络进行了详细的分析，并通过大量的计算测试证明了我们方法的有效性。



## **19. Minerva: A File-Based Ransomware Detector**

Minerva：基于文件的勒索软件检测器 cs.CR

14 pages

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2301.11050v2) [paper-pdf](http://arxiv.org/pdf/2301.11050v2)

**Authors**: Dorjan Hitaj, Giulio Pagnotta, Fabio De Gaspari, Lorenzo De Carli, Luigi V. Mancini

**Abstract**: Ransomware attacks have caused billions of dollars in damages in recent years, and are expected to cause billions more in the future. Consequently, significant effort has been devoted to ransomware detection and mitigation. Behavioral-based ransomware detection approaches have garnered considerable attention recently. These behavioral detectors typically rely on process-based behavioral profiles to identify malicious behaviors. However, with an increasing body of literature highlighting the vulnerability of such approaches to evasion attacks, a comprehensive solution to the ransomware problem remains elusive. This paper presents Minerva, a novel robust approach to ransomware detection. Minerva is engineered to be robust by design against evasion attacks, with architectural and feature selection choices informed by their resilience to adversarial manipulation. We conduct a comprehensive analysis of Minerva across a diverse spectrum of ransomware types, encompassing unseen ransomware as well as variants designed specifically to evade Minerva. Our evaluation showcases the ability of Minerva to accurately identify ransomware, generalize to unseen threats, and withstand evasion attacks. Furthermore, Minerva achieves remarkably low detection times, enabling the adoption of data loss prevention techniques with near-zero overhead.

摘要: 近年来，勒索软件攻击已经造成了数十亿美元的损失，预计未来还会造成数十亿美元的损失。因此，在勒索软件检测和缓解方面投入了大量努力。基于行为的勒索软件检测方法最近得到了相当大的关注。这些行为检测器通常依赖基于进程的行为配置文件来识别恶意行为。然而，随着越来越多的文献强调这种方法对逃避攻击的脆弱性，勒索软件问题的全面解决方案仍然难以找到。提出了一种新的稳健的勒索软件检测方法Minerva。Minerva的设计是针对躲避攻击而设计的，其架构和功能选择取决于其对对手操纵的弹性。我们对Minerva进行了全面的分析，包括各种勒索软件类型，包括看不见的勒索软件以及专门为规避Minerva而设计的变体。我们的评估展示了Minerva准确识别勒索软件、概括为看不见的威胁和抵御逃避攻击的能力。此外，Minerva实现了极低的检测时间，从而能够以几乎为零的开销采用数据丢失预防技术。



## **20. Adversarial Identity Injection for Semantic Face Image Synthesis**

用于语义人脸图像合成的对抗身份注入 cs.CV

Paper accepted at CVPR 2024 Biometrics Workshop

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10408v1) [paper-pdf](http://arxiv.org/pdf/2404.10408v1)

**Authors**: Giuseppe Tarollo, Tomaso Fontanini, Claudio Ferrari, Guido Borghi, Andrea Prati

**Abstract**: Nowadays, deep learning models have reached incredible performance in the task of image generation. Plenty of literature works address the task of face generation and editing, with human and automatic systems that struggle to distinguish what's real from generated. Whereas most systems reached excellent visual generation quality, they still face difficulties in preserving the identity of the starting input subject. Among all the explored techniques, Semantic Image Synthesis (SIS) methods, whose goal is to generate an image conditioned on a semantic segmentation mask, are the most promising, even though preserving the perceived identity of the input subject is not their main concern. Therefore, in this paper, we investigate the problem of identity preservation in face image generation and present an SIS architecture that exploits a cross-attention mechanism to merge identity, style, and semantic features to generate faces whose identities are as similar as possible to the input ones. Experimental results reveal that the proposed method is not only suitable for preserving the identity but is also effective in the face recognition adversarial attack, i.e. hiding a second identity in the generated faces.

摘要: 如今，深度学习模型在图像生成任务中已经达到了令人难以置信的性能。许多文学作品都涉及人脸生成和编辑的任务，人类和自动系统很难区分什么是真实的，什么是生成的。虽然大多数系统达到了极好的视觉生成质量，但它们在保留起始输入主体的身份方面仍然面临困难。在所有已探索的技术中，语义图像合成(SIS)方法是最有前途的，其目标是生成基于语义分割掩模的图像，尽管保持输入主体的感知身份并不是它们的主要考虑因素。因此，在本文中，我们研究了人脸图像生成中的身份保存问题，并提出了一种利用交叉注意机制合并身份、风格和语义特征来生成身份尽可能与输入人脸相似的人脸的SIS体系结构。实验结果表明，该方法不仅适用于保存身份，而且对人脸识别的敌意攻击也是有效的，即在生成的人脸中隐藏第二个身份。



## **21. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

通过错误纠正代码对人工智能生成的文本进行可证明鲁棒的多位水印 cs.CR

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2401.16820v2) [paper-pdf](http://arxiv.org/pdf/2401.16820v2)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.

摘要: 大型语言模型(LLM)因其生成类似人类语言的文本的非凡能力而被广泛使用。然而，它们可能被犯罪分子滥用来创造欺骗性内容，如假新闻和钓鱼电子邮件，这引发了伦理问题。水印是缓解LLMS误用的一项关键技术，它将水印(如比特串)嵌入到LLM生成的文本中。因此，这使得能够检测由LLM生成的文本以及将生成的文本跟踪到特定用户。现有水印技术的主要局限性是不能准确或高效地从文本中提取水印，特别是当水印是长比特串的时候。这一关键限制阻碍了它们在现实世界应用程序中的部署，例如，跟踪生成的文本到特定用户。为了解决这一问题，提出了一种新的基于文本纠错码的LLM文本水印方法。我们提供了强有力的理论分析，证明了在有界的敌意单词/令牌编辑(插入、删除和替换)下，我们的方法可以正确地提取水印，提供了可证明的健壮性保证。这一突破也被我们广泛的实验结果所证明。实验表明，在基准数据集上，我们的方法在准确率和稳健性方面都大大优于现有的基线。例如，当将长度为12的比特串嵌入到200个标记生成的文本中时，我们的方法获得了令人印象深刻的匹配率$98.4\$，超过了Yoo等人的性能。(最新基线)为85.6美元。在对200个单词的文本进行50个标记的复制粘贴攻击时，我们的方法保持了相当高的匹配率为90.8美元，而Yoo等人的匹配率是90.8美元。降至65美元以下。



## **22. Towards a Novel Perspective on Adversarial Examples Driven by Frequency**

以新的视角看待由频率驱动的对抗性例子 cs.LG

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10202v1) [paper-pdf](http://arxiv.org/pdf/2404.10202v1)

**Authors**: Zhun Zhang, Yi Zeng, Qihe Liu, Shijie Zhou

**Abstract**: Enhancing our understanding of adversarial examples is crucial for the secure application of machine learning models in real-world scenarios. A prevalent method for analyzing adversarial examples is through a frequency-based approach. However, existing research indicates that attacks designed to exploit low-frequency or high-frequency information can enhance attack performance, leading to an unclear relationship between adversarial perturbations and different frequency components. In this paper, we seek to demystify this relationship by exploring the characteristics of adversarial perturbations within the frequency domain. We employ wavelet packet decomposition for detailed frequency analysis of adversarial examples and conduct statistical examinations across various frequency bands. Intriguingly, our findings indicate that significant adversarial perturbations are present within the high-frequency components of low-frequency bands. Drawing on this insight, we propose a black-box adversarial attack algorithm based on combining different frequency bands. Experiments conducted on multiple datasets and models demonstrate that combining low-frequency bands and high-frequency components of low-frequency bands can significantly enhance attack efficiency. The average attack success rate reaches 99\%, surpassing attacks that utilize a single frequency segment. Additionally, we introduce the normalized disturbance visibility index as a solution to the limitations of $L_2$ norm in assessing continuous and discrete perturbations.

摘要: 提高我们对对抗性例子的理解对于机器学习模型在真实世界场景中的安全应用至关重要。分析对抗性例子的一种流行方法是通过基于频率的方法。然而，现有的研究表明，利用低频或高频信息设计的攻击可以提高攻击性能，导致敌方扰动与不同频率分量之间的关系不清楚。在本文中，我们试图通过探索对抗性扰动在频域内的特征来揭开这种关系的神秘面纱。我们使用小波包分解对对抗性样本进行详细的频率分析，并在不同的频段上进行统计检验。有趣的是，我们的发现表明，在低频段的高频分量中存在显著的对抗性扰动。在此基础上，提出了一种基于不同频段组合的黑盒对抗攻击算法。在多个数据集和模型上进行的实验表明，结合低频段和低频段的高频分量可以显著提高攻击效率。平均攻击成功率达到99%，超过了利用单个频段的攻击。此外，我们引入了归一化扰动能见度指数作为解决$L_2$范数在评估连续和离散扰动方面的局限性。



## **23. Ti-Patch: Tiled Physical Adversarial Patch for no-reference video quality metrics**

Ti-patch：用于无参考视频质量指标的切片物理对抗补丁 cs.CV

Accepted to WAIT AINL 2024

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09961v1) [paper-pdf](http://arxiv.org/pdf/2404.09961v1)

**Authors**: Victoria Leonenkova, Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Objective no-reference image- and video-quality metrics are crucial in many computer vision tasks. However, state-of-the-art no-reference metrics have become learning-based and are vulnerable to adversarial attacks. The vulnerability of quality metrics imposes restrictions on using such metrics in quality control systems and comparing objective algorithms. Also, using vulnerable metrics as a loss for deep learning model training can mislead training to worsen visual quality. Because of that, quality metrics testing for vulnerability is a task of current interest. This paper proposes a new method for testing quality metrics vulnerability in the physical space. To our knowledge, quality metrics were not previously tested for vulnerability to this attack; they were only tested in the pixel space. We applied a physical adversarial Ti-Patch (Tiled Patch) attack to quality metrics and did experiments both in pixel and physical space. We also performed experiments on the implementation of physical adversarial wallpaper. The proposed method can be used as additional quality metrics in vulnerability evaluation, complementing traditional subjective comparison and vulnerability tests in the pixel space. We made our code and adversarial videos available on GitHub: https://github.com/leonenkova/Ti-Patch.

摘要: 目的在许多计算机视觉任务中，无参考图像和视频质量度量是至关重要的。然而，最先进的无参考指标已经成为基于学习的指标，很容易受到对手的攻击。质量指标的脆弱性对在质量控制系统中使用此类指标和比较目标算法施加了限制。此外，使用易受攻击的指标作为深度学习模型训练的损失可能会误导训练，使视觉质量恶化。正因为如此，针对漏洞的质量度量测试是当前感兴趣的任务。提出了一种在物理空间中测试质量度量脆弱性的新方法。据我们所知，质量指标之前没有针对这种攻击的脆弱性进行过测试；它们只在像素空间进行了测试。我们将物理对抗性的TiPatch(平铺补丁)攻击应用到质量度量中，并在像素和物理空间上进行了实验。我们还对物理对抗墙纸的实现进行了实验。该方法可以作为脆弱性评估中的附加质量度量，补充了传统的主观比较法和像素空间脆弱性测试的不足。我们在gihub上提供了我们的代码和对抗性视频：https://github.com/leonenkova/Ti-Patch.



## **24. Larger-scale Nakamoto-style Blockchains Don't Necessarily Offer Better Security**

更大规模的中本式区块链不一定提供更好的安全性 cs.CR

IEEE Symposium on Security and Privacy (IEEE SP), 2024

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09895v1) [paper-pdf](http://arxiv.org/pdf/2404.09895v1)

**Authors**: Jannik Albrecht, Sebastien Andreina, Frederik Armknecht, Ghassan Karame, Giorgia Marson, Julian Willingmann

**Abstract**: Extensive research on Nakamoto-style consensus protocols has shown that network delays degrade the security of these protocols. Established results indicate that, perhaps surprisingly, maximal security is achieved when the network is as small as two nodes due to increased delays in larger networks. This contradicts the very foundation of blockchains, namely that decentralization improves security. In this paper, we take a closer look at how the network scale affects security of Nakamoto-style blockchains. We argue that a crucial aspect has been neglected in existing security models: the larger the network, the harder it is for an attacker to control a significant amount of power. To this end, we introduce a probabilistic corruption model to express the increasing difficulty for an attacker to corrupt resources in larger networks. Based on our model, we analyze the impact of the number of nodes on the (maximum) network delay and the fraction of adversarial power. In particular, we show that (1) increasing the number of nodes eventually violates security, but (2) relying on a small number of nodes does not provide decent security provisions either. We then validate our analysis by means of an empirical evaluation emulating hundreds of thousands of nodes in deployments such as Bitcoin, Monero, Cardano, and Ethereum Classic. Based on our empirical analysis, we concretely analyze the impact of various real-world parameters and configurations on the consistency bounds in existing deployments and on the adversarial power that can be tolerated while providing security. As far as we are aware, this is the first work that analytically and empirically explores the real-world tradeoffs achieved by current popular Nakamoto-style deployments.

摘要: 对Nakamoto类一致性协议的广泛研究表明，网络延迟降低了这些协议的安全性。已建立的结果表明，可能令人惊讶的是，当网络小到两个节点时，由于较大网络中的延迟增加，可以获得最大的安全性。这与区块链的根本基础相矛盾，即去中心化提高了安全性。在本文中，我们仔细研究了网络规模对Nakamoto式区块链安全性的影响。我们认为，现有安全模型忽略了一个关键方面：网络越大，攻击者就越难控制大量的电力。为此，我们引入了一个概率破坏模型来表示攻击者破坏大型网络中的资源的难度不断增加。基于我们的模型，我们分析了节点数量对网络最大时延和对抗能力分数的影响。特别是，我们证明了(1)增加节点数量最终会违反安全性，但(2)依赖于少量节点也不能提供像样的安全保障。然后，我们通过对比特币、Monero、Cardano和Etherum Classic等部署中的数十万个节点进行模拟的经验评估来验证我们的分析。在实证分析的基础上，我们具体分析了各种真实世界参数和配置对现有部署中一致性界限的影响，以及对提供安全时所能容忍的对抗能力的影响。据我们所知，这是第一个以分析和经验的方式探索当前流行的Nakamoto式部署所实现的现实世界权衡的工作。



## **25. Adversarial Nibbler: An Open Red-Teaming Method for Identifying Diverse Harms in Text-to-Image Generation**

对抗性Nibbler：一种用于识别文本到图像生成中各种伤害的开放式红团队方法 cs.CY

15 pages, 6 figures

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2403.12075v2) [paper-pdf](http://arxiv.org/pdf/2403.12075v2)

**Authors**: Jessica Quaye, Alicia Parrish, Oana Inel, Charvi Rastogi, Hannah Rose Kirk, Minsuk Kahng, Erin van Liemt, Max Bartolo, Jess Tsang, Justin White, Nathan Clement, Rafael Mosquera, Juan Ciro, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: With the rise of text-to-image (T2I) generative AI models reaching wide audiences, it is critical to evaluate model robustness against non-obvious attacks to mitigate the generation of offensive images. By focusing on ``implicitly adversarial'' prompts (those that trigger T2I models to generate unsafe images for non-obvious reasons), we isolate a set of difficult safety issues that human creativity is well-suited to uncover. To this end, we built the Adversarial Nibbler Challenge, a red-teaming methodology for crowdsourcing a diverse set of implicitly adversarial prompts. We have assembled a suite of state-of-the-art T2I models, employed a simple user interface to identify and annotate harms, and engaged diverse populations to capture long-tail safety issues that may be overlooked in standard testing. The challenge is run in consecutive rounds to enable a sustained discovery and analysis of safety pitfalls in T2I models.   In this paper, we present an in-depth account of our methodology, a systematic study of novel attack strategies and discussion of safety failures revealed by challenge participants. We also release a companion visualization tool for easy exploration and derivation of insights from the dataset. The first challenge round resulted in over 10k prompt-image pairs with machine annotations for safety. A subset of 1.5k samples contains rich human annotations of harm types and attack styles. We find that 14% of images that humans consider harmful are mislabeled as ``safe'' by machines. We have identified new attack strategies that highlight the complexity of ensuring T2I model robustness. Our findings emphasize the necessity of continual auditing and adaptation as new vulnerabilities emerge. We are confident that this work will enable proactive, iterative safety assessments and promote responsible development of T2I models.

摘要: 随着文本到图像(T2I)生成式人工智能模型的兴起，评估模型对非明显攻击的稳健性以减少攻击性图像的生成至关重要。通过关注“隐含的对抗性”提示(那些由于不明显的原因触发T2I模型生成不安全图像的提示)，我们隔离了一组人类创造力非常适合揭示的困难安全问题。为此，我们建立了对抗性Nibbler挑战赛，这是一种用于众包各种隐含对抗性提示的红团队方法论。我们组装了一套最先进的T2I模型，使用简单的用户界面来识别和注释危害，并让不同的人群参与捕获标准测试中可能被忽视的长尾安全问题。该挑战赛分连续几轮进行，以持续发现和分析T2I型号的安全隐患。在这篇文章中，我们介绍了我们的方法，对新的攻击策略进行了系统的研究，并讨论了挑战参与者揭示的安全故障。我们还发布了一个配套的可视化工具，用于轻松探索和从数据集获得洞察力。第一轮挑战赛产生了10000多个带有机器注释的提示图像对，以确保安全。1.5K样本的子集包含丰富的危害类型和攻击风格的人类注释。我们发现，在人类认为有害的图像中，14%被机器错误地贴上了“安全”的标签。我们已经确定了新的攻击策略，这些策略突出了确保T2I模型健壮性的复杂性。我们的发现强调了随着新漏洞的出现而持续审计和适应的必要性。我们相信，这项工作将使主动、迭代的安全评估成为可能，并促进负责任的T2I模型的开发。



## **26. VFLGAN: Vertical Federated Learning-based Generative Adversarial Network for Vertically Partitioned Data Publication**

VFLGAN：用于垂直分区数据发布的基于垂直联邦学习的生成对抗网络 cs.LG

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09722v1) [paper-pdf](http://arxiv.org/pdf/2404.09722v1)

**Authors**: Xun Yuan, Yang Yang, Prosanta Gope, Aryan Pasikhani, Biplab Sikdar

**Abstract**: In the current artificial intelligence (AI) era, the scale and quality of the dataset play a crucial role in training a high-quality AI model. However, good data is not a free lunch and is always hard to access due to privacy regulations like the General Data Protection Regulation (GDPR). A potential solution is to release a synthetic dataset with a similar distribution to that of the private dataset. Nevertheless, in some scenarios, it has been found that the attributes needed to train an AI model belong to different parties, and they cannot share the raw data for synthetic data publication due to privacy regulations. In PETS 2023, Xue et al. proposed the first generative adversary network-based model, VertiGAN, for vertically partitioned data publication. However, after thoroughly investigating, we found that VertiGAN is less effective in preserving the correlation among the attributes of different parties. This article proposes a Vertical Federated Learning-based Generative Adversarial Network, VFLGAN, for vertically partitioned data publication to address the above issues. Our experimental results show that compared with VertiGAN, VFLGAN significantly improves the quality of synthetic data. Taking the MNIST dataset as an example, the quality of the synthetic dataset generated by VFLGAN is 3.2 times better than that generated by VertiGAN w.r.t. the Fr\'echet Distance. We also designed a more efficient and effective Gaussian mechanism for the proposed VFLGAN to provide the synthetic dataset with a differential privacy guarantee. On the other hand, differential privacy only gives the upper bound of the worst-case privacy guarantee. This article also proposes a practical auditing scheme that applies membership inference attacks to estimate privacy leakage through the synthetic dataset.

摘要: 在当前的人工智能(AI)时代，数据集的规模和质量对培养高质量的AI模型起着至关重要的作用。然而，好的数据不是免费的午餐，由于隐私法规，如一般数据保护条例(GDPR)，总是很难访问的。一种可能的解决方案是发布具有类似于私有数据集的分布的合成数据集。然而，在一些场景中，人们发现训练AI模型所需的属性属于不同的各方，由于隐私法规的原因，他们无法共享用于合成数据发布的原始数据。在《宠物2023》中，薛等人。提出了第一个基于生成性对手网络的垂直分割数据发布模型Vertigan。然而，经过深入调查，我们发现Vertigan在保留不同各方属性之间的相关性方面效果较差。针对上述问题，本文提出了一种基于垂直联合学习的生成性对抗网络，用于垂直分割的数据发布。我们的实验结果表明，与Vertigan相比，VFLGAN显著提高了合成数据的质量。以MNIST数据集为例，VFLGAN生成的合成数据集的质量是Vertigan w.r.t.生成的合成数据集的3.2倍。本文还设计了一种更有效的高斯机制来为合成数据集提供差分隐私保证，而差分隐私只给出了最坏情况下隐私保证的上界。本文还提出了一种实用的审计方案，该方案应用成员关系推理攻击来估计合成数据集的隐私泄露。



## **27. A Survey of Neural Network Robustness Assessment in Image Recognition**

图像识别中神经网络鲁棒性评估综述 cs.CV

Corrected typos and grammatical errors in Section 5

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.08285v2) [paper-pdf](http://arxiv.org/pdf/2404.08285v2)

**Authors**: Jie Wang, Jun Ai, Minyan Lu, Haoran Su, Dan Yu, Yutao Zhang, Junda Zhu, Jingyu Liu

**Abstract**: In recent years, there has been significant attention given to the robustness assessment of neural networks. Robustness plays a critical role in ensuring reliable operation of artificial intelligence (AI) systems in complex and uncertain environments. Deep learning's robustness problem is particularly significant, highlighted by the discovery of adversarial attacks on image classification models. Researchers have dedicated efforts to evaluate robustness in diverse perturbation conditions for image recognition tasks. Robustness assessment encompasses two main techniques: robustness verification/ certification for deliberate adversarial attacks and robustness testing for random data corruptions. In this survey, we present a detailed examination of both adversarial robustness (AR) and corruption robustness (CR) in neural network assessment. Analyzing current research papers and standards, we provide an extensive overview of robustness assessment in image recognition. Three essential aspects are analyzed: concepts, metrics, and assessment methods. We investigate the perturbation metrics and range representations used to measure the degree of perturbations on images, as well as the robustness metrics specifically for the robustness conditions of classification models. The strengths and limitations of the existing methods are also discussed, and some potential directions for future research are provided.

摘要: 近年来，神经网络的稳健性评估受到了极大的关注。稳健性对于确保人工智能系统在复杂和不确定环境中的可靠运行起着至关重要的作用。深度学习的稳健性问题尤其显著，突出表现在发现了对图像分类模型的敌意攻击。研究人员致力于评估图像识别任务在不同扰动条件下的稳健性。健壮性评估包括两个主要技术：针对蓄意敌对攻击的健壮性验证/认证和针对随机数据损坏的健壮性测试。在这项调查中，我们提出了在神经网络评估中的对抗稳健性(AR)和腐败稳健性(CR)的详细检查。通过分析现有的研究文献和标准，我们对图像识别中的稳健性评估进行了广泛的综述。分析了三个基本方面：概念、度量和评估方法。我们研究了用于度量图像扰动程度的扰动度量和范围表示，以及专门针对分类模型的稳健性条件的稳健性度量。文中还讨论了现有方法的优点和局限性，并对未来的研究方向进行了展望。



## **28. Less is More: Understanding Word-level Textual Adversarial Attack via n-gram Frequency Descend**

少即是多：通过n-gram频率分类了解单词级文本对抗攻击 cs.CL

To be published in: 2024 IEEE Conference on Artificial Intelligence  (CAI 2024)

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2302.02568v4) [paper-pdf](http://arxiv.org/pdf/2302.02568v4)

**Authors**: Ning Lu, Shengcai Liu, Zhirui Zhang, Qi Wang, Haifeng Liu, Ke Tang

**Abstract**: Word-level textual adversarial attacks have demonstrated notable efficacy in misleading Natural Language Processing (NLP) models. Despite their success, the underlying reasons for their effectiveness and the fundamental characteristics of adversarial examples (AEs) remain obscure. This work aims to interpret word-level attacks by examining their $n$-gram frequency patterns. Our comprehensive experiments reveal that in approximately 90\% of cases, word-level attacks lead to the generation of examples where the frequency of $n$-grams decreases, a tendency we term as the $n$-gram Frequency Descend ($n$-FD). This finding suggests a straightforward strategy to enhance model robustness: training models using examples with $n$-FD. To examine the feasibility of this strategy, we employed the $n$-gram frequency information, as an alternative to conventional loss gradients, to generate perturbed examples in adversarial training. The experiment results indicate that the frequency-based approach performs comparably with the gradient-based approach in improving model robustness. Our research offers a novel and more intuitive perspective for understanding word-level textual adversarial attacks and proposes a new direction to improve model robustness.

摘要: 词级文本敌意攻击在误导自然语言处理(NLP)模型方面显示出显著的效果。尽管它们取得了成功，但其有效性的根本原因和对抗性例子的基本特征仍然不清楚。这项工作旨在通过检查其$n$-gram频率模式来解释词级攻击。我们的综合实验表明，在大约90%的情况下，词级攻击导致$n$-gram的频率下降，这种趋势我们称之为$n$-gram频率下降($n$-fd)。这一发现提出了一种增强模型稳健性的简单策略：使用具有$n$-fd的示例来训练模型。为了检验这一策略的可行性，我们使用了$n$-gram频率信息作为传统损失梯度的替代，以生成对抗性训练中的扰动示例。实验结果表明，基于频率的方法在提高模型稳健性方面与基于梯度的方法具有相当的效果。我们的研究为理解词级文本对抗攻击提供了一个新的、更直观的视角，并为提高模型的稳健性提出了新的方向。



## **29. Black-box Adversarial Transferability: An Empirical Study in Cybersecurity Perspective**

黑匣子对抗性可转让性：网络安全视角的实证研究 cs.CR

Submitted to Computer & Security (Elsevier)

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.10796v1) [paper-pdf](http://arxiv.org/pdf/2404.10796v1)

**Authors**: Khushnaseeb Roshan, Aasim Zafar

**Abstract**: The rapid advancement of artificial intelligence within the realm of cybersecurity raises significant security concerns. The vulnerability of deep learning models in adversarial attacks is one of the major issues. In adversarial machine learning, malicious users try to fool the deep learning model by inserting adversarial perturbation inputs into the model during its training or testing phase. Subsequently, it reduces the model confidence score and results in incorrect classifications. The novel key contribution of the research is to empirically test the black-box adversarial transferability phenomena in cyber attack detection systems. It indicates that the adversarial perturbation input generated through the surrogate model has a similar impact on the target model in producing the incorrect classification. To empirically validate this phenomenon, surrogate and target models are used. The adversarial perturbation inputs are generated based on the surrogate-model for which the hacker has complete information. Based on these adversarial perturbation inputs, both surrogate and target models are evaluated during the inference phase. We have done extensive experimentation over the CICDDoS-2019 dataset, and the results are classified in terms of various performance metrics like accuracy, precision, recall, and f1-score. The findings indicate that any deep learning model is highly susceptible to adversarial attacks, even if the attacker does not have access to the internal details of the target model. The results also indicate that white-box adversarial attacks have a severe impact compared to black-box adversarial attacks. There is a need to investigate and explore adversarial defence techniques to increase the robustness of the deep learning models against adversarial attacks.

摘要: 人工智能在网络安全领域的快速发展引发了重大的安全担忧。深度学习模型在对抗性攻击中的脆弱性是主要问题之一。在对抗性机器学习中，恶意用户试图通过在模型的训练或测试阶段向模型中插入对抗性扰动输入来欺骗深度学习模型。随后，它降低了模型的置信度分数，并导致错误的分类。该研究的新的关键贡献是对网络攻击检测系统中的黑箱对抗可转移性现象进行了经验测试。这表明，通过代理模型生成的对抗性扰动输入在产生错误分类方面对目标模型具有类似的影响。为了对这一现象进行实证验证，我们使用了代理模型和目标模型。敌意扰动输入是基于黑客具有完整信息的代理模型生成的。基于这些对抗性扰动输入，在推理阶段对代理模型和目标模型进行评估。我们在CICDDoS-2019数据集上进行了广泛的实验，并根据准确率、精确度、召回率和F1得分等各种性能指标对结果进行了分类。研究结果表明，任何深度学习模型都非常容易受到对抗性攻击，即使攻击者无法访问目标模型的内部详细信息。研究结果还表明，与黑盒对抗性攻击相比，白盒对抗性攻击具有严重的影响。需要研究和探索对抗性防御技术，以提高深度学习模型对对抗性攻击的稳健性。



## **30. SpamDam: Towards Privacy-Preserving and Adversary-Resistant SMS Spam Detection**

SpamDam：迈向保护隐私和抗对手的短信垃圾邮件检测 cs.CR

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09481v1) [paper-pdf](http://arxiv.org/pdf/2404.09481v1)

**Authors**: Yekai Li, Rufan Zhang, Wenxin Rong, Xianghang Mi

**Abstract**: In this study, we introduce SpamDam, a SMS spam detection framework designed to overcome key challenges in detecting and understanding SMS spam, such as the lack of public SMS spam datasets, increasing privacy concerns of collecting SMS data, and the need for adversary-resistant detection models. SpamDam comprises four innovative modules: an SMS spam radar that identifies spam messages from online social networks(OSNs); an SMS spam inspector for statistical analysis; SMS spam detectors(SSDs) that enable both central training and federated learning; and an SSD analyzer that evaluates model resistance against adversaries in realistic scenarios. Leveraging SpamDam, we have compiled over 76K SMS spam messages from Twitter and Weibo between 2018 and 2023, forming the largest dataset of its kind. This dataset has enabled new insights into recent spam campaigns and the training of high-performing binary and multi-label classifiers for spam detection. Furthermore, effectiveness of federated learning has been well demonstrated to enable privacy-preserving SMS spam detection. Additionally, we have rigorously tested the adversarial robustness of SMS spam detection models, introducing the novel reverse backdoor attack, which has shown effectiveness and stealthiness in practical tests.

摘要: 在这项研究中，我们介绍了SpamDam，一个短信垃圾邮件检测框架，旨在克服在检测和理解短信垃圾邮件方面的关键挑战，例如缺乏公开的短信垃圾数据集，收集短信数据时增加的隐私问题，以及需要抵抗对手的检测模型。SpamDam包括四个创新模块：识别来自在线社交网络(OSN)的垃圾邮件的短信垃圾邮件雷达；用于统计分析的短信垃圾邮件检查器；启用中央培训和联合学习的短信垃圾邮件检测器(SSD)；以及评估现实场景中对对手的模型抵抗力的SSD分析器。2018年至2023年，我们利用SpamDam收集了超过7.6万条推特和微博的垃圾短信，形成了最大的垃圾短信数据集。这一数据集使人们能够对最近的垃圾邮件活动有新的见解，并培训用于垃圾邮件检测的高性能二进制和多标签分类器。此外，联合学习的有效性已经被很好地证明能够实现隐私保护的短信垃圾检测。此外，我们严格测试了短信垃圾邮件检测模型的攻击健壮性，引入了新颖的反向后门攻击，在实际测试中显示了有效性和隐蔽性。



## **31. Crooked indifferentiability of the Feistel Construction**

费斯特尔结构的扭曲不可区分性 cs.CR

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09450v1) [paper-pdf](http://arxiv.org/pdf/2404.09450v1)

**Authors**: Alexander Russell, Qiang Tang, Jiadong Zhu

**Abstract**: The Feistel construction is a fundamental technique for building pseudorandom permutations and block ciphers. This paper shows that a simple adaptation of the construction is resistant, even to algorithm substitution attacks -- that is, adversarial subversion -- of the component round functions. Specifically, we establish that a Feistel-based construction with more than $2000n/\log(1/\epsilon)$ rounds can transform a subverted random function -- which disagrees with the original one at a small fraction (denoted by $\epsilon$) of inputs -- into an object that is \emph{crooked-indifferentiable} from a random permutation, even if the adversary is aware of all the randomness used in the transformation. We also provide a lower bound showing that the construction cannot use fewer than $2n/\log(1/\epsilon)$ rounds to achieve crooked-indifferentiable security.

摘要: Feistel构造是构建伪随机排列和分组密码的基本技术。本文表明，结构的简单调整即使能抵抗组件轮函数的算法替代攻击（即对抗性颠覆）也是如此。具体来说，我们确定基于费斯特的建筑价值超过2000美元n/\log（1/\）$ rounds可以转换一个颠覆的随机函数--它与原始函数有一小部分不一致（由$\$表示）的输入--进入来自随机排列的\{crooked-indexable}的对象，即使对手知道转换中使用的所有随机性。我们还提供了一个下限，表明该构造不能使用少于2n/\log（1/\）$轮来实现弯曲不可区分的安全性。



## **32. Watermark-embedded Adversarial Examples for Copyright Protection against Diffusion Models**

针对扩散模型的版权保护嵌入水印的对抗示例 cs.CV

**SubmitDate**: 2024-04-15    [abs](http://arxiv.org/abs/2404.09401v1) [paper-pdf](http://arxiv.org/pdf/2404.09401v1)

**Authors**: Peifei Zhu, Tsubasa Takahashi, Hirokatsu Kataoka

**Abstract**: Diffusion Models (DMs) have shown remarkable capabilities in various image-generation tasks. However, there are growing concerns that DMs could be used to imitate unauthorized creations and thus raise copyright issues. To address this issue, we propose a novel framework that embeds personal watermarks in the generation of adversarial examples. Such examples can force DMs to generate images with visible watermarks and prevent DMs from imitating unauthorized images. We construct a generator based on conditional adversarial networks and design three losses (adversarial loss, GAN loss, and perturbation loss) to generate adversarial examples that have subtle perturbation but can effectively attack DMs to prevent copyright violations. Training a generator for a personal watermark by our method only requires 5-10 samples within 2-3 minutes, and once the generator is trained, it can generate adversarial examples with that watermark significantly fast (0.2s per image). We conduct extensive experiments in various conditional image-generation scenarios. Compared to existing methods that generate images with chaotic textures, our method adds visible watermarks on the generated images, which is a more straightforward way to indicate copyright violations. We also observe that our adversarial examples exhibit good transferability across unknown generative models. Therefore, this work provides a simple yet powerful way to protect copyright from DM-based imitation.

摘要: 扩散模型(DM)在各种图像生成任务中表现出了卓越的能力。然而，人们越来越担心，DM可能被用来模仿未经授权的创作，从而引发版权问题。为了解决这个问题，我们提出了一种新的框架，在对抗性例子的生成中嵌入个人水印。这样的例子可以迫使DM生成带有可见水印的图像，并防止DM模仿未经授权的图像。我们构造了一个基于条件对抗网络的生成器，并设计了三种损失(对抗损失、GAN损失和扰动损失)来生成具有微妙扰动但可以有效攻击DM以防止版权侵犯的对抗实例。在2-3分钟内，我们的方法只需要5-10个样本就可以训练生成个人水印的生成器，并且一旦训练生成器，它可以显著地快速地生成带有该水印的对抗性示例(每幅图像0.2s)。我们在各种有条件的图像生成场景中进行了广泛的实验。与现有的带有混沌纹理的图像生成方法相比，我们的方法在生成的图像上添加了可见的水印，这是一种更直接的方式来指示侵犯版权的行为。我们还观察到，我们的对抗性例子显示出良好的跨未知生成模型的可转移性。因此，这部作品提供了一种简单而强大的方式来保护版权免受基于DM的模仿。



## **33. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies**

通过比例定律和人际关系研究的对抗稳健性限制 cs.LG

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2404.09349v1) [paper-pdf](http://arxiv.org/pdf/2404.09349v1)

**Authors**: Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, Bhavya Kailkhura

**Abstract**: This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. Taking CIFAR10 as an example, SOTA clean accuracy is about $100$%, but SOTA robustness to $\ell_{\infty}$-norm bounded perturbations barely exceeds $70$%. To understand this gap, we analyze how model size, dataset size, and synthetic data quality affect robustness by developing the first scaling laws for adversarial training. Our scaling laws reveal inefficiencies in prior art and provide actionable feedback to advance the field. For instance, we discovered that SOTA methods diverge notably from compute-optimal setups, using excess compute for their level of robustness. Leveraging a compute-efficient setup, we surpass the prior SOTA with $20$% ($70$%) fewer training (inference) FLOPs. We trained various compute-efficient models, with our best achieving $74$% AutoAttack accuracy ($+3$% gain). However, our scaling laws also predict robustness slowly grows then plateaus at $90$%: dwarfing our new SOTA by scaling is impractical, and perfect robustness is impossible. To better understand this predicted limit, we carry out a small-scale human evaluation on the AutoAttack data that fools our top-performing model. Concerningly, we estimate that human performance also plateaus near $90$%, which we show to be attributable to $\ell_{\infty}$-constrained attacks' generation of invalid images not consistent with their original labels. Having characterized limiting roadblocks, we outline promising paths for future research.

摘要: 本文回顾了一个简单、研究已久但仍未解决的问题，即使图像分类器对不可察觉的扰动具有健壮性。以CIFAR10为例，SOTA的清洁精度约为$100$%，但对$\ell_{inty}$-范数有界摄动的鲁棒性仅略高于$70$%。为了理解这一差距，我们分析了模型大小、数据集大小和合成数据质量如何通过开发用于对抗性训练的第一个缩放规则来影响稳健性。我们的比例法则揭示了现有技术中的低效，并提供了可操作的反馈来推动该领域的发展。例如，我们发现SOTA方法与计算最优设置明显不同，使用过量计算作为其健壮性级别。利用高效计算的设置，我们比以前的SOTA少了20美元%(70美元%)的培训(推理)失败。我们训练了各种计算效率高的模型，最大限度地达到了$74$%的AutoAttack精度($+3$%的收益)。然而，我们的定标法则也预测稳健性在90美元时缓慢增长然后停滞不前：通过定标来使我们的新SOTA相形见绌是不切实际的，而且完美的稳健性是不可能的。为了更好地理解这一预测极限，我们对AutoAttack数据进行了小规模的人工评估，该评估愚弄了我们的最佳模型。令人担忧的是，我们估计人类的性能也停滞不前近90$%，我们表明这归因于$受限攻击生成的无效图像与其原始标签不一致。在描述了限制障碍的特征之后，我们概述了未来研究的有希望的道路。



## **34. FaceCat: Enhancing Face Recognition Security with a Unified Generative Model Framework**

FaceCat：通过统一的生成模型框架增强面部识别安全性 cs.CV

Under review

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2404.09193v1) [paper-pdf](http://arxiv.org/pdf/2404.09193v1)

**Authors**: Jiawei Chen, Xiao Yang, Yinpeng Dong, Hang Su, Jianteng Peng, Zhaoxia Yin

**Abstract**: Face anti-spoofing (FAS) and adversarial detection (FAD) have been regarded as critical technologies to ensure the safety of face recognition systems. As a consequence of their limited practicality and generalization, some existing methods aim to devise a framework capable of concurrently detecting both threats to address the challenge. Nevertheless, these methods still encounter challenges of insufficient generalization and suboptimal robustness, potentially owing to the inherent drawback of discriminative models. Motivated by the rich structural and detailed features of face generative models, we propose FaceCat which utilizes the face generative model as a pre-trained model to improve the performance of FAS and FAD. Specifically, FaceCat elaborately designs a hierarchical fusion mechanism to capture rich face semantic features of the generative model. These features then serve as a robust foundation for a lightweight head, designed to execute FAS and FAD tasks simultaneously. As relying solely on single-modality data often leads to suboptimal performance, we further propose a novel text-guided multi-modal alignment strategy that utilizes text prompts to enrich feature representation, thereby enhancing performance. For fair evaluations, we build a comprehensive protocol with a wide range of 28 attack types to benchmark the performance. Extensive experiments validate the effectiveness of FaceCat generalizes significantly better and obtains excellent robustness against input transformations.

摘要: 人脸反欺骗(FAS)和对抗检测(FAD)被认为是确保人脸识别系统安全的关键技术。由于实用性和普遍性有限，一些现有方法旨在设计一个能够同时检测这两种威胁的框架，以应对这一挑战。然而，这些方法仍然面临着泛化不足和鲁棒性不佳的挑战，这可能是由于判别模型的固有缺陷。考虑到人脸生成模型具有丰富的结构和细节特征，本文提出了一种基于人脸生成模型的FaceCat算法，该算法利用人脸生成模型作为预训练模型来提高Fas和FAD的性能。具体地说，FaceCat精心设计了一种分层融合机制来捕捉生成模型丰富的人脸语义特征。这些功能为轻量级头部奠定了坚实的基础，旨在同时执行FAS和FAD任务。由于单纯依赖单一通道数据往往导致性能不佳，我们进一步提出了一种文本引导的多通道对齐策略，该策略利用文本提示来丰富特征表示，从而提高了性能。为了公平评估，我们构建了一个包含28种攻击类型的全面协议来对性能进行基准测试。大量实验验证了FaceCat算法的有效性，其泛化能力显着提高，对输入变换具有良好的鲁棒性。



## **35. Annealing Self-Distillation Rectification Improves Adversarial Training**

自我蒸馏修正改进对抗训练 cs.LG

Accepted to ICLR 2024

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2305.12118v2) [paper-pdf](http://arxiv.org/pdf/2305.12118v2)

**Authors**: Yu-Yu Wu, Hung-Jui Wang, Shang-Tse Chen

**Abstract**: In standard adversarial training, models are optimized to fit one-hot labels within allowable adversarial perturbation budgets. However, the ignorance of underlying distribution shifts brought by perturbations causes the problem of robust overfitting. To address this issue and enhance adversarial robustness, we analyze the characteristics of robust models and identify that robust models tend to produce smoother and well-calibrated outputs. Based on the observation, we propose a simple yet effective method, Annealing Self-Distillation Rectification (ADR), which generates soft labels as a better guidance mechanism that accurately reflects the distribution shift under attack during adversarial training. By utilizing ADR, we can obtain rectified distributions that significantly improve model robustness without the need for pre-trained models or extensive extra computation. Moreover, our method facilitates seamless plug-and-play integration with other adversarial training techniques by replacing the hard labels in their objectives. We demonstrate the efficacy of ADR through extensive experiments and strong performances across datasets.

摘要: 在标准的对抗性训练中，模型经过优化，以适应允许的对抗性扰动预算内的单一热门标签。然而，对扰动带来的潜在分布漂移的忽视导致了稳健过拟合的问题。为了解决这一问题并增强对手的稳健性，我们分析了稳健模型的特征，并发现稳健模型往往会产生更平滑和校准良好的输出。在此基础上，我们提出了一种简单而有效的方法--退火法自蒸馏纠错(ADR)，该方法生成软标签作为一种更好的指导机制，准确地反映了对抗训练中攻击下的分布变化。通过利用ADR，我们可以获得显著提高模型稳健性的校正分布，而不需要预先训练的模型或大量的额外计算。此外，我们的方法通过替换目标中的硬标签，促进了与其他对抗性训练技术的无缝即插即用集成。我们通过广泛的实验和在数据集上的强劲表现证明了ADR的有效性。



## **36. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

IRAD：隐式表示驱动的图像恢复对抗对抗攻击 cs.CV

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2310.11890v3) [paper-pdf](http://arxiv.org/pdf/2310.11890v3)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.

摘要: 我们引入了一种新的方法来对抗敌意攻击，即图像重采样。图像重采样将离散图像转换为新图像，模拟由几何变换指定的场景重新捕获或重新渲染的过程。我们的想法背后的基本原理是，图像重采样可以在保留基本语义信息的同时减轻对抗性扰动的影响，从而在防御对抗性攻击方面具有固有的优势。为了验证这一概念，我们提出了一种利用图像重采样来防御敌意攻击的综合研究。我们已经开发了使用内插策略和协调移动量的基本重采样方法。我们的分析表明，这些基本方法可以部分缓解对抗性攻击。然而，它们也有明显的局限性：清晰图像的准确性显著下降，而对抗性例子的准确性提高并不显著。我们提出了隐式表示驱动的图像重采样(IRAD)来克服这些限制。首先，我们构造了一个隐式连续表示，它使我们能够表示连续坐标空间内的任何输入图像。其次，我们介绍了SampleNet，它可以根据不同的输入自动生成像素方向的移位以进行重采样。此外，我们可以将我们的方法扩展到最先进的基于扩散的方法，以更少的时间步骤加速它，同时保持其防御能力。大量实验表明，该方法在保持对清晰图像的较高准确率的同时，显著增强了不同深度模型对各种攻击的抵抗能力。



## **37. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

22 pages, 6 figures

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.09005v1) [paper-pdf](http://arxiv.org/pdf/2404.09005v1)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别是，我们的工作是安全的，可以抵抗对Jia等人最近的工作的两次攻击。[2021]并将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **38. Stability and Generalization in Free Adversarial Training**

自由对抗训练的稳定性和概括性 cs.LG

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.08980v1) [paper-pdf](http://arxiv.org/pdf/2404.08980v1)

**Authors**: Xiwei Cheng, Kexin Fu, Farzan Farnia

**Abstract**: While adversarial training methods have resulted in significant improvements in the deep neural nets' robustness against norm-bounded adversarial perturbations, their generalization performance from training samples to test data has been shown to be considerably worse than standard empirical risk minimization methods. Several recent studies seek to connect the generalization behavior of adversarially trained classifiers to various gradient-based min-max optimization algorithms used for their training. In this work, we study the generalization performance of adversarial training methods using the algorithmic stability framework. Specifically, our goal is to compare the generalization performance of the vanilla adversarial training scheme fully optimizing the perturbations at every iteration vs. the free adversarial training simultaneously optimizing the norm-bounded perturbations and classifier parameters. Our proven generalization bounds indicate that the free adversarial training method could enjoy a lower generalization gap between training and test samples due to the simultaneous nature of its min-max optimization algorithm. We perform several numerical experiments to evaluate the generalization performance of vanilla, fast, and free adversarial training methods. Our empirical findings also show the improved generalization performance of the free adversarial training method and further demonstrate that the better generalization result could translate to greater robustness against black-box attack schemes. The code is available at https://github.com/Xiwei-Cheng/Stability_FreeAT.

摘要: 虽然对抗性训练方法显著提高了深度神经网络对范数有界对抗性扰动的稳健性，但从训练样本到测试数据的泛化性能已被证明比标准的经验风险最小化方法要差得多。最近的一些研究试图将对抗性训练的分类器的泛化行为与用于其训练的各种基于梯度的最小-最大优化算法联系起来。在这项工作中，我们使用算法稳定性框架来研究对抗性训练方法的泛化性能。具体地说，我们的目标是比较香草对抗训练方案和自由对抗训练方案的泛化性能，前者在每次迭代时完全优化扰动，后者同时优化范数有界扰动和分类器参数。我们证明的泛化界表明，由于其最小-最大优化算法的同时性质，自由对抗性训练方法可以享受较小的训练样本和测试样本之间的泛化差距。我们进行了一些数值实验来评估普通的、快速的和自由的对抗性训练方法的泛化性能。我们的实验结果还表明，自由对抗性训练方法的泛化性能有所改善，并进一步证明了更好的泛化结果可以转化为对黑盒攻击方案更好的鲁棒性。代码可在https://github.com/Xiwei-Cheng/Stability_FreeAT.上获得



## **39. Multimodal Attack Detection for Action Recognition Models**

动作识别模型的多模式攻击检测 cs.CR

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.10790v1) [paper-pdf](http://arxiv.org/pdf/2404.10790v1)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Adversarial machine learning attacks on video action recognition models is a growing research area and many effective attacks were introduced in recent years. These attacks show that action recognition models can be breached in many ways. Hence using these models in practice raises significant security concerns. However, there are very few works which focus on defending against or detecting attacks. In this work, we propose a novel universal detection method which is compatible with any action recognition model. In our extensive experiments, we show that our method consistently detects various attacks against different target models with high true positive rates while satisfying very low false positive rates. Tested against four state-of-the-art attacks targeting four action recognition models, the proposed detector achieves an average AUC of 0.911 over 16 test cases while the best performance achieved by the existing detectors is 0.645 average AUC. This 41.2% improvement is enabled by the robustness of the proposed detector to varying attack methods and target models. The lowest AUC achieved by our detector across the 16 test cases is 0.837 while the competing detector's performance drops as low as 0.211. We also show that the proposed detector is robust to varying attack strengths. In addition, we analyze our method's real-time performance with different hardware setups to demonstrate its potential as a practical defense mechanism.

摘要: 针对视频动作识别模型的对抗性机器学习攻击是一个不断发展的研究领域，近年来出现了许多有效的攻击方法。这些攻击表明，动作识别模型可以在许多方面被攻破。因此，在实践中使用这些模型会引起重大的安全问题。然而，针对攻击防御或检测的研究成果却很少。在这项工作中，我们提出了一种新的通用检测方法，该方法兼容任何动作识别模型。在我们的大量实验中，我们的方法一致地检测到针对不同目标模型的各种攻击，具有很高的真阳性率，同时满足很低的假阳性率。在针对4个动作识别模型的4种最新攻击下，提出的检测器在16个测试用例上的平均AUC为0.911，而现有检测器的最佳性能为0.645平均AUC.这41.2%的改进得益于所提出的检测器对不同攻击方法和目标模型的稳健性。我们的检测器在16个测试用例中实现的最低AUC值为0.837，而竞争对手的检测器的性能下降到0.211。我们还证明了所提出的检测器对不同的攻击强度具有很强的鲁棒性。此外，我们分析了我们的方法在不同硬件设置下的实时性能，以展示其作为一种实用的防御机制的潜力。



## **40. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

MixedNUTS：通过非线性混合分类器实现免训练的准确性-稳健性平衡 cs.LG

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2402.02263v3) [paper-pdf](http://arxiv.org/pdf/2402.02263v3)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.

摘要: 对抗性的稳健性往往是以降低精度为代价的，这阻碍了稳健分类模型的实际应用。基于培训的更好权衡的解决方案受到与已经培训的高性能大型模型不兼容的限制，因此有必要探索无需培训的整体方法。我们观察到，稳健模型在正确预测中的信心比基于干净和敌对数据的不正确预测更有信心，我们推测，放大这种“良性置信度属性”可以在整体设置中调和准确性和稳健性。为了实现这一点，我们提出了一种无需训练的方法“MixedNUTS”，其中稳健分类器和标准非稳健分类器的输出逻辑通过只有三个参数的非线性变换来处理，并通过有效的算法进行优化。MixedNUTS然后将转换后的Logit转换为概率，并将它们混合为整体输出。在CIFAR-10、CIFAR-100和ImageNet数据集上，自定义强自适应攻击的实验结果表明，MixedNUTS的精确度和接近SOTA的稳健性都得到了极大的提高--它将CIFAR-100的干净精确度提高了7.86个点，而健壮精确度仅牺牲了0.87个点。



## **41. Adversarial Patterns: Building Robust Android Malware Classifiers**

对抗模式：构建稳健的Android恶意软件分类器 cs.CR

survey

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2203.02121v2) [paper-pdf](http://arxiv.org/pdf/2203.02121v2)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstract**: Machine learning models are increasingly being adopted across various fields, such as medicine, business, autonomous vehicles, and cybersecurity, to analyze vast amounts of data, detect patterns, and make predictions or recommendations. In the field of cybersecurity, these models have made significant improvements in malware detection. However, despite their ability to understand complex patterns from unstructured data, these models are susceptible to adversarial attacks that perform slight modifications in malware samples, leading to misclassification from malignant to benign. Numerous defense approaches have been proposed to either detect such adversarial attacks or improve model robustness. These approaches have resulted in a multitude of attack and defense techniques and the emergence of a field known as `adversarial machine learning.' In this survey paper, we provide a comprehensive review of adversarial machine learning in the context of Android malware classifiers. Android is the most widely used operating system globally and is an easy target for malicious agents. The paper first presents an extensive background on Android malware classifiers, followed by an examination of the latest advancements in adversarial attacks and defenses. Finally, the paper provides guidelines for designing robust malware classifiers and outlines research directions for the future.

摘要: 机器学习模型正越来越多地被应用于医疗、商业、自动驾驶汽车和网络安全等各个领域，以分析海量数据、发现模式并做出预测或建议。在网络安全领域，这些模型在恶意软件检测方面取得了重大改进。然而，尽管这些模型能够从非结构化数据中理解复杂的模式，但它们容易受到对手攻击，这些攻击对恶意软件样本进行轻微修改，导致从恶性到良性的错误分类。已经提出了许多防御方法来检测这种对抗性攻击或提高模型的稳健性。这些方法导致了大量的攻击和防御技术，并出现了一个被称为“对抗性机器学习”的领域。在这篇调查论文中，我们对Android恶意软件分类器背景下的对抗性机器学习进行了全面的回顾。Android是全球使用最广泛的操作系统，很容易成为恶意代理的攻击目标。本文首先介绍了Android恶意软件分类器的广泛背景，然后研究了对抗性攻击和防御的最新进展。最后，本文为设计健壮的恶意软件分类器提供了指导，并概述了未来的研究方向。



## **42. PASA: Attack Agnostic Unsupervised Adversarial Detection using Prediction & Attribution Sensitivity Analysis**

PASA：使用预测和归因敏感性分析的攻击不可知无监督对抗检测 cs.CR

9th IEEE European Symposium on Security and Privacy

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.10789v1) [paper-pdf](http://arxiv.org/pdf/2404.10789v1)

**Authors**: Dipkamal Bhusal, Md Tanvirul Alam, Monish K. Veerabhadran, Michael Clifford, Sara Rampazzi, Nidhi Rastogi

**Abstract**: Deep neural networks for classification are vulnerable to adversarial attacks, where small perturbations to input samples lead to incorrect predictions. This susceptibility, combined with the black-box nature of such networks, limits their adoption in critical applications like autonomous driving. Feature-attribution-based explanation methods provide relevance of input features for model predictions on input samples, thus explaining model decisions. However, we observe that both model predictions and feature attributions for input samples are sensitive to noise. We develop a practical method for this characteristic of model prediction and feature attribution to detect adversarial samples. Our method, PASA, requires the computation of two test statistics using model prediction and feature attribution and can reliably detect adversarial samples using thresholds learned from benign samples. We validate our lightweight approach by evaluating the performance of PASA on varying strengths of FGSM, PGD, BIM, and CW attacks on multiple image and non-image datasets. On average, we outperform state-of-the-art statistical unsupervised adversarial detectors on CIFAR-10 and ImageNet by 14\% and 35\% ROC-AUC scores, respectively. Moreover, our approach demonstrates competitive performance even when an adversary is aware of the defense mechanism.

摘要: 用于分类的深度神经网络很容易受到敌意攻击，在这种攻击中，输入样本的微小扰动会导致错误的预测。这种敏感性，再加上这类网络的黑匣子性质，限制了它们在自动驾驶等关键应用中的应用。基于特征属性的解释方法为输入样本上的模型预测提供了输入特征的相关性，从而解释了模型决策。然而，我们观察到输入样本的模型预测和特征属性都对噪声敏感。针对模型预测和特征属性的这一特点，我们提出了一种实用的检测敌意样本的方法。我们的PASA方法需要使用模型预测和特征属性来计算两个测试统计量，并且可以使用从良性样本学习的阈值来可靠地检测敌意样本。我们通过评估PASA对多个图像和非图像数据集上不同强度的FGSM、PGD、BIM和CW攻击的性能来验证我们的轻量级方法。平均而言，我们在CIFAR-10和ImageNet上的ROC-AUC得分分别比最先进的统计非监督对手检测器高出14和35分。此外，我们的方法即使在对手知道防御机制的情况下也表现出了竞争性的性能。



## **43. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

越狱镜头：针对大型语言模型的越狱攻击的视觉分析 cs.CR

Submitted to VIS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08793v1) [paper-pdf](http://arxiv.org/pdf/2404.08793v1)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Minfeng Zhu, Wei Zhang, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.

摘要: 大型语言模型(LLM)的激增突显了人们对其安全漏洞的担忧，特别是针对越狱攻击的担忧，在越狱攻击中，对手设计越狱提示以绕过安全机制，防止潜在的滥用。解决这些问题需要对越狱提示进行全面分析，以评估LLMS的防御能力并确定潜在的弱点。然而，评估越狱性能和理解提示特征的复杂性使得这一分析变得费力。我们与领域专家合作来表征问题，并提出一个LLM辅助的框架来简化分析过程。它提供自动越狱评估，以便于对提示中的组件和关键字进行性能评估和支持分析。基于该框架，我们设计了一个可视化分析系统JailBreakLens，使用户能够针对目标模型探索越狱性能，对提示特征进行多层次分析，并对提示实例进行精化以验证结果。通过案例研究、技术评估和专家访谈，我们展示了我们的系统在帮助用户评估模型安全性和识别模型弱点方面的有效性。



## **44. What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?**

状态对抗多智能体强化学习的解决方案是什么？ cs.AI

Accepted by Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2212.02705v5) [paper-pdf](http://arxiv.org/pdf/2212.02705v5)

**Authors**: Songyang Han, Sanbao Su, Sihong He, Shuo Han, Haizhao Yang, Shaofeng Zou, Fei Miao

**Abstract**: Various methods for Multi-Agent Reinforcement Learning (MARL) have been developed with the assumption that agents' policies are based on accurate state information. However, policies learned through Deep Reinforcement Learning (DRL) are susceptible to adversarial state perturbation attacks. In this work, we propose a State-Adversarial Markov Game (SAMG) and make the first attempt to investigate different solution concepts of MARL under state uncertainties. Our analysis shows that the commonly used solution concepts of optimal agent policy and robust Nash equilibrium do not always exist in SAMGs. To circumvent this difficulty, we consider a new solution concept called robust agent policy, where agents aim to maximize the worst-case expected state value. We prove the existence of robust agent policy for finite state and finite action SAMGs. Additionally, we propose a Robust Multi-Agent Adversarial Actor-Critic (RMA3C) algorithm to learn robust policies for MARL agents under state uncertainties. Our experiments demonstrate that our algorithm outperforms existing methods when faced with state perturbations and greatly improves the robustness of MARL policies. Our code is public on https://songyanghan.github.io/what_is_solution/.

摘要: 多智能体强化学习(MAIL)的各种方法都是在假设智能体的策略基于准确的状态信息的基础上提出的。然而，通过深度强化学习(DRL)学习的策略容易受到对抗性状态扰动攻击。在这项工作中，我们提出了一种状态-对手马尔可夫博弈(SAMG)，并首次尝试研究了状态不确定条件下Marl的不同解概念。我们的分析表明，最优代理策略和稳健纳什均衡等解的概念在SAMG中并不总是存在的。为了规避这一困难，我们考虑了一个新的解决方案概念，称为稳健代理策略，其中代理的目标是最大化最坏情况下的预期状态值。我们证明了有限状态和有限动作SAMG的鲁棒代理策略的存在性。此外，我们还提出了一种健壮的多智能体对抗行为者-批评者(RMA3C)算法来学习状态不确定条件下MAIL智能体的健壮策略。实验表明，该算法在面对状态扰动时的性能优于已有方法，并大大提高了MAIL策略的稳健性。我们的代码在https://songyanghan.github.io/what_is_solution/.上是公开的



## **45. Mayhem: Targeted Corruption of Register and Stack Variables**

混乱：寄存器和堆栈变量的有针对性的腐败 cs.CR

ACM ASIACCS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.02545v2) [paper-pdf](http://arxiv.org/pdf/2309.02545v2)

**Authors**: Andrew J. Adiletta, M. Caner Tol, Yarkın Doröz, Berk Sunar

**Abstract**: In the past decade, many vulnerabilities were discovered in microarchitectures which yielded attack vectors and motivated the study of countermeasures. Further, architectural and physical imperfections in DRAMs led to the discovery of Rowhammer attacks which give an adversary power to introduce bit flips in a victim's memory space. Numerous studies analyzed Rowhammer and proposed techniques to prevent it altogether or to mitigate its effects.   In this work, we push the boundary and show how Rowhammer can be further exploited to inject faults into stack variables and even register values in a victim's process. We achieve this by targeting the register value that is stored in the process's stack, which subsequently is flushed out into the memory, where it becomes vulnerable to Rowhammer. When the faulty value is restored into the register, it will end up used in subsequent iterations. The register value can be stored in the stack via latent function calls in the source or by actively triggering signal handlers. We demonstrate the power of the findings by applying the techniques to bypass SUDO and SSH authentication. We further outline how MySQL and other cryptographic libraries can be targeted with the new attack vector. There are a number of challenges this work overcomes with extensive experimentation before coming together to yield an end-to-end attack on an OpenSSL digital signature: achieving co-location with stack and register variables, with synchronization provided via a blocking window. We show that stack and registers are no longer safe from the Rowhammer attack.

摘要: 在过去的十年中，微体系结构中发现了许多漏洞，这些漏洞产生了攻击载体，并推动了对抗措施的研究。此外，DRAM的结构和物理缺陷导致了Rowhammer攻击的发现，这种攻击使对手有能力在受害者的记忆空间中引入比特翻转。许多研究分析了Rowhammer，并提出了完全预防或减轻其影响的技术。在这项工作中，我们突破了界限，并展示了如何进一步利用Rowhammer向受害者进程中的堆栈变量甚至寄存器值注入错误。我们通过锁定存储在进程堆栈中的寄存器值来实现这一点，该寄存器值随后被刷新到内存中，在内存中变得容易受到Rowhammer的攻击。当故障值恢复到寄存器中时，它将在后续迭代中使用。寄存器值可以通过源代码中的潜在函数调用或通过主动触发信号处理程序存储在堆栈中。我们通过应用绕过SUDO和SSH身份验证的技术来演示这些发现的威力。我们进一步概述了如何利用新的攻击载体将MySQL和其他加密库作为目标。这项工作通过广泛的实验克服了许多挑战，然后结合在一起对OpenSSL数字签名进行端到端攻击：实现堆栈和寄存器变量的协同定位，并通过阻塞窗口提供同步。我们表明堆栈和寄存器不再安全，不再受到Rowhammer攻击。



## **46. On the Robustness of Language Guidance for Low-Level Vision Tasks: Findings from Depth Estimation**

低水平视觉任务语言指导的稳健性：深度估计的发现 cs.CV

Accepted to CVPR 2024. Project webpage:  https://agneetchatterjee.com/robustness_depth_lang/

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08540v1) [paper-pdf](http://arxiv.org/pdf/2404.08540v1)

**Authors**: Agneet Chatterjee, Tejas Gokhale, Chitta Baral, Yezhou Yang

**Abstract**: Recent advances in monocular depth estimation have been made by incorporating natural language as additional guidance. Although yielding impressive results, the impact of the language prior, particularly in terms of generalization and robustness, remains unexplored. In this paper, we address this gap by quantifying the impact of this prior and introduce methods to benchmark its effectiveness across various settings. We generate "low-level" sentences that convey object-centric, three-dimensional spatial relationships, incorporate them as additional language priors and evaluate their downstream impact on depth estimation. Our key finding is that current language-guided depth estimators perform optimally only with scene-level descriptions and counter-intuitively fare worse with low level descriptions. Despite leveraging additional data, these methods are not robust to directed adversarial attacks and decline in performance with an increase in distribution shift. Finally, to provide a foundation for future research, we identify points of failures and offer insights to better understand these shortcomings. With an increasing number of methods using language for depth estimation, our findings highlight the opportunities and pitfalls that require careful consideration for effective deployment in real-world settings

摘要: 单目深度估计的最新进展是通过结合自然语言作为附加指导而取得的。尽管取得了令人印象深刻的成果，但语言先验的影响，特别是在泛化和稳健性方面的影响，仍然有待探索。在本文中，我们通过量化这一先验的影响来解决这一差距，并引入各种方法来对其在各种设置下的有效性进行基准测试。我们生成传达以对象为中心的三维空间关系的“低级别”句子，将它们作为额外的语言先决条件并入其中，并评估它们对深度估计的下游影响。我们的主要发现是，目前的语言引导深度估计器仅在场景级别的描述中表现最佳，而在低级别描述中表现较差。尽管利用了更多的数据，但这些方法对定向对抗性攻击和随着分布变化增加而导致的性能下降不是很健壮。最后，为了为未来的研究提供基础，我们找出了故障点，并提供了洞察力，以更好地理解这些缺点。随着越来越多的方法使用语言进行深度估计，我们的发现突出了需要仔细考虑在现实世界环境中有效部署的机会和陷阱



## **47. VertAttack: Taking advantage of Text Classifiers' horizontal vision**

VertAttack：利用文本分类器的水平视野 cs.CL

14 pages, 4 figures, accepted to NAACL 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08538v1) [paper-pdf](http://arxiv.org/pdf/2404.08538v1)

**Authors**: Jonathan Rusert

**Abstract**: Text classification systems have continuously improved in performance over the years. However, nearly all current SOTA classifiers have a similar shortcoming, they process text in a horizontal manner. Vertically written words will not be recognized by a classifier. In contrast, humans are easily able to recognize and read words written both horizontally and vertically. Hence, a human adversary could write problematic words vertically and the meaning would still be preserved to other humans. We simulate such an attack, VertAttack. VertAttack identifies which words a classifier is reliant on and then rewrites those words vertically. We find that VertAttack is able to greatly drop the accuracy of 4 different transformer models on 5 datasets. For example, on the SST2 dataset, VertAttack is able to drop RoBERTa's accuracy from 94 to 13%. Furthermore, since VertAttack does not replace the word, meaning is easily preserved. We verify this via a human study and find that crowdworkers are able to correctly label 77% perturbed texts perturbed, compared to 81% of the original texts. We believe VertAttack offers a look into how humans might circumvent classifiers in the future and thus inspire a look into more robust algorithms.

摘要: 多年来，文本分类系统在性能上不断提高。然而，目前几乎所有的SOTA分类器都有一个类似的缺点，它们以水平的方式处理文本。分类器无法识别垂直书写的单词。相比之下，人类很容易识别和阅读水平和垂直书写的单词。因此，人类对手可以垂直书写有问题的单词，而其含义仍将保留给其他人类。我们模拟这样的攻击，VertAttack。VertAttack识别分类器依赖的单词，然后垂直重写这些单词。我们发现，VertAttack能够在5个数据集上大幅降低4种不同变压器模型的精度。例如，在Sst2数据集上，VertAttack能够将Roberta的准确率从94%降至13%。此外，由于VertAttack不会取代该词，因此含义很容易保留。我们通过一项人体研究验证了这一点，发现众筹人员能够正确地将77%的受干扰文本标记为受干扰的文本，而原始文本的这一比例为81%。我们相信，VertAttack提供了一个关于人类未来可能如何绕过分类器的展望，从而激发了对更健壮算法的研究。



## **48. Adversarially Robust Spiking Neural Networks Through Conversion**

通过转换的对抗鲁棒尖峰神经网络 cs.NE

Transactions on Machine Learning Research (TMLR), 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2311.09266v2) [paper-pdf](http://arxiv.org/pdf/2311.09266v2)

**Authors**: Ozan Özdenizci, Robert Legenstein

**Abstract**: Spiking neural networks (SNNs) provide an energy-efficient alternative to a variety of artificial neural network (ANN) based AI applications. As the progress in neuromorphic computing with SNNs expands their use in applications, the problem of adversarial robustness of SNNs becomes more pronounced. To the contrary of the widely explored end-to-end adversarial training based solutions, we address the limited progress in scalable robust SNN training methods by proposing an adversarially robust ANN-to-SNN conversion algorithm. Our method provides an efficient approach to embrace various computationally demanding robust learning objectives that have been proposed for ANNs. During a post-conversion robust finetuning phase, our method adversarially optimizes both layer-wise firing thresholds and synaptic connectivity weights of the SNN to maintain transferred robustness gains from the pre-trained ANN. We perform experimental evaluations in a novel setting proposed to rigorously assess the robustness of SNNs, where numerous adaptive adversarial attacks that account for the spike-based operation dynamics are considered. Results show that our approach yields a scalable state-of-the-art solution for adversarially robust deep SNNs with low-latency.

摘要: 尖峰神经网络(SNN)为各种基于人工神经网络(ANN)的人工智能应用提供了一种节能的替代方案。随着SNN在神经形态计算中应用范围的扩大，SNN的对抗健壮性问题变得更加突出。针对目前广泛研究的基于端到端对抗训练的解决方案，我们提出了一种对抗性稳健的神经网络到SNN的转换算法，解决了可扩展的稳健SNN训练方法的局限性。我们的方法提供了一种有效的方法来接受各种计算要求高的健壮学习目标，这些目标已经被提出给神经网络。在转换后的稳健精调阶段，我们的方法相反地优化了SNN的层级激发阈值和突触连接权重，以保持从预训练的ANN中转移的鲁棒性增益。我们在一种新的环境下进行了实验评估，该环境旨在严格评估SNN的健壮性，其中考虑了大量的自适应对手攻击，这些攻击解释了基于尖峰的操作动态。结果表明，我们的方法为具有低延迟的相反健壮的深度SNN提供了一种可扩展的最先进的解决方案。



## **49. Counterfactual Explanations for Face Forgery Detection via Adversarial Removal of Artifacts**

通过对抗性去除伪影进行人脸伪造检测的反事实解释 cs.CV

Accepted to ICME2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08341v1) [paper-pdf](http://arxiv.org/pdf/2404.08341v1)

**Authors**: Yang Li, Songlin Yang, Wei Wang, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Highly realistic AI generated face forgeries known as deepfakes have raised serious social concerns. Although DNN-based face forgery detection models have achieved good performance, they are vulnerable to latest generative methods that have less forgery traces and adversarial attacks. This limitation of generalization and robustness hinders the credibility of detection results and requires more explanations. In this work, we provide counterfactual explanations for face forgery detection from an artifact removal perspective. Specifically, we first invert the forgery images into the StyleGAN latent space, and then adversarially optimize their latent representations with the discrimination supervision from the target detection model. We verify the effectiveness of the proposed explanations from two aspects: (1) Counterfactual Trace Visualization: the enhanced forgery images are useful to reveal artifacts by visually contrasting the original images and two different visualization methods; (2) Transferable Adversarial Attacks: the adversarial forgery images generated by attacking the detection model are able to mislead other detection models, implying the removed artifacts are general. Extensive experiments demonstrate that our method achieves over 90% attack success rate and superior attack transferability. Compared with naive adversarial noise methods, our method adopts both generative and discriminative model priors, and optimize the latent representations in a synthesis-by-analysis way, which forces the search of counterfactual explanations on the natural face manifold. Thus, more general counterfactual traces can be found and better adversarial attack transferability can be achieved.

摘要: 高度逼真的人工智能生成的人脸伪造被称为深度假冒，已经引起了严重的社会关注。虽然基于DNN的人脸伪造检测模型取得了很好的性能，但它们容易受到最新的生成性方法的攻击，这些方法具有较少的伪造痕迹和对抗性攻击。这种泛化和稳健性的限制阻碍了检测结果的可信度，需要更多的解释。在这项工作中，我们从伪影去除的角度为人脸伪造检测提供了反事实的解释。具体地说，我们首先将伪造图像倒置到StyleGAN潜在空间，然后在目标检测模型的判别监督下对其潜在表示进行反向优化。我们从两个方面验证了所提出的解释的有效性：(1)反事实跟踪可视化：增强的伪造图像通过视觉对比原始图像和两种不同的可视化方法来揭示伪像；(2)可转移的对抗性攻击：攻击检测模型生成的对抗性伪造图像能够误导其他检测模型，这意味着去除的伪像是通用的。大量实验表明，该方法具有90%以上的攻击成功率和良好的攻击可转移性。与朴素的对抗性噪声方法相比，该方法同时采用产生式和判别式模型先验，并通过分析综合的方式对潜在表示进行优化，迫使人们在自然人脸流形上寻找反事实的解释。因此，可以发现更一般的反事实痕迹，并且可以实现更好的对抗性攻击可转移性。



## **50. Combating Advanced Persistent Threats: Challenges and Solutions**

应对高级持续威胁：挑战和解决方案 cs.CR

This work has been accepted by IEEE NETWORK in April 2024. 9 pages, 5  figures, 1 table

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.09498v2) [paper-pdf](http://arxiv.org/pdf/2309.09498v2)

**Authors**: Yuntao Wang, Han Liu, Zhendong Li, Zhou Su, Jiliang Li

**Abstract**: The rise of advanced persistent threats (APTs) has marked a significant cybersecurity challenge, characterized by sophisticated orchestration, stealthy execution, extended persistence, and targeting valuable assets across diverse sectors. Provenance graph-based kernel-level auditing has emerged as a promising approach to enhance visibility and traceability within intricate network environments. However, it still faces challenges including reconstructing complex lateral attack chains, detecting dynamic evasion behaviors, and defending smart adversarial subgraphs. To bridge the research gap, this paper proposes an efficient and robust APT defense scheme leveraging provenance graphs, including a network-level distributed audit model for cost-effective lateral attack reconstruction, a trust-oriented APT evasion behavior detection strategy, and a hidden Markov model based adversarial subgraph defense approach. Through prototype implementation and extensive experiments, we validate the effectiveness of our system. Lastly, crucial open research directions are outlined in this emerging field.

摘要: 高级持续性威胁(APT)的兴起标志着一个重大的网络安全挑战，其特征是复杂的协调、隐蔽的执行、延长的持久性以及针对不同行业的宝贵资产。基于起源图的内核级审计已经成为在复杂的网络环境中增强可见性和可跟踪性的一种有前途的方法。然而，它仍然面临着重构复杂的侧向攻击链、检测动态规避行为和防御智能对抗性子图等挑战。为了弥补这一研究空白，本文提出了一种利用起源图的高效、健壮的APT防御方案，包括用于高性价比的横向攻击重构的网络级分布式审计模型、面向信任的APT逃避行为检测策略以及基于隐马尔可夫模型的敌意子图防御方法。通过原型实现和广泛的实验，验证了系统的有效性。最后，概述了这一新兴领域的关键开放研究方向。



