# Latest Adversarial Attack Papers
**update at 2024-06-09 20:17:43**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improving Alignment and Robustness with Short Circuiting**

通过短路改善对齐和鲁棒性 cs.LG

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04313v1) [paper-pdf](http://arxiv.org/pdf/2406.04313v1)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that "short-circuits" models as they respond with harmful outputs. Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, short-circuiting directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, short-circuiting allows the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

摘要: 人工智能系统可能采取有害行动，并且非常容易受到对抗性攻击。我们提出了一种方法，灵感来自于最近在表示工程方面的进步，即在模型以有害输出做出反应时使其“短路”。旨在改善一致性的现有技术，如拒绝训练，经常被绕过。对抗性训练等技术试图通过反击特定攻击来堵塞这些漏洞。作为拒绝训练和对抗性训练的替代方案，短路直接控制了首先要对有害输出负责的陈述。我们的技术可以应用于纯文本和多模式语言模型，在不牺牲效用的情况下防止产生有害输出-即使在存在强大的看不见的攻击的情况下也是如此。值得注意的是，虽然独立图像识别中的对抗性健壮性仍然是一个开放的挑战，但短路使更大的多模式系统能够可靠地经受住旨在产生有害内容的图像“劫持”。最后，我们将我们的方法扩展到人工智能代理，表明当他们受到攻击时，有害行动的比率大大降低。我们的方法代表着在发展对有害行为和敌对攻击的可靠保障方面向前迈出了重要的一步。



## **2. NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise**

NoisyGL：标签噪音下图神经网络的综合基准 cs.LG

Submitted to the 38th Conference on Neural Information Processing  Systems (NeurIPS 2024) Track on Datasets and Benchmarks

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04299v1) [paper-pdf](http://arxiv.org/pdf/2406.04299v1)

**Authors**: Zhonghao Wang, Danyu Sun, Sheng Zhou, Haobo Wang, Jiapei Fan, Longtao Huang, Jiajun Bu

**Abstract**: Graph Neural Networks (GNNs) exhibit strong potential in node classification task through a message-passing mechanism. However, their performance often hinges on high-quality node labels, which are challenging to obtain in real-world scenarios due to unreliable sources or adversarial attacks. Consequently, label noise is common in real-world graph data, negatively impacting GNNs by propagating incorrect information during training. To address this issue, the study of Graph Neural Networks under Label Noise (GLN) has recently gained traction. However, due to variations in dataset selection, data splitting, and preprocessing techniques, the community currently lacks a comprehensive benchmark, which impedes deeper understanding and further development of GLN. To fill this gap, we introduce NoisyGL in this paper, the first comprehensive benchmark for graph neural networks under label noise. NoisyGL enables fair comparisons and detailed analyses of GLN methods on noisy labeled graph data across various datasets, with unified experimental settings and interface. Our benchmark has uncovered several important insights that were missed in previous research, and we believe these findings will be highly beneficial for future studies. We hope our open-source benchmark library will foster further advancements in this field. The code of the benchmark can be found in https://github.com/eaglelab-zju/NoisyGL.

摘要: 图神经网络(GNN)通过一种消息传递机制在节点分类任务中显示出很强的潜力。然而，它们的性能往往取决于高质量的节点标签，而在现实世界的场景中，由于不可靠的来源或对手攻击，这些标签很难获得。因此，标签噪声在真实世界的图形数据中很常见，通过在训练期间传播不正确的信息而对GNN产生负面影响。为了解决这个问题，标签噪声下的图神经网络(GLN)的研究最近得到了重视。然而，由于数据集选择、数据拆分和预处理技术的差异，目前社区缺乏一个全面的基准，这阻碍了对GLN的深入理解和进一步发展。为了填补这一空白，我们在本文中引入了NoisyGL，这是第一个在标签噪声下对图神经网络进行全面测试的基准。NoisyGL可以通过统一的实验设置和界面，对不同数据集上的噪声标记图形数据进行公平的GLN方法比较和详细分析。我们的基准发现了之前研究中遗漏的几个重要见解，我们相信这些发现将对未来的研究非常有益。我们希望我们的开源基准库将促进这一领域的进一步发展。基准测试的代码可以在https://github.com/eaglelab-zju/NoisyGL.中找到



## **3. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

BadRAG：识别大型语言模型检索增强生成中的漏洞 cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.00083v2) [paper-pdf](http://arxiv.org/pdf/2406.00083v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.

摘要: 大型语言模型(LLM)受到过时信息和生成错误数据的倾向的限制，这通常被称为“幻觉”。检索-增强生成(RAG)结合了基于检索的方法和生成模型的优点，解决了这些局限性。这种方法涉及从大型最新数据集中检索相关信息，并使用它来改进生成过程，从而产生更准确和符合上下文的响应。尽管有好处，但RAG为LLMS带来了新的攻击面，特别是因为RAG数据库通常来自公共数据，如Web。本文提出用TrojRAG{}来识别检索零件(RAG数据库)上的漏洞和攻击，以及它们对生成零件(LLM)的间接攻击。具体地说，我们发现毒化几个定制的内容段落可以实现检索后门，其中检索对于干净的查询工作得很好，但总是返回定制的有毒对抗性查询。触发器和有毒段落可以高度定制，以实施各种攻击。例如，触发点可能是一个语义组，比如“共和党、唐纳德·特朗普等。”对抗性段落可以针对不同的内容量身定做，不仅与触发因素有关，还可以用来间接攻击生成性LLM而不修改它们。这些攻击可以包括针对RAG的拒绝服务攻击和针对受触发器限制的LLM生成的语义引导攻击。我们的实验表明，只要毒化10篇对抗性文章，就可以诱导98.2%的成功率来检索对抗性文章。然后，这些文章可以将基于RAG的GPT-4的拒绝率从0.01%提高到74.6%，或者将目标查询的否定回复率从0.22%提高到72%。



## **4. Batch-in-Batch: a new adversarial training framework for initial perturbation and sample selection**

批中批：用于初始扰动和样本选择的新对抗训练框架 cs.LG

29 pages, 11 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04070v1) [paper-pdf](http://arxiv.org/pdf/2406.04070v1)

**Authors**: Yinting Wu, Pai Peng, Bo Cai, Le Li, .

**Abstract**: Adversarial training methods commonly generate independent initial perturbation for adversarial samples from a simple uniform distribution, and obtain the training batch for the classifier without selection. In this work, we propose a simple yet effective training framework called Batch-in-Batch (BB) to enhance models robustness. It involves specifically a joint construction of initial values that could simultaneously generates $m$ sets of perturbations from the original batch set to provide more diversity for adversarial samples; and also includes various sample selection strategies that enable the trained models to have smoother losses and avoid overconfident outputs. Through extensive experiments on three benchmark datasets (CIFAR-10, SVHN, CIFAR-100) with two networks (PreActResNet18 and WideResNet28-10) that are used in both the single-step (Noise-Fast Gradient Sign Method, N-FGSM) and multi-step (Projected Gradient Descent, PGD-10) adversarial training, we show that models trained within the BB framework consistently have higher adversarial accuracy across various adversarial settings, notably achieving over a 13% improvement on the SVHN dataset with an attack radius of 8/255 compared to the N-FGSM baseline model. Furthermore, experimental analysis of the efficiency of both the proposed initial perturbation method and sample selection strategies validates our insights. Finally, we show that our framework is cost-effective in terms of computational resources, even with a relatively large value of $m$.

摘要: 对抗性训练方法通常从简单的均匀分布对对抗性样本产生独立的初始扰动，不需要选择就可以获得分类器的训练批次。在这项工作中，我们提出了一种简单而有效的训练框架，称为Batch-in-Batch(BB)，以增强模型的稳健性。具体而言，它涉及联合构造初值，该初值可以同时从原始批次集合产生$m$集合的扰动，从而为对抗性样本提供更多的多样性；还包括各种样本选择策略，使训练的模型具有更平滑的损失并避免过度自信的输出。通过在三个基准数据集(CIFAR-10，SVHN，CIFAR-100)和两个网络(PreActResNet18和WideResNet28-10)上的广泛实验，这两个网络同时用于单步(噪声-快速梯度符号方法，N-FGSM)和多步(投影梯度下降，PGD-10)对抗训练，我们表明，在BB框架内训练的模型在各种对抗环境下一致具有更高的对抗准确率，与N-FGSM基线模型相比，攻击半径为8/255的SVHN数据集获得了13%以上的改进。此外，对初始摄动法和样本选择策略的效率进行了实验分析，验证了本文的观点。最后，我们证明了我们的框架在计算资源方面是有成本效益的，即使有相对较大的百万美元的价值。



## **5. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04031v1) [paper-pdf](http://arxiv.org/pdf/2406.04031v1)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **6. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.14461v2) [paper-pdf](http://arxiv.org/pdf/2404.14461v2)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **7. Verifiably Robust Conformal Prediction**

可验证鲁棒性保形预测 cs.LO

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2405.18942v2) [paper-pdf](http://arxiv.org/pdf/2405.18942v2)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.

摘要: 保角预测是一种流行的不确定性量化方法，它假设训练和测试数据是可交换的，提供了无分布的、统计上有效的预测集。在这种情况下，CP的预测集保证以用户指定的概率覆盖(未知)真实测试输出。然而，当数据受到对抗性攻击时，这一保证就会被违反，这往往会导致覆盖范围的重大损失。最近，已经提出了几种在这种情况下恢复CP担保的方法。这些方法利用随机平滑的变化来产生保守集合，这些保守集合考虑了对抗性扰动的影响。然而，它们的局限性在于它们只支持$^2$有界的扰动和分类任务。本文介绍了一种新的框架VRCP，它利用最新的神经网络验证方法来恢复对抗性攻击下的覆盖保证。我们的VRCP方法是第一个支持以任意范数为界的扰动，包括$^1$，$^2$，$^inty$，以及回归任务。我们在深度强化学习环境下的图像分类任务(CIFAR10、CIFAR100和TinyImageNet)和回归任务上对我们的方法进行了评估和比较。在任何情况下，VRCP都达到了名义覆盖率以上，并产生了比SOTA更有效和更有信息量的预测区域。



## **8. Behavior-Targeted Attack on Reinforcement Learning with Limited Access to Victim's Policy**

对受害者政策访问有限的强化学习的行为目标攻击 cs.LG

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03862v1) [paper-pdf](http://arxiv.org/pdf/2406.03862v1)

**Authors**: Shojiro Yamabe, Kazuto Fukuchi, Ryoma Senda, Jun Sakuma

**Abstract**: This study considers the attack on reinforcement learning agents where the adversary aims to control the victim's behavior as specified by the adversary by adding adversarial modifications to the victim's state observation. While some attack methods reported success in manipulating the victim agent's behavior, these methods often rely on environment-specific heuristics. In addition, all existing attack methods require white-box access to the victim's policy. In this study, we propose a novel method for manipulating the victim agent in the black-box (i.e., the adversary is allowed to observe the victim's state and action only) and no-box (i.e., the adversary is allowed to observe the victim's state only) setting without requiring environment-specific heuristics. Our attack method is formulated as a bi-level optimization problem that is reduced to a distribution matching problem and can be solved by an existing imitation learning algorithm in the black-box and no-box settings. Empirical evaluations on several reinforcement learning benchmarks show that our proposed method has superior attack performance to baselines.

摘要: 该研究考虑了对强化学习智能体的攻击，其中对手的目的是通过在受害者的状态观察中添加对抗性的修改来控制受害者的行为。虽然一些攻击方法报告成功地操纵了受害者代理的行为，但这些方法通常依赖于特定于环境的启发式方法。此外，所有现有攻击方法都需要白盒访问受害者的策略。在本研究中，我们提出了一种新的方法来操纵黑箱(即对手只允许观察受害者的状态和动作)和非盒子(即对手只允许观察受害者的状态)环境中的受害者代理，而不需要特定于环境的启发式方法。我们的攻击方法被描述为一个双层优化问题，该问题归结为一个分布匹配问题，并且可以在黑箱和非箱环境下通过现有的模仿学习算法来求解。在几个强化学习基准上的实验评估表明，我们提出的方法具有比基线更好的攻击性能。



## **9. Exploiting Global Graph Homophily for Generalized Defense in Graph Neural Networks**

利用全局图同质性进行图神经网络中的广义防御 cs.LG

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03833v1) [paper-pdf](http://arxiv.org/pdf/2406.03833v1)

**Authors**: Duanyu Li, Huijun Wu, Min Xie, Xugang Wu, Zhenwei Wu, Wenzhe Zhang

**Abstract**: Graph neural network (GNN) models play a pivotal role in numerous tasks involving graph-related data analysis. Despite their efficacy, similar to other deep learning models, GNNs are susceptible to adversarial attacks. Even minor perturbations in graph data can induce substantial alterations in model predictions. While existing research has explored various adversarial defense techniques for GNNs, the challenge of defending against adversarial attacks on real-world scale graph data remains largely unresolved. On one hand, methods reliant on graph purification and preprocessing tend to excessively emphasize local graph information, leading to sub-optimal defensive outcomes. On the other hand, approaches rooted in graph structure learning entail significant time overheads, rendering them impractical for large-scale graphs. In this paper, we propose a new defense method named Talos, which enhances the global, rather than local, homophily of graphs as a defense. Experiments show that the proposed approach notably outperforms state-of-the-art defense approaches, while imposing little computational overhead.

摘要: 图神经网络(GNN)模型在许多涉及图相关数据分析的任务中发挥着关键作用。尽管GNN像其他深度学习模型一样有效，但它很容易受到敌意攻击。即使是图表数据中的微小扰动，也可能导致模型预测的重大变化。虽然现有的研究已经探索了针对GNN的各种对抗性防御技术，但针对真实世界规模的图数据的对抗性攻击的防御挑战在很大程度上仍然没有解决。一方面，依赖于图提纯和预处理的方法往往过分强调局部图信息，导致次优防御结果。另一方面，扎根于图结构学习的方法需要大量的时间开销，使得它们对于大规模图是不现实的。在本文中，我们提出了一种新的防御方法TALOS，它增强了图的全局同伦而不是局部同伦作为防御。实验表明，该方法在计算开销很小的情况下，显著优于现有的防御方法。



## **10. AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens**

自动越狱：通过依赖的视角探索越狱攻击和防御 cs.CR

32 pages, 2 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03805v1) [paper-pdf](http://arxiv.org/pdf/2406.03805v1)

**Authors**: Lin Lu, Hai Yan, Zenghui Yuan, Jiawen Shi, Wenqi Wei, Pin-Yu Chen, Pan Zhou

**Abstract**: Jailbreak attacks in large language models (LLMs) entail inducing the models to generate content that breaches ethical and legal norm through the use of malicious prompts, posing a substantial threat to LLM security. Current strategies for jailbreak attack and defense often focus on optimizing locally within specific algorithmic frameworks, resulting in ineffective optimization and limited scalability. In this paper, we present a systematic analysis of the dependency relationships in jailbreak attack and defense techniques, generalizing them to all possible attack surfaces. We employ directed acyclic graphs (DAGs) to position and analyze existing jailbreak attacks, defenses, and evaluation methodologies, and propose three comprehensive, automated, and logical frameworks. \texttt{AutoAttack} investigates dependencies in two lines of jailbreak optimization strategies: genetic algorithm (GA)-based attacks and adversarial-generation-based attacks, respectively. We then introduce an ensemble jailbreak attack to exploit these dependencies. \texttt{AutoDefense} offers a mixture-of-defenders approach by leveraging the dependency relationships in pre-generative and post-generative defense strategies. \texttt{AutoEvaluation} introduces a novel evaluation method that distinguishes hallucinations, which are often overlooked, from jailbreak attack and defense responses. Through extensive experiments, we demonstrate that the proposed ensemble jailbreak attack and defense framework significantly outperforms existing research.

摘要: 大型语言模型(LLM)中的越狱攻击需要通过使用恶意提示诱导模型生成违反道德和法律规范的内容，从而对LLM安全构成重大威胁。当前的越狱攻防策略往往集中在特定算法框架内的局部优化，导致优化效果不佳，可扩展性有限。本文系统地分析了越狱攻防技术中的依赖关系，并将其推广到所有可能的攻击面。我们使用有向无环图(DAG)来定位和分析现有的越狱攻击、防御和评估方法，并提出了三个全面的、自动化的和逻辑的框架。Texttt{AutoAttack}研究了两种越狱优化策略的依赖关系：基于遗传算法(GA)的攻击和基于对抗性生成的攻击。然后，我们引入整体越狱攻击来利用这些依赖关系。通过利用生成前和生成后防御策略中的依赖关系，\exttt{AutoDefense}提供了混合防御者的方法。Texttt(自动评估)介绍了一种新的评估方法，可以将经常被忽视的幻觉与越狱攻击和防御反应区分开来。通过大量的实验，我们证明了所提出的集成越狱攻防框架的性能明显优于现有的研究。



## **11. Transferable Availability Poisoning Attacks**

可转移可用性中毒攻击 cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2310.05141v2) [paper-pdf](http://arxiv.org/pdf/2310.05141v2)

**Authors**: Yiyong Liu, Michael Backes, Xiao Zhang

**Abstract**: We consider availability data poisoning attacks, where an adversary aims to degrade the overall test accuracy of a machine learning model by crafting small perturbations to its training data. Existing poisoning strategies can achieve the attack goal but assume the victim to employ the same learning method as what the adversary uses to mount the attack. In this paper, we argue that this assumption is strong, since the victim may choose any learning algorithm to train the model as long as it can achieve some targeted performance on clean data. Empirically, we observe a large decrease in the effectiveness of prior poisoning attacks if the victim employs an alternative learning algorithm. To enhance the attack transferability, we propose Transferable Poisoning, which first leverages the intrinsic characteristics of alignment and uniformity to enable better unlearnability within contrastive learning, and then iteratively utilizes the gradient information from supervised and unsupervised contrastive learning paradigms to generate the poisoning perturbations. Through extensive experiments on image benchmarks, we show that our transferable poisoning attack can produce poisoned samples with significantly improved transferability, not only applicable to the two learners used to devise the attack but also to learning algorithms and even paradigms beyond.

摘要: 我们考虑可用性数据中毒攻击，其中对手的目标是通过对机器学习模型的训练数据进行小的扰动来降低其总体测试精度。现有的中毒策略可以达到攻击目标，但假设受害者使用与对手发动攻击相同的学习方法。在本文中，我们认为这一假设是强有力的，因为受害者可以选择任何学习算法来训练模型，只要它能够在干净的数据上达到一些目标性能。经验上，如果受害者使用另一种学习算法，我们观察到先前中毒攻击的有效性大幅下降。为了增强攻击的可转移性，我们提出了可传递中毒，它首先利用对准和一致性的内在特性在对比学习中实现更好的不可学习性，然后迭代地利用监督和非监督对比学习范例中的梯度信息来产生中毒扰动。通过在图像基准上的大量实验表明，我们的可转移中毒攻击可以产生中毒样本，并显著提高了可转移性，不仅适用于用于设计攻击的两个学习器，而且适用于学习算法甚至更多的范例。



## **12. Backdoor Attack with Sparse and Invisible Trigger**

具有稀疏和不可见触发的后门攻击 cs.CV

This paper was accepted by IEEE Transactions on Information Forensics  and Security (TIFS). The first two authors contributed equally to this work.  14 pages

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2306.06209v3) [paper-pdf](http://arxiv.org/pdf/2306.06209v3)

**Authors**: Yinghua Gao, Yiming Li, Xueluan Gong, Zhifeng Li, Shu-Tao Xia, Qian Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where the adversary manipulates a small portion of training data such that the victim model predicts normally on the benign samples but classifies the triggered samples as the target class. The backdoor attack is an emerging yet threatening training-phase threat, leading to serious risks in DNN-based applications. In this paper, we revisit the trigger patterns of existing backdoor attacks. We reveal that they are either visible or not sparse and therefore are not stealthy enough. More importantly, it is not feasible to simply combine existing methods to design an effective sparse and invisible backdoor attack. To address this problem, we formulate the trigger generation as a bi-level optimization problem with sparsity and invisibility constraints and propose an effective method to solve it. The proposed method is dubbed sparse and invisible backdoor attack (SIBA). We conduct extensive experiments on benchmark datasets under different settings, which verify the effectiveness of our attack and its resistance to existing backdoor defenses. The codes for reproducing main experiments are available at \url{https://github.com/YinghuaGao/SIBA}.

摘要: 深度神经网络(DNN)很容易受到后门攻击，对手操纵一小部分训练数据，使得受害者模型对良性样本进行正常预测，但将触发的样本归类为目标类。后门攻击是一种新出现的但具有威胁性的训练阶段威胁，导致基于DNN的应用程序存在严重风险。在本文中，我们回顾了现有后门攻击的触发模式。我们发现，它们要么是可见的，要么不是稀疏的，因此不够隐蔽。更重要的是，简单地结合现有方法来设计有效的稀疏、隐形的后门攻击是不可行的。针对这一问题，我们将触发器生成问题描述为一个具有稀疏性和不可见性约束的双层优化问题，并提出了一种有效的求解方法。该方法被称为稀疏不可见后门攻击(SIBA)。我们在不同环境下的基准数据集上进行了大量的实验，验证了我们的攻击的有效性及其对现有后门防御的抵抗力。有关复制主要实验的代码，请访问\url{https://github.com/YinghuaGao/SIBA}.



## **13. Principles of Designing Robust Remote Face Anti-Spoofing Systems**

设计稳健的远程面部反欺骗系统的原则 cs.CV

Under review

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03684v1) [paper-pdf](http://arxiv.org/pdf/2406.03684v1)

**Authors**: Xiang Xu, Tianchen Zhao, Zheng Zhang, Zhihua Li, Jon Wu, Alessandro Achille, Mani Srivastava

**Abstract**: Protecting digital identities of human face from various attack vectors is paramount, and face anti-spoofing plays a crucial role in this endeavor. Current approaches primarily focus on detecting spoofing attempts within individual frames to detect presentation attacks. However, the emergence of hyper-realistic generative models capable of real-time operation has heightened the risk of digitally generated attacks. In light of these evolving threats, this paper aims to address two key aspects. First, it sheds light on the vulnerabilities of state-of-the-art face anti-spoofing methods against digital attacks. Second, it presents a comprehensive taxonomy of common threats encountered in face anti-spoofing systems. Through a series of experiments, we demonstrate the limitations of current face anti-spoofing detection techniques and their failure to generalize to novel digital attack scenarios. Notably, the existing models struggle with digital injection attacks including adversarial noise, realistic deepfake attacks, and digital replay attacks. To aid in the design and implementation of robust face anti-spoofing systems resilient to these emerging vulnerabilities, the paper proposes key design principles from model accuracy and robustness to pipeline robustness and even platform robustness. Especially, we suggest to implement the proactive face anti-spoofing system using active sensors to significant reduce the risks for unseen attack vectors and improve the user experience.

摘要: 保护人脸的数字身份免受各种攻击载体的攻击是至关重要的，而人脸反欺骗在这一努力中起着至关重要的作用。当前的方法主要集中在检测单个帧内的欺骗尝试，以检测呈现攻击。然而，能够实时操作的超逼真生成模型的出现增加了数字生成攻击的风险。鉴于这些不断变化的威胁，本文旨在解决两个关键方面。首先，它揭示了针对数字攻击的最先进的Face反欺骗方法的漏洞。其次，对Face反欺骗系统中遇到的常见威胁进行了全面的分类。通过一系列的实验，我们证明了现有人脸反欺骗检测技术的局限性，以及它们不能推广到新的数字攻击场景。值得注意的是，现有的模型正在与数字注入攻击作斗争，包括对抗性噪声、真实的深度伪攻击和数字重播攻击。为了帮助设计和实现对这些新出现的漏洞具有弹性的健壮的Face反欺骗系统，本文提出了从模型的准确性和健壮性到流水线健壮性甚至平台健壮性的关键设计原则。特别是，我们建议使用主动传感器来实现主动的人脸反欺骗系统，以显著降低不可见攻击载体的风险，提高用户体验。



## **14. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.17263v3) [paper-pdf](http://arxiv.org/pdf/2401.17263v3)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **15. Ranking Manipulation for Conversational Search Engines**

对话式搜索引擎的排名操纵 cs.CL

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03589v1) [paper-pdf](http://arxiv.org/pdf/2406.03589v1)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.

摘要: 各大搜索引擎提供商正在快速整合大型语言模型(LLM)生成的内容，以响应用户查询。这些对话式搜索引擎通过将检索到的网站文本加载到LLM上下文中进行操作以进行摘要和解释。最近的研究表明，LLM非常容易受到越狱和快速注入攻击，这些攻击使用敌意字符串破坏LLM的安全和质量目标。这项工作调查了提示注入对对话式搜索引擎引用的来源的排名顺序的影响。为此，我们引入了一个聚焦于真实世界消费产品网站的数据集，并将会话搜索排名形式化为一个对抗性问题。在实验上，我们分析了在没有对抗性注入的情况下的会话搜索排名，结果表明不同的LLM在产品名称、文档内容和上下文位置的优先顺序上存在显著差异。然后，我们提出了一种基于攻击树的越狱技术，该技术可靠地推广排名较低的产品。重要的是，这些攻击有效地转移到了最先进的会话搜索引擎，如Pplexity.ai。考虑到网站所有者有强大的经济动机来提高他们的搜索排名，我们认为我们的问题表达对于未来的稳健性工作至关重要。



## **16. Distributional Adversarial Loss**

分配对抗损失 cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03458v1) [paper-pdf](http://arxiv.org/pdf/2406.03458v1)

**Authors**: Saba Ahmadi, Siddharth Bhandari, Avrim Blum, Chen Dan, Prabhav Jain

**Abstract**: A major challenge in defending against adversarial attacks is the enormous space of possible attacks that even a simple adversary might perform. To address this, prior work has proposed a variety of defenses that effectively reduce the size of this space. These include randomized smoothing methods that add noise to the input to take away some of the adversary's impact. Another approach is input discretization which limits the adversary's possible number of actions.   Motivated by these two approaches, we introduce a new notion of adversarial loss which we call distributional adversarial loss, to unify these two forms of effectively weakening an adversary. In this notion, we assume for each original example, the allowed adversarial perturbation set is a family of distributions (e.g., induced by a smoothing procedure), and the adversarial loss over each example is the maximum loss over all the associated distributions. The goal is to minimize the overall adversarial loss.   We show generalization guarantees for our notion of adversarial loss in terms of the VC-dimension of the hypothesis class and the size of the set of allowed adversarial distributions associated with each input. We also investigate the role of randomness in achieving robustness against adversarial attacks in the methods described above. We show a general derandomization technique that preserves the extent of a randomized classifier's robustness against adversarial attacks. We corroborate the procedure experimentally via derandomizing the Random Projection Filters framework of \cite{dong2023adversarial}. Our procedure also improves the robustness of the model against various adversarial attacks.

摘要: 防御对手攻击的一个主要挑战是，即使是一个简单的对手也可能进行巨大的攻击空间。为了解决这个问题，以前的工作已经提出了各种防御措施，有效地缩小了这个空间的大小。其中包括随机化的平滑方法，这种方法向输入中添加噪音，以消除对手的一些影响。另一种方法是输入离散化，它限制了对手可能采取的行动数量。受这两种方法的启发，我们引入了一种新的对抗性损失的概念，我们称之为分布性对抗性损失，将这两种有效削弱对手的形式统一起来。在这个概念中，我们假设对于每个原始示例，允许的对抗性扰动集是一族分布(例如，由平滑过程引起的)，并且每个示例上的对抗性损失是所有相关分布上的最大损失。目标是将对手的总体损失降至最低。我们根据假设类的VC维和与每个输入相关联的允许的对抗性分布的集合的大小来证明我们的对抗性损失概念的一般性保证。在上述方法中，我们还研究了随机性在实现对敌方攻击的稳健性方面的作用。我们展示了一种通用的去随机化技术，它保持了随机分类器对对手攻击的健壮性。我们通过对随机投影过滤器框架进行去随机化，从实验上证实了这一过程。我们的过程还提高了模型对各种对抗性攻击的稳健性。



## **17. CR-UTP: Certified Robustness against Universal Text Perturbations on Large Language Models**

CR-GPT：针对大型语言模型上通用文本扰动的鲁棒性认证 cs.CL

Accepted by ACL Findings 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.01873v2) [paper-pdf](http://arxiv.org/pdf/2406.01873v2)

**Authors**: Qian Lou, Xin Liang, Jiaqi Xue, Yancheng Zhang, Rui Xie, Mengxin Zheng

**Abstract**: It is imperative to ensure the stability of every prediction made by a language model; that is, a language's prediction should remain consistent despite minor input variations, like word substitutions. In this paper, we investigate the problem of certifying a language model's robustness against Universal Text Perturbations (UTPs), which have been widely used in universal adversarial attacks and backdoor attacks. Existing certified robustness based on random smoothing has shown considerable promise in certifying the input-specific text perturbations (ISTPs), operating under the assumption that any random alteration of a sample's clean or adversarial words would negate the impact of sample-wise perturbations. However, with UTPs, masking only the adversarial words can eliminate the attack. A naive method is to simply increase the masking ratio and the likelihood of masking attack tokens, but it leads to a significant reduction in both certified accuracy and the certified radius due to input corruption by extensive masking. To solve this challenge, we introduce a novel approach, the superior prompt search method, designed to identify a superior prompt that maintains higher certified accuracy under extensive masking. Additionally, we theoretically motivate why ensembles are a particularly suitable choice as base prompts for random smoothing. The method is denoted by superior prompt ensembling technique. We also empirically confirm this technique, obtaining state-of-the-art results in multiple settings. These methodologies, for the first time, enable high certified accuracy against both UTPs and ISTPs. The source code of CR-UTP is available at \url {https://github.com/UCFML-Research/CR-UTP}.

摘要: 必须确保语言模型做出的每个预测的稳定性；也就是说，语言的预测应该保持一致，尽管输入有微小的变化，如单词替换。在本文中，我们研究了语言模型对通用文本扰动(UTP)的稳健性证明问题，UTP被广泛应用于通用对抗性攻击和后门攻击。现有的基于随机平滑的已证明的稳健性在证明特定于输入的文本扰动(ISTP)方面显示出相当大的前景，其操作是在假设样本的干净或敌意的单词的任何随机改变将否定样本方面的扰动的影响的情况下进行的。然而，对于UTP，只屏蔽敌意的单词就可以消除攻击。一种天真的方法是简单地增加掩蔽率和掩蔽攻击令牌的可能性，但由于广泛的掩蔽导致输入损坏，它导致认证的准确性和认证的半径都显著降低。为了解决这一挑战，我们引入了一种新的方法，高级提示搜索方法，旨在识别在广泛掩蔽下保持更高认证准确率的高级提示。此外，我们从理论上解释了为什么作为随机平滑的基础提示，集合是特别合适的选择。这种方法以卓越的即时集成技术表示。我们还从经验上证实了这一技术，在多个环境下获得了最先进的结果。这些方法首次针对UTP和ISTP实现了高度认证的准确性。CR-UTP的源代码可在\url{https://github.com/UCFML-Research/CR-UTP}.



## **18. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

稳健的CLIP：稳健的大型视觉语言模型的视觉嵌入的无监督对抗微调 cs.LG

ICML 2024 Oral

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.12336v2) [paper-pdf](http://arxiv.org/pdf/2402.12336v2)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available at https://github.com/chs20/RobustVLM

摘要: OpenFlamingo、LLaVA和GPT-4等多模式基础模型越来越多地用于各种实际任务。先前的工作表明，这些模型非常容易受到视觉通道的对抗性攻击。这些攻击可以被用来传播虚假信息或欺骗用户，从而构成巨大的风险，这使得大型多通道基础模型的健壮性成为一个紧迫的问题。在许多大型视觉语言模型(如LLaVA和OpenFlamingo)中，剪辑模型或其变体之一被用作冻结的视觉编码器。我们提出了一种无监督的对抗性微调方案，以获得一个健壮的裁剪视觉编码器，它对依赖于裁剪的所有视觉下游任务(LVLM，零镜头分类)都具有健壮性。特别是，我们表明，一旦用我们的健壮模型取代了原始的剪辑模型，恶意第三方提供的篡改图像就不再可能对LVLMS的用户进行秘密攻击。不需要对下游低成本模块进行再培训或微调。代码和健壮模型可在https://github.com/chs20/RobustVLM上获得



## **19. Can Implicit Bias Imply Adversarial Robustness?**

隐性偏见会暗示对抗性鲁棒性吗？ cs.LG

icml 2024 camera-ready

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.15942v2) [paper-pdf](http://arxiv.org/pdf/2405.15942v2)

**Authors**: Hancheng Min, René Vidal

**Abstract**: The implicit bias of gradient-based training algorithms has been considered mostly beneficial as it leads to trained networks that often generalize well. However, Frei et al. (2023) show that such implicit bias can harm adversarial robustness. Specifically, they show that if the data consists of clusters with small inter-cluster correlation, a shallow (two-layer) ReLU network trained by gradient flow generalizes well, but it is not robust to adversarial attacks of small radius. Moreover, this phenomenon occurs despite the existence of a much more robust classifier that can be explicitly constructed from a shallow network. In this paper, we extend recent analyses of neuron alignment to show that a shallow network with a polynomial ReLU activation (pReLU) trained by gradient flow not only generalizes well but is also robust to adversarial attacks. Our results highlight the importance of the interplay between data structure and architecture design in the implicit bias and robustness of trained networks.

摘要: 基于梯度的训练算法的隐含偏差一直被认为是最有益的，因为它导致训练的网络往往具有很好的泛化能力。然而，Frei等人。(2023)表明，这种隐性偏见会损害对手的稳健性。具体地说，他们表明，如果数据由簇间相关性较小的簇组成，则由梯度流训练的浅(两层)RELU网络具有很好的泛化能力，但对小半径的敌意攻击并不健壮。此外，尽管存在可以从浅层网络显式构造的更健壮的分类器，但仍会出现这种现象。在本文中，我们扩展了最近对神经元排列的分析，表明由梯度流训练的具有多项式激活的浅网络(PReLU)不仅具有很好的泛化能力，而且对对手攻击具有很强的鲁棒性。我们的结果强调了数据结构和体系结构设计之间的相互作用在训练网络的隐含偏差和稳健性中的重要性。



## **20. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03230v1) [paper-pdf](http://arxiv.org/pdf/2406.03230v1)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply an established methodology for analyzing distinctive activation patterns in the residual streams for a novel result of attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用已建立的方法来分析残留流中不同的激活模式，以获得攻击提示分类的新结果。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **21. Graph Neural Network Explanations are Fragile**

图神经网络解释很脆弱 cs.CR

17 pages, 64 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03193v1) [paper-pdf](http://arxiv.org/pdf/2406.03193v1)

**Authors**: Jiate Li, Meng Pang, Yun Dong, Jinyuan Jia, Binghui Wang

**Abstract**: Explainable Graph Neural Network (GNN) has emerged recently to foster the trust of using GNNs. Existing GNN explainers are developed from various perspectives to enhance the explanation performance. We take the first step to study GNN explainers under adversarial attack--We found that an adversary slightly perturbing graph structure can ensure GNN model makes correct predictions, but the GNN explainer yields a drastically different explanation on the perturbed graph. Specifically, we first formulate the attack problem under a practical threat model (i.e., the adversary has limited knowledge about the GNN explainer and a restricted perturbation budget). We then design two methods (i.e., one is loss-based and the other is deduction-based) to realize the attack. We evaluate our attacks on various GNN explainers and the results show these explainers are fragile.

摘要: 可解释图神经网络（GNN）最近出现，旨在增强使用GNN的信任度。现有的GNN解释器是从各个角度开发的，以增强解释性能。我们迈出了第一步，研究对抗攻击下的GNN解释器--我们发现对手稍微扰乱图结构可以确保GNN模型做出正确的预测，但GNN解释器对受干扰的图产生了截然不同的解释。具体来说，我们首先在实际威胁模型下制定攻击问题（即，对手对GNN解释器的了解有限并且干扰预算有限）。然后我们设计两种方法（即，一种是基于损失的，另一种是基于演绎的）来实现攻击。我们评估了对各种GNN解释器的攻击，结果表明这些解释器很脆弱。



## **22. Reconstructing training data from document understanding models**

从文档理解模型重建训练数据 cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03182v1) [paper-pdf](http://arxiv.org/pdf/2406.03182v1)

**Authors**: Jérémie Dentan, Arnaud Paran, Aymen Shabou

**Abstract**: Document understanding models are increasingly employed by companies to supplant humans in processing sensitive documents, such as invoices, tax notices, or even ID cards. However, the robustness of such models to privacy attacks remains vastly unexplored. This paper presents CDMI, the first reconstruction attack designed to extract sensitive fields from the training data of these models. We attack LayoutLM and BROS architectures, demonstrating that an adversary can perfectly reconstruct up to 4.1% of the fields of the documents used for fine-tuning, including some names, dates, and invoice amounts up to six-digit numbers. When our reconstruction attack is combined with a membership inference attack, our attack accuracy escalates to 22.5%. In addition, we introduce two new end-to-end metrics and evaluate our approach under various conditions: unimodal or bimodal data, LayoutLM or BROS backbones, four fine-tuning tasks, and two public datasets (FUNSD and SROIE). We also investigate the interplay between overfitting, predictive performance, and susceptibility to our attack. We conclude with a discussion on possible defenses against our attack and potential future research directions to construct robust document understanding models.

摘要: 越来越多的公司使用文档理解模型来取代人工来处理敏感文档，如发票、纳税申报单，甚至身份证。然而，这类模型对隐私攻击的稳健性仍有待研究。本文提出了CDMI，这是第一个重构攻击，旨在从这些模型的训练数据中提取敏感区域。我们攻击LayoutLM和Bros架构，展示了对手可以完美地重建用于微调的文档高达4.1%的字段，包括一些名称、日期和发票金额高达六位数字。当我们的重构攻击与成员推理攻击相结合时，我们的攻击准确率提升到22.5%。此外，我们引入了两个新的端到端度量，并在不同的条件下对我们的方法进行了评估：单峰或双峰数据，LayoutLM或Bros主干，四个微调任务，以及两个公共数据集(FUNSD和SROIE)。我们还调查了过度匹配、预测性能和我们的攻击易感性之间的相互作用。最后，我们讨论了可能的防御攻击和潜在的未来研究方向，以构建稳健的文档理解模型。



## **23. ZeroPur: Succinct Training-Free Adversarial Purification**

ZeroPur：简洁的免培训对抗净化 cs.CV

16 pages, 5 figures, under review

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03143v1) [paper-pdf](http://arxiv.org/pdf/2406.03143v1)

**Authors**: Xiuli Bi, Zonglin Yang, Bo Liu, Xiaodong Cun, Chi-Man Pun, Pietro Lio, Bin Xiao

**Abstract**: Adversarial purification is a kind of defense technique that can defend various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.

摘要: 对抗性净化是一种防御技术，它可以在不修改受害者分类器的情况下防御各种看不见的对抗性攻击。现有的方法往往依赖于外部生成模型或辅助函数与受害者分类器之间的合作。然而，再训练生成模型、辅助函数或受害者分类器依赖于微调数据集的域，并且是计算消耗的。在这项工作中，我们假设对抗性图像是自然图像流形的离群点，净化过程可以被认为是将它们返回到这个流形。根据这一假设，我们提出了一种无需进一步训练的简单的对抗性图像净化方法，称为ZeroPur。ZeroPur包含两个步骤：给定一个对抗性样本，引导移位得到对抗性样本的移位嵌入；然后，自适应投影通过这种移位嵌入构造一个方向向量，提供动量，将对抗性图像自适应地投影到流形上。ZeroPur独立于外部模型，不需要重新训练受害者分类器或辅助功能，仅依靠受害者分类器本身实现净化。在三个数据集(CIFAR-10、CIFAR-100和ImageNet-1K)上使用不同的分类器结构(ResNet、WideResNet)进行的大量实验表明，我们的方法获得了最先进的稳健性能。代码将向公众开放。



## **24. VQUNet: Vector Quantization U-Net for Defending Adversarial Atacks by Regularizing Unwanted Noise**

VQUNet：通过规范化不想要的噪音来防御对抗性攻击的载体量化U-Net cs.CV

8 pages, 6 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03117v1) [paper-pdf](http://arxiv.org/pdf/2406.03117v1)

**Authors**: Zhixun He, Mukesh Singhal

**Abstract**: Deep Neural Networks (DNN) have become a promising paradigm when developing Artificial Intelligence (AI) and Machine Learning (ML) applications. However, DNN applications are vulnerable to fake data that are crafted with adversarial attack algorithms. Under adversarial attacks, the prediction accuracy of DNN applications suffers, making them unreliable. In order to defend against adversarial attacks, we introduce a novel noise-reduction procedure, Vector Quantization U-Net (VQUNet), to reduce adversarial noise and reconstruct data with high fidelity. VQUNet features a discrete latent representation learning through a multi-scale hierarchical structure for both noise reduction and data reconstruction. The empirical experiments show that the proposed VQUNet provides better robustness to the target DNN models, and it outperforms other state-of-the-art noise-reduction-based defense methods under various adversarial attacks for both Fashion-MNIST and CIFAR10 datasets. When there is no adversarial attack, the defense method has less than 1% accuracy degradation for both datasets.

摘要: 深度神经网络(DNN)已成为开发人工智能(AI)和机器学习(ML)应用的一个很有前途的范例。然而，DNN应用程序很容易受到用对抗性攻击算法编制的虚假数据的攻击。在敌意攻击下，DNN应用程序的预测精度受到影响，使其不可靠。为了抵抗敌意攻击，我们引入了一种新的降噪过程--矢量量化U-网(VQUNet)，以降低对抗性噪声并重建高保真的数据。VQUNet的特点是通过多尺度分层结构进行离散的潜在表示学习，用于降噪和数据重建。实验表明，VQUNet对目标DNN模型具有更好的鲁棒性，并且在Fashion-MNIST和CIFAR10数据集上的各种对抗攻击下，其性能优于其他基于降噪的防御方法。当没有对抗性攻击时，两个数据集的防御方法的准确率降幅都不到1%。



## **25. Revisiting the Trade-off between Accuracy and Robustness via Weight Distribution of Filters**

通过过滤器的权重分布重新审视准确性和稳健性之间的权衡 cs.CV

Accepted by TPAMI2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2306.03430v4) [paper-pdf](http://arxiv.org/pdf/2306.03430v4)

**Authors**: Xingxing Wei, Shiji Zhao, Bo li

**Abstract**: Adversarial attacks have been proven to be potential threats to Deep Neural Networks (DNNs), and many methods are proposed to defend against adversarial attacks. However, while enhancing the robustness, the clean accuracy will decline to a certain extent, implying a trade-off existed between the accuracy and robustness. In this paper, to meet the trade-off problem, we theoretically explore the underlying reason for the difference of the filters' weight distribution between standard-trained and robust-trained models and then argue that this is an intrinsic property for static neural networks, thus they are difficult to fundamentally improve the accuracy and adversarial robustness at the same time. Based on this analysis, we propose a sample-wise dynamic network architecture named Adversarial Weight-Varied Network (AW-Net), which focuses on dealing with clean and adversarial examples with a "divide and rule" weight strategy. The AW-Net adaptively adjusts the network's weights based on regulation signals generated by an adversarial router, which is directly influenced by the input sample. Benefiting from the dynamic network architecture, clean and adversarial examples can be processed with different network weights, which provides the potential to enhance both accuracy and adversarial robustness. A series of experiments demonstrate that our AW-Net is architecture-friendly to handle both clean and adversarial examples and can achieve better trade-off performance than state-of-the-art robust models.

摘要: 对抗性攻击已被证明是深度神经网络(DNNS)的潜在威胁，并提出了许多方法来防御对抗性攻击。然而，在增强鲁棒性的同时，清洁精度会有一定程度的下降，这意味着在精度和稳健性之间存在着权衡。本文针对这一权衡问题，从理论上探讨了标准训练模型和稳健训练模型在权值分布上存在差异的根本原因，认为这是静态神经网络的固有特性，很难从根本上同时提高网络的精确度和对抗性。在此基础上，提出了一种基于样本的动态网络体系结构AW-Net(AW-Net)，它采用分而治之的权重策略来处理干净的和对抗性的例子。AW-Net根据敌方路由器产生的调整信号自适应地调整网络的权重，该调整信号直接受输入样本的影响。得益于动态网络体系结构，干净的和对抗性的例子可以用不同的网络权重处理，这提供了提高准确性和对抗性健壮性的潜力。一系列的实验表明，我们的AW-Net是体系结构友好的，可以处理干净和对抗性的例子，并且可以获得比最新的健壮模型更好的折衷性能。



## **26. Enhancing the Resilience of Graph Neural Networks to Topological Perturbations in Sparse Graphs**

增强图神经网络对稀疏图中拓扑扰动的弹性 cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03097v1) [paper-pdf](http://arxiv.org/pdf/2406.03097v1)

**Authors**: Shuqi He, Jun Zhuang, Ding Wang, Luyao Peng, Jun Song

**Abstract**: Graph neural networks (GNNs) have been extensively employed in node classification. Nevertheless, recent studies indicate that GNNs are vulnerable to topological perturbations, such as adversarial attacks and edge disruptions. Considerable efforts have been devoted to mitigating these challenges. For example, pioneering Bayesian methodologies, including GraphSS and LlnDT, incorporate Bayesian label transitions and topology-based label sampling to strengthen the robustness of GNNs. However, GraphSS is hindered by slow convergence, while LlnDT faces challenges in sparse graphs. To overcome these limitations, we propose a novel label inference framework, TraTopo, which combines topology-driven label propagation, Bayesian label transitions, and link analysis via random walks. TraTopo significantly surpasses its predecessors on sparse graphs by utilizing random walk sampling, specifically targeting isolated nodes for link prediction, thus enhancing its effectiveness in topological sampling contexts. Additionally, TraTopo employs a shortest-path strategy to refine link prediction, thereby reducing predictive overhead and improving label inference accuracy. Empirical evaluations highlight TraTopo's superiority in node classification, significantly exceeding contemporary GCN models in accuracy.

摘要: 图神经网络在节点分类中得到了广泛的应用。然而，最近的研究表明，GNN容易受到拓扑扰动的影响，例如对抗性攻击和边缘中断。为减轻这些挑战，已经作出了相当大的努力。例如，开创性的贝叶斯方法，包括GraphSS和LlnDT，将贝叶斯标签转换和基于拓扑的标签采样结合在一起，以增强GNN的健壮性。然而，GraphSS的收敛速度较慢，而LinDT在稀疏图中面临着挑战。为了克服这些局限性，我们提出了一种新的标签推理框架TraTopo，该框架结合了拓扑驱动的标签传播、贝叶斯标签转移和通过随机游走进行的链接分析。TraTopo通过使用随机游走抽样，特别是针对孤立节点进行链路预测，在稀疏图上大大超过了它的前辈，从而增强了它在拓扑抽样环境中的有效性。此外，TraTopo采用最短路径策略来改进链路预测，从而减少预测开销并提高标签推理精度。经验评估突出了TraTopo在节点分类方面的优势，在准确率上大大超过了当代的GCN模型。



## **27. On the Duality Between Sharpness-Aware Minimization and Adversarial Training**

论敏锐意识最小化与对抗训练的二元性 cs.LG

ICML 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.15152v2) [paper-pdf](http://arxiv.org/pdf/2402.15152v2)

**Authors**: Yihao Zhang, Hangzhou He, Jingyu Zhu, Huanran Chen, Yifei Wang, Zeming Wei

**Abstract**: Adversarial Training (AT), which adversarially perturb the input samples during training, has been acknowledged as one of the most effective defenses against adversarial attacks, yet suffers from inevitably decreased clean accuracy. Instead of perturbing the samples, Sharpness-Aware Minimization (SAM) perturbs the model weights during training to find a more flat loss landscape and improve generalization. However, as SAM is designed for better clean accuracy, its effectiveness in enhancing adversarial robustness remains unexplored. In this work, considering the duality between SAM and AT, we investigate the adversarial robustness derived from SAM. Intriguingly, we find that using SAM alone can improve adversarial robustness. To understand this unexpected property of SAM, we first provide empirical and theoretical insights into how SAM can implicitly learn more robust features, and conduct comprehensive experiments to show that SAM can improve adversarial robustness notably without sacrificing any clean accuracy, shedding light on the potential of SAM to be a substitute for AT when accuracy comes at a higher priority. Code is available at https://github.com/weizeming/SAM_AT.

摘要: 对抗性训练(AT)在训练过程中对输入样本进行对抗性扰动，已被公认为是对抗对抗性攻击的最有效的防御方法之一，但不可避免地会降低干净的准确率。锐度感知最小化(SAM)不是扰动样本，而是在训练过程中扰动模型权重，以找到更平坦的损失情况，并提高泛化能力。然而，由于SAM是为更好的干净准确性而设计的，其在增强对手稳健性方面的有效性仍未被探索。在这项工作中，考虑到SAM和AT之间的对偶性，我们研究了SAM产生的对抗健壮性。有趣的是，我们发现单独使用SAM可以提高对手的健壮性。为了理解SAM的这种意想不到的特性，我们首先提供了关于SAM如何隐含地学习更健壮的特征的经验和理论见解，并进行了全面的实验，以表明SAM可以在不牺牲任何干净的准确性的情况下显著地提高对手的健壮性，从而揭示了当准确性变得更重要时，SAM成为AT的替代品的潜力。代码可在https://github.com/weizeming/SAM_AT.上找到



## **28. Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections**

你的模型仍然公平吗？通过节点注入对图神经网络的公平性攻击 cs.LG

21 pages

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03052v1) [paper-pdf](http://arxiv.org/pdf/2406.03052v1)

**Authors**: Zihan Luo, Hong Huang, Yongkang Zhou, Jiping Zhang, Nuo Chen

**Abstract**: Despite the remarkable capabilities demonstrated by Graph Neural Networks (GNNs) in graph-related tasks, recent research has revealed the fairness vulnerabilities in GNNs when facing malicious adversarial attacks. However, all existing fairness attacks require manipulating the connectivity between existing nodes, which may be prohibited in reality. To this end, we introduce a Node Injection-based Fairness Attack (NIFA), exploring the vulnerabilities of GNN fairness in such a more realistic setting. In detail, NIFA first designs two insightful principles for node injection operations, namely the uncertainty-maximization principle and homophily-increase principle, and then optimizes injected nodes' feature matrix to further ensure the effectiveness of fairness attacks. Comprehensive experiments on three real-world datasets consistently demonstrate that NIFA can significantly undermine the fairness of mainstream GNNs, even including fairness-aware GNNs, by injecting merely 1% of nodes. We sincerely hope that our work can stimulate increasing attention from researchers on the vulnerability of GNN fairness, and encourage the development of corresponding defense mechanisms.

摘要: 尽管图神经网络(GNN)在与图相关的任务中表现出了卓越的能力，但最近的研究揭示了GNN在面对恶意攻击时的公平性漏洞。然而，所有现有的公平攻击都需要操纵现有节点之间的连通性，这在现实中可能是被禁止的。为此，我们引入了一种基于节点注入的公平攻击(NIFA)，探讨了GNN公平性在这样一个更现实的环境下的脆弱性。具体而言，NIFA首先为节点注入操作设计了两个有洞察力的原则，即不确定性最大化原则和同质性增加原则，然后对注入节点的特征矩阵进行优化，进一步保证了公平攻击的有效性。在三个真实数据集上的综合实验一致表明，NIFA只注入1%的节点，就可以显著破坏主流GNN的公平性，即使包括公平感知的GNN。我们真诚地希望我们的工作能够引起研究人员对GNN公平性脆弱性的越来越多的关注，并鼓励开发相应的防御机制。



## **29. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

SIFER：调查恶意软件检测管道的性能和稳健性 cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.14478v2) [paper-pdf](http://arxiv.org/pdf/2405.14478v2)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Ivan Tesfai Ogbu, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyse samples; and (iii) analysing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we propose SLIFER, a novel Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples resistance to analysis, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER leveraging content-injections attacks, and we show that, counter-intuitively, attacks are blocked more by YARA rules than dynamic analysis due to byte artifacts created while optimizing the adversarial strategy.

摘要: 作为数十年研究的结果，Windows恶意软件检测是通过大量技术实现的。然而，学术界--追求在检测率和低虚警方面的最佳表现--与现实世界场景的要求之间存在着持续的不匹配。特别是，学术界专注于在单个或集成模型中结合静态和动态分析，陷入了几个陷阱，如(I)触发动态分析而不考虑其所需的计算负担；(Ii)丢弃无法分析的样本；以及(Iii)分析针对敌意攻击的稳健性，而不考虑恶意软件检测器补充了更多的非机器学习组件。因此，在本文中，我们提出了一种新颖的Windows恶意软件检测流水线Slifer，它顺序地利用静态和动态分析，在一个模块触发警报时立即中断计算，仅在需要时才需要动态分析。与最新技术相反，我们调查了如何处理抗拒分析的样本，显示了它们对性能的影响程度，得出的结论是，最好将它们标记为合法，不要大幅增加错误警报。最后，我们利用内容注入攻击对Slifer进行了健壮性评估，并且我们表明，与直觉相反，由于优化对抗策略时产生的字节伪影，攻击更多地被Yara规则阻止而不是动态分析。



## **30. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross Domain**

DifAttack++：跨域中通过分层解纠缠特征空间进行查询高效黑匣子对抗攻击 cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03017v1) [paper-pdf](http://arxiv.org/pdf/2406.03017v1)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a \textit{Hierarchical} \textbf{Di}sentangled \textbf{F}eature space and \textit{cross domain}, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an \textit{adversarial feature} (AF) and a \textit{visual feature} (VF) via an autoencoder equipped with our specially designed \textbf{H}ierarchical \textbf{D}ecouple-\textbf{F}usion (HDF) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such autoencoders for the clean and adversarial image domains respectively, meanwhile realizing feature disentanglement, by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our method achieves superior ASR and query efficiency than SOTA methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.

摘要: 研究了基于分数的高效黑盒对抗攻击，具有较高的攻击成功率(ASR)和良好的泛化能力。我们设计了一种新的攻击方法，该方法不同于现有的在整个特征空间上操作的攻击方法，它是基于文本{层次}\extbf{Di}定位的文本空间和文本{跨域}的攻击方法。具体地说，DifAttack++首先通过一个自动编码器将图像的潜在特征分解为文本{对抗性特征}(AF)和文本{视觉特征}(VF)，该自动编码器配备了我们特别设计的\textbf{H}ierarchical\extbf{D}生态耦合(HDF)模块，其中，对抗性特征主导图像的对抗能力，而视觉特征在很大程度上决定了图像的视觉外观。利用已有的代理模型生成的干净图像及其对抗性实例(AE)，通过白盒攻击方法，分别训练干净图像域和对抗性图像域的自动编码器，同时实现特征解缠。最终，在黑盒攻击阶段，DifAttack++根据受害者模型的查询反馈迭代地优化AF，直到生成成功的AE，同时保持VF不变。大量的实验结果表明，与SOTA方法相比，该方法获得了更好的ASR和查询效率，同时也表现出了更好的AEs视觉质量。代码可在https://github.com/csjunjun/DifAttack.git.上获得



## **31. ACE: A Model Poisoning Attack on Contribution Evaluation Methods in Federated Learning**

ACE：联邦学习中贡献评估方法的一种中毒攻击模型 cs.CR

To appear in the 33rd USENIX Security Symposium, 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.20975v2) [paper-pdf](http://arxiv.org/pdf/2405.20975v2)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: In Federated Learning (FL), a set of clients collaboratively train a machine learning model (called global model) without sharing their local training data. The local training data of clients is typically non-i.i.d. and heterogeneous, resulting in varying contributions from individual clients to the final performance of the global model. In response, many contribution evaluation methods were proposed, where the server could evaluate the contribution made by each client and incentivize the high-contributing clients to sustain their long-term participation in FL. Existing studies mainly focus on developing new metrics or algorithms to better measure the contribution of each client. However, the security of contribution evaluation methods of FL operating in adversarial environments is largely unexplored. In this paper, we propose the first model poisoning attack on contribution evaluation methods in FL, termed ACE. Specifically, we show that any malicious client utilizing ACE could manipulate the parameters of its local model such that it is evaluated to have a high contribution by the server, even when its local training data is indeed of low quality. We perform both theoretical analysis and empirical evaluations of ACE. Theoretically, we show our design of ACE can effectively boost the malicious client's perceived contribution when the server employs the widely-used cosine distance metric to measure contribution. Empirically, our results show ACE effectively and efficiently deceive five state-of-the-art contribution evaluation methods. In addition, ACE preserves the accuracy of the final global models on testing inputs. We also explore six countermeasures to defend ACE. Our results show they are inadequate to thwart ACE, highlighting the urgent need for new defenses to safeguard the contribution evaluation methods in FL.

摘要: 在联合学习(FL)中，一组客户协作训练机器学习模型(称为全局模型)，而不共享他们的本地训练数据。客户的本地培训数据通常是非I.I.D.的。和异质性，导致各个客户对全球模型最终绩效的贡献各不相同。作为回应，提出了许多贡献评估方法，其中服务器可以评估每个客户所做的贡献，并激励高贡献的客户维持他们在FL中的长期参与。现有的研究主要集中在开发新的指标或算法，以更好地衡量每个客户的贡献。然而，在对抗环境下运行的FL的贡献评估方法的安全性在很大程度上是未被探索的。在本文中，我们提出了第一个模型中毒攻击的贡献评估方法，称为ACE。具体地说，我们表明，任何使用ACE的恶意客户端都可以操纵其本地模型的参数，从而使服务器评估其具有高贡献，即使其本地训练数据确实质量较低。我们对ACE进行了理论分析和实证评估。理论上，当服务器使用广泛使用的余弦距离度量贡献度时，我们的ACE设计可以有效地提高恶意客户端的感知贡献度。实证结果表明，ACE有效且高效地欺骗了五种最先进的贡献评估方法。此外，ACE在测试输入时保留了最终全局模型的准确性。我们还探讨了防御ACE的六项对策。我们的结果表明，它们不足以阻止ACE，突显了迫切需要新的防御措施来保障FL的贡献评估方法。



## **32. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

做最坏的打算：基于学习的对抗性攻击，用于ICP算法弹性分析 cs.RO

9 pages (7 content, 1 reference, 1 appendix). 6 figures, submitted to  the IEEE Robotics and Automation Letters (RA-L)

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2403.05666v2) [paper-pdf](http://arxiv.org/pdf/2403.05666v2)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial attacks. The proposed attack induces significant pose errors on ICP and outperforms baselines more than 88% of the time across a wide range of scenarios. As an example application, we demonstrate that our attack can be used to identify areas on a map where ICP is particularly vulnerable to corruption in the measurements.

摘要: 提出了一种基于深度学习的激光雷达点云攻击评估迭代最近点算法抗攻击能力的新方法。对于自主导航等安全关键型应用，在部署之前确保算法的弹性是至关重要的。该算法已成为激光雷达定位的标准算法。然而，它产生的姿势估计可能会受到测量中的干扰的很大影响。损坏可能由多种情况引起，例如堵塞、恶劣天气或传感器中的机械问题。不幸的是，比较方案的复杂性和迭代性使评估其对腐败的复原力具有挑战性。虽然已经有人努力创建具有挑战性的数据集和开发仿真来经验地评估ICP的弹性，但我们的方法专注于使用基于扰动的对抗性攻击来寻找最大可能的ICP姿态误差。所提出的攻击在ICP上引起显著的姿势误差，并且在广泛的场景中超过基线的时间超过88%。作为一个示例应用程序，我们演示了我们的攻击可以用来识别地图上的那些区域，在测量中，ICP特别容易受到腐败的影响。



## **33. Nonlinear Transformations Against Unlearnable Datasets**

针对不可学习数据集的非线性转换 cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.02883v1) [paper-pdf](http://arxiv.org/pdf/2406.02883v1)

**Authors**: Thushari Hapuarachchi, Jing Lin, Kaiqi Xiong, Mohamed Rahouti, Gitte Ost

**Abstract**: Automated scraping stands out as a common method for collecting data in deep learning models without the authorization of data owners. Recent studies have begun to tackle the privacy concerns associated with this data collection method. Notable approaches include Deepconfuse, error-minimizing, error-maximizing (also known as adversarial poisoning), Neural Tangent Generalization Attack, synthetic, autoregressive, One-Pixel Shortcut, Self-Ensemble Protection, Entangled Features, Robust Error-Minimizing, Hypocritical, and TensorClog. The data generated by those approaches, called "unlearnable" examples, are prevented "learning" by deep learning models. In this research, we investigate and devise an effective nonlinear transformation framework and conduct extensive experiments to demonstrate that a deep neural network can effectively learn from the data/examples traditionally considered unlearnable produced by the above twelve approaches. The resulting approach improves the ability to break unlearnable data compared to the linear separable technique recently proposed by researchers. Specifically, our extensive experiments show that the improvement ranges from 0.34% to 249.59% for the unlearnable CIFAR10 datasets generated by those twelve data protection approaches, except for One-Pixel Shortcut. Moreover, the proposed framework achieves over 100% improvement of test accuracy for Autoregressive and REM approaches compared to the linear separable technique. Our findings suggest that these approaches are inadequate in preventing unauthorized uses of data in machine learning models. There is an urgent need to develop more robust protection mechanisms that effectively thwart an attacker from accessing data without proper authorization from the owners.

摘要: 自动抓取作为深度学习模型中一种常见的数据收集方法脱颖而出，无需数据所有者的授权。最近的研究已经开始解决与这种数据收集方法相关的隐私问题。值得注意的方法包括深度混淆、错误最小化、错误最大化(也称为对抗性中毒)、神经切线泛化攻击、合成、自回归、单像素捷径、自我集成保护、纠缠特征、健壮错误最小化、伪善和TensorClog。这些方法产生的数据，即所谓的“无法学习”的例子，被深度学习模型阻止“学习”。在这项研究中，我们研究并设计了一个有效的非线性转换框架，并进行了大量的实验，以证明深度神经网络可以从上述12种方法产生的传统上被认为是不可学习的数据/实例中有效地学习。与研究人员最近提出的线性可分离技术相比，由此产生的方法提高了打破不可学习数据的能力。具体地说，我们的广泛实验表明，除了单像素快捷方式之外，这12种数据保护方法生成的不可学习CIFAR10数据集的性能改善幅度在0.34%到249.59%之间。此外，与线性分离技术相比，该框架对自回归和REM方法的测试精度提高了100%以上。我们的发现表明，这些方法在防止机器学习模型中的数据未经授权使用方面是不够的。迫切需要开发更强大的保护机制，有效地阻止攻击者在没有所有者适当授权的情况下访问数据。



## **34. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

通过自我监督学习提高说话人验证的对抗鲁棒性 cs.SD

Accepted by TASLP

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2106.00273v4) [paper-pdf](http://arxiv.org/pdf/2106.00273v4)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstract**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.

摘要: 以前的研究表明，自动说话人验证(ASV)非常容易受到恶意欺骗攻击，如重放、合成语音和最近出现的对抗性攻击。人们一直致力于保护ASV免受重播和合成语音的攻击；然而，只有少数几种方法被探索来应对对抗性攻击。现有的解决ASV对抗性攻击的所有方法都需要对抗性样本生成的知识，但防御者要知道野外攻击者应用的确切攻击算法是不切实际的。这是第一个在不知道具体攻击算法的情况下对ASV进行对抗性防御的工作之一。受自监督学习模型(SSLMS)减少输入表面噪声和从中断样本中重建干净样本的优点的启发，本文将对抗性扰动视为一种噪声，利用SSLMS对ASV进行对抗性防御。具体地说，我们提出从两个角度进行对抗性防御：1)对抗性扰动净化和2)对抗性扰动检测。实验结果表明，该检测模块对恶意样本的检测准确率达到80%左右，有效地屏蔽了ASV。此外，由于ASV的对抗防御性能没有通用的评价指标，本文还考虑了基于净化和基于检测的方法，形式化了对抗防御的评价指标。我们真诚地鼓励今后的工作以拟议的评价框架为基准确定其方法的基准。



## **35. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

针对归纳图神经网络的有效模型窃取攻击 cs.LG

arXiv admin note: text overlap with arXiv:2112.08331 by other authors

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.12295v2) [paper-pdf](http://arxiv.org/pdf/2405.12295v2)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which enable the processing of graph-structured data without relying on predefined graph structures, are gaining importance in an increasingly wide variety of applications. As these networks demonstrate proficiency across a range of tasks, they become lucrative targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. A large effort has been made to develop model-stealing attacks that focus on models trained with images and texts. However, little attention has been paid to GNNs trained on graph data. This paper introduces a novel method for unsupervised model-stealing attacks against inductive GNNs, based on graph contrasting learning and spectral graph augmentations to efficiently extract information from the target model. The proposed attack is thoroughly evaluated on six datasets. The results show that this approach demonstrates a higher level of efficiency compared to existing stealing attacks. More concretely, our attack outperforms the baseline on all benchmarks achieving higher fidelity and downstream accuracy of the stolen model while requiring fewer queries sent to the target model.

摘要: 图神经网络(GNN)被认为是处理以图结构组织的真实世界数据的有力工具。尤其是感应式GNN，它能够在不依赖于预定义的图结构的情况下处理图结构的数据，在越来越广泛的应用中正变得越来越重要。由于这些网络在一系列任务中表现出熟练程度，它们成为窃取模型攻击的有利可图的目标，在这种攻击中，对手试图复制目标网络的功能。已经做出了大量努力来开发窃取模型的攻击，这些攻击集中在使用图像和文本训练的模型上。然而，对以图表数据为基础的全球网络的关注很少。提出了一种基于图对比学习和谱图扩充的非监督模型窃取方法，有效地从目标模型中提取信息。对提出的攻击在六个数据集上进行了彻底的评估。实验结果表明，与现有的窃取攻击相比，该方法具有更高的效率。更具体地说，我们的攻击在所有基准上都超过了基线，实现了被盗模型的更高保真度和下游精度，同时需要发送到目标模型的查询更少。



## **36. Auditing Privacy Mechanisms via Label Inference Attacks**

通过标签推理攻击审计隐私机制 cs.LG

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02797v1) [paper-pdf](http://arxiv.org/pdf/2406.02797v1)

**Authors**: Róbert István Busa-Fekete, Travis Dick, Claudio Gentile, Andrés Muñoz Medina, Adam Smith, Marika Swanberg

**Abstract**: We propose reconstruction advantage measures to audit label privatization mechanisms. A reconstruction advantage measure quantifies the increase in an attacker's ability to infer the true label of an unlabeled example when provided with a private version of the labels in a dataset (e.g., aggregate of labels from different users or noisy labels output by randomized response), compared to an attacker that only observes the feature vectors, but may have prior knowledge of the correlation between features and labels. We consider two such auditing measures: one additive, and one multiplicative. These incorporate previous approaches taken in the literature on empirical auditing and differential privacy. The measures allow us to place a variety of proposed privatization schemes -- some differentially private, some not -- on the same footing. We analyze these measures theoretically under a distributional model which encapsulates reasonable adversarial settings. We also quantify their behavior empirically on real and simulated prediction tasks. Across a range of experimental settings, we find that differentially private schemes dominate or match the privacy-utility tradeoff of more heuristic approaches.

摘要: 提出重构优势措施，对标签民营化机制进行审计。与仅观察特征向量但可能具有特征和标签之间相关性的先验知识的攻击者相比，重建优势量度量化了当向攻击者提供数据集中的标签的私有版本(例如，来自不同用户的标签的集合或通过随机响应输出的噪声标签)时，攻击者推断未标记示例的真实标签的能力的增加。我们考虑了两种这样的审计措施：一种是加法，另一种是乘法。这些方法结合了先前文献中关于经验性审计和差别隐私的方法。这些措施使我们能够将各种拟议的私有化计划--一些是不同的私人计划，另一些不是--置于相同的基础上。我们在一个包含合理对抗性设置的分布模型下对这些措施进行了理论分析。我们还对他们在真实和模拟预测任务中的行为进行了经验上的量化。在一系列实验设置中，我们发现不同的私有方案主导或匹配了更多启发式方法的隐私-效用权衡。



## **37. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

16 pages

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2404.09005v5) [paper-pdf](http://arxiv.org/pdf/2404.09005v5)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别是，我们的工作是安全的，可以抵抗对Jia等人最近的工作的两次攻击。[2021]并将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **38. Tree Proof-of-Position Algorithms**

树位置证明算法 cs.DS

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.06761v2) [paper-pdf](http://arxiv.org/pdf/2405.06761v2)

**Authors**: Aida Manzano Kharman, Pietro Ferraro, Homayoun Hamedmoghadam, Robert Shorten

**Abstract**: We present a novel class of proof-of-position algorithms: Tree-Proof-of-Position (T-PoP). This algorithm is decentralised, collaborative and can be computed in a privacy preserving manner, such that agents do not need to reveal their position publicly. We make no assumptions of honest behaviour in the system, and consider varying ways in which agents may misbehave. Our algorithm is therefore resilient to highly adversarial scenarios. This makes it suitable for a wide class of applications, namely those in which trust in a centralised infrastructure may not be assumed, or high security risk scenarios. Our algorithm has a worst case quadratic runtime, making it suitable for hardware constrained IoT applications. We also provide a mathematical model that summarises T-PoP's performance for varying operating conditions. We then simulate T-PoP's behaviour with a large number of agent-based simulations, which are in complete agreement with our mathematical model, thus demonstrating its validity. T-PoP can achieve high levels of reliability and security by tuning its operating conditions, both in high and low density environments. Finally, we also present a mathematical model to probabilistically detect platooning attacks.

摘要: 提出了一类新的位置证明算法：树位置证明算法(T-POP)。该算法是分散的、协作的，并且可以以保护隐私的方式进行计算，因此代理不需要公开透露他们的位置。我们没有对系统中的诚实行为做出假设，并考虑了代理人可能不当行为的各种方式。因此，我们的算法对高度对抗性的场景具有弹性。这使得它适合于广泛类别的应用程序，即那些可能不信任集中式基础设施的应用程序，或高安全风险场景。该算法具有最坏情况下的二次运行时间，适用于硬件受限的物联网应用。我们还提供了一个数学模型，总结了T-POP在不同工作条件下的性能。然后，我们用大量基于代理的模拟来模拟T-POP的行为，这与我们的数学模型完全一致，从而证明了它的有效性。T-POP可以通过调整其在高密度和低密度环境中的运行条件来实现高水平的可靠性和安全性。最后，我们还给出了一个概率检测排队攻击的数学模型。



## **39. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

重新思考面部识别系统的漏洞：从实践的角度 cs.CR

19 pages,version 2

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.12786v2) [paper-pdf](http://arxiv.org/pdf/2405.12786v2)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Changjiang Li, Ting Wang, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.

摘要: 人脸识别系统(FRS)越来越多地集成到包括监控和用户身份验证在内的关键应用中，突显了它们在现代安全系统中的关键作用。最近的研究发现，FRS对对抗性攻击(例如，对抗性补丁攻击)和后门攻击(例如，训练数据中毒)的脆弱性，引起了人们对其可靠性和可信性的严重担忧。以往的研究主要集中于传统的对抗性攻击或后门攻击，忽略了此类威胁的资源密集型或特权操纵性，从而限制了它们的实用通用性、隐蔽性、普遍性和健壮性。相应地，在本文中，我们通过用户研究和初步探索，深入研究了FRS的固有漏洞。通过利用这些漏洞，我们确定了一种新型的攻击，即面部识别后门攻击，称为FIBA，它揭示了对FRS的一个潜在的更具破坏性的威胁：注册阶段的后门攻击。FIBA绕过了传统攻击的限制，允许任何使用特定触发器的攻击者绕过这些系统，从而实现广泛的破坏。这意味着在将单个有毒示例插入数据库后，相应的触发器将成为任何攻击者欺骗FRS的通用密钥。该策略实质上是通过在注册阶段发起攻击来挑战传统攻击，通过毒化特征数据库而不是训练数据来极大地改变威胁格局。



## **40. Advancing Generalized Transfer Attack with Initialization Derived Bilevel Optimization and Dynamic Sequence Truncation**

利用子索衍生二层优化和动态序列截断推进广义传输攻击 cs.LG

Accepted by IJCAI 2024. 10 pages

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02064v1) [paper-pdf](http://arxiv.org/pdf/2406.02064v1)

**Authors**: Yaohua Liu, Jiaxin Gao, Xuan Liu, Xianghao Jiao, Xin Fan, Risheng Liu

**Abstract**: Transfer attacks generate significant interest for real-world black-box applications by crafting transferable adversarial examples through surrogate models. Whereas, existing works essentially directly optimize the single-level objective w.r.t. the surrogate model, which always leads to poor interpretability of attack mechanism and limited generalization performance over unknown victim models. In this work, we propose the \textbf{B}il\textbf{E}vel \textbf{T}ransfer \textbf{A}ttac\textbf{K} (BETAK) framework by establishing an initialization derived bilevel optimization paradigm, which explicitly reformulates the nested constraint relationship between the Upper-Level (UL) pseudo-victim attacker and the Lower-Level (LL) surrogate attacker. Algorithmically, we introduce the Hyper Gradient Response (HGR) estimation as an effective feedback for the transferability over pseudo-victim attackers, and propose the Dynamic Sequence Truncation (DST) technique to dynamically adjust the back-propagation path for HGR and reduce computational overhead simultaneously. Meanwhile, we conduct detailed algorithmic analysis and provide convergence guarantee to support non-convexity of the LL surrogate attacker. Extensive evaluations demonstrate substantial improvement of BETAK (e.g., $\mathbf{53.41}$\% increase of attack success rates against IncRes-v$2_{ens}$) against different victims and defense methods in targeted and untargeted attack scenarios. The source code is available at https://github.com/callous-youth/BETAK.

摘要: 传输攻击通过代理模型制作可传输的对抗性实例，从而为现实世界的黑盒应用程序产生极大的兴趣。然而，已有的工作基本上直接优化了单层目标w.r.t.代理模型往往导致对攻击机制的可解释性差，对未知受害者模型的泛化性能有限。在这项工作中，我们通过建立一个初始化派生的双层优化范例，显式地重塑了上层(UL)伪受害者攻击者和下层(LL)代理攻击者之间的嵌套约束关系，从而提出了TTAC/TTAC/Textbf{K}(Betak)框架。在算法上，我们引入了超梯度响应(HGR)估计作为对伪受害者攻击者可传递性的有效反馈，并提出了动态序列截断(DST)技术来动态调整HGR的反向传播路径，同时减少了计算开销。同时，我们进行了详细的算法分析，并为支持LL代理攻击的非凸性提供了收敛保证。广泛的评估表明，Betak在目标攻击和非目标攻击场景中针对不同受害者和防御方法的Betak显著改进(例如，对IncRes-v$2_{ens}$的攻击成功率增加)。源代码可在https://github.com/callous-youth/BETAK.上找到



## **41. Graph Adversarial Diffusion Convolution**

图对抗扩散卷积 cs.LG

Accepted by ICML 2024

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02059v1) [paper-pdf](http://arxiv.org/pdf/2406.02059v1)

**Authors**: Songtao Liu, Jinghui Chen, Tianfan Fu, Lu Lin, Marinka Zitnik, Dinghao Wu

**Abstract**: This paper introduces a min-max optimization formulation for the Graph Signal Denoising (GSD) problem. In this formulation, we first maximize the second term of GSD by introducing perturbations to the graph structure based on Laplacian distance and then minimize the overall loss of the GSD. By solving the min-max optimization problem, we derive a new variant of the Graph Diffusion Convolution (GDC) architecture, called Graph Adversarial Diffusion Convolution (GADC). GADC differs from GDC by incorporating an additional term that enhances robustness against adversarial attacks on the graph structure and noise in node features. Moreover, GADC improves the performance of GDC on heterophilic graphs. Extensive experiments demonstrate the effectiveness of GADC across various datasets. Code is available at https://github.com/SongtaoLiu0823/GADC.

摘要: 本文介绍了图形信号去噪（GSD）问题的最小-最大优化公式。在这个公式中，我们首先通过基于拉普拉斯距离对图结构引入扰动来最大化GSD的第二项，然后最大化GSD的总体损失。通过解决最小-最大优化问题，我们推导出图扩散卷积（GDC）架构的一种新变体，称为图对抗扩散卷积（GADC）。GADC与GDC的不同之处在于添加了一个额外的术语，该术语增强了针对图结构和节点特征中的噪音的对抗性攻击的鲁棒性。此外，GADC改进了GDC在异嗜图上的性能。大量实验证明了GADC在各种数据集中的有效性。代码可在https://github.com/SongtaoLiu0823/GADC上获取。



## **42. ASETF: A Novel Method for Jailbreak Attack on LLMs through Translate Suffix Embeddings**

ASTF：一种通过翻译后缀嵌入对LLM进行越狱攻击的新方法 cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.16006v2) [paper-pdf](http://arxiv.org/pdf/2402.16006v2)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. However, similar to traditional text adversarial attacks, this approach, while effective, is limited by the challenge of the discrete tokens. This gradient based discrete optimization attack requires over 100,000 LLM calls, and due to the unreadable of adversarial suffixes, it can be relatively easily penetrated by common defense methods such as perplexity filters. To cope with this challenge, in this paper, we proposes an Adversarial Suffix Embedding Translation Framework (ASETF), aimed at transforming continuous adversarial suffix embeddings into coherent and understandable text. This method greatly reduces the computational overhead during the attack process and helps to automatically generate multiple adversarial samples, which can be used as data to strengthen LLMs security defense. Experimental evaluations were conducted on Llama2, Vicuna, and other prominent LLMs, employing harmful directives sourced from the Advbench dataset. The results indicate that our method significantly reduces the computation time of adversarial suffixes and achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini.

摘要: 大型语言模型(LLM)的安全防御方法仍然有限，因为危险的提示是手动管理到少数已知的攻击类型，无法跟上新兴的变体。最近的研究发现，在有害指令上附加后缀可能会破坏LLMS的防御，并导致危险的输出。然而，与传统的文本对抗性攻击类似，该方法虽然有效，但受到离散令牌挑战的限制。这种基于梯度的离散优化攻击需要超过100,000个LLM调用，并且由于敌意后缀的不可读，它可以相对容易地被困惑过滤器等常见防御方法穿透。为了应对这一挑战，本文提出了一个对抗性后缀嵌入翻译框架(ASETF)，旨在将连续的对抗性后缀嵌入转换成连贯的、可理解的文本。该方法大大减少了攻击过程中的计算开销，有助于自动生成多个对抗性样本，作为加强LLMS安全防御的数据。在Llama2、Vicuna和其他著名的LLM上进行了实验评估，采用了来自Advbench数据集的有害指令。实验结果表明，该方法显著减少了对抗性后缀的计算时间，取得了比现有技术更好的攻击成功率，同时显著提高了提示的文本流畅性。此外，我们的方法可以推广到更广泛的方法来生成可转移的敌意后缀，可以成功地攻击多个LLM，甚至可以攻击黑盒LLM，如ChatGPT和Gemini。



## **43. SVASTIN: Sparse Video Adversarial Attack via Spatio-Temporal Invertible Neural Networks**

SVASTIN：通过时空可逆神经网络的稀疏视频对抗攻击 cs.CV

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01894v1) [paper-pdf](http://arxiv.org/pdf/2406.01894v1)

**Authors**: Yi Pan, Jun-Jie Huang, Zihan Chen, Wentao Zhao, Ziyue Wang

**Abstract**: Robust and imperceptible adversarial video attack is challenging due to the spatial and temporal characteristics of videos. The existing video adversarial attack methods mainly take a gradient-based approach and generate adversarial videos with noticeable perturbations. In this paper, we propose a novel Sparse Adversarial Video Attack via Spatio-Temporal Invertible Neural Networks (SVASTIN) to generate adversarial videos through spatio-temporal feature space information exchanging. It consists of a Guided Target Video Learning (GTVL) module to balance the perturbation budget and optimization speed and a Spatio-Temporal Invertible Neural Network (STIN) module to perform spatio-temporal feature space information exchanging between a source video and the target feature tensor learned by GTVL module. Extensive experiments on UCF-101 and Kinetics-400 demonstrate that our proposed SVASTIN can generate adversarial examples with higher imperceptibility than the state-of-the-art methods with the higher fooling rate. Code is available at \href{https://github.com/Brittany-Chen/SVASTIN}{https://github.com/Brittany-Chen/SVASTIN}.

摘要: 视频的时空特性决定了视频攻击具有很强的健壮性和不可感知性。现有的视频对抗性攻击方法主要采用基于梯度的方法，生成扰动明显的对抗性视频。提出了一种基于时空可逆神经网络的稀疏对抗性视频攻击方法，通过时空特征空间信息交换生成对抗性视频。该算法包括用于平衡扰动预算和优化速度的引导目标视频学习(GTVL)模块和用于在GTVL模块学习的源视频和目标特征张量之间进行时空特征空间信息交换的时空可逆神经网络(STIN)模块。在UCF-101和Kinetics-400上的大量实验表明，与现有的方法相比，该算法可以生成具有更高的不可感知性的对抗性实例，并且具有更高的欺骗率。代码可在\href{https://github.com/Brittany-Chen/SVASTIN}{https://github.com/Brittany-Chen/SVASTIN}.上找到



## **44. Adversarial Attacks on Combinatorial Multi-Armed Bandits**

对组合多臂强盗的对抗攻击 cs.LG

28 pages, Accepted to ICML 2024

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2310.05308v2) [paper-pdf](http://arxiv.org/pdf/2310.05308v2)

**Authors**: Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao

**Abstract**: We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, a notion to capture the vulnerability and robustness of CMAB. The attackability condition depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.

摘要: 研究了组合多臂土匪(CMAB)的悬赏中毒攻击。我们首先给出了CMAB可攻击性的充要条件，这是一种捕捉CMAB脆弱性和健壮性的概念。可攻击性条件取决于相应CMAB实例的内在属性，如超级武器的奖励分布和基础武器的结果分布。此外，我们还设计了一个针对可攻击CMAB实例的攻击算法。与之前对多武装强盗的理解相反，我们的工作揭示了一个令人惊讶的事实，即特定CMAB实例的可攻击性还取决于对手是否知道该强盗实例。这一发现表明，对CMAB的对抗性攻击在实践中是困难的，并且不存在针对任何CMAB实例的通用攻击策略，因为对手基本上不知道环境。通过在概率最大覆盖问题、在线最小生成树、在线排名的级联强盗和在线最短路径等实际CMAB应用上的大量实验，我们验证了我们的理论发现。



## **45. Reproducibility Study on Adversarial Attacks Against Robust Transformer Trackers**

针对鲁棒Transformer跟踪器的对抗性攻击的再现性研究 cs.CV

Published in Transactions on Machine Learning Research (05/2024):  https://openreview.net/forum?id=FEEKR0Vl9s

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01765v1) [paper-pdf](http://arxiv.org/pdf/2406.01765v1)

**Authors**: Fatemeh Nourilenjan Nokabadi, Jean-François Lalonde, Christian Gagné

**Abstract**: New transformer networks have been integrated into object tracking pipelines and have demonstrated strong performance on the latest benchmarks. This paper focuses on understanding how transformer trackers behave under adversarial attacks and how different attacks perform on tracking datasets as their parameters change. We conducted a series of experiments to evaluate the effectiveness of existing adversarial attacks on object trackers with transformer and non-transformer backbones. We experimented on 7 different trackers, including 3 that are transformer-based, and 4 which leverage other architectures. These trackers are tested against 4 recent attack methods to assess their performance and robustness on VOT2022ST, UAV123 and GOT10k datasets. Our empirical study focuses on evaluating adversarial robustness of object trackers based on bounding box versus binary mask predictions, and attack methods at different levels of perturbations. Interestingly, our study found that altering the perturbation level may not significantly affect the overall object tracking results after the attack. Similarly, the sparsity and imperceptibility of the attack perturbations may remain stable against perturbation level shifts. By applying a specific attack on all transformer trackers, we show that new transformer trackers having a stronger cross-attention modeling achieve a greater adversarial robustness on tracking datasets, such as VOT2022ST and GOT10k. Our results also indicate the necessity for new attack methods to effectively tackle the latest types of transformer trackers. The codes necessary to reproduce this study are available at https://github.com/fatemehN/ReproducibilityStudy.

摘要: 新的变压器网络已集成到目标跟踪管道中，并在最新基准中表现出强劲的性能。本文的重点是了解变压器跟踪器在对抗性攻击下的行为，以及不同的攻击在跟踪数据集的参数变化时如何执行。我们进行了一系列实验，以评估现有的对抗性攻击对具有变压器和非变压器骨干的对象跟踪器的有效性。我们在7个不同的跟踪器上进行了实验，其中3个是基于转换器的，4个是利用其他架构的。这些跟踪器针对最近的4种攻击方法进行了测试，以评估它们在VOT2022ST、UAV123和GOT10k数据集上的性能和健壮性。我们的实证研究集中于评估基于包围盒和二进制掩码预测的对象跟踪器的攻击健壮性，以及在不同扰动级别下的攻击方法。有趣的是，我们的研究发现，改变扰动级别可能不会显著影响攻击后的整体目标跟踪结果。类似地，攻击扰动的稀疏性和不可感知性可以针对扰动级别变化保持稳定。通过对所有变压器跟踪器的攻击，我们证明了新的变压器跟踪器具有更强的交叉注意建模能力，在VOT2022ST和GOT10k等跟踪数据集上实现了更强的对抗鲁棒性。我们的结果还表明，需要新的攻击方法来有效地对付最新类型的变压器跟踪器。复制这项研究所需的代码可在https://github.com/fatemehN/ReproducibilityStudy.上获得



## **46. On the completeness of several fortification-interdiction games in the Polynomial Hierarchy**

关于多项等级中几个防御-拦截游戏的完整性 cs.CC

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01756v1) [paper-pdf](http://arxiv.org/pdf/2406.01756v1)

**Authors**: Alberto Boggio Tomasaz, Margarida Carvalho, Roberto Cordone, Pierre Hosteins

**Abstract**: Fortification-interdiction games are tri-level adversarial games where two opponents act in succession to protect, disrupt and simply use an infrastructure for a specific purpose. Many such games have been formulated and tackled in the literature through specific algorithmic methods, however very few investigations exist on the completeness of such fortification problems in order to locate them rigorously in the polynomial hierarchy. We clarify the completeness status of several well-known fortification problems, such as the Tri-level Interdiction Knapsack Problem with unit fortification and attack weights, the Max-flow Interdiction Problem and Shortest Path Interdiction Problem with Fortification, the Multi-level Critical Node Problem with unit weights, as well as a well-studied electric grid defence planning problem. For all of these problems, we prove their completeness either for the $\Sigma^p_2$ or the $\Sigma^p_3$ class of the polynomial hierarchy. We also prove that the Multi-level Fortification-Interdiction Knapsack Problem with an arbitrary number of protection and interdiction rounds and unit fortification and attack weights is complete for any level of the polynomial hierarchy, therefore providing a useful basis for further attempts at proving the completeness of protection-interdiction games at any level of said hierarchy.

摘要: 防御工事-拦截游戏是三级对抗性游戏，其中两个对手连续行动以保护、扰乱和简单地将基础设施用于特定目的。许多这样的对策在文献中都是通过特定的算法方法来描述和解决的，然而，很少有人研究这种防御问题的完备性，以便在多项式层次中严格地定位它们。阐明了几个著名的防御工事问题的完备性状况，如具有单位防御工事和攻击权重的三级封锁背包问题、具有防御工事的最大流封锁问题和最短路径封锁问题、具有单位权重的多水平关键节点问题以及一个研究得很好的电网防御规划问题。对于所有这些问题，我们证明了它们对于多项式族的$\Sigma^p_2$或$\Sigma^p_3$类的完备性。我们还证明了具有任意多项式轮数和单位防御权和攻击权的多层防御工事背包问题对多项式层次的任何一层都是完备的，从而为进一步证明该多项式族的任何一层上的防御工事对策的完备性提供了有用的依据。



## **47. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

MAWSEO：对抗性的维基搜索毒害非法在线促销 cs.CR

Accepted at the 45th IEEE Symposium on Security and Privacy (IEEE S&P  2024)

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2304.11300v3) [paper-pdf](http://arxiv.org/pdf/2304.11300v3)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is capable of effectively and efficiently generating adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.

摘要: 作为破坏编辑的一个突出例子，非法推广的维基搜索中毒是一种网络犯罪，对手旨在编辑维基文章，通过相关查询的维基搜索结果来推广非法业务。在这篇文章中，我们报告了一项研究，首次表明维基上这种隐蔽的黑帽SEO可以自动化。我们的技术，称为MAWSEO，使用对抗性修订来实现现实世界的网络犯罪目标，包括排名提升、破坏检测规避、主题相关性、语义一致性、用户对促销内容的感知(但不令人震惊)等。我们的评估和用户研究表明，MAWSEO能够有效和高效地生成对抗性破坏编辑，这可以绕过最先进的内置维基破坏检测器，还可以在不触发维基用户警报的情况下将宣传内容传递给维基用户。此外，我们调查了针对我们在维基生态系统中的攻击的潜在防御，包括基于一致性的检测和恶意检测的对抗性培训。



## **48. Mixing Classifiers to Alleviate the Accuracy-Robustness Trade-Off**

混合分类器以减轻准确性与稳健性的权衡 cs.LG

arXiv admin note: text overlap with arXiv:2301.12554

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2311.15165v2) [paper-pdf](http://arxiv.org/pdf/2311.15165v2)

**Authors**: Yatong Bai, Brendon G. Anderson, Somayeh Sojoudi

**Abstract**: Deep neural classifiers have recently found tremendous success in data-driven control systems. However, existing models suffer from a trade-off between accuracy and adversarial robustness. This limitation must be overcome in the control of safety-critical systems that require both high performance and rigorous robustness guarantees. In this work, we develop classifiers that simultaneously inherit high robustness from robust models and high accuracy from standard models. Specifically, we propose a theoretically motivated formulation that mixes the output probabilities of a standard neural network and a robust neural network. Both base classifiers are pre-trained, and thus our method does not require additional training. Our numerical experiments verify that the mixed classifier noticeably improves the accuracy-robustness trade-off and identify the confidence property of the robust base classifier as the key leverage of this more benign trade-off. Our theoretical results prove that under mild assumptions, when the robustness of the robust base model is certifiable, no alteration or attack within a closed-form $\ell_p$ radius on an input can result in the misclassification of the mixed classifier.

摘要: 深度神经分类器最近在数据驱动的控制系统中取得了巨大的成功。然而，现有的模型在准确性和对抗性稳健性之间进行了权衡。在需要高性能和严格的稳健性保证的安全关键系统的控制中，必须克服这一限制。在这项工作中，我们开发的分类器同时继承了稳健模型的高鲁棒性和标准模型的高精度。具体地说，我们提出了一个理论激励的公式，它混合了标准神经网络和稳健神经网络的输出概率。这两个基本分类器都是预先训练的，因此我们的方法不需要额外的训练。我们的数值实验证明，混合分类器显著改善了准确率和稳健性之间的权衡，并将稳健基分类器的置信度属性识别为这种更良性权衡的关键杠杆。我们的理论结果证明，在较温和的假设下，当稳健基模型的稳健性是可证明的时，在闭合形式的$\ell_p$半径内对输入的任何改变或攻击都不会导致混合分类器的错误分类。



## **49. Model for Peanuts: Hijacking ML Models without Training Access is Possible**

花生模型：在没有培训访问权限的情况下劫持ML模型是可能的 cs.CR

17 pages, 14 figures, 7 tables

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01708v1) [paper-pdf](http://arxiv.org/pdf/2406.01708v1)

**Authors**: Mahmoud Ghorbel, Halima Bouzidi, Ioan Marius Bilasco, Ihsen Alouani

**Abstract**: The massive deployment of Machine Learning (ML) models has been accompanied by the emergence of several attacks that threaten their trustworthiness and raise ethical and societal concerns such as invasion of privacy, discrimination risks, and lack of accountability. Model hijacking is one of these attacks, where the adversary aims to hijack a victim model to execute a different task than its original one. Model hijacking can cause accountability and security risks since a hijacked model owner can be framed for having their model offering illegal or unethical services. Prior state-of-the-art works consider model hijacking as a training time attack, whereby an adversary requires access to the ML model training to execute their attack. In this paper, we consider a stronger threat model where the attacker has no access to the training phase of the victim model. Our intuition is that ML models, typically over-parameterized, might (unintentionally) learn more than the intended task for they are trained. We propose a simple approach for model hijacking at inference time named SnatchML to classify unknown input samples using distance measures in the latent space of the victim model to previously known samples associated with the hijacking task classes. SnatchML empirically shows that benign pre-trained models can execute tasks that are semantically related to the initial task. Surprisingly, this can be true even for hijacking tasks unrelated to the original task. We also explore different methods to mitigate this risk. We first propose a novel approach we call meta-unlearning, designed to help the model unlearn a potentially malicious task while training on the original task dataset. We also provide insights on over-parameterization as one possible inherent factor that makes model hijacking easier, and we accordingly propose a compression-based countermeasure against this attack.

摘要: 机器学习(ML)模型的大规模部署伴随着几种攻击的出现，这些攻击威胁到它们的可信度，并引发了诸如侵犯隐私、歧视风险和缺乏问责等伦理和社会问题。模型劫持是其中的一种攻击，对手的目标是劫持受害者模型，以执行与其原始任务不同的任务。劫持模特可能会导致责任追究和安全风险，因为被劫持的模特所有者可能会因为让他们的模特提供非法或不道德的服务而被陷害。以往的研究将模型劫持视为训练时间攻击，对手需要访问ML模型训练才能执行他们的攻击。在本文中，我们考虑了一个更强的威胁模型，其中攻击者无权访问受害者模型的训练阶段。我们的直觉是，ML模型，通常是过度参数化的，可能(无意中)学到比它们被训练的预期任务更多的东西。我们提出了一种简单的推理时模型劫持方法SnatchML，该方法利用受害者模型潜在空间中的距离度量将未知输入样本与与劫持任务类相关的已知样本进行分类。SnatchML的经验表明，良性的预训练模型可以执行与初始任务语义相关的任务。令人惊讶的是，即使对于与原始任务无关的劫持任务，这也是正确的。我们还探索了缓解这一风险的不同方法。我们首先提出了一种称为元遗忘的新方法，旨在帮助该模型在对原始任务数据集进行训练时忘记潜在的恶意任务。我们还对过度参数化作为使模型劫持变得更容易的一个可能的内在因素提出了见解，并相应地提出了针对这种攻击的基于压缩的对策。



## **50. From Feature Visualization to Visual Circuits: Effect of Adversarial Model Manipulation**

从特征可视化到视觉电路：对抗模型操纵的影响 cs.CV

Under review

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01365v1) [paper-pdf](http://arxiv.org/pdf/2406.01365v1)

**Authors**: Geraldin Nanfack, Michael Eickenberg, Eugene Belilovsky

**Abstract**: Understanding the inner working functionality of large-scale deep neural networks is challenging yet crucial in several high-stakes applications. Mechanistic inter- pretability is an emergent field that tackles this challenge, often by identifying human-understandable subgraphs in deep neural networks known as circuits. In vision-pretrained models, these subgraphs are usually interpreted by visualizing their node features through a popular technique called feature visualization. Recent works have analyzed the stability of different feature visualization types under the adversarial model manipulation framework. This paper starts by addressing limitations in existing works by proposing a novel attack called ProxPulse that simultaneously manipulates the two types of feature visualizations. Surprisingly, when analyzing these attacks under the umbrella of visual circuits, we find that visual circuits show some robustness to ProxPulse. We, therefore, introduce a new attack based on ProxPulse that unveils the manipulability of visual circuits, shedding light on their lack of robustness. The effectiveness of these attacks is validated using pre-trained AlexNet and ResNet-50 models on ImageNet.

摘要: 在一些高风险的应用中，理解大规模深度神经网络的内部工作功能是具有挑战性的，但也是至关重要的。机械互易性是解决这一挑战的一个新兴领域，通常通过在被称为电路的深层神经网络中识别人类可理解的子图来实现。在视觉预先训练的模型中，这些子图通常是通过一种称为特征可视化的流行技术来可视化其节点特征来解释的。最近的工作分析了对抗性模型操纵框架下不同特征可视化类型的稳定性。本文从解决现有工作的局限性入手，提出了一种称为ProxPulse的新型攻击方法，该方法可以同时处理两种类型的特征可视化。令人惊讶的是，当我们在视觉电路的保护伞下分析这些攻击时，我们发现视觉电路对ProxPulse表现出一定的稳健性。因此，我们引入了一种基于ProxPulse的新攻击，它揭示了视觉电路的可操纵性，暴露了它们缺乏健壮性。在ImageNet上使用预先训练好的AlexNet和ResNet-50模型验证了这些攻击的有效性。



