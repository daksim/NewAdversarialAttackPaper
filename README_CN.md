# Latest Adversarial Attack Papers
**update at 2024-10-16 11:23:06**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11782v1) [paper-pdf](http://arxiv.org/pdf/2410.11782v1)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **2. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

Phantom：对检索增强语言生成的通用触发攻击 cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.20485v2) [paper-pdf](http://arxiv.org/pdf/2405.20485v2)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama, and show that they transfer to GPT-3.5 Turbo and GPT-4. Finally, we successfully conducted a Phantom attack on NVIDIA's black-box production RAG system, "Chat with RTX".

摘要: 检索增强生成(RAG)通过锚定、调整和个性化对最相关的知识源的响应来扩展现代大型语言模型(LLMS)的能力。它在聊天机器人应用程序中特别有用，允许开发人员定制LLM输出，而无需昂贵的再培训。尽管RAG系统在各种应用中具有重要的实用价值，但它带来了新的安全风险。在这项工作中，我们提出了新的攻击向量，允许攻击者将单个恶意文档注入RAG系统的知识库，并发动后门中毒攻击。我们设计了Phantom，这是一个针对RAG系统的通用两阶段优化框架，它手工制作了一个恶意中毒文档，导致模型输出中的完整性破坏。首先，文档被构建为仅在受害者的查询中出现特定的令牌触发序列时才检索。其次，通过精心设计的敌意文本进一步优化了文档，这些文本在LLM输出上诱导了各种敌意目标，包括拒绝回答、声誉损害、侵犯隐私和有害行为。我们演示了我们对多个LLM体系结构的攻击，包括Gema、Vicuna和Llama，并表明它们可以传输到GPT-3.5Turbo和GPT-4。最后，我们成功地对NVIDIA的黑匣子生产RAG系统“与腾讯通聊天”进行了幻影攻击。



## **3. Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor**

通过注入主动防御后门来缓解后门攻击 cs.CR

Accepted by NeurIPS 2024. 32 pages, 7 figures, 28 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2405.16112v2) [paper-pdf](http://arxiv.org/pdf/2405.16112v2)

**Authors**: Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Data-poisoning backdoor attacks are serious security threats to machine learning models, where an adversary can manipulate the training dataset to inject backdoors into models. In this paper, we focus on in-training backdoor defense, aiming to train a clean model even when the dataset may be potentially poisoned. Unlike most existing methods that primarily detect and remove/unlearn suspicious samples to mitigate malicious backdoor attacks, we propose a novel defense approach called PDB (Proactive Defensive Backdoor). Specifically, PDB leverages the home-field advantage of defenders by proactively injecting a defensive backdoor into the model during training. Taking advantage of controlling the training process, the defensive backdoor is designed to suppress the malicious backdoor effectively while remaining secret to attackers. In addition, we introduce a reversible mapping to determine the defensive target label. During inference, PDB embeds a defensive trigger in the inputs and reverses the model's prediction, suppressing malicious backdoor and ensuring the model's utility on the original task. Experimental results across various datasets and models demonstrate that our approach achieves state-of-the-art defense performance against a wide range of backdoor attacks. The code is available at https://github.com/shawkui/Proactive_Defensive_Backdoor.

摘要: 数据中毒后门攻击是对机器学习模型的严重安全威胁，攻击者可以操纵训练数据集向模型注入后门。在本文中，我们将重点放在训练中的后门防御上，目的是训练一个干净的模型，即使数据集可能被毒化。不同于现有的大多数方法主要是检测和删除/取消学习可疑样本来缓解恶意后门攻击，我们提出了一种称为主动防御后门的新防御方法。具体地说，PDB通过在训练期间主动向模型中注入防守后门来利用后卫的主场优势。利用控制训练过程的优势，防御性后门被设计成在对攻击者保密的同时有效地抑制恶意后门。此外，我们还引入了一种可逆映射来确定防御目标标签。在推理过程中，PDB在输入中嵌入一个防御触发器，逆转模型的预测，抑制恶意后门，确保模型在原始任务上的实用性。在不同的数据集和模型上的实验结果表明，我们的方法在抵抗广泛的后门攻击时获得了最先进的防御性能。代码可在https://github.com/shawkui/Proactive_Defensive_Backdoor.上获得



## **4. GSE: Group-wise Sparse and Explainable Adversarial Attacks**

GSE：分组稀疏和可解释的对抗性攻击 cs.CV

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2311.17434v2) [paper-pdf](http://arxiv.org/pdf/2311.17434v2)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2-$quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2-$norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved explainability, and a $100\%$ attack success rate.

摘要: 稀疏敌意攻击通过最小的像素扰动来欺骗深度神经网络(DNN)，这种扰动通常由$\ell_0$范数来正则化。最近的努力已经用结构稀疏性正则化规则取代了这一规范，例如核集团规范，以制定群组稀疏对抗性攻击。因此，由此产生的扰动是可以解释的，并具有重要的实际意义，揭示了DNN更大的脆弱性。然而，精心设计这样的攻击构成了一个优化挑战，因为它涉及到计算非凸目标内的像素组的规范。我们通过提出一个两阶段算法来解决这个问题，该算法在图像的语义有意义的区域内生成分组稀疏攻击。首先，我们使用为非凸规划量身定做的$1/2-$拟正态近似算子来优化拟正态对抗性损失。随后，算法过渡到投影的内斯特罗夫加速梯度下降，并对摄动幅度应用$2-$范数正则化。在CIFAR-10和ImageNet数据集上的严格评估表明，组内稀疏性显著增加，例如，CIFAR-10上的稀疏度为50.9美元，ImageNet上的稀疏度为38.4美元(平均案例，有针对性的攻击)。伴随着这种性能改进的是显著更快的计算时间、更好的可解释性以及$100\$攻击成功率。



## **5. Efficient and Effective Universal Adversarial Attack against Vision-Language Pre-training Models**

针对视觉语言预训练模型的高效且有效的通用对抗攻击 cs.CV

11 pages

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11639v1) [paper-pdf](http://arxiv.org/pdf/2410.11639v1)

**Authors**: Fan Yang, Yihao Huang, Kailong Wang, Ling Shi, Geguang Pu, Yang Liu, Haoyu Wang

**Abstract**: Vision-language pre-training (VLP) models, trained on large-scale image-text pairs, have become widely used across a variety of downstream vision-and-language (V+L) tasks. This widespread adoption raises concerns about their vulnerability to adversarial attacks. Non-universal adversarial attacks, while effective, are often impractical for real-time online applications due to their high computational demands per data instance. Recently, universal adversarial perturbations (UAPs) have been introduced as a solution, but existing generator-based UAP methods are significantly time-consuming. To overcome the limitation, we propose a direct optimization-based UAP approach, termed DO-UAP, which significantly reduces resource consumption while maintaining high attack performance. Specifically, we explore the necessity of multimodal loss design and introduce a useful data augmentation strategy. Extensive experiments conducted on three benchmark VLP datasets, six popular VLP models, and three classical downstream tasks demonstrate the efficiency and effectiveness of DO-UAP. Specifically, our approach drastically decreases the time consumption by 23-fold while achieving a better attack performance.

摘要: 视觉-语言预训练模型是在大规模图文对上训练的，已被广泛应用于各种下游视觉与语言(V+L)任务。这种广泛的采用引起了人们对它们易受对手攻击的担忧。非通用对抗性攻击虽然有效，但对于实时在线应用程序来说往往是不切实际的，因为它们对每个数据实例的计算要求很高。最近，通用对抗扰动(UAP)被引入作为解决方案，但现有的基于生成器的UAP方法非常耗时。为了克服这一局限性，我们提出了一种基于直接优化的UAP方法，称为DO-UAP，它在保持高攻击性能的同时显著减少了资源消耗。具体地说，我们探讨了多峰损失设计的必要性，并介绍了一种有用的数据增强策略。在三个基准VLP数据集、六个流行的VLP模型和三个经典下游任务上的广泛实验证明了DO-UAP的效率和有效性。具体地说，我们的方法大大减少了23倍的时间消耗，同时实现了更好的攻击性能。



## **6. Information Importance-Aware Defense against Adversarial Attack for Automatic Modulation Classification:An XAI-Based Approach**

自动调制分类的信息重要性感知对抗攻击防御：一种基于XAI的方法 eess.SP

Accepted by WCSP 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11608v1) [paper-pdf](http://arxiv.org/pdf/2410.11608v1)

**Authors**: Jingchun Wang, Peihao Dong, Fuhui Zhou, Qihui Wu

**Abstract**: Deep learning (DL) has significantly improved automatic modulation classification (AMC) by leveraging neural networks as the feature extractor.However, as the DL-based AMC becomes increasingly widespread, it is faced with the severe secure issue from various adversarial attacks. Existing defense methods often suffer from the high computational cost, intractable parameter tuning, and insufficient robustness.This paper proposes an eXplainable artificial intelligence (XAI) defense approach, which uncovers the negative information caused by the adversarial attack through measuring the importance of input features based on the SHapley Additive exPlanations (SHAP).By properly removing the negative information in adversarial samples and then fine-tuning(FT) the model, the impact of the attacks on the classification result can be mitigated.Experimental results demonstrate that the proposed SHAP-FT improves the classification performance of the model by 15%-20% under different attack levels,which not only enhances model robustness against various attack levels but also reduces the resource consumption, validating its effectiveness in safeguarding communication networks.

摘要: 深度学习利用神经网络作为特征提取工具，极大地改善了自动调制分类算法的性能，但随着基于深度学习的自动调制分类算法的应用越来越广泛，它也面临着严峻的安全问题。针对现有防御方法计算量大、参数整定困难、鲁棒性不足等问题，提出了一种基于Shapley附加解释(Shap)的可解释人工智能(XAI)防御方法。该方法基于Shapley附加解释(Shap)度量输入特征的重要性来揭示敌方攻击带来的负面信息，通过适当去除敌方样本中的负面信息并对模型进行微调，可以缓解攻击对分类结果的影响。实验结果表明，在不同攻击级别下，Shap-FT使模型的分类性能提高了15%-20%。这不仅增强了模型对各种攻击级别的健壮性，而且降低了资源消耗，验证了其在保护通信网络方面的有效性。



## **7. RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation**

RAUCA：通过稳健而准确的伪装生成对车辆检测器的新型物理对抗攻击 cs.CV

12 pages. In Proceedings of the Forty-first International Conference  on Machine Learning (ICML), Vienna, Austria, July 21-27, 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2402.15853v2) [paper-pdf](http://arxiv.org/pdf/2402.15853v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle, resulting in suboptimal attack performance. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, Neural Renderer Plus (NRP), which can accurately project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA consistently outperforms existing methods in both simulation and real-world settings.

摘要: 对抗伪装是一种广泛使用的针对车辆探测器的物理攻击，具有多视点攻击性能的优势。一种有希望的方法包括使用可微神经呈现器通过梯度反向传播来促进对抗性伪装优化。然而，现有的方法往往难以捕捉渲染过程中的环境特征，或者生成能够精确映射到目标车辆的对抗性纹理，导致攻击性能次优。此外，这些方法忽略了不同的天气条件，降低了在不同天气情况下产生的伪装效果。为了应对这些挑战，我们提出了一种健壮而准确的伪装生成方法，即Ruca。Ruca的核心是一个新的神经渲染组件-神经渲染器Plus(NRP)，它可以准确地投影车辆纹理，并渲染具有照明和天气等环境特征的图像。此外，我们还集成了一个用于伪装生成的多天气数据集，利用NRP来增强攻击的健壮性。在六个流行的目标探测器上的实验结果表明，无论是在模拟环境中还是在现实世界中，Ruca的性能都一致优于现有的方法。



## **8. Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation**

破译混乱：通过对抗性提示翻译增强越狱攻击 cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11317v1) [paper-pdf](http://arxiv.org/pdf/2410.11317v1)

**Authors**: Qizhang Li, Xiaochen Yang, Wangmeng Zuo, Yiwen Guo

**Abstract**: Automatic adversarial prompt generation provides remarkable success in jailbreaking safely-aligned large language models (LLMs). Existing gradient-based attacks, while demonstrating outstanding performance in jailbreaking white-box LLMs, often generate garbled adversarial prompts with chaotic appearance. These adversarial prompts are difficult to transfer to other LLMs, hindering their performance in attacking unknown victim models. In this paper, for the first time, we delve into the semantic meaning embedded in garbled adversarial prompts and propose a novel method that "translates" them into coherent and human-readable natural language adversarial prompts. In this way, we can effectively uncover the semantic information that triggers vulnerabilities of the model and unambiguously transfer it to the victim model, without overlooking the adversarial information hidden in the garbled text, to enhance jailbreak attacks. It also offers a new approach to discovering effective designs for jailbreak prompts, advancing the understanding of jailbreak attacks. Experimental results demonstrate that our method significantly improves the success rate of jailbreak attacks against various safety-aligned LLMs and outperforms state-of-the-arts by large margins. With at most 10 queries, our method achieves an average attack success rate of 81.8% in attacking 7 commercial closed-source LLMs, including GPT and Claude-3 series, on HarmBench. Our method also achieves over 90% attack success rates against Llama-2-Chat models on AdvBench, despite their outstanding resistance to jailbreak attacks. Code at: https://github.com/qizhangli/Adversarial-Prompt-Translator.

摘要: 自动对抗性提示生成在越狱安全对齐的大型语言模型(LLM)方面取得了显着的成功。现有的基于梯度的攻击虽然在越狱白盒LLM中表现出出色的性能，但往往会产生外观混乱的乱码对抗性提示。这些对抗性提示很难转移到其他LLM上，阻碍了它们在攻击未知受害者模型时的表现。在本文中，我们首次深入研究了混淆的对抗性提示中所蕴含的语义，并提出了一种新的方法，将它们“翻译”成连贯的、人类可读的自然语言对抗性提示。通过这种方式，我们可以有效地发现触发模型漏洞的语义信息，并毫不含糊地将其传递给受害者模型，而不会忽视隐藏在乱码文本中的对抗性信息，以增强越狱攻击。它还提供了一种新的方法来发现有效的越狱提示设计，促进了对越狱攻击的理解。实验结果表明，我们的方法显著提高了对各种安全对齐LLM的越狱攻击成功率，并且远远超过了最新的技术水平。在最多10个查询的情况下，我们的方法在HarmBch上攻击包括GPT和Claude-3系列在内的7个商业闭源LLM，平均攻击成功率为81.8%。我们的方法对AdvBtch上的Llama-2-Chat模型的攻击成功率也达到了90%以上，尽管它们对越狱攻击具有出色的抵抗力。代码：https://github.com/qizhangli/Adversarial-Prompt-Translator.



## **9. On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning**

关于测试时间适应的对抗风险：对现实测试时间数据中毒的调查 cs.LG

19 pages, 4 figures, 8 tables

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.04682v2) [paper-pdf](http://arxiv.org/pdf/2410.04682v2)

**Authors**: Yongyi Su, Yushu Li, Nanqing Liu, Kui Jia, Xulei Yang, Chuan-Sheng Foo, Xun Xu

**Abstract**: Test-time adaptation (TTA) updates the model weights during the inference stage using testing data to enhance generalization. However, this practice exposes TTA to adversarial risks. Existing studies have shown that when TTA is updated with crafted adversarial test samples, also known as test-time poisoned data, the performance on benign samples can deteriorate. Nonetheless, the perceived adversarial risk may be overstated if the poisoned data is generated under overly strong assumptions. In this work, we first review realistic assumptions for test-time data poisoning, including white-box versus grey-box attacks, access to benign data, attack budget, and more. We then propose an effective and realistic attack method that better produces poisoned samples without access to benign samples, and derive an effective in-distribution attack objective. We also design two TTA-aware attack objectives. Our benchmarks of existing attack methods reveal that the TTA methods are more robust than previously believed. In addition, we analyze effective defense strategies to help develop adversarially robust TTA methods.

摘要: 测试时间自适应(TTA)在推理阶段使用测试数据更新模型权重，以增强泛化能力。然而，这种做法使TTA面临对抗性风险。现有研究表明，当使用精心编制的对抗性测试样本(也称为测试时间中毒数据)更新TTA时，良性样本的性能可能会恶化。然而，如果有毒数据是在过于强烈的假设下产生的，那么感知到的对抗性风险可能被夸大了。在这项工作中，我们首先回顾测试时间数据中毒的现实假设，包括白盒攻击与灰盒攻击、对良性数据的访问、攻击预算等。然后，我们提出了一种有效且现实的攻击方法，在不访问良性样本的情况下更好地产生有毒样本，并推导出有效的分布内攻击目标。我们还设计了两个TTA感知攻击目标。我们对现有攻击方法的基准测试表明，TTA方法比之前认为的更健壮。此外，我们分析了有效的防御策略，以帮助开发对抗健壮的TTA方法。



## **10. BRC20 Pinning Attack**

BRRC 20钉扎攻击 cs.CR

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11295v1) [paper-pdf](http://arxiv.org/pdf/2410.11295v1)

**Authors**: Minfeng Qi, Qin Wang, Zhipeng Wang, Lin Zhong, Tianqing Zhu, Shiping Chen, William Knottenbelt

**Abstract**: BRC20 tokens are a type of non-fungible asset on the Bitcoin network. They allow users to embed customized content within Bitcoin satoshis. The related token frenzy has reached a market size of USD 3,650b over the past year (2023Q3-2024Q3). However, this intuitive design has not undergone serious security scrutiny.   We present the first in-depth analysis of the BRC20 transfer mechanism and identify a critical attack vector. A typical BRC20 transfer involves two bundled on-chain transactions with different fee levels: the first (i.e., Tx1) with a lower fee inscribes the transfer request, while the second (i.e., Tx2) with a higher fee finalizes the actual transfer. We find that an adversary can exploit this by sending a manipulated fee transaction (falling between the two fee levels), which allows Tx1 to be processed while Tx2 remains pinned in the mempool. This locks the BRC20 liquidity and disrupts normal transfers for users. We term this BRC20 pinning attack.   Our attack exposes an inherent design flaw that can be applied to 90+% inscription-based tokens within the Bitcoin ecosystem.   We also conducted the attack on Binance's ORDI hot wallet (the most prevalent BRC20 token and the most active wallet), resulting in a temporary suspension of ORDI withdrawals on Binance for 3.5 hours, which were shortly resumed after our communication.

摘要: BRC20代币是比特币网络上的一种不可替代资产。它们允许用户在比特币Satoshis中嵌入定制内容。在过去的一年里(2023Q3-2024Q3)，相关的代币狂潮已经达到了3.65万亿美元的市场规模。然而，这种直观的设计并没有经过严格的安全审查。我们首次深入分析了BRC20的传输机制，并确定了一个关键的攻击载体。典型的BRC20转移涉及两个不同费用水平的捆绑链上交易：第一个费用较低的(即TX1)记录转移请求，而第二个(即Tx2)费用较高的完成实际转移。我们发现，对手可以通过发送被操纵的费用事务(介于两个费用水平之间)来利用这一点，这允许在Tx1被处理的同时Tx2仍然被固定在内存池中。这锁定了BRC20的流动性，并扰乱了用户的正常转账。我们称之为BRC20钉住攻击。我们的攻击暴露了一个固有的设计缺陷，该缺陷可以应用于比特币生态系统中90%以上的铭文令牌。我们还对Binance的Ordi热钱包(最流行的BRC20代币和最活跃的钱包)进行了攻击，导致Binance上的Ordi提款暂时暂停3.5小时，并在我们沟通后不久恢复。



## **11. Cognitive Overload Attack:Prompt Injection for Long Context**

认知过载攻击：长上下文的提示注入 cs.CL

40 pages, 31 Figures

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11272v1) [paper-pdf](http://arxiv.org/pdf/2410.11272v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan, Amin Karbasi

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in performing tasks across various domains without needing explicit retraining. This capability, known as In-Context Learning (ICL), while impressive, exposes LLMs to a variety of adversarial prompts and jailbreaks that manipulate safety-trained LLMs into generating undesired or harmful output. In this paper, we propose a novel interpretation of ICL in LLMs through the lens of cognitive neuroscience, by drawing parallels between learning in human cognition with ICL. We applied the principles of Cognitive Load Theory in LLMs and empirically validate that similar to human cognition, LLMs also suffer from cognitive overload a state where the demand on cognitive processing exceeds the available capacity of the model, leading to potential errors. Furthermore, we demonstrated how an attacker can exploit ICL to jailbreak LLMs through deliberately designed prompts that induce cognitive overload on LLMs, thereby compromising the safety mechanisms of LLMs. We empirically validate this threat model by crafting various cognitive overload prompts and show that advanced models such as GPT-4, Claude-3.5 Sonnet, Claude-3 OPUS, Llama-3-70B-Instruct, Gemini-1.0-Pro, and Gemini-1.5-Pro can be successfully jailbroken, with attack success rates of up to 99.99%. Our findings highlight critical vulnerabilities in LLMs and underscore the urgency of developing robust safeguards. We propose integrating insights from cognitive load theory into the design and evaluation of LLMs to better anticipate and mitigate the risks of adversarial attacks. By expanding our experiments to encompass a broader range of models and by highlighting vulnerabilities in LLMs' ICL, we aim to ensure the development of safer and more reliable AI systems.

摘要: 大型语言模型(LLM)已经显示出在不需要明确的再培训的情况下执行跨领域任务的显著能力。这种被称为情景学习(ICL)的能力虽然令人印象深刻，但会使LLM暴露在各种对抗性提示和越狱之下，这些提示和越狱操作经过安全培训的LLM产生不需要的或有害的输出。在这篇文章中，我们提出了一种新的解释，从认知神经科学的角度，通过将人类认知中的学习与ICL相提并论，对LLMS中的ICL做出了新的解释。我们将认知负荷理论的原理应用到LLMS中，并实证验证了与人类认知类似，LLMS也存在认知过载，即认知加工需求超过模型的可用能力，从而导致潜在错误。此外，我们演示了攻击者如何通过故意设计的提示来利用ICL来越狱LLM，这些提示会导致LLM上的认知过载，从而危及LLMS的安全机制。我们通过制作不同的认知过载提示对该威胁模型进行了实证验证，结果表明，GPT-4、Claude-3.5十四行诗、Claude-3 opus、Llama-3-70B-Indict、Gemini-1.0-Pro和Gemini-1.5-Pro等高级模型可以成功越狱，攻击成功率高达99.99%。我们的发现突显了低土地管理制度的严重脆弱性，并强调了制定强有力的保障措施的紧迫性。我们建议将认知负荷理论的见解融入到LLMS的设计和评估中，以更好地预测和减轻对手攻击的风险。通过扩大我们的实验以涵盖更广泛的模型，并通过突出LLMS ICL中的漏洞，我们的目标是确保开发出更安全、更可靠的人工智能系统。



## **12. Adversarially Guided Stateful Defense Against Backdoor Attacks in Federated Deep Learning**

联合深度学习中针对后门攻击的敌对引导状态防御 cs.LG

16 pages, Accepted at ACSAC 2024

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11205v1) [paper-pdf](http://arxiv.org/pdf/2410.11205v1)

**Authors**: Hassan Ali, Surya Nepal, Salil S. Kanhere, Sanjay Jha

**Abstract**: Recent works have shown that Federated Learning (FL) is vulnerable to backdoor attacks. Existing defenses cluster submitted updates from clients and select the best cluster for aggregation. However, they often rely on unrealistic assumptions regarding client submissions and sampled clients population while choosing the best cluster. We show that in realistic FL settings, state-of-the-art (SOTA) defenses struggle to perform well against backdoor attacks in FL. To address this, we highlight that backdoored submissions are adversarially biased and overconfident compared to clean submissions. We, therefore, propose an Adversarially Guided Stateful Defense (AGSD) against backdoor attacks on Deep Neural Networks (DNNs) in FL scenarios. AGSD employs adversarial perturbations to a small held-out dataset to compute a novel metric, called the trust index, that guides the cluster selection without relying on any unrealistic assumptions regarding client submissions. Moreover, AGSD maintains a trust state history of each client that adaptively penalizes backdoored clients and rewards clean clients. In realistic FL settings, where SOTA defenses mostly fail to resist attacks, AGSD mostly outperforms all SOTA defenses with minimal drop in clean accuracy (5% in the worst-case compared to best accuracy) even when (a) given a very small held-out dataset -- typically AGSD assumes 50 samples (<= 0.1% of the training data) and (b) no heldout dataset is available, and out-of-distribution data is used instead. For reproducibility, our code will be openly available at: https://github.com/hassanalikhatim/AGSD.

摘要: 最近的研究表明，联邦学习(FL)容易受到后门攻击。现有防御对来自客户端的提交的更新进行集群，并选择最佳集群进行聚合。然而，在选择最佳聚类时，他们往往依赖于关于客户提交和抽样客户总体的不切实际的假设。我们展示了在现实的FL环境中，最先进的(SOTA)防御在FL的后门攻击中表现得很好。为了解决这个问题，我们强调，与干净的提交相比，落后的提交是相反的偏见和过度自信。因此，我们提出了一种针对FL场景下的深层神经网络(DNN)后门攻击的对抗性引导状态防御(AGSD)。AGSD对一个较小的坚持数据集使用对抗性扰动来计算一个新的度量，称为信任指数，该度量指导集群选择，而不依赖于任何关于客户提交的不切实际的假设。此外，AGSD维护每个客户端的信任状态历史，该历史自适应地惩罚落后的客户端并奖励干净的客户端。在现实的FL设置中，SOTA防御系统大多无法抵抗攻击，AGSD的性能大多优于所有SOTA防御系统，清洁精度下降最小(与最佳精度相比，在最差情况下为5%)，即使在以下情况下也是如此：(A)给定一个非常小的坚持数据集--通常AGSD假设50个样本(<=0.1%的训练数据)，以及(B)没有可用的坚持数据集，而改用非分布数据。为了重现性，我们的代码将在以下网址公开提供：https://github.com/hassanalikhatim/AGSD.



## **13. Fast Second-Order Online Kernel Learning through Incremental Matrix Sketching and Decomposition**

通过增量矩阵绘制和分解进行快速二阶在线核心学习 cs.LG

**SubmitDate**: 2024-10-15    [abs](http://arxiv.org/abs/2410.11188v1) [paper-pdf](http://arxiv.org/pdf/2410.11188v1)

**Authors**: Dongxie Wen, Xiao Zhang, Zhewei Wei

**Abstract**: Online Kernel Learning (OKL) has attracted considerable research interest due to its promising predictive performance in streaming environments. Second-order approaches are particularly appealing for OKL as they often offer substantial improvements in regret guarantees. However, existing second-order OKL approaches suffer from at least quadratic time complexity with respect to the pre-set budget, rendering them unsuitable for meeting the real-time demands of large-scale streaming recommender systems. The singular value decomposition required to obtain explicit feature mapping is also computationally expensive due to the complete decomposition process. Moreover, the absence of incremental updates to manage approximate kernel space causes these algorithms to perform poorly in adversarial environments and real-world streaming recommendation datasets. To address these issues, we propose FORKS, a fast incremental matrix sketching and decomposition approach tailored for second-order OKL. FORKS constructs an incremental maintenance paradigm for second-order kernelized gradient descent, which includes incremental matrix sketching for kernel approximation and incremental matrix decomposition for explicit feature mapping construction. Theoretical analysis demonstrates that FORKS achieves a logarithmic regret guarantee on par with other second-order approaches while maintaining a linear time complexity w.r.t. the budget, significantly enhancing efficiency over existing approaches. We validate the performance of FORKS through extensive experiments conducted on real-world streaming recommendation datasets, demonstrating its superior scalability and robustness against adversarial attacks.

摘要: 在线核学习(Online Kernel Learning，OKL)因其在流媒体环境中良好的预测性能而引起了广泛的研究兴趣。二阶方法对OKL特别有吸引力，因为它们经常在后悔保证方面提供实质性的改善。然而，现有的二阶OKL方法至少存在相对于预先设定的预算的二次时间复杂度，不适合满足大规模流媒体推荐系统的实时需求。由于整个分解过程，获得显式特征映射所需的奇异值分解在计算上也是昂贵的。此外，由于缺乏增量更新来管理近似的内核空间，导致这些算法在敌对环境和真实的流媒体推荐数据集上表现不佳。为了解决这些问题，我们提出了Forks，一种为二阶OKL量身定做的快速增量矩阵绘制和分解方法。Forks构造了一种用于二阶核化梯度下降的增量维护范式，其中包括用于核逼近的增量矩阵草图和用于显式特征映射构造的增量矩阵分解。理论分析表明，Forks在保持线性时间复杂度的同时，获得了与其他二阶方法相同的对数错误保证。与现有办法相比，大大提高了效率。我们通过在真实的流媒体推荐数据集上进行的大量实验验证了Forks的性能，展示了其优越的可扩展性和对对手攻击的健壮性。



## **14. Sensor Deprivation Attacks for Stealthy UAV Manipulation**

隐形无人机操纵的传感器破坏攻击 cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.11131v1) [paper-pdf](http://arxiv.org/pdf/2410.11131v1)

**Authors**: Alessandro Erba, John H. Castellanos, Sahil Sihag, Saman Zonouz, Nils Ole Tippenhauer

**Abstract**: Unmanned Aerial Vehicles autonomously perform tasks with the use of state-of-the-art control algorithms. These control algorithms rely on the freshness and correctness of sensor readings. Incorrect control actions lead to catastrophic destabilization of the process.   In this work, we propose a multi-part \emph{Sensor Deprivation Attacks} (SDAs), aiming to stealthily impact process control via sensor reconfiguration. In the first part, the attacker will inject messages on local buses that connect to the sensor. The injected message reconfigures the sensors, e.g.,~to suspend the sensing. In the second part, those manipulation primitives are selectively used to cause adversarial sensor values at the controller, transparently to the data consumer. In the third part, the manipulated sensor values lead to unwanted control actions (e.g. a drone crash). We experimentally investigate all three parts of our proposed attack. Our findings show that i)~reconfiguring sensors can have surprising effects on reported sensor values, and ii)~the attacker can stall the overall Kalman Filter state estimation, leading to a complete stop of control computations. As a result, the UAV becomes destabilized, leading to a crash or significant deviation from its planned trajectory (over 30 meters). We also propose an attack synthesis methodology that optimizes the timing of these SDA manipulations, maximizing their impact. Notably, our results demonstrate that these SDAs evade detection by state-of-the-art UAV anomaly detectors.   Our work shows that attacks on sensors are not limited to continuously inducing random measurements, and demonstrate that sensor reconfiguration can completely stall the drone controller. In our experiments, state-of-the-art UAV controller software and countermeasures are unable to handle such manipulations. Hence, we also discuss new corresponding countermeasures.

摘要: 无人驾驶飞行器使用最先进的控制算法自主执行任务。这些控制算法依赖于传感器读数的新鲜度和准确性。不正确的控制行动会导致过程的灾难性不稳定。在这项工作中，我们提出了一种多部分的传感器剥夺攻击(SDAS)，旨在通过传感器重构来秘密影响过程控制。在第一部分中，攻击者将在连接到传感器的本地总线上注入消息。注入的消息重新配置传感器，例如，暂停侦听。在第二部分中，这些操作原语被选择性地用于在控制器处产生对抗性的传感器值，对数据消费者是透明的。在第三部分中，被操纵的传感器数值会导致不需要的控制动作(例如无人机坠毁)。我们对我们提议的攻击的所有三个部分进行了实验研究。我们的发现表明，i)~重新配置传感器可以对报告的传感器值产生惊人的影响，以及ii)~攻击者可以停止整个卡尔曼滤波状态估计，导致控制计算完全停止。结果，无人机变得不稳定，导致坠毁或严重偏离其计划轨迹(超过30米)。我们还提出了一种攻击综合方法，优化了这些SDA操作的时机，最大化了它们的影响。值得注意的是，我们的结果表明，这些SDA躲避了最先进的无人机异常检测器的检测。我们的工作表明，对传感器的攻击并不局限于连续诱导随机测量，并证明了传感器重新配置可以完全阻止无人机控制器。在我们的实验中，最先进的无人机控制器软件和对策无法处理这种操作。因此，我们还讨论了新的相应对策。



## **15. Denial-of-Service Poisoning Attacks against Large Language Models**

针对大型语言模型的拒绝服务中毒攻击 cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10760v1) [paper-pdf](http://arxiv.org/pdf/2410.10760v1)

**Authors**: Kuofeng Gao, Tianyu Pang, Chao Du, Yong Yang, Shu-Tao Xia, Min Lin

**Abstract**: Recent studies have shown that LLMs are vulnerable to denial-of-service (DoS) attacks, where adversarial inputs like spelling errors or non-semantic prompts trigger endless outputs without generating an [EOS] token. These attacks can potentially cause high latency and make LLM services inaccessible to other users or tasks. However, when there are speech-to-text interfaces (e.g., voice commands to a robot), executing such DoS attacks becomes challenging, as it is difficult to introduce spelling errors or non-semantic prompts through speech. A simple DoS attack in these scenarios would be to instruct the model to "Keep repeating Hello", but we observe that relying solely on natural instructions limits output length, which is bounded by the maximum length of the LLM's supervised finetuning (SFT) data. To overcome this limitation, we propose poisoning-based DoS (P-DoS) attacks for LLMs, demonstrating that injecting a single poisoned sample designed for DoS purposes can break the output length limit. For example, a poisoned sample can successfully attack GPT-4o and GPT-4o mini (via OpenAI's finetuning API) using less than $1, causing repeated outputs up to the maximum inference length (16K tokens, compared to 0.5K before poisoning). Additionally, we perform comprehensive ablation studies on open-source LLMs and extend our method to LLM agents, where attackers can control both the finetuning dataset and algorithm. Our findings underscore the urgent need for defenses against P-DoS attacks to secure LLMs. Our code is available at https://github.com/sail-sg/P-DoS.

摘要: 最近的研究表明，LLMS容易受到拒绝服务(DoS)攻击，即拼写错误或非语义提示等敌意输入会触发无休止的输出，而不会生成[EOS]令牌。这些攻击可能会导致高延迟，并使其他用户或任务无法访问LLM服务。然而，当存在语音到文本的接口(例如，对机器人的语音命令)时，执行这种DoS攻击变得具有挑战性，因为很难通过语音引入拼写错误或非语义提示。在这些场景中，一个简单的DoS攻击是指示模型“不断重复Hello”，但我们观察到，仅依赖自然指令会限制输出长度，而输出长度受LLM的监督微调(SFT)数据的最大长度的限制。为了克服这一局限性，我们提出了针对LLMS的基于中毒的DoS(P-DoS)攻击，证明了注入单个为DoS目的而设计的有毒样本可以打破输出长度限制。例如，中毒的样本可以使用不到1美元的成本成功攻击GPT-4o和GPT-4o mini(通过OpenAI的Finetuning API)，导致重复输出到最大推理长度(16K令牌，而中毒前为0.5K)。此外，我们在开源LLMS上进行了全面的烧蚀研究，并将我们的方法扩展到LLM代理，其中攻击者可以控制精调数据集和算法。我们的发现强调了防御P-DoS攻击以确保LLM安全的迫切需要。我们的代码可以在https://github.com/sail-sg/P-DoS.上找到



## **16. Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings**

使用Lyapunov稳定嵌入的对抗鲁棒性分布外检测 cs.LG

Code and pre-trained models are available at  https://github.com/AdaptiveMotorControlLab/AROS

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10744v1) [paper-pdf](http://arxiv.org/pdf/2410.10744v1)

**Authors**: Hossein Mirzaei, Mackenzie W. Mathis

**Abstract**: Despite significant advancements in out-of-distribution (OOD) detection, existing methods still struggle to maintain robustness against adversarial attacks, compromising their reliability in critical real-world applications. Previous studies have attempted to address this challenge by exposing detectors to auxiliary OOD datasets alongside adversarial training. However, the increased data complexity inherent in adversarial training, and the myriad of ways that OOD samples can arise during testing, often prevent these approaches from establishing robust decision boundaries. To address these limitations, we propose AROS, a novel approach leveraging neural ordinary differential equations (NODEs) with Lyapunov stability theorem in order to obtain robust embeddings for OOD detection. By incorporating a tailored loss function, we apply Lyapunov stability theory to ensure that both in-distribution (ID) and OOD data converge to stable equilibrium points within the dynamical system. This approach encourages any perturbed input to return to its stable equilibrium, thereby enhancing the model's robustness against adversarial perturbations. To not use additional data, we generate fake OOD embeddings by sampling from low-likelihood regions of the ID data feature space, approximating the boundaries where OOD data are likely to reside. To then further enhance robustness, we propose the use of an orthogonal binary layer following the stable feature space, which maximizes the separation between the equilibrium points of ID and OOD samples. We validate our method through extensive experiments across several benchmarks, demonstrating superior performance, particularly under adversarial attacks. Notably, our approach improves robust detection performance from 37.8% to 80.1% on CIFAR-10 vs. CIFAR-100 and from 29.0% to 67.0% on CIFAR-100 vs. CIFAR-10.

摘要: 尽管在分发外(OOD)检测方面有了很大的进步，但现有的方法仍然难以保持对对手攻击的健壮性，从而影响了它们在关键现实应用中的可靠性。以前的研究试图通过将探测器暴露于辅助OOD数据集以及对抗性训练来解决这一挑战。然而，对抗性训练中固有的增加的数据复杂性，以及OOD样本在测试过程中可能出现的各种方式，往往阻碍这些方法建立稳健的决策边界。为了克服这些局限性，我们提出了一种新的方法AROS，它利用Lyapunov稳定性定理来利用神经常微分方程组(节点)来获得用于OOD检测的稳健嵌入。通过引入定制的损失函数，我们应用Lyapunov稳定性理论来确保内分布(ID)和OOD数据都收敛到动力系统中的稳定平衡点。这种方法鼓励任何扰动的输入返回到其稳定的平衡，从而增强模型对对抗性扰动的稳健性。为了不使用额外的数据，我们通过从ID数据特征空间的低似然区域采样，逼近OOD数据可能驻留的边界来生成虚假的OOD嵌入。为了进一步增强稳健性，我们提出了在稳定特征空间之后使用一个正交二值层，最大化了ID和OOD样本的平衡点之间的分离。我们通过在几个基准上的大量实验来验证我们的方法，展示了优越的性能，特别是在对抗性攻击下。值得注意的是，我们的方法将健壮性检测性能从CIFAR-10上的37.8%提高到CIFAR-100上的80.1%，以及CIFAR-100上的29.0%到CIFAR-10上的67.0%。



## **17. Towards Calibrated Losses for Adversarial Robust Reject Option Classification**

走向对抗性的校准损失稳健的暂停期权分类 cs.LG

Accepted at Asian Conference on Machine Learning (ACML) , 2024

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10736v1) [paper-pdf](http://arxiv.org/pdf/2410.10736v1)

**Authors**: Vrund Shah, Tejas Chaudhari, Naresh Manwani

**Abstract**: Robustness towards adversarial attacks is a vital property for classifiers in several applications such as autonomous driving, medical diagnosis, etc. Also, in such scenarios, where the cost of misclassification is very high, knowing when to abstain from prediction becomes crucial. A natural question is which surrogates can be used to ensure learning in scenarios where the input points are adversarially perturbed and the classifier can abstain from prediction? This paper aims to characterize and design surrogates calibrated in "Adversarial Robust Reject Option" setting. First, we propose an adversarial robust reject option loss $\ell_{d}^{\gamma}$ and analyze it for the hypothesis set of linear classifiers ($\mathcal{H}_{\textrm{lin}}$). Next, we provide a complete characterization result for any surrogate to be $(\ell_{d}^{\gamma},\mathcal{H}_{\textrm{lin}})$- calibrated. To demonstrate the difficulty in designing surrogates to $\ell_{d}^{\gamma}$, we show negative calibration results for convex surrogates and quasi-concave conditional risk cases (these gave positive calibration in adversarial setting without reject option). We also empirically argue that Shifted Double Ramp Loss (DRL) and Shifted Double Sigmoid Loss (DSL) satisfy the calibration conditions. Finally, we demonstrate the robustness of shifted DRL and shifted DSL against adversarial perturbations on a synthetically generated dataset.

摘要: 对敌意攻击的稳健性是分类器在一些应用中的重要特性，例如自动驾驶、医疗诊断等。此外，在这种情况下，错误分类的成本非常高，知道何时放弃预测变得至关重要。一个自然的问题是，在输入点受到相反干扰并且分类器可以避免预测的情况下，可以使用哪些代理来确保学习？本文旨在刻画和设计在“对抗性稳健拒绝选项”设置下校准的代理。首先，针对线性分类器的假设集($\Mathcal{H}_{\tExtrm{Lin}}$)，提出了一种对抗性稳健拒绝期权损失$\ell_(D)^{\Gamma}$，并对其进行了分析。接下来，我们给出了任一代理是$(\ell_{d}^{\Gamma}，\Mathcal{H}_{\tExtrm{Lin}})-校准的完整刻画结果。为了证明代理设计的难度，我们给出了凸代理和准凹条件风险情形的负校准结果(在没有拒绝选项的对抗性环境下给出了正校准)。我们还从经验上论证了移位双斜坡损耗(DRL)和移位双Sigmoid损耗(DSL)满足校准条件。最后，我们在合成数据集上证明了移位DRL和移位DSL对敌意扰动的稳健性。



## **18. Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues**

脱轨自己：通过自我发现的线索进行多回合LLM越狱攻击 cs.CL

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10700v1) [paper-pdf](http://arxiv.org/pdf/2410.10700v1)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: This study exposes the safety vulnerabilities of Large Language Models (LLMs) in multi-turn interactions, where malicious users can obscure harmful intents across several queries. We introduce ActorAttack, a novel multi-turn attack method inspired by actor-network theory, which models a network of semantically linked actors as attack clues to generate diverse and effective attack paths toward harmful targets. ActorAttack addresses two main challenges in multi-turn attacks: (1) concealing harmful intents by creating an innocuous conversation topic about the actor, and (2) uncovering diverse attack paths towards the same harmful target by leveraging LLMs' knowledge to specify the correlated actors as various attack clues. In this way, ActorAttack outperforms existing single-turn and multi-turn attack methods across advanced aligned LLMs, even for GPT-o1. We will publish a dataset called SafeMTData, which includes multi-turn adversarial prompts and safety alignment data, generated by ActorAttack. We demonstrate that models safety-tuned using our safety dataset are more robust to multi-turn attacks. Code is available at https://github.com/renqibing/ActorAttack.

摘要: 这项研究揭示了大型语言模型(LLM)在多轮交互中的安全漏洞，在这种交互中，恶意用户可以通过几个查询来掩盖有害意图。我们引入了ActorAttack，这是一种受行动者-网络理论启发的新型多回合攻击方法，它将语义上联系在一起的行动者网络建模为攻击线索，以生成针对有害目标的多样化和有效的攻击路径。ActorAttack解决了多轮攻击中的两个主要挑战：(1)通过创建关于参与者的无害对话主题来隐藏有害意图；(2)通过利用LLMS的知识将相关的参与者指定为各种攻击线索，揭示针对同一有害目标的不同攻击路径。通过这种方式，ActorAttack在高级对准LLM上的表现优于现有的单回合和多回合攻击方法，即使对于GPT-o1也是如此。我们将发布一个名为SafeMTData的数据集，其中包括由ActorAttack生成的多轮对抗性提示和安全对齐数据。我们证明，使用我们的安全数据集进行安全调整的模型对多轮攻击更具健壮性。代码可在https://github.com/renqibing/ActorAttack.上找到



## **19. Enhancing Robustness in Deep Reinforcement Learning: A Lyapunov Exponent Approach**

增强深度强化学习的鲁棒性：一种李雅普诺夫指数方法 cs.LG

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10674v1) [paper-pdf](http://arxiv.org/pdf/2410.10674v1)

**Authors**: Rory Young, Nicolas Pugeault

**Abstract**: Deep reinforcement learning agents achieve state-of-the-art performance in a wide range of simulated control tasks. However, successful applications to real-world problems remain limited. One reason for this dichotomy is because the learned policies are not robust to observation noise or adversarial attacks. In this paper, we investigate the robustness of deep RL policies to a single small state perturbation in deterministic continuous control tasks. We demonstrate that RL policies can be deterministically chaotic as small perturbations to the system state have a large impact on subsequent state and reward trajectories. This unstable non-linear behaviour has two consequences: First, inaccuracies in sensor readings, or adversarial attacks, can cause significant performance degradation; Second, even policies that show robust performance in terms of rewards may have unpredictable behaviour in practice. These two facets of chaos in RL policies drastically restrict the application of deep RL to real-world problems. To address this issue, we propose an improvement on the successful Dreamer V3 architecture, implementing a Maximal Lyapunov Exponent regularisation. This new approach reduces the chaotic state dynamics, rendering the learnt policies more resilient to sensor noise or adversarial attacks and thereby improving the suitability of Deep Reinforcement Learning for real-world applications.

摘要: 深度强化学习代理在广泛的模拟控制任务中实现最先进的性能。然而，对现实世界问题的成功应用仍然有限。这种二分法的一个原因是，学习的策略对观察噪声或对抗性攻击不是很健壮。本文研究了在确定性连续控制任务中，深度RL策略对单个小状态扰动的鲁棒性。我们证明了RL策略可以是确定性混沌的，因为系统状态的微小扰动对随后的状态和奖励轨迹有很大的影响。这种不稳定的非线性行为有两个后果：第一，传感器读数的不准确或敌意攻击可能导致性能显著下降；第二，即使是在奖励方面表现强劲的策略，在实践中也可能有不可预测的行为。RL政策中的这两个方面的混乱极大地限制了深度RL在现实世界问题中的应用。为了解决这个问题，我们提出了对成功的Dreamer V3架构的改进，实现了最大Lyapunov指数正则化。这种新方法降低了混沌状态动态，使学习到的策略对传感器噪声或敌意攻击更具弹性，从而提高了深度强化学习在实际应用中的适用性。



## **20. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

正规的鲁棒可靠的学习者和实例有针对性的攻击 cs.LG

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10572v1) [paper-pdf](http://arxiv.org/pdf/2410.10572v1)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.

摘要: 针对实例的数据中毒攻击，即对手破坏训练集以在特定测试点上引发错误，已经引起了严重的担忧。Balcan等人(2022)提出了一种应对这一挑战的方法，定义了稳健可靠的学习者的概念，即使在存在数据中毒攻击的情况下，也可以在定义明确的假设下提供逐个实例的正确性保证。然后，对于对数凹分布上的线性分隔符的情况，他们给出了一个通用的最优(但计算效率低)鲁棒可靠的学习器以及一个计算高效的算法。在这项工作中，我们解决了Balcan等人(2022)留下的两个挑战。首先，对于高度灵活的假设类，Balcan等人(2022)中的稳健可靠学习者的定义变得空洞：如果在训练集上存在两个都是零误差的分类器h_0，h_1\in H，使得h_0(X)\neq h_1(X)，那么稳健可靠的学习者必须在x上弃权。我们通过定义一个修正的正则化稳健可靠学习者的概念来解决这个问题，它允许在这种情况下非平凡的陈述。其次，Balcan等人(2022)的通用算法需要在每个测试点x上重新运行ERM预言(本质上是重新训练分类器)，即使ERM可以有效地实施，这通常也是不切实际的。为了解决这个问题，我们证明了，至少在某些有趣的情况下，我们可以设计算法，通过使用动态算法设计的技术，在训练时间内产生时间上的次线性输出。



## **21. ROSAR: An Adversarial Re-Training Framework for Robust Side-Scan Sonar Object Detection**

ROSAR：用于稳健侧扫声纳目标检测的对抗性重新训练框架 cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10554v1) [paper-pdf](http://arxiv.org/pdf/2410.10554v1)

**Authors**: Martin Aubard, László Antal, Ana Madureira, Luis F. Teixeira, Erika Ábrahám

**Abstract**: This paper introduces ROSAR, a novel framework enhancing the robustness of deep learning object detection models tailored for side-scan sonar (SSS) images, generated by autonomous underwater vehicles using sonar sensors. By extending our prior work on knowledge distillation (KD), this framework integrates KD with adversarial retraining to address the dual challenges of model efficiency and robustness against SSS noises. We introduce three novel, publicly available SSS datasets, capturing different sonar setups and noise conditions. We propose and formalize two SSS safety properties and utilize them to generate adversarial datasets for retraining. Through a comparative analysis of projected gradient descent (PGD) and patch-based adversarial attacks, ROSAR demonstrates significant improvements in model robustness and detection accuracy under SSS-specific conditions, enhancing the model's robustness by up to 1.85%. ROSAR is available at https://github.com/remaro-network/ROSAR-framework.

摘要: 本文介绍了ROSAR，这是一种新型框架，可增强深度学习对象检测模型的鲁棒性，该模型专为侧扫描声纳（SS）图像定制，该图像由自主水下航行器使用声纳传感器生成。通过扩展我们之前在知识蒸馏（KD）方面的工作，该框架将KD与对抗性再培训集成起来，以解决模型效率和针对SS噪音的鲁棒性的双重挑战。我们引入了三个新颖的、公开可用的SS数据集，捕捉不同的声纳设置和噪音条件。我们提出并形式化了两个SS安全属性，并利用它们来生成用于再培训的对抗性数据集。通过对投影梯度下降（PVD）和基于补丁的对抗攻击的比较分析，ROSAR展示了在特定SS条件下模型稳健性和检测准确性的显着提高，将模型的稳健性提高了高达1.85%。ROSAR可在https://github.com/remaro-network/ROSAR-framework上获取。



## **22. Generalized Adversarial Code-Suggestions: Exploiting Contexts of LLM-based Code-Completion**

广义对抗代码建议：利用基于LLM的代码完成的上下文 cs.CR

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10526v1) [paper-pdf](http://arxiv.org/pdf/2410.10526v1)

**Authors**: Karl Rubel, Maximilian Noppel, Christian Wressnegger

**Abstract**: While convenient, relying on LLM-powered code assistants in day-to-day work gives rise to severe attacks. For instance, the assistant might introduce subtle flaws and suggest vulnerable code to the user. These adversarial code-suggestions can be introduced via data poisoning and, thus, unknowingly by the model creators. In this paper, we provide a generalized formulation of such attacks, spawning and extending related work in this domain. This formulation is defined over two components: First, a trigger pattern occurring in the prompts of a specific user group, and, second, a learnable map in embedding space from the prompt to an adversarial bait. The latter gives rise to novel and more flexible targeted attack-strategies, allowing the adversary to choose the most suitable trigger pattern for a specific user-group arbitrarily, without restrictions on the pattern's tokens. Our directional-map attacks and prompt-indexing attacks increase the stealthiness decisively. We extensively evaluate the effectiveness of these attacks and carefully investigate defensive mechanisms to explore the limits of generalized adversarial code-suggestions. We find that most defenses unfortunately offer little protection only.

摘要: 虽然方便，但在日常工作中依赖LLM支持的代码助手会引发严重的攻击。例如，助手可能会引入细微的缺陷，并向用户建议易受攻击的代码。这些对抗性代码建议可以通过数据中毒引入，因此，模型创建者会在不知情的情况下引入这些代码建议。在本文中，我们给出了这类攻击的一般形式，产生和扩展了这一领域的相关工作。该公式定义在两个部分上：第一，出现在特定用户组的提示中的触发模式，第二，从提示到敌方诱饵的嵌入空间中的可学习映射。后者产生了新颖和更灵活的定向攻击策略，允许攻击者为特定的用户组任意选择最合适的触发模式，而不限制模式的令牌。我们的方向图攻击和快速索引攻击决定性地增加了隐蔽性。我们广泛评估这些攻击的有效性，并仔细研究防御机制，以探索广义对抗性代码建议的局限性。我们发现，不幸的是，大多数防御措施只提供了很少的保护。



## **23. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks**

（图）神经网络对抗数据中毒和后门攻击的可证明鲁棒性 cs.LG

A preliminary version of this work appeared at the AdvML-Frontiers @  NeurIPS 2024 workshop

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2407.10867v2) [paper-pdf](http://arxiv.org/pdf/2407.10867v2)

**Authors**: Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann

**Abstract**: Generalization of machine learning models can be severely compromised by data poisoning, where adversarial changes are applied to the training data. This vulnerability has led to interest in certifying (i.e., proving) that such changes up to a certain magnitude do not affect test predictions. We, for the first time, certify Graph Neural Networks (GNNs) against poisoning attacks, including backdoors, targeting the node features of a given graph. Our certificates are white-box and based upon $(i)$ the neural tangent kernel, which characterizes the training dynamics of sufficiently wide networks; and $(ii)$ a novel reformulation of the bilevel optimization problem describing poisoning as a mixed-integer linear program. Consequently, we leverage our framework to provide fundamental insights into the role of graph structure and its connectivity on the worst-case robustness behavior of convolution-based and PageRank-based GNNs. We note that our framework is more general and constitutes the first approach to derive white-box poisoning certificates for NNs, which can be of independent interest beyond graph-related tasks.

摘要: 机器学习模型的泛化可能会受到数据中毒的严重影响，在数据中毒中，对训练数据应用对抗性更改。这一漏洞引起了人们对证明(即证明)这样的变化不会影响测试预测的兴趣。我们首次证明了图神经网络(GNN)不会受到毒化攻击，包括针对给定图的节点特征的后门攻击。我们的证书是白盒的，并且基于$(I)$神经正切核，它表征了足够广泛的网络的训练动力学；$(Ii)$是将中毒描述为混合整数线性规划的双层优化问题的新形式。因此，我们利用我们的框架来提供关于图结构及其连通性对基于卷积和基于PageRank的GNN的最坏情况健壮性行为的作用的基本见解。我们注意到，我们的框架更通用，并且构成了第一种为NNS派生白盒中毒证书的方法，这可能是图相关任务之外的独立兴趣。



## **24. Achieving Optimal Breakdown for Byzantine Robust Gossip**

实现拜占庭稳健八卦的最佳细分 math.OC

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10418v1) [paper-pdf](http://arxiv.org/pdf/2410.10418v1)

**Authors**: Renaud Gaucher, Aymeric Dieuleveut, Hadrien Hendrikx

**Abstract**: Distributed approaches have many computational benefits, but they are vulnerable to attacks from a subset of devices transmitting incorrect information. This paper investigates Byzantine-resilient algorithms in a decentralized setting, where devices communicate directly with one another. We investigate the notion of breakdown point, and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce $\mathrm{CG}^+$, an algorithm at the intersection of $\mathrm{ClippedGossip}$ and $\mathrm{NNA}$, two popular approaches for robust decentralized learning. $\mathrm{CG}^+$ meets our upper bound, and thus obtains optimal robustness guarantees, whereas neither of the existing two does. We provide experimental evidence for this gap by presenting an attack tailored to sparse graphs which breaks $\mathrm{NNA}$ but against which $\mathrm{CG}^+$ is robust.

摘要: 分布式方法具有许多计算优势，但它们很容易受到来自传输错误信息的设备子集的攻击。本文研究了去中心化环境中的拜占庭弹性算法，其中设备之间直接通信。我们研究了崩溃点的概念，并给出了去中心化算法可以容忍的对手数量的上限。我们引入了$\mathrm{CG}^+$，这是一种位于$\mathrm{ClipedGossip}$和$\mathrm{NNA}$交叉点的算法，这是两种流行的稳健去中心化学习方法。$\mathrm{CG}^+$满足了我们的上界，从而获得了最佳稳健性保证，而现有两个都没有。我们通过提供针对稀疏图量身定制的攻击来为这一差距提供实验证据，该攻击会破坏$\mathrm{NNA}$，但$\mathrm{CG}^+$对此是稳健的。



## **25. Feature Averaging: An Implicit Bias of Gradient Descent Leading to Non-Robustness in Neural Networks**

特征平均：梯度下降的隐式偏差导致神经网络的非鲁棒性 cs.LG

78 pages, 10 figures

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10322v1) [paper-pdf](http://arxiv.org/pdf/2410.10322v1)

**Authors**: Binghui Li, Zhixuan Pan, Kaifeng Lyu, Jian Li

**Abstract**: In this work, we investigate a particular implicit bias in the gradient descent training process, which we term "Feature Averaging", and argue that it is one of the principal factors contributing to non-robustness of deep neural networks. Despite the existence of multiple discriminative features capable of classifying data, neural networks trained by gradient descent exhibit a tendency to learn the average (or certain combination) of these features, rather than distinguishing and leveraging each feature individually. In particular, we provide a detailed theoretical analysis of the training dynamics of gradient descent in a two-layer ReLU network for a binary classification task, where the data distribution consists of multiple clusters with orthogonal cluster center vectors. We rigorously prove that gradient descent converges to the regime of feature averaging, wherein the weights associated with each hidden-layer neuron represent an average of the cluster centers (each center corresponding to a distinct feature). It leads the network classifier to be non-robust due to an attack that aligns with the negative direction of the averaged features. Furthermore, we prove that, with the provision of more granular supervised information, a two-layer multi-class neural network is capable of learning individual features, from which one can derive a binary classifier with the optimal robustness under our setting. Besides, we also conduct extensive experiments using synthetic datasets, MNIST and CIFAR-10 to substantiate the phenomenon of feature averaging and its role in adversarial robustness of neural networks. We hope the theoretical and empirical insights can provide a deeper understanding of the impact of the gradient descent training on feature learning process, which in turn influences the robustness of the network, and how more detailed supervision may enhance model robustness.

摘要: 在这项工作中，我们研究了梯度下降训练过程中的一种特殊的隐偏差，我们称之为特征平均，并认为它是导致深度神经网络非稳健性的主要因素之一。尽管存在能够对数据进行分类的多个区别性特征，但通过梯度下降训练的神经网络显示出学习这些特征的平均值(或某些组合)的倾向，而不是分别区分和利用每个特征。特别地，我们对二分类任务的两层RELU网络中的梯度下降训练动态进行了详细的理论分析，其中数据分布由具有正交聚类中心向量的多个聚类组成。我们严格地证明了梯度下降收敛于特征平均，其中每个隐含层神经元的权值代表聚类中心的平均值(每个中心对应于一个不同的特征)。由于攻击与平均特征的负方向一致，导致网络分类器不稳健。此外，我们还证明了，在提供更细粒度的监督信息的情况下，两层多类神经网络能够学习个体特征，由此可以得到在我们的设置下具有最优鲁棒性的二进制分类器。此外，我们还使用合成数据集、MNIST和CIFAR-10进行了大量的实验，以证实特征平均现象及其在神经网络对抗健壮性中的作用。我们希望这些理论和经验的见解能够更深入地理解梯度下降训练对特征学习过程的影响，进而影响网络的稳健性，以及更详细的监督如何增强模型的稳健性。



## **26. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation**

DD-RobustBench：数据集蒸馏的对抗稳健性基准 cs.CV

* denotes equal contributions; ^ denotes corresponding author. In  this updated version, we have expanded our research to include more  experiments on various adversarial attack methods and latest dataset  distillation studies. All new results have been incorporated into the  document

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2403.13322v3) [paper-pdf](http://arxiv.org/pdf/2403.13322v3)

**Authors**: Yifan Wu, Jiawei Du, Ping Liu, Yuewei Lin, Wei Xu, Wenqing Cheng

**Abstract**: Dataset distillation is an advanced technique aimed at compressing datasets into significantly smaller counterparts, while preserving formidable training performance. Significant efforts have been devoted to promote evaluation accuracy under limited compression ratio while overlooked the robustness of distilled dataset. In this work, we introduce a comprehensive benchmark that, to the best of our knowledge, is the most extensive to date for evaluating the adversarial robustness of distilled datasets in a unified way. Our benchmark significantly expands upon prior efforts by incorporating a wider range of dataset distillation methods, including the latest advancements such as TESLA and SRe2L, a diverse array of adversarial attack methods, and evaluations across a broader and more extensive collection of datasets such as ImageNet-1K. Moreover, we assessed the robustness of these distilled datasets against representative adversarial attack algorithms like PGD and AutoAttack, while exploring their resilience from a frequency perspective. We also discovered that incorporating distilled data into the training batches of the original dataset can yield to improvement of robustness.

摘要: 数据集精馏是一种高级技术，旨在将数据集压缩成小得多的对应物，同时保持强大的训练性能。人们一直致力于提高有限压缩比下的评估精度，而忽略了提取数据集的稳健性。在这项工作中，我们引入了一个全面的基准，据我们所知，这是到目前为止最广泛的评估提取数据集的对抗稳健性的统一方式。我们的基准显著扩展了之前的工作，纳入了更广泛的数据集蒸馏方法，包括最新的进步，如特斯拉和SRe2L，多种对抗性攻击方法，以及对更广泛的数据集集合(如ImageNet-1K)的评估。此外，我们评估了这些提取的数据集对PGD和AutoAttack等典型对抗性攻击算法的健壮性，同时从频率的角度探讨了它们的弹性。我们还发现，将提取的数据结合到原始数据集的训练批次中可以提高稳健性。



## **27. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

针对大型视觉语言模型的白盒多模式越狱 cs.CV

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2405.17894v2) [paper-pdf](http://arxiv.org/pdf/2405.17894v2)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.

摘要: 大型视觉语言模型(VLM)的最新进展凸显了它们在各种多通道任务中的优越性。然而，VLMS的对抗健壮性还没有得到充分的研究。现有的方法主要通过扰乱图像的单峰对抗性攻击来评估稳健性，同时假设对基于文本的攻击具有内在的弹性。与已有的攻击不同，我们提出了一种更全面的策略，联合攻击文本和图像模式，以利用VLM中更广泛的漏洞。具体地说，我们提出了一个双重优化目标，旨在引导模型产生高毒性的肯定反应。我们的攻击方法首先从随机噪声中优化一个敌意图像前缀，在没有文本输入的情况下产生不同的有害响应，从而使图像充满有毒语义。随后，对抗性文本后缀与对抗性图像前缀集成并共同优化，以最大限度地引起对各种有害指令的肯定响应的概率。所发现的敌意图像前缀和文本后缀统称为通用主密钥(UMK)。当集成到各种恶意查询中时，UMK可以绕过VLM的对齐防御，并导致生成令人反感的内容，即所谓的越狱。实验结果表明，我们的通用攻击策略能够有效地越狱MiniGPT-4，成功率为96%，凸显了VLMS的脆弱性和对新的对齐策略的迫切需求。



## **28. Out-of-Bounding-Box Triggers: A Stealthy Approach to Cheat Object Detectors**

越界盒触发器：作弊物体检测器的隐形方法 cs.CV

ECCV 2024

**SubmitDate**: 2024-10-14    [abs](http://arxiv.org/abs/2410.10091v1) [paper-pdf](http://arxiv.org/pdf/2410.10091v1)

**Authors**: Tao Lin, Lijia Yu, Gaojie Jin, Renjue Li, Peng Wu, Lijun Zhang

**Abstract**: In recent years, the study of adversarial robustness in object detection systems, particularly those based on deep neural networks (DNNs), has become a pivotal area of research. Traditional physical attacks targeting object detectors, such as adversarial patches and texture manipulations, directly manipulate the surface of the object. While these methods are effective, their overt manipulation of objects may draw attention in real-world applications. To address this, this paper introduces a more subtle approach: an inconspicuous adversarial trigger that operates outside the bounding boxes, rendering the object undetectable to the model. We further enhance this approach by proposing the Feature Guidance (FG) technique and the Universal Auto-PGD (UAPGD) optimization strategy for crafting high-quality triggers. The effectiveness of our method is validated through extensive empirical testing, demonstrating its high performance in both digital and physical environments. The code and video will be available at: https://github.com/linToTao/Out-of-bbox-attack.

摘要: 近年来，目标检测系统，特别是基于深度神经网络(DNN)的目标检测系统中对抗鲁棒性的研究已经成为一个重要的研究领域。传统的针对对象检测器的物理攻击，如对抗性补丁和纹理操作，直接操作对象的表面。虽然这些方法是有效的，但它们对对象的公开操作可能会在现实世界的应用程序中引起注意。为了解决这个问题，本文引入了一种更微妙的方法：在边界框外操作的一个不明显的对抗性触发器，使得对象对于模型来说是不可检测的。我们通过提出特征引导(FG)技术和通用自动PGD(UAPGD)优化策略来进一步增强这一方法，以创建高质量的触发器。通过大量的实验测试，验证了该方法的有效性，证明了该方法在数字和物理环境中的高性能。代码和视频将在以下网站上获得：https://github.com/linToTao/Out-of-bbox-attack.



## **29. The Role of Fake Users in Sequential Recommender Systems**

假用户在顺序推荐系统中的角色 cs.IR

10 pages, 2 figures

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09936v1) [paper-pdf](http://arxiv.org/pdf/2410.09936v1)

**Authors**: Filippo Betello

**Abstract**: Sequential Recommender Systems (SRSs) are widely used to model user behavior over time, yet their robustness remains an under-explored area of research. In this paper, we conduct an empirical study to assess how the presence of fake users, who engage in random interactions, follow popular or unpopular items, or focus on a single genre, impacts the performance of SRSs in real-world scenarios. We evaluate two SRS models across multiple datasets, using established metrics such as Normalized Discounted Cumulative Gain (NDCG) and Rank Sensitivity List (RLS) to measure performance. While traditional metrics like NDCG remain relatively stable, our findings reveal that the presence of fake users severely degrades RLS metrics, often reducing them to near-zero values. These results highlight the need for further investigation into the effects of fake users on training data and emphasize the importance of developing more resilient SRSs that can withstand different types of adversarial attacks.

摘要: 顺序推荐系统（SR）被广泛用于对用户随时间的行为进行建模，但其稳健性仍然是一个未充分探索的研究领域。在本文中，我们进行了一项实证研究，以评估虚假用户（参与随机互动、关注流行或不受欢迎的项目或专注于单一类型）的存在如何影响现实世界场景中SR的性能。我们使用标准化贴现累积收益（NDCG）和等级敏感度列表（SLS）等既定指标来评估多个数据集中的两个RS模型来衡量性能。虽然NDCG等传统指标保持相对稳定，但我们的研究结果表明，虚假用户的存在严重降低了SLS指标，通常将其降低到接近零的值。这些结果凸显了进一步调查虚假用户对训练数据的影响的必要性，并强调了开发能够抵御不同类型对抗攻击的更具弹性的SR的重要性。



## **30. Extreme Miscalibration and the Illusion of Adversarial Robustness**

极端失调和对抗稳健性错觉 cs.CL

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2402.17509v3) [paper-pdf](http://arxiv.org/pdf/2402.17509v3)

**Authors**: Vyas Raina, Samson Tan, Volkan Cevher, Aditya Rawal, Sheng Zha, George Karypis

**Abstract**: Deep learning-based Natural Language Processing (NLP) models are vulnerable to adversarial attacks, where small perturbations can cause a model to misclassify. Adversarial Training (AT) is often used to increase model robustness. However, we have discovered an intriguing phenomenon: deliberately or accidentally miscalibrating models masks gradients in a way that interferes with adversarial attack search methods, giving rise to an apparent increase in robustness. We show that this observed gain in robustness is an illusion of robustness (IOR), and demonstrate how an adversary can perform various forms of test-time temperature calibration to nullify the aforementioned interference and allow the adversarial attack to find adversarial examples. Hence, we urge the NLP community to incorporate test-time temperature scaling into their robustness evaluations to ensure that any observed gains are genuine. Finally, we show how the temperature can be scaled during \textit{training} to improve genuine robustness.

摘要: 基于深度学习的自然语言处理(NLP)模型容易受到敌意攻击，其中微小的扰动可能会导致模型错误分类。对抗性训练(AT)通常被用来增强模型的稳健性。然而，我们发现了一个有趣的现象：故意或意外地错误校准模型以干扰对抗性攻击搜索方法的方式掩盖了梯度，从而产生了明显的健壮性增强。我们证明了这种观察到的健壮性增长是健壮性错觉(IOR)，并演示了对手如何执行各种形式的测试时间温度校准来抵消上述干扰，并允许对手攻击找到对手的例子。因此，我们敦促NLP社区将测试时间温度调整纳入其稳健性评估，以确保任何观察到的收益都是真实的。最后，我们展示了如何在训练期间调整温度以提高真正的健壮性。



## **31. Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning**

数据中毒情况下可证明可靠的保形预测集 cs.LG

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09878v1) [paper-pdf](http://arxiv.org/pdf/2410.09878v1)

**Authors**: Yan Scholten, Stephan Günnemann

**Abstract**: Conformal prediction provides model-agnostic and distribution-free uncertainty quantification through prediction sets that are guaranteed to include the ground truth with any user-specified probability. Yet, conformal prediction is not reliable under poisoning attacks where adversaries manipulate both training and calibration data, which can significantly alter prediction sets in practice. As a solution, we propose reliable prediction sets (RPS): the first efficient method for constructing conformal prediction sets with provable reliability guarantees under poisoning. To ensure reliability under training poisoning, we introduce smoothed score functions that reliably aggregate predictions of classifiers trained on distinct partitions of the training data. To ensure reliability under calibration poisoning, we construct multiple prediction sets, each calibrated on distinct subsets of the calibration data. We then aggregate them into a majority prediction set, which includes a class only if it appears in a majority of the individual sets. Both proposed aggregations mitigate the influence of datapoints in the training and calibration data on the final prediction set. We experimentally validate our approach on image classification tasks, achieving strong reliability while maintaining utility and preserving coverage on clean data. Overall, our approach represents an important step towards more trustworthy uncertainty quantification in the presence of data poisoning.

摘要: 保角预测通过预测集提供与模型无关和无分布的不确定性量化，这些预测集保证以任何用户指定的概率包括基本事实。然而，在中毒攻击下，保角预测是不可靠的，其中对手同时操纵训练和校准数据，这在实践中可能会显著改变预测集。作为解决方案，我们提出了可靠预测集(RPS)：在中毒情况下构造具有可证明可靠性保证的共形预测集的第一种有效方法。为了确保在训练中毒情况下的可靠性，我们引入了平滑得分函数，它可靠地聚合了在不同的训练数据分区上训练的分类器的预测。为了确保在校准中毒情况下的可靠性，我们构造了多个预测集，每个预测集都在校准数据的不同子集上进行校准。然后我们将它们聚集到一个多数预测集合中，该集合只包括一个类，当它出现在大多数单独的集合中时。这两种建议的聚合都减轻了训练和校准数据中的数据点对最终预测集的影响。我们在实验上验证了我们的方法在图像分类任务上的有效性，在保持实用性和对干净数据的覆盖率的同时实现了很强的可靠性。总体而言，我们的方法代表着在存在数据中毒的情况下朝着更可信的不确定性量化迈出的重要一步。



## **32. Understanding Robustness of Parameter-Efficient Tuning for Image Classification**

了解图像分类的参数高效调整的稳健性 cs.CV

5 pages, 2 figures. Work in Progress

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09845v1) [paper-pdf](http://arxiv.org/pdf/2410.09845v1)

**Authors**: Jiacheng Ruan, Xian Gao, Suncheng Xiang, Mingye Xie, Ting Liu, Yuzhuo Fu

**Abstract**: Parameter-efficient tuning (PET) techniques calibrate the model's predictions on downstream tasks by freezing the pre-trained models and introducing a small number of learnable parameters. However, despite the numerous PET methods proposed, their robustness has not been thoroughly investigated. In this paper, we systematically explore the robustness of four classical PET techniques (e.g., VPT, Adapter, AdaptFormer, and LoRA) under both white-box attacks and information perturbations. For white-box attack scenarios, we first analyze the performance of PET techniques using FGSM and PGD attacks. Subsequently, we further explore the transferability of adversarial samples and the impact of learnable parameter quantities on the robustness of PET methods. Under information perturbation attacks, we introduce four distinct perturbation strategies, including Patch-wise Drop, Pixel-wise Drop, Patch Shuffle, and Gaussian Noise, to comprehensively assess the robustness of these PET techniques in the presence of information loss. Via these extensive studies, we enhance the understanding of the robustness of PET methods, providing valuable insights for improving their performance in computer vision applications. The code is available at https://github.com/JCruan519/PETRobustness.

摘要: 参数高效调整(PET)技术通过冻结预先训练的模型并引入少量可学习的参数来校准模型对下游任务的预测。然而，尽管提出了许多PET方法，但它们的稳健性还没有得到彻底的研究。在本文中，我们系统地研究了四种经典的PET技术(如VPT、Adapter、AdaptFormer和LORA)在白盒攻击和信息扰动下的稳健性。对于白盒攻击场景，我们首先分析了使用FGSM和PGD攻击的PET技术的性能。随后，我们进一步探讨了对抗性样本的可转移性以及可学习参数对PET方法稳健性的影响。在信息扰动攻击下，我们引入了四种不同的扰动策略，包括Patch-Wise Drop、Pixel-Drop、Patch Shuffle和Gauss Noise，以综合评估这些PET技术在信息丢失情况下的稳健性。通过这些广泛的研究，我们加深了对PET方法稳健性的理解，为提高其在计算机视觉应用中的性能提供了有价值的见解。代码可在https://github.com/JCruan519/PETRobustness.上获得



## **33. Robust 3D Point Clouds Classification based on Declarative Defenders**

基于声明性防御者的稳健3D点云分类 cs.CV

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09691v1) [paper-pdf](http://arxiv.org/pdf/2410.09691v1)

**Authors**: Kaidong Li, Tianxiao Zhang, Chuncong Zhong, Ziming Zhang, Guanghui Wang

**Abstract**: 3D point cloud classification requires distinct models from 2D image classification due to the divergent characteristics of the respective input data. While 3D point clouds are unstructured and sparse, 2D images are structured and dense. Bridging the domain gap between these two data types is a non-trivial challenge to enable model interchangeability. Recent research using Lattice Point Classifier (LPC) highlights the feasibility of cross-domain applicability. However, the lattice projection operation in LPC generates 2D images with disconnected projected pixels. In this paper, we explore three distinct algorithms for mapping 3D point clouds into 2D images. Through extensive experiments, we thoroughly examine and analyze their performance and defense mechanisms. Leveraging current large foundation models, we scrutinize the feature disparities between regular 2D images and projected 2D images. The proposed approaches demonstrate superior accuracy and robustness against adversarial attacks. The generative model-based mapping algorithms yield regular 2D images, further minimizing the domain gap from regular 2D classification tasks. The source code is available at https://github.com/KaidongLi/pytorch-LatticePointClassifier.git.

摘要: 由于各个输入数据的发散特性，三维点云分类需要与二维图像分类不同的模型。虽然三维点云是非结构化的和稀疏的，但二维图像是结构化的和密集的。弥合这两种数据类型之间的域差距是实现模型互换性的一个不小的挑战。最近使用格点分类器(LPC)的研究突出了跨域适用性的可行性。然而，LPC中的晶格投影操作会生成具有断开的投影像素的2D图像。在本文中，我们探索了三种不同的算法来将3D点云映射到2D图像。通过大量的实验，我们对它们的性能和防御机制进行了深入的检测和分析。利用当前的大型基础模型，我们仔细研究了常规2D图像和投影2D图像之间的特征差异。所提出的方法在抵抗敌意攻击时表现出了优越的准确性和鲁棒性。基于生成模型的映射算法生成规则的2D图像，进一步最小化了与常规2D分类任务之间的域差距。源代码可在https://github.com/KaidongLi/pytorch-LatticePointClassifier.git.上找到



## **34. Uncovering Attacks and Defenses in Secure Aggregation for Federated Deep Learning**

揭露联邦深度学习安全聚合中的攻击和防御 cs.CR

**SubmitDate**: 2024-10-13    [abs](http://arxiv.org/abs/2410.09676v1) [paper-pdf](http://arxiv.org/pdf/2410.09676v1)

**Authors**: Yiwei Zhang, Rouzbeh Behnia, Attila A. Yavuz, Reza Ebrahimi, Elisa Bertino

**Abstract**: Federated learning enables the collaborative learning of a global model on diverse data, preserving data locality and eliminating the need to transfer user data to a central server. However, data privacy remains vulnerable, as attacks can target user training data by exploiting the updates sent by users during each learning iteration. Secure aggregation protocols are designed to mask/encrypt user updates and enable a central server to aggregate the masked information. MicroSecAgg (PoPETS 2024) proposes a single server secure aggregation protocol that aims to mitigate the high communication complexity of the existing approaches by enabling a one-time setup of the secret to be re-used in multiple training iterations. In this paper, we identify a security flaw in the MicroSecAgg that undermines its privacy guarantees. We detail the security flaw and our attack, demonstrating how an adversary can exploit predictable masking values to compromise user privacy. Our findings highlight the critical need for enhanced security measures in secure aggregation protocols, particularly the implementation of dynamic and unpredictable masking strategies. We propose potential countermeasures to mitigate these vulnerabilities and ensure robust privacy protection in the secure aggregation frameworks.

摘要: 联合学习实现了对不同数据的全球模型的协作学习，保留了数据的局部性，消除了将用户数据传输到中央服务器的需要。然而，数据隐私仍然很容易受到攻击，因为攻击可以通过利用用户在每次学习迭代期间发送的更新来攻击用户训练数据。安全聚合协议旨在屏蔽/加密用户更新，并使中央服务器能够聚合屏蔽的信息。MicroSecAgg(PoPETS 2024)提出了一种单服务器安全聚合协议，该协议旨在通过允许一次性设置秘密以在多次训练迭代中重复使用来缓解现有方法的高度通信复杂性。在本文中，我们发现了MicroSecAgg中的一个安全漏洞，该漏洞破坏了其隐私保障。我们详细介绍了安全漏洞和我们的攻击，展示了对手如何利用可预测的掩蔽值来危害用户隐私。我们的发现强调了在安全聚合协议中增强安全措施的迫切需要，特别是实施动态和不可预测的掩蔽策略。我们提出了潜在的对策来缓解这些漏洞，并确保在安全聚合框架中提供强大的隐私保护。



## **35. Unlearn and Burn: Adversarial Machine Unlearning Requests Destroy Model Accuracy**

取消学习和烧毁：对抗性机器取消学习请求破坏模型准确性 cs.CR

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2410.09591v1) [paper-pdf](http://arxiv.org/pdf/2410.09591v1)

**Authors**: Yangsibo Huang, Daogao Liu, Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Milad Nasr, Amer Sinha, Chiyuan Zhang

**Abstract**: Machine unlearning algorithms, designed for selective removal of training data from models, have emerged as a promising approach to growing privacy concerns. In this work, we expose a critical yet underexplored vulnerability in the deployment of unlearning systems: the assumption that the data requested for removal is always part of the original training set. We present a threat model where an attacker can degrade model accuracy by submitting adversarial unlearning requests for data not present in the training set. We propose white-box and black-box attack algorithms and evaluate them through a case study on image classification tasks using the CIFAR-10 and ImageNet datasets, targeting a family of widely used unlearning methods. Our results show extremely poor test accuracy following the attack: 3.6% on CIFAR-10 and 0.4% on ImageNet for white-box attacks, and 8.5% on CIFAR-10 and 1.3% on ImageNet for black-box attacks. Additionally, we evaluate various verification mechanisms to detect the legitimacy of unlearning requests and reveal the challenges in verification, as most of the mechanisms fail to detect stealthy attacks without severely impairing their ability to process valid requests. These findings underscore the urgent need for research on more robust request verification methods and unlearning protocols, should the deployment of machine unlearning systems become more prevalent in the future.

摘要: 机器遗忘算法是为选择性地从模型中移除训练数据而设计的，已成为解决日益增长的隐私问题的一种有前途的方法。在这项工作中，我们暴露了遗忘系统部署中的一个关键但未被探索的漏洞：假设请求移除的数据始终是原始训练集的一部分。我们提出了一个威胁模型，其中攻击者可以通过提交对训练集中不存在的数据的敌意遗忘请求来降低模型的准确性。我们提出了白盒和黑盒攻击算法，并针对一类广泛使用的遗忘方法，使用CIFAR-10和ImageNet数据集对图像分类任务进行了评估。我们的结果表明，攻击后的测试准确率非常低：对于白盒攻击，CIFAR-10和ImageNet的测试准确率分别为3.6%和0.4%，对于黑盒攻击，CIFAR-10和ImageNet的测试准确率分别为8.5%和1.3%。此外，我们对各种验证机制进行了评估，以检测遗忘请求的合法性并揭示验证中的挑战，因为大多数机制无法在不严重损害其处理有效请求的能力的情况下检测到隐蔽攻击。这些发现强调，如果机器遗忘系统的部署在未来变得更加普遍，迫切需要研究更健壮的请求验证方法和遗忘协议。



## **36. Differentially Private and Byzantine-Resilient Decentralized Nonconvex Optimization: System Modeling, Utility, Resilience, and Privacy Analysis**

差异私有和拜占庭弹性去中心化非凸优化：系统建模、效用、弹性和隐私分析 math.OC

13 pages, 13 figures

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2409.18632v5) [paper-pdf](http://arxiv.org/pdf/2409.18632v5)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.

摘要: 隐私泄露和拜占庭失效是影响多智能体系统(MASS)智能决策过程的两个不利因素。考虑到这两个问题的存在，本文研究了一类在Polyak-L条件下的非凸优化问题的解。为了解决这一问题，我们首先识别并构建了对手系统模型。为了增强随机梯度下降算法的稳健性，我们用高斯噪声掩盖局部梯度，并采用弹性聚合方法自中心剪裁(SCC)设计了一种差分私有(DP)分散拜占庭弹性算法DP-SCC-PL，同时实现了差分隐私保护和拜占庭弹性。DP-SCC-PL的收敛分析具有挑战性，因为收敛误差是由隐私保护和拜占庭弹性机制以及非凸松弛机制共同造成的，非凸松弛通过寻找可靠代理聚集前后的不一致度量和最优间隙之间的收缩关系来解决。理论结果表明，DP-SCC-PL算法在合理设计步长的情况下，实现了所有可靠代理之间的一致性和次线性(不精确)收敛。证明了如果不存在隐私问题和拜占庭代理，则可以恢复渐近精确收敛。通过求解满足P-L条件的非凸优化问题，验证了DP-SCC-PL在不同拜占庭攻击下的实用性、抗攻击能力和差分隐私性。



## **37. Minimax rates of convergence for nonparametric regression under adversarial attacks**

对抗攻击下非参数回归的极小极大收敛率 math.ST

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2410.09402v1) [paper-pdf](http://arxiv.org/pdf/2410.09402v1)

**Authors**: Jingfu Peng, Yuhong Yang

**Abstract**: Recent research shows the susceptibility of machine learning models to adversarial attacks, wherein minor but maliciously chosen perturbations of the input can significantly degrade model performance. In this paper, we theoretically analyse the limits of robustness against such adversarial attacks in a nonparametric regression setting, by examining the minimax rates of convergence in an adversarial sup-norm. Our work reveals that the minimax rate under adversarial attacks in the input is the same as sum of two terms: one represents the minimax rate in the standard setting without adversarial attacks, and the other reflects the maximum deviation of the true regression function value within the target function class when subjected to the input perturbations. The optimal rates under the adversarial setup can be achieved by a plug-in procedure constructed from a minimax optimal estimator in the corresponding standard setting. Two specific examples are given to illustrate the established minimax results.

摘要: 最近的研究表明，机器学习模型对敌意攻击很敏感，其中输入的微小但恶意选择的扰动会显著降低模型的性能。本文通过检验对抗性超范数下的极小极大收敛速度，从理论上分析了在非参数回归环境下抵抗此类对抗性攻击的稳健性极限。我们的工作表明，在对抗性攻击下，输入的极小极大率等于两项之和：一项表示没有对抗性攻击的标准设置下的极小极大率，另一项反映了输入扰动下目标函数类内真实回归函数值的最大偏差。对抗性设置下的最优速率可以通过由相应标准设置中的极小极大最优估计器构造的插件程序来实现。给出了两个具体的例子来说明所建立的极小极大结果。



## **38. Targeted Attack Improves Protection against Unauthorized Diffusion Customization**

有针对性的攻击提高了对未经授权的扩散定制的保护 cs.CV

**SubmitDate**: 2024-10-12    [abs](http://arxiv.org/abs/2310.04687v4) [paper-pdf](http://arxiv.org/pdf/2310.04687v4)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu

**Abstract**: Diffusion models build a new milestone for image generation yet raising public concerns, for they can be fine-tuned on unauthorized images for customization. Protection based on adversarial attacks rises to encounter this unauthorized diffusion customization, by adding protective watermarks to images and poisoning diffusion models. However, current protection, leveraging untargeted attacks, does not appear to be effective enough. In this paper, we propose a simple yet effective improvement for the protection against unauthorized diffusion customization by introducing targeted attacks. We show that by carefully selecting the target, targeted attacks significantly outperform untargeted attacks in poisoning diffusion models and degrading the customization image quality. Extensive experiments validate the superiority of our method on two mainstream customization methods of diffusion models, compared to existing protections. To explain the surprising success of targeted attacks, we delve into the mechanism of attack-based protections and propose a hypothesis based on our observation, which enhances the comprehension of attack-based protections. To the best of our knowledge, we are the first to both reveal the vulnerability of diffusion models to targeted attacks and leverage targeted attacks to enhance protection against unauthorized diffusion customization. Our code is available on GitHub: \url{https://github.com/psyker-team/mist-v2}.

摘要: 扩散模型为图像生成建立了一个新的里程碑，但也引起了公众的关注，因为它们可以对未经授权的图像进行微调以进行定制。基于对抗性攻击的保护通过向图像添加保护性水印和毒化扩散模型来遇到这种未经授权的扩散定制。然而，目前利用非目标攻击的保护似乎不够有效。在本文中，我们提出了一种简单而有效的改进，通过引入有针对性的攻击来防止未经授权的扩散定制。我们表明，通过仔细选择目标，目标攻击在毒化扩散模型和降低定制图像质量方面显著优于非目标攻击。大量实验验证了该方法在两种主流扩散模型定制方法上的优越性，并与现有的保护方法进行了比较。为了解释定向攻击的惊人成功，我们深入研究了基于攻击的保护机制，并根据我们的观察提出了一个假设，这增强了对基于攻击的保护的理解。据我们所知，我们是第一个揭示扩散模型对目标攻击的脆弱性，并利用目标攻击来增强对未经授权的扩散定制的保护。我们的代码可在giHub：\url{https://github.com/psyker-team/mist-v2}.



## **39. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

控制耳语：控制语音基础模型的通用声学对抗攻击 cs.SD

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.04482v2) [paper-pdf](http://arxiv.org/pdf/2407.04482v2)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.

摘要: 以灵活的基于语音识别的系统或音频提示的大型语言模型(LLM)的形式启用语音的基础模型正变得越来越受欢迎。这些模型的一个有趣方面是，它们能够使用适当的提示执行自动语音识别(ASR)以外的任务。例如，OpenAI Whisper模型可以执行语音转录和语音翻译。随着音频提示LLMS的发展，有可能出现更大的控制选项。在这项工作中，我们证明了有了这种更大的灵活性，系统可以容易受到模型控制的对抗性攻击。在不访问模型提示的情况下，可以通过适当地改变音频输入来修改系统的行为。为了说明这一风险，我们证明了有可能在任何输入语音信号之前添加一个简短的通用对抗性声学片段，以覆盖ASR基础模型的提示设置。具体地说，我们成功地使用了一个通用的对抗性声学段来控制Whisper始终执行语音翻译，尽管被设置为执行语音转录。总体而言，这项工作展示了一种对多任务语音启用的基础模型的新形式的对抗性攻击，在部署这种形式的模型之前需要考虑这种形式。



## **40. On the Adversarial Transferability of Generalized "Skip Connections"**

广义“跳过连接”的对抗性可转让性 cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08950v1) [paper-pdf](http://arxiv.org/pdf/2410.08950v1)

**Authors**: Yisen Wang, Yichuan Mo, Dongxian Wu, Mingjie Li, Xingjun Ma, Zhouchen Lin

**Abstract**: Skip connection is an essential ingredient for modern deep models to be deeper and more powerful. Despite their huge success in normal scenarios (state-of-the-art classification performance on natural examples), we investigate and identify an interesting property of skip connections under adversarial scenarios, namely, the use of skip connections allows easier generation of highly transferable adversarial examples. Specifically, in ResNet-like models (with skip connections), we find that using more gradients from the skip connections rather than the residual modules according to a decay factor during backpropagation allows one to craft adversarial examples with high transferability. The above method is termed as Skip Gradient Method (SGM). Although starting from ResNet-like models in vision domains, we further extend SGM to more advanced architectures, including Vision Transformers (ViTs) and models with length-varying paths and other domains, i.e. natural language processing. We conduct comprehensive transfer attacks against various models including ResNets, Transformers, Inceptions, Neural Architecture Search, and Large Language Models (LLMs). We show that employing SGM can greatly improve the transferability of crafted attacks in almost all cases. Furthermore, considering the big complexity for practical use, we further demonstrate that SGM can even improve the transferability on ensembles of models or targeted attacks and the stealthiness against current defenses. At last, we provide theoretical explanations and empirical insights on how SGM works. Our findings not only motivate new adversarial research into the architectural characteristics of models but also open up further challenges for secure model architecture design. Our code is available at https://github.com/mo666666/SGM.

摘要: 跳过连接是现代深层模型更深入、更强大的关键因素。尽管它们在正常场景中取得了巨大的成功(在自然示例上的最新分类性能)，但我们调查并识别了对抗性场景下跳过连接的一个有趣属性，即使用跳过连接可以更容易地生成高度可转移的对抗性示例。具体地说，在类ResNet模型(带有跳过连接)中，我们发现在反向传播过程中，根据衰减因子使用来自跳过连接的更多梯度，而不是使用剩余模块，可以创建具有高可转移性的对抗性例子。上述方法被称为跳过梯度法(SGM)。虽然我们从视觉领域中类似ResNet的模型开始，但我们将SGM进一步扩展到更高级的体系结构，包括视觉转换器(VITS)和具有变长度路径的模型以及其他领域，即自然语言处理。我们针对不同的模型进行全面的传输攻击，包括ResNet、Transformers、Inceptions、Neural Architecture Search和Large Language Model(LLM)。我们表明，在几乎所有情况下，使用SGM都可以极大地提高精心设计的攻击的可转移性。此外，考虑到实际应用的巨大复杂性，我们进一步证明了SGM甚至可以提高模型集成或定向攻击的可转换性和对现有防御的隐蔽性。最后，本文对SGM的运行机制进行了理论解释和实证分析。我们的发现不仅激发了对模型体系结构特征的新的对抗性研究，而且也为安全模型体系结构设计开辟了进一步的挑战。我们的代码可以在https://github.com/mo666666/SGM.上找到



## **41. Fragile Giants: Understanding the Susceptibility of Models to Subpopulation Attacks**

脆弱的巨人：了解模型对亚群攻击的敏感性 cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08872v1) [paper-pdf](http://arxiv.org/pdf/2410.08872v1)

**Authors**: Isha Gupta, Hidde Lycklama, Emanuel Opel, Evan Rose, Anwar Hithnawi

**Abstract**: As machine learning models become increasingly complex, concerns about their robustness and trustworthiness have become more pressing. A critical vulnerability of these models is data poisoning attacks, where adversaries deliberately alter training data to degrade model performance. One particularly stealthy form of these attacks is subpopulation poisoning, which targets distinct subgroups within a dataset while leaving overall performance largely intact. The ability of these attacks to generalize within subpopulations poses a significant risk in real-world settings, as they can be exploited to harm marginalized or underrepresented groups within the dataset. In this work, we investigate how model complexity influences susceptibility to subpopulation poisoning attacks. We introduce a theoretical framework that explains how overparameterized models, due to their large capacity, can inadvertently memorize and misclassify targeted subpopulations. To validate our theory, we conduct extensive experiments on large-scale image and text datasets using popular model architectures. Our results show a clear trend: models with more parameters are significantly more vulnerable to subpopulation poisoning. Moreover, we find that attacks on smaller, human-interpretable subgroups often go undetected by these models. These results highlight the need to develop defenses that specifically address subpopulation vulnerabilities.

摘要: 随着机器学习模型变得越来越复杂，人们对其健壮性和可信度的担忧也变得更加紧迫。这些模型的一个关键漏洞是数据中毒攻击，即攻击者故意更改训练数据以降低模型的性能。这些攻击的一种特别隐蔽的形式是子种群中毒，它以数据集中不同的子群为目标，而总体性能基本保持不变。这些攻击在子群体中泛化的能力在现实世界环境中构成了重大风险，因为它们可能被利用来伤害数据集中被边缘化或代表性不足的群体。在这项工作中，我们研究了模型复杂性如何影响对子种群中毒攻击的敏感性。我们介绍了一个理论框架，解释了过度参数化的模型，由于其容量大，可能会无意中记忆和错误分类目标子种群。为了验证我们的理论，我们使用流行的模型架构在大规模的图像和文本数据集上进行了广泛的实验。我们的结果显示了一个明显的趋势：参数越多的模型越容易受到子种群的毒害。此外，我们发现，对较小的、人类可解释的子组的攻击通常不会被这些模型检测到。这些结果突显了开发专门针对亚群体脆弱性的防御措施的必要性。



## **42. The Good, the Bad and the Ugly: Watermarks, Transferable Attacks and Adversarial Defenses**

好的、坏的和丑陋的：水印、可转移攻击和对抗性防御 cs.LG

42 pages, 6 figures, preliminary version published in ICML 2024  (Workshop on Theoretical Foundations of Foundation Models), see  https://openreview.net/pdf?id=WMaFRiggwV

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08864v1) [paper-pdf](http://arxiv.org/pdf/2410.08864v1)

**Authors**: Grzegorz Głuch, Berkant Turan, Sai Ganesh Nagarajan, Sebastian Pokutta

**Abstract**: We formalize and extend existing definitions of backdoor-based watermarks and adversarial defenses as interactive protocols between two players. The existence of these schemes is inherently tied to the learning tasks for which they are designed. Our main result shows that for almost every discriminative learning task, at least one of the two -- a watermark or an adversarial defense -- exists. The term "almost every" indicates that we also identify a third, counterintuitive but necessary option, i.e., a scheme we call a transferable attack. By transferable attack, we refer to an efficient algorithm computing queries that look indistinguishable from the data distribution and fool all efficient defenders. To this end, we prove the necessity of a transferable attack via a construction that uses a cryptographic tool called homomorphic encryption. Furthermore, we show that any task that satisfies our notion of a transferable attack implies a cryptographic primitive, thus requiring the underlying task to be computationally complex. These two facts imply an "equivalence" between the existence of transferable attacks and cryptography. Finally, we show that the class of tasks of bounded VC-dimension has an adversarial defense, and a subclass of them has a watermark.

摘要: 我们将基于后门的水印和对抗性防御的现有定义形式化并扩展为两个参与者之间的交互协议。这些方案的存在与它们设计的学习任务内在地联系在一起。我们的主要结果表明，对于几乎每一项歧视性学习任务，至少存在两种任务中的一种--水印或对抗性防御。术语“几乎每一个”表明，我们还确定了第三种违反直觉但必要的选择，即我们称之为可转移攻击的方案。在可转移攻击中，我们指的是一种计算查询的高效算法，这些查询看起来与数据分布没有区别，并欺骗了所有有效的防御者。为此，我们通过使用称为同态加密的密码工具的构造来证明可转移攻击的必要性。此外，我们证明了任何满足我们的可转移攻击概念的任务都隐含着密码原语，因此要求底层任务在计算上是复杂的。这两个事实暗示了可转移攻击的存在与密码学之间的“等价性”。最后，我们证明了有界VC维的任务类具有对抗防御，并且它们的一个子类具有水印。



## **43. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08827v1) [paper-pdf](http://arxiv.org/pdf/2410.08827v1)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.

摘要: 大型语言模型关于如何执行网络安全攻击、制造生物武器和操纵人类的知识带来了滥用的风险。之前的工作提出了忘记这些知识的方法。从历史上看，目前尚不清楚取消学习技术是否正在从模型权重中删除信息，或者只是使其更难访问。为了解开这两个目标，我们提出了一种对抗评估方法来测试从模型权重中删除信息的情况：我们让攻击者访问一些应该删除的事实，并使用这些事实，攻击者试图从无法从可访问的事实中猜测到的相同分布中恢复其他事实。我们表明，当应用于当前的取消学习方法时，对可访问的事实进行微调可以恢复取消学习前的88%的准确性，揭示了这些方法在从模型权重中删除信息方面的局限性。



## **44. Training on Fake Labels: Mitigating Label Leakage in Split Learning via Secure Dimension Transformation**

假标签培训：通过安全维度转换减轻拆分学习中的标签泄露 cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.09125v1) [paper-pdf](http://arxiv.org/pdf/2410.09125v1)

**Authors**: Yukun Jiang, Peiran Wang, Chengguo Lin, Ziyue Huang, Yong Cheng

**Abstract**: Two-party split learning has emerged as a popular paradigm for vertical federated learning. To preserve the privacy of the label owner, split learning utilizes a split model, which only requires the exchange of intermediate representations (IRs) based on the inputs and gradients for each IR between two parties during the learning process. However, split learning has recently been proven to survive label inference attacks. Though several defense methods could be adopted, they either have limited defensive performance or significantly negatively impact the original mission. In this paper, we propose a novel two-party split learning method to defend against existing label inference attacks while maintaining the high utility of the learned models. Specifically, we first craft a dimension transformation module, SecDT, which could achieve bidirectional mapping between original labels and increased K-class labels to mitigate label leakage from the directional perspective. Then, a gradient normalization algorithm is designed to remove the magnitude divergence of gradients from different classes. We propose a softmax-normalized Gaussian noise to mitigate privacy leakage and make our K unknowable to adversaries. We conducted experiments on real-world datasets, including two binary-classification datasets (Avazu and Criteo) and three multi-classification datasets (MNIST, FashionMNIST, CIFAR-10); we also considered current attack schemes, including direction, norm, spectral, and model completion attacks. The detailed experiments demonstrate our proposed method's effectiveness and superiority over existing approaches. For instance, on the Avazu dataset, the attack AUC of evaluated four prominent attacks could be reduced by 0.4532+-0.0127.

摘要: 两方分裂学习已经成为垂直联合学习的一种流行范式。为了保护标签所有者的隐私，分裂学习使用分裂模型，在学习过程中只需要基于双方之间每个IR的输入和梯度来交换中间表示(IR)。然而，分裂学习最近被证明能够经受住标签推理攻击。虽然可以采取几种防御方法，但它们要么防御性能有限，要么对最初的任务产生重大负面影响。在本文中，我们提出了一种新的两方分裂学习方法来防御现有的标签推理攻击，同时保持了学习模型的高实用性。具体地说，我们首先设计了一个维度转换模块SecDT，该模块可以实现原始标签和增加的K类标签之间的双向映射，从方向上缓解标签泄漏。然后，设计了一种梯度归一化算法，以消除不同类别的梯度的幅度差异。我们提出了一种Softmax归一化的高斯噪声来缓解隐私泄露，并使我们的K对攻击者来说是未知的。我们在真实世界的数据集上进行了实验，包括两个二分类数据集(Avazu和Criteo)和三个多分类数据集(MNIST，FashionMNIST，CIFAR-10)；我们还考虑了现有的攻击方案，包括方向攻击、范数攻击、谱攻击和模型完成攻击。详细的实验证明了我们提出的方法的有效性和优越性。例如，在Avazu数据集上，评估的四种重要攻击的攻击AUC可以减少0.4532+-0.0127。



## **45. Natural Language Induced Adversarial Images**

自然语言引发的对抗图像 cs.CR

Carmera-ready version. To appear in ACM MM 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08620v1) [paper-pdf](http://arxiv.org/pdf/2410.08620v1)

**Authors**: Xiaopei Zhu, Peiyang Xu, Guanning Zeng, Yingpeng Dong, Xiaolin Hu

**Abstract**: Research of adversarial attacks is important for AI security because it shows the vulnerability of deep learning models and helps to build more robust models. Adversarial attacks on images are most widely studied, which include noise-based attacks, image editing-based attacks, and latent space-based attacks. However, the adversarial examples crafted by these methods often lack sufficient semantic information, making it challenging for humans to understand the failure modes of deep learning models under natural conditions. To address this limitation, we propose a natural language induced adversarial image attack method. The core idea is to leverage a text-to-image model to generate adversarial images given input prompts, which are maliciously constructed to lead to misclassification for a target model. To adopt commercial text-to-image models for synthesizing more natural adversarial images, we propose an adaptive genetic algorithm (GA) for optimizing discrete adversarial prompts without requiring gradients and an adaptive word space reduction method for improving query efficiency. We further used CLIP to maintain the semantic consistency of the generated images. In our experiments, we found that some high-frequency semantic information such as "foggy", "humid", "stretching", etc. can easily cause classifier errors. This adversarial semantic information exists not only in generated images but also in photos captured in the real world. We also found that some adversarial semantic information can be transferred to unknown classification tasks. Furthermore, our attack method can transfer to different text-to-image models (e.g., Midjourney, DALL-E 3, etc.) and image classifiers. Our code is available at: https://github.com/zxp555/Natural-Language-Induced-Adversarial-Images.

摘要: 对抗性攻击的研究对人工智能安全具有重要意义，因为它揭示了深度学习模型的脆弱性，有助于建立更健壮的模型。针对图像的对抗性攻击被广泛研究，包括基于噪声的攻击、基于图像编辑的攻击和潜在的基于空间的攻击。然而，这些方法生成的对抗性实例往往缺乏足够的语义信息，这使得人类很难理解自然条件下深度学习模型的失败模式。针对这一局限性，我们提出了一种自然语言诱导的对抗性图像攻击方法。其核心思想是利用文本到图像的模型来生成给定输入提示的对抗性图像，这些提示被恶意构建以导致目标模型的错误分类。为了采用商业的文本到图像模型来合成更多的自然对抗性图像，提出了一种不需要梯度的自适应遗传算法(GA)来优化离散对抗性提示，并提出了一种自适应词空间缩减方法来提高查询效率。我们进一步使用CLIP来保持生成图像的语义一致性。在我们的实验中，我们发现一些高频语义信息，如“雾”、“湿”、“拉伸”等，容易导致分类器错误。这种对抗性的语义信息不仅存在于生成的图像中，而且还存在于现实世界中捕获的照片中。我们还发现，一些对抗性语义信息可以被转移到未知的分类任务中。此外，我们的攻击方法可以转换为不同的文本到图像模型(例如，中途、Dall-E 3等)。和图像分类器。我们的代码请访问：https://github.com/zxp555/Natural-Language-Induced-Adversarial-Images.



## **46. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

ART：文本到图像模型的自动红色团队以保护良性用户 cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2405.19360v3) [paper-pdf](http://arxiv.org/pdf/2405.19360v3)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.

摘要: 由于具有创造内容的能力，大规模的预先训练的生成性模型正在席卷世界。同时，为了保护用户的权利和安全，制定了对这些生成模型的保障措施，其中大部分是为大型语言模型设计的。现有的方法主要针对越狱和对抗性攻击，主要是在恶意提示下对模型的安全性进行评估。最近的研究发现，手动创建的安全提示可能会无意中引发不安全的世代。为了进一步系统地评估文本到图像模型的安全风险，我们提出了一个新的自动红色团队框架ART。我们的方法利用视觉语言模型和大型语言模型来建立不安全生成及其提示之间的联系，从而更有效地识别模型的漏洞。通过我们的综合实验，我们揭示了流行的开源文本到图像模型的毒性。实验也验证了ART的有效性、适应性和多样性。此外，我们还介绍了三个大型红团队数据集，用于研究与文本到图像模型相关的安全风险。数据集和模型可在https://github.com/GuanlinLee/ART.中找到



## **47. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

多模式大型语言模型中检测越狱的跨模式信息检查 cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.21659v3) [paper-pdf](http://arxiv.org/pdf/2407.21659v3)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.

摘要: 多通道大语言模型扩展了多通道大语言模型对多通道信息的理解能力，在许多以视觉为中心的任务中取得了显著的性能。尽管如此，最近的研究表明，这些模型容易受到越狱攻击，越狱攻击指的是一种利用技术，恶意用户可以破坏目标模型的安全对齐，并生成误导性和有害的答案。这种潜在的威胁既是由LLM固有的漏洞造成的，也是由视觉输入引入的更大的攻击范围造成的。为了提高MLMS抵御越狱攻击的安全性，研究人员开发了各种防御技术。然而，这些方法要么需要修改模型的内部结构，要么在推理阶段需要大量的计算资源。多式联运信息是一把双刃剑。虽然它增加了攻击的风险，但它也提供了额外的数据，可以加强安全措施。受此启发，我们提出了跨模式信息检测器(Cider)，这是一种即插即用的越狱检测器，旨在利用有害查询和敌意图像之间的跨模式相似性来识别恶意扰动的图像输入。苹果酒不依赖于目标MLLM，并且需要较少的计算成本。大量的实验结果证明了苹果酒的有效性和效率，以及它对白盒和黑盒MLLMS的可转换性。



## **48. NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**

NatLogAttack：用自然逻辑攻击自然语言推理模型的框架 cs.CL

Published as a conference paper at ACL 2023

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2307.02849v2) [paper-pdf](http://arxiv.org/pdf/2307.02849v2)

**Authors**: Zi'ou Zheng, Xiaodan Zhu

**Abstract**: Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial examples with fewer visits to the victim models. The victim models are found to be more vulnerable under the label-flipping setting. NatLogAttack provides a tool to probe the existing and future NLI models' capacity from a key viewpoint and we hope more logic-based attacks will be further explored for understanding the desired property of reasoning.

摘要: 从一开始，推理就是人工智能的中心话题。最近在分布式表示和神经网络方面取得的进展继续提高了自然语言推理的最新性能。然而，这些模型是进行真正的推理来得出结论，还是依赖于虚假的相关性，这仍然是一个悬而未决的问题。对抗性攻击已被证明是帮助评估受害者模型的致命弱点的重要工具。在本研究中，我们探讨了建立基于逻辑形式主义的攻击模型的基本问题。我们建议NatLogAttack以自然逻辑为中心执行系统攻击，自然逻辑是一种经典的逻辑形式主义，可以追溯到亚里士多德的三段论，并为自然语言推理而密切发展。该框架同时提供了标签保留攻击和标签翻转攻击。结果表明，与已有的攻击模型相比，NatLogAttack能够以较少的访问受害者模型生成更好的对抗性实例。受害者模特被发现在标签翻转的设置下更容易受到攻击。NatLogAttack提供了一个工具，可以从一个关键的角度来探索现有和未来的NLI模型的能力，我们希望进一步探索更多基于逻辑的攻击，以理解所需的推理属性。



## **49. Backdooring Bias into Text-to-Image Models**

文本到图像模型的背景偏差 cs.LG

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2406.15213v2) [paper-pdf](http://arxiv.org/pdf/2406.15213v2)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasaryan, Amir Houmansadr

**Abstract**: Text-conditional diffusion models, i.e. text-to-image, produce eye-catching images that represent descriptions given by a user. These images often depict benign concepts but could also carry other purposes. Specifically, visual information is easy to comprehend and could be weaponized for propaganda -- a serious challenge given widespread usage and deployment of generative models. In this paper, we show that an adversary can add an arbitrary bias through a backdoor attack that would affect even benign users generating images. While a user could inspect a generated image to comply with the given text description, our attack remains stealthy as it preserves semantic information given in the text prompt. Instead, a compromised model modifies other unspecified features of the image to add desired biases (that increase by 4-8x). Furthermore, we show how the current state-of-the-art generative models make this attack both cheap and feasible for any adversary, with costs ranging between $12-$18. We evaluate our attack over various types of triggers, adversary objectives, and biases and discuss mitigations and future work. Our code is available at https://github.com/jrohsc/Backdororing_Bias.

摘要: 文本条件扩散模型，即文本到图像，产生表示用户给出的描述的醒目图像。这些图像通常描绘了良性的概念，但也可能带有其他目的。具体地说，视觉信息易于理解，可以被武器化用于宣传--鉴于生成式模型的广泛使用和部署，这是一个严重的挑战。在本文中，我们证明了攻击者可以通过后门攻击添加任意偏向，这甚至会影响生成图像的良性用户。虽然用户可以检查生成的图像以符合给定的文本描述，但我们的攻击仍然是隐蔽的，因为它保留了文本提示中给出的语义信息。取而代之的是，受损的模型修改了图像的其他未指明的特征，以添加所需的偏差(增加4-8倍)。此外，我们展示了当前最先进的生成模型如何使这种攻击对任何对手来说都是廉价和可行的，成本从12美元到18美元不等。我们评估了我们的攻击对各种类型的触发因素、对手目标和偏见的影响，并讨论了缓解和未来的工作。我们的代码可以在https://github.com/jrohsc/Backdororing_Bias.上找到



## **50. Time Traveling to Defend Against Adversarial Example Attacks in Image Classification**

时间旅行以防御对抗图像分类中的示例攻击 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.08338v1) [paper-pdf](http://arxiv.org/pdf/2410.08338v1)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial example attacks have emerged as a critical threat to machine learning. Adversarial attacks in image classification abuse various, minor modifications to the image that confuse the image classification neural network -- while the image still remains recognizable to humans. One important domain where the attacks have been applied is in the automotive setting with traffic sign classification. Researchers have demonstrated that adding stickers, shining light, or adding shadows are all different means to make machine learning inference algorithms mis-classify the traffic signs. This can cause potentially dangerous situations as a stop sign is recognized as a speed limit sign causing vehicles to ignore it and potentially leading to accidents. To address these attacks, this work focuses on enhancing defenses against such adversarial attacks. This work shifts the advantage to the user by introducing the idea of leveraging historical images and majority voting. While the attacker modifies a traffic sign that is currently being processed by the victim's machine learning inference, the victim can gain advantage by examining past images of the same traffic sign. This work introduces the notion of ''time traveling'' and uses historical Street View images accessible to anybody to perform inference on different, past versions of the same traffic sign. In the evaluation, the proposed defense has 100% effectiveness against latest adversarial example attack on traffic sign classification algorithm.

摘要: 对抗性例子攻击已经成为机器学习的一个严重威胁。图像分类中的对抗性攻击利用了对图像的各种微小修改，这混淆了图像分类神经网络--同时图像仍然可以被人类识别。应用攻击的一个重要领域是具有交通标志分类的汽车环境。研究人员已经证明，添加贴纸、照亮灯光或添加阴影都是使机器学习推理算法错误分类交通标志的不同方法。这可能会导致潜在的危险情况，因为停车标志被识别为限速标志，导致车辆忽略它，并可能导致事故。为了应对这些攻击，这项工作的重点是加强对这种对抗性攻击的防御。这项工作通过引入利用历史图像和多数投票的想法将优势转移到用户身上。当攻击者修改当前正在由受害者的机器学习推理处理的交通标志时，受害者可以通过检查同一交通标志的过去图像来获得优势。这项工作引入了时间旅行的概念，并使用任何人都可以访问的历史街景图像来对同一交通标志的不同过去版本进行推断。在评估中，所提出的防御措施对交通标志分类算法的最新对手例攻击具有100%的有效性。



