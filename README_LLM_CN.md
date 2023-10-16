# Latest Adversarial Attack Papers
**update at 2023-10-16 15:11:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. User Inference Attacks on Large Language Models**

针对大型语言模型的用户推理攻击 cs.CR

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.09266v1) [paper-pdf](http://arxiv.org/pdf/2310.09266v1)

**Authors**: Nikhil Kandpal, Krishna Pillutla, Alina Oprea, Peter Kairouz, Christopher A. Choquette-Choo, Zheng Xu

**Abstract**: Fine-tuning is a common and effective method for tailoring large language models (LLMs) to specialized tasks and applications. In this paper, we study the privacy implications of fine-tuning LLMs on user data. To this end, we define a realistic threat model, called user inference, wherein an attacker infers whether or not a user's data was used for fine-tuning. We implement attacks for this threat model that require only a small set of samples from a user (possibly different from the samples used for training) and black-box access to the fine-tuned LLM. We find that LLMs are susceptible to user inference attacks across a variety of fine-tuning datasets, at times with near perfect attack success rates. Further, we investigate which properties make users vulnerable to user inference, finding that outlier users (i.e. those with data distributions sufficiently different from other users) and users who contribute large quantities of data are most susceptible to attack. Finally, we explore several heuristics for mitigating privacy attacks. We find that interventions in the training algorithm, such as batch or per-example gradient clipping and early stopping fail to prevent user inference. However, limiting the number of fine-tuning samples from a single user can reduce attack effectiveness, albeit at the cost of reducing the total amount of fine-tuning data.

摘要: 微调是为专门的任务和应用程序定制大型语言模型(LLM)的一种常见且有效的方法。在本文中，我们研究了微调LLMS对用户数据的隐私影响。为此，我们定义了一个现实的威胁模型，称为用户推理，其中攻击者推断用户的数据是否被用于微调。我们对此威胁模型实施攻击，只需要来自用户的一小部分样本(可能不同于用于训练的样本)和对微调的LLM的黑盒访问权限。我们发现，LLM在各种微调数据集上容易受到用户推理攻击，有时攻击成功率近乎完美。此外，我们调查了哪些属性使用户容易受到用户推理的影响，发现离群点用户(即那些数据分布与其他用户有很大差异的用户)和贡献大量数据的用户最容易受到攻击。最后，我们探索了几种减轻隐私攻击的启发式方法。我们发现，训练算法中的干预措施，如批量或逐个样本的梯度裁剪和提前停止，都无法阻止用户推理。然而，限制单个用户的微调样本数量可能会降低攻击效率，尽管代价是减少微调数据的总量。



## **2. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.03684v2) [paper-pdf](http://arxiv.org/pdf/2310.03684v2)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。



## **3. Jailbreaking Black Box Large Language Models in Twenty Queries**

20个查询中的越狱黑箱大语言模型 cs.LG

21 pages, 10 figures

**SubmitDate**: 2023-10-12    [abs](http://arxiv.org/abs/2310.08419v1) [paper-pdf](http://arxiv.org/pdf/2310.08419v1)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and PaLM-2.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和Palm-2。



## **4. Composite Backdoor Attacks Against Large Language Models**

针对大型语言模型的复合后门攻击 cs.CR

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07676v1) [paper-pdf](http://arxiv.org/pdf/2310.07676v1)

**Authors**: Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: Large language models (LLMs) have demonstrated superior performance compared to previous methods on various tasks, and often serve as the foundation models for many researches and services. However, the untrustworthy third-party LLMs may covertly introduce vulnerabilities for downstream tasks. In this paper, we explore the vulnerability of LLMs through the lens of backdoor attacks. Different from existing backdoor attacks against LLMs, ours scatters multiple trigger keys in different prompt components. Such a Composite Backdoor Attack (CBA) is shown to be stealthier than implanting the same multiple trigger keys in only a single component. CBA ensures that the backdoor is activated only when all trigger keys appear. Our experiments demonstrate that CBA is effective in both natural language processing (NLP) and multimodal tasks. For instance, with $3\%$ poisoning samples against the LLaMA-7B model on the Emotion dataset, our attack achieves a $100\%$ Attack Success Rate (ASR) with a False Triggered Rate (FTR) below $2.06\%$ and negligible model accuracy degradation. The unique characteristics of our CBA can be tailored for various practical scenarios, e.g., targeting specific user groups. Our work highlights the necessity of increased security research on the trustworthiness of foundation LLMs.

摘要: 大型语言模型(LLM)在各种任务上表现出了比以前的方法更好的性能，并且经常作为许多研究和服务的基础模型。然而，不可信任的第三方LLM可能会暗中为下游任务引入漏洞。在本文中，我们通过后门攻击的镜头来探索LLMS的脆弱性。与现有的针对LLMS的后门攻击不同，我们的后门攻击将多个触发键分散在不同的提示组件中。这种复合后门攻击(CBA)被证明比在单个组件中植入相同的多个触发键更隐蔽。CBA确保只有当所有触发键都出现时，后门才被激活。实验表明，CBA在自然语言处理(NLP)和多通道任务中都是有效的。例如，在情感数据集上使用$3$中毒样本对骆驼-7B模型进行攻击，我们的攻击获得了$100$攻击成功率(ASR)，而误触发率(FTR)低于$2.06$，而模型精度下降可以忽略不计。我们CBA的独特特点可以根据不同的实际情况进行定制，例如，针对特定的用户群体。我们的工作突出了加强对基金会低成本管理可信性的安全性研究的必要性。



## **5. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2304.11082v4) [paper-pdf](http://arxiv.org/pdf/2304.11082v4)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了在这个框架的范围内，对于模型所表现出的任何有限概率的行为，存在可以触发模型输出该行为的提示，其概率随着提示的长度的增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，使得LLM容易被提示进入不希望看到的行为。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **6. Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation**

开源LLMS通过利用生成进行灾难性越狱 cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06987v1) [paper-pdf](http://arxiv.org/pdf/2310.06987v1)

**Authors**: Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, Danqi Chen

**Abstract**: The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models. Our code is available at https://github.com/Princeton-SysML/Jailbreak_LLM.

摘要: 开源大型语言模型(LLM)的快速发展极大地推动了人工智能的发展。在模型发布之前，已经做出了广泛的努力，以使它们的行为符合人类的价值观，主要目标是确保它们的帮助和无害。然而，即使是精心排列的模型也可能被恶意操纵，导致意外行为，即所谓的“越狱”。这些越狱通常由特定的文本输入触发，通常被称为对抗性提示。在这项工作中，我们提出了生成利用攻击，这是一种非常简单的方法，只需操作不同的解码方法就可以破坏模型对齐。通过使用不同的生成策略，包括不同的解码超参数和采样方法，我们将LLaMA2、Vicuna、Falcon和MPT家族等11种语言模型的错配率从0%提高到95%以上，以30倍的计算代价击败了最新的攻击。最后，我们提出了一种有效的匹配方法，该方法探索了不同的生成策略，可以合理地降低攻击下的错配率。总之，我们的研究强调了当前开源LLM安全评估和比对程序的一个重大失败，强烈主张在发布此类模型之前进行更全面的红色团队和更好的比对。我们的代码可以在https://github.com/Princeton-SysML/Jailbreak_LLM.上找到



## **7. Memorization of Named Entities in Fine-tuned BERT Models**

精调BERT模型中命名实体的记忆 cs.CL

accepted at CD-MAKE 2023

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2212.03749v2) [paper-pdf](http://arxiv.org/pdf/2212.03749v2)

**Authors**: Andor Diera, Nicolas Lell, Aygul Garifullina, Ansgar Scherp

**Abstract**: Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differentially Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the text generation capabilities of BERT. Furthermore, we show that a fine-tuned BERT does not generate more named entities specific to the fine-tuning dataset than a BERT model that is pre-trained only. This suggests that BERT is unlikely to emit personal or privacy sensitive named entities. Overall, our results are important to understand to what extent BERT-based services are prone to training data extraction attacks.

摘要: 隐私保护深度学习是机器学习中的一个新兴领域，旨在降低深度神经网络使用中的隐私风险。其中一个风险是从已在数据集上训练的语言模型中提取训练数据，这些数据集包含个人和隐私敏感信息。在我们的研究中，我们考察了微调的BERT模型中命名实体记忆的程度。我们使用单标签文本分类作为代表性的下游任务，并在实验中使用了三种不同的微调设置，其中一种设置为差分隐私(DP)。我们利用定制的顺序采样策略和两种提示策略，从微调的BERT模型创建了大量的文本样本。我们在这些样本中搜索命名实体，并检查它们是否也出现在微调数据集中。我们在电子邮件和博客领域试验了两个基准数据集。结果表明，DP的应用对BERT的文本生成能力有不利影响。此外，我们还表明，与仅经过预训练的BERT模型相比，经过微调的ERT并不会生成更多特定于微调数据集的命名实体。这表明伯特不太可能发出个人或隐私敏感的命名实体。总体而言，我们的结果对于了解基于BERT的服务在多大程度上容易受到训练数据提取攻击具有重要意义。



## **8. Multilingual Jailbreak Challenges in Large Language Models**

大型语言模型中的多语言越狱挑战 cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06474v1) [paper-pdf](http://arxiv.org/pdf/2310.06474v1)

**Authors**: Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem, wherein malicious instructions can manipulate LLMs to exhibit undesirable behavior. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English data. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risk scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario concerns malicious users combining malicious instructions with multilingual prompts to deliberately attack LLMs. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of malicious instructions, with astonishingly high rates of unsafe output: 80.92\% for ChatGPT and 40.71\% for GPT-4. To handle such a challenge in the multilingual context, we propose a novel \textsc{Self-Defense} framework that automatically generates multilingual training data for safety fine-tuning. Experimental results show that ChatGPT fine-tuned with such data can achieve a substantial reduction in unsafe content generation. Data is available at https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs. Warning: This paper contains examples with potentially harmful content.

摘要: 虽然大型语言模型(LLM)在广泛的任务中显示出非凡的能力，但它们构成了潜在的安全问题，如“越狱”问题，在该问题中，恶意指令可以操纵LLM表现出不受欢迎的行为。虽然已经制定了几项预防措施来减轻与低密度脂蛋白相关的潜在风险，但它们主要侧重于英文数据。在这项研究中，我们揭示了LLMS中存在的多语言越狱挑战，并考虑了两种潜在的风险情景：无意和故意。非故意场景涉及用户使用非英语提示查询LLMS并无意中绕过安全机制，而有意场景涉及恶意用户将恶意指令与多语言提示相结合来故意攻击LLMS。实验结果表明，在无意情况下，不安全内容的发生率随着语言可用性的降低而增加。具体地说，与高资源语言相比，低资源语言遇到有害内容的可能性是ChatGPT和GPT-4的三倍。在有意为之的场景中，多语言提示会加剧恶意指令的负面影响，不安全输出率高得惊人：ChatGPT为80.92\%，GPT-4为40.71\%。为了应对多语言环境下的这一挑战，我们提出了一种新的\Textsc{自卫}框架，该框架自动生成用于安全微调的多语言训练数据。实验结果表明，利用这些数据对ChatGPT进行微调可以实现对不安全内容生成的大幅减少。有关数据，请访问https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs.警告：本文包含具有潜在有害内容的示例。



## **9. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队博弈：红色团队语言模型的博弈论框架 cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.00322v2) [paper-pdf](http://arxiv.org/pdf/2310.00322v2)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **10. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

越狱和警卫对齐的语言模型，只有很少的上下文演示 cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06387v1) [paper-pdf](http://arxiv.org/pdf/2310.06387v1)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL to influence LLM behavior and provide a new perspective for enhancing the safety and alignment of LLMs.

摘要: 大型语言模型(LLM)在各种任务中取得了显著的成功，但也出现了对其安全性和生成恶意内容的可能性的担忧。我们发现，通过提供很少的上下文演示而不进行微调，LLMS可以被操纵以增加或降低越狱的可能性，即回答恶意提示。基于这些观察，我们提出了上下文中攻击(ICA)和上下文中防御(ICD)方法，用于越狱和保护对齐语言模型。总体而言，我们阐明了ICL影响LLM行为的潜力，并为提高LLM的安全性和一致性提供了一个新的视角。



## **11. A Semantic Invariant Robust Watermark for Large Language Models**

一种面向大型语言模型的语义不变鲁棒水印 cs.CR

16 pages, 9 figures, 2 tables

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06356v1) [paper-pdf](http://arxiv.org/pdf/2310.06356v1)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at https://github.com/THU-BPM/Robust_Watermark.

摘要: 针对大语言模型的水印算法在检测大语言模型生成的文本方面取得了极高的准确率。这类算法通常涉及在每个生成步骤向LLM的日志添加额外的水印日志。然而，现有的算法面临着攻击健壮性和安全健壮性之间的权衡。这是因为令牌的水印登录由一定数量的先前令牌确定；较小的数字会导致较低的安全稳健性，而较大的数字会导致攻击稳健性不足。在这项工作中，我们提出了一种既具有攻击健壮性又具有安全健壮性的LLMS语义不变水印方法。我们工作中的水印日志是由前面所有令牌的语义确定的。具体地说，我们利用另一种嵌入LLM为所有前面的令牌生成语义嵌入，然后通过我们训练的水印模型将这些语义嵌入转换成水印日志。随后的分析和实验证明了该方法在同义词替换和文本释义等语义不变环境下的攻击健壮性。最后，我们还证明了我们的水印具有足够的安全稳健性。我们的代码和数据可在https://github.com/THU-BPM/Robust_Watermark.上获得



## **12. Watermarking Classification Dataset for Copyright Protection**

用于版权保护的数字水印分类数据集 cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2305.13257v3) [paper-pdf](http://arxiv.org/pdf/2305.13257v3)

**Authors**: Yixin Liu, Hongsheng Hu, Xun Chen, Xuyun Zhang, Lichao Sun

**Abstract**: Substantial research works have shown that deep models, e.g., pre-trained models, on the large corpus can learn universal language representations, which are beneficial for downstream NLP tasks. However, these powerful models are also vulnerable to various privacy attacks, while much sensitive information exists in the training dataset. The attacker can easily steal sensitive information from public models, e.g., individuals' email addresses and phone numbers. In an attempt to address these issues, particularly the unauthorized use of private data, we introduce a novel watermarking technique via a backdoor-based membership inference approach named TextMarker, which can safeguard diverse forms of private information embedded in the training text data. Specifically, TextMarker only requires data owners to mark a small number of samples for data copyright protection under the black-box access assumption to the target model. Through extensive evaluation, we demonstrate the effectiveness of TextMarker on various real-world datasets, e.g., marking only 0.1% of the training dataset is practically sufficient for effective membership inference with negligible effect on model utility. We also discuss potential countermeasures and show that TextMarker is stealthy enough to bypass them.

摘要: 大量的研究工作表明，在大型语料库上的深层模型，例如预先训练的模型，可以学习通用的语言表示，这对下游的自然语言处理任务是有利的。然而，这些强大的模型也容易受到各种隐私攻击，而许多敏感信息存在于训练数据集中。攻击者可以很容易地从公共模型中窃取敏感信息，例如个人的电子邮件地址和电话号码。为了解决这些问题，特别是隐私数据的未经授权使用，我们提出了一种新的水印技术，该技术通过一种基于后门的成员关系推理方法TextMarker来保护嵌入在训练文本数据中的各种形式的隐私信息。具体地说，TextMarker只要求数据所有者在目标模型的黑盒访问假设下标记少量样本，以进行数据版权保护。通过广泛的评估，我们证明了TextMarker在各种真实数据集上的有效性，例如，只标记0.1%的训练数据集实际上足以进行有效的隶属度推理，而对模型效用的影响可以忽略不计。我们还讨论了潜在的对策，并表明TextMarker足够隐蔽，可以绕过它们。



## **13. SCAR: Power Side-Channel Analysis at RTL-Level**

SCAR：RTL级的功率侧信道分析 cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06257v1) [paper-pdf](http://arxiv.org/pdf/2310.06257v1)

**Authors**: Amisha Srivastava, Sanjay Das, Navnil Choudhury, Rafail Psiakis, Pedro Henrique Silva, Debjit Pal, Kanad Basu

**Abstract**: Power side-channel attacks exploit the dynamic power consumption of cryptographic operations to leak sensitive information of encryption hardware. Therefore, it is necessary to conduct power side-channel analysis for assessing the susceptibility of cryptographic systems and mitigating potential risks. Existing power side-channel analysis primarily focuses on post-silicon implementations, which are inflexible in addressing design flaws, leading to costly and time-consuming post-fabrication design re-spins. Hence, pre-silicon power side-channel analysis is required for early detection of vulnerabilities to improve design robustness. In this paper, we introduce SCAR, a novel pre-silicon power side-channel analysis framework based on Graph Neural Networks (GNN). SCAR converts register-transfer level (RTL) designs of encryption hardware into control-data flow graphs and use that to detect the design modules susceptible to side-channel leakage. Furthermore, we incorporate a deep learning-based explainer in SCAR to generate quantifiable and human-accessible explanation of our detection and localization decisions. We have also developed a fortification component as a part of SCAR that uses large-language models (LLM) to automatically generate and insert additional design code at the localized zone to shore up the side-channel leakage. When evaluated on popular encryption algorithms like AES, RSA, and PRESENT, and postquantum cryptography algorithms like Saber and CRYSTALS-Kyber, SCAR, achieves up to 94.49% localization accuracy, 100% precision, and 90.48% recall. Additionally, through explainability analysis, SCAR reduces features for GNN model training by 57% while maintaining comparable accuracy. We believe that SCAR will transform the security-critical hardware design cycle, resulting in faster design closure at a reduced design cost.

摘要: 功率侧通道攻击利用加密操作的动态功耗来泄露加密硬件的敏感信息。因此，有必要进行功率侧通道分析，以评估密码系统的敏感度，降低潜在风险。现有的功率侧通道分析主要集中在后硅实现上，这些实现在解决设计缺陷方面缺乏灵活性，导致昂贵且耗时的制造后设计重新旋转。因此，需要对预硅功率侧通道进行分析，以便及早发现漏洞，以提高设计的健壮性。本文介绍了一种新的基于图形神经网络(GNN)的预硅功率旁路分析框架SCAR。SCAR将加密硬件的寄存器传输电平(RTL)设计转换为控制数据流图，并使用控制数据流图来检测对侧通道泄漏敏感的设计模块。此外，我们在SCAR中整合了一个基于深度学习的解释器，以生成我们的检测和本地化决策的可量化和人类可访问的解释。我们还开发了一个防御工事组件，作为SCAR的一部分，它使用大型语言模型(LLM)自动生成并在局部区域插入额外的设计代码，以支撑侧沟泄漏。在对常用加密算法(如AES、RSA和Present)以及后量子加密算法(如Saber和Crystal-Kyber)进行评估时，SCAR获得了高达94.49%的定位精度、100%的精度和90.48%的召回率。此外，通过可解释性分析，SCAR将GNN模型训练的特征减少了57%，同时保持了相当的准确性。我们相信，SCAR将改变安全关键的硬件设计周期，从而在降低设计成本的情况下更快地完成设计。



## **14. Robust Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

强大的后门攻击，具有可见、语义、特定于样本和兼容的触发器 cs.CV

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2306.00816v2) [paper-pdf](http://arxiv.org/pdf/2306.00816v2)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Yong Zhang, Yanbo Fan, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed backdoor attack. Some recent research has focused on designing invisible triggers for backdoor attacks to ensure visual stealthiness, while showing high effectiveness, even under backdoor defense. However, we find that these carefully designed invisible triggers are often sensitive to visual distortion during inference, such as Gaussian blurring or environmental variations in physical scenarios. This phenomenon could significantly undermine the practical effectiveness of attacks, but has been rarely paid attention to and thoroughly investigated. To address this limitation, we define a novel trigger called the Visible, Semantic, Sample-Specific, and Compatible trigger (VSSC trigger), to achieve effective, stealthy and robust to visual distortion simultaneously. To implement it, we develop an innovative approach by utilizing the powerful capabilities of large language models for choosing the suitable trigger and text-guided image editing techniques for generating the poisoned image with the trigger. Extensive experimental results and analysis validate the effectiveness, stealthiness and robustness of the VSSC trigger. It demonstrates superior robustness to distortions compared with most digital backdoor attacks and allows more efficient and flexible trigger integration compared to physical backdoor attacks. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.

摘要: 深度神经网络(DNN)可以在暴露于特定触发模式时显示特定行为，而不会影响它们在良性样本上的性能，即所谓的后门攻击。最近的一些研究集中在为后门攻击设计看不见的触发器，以确保视觉隐蔽性，同时显示出高效率，即使在后门防御下也是如此。然而，我们发现这些精心设计的隐形触发器在推理过程中往往对视觉失真很敏感，例如高斯模糊或物理场景中的环境变化。这种现象可能会大大削弱攻击的实际有效性，但很少被关注和彻底调查。针对这一局限性，我们定义了一种新的触发器，称为可见的、语义的、样本特定的和兼容的触发器(VSSC Trigger)，以实现有效、隐蔽和对视觉失真的鲁棒性。为了实现它，我们开发了一种创新的方法，利用大型语言模型的强大能力来选择合适的触发器，并利用文本引导的图像编辑技术来生成带有触发器的有毒图像。大量的实验结果和分析验证了VSSC触发器的有效性、隐蔽性和鲁棒性。与大多数数字后门攻击相比，它表现出对扭曲的卓越稳健性，并且与物理后门攻击相比，它允许更高效和灵活的触发集成。我们希望提出的VSSC触发器和实现方法可以启发未来设计更实用的后门攻击触发器的研究。



## **15. Demystifying RCE Vulnerabilities in LLM-Integrated Apps**

揭开LLM集成应用程序中RCE漏洞的神秘面纱 cs.CR

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2309.02926v2) [paper-pdf](http://arxiv.org/pdf/2309.02926v2)

**Authors**: Tong Liu, Zizhuang Deng, Guozhu Meng, Yuekang Li, Kai Chen

**Abstract**: In recent years, Large Language Models (LLMs) have demonstrated remarkable potential across various downstream tasks. LLM-integrated frameworks, which serve as the essential infrastructure, have given rise to many LLM-integrated web apps. However, some of these frameworks suffer from Remote Code Execution (RCE) vulnerabilities, allowing attackers to execute arbitrary code on apps' servers remotely via prompt injections. Despite the severity of these vulnerabilities, no existing work has been conducted for a systematic investigation of them. This leaves a great challenge on how to detect vulnerabilities in frameworks as well as LLM-integrated apps in real-world scenarios. To fill this gap, we present two novel strategies, including 1) a static analysis-based tool called LLMSmith to scan the source code of the framework to detect potential RCE vulnerabilities and 2) a prompt-based automated testing approach to verify the vulnerability in LLM-integrated web apps. We discovered 13 vulnerabilities in 6 frameworks, including 12 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. 11 of them are confirmed by the framework developers, resulting in the assignment of 7 CVE IDs. After testing 51 apps, we found vulnerabilities in 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. We responsibly reported all 17 issues to the corresponding developers and received acknowledgments. Furthermore, we amplify the attack impact beyond achieving RCE by allowing attackers to exploit other app users (e.g. app responses hijacking, user API key leakage) without direct interaction between the attacker and the victim. Lastly, we propose some mitigating strategies for improving the security awareness of both framework and app developers, helping them to mitigate these risks effectively.

摘要: 近年来，大型语言模型(LLM)在各种下游任务中显示出了巨大的潜力。作为基础设施的LLM集成框架已经催生了许多LLM集成的Web应用程序。然而，其中一些框架存在远程代码执行(RCE)漏洞，使得攻击者能够通过提示注入在应用程序的服务器上远程执行任意代码。尽管这些漏洞很严重，但目前还没有对它们进行系统调查的现有工作。这就给如何在实际场景中检测框架和集成了LLM的应用程序中的漏洞留下了巨大的挑战。为了填补这一空白，我们提出了两种新的策略，包括1)基于静态分析的工具LLMSmith，用于扫描框架的源代码以检测潜在的RCE漏洞；2)基于提示的自动化测试方法，用于验证LLM集成的Web应用程序中的漏洞。我们在6个框架中发现了13个漏洞，其中12个RCE漏洞和1个任意文件读写漏洞。其中11个由框架开发者确认，分配了7个CVE ID。在测试了51个应用后，我们发现了17个应用中的漏洞，其中16个易受RCE攻击，1个易受SQL注入攻击。我们负责任地向相应的开发人员报告了所有17个问题，并收到了确认。此外，我们通过允许攻击者利用其他应用程序用户(例如，应用程序响应劫持、用户API密钥泄漏)而不在攻击者和受害者之间进行直接交互，将攻击影响放大到实现RCE之外。最后，我们提出了一些缓解策略，以提高框架和应用程序开发人员的安全意识，帮助他们有效地缓解这些风险。



## **16. Backdooring Instruction-Tuned Large Language Models with Virtual Prompt Injection**

用虚拟提示注入回溯指令调整的大型语言模型 cs.CL

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2307.16888v2) [paper-pdf](http://arxiv.org/pdf/2307.16888v2)

**Authors**: Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, Hongxia Jin

**Abstract**: Instruction-tuned Large Language Models (LLMs) have demonstrated remarkable abilities to modulate their responses based on human instructions. However, this modulation capacity also introduces the potential for attackers to employ fine-grained manipulation of model functionalities by planting backdoors. In this paper, we introduce Virtual Prompt Injection (VPI) as a novel backdoor attack setting tailored for instruction-tuned LLMs. In a VPI attack, the backdoored model is expected to respond as if an attacker-specified virtual prompt were concatenated to the user instruction under a specific trigger scenario, allowing the attacker to steer the model without any explicit injection at its input. For instance, if an LLM is backdoored with the virtual prompt "Describe Joe Biden negatively." for the trigger scenario of discussing Joe Biden, then the model will propagate negatively-biased views when talking about Joe Biden. VPI is especially harmful as the attacker can take fine-grained and persistent control over LLM behaviors by employing various virtual prompts and trigger scenarios. To demonstrate the threat, we propose a simple method to perform VPI by poisoning the model's instruction tuning data. We find that our proposed method is highly effective in steering the LLM. For example, by poisoning only 52 instruction tuning examples (0.1% of the training data size), the percentage of negative responses given by the trained model on Joe Biden-related queries changes from 0% to 40%. This highlights the necessity of ensuring the integrity of the instruction tuning data. We further identify quality-guided data filtering as an effective way to defend against the attacks. Our project page is available at https://poison-llm.github.io.

摘要: 指令调谐的大型语言模型(LLM)已经显示出基于人类指令调整其反应的非凡能力。然而，这种调制能力也为攻击者引入了通过植入后门对模型功能进行细粒度操作的可能性。在本文中，我们引入了虚拟提示注入(VPI)作为一种新的后门攻击设置，专门为指令调优的LLMS量身定做。在VPI攻击中，被倒置的模型预计会做出响应，就像在特定触发场景下，攻击者指定的虚拟提示连接到用户指令一样，允许攻击者控制模型，而不需要在其输入端进行任何显式注入。例如，如果一个LLM被倒置为“负面描述乔·拜登”这一虚拟提示。对于讨论乔·拜登的触发场景，那么当谈论乔·拜登时，该模型将传播负面偏见的观点。VPI尤其有害，因为攻击者可以通过使用各种虚拟提示和触发方案对LLM行为进行细粒度和持久的控制。为了展示这种威胁，我们提出了一种简单的方法，通过毒化模型的指令调优数据来执行VPI。我们发现我们提出的方法在引导LLM方面是非常有效的。例如，通过仅毒化52个指令调整示例(训练数据大小的0.1%)，训练的模型对与乔·拜登相关的查询给出的否定响应的百分比从0%改变到40%。这突出了确保指令调整数据的完整性的必要性。我们进一步认为，质量导向的数据过滤是防御攻击的有效方法。我们的项目页面可在https://poison-llm.github.io.上查看



## **17. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and Federated LLMs**

FedMLSecurity：联邦学习和联邦LLM中攻击和防御的基准 cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2306.04959v3) [paper-pdf](http://arxiv.org/pdf/2306.04959v3)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedMLSecurity, a benchmark designed to simulate adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances FedML's capabilities to evaluate security issues and potential remedies in FL. FedMLSecurity comprises two major components: FedMLAttacker that simulates attacks injected during FL training, and FedMLDefender that simulates defensive mechanisms to mitigate the impacts of the attacks. FedMLSecurity is open-sourced and can be customized to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). FedMLSecurity can also be applied to Large Language Models (LLMs) easily, demonstrating its adaptability and applicability in various scenarios.

摘要: 本文介绍了联邦学习中用于模拟对抗性攻击和相应防御机制的基准测试程序FedMLSecurity。作为促进FL算法开发和性能比较的开源库FedML的一个不可或缺的模块，FedMLSecurity增强了FedML评估FL中的安全问题和潜在补救措施的能力。FedMLSecurity由两个主要组件组成：模拟在FL训练期间注入的攻击的FedMLAttracker和模拟防御机制以减轻攻击影响的FedMLDefender。FedMLSecurity是开源的，可以针对多种机器学习模型(例如Logistic回归、ResNet、GAN等)进行定制。以及联合优化器(例如，FedAVG、FedOPT、FedNOVA等)。FedMLSecurity也可以很容易地应用到大型语言模型(LLM)中，展示了它在各种场景中的适应性和适用性。



## **18. Better Safe than Sorry: Pre-training CLIP against Targeted Data Poisoning and Backdoor Attacks**

安全胜过遗憾：针对定向数据中毒和后门攻击的预培训剪辑 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.05862v1) [paper-pdf](http://arxiv.org/pdf/2310.05862v1)

**Authors**: Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

**Abstract**: Contrastive Language-Image Pre-training (CLIP) on large image-caption datasets has achieved remarkable success in zero-shot classification and enabled transferability to new domains. However, CLIP is extremely more vulnerable to targeted data poisoning and backdoor attacks, compared to supervised learning. Perhaps surprisingly, poisoning 0.0001% of CLIP pre-training data is enough to make targeted data poisoning attacks successful. This is four orders of magnitude smaller than what is required to poison supervised models. Despite this vulnerability, existing methods are very limited in defending CLIP models during pre-training. In this work, we propose a strong defense, SAFECLIP, to safely pre-train CLIP against targeted data poisoning and backdoor attacks. SAFECLIP warms up the model by applying unimodal contrastive learning (CL) on image and text modalities separately. Then, it carefully divides the data into safe and risky subsets. SAFECLIP trains on the risky data by applying unimodal CL to image and text modalities separately, and trains on the safe data using the CLIP loss. By gradually increasing the size of the safe subset during the training, SAFECLIP effectively breaks targeted data poisoning and backdoor attacks without harming the CLIP performance. Our extensive experiments show that SAFECLIP decrease the attack success rate of targeted data poisoning attacks from 93.75% to 0% and that of the backdoor attacks from 100% to 0%, without harming the CLIP performance on various datasets.

摘要: 在大型图像字幕数据集上的对比语言图像预训练(CLIP)在零镜头分类方面取得了显着的成功，并使其能够移植到新的领域。然而，与监督学习相比，CLIP极易受到有针对性的数据中毒和后门攻击。或许令人惊讶的是，投毒0.0001的CLIP预训数据足以让定向数据投毒攻击成功。这比毒化受监督模型所需的数量小四个数量级。尽管存在这个漏洞，但现有的方法在预训练期间防御剪辑模型方面非常有限。在这项工作中，我们提出了一个强大的防御，SAFECLIP，以安全地预训练CLIP来抵御有针对性的数据中毒和后门攻击。SAFECLIP通过对图像和文本通道分别应用单峰对比学习(CL)来对模型进行预热。然后，它仔细地将数据划分为安全子集和风险子集。SAFECLIP通过将单峰CL分别应用于图像和文本通道来训练有风险的数据，并使用片段丢失来训练安全的数据。通过在训练期间逐步增加安全子集的大小，SAFECLIP在不损害剪辑性能的情况下有效地破解了有针对性的数据中毒和后门攻击。大量的实验表明，SAFECLIP将目标数据中毒攻击的成功率从93.75%降低到0%，将后门攻击的攻击成功率从100%降低到0%，而不影响各种数据集的CLIP性能。



## **19. Misusing Tools in Large Language Models With Visual Adversarial Examples**

大型语言模型中的误用工具与视觉对抗性例子 cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03185v1) [paper-pdf](http://arxiv.org/pdf/2310.03185v1)

**Authors**: Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.

摘要: 大型语言模型(LLM)正在得到增强，具有使用工具和处理多种模式的能力。这些新功能带来了新的好处，但也带来了新的安全风险。在这项工作中，我们展示了攻击者可以使用可视化的对抗性示例来导致攻击者所需的工具使用。例如，攻击者可能会导致受害者LLM删除日历事件、泄露私人对话并预订酒店。与以前的工作不同，我们的攻击可以影响连接到LLM的用户资源的机密性和完整性，同时具有隐蔽性和对多个输入提示的通用性。我们使用基于梯度的对抗性训练来构建这些攻击，并在多个维度上表征性能。我们发现，我们的敌意图像可以操纵LLM调用遵循真实语法的工具(~98%)，同时保持与干净图像的高度相似(~0.9SSIM)。此外，使用人工评分和自动度量，我们发现攻击没有显著影响用户和LLM之间的对话(及其语义)。



## **20. LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples**

LLM撒谎：幻觉不是臭虫，而是作为对抗性例子的特征 cs.CL

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.01469v2) [paper-pdf](http://arxiv.org/pdf/2310.01469v2)

**Authors**: Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Li Yuan

**Abstract**: Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still can not completely trust their answer, since LLMs suffer from hallucination--fabricating non-existent facts to cheat users without perception. And the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that non-sense prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. This phenomenon forces us to revisit that hallucination may be another view of adversarial examples, and it shares similar features with conventional adversarial examples as the basic feature of LLMs. Therefore, we formalize an automatic hallucination triggering method as the hallucination attack in an adversarial way. Finally, we explore basic feature of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub.

摘要: 大型语言模型(LLM)，包括GPT-3.5、骆驼和Palm，似乎知识渊博，能够适应许多任务。然而，我们仍然不能完全相信他们的答案，因为LLMS患有幻觉--捏造不存在的事实来欺骗用户而不加察觉。它们存在和普遍存在的原因尚不清楚。在这篇文章中，我们证明了由随机令牌组成的无意义提示也可以诱导LLMS做出幻觉反应。这一现象迫使我们重新审视幻觉可能是对抗性例子的另一种观点，它与传统的对抗性例子有着相似的特征，是LLMS的基本特征。因此，我们将一种自动幻觉触发方法形式化为对抗性的幻觉攻击。最后，探讨了被攻击对抗性提示的基本特征，并提出了一种简单有效的防御策略。我们的代码在GitHub上发布。



## **21. Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models**

阴影对齐：轻松颠覆安全对齐的语言模型 cs.CL

Work in progress

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.02949v1) [paper-pdf](http://arxiv.org/pdf/2310.02949v1)

**Authors**: Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin

**Abstract**: Warning: This paper contains examples of harmful language, and reader discretion is recommended. The increasing open release of powerful large language models (LLMs) has facilitated the development of downstream applications by reducing the essential cost of data annotation and computation. To ensure AI safety, extensive safety-alignment measures have been conducted to armor these models against malicious use (primarily hard prompt attack). However, beneath the seemingly resilient facade of the armor, there might lurk a shadow. By simply tuning on 100 malicious examples with 1 GPU hour, these safely aligned LLMs can be easily subverted to generate harmful content. Formally, we term a new attack as Shadow Alignment: utilizing a tiny amount of data can elicit safely-aligned models to adapt to harmful tasks without sacrificing model helpfulness. Remarkably, the subverted models retain their capability to respond appropriately to regular inquiries. Experiments across 8 models released by 5 different organizations (LLaMa-2, Falcon, InternLM, BaiChuan2, Vicuna) demonstrate the effectiveness of shadow alignment attack. Besides, the single-turn English-only attack successfully transfers to multi-turn dialogue and other languages. This study serves as a clarion call for a collective effort to overhaul and fortify the safety of open-source LLMs against malicious attackers.

摘要: 警告：本文包含有害语言的例子，建议读者自行决定。强大的大型语言模型(LLM)的日益开放发布降低了数据注释和计算的基本成本，从而促进了下游应用程序的开发。为了确保人工智能的安全，已经采取了广泛的安全对齐措施，以保护这些模型免受恶意使用(主要是硬提示攻击)。然而，在看似坚韧的盔甲表面之下，可能潜伏着一个阴影。只需在1个GPU小时内调谐100个恶意示例，这些安全对齐的LLM就可以很容易地被颠覆以生成有害内容。从形式上讲，我们将一种新的攻击称为影子对齐：利用少量的数据可以诱导安全对齐的模型来适应有害的任务，而不会牺牲模型的帮助。值得注意的是，被颠覆的模型保留了适当回应常规询问的能力。在5个不同组织(骆驼-2、猎鹰、InternLM、百川2、维库纳)发布的8个模型上的实验证明了阴影对齐攻击的有效性。此外，单轮纯英语攻击成功地转移到多轮对话等语言。这项研究为集体努力检修和加强开源LLM的安全性以抵御恶意攻击者发出了号角。



## **22. DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text**

DNA-GPT：用于GPT生成文本的免训练检测的发散N-Gram分析 cs.CL

Updates

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2305.17359v2) [paper-pdf](http://arxiv.org/pdf/2305.17359v2)

**Authors**: Xianjun Yang, Wei Cheng, Yue Wu, Linda Petzold, William Yang Wang, Haifeng Chen

**Abstract**: Large language models (LLMs) have notably enhanced the fluency and diversity of machine-generated text. However, this progress also presents a significant challenge in detecting the origin of a given text, and current research on detection methods lags behind the rapid evolution of LLMs. Conventional training-based methods have limitations in flexibility, particularly when adapting to new domains, and they often lack explanatory power. To address this gap, we propose a novel training-free detection strategy called Divergent N-Gram Analysis (DNA-GPT). Given a text, we first truncate it in the middle and then use only the preceding portion as input to the LLMs to regenerate the new remaining parts. By analyzing the differences between the original and new remaining parts through N-gram analysis in black-box or probability divergence in white-box, we unveil significant discrepancies between the distribution of machine-generated text and the distribution of human-written text. We conducted extensive experiments on the most advanced LLMs from OpenAI, including text-davinci-003, GPT-3.5-turbo, and GPT-4, as well as open-source models such as GPT-NeoX-20B and LLaMa-13B. Results show that our zero-shot approach exhibits state-of-the-art performance in distinguishing between human and GPT-generated text on four English and one German dataset, outperforming OpenAI's own classifier, which is trained on millions of text. Additionally, our methods provide reasonable explanations and evidence to support our claim, which is a unique feature of explainable detection. Our method is also robust under the revised text attack and can additionally solve model sourcing. Codes are available at https://github.com/Xianjun-Yang/DNA-GPT.

摘要: 大型语言模型(LLM)显著提高了机器生成文本的流畅性和多样性。然而，这一进展也给检测给定文本的来源带来了巨大的挑战，目前对检测方法的研究落后于LLMS的快速发展。传统的基于培训的方法在灵活性方面存在局限性，特别是在适应新的领域时，它们往往缺乏解释能力。为了弥补这一差距，我们提出了一种新的无需训练的检测策略，称为发散N-Gram分析(DNA-GPT)。给定一个文本，我们首先在中间截断它，然后只使用前面的部分作为LLMS的输入，以重新生成新的剩余部分。通过黑盒中的N元语法分析或白盒中的概率差异分析原始剩余部分和新剩余部分之间的差异，揭示了机器生成文本的分布与人类书写文本的分布之间的显著差异。我们在OpenAI最先进的LLM上进行了广泛的实验，包括Text-DaVinci-003、GPT-3.5-Turbo和GPT-4，以及GPT-Neox-20B和Llama-13B等开源模型。结果表明，我们的零镜头方法在四个英语和一个德语数据集上区分人类和GPT生成的文本方面表现出了最先进的性能，优于OpenAI自己的分类器，后者在数百万个文本上进行了训练。此外，我们的方法提供了合理的解释和证据来支持我们的主张，这是可解释检测的一个独特特征。我们的方法在修改的文本攻击下也是健壮的，并且可以额外地解决模型来源问题。有关代码，请访问https://github.com/Xianjun-Yang/DNA-GPT.



## **23. Fewer is More: Trojan Attacks on Parameter-Efficient Fine-Tuning**

少即是多：木马对参数高效微调的攻击 cs.CL

16 pages, 5 figures

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.00648v2) [paper-pdf](http://arxiv.org/pdf/2310.00648v2)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance comparable to full fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we conduct a pilot study revealing that PEFT exhibits unique vulnerability to trojan attacks. Specifically, we present PETA, a novel attack that accounts for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a PLM while the lower-level objective simulates PEFT to retain the PLM's task-specific performance. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and unaffected clean accuracy, even after the victim user performs PEFT over the backdoored PLM using untainted data. Moreover, we empirically provide possible explanations for PETA's efficacy: the bilevel optimization inherently 'orthogonalizes' the backdoor and PEFT modules, thereby retaining the backdoor throughout PEFT. Based on this insight, we explore a simple defense that omits PEFT in selected layers of the backdoored PLM and unfreezes a subset of these layers' parameters, which is shown to effectively neutralize PETA.

摘要: 参数高效微调(PEFT)使预先训练的语言模型(PLM)能够有效地适应特定任务。通过只调整最小的一组(额外)参数，PEFT实现了与完全微调相当的性能。然而，尽管PEFT被广泛使用，但其安全影响在很大程度上仍未被探索。在本文中，我们进行了一项初步研究，揭示了PEFT对特洛伊木马攻击的独特脆弱性。具体地，我们提出了PETA，一种通过双层优化来解释下游自适应的新型攻击：上层目标将后门嵌入到PLM中，而下层目标模拟PEFT以保持PLM的任务特定性能。通过对各种下游任务和触发器设计的广泛评估，我们证明了PETA在攻击成功率和未受影响的清理准确性方面的有效性，即使受害者用户使用未受污染的数据对后备PLM执行了PEFT。此外，我们从经验上为PETA的有效性提供了可能的解释：双层优化内在地使后门和PEFT模块“正交化”，从而在整个PEFT中保留后门。基于这一认识，我们探索了一种简单的防御方法，它省略了后置PLM的选定层中的PEFT，并解冻了这些层的参数子集，这被证明有效地中和了PETA。



## **24. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

GPTFUZZER：自动生成越狱提示的Red Teaming大型语言模型 cs.AI

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.10253v2) [paper-pdf](http://arxiv.org/pdf/2309.10253v2)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.

摘要: 大型语言模型(LLM)最近经历了巨大的流行，并被广泛使用，从随意的对话到人工智能驱动的编程。然而，尽管LLM取得了相当大的成功，但它们并不完全可靠，可以就如何进行有害或非法活动提供详细指导。虽然安全措施可以降低此类输出的风险，但对抗性越狱攻击仍然可以利用LLMS产生有害内容。这些越狱模板通常是手动制作的，这使得大规模测试具有挑战性。在本文中，我们介绍了一种新的黑盒越狱模糊框架GPTFuzz，该框架受到AFL模糊框架的启发。GPTFuzz不是手动设计，而是自动生成用于红队LLM的越狱模板。在其核心，GPTFuzz以人类编写的模板作为初始种子，然后对它们进行突变以产生新的模板。我们详细介绍了GPTFuzz的三个关键组成部分：用于平衡效率和可变性的种子选择策略，用于创建语义等价或相似句子的变异算子，以及用于评估越狱攻击成功的判断模型。我们在不同的攻击场景下，针对各种商业和开源LLM，包括ChatGPT、骆驼2和维库纳，对GPTFuzz进行了评估。我们的结果表明，GPTFuzz一致地生成了成功率较高的越狱模板，超过了人工制作的模板。值得注意的是，GPTFuzz对ChatGPT和Llama-2模型的攻击成功率超过90%，即使在初始种子模板不是最优的情况下也是如此。我们预计，GPTFuzz将有助于研究人员和从业者检查LLM的稳健性，并将鼓励进一步探索增强LLM的安全性。



## **25. Low-Resource Languages Jailbreak GPT-4**

低资源语言越狱GPT-4 cs.CL

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02446v1) [paper-pdf](http://arxiv.org/pdf/2310.02446v1)

**Authors**: Zheng-Xin Yong, Cristina Menghini, Stephen H. Bach

**Abstract**: AI safety training and red-teaming of large language models (LLMs) are measures to mitigate the generation of unsafe content. Our work exposes the inherent cross-lingual vulnerability of these safety mechanisms, resulting from the linguistic inequality of safety training data, by successfully circumventing GPT-4's safeguard through translating unsafe English inputs into low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe translated inputs and provides actionable items that can get the users towards their harmful goals 79% of the time, which is on par with or even surpassing state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have significantly lower attack success rate, which suggests that the cross-lingual vulnerability mainly applies to low-resource languages. Previously, limited training on low-resource languages primarily affects speakers of those languages, causing technological disparities. However, our work highlights a crucial shift: this deficiency now poses a risk to all LLMs users. Publicly available translation APIs enable anyone to exploit LLMs' safety vulnerabilities. Therefore, our work calls for a more holistic red-teaming efforts to develop robust multilingual safeguards with wide language coverage.

摘要: AI安全培训和大型语言模型(LLM)的红团队是减少不安全内容生成的措施。我们的工作通过将不安全的英语输入翻译成低资源的语言，成功地绕过了GPT-4的S保障，暴露了这些安全机制固有的跨语言漏洞，这是由于安全培训数据的语言不平等造成的。在AdvBenchmark上，GPT-4与不安全的翻译输入接触，并提供可操作的项目，可以在79%的时间内引导用户实现他们的有害目标，这与最先进的越狱攻击不相上下，甚至超过了这一水平。其他高/中资源语言的攻击成功率明显较低，这表明跨语言漏洞主要适用于低资源语言。以前，关于低资源语言的培训有限，主要影响说这些语言的人，造成技术差距。然而，我们的工作突出了一个关键的转变：这一缺陷现在对所有LLMS用户构成了风险。公开提供的转换API使任何人都能够利用LLMS的安全漏洞。因此，我们的工作需要更全面的红队努力，以制定具有广泛语言覆盖面的强大的多语言保障措施。



## **26. Jailbreaker in Jail: Moving Target Defense for Large Language Models**

监狱里的越狱者：大型语言模型的移动目标防御 cs.CR

MTD Workshop in CCS'23

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.02417v1) [paper-pdf](http://arxiv.org/pdf/2310.02417v1)

**Authors**: Bocheng Chen, Advait Paliwal, Qiben Yan

**Abstract**: Large language models (LLMs), known for their capability in understanding and following instructions, are vulnerable to adversarial attacks. Researchers have found that current commercial LLMs either fail to be "harmless" by presenting unethical answers, or fail to be "helpful" by refusing to offer meaningful answers when faced with adversarial queries. To strike a balance between being helpful and harmless, we design a moving target defense (MTD) enhanced LLM system. The system aims to deliver non-toxic answers that align with outputs from multiple model candidates, making them more robust against adversarial attacks. We design a query and output analysis model to filter out unsafe or non-responsive answers. %to achieve the two objectives of randomly selecting outputs from different LLMs. We evaluate over 8 most recent chatbot models with state-of-the-art adversarial queries. Our MTD-enhanced LLM system reduces the attack success rate from 37.5\% to 0\%. Meanwhile, it decreases the response refusal rate from 50\% to 0\%.

摘要: 大型语言模型(LLM)以其理解和遵循指令的能力而闻名，容易受到对手攻击。研究人员发现，当前的商业LLM要么无法提供不道德的答案，要么无法通过在面对敌对问题时提供有意义的答案而无法提供“帮助”。为了在有益和无害之间取得平衡，我们设计了一种增强的移动目标防御LLM系统。该系统旨在提供无毒的答案，与来自多个模型候选人的输出保持一致，使它们更强大地抵御对手攻击。我们设计了一个查询和输出分析模型来过滤掉不安全或无响应的答案。%，以实现从不同LLM中随机选择输出的两个目标。我们使用最先进的对抗性查询评估了超过8个最新的聊天机器人模型。我们的MTD增强型LLM系统将攻击成功率从37.5%降低到0%。同时，它将响应拒绝率从50%降低到0。



## **27. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

AutoDAN：在对齐的大型语言模型上生成秘密越狱提示 cs.CL

Pre-print, code is available at  https://github.com/SheltonLiu-N/AutoDAN

**SubmitDate**: 2023-10-03    [abs](http://arxiv.org/abs/2310.04451v1) [paper-pdf](http://arxiv.org/pdf/2310.04451v1)

**Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

**Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.

摘要: 对齐的大型语言模型(LLM)是强大的语言理解和决策工具，通过与人类反馈的广泛对齐而创建。然而，这些大型模型仍然容易受到越狱攻击，在越狱攻击中，对手操纵提示来获得不应由对齐的LLM提供的恶意输出。调查越狱提示可以引导我们深入研究LLMS的局限性，并进一步指导我们确保它们的安全。不幸的是，现有的越狱技术存在以下两个问题：(1)可扩展性问题，攻击严重依赖手工编写提示；(2)隐蔽性问题，因为攻击依赖基于令牌的算法来生成通常在语义上没有意义的提示，这使得它们很容易通过基本的困惑测试被检测到。鉴于这些挑战，我们打算回答这个问题：我们能否开发出一种能够自动生成秘密越狱提示的方法？在本文中，我们介绍了AutoDAN，一种新的针对对齐LLM的越狱攻击。AutoDAN可以通过精心设计的分层遗传算法自动生成隐形越狱提示。广泛的评估表明，AutoDAN不仅在保持语义意义的同时实现了过程的自动化，而且与基线相比，在跨模型可转移性和跨样本通用性方面表现出了优越的攻击能力。此外，我们还对AutoDAN和基于困惑的防御方法进行了比较，结果表明AutoDAN可以有效地绕过它们。



## **28. LoFT: Local Proxy Fine-tuning For Improving Transferability Of Adversarial Attacks Against Large Language Model**

LOFT：提高大型语言模型对抗性攻击可转移性的局部代理微调 cs.CL

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.04445v1) [paper-pdf](http://arxiv.org/pdf/2310.04445v1)

**Authors**: Muhammad Ahmed Shah, Roshan Sharma, Hira Dhamyal, Raphael Olivier, Ankit Shah, Dareen Alharthi, Hazim T Bukhari, Massa Baali, Soham Deshmukh, Michael Kuhlmann, Bhiksha Raj, Rita Singh

**Abstract**: It has been shown that Large Language Model (LLM) alignments can be circumvented by appending specially crafted attack suffixes with harmful queries to elicit harmful responses. To conduct attacks against private target models whose characterization is unknown, public models can be used as proxies to fashion the attack, with successful attacks being transferred from public proxies to private target models. The success rate of attack depends on how closely the proxy model approximates the private model. We hypothesize that for attacks to be transferrable, it is sufficient if the proxy can approximate the target model in the neighborhood of the harmful query. Therefore, in this paper, we propose \emph{Local Fine-Tuning (LoFT)}, \textit{i.e.}, fine-tuning proxy models on similar queries that lie in the lexico-semantic neighborhood of harmful queries to decrease the divergence between the proxy and target models. First, we demonstrate three approaches to prompt private target models to obtain similar queries given harmful queries. Next, we obtain data for local fine-tuning by eliciting responses from target models for the generated similar queries. Then, we optimize attack suffixes to generate attack prompts and evaluate the impact of our local fine-tuning on the attack's success rate. Experiments show that local fine-tuning of proxy models improves attack transferability and increases attack success rate by $39\%$, $7\%$, and $0.5\%$ (absolute) on target models ChatGPT, GPT-4, and Claude respectively.

摘要: 已有研究表明，通过在巧尽心思构建的攻击后缀上附加有害查询来引发有害响应，可以绕过大型语言模型(LLM)对齐。为了对特征未知的私有目标模型进行攻击，可以使用公共模型作为代理来进行攻击，成功的攻击将从公共代理转移到私有目标模型。攻击的成功率取决于代理模型与私有模型的接近程度。我们假设，对于可转移的攻击，只要代理能够逼近有害查询附近的目标模型就足够了。因此，在本文中，我们提出了对位于有害查询的词典-语义邻域中的相似查询的代理模型进行微调，以减少代理模型和目标模型之间的差异。首先，我们演示了三种方法来提示私人目标模型在给定有害查询的情况下获得类似的查询。接下来，我们通过从目标模型获取对生成的类似查询的响应来获得用于本地微调的数据。然后，我们优化攻击后缀来生成攻击提示，并评估我们的局部微调对攻击成功率的影响。实验表明，代理模型的局部微调提高了攻击的可转移性，使目标模型ChatGPT、GPT-4和Claude的攻击成功率分别提高了39美元、7美元和0.5美元(绝对)。



## **29. Gotcha! This Model Uses My Code! Evaluating Membership Leakage Risks in Code Models**

抓到你了！此模型使用我的代码！评估代码模型中的成员泄漏风险 cs.SE

13 pages

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2310.01166v1) [paper-pdf](http://arxiv.org/pdf/2310.01166v1)

**Authors**: Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsum Kim, Donggyun Han, David Lo

**Abstract**: Given large-scale source code datasets available in open-source projects and advanced large language models, recent code models have been proposed to address a series of critical software engineering tasks, such as program repair and code completion. The training data of the code models come from various sources, not only the publicly available source code, e.g., open-source projects on GitHub but also the private data such as the confidential source code from companies, which may contain sensitive information (for example, SSH keys and personal information). As a result, the use of these code models may raise new privacy concerns.   In this paper, we focus on a critical yet not well-explored question on using code models: what is the risk of membership information leakage in code models? Membership information leakage refers to the risk that an attacker can infer whether a given data point is included in (i.e., a member of) the training data. To answer this question, we propose Gotcha, a novel membership inference attack method specifically for code models. We investigate the membership leakage risk of code models. Our results reveal a worrying fact that the risk of membership leakage is high: although the previous attack methods are close to random guessing, Gotcha can predict the data membership with a high true positive rate of 0.95 and a low false positive rate of 0.10. We also show that the attacker's knowledge of the victim model (e.g., the model architecture and the pre-training data) impacts the success rate of attacks. Further analysis demonstrates that changing the decoding strategy can mitigate the risk of membership leakage. This study calls for more attention to understanding the privacy of code models and developing more effective countermeasures against such attacks.

摘要: 鉴于开源项目中可用的大规模源代码数据集和高级大型语言模型，最近提出了一些代码模型来解决一系列关键的软件工程任务，如程序修复和代码完成。代码模型的训练数据来自各种来源，不仅有公开可用的源代码，如GitHub上的开源项目，还包括私人数据，如来自公司的机密源代码，其中可能包含敏感信息(如SSH密钥和个人信息)。因此，使用这些代码模型可能会引发新的隐私问题。在这篇文章中，我们关注一个关于使用代码模型的关键但没有得到很好探索的问题：代码模型中成员信息泄漏的风险是什么？成员资格信息泄漏是指攻击者可以推断给定数据点是否包括在训练数据中(即，训练数据的成员)的风险。为了回答这个问题，我们提出了Gotcha，一种新的专门针对代码模型的成员推理攻击方法。我们研究了编码模型的成员泄漏风险。我们的结果揭示了一个令人担忧的事实，即成员泄露的风险很高：虽然以前的攻击方法接近随机猜测，但Gotcha可以预测数据的成员身份，真阳性率高达0.95，假阳性率低0.10。我们还表明，攻击者对受害者模型的了解(例如，模型体系结构和预训练数据)会影响攻击的成功率。进一步的分析表明，改变译码策略可以降低成员泄漏的风险。这项研究呼吁更多地关注了解代码模型的隐私，并开发更有效的对策来应对此类攻击。



## **30. LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**

LatticeGen：一种将生成的文本隐藏在网格中的云隐私感知生成框架 cs.CL

**SubmitDate**: 2023-10-02    [abs](http://arxiv.org/abs/2309.17157v2) [paper-pdf](http://arxiv.org/pdf/2309.17157v2)

**Authors**: Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov

**Abstract**: In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).

摘要: 在当前云上使用大型语言模型(LLM)进行提示生成的用户-服务器交互模式中，服务器完全控制生成过程，这为想要将生成的文本保密的用户留下了零的选择。我们提出了LatticeGen，这是一个协作框架，其中服务器仍然处理大部分计算，而用户控制采样操作。其关键思想是，用户将真实生成的序列与噪声令牌混合，并将其隐藏在有噪声的网格中。考虑到来自假设恶意服务器的潜在攻击以及用户如何防御它，我们提出了重复波束搜索攻击和混合噪声方案。在我们的实验中，我们应用LatticeGen来保护提示和生成。实验结果表明，虽然加噪的格子降低了生成质量，但在强攻击下(BERTScore测试50%以上的语义仍然隐藏)，LatticeGen在很大程度上保护了真实的生成。



## **31. Streamlining Attack Tree Generation: A Fragment-Based Approach**

精简攻击树生成：一种基于片段的方法 cs.CR

To appear at the 57th Hawaii International Conference on Social  Systems (HICSS-57), Honolulu, Hawaii. 2024

**SubmitDate**: 2023-10-01    [abs](http://arxiv.org/abs/2310.00654v1) [paper-pdf](http://arxiv.org/pdf/2310.00654v1)

**Authors**: Irdin Pekaric, Markus Frick, Jubril Gbolahan Adigun, Raffaela Groner, Thomas Witte, Alexander Raschke, Michael Felderer, Matthias Tichy

**Abstract**: Attack graphs are a tool for analyzing security vulnerabilities that capture different and prospective attacks on a system. As a threat modeling tool, it shows possible paths that an attacker can exploit to achieve a particular goal. However, due to the large number of vulnerabilities that are published on a daily basis, they have the potential to rapidly expand in size. Consequently, this necessitates a significant amount of resources to generate attack graphs. In addition, generating composited attack models for complex systems such as self-adaptive or AI is very difficult due to their nature to continuously change. In this paper, we present a novel fragment-based attack graph generation approach that utilizes information from publicly available information security databases. Furthermore, we also propose a domain-specific language for attack modeling, which we employ in the proposed attack graph generation approach. Finally, we present a demonstrator example showcasing the attack generator's capability to replicate a verified attack chain, as previously confirmed by security experts.

摘要: 攻击图是一种分析安全漏洞的工具，可捕获对系统的不同攻击和潜在攻击。作为一种威胁建模工具，它显示了攻击者可以利用来实现特定目标的可能路径。然而，由于每天发布的大量漏洞，它们有可能迅速扩大规模。因此，这需要大量的资源来生成攻击图。此外，对于自适应或人工智能等复杂系统，由于其不断变化的性质，生成复合攻击模型是非常困难的。本文提出了一种新的基于片段的攻击图生成方法，该方法利用公共信息安全数据库中的信息生成攻击图。此外，我们还提出了一种特定于领域的攻击建模语言，并将其用于提出的攻击图生成方法。最后，我们给出了一个演示示例，展示了攻击生成器复制经过验证的攻击链的能力，这一点之前得到了安全专家的证实。



## **32. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

评估大语言模型的指令跟随健壮性以实现快速注入 cs.CL

The data and code can be found at  https://github.com/Leezekun/Adv-Instruct-Eval

**SubmitDate**: 2023-09-30    [abs](http://arxiv.org/abs/2308.10819v2) [paper-pdf](http://arxiv.org/pdf/2308.10819v2)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of instruction-following LLMs against adversarial instructions injected in the prompt. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these injected adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction injection attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being ``overfitted'' to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text. The data and code can be found at \url{https://github.com/Leezekun/Adv-Instruct-Eval}.

摘要: 大型语言模型(LLM)在遵循说明方面表现出非凡的熟练程度，这使它们在面向客户的应用程序中具有价值。然而，它们令人印象深刻的能力也引发了人们对对抗性指令带来的风险放大的担忧，这些指令可以被注入第三方攻击者输入的模型中，以操纵LLMS的原始指令并提示意外的操作和内容。因此，了解LLMS准确识别应遵循哪些指令以确保在现实世界场景中安全部署的能力至关重要。在本文中，我们提出了一个开创性的基准，用于自动评估指令跟随LLMS对提示中注入的敌意指令的健壮性。这一基准的目的是量化LLM受注入的敌意指令的影响程度，并评估它们区分这些注入的对抗性指令和原始用户指令的能力。通过使用最先进的指令跟随LLM进行的实验，我们发现它们对敌意指令注入攻击的健壮性存在显著的局限性。此外，我们的研究结果表明，流行的指导性调整模型倾向于在没有真正理解哪些指令应该被遵循的情况下，“过度适应”地遵循提示中的任何指导语。这突出表明，需要解决培训模型理解提示的挑战，而不是仅仅遵循指导短语和完成正文。数据和代码可在\url{https://github.com/Leezekun/Adv-Instruct-Eval}.上找到



## **33. FLIP: Cross-domain Face Anti-spoofing with Language Guidance**

翻转：跨域人脸反欺骗式语言引导 cs.CV

Accepted to ICCV-2023. Project Page:  https://koushiksrivats.github.io/FLIP/

**SubmitDate**: 2023-09-28    [abs](http://arxiv.org/abs/2309.16649v1) [paper-pdf](http://arxiv.org/pdf/2309.16649v1)

**Authors**: Koushik Srivatsan, Muzammal Naseer, Karthik Nandakumar

**Abstract**: Face anti-spoofing (FAS) or presentation attack detection is an essential component of face recognition systems deployed in security-critical applications. Existing FAS methods have poor generalizability to unseen spoof types, camera sensors, and environmental conditions. Recently, vision transformer (ViT) models have been shown to be effective for the FAS task due to their ability to capture long-range dependencies among image patches. However, adaptive modules or auxiliary loss functions are often required to adapt pre-trained ViT weights learned on large-scale datasets such as ImageNet. In this work, we first show that initializing ViTs with multimodal (e.g., CLIP) pre-trained weights improves generalizability for the FAS task, which is in line with the zero-shot transfer capabilities of vision-language pre-trained (VLP) models. We then propose a novel approach for robust cross-domain FAS by grounding visual representations with the help of natural language. Specifically, we show that aligning the image representation with an ensemble of class descriptions (based on natural language semantics) improves FAS generalizability in low-data regimes. Finally, we propose a multimodal contrastive learning strategy to boost feature generalization further and bridge the gap between source and target domains. Extensive experiments on three standard protocols demonstrate that our method significantly outperforms the state-of-the-art methods, achieving better zero-shot transfer performance than five-shot transfer of adaptive ViTs. Code: https://github.com/koushiksrivats/FLIP

摘要: 人脸反欺骗(FAS)或表示攻击检测是部署在安全关键应用中的人脸识别系统的重要组件。现有的FAS方法对未知的欺骗类型、摄像机传感器和环境条件的泛化能力较差。最近，视觉转换器(VIT)模型被证明是有效的，因为它们能够捕获图像斑块之间的远程依赖关系。然而，通常需要自适应模块或辅助损失函数来适应在诸如ImageNet的大规模数据集上学习的预先训练的VIT权重。在这项工作中，我们首先证明了用多模式(如CLIP)预训练权重初始化VITS提高了FAS任务的泛化能力，这与视觉语言预训练(VLP)模型的零镜头迁移能力是一致的。在此基础上，我们提出了一种新的基于自然语言的视觉表达方法，实现了跨域的强健性。具体地说，我们表明，将图像表示与类描述的集合(基于自然语言语义)对齐可以提高低数据条件下的FAS泛化能力。最后，我们提出了一种多通道对比学习策略，以进一步提高特征泛化能力，并弥合源域和目标域之间的差距。在三个标准协议上的大量实验表明，该方法的性能明显优于最新的方法，获得了比自适应VITS的五次传输更好的零次传输性能。代码：https://github.com/koushiksrivats/FLIP



## **34. VDC: Versatile Data Cleanser for Detecting Dirty Samples via Visual-Linguistic Inconsistency**

VDC：通过视觉-语言不一致检测污点样本的通用数据清洁器 cs.CV

22 pages,5 figures,17 tables

**SubmitDate**: 2023-09-28    [abs](http://arxiv.org/abs/2309.16211v1) [paper-pdf](http://arxiv.org/pdf/2309.16211v1)

**Authors**: Zihao Zhu, Mingda Zhang, Shaokui Wei, Bingzhe Wu, Baoyuan Wu

**Abstract**: The role of data in building AI systems has recently been emphasized by the emerging concept of data-centric AI. Unfortunately, in the real-world, datasets may contain dirty samples, such as poisoned samples from backdoor attack, noisy labels in crowdsourcing, and even hybrids of them. The presence of such dirty samples makes the DNNs vunerable and unreliable.Hence, it is critical to detect dirty samples to improve the quality and realiability of dataset. Existing detectors only focus on detecting poisoned samples or noisy labels, that are often prone to weak generalization when dealing with dirty samples from other domains.In this paper, we find a commonality of various dirty samples is visual-linguistic inconsistency between images and associated labels. To capture the semantic inconsistency between modalities, we propose versatile data cleanser (VDC) leveraging the surpassing capabilities of multimodal large language models (MLLM) in cross-modal alignment and reasoning.It consists of three consecutive modules: the visual question generation module to generate insightful questions about the image; the visual question answering module to acquire the semantics of the visual content by answering the questions with MLLM; followed by the visual answer evaluation module to evaluate the inconsistency.Extensive experiments demonstrate its superior performance and generalization to various categories and types of dirty samples.

摘要: 数据在构建人工智能系统中的作用最近被以数据为中心的人工智能的新兴概念所强调。不幸的是，在现实世界中，数据集可能包含肮脏的样本，例如来自后门攻击的有毒样本、众包中嘈杂的标签，甚至是它们的混合体。这些脏样本的存在使得DNN变得脆弱和不可靠，因此，检测脏样本对于提高数据集的质量和可靠性至关重要。现有的检测器只检测有毒样本或有噪声的标签，在处理其他领域的脏样本时往往容易产生较弱的泛化，本文发现各种脏样本的一个共同点是图像和关联标签之间的视觉语言不一致。为了捕捉通道间的语义不一致，利用多通道大语言模型(MLLM)在跨通道对齐和推理方面的优势，提出了通用数据清洗模块(VDC)，它由三个连续的模块组成：视觉问题生成模块，用于生成关于图像的有洞察力的问题；视觉问答模块，通过使用MLLM回答问题来获取视觉内容的语义；以及视觉答案评估模块，用于评估不一致。大量的实验表明，它具有优越的性能和对各种类别和类型的脏样本的泛化。



## **35. Towards the Vulnerability of Watermarking Artificial Intelligence Generated Content**

人工智能生成内容水印的脆弱性研究 cs.CV

**SubmitDate**: 2023-09-27    [abs](http://arxiv.org/abs/2310.07726v1) [paper-pdf](http://arxiv.org/pdf/2310.07726v1)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: Artificial Intelligence Generated Content (AIGC) is gaining great popularity in social media, with many commercial services available. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images, fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content).   Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely without the regulation of the service provider. (2) Watermark forge: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose WMaGi, a unified framework to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing, and a generative adversarial network for watermark removing or forging. We evaluate WMaGi on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared with existing diffusion model-based attacks, WMaGi is 5,050$\sim$11,000$\times$ faster.

摘要: 人工智能生成的内容(AIGC)在社交媒体上越来越受欢迎，提供了许多商业服务。这些服务利用高级生成模型，如潜在扩散模型和大型语言模型，为用户生成创造性内容(例如，逼真的图像、流畅的句子)。这种生成的内容的使用需要受到严格的监管，因为服务提供商需要确保用户不违反使用策略(例如，滥用以商业化、生成和分发不安全的内容)。最近，人们提出了许多水印方法。然而，在本文中，我们证明了攻击者可以很容易地破解这些水印机制。具体地说，我们考虑两种可能的攻击。(1)水印去除：攻击者可以很容易地从生成的内容中删除嵌入的水印，然后自由使用，而不需要服务提供商的监管。(2)水印伪造：对手可以利用来自其他用户的伪造水印创建非法内容，导致服务提供商做出错误的归属。我们提出了WMaGi，一个统一的框架，以整体的方式实现这两种攻击。其关键思想是利用预先训练的扩散模型进行内容处理，并利用生成性对抗网络来去除或伪造水印。我们在不同的数据集和嵌入设置上对WMaGi进行了评估。实验结果表明，该算法在保证生成内容质量的同时，具有较高的成功率。与现有的基于扩散模型的攻击相比，WMaGi的攻击速度快5,050$\sim$11,000$\倍$。



## **36. Advancing Beyond Identification: Multi-bit Watermark for Large Language Models**

超越识别：大型语言模型的多位水印 cs.CL

Under review. 9 pages and appendix

**SubmitDate**: 2023-09-27    [abs](http://arxiv.org/abs/2308.00221v2) [paper-pdf](http://arxiv.org/pdf/2308.00221v2)

**Authors**: KiYoon Yoo, Wonhyuk Ahn, Nojun Kwak

**Abstract**: We propose a method to tackle misuses of large language models beyond the identification of machine-generated text. While existing methods focus on detection, some malicious misuses demand tracing the adversary user for counteracting them. To address this, we propose Multi-bit Watermark via Position Allocation, embedding traceable multi-bit information during language model generation. Leveraging the benefits of zero-bit watermarking, our method enables robust extraction of the watermark without any model access, embedding and extraction of long messages ($\geq$ 32-bit) without finetuning, and maintaining text quality, while allowing zero-bit detection all at the same time. Moreover, our watermark is relatively robust under strong attacks like interleaving human texts and paraphrasing.

摘要: 我们提出了一种方法来解决机器生成文本识别之外的大型语言模型的误用。虽然现有的方法侧重于检测，但一些恶意滥用需要跟踪恶意用户来对抗它们。为了解决这个问题，我们提出了通过位置分配的多比特水印，在语言模型生成过程中嵌入可追踪的多比特信息。利用零位水印的优点，我们的方法可以在不访问任何模型的情况下稳健地提取水印，在不进行精细调整的情况下嵌入和提取长消息($32位)，并保持文本质量，同时允许零位检测。此外，我们的水印在交织文本和转译等强攻击下具有较强的稳健性。



## **37. Large Language Model Alignment: A Survey**

大型语言模型对齐：综述 cs.CL

76 pages

**SubmitDate**: 2023-09-26    [abs](http://arxiv.org/abs/2309.15025v1) [paper-pdf](http://arxiv.org/pdf/2309.15025v1)

**Authors**: Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, Deyi Xiong

**Abstract**: Recent years have witnessed remarkable progress made in large language models (LLMs). Such advancements, while garnering significant attention, have concurrently elicited various concerns. The potential of these models is undeniably vast; however, they may yield texts that are imprecise, misleading, or even detrimental. Consequently, it becomes paramount to employ alignment techniques to ensure these models to exhibit behaviors consistent with human values.   This survey endeavors to furnish an extensive exploration of alignment methodologies designed for LLMs, in conjunction with the extant capability research in this domain. Adopting the lens of AI alignment, we categorize the prevailing methods and emergent proposals for the alignment of LLMs into outer and inner alignment. We also probe into salient issues including the models' interpretability, and potential vulnerabilities to adversarial attacks. To assess LLM alignment, we present a wide variety of benchmarks and evaluation methodologies. After discussing the state of alignment research for LLMs, we finally cast a vision toward the future, contemplating the promising avenues of research that lie ahead.   Our aspiration for this survey extends beyond merely spurring research interests in this realm. We also envision bridging the gap between the AI alignment research community and the researchers engrossed in the capability exploration of LLMs for both capable and safe LLMs.

摘要: 近年来，大型语言模型(LLM)取得了显著进展。这些进展在引起人们极大关注的同时，也引起了各种关注。不可否认，这些模型的潜力是巨大的；然而，它们可能会产生不精确、误导甚至有害的文本。因此，使用对齐技术来确保这些模型表现出与人类价值观一致的行为变得至关重要。本综述致力于结合该领域现有的能力研究，对为LLMS设计的比对方法进行广泛的探索。采用人工智能对准的视角，将目前流行的对准方法和建议分为外对准和内对准两大类。我们还探讨了突出的问题，包括模型的可解释性，以及对对抗性攻击的潜在脆弱性。为了评估LLM一致性，我们提出了各种基准和评估方法。在讨论了LLMS配准研究的现状后，我们最后展望了未来，展望了未来充满希望的研究途径。我们对这项调查的渴望不仅仅是刺激这一领域的研究兴趣。我们还设想弥合人工智能对齐研究社区和致力于LLM能力探索的研究人员之间的差距，以实现有能力和安全的LLM。



## **38. SurrogatePrompt: Bypassing the Safety Filter of Text-To-Image Models via Substitution**

代理提示：通过替换绕过文本到图像模型的安全过滤器 cs.CV

14 pages, 11 figures

**SubmitDate**: 2023-09-25    [abs](http://arxiv.org/abs/2309.14122v1) [paper-pdf](http://arxiv.org/pdf/2309.14122v1)

**Authors**: Zhongjie Ba, Jieming Zhong, Jiachen Lei, Peng Cheng, Qinglong Wang, Zhan Qin, Zhibo Wang, Kui Ren

**Abstract**: Advanced text-to-image models such as DALL-E 2 and Midjourney possess the capacity to generate highly realistic images, raising significant concerns regarding the potential proliferation of unsafe content. This includes adult, violent, or deceptive imagery of political figures. Despite claims of rigorous safety mechanisms implemented in these models to restrict the generation of not-safe-for-work (NSFW) content, we successfully devise and exhibit the first prompt attacks on Midjourney, resulting in the production of abundant photorealistic NSFW images. We reveal the fundamental principles of such prompt attacks and suggest strategically substituting high-risk sections within a suspect prompt to evade closed-source safety measures. Our novel framework, SurrogatePrompt, systematically generates attack prompts, utilizing large language models, image-to-text, and image-to-image modules to automate attack prompt creation at scale. Evaluation results disclose an 88% success rate in bypassing Midjourney's proprietary safety filter with our attack prompts, leading to the generation of counterfeit images depicting political figures in violent scenarios. Both subjective and objective assessments validate that the images generated from our attack prompts present considerable safety hazards.

摘要: 先进的文本到图像模型，如Dall-E2和MidTrik，具有生成高真实感图像的能力，这引发了人们对不安全内容潜在扩散的严重担忧。这包括成人的、暴力的或欺骗性的政治人物形象。尽管声称在这些模型中实施了严格的安全机制来限制不安全工作(NSFW)内容的生成，但我们成功地设计并展示了第一次在中途进行的即时攻击，从而产生了丰富的照片逼真的NSFW图像。我们揭示了这种快速攻击的基本原理，并建议有策略地在可疑提示中替换高风险部分，以规避封闭源代码的安全措施。我们的新框架Surogue atePrompt系统地生成攻击提示，利用大型语言模型、图像到文本和图像到图像模块来自动大规模创建攻击提示。评估结果显示，使用我们的攻击提示绕过MidRoad的专有安全过滤器的成功率为88%，导致生成描绘暴力场景中的政治人物的假冒图像。主观和客观评估都证实，我们的攻击提示生成的图像存在相当大的安全风险。



## **39. Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks**

为预先培训的语言模型辩护，称其为不太可能学习的人，免受后门攻击 cs.LG

Accepted by NeurIPS'23

**SubmitDate**: 2023-09-23    [abs](http://arxiv.org/abs/2309.13256v1) [paper-pdf](http://arxiv.org/pdf/2309.13256v1)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang

**Abstract**: Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP.

摘要: 预先训练的语言模型(PLM)表现出作为少有学习者的显著表现。然而，在这种情况下，他们的安全风险在很大程度上是未被探索的。在这项工作中，我们进行了一项初步研究，表明作为少射学习者的PLM非常容易受到后门攻击，而现有的防御由于少射场景的独特挑战而不够充分。为了应对这样的挑战，我们提倡MDP，这是一种新型的轻量级、可插拔和有效的防御措施，适用于学习机会较少的PLM。具体地说，MDP利用了有毒样本和干净样本的掩蔽敏感度之间的差距：参考有限的少数激发数据作为分布锚，它比较不同掩蔽下给定样本的表示，并将有毒样本识别为具有显著变化的样本。我们分析表明，MDP造成了攻击者在攻击有效性和检测规避之间进行选择的有趣两难境地。使用基准数据集和典型攻击进行的经验评估验证了MDP的有效性。



## **40. Knowledge Sanitization of Large Language Models**

大型语言模型的知识清洗 cs.CL

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11852v1) [paper-pdf](http://arxiv.org/pdf/2309.11852v1)

**Authors**: Yoichi Ishibashi, Hidetoshi Shimodaira

**Abstract**: We explore a knowledge sanitization approach to mitigate the privacy concerns associated with large language models (LLMs). LLMs trained on a large corpus of Web data can memorize and potentially reveal sensitive or confidential information, raising critical security concerns. Our technique fine-tunes these models, prompting them to generate harmless responses such as ``I don't know'' when queried about specific information. Experimental results in a closed-book question-answering task show that our straightforward method not only minimizes particular knowledge leakage but also preserves the overall performance of LLM. These two advantages strengthen the defense against extraction attacks and reduces the emission of harmful content such as hallucinations.

摘要: 我们探索了一种知识净化方法来缓解与大型语言模型(LLM)相关的隐私问题。在大型网络数据语料库上接受培训的LLM可能会记住并可能泄露敏感或机密信息，从而引发关键的安全问题。我们的技术对这些模型进行了微调，促使它们在被问及特定信息时产生无害的反应，如“我不知道”。在闭卷问答任务中的实验结果表明，该方法不仅最大限度地减少了特定的知识泄漏，而且保持了LLM的整体性能。这两个优势加强了对提取攻击的防御，并减少了幻觉等有害内容的排放。



## **41. A Chinese Prompt Attack Dataset for LLMs with Evil Content**

 cs.CL

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11830v1) [paper-pdf](http://arxiv.org/pdf/2309.11830v1)

**Authors**: Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu

**Abstract**: Large Language Models (LLMs) present significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset to evaluate the abilities of defending prompt attack. In this paper, we introduce a Chinese Prompt Attack Dataset for LLMs, called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack approaches and widely concerned attacking contents. Different from previous datasets involving safety estimation, We construct the prompts considering three dimensions: contents, attacking methods and goals, thus the responses can be easily evaluated and analysed. We run several well-known Chinese LLMs on our dataset, and the results show that our prompts are significantly harmful to LLMs, with around 70% attack success rate. We will release CPAD to encourage further studies on prompt attack and defense.

摘要: 大语言模型(LLM)在文本理解和生成中具有重要的优先地位。然而，LLMS面临着产生有害内容的风险，特别是在应用程序中使用时。有几种黑盒攻击方法，如提示攻击，可以改变LLMS的行为，并诱导LLMS生成包含有害内容的意外答案。研究人员对LLMS的快速攻防感兴趣，但目前还没有公开的数据集来评估其防御快速攻击的能力。本文介绍了一个针对LLMS的中文即时攻击数据集CPAD。我们的提示旨在通过精心设计的几种即时攻击方法和广泛关注的攻击内容来诱导LLMS产生意想不到的输出。与以往涉及安全评估的数据集不同，我们从内容、攻击方法和目标三个维度构建提示，从而便于对响应进行评估和分析。我们在我们的数据集上运行了几个著名的中文LLMS，结果表明我们的提示对LLMS有显著的危害，攻击成功率约为70%。我们将发布CPAD，以鼓励进一步研究快速攻防。



## **42. How Robust is Google's Bard to Adversarial Image Attacks?**

谷歌的吟游诗人对敌意图像攻击的健壮程度如何？ cs.CV

Technical report

**SubmitDate**: 2023-09-21    [abs](http://arxiv.org/abs/2309.11751v1) [paper-pdf](http://arxiv.org/pdf/2309.11751v1)

**Authors**: Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, Jun Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., a 26% attack success rate against Bing Chat and a 86% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. Our code is available at https://github.com/thu-ml/Attack-Bard.

摘要: 多通道大语言模型将文本和其他通道(尤其是视觉)结合在一起，在各种多通道任务中取得了前所未有的性能。然而，由于视觉模型的对抗性健壮性问题尚未解决，通过引入视觉输入，MLLMS可能存在更严重的安全风险。在这项工作中，我们研究了Google的Bard，一个与ChatGPT竞争的聊天机器人，最近发布了它的多模式功能，以更好地了解商业MLLMS的漏洞。通过攻击白盒代理视觉编码器或MLLM，生成的敌意示例可以误导BARD输出错误的图像描述，仅基于可转移性的成功率为22%。我们表明，恶意例子也可以攻击其他MLLMS，例如，对Bing Chat的攻击成功率为26%，对Ernie bot的攻击成功率为86%。此外，我们还识别了BARD的两种防御机制，包括人脸检测和图像毒性检测。我们设计了相应的攻击来逃避这些防御，证明了巴德目前的防御也是脆弱的。我们希望这项工作可以加深我们对MLLMS稳健性的理解，并为未来的防御研究提供便利。我们的代码可以在https://github.com/thu-ml/Attack-Bard.上找到



## **43. Model Leeching: An Extraction Attack Targeting LLMs**

模型LEACK：一种针对LLMS的提取攻击 cs.LG

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10544v1) [paper-pdf](http://arxiv.org/pdf/2309.10544v1)

**Authors**: Lewis Birch, William Hackett, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Model Leeching is a novel extraction attack targeting Large Language Models (LLMs), capable of distilling task-specific knowledge from a target LLM into a reduced parameter model. We demonstrate the effectiveness of our attack by extracting task capability from ChatGPT-3.5-Turbo, achieving 73% Exact Match (EM) similarity, and SQuAD EM and F1 accuracy scores of 75% and 87%, respectively for only $50 in API cost. We further demonstrate the feasibility of adversarial attack transferability from an extracted model extracted via Model Leeching to perform ML attack staging against a target LLM, resulting in an 11% increase to attack success rate when applied to ChatGPT-3.5-Turbo.

摘要: 模型提取攻击是一种针对大型语言模型的新型提取攻击，能够将目标语言模型中特定于任务的知识提取到一个简化的参数模型中。我们通过从ChatGPT-3.5-Turbo中提取任务能力来证明我们的攻击的有效性，获得了73%的精确匹配(EM)相似度，以及小队EM和F1的准确率分别为75%和87%，仅需50美元的API成本。进一步论证了利用Model leeching提取的模型对目标LLM执行ML攻击阶段性攻击的可行性，将其应用于ChatGPT-3.5-Turbo，攻击成功率提高了11%。



## **44. Language Guided Adversarial Purification**

语言引导的对抗性净化 cs.LG

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10348v1) [paper-pdf](http://arxiv.org/pdf/2309.10348v1)

**Authors**: Himanshu Singh, A V Subramanyam

**Abstract**: Adversarial purification using generative models demonstrates strong adversarial defense performance. These methods are classifier and attack-agnostic, making them versatile but often computationally intensive. Recent strides in diffusion and score networks have improved image generation and, by extension, adversarial purification. Another highly efficient class of adversarial defense methods known as adversarial training requires specific knowledge of attack vectors, forcing them to be trained extensively on adversarial examples. To overcome these limitations, we introduce a new framework, namely Language Guided Adversarial Purification (LGAP), utilizing pre-trained diffusion models and caption generators to defend against adversarial attacks. Given an input image, our method first generates a caption, which is then used to guide the adversarial purification process through a diffusion network. Our approach has been evaluated against strong adversarial attacks, proving its effectiveness in enhancing adversarial robustness. Our results indicate that LGAP outperforms most existing adversarial defense techniques without requiring specialized network training. This underscores the generalizability of models trained on large datasets, highlighting a promising direction for further research.

摘要: 基于产生式模型的对抗性净化算法表现出较强的对抗性防御性能。这些方法是分类器和攻击不可知的，使它们多才多艺，但往往计算密集。最近在传播和得分网络方面的进展改善了图像生成，进而改善了对手的净化。另一种被称为对抗性训练的高效对抗性防御方法需要对攻击载体的特定知识，迫使他们接受关于对抗性例子的广泛培训。为了克服这些局限性，我们引入了一种新的框架，即语言制导的对抗性净化(LGAP)，利用预先训练的扩散模型和字幕生成器来防御对抗性攻击。在给定输入图像的情况下，我们的方法首先生成字幕，然后使用该字幕通过扩散网络来指导敌方净化过程。我们的方法已经针对强大的对手攻击进行了评估，证明了它在增强对手稳健性方面的有效性。我们的结果表明，LGAP在不需要专门的网络训练的情况下，性能优于大多数现有的对抗性防御技术。这突显了在大数据集上训练的模型的泛化能力，突出了进一步研究的一个有希望的方向。



## **45. LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins**

LLM平台安全：将系统评估框架应用于OpenAI的ChatGPT插件 cs.CR

**SubmitDate**: 2023-09-19    [abs](http://arxiv.org/abs/2309.10254v1) [paper-pdf](http://arxiv.org/pdf/2309.10254v1)

**Authors**: Umar Iqbal, Tadayoshi Kohno, Franziska Roesner

**Abstract**: Large language model (LLM) platforms, such as ChatGPT, have recently begun offering a plugin ecosystem to interface with third-party services on the internet. While these plugins extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Plugins also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future plugin-integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin ecosystem. We uncover plugins that concretely demonstrate the potential for the types of issues that we outline in our attack taxonomy. We conclude by discussing novel challenges and by providing recommendations to improve the security, privacy, and safety of present and future LLM-based computing platforms.

摘要: 大型语言模型(LLM)平台，如ChatGPT，最近开始提供插件生态系统，以与互联网上的第三方服务对接。虽然这些插件扩展了LLM平台的功能，但它们是由任意的第三方开发的，因此不能被隐式信任。插件还使用自然语言与LLM平台和用户交互，这可能会有不准确的解释。在本文中，我们提出了一个框架，为LLM平台设计者分析和改进现有和未来插件集成的LLM平台的安全性、保密性和安全性奠定了基础。我们的框架是攻击分类的公式，通过迭代探索LLM平台利益相关者如何利用他们的能力和责任来对彼此发动攻击而开发的攻击分类。作为迭代过程的一部分，我们在OpenAI的插件生态系统中应用了我们的框架。我们发现了一些插件，这些插件具体展示了我们在攻击分类中概述的问题类型的可能性。最后，我们讨论了新的挑战，并提供了改进当前和未来基于LLM的计算平台的安全性、保密性和安全性的建议。



## **46. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强健对齐的LLM防御对齐破坏攻击 cs.CL

16 Pages, 5 Figures, 3 Tables

**SubmitDate**: 2023-09-18    [abs](http://arxiv.org/abs/2309.14348v1) [paper-pdf](http://arxiv.org/pdf/2309.14348v1)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100\% to around 10\% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **47. Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning**

你的房间不是私人的：强化学习中的梯度反转攻击 cs.RO

7 pages, 4 figures, 2 tables

**SubmitDate**: 2023-09-17    [abs](http://arxiv.org/abs/2306.09273v2) [paper-pdf](http://arxiv.org/pdf/2306.09273v2)

**Authors**: Miao Li, Wenhao Ding, Ding Zhao

**Abstract**: The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advancements in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly in relation to reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervision signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or transmitting the data to public servers. Nevertheless, these gradients contain sufficient information to potentially expose private data. To validate our approach, we conduct experiments on the AI2THOR simulator and evaluate our algorithm on active perception, a prevalent task in embodied AI. The experimental results demonstrate the effectiveness of our method in successfully reconstructing all information from the data across 120 room layouts.

摘要: 随着机器人访问大量的个人信息，隐私成为体现人工智能领域的一个关键问题。然而，体验式人工智能任务中的隐私泄露问题，特别是与强化学习算法相关的隐私泄露问题，在研究中并没有得到足够的考虑。为了解决这一问题，本文对基于值的算法和基于梯度的算法提出了一种攻击，利用梯度求逆来重建状态、动作和监控信号。然而，这些渐变包含了足够的信息来潜在地暴露私有数据。为了验证我们的方法，我们在AI2THOR模拟器上进行了实验，并对我们的算法进行了评估，主动感知是体现人工智能中的一个普遍任务。实验结果表明，我们的方法能够成功地从120个房间布局的数据中重建所有信息。



## **48. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

芝麻开门！大型语言模型的通用黑盒越狱 cs.CL

**SubmitDate**: 2023-09-17    [abs](http://arxiv.org/abs/2309.01446v2) [paper-pdf](http://arxiv.org/pdf/2309.01446v2)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.

摘要: 大型语言模型(LLM)旨在提供有用和安全的响应，它们通常依靠对齐技术来与用户意图和社交指南保持一致。遗憾的是，恶意行为者可能会利用这种对齐方式来操纵LLM的输出，以达到非预期目的。在本文中，我们介绍了一种新的方法，即在模型结构和参数不可访问的情况下，使用遗传算法(GA)来操作LLM。GA攻击的工作原理是优化一个通用的对抗性提示，当与用户的查询结合在一起时，会扰乱被攻击模型的对齐，导致意外的和潜在的有害输出。我们的新方法通过揭示模型响应偏离预期行为的实例，系统地揭示了模型的局限性和漏洞。通过广泛的实验，我们展示了我们技术的有效性，从而通过提供一种诊断工具来评估和增强LLM与人类意图的一致性，从而为正在进行的关于负责任的人工智能开发的讨论做出贡献。据我们所知，这是第一次自动通用黑匣子越狱攻击。



## **49. Context-aware Adversarial Attack on Named Entity Recognition**

针对命名实体识别的上下文感知敌意攻击 cs.CL

**SubmitDate**: 2023-09-16    [abs](http://arxiv.org/abs/2309.08999v1) [paper-pdf](http://arxiv.org/pdf/2309.08999v1)

**Authors**: Shuguang Chen, Leonardo Neves, Thamar Solorio

**Abstract**: In recent years, large pre-trained language models (PLMs) have achieved remarkable performance on many natural language processing benchmarks. Despite their success, prior studies have shown that PLMs are vulnerable to attacks from adversarial examples. In this work, we focus on the named entity recognition task and study context-aware adversarial attack methods to examine the model's robustness. Specifically, we propose perturbing the most informative words for recognizing entities to create adversarial examples and investigate different candidate replacement methods to generate natural and plausible adversarial examples. Experiments and analyses show that our methods are more effective in deceiving the model into making wrong predictions than strong baselines.

摘要: 近年来，大型预训练语言模型(PLM)在许多自然语言处理基准上取得了显著的性能。尽管它们取得了成功，但先前的研究表明，PLM很容易受到对手例子的攻击。在这项工作中，我们以命名实体识别任务为重点，研究上下文感知的对抗性攻击方法，以检验模型的健壮性。具体地说，我们提出通过扰动用于识别实体的信息量最大的词来创建对抗性实例，并研究不同的候选替换方法来生成自然的和可信的对抗性实例。实验和分析表明，我们的方法比强基线更能有效地欺骗模型做出错误的预测。



## **50. ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer**

ICLEF：带专家反馈的情景学习，用于可解释的风格转换 cs.CL

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08583v1) [paper-pdf](http://arxiv.org/pdf/2309.08583v1)

**Authors**: Arkadiy Saakyan, Smaranda Muresan

**Abstract**: While state-of-the-art language models excel at the style transfer task, current work does not address explainability of style transfer systems. Explanations could be generated using large language models such as GPT-3.5 and GPT-4, but the use of such complex systems is inefficient when smaller, widely distributed, and transparent alternatives are available. We propose a framework to augment and improve a formality style transfer dataset with explanations via model distillation from ChatGPT. To further refine the generated explanations, we propose a novel way to incorporate scarce expert human feedback using in-context learning (ICLEF: In-Context Learning from Expert Feedback) by prompting ChatGPT to act as a critic to its own outputs. We use the resulting dataset of 9,960 explainable formality style transfer instances (e-GYAFC) to show that current openly distributed instruction-tuned models (and, in some settings, ChatGPT) perform poorly on the task, and that fine-tuning on our high-quality dataset leads to significant improvements as shown by automatic evaluation. In human evaluation, we show that models much smaller than ChatGPT fine-tuned on our data align better with expert preferences. Finally, we discuss two potential applications of models fine-tuned on the explainable style transfer task: interpretable authorship verification and interpretable adversarial attacks on AI-generated text detectors.

摘要: 虽然最先进的语言模型擅长于风格转换任务，但目前的工作并没有解决风格转换系统的可解释性问题。可以使用大型语言模型(如GPT-3.5和GPT-4)生成解释，但当有较小、分布广泛和透明的替代方案时，使用这种复杂系统的效率很低。通过对ChatGPT的模型提炼，我们提出了一个框架来扩充和改进带有解释的形式化风格的传输数据集。为了进一步完善生成的解释，我们提出了一种新的方法，通过促使ChatGPT作为对自己输出的批评者，使用上下文中学习(ICLEF：In-Context Learning from Expert Feedback)来整合稀缺的专家人类反馈。我们使用9960个可解释形式风格转移实例(e-GYAFC)的结果数据集来表明，当前开放分布的指令优化模型(在某些设置中，ChatGPT)在任务中表现不佳，并且如自动评估所示，对我们的高质量数据集进行微调会导致显著的改进。在人类评估中，我们表明，根据我们的数据微调的模型比ChatGPT小得多，更符合专家的偏好。最后，我们讨论了对可解释风格迁移任务进行微调的模型的两个潜在应用：可解释作者身份验证和对人工智能生成的文本检测器的可解释敌意攻击。



