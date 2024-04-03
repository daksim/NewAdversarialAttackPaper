# Latest Large Language Model Attack Papers
**update at 2024-04-03 21:08:06**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Topic-based Watermarks for LLM-Generated Text**

基于主题的LLM文本水印 cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02138v1) [paper-pdf](http://arxiv.org/pdf/2404.02138v1)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked LLM. Inspired from previous work, we propose using a pair of lists (that are generated based on the specified extracted topic(s)) that specify certain tokens to be included or excluded while generating the watermarked output of the LLM. Using the proposed watermarking algorithm, we show the practicality of a watermark detection algorithm. Furthermore, we discuss a wide range of attacks that can emerge against watermarking algorithms for LLMs and the benefit of the proposed watermarking scheme for the feasibility of modeling a potential attacker considering its benefit vs. loss.

摘要: 最近大型语言模型(LLM)的进步导致了与人类生成的文本相比难以区分的文本输出。水印算法是一种潜在的工具，它通过在LLM生成的输出中嵌入可检测的签名来区分LLM生成的文本和人类生成的文本。然而，目前的水印方案对已知的针对水印算法的攻击缺乏稳健性。此外，考虑到LLM每天生成数万个文本输出，并且水印算法需要记住它生成的每个输出才能使检测工作，因此它们是不切实际的。在这项工作中，针对现有水印方案的局限性，我们提出了一种基于主题的LLMS水印算法的概念。该算法基于提取的输入提示主题或非水印LLM的输出主题，确定如何为带水印的LLM输出生成令牌。受以前工作的启发，我们建议使用一对列表(基于指定的提取主题(S)生成)，这些列表指定在生成LLM的水印输出时要包括或排除的某些标记。利用所提出的水印算法，我们展示了水印检测算法的实用性。此外，我们讨论了针对LLMS的水印算法可能出现的各种攻击，以及所提出的水印方案的好处，以考虑其利弊来对潜在攻击者进行建模的可行性。



## **2. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

开源LLM有多值得信赖？恶意示威下的评估显示其脆弱性 cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.

摘要: 开源大型语言模型(LLM)的快速发展显著地推动了人工智能的发展。然而，人们对他们的可信性仍知之甚少。在缺乏足够可信度的情况下大规模部署这些模型可能会带来重大风险，这突显了迅速发现这些问题的必要性。在这项工作中，我们对开源LLM的可信性进行了对抗性评估，从八个不同的方面对它们进行了仔细的审查，包括毒性、刻板印象、伦理、幻觉、公平性、奉承、隐私和对对抗性演示的健壮性。我们提出了AdvCoU，一种基于话语的扩展链(CUU)提示策略，它结合了精心制作的恶意演示来进行可信度攻击。我们广泛的实验涵盖了最近一系列具有代表性的开源LLM，包括Vicuna、MPT、Falcon、Mistral和Llama 2。经验结果强调了我们攻击策略在不同方面的有效性。更有趣的是，我们的结果分析显示，在一般NLP任务中性能优越的模型并不总是具有更大的可信度；事实上，较大的模型可能更容易受到攻击。此外，经过指令调整、专注于指令遵循的模型往往更容易受到影响，尽管针对安全对齐的微调LLM被证明在减轻对手信任攻击方面是有效的。



## **3. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

人性化机器生成内容：通过对抗攻击规避AI文本检测 cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.

摘要: 随着大型语言模型(LLM)的发展，面对虚假信息传播、知识产权保护和防止学术剽窃等恶意使用案例，检测文本是否由机器生成变得越来越具有挑战性。虽然训练有素的文本检测器在看不见的测试数据上表现出了良好的性能，但最近的研究表明，这些检测器在处理诸如释义等敌意攻击时存在漏洞。在本文中，我们提出了一个更广泛类别的对抗性攻击的框架，旨在对机器生成的内容执行微小的扰动以逃避检测。我们考虑了两种攻击环境：白盒和黑盒，并在动态场景中使用对抗性学习来评估当前检测模型对此类攻击的稳健性的潜在增强。实验结果表明，当前的检测模型可以在短短10秒内被攻破，导致机器生成的文本被错误分类为人类书写的内容。此外，我们还探讨了改进模型在迭代对抗学习中的稳健性的前景。虽然在模型稳健性方面观察到了一些改进，但实际应用仍然面临着巨大的挑战。这些发现为人工智能文本检测器的未来发展指明了方向，强调了需要更准确和更稳健的检测方法。



## **4. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

很好，现在写一篇关于这一点的文章：渐强多转LLM越狱攻击 cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01833v1) [paper-pdf](http://arxiv.org/pdf/2404.01833v1)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as "jailbreaks", seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies, progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we introduce Crescendomation, a tool that automates the Crescendo attack, and our evaluation showcases its effectiveness against state-of-the-art models.

摘要: 大型语言模型(LLM)的受欢迎程度显著提高，并越来越多地被多个应用程序采用。这些低成本管理机构强烈反对从事非法或不道德的话题，以此作为避免造成负责任的人工智能损害的一种手段。然而，最近一系列被称为“越狱”的袭击试图克服这一趋势。直观地说，越狱攻击的目的是缩小模型可以做的事情和它愿意做的事情之间的差距。在这篇文章中，我们介绍了一种新的越狱攻击，称为Crescendo。与现有的越狱方法不同，Cresendo是一种多转弯越狱方法，它以一种看似良性的方式与模型交互。它以关于手头任务的一般提示或问题开始，然后通过参考模型的答复逐步升级对话，逐步导致成功越狱。我们在包括ChatGPT、Gemini Pro、Gemini-Ultra、Llama-2 70b Chat和Anthropic Chat在内的各种公共系统上对Cresendo进行了评估。我们的结果证明了Crescendo的强大效力，它在所有评估的模型和任务中都实现了高攻击成功率。此外，我们还介绍了Cresendomation，这是一种自动化Cresendo攻击的工具，我们的评估展示了它对最先进模型的有效性。



## **5. LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**

LatticeGen：一个将生成文本隐藏在网格中的协作框架，用于云上的隐私感知生成 cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2309.17157v4) [paper-pdf](http://arxiv.org/pdf/2309.17157v4)

**Authors**: Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov

**Abstract**: In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).

摘要: 在当前云上使用大型语言模型(LLM)进行提示生成的用户-服务器交互模式中，服务器完全控制生成过程，这为想要将生成的文本保密的用户留下了零的选择。我们提出了LatticeGen，这是一个协作框架，其中服务器仍然处理大部分计算，而用户控制采样操作。其关键思想是，用户将真实生成的序列与噪声令牌混合，并将其隐藏在有噪声的网格中。考虑到来自假设恶意服务器的潜在攻击以及用户如何防御它，我们提出了重复波束搜索攻击和混合噪声方案。在我们的实验中，我们应用LatticeGen来保护提示和生成。实验结果表明，虽然加噪的格子降低了生成质量，但在强攻击下(BERTScore测试50%以上的语义仍然隐藏)，LatticeGen在很大程度上保护了真实的生成。



## **6. Privacy Backdoors: Enhancing Membership Inference through Poisoning Pre-trained Models**

隐私后门：通过中毒预训练模型增强成员推断 cs.CR

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2404.01231v1) [paper-pdf](http://arxiv.org/pdf/2404.01231v1)

**Authors**: Yuxin Wen, Leo Marchyok, Sanghyun Hong, Jonas Geiping, Tom Goldstein, Nicholas Carlini

**Abstract**: It is commonplace to produce application-specific models by fine-tuning large pre-trained models using a small bespoke dataset. The widespread availability of foundation model checkpoints on the web poses considerable risks, including the vulnerability to backdoor attacks. In this paper, we unveil a new vulnerability: the privacy backdoor attack. This black-box privacy attack aims to amplify the privacy leakage that arises when fine-tuning a model: when a victim fine-tunes a backdoored model, their training data will be leaked at a significantly higher rate than if they had fine-tuned a typical model. We conduct extensive experiments on various datasets and models, including both vision-language models (CLIP) and large language models, demonstrating the broad applicability and effectiveness of such an attack. Additionally, we carry out multiple ablation studies with different fine-tuning methods and inference strategies to thoroughly analyze this new threat. Our findings highlight a critical privacy concern within the machine learning community and call for a reevaluation of safety protocols in the use of open-source pre-trained models.

摘要: 通过使用小型定制数据集微调大型预先训练的模型来生成特定于应用程序的模型是很常见的。网络上广泛存在的基础模型检查点构成了相当大的风险，包括易受后门攻击。在本文中，我们揭示了一个新的漏洞：隐私后门攻击。这种黑匣子隐私攻击旨在放大微调模型时出现的隐私泄露：当受害者微调过时的模型时，他们的训练数据泄露的速度将比他们微调典型模型时高得多。我们在各种数据集和模型上进行了广泛的实验，包括视觉语言模型(CLIP)和大型语言模型，证明了这种攻击的广泛适用性和有效性。此外，我们用不同的微调方法和推理策略进行了多个烧蚀研究，以深入分析这一新的威胁。我们的发现突出了机器学习社区中一个关键的隐私问题，并呼吁重新评估使用开放源码预先训练的模型的安全协议。



## **7. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队游戏：红色团队语言模型的游戏理论框架 cs.CL

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2310.00322v3) [paper-pdf](http://arxiv.org/pdf/2310.00322v3)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **8. Fake Alignment: Are LLMs Really Aligned Well?**

假对齐：LLM真的对齐好吗？ cs.CL

Accepted to the NAACL 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2311.05915v3) [paper-pdf](http://arxiv.org/pdf/2311.05915v3)

**Authors**: Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, Yu-Gang Jiang, Yu Qiao, Yingchun Wang

**Abstract**: The growing awareness of safety concerns in large language models (LLMs) has sparked considerable interest in the evaluation of safety. This study investigates an under-explored issue about the evaluation of LLMs, namely the substantial discrepancy in performance between multiple-choice questions and open-ended questions. Inspired by research on jailbreak attack patterns, we argue this is caused by mismatched generalization. That is, LLM only remembers the answer style for open-ended safety questions, which makes it unable to solve other forms of safety tests. We refer to this phenomenon as fake alignment and construct a comparative benchmark to empirically verify its existence in LLMs. We introduce a Fake alIgNment Evaluation (FINE) framework and two novel metrics--Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimation. Applying FINE to 14 widely-used LLMs reveals several models with purported safety are poorly aligned in practice. Subsequently, we found that multiple-choice format data can also be used as high-quality contrast distillation-based fine-tuning data, which can strongly improve the alignment consistency of LLMs with minimal fine-tuning overhead. For data and code, see https://github.com/AIFlames/Fake-Alignment.

摘要: 随着人们对大型语言模型(LLM)中安全问题的日益关注，人们对安全评估产生了极大的兴趣。本研究探讨了多项选择题和开放式题在多项选择题和开放式题之间存在的显著差异，这是一个尚未被充分探讨的问题。受越狱攻击模式研究的启发，我们认为这是由不匹配的泛化造成的。也就是说，LLM只记住了开放式安全问题的答案风格，这使得它无法解决其他形式的安全测试。我们将这种现象称为伪对齐，并构建了一个比较基准来实证验证这种现象在低密度脂蛋白中的存在。提出了一种伪对齐评估(FINE)框架和两种新的度量方法--一致性分数(CS)和一致安全分数(CS)，它们联合评估两种互补的评估形式来量化伪对齐并获得正确的性能估计。将FINE应用于14个广泛使用的LLM，发现几种声称安全的模型在实践中不太一致。随后，我们发现多项选择格式的数据也可以作为基于对比蒸馏的高质量微调数据，这可以以最小的微调开销有力地提高LLMS的对准一致性。有关数据和代码，请参阅https://github.com/AIFlames/Fake-Alignment.



## **9. VDC: Versatile Data Cleanser based on Visual-Linguistic Inconsistency by Multimodal Large Language Models**

VDC：基于多模态大型语言模型的视觉语言不一致性的通用数据清理器 cs.CV

Accepted to ICLR 2024

**SubmitDate**: 2024-04-01    [abs](http://arxiv.org/abs/2309.16211v2) [paper-pdf](http://arxiv.org/pdf/2309.16211v2)

**Authors**: Zihao Zhu, Mingda Zhang, Shaokui Wei, Bingzhe Wu, Baoyuan Wu

**Abstract**: The role of data in building AI systems has recently been emphasized by the emerging concept of data-centric AI. Unfortunately, in the real-world, datasets may contain dirty samples, such as poisoned samples from backdoor attack, noisy labels in crowdsourcing, and even hybrids of them. The presence of such dirty samples makes the DNNs vunerable and unreliable.Hence, it is critical to detect dirty samples to improve the quality and realiability of dataset. Existing detectors only focus on detecting poisoned samples or noisy labels, that are often prone to weak generalization when dealing with dirty samples from other domains.In this paper, we find a commonality of various dirty samples is visual-linguistic inconsistency between images and associated labels. To capture the semantic inconsistency between modalities, we propose versatile data cleanser (VDC) leveraging the surpassing capabilities of multimodal large language models (MLLM) in cross-modal alignment and reasoning.It consists of three consecutive modules: the visual question generation module to generate insightful questions about the image; the visual question answering module to acquire the semantics of the visual content by answering the questions with MLLM; followed by the visual answer evaluation module to evaluate the inconsistency.Extensive experiments demonstrate its superior performance and generalization to various categories and types of dirty samples. The code is available at \url{https://github.com/zihao-ai/vdc}.

摘要: 数据在构建人工智能系统中的作用最近被以数据为中心的人工智能的新兴概念所强调。不幸的是，在现实世界中，数据集可能包含肮脏的样本，例如来自后门攻击的有毒样本、众包中嘈杂的标签，甚至是它们的混合体。这些脏样本的存在使得DNN变得脆弱和不可靠，因此，检测脏样本对于提高数据集的质量和可靠性至关重要。现有的检测器只检测有毒样本或有噪声的标签，在处理其他领域的脏样本时往往容易产生较弱的泛化，本文发现各种脏样本的一个共同点是图像和关联标签之间的视觉语言不一致。为了捕捉通道间的语义不一致，利用多通道大语言模型(MLLM)在跨通道对齐和推理方面的优势，提出了通用数据清洗模块(VDC)，它由三个连续的模块组成：视觉问题生成模块，用于生成关于图像的有洞察力的问题；视觉问答模块，通过使用MLLM回答问题来获取视觉内容的语义；以及视觉答案评估模块，用于评估不一致。大量的实验表明，它具有优越的性能和对各种类别和类型的脏样本的泛化。代码可在\url{https://github.com/zihao-ai/vdc}.



## **10. Dialectical Alignment: Resolving the Tension of 3H and Security Threats of LLMs**

辩证对齐：化解3H紧张与LLM安全威胁 cs.CL

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2404.00486v1) [paper-pdf](http://arxiv.org/pdf/2404.00486v1)

**Authors**: Shu Yang, Jiayuan Su, Han Jiang, Mengdi Li, Keyuan Cheng, Muhammad Asif Ali, Lijie Hu, Di Wang

**Abstract**: With the rise of large language models (LLMs), ensuring they embody the principles of being helpful, honest, and harmless (3H), known as Human Alignment, becomes crucial. While existing alignment methods like RLHF, DPO, etc., effectively fine-tune LLMs to match preferences in the preference dataset, they often lead LLMs to highly receptive human input and external evidence, even when this information is poisoned. This leads to a tendency for LLMs to be Adaptive Chameleons when external evidence conflicts with their parametric memory. This exacerbates the risk of LLM being attacked by external poisoned data, which poses a significant security risk to LLM system applications such as Retrieval-augmented generation (RAG). To address the challenge, we propose a novel framework: Dialectical Alignment (DA), which (1) utilizes AI feedback to identify optimal strategies for LLMs to navigate inter-context conflicts and context-memory conflicts with different external evidence in context window (i.e., different ratios of poisoned factual contexts); (2) constructs the SFT dataset as well as the preference dataset based on the AI feedback and strategies above; (3) uses the above datasets for LLM alignment to defense poisoned context attack while preserving the effectiveness of in-context knowledge editing. Our experiments show that the dialectical alignment model improves poisoned data attack defense by 20 and does not require any additional prompt engineering or prior declaration of ``you may be attacked`` to the LLMs' context window.

摘要: 随着大型语言模型(LLM)的兴起，确保它们体现了有益、诚实和无害(3H)的原则，即所谓的人类对齐，变得至关重要。虽然现有的比对方法，如RLHF、DPO等，可以有效地微调LLM以匹配偏好数据集中的偏好，但它们往往会导致LLM获得高度接受的人类输入和外部证据，即使这些信息是有毒的。这导致当外部证据与其参数记忆冲突时，LLM有成为自适应变色龙的趋势。这加剧了LLM受到外部有毒数据攻击的风险，这对LLM系统应用程序(如检索增强生成(RAG))构成了重大的安全风险。为了应对这一挑战，我们提出了一种新的框架：辩证对齐(DA)，它(1)利用人工智能反馈来确定LLM在上下文窗口中导航不同外部证据(即不同比例的有毒事实上下文)时的上下文间冲突和上下文-记忆冲突的最佳策略；(2)基于上述AI反馈和策略构建SFT数据集以及偏好数据集；(3)使用上述数据集进行LLM对齐，以防御有毒上下文攻击，同时保持上下文中知识编辑的有效性。我们的实验表明，辩证对齐模型将有毒数据攻击防御提高了20%，并且不需要任何额外的提示工程或预先声明``您可能被攻击``到LLMS上下文窗口。



## **11. Composite Backdoor Attacks Against Large Language Models**

大型语言模型的复合后门攻击 cs.CR

To Appear in Findings of the Association for Computational  Linguistics: NAACL 2024, June 2024

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2310.07676v2) [paper-pdf](http://arxiv.org/pdf/2310.07676v2)

**Authors**: Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: Large language models (LLMs) have demonstrated superior performance compared to previous methods on various tasks, and often serve as the foundation models for many researches and services. However, the untrustworthy third-party LLMs may covertly introduce vulnerabilities for downstream tasks. In this paper, we explore the vulnerability of LLMs through the lens of backdoor attacks. Different from existing backdoor attacks against LLMs, ours scatters multiple trigger keys in different prompt components. Such a Composite Backdoor Attack (CBA) is shown to be stealthier than implanting the same multiple trigger keys in only a single component. CBA ensures that the backdoor is activated only when all trigger keys appear. Our experiments demonstrate that CBA is effective in both natural language processing (NLP) and multimodal tasks. For instance, with $3\%$ poisoning samples against the LLaMA-7B model on the Emotion dataset, our attack achieves a $100\%$ Attack Success Rate (ASR) with a False Triggered Rate (FTR) below $2.06\%$ and negligible model accuracy degradation. Our work highlights the necessity of increased security research on the trustworthiness of foundation LLMs.

摘要: 大型语言模型(LLM)在各种任务上表现出了比以前的方法更好的性能，并且经常作为许多研究和服务的基础模型。然而，不可信任的第三方LLM可能会暗中为下游任务引入漏洞。在本文中，我们通过后门攻击的镜头来探索LLMS的脆弱性。与现有的针对LLMS的后门攻击不同，我们的后门攻击将多个触发键分散在不同的提示组件中。这种复合后门攻击(CBA)被证明比仅在单个组件中植入相同的多个触发键更隐蔽。CBA确保只有当所有触发键都出现时，后门才被激活。实验表明，CBA在自然语言处理(NLP)和多通道任务中都是有效的。例如，在情感数据集上使用$3$中毒样本对骆驼-7B模型进行攻击，我们的攻击获得了$100$攻击成功率(ASR)，而误触发率(FTR)低于$2.06$，而模型精度下降可以忽略不计。我们的工作突出了加强对基金会低成本管理可信性的安全性研究的必要性。



## **12. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

通过对抗攻击生成LLM抵抗数学单词问题 cs.CL

Code/data: https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-03-30    [abs](http://arxiv.org/abs/2402.17916v2) [paper-pdf](http://arxiv.org/pdf/2402.17916v2)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure, offering a nuanced view into model's limitation.

摘要: 大型语言模型(LLM)极大地改变了教育格局。由于目前的抄袭检测工具难以跟上LLMS的快速进步，教育界面临着在LLMS存在的情况下评估学生真正的问题解决能力的挑战。在这项工作中，我们探索了一种确保公平评价的新范式--生成对抗性实例，它保留了用于评价的原始问题的结构和难度，但无法用LLMS解决。聚焦于数学应用题领域，我们利用抽象语法树来结构化地生成对抗性实例，这些实例通过简单地编辑问题中的数值来导致LLMS产生不正确的答案。我们在各种开源和闭源的LLM上进行了实验，定量和定性地证明了我们的方法显著降低了他们的数学问题解决能力。我们识别了LLM之间的共同漏洞，并提出了一种具有成本效益的方法来攻击高成本模型。此外，我们对数学问题进行了自动分析，并调查了失败的原因，为模型的局限性提供了一个细微的视角。



## **13. PETA: Parameter-Efficient Trojan Attacks**

PETA：参数高效木马攻击 cs.CL

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.00648v5) [paper-pdf](http://arxiv.org/pdf/2310.00648v5)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance that is comparable to standard fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we take the initial steps and present PETA, a novel trojan attack that compromises the weights of PLMs by accounting for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a model while the lower-level objective simulates PEFT to both retain the PLM's task-specific performance and ensure that the backdoor persists after fine-tuning. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and clean accuracy, even when the attacker does not have full knowledge of the victim user's training process.

摘要: 参数高效微调(PEFT)使预先训练的语言模型(PLM)能够有效地适应特定任务。通过只调整最小的一组(额外)参数，PEFT实现了与标准微调相当的性能。然而，尽管PEFT被广泛使用，但其安全影响在很大程度上仍未被探索。在本文中，我们采取了初步的步骤，并提出了一种新的木马攻击PETA，它通过双层优化考虑下游适应来折衷PLM的权重：上层目标将后门嵌入到模型中，而下层目标模拟PEFT，既保留了PLM的特定任务性能，又确保了微调后后门的存在。通过对各种下游任务和触发器设计的广泛评估，我们证明了PETA在攻击成功率和干净准确性方面的有效性，即使攻击者并不完全了解受害者用户的培训过程。



## **14. Detoxifying Large Language Models via Knowledge Editing**

基于知识编辑的大型语言模型解化 cs.CL

Ongoing work. Project website:  https://zjunlp.github.io/project/SafeEdit Due to the specificity of the  knowledge editing setting, we revise Tables 1 and 3 to present a fair  comparison of experimental results. More experimental results will be updated  soon

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.14472v2) [paper-pdf](http://arxiv.org/pdf/2403.14472v2)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxify approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.

摘要: 本文研究了利用知识编辑技术对大型语言模型进行去毒处理。我们构建了一个涵盖9个不安全类别、具有各种强大的攻击提示的基准SafeEdit，并配备了全面的度量来进行系统评估。我们用几种知识编辑方法进行了实验，表明知识编辑有可能在对一般性能影响有限的情况下有效地对LLM进行解毒。然后，我们提出了一个简单而有效的基线，称为术中神经监测解毒(DINM)，仅通过一个实例在几个调整步骤内降低LLMS的毒性。我们进一步深入分析了各种解毒方法的内在机制，证明了以前的方法如SFT和DPO可能只是抑制了毒性参数的激活，而DINM在一定程度上减轻了毒性参数的毒性，做出了永久性的调整。我们希望这些洞察力能够为未来开发戒毒方法的工作和LLMS的潜在知识机制提供帮助。代码和基准测试可在https://github.com/zjunlp/EasyEdit.上获得



## **15. Evolving Assembly Code in an Adversarial Environment**

对抗环境下的汇编代码演变 cs.NE

9 pages, 5 figures, 6 listings

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19489v1) [paper-pdf](http://arxiv.org/pdf/2403.19489v1)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve assembly code for the CodeGuru competition. The competition's goal is to create a survivor -- an assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. In addition, we compare our approach with a Large-Language Model, demonstrating that the latter cannot generate a survivor that can win at any competition. This work has important applications for cyber-security, as we utilize evolution to detect weaknesses in survivors. The assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.

摘要: 在这项工作中，我们为CodeGuru竞赛演变汇编代码。这项竞赛的目标是创建一个幸存者--一个在共享内存中运行时间最长的汇编程序，通过抵抗对手幸存者的攻击并找到他们的弱点。对于进化的顶级解算器，我们为汇编语言指定了Backus范式(BNF)，并使用遗传编程(GP)从头开始合成代码。我们通过运行CodeGuru游戏来评估幸存者，以对抗人类编写的获胜幸存者。我们的演进计划发现了他们所针对的计划中的弱点，并利用了这些弱点。此外，我们将我们的方法与大语言模型进行比较，表明后者无法产生能够在任何竞争中获胜的幸存者。这项工作在网络安全方面有重要的应用，因为我们利用进化论来检测幸存者的弱点。程序集BNF是独立于域的；因此，通过修改适应度函数，它可以检测代码弱点并帮助修复它们。最后，CodeGuru竞赛为分析对抗性环境中的GP和代码演化提供了一个新的平台。为了支持这方面的进一步研究，我们对进化的幸存者和发现的弱点进行了彻底的定性分析。



## **16. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

JailbreakBench：一个大型语言模型的开放鲁棒性基准测试 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2404.01318v1) [paper-pdf](http://arxiv.org/pdf/2404.01318v1)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) a new jailbreaking dataset containing 100 unique behaviors, which we call JBB-Behaviors; (2) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak Bch，这是一个开源基准测试，具有以下组件：(1)包含100个独特行为的新越狱数据集，我们称之为JBB行为；(2)不断发展的最新对手提示存储库，我们称为越狱人工制品；(3)标准化评估框架，其中包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)跟踪各种LLM攻击和防御性能的排行榜。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。随着时间的推移，我们将扩大和调整基准，以反映研究界的技术和方法进步。



## **17. Data Poisoning for In-context Learning**

基于上下文学习的数据中毒 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2402.02160v2) [paper-pdf](http://arxiv.org/pdf/2402.02160v2)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.

摘要: 在大型语言模型(LLM)领域，情境学习(ICL)因其适应新任务的创新能力而被公认，它依赖于例子而不是再培训或微调。本文深入研究了ICL对数据中毒攻击的易感性这一关键问题，这是一个尚未完全探索的领域。我们想知道ICL是否易受攻击，因为对手能够操纵示例数据来降低模型性能。为了解决这个问题，我们引入了ICLPoison，这是一个专门的攻击框架，旨在利用ICL的学习机制。我们的方法独特地使用离散文本扰动来战略性地影响ICL过程中LLM的隐藏状态。我们概述了在我们的框架下实施攻击的三种具有代表性的战略，每种战略都在各种模型和任务中进行了严格的评估。我们的全面测试，包括对复杂的GPT-4模型的试验，表明ICL的性能在我们的框架下受到了严重影响。这些发现表明，迫切需要增强防御机制，以保障依赖于情景学习的应用程序中LLMS的完整性和可靠性。



## **18. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

LLM会话安全的攻击、防御与评估：一项调查 cs.CL

Accepted to NAACL 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2402.09283v3) [paper-pdf](http://arxiv.org/pdf/2402.09283v3)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.

摘要: 大型语言模型（LLM）现在在会话应用中很常见。然而，它们被滥用以产生有害反应的风险引起了严重的社会关注，并刺激了最近对LLM会话安全的研究。因此，在本次调查中，我们提供了一个全面的概述最近的研究，涵盖了LLM会话安全的三个关键方面：攻击，防御和评估。我们的目标是提供一个结构化的摘要，以提高对LLM会话安全的理解，并鼓励进一步调查这一重要主题。为了便于参考，我们根据我们的分类法对本次调查中提到的所有研究进行了分类，可在www.example.com上查阅。



## **19. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱陷阱可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2311.08268v3) [paper-pdf](http://arxiv.org/pdf/2311.08268v3)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **20. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\textit {LinkPrompt}$：基于XSLT语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.16432v2) [paper-pdf](http://arxiv.org/pdf/2403.16432v2)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。



## **21. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑盒大语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2310.12214v6) [paper-pdf](http://arxiv.org/pdf/2310.12214v6)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Zhikun Zhang

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **22. Exploring the Deceptive Power of LLM-Generated Fake News: A Study of Real-World Detection Challenges**

探索LLM生成的假新闻的欺骗力量：现实世界的检测挑战研究 cs.CL

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18249v1) [paper-pdf](http://arxiv.org/pdf/2403.18249v1)

**Authors**: Yanshen Sun, Jianfeng He, Limeng Cui, Shuo Lei, Chang-Tien Lu

**Abstract**: Recent advancements in Large Language Models (LLMs) have enabled the creation of fake news, particularly in complex fields like healthcare. Studies highlight the gap in the deceptive power of LLM-generated fake news with and without human assistance, yet the potential of prompting techniques has not been fully explored. Thus, this work aims to determine whether prompting strategies can effectively narrow this gap. Current LLM-based fake news attacks require human intervention for information gathering and often miss details and fail to maintain context consistency. Therefore, to better understand threat tactics, we propose a strong fake news attack method called conditional Variational-autoencoder-Like Prompt (VLPrompt). Unlike current methods, VLPrompt eliminates the need for additional data collection while maintaining contextual coherence and preserving the intricacies of the original text. To propel future research on detecting VLPrompt attacks, we created a new dataset named VLPrompt fake news (VLPFN) containing real and fake texts. Our experiments, including various detection methods and novel human study metrics, were conducted to assess their performance on our dataset, yielding numerous findings.

摘要: 最近大型语言模型(LLM)的进步使假新闻的创造成为可能，特别是在医疗保健等复杂领域。研究突显了在有无人工帮助的情况下，LLM生成的假新闻的欺骗力存在差距，但提示技术的潜力尚未得到充分挖掘。因此，本研究旨在确定激励策略是否能有效缩小这一差距。目前基于LLM的假新闻攻击需要人工干预来收集信息，并且经常错过细节，无法保持上下文一致性。因此，为了更好地理解威胁策略，我们提出了一种强大的假新闻攻击方法，称为条件变分自动编码式提示(VLPrompt)。与目前的方法不同，VLPrompt消除了对额外数据收集的需要，同时保持了上下文的连贯性和原始文本的错综复杂。为了推动未来对VLPrompt攻击检测的研究，我们创建了一个新的数据集，名为VLPrompt假新闻(VLPFN)，包含真实和虚假的文本。我们的实验，包括各种检测方法和新的人体研究指标，被用来评估它们在我们的数据集上的性能，产生了许多发现。



## **23. Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks**

欺骗法学硕士到不服从：形式化，分析和检测越狱 cs.CL

Accepted at LREC-COLING 2024 - The 2024 Joint International  Conference on Computational Linguistics, Language Resources and Evaluation

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2305.14965v4) [paper-pdf](http://arxiv.org/pdf/2305.14965v4)

**Authors**: Abhinav Rao, Sachin Vashistha, Atharva Naik, Somak Aditya, Monojit Choudhury

**Abstract**: Recent explorations with commercial Large Language Models (LLMs) have shown that non-expert users can jailbreak LLMs by simply manipulating their prompts; resulting in degenerate output behavior, privacy and security breaches, offensive outputs, and violations of content regulator policies. Limited studies have been conducted to formalize and analyze these attacks and their mitigations. We bridge this gap by proposing a formalism and a taxonomy of known (and possible) jailbreaks. We survey existing jailbreak methods and their effectiveness on open-source and commercial LLMs (such as GPT-based models, OPT, BLOOM, and FLAN-T5-XXL). We further discuss the challenges of jailbreak detection in terms of their effectiveness against known attacks. For further analysis, we release a dataset of model outputs across 3700 jailbreak prompts over 4 tasks.

摘要: 最近对商业大型语言模型（LLM）的探索表明，非专家用户可以通过简单地操纵他们的提示来越狱LLM；导致退化的输出行为、隐私和安全漏洞、攻击性输出以及违反内容监管政策。已经进行了有限的研究，以正规化和分析这些攻击及其缓解措施。我们通过提出一种形式主义和已知（和可能的）越狱分类来弥合这一差距。我们调查了现有的越狱方法及其在开源和商业LLM上的有效性（如基于GPL的模型、OPT、BLOOM和FLAN—T5—XXL）。我们进一步讨论了越狱检测在对抗已知攻击的有效性方面的挑战。为了进一步分析，我们发布了一个模型输出数据集，该数据集涵盖了4个任务的3700个越狱提示。



## **24. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

基于优化的LLM—as—a—Judge快速注入攻击 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17710v1) [paper-pdf](http://arxiv.org/pdf/2403.17710v1)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. Through extensive experiments, we showcase the capability of JudgeDeceiver in altering decision outcomes across various cases, highlighting the vulnerability of LLM-as-a-Judge systems to the optimization-based prompt injection attack.

摘要: LLM-as-a-Court是一种新的解决方案，它可以使用大型语言模型(LLM)来评估文本信息。基于现有的研究，LLMS在提供一种令人信服的替代传统的人类评估方面表现出显著的性能。然而，这些系统对快速注入攻击的健壮性仍然是一个悬而未决的问题。在这项工作中，我们介绍了一种新的基于优化的快速注入攻击，该攻击是针对LLM-as-a-Court定制的。我们的方法为攻击LLM-as-a-Court的决策过程制定了一个精确的优化目标，并利用优化算法高效地自动生成对抗序列，实现了对模型评估的有针对性和有效的操作。与手工即时注入攻击相比，我们的方法表现出更好的有效性，对基于LLM的判断系统的现有安全范例提出了重大挑战。通过大量的实验，我们展示了JudgeDeceiver在改变不同案件的决策结果方面的能力，突出了LLM-as-a-Court系统对基于优化的即时注入攻击的脆弱性。



## **25. Targeted Visualization of the Backbone of Encoder LLMs**

编码器LLM骨干的目标可视化 cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18872v1) [paper-pdf](http://arxiv.org/pdf/2403.18872v1)

**Authors**: Isaac Roberts, Alexander Schulz, Luca Hermes, Barbara Hammer

**Abstract**: Attention based Large Language Models (LLMs) are the state-of-the-art in natural language processing (NLP). The two most common architectures are encoders such as BERT, and decoders like the GPT models. Despite the success of encoder models, on which we focus in this work, they also bear several risks, including issues with bias or their susceptibility for adversarial attacks, signifying the necessity for explainable AI to detect such issues. While there does exist various local explainability methods focusing on the prediction of single inputs, global methods based on dimensionality reduction for classification inspection, which have emerged in other domains and that go further than just using t-SNE in the embedding space, are not widely spread in NLP.   To reduce this gap, we investigate the application of DeepView, a method for visualizing a part of the decision function together with a data set in two dimensions, to the NLP domain. While in previous work, DeepView has been used to inspect deep image classification models, we demonstrate how to apply it to BERT-based NLP classifiers and investigate its usability in this domain, including settings with adversarially perturbed input samples and pre-trained, fine-tuned, and multi-task models.

摘要: 基于注意力的大语言模型(LLM)是自然语言处理(NLP)领域的前沿技术。两种最常见的架构是编码器(如BERT)和解码器(如GPT模型)。尽管我们在本工作中重点关注的编码器模型取得了成功，但它们也存在几个风险，包括偏见或它们对对抗性攻击的敏感性问题，这意味着有必要使用可解释的人工智能来检测此类问题。虽然有各种局部可解释方法专注于单输入预测，但在其他领域出现的基于降维的全局分类检测方法并没有在NLP中广泛推广，这些方法比仅仅使用嵌入空间中的t-SNE更深入。为了缩小这一差距，我们研究了DeepView在NLP领域的应用，DeepView是一种将决策函数的一部分与二维数据集一起可视化的方法。在以前的工作中，DeepView已经被用来检查深度图像分类模型，我们演示了如何将其应用于基于BERT的NLP分类器，并研究了它在该领域的可用性，包括设置了相反扰动的输入样本和预先训练的、微调的和多任务模型。



## **26. CYGENT: A cybersecurity conversational agent with log summarization powered by GPT-3**

CYGENT：一个网络安全会话代理，具有由GPT—3提供支持的日志摘要 cs.CR

7 pages, 9 figures

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.17160v1) [paper-pdf](http://arxiv.org/pdf/2403.17160v1)

**Authors**: Prasasthy Balasubramanian, Justin Seby, Panos Kostakos

**Abstract**: In response to the escalating cyber-attacks in the modern IT and IoT landscape, we developed CYGENT, a conversational agent framework powered by GPT-3.5 turbo model, designed to aid system administrators in ensuring optimal performance and uninterrupted resource availability. This study focuses on fine-tuning GPT-3 models for cybersecurity tasks, including conversational AI and generative AI tailored specifically for cybersecurity operations. CYGENT assists users by providing cybersecurity information, analyzing and summarizing uploaded log files, detecting specific events, and delivering essential instructions. The conversational agent was developed based on the GPT-3.5 turbo model. We fine-tuned and validated summarizer models (GPT3) using manually generated data points. Using this approach, we achieved a BERTscore of over 97%, indicating GPT-3's enhanced capability in summarizing log files into human-readable formats and providing necessary information to users. Furthermore, we conducted a comparative analysis of GPT-3 models with other Large Language Models (LLMs), including CodeT5-small, CodeT5-base, and CodeT5-base-multi-sum, with the objective of analyzing log analysis techniques. Our analysis consistently demonstrated that Davinci (GPT-3) model outperformed all other LLMs, showcasing higher performance. These findings are crucial for improving human comprehension of logs, particularly in light of the increasing numbers of IoT devices. Additionally, our research suggests that the CodeT5-base-multi-sum model exhibits comparable performance to Davinci to some extent in summarizing logs, indicating its potential as an offline model for this task.

摘要: 为了应对现代IT和物联网环境中不断升级的网络攻击，我们开发了基于GPT-3.5 Turbo模型的会话代理框架CyGENT，旨在帮助系统管理员确保最佳性能和不间断的资源可用性。这项研究的重点是针对网络安全任务微调GPT-3模型，包括专门为网络安全操作量身定做的对话式人工智能和生成式人工智能。CyGENT通过提供网络安全信息、分析和汇总上传的日志文件、检测特定事件和提供基本说明来帮助用户。会话代理是在GPT-3.5涡轮机型的基础上开发的。我们使用手动生成的数据点对汇总器模型(GPT3)进行了微调和验证。使用该方法，我们获得了97%以上的BERT分数，这表明GPT-3的S增强了将日志文件摘要为人类可读格式并向用户提供必要信息的能力。此外，我们还将GPT-3模型与其他大型语言模型(包括CodeT5-Small、CodeT5-BASE和CodeT5-BASE-MULTSUM)进行了比较分析，目的是分析日志分析技术。我们的分析始终表明，Davinci(GPT-3)模型的性能优于所有其他LLM，表现出更高的性能。这些发现对于提高人类对日志的理解至关重要，特别是在物联网设备数量不断增加的情况下。此外，我们的研究表明，CodeT5基多和模型在总结日志方面在一定程度上表现出与Davinci相当的性能，表明其作为这一任务的离线模型的潜力。



## **27. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**

InjectAgent：基准测试工具集成大型语言模型代理中的间接提示注入 cs.CL

28 pages, 5 figures, 9 tables

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.02691v2) [paper-pdf](http://arxiv.org/pdf/2403.02691v2)

**Authors**: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

**Abstract**: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable to IPI attacks, with ReAct-prompted GPT-4 vulnerable to attacks 24% of the time. Further investigation into an enhanced setting, where the attacker instructions are reinforced with a hacking prompt, shows additional increases in success rates, nearly doubling the attack success rate on the ReAct-prompted GPT-4. Our findings raise questions about the widespread deployment of LLM Agents. Our benchmark is available at https://github.com/uiuc-kang-lab/InjecAgent.

摘要: 最近的工作将LLMS体现为代理，允许它们访问工具、执行操作并与外部内容(例如，电子邮件或网站)交互。然而，外部内容会带来间接提示注入(IPI)攻击的风险，在IPI攻击中，恶意指令被嵌入到LLMS处理的内容中，目的是操纵这些代理执行针对用户的有害操作。鉴于此类攻击的潜在严重后果，建立评估和减轻这些风险的基准势在必行。在这项工作中，我们引入了InjecAgent，这是一个旨在评估工具集成的LLM代理对IPI攻击的脆弱性的基准测试。InjecAgent由1,054个测试用例组成，涵盖17个不同的用户工具和62个攻击者工具。我们将攻击意图分为两种主要类型：直接伤害用户和泄露私人数据。我们评估了30种不同的LLM代理，表明代理容易受到IPI攻击，其中反应提示的GPT-4在24%的时间内容易受到攻击。对增强设置的进一步调查显示，成功率进一步提高，反应提示GPT-4的攻击成功率几乎翻了一番。在增强设置中，攻击者的指令通过黑客提示得到加强。我们的发现对LLM特工的广泛部署提出了质疑。我们的基准测试可从https://github.com/uiuc-kang-lab/InjecAgent.获得



## **28. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2402.09132v3) [paper-pdf](http://arxiv.org/pdf/2402.09132v3)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **29. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统文献综述 cs.CR

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.14280v2) [paper-pdf](http://arxiv.org/pdf/2403.14280v2)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型(LLM)在涉及区块链安全(BS)的各个领域中已成为强大的工具。最近的几项研究正在探索将LLMS应用于BS。然而，对于低成本管理的全部应用范围、影响以及对区块链安全的潜在限制，我们的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。作为LLM在区块链安全方面应用的首次综述，本研究旨在全面分析现有研究，阐明LLM如何为增强区块链系统的安全性做出贡献。通过对学术著作的深入研究，我们深入研究了LLMS在区块链安全的各个方面的整合。我们探讨了LLMS增强区块链安全的机制，包括它们在智能合同审计、身份验证、异常检测、漏洞修复等方面的应用。此外，考虑到可扩展性、隐私问题和敌意攻击等因素，我们严格评估了利用LLM实现区块链安全所面临的挑战和限制。我们的审查揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了有价值的见解。



## **30. LLM Paternity Test: Generated Text Detection with LLM Genetic Inheritance**

LLM亲子鉴定：基于LLM遗传的生成文本检测 cs.CL

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2305.12519v2) [paper-pdf](http://arxiv.org/pdf/2305.12519v2)

**Authors**: Xiao Yu, Yuang Qi, Kejiang Chen, Guoqiang Chen, Xi Yang, Pengyuan Zhu, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs) can generate texts that carry the risk of various misuses, including plagiarism, planting fake reviews on e-commerce platforms, or creating inflammatory false tweets. Detecting whether a text is machine-generated has thus become increasingly important. While existing detection methods exhibit superior performance, they often lack generalizability due to their heavy dependence on training data. To alleviate this problem, we propose a model-related generated text detection method, the LLM Paternity Test (LLM-Pat). Specifically, given any candidate text (\textit{child}), LLM-Pat employs an intermediary LLM (\textit{parent}) to reconstruct a \textit{sibling} text corresponding to the given text and then measures the similarity between candidate texts and their sibling texts. High similarity indicates that the candidate text is machine-generated, akin to genetic traits. We have constructed datasets encompassing four scenarios: student responses in educational settings, news creation, academic paper writing, and social media bots to assess the performance of LLM-Pat. The experiments show that LLM-Pat outperforms the existing detection methods and is more robust against paraphrasing attacks and re-translating attacks. Besides, LLM-Pat can also be used to trace which large language model the text was generated by. The constructed dataset and code will be released to benefit the community.

摘要: 大型语言模型(LLM)可以生成带有各种误用风险的文本，包括抄袭、在电子商务平台上种植虚假评论或制造煽动性的虚假推文。因此，检测文本是否是机器生成的变得越来越重要。虽然现有的检测方法表现出了优越的性能，但由于它们对训练数据的严重依赖，往往缺乏泛化能力。为了缓解这一问题，我们提出了一种与模型相关的生成文本检测方法--LLM父子关系测试(LLM-PAT)。具体地说，对于给定的候选文本，LLM-PAT采用一个中间的LLM来重构与给定文本相对应的文本，然后度量候选文本与其兄弟文本之间的相似度。高相似度表明候选文本是机器生成的，类似于遗传特征。我们构建了包含四个场景的数据集：学生在教育环境中的反应、新闻创作、学术论文写作和社交媒体机器人，以评估LLM-PAT的性能。实验表明，LLM-PAT的性能优于现有的检测方法，并且对释义攻击和重译攻击具有更好的鲁棒性。此外，LLM-PAT还可以用于跟踪文本是由哪个大型语言模型生成的。构建的数据集和代码将发布，使社区受益。



## **31. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

突破防御：大型语言模型攻击的比较研究 cs.CR

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.04786v2) [paper-pdf](http://arxiv.org/pdf/2403.04786v2)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.

摘要: 大型语言模型(LLM)已经成为自然语言处理(NLP)领域的基石，在理解和生成类似人类的文本方面提供了变革性的能力。然而，随着它们的日益突出，这些模型的安全和漏洞方面已经引起了极大的关注。本文对各种形式的针对LLMS的攻击进行了全面的综述，讨论了这些攻击的性质和机制、它们的潜在影响以及当前的防御策略。我们深入探讨了旨在操纵模型输出的对抗性攻击、影响模型训练的数据中毒以及与训练数据利用相关的隐私问题等主题。文中还探讨了不同攻击方法的有效性，LLMS对这些攻击的恢复能力，以及对模型完整性和用户信任的影响。通过检查最新的研究，我们提供了对LLM漏洞和防御机制的当前情况的见解。我们的目标是提供对LLM攻击的细微差别的理解，培养人工智能社区的意识，并激发强大的解决方案，以减轻未来发展中的这些风险。



## **32. A hybrid LLM workflow can help identify user privilege related variables in programs of any size**

混合LLM工作流可以帮助识别任何规模的程序中的用户权限相关变量 cs.CR

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15723v1) [paper-pdf](http://arxiv.org/pdf/2403.15723v1)

**Authors**: Haizhou Wang, Zhilong Wang, Peng Liu

**Abstract**: Many programs involves operations and logic manipulating user privileges, which is essential for the security of an organization. Therefore, one common malicious goal of attackers is to obtain or escalate the privileges, causing privilege leakage. To protect the program and the organization against privilege leakage attacks, it is important to eliminate the vulnerabilities which can be exploited to achieve such attacks. Unfortunately, while memory vulnerabilities are less challenging to find, logic vulnerabilities are much more imminent, harmful and difficult to identify. Accordingly, many analysts choose to find user privilege related (UPR) variables first as start points to investigate the code where the UPR variables may be used to see if there exists any vulnerabilities, especially the logic ones. In this paper, we introduce a large language model (LLM) workflow that can assist analysts in identifying such UPR variables, which is considered to be a very time-consuming task. Specifically, our tool will audit all the variables in a program and output a UPR score, which is the degree of relationship (closeness) between the variable and user privileges, for each variable. The proposed approach avoids the drawbacks introduced by directly prompting a LLM to find UPR variables by focusing on leverage the LLM at statement level instead of supplying LLM with very long code snippets. Those variables with high UPR scores are essentially potential UPR variables, which should be manually investigated. Our experiments show that using a typical UPR score threshold (i.e., UPR score >0.8), the false positive rate (FPR) is only 13.49%, while UPR variable found is significantly more than that of the heuristic based method.

摘要: 许多程序涉及操纵用户权限的操作和逻辑，这对组织的安全至关重要。因此，攻击者的一个常见恶意目标是获取或提升权限，从而导致权限泄漏。为了保护程序和组织免受权限泄漏攻击，重要的是消除可被利用来实现此类攻击的漏洞。不幸的是，虽然发现内存漏洞的难度较小，但逻辑漏洞更迫在眉睫、危害更大、更难识别。因此，许多分析人员选择首先找到用户权限相关(UPR)变量作为起点，以调查代码中可能使用UPR变量的地方是否存在任何漏洞，特别是逻辑漏洞。在本文中，我们介绍了一个大型语言模型(LLM)工作流，它可以帮助分析师识别这样的UPR变量，这被认为是一项非常耗时的任务。具体地说，我们的工具将审计程序中的所有变量，并为每个变量输出UPR分数，这是变量和用户权限之间的关系(密切程度)。提出的方法避免了直接提示LLM查找UPR变量的缺点，方法是专注于在语句级利用LLM，而不是向LLM提供非常长的代码片段。那些UPR得分较高的变量本质上是潜在的UPR变量，应手动调查。实验表明，使用典型的UPR评分阈值(即UPR评分>0.8)，错误正确率仅为13.49%，而UPR变量的发现明显多于基于启发式的方法。



## **33. LogPrécis: Unleashing Language Models for Automated Malicious Log Analysis**

LogPrécis：释放用于自动恶意日志分析的语言模型 cs.CR

18 pages, Computer&Security  (https://www.sciencedirect.com/science/article/pii/S0167404824001068), code  available at https://github.com/SmartData-Polito/logprecis, models available  at https://huggingface.co/SmartDataPolito

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2307.08309v3) [paper-pdf](http://arxiv.org/pdf/2307.08309v3)

**Authors**: Matteo Boffa, Rodolfo Vieira Valentim, Luca Vassio, Danilo Giordano, Idilio Drago, Marco Mellia, Zied Ben Houidi

**Abstract**: The collection of security-related logs holds the key to understanding attack behaviors and diagnosing vulnerabilities. Still, their analysis remains a daunting challenge. Recently, Language Models (LMs) have demonstrated unmatched potential in understanding natural and programming languages. The question arises whether and how LMs could be also useful for security experts since their logs contain intrinsically confused and obfuscated information. In this paper, we systematically study how to benefit from the state-of-the-art in LM to automatically analyze text-like Unix shell attack logs. We present a thorough design methodology that leads to LogPr\'ecis. It receives as input raw shell sessions and automatically identifies and assigns the attacker tactic to each portion of the session, i.e., unveiling the sequence of the attacker's goals. We demonstrate LogPr\'ecis capability to support the analysis of two large datasets containing about 400,000 unique Unix shell attacks. LogPr\'ecis reduces them into about 3,000 fingerprints, each grouping sessions with the same sequence of tactics. The abstraction it provides lets the analyst better understand attacks, identify fingerprints, detect novelty, link similar attacks, and track families and mutations. Overall, LogPr\'ecis, released as open source, paves the way for better and more responsive defense against cyberattacks.

摘要: 安全相关日志的收集是了解攻击行为和诊断漏洞的关键。尽管如此，他们的分析仍然是一个艰巨的挑战。最近，语言模型(LMS)在理解自然语言和编程语言方面显示出无与伦比的潜力。问题是，LMS是否以及如何也对安全专家有用，因为他们的日志包含本质上令人困惑和混淆的信息。在本文中，我们系统地研究了如何利用LM的最新技术来自动分析类似文本的Unix外壳攻击日志。我们提出了一种通向LogPr‘ECIS的全面设计方法。它接收原始外壳会话作为输入，并自动识别攻击者的战术并将其分配给会话的每个部分，即揭示攻击者的目标序列。我们演示了LogPr的ECIS功能，以支持对包含约400,000个唯一Unix外壳攻击的两个大型数据集的分析。LogPr‘’ECIS将它们简化为大约3,000个指纹，每个指纹使用相同的战术序列对会话进行分组。它提供的抽象使分析师能够更好地了解攻击、识别指纹、检测新颖性、链接类似攻击以及跟踪家族和突变。总体而言，作为开源发布的LogPr‘ECIS为更好、更具响应性的网络攻击防御铺平了道路。



## **34. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images**

用冗长图像诱导大型视觉语言模型的高能量延迟 cs.CV

Accepted by ICLR 2024

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2401.11170v2) [paper-pdf](http://arxiv.org/pdf/2401.11170v2)

**Authors**: Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, Wei Liu

**Abstract**: Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications. Our code is available at https://github.com/KuofengGao/Verbose_Images.

摘要: 大型视觉语言模型(VLM)，如GPT-4，已经在各种多模式任务中取得了出色的性能。然而，VLMS的部署需要大量的能源消耗和计算资源。一旦攻击者在VLMS的推理过程中恶意导致高能耗和高延迟时间(能量延迟成本)，就会耗尽计算资源。在本文中，我们探索了关于VLMS可用性的攻击面，目的是在VLMS的推理过程中引入高能量延迟代价。我们发现，可以通过最大化生成序列的长度来控制VLMS推理过程中的高能量延迟代价。为此，我们提出了冗长的图像，目的是在推理过程中制作一种不可察觉的扰动来诱导VLM生成长句子。具体而言，我们设计了三个损失目标。首先，提出了一种延迟序列结束(EOS)令牌发生的损失，其中EOS令牌是VLM停止生成更多令牌的信号。此外，还提出了一种不确定性损失和令牌分集损失，分别增加了每个生成令牌的不确定性和整个生成序列中所有令牌之间的多样性，从而打破了令牌级和序列级的输出相关性。在此基础上，提出了一种时间权值调整算法，可以有效地平衡这些损失。大量实验表明，在MS-Coco和ImageNet数据集上，与原始图像相比，我们的冗长图像可以使生成的序列长度分别增加7.87倍和8.56倍，这给各种应用带来了潜在的挑战。我们的代码可以在https://github.com/KuofengGao/Verbose_Images.上找到



## **35. Self-Guard: Empower the LLM to Safeguard Itself**

Self—Guard：授权LLM保护自己 cs.CL

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2310.15851v2) [paper-pdf](http://arxiv.org/pdf/2310.15851v2)

**Authors**: Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong

**Abstract**: The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.

摘要: 越狱攻击可以绕过大型语言模型(LLM)的安全措施，生成有害内容。这种对LLM的滥用已经导致了负面的社会后果。目前，解决越狱攻击的主要方法有两种：安全培训和保障措施。安全培训的重点是对LLM进行进一步培训，以提高其安全性。另一方面，保障措施涉及实施外部模型或过滤器，以防止有害输出。然而，安全培训在适应新攻击类型的能力方面存在限制，往往会导致模型性能下降。事实证明，保障措施的帮助有限。为了解决这些问题，我们提出了一种名为Self-Guard的新方法，它结合了两种安全方法的优点。自我保护包括两个阶段。在第一阶段，我们增强了模型评估有害内容的能力，在第二阶段，我们指示模型对其自身的响应进行一致的有害内容检测。实验证明，Self-Guard对越狱攻击具有很强的抵抗力。在坏案例分析中，我们发现LLM偶尔会对有害查询提供无害的响应。此外，我们在安全培训前后评估了LLM的一般能力，提供了自我保护不会导致LLM性能下降的证据。在敏感性测试中，Self-Guard不仅可以避免在LLM中诱导过度敏感，而且甚至可以缓解这一问题。



## **36. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

闭上眼睛，安全开启：通过图像到文本转换保护多模式LLM cs.CV

Project Page: https://gyhdog99.github.io/projects/ecso/

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.09572v2) [paper-pdf](http://arxiv.org/pdf/2403.09572v2)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities, which, however, are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed due to the introduction of image features. To construct robust MLLMs, we propose ECSO(Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that our ECSO enhances model safety significantly (e.g., a 37.6% improvement on the MM-SafetyBench (SD+OCR), and 71.3% on VLSafe for the LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.

摘要: 多通道大语言模型(MLLM)已经显示出令人印象深刻的推理能力，然而，它们也比它们的前身更容易受到越狱攻击。虽然仍然能够检测到不安全的响应，但我们观察到，由于引入了图像特征，MLLMS中预先对准的LLM的安全机制可以很容易地绕过。为了构造稳健的MLLMS，我们提出了一种新的无需训练的保护方法ECSO(Eyes Closed，Safe On)，该方法利用MLLMS固有的安全意识，通过自适应地将不安全的图像转换为文本来激活MLLMS中预对准的LLMS的本质安全机制，从而产生更安全的响应。在五个最先进的(SOTA)MLLM上的实验表明，我们的ECSO显著增强了模型安全性(例如，对于LLaVA-1.5-7B，MM-SafetyBch(SD+OCR)的性能提高了37.6%，VLSafe的性能提高了71.3%)，同时保持了常见MLLM基准的实用结果。此外，我们还证明了ECSO可以作为数据引擎来生成用于MLLM比对的监督精调(SFT)数据，而无需额外的人工干预。



## **37. Risk and Response in Large Language Models: Evaluating Key Threat Categories**

大型语言模型中的风险和响应：评估关键威胁类别 cs.CL

19 pages, 14 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.14988v1) [paper-pdf](http://arxiv.org/pdf/2403.14988v1)

**Authors**: Bahareh Harandizadeh, Abel Salinas, Fred Morstatter

**Abstract**: This paper explores the pressing issue of risk assessment in Large Language Models (LLMs) as they become increasingly prevalent in various applications. Focusing on how reward models, which are designed to fine-tune pretrained LLMs to align with human values, perceive and categorize different types of risks, we delve into the challenges posed by the subjective nature of preference-based training data. By utilizing the Anthropic Red-team dataset, we analyze major risk categories, including Information Hazards, Malicious Uses, and Discrimination/Hateful content. Our findings indicate that LLMs tend to consider Information Hazards less harmful, a finding confirmed by a specially developed regression model. Additionally, our analysis shows that LLMs respond less stringently to Information Hazards compared to other risks. The study further reveals a significant vulnerability of LLMs to jailbreaking attacks in Information Hazard scenarios, highlighting a critical security concern in LLM risk assessment and emphasizing the need for improved AI safety measures.

摘要: 本文探讨了大型语言模型(LLMS)中风险评估的紧迫问题，因为它们在各种应用中日益普遍。我们聚焦于奖励模型，这些模型旨在微调预先训练的LLM以与人类价值观保持一致，如何感知和分类不同类型的风险，深入探讨基于偏好的训练数据的主观性质带来的挑战。通过利用人类红队数据集，我们分析了主要的风险类别，包括信息危害、恶意使用和歧视/仇恨内容。我们的发现表明，LLM倾向于认为信息风险的危害性较小，这一发现得到了专门开发的回归模型的证实。此外，我们的分析表明，与其他风险相比，低成本管理对信息风险的反应不那么严格。这项研究进一步揭示了LLMS在信息危险情景下对越狱攻击的严重脆弱性，突显了LLM风险评估中的一个关键安全问题，并强调了改进人工智能安全措施的必要性。



## **38. Privacy-Preserving End-to-End Spoken Language Understanding**

隐私保护的端到端口语理解 cs.CR

Accepted by IJCAI

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15510v1) [paper-pdf](http://arxiv.org/pdf/2403.15510v1)

**Authors**: Yinggui Wang, Wei Huang, Le Yang

**Abstract**: Spoken language understanding (SLU), one of the key enabling technologies for human-computer interaction in IoT devices, provides an easy-to-use user interface. Human speech can contain a lot of user-sensitive information, such as gender, identity, and sensitive content. New types of security and privacy breaches have thus emerged. Users do not want to expose their personal sensitive information to malicious attacks by untrusted third parties. Thus, the SLU system needs to ensure that a potential malicious attacker cannot deduce the sensitive attributes of the users, while it should avoid greatly compromising the SLU accuracy. To address the above challenge, this paper proposes a novel SLU multi-task privacy-preserving model to prevent both the speech recognition (ASR) and identity recognition (IR) attacks. The model uses the hidden layer separation technique so that SLU information is distributed only in a specific portion of the hidden layer, and the other two types of information are removed to obtain a privacy-secure hidden layer. In order to achieve good balance between efficiency and privacy, we introduce a new mechanism of model pre-training, namely joint adversarial training, to further enhance the user privacy. Experiments over two SLU datasets show that the proposed method can reduce the accuracy of both the ASR and IR attacks close to that of a random guess, while leaving the SLU performance largely unaffected.

摘要: 口语理解(SLU)是物联网设备中实现人机交互的关键技术之一，它提供了易于使用的用户界面。人类语音可能包含大量用户敏感信息，如性别、身份和敏感内容。因此，出现了新类型的安全和隐私违规行为。用户不想将他们的个人敏感信息暴露在不受信任的第三方的恶意攻击下。因此，SLU系统需要确保潜在的恶意攻击者不能推断用户的敏感属性，同时应该避免极大地损害SLU的准确性。针对上述挑战，提出了一种新的SLU多任务隐私保护模型，以同时防止语音识别(ASR)和身份识别(IR)攻击。该模型使用隐层分离技术，使得SLU信息只分布在隐层的特定部分，而其他两种类型的信息被移除以获得隐私安全的隐层。为了在效率和隐私之间取得良好的平衡，我们引入了一种新的模型预训练机制，即联合对抗性训练，以进一步增强用户的隐私。在两个SLU数据集上的实验表明，该方法可以将ASR和IR攻击的准确率降低到接近随机猜测的水平，而SLU的性能基本不受影响。



## **39. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

BadCLIP：触发感知提示学习，以应对CLIP上的后门攻击 cs.CV

14 pages, 6 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2311.16194v2) [paper-pdf](http://arxiv.org/pdf/2311.16194v2)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.

摘要: 对比视觉语言预训练，也称为CLIP，在解决下游图像识别任务方面显示出了良好的效果。然而，最近的研究表明，夹子模型可以植入一个面向下游的后门。在下游任务上，一个受害者模型在干净的样本上执行得很好，但只要出现特定的触发器，就会预测特定的目标类。对于注入后门，现有的攻击依赖于大量的额外数据来恶意微调整个预先训练的剪辑模型，这使得它们不适用于数据有限的场景。在这项工作中，受最近可学习提示的成功的启发，我们通过在快速学习阶段向CLIP模型注入后门来解决这个问题。我们的方法BadCLIP是建立在对CLIP的后门攻击中的一种新颖而有效的机制上的，即通过触发器同时影响图像和文本编码器。它由应用于图像的可学习触发器和触发器感知上下文生成器组成，使得触发器可以通过触发器感知提示改变文本特征，从而产生强大且可泛化的攻击。在11个数据集上进行的大量实验证明，BadCLIP的清洁准确率与先进的快速学习方法相似，在大多数情况下攻击成功率高于99%。BadCLIP还可以泛化到看不见的类，在跨数据集和跨域设置下表现出很强的泛化能力。



## **40. Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model**

揭开印刷欺骗：大型视觉语言模型中印刷漏洞的透视 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2402.19150v2) [paper-pdf](http://arxiv.org/pdf/2402.19150v2)

**Authors**: Hao Cheng, Erjia Xiao, Jindong Gu, Le Yang, Jinhao Duan, Jize Zhang, Jiahang Cao, Kaidi Xu, Renjing Xu

**Abstract**: Large Vision-Language Models (LVLMs) rely on vision encoders and Large Language Models (LLMs) to exhibit remarkable capabilities on various multi-modal tasks in the joint space of vision and language. However, the Typographic Attack, which disrupts vision-language models (VLMs) such as Contrastive Language-Image Pretraining (CLIP), has also been expected to be a security threat to LVLMs. Firstly, we verify typographic attacks on current well-known commercial and open-source LVLMs and uncover the widespread existence of this threat. Secondly, to better assess this vulnerability, we propose the most comprehensive and largest-scale Typographic Dataset to date. The Typographic Dataset not only considers the evaluation of typographic attacks under various multi-modal tasks but also evaluates the effects of typographic attacks, influenced by texts generated with diverse factors. Based on the evaluation results, we investigate the causes why typographic attacks may impact VLMs and LVLMs, leading to three highly insightful discoveries. By the examination of our discoveries and experimental validation in the Typographic Dataset, we reduce the performance degradation from $42.07\%$ to $13.90\%$ when LVLMs confront typographic attacks.

摘要: 大视觉-语言模型依赖于视觉编码器和大语言模型，在视觉和语言的联合空间中表现出对各种多通道任务的卓越能力。然而，打乱视觉语言模型(如对比语言图像预训练(CLIP))的排版攻击也被认为是对视觉语言模型的安全威胁。首先，我们验证了对当前著名的商业和开源LVLM的排版攻击，并揭示了这种威胁的广泛存在。其次，为了更好地评估这个漏洞，我们提出了迄今为止最全面和最大规模的排版数据集。排版数据集不仅考虑了各种多模式任务下排版攻击的评估，而且还评估了受多种因素生成的文本影响的排版攻击的效果。基于评估结果，我们调查了排版攻击可能影响VLM和LVLM的原因，导致了三个非常有洞察力的发现。通过检验我们的发现和在排版数据集中的实验验证，我们将当LVLMS遇到排版攻击时的性能下降从42.07美元减少到13.90美元。



## **41. $\nabla τ$: Gradient-based and Task-Agnostic machine Unlearning**

$\nabla τ $：基于任务和任务不可知的机器学习 cs.LG

14 pages, 2 figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14339v1) [paper-pdf](http://arxiv.org/pdf/2403.14339v1)

**Authors**: Daniel Trippa, Cesare Campagnano, Maria Sofia Bucarelli, Gabriele Tolomei, Fabrizio Silvestri

**Abstract**: Machine Unlearning, the process of selectively eliminating the influence of certain data examples used during a model's training, has gained significant attention as a means for practitioners to comply with recent data protection regulations. However, existing unlearning methods face critical drawbacks, including their prohibitively high cost, often associated with a large number of hyperparameters, and the limitation of forgetting only relatively small data portions. This often makes retraining the model from scratch a quicker and more effective solution. In this study, we introduce Gradient-based and Task-Agnostic machine Unlearning ($\nabla \tau$), an optimization framework designed to remove the influence of a subset of training data efficiently. It applies adaptive gradient ascent to the data to be forgotten while using standard gradient descent for the remaining data. $\nabla \tau$ offers multiple benefits over existing approaches. It enables the unlearning of large sections of the training dataset (up to 30%). It is versatile, supporting various unlearning tasks (such as subset forgetting or class removal) and applicable across different domains (images, text, etc.). Importantly, $\nabla \tau$ requires no hyperparameter adjustments, making it a more appealing option than retraining the model from scratch. We evaluate our framework's effectiveness using a set of well-established Membership Inference Attack metrics, demonstrating up to 10% enhancements in performance compared to state-of-the-art methods without compromising the original model's accuracy.

摘要: 机器遗忘是有选择地消除模型训练期间使用的某些数据示例的影响的过程，作为从业者遵守最新数据保护法规的一种手段，已受到极大关注。然而，现有的遗忘方法面临着严重的缺陷，包括成本高得令人望而却步，往往与大量的超参数相关，以及仅忘记相对较小的数据部分的限制。这通常会使从头开始重新培训模型成为更快、更有效的解决方案。在这项研究中，我们介绍了基于梯度和任务无关的机器遗忘($\nabla\tau$)，这是一个优化框架，旨在有效地消除训练数据子集的影响。它对要遗忘的数据应用自适应梯度上升，而对剩余数据使用标准梯度下降。与现有方法相比，$\nabla\tau$具有多种优势。它允许忘记训练数据集的大段(高达30%)。它是通用的，支持各种遗忘任务(如子集遗忘或类移除)，并适用于不同的域(图像、文本等)。重要的是，$\nabla\tau$不需要调整超参数，这使它成为比从头开始重新训练模型更有吸引力的选择。我们使用一组完善的成员关系推理攻击度量来评估我们框架的有效性，在不影响原始模型准确性的情况下，与最先进的方法相比，性能最多提高了10%。



## **42. FMM-Attack: A Flow-based Multi-modal Adversarial Attack on Video-based LLMs**

FMM—Attack：一种基于流的多模式对抗性视频LLM攻击 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13507v2) [paper-pdf](http://arxiv.org/pdf/2403.13507v2)

**Authors**: Jinmin Li, Kuofeng Gao, Yang Bai, Jingyun Zhang, Shu-tao Xia, Yisen Wang

**Abstract**: Despite the remarkable performance of video-based large language models (LLMs), their adversarial threat remains unexplored. To fill this gap, we propose the first adversarial attack tailored for video-based LLMs by crafting flow-based multi-modal adversarial perturbations on a small fraction of frames within a video, dubbed FMM-Attack. Extensive experiments show that our attack can effectively induce video-based LLMs to generate incorrect answers when videos are added with imperceptible adversarial perturbations. Intriguingly, our FMM-Attack can also induce garbling in the model output, prompting video-based LLMs to hallucinate. Overall, our observations inspire a further understanding of multi-modal robustness and safety-related feature alignment across different modalities, which is of great importance for various large multi-modal models. Our code is available at https://github.com/THU-Kingmin/FMM-Attack.

摘要: 尽管基于视频的大型语言模型（LLM）表现出色，但它们的对抗性威胁仍未得到探索。为了填补这一空白，我们提出了第一个针对基于视频的LLM的对抗攻击，通过在视频中的一小部分帧上制作基于流的多模式对抗干扰，称为FMM攻击。大量的实验表明，我们的攻击可以有效地诱导基于视频的LLM生成错误的答案时，视频中添加了不可感知的对抗干扰。有趣的是，我们的FM—Attack还可以在模型输出中引起混乱，促使基于视频的LLM产生幻觉。总的来说，我们的观察结果激发了人们对不同模态的多模态鲁棒性和安全相关特性对齐的进一步理解，这对各种大型多模态模型非常重要。我们的代码可在www.example.com获得。



## **43. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

AutoDAN：在对齐的大型语言模型上生成隐蔽的越狱脚本 cs.CL

Published as a conference paper at ICLR 2024. Code is available at  https://github.com/SheltonLiu-N/AutoDAN

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2310.04451v2) [paper-pdf](http://arxiv.org/pdf/2310.04451v2)

**Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

**Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.

摘要: 对齐的大型语言模型(LLM)是强大的语言理解和决策工具，通过与人类反馈的广泛对齐而创建。然而，这些大型模型仍然容易受到越狱攻击，在越狱攻击中，对手操纵提示来获得不应由对齐的LLM提供的恶意输出。调查越狱提示可以引导我们深入研究LLMS的局限性，并进一步指导我们确保它们的安全。不幸的是，现有的越狱技术存在以下两个问题：(1)可扩展性问题，攻击严重依赖手工编写提示；(2)隐蔽性问题，因为攻击依赖基于令牌的算法来生成通常在语义上没有意义的提示，这使得它们很容易通过基本的困惑测试被检测到。鉴于这些挑战，我们打算回答这个问题：我们能否开发出一种能够自动生成秘密越狱提示的方法？在本文中，我们介绍了AutoDAN，一种新的针对对齐LLM的越狱攻击。AutoDAN可以通过精心设计的分层遗传算法自动生成隐形越狱提示。广泛的评估表明，AutoDAN不仅在保持语义意义的同时实现了过程的自动化，而且与基线相比，在跨模型可转移性和跨样本通用性方面表现出了优越的攻击能力。此外，我们还对AutoDAN和基于困惑的防御方法进行了比较，结果表明AutoDAN可以有效地绕过它们。



## **44. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

大语言模型（LLM）安全性和隐私性调查：好、坏和丑 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2312.02003v3) [paper-pdf](http://arxiv.org/pdf/2312.02003v3)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.

摘要: 大型语言模型(LLM)，如ChatGPT和BARD，彻底改变了自然语言的理解和生成。它们具有深刻的语言理解能力、类似人类的文本生成能力、上下文意识和强大的解决问题的技能，使它们在各个领域(例如，搜索引擎、客户支持、翻译)具有无价的价值。与此同时，LLMS也在安全界获得了吸引力，揭示了安全漏洞，并在与安全相关的任务中展示了它们的潜力。本文探讨了LLMS与安全和隐私的交集。具体地说，我们调查了LLM如何对安全和隐私产生积极影响，与其使用相关的潜在风险和威胁，以及LLM中的固有漏洞。通过全面的文献综述，本文将这些论文分为“好的”(有益的LLM应用程序)、“坏的”(攻击性应用程序)和“丑陋的”(LLM的漏洞及其防御)。我们有一些有趣的发现。例如，LLM已被证明可以增强代码安全性(代码漏洞检测)和数据隐私(数据机密性保护)，性能优于传统方法。然而，由于它们类似人类的推理能力，它们也可以被利用来进行各种攻击(特别是用户级的攻击)。我们已经确定了需要进一步研究的领域。例如，对模型和参数提取攻击的研究是有限的，而且往往是理论上的，受到LLM参数规模和保密性的阻碍。安全的指令调优是一个新的发展，需要更多的探索。我们希望我们的工作能够揭示小岛屿发展中国家加强和危害网络安全的潜力。



## **45. Defending Against Indirect Prompt Injection Attacks With Spotlighting**

利用聚光灯防御间接即时注入攻击 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14720v1) [paper-pdf](http://arxiv.org/pdf/2403.14720v1)

**Authors**: Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre Kiciman

**Abstract**: Large Language Models (LLMs), while powerful, are built and trained to process a single text input. In common applications, multiple inputs can be processed by concatenating them together into a single stream of text. However, the LLM is unable to distinguish which sections of prompt belong to various input sources. Indirect prompt injection attacks take advantage of this vulnerability by embedding adversarial instructions into untrusted data being processed alongside user commands. Often, the LLM will mistake the adversarial instructions as user commands to be followed, creating a security vulnerability in the larger system. We introduce spotlighting, a family of prompt engineering techniques that can be used to improve LLMs' ability to distinguish among multiple sources of input. The key insight is to utilize transformations of an input to provide a reliable and continuous signal of its provenance. We evaluate spotlighting as a defense against indirect prompt injection attacks, and find that it is a robust defense that has minimal detrimental impact to underlying NLP tasks. Using GPT-family models, we find that spotlighting reduces the attack success rate from greater than {50}\% to below {2}\% in our experiments with minimal impact on task efficacy.

摘要: 大型语言模型(LLM)虽然功能强大，但其构建和训练都是为了处理单个文本输入。在常见的应用程序中，可以通过将多个输入连接成单个文本流来处理它们。然而，LLM无法区分提示的哪些部分属于不同的输入源。间接提示注入攻击通过在与用户命令一起处理的不可信数据中嵌入敌意指令来利用此漏洞。通常，LLM会将敌意指令误认为需要遵循的用户命令，从而在更大的系统中造成安全漏洞。我们介绍了聚光灯，这是一系列快速工程技术，可用于提高LLMS区分多个输入源的能力。关键的洞察力是利用输入的转换来提供关于其来源的可靠和连续的信号。我们评估聚光灯作为对间接即时注入攻击的防御，并发现它是一种健壮的防御，对底层NLP任务的有害影响最小。使用GPT家族模型，我们发现在我们的实验中，聚光灯将攻击成功率从大于{50}\%降低到低于{2}\%，而对任务效能的影响最小。



## **46. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

AttackEval：如何评估大型语言模型越狱攻击的有效性 cs.CL

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2401.09002v3) [paper-pdf](http://arxiv.org/pdf/2401.09002v3)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

摘要: 在我们的研究中，我们开创了一种新的方法来评估越狱攻击对大型语言模型(如GPT-4和LLaMa2)的有效性，不同于传统的专注于健壮性的二进制评估。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供了一个独特的视角，能够对攻击效果进行更全面和细微的评估，并使攻击者能够更好地了解他们的攻击提示。此外，我们还开发了专门为越狱任务量身定做的全面地面事实数据集。这一数据集不仅是我们当前研究的重要基准，而且还为未来的研究奠定了基础资源，使这一不断发展的领域能够进行一致和比较的分析。通过与传统评估方法的细致比较，我们发现我们的评估符合基线的趋势，同时提供了更深入和详细的评估。我们相信，通过准确评估越狱任务中攻击提示的有效性，我们的工作为评估快速注射领域中更广泛的类似甚至更复杂的任务奠定了坚实的基础，这可能会给这一领域带来革命性的变化。



## **47. BadEdit: Backdooring large language models by model editing**

BadEdit：通过模型编辑后台处理大型语言模型 cs.CR

ICLR 2024

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13355v1) [paper-pdf](http://arxiv.org/pdf/2403.13355v1)

**Authors**: Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, Yang Liu

**Abstract**: Mainstream backdoor attack methods typically demand substantial tuning data for poisoning, limiting their practicality and potentially degrading the overall performance when applied to Large Language Models (LLMs). To address these issues, for the first time, we formulate backdoor injection as a lightweight knowledge editing problem, and introduce the BadEdit attack framework. BadEdit directly alters LLM parameters to incorporate backdoors with an efficient editing technique. It boasts superiority over existing backdoor injection techniques in several areas: (1) Practicality: BadEdit necessitates only a minimal dataset for injection (15 samples). (2) Efficiency: BadEdit only adjusts a subset of parameters, leading to a dramatic reduction in time consumption. (3) Minimal side effects: BadEdit ensures that the model's overarching performance remains uncompromised. (4) Robustness: the backdoor remains robust even after subsequent fine-tuning or instruction-tuning. Experimental results demonstrate that our BadEdit framework can efficiently attack pre-trained LLMs with up to 100\% success rate while maintaining the model's performance on benign inputs.

摘要: 主流后门攻击方法通常需要大量调整数据以进行中毒，这限制了它们的实用性，并可能在应用于大型语言模型(LLM)时降低整体性能。为了解决这些问题，我们首次将后门注入描述为一个轻量级的知识编辑问题，并引入了BadEdit攻击框架。BadEDIT直接更改LLM参数，将后门与高效的编辑技术结合在一起。它在几个方面优于现有的后门注入技术：(1)实用性：BadEdit只需要一个最小的注入数据集(15个样本)。(2)效率：BadEDIT只调整部分参数，大大减少了时间消耗。(3)副作用最小：BadEdit可确保模型的总体性能不受影响。(4)健壮性：即使在随后的微调或指令调优之后，后门仍然保持健壮。实验结果表明，我们的BadEdit框架可以有效地攻击预先训练的LLMS，成功率高达100%，同时保持了模型在良性输入下的性能。



## **48. Mapping LLM Security Landscapes: A Comprehensive Stakeholder Risk Assessment Proposal**

绘制LLM安全景观图：全面的利益相关者风险评估提案 cs.CR

10 pages, 1 figure, 3 tables

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13309v1) [paper-pdf](http://arxiv.org/pdf/2403.13309v1)

**Authors**: Rahul Pankajakshan, Sumitra Biswal, Yuvaraj Govindarajulu, Gilad Gressel

**Abstract**: The rapid integration of Large Language Models (LLMs) across diverse sectors has marked a transformative era, showcasing remarkable capabilities in text generation and problem-solving tasks. However, this technological advancement is accompanied by significant risks and vulnerabilities. Despite ongoing security enhancements, attackers persistently exploit these weaknesses, casting doubts on the overall trustworthiness of LLMs. Compounding the issue, organisations are deploying LLM-integrated systems without understanding the severity of potential consequences. Existing studies by OWASP and MITRE offer a general overview of threats and vulnerabilities but lack a method for directly and succinctly analysing the risks for security practitioners, developers, and key decision-makers who are working with this novel technology. To address this gap, we propose a risk assessment process using tools like the OWASP risk rating methodology which is used for traditional systems. We conduct scenario analysis to identify potential threat agents and map the dependent system components against vulnerability factors. Through this analysis, we assess the likelihood of a cyberattack. Subsequently, we conduct a thorough impact analysis to derive a comprehensive threat matrix. We also map threats against three key stakeholder groups: developers engaged in model fine-tuning, application developers utilizing third-party APIs, and end users. The proposed threat matrix provides a holistic evaluation of LLM-related risks, enabling stakeholders to make informed decisions for effective mitigation strategies. Our outlined process serves as an actionable and comprehensive tool for security practitioners, offering insights for resource management and enhancing the overall system security.

摘要: 跨不同部门的大型语言模型(LLM)的快速整合标志着一个变革时代的到来，展示了在文本生成和解决问题任务方面的非凡能力。然而，这种技术进步也伴随着重大的风险和脆弱性。尽管正在进行安全增强，攻击者仍不断地利用这些弱点，使人对低成本管理系统的整体可信度产生怀疑。让问题变得更加复杂的是，组织在部署LLM集成系统时，并不了解潜在后果的严重性。OWASP和MITRE的现有研究提供了对威胁和漏洞的总体概述，但缺乏直接和简洁地分析使用这一新技术的安全从业者、开发人员和关键决策者的风险的方法。为了解决这一差距，我们提出了一个风险评估过程，使用传统系统使用的OWASP风险评级方法等工具。我们进行场景分析，以确定潜在的威胁代理，并将依赖的系统组件与漏洞因素进行映射。通过此分析，我们评估了网络攻击的可能性。随后，我们进行了彻底的影响分析，以得出一个全面的威胁矩阵。我们还将威胁映射到三个关键的利益相关者群体：从事模型微调的开发人员、使用第三方API的应用程序开发人员和最终用户。拟议的威胁矩阵提供了对LLM相关风险的全面评估，使利益相关者能够做出明智的决策，制定有效的缓解战略。我们概述的流程为安全从业者提供了一种可操作的综合工具，为资源管理和增强整体系统安全性提供了见解。



## **49. Bypassing LLM Watermarks with Color-Aware Substitutions**

使用颜色感知替代品来处理LLM水印 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.14719v1) [paper-pdf](http://arxiv.org/pdf/2403.14719v1)

**Authors**: Qilong Wu, Varun Chandrasekaran

**Abstract**: Watermarking approaches are proposed to identify if text being circulated is human or large language model (LLM) generated. The state-of-the-art watermarking strategy of Kirchenbauer et al. (2023a) biases the LLM to generate specific (``green'') tokens. However, determining the robustness of this watermarking method is an open problem. Existing attack methods fail to evade detection for longer text segments. We overcome this limitation, and propose {\em Self Color Testing-based Substitution (SCTS)}, the first ``color-aware'' attack. SCTS obtains color information by strategically prompting the watermarked LLM and comparing output tokens frequencies. It uses this information to determine token colors, and substitutes green tokens with non-green ones. In our experiments, SCTS successfully evades watermark detection using fewer number of edits than related work. Additionally, we show both theoretically and empirically that SCTS can remove the watermark for arbitrarily long watermarked text.

摘要: 提出了一种用于识别流传的文本是人类文本还是大型语言模型(LLM)文本的水印方法。Kirchenbauer等人的最新水印策略。(2023a)偏置LLM以生成特定的(‘’绿色‘’)令牌。然而，确定这种水印方法的稳健性是一个开放的问题。现有的攻击方法无法逃避对较长文本段的检测。我们克服了这一局限性，提出了基于自颜色测试的替换(SCTS)，这是第一个“颜色感知”攻击。SCTS通过策略性地提示带水印的LLM和比较输出标记的频率来获得颜色信息。它使用此信息来确定令牌颜色，并用非绿色令牌替换绿色令牌。在我们的实验中，与相关工作相比，SCTS使用更少的编辑次数成功地避开了水印检测。此外，我们在理论和实验上都证明了SCTS可以去除任意长度水印文本的水印。



## **50. Review of Generative AI Methods in Cybersecurity**

网络安全中的生成性人工智能方法综述 cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.

摘要: 在过去的十年里，人工智能(AI)变得越来越流行，特别是随着ChatGPT、Gemini和Dall-E等聊天机器人的使用。随着这一崛起，大型语言模型(LLM)和生成性人工智能(GenAI)也在日常使用中变得更加普遍。这些进展加强了网络安全的防御态势，也为对手开辟了新的攻击途径。本文全面概述了GenAI当前最先进的部署，包括攻击、越狱以及快速注射和反向心理学的应用。本文还提供了GenAI在网络犯罪中的各种应用，如自动黑客、网络钓鱼电子邮件、社会工程、反向密码学、创建攻击负载和创建恶意软件。GenAI可以通过数据集构建、安全代码开发、威胁情报、防御措施、报告和网络攻击检测等策略，显著提高防御性网络安全流程的自动化程度。在这项研究中，我们建议未来的研究应专注于发展强大的伦理规范和创新的防御机制，以解决GenAI目前造成的问题，并进一步鼓励对其未来在网络安全中的应用采取公正的方法。此外，我们强调跨学科方法的重要性，以进一步弥合科学发展和伦理考量之间的差距。



