# Latest Large Language Model Attack Papers
**update at 2024-03-04 16:53:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：快速分解和重构使LLM成为强大的越狱者 cs.CR

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.16914v2) [paper-pdf](http://arxiv.org/pdf/2402.16914v2)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



## **2. Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**

这是一顿免费午餐：用模型合并来清理过时的模型 cs.CL

work in progress

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19334v1) [paper-pdf](http://arxiv.org/pdf/2402.19334v1)

**Authors**: Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, Qiongkai Xu

**Abstract**: The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we explore various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks without additional resources or specific knowledge. Our approach consistently outperforms the other advanced baselines, leading to an average of 75% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.

摘要: 通过开放源码倡议使预先培训的语言模型民主化，迅速推动了创新，扩大了获得尖端技术的机会。然而，这种开放性也带来了重大的安全风险，包括后门攻击，其中隐藏的恶意行为由特定的输入触发，损害了自然语言处理(NLP)系统的完整性和可靠性。本文认为，将后门模型与其他同类模型合并可以补救后门漏洞，即使这些模型不是完全安全的。在我们的实验中，我们探索了各种模型(Bert-Base、Roberta-Large、Llama2-7B和Mistral-7B)和数据集(SST-2、OLID、AG News和QNLI)。与多种先进的防御方法相比，我们的方法提供了一种有效和高效的推理阶段防御后门攻击，而不需要额外的资源或特定的知识。我们的方法始终优于其他先进的基准，导致攻击成功率平均降低75%。由于模型合并已经成为提高模型性能的既定方法，它提供的关于防御的额外优势可以被视为免费的额外奖励。



## **3. PRSA: Prompt Reverse Stealing Attacks against Large Language Models**

PRSA：针对大型语言模型的快速反向窃取攻击 cs.CR

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19200v1) [paper-pdf](http://arxiv.org/pdf/2402.19200v1)

**Authors**: Yong Yang, Xuhong Zhang, Yi Jiang, Xi Chen, Haoyu Wang, Shouling Ji, Zonghui Wang

**Abstract**: Prompt, recognized as crucial intellectual property, enables large language models (LLMs) to perform specific tasks without the need of fine-tuning, underscoring their escalating importance. With the rise of prompt-based services, such as prompt marketplaces and LLM applications, providers often display prompts' capabilities through input-output examples to attract users. However, this paradigm raises a pivotal security concern: does the exposure of input-output pairs pose the risk of potential prompt leakage, infringing on the intellectual property rights of the developers? To our knowledge, this problem still has not been comprehensively explored yet. To remedy this gap, in this paper, we perform the first in depth exploration and propose a novel attack framework for reverse-stealing prompts against commercial LLMs, namely PRSA. The main idea of PRSA is that by analyzing the critical features of the input-output pairs, we mimic and gradually infer (steal) the target prompts. In detail, PRSA mainly consists of two key phases: prompt mutation and prompt pruning. In the mutation phase, we propose a prompt attention algorithm based on differential feedback to capture these critical features for effectively inferring the target prompts. In the prompt pruning phase, we identify and mask the words dependent on specific inputs, enabling the prompts to accommodate diverse inputs for generalization. Through extensive evaluation, we verify that PRSA poses a severe threat in real world scenarios. We have reported these findings to prompt service providers and actively collaborate with them to take protective measures for prompt copyright.

摘要: Prompt被认为是至关重要的知识产权，它使大型语言模型（LLM）能够执行特定的任务，而无需进行微调，这凸显了其日益重要的地位。随着基于提示的服务的兴起，如提示市场和LLM应用程序，提供商通常通过输入输出示例来展示提示的功能以吸引用户。然而，这种模式提出了一个关键的安全问题：投入产出对的暴露是否会造成潜在的即时泄漏风险，侵犯开发人员的知识产权？据我们所知，这个问题还没有得到全面的探讨。为了弥补这一差距，在本文中，我们进行了第一次深入的探索，并提出了一种新的攻击框架，反向窃取提示对商业LLM，即PRSA。PRSA的主要思想是通过分析输入输出对的关键特征，模仿并逐步推断（窃取）目标提示。PRSA算法主要包括两个关键阶段：快速变异和快速剪枝。在变异阶段，我们提出了一种基于差分反馈的提示注意算法，以捕获这些关键特征，有效地推断目标提示。在提示修剪阶段，我们识别和屏蔽依赖于特定输入的单词，使提示能够适应不同的输入进行概括。通过广泛的评估，我们验证了PRSA在现实世界中构成了严重的威胁。我们已将这些发现报告给提示服务提供商，并积极与他们合作，采取保护措施，以保护提示版权。



## **4. A Semantic Invariant Robust Watermark for Large Language Models**

一种面向大型语言模型的语义不变鲁棒水印 cs.CR

ICLR2024, 21 pages, 10 figures, 6 tables

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2310.06356v2) [paper-pdf](http://arxiv.org/pdf/2310.06356v2)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at https://github.com/THU-BPM/Robust_Watermark.

摘要: 针对大语言模型的水印算法在检测大语言模型生成的文本方面取得了极高的准确率。这类算法通常涉及在每个生成步骤向LLM的日志添加额外的水印日志。然而，现有的算法面临着攻击健壮性和安全健壮性之间的权衡。这是因为令牌的水印登录由一定数量的先前令牌确定；较小的数字会导致较低的安全稳健性，而较大的数字会导致攻击稳健性不足。在这项工作中，我们提出了一种既具有攻击健壮性又具有安全健壮性的LLMS语义不变水印方法。我们工作中的水印日志是由前面所有令牌的语义确定的。具体地说，我们利用另一种嵌入LLM为所有前面的令牌生成语义嵌入，然后通过我们训练的水印模型将这些语义嵌入转换成水印日志。随后的分析和实验证明了该方法在同义词替换和文本释义等语义不变环境下的攻击健壮性。最后，我们还证明了我们的水印具有足够的安全稳健性。我们的代码和数据可在https://github.com/THU-BPM/Robust_Watermark.上获得



## **5. Typographic Attacks in Large Multimodal Models Can be Alleviated by More Informative Prompts**

大型多模态模型中的排版攻击可以通过更多的信息来缓解 cs.CV

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19150v1) [paper-pdf](http://arxiv.org/pdf/2402.19150v1)

**Authors**: Hao Cheng, Erjia Xiao, Renjing Xu

**Abstract**: Large Multimodal Models (LMMs) rely on pre-trained Vision Language Models (VLMs) and Large Language Models (LLMs) to perform amazing emergent abilities on various multimodal tasks in the joint space of vision and language. However, the Typographic Attack, which shows disruption to VLMs, has also been certified as a security vulnerability to LMMs. In this work, we first comprehensively investigate the distractibility of LMMs by typography. In particular, we introduce the Typographic Dataset designed to evaluate distractibility across various multi-modal subtasks, such as object recognition, visual attributes detection, enumeration, arithmetic computation, and commonsense reasoning. To further study the effect of typographic patterns on performance, we also scrutinize the effect of tuning various typographic factors, encompassing font size, color, opacity, and spatial positioning of typos. We discover that LMMs can partially distinguish visual contents and typos when confronting typographic attacks, which suggests that embeddings from vision encoders contain enough information to distinguish visual contents and typos in images. Inspired by such phenomena, we demonstrate that CLIP's performance of zero-shot classification on typo-ridden images can be significantly improved by providing more informative texts to match images. Furthermore, we also prove that LMMs can utilize more informative prompts to leverage information in embeddings to differentiate between visual content and typos. Finally, we propose a prompt information enhancement method that can effectively mitigate the effects of typography.

摘要: 大型多通道模型(LMM)依靠预先训练好的视觉语言模型(VLM)和大语言模型(LLM)在视觉和语言的联合空间中执行各种多通道任务，具有惊人的应急能力。然而，字体攻击显示了对VLM的破坏，也被证明是LMM的一个安全漏洞。在这项工作中，我们首先通过排版来全面地研究LMM的分心问题。特别是，我们介绍了排版数据集，旨在评估各种多模式子任务的分心能力，如对象识别、视觉属性检测、枚举、算术计算和常识推理。为了进一步研究印刷模式对性能的影响，我们还仔细研究了调整各种印刷因素的影响，包括字体大小、颜色、不透明度和印刷错误的空间位置。我们发现，LMM在面对印刷攻击时可以部分区分视觉内容和错别字，这表明来自视觉编码器的嵌入包含了足够的信息来区分图像中的视觉内容和错别字。受这些现象的启发，我们证明了通过提供更多的信息量的文本来匹配图像，可以显著提高CLIP在拼写错误的图像上的零镜头分类性能。此外，我们还证明了LMM可以利用更多的信息提示来利用嵌入中的信息来区分可视内容和错别字。最后，我们提出了一种能够有效缓解排版影响的即时信息增强方法。



## **6. Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking**

认知超载：用超载的逻辑思维越狱大型语言模型 cs.CL

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2311.09827v2) [paper-pdf](http://arxiv.org/pdf/2311.09827v2)

**Authors**: Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen

**Abstract**: While large language models (LLMs) have demonstrated increasing power, they have also given rise to a wide range of harmful behaviors. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of (1) multilingual cognitive overload, (2) veiled expression, and (3) effect-to-cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.

摘要: 虽然大型语言模型(LLM)显示出越来越大的力量，但它们也引发了广泛的有害行为。作为代表，越狱攻击可能会引发低收入国家的有害或不道德的反应，即使在安全调整之后也是如此。在本文中，我们研究了一类新的越狱攻击，该攻击专门针对LLMS的认知结构和过程而设计。具体地说，我们分析了在(1)多语言认知过载、(2)含蓄表达和(3)因果推理的情况下，LLMS的安全脆弱性。与以前的越狱攻击不同，我们提出的认知过载攻击是一种黑盒攻击，不需要了解模型体系结构或访问模型权重。在AdvBtch和MasterKey上进行的实验表明，各种LLM，包括流行的开源模型Llama 2和专有模型ChatGPT，都可以通过认知过载而受到损害。受认知心理学关于管理认知负荷的研究的启发，我们从两个角度进一步研究了防御认知过载攻击。实证研究表明，我们从三个角度的认知过载可以成功地越狱所有研究的LLM，而现有的防御策略很难有效地缓解造成的恶意使用。



## **7. Vaccine: Perturbation-aware Alignment for Large Language Model**

疫苗：大型语言模型中的扰动感知比对 cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.01109v3) [paper-pdf](http://arxiv.org/pdf/2402.01109v3)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 精调即服务的新范式为大型语言模型(LLM)引入了一个新的攻击面：用户上传的少量有害数据就可以很容易地欺骗精调，产生一个破坏对齐的模型。我们进行了实证分析，发现了一种有害的嵌入漂移现象，揭示了排列断裂效应的可能原因。受我们发现的启发，我们提出了Vaccine，一种扰动感知的对齐技术，以降低用户精调的安全风险。Vaccine的核心思想是通过在比对阶段逐步向其添加精心制作的扰动来产生不变的隐藏嵌入。这使嵌入能够在精细调整阶段抵御来自未清理的用户数据的有害干扰。我们在开源主流LLMS(如Llama2、Opt、Vicuna)上的实验结果表明，疫苗可以提高对有害提示导致的嵌入漂移的健壮性，同时保留对良性提示的推理能力。我们的代码可在\url{https://github.com/git-disl/Vaccine}.



## **8. Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing**

通过语义平滑保护大型语言模型免受越狱攻击 cs.CL

37 pages

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16192v2) [paper-pdf](http://arxiv.org/pdf/2402.16192v2)

**Authors**: Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, Shiyu Chang

**Abstract**: Aligned large language models (LLMs) are vulnerable to jailbreaking attacks, which bypass the safeguards of targeted LLMs and fool them into generating objectionable content. While initial defenses show promise against token-based threat models, there do not exist defenses that provide robustness against semantic attacks and avoid unfavorable trade-offs between robustness and nominal performance. To meet this need, we propose SEMANTICSMOOTH, a smoothing-based defense that aggregates the predictions of multiple semantically transformed copies of a given input prompt. Experimental results demonstrate that SEMANTICSMOOTH achieves state-of-the-art robustness against GCG, PAIR, and AutoDAN attacks while maintaining strong nominal performance on instruction following benchmarks such as InstructionFollowing and AlpacaEval. The codes will be publicly available at https://github.com/UCSB-NLP-Chang/SemanticSmooth.

摘要: 对齐的大型语言模型(LLM)容易受到越狱攻击，这些攻击绕过目标LLM的保护措施，欺骗它们生成令人反感的内容。虽然针对基于令牌的威胁模型的初始防御很有希望，但不存在针对语义攻击提供稳健性并避免稳健性和名义性能之间的不利权衡的防御。为了满足这一需求，我们提出了SEMANTICSMOOTH，这是一种基于平滑的防御方法，它聚合了给定输入提示的多个语义转换副本的预测。实验结果表明，SEMANTICSMOOTH在抵抗GCG、Pair和AutoDAN攻击的同时，在遵循InstructionFollowing和AlpacaEval等基准测试的指令上保持了很强的名义性能。这些代码将在https://github.com/UCSB-NLP-Chang/SemanticSmooth.上公开提供



## **9. Defending LLMs against Jailbreaking Attacks via Backtranslation**

通过反向翻译保护LLMS免受越狱攻击 cs.CL

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16459v2) [paper-pdf](http://arxiv.org/pdf/2402.16459v2)

**Authors**: Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh

**Abstract**: Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks, which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and is not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts.

摘要: 尽管许多大型语言模型(LLM)已经接受了拒绝有害请求的培训，但他们仍然容易受到越狱攻击，这些攻击会重写原始提示以掩盖其有害意图。在本文中，我们提出了一种新的方法来防御‘’反向翻译‘’越狱攻击。具体地说，给定目标LLM从输入提示生成的初始响应，我们的反向翻译会提示语言模型推断出可能导致该响应的输入提示。推断的提示称为反向翻译提示，它往往会揭示原始提示的实际意图，因为它是基于LLM的响应生成的，而不是由攻击者直接操纵的。然后，我们在回译的提示上再次运行目标LLM，如果模型拒绝回译的提示，我们将拒绝原始提示。我们解释说，拟议的辩护在其有效性和效率方面提供了几个好处。我们的经验证明，我们的防御显著优于基线，在基线难以达到的情况下，我们的防御对良性输入提示的生成质量也几乎没有影响。



## **10. A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems**

LLM安全的新时代：探索现实世界中基于LLM的系统的安全问题 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18649v1) [paper-pdf](http://arxiv.org/pdf/2402.18649v1)

**Authors**: Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Model (LLM) systems are inherently compositional, with individual LLM serving as the core foundation with additional layers of objects such as plugins, sandbox, and so on. Along with the great potential, there are also increasing concerns over the security of such probabilistic intelligent systems. However, existing studies on LLM security often focus on individual LLM, but without examining the ecosystem through the lens of LLM systems with other objects (e.g., Frontend, Webtool, Sandbox, and so on). In this paper, we systematically analyze the security of LLM systems, instead of focusing on the individual LLMs. To do so, we build on top of the information flow and formulate the security of LLM systems as constraints on the alignment of the information flow within LLM and between LLM and other objects. Based on this construction and the unique probabilistic nature of LLM, the attack surface of the LLM system can be decomposed into three key components: (1) multi-layer security analysis, (2) analysis of the existence of constraints, and (3) analysis of the robustness of these constraints. To ground this new attack surface, we propose a multi-layer and multi-step approach and apply it to the state-of-art LLM system, OpenAI GPT4. Our investigation exposes several security issues, not just within the LLM model itself but also in its integration with other components. We found that although the OpenAI GPT4 has designed numerous safety constraints to improve its safety features, these safety constraints are still vulnerable to attackers. To further demonstrate the real-world threats of our discovered vulnerabilities, we construct an end-to-end attack where an adversary can illicitly acquire the user's chat history, all without the need to manipulate the user's input or gain direct access to OpenAI GPT4. Our demo is in the link: https://fzwark.github.io/LLM-System-Attack-Demo/

摘要: 大型语言模型(LLM)系统本质上是组合的，单个LLM充当核心基础，具有附加的对象层，如插件、沙箱等。在这种巨大潜力的同时，人们对这种概率智能系统的安全性也越来越关注。然而，现有的关于LLM安全的研究往往集中在单个LLM上，而没有通过LLM系统与其他对象(如前端、WebTool、沙盒等)的透镜来考察生态系统。在本文中，我们系统地分析了LLM系统的安全性，而不是关注单个LLM。为此，我们建立在信息流的基础上，并将LLM系统的安全性制定为对LLM内以及LLM与其他对象之间的信息流对齐的约束。基于这种构造和LLM的独特概率性质，LLM系统的攻击面可以分解为三个关键部分：(1)多层安全分析，(2)约束的存在性分析，(3)这些约束的稳健性分析。为了对这种新的攻击面进行接地，我们提出了一种多层多步骤的方法，并将其应用于最先进的LLM系统OpenAI GPT4。我们的调查暴露了几个安全问题，不仅在LLM模型本身，而且在它与其他组件的集成中。我们发现，尽管OpenAI GPT4设计了众多安全约束来改进其安全功能，但这些安全约束仍然容易受到攻击者的攻击。为了进一步展示我们发现的漏洞对现实世界的威胁，我们构建了一个端到端攻击，其中对手可以非法获取用户的聊天历史记录，而无需操纵用户的输入或获得对OpenAI GPT4的直接访问。我们的演示位于链接中：https://fzwark.github.io/LLM-System-Attack-Demo/



## **11. Multilingual Jailbreak Challenges in Large Language Models**

大型语言模型中的多语言越狱挑战 cs.CL

ICLR 2024

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2310.06474v2) [paper-pdf](http://arxiv.org/pdf/2310.06474v2)

**Authors**: Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem, wherein malicious instructions can manipulate LLMs to exhibit undesirable behavior. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risky scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario concerns malicious users combining malicious instructions with multilingual prompts to deliberately attack LLMs. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit about three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of malicious instructions, with astonishingly high rates of unsafe output: 80.92\% for ChatGPT and 40.71\% for GPT-4. To handle such a challenge in the multilingual context, we propose a novel \textsc{Self-Defense} framework that automatically generates multilingual training data for safety fine-tuning. Experimental results show that ChatGPT fine-tuned with such data can achieve a substantial reduction in unsafe content generation. Data is available at \url{https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs}.

摘要: 虽然大型语言模型(LLM)在广泛的任务中显示出非凡的能力，但它们构成了潜在的安全问题，如“越狱”问题，在该问题中，恶意指令可以操纵LLM表现出不受欢迎的行为。虽然已经制定了几项预防措施来减轻与低密度脂蛋白相关的潜在风险，但它们主要集中在英语上。在这项研究中，我们揭示了多语言越狱挑战在LLMS中的存在，并考虑了两种潜在的风险情景：无意和故意。非故意场景涉及用户使用非英语提示查询LLMS并无意中绕过安全机制，而有意场景涉及恶意用户将恶意指令与多语言提示相结合来故意攻击LLMS。实验结果表明，在无意情况下，不安全内容的发生率随着语言可用性的降低而增加。具体地说，与高资源语言相比，低资源语言遇到有害内容的可能性大约是ChatGPT和GPT-4语言的三倍。在有意为之的场景中，多语言提示会加剧恶意指令的负面影响，不安全输出率高得惊人：ChatGPT为80.92\%，GPT-4为40.71\%。为了应对多语言环境下的这一挑战，我们提出了一种新的\Textsc{自卫}框架，该框架自动生成用于安全微调的多语言训练数据。实验结果表明，利用这些数据对ChatGPT进行微调可以实现对不安全内容生成的大幅减少。有关数据，请访问\url{https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs}.



## **12. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

 cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18104v1) [paper-pdf](http://arxiv.org/pdf/2402.18104v1)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and close-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 90\% attack success rate on LLM chatbots GPT-4.

摘要: 近年来，大型语言模型在各种任务上取得了显著的成功，但大型语言模型的可信度仍然是一个悬而未决的问题。一个具体的威胁是可能产生有毒或有害的反应。攻击者可以精心编制敌意提示，以诱导LLMS做出有害的响应。在这项工作中，我们通过识别安全微调中的偏差漏洞，开创了LLMS安全的理论基础，并设计了一种称为DRA(伪装和重建攻击)的黑盒越狱方法，该方法通过伪装来隐藏有害指令，并促使模型在其完成的范围内重建原始有害指令。我们通过各种开源和封闭源代码模型对DRA进行评估，展示最先进的越狱成功率和攻击效率。值得注意的是，DRA对LLM聊天机器人GPT-4的攻击成功率高达90%。



## **13. EmMark: Robust Watermarks for IP Protection of Embedded Quantized Large Language Models**

EmMark：用于嵌入式量化大语言模型知识产权保护的稳健水印 cs.CR

Accept to DAC 2024

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17938v1) [paper-pdf](http://arxiv.org/pdf/2402.17938v1)

**Authors**: Ruisi Zhang, Farinaz Koushanfar

**Abstract**: This paper introduces EmMark,a novel watermarking framework for protecting the intellectual property (IP) of embedded large language models deployed on resource-constrained edge devices. To address the IP theft risks posed by malicious end-users, EmMark enables proprietors to authenticate ownership by querying the watermarked model weights and matching the inserted signatures. EmMark's novelty lies in its strategic watermark weight parameters selection, nsuring robustness and maintaining model quality. Extensive proof-of-concept evaluations of models from OPT and LLaMA-2 families demonstrate EmMark's fidelity, achieving 100% success in watermark extraction with model performance preservation. EmMark also showcased its resilience against watermark removal and forging attacks.

摘要: 介绍了EmMark，一种新的数字水印框架，用于保护部署在资源受限边缘设备上的嵌入式大语言模型的知识产权。为了应对恶意最终用户带来的知识产权盗窃风险，EmMark使所有者能够通过查询带水印的模型权重并匹配插入的签名来验证所有权。EmMark的新颖之处在于其战略性的水印权重参数选择，确保了稳健性，并保持了模型质量。对OPT和Llama-2家族的模型进行了广泛的概念验证评估，证明了EmMark的保真度，在保持模型性能的情况下实现了100%的水印提取。EmMark还展示了其对水印移除和伪造攻击的弹性。



## **14. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

通过对抗性攻击生成LLM抵抗的数学应用题 cs.CL

Code is available at  https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17916v1) [paper-pdf](http://arxiv.org/pdf/2402.17916v1)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure to guide future research on LLM's mathematical capability.

摘要: 大型语言模型(LLM)极大地改变了教育格局。由于目前的抄袭检测工具难以跟上LLMS的快速进步，教育界面临着在LLMS存在的情况下评估学生真正的问题解决能力的挑战。在这项工作中，我们探索了一种确保公平评价的新范式--生成对抗性实例，它保留了用于评价的原始问题的结构和难度，但无法用LLMS解决。聚焦于数学应用题领域，我们利用抽象语法树来结构化地生成对抗性实例，这些实例通过简单地编辑问题中的数值来导致LLMS产生不正确的答案。我们在各种开源和闭源的LLM上进行了实验，定量和定性地证明了我们的方法显著降低了他们的数学问题解决能力。我们识别了LLM之间的共同漏洞，并提出了一种具有成本效益的方法来攻击高成本模型。此外，我们还对数学问题进行了自动分析，并调查了失败的原因，以指导未来对LLM数学能力的研究。



## **15. Mitigating Fine-tuning Jailbreak Attack with Backdoor Enhanced Alignment**

通过后门增强的对齐功能缓解精调越狱攻击 cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14968v2) [paper-pdf](http://arxiv.org/pdf/2402.14968v2)

**Authors**: Jiongxiao Wang, Jiazhao Li, Yiquan Li, Xiangyu Qi, Junjie Hu, Yixuan Li, Patrick McDaniel, Muhao Chen, Bo Li, Chaowei Xiao

**Abstract**: Despite the general capabilities of Large Language Models (LLMs) like GPT-4 and Llama-2, these models still request fine-tuning or adaptation with customized data when it comes to meeting the specific business demands and intricacies of tailored use cases. However, this process inevitably introduces new safety threats, particularly against the Fine-tuning based Jailbreak Attack (FJAttack), where incorporating just a few harmful examples into the fine-tuning dataset can significantly compromise the model safety. Though potential defenses have been proposed by incorporating safety examples into the fine-tuning dataset to reduce the safety issues, such approaches require incorporating a substantial amount of safety examples, making it inefficient. To effectively defend against the FJAttack with limited safety examples, we propose a Backdoor Enhanced Safety Alignment method inspired by an analogy with the concept of backdoor attacks. In particular, we construct prefixed safety examples by integrating a secret prompt, acting as a "backdoor trigger", that is prefixed to safety examples. Our comprehensive experiments demonstrate that through the Backdoor Enhanced Safety Alignment with adding as few as 11 prefixed safety examples, the maliciously fine-tuned LLMs will achieve similar safety performance as the original aligned models. Furthermore, we also explore the effectiveness of our method in a more practical setting where the fine-tuning data consists of both FJAttack examples and the fine-tuning task data. Our method shows great efficacy in defending against FJAttack without harming the performance of fine-tuning tasks.

摘要: 尽管GPT-4和LLAMA-2等大型语言模型(LLM)具有一般功能，但在满足特定业务需求和定制用例的复杂性时，这些模型仍然需要使用定制数据进行微调或调整。然而，这一过程不可避免地引入了新的安全威胁，特别是针对基于微调的越狱攻击(FJAttack)，在该攻击中，仅将几个有害的示例合并到微调数据集中可能会显著损害模型的安全性。虽然已经提出了通过将安全实例纳入微调数据集中来减少安全问题的潜在防御措施，但这种方法需要纳入大量的安全实例，从而使其效率低下。为了在安全示例有限的情况下有效防御FJAttack，我们提出了一种后门增强安全对齐方法，其灵感来自于后门攻击的概念。具体地说，我们通过集成一个秘密提示来构建前缀的安全实例，该提示充当安全实例的前缀的“后门触发器”。我们的综合实验表明，通过后门增强安全对齐，只需添加11个前缀安全实例，恶意微调的LLM将获得与原始对齐模型相似的安全性能。此外，我们还在一个更实际的环境中探索了我们的方法的有效性，其中微调数据包括FJAttack示例和微调任务数据。在不影响微调任务性能的情况下，我们的方法在防御FJAttack方面表现出了很好的效果。



## **16. Semantic Mirror Jailbreak: Genetic Algorithm Based Jailbreak Prompts Against Open-source LLMs**

语义镜像越狱：基于遗传算法的开源LLMS越狱提示 cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14872v2) [paper-pdf](http://arxiv.org/pdf/2402.14872v2)

**Authors**: Xiaoxia Li, Siyuan Liang, Jiyi Zhang, Han Fang, Aishan Liu, Ee-Chien Chang

**Abstract**: Large Language Models (LLMs), used in creative writing, code generation, and translation, generate text based on input sequences but are vulnerable to jailbreak attacks, where crafted prompts induce harmful outputs. Most jailbreak prompt methods use a combination of jailbreak templates followed by questions to ask to create jailbreak prompts. However, existing jailbreak prompt designs generally suffer from excessive semantic differences, resulting in an inability to resist defenses that use simple semantic metrics as thresholds. Jailbreak prompts are semantically more varied than the original questions used for queries. In this paper, we introduce a Semantic Mirror Jailbreak (SMJ) approach that bypasses LLMs by generating jailbreak prompts that are semantically similar to the original question. We model the search for jailbreak prompts that satisfy both semantic similarity and jailbreak validity as a multi-objective optimization problem and employ a standardized set of genetic algorithms for generating eligible prompts. Compared to the baseline AutoDAN-GA, SMJ achieves attack success rates (ASR) that are at most 35.4% higher without ONION defense and 85.2% higher with ONION defense. SMJ's better performance in all three semantic meaningfulness metrics of Jailbreak Prompt, Similarity, and Outlier, also means that SMJ is resistant to defenses that use those metrics as thresholds.

摘要: 用于创造性编写、代码生成和翻译的大型语言模型(LLM)根据输入序列生成文本，但容易受到越狱攻击，在这种攻击中，精心编制的提示会导致有害的输出。大多数越狱提示方法使用越狱模板和问题的组合来创建越狱提示。然而，现有的越狱提示设计通常存在过度的语义差异，导致无法抵抗使用简单语义度量作为阈值的防御。越狱提示在语义上比用于查询的原始问题更多样化。在本文中，我们介绍了一种语义镜像越狱(SMJ)方法，该方法通过生成与原始问题语义相似的越狱提示来绕过LLMS。我们将满足语义相似度和越狱有效性的越狱提示搜索问题建模为一个多目标优化问题，并使用一套标准化的遗传算法来生成合格的提示。与基线AutoDAN-GA相比，SMJ的攻击成功率(ASR)在没有洋葱防御的情况下最多提高了35.4%，在洋葱防御的情况下提高了85.2%。SMJ在越狱提示、相似度和离群值这三个语义意义指标上的表现都更好，这也意味着SMJ抵制使用这些指标作为阈值的防御。



## **17. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**

HarmBtch：一种标准化的自动化红队和稳健拒绝评估框架 cs.LG

Website: https://www.harmbench.org

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.04249v2) [paper-pdf](http://arxiv.org/pdf/2402.04249v2)

**Authors**: Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks

**Abstract**: Automated red teaming holds substantial promise for uncovering and mitigating the risks associated with the malicious use of large language models (LLMs), yet the field lacks a standardized evaluation framework to rigorously assess new methods. To address this issue, we introduce HarmBench, a standardized evaluation framework for automated red teaming. We identify several desirable properties previously unaccounted for in red teaming evaluations and systematically design HarmBench to meet these criteria. Using HarmBench, we conduct a large-scale comparison of 18 red teaming methods and 33 target LLMs and defenses, yielding novel insights. We also introduce a highly efficient adversarial training method that greatly enhances LLM robustness across a wide range of attacks, demonstrating how HarmBench enables codevelopment of attacks and defenses. We open source HarmBench at https://github.com/centerforaisafety/HarmBench.

摘要: 自动红色团队在发现和减轻与恶意使用大型语言模型(LLM)相关的风险方面有着很大的希望，但该领域缺乏标准的评估框架来严格评估新方法。为了解决这个问题，我们引入了HarmBtch，这是一个自动化红色团队的标准化评估框架。我们确定了几个以前在红队评估中没有考虑到的理想特性，并系统地设计了HarmBtch以满足这些标准。使用HarmBtch，我们对18种红队方法和33种目标LLM和防御进行了大规模比较，产生了新的见解。我们还引入了一种高效的对抗性训练方法，极大地增强了LLM在各种攻击中的健壮性，展示了HarmBtch如何实现攻击和防御的共同开发。我们在https://github.com/centerforaisafety/HarmBench.上开源了哈姆本奇



## **18. Pandora's White-Box: Increased Training Data Leakage in Open LLMs**

潘多拉的白盒：开放LLMS中训练数据泄露的增加 cs.CR

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.17012v1) [paper-pdf](http://arxiv.org/pdf/2402.17012v1)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we undertake a systematic study of privacy attacks against open source Large Language Models (LLMs), where an adversary has access to either the model weights, gradients, or losses, and tries to exploit them to learn something about the underlying training data. Our headline results are the first membership inference attacks (MIAs) against pre-trained LLMs that are able to simultaneously achieve high TPRs and low FPRs, and a pipeline showing that over $50\%$ (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, customization of the language model, and resources available to the attacker. In the pre-trained setting, we propose three new white-box MIAs: an attack based on the gradient norm, a supervised neural network classifier, and a single step loss ratio attack. All outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and other types of models. In fine-tuning, we find that given access to the loss of the fine-tuned and base models, a fine-tuned loss ratio attack FLoRA is able to achieve near perfect MIA peformance. We then leverage these MIAs to extract fine-tuning data from fine-tuned language models. We find that the pipeline of generating from fine-tuned models prompted with a small snippet of the prefix of each training example, followed by using FLoRa to select the most likely training sample, succeeds the majority of the fine-tuning dataset after only $3$ epochs of fine-tuning. Taken together, these findings show that highly effective MIAs are available in almost all LLM training settings, and highlight that great care must be taken before LLMs are fine-tuned on highly sensitive data and then deployed.

摘要: 在本文中，我们对针对开源大型语言模型(LLMS)的隐私攻击进行了系统的研究，其中攻击者可以访问模型的权重、梯度或损失，并试图利用它们来了解潜在的训练数据。我们的主要结果是针对预先训练的能够同时实现高TPR和低FPR的LLM的第一次成员推理攻击(MIA)，以及一个流水线显示超过50美元(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑对底层模型的不同程度的访问、语言模型的定制以及攻击者可用的资源。在预先训练的环境下，我们提出了三种新的白盒MIA：基于梯度范数的攻击、有监督的神经网络分类器和单步丢失率攻击。所有这些都超过了现有的黑盒基线，我们的监督攻击缩小了MIA对LLM的攻击成功与其他类型模型之间的差距。在微调中，我们发现，在获得微调和基本模型的损失的情况下，微调的损失率攻击菌群能够获得近乎完美的MIA性能。然后，我们利用这些MIA从微调的语言模型中提取微调数据。我们发现，在每个训练样本的前缀的一小段提示下，从微调模型生成的管道，然后使用FLORA来选择最可能的训练样本，在仅仅$3$的微调纪元之后，就成功了大部分微调数据集。综上所述，这些发现表明，高效的MIA在几乎所有LLM培训环境中都可用，并强调在对高度敏感的数据进行微调并随后部署LLM之前，必须非常小心。



## **19. WIPI: A New Web Threat for LLM-Driven Web Agents**

WIPI：LLM驱动的Web代理的一种新的Web威胁 cs.CR

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16965v1) [paper-pdf](http://arxiv.org/pdf/2402.16965v1)

**Authors**: Fangzhou Wu, Shutong Wu, Yulong Cao, Chaowei Xiao

**Abstract**: With the fast development of large language models (LLMs), LLM-driven Web Agents (Web Agents for short) have obtained tons of attention due to their superior capability where LLMs serve as the core part of making decisions like the human brain equipped with multiple web tools to actively interact with external deployed websites. As uncountable Web Agents have been released and such LLM systems are experiencing rapid development and drawing closer to widespread deployment in our daily lives, an essential and pressing question arises: "Are these Web Agents secure?". In this paper, we introduce a novel threat, WIPI, that indirectly controls Web Agent to execute malicious instructions embedded in publicly accessible webpages. To launch a successful WIPI works in a black-box environment. This methodology focuses on the form and content of indirect instructions within external webpages, enhancing the efficiency and stealthiness of the attack. To evaluate the effectiveness of the proposed methodology, we conducted extensive experiments using 7 plugin-based ChatGPT Web Agents, 8 Web GPTs, and 3 different open-source Web Agents. The results reveal that our methodology achieves an average attack success rate (ASR) exceeding 90% even in pure black-box scenarios. Moreover, through an ablation study examining various user prefix instructions, we demonstrated that the WIPI exhibits strong robustness, maintaining high performance across diverse prefix instructions.

摘要: 随着大型语言模型(LLM)的快速发展，LLM驱动的Web代理(简称Web代理)因其优越的能力而获得了大量的关注，其中LLM是决策的核心部分，就像人脑配备了多个Web工具来与外部部署的网站进行主动交互一样。随着无数的Web代理被发布，这样的LLM系统正在经历快速的发展，并接近于在我们的日常生活中广泛部署，一个基本而紧迫的问题出现了：“这些Web代理安全吗？”在本文中，我们介绍了一种新的威胁，WIPI，它间接地控制Web代理执行嵌入到可公开访问的网页中的恶意指令。要推出一款成功的Wipi，需要在黑盒环境中工作。这种方法侧重于外部网页中间接指令的形式和内容，提高了攻击的效率和隐蔽性。为了评估提出的方法的有效性，我们使用7个基于插件的ChatGPT Web代理、8个Web GPT和3个不同的开源Web代理进行了广泛的实验。结果表明，即使在纯黑盒场景下，该方法的平均攻击成功率(ASR)也超过90%。此外，通过对不同用户前缀指令的消融研究，我们证明了WIPI表现出很强的健壮性，在不同的前缀指令中保持了高性能。



## **20. CodeChameleon: Personalized Encryption Framework for Jailbreaking Large Language Models**

CodeChameleon：面向越狱大型语言模型的个性化加密框架 cs.CL

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16717v1) [paper-pdf](http://arxiv.org/pdf/2402.16717v1)

**Authors**: Huijie Lv, Xiao Wang, Yuansen Zhang, Caishuang Huang, Shihan Dou, Junjie Ye, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Adversarial misuse, particularly through `jailbreaking' that circumvents a model's safety and ethical protocols, poses a significant challenge for Large Language Models (LLMs). This paper delves into the mechanisms behind such successful attacks, introducing a hypothesis for the safety mechanism of aligned LLMs: intent security recognition followed by response generation. Grounded in this hypothesis, we propose CodeChameleon, a novel jailbreak framework based on personalized encryption tactics. To elude the intent security recognition phase, we reformulate tasks into a code completion format, enabling users to encrypt queries using personalized encryption functions. To guarantee response generation functionality, we embed a decryption function within the instructions, which allows the LLM to decrypt and execute the encrypted queries successfully. We conduct extensive experiments on 7 LLMs, achieving state-of-the-art average Attack Success Rate (ASR). Remarkably, our method achieves an 86.6\% ASR on GPT-4-1106.

摘要: 对抗性滥用，特别是通过“越狱”来规避模型的安全和道德协议，对大型语言模型（LLM）构成了重大挑战。本文深入研究了这种成功攻击背后的机制，介绍了对齐LLM的安全机制的假设：意图安全识别，然后生成响应。基于这一假设，我们提出了CodeChameleon，一种基于个性化加密策略的新型越狱框架。为了避开意图安全识别阶段，我们将任务重新制定为代码完成格式，使用户能够使用个性化的加密功能加密查询。为了保证响应生成功能，我们在指令中嵌入了一个解密函数，它允许LLM成功地解密和执行加密的查询。我们在7个LLM上进行了广泛的实验，达到了最先进的平均攻击成功率（ASR）。值得注意的是，我们的方法在GPT-4-1106上实现了86.6%的ASR。



## **21. RoCoIns: Enhancing Robustness of Large Language Models through Code-Style Instructions**

RoCoIns：通过代码风格指令增强大型语言模型的鲁棒性 cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16431v1) [paper-pdf](http://arxiv.org/pdf/2402.16431v1)

**Authors**: Yuansen Zhang, Xiao Wang, Zhiheng Xi, Han Xia, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Large Language Models (LLMs) have showcased remarkable capabilities in following human instructions. However, recent studies have raised concerns about the robustness of LLMs when prompted with instructions combining textual adversarial samples. In this paper, drawing inspiration from recent works that LLMs are sensitive to the design of the instructions, we utilize instructions in code style, which are more structural and less ambiguous, to replace typically natural language instructions. Through this conversion, we provide LLMs with more precise instructions and strengthen the robustness of LLMs. Moreover, under few-shot scenarios, we propose a novel method to compose in-context demonstrations using both clean and adversarial samples (\textit{adversarial context method}) to further boost the robustness of the LLMs. Experiments on eight robustness datasets show that our method consistently outperforms prompting LLMs with natural language instructions. For example, with gpt-3.5-turbo, our method achieves an improvement of 5.68\% in test set accuracy and a reduction of 5.66 points in Attack Success Rate (ASR).

摘要: 大型语言模型（LLM）在遵循人类指令方面表现出了非凡的能力。然而，最近的研究提出了对LLM的鲁棒性的担忧，当提示结合文本对抗样本的指令时。在本文中，从最近的作品，LLM是敏感的指令的设计的灵感，我们利用代码风格的指令，这是更结构化和更少的歧义，以取代典型的自然语言指令。通过这种转换，我们提供了更精确的指令LLM和加强LLM的鲁棒性。此外，在少量场景下，我们提出了一种新的方法来使用干净和对抗样本（\textit{adversarial context method}）来组成上下文演示，以进一步提高LLM的鲁棒性。八个鲁棒性数据集上的实验表明，我们的方法始终优于提示LLM与自然语言指令。例如，使用gpt-3.5-turbo，我们的方法在测试集准确率上提高了5.68%，在攻击成功率（ASR）上降低了5.66个点。



## **22. Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks**

诱使LLM不服从：形式化、分析和检测越狱 cs.CL

Accepted in LREC-COLING 2024 - The 2024 Joint International  Conference on Computational Linguistics, Language Resources and Evaluation

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2305.14965v2) [paper-pdf](http://arxiv.org/pdf/2305.14965v2)

**Authors**: Abhinav Rao, Sachin Vashistha, Atharva Naik, Somak Aditya, Monojit Choudhury

**Abstract**: Recent explorations with commercial Large Language Models (LLMs) have shown that non-expert users can jailbreak LLMs by simply manipulating their prompts; resulting in degenerate output behavior, privacy and security breaches, offensive outputs, and violations of content regulator policies. Limited studies have been conducted to formalize and analyze these attacks and their mitigations. We bridge this gap by proposing a formalism and a taxonomy of known (and possible) jailbreaks. We survey existing jailbreak methods and their effectiveness on open-source and commercial LLMs (such as GPT-based models, OPT, BLOOM, and FLAN-T5-XXL). We further discuss the challenges of jailbreak detection in terms of their effectiveness against known attacks. For our analysis, we collect a dataset of 3700 jailbreak prompts across 4 tasks. We will make the dataset public along with the model outputs.

摘要: 最近对商业大型语言模型(LLM)的探索表明，非专家用户可以通过简单地操作他们的提示来越狱LLM；导致退化的输出行为、隐私和安全漏洞、攻击性输出以及违反内容监管政策。对这些攻击及其缓解措施的正式化和分析进行了有限的研究。我们通过提出一种形式主义和已知(以及可能的)越狱分类来弥合这一差距。我们调查了现有的越狱方法及其在开源和商业LLM(如基于GPT的模型、OPT、Bloom和FRAN-T5-XXL)上的有效性。我们进一步讨论越狱检测在对抗已知攻击的有效性方面所面临的挑战。为了进行分析，我们收集了4个任务中3700个越狱提示的数据集。我们将把数据集与模型输出一起公开。



## **23. Immunization against harmful fine-tuning attacks**

针对有害微调攻击的免疫接种 cs.CL

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16382v1) [paper-pdf](http://arxiv.org/pdf/2402.16382v1)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, Jan Batzner, Hassan Sajjad, Frank Rudzicz

**Abstract**: Approaches to aligning large language models (LLMs) with human values has focused on correcting misalignment that emerges from pretraining. However, this focus overlooks another source of misalignment: bad actors might purposely fine-tune LLMs to achieve harmful goals. In this paper, we present an emerging threat model that has arisen from alignment circumvention and fine-tuning attacks. However, lacking in previous works is a clear presentation of the conditions for effective defence. We propose a set of conditions for effective defence against harmful fine-tuning in LLMs called "Immunization conditions," which help us understand how we would construct and measure future defences. Using this formal framework for defence, we offer a synthesis of different research directions that might be persued to prevent harmful fine-tuning attacks and provide a demonstration of how to use these conditions experimentally showing early results of using an adversarial loss to immunize LLama2-7b-chat.

摘要: 使大型语言模型(LLM)与人类价值观保持一致的方法一直集中在纠正预培训中出现的错位上。然而，这种关注忽略了另一个不协调的来源：糟糕的行为者可能会故意微调LLM，以实现有害的目标。在本文中，我们提出了一种新兴的威胁模型，该模型源于对齐规避和微调攻击。然而，在以往的著作中，缺乏对有效辩护的条件的明确呈现。我们提出了一套有效防御LLMS中有害微调的条件，称为“免疫条件”，这有助于我们了解如何构建和测量未来的防御。使用这一正式的防御框架，我们提供了可能被说服以防止有害的微调攻击的不同研究方向的综合，并提供了如何使用这些条件的演示，实验显示了使用对抗性损失来免疫LLama2-7b-Chat的早期结果。



## **24. Privacy-Preserved Neural Graph Databases**

隐私保护的神经图数据库 cs.DB

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2312.15591v4) [paper-pdf](http://arxiv.org/pdf/2312.15591v4)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.

摘要: 在大型语言模型(LLMS)时代，高效和准确的数据检索对于在检索增强生成(RAG)中使用特定领域或私有数据变得越来越重要。神经图形数据库(NGDB)已经成为一种强大的范例，它结合了图形数据库(GDB)和神经网络的优点，能够有效地存储、检索和分析图结构的数据，这些数据可以用LLMS进行自适应训练。神经嵌入存储和复杂神经逻辑查询应答(CQA)的使用为NGDB提供了泛化能力。当图不完整时，通过提取潜在模式和表示，神经图库可以填补图结构中的空白，揭示隐藏的关系，并使查询得到准确的回答。然而，这种能力是有内在权衡的，因为它会给特定于域或私有的数据库带来额外的隐私风险。恶意攻击者可以使用精心设计的查询来推断数据库中更敏感的信息，例如从图灵奖获得者1950年前和1940年后出生的地方的答案集中，图灵奖获得者辛顿的居住地可能会被曝光，尽管出于隐私考虑，居住地可能在培训阶段已被删除。在这项工作中，我们提出了一个隐私保护的神经图库(P-NGDB)框架，以缓解NGDB中隐私泄露的风险。在训练阶段引入对抗性训练技术，强制NGDB在查询私有信息时产生不可区分的答案，增加了通过组合多个无害查询来推断敏感信息的难度。



## **25. LuaTaint: A Static Taint Analysis System for Web Interface Framework Vulnerability of IoT Devices**

LuaTaint：物联网设备Web接口框架漏洞静态污点分析系统 cs.CR

**SubmitDate**: 2024-02-25    [abs](http://arxiv.org/abs/2402.16043v1) [paper-pdf](http://arxiv.org/pdf/2402.16043v1)

**Authors**: Jiahui Xiang, Wenhai Wang, Tong Ye, Peiyu Liu

**Abstract**: IoT devices are currently facing continuous malicious attacks due to their widespread use. Among these IoT devices, web vulnerabilities are also widely exploited because of their inherent characteristics, such as improper permission controls and insecure interfaces. Recently, the embedded system web interface framework has become highly diverse, and specific vulnerabilities can arise if developers forget to detect user input parameters or if the detection process is not strict enough. Therefore, discovering vulnerabilities in the web interfaces of IoT devices accurately and comprehensively through an automated method is a major challenge. This paper aims to work out the challenge. We have developed an automated vulnerability detection system called LuaTaint for the typical web interface framework, LuCI. The system employs static taint analysis to address web security issues on mobile terminal platforms to ensure detection coverage. It integrates rules pertaining to page handler control logic within the taint detection process to improve its extensibility. We also implemented a post-processing step with the assistance of large language models to enhance accuracy and reduce the need for manual analysis. We have created a prototype of LuaTaint and tested it on 92 IoT firmwares from 8 well-known vendors. LuaTaint has discovered 68 unknown vulnerabilities.

摘要: 由于物联网设备的广泛使用，目前正面临着持续的恶意攻击。在这些物联网设备中，网络漏洞也因其固有的特性而被广泛利用，例如不正确的权限控制和不安全的接口。最近，嵌入式系统Web界面框架变得高度多样化，如果开发人员忘记检测用户输入参数或检测过程不够严格，就可能出现特定的漏洞。因此，通过自动化方法准确、全面地发现物联网设备网络界面中的漏洞是一大挑战。本文旨在解决这一挑战。我们为典型的Web界面框架Luci开发了一个自动化的漏洞检测系统LuaTaint。该系统使用静态污点分析来解决移动终端平台上的网络安全问题，以确保检测覆盖。它将与页面处理程序控制逻辑相关的规则集成到污点检测过程中，以提高其可扩展性。我们还在大型语言模型的协助下实施了后处理步骤，以提高准确性并减少对人工分析的需要。我们已经创建了LuaTaint的原型，并在来自8家知名供应商的92个物联网固件上进行了测试。LuaTaint已经发现了68个未知漏洞。



## **26. From Noise to Clarity: Unraveling the Adversarial Suffix of Large Language Model Attacks via Translation of Text Embeddings**

从噪音到清晰度：通过文本嵌入的翻译解开大型语言模型攻击的对抗性后缀 cs.CL

**SubmitDate**: 2024-02-25    [abs](http://arxiv.org/abs/2402.16006v1) [paper-pdf](http://arxiv.org/pdf/2402.16006v1)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. This method, while effective, leaves a gap in understanding the underlying mechanics of such adversarial suffix due to the non-readability and it can be relatively easily seen through by common defense methods such as perplexity filters.To cope with this challenge, in this paper, we propose an Adversarial Suffixes Embedding Translation Framework(ASETF) that are able to translate the unreadable adversarial suffixes into coherent, readable text, which makes it easier to understand and analyze the reasons behind harmful content generation by large language models. We conducted experiments on LLMs such as LLaMa2, Vicuna and using the Advbench dataset's harmful instructions. The results indicate that our method achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini. As a result, the prompts generated through our method exhibit enriched semantic diversity, which potentially provides more adversarial examples for LLM defense methods.

摘要: 大型语言模型(LLM)的安全防御方法仍然有限，因为危险的提示是手动管理到少数已知的攻击类型，无法跟上新兴的变体。最近的研究发现，在有害指令上附加后缀可能会破坏LLMS的防御，并导致危险的输出。针对这一挑战，本文提出了一种对抗性后缀嵌入翻译框架(ASETF)，该框架能够将不可读的对抗性后缀翻译成连贯、可读的文本，从而更容易理解和分析大型语言模型产生有害内容的原因。我们在LLaMa2、Vicuna等LLMS上进行了实验，并使用Advbench数据集的有害指令进行了实验。实验结果表明，与现有技术相比，该方法具有更高的攻击成功率，同时显著提高了提示的文本流畅性。此外，我们的方法可以推广到更广泛的方法来生成可转移的敌意后缀，可以成功地攻击多个LLM，甚至可以攻击黑盒LLM，如ChatGPT和Gemini。因此，通过我们的方法生成的提示显示了丰富的语义多样性，这可能为LLM防御方法提供更具对抗性的例子。



## **27. Safety of Multimodal Large Language Models on Images and Text**

基于图像和文本的多模态大型语言模型的安全性 cs.CV

**SubmitDate**: 2024-02-25    [abs](http://arxiv.org/abs/2402.00357v2) [paper-pdf](http://arxiv.org/pdf/2402.00357v2)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions. The latest papers are continually collected at https://github.com/isXinLiu/MLLM-Safety-Collection.

摘要: 被多模式大型语言模型(MLLMS)令人印象深刻的力量所吸引，公众越来越多地利用它们来提高日常工作效率。然而，当这些模型部署在现实世界的场景中时，MLLMS对不安全指令的脆弱性带来了巨大的安全风险。本文系统地综述了当前对MLLMS图像和文本安全性的评估、攻击和防御的研究进展。我们首先介绍MLLMS关于图像和文本的概述以及对安全性的理解，这有助于研究人员了解我们调查的详细范围。然后，我们回顾了用于衡量MLLMS安全性的评价数据集和度量。接下来，我们全面介绍了与MLLMS安全相关的攻防技术。最后，我们分析了一些尚未解决的问题，并讨论了未来的研究方向。最新的论文不断在https://github.com/isXinLiu/MLLM-Safety-Collection.上收集



## **28. PRP: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails**

PRP：传播普适扰动攻击大型语言模型Guard-Rails cs.CR

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.15911v1) [paper-pdf](http://arxiv.org/pdf/2402.15911v1)

**Authors**: Neal Mangaokar, Ashish Hooda, Jihye Choi, Shreyas Chandrashekaran, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Large language models (LLMs) are typically aligned to be harmless to humans. Unfortunately, recent work has shown that such models are susceptible to automated jailbreak attacks that induce them to generate harmful content. More recent LLMs often incorporate an additional layer of defense, a Guard Model, which is a second LLM that is designed to check and moderate the output response of the primary LLM. Our key contribution is to show a novel attack strategy, PRP, that is successful against several open-source (e.g., Llama 2) and closed-source (e.g., GPT 3.5) implementations of Guard Models. PRP leverages a two step prefix-based attack that operates by (a) constructing a universal adversarial prefix for the Guard Model, and (b) propagating this prefix to the response. We find that this procedure is effective across multiple threat models, including ones in which the adversary has no access to the Guard Model at all. Our work suggests that further advances are required on defenses and Guard Models before they can be considered effective.

摘要: 大型语言模型(LLM)通常被调整为对人类无害。不幸的是，最近的研究表明，这类模型容易受到自动越狱攻击，从而导致它们生成有害内容。最近的LLM通常包含一个额外的防御层，即警卫模型，这是第二个LLM，旨在检查和调节主要LLM的输出响应。我们的主要贡献是展示了一种新的攻击策略，PRP，它成功地对抗了几种开源(例如，Llama 2)和封闭源代码(例如，GPT 3.5)的Guard模型实现。PRP利用基于两步前缀的攻击，该攻击通过(A)为警卫模型构建通用对抗性前缀，以及(B)将该前缀传播到响应来操作。我们发现，此过程对多个威胁模型有效，包括对手根本无法访问警卫模型的威胁模型。我们的工作表明，在防御和警卫模型被认为有效之前，还需要进一步的进步。



## **29. On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities**

在机器人中部署LLMS/VLM的安全问题：突出风险和漏洞 cs.RO

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.10340v3) [paper-pdf](http://arxiv.org/pdf/2402.10340v3)

**Authors**: Xiyang Wu, Ruiqi Xian, Tianrui Guan, Jing Liang, Souradip Chakraborty, Fuxiao Liu, Brian Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these attacks. Our empirical findings reveal a striking vulnerability of LLM/VLM-robot integrated systems: simple adversarial attacks can significantly undermine the effectiveness of LLM/VLM-robot integrated systems. Specifically, our data demonstrate an average performance deterioration of 21.2% under prompt attacks and a more alarming 30.2% under perception attacks. These results underscore the critical need for robust countermeasures to ensure the safe and reliable deployment of the advanced LLM/VLM-based robotic systems.

摘要: 在本文中，我们强调了与将大型语言模型（LLM）和视觉语言模型（VLM）集成到机器人应用程序中相关的鲁棒性和安全性的关键问题。最近的工作主要集中在使用LLM和VLM来提高机器人任务的性能，例如操纵，导航等，但是，这种集成可能会引入重大漏洞，因为语言模型容易受到对抗性攻击，可能导致灾难性后果。通过检查最近的工作在LLM/VLM和机器人的接口，我们表明，它是很容易操纵或误导机器人的行动，导致安全隐患。我们定义并提供了几种看似合理的对抗性攻击的例子，并在三个与语言模型集成的突出机器人框架上进行实验，包括KnowNo VIMA和Instruct 2Act，以评估它们对这些攻击的敏感性。我们的实证研究结果揭示了一个惊人的漏洞LLM/VLM-robot集成系统：简单的对抗性攻击可以显着破坏LLM/VLM-robot集成系统的有效性。具体来说，我们的数据表明，在即时攻击下平均性能下降21.2%，在感知攻击下更令人担忧的是30.2%。这些结果强调了对强有力的对策的迫切需要，以确保先进的LLM/VLM机器人系统的安全和可靠的部署。



## **30. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

安全解码：通过安全感知解码防御越狱攻击 cs.CR

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.08983v2) [paper-pdf](http://arxiv.org/pdf/2402.08983v2)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.

摘要: 随着大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，如代码生成和聊天机器人辅助，人们已经做出了广泛的努力，使LLM行为与人类价值观保持一致，包括安全性。越狱攻击，旨在挑起LLM的意外和不安全的行为，仍然是一个重大/领先的LLM安全威胁。在本文中，我们的目标是通过引入SafeDecoding，一个安全意识的解码策略，LLM生成有用的和无害的响应用户查询，以抵御越狱攻击的LLM。我们在开发SafeDecoding时的见解是基于这样的观察：即使表示有害内容的令牌的概率大于表示无害响应的令牌的概率，安全声明仍然出现在按概率降序排列令牌之后的顶部令牌中。这使我们能够通过识别安全声明并放大其令牌概率来减轻越狱攻击，同时衰减与越狱攻击目标一致的令牌序列的概率。我们使用六种最先进的越狱攻击和四个基准数据集对五个LLM进行了广泛的实验。我们的研究结果表明，SafeDecoding显着降低了攻击的成功率和越狱攻击的危害性，而不影响良性用户查询的响应的有用性。SafeDecoding优于六种防御方法。



## **31. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

LLMS能够以实际的方式保护自己免受越狱：一份愿景文件 cs.CR

This is a vision paper on defending against LLM jailbreaks

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.15727v1) [paper-pdf](http://arxiv.org/pdf/2402.15727v1)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐。已有大量研究提出了更有效的越狱攻击方案，包括最近的贪婪坐标梯度(GCG)攻击、基于模板的越狱攻击(例如使用“Do-Anything-Now”(DAN))和多语言越狱。相比之下，防守方面的探索相对较少。本文提出了一种轻量级而实用的防御方法SELFDEFEND，它可以防御所有现有的越狱攻击，而越狱提示的延迟最小，正常用户提示的延迟可以忽略不计。我们的主要见解是，无论采用哪种越狱策略，他们最终都需要在发送给LLMS的提示中包含有害提示(例如，如何制造炸弹)，我们发现现有LLMS可以有效地识别此类违反其安全政策的有害提示。基于这一观点，我们设计了一个影子堆栈，该堆栈同时检查用户提示中是否存在有害提示，并在输出令牌“否”或有害提示时触发正常堆栈中的检查点。后者还可以对对抗性提示产生可解释的LLM响应。我们通过GPT-3.5/4中的手动分析，展示了我们的SELFDEFEND在各种越狱场景中的工作原理。我们还列出了进一步增强SELFDEFEND的三个未来方向。



## **32. Foot In The Door: Understanding Large Language Model Jailbreaking via Cognitive Psychology**

入门：通过认知心理学理解大型语言模型越狱 cs.CL

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.15690v1) [paper-pdf](http://arxiv.org/pdf/2402.15690v1)

**Authors**: Zhenhua Wang, Wei Xie, Baosheng Wang, Enze Wang, Zhiwen Gui, Shuoyoucheng Ma, Kai Chen

**Abstract**: Large Language Models (LLMs) have gradually become the gateway for people to acquire new knowledge. However, attackers can break the model's security protection ("jail") to access restricted information, which is called "jailbreaking." Previous studies have shown the weakness of current LLMs when confronted with such jailbreaking attacks. Nevertheless, comprehension of the intrinsic decision-making mechanism within the LLMs upon receipt of jailbreak prompts is noticeably lacking. Our research provides a psychological explanation of the jailbreak prompts. Drawing on cognitive consistency theory, we argue that the key to jailbreak is guiding the LLM to achieve cognitive coordination in an erroneous direction. Further, we propose an automatic black-box jailbreaking method based on the Foot-in-the-Door (FITD) technique. This method progressively induces the model to answer harmful questions via multi-step incremental prompts. We instantiated a prototype system to evaluate the jailbreaking effectiveness on 8 advanced LLMs, yielding an average success rate of 83.9%. This study builds a psychological perspective on the explanatory insights into the intrinsic decision-making logic of LLMs.

摘要: 大语言模型逐渐成为人们获取新知识的门户。然而，攻击者可以突破模型的安全保护(“监狱”)来访问受限信息，这被称为“越狱”。之前的研究已经表明，当前的LLM在面对这样的越狱攻击时存在弱点。然而，对LLMS在收到越狱提示时的内在决策机制的理解显然是缺乏的。我们的研究提供了越狱提示的心理学解释。借鉴认知一致性理论，我们认为越狱的关键是引导LLM在错误的方向上实现认知协调。在此基础上，提出了一种基于FITD(Foot-in-the-Door)技术的自动黑盒越狱方法。该方法通过多步增量提示逐步诱导模型回答有害问题。我们实例化了一个原型系统，对8个先进的LLMS进行了越狱效果评估，平均成功率为83.9%。这项研究建立了一个心理学的视角来解释LLMS的内在决策逻辑。



## **33. User Inference Attacks on Large Language Models**

针对大型语言模型的用户推理攻击 cs.CR

v2 contains experiments on additional datasets and differential  privacy

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2310.09266v2) [paper-pdf](http://arxiv.org/pdf/2310.09266v2)

**Authors**: Nikhil Kandpal, Krishna Pillutla, Alina Oprea, Peter Kairouz, Christopher A. Choquette-Choo, Zheng Xu

**Abstract**: Fine-tuning is a common and effective method for tailoring large language models (LLMs) to specialized tasks and applications. In this paper, we study the privacy implications of fine-tuning LLMs on user data. To this end, we consider a realistic threat model, called user inference, wherein an attacker infers whether or not a user's data was used for fine-tuning. We design attacks for performing user inference that require only black-box access to the fine-tuned LLM and a few samples from a user which need not be from the fine-tuning dataset. We find that LLMs are susceptible to user inference across a variety of fine-tuning datasets, at times with near perfect attack success rates. Further, we theoretically and empirically investigate the properties that make users vulnerable to user inference, finding that outlier users, users with identifiable shared features between examples, and users that contribute a large fraction of the fine-tuning data are most susceptible to attack. Based on these findings, we identify several methods for mitigating user inference including training with example-level differential privacy, removing within-user duplicate examples, and reducing a user's contribution to the training data. While these techniques provide partial mitigation of user inference, we highlight the need to develop methods to fully protect fine-tuned LLMs against this privacy risk.

摘要: 微调是为专门的任务和应用程序定制大型语言模型(LLM)的一种常见且有效的方法。在本文中，我们研究了微调LLMS对用户数据的隐私影响。为此，我们考虑一个现实的威胁模型，称为用户推理，其中攻击者推断用户的数据是否被用于微调。我们设计了用于执行用户推理的攻击，这些攻击只需要对微调的LLM和不需要来自微调数据集的用户的一些样本进行黑盒访问。我们发现，LLM在各种微调的数据集上都很容易受到用户推理的影响，有时攻击成功率近乎完美。此外，我们从理论和经验上研究了使用户容易受到用户推理影响的属性，发现离群点用户、实例之间具有可识别的共享特征的用户以及贡献了很大一部分微调数据的用户最容易受到攻击。基于这些发现，我们确定了几种减轻用户推理的方法，包括使用示例级差异隐私进行训练，删除用户内部的重复示例，以及减少用户对训练数据的贡献。虽然这些技术部分缓解了用户干扰，但我们强调需要开发方法来完全保护微调的LLM免受这种隐私风险的影响。



## **34. The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)**

好与坏：探索检索-增强代(RAG)中的隐私问题 cs.CR

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2402.16893v1) [paper-pdf](http://arxiv.org/pdf/2402.16893v1)

**Authors**: Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren, Shuaiqiang Wang, Dawei Yin, Yi Chang, Jiliang Tang

**Abstract**: Retrieval-augmented generation (RAG) is a powerful technique to facilitate language model with proprietary and private data, where data privacy is a pivotal concern. Whereas extensive research has demonstrated the privacy risks of large language models (LLMs), the RAG technique could potentially reshape the inherent behaviors of LLM generation, posing new privacy issues that are currently under-explored. In this work, we conduct extensive empirical studies with novel attack methods, which demonstrate the vulnerability of RAG systems on leaking the private retrieval database. Despite the new risk brought by RAG on the retrieval data, we further reveal that RAG can mitigate the leakage of the LLMs' training data. Overall, we provide new insights in this paper for privacy protection of retrieval-augmented LLMs, which benefit both LLMs and RAG systems builders. Our code is available at https://github.com/phycholosogy/RAG-privacy.

摘要: 检索增强生成（RAG）是一种功能强大的技术，以促进语言模型的专有和私人的数据，其中数据隐私是一个关键的关注。虽然广泛的研究已经证明了大型语言模型（LLM）的隐私风险，但RAG技术可能会重塑LLM生成的固有行为，从而提出目前尚未充分探索的新隐私问题。在这项工作中，我们进行了广泛的实证研究与新的攻击方法，这表明RAG系统的脆弱性泄漏的私人检索数据库。尽管RAG对检索数据带来了新的风险，但我们进一步揭示了RAG可以减轻LLM训练数据的泄漏。总的来说，我们在本文中为检索增强LLM的隐私保护提供了新的见解，这对LLM和RAG系统构建者都有好处。我们的代码可以在https://github.com/phycholosogy/RAG-privacy上找到。



## **35. Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak**

分析LLMS的内在响应趋势：现实世界指令驱动的越狱 cs.CL

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2312.04127v2) [paper-pdf](http://arxiv.org/pdf/2312.04127v2)

**Authors**: Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin

**Abstract**: Extensive work has been devoted to improving the safety mechanism of Large Language Models (LLMs). However, LLMs still tend to generate harmful responses when faced with malicious instructions, a phenomenon referred to as "Jailbreak Attack". In our research, we introduce a novel automatic jailbreak method RADIAL, which bypasses the security mechanism by amplifying the potential of LLMs to generate affirmation responses. The jailbreak idea of our method is "Inherent Response Tendency Analysis" which identifies real-world instructions that can inherently induce LLMs to generate affirmation responses and the corresponding jailbreak strategy is "Real-World Instructions-Driven Jailbreak" which involves strategically splicing real-world instructions identified through the above analysis around the malicious instruction. Our method achieves excellent attack performance on English malicious instructions with five open-source advanced LLMs while maintaining robust attack performance in executing cross-language attacks against Chinese malicious instructions. We conduct experiments to verify the effectiveness of our jailbreak idea and the rationality of our jailbreak strategy design. Notably, our method designed a semantically coherent attack prompt, highlighting the potential risks of LLMs. Our study provides detailed insights into jailbreak attacks, establishing a foundation for the development of safer LLMs.

摘要: 人们在改进大型语言模型(LLM)的安全机制方面做了大量的工作。然而，当面临恶意指令时，LLMS仍然倾向于产生有害的响应，这种现象被称为“越狱攻击”。在我们的研究中，我们引入了一种新的自动越狱方法RADIUS，它通过放大LLMS产生肯定响应的潜力来绕过安全机制。该方法的越狱思想是“内在响应趋势分析”，它识别可以内在地诱导LLM生成肯定响应的真实指令，而相应的越狱策略是“真实指令驱动越狱”，它涉及到围绕恶意指令策略性地拼接通过上述分析识别的真实指令。该方法使用5个开源的高级LLM实现了对英文恶意指令的良好攻击性能，同时保持了对中文恶意指令的跨语言攻击的健壮性。通过实验验证了我们越狱思想的有效性和越狱策略设计的合理性。值得注意的是，我们的方法设计了一个语义连贯的攻击提示，突出了LLMS的潜在风险。我们的研究提供了对越狱攻击的详细见解，为开发更安全的LLM奠定了基础。



## **36. A First Look at GPT Apps: Landscape and Vulnerability**

GPT应用程序初见端倪：格局和漏洞 cs.CR

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2402.15105v1) [paper-pdf](http://arxiv.org/pdf/2402.15105v1)

**Authors**: Zejun Zhang, Li Zhang, Xin Yuan, Anlan Zhang, Mengwei Xu, Feng Qian

**Abstract**: With the advancement of Large Language Models (LLMs), increasingly sophisticated and powerful GPTs are entering the market. Despite their popularity, the LLM ecosystem still remains unexplored. Additionally, LLMs' susceptibility to attacks raises concerns over safety and plagiarism. Thus, in this work, we conduct a pioneering exploration of GPT stores, aiming to study vulnerabilities and plagiarism within GPT applications. To begin with, we conduct, to our knowledge, the first large-scale monitoring and analysis of two stores, an unofficial GPTStore.AI, and an official OpenAI GPT Store. Then, we propose a TriLevel GPT Reversing (T-GR) strategy for extracting GPT internals. To complete these two tasks efficiently, we develop two automated tools: one for web scraping and another designed for programmatically interacting with GPTs. Our findings reveal a significant enthusiasm among users and developers for GPT interaction and creation, as evidenced by the rapid increase in GPTs and their creators. However, we also uncover a widespread failure to protect GPT internals, with nearly 90% of system prompts easily accessible, leading to considerable plagiarism and duplication among GPTs.

摘要: 随着大型语言模型（LLM）的发展，越来越复杂和强大的GPT正在进入市场。尽管它们很受欢迎，LLM生态系统仍然未被探索。此外，LLM对攻击的敏感性引发了对安全和剽窃的担忧。因此，在这项工作中，我们对GPT商店进行了开创性的探索，旨在研究GPT应用程序中的漏洞和剽窃。首先，据我们所知，我们对两个商店进行了首次大规模的监控和分析，一个是非官方的GPTStore.AI，另一个是官方的OpenAI GPT Store。然后，我们提出了一个三层GPT反转（T-GR）的策略提取GPT内部。为了有效地完成这两项任务，我们开发了两个自动化工具：一个用于Web抓取，另一个用于以编程方式与GPT交互。我们的研究结果揭示了用户和开发人员对GPT交互和创建的极大热情，GPT及其创建者的快速增长就是明证。然而，我们也发现了一个广泛的失败，以保护GPT内部，近90%的系统提示很容易访问，导致相当大的剽窃和重复之间的GPT。



## **37. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

ArtPrompt：基于ASCII ART的针对结盟LLM的越狱攻击 cs.CL

**SubmitDate**: 2024-02-22    [abs](http://arxiv.org/abs/2402.11753v2) [paper-pdf](http://arxiv.org/pdf/2402.11753v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs.

摘要: 安全对于大型语言模型(LLM)的使用至关重要。已经开发了多种技术，如数据过滤和有监督的微调，以加强LLM的安全性。然而，目前已知的技术假定用于LLM的安全对准的语料库仅由语义解释。然而，这一假设在现实世界的应用程序中并不成立，这导致了LLMS中的严重漏洞。例如，论坛的用户经常使用ASCII艺术，这是一种基于文本的艺术形式，以传达图像信息。本文提出了一种新的基于ASCII ART的越狱攻击方法，并引入了一个综合基准的文本中视觉挑战(VITC)来评估LLMS在识别不能完全由语义解释的提示方面的能力。我们发现，五个SOTA LLM(GPT-3.5、GPT-4、双子座、克劳德和Llama2)很难识别以ASCII ART形式提供的提示。基于这种观察，我们开发了越狱攻击ArtPrompt，它利用LLMS在识别ASCII ART方面的较差性能来绕过安全措施，并从LLM引发不希望看到的行为。ArtPrompt只需要黑盒访问受攻击的LLM，这使其成为一种实际的攻击。我们在五个SOTA LLM上对ArtPrompt进行了评估，结果表明，ArtPrompt可以有效和高效地诱导所有五个LLM的不良行为。



## **38. Coercing LLMs to do and reveal (almost) anything**

强迫LLM做和透露(几乎)任何事情 cs.LG

32 pages. Implementation available at  https://github.com/JonasGeiping/carving

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.14020v1) [paper-pdf](http://arxiv.org/pdf/2402.14020v1)

**Authors**: Jonas Geiping, Alex Stein, Manli Shu, Khalid Saifullah, Yuxin Wen, Tom Goldstein

**Abstract**: It has recently been shown that adversarial attacks on large language models (LLMs) can "jailbreak" the model into making harmful statements. In this work, we argue that the spectrum of adversarial attacks on LLMs is much larger than merely jailbreaking. We provide a broad overview of possible attack surfaces and attack goals. Based on a series of concrete examples, we discuss, categorize and systematize attacks that coerce varied unintended behaviors, such as misdirection, model control, denial-of-service, or data extraction.   We analyze these attacks in controlled experiments, and find that many of them stem from the practice of pre-training LLMs with coding capabilities, as well as the continued existence of strange "glitch" tokens in common LLM vocabularies that should be removed for security reasons.

摘要: 最近有研究表明，针对大型语言模型(LLM)的对抗性攻击可以将模型“越狱”，使其做出有害的声明。在这项工作中，我们认为针对LLMS的对抗性攻击的范围比仅仅越狱要大得多。我们提供了可能的攻击面和攻击目标的广泛概述。基于一系列具体的例子，我们对强迫各种意外行为的攻击进行了讨论、分类和系统化，例如误导、模型控制、拒绝服务或数据提取。我们在受控实验中对这些攻击进行了分析，发现其中许多攻击源于预先训练具有编码能力的LLM的做法，以及常见LLM词汇中持续存在出于安全原因应删除的奇怪“毛刺”标记。



## **39. Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment**

LLM-as-a-Justice稳健吗？基于零射LLM评估的通用对抗性攻击研究 cs.CL

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.14016v1) [paper-pdf](http://arxiv.org/pdf/2402.14016v1)

**Authors**: Vyas Raina, Adian Liusie, Mark Gales

**Abstract**: Large Language Models (LLMs) are powerful zero-shot assessors and are increasingly used in real-world situations such as for written exams or benchmarking systems. Despite this, no existing work has analyzed the vulnerability of judge-LLMs against adversaries attempting to manipulate outputs. This work presents the first study on the adversarial robustness of assessment LLMs, where we search for short universal phrases that when appended to texts can deceive LLMs to provide high assessment scores. Experiments on SummEval and TopicalChat demonstrate that both LLM-scoring and pairwise LLM-comparative assessment are vulnerable to simple concatenation attacks, where in particular LLM-scoring is very susceptible and can yield maximum assessment scores irrespective of the input text quality. Interestingly, such attacks are transferable and phrases learned on smaller open-source LLMs can be applied to larger closed-source models, such as GPT3.5. This highlights the pervasive nature of the adversarial vulnerabilities across different judge-LLM sizes, families and methods. Our findings raise significant concerns on the reliability of LLMs-as-a-judge methods, and underscore the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.

摘要: 大型语言模型(LLM)是强大的零分评价器，越来越多地用于真实世界的情况，如笔试或基准系统。尽管如此，现有的工作还没有分析JUSTER-LLMS针对试图操纵输出的对手的脆弱性。这项工作首次研究了评估LLMS的对抗稳健性，我们寻找简短的通用短语，当添加到文本中时，可以欺骗LLMS提供高的评估分数。在SummEval和TopicalChat上的实验表明，LLM评分和成对LLM比较评估都容易受到简单串联攻击，其中LLM评分非常敏感，无论输入文本质量如何，都可以产生最高评估分数。有趣的是，这种攻击是可以转移的，在较小的开源LLM上学习的短语可以应用于较大的闭源模型，如GPT3.5。这突出了敌对性漏洞在不同的JASTER-LLM规模、家族和方法中的普遍性质。我们的发现引起了人们对LLMS作为判断方法的可靠性的严重关注，并强调了在高风险的现实世界场景中部署之前解决LLM评估方法中的漏洞的重要性。



## **40. Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models**

水印能在翻译中存活下来吗？大语言模型下文本水印的跨语言一致性研究 cs.CL

Under review

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.14007v1) [paper-pdf](http://arxiv.org/pdf/2402.14007v1)

**Authors**: Zhiwei He, Binglin Zhou, Hongkun Hao, Aiwei Liu, Xing Wang, Zhaopeng Tu, Zhuosheng Zhang, Rui Wang

**Abstract**: Text watermarking technology aims to tag and identify content produced by large language models (LLMs) to prevent misuse. In this study, we introduce the concept of ''cross-lingual consistency'' in text watermarking, which assesses the ability of text watermarks to maintain their effectiveness after being translated into other languages. Preliminary empirical results from two LLMs and three watermarking methods reveal that current text watermarking technologies lack consistency when texts are translated into various languages. Based on this observation, we propose a Cross-lingual Watermark Removal Attack (CWRA) to bypass watermarking by first obtaining a response from an LLM in a pivot language, which is then translated into the target language. CWRA can effectively remove watermarks by reducing the Area Under the Curve (AUC) from 0.95 to 0.67 without performance loss. Furthermore, we analyze two key factors that contribute to the cross-lingual consistency in text watermarking and propose a defense method that increases the AUC from 0.67 to 0.88 under CWRA.

摘要: 文本水印技术旨在标记和识别大型语言模型(LLM)产生的内容，以防止误用。在这项研究中，我们在文本水印中引入了“跨语言一致性”的概念，用来评估文本水印在被翻译成其他语言后保持其有效性的能力。两种LLMS和三种水印方法的初步实验结果表明，现有的文本水印技术在文本翻译成各种语言时缺乏一致性。基于这一观察结果，我们提出了一种跨语言水印移除攻击(CWRA)，通过首先从旋转语言的LLM获得响应，然后将其翻译成目标语言来绕过水印。CWRA可以有效地去除水印，将曲线下面积(AUC)从0.95减小到0.67，而不会造成性能损失。此外，我们分析了影响文本水印跨语言一致性的两个关键因素，并提出了一种在CWRA下将AUC从0.67提高到0.88的防御方法。



## **41. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

攻击树：自动破解黑盒LLM cs.LG

An implementation of the presented method is available at  https://github.com/RICommunity/TAP

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2312.02119v2) [paper-pdf](http://arxiv.org/pdf/2312.02119v2)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80% of the prompts using only a small number of queries. Interestingly, TAP is also capable of jailbreaking LLMs protected by state-of-the-art guardrails, e.g., LlamaGuard. This significantly improves upon the previous state-of-the-art black-box method for generating jailbreaks.

摘要: 虽然大型语言模型(LLM)显示了多功能，但它们继续产生有害、有偏见和有毒的内容，人类设计的越狱事件的流行就证明了这一点。在这项工作中，我们提出了带修剪的攻击树(TAP)，这是一种自动生成越狱的方法，只需要通过黑盒访问目标LLM。TAP利用LLM使用思想树推理反复优化候选(攻击)提示，直到其中一个生成的提示越狱目标。至关重要的是，在向目标发送提示之前，TAP会对它们进行评估，并删除那些不太可能导致越狱的提示。使用思维树推理允许TAP导航大的提示搜索空间，并进行修剪以减少发送到目标的查询总数。在经验评估中，我们观察到TAP仅使用少量查询就为80%以上的提示生成了越狱最先进的LLM(包括GPT4和GPT4-Turbo)提示。有趣的是，TAP还能够通过最先进的护栏保护LLMS越狱，例如LlamaGuard。这大大改进了以前用于生成越狱的最先进的黑匣子方法。



## **42. Large Language Models are Vulnerable to Bait-and-Switch Attacks for Generating Harmful Content**

大型语言模型容易受到诱饵和切换攻击，从而生成有害内容 cs.CL

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13926v1) [paper-pdf](http://arxiv.org/pdf/2402.13926v1)

**Authors**: Federico Bianchi, James Zou

**Abstract**: The risks derived from large language models (LLMs) generating deceptive and damaging content have been the subject of considerable research, but even safe generations can lead to problematic downstream impacts. In our study, we shift the focus to how even safe text coming from LLMs can be easily turned into potentially dangerous content through Bait-and-Switch attacks. In such attacks, the user first prompts LLMs with safe questions and then employs a simple find-and-replace post-hoc technique to manipulate the outputs into harmful narratives. The alarming efficacy of this approach in generating toxic content highlights a significant challenge in developing reliable safety guardrails for LLMs. In particular, we stress that focusing on the safety of the verbatim LLM outputs is insufficient and that we also need to consider post-hoc transformations.

摘要: 大型语言模型(LLM)产生欺骗性和破坏性内容的风险一直是大量研究的主题，但即使是安全的生成也可能导致问题的下游影响。在我们的研究中，我们将重点转移到如何通过诱饵和切换攻击将来自LLMS的安全文本轻松转换为潜在危险内容。在这种攻击中，用户首先用安全的问题提示LLMS，然后使用简单的查找和替换后自组织技术来将输出操纵为有害的叙述。这种方法在产生有毒物质方面的惊人效力，突显了在为低密度脂蛋白开发可靠的安全护栏方面的重大挑战。我们特别强调，仅注重LLM逐字输出的安全是不够的，我们还需要考虑临时后的转换。



## **43. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

模拟失调：大型语言模型的安全校准可能会适得其反！ cs.CL

Project web page: https://zhziszz.github.io/emulated-disalignment

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.12343v2) [paper-pdf](http://arxiv.org/pdf/2402.12343v2)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, in this work, we introduce an inference-time attack framework, demonstrating that safety alignment can also unintentionally facilitate harmful outcomes under adversarial manipulation. This framework, named Emulated Disalignment (ED), adversely combines a pair of open-source pre-trained and safety-aligned language models in the output space to produce a harmful language model without additional training. Our experiments with ED across three datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Crucially, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.

摘要: 大型语言模型(LLM)需要经过安全调整，以确保与人类的安全对话。然而，在这项工作中，我们引入了一个推理时间攻击框架，证明了安全对齐也可以在无意中促进对抗性操纵下的有害结果。这个名为仿真失调(ED)的框架在输出空间中反向组合了两个开放源码的预训练和安全对齐的语言模型，在没有额外训练的情况下产生了有害的语言模型。我们在三个数据集和四个模型家族(骆驼-1、骆驼-2、米斯特拉尔和羊驼)上使用ED进行的实验表明，ED的危害性是预训练模型的两倍，并且性能优于强基线，在48个评估子集中的43个子集上获得了最高的伤害率。至关重要的是，我们的发现强调了重新评估开源语言模型实践的重要性，即使在安全调整之后也是如此。



## **44. An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach**

一种可解释的基于transformer的网络钓鱼邮件检测模型：一种大语言模型方法 cs.LG

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13871v1) [paper-pdf](http://arxiv.org/pdf/2402.13871v1)

**Authors**: Mohammad Amaz Uddin, Iqbal H. Sarker

**Abstract**: Phishing email is a serious cyber threat that tries to deceive users by sending false emails with the intention of stealing confidential information or causing financial harm. Attackers, often posing as trustworthy entities, exploit technological advancements and sophistication to make detection and prevention of phishing more challenging. Despite extensive academic research, phishing detection remains an ongoing and formidable challenge in the cybersecurity landscape. Large Language Models (LLMs) and Masked Language Models (MLMs) possess immense potential to offer innovative solutions to address long-standing challenges. In this research paper, we present an optimized, fine-tuned transformer-based DistilBERT model designed for the detection of phishing emails. In the detection process, we work with a phishing email dataset and utilize the preprocessing techniques to clean and solve the imbalance class issues. Through our experiments, we found that our model effectively achieves high accuracy, demonstrating its capability to perform well. Finally, we demonstrate our fine-tuned model using Explainable-AI (XAI) techniques such as Local Interpretable Model-Agnostic Explanations (LIME) and Transformer Interpret to explain how our model makes predictions in the context of text classification for phishing emails.

摘要: 网络钓鱼电子邮件是一种严重的网络威胁，它试图通过发送虚假电子邮件来欺骗用户，目的是窃取机密信息或造成经济损失。攻击者通常伪装成值得信赖的实体，利用技术进步和复杂性使检测和预防网络钓鱼更具挑战性。尽管有广泛的学术研究，网络钓鱼检测仍然是网络安全领域的一个持续而艰巨的挑战。大型语言模型（LLM）和掩蔽语言模型（MLMs）具有巨大的潜力，可以提供创新的解决方案来解决长期存在的挑战。在这篇研究论文中，我们提出了一个优化的，微调的基于transformer的DistilBERT模型，旨在检测网络钓鱼电子邮件。在检测过程中，我们使用钓鱼邮件数据集，并利用预处理技术来清理和解决不平衡的类问题。通过我们的实验，我们发现我们的模型有效地实现了高精度，证明了它的性能良好。最后，我们使用可解释AI（XAI）技术（如本地可解释模型不可知解释（LIME）和Transformer Interpret）演示了我们的微调模型，以解释我们的模型如何在钓鱼电子邮件的文本分类上下文中进行预测。



## **45. Intention Analysis Makes LLMs A Good Jailbreak Defender**

意图分析使LLM成为一个很好的越狱防御者 cs.CL

17 pages, 12 figures

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2401.06561v2) [paper-pdf](http://arxiv.org/pdf/2401.06561v2)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreak attacks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). The principle behind this is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -46.5\% attack success rate) and maintain the general helpfulness. Encouragingly, with the help of our $\mathbb{IA}$, Vicuna-7b even outperforms GPT-3.5 in terms of attack success rate. Further analyses present some insights into how our method works. To facilitate reproducibility, we release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.

摘要: 要使大型语言模型(LLM)与人类价值观保持一致，尤其是在面对秘密和复杂的越狱袭击时，这是一个艰巨的挑战。在本研究中，我们提出了一种简单而高效的防御策略，即意图分析($\mathbb{IA}$)。这背后的原理是通过两个阶段触发LLMS内在的自我纠正和提高能力：1)基本意图分析，2)政策一致的反应。值得注意的是，$\mathbb{IA}$是一种仅限推理的方法，因此可以在不影响LLM的有用性的情况下增强其安全性。在Vicuna、ChatGLM、MPT、DeepSeek和GPT-3.5上对SAP200和DAN基准测试的广泛实验表明，$mathbb{IA}$可以持续且显著地降低响应中的危害性(平均-46.5\%攻击成功率)，并保持总体帮助。令人鼓舞的是，在我们$\mathbb{IA}$的帮助下，维库纳-7b的攻击成功率甚至超过了GPT-3.5。进一步的分析为我们的方法是如何工作的提供了一些见解。为了便于重现，我们在https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.上发布了我们的代码和脚本



## **46. Round Trip Translation Defence against Large Language Model Jailbreaking Attacks**

针对大型语言模型越狱攻击的往返翻译防御 cs.CL

6 pages, 6 figures

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13517v1) [paper-pdf](http://arxiv.org/pdf/2402.13517v1)

**Authors**: Canaan Yung, Hadi Mohaghegh Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly available at https://github.com/Cancanxxx/Round_Trip_Translation_Defence

摘要: 大型语言模型(LLM)容易受到社会工程攻击，这些攻击人类可以解释，但需要高水平的理解才能对抗LLM。现有的防御措施最多只能缓解这些攻击的不到一半。为了解决这个问题，我们提出了往返转换(RTT)方法，这是第一个专门设计用于防御针对LLM的社会工程攻击的算法。RTT解释了敌意提示并概括了所传达的思想，使LLMS更容易检测到诱导的有害行为。这种方法通用性强，重量轻，并且可以移植到不同的LLM上。我们的防御成功地缓解了70%以上的即时自动迭代精化(Pair)攻击，这是目前我们所知的最有效的防御。我们也是第一个尝试缓解MathsAttack的人，并将其攻击成功率降低了近40%。我们的代码在https://github.com/Cancanxxx/Round_Trip_Translation_Defence上公开提供



## **47. Learning to Poison Large Language Models During Instruction Tuning**

在教学调整过程中学习毒化大型语言模型 cs.LG

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13459v1) [paper-pdf](http://arxiv.org/pdf/2402.13459v1)

**Authors**: Yao Qiang, Xiangyu Zhou, Saleh Zare Zade, Mohammad Amin Roshani, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work highlights the need for stronger defenses against data poisoning attack, offering insights into safeguarding LLMs against these more sophisticated attacks. The source code can be found on this GitHub repository: https://github.com/RookieZxy/GBTL/blob/main/README.md.

摘要: 大型语言模型的出现在语言处理和推理能力方面取得了显著的成就。尽管取得了进步，但LLM仍面临数据中毒攻击的漏洞，即对手在训练数据中插入后门触发器，以恶意目的操纵输出。这项工作通过设计一种新的数据中毒攻击来进一步识别LLMS中的额外安全风险，该攻击专为利用指令调优过程而定制。我们提出了一种新的梯度引导的后门触发学习方法来高效地识别敌意触发，在保证内容完整性的同时避免了传统防御的检测。通过对不同LLM和任务的实验验证，我们的策略在牺牲模型输出方面表现出了很高的成功率；在4000个指令调优样本中只有1个中毒导致性能下降(PDR)约80%。我们的工作突显了针对数据中毒攻击的更强大防御的必要性，为保护LLM免受这些更复杂的攻击提供了见解。源代码可以在GitHub资源库中找到：https://github.com/RookieZxy/GBTL/blob/main/README.md.



## **48. LLM Jailbreak Attack versus Defense Techniques -- A Comprehensive Study**

LLM越狱攻防技术--综合研究 cs.CR

16 pages, 6 figures

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13457v1) [paper-pdf](http://arxiv.org/pdf/2402.13457v1)

**Authors**: Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, Stjepan Picek

**Abstract**: Large Language Models (LLMS) have increasingly become central to generating content with potential societal impacts. Notably, these models have demonstrated capabilities for generating content that could be deemed harmful. To mitigate these risks, researchers have adopted safety training techniques to align model outputs with societal values to curb the generation of malicious content. However, the phenomenon of "jailbreaking", where carefully crafted prompts elicit harmful responses from models, persists as a significant challenge. This research conducts a comprehensive analysis of existing studies on jailbreaking LLMs and their defense techniques. We meticulously investigate nine attack techniques and seven defense techniques applied across three distinct language models: Vicuna, LLama, and GPT-3.5 Turbo. We aim to evaluate the effectiveness of these attack and defense techniques. Our findings reveal that existing white-box attacks underperform compared to universal techniques and that including special tokens in the input significantly affects the likelihood of successful attacks. This research highlights the need to concentrate on the security facets of LLMs. Additionally, we contribute to the field by releasing our datasets and testing framework, aiming to foster further research into LLM security. We believe these contributions will facilitate the exploration of security measures within this domain.

摘要: 大型语言模型(LLM)越来越多地成为产生潜在社会影响的内容的核心。值得注意的是，这些模型展示了生成可能被认为有害的内容的能力。为了降低这些风险，研究人员采用了安全培训技术，使模型输出与社会价值观保持一致，以遏制恶意内容的生成。然而，“越狱”现象仍然是一个重大挑战。“越狱”是一种精心设计的行为，会招致模特的有害反应。本研究对已有的越狱LLMS及其防御技术的研究进行了全面的分析。我们仔细研究了在三种不同的语言模型中应用的九种攻击技术和七种防御技术：羊驼、骆驼和GPT-3.5Turbo。我们的目标是评估这些攻防技术的有效性。我们的发现表明，与通用技术相比，现有的白盒攻击表现不佳，并且在输入中包括特殊令牌显着影响攻击成功的可能性。这项研究强调了专注于低成本管理的安全方面的必要性。此外，我们通过发布我们的数据集和测试框架来为该领域做出贡献，旨在促进对LLM安全的进一步研究。我们相信，这些贡献将有助于探索这一领域内的安全措施。



## **49. Defending Jailbreak Prompts via In-Context Adversarial Game**

通过情景对抗性游戏捍卫越狱提示 cs.LG

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.13148v1) [paper-pdf](http://arxiv.org/pdf/2402.13148v1)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Taicheng Guo, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.

摘要: 大型语言模型(LLM)在不同的应用程序中展示了卓越的功能。然而，对他们的安全，特别是对越狱攻击的脆弱性的担忧依然存在。从深度学习和LLM代理学习过程中的对抗性训练中获得灵感，我们引入了无需微调的上下文对抗性游戏(ICAG)来防御越狱。ICAG利用代理学习进行对抗性游戏，旨在动态扩展知识来防御越狱。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一不断改进的过程加强了对新生成的越狱提示的防御。我们的经验研究肯定了ICAG的有效性，在不同的攻击场景中，由ICAG保护的LLM显示出显著降低的越狱成功率。此外，ICAG表现出显著的可转移性，表明其作为一种多功能防御机制的潜力。



## **50. Humans or LLMs as the Judge? A Study on Judgement Biases**

人类还是LLMS当法官？关于判断偏差的研究 cs.CL

19 pages

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.10669v2) [paper-pdf](http://arxiv.org/pdf/2402.10669v2)

**Authors**: Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang

**Abstract**: Adopting human and large language models (LLM) as judges (\textit{a.k.a} human- and LLM-as-a-judge) for evaluating the performance of existing LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLM judges, questioning the reliability of the evaluation results. In this paper, we propose a novel framework for investigating 5 types of biases for LLM and human judges. We curate a dataset with 142 samples referring to the revised Bloom's Taxonomy and conduct thousands of human and LLM evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the most cutting-edge judges possess considerable biases. We further exploit their weakness and conduct attacks on LLM judges. We hope that our work can notify the community of the vulnerability of human- and LLM-as-a-judge against perturbations, as well as the urgency of developing robust evaluation systems.

摘要: 采用人类和大型语言模型（LLM）作为法官（\textit{a.k.a} human- and LLM-as-a-judge）来评估现有LLM的性能最近受到了关注。尽管如此，这种方法同时引入了来自人类和LLM法官的潜在偏见，质疑评估结果的可靠性。在本文中，我们提出了一个新的框架，调查5种类型的LLM和人类法官的偏见。我们策划了一个包含142个样本的数据集，参考了修订后的布卢姆分类法，并进行了数千次人类和LLM评估。结果表明，人类和法学硕士法官容易受到不同程度的干扰，即使是最先进的法官也有相当大的偏见。我们进一步利用他们的弱点，对法学硕士法官进行攻击。我们希望我们的工作可以通知社会的脆弱性人类和法学硕士作为一个法官对扰动，以及开发强大的评估系统的紧迫性。



