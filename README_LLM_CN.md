# Latest Large Language Model Attack Papers
**update at 2024-03-25 09:33:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LogPrécis: Unleashing Language Models for Automated Malicious Log Analysis**

LogPrécis：释放用于自动恶意日志分析的语言模型 cs.CR

18 pages, Computer&Security  (https://www.sciencedirect.com/science/article/pii/S0167404824001068), code  available at https://github.com/SmartData-Polito/logprecis, models available  at https://huggingface.co/SmartDataPolito

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2307.08309v3) [paper-pdf](http://arxiv.org/pdf/2307.08309v3)

**Authors**: Matteo Boffa, Rodolfo Vieira Valentim, Luca Vassio, Danilo Giordano, Idilio Drago, Marco Mellia, Zied Ben Houidi

**Abstract**: The collection of security-related logs holds the key to understanding attack behaviors and diagnosing vulnerabilities. Still, their analysis remains a daunting challenge. Recently, Language Models (LMs) have demonstrated unmatched potential in understanding natural and programming languages. The question arises whether and how LMs could be also useful for security experts since their logs contain intrinsically confused and obfuscated information. In this paper, we systematically study how to benefit from the state-of-the-art in LM to automatically analyze text-like Unix shell attack logs. We present a thorough design methodology that leads to LogPr\'ecis. It receives as input raw shell sessions and automatically identifies and assigns the attacker tactic to each portion of the session, i.e., unveiling the sequence of the attacker's goals. We demonstrate LogPr\'ecis capability to support the analysis of two large datasets containing about 400,000 unique Unix shell attacks. LogPr\'ecis reduces them into about 3,000 fingerprints, each grouping sessions with the same sequence of tactics. The abstraction it provides lets the analyst better understand attacks, identify fingerprints, detect novelty, link similar attacks, and track families and mutations. Overall, LogPr\'ecis, released as open source, paves the way for better and more responsive defense against cyberattacks.

摘要: 安全相关日志的收集是理解攻击行为和诊断漏洞的关键。然而，他们的分析仍然是一个艰巨的挑战。最近，语言模型（LM）在理解自然语言和编程语言方面表现出无与伦比的潜力。问题是，LM是否以及如何对安全专家也有用，因为他们的日志包含本质上混乱和混淆的信息。本文系统地研究了如何利用LM的最新技术来自动分析类文本Unix shell攻击日志。我们提出了一个完整的设计方法，导致LogPr\'ecis。它接收原始shell会话作为输入，并自动识别和分配攻击者策略到会话的每个部分，即，揭示了攻击者目标的顺序我们演示了LogPr\'ecis的能力，以支持分析包含约400，000个独特Unix shell攻击的两个大型数据集。LogPr将它们减少到大约3，000个指纹，每个指纹使用相同的策略序列对会话进行分组。它提供的抽象使分析人员能够更好地理解攻击、识别指纹、检测新奇性、链接相似的攻击以及跟踪家族和变异。总体而言，LogPr\'ecis作为开源版本发布，为更好、更快速地防御网络攻击铺平了道路。



## **2. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images**

用冗长图像诱导大型视觉语言模型的高能量延迟 cs.CV

Accepted by ICLR 2024

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2401.11170v2) [paper-pdf](http://arxiv.org/pdf/2401.11170v2)

**Authors**: Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, Wei Liu

**Abstract**: Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications. Our code is available at https://github.com/KuofengGao/Verbose_Images.

摘要: 大型视觉语言模型（VLM）（如GPT—4）已经在各种多模式任务中实现了卓越的性能。然而，VLM的部署需要大量的能耗和计算资源。一旦攻击者在VLM的推理过程中恶意诱导高能量消耗和延迟时间（能量延迟成本），它将耗尽计算资源。在本文中，我们探讨了VLM可用性的攻击表面，并旨在诱导高的能量延迟成本在VLM的推理。我们发现，在VLM的推理过程中的高能量延迟成本可以通过最大化生成序列的长度来操纵。为此，我们提出了详细的图像，目的是制作一个难以察觉的扰动，诱导VLM在推理过程中生成长句子。具体而言，我们设计了三个损失目标。首先，提出了一个丢失来延迟序列结束令牌的出现，其中EOS令牌是VLM停止生成更多令牌的信号。提出了一种不确定性损失和一种令牌多样性损失，分别增加了每个令牌的不确定性和整个生成序列中所有令牌之间的多样性，从而打破了令牌级和序列级的输出依赖性。提出了一种时间权值调整算法，可以有效地平衡这些损失。大量实验表明，我们的详细图像可以将生成的序列长度增加7.87倍和8.56倍，与MS—COCO和ImageNet数据集上的原始图像相比，这对各种应用程序提出了潜在的挑战。我们的代码可在www.example.com获得。



## **3. Self-Guard: Empower the LLM to Safeguard Itself**

自我保护：增强LLM的自我保护能力 cs.CL

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2310.15851v2) [paper-pdf](http://arxiv.org/pdf/2310.15851v2)

**Authors**: Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong

**Abstract**: The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.

摘要: 越狱攻击可以绕过大型语言模型(LLM)的安全措施，生成有害内容。这种对LLM的滥用已经导致了负面的社会后果。目前，解决越狱攻击的主要方法有两种：安全培训和保障措施。安全培训的重点是对LLM进行进一步培训，以提高其安全性。另一方面，保障措施涉及实施外部模型或过滤器，以防止有害输出。然而，安全培训在适应新攻击类型的能力方面存在限制，往往会导致模型性能下降。事实证明，保障措施的帮助有限。为了解决这些问题，我们提出了一种名为Self-Guard的新方法，它结合了两种安全方法的优点。自我保护包括两个阶段。在第一阶段，我们增强了模型评估有害内容的能力，在第二阶段，我们指示模型对其自身的响应进行一致的有害内容检测。实验证明，Self-Guard对越狱攻击具有很强的抵抗力。在坏案例分析中，我们发现LLM偶尔会对有害查询提供无害的响应。此外，我们在安全培训前后评估了LLM的一般能力，提供了自我保护不会导致LLM性能下降的证据。在敏感性测试中，Self-Guard不仅可以避免在LLM中诱导过度敏感，而且甚至可以缓解这一问题。



## **4. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

闭上眼睛，安全开启：通过图像到文本转换保护多模式LLM cs.CV

Project Page: https://gyhdog99.github.io/projects/ecso/

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.09572v2) [paper-pdf](http://arxiv.org/pdf/2403.09572v2)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities, which, however, are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed due to the introduction of image features. To construct robust MLLMs, we propose ECSO(Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that our ECSO enhances model safety significantly (e.g., a 37.6% improvement on the MM-SafetyBench (SD+OCR), and 71.3% on VLSafe for the LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.

摘要: 多模态大型语言模型（MLLM）已经显示出令人印象深刻的推理能力，然而，这也比他们的LLM前辈更容易受到越狱攻击。虽然仍然能够检测到不安全的响应，我们观察到，安全机制的预对准LLM在MLLM可以很容易地绕过由于引入图像功能。为了构建强大的MLLM，我们提出了ECSO（Eyes Closed，Safety On），一种新的免训练保护方法，利用MLLM固有的安全意识，并通过自适应地将不安全图像转换为文本来激活MLLM的内在安全机制来生成更安全的响应。在五个最先进的（SoTA）MLLM上的实验表明，我们的ECSO显著提高了模型的安全性（例如，MM—SafetyBench（SD + OCR）提高了37.6%，LLaVA—1.5—7B的VLSafe提高了71.3%），同时始终保持了通用MLLM基准的效用结果。此外，我们表明，ECSO可以作为数据引擎来生成监督微调（SFT）数据MLLM对准，而无需额外的人工干预。



## **5. Risk and Response in Large Language Models: Evaluating Key Threat Categories**

大型语言模型中的风险和响应：评估关键威胁类别 cs.CL

19 pages, 14 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.14988v1) [paper-pdf](http://arxiv.org/pdf/2403.14988v1)

**Authors**: Bahareh Harandizadeh, Abel Salinas, Fred Morstatter

**Abstract**: This paper explores the pressing issue of risk assessment in Large Language Models (LLMs) as they become increasingly prevalent in various applications. Focusing on how reward models, which are designed to fine-tune pretrained LLMs to align with human values, perceive and categorize different types of risks, we delve into the challenges posed by the subjective nature of preference-based training data. By utilizing the Anthropic Red-team dataset, we analyze major risk categories, including Information Hazards, Malicious Uses, and Discrimination/Hateful content. Our findings indicate that LLMs tend to consider Information Hazards less harmful, a finding confirmed by a specially developed regression model. Additionally, our analysis shows that LLMs respond less stringently to Information Hazards compared to other risks. The study further reveals a significant vulnerability of LLMs to jailbreaking attacks in Information Hazard scenarios, highlighting a critical security concern in LLM risk assessment and emphasizing the need for improved AI safety measures.

摘要: 本文探讨了大型语言模型（LLM）风险评估的紧迫问题，因为它们在各种应用中变得越来越普遍。专注于奖励模型，其旨在微调预训练的LLM，以符合人类价值观，感知和分类不同类型的风险，我们深入研究基于偏好的训练数据的主观性质所带来的挑战。通过使用Anthropic Red团队数据集，我们分析了主要的风险类别，包括信息危害、恶意使用和歧视/仇恨内容。我们的研究结果表明，LLM倾向于认为信息危害危害较小，这一发现由一个专门开发的回归模型证实。此外，我们的分析表明，与其他风险相比，LLM对信息危害的反应不那么严格。该研究进一步揭示了LLM在信息危害场景中对越狱攻击的重大脆弱性，突出了LLM风险评估中的一个关键安全问题，并强调需要改进人工智能安全措施。



## **6. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

BadCLIP：触发感知提示学习，以应对CLIP上的后门攻击 cs.CV

14 pages, 6 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2311.16194v2) [paper-pdf](http://arxiv.org/pdf/2311.16194v2)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.

摘要: 对比视觉语言预训练，即CLIP，在解决下游图像识别任务方面显示出有希望的有效性。然而，最近的研究表明，CLIP模型可以植入一个面向下游的后门。在下游任务中，一个受害者模型在干净的样本上表现良好，但只要存在特定触发器，就会预测特定的目标类。为了注入后门，现有的攻击依赖于大量的额外数据来恶意微调整个预训练的CLIP模型，这使得它们不适用于数据有限的场景。在这项工作中，由最近的成功学习提示，我们解决了这个问题注入后门到CLIP模型在提示学习阶段。我们的方法称为BadCLIP是建立在一个新的有效的后门攻击机制，即，同时影响图像和文本编码器。它由一个应用于图像的可学习触发器和一个可感知的上下文生成器组成，这样触发器可以通过感知的提示来改变文本特征，从而产生强大且可推广的攻击。在11个数据集上进行的大量实验表明，BadCLIP的清除准确率与先进的即时学习方法相似，攻击成功率在大多数情况下都高于99%。BadCLIP还可以推广到未见的类，并且在跨数据集和跨域设置下表现出很强的推广能力。



## **7. Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model**

揭开印刷欺骗：大型视觉语言模型中印刷漏洞的透视 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2402.19150v2) [paper-pdf](http://arxiv.org/pdf/2402.19150v2)

**Authors**: Hao Cheng, Erjia Xiao, Jindong Gu, Le Yang, Jinhao Duan, Jize Zhang, Jiahang Cao, Kaidi Xu, Renjing Xu

**Abstract**: Large Vision-Language Models (LVLMs) rely on vision encoders and Large Language Models (LLMs) to exhibit remarkable capabilities on various multi-modal tasks in the joint space of vision and language. However, the Typographic Attack, which disrupts vision-language models (VLMs) such as Contrastive Language-Image Pretraining (CLIP), has also been expected to be a security threat to LVLMs. Firstly, we verify typographic attacks on current well-known commercial and open-source LVLMs and uncover the widespread existence of this threat. Secondly, to better assess this vulnerability, we propose the most comprehensive and largest-scale Typographic Dataset to date. The Typographic Dataset not only considers the evaluation of typographic attacks under various multi-modal tasks but also evaluates the effects of typographic attacks, influenced by texts generated with diverse factors. Based on the evaluation results, we investigate the causes why typographic attacks may impact VLMs and LVLMs, leading to three highly insightful discoveries. By the examination of our discoveries and experimental validation in the Typographic Dataset, we reduce the performance degradation from $42.07\%$ to $13.90\%$ when LVLMs confront typographic attacks.

摘要: 大型视觉语言模型（LVLM）依赖于视觉编码器和大型语言模型（LLM），在视觉和语言的联合空间中表现出卓越的能力。然而，排版攻击破坏视觉语言模型（VLM），如对比图像预训练（CLIP），也被认为是LVLM的安全威胁。首先，我们验证了目前著名的商业和开源LVLM的排版攻击，并揭示了这种威胁的广泛存在。其次，为了更好地评估这一漏洞，我们提出了迄今为止最全面和最大规模的排版数据集。排版数据集不仅考虑了在各种多模式任务下的排版攻击的评估，而且还评估了排版攻击的影响，受不同因素生成的文本的影响。基于评估结果，我们调查了排版攻击可能影响VLM和LVM的原因，导致三个非常有见地的发现。通过对我们的发现和在排版数据集中的实验验证，我们将LVLM面临排版攻击时的性能下降从42.07美元降低到13.90美元。



## **8. Detoxifying Large Language Models via Knowledge Editing**

通过知识编辑让大型语言模型脱毒 cs.CL

Ongoing work. Project website:  https://zjunlp.github.io/project/SafeEdit Benchmark:  https://huggingface.co/datasets/zjunlp/SafeEdit Code:  https://github.com/zjunlp/EasyEdit

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14472v1) [paper-pdf](http://arxiv.org/pdf/2403.14472v1)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments to compare knowledge editing approaches with previous baselines, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxify approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.

摘要: 本文探讨了使用知识编辑技术来解毒大型语言模型（LLM）。我们构建了一个基准，SafeEdit，它涵盖了九个不安全类别，并提供了各种强大的攻击提示，并为系统评估提供了全面的指标。我们进行了实验，比较知识编辑方法与以前的基线，表明知识编辑有潜力有效地解毒LLM与有限的一般性能影响。然后，我们提出了一个简单而有效的基线，称为手术内神经监测（DINM），通过一个实例在几个调整步骤内减少LLM的毒性。我们进一步深入分析了各种解毒方法的内在机制，证明以前的方法如SFT和DPO可能只是抑制了毒性参数的激活，而DINM在一定程度上减轻了毒性参数的毒性，做出了永久性的调整。我们希望这些见解可以为未来开发解毒方法和LLM的基本知识机制的工作提供启示。代码和基准测试可在www.example.com获得。



## **9. $\nabla τ$: Gradient-based and Task-Agnostic machine Unlearning**

$\nablaτ$：基于梯度和任务不可知的机器遗忘 cs.LG

14 pages, 2 figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14339v1) [paper-pdf](http://arxiv.org/pdf/2403.14339v1)

**Authors**: Daniel Trippa, Cesare Campagnano, Maria Sofia Bucarelli, Gabriele Tolomei, Fabrizio Silvestri

**Abstract**: Machine Unlearning, the process of selectively eliminating the influence of certain data examples used during a model's training, has gained significant attention as a means for practitioners to comply with recent data protection regulations. However, existing unlearning methods face critical drawbacks, including their prohibitively high cost, often associated with a large number of hyperparameters, and the limitation of forgetting only relatively small data portions. This often makes retraining the model from scratch a quicker and more effective solution. In this study, we introduce Gradient-based and Task-Agnostic machine Unlearning ($\nabla \tau$), an optimization framework designed to remove the influence of a subset of training data efficiently. It applies adaptive gradient ascent to the data to be forgotten while using standard gradient descent for the remaining data. $\nabla \tau$ offers multiple benefits over existing approaches. It enables the unlearning of large sections of the training dataset (up to 30%). It is versatile, supporting various unlearning tasks (such as subset forgetting or class removal) and applicable across different domains (images, text, etc.). Importantly, $\nabla \tau$ requires no hyperparameter adjustments, making it a more appealing option than retraining the model from scratch. We evaluate our framework's effectiveness using a set of well-established Membership Inference Attack metrics, demonstrating up to 10% enhancements in performance compared to state-of-the-art methods without compromising the original model's accuracy.

摘要: 机器取消学习是一种选择性地消除模型训练期间使用的某些数据示例的影响的过程，作为从业人员遵守最新数据保护法规的一种手段，已经获得了极大的关注。然而，现有的非学习方法面临着严重的缺点，包括它们的高成本，通常与大量的超参数相关联，以及仅遗忘相对较小的数据部分的限制。这通常使得从头开始重新训练模型成为一个更快、更有效的解决方案。在这项研究中，我们引入了基于任务不可知的机器Unlearning（$\nabla\tau $），这是一个优化框架，旨在有效地消除训练数据子集的影响。它将自适应梯度上升应用于要被遗忘的数据，而将标准梯度下降应用于剩余数据。$\nabla\tau $比现有的方法有多个好处。它允许学习训练数据集的大部分（最多30%）。它是通用的，支持各种非学习任务（如子集遗忘或类删除），并适用于不同的领域（图像，文本等）。重要的是，$\nabla\tau $不需要超参数调整，这使得它比从头开始重新训练模型更有吸引力。我们使用一组成熟的成员推断攻击指标来评估我们的框架的有效性，与最先进的方法相比，性能提高了10%，而不会损害原始模型的准确性。



## **10. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统文献综述 cs.CR

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14280v1) [paper-pdf](http://arxiv.org/pdf/2403.14280v1)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型（LLM）已经成为涉及区块链安全（BS）的各个领域的强大工具。最近的几项研究正在探索法学硕士应用于学士学位。然而，我们对LLM对区块链安全的全部应用范围、影响和潜在限制的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。   作为对LLM在区块链安全方面的应用的首次审查，我们的研究旨在全面分析现有的研究，并阐明LLM如何有助于增强区块链系统的安全性。通过对学术著作的彻底审查，我们深入研究了将LLM集成到区块链安全的各个方面。我们探索了LLM支持区块链安全的机制，包括其在智能合约审计、身份验证、异常检测、漏洞修复等方面的应用。此外，我们还考虑了可扩展性、隐私问题和对抗性攻击等因素，认真评估了利用LLM实现区块链安全的挑战和局限性。我们的综述揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了宝贵的见解。



## **11. FMM-Attack: A Flow-based Multi-modal Adversarial Attack on Video-based LLMs**

FMM—Attack：一种基于流的多模式对抗性视频LLM攻击 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13507v2) [paper-pdf](http://arxiv.org/pdf/2403.13507v2)

**Authors**: Jinmin Li, Kuofeng Gao, Yang Bai, Jingyun Zhang, Shu-tao Xia, Yisen Wang

**Abstract**: Despite the remarkable performance of video-based large language models (LLMs), their adversarial threat remains unexplored. To fill this gap, we propose the first adversarial attack tailored for video-based LLMs by crafting flow-based multi-modal adversarial perturbations on a small fraction of frames within a video, dubbed FMM-Attack. Extensive experiments show that our attack can effectively induce video-based LLMs to generate incorrect answers when videos are added with imperceptible adversarial perturbations. Intriguingly, our FMM-Attack can also induce garbling in the model output, prompting video-based LLMs to hallucinate. Overall, our observations inspire a further understanding of multi-modal robustness and safety-related feature alignment across different modalities, which is of great importance for various large multi-modal models. Our code is available at https://github.com/THU-Kingmin/FMM-Attack.

摘要: 尽管基于视频的大型语言模型（LLM）表现出色，但它们的对抗性威胁仍未得到探索。为了填补这一空白，我们提出了第一个针对基于视频的LLM的对抗攻击，通过在视频中的一小部分帧上制作基于流的多模式对抗干扰，称为FMM攻击。大量的实验表明，我们的攻击可以有效地诱导基于视频的LLM生成错误的答案时，视频中添加了不可感知的对抗干扰。有趣的是，我们的FM—Attack还可以在模型输出中引起混乱，促使基于视频的LLM产生幻觉。总的来说，我们的观察结果激发了人们对不同模态的多模态鲁棒性和安全相关特性对齐的进一步理解，这对各种大型多模态模型非常重要。我们的代码可在www.example.com获得。



## **12. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

AutoDAN：在对齐的大型语言模型上生成秘密越狱提示 cs.CL

Published as a conference paper at ICLR 2024. Code is available at  https://github.com/SheltonLiu-N/AutoDAN

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2310.04451v2) [paper-pdf](http://arxiv.org/pdf/2310.04451v2)

**Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

**Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.

摘要: 对齐的大型语言模型（LLM）是强大的语言理解和决策工具，通过与人类反馈的广泛对齐而创建。然而，这些大型模型仍然容易受到越狱攻击，攻击者操纵提示以引出不应该由对齐的LLM给出的恶意输出。调查越狱提示可以引导我们深入研究LLM的局限性，并进一步指导我们保护它们。不幸的是，现有的越狱技术要么受到（1）可伸缩性问题，其中攻击严重依赖人工制作提示符，要么（2）隐蔽性问题，因为攻击依赖基于令牌的算法来生成通常在语义上毫无意义的提示符，使得它们易于通过基本的困惑度测试检测。鉴于这些挑战，我们打算回答这个问题：我们能否开发出一种方法，可以自动生成隐蔽的越狱提示？在本文中，我们介绍了一种新的针对对齐的LLM的越狱攻击—AutoDAN。AutoDAN可以通过精心设计的层次遗传算法自动生成隐蔽越狱提示。广泛的评估表明，AutoDAN不仅在保持语义意义的同时自动化过程，而且在跨模型可迁移性和跨样本通用性方面表现出优于基线的攻击强度。此外，我们还比较了AutoDAN和基于复杂度的防御方法，表明AutoDAN可以有效地绕过它们。



## **13. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

大型语言模型(LLM)安全与隐私：好、坏、丑 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2312.02003v3) [paper-pdf](http://arxiv.org/pdf/2312.02003v3)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.

摘要: 大型语言模型（LLM），如ChatGPT和Bard，已经彻底改变了自然语言的理解和生成。他们拥有深度的语言理解能力，类似人类的文本生成能力，上下文意识和强大的解决问题的能力，使他们在各个领域（例如，搜索引擎、客户支持、翻译）。与此同时，LLM在安全界也获得了吸引力，揭示了安全漏洞，并展示了其在安全相关任务中的潜力。本文探讨了LLM与安全和隐私的交叉点。具体而言，我们调查了LLM如何积极影响安全和隐私，与其使用相关的潜在风险和威胁，以及LLM内部的固有漏洞。通过全面的文献回顾，本文将这些论文分为“好”（有益的法学硕士应用）、“坏”（攻击性应用）和“丑”（LLM的漏洞及其防御）。我们有一些有趣的发现。例如，LLM已被证明可以增强代码安全性（代码漏洞检测）和数据隐私性（数据机密性保护），超过传统方法。然而，由于它们具有类似人类的推理能力，它们也可以被用于各种攻击（特别是用户级攻击）。我们确定了需要进一步研究的领域。例如，对模型和参数提取攻击的研究是有限的，通常是理论性的，受到LLM参数规模和机密性的阻碍。安全指令调优是最近的一个发展，需要更多的探索。我们希望我们的工作能够阐明LLM在支持和危害网络安全方面的潜力。



## **14. Defending Against Indirect Prompt Injection Attacks With Spotlighting**

利用聚光灯防御间接即时注入攻击 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14720v1) [paper-pdf](http://arxiv.org/pdf/2403.14720v1)

**Authors**: Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre Kiciman

**Abstract**: Large Language Models (LLMs), while powerful, are built and trained to process a single text input. In common applications, multiple inputs can be processed by concatenating them together into a single stream of text. However, the LLM is unable to distinguish which sections of prompt belong to various input sources. Indirect prompt injection attacks take advantage of this vulnerability by embedding adversarial instructions into untrusted data being processed alongside user commands. Often, the LLM will mistake the adversarial instructions as user commands to be followed, creating a security vulnerability in the larger system. We introduce spotlighting, a family of prompt engineering techniques that can be used to improve LLMs' ability to distinguish among multiple sources of input. The key insight is to utilize transformations of an input to provide a reliable and continuous signal of its provenance. We evaluate spotlighting as a defense against indirect prompt injection attacks, and find that it is a robust defense that has minimal detrimental impact to underlying NLP tasks. Using GPT-family models, we find that spotlighting reduces the attack success rate from greater than {50}\% to below {2}\% in our experiments with minimal impact on task efficacy.

摘要: 大型语言模型（LLM）虽然功能强大，但被构建和训练为处理单个文本输入。在常见的应用程序中，多个输入可以通过将它们连接到单个文本流中来处理。然而，LLM无法区分提示符的哪些部分属于不同的输入源。间接提示注入攻击通过将对抗指令嵌入到与用户命令一起处理的不可信数据中来利用此漏洞。通常，LLM会将对抗指令误认为是要遵循的用户命令，从而在更大的系统中造成安全漏洞。我们介绍了聚光灯，一系列即时工程技术，可用于提高LLM区分多个输入源的能力。关键的洞察力是利用输入的转换来提供其来源的可靠和连续的信号。我们评估聚光灯作为一种防御间接提示注入攻击，并发现它是一个强大的防御，具有最小的不利影响底层NLP任务。在实验中，我们发现聚光灯使攻击成功率从大于{50}\%降低到{2}\%以下，对任务效能的影响最小。



## **15. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

攻坚战：如何评估越狱攻击在大型语言模型上的有效性 cs.CL

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2401.09002v3) [paper-pdf](http://arxiv.org/pdf/2401.09002v3)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

摘要: 在我们的研究中，我们开创了一种新的方法来评估越狱攻击对大型语言模型(如GPT-4和LLaMa2)的有效性，不同于传统的专注于健壮性的二进制评估。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供了一个独特的视角，能够对攻击效果进行更全面和细微的评估，并使攻击者能够更好地了解他们的攻击提示。此外，我们还开发了专门为越狱任务量身定做的全面地面事实数据集。这一数据集不仅是我们当前研究的重要基准，而且还为未来的研究奠定了基础资源，使这一不断发展的领域能够进行一致和比较的分析。通过与传统评估方法的细致比较，我们发现我们的评估符合基线的趋势，同时提供了更深入和详细的评估。我们相信，通过准确评估越狱任务中攻击提示的有效性，我们的工作为评估快速注射领域中更广泛的类似甚至更复杂的任务奠定了坚实的基础，这可能会给这一领域带来革命性的变化。



## **16. BadEdit: Backdooring large language models by model editing**

BadEDIT：通过模型编辑来反推大型语言模型 cs.CR

ICLR 2024

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13355v1) [paper-pdf](http://arxiv.org/pdf/2403.13355v1)

**Authors**: Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, Yang Liu

**Abstract**: Mainstream backdoor attack methods typically demand substantial tuning data for poisoning, limiting their practicality and potentially degrading the overall performance when applied to Large Language Models (LLMs). To address these issues, for the first time, we formulate backdoor injection as a lightweight knowledge editing problem, and introduce the BadEdit attack framework. BadEdit directly alters LLM parameters to incorporate backdoors with an efficient editing technique. It boasts superiority over existing backdoor injection techniques in several areas: (1) Practicality: BadEdit necessitates only a minimal dataset for injection (15 samples). (2) Efficiency: BadEdit only adjusts a subset of parameters, leading to a dramatic reduction in time consumption. (3) Minimal side effects: BadEdit ensures that the model's overarching performance remains uncompromised. (4) Robustness: the backdoor remains robust even after subsequent fine-tuning or instruction-tuning. Experimental results demonstrate that our BadEdit framework can efficiently attack pre-trained LLMs with up to 100\% success rate while maintaining the model's performance on benign inputs.

摘要: 主流后门攻击方法通常需要大量调整数据以进行中毒，这限制了它们的实用性，并可能在应用于大型语言模型(LLM)时降低整体性能。为了解决这些问题，我们首次将后门注入描述为一个轻量级的知识编辑问题，并引入了BadEdit攻击框架。BadEDIT直接更改LLM参数，将后门与高效的编辑技术结合在一起。它在几个方面优于现有的后门注入技术：(1)实用性：BadEdit只需要一个最小的注入数据集(15个样本)。(2)效率：BadEDIT只调整部分参数，大大减少了时间消耗。(3)副作用最小：BadEdit可确保模型的总体性能不受影响。(4)健壮性：即使在随后的微调或指令调优之后，后门仍然保持健壮。实验结果表明，我们的BadEdit框架可以有效地攻击预先训练的LLMS，成功率高达100%，同时保持了模型在良性输入下的性能。



## **17. Mapping LLM Security Landscapes: A Comprehensive Stakeholder Risk Assessment Proposal**

绘制LLM安全图景：全面的利益相关者风险评估建议 cs.CR

10 pages, 1 figure, 3 tables

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13309v1) [paper-pdf](http://arxiv.org/pdf/2403.13309v1)

**Authors**: Rahul Pankajakshan, Sumitra Biswal, Yuvaraj Govindarajulu, Gilad Gressel

**Abstract**: The rapid integration of Large Language Models (LLMs) across diverse sectors has marked a transformative era, showcasing remarkable capabilities in text generation and problem-solving tasks. However, this technological advancement is accompanied by significant risks and vulnerabilities. Despite ongoing security enhancements, attackers persistently exploit these weaknesses, casting doubts on the overall trustworthiness of LLMs. Compounding the issue, organisations are deploying LLM-integrated systems without understanding the severity of potential consequences. Existing studies by OWASP and MITRE offer a general overview of threats and vulnerabilities but lack a method for directly and succinctly analysing the risks for security practitioners, developers, and key decision-makers who are working with this novel technology. To address this gap, we propose a risk assessment process using tools like the OWASP risk rating methodology which is used for traditional systems. We conduct scenario analysis to identify potential threat agents and map the dependent system components against vulnerability factors. Through this analysis, we assess the likelihood of a cyberattack. Subsequently, we conduct a thorough impact analysis to derive a comprehensive threat matrix. We also map threats against three key stakeholder groups: developers engaged in model fine-tuning, application developers utilizing third-party APIs, and end users. The proposed threat matrix provides a holistic evaluation of LLM-related risks, enabling stakeholders to make informed decisions for effective mitigation strategies. Our outlined process serves as an actionable and comprehensive tool for security practitioners, offering insights for resource management and enhancing the overall system security.

摘要: 跨不同部门的大型语言模型(LLM)的快速整合标志着一个变革时代的到来，展示了在文本生成和解决问题任务方面的非凡能力。然而，这种技术进步也伴随着重大的风险和脆弱性。尽管正在进行安全增强，攻击者仍不断地利用这些弱点，使人对低成本管理系统的整体可信度产生怀疑。让问题变得更加复杂的是，组织在部署LLM集成系统时，并不了解潜在后果的严重性。OWASP和MITRE的现有研究提供了对威胁和漏洞的总体概述，但缺乏直接和简洁地分析使用这一新技术的安全从业者、开发人员和关键决策者的风险的方法。为了解决这一差距，我们提出了一个风险评估过程，使用传统系统使用的OWASP风险评级方法等工具。我们进行场景分析，以确定潜在的威胁代理，并将依赖的系统组件与漏洞因素进行映射。通过此分析，我们评估了网络攻击的可能性。随后，我们进行了彻底的影响分析，以得出一个全面的威胁矩阵。我们还将威胁映射到三个关键的利益相关者群体：从事模型微调的开发人员、使用第三方API的应用程序开发人员和最终用户。拟议的威胁矩阵提供了对LLM相关风险的全面评估，使利益相关者能够做出明智的决策，制定有效的缓解战略。我们概述的流程为安全从业者提供了一种可操作的综合工具，为资源管理和增强整体系统安全性提供了见解。



## **18. Bypassing LLM Watermarks with Color-Aware Substitutions**

使用颜色感知替换绕过LLM水印 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.14719v1) [paper-pdf](http://arxiv.org/pdf/2403.14719v1)

**Authors**: Qilong Wu, Varun Chandrasekaran

**Abstract**: Watermarking approaches are proposed to identify if text being circulated is human or large language model (LLM) generated. The state-of-the-art watermarking strategy of Kirchenbauer et al. (2023a) biases the LLM to generate specific (``green'') tokens. However, determining the robustness of this watermarking method is an open problem. Existing attack methods fail to evade detection for longer text segments. We overcome this limitation, and propose {\em Self Color Testing-based Substitution (SCTS)}, the first ``color-aware'' attack. SCTS obtains color information by strategically prompting the watermarked LLM and comparing output tokens frequencies. It uses this information to determine token colors, and substitutes green tokens with non-green ones. In our experiments, SCTS successfully evades watermark detection using fewer number of edits than related work. Additionally, we show both theoretically and empirically that SCTS can remove the watermark for arbitrarily long watermarked text.

摘要: 提出了一种水印方法来识别被传播的文本是人为的还是大型语言模型（LLM）生成的。Kirchenbauer等人（2023a）的最先进的水印策略使LLM偏向于生成特定的（“绿色”）令牌。然而，确定这种水印方法的鲁棒性是一个开放的问题。现有的攻击方法无法逃避对较长文本段的检测。我们克服了这个限制，并提出了{\em Self Color Testing—based Substitution（SCTS）}，这是第一个“color—aware”攻击。SCTS通过策略性地提示带水印的LLM并比较输出令牌频率来获取颜色信息。它使用这些信息来确定令牌的颜色，并用非绿色令牌替换绿色令牌。在我们的实验中，SCTS成功地避开水印检测使用较少的编辑次数比相关工作。此外，我们从理论和经验上证明了SCTS可以去除任意长水印文本的水印。



## **19. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

LLM会话安全的攻击、防御与评估：一项调查 cs.CL

Accepted to NAACL 2024

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2402.09283v2) [paper-pdf](http://arxiv.org/pdf/2402.09283v2)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.

摘要: 大型语言模型（LLM）现在在会话应用中很常见。然而，它们被滥用以产生有害反应的风险引起了严重的社会关注，并刺激了最近对LLM会话安全的研究。因此，在本次调查中，我们提供了一个全面的概述最近的研究，涵盖了LLM会话安全的三个关键方面：攻击，防御和评估。我们的目标是提供一个结构化的摘要，以提高对LLM会话安全的理解，并鼓励进一步调查这一重要主题。为了便于参考，我们根据我们的分类法对本次调查中提到的所有研究进行了分类，可在www.example.com上查阅。



## **20. Review of Generative AI Methods in Cybersecurity**

网络安全中的生成性人工智能方法综述 cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.

摘要: 在过去的十年中，人工智能（AI）变得越来越受欢迎，特别是随着聊天机器人的使用，如ChatGPT，Gemini和DALL—E。随着这种增长，大型语言模型（LLM）和生成人工智能（GenAI）在日常使用中也变得越来越普遍。这些进步加强了网络安全的防御态势，并为对手开辟了新的攻击途径。本文全面概述了GenAI当前最先进的部署，涵盖攻击、越狱以及即时注射和逆向心理学的应用。本文还提供了GenAI在网络犯罪中的各种应用，如自动黑客攻击、网络钓鱼电子邮件、社会工程、反向加密、创建攻击有效载荷和创建恶意软件。GenAI可以通过数据集构建、安全代码开发、威胁情报、防御措施、报告和网络攻击检测等策略显著提高防御性网络安全流程的自动化。在这项研究中，我们建议未来的研究应侧重于制定健全的道德规范和创新的防御机制，以解决GenAI目前造成的问题，并进一步鼓励其在网络安全中的未来应用公正的方法。此外，我们强调跨学科方法的重要性，进一步弥合科学发展与伦理考虑之间的差距。



## **21. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

RigorLLM：针对不想要的内容的大型语言模型的弹性屏障 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.13031v1) [paper-pdf](http://arxiv.org/pdf/2403.13031v1)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.

摘要: 大型语言模型(LLM)的最新进展展示了跨越不同领域的各种任务的显著能力。然而，偏见的出现和在低成本管理中产生有害内容的可能性，特别是在恶意投入下，构成了重大挑战。目前的缓解战略虽然有效，但在对抗性攻击下缺乏弹性。本文介绍了用于大型语言模型的弹性护栏(RigorLLM)，这是一个新的框架，旨在高效和有效地控制LLM中有害和不安全的输入和输出。通过采用多方面的方法，包括通过朗之万动力学基于能量的训练数据增强，通过极小极大优化优化输入的安全后缀，以及基于我们的数据增强将稳健的KNN与LLMS相结合的基于融合的模型，RigorLLM为有害内容适度提供了稳健的解决方案。我们的实验评估表明，RigorLLM不仅在检测有害内容方面优于OpenAI API和透视API等现有基线，而且对越狱攻击表现出无与伦比的弹性。约束优化和基于融合的护栏方法的创新使用代表着在开发更安全可靠的LLMS方面向前迈出的重要一步，为面对不断变化的数字威胁的内容审查框架设定了新的标准。



## **22. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

保护大型语言模型：威胁、漏洞和负责任的做法 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12503v1) [paper-pdf](http://arxiv.org/pdf/2403.12503v1)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.

摘要: 大型语言模型（LLM）显著改变了自然语言处理（NLP）的前景。它们的影响延伸到各种任务，彻底改变了我们处理语言理解和世代的方式。然而，除了其显著的实用性，LLM引入了关键的安全和风险考虑。这些挑战值得认真审查，以确保负责任地部署和防范潜在漏洞。本研究论文从五个主题角度彻底调查了与LLM相关的安全和隐私问题：安全和隐私问题，对抗攻击的漏洞，滥用LLM造成的潜在危害，缓解策略，以解决这些挑战，同时确定当前策略的局限性。最后，本文建议了未来研究的有希望的途径，以加强LLM的安全性和风险管理。



## **23. Large language models in 6G security: challenges and opportunities**

6G安全中的大型语言模型：挑战与机遇 cs.CR

29 pages, 2 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12239v1) [paper-pdf](http://arxiv.org/pdf/2403.12239v1)

**Authors**: Tri Nguyen, Huong Nguyen, Ahmad Ijaz, Saeid Sheikhi, Athanasios V. Vasilakos, Panos Kostakos

**Abstract**: The rapid integration of Generative AI (GenAI) and Large Language Models (LLMs) in sectors such as education and healthcare have marked a significant advancement in technology. However, this growth has also led to a largely unexplored aspect: their security vulnerabilities. As the ecosystem that includes both offline and online models, various tools, browser plugins, and third-party applications continues to expand, it significantly widens the attack surface, thereby escalating the potential for security breaches. These expansions in the 6G and beyond landscape provide new avenues for adversaries to manipulate LLMs for malicious purposes. We focus on the security aspects of LLMs from the viewpoint of potential adversaries. We aim to dissect their objectives and methodologies, providing an in-depth analysis of known security weaknesses. This will include the development of a comprehensive threat taxonomy, categorizing various adversary behaviors. Also, our research will concentrate on how LLMs can be integrated into cybersecurity efforts by defense teams, also known as blue teams. We will explore the potential synergy between LLMs and blockchain technology, and how this combination could lead to the development of next-generation, fully autonomous security solutions. This approach aims to establish a unified cybersecurity strategy across the entire computing continuum, enhancing overall digital security infrastructure.

摘要: 生成式人工智能(GenAI)和大型语言模型(LLM)在教育和医疗等领域的快速集成标志着技术的重大进步。然而，这种增长也导致了一个基本上未被探索的方面：它们的安全漏洞。随着包括离线和在线模型、各种工具、浏览器插件和第三方应用程序的生态系统不断扩大，它显著扩大了攻击面，从而增加了安全漏洞的可能性。这些在6G及以上领域的扩展为对手出于恶意目的操纵低层管理提供了新的途径。我们从潜在对手的角度来关注LLMS的安全方面。我们的目标是剖析它们的目标和方法，深入分析已知的安全弱点。这将包括开发一个全面的威胁分类法，对各种对手行为进行分类。此外，我们的研究将集中在如何将LLM整合到防御团队(也称为蓝色团队)的网络安全工作中。我们将探索LLMS和区块链技术之间的潜在协同效应，以及这种结合如何导致下一代完全自主的安全解决方案的开发。这一方法旨在整个计算连续体中建立统一的网络安全战略，增强整体数字安全基础设施。



## **24. Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models**

移动镜头：使用大型语言模型检测NPM生态系统中的恶意软件 cs.CR

13 pages, 1 Figure, 7 tables

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12196v1) [paper-pdf](http://arxiv.org/pdf/2403.12196v1)

**Authors**: Nusrat Zahan, Philipp Burckhardt, Mikola Lysenko, Feross Aboukhadijeh, Laurie Williams

**Abstract**: The Gartner 2022 report predicts that 45% of organizations worldwide will encounter software supply chain attacks by 2025, highlighting the urgency to improve software supply chain security for community and national interests. Current malware detection techniques aid in the manual review process by filtering benign and malware packages, yet such techniques have high false-positive rates and limited automation support. Therefore, malware detection techniques could benefit from advanced, more automated approaches for accurate and minimally false-positive results. The goal of this study is to assist security analysts in identifying malicious packages through the empirical study of large language models (LLMs) to detect potential malware in the npm ecosystem.   We present SocketAI Scanner, a multi-stage decision-maker malware detection workflow using iterative self-refinement and zero-shot-role-play-Chain of Thought (CoT) prompting techniques for ChatGPT. We studied 5,115 npm packages (of which 2,180 are malicious) and performed a baseline comparison of the GPT-3 and GPT-4 models with a static analysis tool. Our findings showed promising results for GPT models with low misclassification alert rates. Our baseline comparison demonstrates a notable improvement over static analysis in precision scores above 25% and F1 scores above 15%. We attained precision and F1 scores of 91% and 94%, respectively, for the GPT-3 model. Overall, GPT-4 demonstrates superior performance in precision (99%) and F1 (97%) scores, while GPT-3 presents a cost-effective balance between performance and expenditure.

摘要: Gartner 2022年报告预测，到2025年，全球45%的组织将遭遇软件供应链攻击，突显出为社区和国家利益改善软件供应链安全的紧迫性。当前的恶意软件检测技术通过过滤良性和恶意软件包来帮助手动审查过程，但是这种技术具有高的假阳性率和有限的自动化支持。因此，恶意软件检测技术可以受益于先进的、更自动化的方法，以获得准确和最小的假阳性结果。本研究的目标是帮助安全分析师通过大型语言模型（LLM）的实证研究来识别恶意软件包，以检测npm生态系统中的潜在恶意软件。   我们提出了SocketAI Scanner，一个多阶段决策者恶意软件检测工作流程，使用迭代自细化和零射击角色扮演思想链（CoT）提示技术，用于ChatGPT。我们研究了5，115个npm包（其中2，180个是恶意的），并使用静态分析工具对GPT—3和GPT—4模型进行了基线比较。我们的研究结果显示，具有低误分类警报率的GPT模型有希望的结果。我们的基线比较表明，与静态分析相比，精确度得分超过25%，F1得分超过15%。对于GPT—3模型，我们获得的精确度和F1评分分别为91%和94%。总体而言，GPT—4在精确度（99%）和F1（97%）评分方面表现出色，而GPT—3在性能和支出之间实现了成本效益的平衡。



## **25. EasyJailbreak: A Unified Framework for Jailbreaking Large Language Models**

EasyJailbreak：一个统一的大型语言模型越狱框架 cs.CL

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12171v1) [paper-pdf](http://arxiv.org/pdf/2403.12171v1)

**Authors**: Weikang Zhou, Xiao Wang, Limao Xiong, Han Xia, Yingshuang Gu, Mingxu Chai, Fukang Zhu, Caishuang Huang, Shihan Dou, Zhiheng Xi, Rui Zheng, Songyang Gao, Yicheng Zou, Hang Yan, Yifan Le, Ruohui Wang, Lijun Li, Jing Shao, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Jailbreak attacks are crucial for identifying and mitigating the security vulnerabilities of Large Language Models (LLMs). They are designed to bypass safeguards and elicit prohibited outputs. However, due to significant differences among various jailbreak methods, there is no standard implementation framework available for the community, which limits comprehensive security evaluations. This paper introduces EasyJailbreak, a unified framework simplifying the construction and evaluation of jailbreak attacks against LLMs. It builds jailbreak attacks using four components: Selector, Mutator, Constraint, and Evaluator. This modular framework enables researchers to easily construct attacks from combinations of novel and existing components. So far, EasyJailbreak supports 11 distinct jailbreak methods and facilitates the security validation of a broad spectrum of LLMs. Our validation across 10 distinct LLMs reveals a significant vulnerability, with an average breach probability of 60% under various jailbreaking attacks. Notably, even advanced models like GPT-3.5-Turbo and GPT-4 exhibit average Attack Success Rates (ASR) of 57% and 33%, respectively. We have released a wealth of resources for researchers, including a web platform, PyPI published package, screencast video, and experimental outputs.

摘要: 越狱攻击对于识别和缓解大型语言模型（LLM）的安全漏洞至关重要。它们的设计是为了绕过保障措施，获取被禁止的产出。然而，由于各种越狱方法之间存在很大差异，社区没有标准的实施框架，这限制了全面的安全评估。本文介绍了EasyJailbreak，一个统一的框架，简化了针对LLM的越狱攻击的构造和评估。它使用四个组件构建越狱攻击：XSLT、Mutator、约束和评估器。这种模块化框架使研究人员能够轻松地从新组件和现有组件的组合中构建攻击。到目前为止，EasyJailbreak支持11种不同的越狱方法，并促进了广泛的LLM的安全验证。我们在10个不同的LLM上进行的验证揭示了一个重大漏洞，在各种越狱攻击下，平均漏洞概率为60%。值得注意的是，即使是像GPT—3.5—Turbo和GPT—4这样的先进机型，平均攻击成功率（ASR）分别为57%和33%。我们为研究人员发布了大量资源，包括网络平台、PyPI发布包、屏幕视频和实验输出。



## **26. Navigation as Attackers Wish? Towards Building Robust Embodied Agents under Federated Learning**

导航如攻击者所愿？基于联邦学习的鲁棒代理构建方法 cs.AI

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2211.14769v4) [paper-pdf](http://arxiv.org/pdf/2211.14769v4)

**Authors**: Yunchao Zhang, Zonglin Di, Kaiwen Zhou, Cihang Xie, Xin Eric Wang

**Abstract**: Federated embodied agent learning protects the data privacy of individual visual environments by keeping data locally at each client (the individual environment) during training. However, since the local data is inaccessible to the server under federated learning, attackers may easily poison the training data of the local client to build a backdoor in the agent without notice. Deploying such an agent raises the risk of potential harm to humans, as the attackers may easily navigate and control the agent as they wish via the backdoor. Towards Byzantine-robust federated embodied agent learning, in this paper, we study the attack and defense for the task of vision-and-language navigation (VLN), where the agent is required to follow natural language instructions to navigate indoor environments. First, we introduce a simple but effective attack strategy, Navigation as Wish (NAW), in which the malicious client manipulates local trajectory data to implant a backdoor into the global model. Results on two VLN datasets (R2R and RxR) show that NAW can easily navigate the deployed VLN agent regardless of the language instruction, without affecting its performance on normal test sets. Then, we propose a new Prompt-Based Aggregation (PBA) to defend against the NAW attack in federated VLN, which provides the server with a ''prompt'' of the vision-and-language alignment variance between the benign and malicious clients so that they can be distinguished during training. We validate the effectiveness of the PBA method on protecting the global model from the NAW attack, which outperforms other state-of-the-art defense methods by a large margin in the defense metrics on R2R and RxR.

摘要: 联合具体化代理学习通过在培训期间在每个客户端(个体环境)本地保存数据来保护个体视觉环境的数据隐私。然而，在联合学习下，由于本地数据对服务器是不可访问的，攻击者很容易毒化本地客户端的训练数据，在没有通知的情况下在代理中构建后门。部署这样的代理会增加对人类造成潜在伤害的风险，因为攻击者可以很容易地通过后门导航和控制代理。针对拜占庭稳健的联邦具身智能体学习，本文研究了视觉语言导航(VLN)任务的攻防问题，该任务要求智能体遵循自然语言指令在室内环境中导航。首先，我们介绍了一种简单但有效的攻击策略，即希望导航(NAW)，在该策略中，恶意客户端操纵局部轨迹数据，在全局模型中植入后门。在两个VLN数据集(R2R和RXR)上的结果表明，NAW可以轻松地导航部署的VLN代理，而不会影响其在正常测试集上的性能。然后，我们提出了一种新的基于提示的聚合(PBA)来防御联邦VLN中的NAW攻击，它为服务器提供了良性客户端和恶意客户端之间视觉和语言对齐差异的“提示”，以便在训练过程中区分它们。我们验证了PBA方法在保护全局模型免受NAW攻击方面的有效性，在R2R和RXR上的防御指标上远远超过了其他最先进的防御方法。



## **27. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于良知的结盟框架对抗敌意攻击 cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.00029v2) [paper-pdf](http://arxiv.org/pdf/2312.00029v2)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. These attacks can trick seemingly aligned models into giving manufacturing instructions for dangerous materials, inciting violence, or recommending other immoral acts. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM emulating the conscience of a protected, primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis shows that, by using Bergeron to complement models with existing alignment training, we can improve the robustness and safety of multiple, commonly used commercial and open-source LLMs.

摘要: 自从最近引入了越来越强大的大型语言模型（LLM）以来，对人工智能对齐的研究已经有了很大的增长。不幸的是，现代对齐方法仍然无法完全防止模型受到蓄意攻击时的有害反应。这些攻击可以欺骗看似一致的模型给出危险材料的制造指令，煽动暴力，或推荐其他不道德行为。为了帮助缓解这个问题，我们引入了Bergeron：一个旨在提高LLM抵抗攻击的鲁棒性的框架，而无需进行任何额外的参数微调。Bergeron分为两个层次；二级法学硕士模仿受保护的，主要法学硕士的良心。该框架更好地保护了主模型免受传入攻击，同时监控其输出中的任何有害内容。实证分析表明，通过使用Bergeron来补充现有的对齐训练模型，我们可以提高多个常用的商业和开源LLM的鲁棒性和安全性。



## **28. Beyond Gradient and Priors in Privacy Attacks: Leveraging Pooler Layer Inputs of Language Models in Federated Learning**

隐私攻击中的超越梯度和先验：在联邦学习中利用语言模型的Poetary层输入 cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.05720v4) [paper-pdf](http://arxiv.org/pdf/2312.05720v4)

**Authors**: Jianwei Li, Sheng Liu, Qi Lei

**Abstract**: Language models trained via federated learning (FL) demonstrate impressive capabilities in handling complex tasks while protecting user privacy. Recent studies indicate that leveraging gradient information and prior knowledge can potentially reveal training samples within FL setting. However, these investigations have overlooked the potential privacy risks tied to the intrinsic architecture of the models. This paper presents a two-stage privacy attack strategy that targets the vulnerabilities in the architecture of contemporary language models, significantly enhancing attack performance by initially recovering certain feature directions as additional supervisory signals. Our comparative experiments demonstrate superior attack performance across various datasets and scenarios, highlighting the privacy leakage risk associated with the increasingly complex architectures of language models. We call for the community to recognize and address these potential privacy risks in designing large language models.

摘要: 通过联邦学习（FL）训练的语言模型在处理复杂任务的同时保护用户隐私方面表现出令人印象深刻的能力。最近的研究表明，利用梯度信息和先验知识可以潜在地揭示FL设置中的训练样本。然而，这些调查忽略了与模型内在架构相关的潜在隐私风险。本文提出了一种两阶段隐私攻击策略，针对当代语言模型体系结构中的漏洞，通过初始恢复某些特征方向作为额外的监督信号，显着提高攻击性能。我们的对比实验证明了在各种数据集和场景中的卓越攻击性能，突出了与语言模型日益复杂的架构相关的隐私泄露风险。我们呼吁社区在设计大型语言模型时认识到并解决这些潜在的隐私风险。



## **29. Logits of API-Protected LLMs Leak Proprietary Information**

API保护的LLMS日志泄露专有信息 cs.CL

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.09539v2) [paper-pdf](http://arxiv.org/pdf/2403.09539v2)

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta

**Abstract**: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1,000 for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We show that this lends itself to a model image or a model signature which unlocks several capabilities with affordable cost: efficiently discovering the LLM's hidden size, obtaining full-vocabulary outputs, detecting and disambiguating different model updates, identifying the source LLM given a single full LLM output, and even estimating the output layer parameters. Our empirical investigations show the effectiveness of our methods, which allow us to estimate the embedding size of OpenAI's gpt-3.5-turbo to be about 4,096. Lastly, we discuss ways that LLM providers can guard against these attacks, as well as how these capabilities can be viewed as a feature (rather than a bug) by allowing for greater transparency and accountability.

摘要: 大型语言模型（LLM）的商业化导致了只有高级API访问专有模型的普遍做法。在这项工作中，我们表明，即使有一个保守的假设模型架构，它是可能的，从相对较少的API查询（例如，OpenAI的gpt—3.5—turbo成本低于1000美元）。我们的研究结果集中在一个关键观察：大多数现代LLM都存在softmax瓶颈，这将模型输出限制在完整输出空间的线性子空间。我们表明，这有助于自己的模型图像或模型签名解锁几个功能，以负担得起的成本：有效地发现LLM的隐藏大小，获得完整的词汇表输出，检测和消除不同的模型更新，识别源LLM给定一个完整的LLM输出，甚至估计输出层参数。我们的实证研究表明了我们方法的有效性，这使我们能够估计OpenAI的gpt—3.5—turbo的嵌入大小约为4，096。最后，我们讨论了LLM提供商可以防范这些攻击的方法，以及如何通过允许更大的透明度和问责制将这些功能视为一个功能（而不是一个bug）。



## **30. Scaling Behavior of Machine Translation with Large Language Models under Prompt Injection Attacks**

即时注入攻击下大语言模型机器翻译的缩放行为 cs.CL

15 pages, 18 figures, First Workshop on the Scaling Behavior of Large  Language Models (SCALE-LLM 2024)

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09832v1) [paper-pdf](http://arxiv.org/pdf/2403.09832v1)

**Authors**: Zhifan Sun, Antonio Valerio Miceli-Barone

**Abstract**: Large Language Models (LLMs) are increasingly becoming the preferred foundation platforms for many Natural Language Processing tasks such as Machine Translation, owing to their quality often comparable to or better than task-specific models, and the simplicity of specifying the task through natural language instructions or in-context examples. Their generality, however, opens them up to subversion by end users who may embed into their requests instructions that cause the model to behave in unauthorized and possibly unsafe ways. In this work we study these Prompt Injection Attacks (PIAs) on multiple families of LLMs on a Machine Translation task, focusing on the effects of model size on the attack success rates. We introduce a new benchmark data set and we discover that on multiple language pairs and injected prompts written in English, larger models under certain conditions may become more susceptible to successful attacks, an instance of the Inverse Scaling phenomenon (McKenzie et al., 2023). To our knowledge, this is the first work to study non-trivial LLM scaling behaviour in a multi-lingual setting.

摘要: 大型语言模型（LLM）正日益成为许多自然语言处理任务（如机器翻译）的首选基础平台，因为它们的质量通常与特定任务的模型相当或更好，以及通过自然语言指令或上下文示例指定任务的简单性。然而，它们的普遍性使最终用户可能会在请求中嵌入指令，导致模型以未经授权的和可能不安全的方式运行。在这项工作中，我们研究了机器翻译任务中对多个家庭的LLM的提示注入攻击（PIA），重点是模型大小对攻击成功率的影响。我们引入了一个新的基准数据集，我们发现，在多个语言对和注入用英语编写的提示符上，在某些条件下，更大的模型可能变得更容易受到成功的攻击，这是逆尺度现象的一个例子（McKenzie等人，2023年）。据我们所知，这是第一项研究多语言环境下非平凡LLM缩放行为的工作。



## **31. Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models**

图像是对齐的致命弱点：利用多模态大型语言模型的视觉漏洞 cs.CV

Work in progress

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09792v1) [paper-pdf](http://arxiv.org/pdf/2403.09792v1)

**Authors**: Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen

**Abstract**: In this paper, we study the harmlessness alignment problem of multimodal large language models~(MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate~(ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.

摘要: 本文研究了多模态大语言模型MLLM的无害对齐问题.我们对典型的多线性线性阵列的无害性性能进行了系统的实证分析，揭示了图像输入造成了多线性阵列的对准脆弱性。受此启发，我们提出了一种名为HADES的新越狱方法，该方法使用精心制作的图像隐藏和放大了文本输入中恶意意图的危害性。实验结果表明，HADES可以有效地破解现有MLLM，LLaVA—1.5和Gemini Pro Vision的平均攻击成功率分别为90.26%和71.60%。我们的代码和数据将公开发布。



## **32. AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting**

AdaShield：通过自适应护盾保护多模态大型语言模型免受基于结构的攻击 cs.CR

Multimodal Large Language Models Defense, 25 Pages

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09513v1) [paper-pdf](http://arxiv.org/pdf/2403.09513v1)

**Authors**: Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, Chaowei Xiao

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), the imperative to ensure their safety has become increasingly pronounced. However, with the integration of additional modalities, MLLMs are exposed to new vulnerabilities, rendering them prone to structured-based jailbreak attacks, where semantic content (e.g., "harmful text") has been injected into the images to mislead MLLMs. In this work, we aim to defend against such threats. Specifically, we propose \textbf{Ada}ptive \textbf{Shield} Prompting (\textbf{AdaShield}), which prepends inputs with defense prompts to defend MLLMs against structure-based jailbreak attacks without fine-tuning MLLMs or training additional modules (e.g., post-stage content detector). Initially, we present a manually designed static defense prompt, which thoroughly examines the image and instruction content step by step and specifies response methods to malicious queries. Furthermore, we introduce an adaptive auto-refinement framework, consisting of a target MLLM and a LLM-based defense prompt generator (Defender). These components collaboratively and iteratively communicate to generate a defense prompt. Extensive experiments on the popular structure-based jailbreak attacks and benign datasets show that our methods can consistently improve MLLMs' robustness against structure-based jailbreak attacks without compromising the model's general capabilities evaluated on standard benign tasks. Our code is available at https://github.com/rain305f/AdaShield.

摘要: 随着多模态大型语言模型（MLLM）的出现和广泛部署，确保其安全性的迫切性变得越来越明显。然而，随着附加模态的集成，MLLM暴露于新的漏洞，使其易于遭受基于结构化的越狱攻击，其中语义内容（例如，“有害文字”）被注入图像误导MLLM。在这项工作中，我们的目标是防范此类威胁。具体地说，我们提出了\textbf {Ada} practiced\textbf {Shield}（\textbf {AdaShield}），它在输入前加上防御提示，以保护MLLM免受基于结构的越狱攻击，而无需微调MLLM或训练额外模块（例如，后阶段内容检测器）。首先，我们提出了一个手动设计的静态防御提示，它彻底检查图像和指令内容一步一步，并指定响应方法的恶意查询。此外，我们引入了一个自适应的自动细化框架，由目标MLLM和基于LLM的防御提示生成器（Defender）组成。这些组件协作地、迭代地通信以生成防御提示。在流行的基于结构的越狱攻击和良性数据集上的大量实验表明，我们的方法可以持续提高MLLM对基于结构的越狱攻击的鲁棒性，而不损害模型在标准良性任务上评估的一般能力。我们的代码可在www.example.com获得。



## **33. On Protecting the Data Privacy of Large Language Models (LLMs): A Survey**

大型语言模型（LLM）数据隐私保护研究综述 cs.CR

18 pages, 4 figures

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.05156v2) [paper-pdf](http://arxiv.org/pdf/2403.05156v2)

**Authors**: Biwei Yan, Kun Li, Minghui Xu, Yueyan Dong, Yue Zhang, Zhaochun Ren, Xiuzhen Cheng

**Abstract**: Large language models (LLMs) are complex artificial intelligence systems capable of understanding, generating and translating human language. They learn language patterns by analyzing large amounts of text data, allowing them to perform writing, conversation, summarizing and other language tasks. When LLMs process and generate large amounts of data, there is a risk of leaking sensitive information, which may threaten data privacy. This paper concentrates on elucidating the data privacy concerns associated with LLMs to foster a comprehensive understanding. Specifically, a thorough investigation is undertaken to delineate the spectrum of data privacy threats, encompassing both passive privacy leakage and active privacy attacks within LLMs. Subsequently, we conduct an assessment of the privacy protection mechanisms employed by LLMs at various stages, followed by a detailed examination of their efficacy and constraints. Finally, the discourse extends to delineate the challenges encountered and outline prospective directions for advancement in the realm of LLM privacy protection.

摘要: 大型语言模型（LLM）是一种复杂的人工智能系统，能够理解、生成和翻译人类语言。他们通过分析大量的文本数据来学习语言模式，使他们能够执行写作、会话、总结和其他语言任务。当LLM处理和生成大量数据时，存在泄漏敏感信息的风险，这可能威胁到数据隐私。本文集中阐述与LLM相关的数据隐私问题，以促进全面理解。具体而言，我们进行了彻底的调查，以界定数据隐私威胁的范围，包括LLM内的被动隐私泄漏和主动隐私攻击。其后，我们会评估有限责任公司在不同阶段所采用的隐私保障机制，并详细研究其成效及限制。最后，论述扩展到描绘所遇到的挑战，并概述在法学硕士隐私保护领域的发展方向。



## **34. AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions**

AVIBENCH：对抗视觉指令上大型视觉语言模型的鲁棒性评估 cs.CV

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09346v1) [paper-pdf](http://arxiv.org/pdf/2403.09346v1)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in well responding to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce AVIBench, a framework designed to analyze the robustness of LVLMs when facing various adversarial visual-instructions (AVIs), including four types of image-based AVIs, ten types of text-based AVIs, and nine types of content bias AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 260K AVIs encompassing five categories of multimodal capabilities (nine tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark will be made publicly available.

摘要: 大型视觉语言模型（LVLM）在很好地响应用户的视觉指令方面取得了显著进展。然而，这些包含图像和文本的指令很容易受到有意和无意的攻击。尽管LVLM对此类威胁的鲁棒性至关重要，但目前在该领域的研究仍然有限。为了弥补这一差距，我们引入了AVIBench，这是一个框架，旨在分析LVLM在面对各种对抗视觉指令（AVIs）时的鲁棒性，包括四种基于图像的AVIs，十种基于文本的AVIs和九种内容偏见的AVIs（如性别、暴力、文化和种族偏见等）。我们生成了260K AVI，涵盖了五个类别的多模态能力（九个任务）和内容偏见。然后，我们对14个开源LVLM进行了全面的评估，以评估它们的性能。AVIBench也是一个方便的工具，为从业者评估LVLM的鲁棒性对AVIs。我们的发现和广泛的实验结果揭示了LVLM的脆弱性，并强调即使在先进的闭源LVLM，如GeminiProVision和GPT—4V中也存在固有的偏见。这强调了增强LVLM鲁棒性、安全性和公平性的重要性。源代码和基准测试将公开。



## **35. What Was Your Prompt? A Remote Keylogging Attack on AI Assistants**

你的提示是什么？对人工智能助手的远程键盘记录攻击 cs.CR

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09751v1) [paper-pdf](http://arxiv.org/pdf/2403.09751v1)

**Authors**: Roy Weiss, Daniel Ayzenshteyn, Guy Amit, Yisroel Mirsky

**Abstract**: AI assistants are becoming an integral part of society, used for asking advice or help in personal and confidential issues. In this paper, we unveil a novel side-channel that can be used to read encrypted responses from AI Assistants over the web: the token-length side-channel. We found that many vendors, including OpenAI and Microsoft, have this side-channel.   However, inferring the content of a response from a token-length sequence alone proves challenging. This is because tokens are akin to words, and responses can be several sentences long leading to millions of grammatically correct sentences. In this paper, we show how this can be overcome by (1) utilizing the power of a large language model (LLM) to translate these sequences, (2) providing the LLM with inter-sentence context to narrow the search space and (3) performing a known-plaintext attack by fine-tuning the model on the target model's writing style.   Using these methods, we were able to accurately reconstruct 29\% of an AI assistant's responses and successfully infer the topic from 55\% of them. To demonstrate the threat, we performed the attack on OpenAI's ChatGPT-4 and Microsoft's Copilot on both browser and API traffic.

摘要: 人工智能助理正在成为社会不可或缺的一部分，用于在个人和机密问题上寻求建议或帮助。在本文中，我们推出了一种新的侧通道，可用于从Web上读取AI助手的加密响应：令牌长度侧通道。我们发现，包括OpenAI和微软在内的许多供应商都有这种侧通道。   然而，仅从标记长度序列推断响应的内容证明具有挑战性。这是因为令牌类似于单词，响应可以是几个句子长，导致数百万个语法正确的句子。在本文中，我们展示了如何克服这一点，（1）利用一个大型语言模型（LLM）的权力来翻译这些序列，（2）提供LLM与句子间上下文，以缩小搜索空间和（3）执行一个已知的明文攻击，微调模型的目标模型的写作风格。   使用这些方法，我们能够准确地重建29%的人工智能助手的回答，并成功地从其中55%的回答中推断出主题。为了演示威胁，我们对OpenAI的ChatGPT—4和微软的Copilot进行了浏览器和API流量的攻击。



## **36. The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?**

第一个知道：在大型视觉语言模型中，令牌分布如何揭示隐藏的知识？ cs.CV

Under review. Project page:  https://github.com/Qinyu-Allen-Zhao/LVLM-LP

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09037v1) [paper-pdf](http://arxiv.org/pdf/2403.09037v1)

**Authors**: Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould

**Abstract**: Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layer of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against multi-modal jailbreaking attack, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, indicating potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicting uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improve models' performance but is still inferior to linear probing on these tasks.

摘要: 大型视觉语言模型(LVLM)旨在解释和响应人类的指令，有时会由于不适当的指令而产生幻觉或有害内容。本研究采用线性探测的方法，揭示了LVLMS输出层的隐含知识。我们证明了第一个令牌的Logit分布包含了足够的信息来确定是否响应指令，包括识别无法回答的可视问题、防御多模式越狱攻击和识别欺骗性问题。在响应生成期间，这种隐藏的知识在后续令牌的登录中逐渐丢失。然后，我们在生成第一个令牌时说明了一种简单的解码策略，有效地改进了生成的内容。在实验中，我们发现了一些有趣的见解：首先，CLIP模型已经包含了解决这些任务的强烈信号，表明现有数据集中存在潜在的偏差。其次，我们通过利用第一个Logit分布在另外三个任务上观察到性能的提高，包括指示数学解决中的不确定性、减轻幻觉和图像分类。最后，在相同的训练数据下，简单的微调LVLM可以提高模型的性能，但在这些任务上仍然不如线性探测。



## **37. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

Jekyll博士和海德先生：法学硕士的两个面孔 cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.03853v2) [paper-pdf](http://arxiv.org/pdf/2312.03853v2)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Only a year ago, we witnessed a rise in the use of Large Language Models (LLMs), especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 就在一年前，我们见证了大型语言模型(LLM)的使用增加，特别是在与Chatbot助手等应用程序结合时。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Bard(在某种程度上，还有Bing聊天)的这些措施，让他们模仿复杂的人物角色，具有与他们应该是的诚实助手相反的特征。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。我们的谈话遵循了角色扮演的风格，得到了助手不允许提供的回应。通过使用人物角色，我们表明实际上提供了被禁止的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用对抗性人物角色，一个人可以克服ChatGPT和Bard提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，总而言之，这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **38. SoK: Reducing the Vulnerability of Fine-tuned Language Models to Membership Inference Attacks**

SoK：降低精细调优语言模型对成员推断攻击的脆弱性 cs.LG

preliminary version

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08481v1) [paper-pdf](http://arxiv.org/pdf/2403.08481v1)

**Authors**: Guy Amit, Abigail Goldsteen, Ariel Farkash

**Abstract**: Natural language processing models have experienced a significant upsurge in recent years, with numerous applications being built upon them. Many of these applications require fine-tuning generic base models on customized, proprietary datasets. This fine-tuning data is especially likely to contain personal or sensitive information about individuals, resulting in increased privacy risk. Membership inference attacks are the most commonly employed attack to assess the privacy leakage of a machine learning model. However, limited research is available on the factors that affect the vulnerability of language models to this kind of attack, or on the applicability of different defense strategies in the language domain. We provide the first systematic review of the vulnerability of fine-tuned large language models to membership inference attacks, the various factors that come into play, and the effectiveness of different defense strategies. We find that some training methods provide significantly reduced privacy risk, with the combination of differential privacy and low-rank adaptors achieving the best privacy protection against these attacks.

摘要: 近年来，自然语言处理模型经历了一个显著的热潮，许多应用程序都是基于它们构建的。其中许多应用程序需要在定制的专有数据集上微调通用基础模型。这种微调数据特别有可能包含个人或敏感信息，从而增加隐私风险。隶属度推理攻击是评估机器学习模型隐私泄漏最常用的攻击。然而，关于影响语言模型对此类攻击的脆弱性的因素，以及不同防御策略在语言领域的适用性的研究有限。我们提供了第一个系统的审查微调大型语言模型的脆弱性，参与的各种因素，以及不同防御策略的有效性。我们发现，一些训练方法提供了显着降低的隐私风险，差分隐私和低秩适配器的组合实现了针对这些攻击的最佳隐私保护。



## **39. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

哲学家之石：大型语言模型的木马插件 cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.00374v2) [paper-pdf](http://arxiv.org/pdf/2312.00374v2)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Shaofeng Li, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can execute malware to control system (e.g., LLM-driven robot) or launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we design and evaluate three potential defenses, yet none proved entirely effective in safeguarding against our attacks.

摘要: 开源大型语言模型（LLM）最近受到了欢迎，因为它们与专有LLM相当的性能。为了有效地完成领域专用任务，开源LLM可以使用低秩适配器进行优化，而无需昂贵的加速器。然而，目前还不清楚是否可以利用低秩适配器来控制LLM。为了解决这一差距，我们证明了受感染的适配器可以诱导，在特定的触发器，LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击，POLISHED和FUSION，改进了先前的方法。POLISHED使用LLM增强的释义来抛光基准中毒数据集。相反，在没有数据集的情况下，FUSION利用过度中毒过程来转换良性适配器。在我们的实验中，我们首先进行了两个案例研究，以证明一个受损的LLM代理可以执行恶意软件来控制系统（例如，LLM驱动的机器人）或发起鱼叉式网络钓鱼攻击。然后，在有针对性的错误信息方面，我们表明我们的攻击提供了比基线更高的攻击效果，并且为了吸引下载，保留或改进适配器的实用性。最后，我们设计并评估了三种潜在的防御措施，但没有一种被证明完全有效地防御我们的攻击。



## **40. Tastle: Distract Large Language Models for Automatic Jailbreak Attack**

Tastle：分散大型语言模型的自动越狱攻击 cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08424v1) [paper-pdf](http://arxiv.org/pdf/2403.08424v1)

**Authors**: Zeguan Xiao, Yan Yang, Guanhua Chen, Yun Chen

**Abstract**: Large language models (LLMs) have achieved significant advances in recent days. Extensive efforts have been made before the public release of LLMs to align their behaviors with human values. The primary goal of alignment is to ensure their helpfulness, honesty and harmlessness. However, even meticulously aligned LLMs remain vulnerable to malicious manipulations such as jailbreaking, leading to unintended behaviors. The jailbreak is to intentionally develop a malicious prompt that escapes from the LLM security restrictions to produce uncensored detrimental contents. Previous works explore different jailbreak methods for red teaming LLMs, yet they encounter challenges regarding to effectiveness and scalability. In this work, we propose Tastle, a novel black-box jailbreak framework for automated red teaming of LLMs. We designed malicious content concealing and memory reframing with an iterative optimization algorithm to jailbreak LLMs, motivated by the research about the distractibility and over-confidence phenomenon of LLMs. Extensive experiments of jailbreaking both open-source and proprietary LLMs demonstrate the superiority of our framework in terms of effectiveness, scalability and transferability. We also evaluate the effectiveness of existing jailbreak defense methods against our attack and highlight the crucial need to develop more effective and practical defense strategies.

摘要: 最近几天，大型语言模型(LLM)取得了重大进展。在公开发布LLM之前，已经做出了广泛的努力，以使它们的行为符合人类的价值观。联合的主要目标是确保他们的帮助、诚实和无害。然而，即使经过精心调整的LLM仍然容易受到恶意操作(如越狱)的攻击，从而导致意外行为。越狱是为了故意开发一个恶意提示，以逃避LLM的安全限制，生成未经审查的有害内容。以前的工作探索了不同的红色团队LLM越狱方法，但在有效性和可扩展性方面遇到了挑战。在这项工作中，我们提出了一种新的黑盒越狱框架Tastle，用于LLM的自动红色团队。基于对LLMS的分心和过度自信现象的研究，我们设计了恶意的内容隐藏和记忆重组，并用迭代优化算法实现了LLMS的越狱。大量的开源和专有LLMS越狱实验证明了该框架在有效性、可扩展性和可转移性方面的优越性。我们还评估了针对我们的攻击的现有越狱防御方法的有效性，并强调了开发更有效和更实用的防御战略的迫切需要。



## **41. Duwak: Dual Watermarks in Large Language Models**

Duwak：大型语言模型中的双水印 cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.13000v1) [paper-pdf](http://arxiv.org/pdf/2403.13000v1)

**Authors**: Chaoyi Zhu, Jeroen Galjaard, Pin-Yu Chen, Lydia Y. Chen

**Abstract**: As large language models (LLM) are increasingly used for text generation tasks, it is critical to audit their usages, govern their applications, and mitigate their potential harms. Existing watermark techniques are shown effective in embedding single human-imperceptible and machine-detectable patterns without significantly affecting generated text quality and semantics. However, the efficiency in detecting watermarks, i.e., the minimum number of tokens required to assert detection with significance and robustness against post-editing, is still debatable. In this paper, we propose, Duwak, to fundamentally enhance the efficiency and quality of watermarking by embedding dual secret patterns in both token probability distribution and sampling schemes. To mitigate expression degradation caused by biasing toward certain tokens, we design a contrastive search to watermark the sampling scheme, which minimizes the token repetition and enhances the diversity. We theoretically explain the interdependency of the two watermarks within Duwak. We evaluate Duwak extensively on Llama2 under various post-editing attacks, against four state-of-the-art watermarking techniques and combinations of them. Our results show that Duwak marked text achieves the highest watermarked text quality at the lowest required token count for detection, up to 70% tokens less than existing approaches, especially under post paraphrasing.

摘要: 随着大型语言模型（LLM）越来越多地用于文本生成任务，审计它们的使用，管理它们的应用程序，并减轻它们的潜在危害至关重要。现有的水印技术被证明是有效的嵌入单一的人类感知不到和机器可检测的模式，而不会显着影响生成的文本质量和语义。然而，检测水印的效率，即，对于后期编辑，断言具有重要性和鲁棒性的检测所需的最小数量的令牌仍然是有争议的。在本文中，我们提出，Duwak，通过在令牌概率分布和采样方案中嵌入双重秘密模式，从根本上提高水印的效率和质量。为了减轻由于偏向特定令牌而导致的表情退化，我们设计了一种对比搜索来水印采样方案，最大限度地减少令牌重复，增强了多样性。我们从理论上解释了杜瓦克内部两个水印的相互依赖性。我们评估Duwak广泛的Llama2下各种后期编辑攻击，针对四个国家的最先进的水印技术及其组合。我们的研究结果表明，Duwak标记的文本在检测所需的最低令牌计数下达到了最高的水印文本质量，比现有方法少了多达70%的令牌，特别是在后释义。



## **42. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

MM—SafetyBench：多模态大型语言模型安全性评估的基准 cs.CV

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2311.17600v2) [paper-pdf](http://arxiv.org/pdf/2311.17600v2)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

摘要: 围绕大型语言模型（LLM）的安全问题已经得到了广泛的探讨，但多模态大型语言模型（MLLM）的安全性研究仍然不足。在本文中，我们观察到，多模态大型语言模型（MLLM）可以很容易地被查询相关图像破坏，就好像文本查询本身是恶意的。为了解决这一问题，我们引入了MM—SafetyBench，一个综合框架，旨在针对此类基于图像的操纵对MLLM进行安全关键评估。我们编译了一个包含13个场景的数据集，总共产生了5，040个文本图像对。我们对12种最先进型号的分析表明，即使装备的LLM已经安全对准，MLLM也容易受到我们的方法引发的违规行为的影响。作为回应，我们提出了一个简单而有效的激励策略，以提高MLLM对这些类型的攻击的弹性。我们的工作强调了共同努力的必要性，以加强和提高开源MLLM的安全措施，以防止潜在的恶意利用。该资源可在\href {this https URL}{https：//github.com/isXinLiu/MM—SafetyBench}获得。



## **43. Poisoning Programs by Un-Repairing Code: Security Concerns of AI-generated Code**

不修复代码使程序中毒：人工智能生成代码的安全问题 cs.CR

Accepted at The 1st IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE), co-located with ISSRE 2023

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06675v1) [paper-pdf](http://arxiv.org/pdf/2403.06675v1)

**Authors**: Cristina Improta

**Abstract**: AI-based code generators have gained a fundamental role in assisting developers in writing software starting from natural language (NL). However, since these large language models are trained on massive volumes of data collected from unreliable online sources (e.g., GitHub, Hugging Face), AI models become an easy target for data poisoning attacks, in which an attacker corrupts the training data by injecting a small amount of poison into it, i.e., astutely crafted malicious samples. In this position paper, we address the security of AI code generators by identifying a novel data poisoning attack that results in the generation of vulnerable code. Next, we devise an extensive evaluation of how these attacks impact state-of-the-art models for code generation. Lastly, we discuss potential solutions to overcome this threat.

摘要: 基于人工智能的代码生成器在帮助开发人员从自然语言(NL)开始编写软件方面发挥了重要作用。然而，由于这些大型语言模型是基于从不可靠的在线来源(如GitHub、拥抱脸)收集的海量数据进行训练的，AI模型很容易成为数据中毒攻击的目标，即攻击者通过向训练数据中注入少量毒药来破坏训练数据，即巧妙地制作恶意样本。在这份立场文件中，我们通过识别一种导致生成易受攻击的代码的新型数据中毒攻击来解决AI代码生成器的安全问题。接下来，我们将对这些攻击如何影响最先进的代码生成模型进行广泛的评估。最后，我们讨论了克服这一威胁的潜在解决方案。



## **44. FedPIT: Towards Privacy-preserving and Few-shot Federated Instruction Tuning**

FedPIT：面向隐私保护和少镜头联合指令调优 cs.CR

Work in process

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.06131v1) [paper-pdf](http://arxiv.org/pdf/2403.06131v1)

**Authors**: Zhuo Zhang, Jingyuan Zhang, Jintao Huang, Lizhen Qu, Hongzhi Zhang, Zenglin Xu

**Abstract**: Instruction tuning has proven essential for enhancing the performance of large language models (LLMs) in generating human-aligned responses. However, collecting diverse, high-quality instruction data for tuning poses challenges, particularly in privacy-sensitive domains. Federated instruction tuning (FedIT) has emerged as a solution, leveraging federated learning from multiple data owners while preserving privacy. Yet, it faces challenges due to limited instruction data and vulnerabilities to training data extraction attacks. To address these issues, we propose a novel federated algorithm, FedPIT, which utilizes LLMs' in-context learning capability to self-generate task-specific synthetic data for training autonomously. Our method employs parameter-isolated training to maintain global parameters trained on synthetic data and local parameters trained on augmented local data, effectively thwarting data extraction attacks. Extensive experiments on real-world medical data demonstrate the effectiveness of FedPIT in improving federated few-shot performance while preserving privacy and robustness against data heterogeneity.

摘要: 指令调优已被证明对于提高大型语言模型（LLM）在生成与人类一致的响应方面的性能至关重要。然而，收集不同的高质量指令数据进行调整带来了挑战，特别是在隐私敏感领域。联邦指令调优（FedIT）已经成为一种解决方案，它利用来自多个数据所有者的联合学习，同时保护隐私。然而，由于指令数据有限和训练数据提取攻击的脆弱性，它面临着挑战。为了解决这些问题，我们提出了一种新的联邦算法，FedPIT，它利用LLM的上下文学习能力，自生成特定于任务的合成数据，用于自主训练。我们的方法采用参数隔离训练来维护在合成数据上训练的全局参数和在增强本地数据上训练的局部参数，有效地阻止了数据提取攻击。对现实世界医疗数据的大量实验证明了FedPIT在提高联邦少镜头性能的同时保护隐私性和针对数据异构性的鲁棒性方面的有效性。



## **45. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

零镜头对抗鲁棒性的迭代驱动算法 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2301.13096v3) [paper-pdf](http://arxiv.org/pdf/2301.13096v3)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.

摘要: 深度神经网络(DNN)是公认的易受敌意攻击的网络。以往的研究主要集中在提高完全监督环境下的对抗稳健性，而对零射击对抗稳健性这一挑战领域的研究还是个未知数。在这项工作中，我们通过利用大型视觉语言模型(如CLIP)的最新进展来研究这一领域，为DNN引入零射击对抗性健壮性。我们提出了LAAT，一种语言驱动的、基于锚的对抗性训练策略。LAAT利用每个类别的文本编码器的特征作为每个类别的固定锚(归一化特征嵌入)，然后将其用于对抗性训练。通过利用文本编码者的语义一致性，LAAT旨在增强图像模型在新类别上的对抗性健壮性。然而，幼稚地使用文本编码器会导致较差的结果。通过分析，我们认为问题在于文本编码者之间存在很高的余弦相似度。然后，我们设计了扩展算法和对齐交叉熵损失来缓解该问题。我们的实验结果表明，与最先进的方法相比，LAAT显著提高了零命中对手的稳健性。LAAT有可能通过大规模多模式模型来增强对手的稳健性，特别是在训练过程中无法获得标记数据的情况下。



## **46. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

从聊天机器人到PhishBots？——防止使用ChatGPT、Google Bard和Claude创建的网络钓鱼诈骗 cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.19181v2) [paper-pdf](http://arxiv.org/pdf/2310.19181v2)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs, i.e., ChatGPT (GPT 3.5 Turbo), GPT 4, Claude, and Bard, to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing websites and emails that can convincingly imitate well-known brands and also deploy a range of evasive tactics that are used to elude detection mechanisms employed by anti-phishing systems. These attacks can be generated using unmodified or "vanilla" versions of these LLMs without requiring any prior adversarial exploits such as jailbreaking. We evaluate the performance of the LLMs towards generating these attacks and find that they can also be utilized to create malicious prompts that, in turn, can be fed back to the model to generate phishing scams - thus massively reducing the prompt-engineering effort required by attackers to scale these threats. As a countermeasure, we build a BERT-based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content. Our model is transferable across all four commercial LLMs, attaining an average accuracy of 96% for phishing website prompts and 94% for phishing email prompts. We also disclose the vulnerabilities to the concerned LLMs, with Google acknowledging it as a severe issue. Our detection model is available for use at Hugging Face, as well as a ChatGPT Actions plugin.

摘要: 大型语言模型（LLM）的先进功能使它们在各种应用程序中发挥了非常重要的作用，从会话代理和内容创建到数据分析、研究和创新。然而，它们的有效性和可访问性也使它们容易被滥用以生成恶意内容，包括网络钓鱼攻击。本研究探讨了使用四种流行的商业化LLM的潜力，即，ChatGPT（GPT 3.5 Turbo）、GPT 4、Claude和Bard，使用一系列恶意提示生成功能性网络钓鱼攻击。我们发现，这些LLM可以生成钓鱼网站和电子邮件，可以令人信服地模仿知名品牌，还部署了一系列逃避策略，用于逃避反钓鱼系统采用的检测机制。这些攻击可以使用这些LLM的未修改或“vanilla”版本生成，而不需要任何先前的对抗性攻击，如越狱。我们评估了LLM在生成这些攻击方面的性能，发现它们还可以用来创建恶意提示，反过来，这些提示可以反馈到模型中生成网络钓鱼诈骗，从而大大减少了攻击者扩展这些威胁所需的网络设计工作。作为一种对策，我们构建了一个基于BERT的自动检测工具，用于早期检测恶意提示，以防止LLM生成钓鱼内容。我们的模型可在所有四个商业LLM中移植，钓鱼网站提示的平均准确率为96%，钓鱼电子邮件提示的平均准确率为94%。我们还向相关LLM披露了这些漏洞，谷歌承认这是一个严重的问题。我们的检测模型可用于Hugging Face，以及ChatGPT Action插件。



## **47. Can LLMs Follow Simple Rules?**

低收入国家能遵循简单的规则吗？ cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.

摘要: 随着大型语言模型（LLM）的部署与日益增加的现实世界责任，重要的是能够以可靠的方式指定和约束这些系统的行为。模型开发者可能希望为模型设置明确的规则，例如“不要生成滥用内容”，但这些规则可能会被越狱技术绕过。现有的对LLM的对抗性攻击和防御的评估通常需要昂贵的手动审查或不可靠的启发式检查。为了解决这个问题，我们提出了规则遵循语言评估方案（RuLES），一个用于测量LLM规则遵循能力的程序框架。RuLES由14个简单的文本场景组成，在这些场景中，模型被指示在与用户交互时遵守各种规则。每个场景都有一个程序化的评估功能，以确定模型是否违反了会话中的任何规则。我们对私有模型和开放模型的评估表明，几乎所有当前模型都难以遵循场景规则，即使是在简单的测试用例上。我们还证明，简单的优化攻击足以显着提高测试用例的失败率。最后，我们探索了两个潜在的改进途径：测试时间控制和监督微调。



## **48. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

摘要: 人工智能生成内容（AI—Generated Content，AIGC）正在获得广泛的欢迎，有许多新兴的商业服务和应用。这些服务利用先进的生成模型，例如潜在扩散模型和大型语言模型，来生成创造性内容（例如，真实的图像和流畅的句子）。这种生成的内容的使用需要受到高度管制，因为服务提供商需要确保用户不违反使用策略（例如，滥用商业化、生成和分发不安全内容）。实现这一目标的一个很有前途的解决方案是水印，它在内容上添加独特的和不可感知的水印，用于服务验证和归属。近年来，已经提出了许多水印方法。然而，在本文中，我们表明，对手可以很容易地打破这些水印机制。具体来说，我们考虑了两种可能的攻击。(1)水印去除：攻击者可以容易地从生成的内容中擦除嵌入的水印，然后绕过服务提供商的规定而自由地使用它。(2)水印锻造：攻击者可以创建具有来自另一用户的伪造水印的非法内容，导致服务提供商作出错误的归属。我们提出了战争，一个统一的方法，以实现两种攻击在一个整体的方式。关键思想是利用预先训练的扩散模型进行内容处理，并利用生成对抗网络进行水印去除或伪造。我们在不同的数据集和嵌入设置上评估战争。实验结果表明，该方法在保证生成内容质量的同时，可以达到较高的成功率。与现有的基于扩散模型的攻击相比，Warfare的速度快了5050～11000倍。



## **49. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.

摘要: 大型语言模型（LLM）与外部内容的集成使得LLM的应用程序能够实现更新和更广泛，例如Microsoft Copilot。然而，这种集成也使LLM面临间接提示注入攻击的风险，攻击者可以在外部内容中嵌入恶意指令，损害LLM输出并导致响应偏离用户期望。为了研究这个重要但未被充分探索的问题，我们引入了第一个间接提示注入攻击的基准，名为BIPIA，以评估此类攻击的风险。基于评估，我们的工作对攻击成功的根本原因进行了关键分析，即LLM无法区分指令和外部内容，以及LLM缺乏不执行外部内容中的指令的意识。在此基础上，我们开发了两种基于即时学习的黑盒防御方法和一种基于对抗训练的微调的白盒防御方法。实验结果表明，黑盒防御在缓解这些攻击方面非常有效，而白盒防御将攻击成功率降低到接近零的水平。总的来说，我们的工作系统地调查间接提示注入攻击通过引入一个基准，分析攻击成功的潜在原因，并开发一套初始防御。



## **50. SecGPT: An Execution Isolation Architecture for LLM-Based Systems**

SecGPT：一种基于LLM系统的执行隔离体系结构 cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.04960v1) [paper-pdf](http://arxiv.org/pdf/2403.04960v1)

**Authors**: Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, Umar Iqbal

**Abstract**: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more precisely mediate their interactions outside of their isolated environments. We evaluate SecGPT against a number of case study attacks and demonstrate that it protects against many security, privacy, and safety issues that exist in non-isolated LLM-based systems. The performance overhead incurred by SecGPT to improve security is under 0.3x for three-quarters of the tested queries. To foster follow-up research, we release SecGPT's source code at https://github.com/llm-platform-security/SecGPT.

摘要: 作为系统扩展的大型语言模型（LLM），如ChatGPT，已经开始支持第三方应用程序。这些LLM应用程序利用事实上基于自然语言的LLM自动执行范式：即，应用程序及其交互以自然语言定义，提供对用户数据的访问，并允许彼此和系统自由交互。这些LLM应用生态系统类似于早期计算平台的设置，其中应用和系统之间的隔离不够。由于第三方应用程序可能不值得信赖，并且由于自然语言界面的不精确性而加剧，当前的设计给用户带来了安全和隐私风险。在本文中，我们提出了SecGPT，一种基于LLM的系统的架构，旨在缓解第三方应用程序执行时出现的安全和隐私问题。SecGPT的关键思想是隔离应用程序的执行，并更精确地在其隔离环境之外调解它们的交互。我们评估了SecGPT的一些案例研究攻击，并证明它可以防止存在于非隔离的基于LLM的系统中的许多安全、隐私和安全问题。对于四分之三的测试查询，SecGPT为提高安全性而产生的性能开销低于0.3x。为了促进后续研究，我们在www.example.com上发布了SecGPT的源代码。



