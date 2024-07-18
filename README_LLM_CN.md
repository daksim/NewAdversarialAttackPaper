# Latest Large Language Model Attack Papers
**update at 2024-07-18 16:49:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Counterfactual Explainable Incremental Prompt Attack Analysis on Large Language Models**

大型语言模型的反事实可解释增量提示攻击分析 cs.CR

23 pages, 6 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.09292v2) [paper-pdf](http://arxiv.org/pdf/2407.09292v2)

**Authors**: Dong Shu, Mingyu Jin, Tianle Chen, Chong Zhang, Yongfeng Zhang

**Abstract**: This study sheds light on the imperative need to bolster safety and privacy measures in large language models (LLMs), such as GPT-4 and LLaMA-2, by identifying and mitigating their vulnerabilities through explainable analysis of prompt attacks. We propose Counterfactual Explainable Incremental Prompt Attack (CEIPA), a novel technique where we guide prompts in a specific manner to quantitatively measure attack effectiveness and explore the embedded defense mechanisms in these models. Our approach is distinctive for its capacity to elucidate the reasons behind the generation of harmful responses by LLMs through an incremental counterfactual methodology. By organizing the prompt modification process into four incremental levels: (word, sentence, character, and a combination of character and word) we facilitate a thorough examination of the susceptibilities inherent to LLMs. The findings from our study not only provide counterfactual explanation insight but also demonstrate that our framework significantly enhances the effectiveness of attack prompts.

摘要: 这项研究揭示了通过对提示攻击进行解释性分析来识别和缓解大型语言模型(LLM)(如GPT-4和LLAMA-2)中的安全和隐私措施的迫切需要。我们提出了反事实可解释增量提示攻击(CEIPA)，这是一种新的技术，我们以特定的方式引导提示来定量地衡量攻击效果，并探索这些模型中嵌入的防御机制。我们的方法的独特之处在于，它能够通过循序渐进的反事实方法论来阐明小岛屿发展中国家产生有害反应背后的原因。通过将即时修改过程组织成四个递增级别：(单词、句子、字符和字符和单词的组合)，我们有助于彻底检查LLMS固有的易感性。我们的研究结果不仅提供了反事实的解释洞察力，而且证明了我们的框架显着提高了攻击提示的有效性。



## **2. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.06134v2) [paper-pdf](http://arxiv.org/pdf/2405.06134v2)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<|endoftext|>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<|endoftext|>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在其词汇表中加入了“特殊标记”，如$\exttt{<|endoftext|>}$，以指导其语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{}$标记的通用声学实现，当该标记被预先添加到任何语音信号时，鼓励模型忽略语音而只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **3. Security Matrix for Multimodal Agents on Mobile Devices: A Systematic and Proof of Concept Study**

移动设备上多模式代理的安全矩阵：系统性的概念验证研究 cs.CR

Preprint. Work in progress

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.09295v2) [paper-pdf](http://arxiv.org/pdf/2407.09295v2)

**Authors**: Yulong Yang, Xinshan Yang, Shuaidong Li, Chenhao Lin, Zhengyu Zhao, Chao Shen, Tianwei Zhang

**Abstract**: The rapid progress in the reasoning capability of the Multi-modal Large Language Models (MLLMs) has triggered the development of autonomous agent systems on mobile devices. MLLM-based mobile agent systems consist of perception, reasoning, memory, and multi-agent collaboration modules, enabling automatic analysis of user instructions and the design of task pipelines with only natural language and device screenshots as inputs. Despite the increased human-machine interaction efficiency, the security risks of MLLM-based mobile agent systems have not been systematically studied. Existing security benchmarks for agents mainly focus on Web scenarios, and the attack techniques against MLLMs are also limited in the mobile agent scenario. To close these gaps, this paper proposes a mobile agent security matrix covering 3 functional modules of the agent systems. Based on the security matrix, this paper proposes 4 realistic attack paths and verifies these attack paths through 8 attack methods. By analyzing the attack results, this paper reveals that MLLM-based mobile agent systems are not only vulnerable to multiple traditional attacks, but also raise new security concerns previously unconsidered. This paper highlights the need for security awareness in the design of MLLM-based systems and paves the way for future research on attacks and defense methods.

摘要: 多通道大语言模型(MLLMS)在推理能力上的快速发展引发了移动设备上自主代理系统的发展。基于MLLM的移动代理系统由感知、推理、记忆和多代理协作模块组成，支持用户指令的自动分析和任务流水线的设计，只需输入自然语言和设备截图。尽管提高了人机交互效率，但基于MLLM的移动代理系统的安全风险还没有得到系统的研究。现有的代理安全基准测试主要集中在Web场景中，针对移动代理场景中MLLMS的攻击技术也很有限。为了弥补这些差距，本文提出了一种覆盖代理系统3个功能模块的移动代理安全矩阵。基于安全矩阵，提出了4条真实的攻击路径，并通过8种攻击方法对这些攻击路径进行了验证。通过对攻击结果的分析，本文揭示了基于MLLM的移动代理系统不仅容易受到多种传统攻击的攻击，而且还提出了以前没有考虑到的新的安全问题。本文强调了在基于MLLM的系统设计中安全意识的必要性，并为未来攻击和防御方法的研究铺平了道路。



## **4. DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation**

DistillSeq：使用知识蒸馏在大型语言模型中进行安全一致测试的框架 cs.SE

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.10106v2) [paper-pdf](http://arxiv.org/pdf/2407.10106v2)

**Authors**: Mingke Yang, Yuqi Chen, Yi Liu, Ling Shi

**Abstract**: Large Language Models (LLMs) have showcased their remarkable capabilities in diverse domains, encompassing natural language understanding, translation, and even code generation. The potential for LLMs to generate harmful content is a significant concern. This risk necessitates rigorous testing and comprehensive evaluation of LLMs to ensure safe and responsible use. However, extensive testing of LLMs requires substantial computational resources, making it an expensive endeavor. Therefore, exploring cost-saving strategies during the testing phase is crucial to balance the need for thorough evaluation with the constraints of resource availability. To address this, our approach begins by transferring the moderation knowledge from an LLM to a small model. Subsequently, we deploy two distinct strategies for generating malicious queries: one based on a syntax tree approach, and the other leveraging an LLM-based method. Finally, our approach incorporates a sequential filter-test process designed to identify test cases that are prone to eliciting toxic responses. Our research evaluated the efficacy of DistillSeq across four LLMs: GPT-3.5, GPT-4.0, Vicuna-13B, and Llama-13B. In the absence of DistillSeq, the observed attack success rates on these LLMs stood at 31.5% for GPT-3.5, 21.4% for GPT-4.0, 28.3% for Vicuna-13B, and 30.9% for Llama-13B. However, upon the application of DistillSeq, these success rates notably increased to 58.5%, 50.7%, 52.5%, and 54.4%, respectively. This translated to an average escalation in attack success rate by a factor of 93.0% when compared to scenarios without the use of DistillSeq. Such findings highlight the significant enhancement DistillSeq offers in terms of reducing the time and resource investment required for effectively testing LLMs.

摘要: 大型语言模型(LLM)已经在不同的领域展示了它们非凡的能力，包括自然语言理解、翻译，甚至代码生成。低密度脂蛋白产生有害内容的可能性是一个重大关切。这种风险需要对LLMS进行严格的测试和全面评估，以确保安全和负责任地使用。然而，大规模的LLMS测试需要大量的计算资源，这使得它成为一项昂贵的工作。因此，在测试阶段探索节约成本的策略对于平衡彻底评估的需要和资源可用性的限制至关重要。为了解决这个问题，我们的方法首先将适度知识从LLM转移到一个小模型。随后，我们部署了两种不同的策略来生成恶意查询：一种基于语法树方法，另一种利用基于LLM的方法。最后，我们的方法结合了一个顺序的过滤测试过程，旨在识别容易引发有毒反应的测试用例。我们的研究评估了DistillSeq在四种低密度脂蛋白上的疗效：GPT-3.5、GPT-4.0、Vicuna-13B和Llama-13B。在没有DistillSeq的情况下，观察到的对这些LLMS的攻击成功率GPT-3.5为31.5%，GPT-4.0为21.4%，Vicuna-13B为28.3%，Llama-13B为30.9%。然而，在应用DistillSeq后，这些成功率分别显著增加到58.5%、50.7%、52.5%和54.4%。与未使用DistillSeq的情况相比，这意味着攻击成功率平均提升了93.0%。这些发现突出了DistillSeq在减少有效测试LLM所需的时间和资源投资方面所提供的显著增强。



## **5. The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?**

首先知道的：代币分布如何揭示大型视觉语言模型中隐藏的知识？ cs.CV

ECCV 2024. Project page: https://github.com/Qinyu-Allen-Zhao/LVLM-LP

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2403.09037v2) [paper-pdf](http://arxiv.org/pdf/2403.09037v2)

**Authors**: Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould

**Abstract**: Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layers of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against jailbreaking attacks, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, which indicates potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicating uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improves models' performance but is still inferior to linear probing on these tasks.

摘要: 大型视觉语言模型(LVLM)旨在解释和响应人类的指令，有时会由于不适当的指令而产生幻觉或有害内容。这项研究使用线性探测来揭示LVLMS输出层的隐藏知识。我们证明了第一个令牌的Logit分布包含了足够的信息来确定是否响应指令，包括识别无法回答的可视问题、防御越狱攻击和识别欺骗性问题。在响应生成期间，这种隐藏的知识在后续令牌的登录中逐渐丢失。然后，我们在生成第一个令牌时说明了一种简单的解码策略，有效地改进了生成的内容。在实验中，我们发现了一些有趣的见解：首先，CLIP模型已经包含了解决这些任务的强烈信号，这表明现有数据集中存在潜在的偏差。其次，我们观察到在三个额外的任务中，利用第一个Logit分布，包括指示数学解题中的不确定性、减轻幻觉和图像分类，性能有所提高。最后，在相同的训练数据下，简单地对LVLM进行微调可以提高模型的性能，但在这些任务上仍然不如线性探测。



## **6. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2404.19287v2) [paper-pdf](http://arxiv.org/pdf/2404.19287v2)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在各种下游任务中表现出令人印象深刻的泛化性能，但它们仍然容易受到对手的攻击。虽然以前的研究主要集中在提高图像编码器的对抗健壮性以防止对图像的攻击，但对基于文本的和多模式攻击的探索在很大程度上被忽视了。在这项工作中，我们启动了第一个已知和全面的努力，以研究适应视觉语言模型的对手在多模式攻击下的稳健性。首先，我们介绍了一种多模式攻击策略，并研究了不同攻击的影响。然后，我们提出了一种多模式对抗性训练损失，将干净和对抗性的文本嵌入与对抗性和干净的视觉特征相结合，以增强CLIP图像和文本编码者的对抗性健壮性。在两个任务的15个数据集上的大量实验表明，我们的方法显著地提高了CLIP的对抗健壮性。有趣的是，我们发现，与仅针对基于图像的攻击进行微调的模型相比，针对多模式攻击进行微调的模型表现出更强的稳健性，甚至在图像攻击的背景下也是如此，这可能为增强VLM的安全性开辟新的可能性。



## **7. Turning Generative Models Degenerate: The Power of Data Poisoning Attacks**

使生成模型退化：数据中毒攻击的力量 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12281v1) [paper-pdf](http://arxiv.org/pdf/2407.12281v1)

**Authors**: Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Farhan Ahmed, Ling Cai, Nathalie Baracaldo

**Abstract**: The increasing use of large language models (LLMs) trained by third parties raises significant security concerns. In particular, malicious actors can introduce backdoors through poisoning attacks to generate undesirable outputs. While such attacks have been extensively studied in image domains and classification tasks, they remain underexplored for natural language generation (NLG) tasks. To address this gap, we conduct an investigation of various poisoning techniques targeting the LLM's fine-tuning phase via prefix-tuning, a Parameter Efficient Fine-Tuning (PEFT) method. We assess their effectiveness across two generative tasks: text summarization and text completion; and we also introduce new metrics to quantify the success and stealthiness of such NLG poisoning attacks. Through our experiments, we find that the prefix-tuning hyperparameters and trigger designs are the most crucial factors to influence attack success and stealthiness. Moreover, we demonstrate that existing popular defenses are ineffective against our poisoning attacks. Our study presents the first systematic approach to understanding poisoning attacks targeting NLG tasks during fine-tuning via PEFT across a wide range of triggers and attack settings. We hope our findings will aid the AI security community in developing effective defenses against such threats.

摘要: 越来越多地使用由第三方培训的大型语言模型(LLM)引起了严重的安全问题。特别是，恶意攻击者可以通过投毒攻击引入后门，以生成不良输出。虽然这类攻击已经在图像域和分类任务中得到了广泛的研究，但它们在自然语言生成(NLG)任务中仍然没有得到充分的研究。为了解决这一差距，我们通过前缀调整(一种参数高效微调(PEFT)方法)对针对LLM微调阶段的各种中毒技术进行了调查。我们通过两个生成性任务来评估它们的有效性：文本摘要和文本完成；我们还引入了新的度量来量化此类NLG中毒攻击的成功和隐蔽性。通过实验，我们发现前缀调整超参数和触发器设计是影响攻击成功和隐身的最关键因素。此外，我们证明，现有的大众防御系统对我们的中毒攻击是无效的。我们的研究提出了第一种系统的方法来理解针对NLG任务的中毒攻击，在通过PEFT对广泛的触发器和攻击设置进行微调期间。我们希望我们的发现将有助于人工智能安全界开发针对此类威胁的有效防御措施。



## **8. Uncertainty is Fragile: Manipulating Uncertainty in Large Language Models**

不确定性是脆弱的：在大型语言模型中操纵不确定性 cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.11282v2) [paper-pdf](http://arxiv.org/pdf/2407.11282v2)

**Authors**: Qingcheng Zeng, Mingyu Jin, Qinkai Yu, Zhenting Wang, Wenyue Hua, Zihao Zhou, Guangyan Sun, Yanda Meng, Shiqing Ma, Qifan Wang, Felix Juefei-Xu, Kaize Ding, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: Large Language Models (LLMs) are employed across various high-stakes domains, where the reliability of their outputs is crucial. One commonly used method to assess the reliability of LLMs' responses is uncertainty estimation, which gauges the likelihood of their answers being correct. While many studies focus on improving the accuracy of uncertainty estimations for LLMs, our research investigates the fragility of uncertainty estimation and explores potential attacks. We demonstrate that an attacker can embed a backdoor in LLMs, which, when activated by a specific trigger in the input, manipulates the model's uncertainty without affecting the final output. Specifically, the proposed backdoor attack method can alter an LLM's output probability distribution, causing the probability distribution to converge towards an attacker-predefined distribution while ensuring that the top-1 prediction remains unchanged. Our experimental results demonstrate that this attack effectively undermines the model's self-evaluation reliability in multiple-choice questions. For instance, we achieved a 100 attack success rate (ASR) across three different triggering strategies in four models. Further, we investigate whether this manipulation generalizes across different prompts and domains. This work highlights a significant threat to the reliability of LLMs and underscores the need for future defenses against such attacks. The code is available at https://github.com/qcznlp/uncertainty_attack.

摘要: 大型语言模型(LLM)被用于各种高风险领域，在这些领域中，其输出的可靠性至关重要。评估LLMS回答可靠性的一种常用方法是不确定性估计，它衡量他们回答正确的可能性。虽然许多研究都集中在提高LLMS不确定性估计的准确性上，但我们的研究调查了不确定性估计的脆弱性，并探索了潜在的攻击。我们演示了攻击者可以在LLMS中嵌入后门，当它被输入中的特定触发器激活时，在不影响最终输出的情况下操纵模型的不确定性。具体地说，提出的后门攻击方法可以改变LLM的输出概率分布，使概率分布收敛到攻击者预定义的分布，同时确保TOP-1预测保持不变。我们的实验结果表明，这种攻击有效地破坏了该模型在选择题中的自我评价可靠性。例如，我们在四个模型中的三种不同触发策略中实现了100%的攻击成功率(ASR)。此外，我们还研究了这种操作是否适用于不同的提示和域。这项工作突出了对LLMS可靠性的重大威胁，并强调了今后对这类攻击采取防御措施的必要性。代码可在https://github.com/qcznlp/uncertainty_attack.上获得



## **9. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2404.01318v4) [paper-pdf](http://arxiv.org/pdf/2404.01318v4)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **10. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

通过对基于LLM的排队模型的对抗攻击探索决策级的鲁棒性 cs.MM

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2405.19802v3) [paper-pdf](http://arxiv.org/pdf/2405.19802v3)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.

摘要: 具身智能使特工具有深刻的感知力，使他们能够以与现实世界情况密切一致的方式做出反应。大型语言模型(LLM)深入研究语言指令，在为复杂任务制定计划方面发挥着至关重要的作用。因此，基于LLM的具体化模型进一步增强了代理理解和处理信息的能力。然而，这种融合也带来了追求高智商的新挑战。具体地说，攻击者可以通过更改提示来操纵LLMS生成无关甚至恶意的输出。面对这一挑战，我们注意到明显缺乏全面评估基于LLM的体现模型的稳健性所必需的多模式数据集。因此，我们构建了专门为健壮性评估量身定做的具体化智能机器人攻击数据集(Eirad)。此外，设计了两种攻击策略，包括非定向攻击和定向攻击，以有效地模拟一系列不同的攻击场景。同时，在攻击过程中，为了更准确地确定我们的方法在攻击基于LLM的体现模型上是否成功，我们设计了一种新的利用BLIP2模型的攻击成功评估方法。考虑到GCG算法在攻击中的时间和成本密集性，我们设计了一种基于不同目标任务的快速后缀初始化方案，从而加快了收敛过程。实验结果表明，我们的方法在攻击基于LLM的具体模型时表现出了较高的攻击成功率，表明这些模型具有较低的决策级健壮性。



## **11. Robust Utility-Preserving Text Anonymization Based on Large Language Models**

基于大型语言模型的鲁棒性保留效用的文本解析 cs.CL

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11770v1) [paper-pdf](http://arxiv.org/pdf/2407.11770v1)

**Authors**: Tianyu Yang, Xiaodan Zhu, Iryna Gurevych

**Abstract**: Text anonymization is crucial for sharing sensitive data while maintaining privacy. Existing techniques face the emerging challenges of re-identification attack ability of Large Language Models (LLMs), which have shown advanced capability in memorizing detailed information and patterns as well as connecting disparate pieces of information. In defending against LLM-based re-identification attacks, anonymization could jeopardize the utility of the resulting anonymized data in downstream tasks -- the trade-off between privacy and data utility requires deeper understanding within the context of LLMs. This paper proposes a framework composed of three LLM-based components -- a privacy evaluator, a utility evaluator, and an optimization component, which work collaboratively to perform anonymization. To provide a practical model for large-scale and real-time environments, we distill the anonymization capabilities into a lightweight model using Direct Preference Optimization (DPO). Extensive experiments demonstrate that the proposed models outperform baseline models, showing robustness in reducing the risk of re-identification while preserving greater data utility in downstream tasks. Our code and dataset are available at https://github.com/UKPLab/arxiv2024-rupta.

摘要: 文本匿名化对于在共享敏感数据的同时保持隐私至关重要。现有技术面临着大型语言模型(LLM)重新识别攻击能力的新挑战，LLM在记忆详细信息和模式以及连接不同的信息片段方面显示出先进的能力。在防御基于LLM的重新识别攻击时，匿名化可能会危及产生的匿名数据在下游任务中的效用--隐私和数据效用之间的权衡需要在LLM的背景下更深入地理解。本文提出了一个由三个基于LLM的组件--隐私评估器、效用评估器和优化组件组成的框架，它们协同工作来执行匿名化。为了提供一个适用于大规模实时环境的实用模型，我们使用直接偏好优化(DPO)将匿名化能力提炼成一个轻量级模型。大量的实验表明，所提出的模型优于基线模型，在降低重新识别风险的同时，在下游任务中保持了更大的数据效用。我们的代码和数据集可在https://github.com/UKPLab/arxiv2024-rupta.上获得



## **12. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

使用大型语言模型（LLM）学习图形：深入研究模型稳健性 cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.12068v1) [paper-pdf](http://arxiv.org/pdf/2407.12068v1)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了显著的性能。最近，已经开发了几个基于LLMS的管道来增强对具有文本属性的图形的学习，展示了良好的性能。然而，众所周知，图是容易受到敌意攻击的，而且目前还不清楚LLM是否表现出关于图的学习的健壮性。为了解决这一差距，我们的工作旨在探索LLMS在对抗性攻击图的背景下的潜力。具体地说，我们从两个维度考察了对图结构和文本扰动的稳健性：作为增强器的LLMS和作为预测者的LLMS。通过广泛的实验，我们发现，与浅层模型相比，LLMS-as-Enhancer和LLMS-as-Predicator对结构和文本攻击都具有更好的稳健性。基于这些发现，我们进行了额外的分析，以探讨潜在的原因。此外，我们开放了我们的基准图书馆，以促进快速和公平的评估，并鼓励这一领域正在进行的创新研究。



## **13. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

Entrobust：评估大型语言模型在对抗性预测上的稳健性 cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.04528v5) [paper-pdf](http://arxiv.org/pdf/2306.04528v5)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptRobust, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks including sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present a comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptRobust，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些对抗性提示旨在模仿打字或同义词等看似合理的用户错误，旨在评估微小的偏差如何在保持语义完整性的同时影响LLM结果。然后，这些提示被用于各种任务，包括情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4788个对抗性提示，仔细评估了8个任务和13个数据集。我们的研究结果表明，当代的LLM对敌意提示并不健壮。此外，我们给出了一个全面的分析，以理解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。



## **14. Static Detection of Filesystem Vulnerabilities in Android Systems**

Android系统中文件库漏洞的静态检测 cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.11279v1) [paper-pdf](http://arxiv.org/pdf/2407.11279v1)

**Authors**: Yu-Tsung Lee, Hayawardh Vijayakumar, Zhiyun Qian, Trent Jaeger

**Abstract**: Filesystem vulnerabilities persist as a significant threat to Android systems, despite various proposed defenses and testing techniques. The complexity of program behaviors and access control mechanisms in Android systems makes it challenging to effectively identify these vulnerabilities. In this paper, we present PathSentinel, which overcomes the limitations of previous techniques by combining static program analysis and access control policy analysis to detect three types of filesystem vulnerabilities: path traversals, hijacking vulnerabilities, and luring vulnerabilities. By unifying program and access control policy analysis, PathSentinel identifies attack surfaces accurately and prunes many impractical attacks to generate input payloads for vulnerability testing. To streamline vulnerability validation, PathSentinel leverages large language models (LLMs) to generate targeted exploit code based on the identified vulnerabilities and generated input payloads. The LLMs serve as a tool to reduce the engineering effort required for writing test applications, demonstrating the potential of combining static analysis with LLMs to enhance the efficiency of exploit generation and vulnerability validation. Evaluation on Android 12 and 14 systems from Samsung and OnePlus demonstrates PathSentinel's effectiveness, uncovering 51 previously unknown vulnerabilities among 217 apps with only 2 false positives. These results underscore the importance of combining program and access control policy analysis for accurate vulnerability detection and highlight the promising direction of integrating LLMs for automated exploit generation, providing a comprehensive approach to enhancing the security of Android systems against filesystem vulnerabilities.

摘要: 尽管提出了各种防御和测试技术，但文件系统漏洞仍然是对Android系统的重大威胁。Android系统中程序行为和访问控制机制的复杂性使得有效识别这些漏洞具有挑战性。在本文中，我们提出了Path Sentinel，它通过将静态程序分析和访问控制策略分析相结合来检测三种类型的文件系统漏洞：路径遍历漏洞、劫持漏洞和引诱漏洞，从而克服了以往技术的局限性。通过统一程序和访问控制策略分析，PathSentinel准确识别攻击面，并修剪许多不切实际的攻击，以生成用于漏洞测试的输入有效负载。为了简化漏洞验证，PathSentinel利用大型语言模型(LLM)根据识别的漏洞和生成的输入有效负载生成目标攻击代码。LLM可用作减少编写测试应用程序所需的工程工作量的工具，展示了将静态分析与LLM相结合以提高漏洞生成和漏洞验证效率的潜力。三星和OnePlus对Android 12和14系统的评估展示了PathSentinel的有效性，在217个只有2个误报的应用程序中发现了51个以前未知的漏洞。这些结果强调了将程序和访问控制策略分析相结合以进行准确的漏洞检测的重要性，并突出了集成LLMS以实现自动利用漏洞生成的前景，为增强Android系统针对文件系统漏洞的安全性提供了一种全面的方法。



## **15. Did the Neurons Read your Book? Document-level Membership Inference for Large Language Models**

神经元读过你的书吗？大型语言模型的文档级成员推断 cs.CL

Accepted at 33rd USENIX Security Symposium (USENIX Security 2024)

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2310.15007v2) [paper-pdf](http://arxiv.org/pdf/2310.15007v2)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: With large language models (LLMs) poised to become embedded in our daily lives, questions are starting to be raised about the data they learned from. These questions range from potential bias or misinformation LLMs could retain from their training data to questions of copyright and fair use of human-generated text. However, while these questions emerge, developers of the recent state-of-the-art LLMs become increasingly reluctant to disclose details on their training corpus. We here introduce the task of document-level membership inference for real-world LLMs, i.e. inferring whether the LLM has seen a given document during training or not. First, we propose a procedure for the development and evaluation of document-level membership inference for LLMs by leveraging commonly used data sources for training and the model release date. We then propose a practical, black-box method to predict document-level membership and instantiate it on OpenLLaMA-7B with both books and academic papers. We show our methodology to perform very well, reaching an AUC of 0.856 for books and 0.678 for papers. We then show our approach to outperform the sentence-level membership inference attacks used in the privacy literature for the document-level membership task. We further evaluate whether smaller models might be less sensitive to document-level inference and show OpenLLaMA-3B to be approximately as sensitive as OpenLLaMA-7B to our approach. Finally, we consider two mitigation strategies and find the AUC to slowly decrease when only partial documents are considered but to remain fairly high when the model precision is reduced. Taken together, our results show that accurate document-level membership can be inferred for LLMs, increasing the transparency of technology poised to change our lives.

摘要: 随着大型语言模型(LLM)即将嵌入我们的日常生活，人们开始对它们从中学习到的数据提出质疑。这些问题从小岛屿发展中国家可能从其培训数据中保留的潜在偏见或错误信息，到版权和合理使用人类生成的文本的问题。然而，当这些问题浮出水面时，最近最先进的LLM的开发商越来越不愿透露他们的培训语料库的细节。我们在这里介绍了真实世界LLM的文档级成员关系推理任务，即推断LLM在训练期间是否看到过给定的文档。首先，我们提出了一种通过利用用于训练的常用数据源和模型发布日期来开发和评估LLMS的文档级成员资格推理的过程。然后，我们提出了一种实用的黑盒方法来预测文档级成员资格，并在OpenLLaMA-7B上用书籍和学术论文进行了实例化。我们的方法表现得非常好，图书的AUC达到0.856，论文的AUC达到0.678。然后，我们展示了我们的方法，以优于在隐私文献中用于文档级成员身份任务的句子级别成员身份推理攻击。我们进一步评估了较小的模型是否对文档级推理不那么敏感，并表明OpenLLaMA-3B对我们的方法大约与OpenLLaMA-7B一样敏感。最后，我们考虑了两种缓解策略，发现当只考虑部分文档时，AUC缓慢下降，但当模型精度降低时，AUC保持相当高的水平。综上所述，我们的结果表明，可以为LLMS推断出准确的文档级成员资格，从而增加了即将改变我们生活的技术的透明度。



## **16. SLIP: Securing LLMs IP Using Weights Decomposition**

SIP：使用权重分解保护LLM IP cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10886v1) [paper-pdf](http://arxiv.org/pdf/2407.10886v1)

**Authors**: Yehonathan Refael, Adam Hakim, Lev Greenberg, Tal Aviv, Satya Lokam, Ben Fishman, Shachar Seidman

**Abstract**: Large language models (LLMs) have recently seen widespread adoption, in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting enormous investments by their owners. Moreover, the high cost of cloud-based deployment has driven interest towards deployment to edge devices, yet this risks exposing valuable parameters to theft and unauthorized use. Current methods to protect models' IP on the edge have limitations in terms of practicality, loss in accuracy, or suitability to requirements. In this paper, we introduce a novel hybrid inference algorithm, named SLIP, designed to protect edge-deployed models from theft. SLIP is the first hybrid protocol that is both practical for real-world applications and provably secure, while having zero accuracy degradation and minimal impact on latency. It involves partitioning the model between two computing resources, one secure but expensive, and another cost-effective but vulnerable. This is achieved through matrix decomposition, ensuring that the secure resource retains a maximally sensitive portion of the model's IP while performing a minimal amount of computations, and vice versa for the vulnerable resource. Importantly, the protocol includes security guarantees that prevent attackers from exploiting the partition to infer the secured information. Finally, we present experimental results that show the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.

摘要: 大型语言模型(LLM)最近在学术界和工业界都得到了广泛采用。随着这些模式的发展，它们成为有价值的知识产权(IP)，反映了其所有者的巨额投资。此外，基于云的部署的高昂成本推动了对部署到边缘设备的兴趣，但这可能会使宝贵的参数面临被盗和未经授权使用的风险。目前保护模型边缘知识产权的方法在实用性、精确度损失或对要求的适应性方面都存在局限性。在本文中，我们提出了一种新的混合推理算法，称为SLIP，旨在防止边部署的模型被盗。SLIP是第一个混合协议，它既适用于现实世界的应用，又可证明是安全的，同时具有零精度降级和对延迟的最小影响。它涉及在两种计算资源之间划分模型，一种是安全但昂贵的计算资源，另一种是经济高效但脆弱的计算资源。这是通过矩阵分解实现的，确保安全资源保留模型IP的最敏感部分，同时执行最少量的计算，反之亦然。重要的是，该协议包括安全保证，以防止攻击者利用分区来推断安全信息。最后，我们给出了实验结果，证明了该方法的健壮性和有效性，将其定位为一种引人注目的保护LLM的解决方案。



## **17. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

TAPI：针对代码LLM的目标特定和对抗性即时注入 cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.09164v2) [paper-pdf](http://arxiv.org/pdf/2407.09164v2)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate enhancement of up to 89.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛并成功地用于简化和促进代码编程。使用这些工具，开发人员可以根据不完整的代码和自然语言提示轻松生成所需的完整功能代码。然而，一些开创性的工作表明，这些代码LLM也容易受到攻击，例如，抵御后门和对手攻击。前者可以通过毒化训练数据或模型参数来诱导LLMS响应插入恶意代码片段的触发器，而后者可以手工创建恶意输入代码来降低生成代码的质量。然而，这两种攻击方法都有潜在的局限性：后门攻击依赖于控制模型训练过程，而对抗性攻击则难以实现特定的恶意目的。为了继承后门攻击和对抗性攻击的优点，提出了一种新的针对Code LLMS的攻击范式，即目标特定和对抗性提示注入(TAPI)。TAPI生成不可读的注释，其中包含有关恶意指令的信息，并将它们作为触发器隐藏在外部源代码中。当用户利用Code LLMS来完成包含触发器的代码时，模型将在特定位置生成攻击者指定的恶意代码片段。我们在三个典型的恶意目标和七个案例下评估了我们的TAPI攻击对四个有代表性的LLM的攻击。结果表明，该方法具有很强的威胁性(攻击成功率提高高达89.3%)和隐蔽性(在触发设计中平均节省53.1%的令牌)。特别是，我们成功地攻击了一些著名的部署代码完成集成应用程序，包括CodeGeex和Github Copilot。这进一步证实了我们攻击的现实威胁。



## **18. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

闭上眼睛，安全：通过图像到文本转换保护多模式LLM cs.CV

ECCV2024 (Project Page: https://gyhdog99.github.io/projects/ecso/)

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2403.09572v3) [paper-pdf](http://arxiv.org/pdf/2403.09572v3)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting the unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed with the introduction of image features. To construct robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO enhances model safety significantly (e.g.,, 37.6% improvement on the MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.

摘要: 多通道大型语言模型(MLLMS)已经显示出令人印象深刻的推理能力。然而，他们也比他们的LLM前辈更容易受到越狱攻击。虽然仍然能够检测到不安全的响应，但我们观察到，通过引入图像特征，可以很容易地绕过MLLMS中预对准LLM的安全机制。为了构造稳健的MLLMS，我们提出了一种新的无需训练的保护方法ECSO(Eyes Closed，Safe On)，该方法利用MLLMS固有的安全意识，通过自适应地将不安全的图像转换为文本来激活MLLMS中预对准的LLMS的固有安全机制，从而产生更安全的响应。在五个最先进的(SOTA)MLLM上的实验表明，ECSO显著提高了模型安全性(例如，在MM-SafetyBch(SD+OCR)的基础上改进了37.6%，在使用LLaVA-1.5-7B的VLSafe上改进了71.3%)，同时保持了常见MLLM基准的实用结果。此外，我们还证明了ECSO可以作为数据引擎来生成用于MLLM比对的监督精调(SFT)数据，而无需额外的人工干预。



## **19. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

潘多拉的白盒：大型语言模型中的精确训练数据检测和提取 cs.CR

Found software bug in experiments, withdrawing in order to address  and update results

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2402.17012v4) [paper-pdf](http://arxiv.org/pdf/2402.17012v4)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.

摘要: 在本文中，我们开发了针对大型语言模型(LLM)的最先进的隐私攻击，其中对该模型具有一定访问权限的对手试图了解一些关于潜在训练数据的信息。我们的主要结果是针对预先训练的LLM的新成员推理攻击(MIA)，其性能比基线攻击高数百倍，并且管道显示超过50%(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑了不同程度的访问底层模型、预训练和微调数据，以及MIA和训练数据提取。对于预训练数据，我们提出了两个新的MIA：一个有监督的神经网络分类器，它基于(降维)模型梯度预测训练数据的成员资格，以及这种攻击的一个变体，它只需要通过利用LLMS上最近的模型窃取工作来对模型进行Logit访问。据我们所知，这是第一个明确纳入模型窃取信息的MIA。这两种攻击都超过了现有的黑盒基线，我们的监督攻击缩小了针对LLMS的MIA攻击成功与针对其他机器学习模型的已知最强攻击之间的差距。在微调中，我们发现基于基本模型和微调模型之间的损失比率的简单攻击能够获得近乎完美的MIA性能；然后，我们利用我们的MIA从微调的Pythia和Llama模型中提取很大一部分微调数据集。我们的代码可以在githorb.com/Safr-ai-lab/pandora-llm上找到。



## **20. Merging Improves Self-Critique Against Jailbreak Attacks**

合并提高了对越狱袭击的自我批评 cs.CL

Published at ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2406.07188v2) [paper-pdf](http://arxiv.org/pdf/2406.07188v2)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .

摘要: 大型语言模型（LLM）对越狱攻击等对抗性操纵的稳健性仍然是一个重大挑战。在这项工作中，我们提出了一种增强LLM自我批评能力的方法，并根据净化的合成数据进一步对其进行微调。这是通过添加一个可以与原始模型合并的外部批评者模型来实现的，从而增强自我批评能力并提高LLM对对抗提示反应的稳健性。我们的结果表明，合并和自我批评的结合可以显着降低对手的攻击成功率，从而提供一种有希望的针对越狱攻击的防御机制。代码、数据和模型在https://github.com/vicgalle/merging-self-critique-jailbreaks上发布。



## **21. Watermarking Text Data on Large Language Models for Dataset Copyright**

在大型语言模型上对文本数据进行水印以获得数据集版权 cs.CR

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2305.13257v4) [paper-pdf](http://arxiv.org/pdf/2305.13257v4)

**Authors**: Yixin Liu, Hongsheng Hu, Xun Chen, Xuyun Zhang, Lichao Sun

**Abstract**: Substantial research works have shown that deep models, e.g., pre-trained models, on the large corpus can learn universal language representations, which are beneficial for downstream NLP tasks. However, these powerful models are also vulnerable to various privacy attacks, while much sensitive information exists in the training dataset. The attacker can easily steal sensitive information from public models, e.g., individuals' email addresses and phone numbers. In an attempt to address these issues, particularly the unauthorized use of private data, we introduce a novel watermarking technique via a backdoor-based membership inference approach named TextMarker, which can safeguard diverse forms of private information embedded in the training text data. Specifically, TextMarker only requires data owners to mark a small number of samples for data copyright protection under the black-box access assumption to the target model. Through extensive evaluation, we demonstrate the effectiveness of TextMarker on various real-world datasets, e.g., marking only 0.1% of the training dataset is practically sufficient for effective membership inference with negligible effect on model utility. We also discuss potential countermeasures and show that TextMarker is stealthy enough to bypass them.

摘要: 大量的研究工作表明，在大型语料库上的深层模型，例如预先训练的模型，可以学习通用的语言表示，这对下游的自然语言处理任务是有利的。然而，这些强大的模型也容易受到各种隐私攻击，而许多敏感信息存在于训练数据集中。攻击者可以很容易地从公共模型中窃取敏感信息，例如个人的电子邮件地址和电话号码。为了解决这些问题，特别是隐私数据的未经授权使用，我们提出了一种新的水印技术，该技术通过一种基于后门的成员关系推理方法TextMarker来保护嵌入在训练文本数据中的各种形式的隐私信息。具体地说，TextMarker只要求数据所有者在目标模型的黑盒访问假设下标记少量样本，以进行数据版权保护。通过广泛的评估，我们证明了TextMarker在各种真实数据集上的有效性，例如，只标记0.1%的训练数据集实际上足以进行有效的隶属度推理，而对模型效用的影响可以忽略不计。我们还讨论了潜在的对策，并表明TextMarker足够隐蔽，可以绕过它们。



## **22. Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training**

每当您感到不安全时就拒绝：通过脱钩拒绝培训提高LLM的安全性 cs.CL

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09121v1) [paper-pdf](http://arxiv.org/pdf/2407.09121v1)

**Authors**: Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Jiahao Xu, Tian Liang, Pinjia He, Zhaopeng Tu

**Abstract**: This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models' ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses well-known models such as GPT-4 in defending against attacks. Importantly, our approach successfully defends recent advanced attack methods (e.g., CodeAttack) that have jailbroken GPT-4 and LLaMA3-70B-Instruct. Our code and data can be found at https://github.com/RobustNLP/DeRTa.

摘要: 这项研究通过识别和处理安全调整数据中的拒绝位置偏差，解决了大型语言模型(LLM)安全调整实践中的一个关键漏洞，该偏差损害了模型适当拒绝生成不安全内容的能力。我们引入了一种新的方法，去耦合拒绝训练(DeRTa)，旨在使LLMS能够在任何响应位置拒绝遵守有害提示，从而显著增强其安全能力。DeRTA引入了两个新的组件：(1)带有有害响应前缀的最大似然估计(MLE)，它通过在安全响应的开始添加一段有害响应来训练模型识别和避免不安全内容，以及(2)增强转移优化(RTO)，它使模型能够在整个有害响应序列中一致地从潜在危害转换到安全拒绝。我们使用LLaMA3和Mistral模型家族对六种攻击场景进行的经验评估表明，我们的方法不仅在不影响性能的情况下提高了模型的安全性，而且在防御攻击方面优于GPT-4等知名模型。重要的是，我们的方法成功地防御了最近的高级攻击方法(例如，CodeAttack)，这些方法已经越狱了GPT-4和LLaMA3-70B-指令。我们的代码和数据可在https://github.com/RobustNLP/DeRTa.上找到



## **23. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

github url added

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2406.14393v2) [paper-pdf](http://arxiv.org/pdf/2406.14393v2)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个用于自动化红色分组的系统，可以针对各种目标对齐的LLM生成对抗提示。ReMiss在AdvBench基准上实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。与以前的方法相比，详细的分析强调了拟议的奖励错误指定目标所带来的独特优势。



## **24. A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**

大型视觉语言模型攻击调查：资源、进展和未来趋势 cs.CV

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.07403v2) [paper-pdf](http://arxiv.org/pdf/2407.07403v2)

**Authors**: Daizong Liu, Mingyu Yang, Xiaoye Qu, Pan Zhou, Yu Cheng, Wei Hu

**Abstract**: With the significant development of large models in recent years, Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding and reasoning tasks. Compared to traditional Large Language Models (LLMs), LVLMs present great potential and challenges due to its closer proximity to the multi-resource real-world applications and the complexity of multi-modal processing. However, the vulnerability of LVLMs is relatively underexplored, posing potential security risks in daily usage. In this paper, we provide a comprehensive review of the various forms of existing LVLM attacks. Specifically, we first introduce the background of attacks targeting LVLMs, including the attack preliminary, attack challenges, and attack resources. Then, we systematically review the development of LVLM attack methods, such as adversarial attacks that manipulate model outputs, jailbreak attacks that exploit model vulnerabilities for unauthorized actions, prompt injection attacks that engineer the prompt type and pattern, and data poisoning that affects model training. Finally, we discuss promising research directions in the future. We believe that our survey provides insights into the current landscape of LVLM vulnerabilities, inspiring more researchers to explore and mitigate potential safety issues in LVLM developments. The latest papers on LVLM attacks are continuously collected in https://github.com/liudaizong/Awesome-LVLM-Attack.

摘要: 近年来，随着大型模型的显著发展，大型视觉语言模型在广泛的多通道理解和推理任务中表现出了卓越的能力。与传统的大语言模型相比，大语言模型因其更接近多资源的实际应用和多模式处理的复杂性而显示出巨大的潜力和挑战。然而，LVLMS的脆弱性相对较少，在日常使用中存在潜在的安全风险。在本文中，我们对现有的各种形式的LVLM攻击进行了全面的回顾。具体地说，我们首先介绍了针对LVLMS的攻击背景，包括攻击准备、攻击挑战和攻击资源。然后，我们系统地回顾了LVLM攻击方法的发展，如操纵模型输出的对抗性攻击，利用模型漏洞进行未经授权操作的越狱攻击，设计提示类型和模式的提示注入攻击，以及影响模型训练的数据中毒。最后，我们讨论了未来的研究方向。我们相信，我们的调查提供了对LVLM漏洞现状的洞察，激励更多的研究人员探索和缓解LVLM开发中的潜在安全问题。有关LVLm攻击的最新论文在https://github.com/liudaizong/Awesome-LVLM-Attack.上不断收集



## **25. Tactics, Techniques, and Procedures (TTPs) in Interpreted Malware: A Zero-Shot Generation with Large Language Models**

解释恶意软件中的策略、技术和程序（TTP）：具有大型语言模型的零攻击生成 cs.CR

19 pages, 11 figures

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08532v1) [paper-pdf](http://arxiv.org/pdf/2407.08532v1)

**Authors**: Ying Zhang, Xiaoyan Zhou, Hui Wen, Wenjia Niu, Jiqiang Liu, Haining Wang, Qiang Li

**Abstract**: Nowadays, the open-source software (OSS) ecosystem suffers from security threats of software supply chain (SSC) attacks. Interpreted OSS malware plays a vital role in SSC attacks, as criminals have an arsenal of attack vectors to deceive users into installing malware and executing malicious activities. In this paper, we introduce tactics, techniques, and procedures (TTPs) proposed by MITRE ATT\&CK into the interpreted malware analysis to characterize different phases of an attack lifecycle. Specifically, we propose GENTTP, a zero-shot approach to extracting a TTP of an interpreted malware package. GENTTP leverages large language models (LLMs) to automatically generate a TTP, where the input is a malicious package, and the output is a deceptive tactic and an execution tactic of attack vectors. To validate the effectiveness of GENTTP, we collect two datasets for evaluation: a dataset with ground truth labels and a large dataset in the wild. Experimental results show that GENTTP can generate TTPs with high accuracy and efficiency. To demonstrate GENTTP's benefits, we build an LLM-based Chatbot from 3,700+ PyPI malware's TTPs. We further conduct a quantitative analysis of malware's TTPs at a large scale. Our main findings include: (1) many OSS malicious packages share a relatively stable TTP, even with the increasing emergence of malware and attack campaigns, (2) a TTP reflects characteristics of a malware-based attack, and (3) an attacker's intent behind the malware is linked to a TTP.

摘要: 目前，开源软件生态系统受到软件供应链(SSC)攻击的安全威胁。解释的OSS恶意软件在SSC攻击中扮演着至关重要的角色，因为犯罪分子拥有大量的攻击媒介，可以欺骗用户安装恶意软件并执行恶意活动。在本文中，我们将MITRE ATT-CK提出的策略、技术和过程(TTP)引入到解释恶意软件分析中，以表征攻击生命周期的不同阶段。具体地说，我们提出了GENTTP，一种零命中率的方法来提取解释的恶意软件包的TTP。GENTTP利用大型语言模型(LLM)自动生成TTP，输入为恶意包，输出为攻击向量的欺骗策略和执行策略。为了验证GENTTP的有效性，我们收集了两个数据集进行评估：一个是带有地面真实标签的数据集，另一个是野外的大型数据集。实验结果表明，GENTTP算法能够生成高精度、高效率的TTP。为了展示GENTTP的好处，我们从3700多个PyPI恶意软件的TTP构建了一个基于LLM的聊天机器人。我们进一步对恶意软件的TTP进行了大规模的定量分析。我们的主要发现包括：(1)许多OSS恶意包共享相对稳定的TTP，即使恶意软件和攻击活动越来越多；(2)TTP反映了基于恶意软件的攻击的特征；(3)恶意软件背后的攻击者意图与TTP有关。



## **26. Virtual Context: Enhancing Jailbreak Attacks with Special Token Injection**

虚拟上下文：通过特殊代币注入增强越狱攻击 cs.CR

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2406.19845v2) [paper-pdf](http://arxiv.org/pdf/2406.19845v2)

**Authors**: Yuqi Zhou, Lin Lu, Hanchi Sun, Pan Zhou, Lichao Sun

**Abstract**: Jailbreak attacks on large language models (LLMs) involve inducing these models to generate harmful content that violates ethics or laws, posing a significant threat to LLM security. Current jailbreak attacks face two main challenges: low success rates due to defensive measures and high resource requirements for crafting specific prompts. This paper introduces Virtual Context, which leverages special tokens, previously overlooked in LLM security, to improve jailbreak attacks. Virtual Context addresses these challenges by significantly increasing the success rates of existing jailbreak methods and requiring minimal background knowledge about the target model, thus enhancing effectiveness in black-box settings without additional overhead. Comprehensive evaluations show that Virtual Context-assisted jailbreak attacks can improve the success rates of four widely used jailbreak methods by approximately 40% across various LLMs. Additionally, applying Virtual Context to original malicious behaviors still achieves a notable jailbreak effect. In summary, our research highlights the potential of special tokens in jailbreak attacks and recommends including this threat in red-teaming testing to comprehensively enhance LLM security.

摘要: 针对大型语言模型(LLM)的越狱攻击涉及诱导这些模型生成违反道德或法律的有害内容，对LLM安全构成重大威胁。目前的越狱攻击面临两个主要挑战：防御性措施导致的成功率较低，以及制作特定提示所需的资源较高。本文介绍了虚拟上下文技术，它利用了以前在LLM安全中被忽视的特殊令牌来改进越狱攻击。虚拟环境通过显著提高现有越狱方法的成功率和只需要最少的目标模型背景知识来解决这些挑战，从而在不增加额外开销的情况下提高黑箱设置的效率。综合评估表明，虚拟情境辅助越狱攻击可以将四种广泛使用的越狱方法的成功率提高约40%。此外，将虚拟情境应用于原始恶意行为仍然可以达到显著的越狱效果。综上所述，我们的研究强调了特殊令牌在越狱攻击中的潜力，并建议将此威胁包括在红团队测试中，以全面增强LLM安全。



## **27. A Comprehensive Survey on the Security of Smart Grid: Challenges, Mitigations, and Future Research Opportunities**

智能电网安全性全面调查：挑战、缓解措施和未来研究机会 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07966v1) [paper-pdf](http://arxiv.org/pdf/2407.07966v1)

**Authors**: Arastoo Zibaeirad, Farnoosh Koleini, Shengping Bi, Tao Hou, Tao Wang

**Abstract**: In this study, we conduct a comprehensive review of smart grid security, exploring system architectures, attack methodologies, defense strategies, and future research opportunities. We provide an in-depth analysis of various attack vectors, focusing on new attack surfaces introduced by advanced components in smart grids. The review particularly includes an extensive analysis of coordinated attacks that incorporate multiple attack strategies and exploit vulnerabilities across various smart grid components to increase their adverse impact, demonstrating the complexity and potential severity of these threats. Following this, we examine innovative detection and mitigation strategies, including game theory, graph theory, blockchain, and machine learning, discussing their advancements in counteracting evolving threats and associated research challenges. In particular, our review covers a thorough examination of widely used machine learning-based mitigation strategies, analyzing their applications and research challenges spanning across supervised, unsupervised, semi-supervised, ensemble, and reinforcement learning. Further, we outline future research directions and explore new techniques and concerns. We first discuss the research opportunities for existing and emerging strategies, and then explore the potential role of new techniques, such as large language models (LLMs), and the emerging threat of adversarial machine learning in the future of smart grid security.

摘要: 在这项研究中，我们对智能电网安全进行了全面的回顾，探索了系统架构、攻击方法、防御策略和未来的研究机会。我们深入分析了各种攻击载体，重点分析了智能电网中先进组件引入的新攻击面。审查特别包括对协调攻击的广泛分析，这些攻击整合了多种攻击策略，并利用各种智能电网组件的漏洞来增加其不利影响，从而展示了这些威胁的复杂性和潜在严重性。随后，我们研究了创新的检测和缓解策略，包括博弈论、图论、区块链和机器学习，讨论了它们在应对不断演变的威胁和相关研究挑战方面的进展。特别是，我们的综述涵盖了广泛使用的基于机器学习的缓解策略的彻底检查，分析了它们在监督、非监督、半监督、集成和强化学习中的应用和研究挑战。此外，我们概述了未来的研究方向，并探索了新的技术和关注的问题。我们首先讨论了现有和新兴策略的研究机会，然后探讨了新技术的潜在作用，如大型语言模型(LLMS)，以及未来智能电网安全中对抗性机器学习的新威胁。



## **28. Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities**

基于LLM的多智能体社区中操纵知识的泛滥传播 cs.CL

18 Pages, working in progress

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07791v1) [paper-pdf](http://arxiv.org/pdf/2407.07791v1)

**Authors**: Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, Gongshen Liu

**Abstract**: The rapid adoption of large language models (LLMs) in multi-agent systems has highlighted their impressive capabilities in various applications, such as collaborative problem-solving and autonomous negotiation. However, the security implications of these LLM-based multi-agent systems have not been thoroughly investigated, particularly concerning the spread of manipulated knowledge. In this paper, we investigate this critical issue by constructing a detailed threat model and a comprehensive simulation environment that mirrors real-world multi-agent deployments in a trusted platform. Subsequently, we propose a novel two-stage attack method involving Persuasiveness Injection and Manipulated Knowledge Injection to systematically explore the potential for manipulated knowledge (i.e., counterfactual and toxic knowledge) spread without explicit prompt manipulation.   Our method leverages the inherent vulnerabilities of LLMs in handling world knowledge, which can be exploited by attackers to unconsciously spread fabricated information. Through extensive experiments, we demonstrate that our attack method can successfully induce LLM-based agents to spread both counterfactual and toxic knowledge without degrading their foundational capabilities during agent communication. Furthermore, we show that these manipulations can persist through popular retrieval-augmented generation frameworks, where several benign agents store and retrieve manipulated chat histories for future interactions. This persistence indicates that even after the interaction has ended, the benign agents may continue to be influenced by manipulated knowledge. Our findings reveal significant security risks in LLM-based multi-agent systems, emphasizing the imperative need for robust defenses against manipulated knowledge spread, such as introducing ``guardian'' agents and advanced fact-checking tools.

摘要: 大型语言模型在多智能体系统中的迅速应用凸显了其在协作问题求解、自主谈判等方面的应用能力。然而，这些基于LLM的多智能体系统的安全含义还没有得到彻底的研究，特别是关于被操纵的知识的传播。在本文中，我们通过构建一个详细的威胁模型和一个全面的模拟环境来研究这一关键问题，该环境反映了可信平台中真实世界的多代理部署。随后，我们提出了一种新的两阶段攻击方法，包括说服力注入和被操纵的知识注入，以系统地探索被操纵的知识(即反事实和有毒知识)在没有明确的即时操纵的情况下传播的可能性。我们的方法利用了LLMS在处理世界知识方面的固有漏洞，攻击者可以利用这些漏洞来不知不觉地传播伪造的信息。通过大量的实验，我们证明了我们的攻击方法可以成功地诱导基于LLM的代理传播反事实和有毒知识，而不会降低其在代理通信中的基础能力。此外，我们还表明，这些操作可以在流行的检索增强生成框架中持续存在，在这些框架中，几个良性代理存储和检索被操纵的聊天历史，以便将来进行交互。这种持久性表明，即使在互动结束后，良性代理人仍可能继续受到操纵知识的影响。我们的发现揭示了基于LLM的多代理系统中的重大安全风险，强调了对被操纵的知识传播采取强有力的防御措施的迫切需要，例如引入“监护人”代理和先进的事实核查工具。



## **29. SecureReg: Combining NLP and MLP for Enhanced Detection of Malicious Domain Name Registrations**

SecureReg：结合NLP和MLP增强恶意域名注册检测 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2401.03196v3) [paper-pdf](http://arxiv.org/pdf/2401.03196v3)

**Authors**: Furkan Çolhak, Mert İlhan Ecevit, Hasan Dağ, Reiner Creutzburg

**Abstract**: The escalating landscape of cyber threats, characterized by the registration of thousands of new domains daily for large-scale Internet attacks such as spam, phishing, and drive-by downloads, underscores the imperative for innovative detection methodologies. This paper introduces a cutting-edge approach for identifying suspicious domains at the onset of the registration process. The accompanying data pipeline generates crucial features by comparing new domains to registered domains, emphasizing the crucial similarity score. The proposed system analyzes semantic and numerical attributes by leveraging a novel combination of Natural Language Processing (NLP) techniques, including a pretrained CANINE model and Multilayer Perceptron (MLP) models, providing a robust solution for early threat detection. This integrated Pretrained NLP (CANINE) + MLP model showcases the outstanding performance, surpassing both individual pretrained NLP models and standalone MLP models. With an F1 score of 84.86\% and an accuracy of 84.95\% on the SecureReg dataset, it effectively detects malicious domain registrations. The findings demonstrate the effectiveness of the integrated approach and contribute to the ongoing efforts to develop proactive strategies to mitigate the risks associated with illicit online activities through the early identification of suspicious domain registrations.

摘要: 不断升级的网络威胁格局，其特点是每天注册数千个新域名，用于大规模互联网攻击，如垃圾邮件、网络钓鱼和路过下载，这突显了创新检测方法的必要性。本文介绍了一种在注册过程开始时识别可疑域名的最新方法。伴随而来的数据管道通过将新域名与注册域名进行比较来生成关键特征，强调关键的相似性得分。该系统通过利用自然语言处理(NLP)技术的新组合来分析语义和数字属性，包括预先训练的犬类模型和多层感知器(MLP)模型，为早期威胁检测提供了稳健的解决方案。这一集成的预训练NLP(犬类)+MLP模型展示了卓越的性能，超过了单独的预训练NLP模型和独立的MLP模型。它在SecureReg数据集上的F1得分为84.86\%，准确率为84.95\%，可以有效地检测恶意域注册。调查结果证明了综合办法的有效性，并有助于不断努力制定积极主动的战略，通过及早查明可疑域名注册来减轻与非法在线活动有关的风险。



## **30. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

评估大型语言模型基于检索的上下文学习的对抗鲁棒性 cs.CL

COLM 2024, 29 pages, 6 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2405.15984v2) [paper-pdf](http://arxiv.org/pdf/2405.15984v2)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl

摘要: 随着大型语言模型的出现，如Llama和OpenAI GPT-3，情景中学习(ICL)因其有效性和高效性而受到广泛关注。但是，ICL对用于对提示符中的演示进行编码的选择、顺序和形容词非常敏感。检索增强的ICL方法试图通过利用检索器来提取语义相关的示例作为演示来解决这个问题。虽然这种方法可以产生更准确的结果，但它对各种类型的对抗性攻击的稳健性，包括对测试样本、演示和检索数据的扰动，仍然没有得到充分的研究。我们的研究表明，检索增强模型可以增强对测试样本攻击的健壮性，性能优于普通ICL，攻击成功率(ASR)降低4.87%；然而，它们在演示中表现出过度自信，导致演示攻击的ASR提高了2%。对抗性训练可以帮助提高ICL方法对对抗性攻击的稳健性；然而，在LLMS的背景下，这样的训练方案可能代价太高。作为另一种选择，我们引入了一种有效的无需训练的对抗防御方法DARD，它用被攻击的样本丰富了样本库。我们表明，DARD在性能和健壮性方面都有改进，ASR比基准降低了15%。发布代码和数据是为了鼓励进一步的研究：https://github.com/simonucl/adv-retreival-icl



## **31. Was it Slander? Towards Exact Inversion of Generative Language Models**

是诽谤吗？迈向生成语言模型的精确倒置 cs.CR

4 pages, 3 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.11059v1) [paper-pdf](http://arxiv.org/pdf/2407.11059v1)

**Authors**: Adrians Skapars, Edoardo Manino, Youcheng Sun, Lucas C. Cordeiro

**Abstract**: Training large language models (LLMs) requires a substantial investment of time and money. To get a good return on investment, the developers spend considerable effort ensuring that the model never produces harmful and offensive outputs. However, bad-faith actors may still try to slander the reputation of an LLM by publicly reporting a forged output. In this paper, we show that defending against such slander attacks requires reconstructing the input of the forged output or proving that it does not exist. To do so, we propose and evaluate a search based approach for targeted adversarial attacks for LLMs. Our experiments show that we are rarely able to reconstruct the exact input of an arbitrary output, thus demonstrating that LLMs are still vulnerable to slander attacks.

摘要: 训练大型语言模型（LLM）需要投入大量的时间和金钱。为了获得良好的投资回报，开发人员花费了相当多的精力确保模型永远不会产生有害和令人反感的输出。然而，恶意行为者仍然可能试图通过公开报告伪造的成果来诽谤法学硕士的声誉。在本文中，我们表明，防御此类诽谤攻击需要重建伪造输出的输入或证明它不存在。为此，我们提出并评估了一种基于搜索的方法，用于针对LLM的有针对性的对抗攻击。我们的实验表明，我们很少能够重建任意输出的确切输入，从而证明LLM仍然容易受到诽谤攻击。



## **32. The Ethics of Interaction: Mitigating Security Threats in LLMs**

互动伦理：缓解LLC中的安全威胁 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2401.12273v2) [paper-pdf](http://arxiv.org/pdf/2401.12273v2)

**Authors**: Ashutosh Kumar, Shiv Vignesh Murthy, Sagarika Singh, Swathy Ragupathy

**Abstract**: This paper comprehensively explores the ethical challenges arising from security threats to Large Language Models (LLMs). These intricate digital repositories are increasingly integrated into our daily lives, making them prime targets for attacks that can compromise their training data and the confidentiality of their data sources. The paper delves into the nuanced ethical repercussions of such security threats on society and individual privacy. We scrutinize five major threats--prompt injection, jailbreaking, Personal Identifiable Information (PII) exposure, sexually explicit content, and hate-based content--going beyond mere identification to assess their critical ethical consequences and the urgency they create for robust defensive strategies. The escalating reliance on LLMs underscores the crucial need for ensuring these systems operate within the bounds of ethical norms, particularly as their misuse can lead to significant societal and individual harm. We propose conceptualizing and developing an evaluative tool tailored for LLMs, which would serve a dual purpose: guiding developers and designers in preemptive fortification of backend systems and scrutinizing the ethical dimensions of LLM chatbot responses during the testing phase. By comparing LLM responses with those expected from humans in a moral context, we aim to discern the degree to which AI behaviors align with the ethical values held by a broader society. Ultimately, this paper not only underscores the ethical troubles presented by LLMs; it also highlights a path toward cultivating trust in these systems.

摘要: 本文全面探讨了大型语言模型(LLM)面临的安全威胁所带来的伦理挑战。这些错综复杂的数字仓库越来越多地融入我们的日常生活，使它们成为攻击的主要目标，这些攻击可能会危及它们的训练数据及其数据源的机密性。本文深入探讨了此类安全威胁对社会和个人隐私的微妙伦理影响。我们仔细审查了五大威胁--即时注射、越狱、个人身份信息(PII)曝光、露骨的性内容和基于仇恨的内容--不仅仅是识别，还评估了它们的关键伦理后果以及它们为强有力的防御战略带来的紧迫性。对LLMS的依赖不断升级，突显了确保这些系统在道德规范范围内运作的迫切需要，特别是考虑到滥用LLMS可能导致重大的社会和个人伤害。我们建议概念化和开发一个为LLMS量身定做的评估工具，这将具有双重目的：指导开发人员和设计人员先发制人地防御后端系统，并在测试阶段仔细审查LLM聊天机器人响应的伦理维度。通过将LLM的反应与人类在道德背景下的预期反应进行比较，我们的目标是辨别人工智能行为与更广泛社会所持有的伦理价值观的一致程度。最后，这篇论文不仅强调了低成本管理带来的伦理问题，还强调了一条培养对这些系统的信任的途径。



## **33. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

稳健的神经信息检索：对抗性和非分布性的角度 cs.IR

Survey paper

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06992v1) [paper-pdf](http://arxiv.org/pdf/2407.06992v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.

摘要: 神经信息检索(IR)模型的最新进展显著提高了它们在各种IR任务中的有效性。这些模型的稳健性对于确保它们在实践中的可靠性至关重要，也引起了人们的极大关注。随着对稳健IR的广泛研究的提出，我们认为现在是巩固当前状况、从现有方法中收集见解并为未来发展奠定基础的好时机。我们认为信息检索的稳健性是一个多方面的概念，强调了它对对抗攻击、分布外(OOD)场景和性能差异的必要性。以对抗性和面向对象的稳健性为重点，我们分别剖析了密集检索模型(DRM)和神经排名模型(NRM)的稳健性解决方案，将它们识别为神经IR管道的关键组件。我们提供了对现有方法、数据集和评估度量的深入讨论，揭示了大型语言模型时代的挑战和未来方向。据我们所知，这是关于神经IR模型稳健性的第一次全面调查，我们还将在SIGIR2024\url{https://sigir2024-robust-information-retrieval.github.io}.上进行我们的第一次教程演示在组织现有工作的同时，我们还介绍了稳健IR基准(BSTIR)，这是一个用于稳健神经信息检索的异质评估基准，可在\url{https://github.com/Davion-Liu/BestIR}.希望本研究为今后研究信息检索模型的健壮性提供有用的线索，并为开发可信搜索引擎\url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.提供帮助



## **34. A hybrid LLM workflow can help identify user privilege related variables in programs of any size**

混合LLM工作流程可以帮助识别任何规模的程序中的用户特权相关变量 cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2403.15723v2) [paper-pdf](http://arxiv.org/pdf/2403.15723v2)

**Authors**: Haizhou Wang, Zhilong Wang, Peng Liu

**Abstract**: Many programs involves operations and logic manipulating user privileges, which is essential for the security of an organization. Therefore, one common malicious goal of attackers is to obtain or escalate the privileges, causing privilege leakage. To protect the program and the organization against privilege leakage attacks, it is important to eliminate the vulnerabilities which can be exploited to achieve such attacks. Unfortunately, while memory vulnerabilities are less challenging to find, logic vulnerabilities are much more imminent, harmful and difficult to identify. Accordingly, many analysts choose to find user privilege related (UPR) variables first as start points to investigate the code where the UPR variables may be used to see if there exists any vulnerabilities, especially the logic ones. In this paper, we introduce a large language model (LLM) workflow that can assist analysts in identifying such UPR variables, which is considered to be a very time-consuming task. Specifically, our tool will audit all the variables in a program and output a UPR score, which is the degree of relationship (closeness) between the variable and user privileges, for each variable. The proposed approach avoids the drawbacks introduced by directly prompting a LLM to find UPR variables by focusing on leverage the LLM at statement level instead of supplying LLM with very long code snippets. Those variables with high UPR scores are essentially potential UPR variables, which should be manually investigated. Our experiments show that using a typical UPR score threshold (i.e., UPR score >0.8), the false positive rate (FPR) is only 13.49%, while UPR variable found is significantly more than that of the heuristic based method.

摘要: 许多程序涉及操纵用户权限的操作和逻辑，这对组织的安全至关重要。因此，攻击者的一个常见恶意目标是获取或提升权限，从而导致权限泄漏。为了保护程序和组织免受权限泄漏攻击，重要的是消除可被利用来实现此类攻击的漏洞。不幸的是，虽然发现内存漏洞的难度较小，但逻辑漏洞更迫在眉睫、危害更大、更难识别。因此，许多分析人员选择首先找到用户权限相关(UPR)变量作为起点，以调查代码中可能使用UPR变量的地方是否存在任何漏洞，特别是逻辑漏洞。在本文中，我们介绍了一个大型语言模型(LLM)工作流，它可以帮助分析师识别这样的UPR变量，这被认为是一项非常耗时的任务。具体地说，我们的工具将审计程序中的所有变量，并为每个变量输出UPR分数，这是变量和用户权限之间的关系(密切程度)。提出的方法避免了直接提示LLM查找UPR变量的缺点，方法是专注于在语句级利用LLM，而不是向LLM提供非常长的代码片段。那些UPR得分较高的变量本质上是潜在的UPR变量，应手动调查。实验表明，使用典型的UPR评分阈值(即UPR评分>0.8)，错误正确率仅为13.49%，而UPR变量的发现明显多于基于启发式的方法。



## **35. Does CLIP Know My Face?**

CLIP认识我的脸吗？ cs.LG

Published in the Journal of Artificial Intelligence Research (JAIR)

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2209.07341v4) [paper-pdf](http://arxiv.org/pdf/2209.07341v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data have become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.

摘要: 随着深度学习在各种应用中的兴起，围绕训练数据保护的隐私问题已经成为一个关键的研究领域。鉴于以往的研究主要集中于单通道模型中的隐私风险，我们引入了一种新的方法来评估多通道模型的隐私，特别是像CLIP这样的视觉语言模型。提出的身份推断攻击(IDIA)通过用同一人的图像查询模型来揭示该人是否包括在训练数据中。让模型从各种各样的可能的文本标签中进行选择，该模型显示它是否识别出这个人，因此，它被用于训练。我们在CLIP上的大规模实验表明，用于训练的个体可以非常准确地识别。我们确认，该模型已经学会了将姓名与所描述的个人相关联，这意味着存在可被对手提取的敏感信息。我们的结果强调了在大规模模型中加强隐私保护的必要性，并建议可以使用IDIA来证明未经授权使用数据进行培训和执行隐私法。



## **36. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.03230v3) [paper-pdf](http://arxiv.org/pdf/2406.03230v3)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **37. Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment**

暴露隐私差距：对LLM一致偏好数据的会员推断攻击 cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06443v1) [paper-pdf](http://arxiv.org/pdf/2407.06443v1)

**Authors**: Qizhang Feng, Siva Rajesh Kasa, Hyokun Yun, Choon Hui Teo, Sravan Babu Bodapati

**Abstract**: Large Language Models (LLMs) have seen widespread adoption due to their remarkable natural language capabilities. However, when deploying them in real-world settings, it is important to align LLMs to generate texts according to acceptable human standards. Methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO) have made significant progress in refining LLMs using human preference data. However, the privacy concerns inherent in utilizing such preference data have yet to be adequately studied. In this paper, we investigate the vulnerability of LLMs aligned using human preference datasets to membership inference attacks (MIAs), highlighting the shortcomings of previous MIA approaches with respect to preference data. Our study has two main contributions: first, we introduce a novel reference-based attack framework specifically for analyzing preference data called PREMIA (\uline{Pre}ference data \uline{MIA}); second, we provide empirical evidence that DPO models are more vulnerable to MIA compared to PPO models. Our findings highlight gaps in current privacy-preserving practices for LLM alignment.

摘要: 大型语言模型(LLM)因其卓越的自然语言能力而被广泛采用。但是，在实际环境中部署它们时，重要的是对齐LLM以根据可接受的人类标准生成文本。最近策略优化(PPO)和直接偏好优化(DPO)等方法在利用人类偏好数据提炼LLMS方面取得了重大进展。然而，利用这种偏好数据所固有的隐私问题还没有得到充分的研究。在本文中，我们研究了使用人类偏好数据集对齐的最小似然模型对成员关系推理攻击(MIA)的脆弱性，强调了以往MIA方法在偏好数据方面的不足。我们的研究有两个主要贡献：第一，我们引入了一种新的基于参考的攻击框架，专门用于分析偏好数据，称为PremiA(PREMIA)；第二，我们提供了经验证据，表明DPO模型比PPO模型更容易受到MIA的影响。我们的发现突显了当前LLM比对隐私保护实践中的差距。



## **38. If You Don't Understand It, Don't Use It: Eliminating Trojans with Filters Between Layers**

如果你不明白，就不要使用它：用层之间的过滤器消除特洛伊木马 cs.LG

11 pages, 6 figures

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06411v1) [paper-pdf](http://arxiv.org/pdf/2407.06411v1)

**Authors**: Adriano Hernandez

**Abstract**: Large language models (LLMs) sometimes exhibit dangerous unintended behaviors. Finding and fixing these is challenging because the attack surface is massive -- it is not tractable to exhaustively search for all possible inputs that may elicit such behavior. One specific and particularly challenging case is that if data-poisoning-injected trojans, since there is no way to know what they are to search for them. To our knowledge, there is no generally applicable method to unlearn unknown trojans injected during pre-training. This work seeks to provide a general purpose recipe (filters) and a specific implementation (LoRA) filters that work in practice on small to medium sized models. The focus is primarily empirical, though some perplexing behavior opens the door to the fundamental question of how LLMs store and process information. Not unexpectedly, we find that our filters work best on the residual stream and the latest layers.

摘要: 大型语言模型（LLM）有时会表现出危险的非预期行为。找到和修复这些问题具有挑战性，因为攻击面是巨大的--无法彻底搜索可能引发此类行为的所有可能输入。一个具体且特别具有挑战性的情况是，如果数据中毒注入了特洛伊木马，因为没有办法知道它们是什么来搜索它们。据我们所知，没有普遍适用的方法可以消除训练前注射的未知特洛伊木马。这项工作旨在提供通用配方（过滤器）和特定实现（LoRA）过滤器，这些过滤器实际适用于中小型模型。重点主要是经验性的，尽管一些令人困惑的行为为LLM如何存储和处理信息的基本问题打开了大门。毫不奇怪，我们发现我们的过滤器对剩余流和最新层的效果最好。



## **39. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2401.17263v4) [paper-pdf](http://arxiv.org/pdf/2401.17263v4)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **40. Adaptive and robust watermark against model extraction attack**

抗模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **41. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **42. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2405.13401v4) [paper-pdf](http://arxiv.org/pdf/2405.13401v4)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **43. BadCLM: Backdoor Attack in Clinical Language Models for Electronic Health Records**

BadCLM：电子健康记录临床语言模型中的后门攻击 cs.CL

AMIA 2024

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05213v1) [paper-pdf](http://arxiv.org/pdf/2407.05213v1)

**Authors**: Weimin Lyu, Zexin Bi, Fusheng Wang, Chao Chen

**Abstract**: The advent of clinical language models integrated into electronic health records (EHR) for clinical decision support has marked a significant advancement, leveraging the depth of clinical notes for improved decision-making. Despite their success, the potential vulnerabilities of these models remain largely unexplored. This paper delves into the realm of backdoor attacks on clinical language models, introducing an innovative attention-based backdoor attack method, BadCLM (Bad Clinical Language Models). This technique clandestinely embeds a backdoor within the models, causing them to produce incorrect predictions when a pre-defined trigger is present in inputs, while functioning accurately otherwise. We demonstrate the efficacy of BadCLM through an in-hospital mortality prediction task with MIMIC III dataset, showcasing its potential to compromise model integrity. Our findings illuminate a significant security risk in clinical decision support systems and pave the way for future endeavors in fortifying clinical language models against such vulnerabilities.

摘要: 集成到电子健康记录(EHR)中用于临床决策支持的临床语言模型的出现标志着一项重大进步，它利用临床笔记的深度来改进决策。尽管取得了成功，但这些模型的潜在漏洞在很大程度上仍未被发掘。针对临床语言模型的后门攻击问题，提出了一种新的基于注意力的后门攻击方法BadCLM(Bad Clinic Language Model)。这种技术在模型中秘密地嵌入了一个后门，当输入中存在预定义的触发器时，导致模型产生错误的预测，而在其他情况下则准确地发挥作用。我们通过使用MIMIC III数据集的医院内死亡率预测任务，展示了BadCLM的有效性，展示了其损害模型完整性的潜力。我们的发现揭示了临床决策支持系统中的一个重大安全风险，并为未来加强临床语言模型以抵御此类漏洞的努力铺平了道路。



## **44. LLMCloudHunter: Harnessing LLMs for Automated Extraction of Detection Rules from Cloud-Based CTI**

LLMCloudHunter：利用LLM从基于云的RTI自动提取检测规则 cs.CR

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05194v1) [paper-pdf](http://arxiv.org/pdf/2407.05194v1)

**Authors**: Yuval Schwartz, Lavi Benshimol, Dudu Mimran, Yuval Elovici, Asaf Shabtai

**Abstract**: As the number and sophistication of cyber attacks have increased, threat hunting has become a critical aspect of active security, enabling proactive detection and mitigation of threats before they cause significant harm. Open-source cyber threat intelligence (OS-CTI) is a valuable resource for threat hunters, however, it often comes in unstructured formats that require further manual analysis. Previous studies aimed at automating OSCTI analysis are limited since (1) they failed to provide actionable outputs, (2) they did not take advantage of images present in OSCTI sources, and (3) they focused on on-premises environments, overlooking the growing importance of cloud environments. To address these gaps, we propose LLMCloudHunter, a novel framework that leverages large language models (LLMs) to automatically generate generic-signature detection rule candidates from textual and visual OSCTI data. We evaluated the quality of the rules generated by the proposed framework using 12 annotated real-world cloud threat reports. The results show that our framework achieved a precision of 92% and recall of 98% for the task of accurately extracting API calls made by the threat actor and a precision of 99% with a recall of 98% for IoCs. Additionally, 99.18% of the generated detection rule candidates were successfully compiled and converted into Splunk queries.

摘要: 随着网络攻击的数量和复杂性的增加，威胁追捕已经成为主动安全的一个关键方面，能够在威胁造成重大伤害之前主动检测和缓解威胁。开源网络威胁情报(OS-CTI)对于威胁猎人来说是一种宝贵的资源，然而，它通常是非结构化的格式，需要进一步的手动分析。以前旨在自动化OSCTI分析的研究是有限的，因为(1)它们未能提供可操作的输出，(2)它们没有利用OSCTI源中存在的图像，以及(3)它们专注于内部部署环境，忽视了云环境日益增长的重要性。为了弥补这些不足，我们提出了LLMCloudHunter，这是一个新的框架，它利用大型语言模型(LLM)来从文本和可视OSCTI数据自动生成通用签名检测规则候选。我们使用12个带注释的真实云威胁报告评估了所提出的框架生成的规则的质量。实验结果表明，该框架对于准确提取威胁行为人的API调用的准确率为92%，召回率为98%，对于IOC的准确率为99%，召回率为98%。此外，99.18%的生成检测规则候选被成功编译并转换为Splunk查询。



## **45. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

关于评估带有水印的机器生成文本在对抗性攻击下的性能 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04794v1) [paper-pdf](http://arxiv.org/pdf/2407.04794v1)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.

摘要: 大型语言模型(LLM)在各种应用中表现出色，包括文本生成和复杂任务。然而，LLMS的滥用引发了人们对它们产生的内容的真实性和伦理影响的担忧，例如深度假新闻、学术欺诈和侵犯版权。在机器生成的文本中嵌入可识别标记的水印技术，通过允许内容验证和来源追踪，为这些问题提供了一种有前途的解决方案。遗憾的是，目前的LLM水印方案在潜在的水印去除攻击下的稳健性还没有得到全面的研究。为了填补这一空白，本文首先对主流的机器生成文本水印算法和去除攻击进行了系统的梳理，然后将其分为前文本类(文本生成前)和后文本类(文本生成后)，以便进行多样化的分析。在我们的实验中，我们评估了87个场景中的8个水印(5个前置文本，3个后置文本)和12个攻击(2个前置文本，10个后置文本)。评估结果表明：(1)KGW和指数水印具有高的文本质量和水印保留率，但仍然容易受到大多数攻击；(2)后文本攻击被发现比前文本攻击更有效和实用；(3)前文本水印通常更不可察觉，因为它们不像后文本水印那样改变文本的流畅性；(4)此外，组合攻击方法可以显著提高攻击效果，突出了对更健壮的水印解决方案的需求。我们的研究强调了当前技术的脆弱性，以及开发更具弹性的方案的必要性。



## **46. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

控制耳语：控制语音基础模型的通用声学对抗攻击 cs.SD

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04482v1) [paper-pdf](http://arxiv.org/pdf/2407.04482v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.

摘要: 以灵活的基于语音识别的系统或音频提示的大型语言模型(LLM)的形式启用语音的基础模型正变得越来越受欢迎。这些模型的一个有趣方面是，它们能够使用适当的提示执行自动语音识别(ASR)以外的任务。例如，OpenAI Whisper模型可以执行语音转录和语音翻译。随着音频提示LLMS的发展，有可能出现更大的控制选项。在这项工作中，我们证明了有了这种更大的灵活性，系统可以容易受到模型控制的对抗性攻击。在不访问模型提示的情况下，可以通过适当地改变音频输入来修改系统的行为。为了说明这一风险，我们证明了有可能在任何输入语音信号之前添加一个简短的通用对抗性声学片段，以覆盖ASR基础模型的提示设置。具体地说，我们成功地使用了一个通用的对抗性声学段来控制Whisper始终执行语音翻译，尽管被设置为执行语音转录。总体而言，这项工作展示了一种对多任务语音启用的基础模型的新形式的对抗性攻击，在部署这种形式的模型之前需要考虑这种形式。



## **47. Waterfall: Framework for Robust and Scalable Text Watermarking**

瀑布：稳健且可扩展的文本水印框架 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04411v1) [paper-pdf](http://arxiv.org/pdf/2407.04411v1)

**Authors**: Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, Bryan Kian Hsiang Low

**Abstract**: Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and also showed how it could be directly applied to the watermarking of code.

摘要: 保护文章和代码等文本的知识产权(IP)越来越重要，特别是在可能进行复杂攻击的情况下，例如利用大型语言模型(LLM)进行释义，甚至未经授权对受版权保护的文本进行LLM培训，以侵犯此类IP。然而，现有的文本水印方法对此类攻击不够健壮，也不能扩展到数百万用户进行实际实现。在本文中，我们提出了瀑布，这是第一个无训练的文本水印框架，适用于LLMS支持的多种文本类型(例如，文章、代码)和语言，用于一般文本和LLM数据来源。瀑布由几项关键创新组成，例如第一个使用LLM作为水印解释程序，以及在实现强大的可验证性和可扩展性方面出人意料地有效的技术组合。实验证明，与SOTA的文章-文本水印方法相比，瀑布算法具有更好的可扩展性、健壮的可验证性和计算效率，并且可以直接应用于代码的水印。



## **48. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

针对大型语言模型的越狱攻击和防御：调查 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04295v1) [paper-pdf](http://arxiv.org/pdf/2407.04295v1)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.

摘要: 大型语言模型(LLMS)在问答、翻译、代码补全等文本生成任务中表现出色。然而，LLMS的过度协助带来了越狱的挑战，这导致该模型通过设计敌意提示来生成针对使用策略和社会的恶意响应。随着利用LLMS中不同漏洞的越狱攻击方法的出现，相应的安全对齐措施也在不断发展。在本文中，我们提出了一个全面和详细的分类越狱攻防方法。例如，根据目标模型的透明性，将攻击方法分为黑盒攻击和白盒攻击。同时，我们将防御方法分为提示级防御和模型级防御。此外，我们还将这些攻击和防御方法进一步细分为不同的子类，并提供了一个连贯的图来说明它们之间的关系。我们还对现有的评估方法进行了调查，并从不同的角度对它们进行了比较。我们的发现旨在启发未来在保护LLM免受对手攻击方面的研究和实际实现。最重要的是，尽管越狱在社区中仍然是一个重要的问题，但我们相信我们的工作增进了对这个领域的了解，并为开发更安全的LLM提供了基础。



## **49. Defending Jailbreak Prompts via In-Context Adversarial Game**

通过上下文对抗游戏为越狱辩护 cs.LG

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2402.13148v2) [paper-pdf](http://arxiv.org/pdf/2402.13148v2)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.

摘要: 大型语言模型(LLM)在不同的应用程序中展示了卓越的功能。然而，对他们的安全，特别是对越狱攻击的脆弱性的担忧依然存在。从深度学习和LLM代理学习过程中的对抗性训练中获得灵感，我们引入了无需微调的上下文对抗性游戏(ICAG)来防御越狱。ICAG利用代理学习进行对抗性游戏，旨在动态扩展知识来防御越狱。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一不断改进的过程加强了对新生成的越狱提示的防御。我们的经验研究肯定了ICAG的有效性，在不同的攻击场景中，由ICAG保护的LLM显示出显著降低的越狱成功率。此外，ICAG表现出显著的可转移性，表明其作为一种多功能防御机制的潜力。



## **50. Defense Against Syntactic Textual Backdoor Attacks with Token Substitution**

利用令牌替换防御语法文本后门攻击 cs.CL

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04179v1) [paper-pdf](http://arxiv.org/pdf/2407.04179v1)

**Authors**: Xinglin Li, Xianwen He, Yao Li, Minhao Cheng

**Abstract**: Textual backdoor attacks present a substantial security risk to Large Language Models (LLM). It embeds carefully chosen triggers into a victim model at the training stage, and makes the model erroneously predict inputs containing the same triggers as a certain class. Prior backdoor defense methods primarily target special token-based triggers, leaving syntax-based triggers insufficiently addressed. To fill this gap, this paper proposes a novel online defense algorithm that effectively counters syntax-based as well as special token-based backdoor attacks. The algorithm replaces semantically meaningful words in sentences with entirely different ones but preserves the syntactic templates or special tokens, and then compares the predicted labels before and after the substitution to determine whether a sentence contains triggers. Experimental results confirm the algorithm's performance against these two types of triggers, offering a comprehensive defense strategy for model integrity.

摘要: 文本后门攻击给大型语言模型（LLM）带来了巨大的安全风险。它在训练阶段将精心选择的触发器嵌入到受害者模型中，并使模型错误地预测包含与某个类别相同触发器的输入。先前的后门防御方法主要针对特殊的基于令牌的触发器，而基于语法的触发器则没有得到充分解决。为了填补这一空白，本文提出了一种新颖的在线防御算法，可以有效地对抗基于语法以及特殊的基于令牌的后门攻击。该算法用完全不同的单词替换句子中具有语义意义的单词，但保留语法模板或特殊标记，然后比较替换前后的预测标签，以确定句子是否包含触发器。实验结果证实了该算法针对这两种类型触发器的性能，为模型完整性提供了全面的防御策略。



