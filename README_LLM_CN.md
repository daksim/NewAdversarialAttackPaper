# Latest Large Language Model Attack Papers
**update at 2024-06-11 10:36:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LLM Dataset Inference: Did you train on my dataset?**

LLM数据集推理：您使用我的数据集进行训练了吗？ cs.LG

Code is available at  \href{https://github.com/pratyushmaini/llm_dataset_inference/

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06443v1) [paper-pdf](http://arxiv.org/pdf/2406.06443v1)

**Authors**: Pratyush Maini, Hengrui Jia, Nicolas Papernot, Adam Dziedzic

**Abstract**: The proliferation of large language models (LLMs) in the real world has come with a rise in copyright cases against companies for training their models on unlicensed data from the internet. Recent works have presented methods to identify if individual text sequences were members of the model's training data, known as membership inference attacks (MIAs). We demonstrate that the apparent success of these MIAs is confounded by selecting non-members (text sequences not used for training) belonging to a different distribution from the members (e.g., temporally shifted recent Wikipedia articles compared with ones used to train the model). This distribution shift makes membership inference appear successful. However, most MIA methods perform no better than random guessing when discriminating between members and non-members from the same distribution (e.g., in this case, the same period of time). Even when MIAs work, we find that different MIAs succeed at inferring membership of samples from different distributions. Instead, we propose a new dataset inference method to accurately identify the datasets used to train large language models. This paradigm sits realistically in the modern-day copyright landscape, where authors claim that an LLM is trained over multiple documents (such as a book) written by them, rather than one particular paragraph. While dataset inference shares many of the challenges of membership inference, we solve it by selectively combining the MIAs that provide positive signal for a given distribution, and aggregating them to perform a statistical test on a given dataset. Our approach successfully distinguishes the train and test sets of different subsets of the Pile with statistically significant p-values < 0.1, without any false positives.

摘要: 随着大型语言模型(LLM)在现实世界中的激增，针对利用互联网上未经许可的数据对其模型进行培训的公司的版权案件也有所增加。最近的工作提出了一些方法来识别单个文本序列是否是模型训练数据的成员，称为成员关系推理攻击(MIA)。我们证明，通过选择属于与成员不同分布的非成员(不用于训练的文本序列)(例如，与用于训练模型的文章相比，最近的维基百科文章在时间上发生了偏移)，这些MIA的表面成功被混淆了。这种分布转移使成员关系推断看起来很成功。然而，大多数MIA方法在区分来自相同分布(例如，在这种情况下，相同的时间段)的成员和非成员时，并不比随机猜测更好。即使当MIA起作用时，我们发现不同的MIA成功地推断出来自不同分布的样本的成员资格。相反，我们提出了一种新的数据集推理方法，以准确识别用于训练大型语言模型的数据集。这种模式现实地存在于现代版权领域，作者声称，LLM是针对他们编写的多个文档(如一本书)进行培训的，而不是针对一个特定的段落。虽然数据集推理与成员关系推理一样存在许多挑战，但我们通过选择性地组合为给定分布提供正信号的MIA，并将它们聚合起来对给定数据集执行统计测试来解决这一问题。我们的方法成功地区分了具有统计意义的p值<0.1的堆的不同子集的训练集和测试集，没有任何误报。



## **2. Are you still on track!? Catching LLM Task Drift with Activations**

你还在正轨上吗！？通过激活捕捉LLM任务漂移 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.00799v2) [paper-pdf](http://arxiv.org/pdf/2406.00799v2)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.

摘要: 大型语言模型(LLM)通常用于检索增强的应用程序中，以协调任务并处理来自用户和其他来源的输入。这些输入，即使是在单个LLM交互中，也可以来自各种来源，具有不同的可信度和出处。这为即时注入攻击打开了大门，在这种情况下，LLM接收来自假定仅限数据的来源的指令并对其采取行动，从而偏离用户的原始指令。我们将其定义为任务漂移，并建议通过扫描和分析LLM的激活来捕获它。我们比较LLM在处理外部输入之前和之后的激活，以检测该输入是否导致指令漂移。我们开发了两种探测方法，发现简单地使用线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们表明，这种方法对于看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。我们的设置不需要对LLM进行任何修改(例如，微调)或任何文本生成，从而最大限度地提高可部署性和成本效益，并避免依赖不可靠的模型输出。为了促进未来对基于激活的任务检测、解码和可解释性的研究，我们将发布我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自4个SOTA语言模型的表示和检测工具。



## **3. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

让他们问答：通过伪装和重建在很短的时间内越狱大型语言模型 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.18104v2) [paper-pdf](http://arxiv.org/pdf/2402.18104v2)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and closed-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 91.1% attack success rate on OpenAI GPT-4 chatbot.

摘要: 近年来，大型语言模型（LLM）在各种任务中取得了显着的成功，但LLM的可信度仍然是一个悬而未决的问题。一个具体的威胁是可能产生有毒或有害反应。攻击者可以设计对抗性提示，引发LLM的有害反应。在这项工作中，我们通过识别安全微调中的偏见漏洞，开创了LLM安全的理论基础，并设计了一种名为“伪装和重建攻击”的黑匣子越狱方法，通过伪装隐藏有害指令，并促使模型在完成时重建原始有害指令。我们评估各种开源和开源模型的NPS，展示最先进的越狱成功率和攻击效率。值得注意的是，Inbox对OpenAI GPT-4聊天机器人的攻击成功率为91.1%。



## **4. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models**

CARES：医学视觉语言模型可信度的综合基准 cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06007v1) [paper-pdf](http://arxiv.org/pdf/2406.06007v1)

**Authors**: Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, Wenhao Zheng, Zhaoyang Wang, Xiao Wang, Xuchao Zhang, Chetan Bansal, Marc Niethammer, Junzhou Huang, Hongtu Zhu, Yun Li, Jimeng Sun, Zongyuan Ge, Gang Li, James Zou, Huaxiu Yao

**Abstract**: Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://github.com/richard-peng-xia/CARES.

摘要: 人工智能对医疗应用产生了重大影响，特别是随着医学大视觉语言模型(Med-LVLMS)的出现，引发了对自动化和个性化医疗保健未来的乐观情绪。然而，MED-LVLMS的可信性仍未得到验证，对未来的模型部署构成重大风险。在本文中，我们引入CARE，旨在全面评估医学领域的MED-LVLMS的可信性。我们从可信性、公平性、安全性、隐私性和健壮性五个维度评估Med-LVLM的可信性。CARE包括约41K个封闭式和开放式格式的问答对，涵盖16种医学影像模式和27个解剖区域。我们的分析表明，这些模型始终表现出对可信度的担忧，经常表现出事实上的不准确，并且未能在不同的人口群体中保持公平。此外，他们很容易受到攻击，并表现出缺乏隐私意识。我们在https://github.com/richard-peng-xia/CARES.中公开发布我们的基准测试和代码



## **5. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

审查链：检测大型语言模型的后门攻击 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05948v1) [paper-pdf](http://arxiv.org/pdf/2406.05948v1)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Backdoor attacks present significant threats to Large Language Models (LLMs), particularly with the rise of third-party services that offer API integration and prompt engineering. Untrustworthy third parties can plant backdoors into LLMs and pose risks to users by embedding malicious instructions into user queries. The backdoor-compromised LLM will generate malicious output when and input is embedded with a specific trigger predetermined by an attacker. Traditional defense strategies, which primarily involve model parameter fine-tuning and gradient calculation, are inadequate for LLMs due to their extensive computational and clean data requirements. In this paper, we propose a novel solution, Chain-of-Scrutiny (CoS), to address these challenges. Backdoor attacks fundamentally create a shortcut from the trigger to the target output, thus lack reasoning support. Accordingly, CoS guides the LLMs to generate detailed reasoning steps for the input, then scrutinizes the reasoning process to ensure consistency with the final answer. Any inconsistency may indicate an attack. CoS only requires black-box access to LLM, offering a practical defense, particularly for API-accessible LLMs. It is user-friendly, enabling users to conduct the defense themselves. Driven by natural language, the entire defense process is transparent to users. We validate the effectiveness of CoS through extensive experiments across various tasks and LLMs. Additionally, experiments results shows CoS proves more beneficial for more powerful LLMs.

摘要: 后门攻击对大型语言模型(LLM)构成了重大威胁，特别是随着提供API集成和快速工程的第三方服务的兴起。不可信的第三方可以在LLMS中植入后门，并通过在用户查询中嵌入恶意指令来给用户带来风险。当AND INPUT嵌入攻击者预先确定的特定触发器时，受后门攻击的LLM将生成恶意输出。传统的防御策略主要涉及模型参数微调和梯度计算，但由于其计算量大、数据清晰，不适用于LLMS。在本文中，我们提出了一种新的解决方案，即审查链(CoS)，以应对这些挑战。后门攻击从根本上创建了一条从触发器到目标输出的捷径，因此缺乏推理支持。相应地，COS指导LLMS为输入生成详细的推理步骤，然后仔细检查推理过程以确保与最终答案一致。任何不一致都可能表示发生了攻击。CoS只需要黑盒访问LLM，提供了实用的防御，特别是对于API可访问的LLM。它是用户友好的，使用户能够自己进行防御。在自然语言的驱动下，整个防御过程对用户是透明的。我们通过跨各种任务和LLM的大量实验验证了CoS的有效性。此外，实验结果表明，CoS被证明对更强大的LLM更有利。



## **6. Safety Alignment Should Be Made More Than Just a Few Tokens Deep**

安全调整不应仅仅深入一些代币 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05946v1) [paper-pdf](http://arxiv.org/pdf/2406.05946v1)

**Authors**: Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, Peter Henderson

**Abstract**: The safety alignment of current Large Language Models (LLMs) is vulnerable. Relatively simple attacks, or even benign fine-tuning, can jailbreak aligned models. We argue that many of these vulnerabilities are related to a shared underlying issue: safety alignment can take shortcuts, wherein the alignment adapts a model's generative distribution primarily over only its very first few output tokens. We refer to this issue as shallow safety alignment. In this paper, we present case studies to explain why shallow safety alignment can exist and provide evidence that current aligned LLMs are subject to this issue. We also show how these findings help explain multiple recently discovered vulnerabilities in LLMs, including the susceptibility to adversarial suffix attacks, prefilling attacks, decoding parameter attacks, and fine-tuning attacks. Importantly, we discuss how this consolidated notion of shallow safety alignment sheds light on promising research directions for mitigating these vulnerabilities. For instance, we show that deepening the safety alignment beyond just the first few tokens can often meaningfully improve robustness against some common exploits. Finally, we design a regularized finetuning objective that makes the safety alignment more persistent against fine-tuning attacks by constraining updates on initial tokens. Overall, we advocate that future safety alignment should be made more than just a few tokens deep.

摘要: 当前大型语言模型(LLM)的安全对齐是易受攻击的。相对简单的攻击，甚至是温和的微调，都可以让结盟的模型越狱。我们认为，这些漏洞中的许多都与一个共同的潜在问题有关：安全对齐可以走捷径，其中对齐主要适应模型的生成性分布，仅在其最初的几个输出令牌上。我们将这个问题称为浅层安全对准。在这篇文章中，我们提供了案例研究来解释为什么浅层安全对准可以存在，并提供证据表明当前对准的LLM受到这个问题的影响。我们还展示了这些发现如何帮助解释LLMS中最近发现的多个漏洞，包括对敌意后缀攻击、预填充攻击、解码参数攻击和微调攻击的敏感性。重要的是，我们讨论了浅层安全对齐这一统一概念如何揭示了缓解这些漏洞的有前途的研究方向。例如，我们表明，除了最初的几个令牌之外，深化安全对齐通常可以有意义地提高对一些常见漏洞的健壮性。最后，我们设计了一个正则化的精调目标，通过限制对初始令牌的更新，使安全对齐更持久地抵抗微调攻击。总体而言，我们主张未来的安全调整应该不仅仅是几个标志的深度。



## **7. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2402.06255v2) [paper-pdf](http://arxiv.org/pdf/2402.06255v2)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both black-box and white-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在有害内容过滤或启发式防御提示设计上。然而，如何通过提示实现内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对黑盒攻击和白盒攻击都是有效的，在保持模型对良性任务的实用性的同时，将高级攻击的成功率降低到近0。所提出的防御策略只需要很少的计算开销，为未来在LLM安全方面的探索开辟了新的前景。我们的代码可以在https://github.com/rain152/PAT.上找到



## **8. Adaptive Text Watermark for Large Language Models**

大型语言模型的自适应文本水印 cs.CL

ICML2024

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2401.13927v2) [paper-pdf](http://arxiv.org/pdf/2401.13927v2)

**Authors**: Yepeng Liu, Yuheng Bu

**Abstract**: The advancement of Large Language Models (LLMs) has led to increasing concerns about the misuse of AI-generated text, and watermarking for LLM-generated text has emerged as a potential solution. However, it is challenging to generate high-quality watermarked text while maintaining strong security, robustness, and the ability to detect watermarks without prior knowledge of the prompt or model. This paper proposes an adaptive watermarking strategy to address this problem. To improve the text quality and maintain robustness, we adaptively add watermarking to token distributions with high entropy measured using an auxiliary model and keep the low entropy token distributions untouched. For the sake of security and to further minimize the watermark's impact on text quality, instead of using a fixed green/red list generated from a random secret key, which can be vulnerable to decryption and forgery, we adaptively scale up the output logits in proportion based on the semantic embedding of previously generated text using a well designed semantic mapping model. Our experiments involving various LLMs demonstrate that our approach achieves comparable robustness performance to existing watermark methods. Additionally, the text generated by our method has perplexity comparable to that of \emph{un-watermarked} LLMs while maintaining security even under various attacks.

摘要: 大型语言模型(LLM)的发展引起了人们对人工智能生成文本滥用的日益关注，LLM生成文本的水印已经成为一种潜在的解决方案。然而，在不预先知道提示或模型的情况下，在保持强大的安全性、健壮性和检测水印的能力的同时，生成高质量的水印文本是具有挑战性的。针对这一问题，本文提出了一种自适应水印策略。为了提高文本质量和保持稳健性，我们在使用辅助模型测量的高熵的令牌分布上自适应地添加水印，而对低熵的令牌分布保持不变。为了保证安全性，并进一步减小水印对文本质量的影响，不使用由随机密钥生成的易被解密和伪造的固定绿/红列表，而是使用设计良好的语义映射模型，基于先前生成的文本的语义嵌入，按比例自适应地放大输出逻辑。实验表明，该方法具有与现有水印方法相当的稳健性。此外，该方法生成的文本具有与未加水印的LLMS相当的复杂性，同时即使在各种攻击下也能保持安全性。



## **9. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05498v1) [paper-pdf](http://arxiv.org/pdf/2406.05498v1)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into four major categories: optimization-based attacks such as Greedy Coordinate Gradient (GCG), jailbreak template-based attacks such as "Do-Anything-Now", advanced indirect attacks like DrAttack, and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delay to user prompts, as well as be compatible with both open-source and closed-source LLMs.   Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. Our measurements show that SelfDefend enables GPT-3.5 to suppress the attack success rate (ASR) by 8.97-95.74% (average: 60%) and GPT-4 by even 36.36-100% (average: 83%), while incurring negligible effects on normal queries. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform four SOTA defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to targeted GCG and prompt injection attacks.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为四大类：基于优化的攻击(如贪婪坐标梯度(GCG))、基于模板的攻击(如Do-Anything-Now)、高级间接攻击(如DrAttack)和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend，该框架通过建立一个影子LLM防御实例来同时保护正常堆栈中的目标LLM实例，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM(目标和防御LLM)能够识别用户查询中的有害提示或意图，我们使用所有主要越狱攻击中常用的GPT-3.5/4模型进行了经验验证。我们的测试表明，SelfDefend使GPT-3.5的攻击成功率(ASR)降低了8.97-95.74%(平均：60%)，GPT-4的攻击成功率(ASR)降低了36.36-100%(平均：83%)，而对正常查询的影响可以忽略不计。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。这些型号的性能超过了四种SOTA防御系统，并与基于GPT-4的SelfDefend的性能相当，额外延迟明显更低。我们的实验还表明，调整后的模型对目标GCG攻击和即时注入攻击具有较强的鲁棒性。



## **10. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

MM-SafetyBench：多模式大型语言模型安全评估的基准 cs.CV

The datasets were incomplete as they did not include all the  necessary copyrights

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2311.17600v3) [paper-pdf](http://arxiv.org/pdf/2311.17600v3)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

摘要: 围绕大语言模型的安全问题已经得到了广泛的研究，但多模式大语言模型的安全性仍未得到充分的研究。在本文中，我们观察到多模式大型语言模型(MLLMS)很容易被与查询相关的图像破坏，就好像文本查询本身是恶意的一样。为了解决这一问题，我们引入了MM-SafetyBch，这是一个全面的框架，旨在针对此类基于图像的操作对MLLMS进行安全关键评估。我们汇编了一个包含13个场景的数据集，总共产生了5,040个文本-图像对。我们对12种最先进型号的分析表明，即使配备的LLM已经安全对准，MLLM也容易受到我们的方法引发的漏洞的影响。对此，我们提出了一种简单而有效的提示策略，以增强MLLMS对这些类型攻击的弹性。我们的工作强调了需要齐心协力加强和改进开放源码MLLM的安全措施，以防范潜在的恶意利用。该资源位于\href{此HTTPS URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **11. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一个扰动就足够了：关于针对视觉语言预训练模型生成普遍对抗性扰动 cs.CV

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05491v1) [paper-pdf](http://arxiv.org/pdf/2406.05491v1)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models trained on large-scale image-text pairs have demonstrated unprecedented capability in many practical applications. However, previous studies have revealed that VLP models are vulnerable to adversarial samples crafted by a malicious adversary. While existing attacks have achieved great success in improving attack effect and transferability, they all focus on instance-specific attacks that generate perturbations for each input sample. In this paper, we show that VLP models can be vulnerable to a new class of universal adversarial perturbation (UAP) for all input samples. Although initially transplanting existing UAP algorithms to perform attacks showed effectiveness in attacking discriminative models, the results were unsatisfactory when applied to VLP models. To this end, we revisit the multimodal alignments in VLP model training and propose the Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC). Specifically, we first design a generator that incorporates cross-modal information as conditioning input to guide the training. To further exploit cross-modal interactions, we propose to formulate the training objective as a multimodal contrastive learning paradigm based on our constructed positive and negative image-text pairs. By training the conditional generator with the designed loss, we successfully force the adversarial samples to move away from its original area in the VLP model's feature space, and thus essentially enhance the attacks. Extensive experiments show that our method achieves remarkable attack performance across various VLP models and Vision-and-Language (V+L) tasks. Moreover, C-PGC exhibits outstanding black-box transferability and achieves impressive results in fooling prevalent large VLP models including LLaVA and Qwen-VL.

摘要: 在大规模图文对上训练的视觉语言预训练(VLP)模型在许多实际应用中表现出了前所未有的能力。然而，以前的研究表明，VLP模型容易受到恶意对手制作的敌意样本的攻击。虽然现有的攻击在提高攻击效果和可转移性方面取得了很大的成功，但它们都专注于针对特定实例的攻击，这些攻击会对每个输入样本产生扰动。在本文中，我们证明了VLP模型对于所有输入样本都可能受到一类新的通用对抗性摄动(UAP)的影响。虽然最初移植现有的UAP算法进行攻击对区分模型的攻击是有效的，但将其应用于VLP模型时，效果并不理想。为此，我们回顾了VLP模型训练中的多模式对齐，并提出了具有跨模式条件的对比训练扰动生成器(C-PGC)。具体地说，我们首先设计了一个生成器，它结合了跨通道信息作为条件输入来指导训练。为了进一步开发跨通道互动，我们建议将培训目标制定为基于我们构建的正面和负面图文对的多通道对比学习范式。通过用设计的损失训练条件生成器，我们成功地迫使敌方样本在VLP模型的特征空间中离开其原始区域，从而从根本上增强了攻击。大量的实验表明，我们的方法在不同的视觉和语言(V+L)任务上取得了显著的攻击性能。此外，C-PGC表现出出色的黑盒可转移性，并在愚弄LLaVA和Qwen-VL等流行的大型VLP模型方面取得了令人印象深刻的结果。



## **12. PRSA: PRompt Stealing Attacks against Large Language Models**

PRSA：针对大型语言模型的PRompt窃取攻击 cs.CR

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2402.19200v2) [paper-pdf](http://arxiv.org/pdf/2402.19200v2)

**Authors**: Yong Yang, Changjiang Li, Yi Jiang, Xi Chen, Haoyu Wang, Xuhong Zhang, Zonghui Wang, Shouling Ji

**Abstract**: In recent years, "prompt as a service" has greatly enhanced the utility of large language models (LLMs) by enabling them to perform various downstream tasks efficiently without fine-tuning. This has also increased the commercial value of prompts. However, the potential risk of leakage in these commercialized prompts remains largely underexplored. In this paper, we introduce a novel attack framework, PRSA, designed for prompt stealing attacks against LLMs. The main idea of PRSA is to infer the intent behind a prompt by analyzing its input-output content, enabling the generation of a surrogate prompt that replicates the original's functionality. Specifically, PRSA mainly consists of two key phases: prompt mutation and prompt pruning. In the mutation phase, we propose a prompt attention algorithm based on output difference. The algorithm facilitates the generation of effective surrogate prompts by learning key factors that influence the accurate inference of prompt intent. During the pruning phase, we employ a two-step related word identification strategy to detect and mask words that are highly related to the input, thus improving the generalizability of the surrogate prompts. We verify the actual threat of PRSA through evaluation in both real-world settings, non-interactive and interactive prompt services. The results strongly confirm the PRSA's effectiveness and generalizability. We have reported these findings to prompt service providers and actively collaborate with them to implement defensive measures.

摘要: 近年来，“提示即服务”极大地增强了大型语言模型(LLM)的实用性，使它们能够高效地执行各种下游任务，而无需微调。这也增加了提示的商业价值。然而，这些商业化提示中的潜在泄漏风险在很大程度上仍然没有得到充分的探索。本文提出了一种新的攻击框架--PRSA，用于快速窃取LLMS攻击。PRSA的主要思想是通过分析提示的输入输出内容来推断提示背后的意图，从而能够生成复制原始提示功能的代理提示。具体而言，PRSA算法主要由两个关键阶段组成：快速突变和快速剪枝。在变异阶段，我们提出了一种基于输出差异的快速注意算法。该算法通过学习影响提示意图准确推理的关键因素，有助于生成有效的代理提示。在剪枝阶段，我们采用了两步关联词识别策略来检测和掩蔽与输入高度相关的词，从而提高了代理提示的泛化能力。我们通过在真实世界环境下、非交互和交互提示服务中的评估来验证PRSA的实际威胁。结果有力地证实了PRSA的有效性和泛化能力。我们已将这些发现报告给服务提供商，并积极与他们合作实施防御措施。



## **13. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.03230v2) [paper-pdf](http://arxiv.org/pdf/2406.03230v2)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **14. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

ArtPrompt：针对对齐的LLM的基于ASC艺术的越狱攻击 cs.CL

To appear in ACL 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.11753v4) [paper-pdf](http://arxiv.org/pdf/2402.11753v4)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs. Our code is available at https://github.com/uw-nsl/ArtPrompt.

摘要: 安全对于大型语言模型(LLM)的使用至关重要。已经开发了多种技术，如数据过滤和有监督的微调，以加强LLM的安全性。然而，目前已知的技术假定用于LLM的安全对准的语料库仅由语义解释。然而，这一假设在现实世界的应用程序中并不成立，这导致了LLMS中的严重漏洞。例如，论坛的用户经常使用ASCII艺术，这是一种基于文本的艺术形式，以传达图像信息。本文提出了一种新的基于ASCII ART的越狱攻击方法，并引入了一个综合基准的文本中视觉挑战(VITC)来评估LLMS在识别不能完全由语义解释的提示方面的能力。我们发现，五个SOTA LLM(GPT-3.5、GPT-4、双子座、克劳德和Llama2)很难识别以ASCII ART形式提供的提示。基于这种观察，我们开发了越狱攻击ArtPrompt，它利用LLMS在识别ASCII ART方面的较差性能来绕过安全措施，并从LLM引发不希望看到的行为。ArtPrompt只需要黑盒访问受攻击的LLM，这使其成为一种实际的攻击。我们在五个SOTA LLM上对ArtPrompt进行了评估，结果表明，ArtPrompt可以有效和高效地诱导所有五个LLM的不良行为。我们的代码可以在https://github.com/uw-nsl/ArtPrompt.上找到



## **15. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

SafeDecoding：通过安全意识解码防御越狱攻击 cs.CR

To appear in ACL 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.08983v3) [paper-pdf](http://arxiv.org/pdf/2402.08983v3)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.

摘要: 随着大型语言模型(LLM)越来越多地集成到真实世界的应用中，如代码生成和聊天机器人辅助，人们已经做出了广泛的努力来使LLM的行为与包括安全在内的人类价值观保持一致。越狱攻击旨在挑起LLM的意外和不安全行为，仍然是LLM的重大/主要安全威胁。在本文中，我们的目标是通过引入SafeDecoding来防御LLMS的越狱攻击，SafeDecoding是一种安全感知的解码策略，用于LLMS对用户查询生成有用和无害的响应。我们开发SafeDecoding的洞察力基于这样的观察：即使代表有害内容的令牌的概率大于代表无害响应的令牌的概率，但在按概率降序对令牌进行排序后，安全免责声明仍会出现在排名最靠前的令牌中。这使我们能够通过识别安全免责声明并放大其令牌概率来缓解越狱攻击，同时降低与越狱攻击目标一致的令牌序列的概率。我们使用六个最先进的越狱攻击和四个基准数据集在五个LLM上进行了广泛的实验。我们的结果表明，SafeDecoding在不影响对良性用户查询的响应的帮助的情况下，显著降低了越狱攻击的攻击成功率和危害性。安全解码的性能超过了六种防御方法。



## **16. DepsRAG: Towards Managing Software Dependencies using Large Language Models**

DepsRAG：使用大型语言模型管理软件附属机构 cs.SE

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2405.20455v3) [paper-pdf](http://arxiv.org/pdf/2405.20455v3)

**Authors**: Mohannad Alhanahnah, Yazan Boshmaf, Benoit Baudry

**Abstract**: Managing software dependencies is a crucial maintenance task in software development and is becoming a rapidly growing research field, especially in light of the significant increase in software supply chain attacks. Specialized expertise and substantial developer effort are required to fully comprehend dependencies and reveal hidden properties about the dependencies (e.g., number of dependencies, dependency chains, depth of dependencies).   Recent advancements in Large Language Models (LLMs) allow the retrieval of information from various data sources for response generation, thus providing a new opportunity to uniquely manage software dependencies. To highlight the potential of this technology, we present~\tool, a proof-of-concept Retrieval Augmented Generation (RAG) approach that constructs direct and transitive dependencies of software packages as a Knowledge Graph (KG) in four popular software ecosystems. DepsRAG can answer user questions about software dependencies by automatically generating necessary queries to retrieve information from the KG, and then augmenting the input of LLMs with the retrieved information. DepsRAG can also perform Web search to answer questions that the LLM cannot directly answer via the KG. We identify tangible benefits that DepsRAG can offer and discuss its limitations.

摘要: 管理软件依赖关系是软件开发中一项重要的维护任务，特别是在软件供应链攻击显著增加的情况下，软件依赖关系管理正在成为一个快速增长的研究领域。要完全理解依赖关系并揭示依赖关系的隐藏属性(例如，依赖关系的数量、依赖关系链、依赖关系的深度)，需要专业的专业知识和大量的开发人员工作。大型语言模型(LLM)的最新进展允许从各种数据源检索信息以生成响应，从而为独特地管理软件依赖关系提供了新的机会。为了突出这一技术的潜力，我们提出了一种概念验证检索增强生成(RAG)方法~\Tool，它将软件包的直接和传递依赖构造为四个流行的软件生态系统中的知识图(KG)。DepsRAG可以通过自动生成必要的查询来从KG中检索信息，然后用检索到的信息增强LLMS的输入，来回答用户关于软件依赖性的问题。DepsRAG还可以执行网络搜索，以回答LLM无法通过KG直接回答的问题。我们确定了DepsRAG可以提供的实实在在的好处并讨论了其局限性。



## **17. SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**

SALAD-Bench：大型语言模型的分层和全面的安全基准 cs.CL

Accepted at ACL 2024 Findings

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.05044v4) [paper-pdf](http://arxiv.org/pdf/2402.05044v4)

**Authors**: Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao

**Abstract**: In the rapidly evolving landscape of Large Language Models (LLMs), ensuring robust safety measures is paramount. To meet this crucial need, we propose \emph{SALAD-Bench}, a safety benchmark specifically designed for evaluating LLMs, attack, and defense methods. Distinguished by its breadth, SALAD-Bench transcends conventional benchmarks through its large scale, rich diversity, intricate taxonomy spanning three levels, and versatile functionalities.SALAD-Bench is crafted with a meticulous array of questions, from standard queries to complex ones enriched with attack, defense modifications and multiple-choice. To effectively manage the inherent complexity, we introduce an innovative evaluators: the LLM-based MD-Judge for QA pairs with a particular focus on attack-enhanced queries, ensuring a seamless, and reliable evaluation. Above components extend SALAD-Bench from standard LLM safety evaluation to both LLM attack and defense methods evaluation, ensuring the joint-purpose utility. Our extensive experiments shed light on the resilience of LLMs against emerging threats and the efficacy of contemporary defense tactics. Data and evaluator are released under https://github.com/OpenSafetyLab/SALAD-BENCH.

摘要: 在快速发展的大型语言模型(LLM)环境中，确保可靠的安全措施至关重要。为了满足这一关键需求，我们提出了一种专门为评估LLMS、攻击和防御方法而设计的安全基准。SALAD-BENCH以其广度、丰富的多样性、跨越三个层次的复杂分类和多功能而超越传统基准。SALAD-BENCH精心设计了一系列细致的问题，从标准查询到复杂的问题，丰富了攻击、防御修改和多项选择。为了有效地管理固有的复杂性，我们引入了一种创新的评估器：基于LLM的针对QA对的MD-裁判，特别关注攻击增强的查询，确保无缝和可靠的评估。上述组件将沙拉工作台从标准的LLM安全评估扩展到LLM攻防方法评估，确保了联合用途的实用性。我们广泛的实验揭示了低密度脂蛋白对新出现的威胁的弹性以及当代防御战术的有效性。数据和评估器在https://github.com/OpenSafetyLab/SALAD-BENCH.下发布



## **18. Sales Whisperer: A Human-Inconspicuous Attack on LLM Brand Recommendations**

销售耳语者：对LLM品牌推荐的人性不起眼的攻击 cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04755v1) [paper-pdf](http://arxiv.org/pdf/2406.04755v1)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Large language model (LLM) users might rely on others (e.g., prompting services), to write prompts. However, the risks of trusting prompts written by others remain unstudied. In this paper, we assess the risk of using such prompts on brand recommendation tasks when shopping. First, we found that paraphrasing prompts can result in LLMs mentioning given brands with drastically different probabilities, including a pair of prompts where the probability changes by 100%. Next, we developed an approach that can be used to perturb an original base prompt to increase the likelihood that an LLM mentions a given brand. We designed a human-inconspicuous algorithm that perturbs prompts, which empirically forces LLMs to mention strings related to a brand more often, by absolute improvements up to 78.3%. Our results suggest that our perturbed prompts, 1) are inconspicuous to humans, 2) force LLMs to recommend a target brand more often, and 3) increase the perceived chances of picking targeted brands.

摘要: 大型语言模型（LLM）用户可能会依赖其他人（例如，提示服务），写提示。然而，相信他人撰写的提示的风险仍然没有得到研究。在本文中，我们评估了购物时在品牌推荐任务中使用此类提示的风险。首先，我们发现重述提示可能会导致LLM以截然不同的概率提及给定品牌，包括一对概率变化100%的提示。接下来，我们开发了一种方法，可用于扰乱原始基本提示，以增加LLM提及给定品牌的可能性。我们设计了一种不引人注目的算法，可以扰乱提示，从经验上迫使LLM更频繁地提及与品牌相关的字符串，绝对改进高达78.3%。我们的结果表明，我们的干扰提示，1）对人类来说不明显，2）迫使LLM更频繁地推荐目标品牌，3）增加了选择目标品牌的感知机会。



## **19. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

冷攻击：具有隐蔽性和可控性的越狱LLM cs.LG

Accepted to ICML 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.08679v2) [paper-pdf](http://arxiv.org/pdf/2402.08679v2)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent (suffix) attack with continuation constraint, but also allow us to address new controllable attack settings such as revising a user query adversarially with paraphrasing constraint, and inserting stealthy attacks in context with position constraint. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5, and GPT-4) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.

摘要: 大型语言模型(LLM)的越狱最近受到越来越多的关注。为了全面评估LLM的安全性，必须考虑具有不同属性的越狱，例如上下文连贯性和情绪/风格变化，因此研究可控越狱是有益的，即如何加强对LLM攻击的控制。在本文中，我们形式化地描述了可控攻击生成问题，并将该问题与自然语言处理的一个热门话题--可控文本生成建立了一种新的联系。基于此，我们采用了基于能量的朗之万动力学约束解码算法(COLD)，这是一种最新的、高效的可控文本生成算法，并引入了冷攻击框架，该框架可以在流畅性、隐蔽性、情感和左右一致性等各种控制要求下统一和自动化搜索敌意LLM攻击。冷攻击的可控性导致了新的越狱场景的多样化，这些场景不仅覆盖了生成具有连续约束的流畅(后缀)攻击的标准设置，而且允许我们应对新的可控攻击设置，如使用释义约束对用户查询进行恶意修改，以及在具有位置约束的上下文中插入隐蔽攻击。我们在不同的LLMS(大骆驼-2、米斯特拉尔、维库纳、瓜纳科、GPT-3.5和GPT-4)上的广泛实验表明，冷攻击具有广泛的适用性、很强的可控性、高成功率和攻击可转移性。我们的代码可以在https://github.com/Yu-Fangxu/COLD-Attack.上找到



## **20. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

NLP任务中的安全一致：弱一致摘要作为上下文内攻击 cs.CL

Accepted to ACL2024 main

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2312.06924v2) [paper-pdf](http://arxiv.org/pdf/2312.06924v2)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integrity of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models, Gemini and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

摘要: 最近在平衡大型语言模型(LLM)的有用性和安全性方面的发展提出了一个关键问题：主流NLP任务是否与安全考虑充分一致？我们的研究集中在通过对抗性攻击获得的安全敏感文件上，揭示了各种NLP任务在安全匹配方面的显著差异。例如，LLMS可以有效地汇总恶意的长文档，但通常拒绝翻译它们。这一差异突显了一个以前未知的漏洞：攻击利用安全一致性较弱的任务(如摘要)，可能会潜在地损害传统上被认为更健壮的任务的完整性，如翻译和问答(QA)。此外，同时使用安全性较低的多个NLP任务会增加LLMS无意中处理有害内容的风险。我们在各种安全对齐的LLM中展示了这些漏洞，特别是Llama2型号、Gemini和GPT-4，这表明迫切需要在广泛的NLP任务中加强安全对齐。



## **21. Evaluating the Efficacy of Large Language Models in Identifying Phishing Attempts**

评估大型语言模型在识别网络钓鱼尝试方面的有效性 cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.15485v3) [paper-pdf](http://arxiv.org/pdf/2404.15485v3)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.

摘要: 网络钓鱼是几十年来流行的一种网络犯罪策略，在当今的数字世界中仍然是一个重大威胁。通过利用聪明的社会工程元素和现代技术，网络犯罪以许多个人、企业和组织为目标，以利用信任和安全。这些网络攻击者往往以许多可信的形式伪装成合法的来源。通过巧妙地使用紧急、恐惧、社会证明和其他操纵策略等心理因素，网络钓鱼者可以诱使个人泄露敏感和个性化的信息。基于这一现代技术中普遍存在的问题，本文旨在分析15个大型语言模型(LLM)在检测网络钓鱼尝试方面的有效性，特别是关注一组随机的“419骗局”电子邮件。目标是通过基于预定义标准分析包含电子邮件元数据的文本文件，确定哪些LLM可以准确检测钓鱼电子邮件。实验得出的结论是，ChatGPT 3.5、GPT-3.5-Turbo-Indict和ChatGPT模型在检测钓鱼电子邮件方面最有效。



## **22. Defending LLMs against Jailbreaking Attacks via Backtranslation**

通过反向翻译保护LLM免受越狱攻击 cs.CL

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2402.16459v3) [paper-pdf](http://arxiv.org/pdf/2402.16459v3)

**Authors**: Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh

**Abstract**: Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts. Our implementation is based on our library for LLM jailbreaking defense algorithms at \url{https://github.com/YihanWang617/llm-jailbreaking-defense}, and the code for reproducing our experiments is available at \url{https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation}.

摘要: 尽管许多大型语言模型(LLM)已经接受了拒绝有害请求的培训，但它们仍然容易受到越狱攻击，这些攻击会重写原始提示以掩盖其有害意图。在本文中，我们提出了一种新的方法来防御‘’反向翻译‘’越狱攻击。具体地说，给定目标LLM从输入提示生成的初始响应，我们的反向翻译会提示语言模型推断出可能导致该响应的输入提示。推断的提示称为反向翻译提示，它往往会揭示原始提示的实际意图，因为它是基于LLM的响应生成的，而不是由攻击者直接操纵的。然后，我们在回译的提示上再次运行目标LLM，如果模型拒绝回译的提示，我们将拒绝原始提示。我们解释说，拟议的辩护在其有效性和效率方面提供了几个好处。我们的经验证明，我们的防御显著优于基线，在基线难以达到的情况下，我们的防御对良性输入提示的生成质量也几乎没有影响。我们的实现基于我们在\url{https://github.com/YihanWang617/llm-jailbreaking-defense}，的LLm越狱防御算法库，重现我们实验的代码可以在\url{https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation}.上找到



## **23. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

BadRAG：识别大型语言模型检索增强生成中的漏洞 cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.00083v2) [paper-pdf](http://arxiv.org/pdf/2406.00083v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.

摘要: 大型语言模型(LLM)受到过时信息和生成错误数据的倾向的限制，这通常被称为“幻觉”。检索-增强生成(RAG)结合了基于检索的方法和生成模型的优点，解决了这些局限性。这种方法涉及从大型最新数据集中检索相关信息，并使用它来改进生成过程，从而产生更准确和符合上下文的响应。尽管有好处，但RAG为LLMS带来了新的攻击面，特别是因为RAG数据库通常来自公共数据，如Web。本文提出用TrojRAG{}来识别检索零件(RAG数据库)上的漏洞和攻击，以及它们对生成零件(LLM)的间接攻击。具体地说，我们发现毒化几个定制的内容段落可以实现检索后门，其中检索对于干净的查询工作得很好，但总是返回定制的有毒对抗性查询。触发器和有毒段落可以高度定制，以实施各种攻击。例如，触发点可能是一个语义组，比如“共和党、唐纳德·特朗普等。”对抗性段落可以针对不同的内容量身定做，不仅与触发因素有关，还可以用来间接攻击生成性LLM而不修改它们。这些攻击可以包括针对RAG的拒绝服务攻击和针对受触发器限制的LLM生成的语义引导攻击。我们的实验表明，只要毒化10篇对抗性文章，就可以诱导98.2%的成功率来检索对抗性文章。然后，这些文章可以将基于RAG的GPT-4的拒绝率从0.01%提高到74.6%，或者将目标查询的否定回复率从0.22%提高到72%。



## **24. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04031v1) [paper-pdf](http://arxiv.org/pdf/2406.04031v1)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **25. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

模拟失调：大型语言模型的安全调整可能会适得其反！ cs.CL

ACL 2024

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2402.12343v4) [paper-pdf](http://arxiv.org/pdf/2402.12343v4)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) undergo safety alignment to ensure safe conversations with humans. However, this paper introduces a training-free attack method capable of reversing safety alignment, converting the outcomes of stronger alignment into greater potential for harm by accessing only LLM output token distributions. Specifically, our method achieves this reversal by contrasting the output token distribution of a safety-aligned language model (e.g., Llama-2-chat) against its pre-trained version (e.g., Llama-2), so that the token predictions are shifted towards the opposite direction of safety alignment. We name this method emulated disalignment (ED) because sampling from this contrastive distribution provably emulates the result of fine-tuning to minimize a safety reward. Our experiments with ED across three evaluation datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rates in 43 out of 48 evaluation subsets by a large margin. Eventually, given ED's reliance on language model output token distributions, which particularly compromises open-source models, our findings highlight the need to reassess the open accessibility of language models, even if they have been safety-aligned. Code is available at https://github.com/ZHZisZZ/emulated-disalignment.

摘要: 大型语言模型(LLM)经过安全调整，以确保与人类的安全对话。然而，本文介绍了一种无需训练的攻击方法，该方法能够逆转安全对齐，通过仅访问LLM输出令牌分布来将更强对齐的结果转换为更大的潜在危害。具体地说，我们的方法通过将安全对齐的语言模型(例如，Llama-2-Chat)的输出令牌分布与其预先训练的版本(例如，Llama-2)进行比较，从而使令牌预测向安全对齐的相反方向移动，从而实现了这种逆转。我们将这种方法命名为模拟不对齐(ED)，因为从这种对比分布中进行的采样可以被证明是模拟了微调以最小化安全奖励的结果。我们在三个评估数据集和四个模型家族(骆驼-1、骆驼-2、米斯特拉尔和羊驼)上使用ED进行的实验表明，ED的危害性是预训练模型的两倍，并且性能优于强基线，在48个评估子集中的43个子集上获得了最高的伤害率。最终，鉴于ED对语言模型输出令牌分发的依赖，这尤其损害了开源模型，我们的发现突显了重新评估语言模型的开放可访问性的必要性，即使它们是安全一致的。代码可在https://github.com/ZHZisZZ/emulated-disalignment.上找到



## **26. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.14461v2) [paper-pdf](http://arxiv.org/pdf/2404.14461v2)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **27. AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens**

自动越狱：通过依赖的视角探索越狱攻击和防御 cs.CR

32 pages, 2 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03805v1) [paper-pdf](http://arxiv.org/pdf/2406.03805v1)

**Authors**: Lin Lu, Hai Yan, Zenghui Yuan, Jiawen Shi, Wenqi Wei, Pin-Yu Chen, Pan Zhou

**Abstract**: Jailbreak attacks in large language models (LLMs) entail inducing the models to generate content that breaches ethical and legal norm through the use of malicious prompts, posing a substantial threat to LLM security. Current strategies for jailbreak attack and defense often focus on optimizing locally within specific algorithmic frameworks, resulting in ineffective optimization and limited scalability. In this paper, we present a systematic analysis of the dependency relationships in jailbreak attack and defense techniques, generalizing them to all possible attack surfaces. We employ directed acyclic graphs (DAGs) to position and analyze existing jailbreak attacks, defenses, and evaluation methodologies, and propose three comprehensive, automated, and logical frameworks. \texttt{AutoAttack} investigates dependencies in two lines of jailbreak optimization strategies: genetic algorithm (GA)-based attacks and adversarial-generation-based attacks, respectively. We then introduce an ensemble jailbreak attack to exploit these dependencies. \texttt{AutoDefense} offers a mixture-of-defenders approach by leveraging the dependency relationships in pre-generative and post-generative defense strategies. \texttt{AutoEvaluation} introduces a novel evaluation method that distinguishes hallucinations, which are often overlooked, from jailbreak attack and defense responses. Through extensive experiments, we demonstrate that the proposed ensemble jailbreak attack and defense framework significantly outperforms existing research.

摘要: 大型语言模型(LLM)中的越狱攻击需要通过使用恶意提示诱导模型生成违反道德和法律规范的内容，从而对LLM安全构成重大威胁。当前的越狱攻防策略往往集中在特定算法框架内的局部优化，导致优化效果不佳，可扩展性有限。本文系统地分析了越狱攻防技术中的依赖关系，并将其推广到所有可能的攻击面。我们使用有向无环图(DAG)来定位和分析现有的越狱攻击、防御和评估方法，并提出了三个全面的、自动化的和逻辑的框架。Texttt{AutoAttack}研究了两种越狱优化策略的依赖关系：基于遗传算法(GA)的攻击和基于对抗性生成的攻击。然后，我们引入整体越狱攻击来利用这些依赖关系。通过利用生成前和生成后防御策略中的依赖关系，\exttt{AutoDefense}提供了混合防御者的方法。Texttt(自动评估)介绍了一种新的评估方法，可以将经常被忽视的幻觉与越狱攻击和防御反应区分开来。通过大量的实验，我们证明了所提出的集成越狱攻防框架的性能明显优于现有的研究。



## **28. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.17263v3) [paper-pdf](http://arxiv.org/pdf/2401.17263v3)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **29. Ranking Manipulation for Conversational Search Engines**

对话式搜索引擎的排名操纵 cs.CL

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03589v1) [paper-pdf](http://arxiv.org/pdf/2406.03589v1)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.

摘要: 各大搜索引擎提供商正在快速整合大型语言模型(LLM)生成的内容，以响应用户查询。这些对话式搜索引擎通过将检索到的网站文本加载到LLM上下文中进行操作以进行摘要和解释。最近的研究表明，LLM非常容易受到越狱和快速注入攻击，这些攻击使用敌意字符串破坏LLM的安全和质量目标。这项工作调查了提示注入对对话式搜索引擎引用的来源的排名顺序的影响。为此，我们引入了一个聚焦于真实世界消费产品网站的数据集，并将会话搜索排名形式化为一个对抗性问题。在实验上，我们分析了在没有对抗性注入的情况下的会话搜索排名，结果表明不同的LLM在产品名称、文档内容和上下文位置的优先顺序上存在显著差异。然后，我们提出了一种基于攻击树的越狱技术，该技术可靠地推广排名较低的产品。重要的是，这些攻击有效地转移到了最先进的会话搜索引擎，如Pplexity.ai。考虑到网站所有者有强大的经济动机来提高他们的搜索排名，我们认为我们的问题表达对于未来的稳健性工作至关重要。



## **30. Stealthy Attack on Large Language Model based Recommendation**

对基于大型语言模型的推荐的隐形攻击 cs.CL

ACL 2024 Main

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.14836v2) [paper-pdf](http://arxiv.org/pdf/2402.14836v2)

**Authors**: Jinghao Zhang, Yuting Liu, Qiang Liu, Shu Wu, Guibing Guo, Liang Wang

**Abstract**: Recently, the powerful large language models (LLMs) have been instrumental in propelling the progress of recommender systems (RS). However, while these systems have flourished, their susceptibility to security threats has been largely overlooked. In this work, we reveal that the introduction of LLMs into recommendation models presents new security vulnerabilities due to their emphasis on the textual content of items. We demonstrate that attackers can significantly boost an item's exposure by merely altering its textual content during the testing phase, without requiring direct interference with the model's training process. Additionally, the attack is notably stealthy, as it does not affect the overall recommendation performance and the modifications to the text are subtle, making it difficult for users and platforms to detect. Our comprehensive experiments across four mainstream LLM-based recommendation models demonstrate the superior efficacy and stealthiness of our approach. Our work unveils a significant security gap in LLM-based recommendation systems and paves the way for future research on protecting these systems.

摘要: 近年来，强大的大型语言模型(LLMS)在推动推荐系统(RS)的发展方面发挥了重要作用。然而，尽管这些系统蓬勃发展，但它们对安全威胁的敏感性在很大程度上被忽视了。在这项工作中，我们揭示了将LLMS引入推荐模型中会出现新的安全漏洞，这是因为它们强调项目的文本内容。我们证明，攻击者只需在测试阶段改变项目的文本内容，就可以显著增加项目的曝光率，而不需要直接干预模型的训练过程。此外，攻击具有明显的隐蔽性，因为它不会影响整体推荐性能，而且对文本的修改也很微妙，使得用户和平台很难检测到。我们对四个主流的基于LLM的推荐模型进行了全面的实验，证明了我们的方法具有优越的有效性和隐蔽性。我们的工作揭示了基于LLM的推荐系统中存在的一个显著的安全漏洞，并为未来保护这些系统的研究铺平了道路。



## **31. Improved Techniques for Optimization-Based Jailbreaking on Large Language Models**

基于优化的大型语言模型越狱改进技术 cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.21018v2) [paper-pdf](http://arxiv.org/pdf/2405.21018v2)

**Authors**: Xiaojun Jia, Tianyu Pang, Chao Du, Yihao Huang, Jindong Gu, Yang Liu, Xiaochun Cao, Min Lin

**Abstract**: Large language models (LLMs) are being rapidly developed, and a key component of their widespread deployment is their safety-related alignment. Many red-teaming efforts aim to jailbreak LLMs, where among these efforts, the Greedy Coordinate Gradient (GCG) attack's success has led to a growing interest in the study of optimization-based jailbreaking techniques. Although GCG is a significant milestone, its attacking efficiency remains unsatisfactory. In this paper, we present several improved (empirical) techniques for optimization-based jailbreaks like GCG. We first observe that the single target template of "Sure" largely limits the attacking performance of GCG; given this, we propose to apply diverse target templates containing harmful self-suggestion and/or guidance to mislead LLMs. Besides, from the optimization aspects, we propose an automatic multi-coordinate updating strategy in GCG (i.e., adaptively deciding how many tokens to replace in each step) to accelerate convergence, as well as tricks like easy-to-hard initialisation. Then, we combine these improved technologies to develop an efficient jailbreak method, dubbed I-GCG. In our experiments, we evaluate on a series of benchmarks (such as NeurIPS 2023 Red Teaming Track). The results demonstrate that our improved techniques can help GCG outperform state-of-the-art jailbreaking attacks and achieve nearly 100% attack success rate. The code is released at https://github.com/jiaxiaojunQAQ/I-GCG.

摘要: 大型语言模型(LLM)正在迅速开发，其广泛部署的一个关键组件是与安全相关的一致性。许多红色团队的目标是越狱LLM，其中贪婪坐标梯度(GCG)攻击的成功导致了人们对基于优化的越狱技术的研究越来越感兴趣。虽然GCG是一个重要的里程碑，但其攻击效率仍然不能令人满意。在这篇文章中，我们提出了几种改进的(经验)技术，用于基于优化的越狱，如GCG。我们首先观察到单一目标模板“Sure”在很大程度上限制了GCG的攻击性能；鉴于此，我们建议使用包含有害自我暗示和/或引导的不同目标模板来误导LLM。此外，在优化方面，我们提出了GCG中的自动多坐标更新策略(即自适应地决定每一步需要替换多少个令牌)来加速收敛，以及容易初始化等技巧。然后，我们结合这些改进的技术开发了一种高效的越狱方法，称为I-GCG。在我们的实验中，我们在一系列基准(例如NeurIPS 2023 Red Teaming Track)上进行了评估。结果表明，改进后的技术可以帮助GCG超越最先进的越狱攻击，并获得近100%的攻击成功率。该代码在https://github.com/jiaxiaojunQAQ/I-GCG.上发布



## **32. CR-UTP: Certified Robustness against Universal Text Perturbations on Large Language Models**

CR-GPT：针对大型语言模型上通用文本扰动的鲁棒性认证 cs.CL

Accepted by ACL Findings 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.01873v2) [paper-pdf](http://arxiv.org/pdf/2406.01873v2)

**Authors**: Qian Lou, Xin Liang, Jiaqi Xue, Yancheng Zhang, Rui Xie, Mengxin Zheng

**Abstract**: It is imperative to ensure the stability of every prediction made by a language model; that is, a language's prediction should remain consistent despite minor input variations, like word substitutions. In this paper, we investigate the problem of certifying a language model's robustness against Universal Text Perturbations (UTPs), which have been widely used in universal adversarial attacks and backdoor attacks. Existing certified robustness based on random smoothing has shown considerable promise in certifying the input-specific text perturbations (ISTPs), operating under the assumption that any random alteration of a sample's clean or adversarial words would negate the impact of sample-wise perturbations. However, with UTPs, masking only the adversarial words can eliminate the attack. A naive method is to simply increase the masking ratio and the likelihood of masking attack tokens, but it leads to a significant reduction in both certified accuracy and the certified radius due to input corruption by extensive masking. To solve this challenge, we introduce a novel approach, the superior prompt search method, designed to identify a superior prompt that maintains higher certified accuracy under extensive masking. Additionally, we theoretically motivate why ensembles are a particularly suitable choice as base prompts for random smoothing. The method is denoted by superior prompt ensembling technique. We also empirically confirm this technique, obtaining state-of-the-art results in multiple settings. These methodologies, for the first time, enable high certified accuracy against both UTPs and ISTPs. The source code of CR-UTP is available at \url {https://github.com/UCFML-Research/CR-UTP}.

摘要: 必须确保语言模型做出的每个预测的稳定性；也就是说，语言的预测应该保持一致，尽管输入有微小的变化，如单词替换。在本文中，我们研究了语言模型对通用文本扰动(UTP)的稳健性证明问题，UTP被广泛应用于通用对抗性攻击和后门攻击。现有的基于随机平滑的已证明的稳健性在证明特定于输入的文本扰动(ISTP)方面显示出相当大的前景，其操作是在假设样本的干净或敌意的单词的任何随机改变将否定样本方面的扰动的影响的情况下进行的。然而，对于UTP，只屏蔽敌意的单词就可以消除攻击。一种天真的方法是简单地增加掩蔽率和掩蔽攻击令牌的可能性，但由于广泛的掩蔽导致输入损坏，它导致认证的准确性和认证的半径都显著降低。为了解决这一挑战，我们引入了一种新的方法，高级提示搜索方法，旨在识别在广泛掩蔽下保持更高认证准确率的高级提示。此外，我们从理论上解释了为什么作为随机平滑的基础提示，集合是特别合适的选择。这种方法以卓越的即时集成技术表示。我们还从经验上证实了这一技术，在多个环境下获得了最先进的结果。这些方法首次针对UTP和ISTP实现了高度认证的准确性。CR-UTP的源代码可在\url{https://github.com/UCFML-Research/CR-UTP}.



## **33. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

稳健的CLIP：稳健的大型视觉语言模型的视觉嵌入的无监督对抗微调 cs.LG

ICML 2024 Oral

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.12336v2) [paper-pdf](http://arxiv.org/pdf/2402.12336v2)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available at https://github.com/chs20/RobustVLM

摘要: OpenFlamingo、LLaVA和GPT-4等多模式基础模型越来越多地用于各种实际任务。先前的工作表明，这些模型非常容易受到视觉通道的对抗性攻击。这些攻击可以被用来传播虚假信息或欺骗用户，从而构成巨大的风险，这使得大型多通道基础模型的健壮性成为一个紧迫的问题。在许多大型视觉语言模型(如LLaVA和OpenFlamingo)中，剪辑模型或其变体之一被用作冻结的视觉编码器。我们提出了一种无监督的对抗性微调方案，以获得一个健壮的裁剪视觉编码器，它对依赖于裁剪的所有视觉下游任务(LVLM，零镜头分类)都具有健壮性。特别是，我们表明，一旦用我们的健壮模型取代了原始的剪辑模型，恶意第三方提供的篡改图像就不再可能对LVLMS的用户进行秘密攻击。不需要对下游低成本模块进行再培训或微调。代码和健壮模型可在https://github.com/chs20/RobustVLM上获得



## **34. Text Embedding Inversion Security for Multilingual Language Models**

多语言语言模型的文本嵌入翻转安全性 cs.CL

18 pages, 17 Tables, 6 Figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.12192v4) [paper-pdf](http://arxiv.org/pdf/2401.12192v4)

**Authors**: Yiyi Chen, Heather Lent, Johannes Bjerva

**Abstract**: Textual data is often represented as real-numbered embeddings in NLP, particularly with the popularity of large language models (LLMs) and Embeddings as a Service (EaaS). However, storing sensitive information as embeddings can be susceptible to security breaches, as research shows that text can be reconstructed from embeddings, even without knowledge of the underlying model. While defence mechanisms have been explored, these are exclusively focused on English, leaving other languages potentially exposed to attacks. This work explores LLM security through multilingual embedding inversion. We define the problem of black-box multilingual and cross-lingual inversion attacks, and explore their potential implications. Our findings suggest that multilingual LLMs may be more vulnerable to inversion attacks, in part because English-based defences may be ineffective. To alleviate this, we propose a simple masking defense effective for both monolingual and multilingual models. This study is the first to investigate multilingual inversion attacks, shedding light on the differences in attacks and defenses across monolingual and multilingual settings.

摘要: 文本数据通常在NLP中表示为实数嵌入，特别是随着大型语言模型(LLM)和嵌入即服务(EaaS)的流行。然而，将敏感信息存储为嵌入很容易受到安全漏洞的影响，因为研究表明，即使不知道底层模型，也可以从嵌入中重构文本。虽然已经探索了防御机制，但这些机制完全集中在英语上，使其他语言可能面临攻击。该工作通过多语言嵌入倒置来探索LLM安全性。我们定义了黑盒多语言和跨语言倒置攻击的问题，并探讨了它们的潜在含义。我们的发现表明，多语种LLM可能更容易受到倒置攻击，部分原因是基于英语的防御可能无效。为了缓解这一问题，我们提出了一种简单的掩蔽防御方法，既适用于单语言模型，也适用于多语言模型。这项研究是对多语言倒置攻击的第一次调查，揭示了单语和多语环境下攻击和防御的差异。



## **35. BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents**

BadAgent：在LLM代理中插入并激活后门攻击 cs.CL

Accepted by ACL 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03007v1) [paper-pdf](http://arxiv.org/pdf/2406.03007v1)

**Authors**: Yifei Wang, Dizhan Xue, Shengjie Zhang, Shengsheng Qian

**Abstract**: With the prosperity of large language models (LLMs), powerful LLM-based intelligent agents have been developed to provide customized services with a set of user-defined tools. State-of-the-art methods for constructing LLM agents adopt trained LLMs and further fine-tune them on data for the agent task. However, we show that such methods are vulnerable to our proposed backdoor attacks named BadAgent on various agent tasks, where a backdoor can be embedded by fine-tuning on the backdoor data. At test time, the attacker can manipulate the deployed LLM agents to execute harmful operations by showing the trigger in the agent input or environment. To our surprise, our proposed attack methods are extremely robust even after fine-tuning on trustworthy data. Though backdoor attacks have been studied extensively in natural language processing, to the best of our knowledge, we could be the first to study them on LLM agents that are more dangerous due to the permission to use external tools. Our work demonstrates the clear risk of constructing LLM agents based on untrusted LLMs or data. Our code is public at https://github.com/DPamK/BadAgent

摘要: 随着大型语言模型(LLM)的繁荣，基于大型语言模型的智能代理被开发出来，以提供一套用户定义的工具来提供定制的服务。构建LLM代理的最先进方法采用经过训练的LLM，并根据代理任务的数据进一步微调它们。然而，我们发现这些方法容易受到我们提出的针对各种代理任务的名为BadAgent的后门攻击，其中可以通过对后门数据进行微调来嵌入后门。在测试时，攻击者可以通过在代理输入或环境中显示触发器来操纵部署的LLM代理执行有害操作。令我们惊讶的是，即使在对可信数据进行微调之后，我们提出的攻击方法也非常健壮。虽然后门攻击在自然语言处理中已经得到了广泛的研究，但就我们所知，我们可能是第一个在LLM代理上研究它们的，这些代理由于被允许使用外部工具而更危险。我们的工作证明了基于不可信的LLM或数据构建LLM代理的明显风险。我们的代码在https://github.com/DPamK/BadAgent上是公开的



## **36. Large Language Models are Few-shot Generators: Proposing Hybrid Prompt Algorithm To Generate Webshell Escape Samples**

大型语言模型是少镜头生成器：提出混合提示算法来生成Webshell Escape示例 cs.CR

21 pages, 17 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.07408v2) [paper-pdf](http://arxiv.org/pdf/2402.07408v2)

**Authors**: Mingrui Ma, Lansheng Han, Chunjie Zhou

**Abstract**: The frequent occurrence of cyber-attacks has made webshell attacks and defense gradually become a research hotspot in the field of network security. However, the lack of publicly available benchmark datasets and the over-reliance on manually defined rules for webshell escape sample generation have slowed down the progress of research related to webshell escape sample generation and artificial intelligence (AI)-based webshell detection. To address the drawbacks of weak webshell sample escape capabilities, the lack of webshell datasets with complex malicious features, and to promote the development of webshell detection, we propose the Hybrid Prompt algorithm for webshell escape sample generation with the help of large language models. As a prompt algorithm specifically developed for webshell sample generation, the Hybrid Prompt algorithm not only combines various prompt ideas including Chain of Thought, Tree of Thought, but also incorporates various components such as webshell hierarchical module and few-shot example to facilitate the LLM in learning and reasoning webshell escape strategies. Experimental results show that the Hybrid Prompt algorithm can work with multiple LLMs with excellent code reasoning ability to generate high-quality webshell samples with high Escape Rate (88.61% with GPT-4 model on VirusTotal detection engine) and (Survival Rate 54.98% with GPT-4 model).

摘要: 网络攻击的频繁发生使网络外壳攻击与防御逐渐成为网络安全领域的研究热点。然而，缺乏公开可用的基准数据集，以及过度依赖人工定义的网页外壳逃逸样本生成规则，减缓了与网页外壳逃逸样本生成和基于人工智能(AI)的网页外壳检测相关的研究进展。针对Web外壳样本逃逸能力弱、缺乏具有复杂恶意特征的Web外壳数据集的缺点，为促进Web外壳检测的发展，本文提出了一种基于大型语言模型的Web外壳逃逸样本混合提示生成算法。混合提示算法是专门为Web外壳样本生成而开发的一种提示算法，它不仅结合了链式、树型等多种提示思想，还加入了Web外壳分层模块、少镜头实例等多种组件，方便了LLM在学习和推理Web外壳逃逸策略方面的应用。实验结果表明，混合提示算法能够与具有良好代码推理能力的多个LLMS协同工作，生成高逃逸率(在VirusTotal检测引擎上的GPT-4模型为88.61%)和(GPT-4模型的存活率为54.98%)的高质量Web外壳样本。



## **37. Enhancing Jailbreak Attack Against Large Language Models through Silent Tokens**

通过无声令牌增强针对大型语言模型的越狱攻击 cs.AI

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.20653v2) [paper-pdf](http://arxiv.org/pdf/2405.20653v2)

**Authors**: Jiahao Yu, Haozheng Luo, Jerry Yao-Chieh Hu, Wenbo Guo, Han Liu, Xinyu Xing

**Abstract**: Along with the remarkable successes of Language language models, recent research also started to explore the security threats of LLMs, including jailbreaking attacks. Attackers carefully craft jailbreaking prompts such that a target LLM will respond to the harmful question. Existing jailbreaking attacks require either human experts or leveraging complicated algorithms to craft jailbreaking prompts. In this paper, we introduce BOOST, a simple attack that leverages only the eos tokens. We demonstrate that rather than constructing complicated jailbreaking prompts, the attacker can simply append a few eos tokens to the end of a harmful question. It will bypass the safety alignment of LLMs and lead to successful jailbreaking attacks. We further apply BOOST to four representative jailbreak methods and show that the attack success rates of these methods can be significantly enhanced by simply adding eos tokens to the prompt. To understand this simple but novel phenomenon, we conduct empirical analyses. Our analysis reveals that adding eos tokens makes the target LLM believe the input is much less harmful, and eos tokens have low attention values and do not affect LLM's understanding of the harmful questions, leading the model to actually respond to the questions. Our findings uncover how fragile an LLM is against jailbreak attacks, motivating the development of strong safety alignment approaches.

摘要: 在语言模型取得显著成功的同时，最近的研究也开始探索LLMS的安全威胁，包括越狱攻击。攻击者精心设计越狱提示，以便目标LLM会对这个有害的问题做出回应。现有的越狱攻击要么需要人类专家，要么需要利用复杂的算法来设计越狱提示。在本文中，我们将介绍Boost，这是一种仅利用Eos令牌的简单攻击。我们演示了，攻击者可以简单地在有害问题的末尾添加几个Eos令牌，而不是构建复杂的越狱提示。它将绕过LLMS的安全对准，并导致成功的越狱攻击。我们进一步将Boost应用于四种具有代表性的越狱方法，并表明只需在提示符中添加Eos令牌即可显着提高这些方法的攻击成功率。为了理解这一简单但新颖的现象，我们进行了实证分析。我们的分析表明，添加Eos标记会使目标LLM认为输入的危害要小得多，并且Eos标记的关注值较低，不会影响LLM对有害问题的理解，从而导致模型实际回答问题。我们的发现揭示了LLM对越狱攻击的脆弱性，促使了强大的安全对齐方法的发展。



## **38. Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models**

水印能在翻译中幸存吗？大型语言模型文本水印的跨语言一致性研究 cs.CL

ACL 2024 (main conference)

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.14007v2) [paper-pdf](http://arxiv.org/pdf/2402.14007v2)

**Authors**: Zhiwei He, Binglin Zhou, Hongkun Hao, Aiwei Liu, Xing Wang, Zhaopeng Tu, Zhuosheng Zhang, Rui Wang

**Abstract**: Text watermarking technology aims to tag and identify content produced by large language models (LLMs) to prevent misuse. In this study, we introduce the concept of cross-lingual consistency in text watermarking, which assesses the ability of text watermarks to maintain their effectiveness after being translated into other languages. Preliminary empirical results from two LLMs and three watermarking methods reveal that current text watermarking technologies lack consistency when texts are translated into various languages. Based on this observation, we propose a Cross-lingual Watermark Removal Attack (CWRA) to bypass watermarking by first obtaining a response from an LLM in a pivot language, which is then translated into the target language. CWRA can effectively remove watermarks, decreasing the AUCs to a random-guessing level without performance loss. Furthermore, we analyze two key factors that contribute to the cross-lingual consistency in text watermarking and propose X-SIR as a defense method against CWRA. Code: https://github.com/zwhe99/X-SIR.

摘要: 文本水印技术旨在标记和识别大型语言模型(LLM)产生的内容，以防止误用。在这项研究中，我们在文本水印中引入了跨语言一致性的概念，用来评估文本水印在翻译成其他语言后保持其有效性的能力。两种LLMS和三种水印方法的初步实验结果表明，现有的文本水印技术在文本翻译成各种语言时缺乏一致性。基于这一观察结果，我们提出了一种跨语言水印移除攻击(CWRA)，通过首先从旋转语言的LLM获得响应，然后将其翻译成目标语言来绕过水印。CWRA可以有效地去除水印，在不损失性能的情况下将AUC降低到随机猜测的水平。此外，我们分析了影响文本水印跨语言一致性的两个关键因素，并提出了X-SIR作为一种针对CWRA的防御方法。代码：https://github.com/zwhe99/X-SIR.



## **39. QROA: A Black-Box Query-Response Optimization Attack on LLMs**

QROA：对LLM的黑匣子查询响应优化攻击 cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02044v1) [paper-pdf](http://arxiv.org/pdf/2406.02044v1)

**Authors**: Hussein Jawad, Nicolas J. -B. BRUNEL

**Abstract**: Large Language Models (LLMs) have surged in popularity in recent months, yet they possess concerning capabilities for generating harmful content when manipulated. This study introduces the Query-Response Optimization Attack (QROA), an optimization-based strategy designed to exploit LLMs through a black-box, query-only interaction. QROA adds an optimized trigger to a malicious instruction to compel the LLM to generate harmful content. Unlike previous approaches, QROA does not require access to the model's logit information or any other internal data and operates solely through the standard query-response interface of LLMs. Inspired by deep Q-learning and Greedy coordinate descent, the method iteratively updates tokens to maximize a designed reward function. We tested our method on various LLMs such as Vicuna, Falcon, and Mistral, achieving an Attack Success Rate (ASR) over 80\%. We also tested the model against Llama2-chat, the fine-tuned version of Llama2 designed to resist Jailbreak attacks, achieving good ASR with a suboptimal initial trigger seed. This study demonstrates the feasibility of generating jailbreak attacks against deployed LLMs in the public domain using black-box optimization methods, enabling more comprehensive safety testing of LLMs.

摘要: 近几个月来，大型语言模型(LLM)越来越受欢迎，但它们具有在被操纵时生成有害内容的令人担忧的能力。这项研究介绍了查询-响应优化攻击(QROA)，这是一种基于优化的策略，旨在通过黑盒、仅查询的交互来利用LLMS。QROA向恶意指令添加了优化的触发器，以迫使LLM生成有害内容。与以前的方法不同，QROA不需要访问模型的Logit信息或任何其他内部数据，只通过LLMS的标准查询-响应接口进行操作。受深度Q学习和贪婪坐标下降的启发，该方法迭代更新令牌以最大化所设计的奖励函数。我们在维库纳、猎鹰和米斯特拉尔等不同的LLMS上测试了我们的方法，取得了80%以上的攻击成功率(ASR)。我们还在Llama2-Chat上测试了该模型，Llama2-Chat是Llama2的微调版本，旨在抵抗越狱攻击，使用次优的初始触发种子实现了良好的ASR。这项研究论证了利用黑盒优化方法对部署在公共领域的LLM进行越狱攻击的可行性，从而实现了对LLM进行更全面的安全测试。



## **40. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

Bileve：通过双层签名保护大型语言模型中的文本出处，防止欺骗 cs.CR

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01946v1) [paper-pdf](http://arxiv.org/pdf/2406.01946v1)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability.

摘要: 大型语言模型(LLM)的文本水印通常用于识别机器生成内容的来源，这有望在打击深度虚假或有害内容时评估责任。虽然现有的水印技术通常将健壮性放在免受删除攻击的优先位置，但不幸的是，它们容易受到欺骗性攻击：恶意行为者可以巧妙地更改LLM生成的响应的含义，甚至伪造有害内容，可能会将责任错误地归咎于LLM开发人员。为了克服这一问题，我们提出了一种双层签名方案BiLEVE，该方案通过一种新颖的基于等级的采样策略嵌入细粒度的签名比特用于完整性检查(缓解欺骗攻击)，并在签名无效时嵌入粗粒度的信号来跟踪文本来源(增强了可检测性)。与传统的只输出二进制结果的水印检测器相比，BiLEVE在检测过程中可以区分5种场景，可靠地追踪文本来源和规范LLM。在OPT-1.3B和LLAMA-7B上进行的实验证明了BiLEVE在抵抗欺骗攻击方面的有效性，并增强了可检测性。



## **41. ASETF: A Novel Method for Jailbreak Attack on LLMs through Translate Suffix Embeddings**

ASTF：一种通过翻译后缀嵌入对LLM进行越狱攻击的新方法 cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.16006v2) [paper-pdf](http://arxiv.org/pdf/2402.16006v2)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. However, similar to traditional text adversarial attacks, this approach, while effective, is limited by the challenge of the discrete tokens. This gradient based discrete optimization attack requires over 100,000 LLM calls, and due to the unreadable of adversarial suffixes, it can be relatively easily penetrated by common defense methods such as perplexity filters. To cope with this challenge, in this paper, we proposes an Adversarial Suffix Embedding Translation Framework (ASETF), aimed at transforming continuous adversarial suffix embeddings into coherent and understandable text. This method greatly reduces the computational overhead during the attack process and helps to automatically generate multiple adversarial samples, which can be used as data to strengthen LLMs security defense. Experimental evaluations were conducted on Llama2, Vicuna, and other prominent LLMs, employing harmful directives sourced from the Advbench dataset. The results indicate that our method significantly reduces the computation time of adversarial suffixes and achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini.

摘要: 大型语言模型(LLM)的安全防御方法仍然有限，因为危险的提示是手动管理到少数已知的攻击类型，无法跟上新兴的变体。最近的研究发现，在有害指令上附加后缀可能会破坏LLMS的防御，并导致危险的输出。然而，与传统的文本对抗性攻击类似，该方法虽然有效，但受到离散令牌挑战的限制。这种基于梯度的离散优化攻击需要超过100,000个LLM调用，并且由于敌意后缀的不可读，它可以相对容易地被困惑过滤器等常见防御方法穿透。为了应对这一挑战，本文提出了一个对抗性后缀嵌入翻译框架(ASETF)，旨在将连续的对抗性后缀嵌入转换成连贯的、可理解的文本。该方法大大减少了攻击过程中的计算开销，有助于自动生成多个对抗性样本，作为加强LLMS安全防御的数据。在Llama2、Vicuna和其他著名的LLM上进行了实验评估，采用了来自Advbench数据集的有害指令。实验结果表明，该方法显著减少了对抗性后缀的计算时间，取得了比现有技术更好的攻击成功率，同时显著提高了提示的文本流畅性。此外，我们的方法可以推广到更广泛的方法来生成可转移的敌意后缀，可以成功地攻击多个LLM，甚至可以攻击黑盒LLM，如ChatGPT和Gemini。



## **42. HoneyGPT: Breaking the Trilemma in Terminal Honeypots with Large Language Model**

HoneyGPT：用大型语言模型打破终端蜜罐中的三重困境 cs.CR

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01882v1) [paper-pdf](http://arxiv.org/pdf/2406.01882v1)

**Authors**: Ziyang Wang, Jianzhou You, Haining Wang, Tianwei Yuan, Shichao Lv, Yang Wang, Limin Sun

**Abstract**: Honeypots, as a strategic cyber-deception mechanism designed to emulate authentic interactions and bait unauthorized entities, continue to struggle with balancing flexibility, interaction depth, and deceptive capability despite their evolution over decades. Often they also lack the capability of proactively adapting to an attacker's evolving tactics, which restricts the depth of engagement and subsequent information gathering. Under this context, the emergent capabilities of large language models, in tandem with pioneering prompt-based engineering techniques, offer a transformative shift in the design and deployment of honeypot technologies. In this paper, we introduce HoneyGPT, a pioneering honeypot architecture based on ChatGPT, heralding a new era of intelligent honeypot solutions characterized by their cost-effectiveness, high adaptability, and enhanced interactivity, coupled with a predisposition for proactive attacker engagement. Furthermore, we present a structured prompt engineering framework that augments long-term interaction memory and robust security analytics. This framework, integrating thought of chain tactics attuned to honeypot contexts, enhances interactivity and deception, deepens security analytics, and ensures sustained engagement.   The evaluation of HoneyGPT includes two parts: a baseline comparison based on a collected dataset and a field evaluation in real scenarios for four weeks. The baseline comparison demonstrates HoneyGPT's remarkable ability to strike a balance among flexibility, interaction depth, and deceptive capability. The field evaluation further validates HoneyGPT's efficacy, showing its marked superiority in enticing attackers into more profound interactive engagements and capturing a wider array of novel attack vectors in comparison to existing honeypot technologies.

摘要: 蜜罐作为一种战略性的网络欺骗机制，旨在模拟真实的交互并诱骗未经授权的实体，尽管经过了几十年的演变，但它仍然在灵活性、交互深度和欺骗性能力之间进行权衡。他们往往也缺乏主动适应攻击者不断变化的战术的能力，这限制了接触的深度和随后的信息收集。在这种背景下，大型语言模型的新兴能力与开创性的基于提示的工程技术相结合，为蜜罐技术的设计和部署提供了革命性的转变。在本文中，我们介绍了一种基于ChatGPT的开创性蜜罐架构HoneyGPT，它预示着智能蜜罐解决方案的新时代，其特点是性价比高、适应性强、互动性增强，并易于主动攻击。此外，我们还提出了结构化提示工程框架，以增强长期交互记忆和强大的安全分析能力。该框架整合了与蜜罐环境相协调的链策略思想，增强了互动性和欺骗性，深化了安全分析，并确保了持续参与。HoneyGPT的评估包括两部分：基于收集的数据集的基线比较和为期四周的真实场景下的现场评估。基准比较表明，HoneyGPT在灵活性、交互深度和欺骗性能力之间取得了显著的平衡。现场评估进一步验证了HoneyGPT的有效性，显示出其显著的优势，与现有的蜜罐技术相比，它可以诱使攻击者参与更深入的互动，并捕获更广泛的新型攻击载体。



## **43. Safeguarding Large Language Models: A Survey**

保护大型语言模型：一项调查 cs.CR

under review. arXiv admin note: text overlap with arXiv:2402.01822

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.02622v1) [paper-pdf](http://arxiv.org/pdf/2406.02622v1)

**Authors**: Yi Dong, Ronghui Mu, Yanghao Zhang, Siqi Sun, Tianle Zhang, Changshun Wu, Gaojie Jin, Yi Qi, Jinwei Hu, Jie Meng, Saddek Bensalem, Xiaowei Huang

**Abstract**: In the burgeoning field of Large Language Models (LLMs), developing a robust safety mechanism, colloquially known as "safeguards" or "guardrails", has become imperative to ensure the ethical use of LLMs within prescribed boundaries. This article provides a systematic literature review on the current status of this critical mechanism. It discusses its major challenges and how it can be enhanced into a comprehensive mechanism dealing with ethical issues in various contexts. First, the paper elucidates the current landscape of safeguarding mechanisms that major LLM service providers and the open-source community employ. This is followed by the techniques to evaluate, analyze, and enhance some (un)desirable properties that a guardrail might want to enforce, such as hallucinations, fairness, privacy, and so on. Based on them, we review techniques to circumvent these controls (i.e., attacks), to defend the attacks, and to reinforce the guardrails. While the techniques mentioned above represent the current status and the active research trends, we also discuss several challenges that cannot be easily dealt with by the methods and present our vision on how to implement a comprehensive guardrail through the full consideration of multi-disciplinary approach, neural-symbolic method, and systems development lifecycle.

摘要: 在新兴的大型语言模型(LLM)领域，开发一种强大的安全机制，俗称“保障措施”或“护栏”，已成为确保在规定范围内合乎道德地使用LLM的当务之急。本文对这一关键机制的研究现状进行了系统的文献综述。报告讨论了其面临的主要挑战，以及如何将其加强为在各种情况下处理道德问题的综合机制。首先，本文阐述了主要的LLM服务提供商和开源社区使用的保护机制的现状。然后是评估、分析和增强护栏可能想要强制执行的一些(不想要的)属性的技术，例如幻觉、公平性、隐私等等。在此基础上，我们回顾了规避这些控制(即攻击)、防御攻击和加固护栏的技术。虽然上述技术代表了当前的研究现状和活跃的研究趋势，但我们也讨论了这些方法不容易解决的几个挑战，并提出了我们的愿景，即如何通过充分考虑多学科方法、神经符号方法和系统开发生命周期来实现一个全面的护栏。



## **44. Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**

这是免费午餐：通过模型合并对后门模型进行消毒 cs.CL

accepted to ACL2024 (Findings)

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2402.19334v2) [paper-pdf](http://arxiv.org/pdf/2402.19334v2)

**Authors**: Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, Qiongkai Xu

**Abstract**: The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can significantly remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we verify our hypothesis on various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks on classification and instruction-tuned tasks without additional resources or specific knowledge. Our approach consistently outperforms recent advanced baselines, leading to an average of about 75% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.

摘要: 通过开放源码倡议使预先培训的语言模型民主化，迅速推动了创新，扩大了获得尖端技术的机会。然而，这种开放性也带来了重大的安全风险，包括后门攻击，其中隐藏的恶意行为由特定的输入触发，损害了自然语言处理(NLP)系统的完整性和可靠性。本文认为，将后门模型与其他同类模型合并可以显著补救后门漏洞，即使这些模型不是完全安全的。在我们的实验中，我们在各种模型(Bert-Base、Roberta-Large、Llama2-7B和Mistral-7B)和数据集(SST-2、OLID、AG News和QNLI)上验证了我们的假设。与多种先进的防御方法相比，该方法在不需要额外资源或特定知识的情况下，提供了一种有效的推理阶段防御对分类和指令调优任务的后门攻击。我们的方法始终优于最近的先进基准，导致攻击成功率平均降低约75%。由于模型合并已经成为提高模型性能的既定方法，它提供的关于防御的额外优势可以被视为免费的额外奖励。



## **45. Human vs. Machine: Behavioral Differences Between Expert Humans and Language Models in Wargame Simulations**

人类与机器：战争游戏模拟中专家人类和语言模型之间的行为差异 cs.CY

Updated with new plot and more details

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2403.03407v2) [paper-pdf](http://arxiv.org/pdf/2403.03407v2)

**Authors**: Max Lamparth, Anthony Corso, Jacob Ganz, Oriana Skylar Mastro, Jacquelyn Schneider, Harold Trinkunas

**Abstract**: To some, the advent of artificial intelligence (AI) promises better decision-making and increased military effectiveness while reducing the influence of human error and emotions. However, there is still debate about how AI systems, especially large language models (LLMs), behave compared to humans in high-stakes military decision-making scenarios with the potential for increased risks towards escalation and unnecessary conflicts. To test this potential and scrutinize the use of LLMs for such purposes, we use a new wargame experiment with 107 national security experts designed to look at crisis escalation in a fictional US-China scenario and compare human players to LLM-simulated responses in separate simulations. Wargames have a long history in the development of military strategy and the response of nations to threats or attacks. Here, we show a considerable high-level agreement in the LLM and human responses and significant quantitative and qualitative differences in individual actions and strategic tendencies. These differences depend on intrinsic biases in LLMs regarding the appropriate level of violence following strategic instructions, the choice of LLM, and whether the LLMs are tasked to decide for a team of players directly or first to simulate dialog between players. When simulating the dialog, the discussions lack quality and maintain a farcical harmony. The LLM simulations cannot account for human player characteristics, showing no significant difference even for extreme traits, such as "pacifist" or "aggressive sociopath". Our results motivate policymakers to be cautious before granting autonomy or following AI-based strategy recommendations.

摘要: 对一些人来说，人工智能(AI)的出现保证了更好的决策和更高的军事效力，同时减少了人为错误和情绪的影响。然而，在高风险的军事决策场景中，人工智能系统，特别是大型语言模型(LLM)与人类相比表现如何，可能会增加升级和不必要的冲突的风险，仍存在争议。为了测试这一潜力并仔细审查LLM在此类目的中的使用，我们使用了一个与107名国家安全专家进行的新的军事游戏实验，旨在观察虚构的美国-中国场景中的危机升级，并在不同的模拟中将人类玩家与LLM模拟的反应进行比较。军事演习在军事战略的发展和国家对威胁或攻击的反应方面有着悠久的历史。在这里，我们显示了LLM和人类反应的相当高水平的一致性，以及个别行动和战略趋势的显著数量和质量差异。这些差异取决于LLM中关于遵循战略指令的适当暴力级别的内在偏见，LLM的选择，以及LLM是否被要求直接或首先决定一支球员团队来模拟球员之间的对话。在模拟对话时，讨论缺乏质量，保持了滑稽的和谐。LLM模拟不能解释人类玩家的特征，即使是极端的特征，如“和平主义者”或“攻击性反社会者”，也没有显示出显著的差异。我们的结果促使政策制定者在授予自治权或遵循基于人工智能的战略建议之前保持谨慎。



## **46. PrivacyRestore: Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restoration**

PrivacyRestore：通过隐私删除和恢复在大型语言模型中保留隐私的推理 cs.CR

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01394v1) [paper-pdf](http://arxiv.org/pdf/2406.01394v1)

**Authors**: Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Huiping Zhuang, Cen Chen

**Abstract**: The widespread usage of online Large Language Models (LLMs) inference services has raised significant privacy concerns about the potential exposure of private information in user inputs to eavesdroppers or untrustworthy service providers. Existing privacy protection methods for LLMs suffer from insufficient privacy protection, performance degradation, or severe inference time overhead. In this paper, we propose PrivacyRestore to protect the privacy of user inputs during LLM inference. PrivacyRestore directly removes privacy spans in user inputs and restores privacy information via activation steering during inference. The privacy spans are encoded as restoration vectors. We propose Attention-aware Weighted Aggregation (AWA) which aggregates restoration vectors of all privacy spans in the input into a meta restoration vector. AWA not only ensures proper representation of all privacy spans but also prevents attackers from inferring the privacy spans from the meta restoration vector alone. This meta restoration vector, along with the query with privacy spans removed, is then sent to the server. The experimental results show that PrivacyRestore can protect private information while maintaining acceptable levels of performance and inference efficiency.

摘要: 在线大语言模型(LLMS)推理服务的广泛使用引发了人们对用户输入中的私人信息可能暴露给窃听者或不可信的服务提供商的严重隐私担忧。现有的LLMS隐私保护方法存在隐私保护不足、性能下降或严重的推理时间开销等问题。在本文中，我们提出PrivacyRestore来保护LLM推理过程中用户输入的隐私。PrivacyRestore直接删除用户输入中的隐私范围，并在推理过程中通过激活控制恢复隐私信息。隐私跨度被编码为恢复向量。我们提出了注意力感知加权聚集(AWA)，它将输入中所有隐私跨度的恢复向量聚合成一个元恢复向量。AWA不仅确保了所有隐私范围的正确表示，而且还防止攻击者仅从元恢复向量来推断隐私范围。然后，将该元恢复向量与去除了隐私跨度的查询一起发送到服务器。实验结果表明，PrivacyRestore能够在保持可接受的性能和推理效率的同时保护私有信息。



## **47. Privacy in LLM-based Recommendation: Recent Advances and Future Directions**

基于法学硕士的建议中的隐私：最近的进展和未来的方向 cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01363v1) [paper-pdf](http://arxiv.org/pdf/2406.01363v1)

**Authors**: Sichun Luo, Wei Shao, Yuxuan Yao, Jian Xu, Mingyang Liu, Qintong Li, Bowei He, Maolin Wang, Guanzhi Deng, Hanxu Hou, Xinyi Zhang, Linqi Song

**Abstract**: Nowadays, large language models (LLMs) have been integrated with conventional recommendation models to improve recommendation performance. However, while most of the existing works have focused on improving the model performance, the privacy issue has only received comparatively less attention. In this paper, we review recent advancements in privacy within LLM-based recommendation, categorizing them into privacy attacks and protection mechanisms. Additionally, we highlight several challenges and propose future directions for the community to address these critical problems.

摘要: 如今，大型语言模型（LLM）已与传统推荐模型集成以提高推荐性能。然而，虽然大多数现有作品都专注于提高模型性能，但隐私问题受到的关注相对较少。在本文中，我们回顾了基于LLM的推荐中隐私方面的最新进展，并将其分为隐私攻击和保护机制。此外，我们还强调了几项挑战，并为社区解决这些关键问题提出了未来的方向。



## **48. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

通过对基于LLM的排队模型的对抗攻击探索决策级的鲁棒性 cs.MM

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.19802v2) [paper-pdf](http://arxiv.org/pdf/2405.19802v2)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.

摘要: 具身智能使特工具有深刻的感知力，使他们能够以与现实世界情况密切一致的方式做出反应。大型语言模型(LLM)深入研究语言指令，在为复杂任务制定计划方面发挥着至关重要的作用。因此，基于LLM的具体化模型进一步增强了代理理解和处理信息的能力。然而，这种融合也带来了追求高智商的新挑战。具体地说，攻击者可以通过更改提示来操纵LLMS生成无关甚至恶意的输出。面对这一挑战，我们注意到明显缺乏全面评估基于LLM的体现模型的稳健性所必需的多模式数据集。因此，我们构建了专门为健壮性评估量身定做的具体化智能机器人攻击数据集(Eirad)。此外，设计了两种攻击策略，包括非定向攻击和定向攻击，以有效地模拟一系列不同的攻击场景。同时，在攻击过程中，为了更准确地确定我们的方法在攻击基于LLM的体现模型上是否成功，我们设计了一种新的利用BLIP2模型的攻击成功评估方法。考虑到GCG算法在攻击中的时间和成本密集性，我们设计了一种基于不同目标任务的快速后缀初始化方案，从而加快了收敛过程。实验结果表明，我们的方法在攻击基于LLM的具体模型时表现出了较高的攻击成功率，表明这些模型具有较低的决策级健壮性。



## **49. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本局限性 cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2304.11082v6) [paper-pdf](http://arxiv.org/pdf/2304.11082v6)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了在这个框架的范围内，对于模型所表现出的任何有限概率的行为，存在可以触发模型输出该行为的提示，其概率随着提示的长度的增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，使得LLM容易被提示进入不希望看到的行为。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **50. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

人工智能生成的文本检测器对对抗性扰动是否稳健？ cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01179v1) [paper-pdf](http://arxiv.org/pdf/2406.01179v1)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.

摘要: 大型语言模型(LLM)的广泛使用引发了人们对人工智能生成的文本可能被滥用的担忧，因为这些模型可以生成与人类生成的文本非常相似的内容。目前的人工智能生成文本检测器(AIGT)缺乏对对手扰动的稳健性，即使是字符或单词的微小变化也会导致在区分人工生成文本和人工智能生成文本方面出现逆转。本文研究了现有的AIGT检测方法的稳健性，并介绍了一种新的检测器--暹罗校准重建网络(SCRN)。SCRN使用重构网络来添加和去除文本中的噪声，提取对局部扰动具有鲁棒性的语义表示。我们还提出了一种暹罗校正技术来训练模型，使其在不同的噪声下做出相同的置信度预测，从而提高了模型对对抗性扰动的鲁棒性。在四个公开可用的数据集上的实验表明，SCRN的性能优于所有的基线方法，在对抗性攻击下，其绝对准确率比最佳基线方法提高了6.5-18.25。此外，它在跨域、跨流派和混合来源的场景中表现出出色的泛化能力。代码可在\url{https://github.com/CarlanLark/Robust-AIGC-Detector}.上获得



